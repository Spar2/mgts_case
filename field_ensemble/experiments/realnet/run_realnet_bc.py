"""Варианты B и C поверх уже посчитанного A.

A — голый U-Net, веса (1, 0, 0), порог pooled val FAR 1e-3. Он уже лежит
в artifacts/eval/realnet_synth.json и realnet_real.json. Этот скрипт его не
перевыбирает.

B — веса зафиксированы до карточек и до test: сегментация 0.5, остаток маски
0.25, остаток базовой линии 0.25. Оба добавочных члена — скоры. Вход сети
остаётся каналами предобучения. Порог B — один квантиль FAR 1e-3 фона
pooled val v8+v10 именно этой смеси.

C — те же веса, что у B. Порог каждого файла — квантиль 1e-3 его собственных
валидных пикселей, без меток событий и без карточек.

Маска учится только на фоне train синтетики. AdaBN нет. Карточки читаются
после зон и вариант не выбирают.
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path


def _arg(name, default):
    if name in sys.argv:
        return sys.argv[sys.argv.index(name) + 1]
    return default


_THREADS = _arg("--threads", "2")
os.environ["OMP_NUM_THREADS"] = _THREADS
os.environ["MKL_NUM_THREADS"] = _THREADS
os.environ["OPENBLAS_NUM_THREADS"] = _THREADS
os.environ["NUMEXPR_NUM_THREADS"] = _THREADS

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_realnet as rn
from dts import data, domain, metrics
from experiments.realnet.realnet import (
    MODEL_VERSION, combine_scores, load_pretrained, predict_components, save_model,
    train_mask_head, valid_far_threshold,
)

import torch

# Приор зафиксирован до карточек. Голова — половина, потому что на val она
# одна держала IoU. Четверть — локальный остаток маски (охлаждения и компактные
# очаги). Четверть — остаток своей базовой линии (плато шире 40 м).
W_A = (1.0, 0.0, 0.0)
W_B = (0.5, 0.25, 0.25)


def event(msg: str) -> None:
    line = time.strftime("%H:%M", time.gmtime()) + " realnet-bc: " + msg
    try:
        with open("/workspace/run_logs/_events.log", "a") as f:
            f.write(line + "\n")
    except OSError:
        pass
    print(line, flush=True)


def train_mask(ckpt: Path, cache_v8: str, cache_v10: str, minutes: float, n_adapt: int, tiles_per_file: int):
    model = load_pretrained(ckpt)
    v8_train = data.list_cache(cache_v8, "train")
    v10_train = data.list_cache(cache_v10, "train") if Path(cache_v10).exists() else []
    n8 = min(8, n_adapt, len(v8_train))
    n10 = min(4, max(n_adapt - n8, 0), len(v10_train))
    event(f"маска на synth train-фоне: {n8} файлов v8 + {n10} v10, {minutes:.1f} мин, AdaBN нет")
    tiles, tw, used = [], [], []
    if n8:
        t, w, u = rn.collect_tiles(v8_train, n8, tiles_per_file, seed=0)
        tiles += t
        tw += w
        used += [f"v8:{s}" for s in u]
    if n10:
        t, w, u = rn.collect_tiles(v10_train, n10, tiles_per_file, seed=1)
        tiles += t
        tw += w
        used += [f"v10:{s}" for s in u]
    hist = train_mask_head(model, tiles, tw, minutes=minutes, seed=0, batch=4)
    del tiles, tw
    save_model(model, "artifacts/realnet/model_bc.pt", adapted_on=used, mask_hist=hist,
               source=str(ckpt), use_adabn=False, use_mask=True, use_wide=True,
               weights_b=list(W_B), note="mask head only; unet input is pretrained channels")
    steps = hist[-1].get("step") if hist else 0
    event(f"маска готова, шагов {steps}")
    return model, used, steps


def score_split(model, cache, ver, split, root: Path):
    paths = data.list_cache(cache, split)
    t0 = time.time()
    for i, p in enumerate(paths):
        dst = root / ver / split / p.name
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        rec = rn.build_synth(p, need_wide=True)
        comps = predict_components(model, rec["X"], rec["residual"], rec["sigma"], rec["valid"],
                                   need_mask=True, need_wide=True)
        partial = dst.parent / f"{dst.stem}.partial.npz"
        rn.save_components(partial, comps)
        partial.replace(dst)
        del rec, comps
        if (i + 1) % 5 == 0 or i + 1 == len(paths):
            print(f"{split} {ver} {i+1}/{len(paths)} {time.time()-t0:.0f}s", flush=True)
    return [(root / ver / split / p.name, p, ver) for p in paths]


def pooled_threshold(items, weights, far=1e-3):
    parts = []
    for sp, cp, _ver in items:
        comp = rn.load_components(sp)
        z = np.load(cp)
        valid = np.asarray(z["valid"], bool)
        ev = np.asarray(z["ev"])
        nu = np.asarray(z["nu"])
        bg = valid & (ev == 0) & (nu == 0)
        score = combine_scores(comp[0], comp[1], comp[2], weights)
        parts.append(np.asarray(score[bg], np.float32).ravel())
        del z, comp, score
    bg_scores = np.concatenate(parts) if parts else np.zeros(1, np.float32)
    del parts
    thr = float(metrics.far_threshold(bg_scores, far))
    print(f"thr B n_bg {bg_scores.size} {thr:.6f}", flush=True)
    return thr


def eval_items(items, weights, thr, labels_of, seqs_of, per_file=False):
    by = {}
    rows = []
    for sp, cp, ver in items:
        comp = rn.load_components(sp)
        z = np.load(cp)
        score = combine_scores(comp[0], comp[1], comp[2], weights)
        valid = np.asarray(z["valid"], bool)
        thr_i = valid_far_threshold(score, valid, 1e-3) if per_file else float(thr)
        acc = by.setdefault(ver, rn.new_acc())
        rn.add_frame(acc, score, comp[3], z["valid"], z["ev"], z["nu"], z["x"], z["t"], thr_i,
                     labels_of[ver], Path(cp).stem)
        flagged = (score >= thr_i) & valid
        rows.append(dict(file=Path(cp).stem, ver=ver, threshold=thr_i,
                         alarm_frac=float(flagged.mean()) if valid.size else 0.0))
        del z, comp, score
    out = {}
    pooled = rn.new_acc()
    for ver, acc in by.items():
        out[ver] = rn.finish_acc(acc, seqs_of[ver])
        pooled["tot"] += acc["tot"]
        pooled["hot"] += acc["hot"]
        pooled["cold"] += acc["cold"]
        pooled["far_n"] += acc["far_n"]
        pooled["far_d"] += acc["far_d"]
        pooled["per_seq"].update({f"{ver}:{k}": v for k, v in acc["per_seq"].items()})
        pooled["zones_n"] += acc["zones_n"]
        pooled["zones_false"] += acc["zones_false"]
        pooled["kmh"] += acc["kmh"]
        pooled["ev_rows"] += acc["ev_rows"]
    out["pooled"] = rn.finish_acc(pooled, None)
    if rows:
        ths = np.array([r["threshold"] for r in rows], np.float64)
        fr = np.array([r["alarm_frac"] for r in rows], np.float64)
        summary = dict(n=len(rows), threshold_median=float(np.median(ths)),
                       threshold_min=float(ths.min()), threshold_max=float(ths.max()),
                       alarm_frac_mean=float(fr.mean()))
    else:
        summary = {}
    return out, summary


def copy_variant_a():
    art = Path("artifacts/realnet")
    fig = Path("docs/figures")
    src_all = art / "zones_all.csv"
    if src_all.exists():
        shutil.copyfile(src_all, art / "zones_all_A.csv")
    for p in sorted(art.glob("zones_[0-9].csv")):
        shutil.copyfile(p, art / f"zones_A_{p.stem.split('_')[1]}.csv")
    for p in sorted(fig.glob("realnet_[0-9].png")):
        shutil.copyfile(p, fig / f"realnet_A_{p.stem.split('_')[1]}.png")


def run_real_bc(model, thr_b, real_root, cards):
    root = Path(real_root)
    files = sorted(root.glob("*.csv"))
    if not files:
        raise SystemExit(f"нет csv в {root}")
    grids = [rn.load_real_grid(p, Path("artifacts/realnet/grid")) for p in files]
    names = [g["name"] for g in grids]
    event(f"real B/C на {names}, порог B {thr_b:.4f}, порог C — свой файл")
    try:
        typer = rn.load_typer()
    except Exception as exc:
        typer = None
        print("типизатор не загрузился", exc, flush=True)
    art = Path("artifacts/realnet")
    fig_dir = Path("docs/figures")
    store = {"B": {}, "C": {}}
    summaries = {"B": {}, "C": {}}
    expert = {"B": {}, "C": {}}
    rows_all = {"B": [], "C": []}
    for rec in grids:
        held = rec["name"]
        t0 = time.time()
        x_std, info_std = rn.channelize(rec, need_wide=False)
        _x_w, info_w = rn.channelize(rec, need_wide=True)
        comps = predict_components(model, x_std, info_w["residual"], info_w["sigma_x"], rec["valid"],
                                   need_mask=True, need_wide=True)
        score_b = combine_scores(*comps[:3], W_B)
        thr_c = valid_far_threshold(score_b, rec["valid"], 1e-3)
        specs = (
            ("B", score_b, thr_b, info_w["residual"]),
            ("C", score_b, thr_c, info_w["residual"]),
        )
        for variant, score, thr, residual in specs:
            zones = rn.assemble_zones(score, comps[3], rec["valid"], rec["x"], rec["t"], residual, thr)
            rn.type_zones(typer, zones, x_std, info_std["sigma_x"], rec["x"], rec["t"])
            annotations = []
            try:
                from experiments.realnet.realnet import attach_zone_cv
                annotations = attach_zone_cv(score, residual, rec["valid"], rec["x"], rec["t"], zones)
            except Exception as exc:
                print(f"zone_cv {variant} {held}: {exc}", flush=True)
            rows = list(rn.export.zone_rows(zones, f"{held}.csv", rec["sha256"],
                                            model_version=f"{MODEL_VERSION}-{variant}"))
            rn.export.write_csv(art / f"zones_{variant}_{held}.csv", rows)
            rows_all[variant] += rows
            rn.save_figure(fig_dir / f"realnet_{variant}_{held}.png", rec, residual, score, zones, thr,
                           annotations, variant=variant)
            alarm = [z for z in zones if z.get("alarm", True)]
            cls = {}
            for z in zones:
                cls[z.get("class") or ""] = cls.get(z.get("class") or "", 0) + 1
            flagged = (score >= thr) & rec["valid"]
            wide = [z for z in alarm if z.get("width_m", 0) >= 40]
            cold = [z for z in alarm if z.get("polarity") == "cold"]
            summaries[variant][held] = dict(
                sha256=rec["sha256"], tz_missing=rec["tz_missing"], shape=list(rec["T"].shape),
                hours=round(float(rec["t"][-1] - rec["t"][0]) / 3600.0, 2),
                n_harm=int(info_w["n_harm"]), span_h=round(float(info_w["span_h"]), 2),
                n_zones=len(zones), n_alarm=len(alarm), n_wide_alarm=len(wide), n_cold_alarm=len(cold),
                alarm_fraction=float(flagged[rec["valid"]].mean()) if rec["valid"].any() else 0.0,
                classes=cls, threshold=float(thr),
                weights=dict(seg=W_B[0], mask=W_B[1], wide=W_B[2]),
                sec=round(time.time() - t0, 1))
            described = [rn.describe_zone(z) for z in zones]
            described.sort(key=lambda d: (-d["score"], -d["width_m"]))
            keep = []
            for d in described:
                if len(keep) < 12 or d["width_m"] >= 40 or (d["polarity"] == "cold" and abs(d["peak_dT_C"]) >= 0.4):
                    keep.append(d)
            expert[variant][held] = keep
            store[variant][held] = zones
        event(f"файл {held}: B зон {summaries['B'][held]['n_alarm']} доля {summaries['B'][held]['alarm_fraction']:.4f}; "
              f"C зон {summaries['C'][held]['n_alarm']} доля {summaries['C'][held]['alarm_fraction']:.4f} "
              f"порог {thr_c:.3f}")
        del x_std, comps, score_b
    for variant in ("B", "C"):
        rn.export.write_csv(art / f"zones_all_{variant}.csv", rows_all[variant])
    # Карточки после всех зон. Веса и пороги уже записаны и здесь не меняются.
    card_lines = []
    for cpath in cards:
        with open(cpath) as f:
            card_lines += [json.loads(line) for line in f if line.strip()]
    by_name = {g["name"]: g for g in grids}
    cards_out = {}
    tallies = {}
    for variant in ("B", "C"):
        card_rows = []
        for c in card_lines:
            rec = by_name.get(str(c.get("record")))
            if rec is None:
                card_rows.append(dict(id=c.get("id"), record=str(c.get("record")), presence=c.get("presence"),
                                      status="missing_file"))
                continue
            zones = store[variant][str(c.get("record"))]
            xa, xb, ta, tb = domain.card_bounds(c)
            ix = (rec["x"] >= xa) & (rec["x"] <= xb)
            it = (rec["t"] >= ta) & (rec["t"] <= tb)
            if not ix.any() or not it.any():
                card_rows.append(dict(id=c.get("id"), record=str(c.get("record")), presence=c.get("presence"),
                                      status="box_outside_record", hit_alarm=False, hit_zone=False))
                continue
            alarm = domain.overlapping_zones(zones, xa, xb, ta, tb, True)
            any_z = domain.overlapping_zones(zones, xa, xb, ta, tb, False)
            dom = max(alarm or any_z, key=lambda z: z.get("n_px", 0)) if (alarm or any_z) else None
            card_rows.append(dict(
                id=c.get("id"), record=str(c.get("record")), presence=c.get("presence"), status="ok",
                shape=c.get("shape", ""), notes=c.get("notes", ""),
                hit_alarm=bool(alarm), hit_zone=bool(any_z),
                n_alarm=len(alarm), n_zones=len(any_z),
                dominant_class=None if dom is None else dom.get("class"),
                dominant_polarity=None if dom is None else dom.get("polarity")))
        cards_out[variant] = card_rows
        tallies[variant] = dict(alarm=domain.presence_tally(card_rows, "hit_alarm"),
                                zone=domain.presence_tally(card_rows, "hit_zone"))
        ta = tallies[variant]["alarm"]
        event(f"cards {variant} yes {ta.get('yes_hit_alarm')}/{ta.get('yes_n')} "
              f"no {ta.get('no_hit_alarm')}/{ta.get('no_n')} — описательно, вариант не выбран")
    return dict(files=summaries, cards=cards_out, presence=tallies, expert=expert)


def slim_a(path, key_path):
    if not Path(path).exists():
        return None
    blob = json.load(open(path))
    cur = blob
    for key in key_path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--ckpt", default="artifacts/stage1/unet_v810ft.pt")
    ap.add_argument("--cache-v8", default="artifacts/cache/v8")
    ap.add_argument("--cache-v10", default="artifacts/cache/v10")
    ap.add_argument("--real-root", default="/workspace/repo/src/MGTS_data")
    ap.add_argument("--cards", nargs="*", default=["data/operator_cards/cards_batch1.jsonl",
                                                   "data/operator_cards/cards_batch2.jsonl"])
    ap.add_argument("--adapt-minutes", type=float, default=2.0)
    ap.add_argument("--n-adapt", type=int, default=12)
    ap.add_argument("--tiles-per-file", type=int, default=4)
    ap.add_argument("--redo-mask", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    event("старт B/C, веса приора 0.5/0.25/0.25, вход сети не заменяется")
    model_path = Path("artifacts/realnet/model_bc.pt")
    if model_path.exists() and not a.redo_mask:
        from experiments.realnet.realnet import load_model
        model = load_model(model_path)
        meta = torch.load(model_path, map_location="cpu", weights_only=False)
        used = meta.get("adapted_on", [])
        hist = meta.get("mask_hist") or []
        steps = hist[-1].get("step") if hist else 0
        event(f"маска взята из {model_path}, шагов {steps}")
    else:
        model, used, steps = train_mask(Path(a.ckpt), a.cache_v8, a.cache_v10, a.adapt_minutes,
                                         a.n_adapt, a.tiles_per_file)
    tmp = Path("artifacts/realnet/tmp_bc")
    val_items, test_items = [], []
    for ver, cache in (("v8", a.cache_v8), ("v10", a.cache_v10)):
        if not data.list_cache(cache, "val"):
            continue
        val_items += score_split(model, cache, ver, "val", tmp)
        test_items += score_split(model, cache, ver, "test", tmp)
    event(f"карты val {len(val_items)} test {len(test_items)}")
    thr_b = pooled_threshold(val_items, W_B, 1e-3)
    labels_of, seqs_of = {}, {}
    for ver in ("v8", "v10"):
        try:
            labels_of[ver], seqs_of[ver] = rn.load_meta(ver)
        except FileNotFoundError:
            labels_of[ver], seqs_of[ver] = None, None
    val_b, _sum_b = eval_items(val_items, W_B, thr_b, labels_of, seqs_of, per_file=False)
    val_c, sum_c_val = eval_items(val_items, W_B, None, labels_of, seqs_of, per_file=True)
    test_b, _sum_bt = eval_items(test_items, W_B, thr_b, labels_of, seqs_of, per_file=False)
    test_c, sum_c_test = eval_items(test_items, W_B, None, labels_of, seqs_of, per_file=True)
    event(f"val B IoU {val_b['pooled']['iou']:.4f} порог {thr_b:.4f}; "
          f"val C IoU {val_c['pooled']['iou']:.4f}")
    for ver in ("v8", "v10"):
        if ver in test_b:
            event(f"test {ver} B IoU {test_b[ver]['iou']:.4f} F1 {test_b[ver]['f1']:.4f} "
                  f"FAR {test_b[ver]['far']:.5f}; C IoU {test_c[ver]['iou']:.4f} "
                  f"F1 {test_c[ver]['f1']:.4f} FAR {test_c[ver]['far']:.5f}")
    synth_blob = dict(
        done=True, method="realnet_bc",
        note="Карточки вариант не выбирают. Сравнение found/false на них описательное.",
        prior=dict(seg=W_B[0], mask=W_B[1], wide=W_B[2],
                   rule="fixed before cards: seg keeps half because val IoU selected it; mask 0.25 local residual; wide 0.25 harmonic baseline as an extra score, not a network input"),
        B=dict(weights=dict(seg=W_B[0], mask=W_B[1], wide=W_B[2]), threshold=thr_b,
               threshold_rule="pooled v8+v10 val background far 1e-3 of this mixture",
               val=val_b, test=test_b),
        C=dict(weights=dict(seg=W_B[0], mask=W_B[1], wide=W_B[2]),
               threshold_rule="per-file 1e-3 quantile of valid pixels, no event labels, no cards",
               val=val_c, val_threshold_summary=sum_c_val,
               test=test_c, test_threshold_summary=sum_c_test),
        mask_steps=steps, adapted_on=used, use_adabn=False,
        adapt_input="pretrained channels; mask and harmonic residual are extra scores",
        variant_A="artifacts/eval/realnet_synth.json",
    )
    rn.dump_json(synth_blob, "artifacts/eval/realnet_bc_synth.json")
    removed = 0
    if tmp.exists():
        for p in tmp.rglob("*.npz"):
            p.unlink()
            removed += 1
    event(f"synth B/C записан, временные карты удалены ({removed})")
    copy_variant_a()
    real_blob = run_real_bc(model, thr_b, a.real_root, a.cards)
    real_blob.update(dict(
        done=True, method="realnet_bc", threshold_B=thr_b,
        cards_are_descriptive_only=True,
        selection="A is the synth-val choice (1,0,0). B and C use the fixed prior. Cards were tallied after zones and did not choose a variant.",
        variant_A="artifacts/eval/realnet_real.json",
        reference_real_v8=rn.saved_card_tally("artifacts/eval/real_v8.json"),
        model_version=MODEL_VERSION,
    ))
    rn.dump_json(real_blob, "artifacts/eval/realnet_bc_real.json")
    event("REALNET_BC_DONE")


if __name__ == "__main__":
    main()
