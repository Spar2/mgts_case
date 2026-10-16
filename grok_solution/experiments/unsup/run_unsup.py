"""Leave-one-file-out слепой прогнозист на MGTS 1–6.csv.

Каждый файл оценивает модель, не видевшая его. Порог — квантиль скор-карты
этого файла (доля --far), без карточек и без синтетических меток.
Карточки оператора читаются только в конце, для подсчёта совпадений.
"""
import argparse
import json
import os
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

import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dts import domain, export, features, grid, metrics, real
from experiments.unsup.unsup import (
    BLIND_T, BLIND_X, BOTTLENECK, assemble_zones, describe_zone, save_model,
    score_field, train_forecaster,
)

import torch


def event(msg: str) -> None:
    line = time.strftime("%H:%M", time.gmtime()) + " unsup: " + msg
    try:
        with open("/workspace/grok_logs/_events.log", "a") as f:
            f.write(line + "\n")
    except OSError:
        pass
    print(line, flush=True)


def load_cards(paths):
    cards = []
    for p in paths:
        with open(p) as f:
            cards += [json.loads(l) for l in f if l.strip()]
    return cards


def prepare_record(path, cache_dir: Path):
    path = Path(path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"prepare {path.name}", flush=True)
    # sha до разбора json, чтобы повторный запуск не парсил CSV
    digest = real.sha256(path)
    cache = cache_dir / f"{path.stem}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if str(z["sha256"].item()) == digest:
            print(f"cache {cache.name} {tuple(z['X'].shape)}", flush=True)
            return dict(name=str(z["name"].item()), sha256=digest, tz_missing=bool(z["tz_missing"].item()),
                        x=z["x"], t=z["t"], X=z["X"].astype(np.float32), dT=z["dT"].astype(np.float32),
                        valid=z["valid"].astype(bool), sigma=z["sigma"].astype(np.float32))
    print(f"parse {path.name}", flush=True)
    r = real.load_csv(path)
    print(f"grid {path.name} raw {r['T'].shape}", flush=True)
    p = grid.plan(r["x"], r["t"])
    Tg, vg = grid.resample(r["T"], r["valid"], p, "mean")
    Tg = np.asarray(Tg, np.float32)
    vg = np.asarray(vg, bool) & np.isfinite(Tg)
    print(f"channels {path.name} grid {Tg.shape}", flush=True)
    ch, meta = features.make_channels(Tg, vg)
    out = dict(name=r["name"], sha256=r["sha256"], tz_missing=bool(r["tz_missing"]),
               x=np.asarray(p["x"], np.float64), t=np.asarray(p["t"], np.float64),
               X=ch.astype(np.float32), dT=np.asarray(meta["dT_time"], np.float32),
               valid=vg, sigma=np.asarray(meta["sigma_x"], np.float32))
    np.savez(cache, X=out["X"], dT=out["dT"], valid=out["valid"].astype(np.uint8), x=out["x"], t=out["t"],
             sigma=out["sigma"], sha256=np.array(out["sha256"]), tz_missing=np.array(out["tz_missing"]),
             name=np.array(out["name"]))
    print(f"prepared {out['name']} grid {out['X'].shape} valid {out['valid'].mean():.3f}", flush=True)
    return out


def save_figure(path, rec, score, zones, thr):
    x, t, valid, dT = rec["x"], rec["t"], rec["valid"], rec["dT"]
    fig, ax = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    hours = (t - t[0]) / 3600.0
    ext = [x[0], x[-1], hours[-1] if len(hours) else 0, 0]
    show = np.where(valid, dT, np.nan)
    v = np.nanpercentile(np.abs(show), 99.5) if np.isfinite(show).any() else 1.0
    v = max(float(v), 0.2)
    ax[0].imshow(show, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v, extent=ext, interpolation="nearest")
    ax[0].set_title(f"{rec['name']}.csv: ΔT без синфазной и статической, °C")
    zshow = np.where(valid, score, np.nan)
    vmax = max(float(thr) * 1.5, float(np.nanpercentile(zshow, 99.5)) if np.isfinite(zshow).any() else 1.0)
    ax[1].imshow(zshow, aspect="auto", cmap="magma", vmin=0, vmax=vmax, extent=ext, interpolation="nearest")
    ax[1].set_title(f"робастный |остаток| z, порог {thr:.2f} (квантиль {1e-3:g} файла)")
    for z in zones:
        dur_h = (z["t_end_s"] - z["t_start_s"]) / 3600.0
        width = max(z["end_m"] - z["start_m"], 0.5)
        y0 = (z["t_start_s"] - t[0]) / 3600.0
        color = "#c0392b" if z["polarity"] == "hot" else "#2471a3"
        for a in ax:
            a.add_patch(Rectangle((z["start_m"], y0), width, max(dur_h, 0.02), fill=False, ec=color, lw=0.8))
    ax[0].set_ylabel("время от начала, ч")
    ax[0].set_xlabel("x, м")
    ax[1].set_xlabel("x, м")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def card_rows_for(rec, score, zones, thr, cards):
    rows = []
    for c in cards:
        if str(c.get("record")) != str(rec["name"]):
            continue
        row = domain.score_card(c, rec["x"], rec["t"], rec["valid"], score, zones, thr, [])
        row["shape"] = c.get("shape", "")
        row["notes"] = c.get("notes", "")
        rows.append(row)
    return rows


def supervised_card_replay(prelim_path, cards):
    """Совпадения ансамбля по уже сохранённым зонам real_v8_prelim.json. Обучение не трогаем."""
    if not Path(prelim_path).exists():
        return None
    blob = json.load(open(prelim_path))
    rows = []
    for c in cards:
        rec = str(c.get("record"))
        f = blob.get("files", {}).get(rec)
        if f is None:
            rows.append(dict(id=c.get("id"), record=rec, presence=c.get("presence"), status="missing_file"))
            continue
        xa, xb, ta, tb = domain.card_bounds(c)
        zones = f.get("zones", [])
        hit = bool(domain.overlapping_zones(zones, xa, xb, ta, tb, True))
        rows.append(dict(id=c.get("id"), record=rec, presence=c.get("presence"), status="ok", hit_alarm=hit,
                         cnn_n_alarm_zones=sum(1 for z in zones if z.get("alarm", True)
                                               and z["end_m"] >= xa and z["start_m"] <= xb
                                               and z["t_end_s"] >= ta and z["t_start_s"] <= tb)))
    tally = domain.presence_tally(rows, "hit_alarm")
    # как записано в самом json (cnn_n_alarm_zones карточки), если ключ есть
    saved = blob.get("cards") or []
    if saved:
        yes = [c for c in saved if c.get("presence") == "yes"]
        no = [c for c in saved if c.get("presence") == "no"]
        saved_tally = dict(yes_alarm=sum(1 for c in yes if c.get("cnn_n_alarm_zones")), yes_n=len(yes),
                           no_alarm=sum(1 for c in no if c.get("cnn_n_alarm_zones")), no_n=len(no),
                           yes_base=sum(1 for c in yes if c.get("base_n_zones")),
                           no_base=sum(1 for c in no if c.get("base_n_zones")))
    else:
        saved_tally = None
    return dict(replay=tally, saved_cards=saved_tally, threshold=blob.get("threshold"), op=blob.get("op"))


def reference_synth_iou():
    path = Path("artifacts/eval/stage1_v8_round4.json")
    if not path.exists():
        return None
    blob = json.load(open(path))
    out = {}
    for key, name in (("unet@far_1e-3", "unet"), ("ens_unet_tcn_lgbm@far_1e-3", "ens_unet_tcn_lgbm"),
                      ("ens_cnn3_lgbm@far_1e-3", "ens_cnn3_lgbm"),
                      ("base_median_x3_spatial_z@far_1e-3", "baseline_far_1e-3")):
        block = blob.get("test", {}).get(key)
        if block:
            out[name] = dict(iou=block.get("iou"), f1=block.get("f1"), far=block.get("far"))
    # SSL-AE из раунда 2, если лежит рядом
    r2 = Path("artifacts/eval/stage1_v8_round2.json")
    if r2.exists():
        b2 = json.load(open(r2))
        block = b2.get("test", {}).get("ssl_ae@far_1e-3")
        if block:
            out["ssl_ae"] = dict(iou=block.get("iou"), f1=block.get("f1"), far=block.get("far"))
    return out


def run_real(a, cards):
    root = Path(a.real_root)
    files = sorted(root.glob("*.csv"))
    hold = {s.strip() for s in a.holdouts.split(",") if s.strip()}
    files = [p for p in files if p.stem in hold] if hold else files
    if not files:
        raise SystemExit(f"no csv in {root} for holdouts {a.holdouts}")
    cache = Path(a.out) / "fields"
    recs = [prepare_record(p, cache) for p in files]
    by = {r["name"]: r for r in recs}
    names = [r["name"] for r in recs]
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(a.fig)
    all_rows = []
    report_files = {}
    card_rows = []
    for name in names:
        others = [by[n]["X"] for n in names if n != name]
        if not others:
            raise SystemExit("LOFO needs at least two files")
        event(f"train holdout {name} on {len(others)} files, ≤{a.max_minutes} min, threads {a.threads}")
        t0 = time.time()
        model, hist = train_forecaster(
            others, minutes=a.max_minutes, seed=a.seed + int(name) if str(name).isdigit() else a.seed,
            width=a.width, bottleneck=a.bottleneck, blind_t=a.blind_t, blind_x=a.blind_x,
            batch=a.batch, tile_t=a.tile_t, tile_x=a.tile_x, lr=a.lr, tag=f"[{name}] ")
        train_sec = time.time() - t0
        ckpt = out_dir / f"model_holdout_{name}.pt"
        save_model(model, ckpt, holdout=name, train_sec=train_sec)
        rec = by[name]
        t1 = time.time()
        score, hot = score_field(model, rec["X"])
        score_sec = time.time() - t1
        thr = domain.flag_threshold(score, rec["valid"], a.far)
        zones = assemble_zones(score, hot, rec["valid"], rec["x"], rec["t"], rec["dT"], thr)
        zones_path = out_dir / f"zones_{name}.csv"
        n_csv = export.write_csv(zones_path, export.zone_rows(zones, f"{name}.csv", rec["sha256"], model_version=a.model_version))
        all_rows += list(export.zone_rows(zones, f"{name}.csv", rec["sha256"], model_version=a.model_version))
        fig = fig_dir / f"unsup_{name}.png"
        save_figure(fig, rec, score, zones, thr)
        descs = [describe_zone(z, score, rec["valid"], rec["x"], rec["t"], thr) for z in zones]
        descs_sorted = sorted(descs, key=lambda d: -d["score"])
        top = descs_sorted[:a.top]
        flagged = float((score[rec["valid"]] >= thr).mean()) if rec["valid"].any() else 0.0
        hours = float((rec["t"][-1] - rec["t"][0]) / 3600.0)
        occupied = rec["valid"].any(0)
        km = float(np.ptp(rec["x"][occupied]) / 1000.0) if occupied.any() else 0.0
        final = [h for h in hist if h.get("final")]
        report_files[name] = dict(
            sha256=rec["sha256"], tz_missing=rec["tz_missing"], grid_shape=list(rec["valid"].shape),
            hours=round(hours, 2), km=round(km, 3), threshold=float(thr), far=a.far,
            alarm_fraction=flagged, n_zones=len(zones), train_sec=round(train_sec, 1),
            score_sec=round(score_sec, 1), checkpoint=str(ckpt), zones_csv=str(zones_path),
            figure=str(fig), n_csv_rows=n_csv, probe_final=final[-1] if final else None,
            zones_top=top, n_hot=sum(z["polarity"] == "hot" for z in zones),
            n_cold=sum(z["polarity"] == "cold" for z in zones))
        card_rows += card_rows_for(rec, score, zones, thr, cards)
        event(f"holdout {name}: zones {len(zones)} thr {thr:.3f} flag {flagged:.4f} train {train_sec:.0f}s")
        # скор-карта больше не нужна
        del score, hot, model
    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", newline="") as f:
        cols = ["file", "hours", "km", "threshold", "alarm_fraction", "n_zones", "n_hot", "n_cold", "train_sec"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for name, info in report_files.items():
            w.writerow(dict(file=name, hours=info["hours"], km=info["km"], threshold=round(info["threshold"], 4),
                            alarm_fraction=round(info["alarm_fraction"], 6), n_zones=info["n_zones"],
                            n_hot=info["n_hot"], n_cold=info["n_cold"], train_sec=info["train_sec"]))
    export.write_csv(out_dir / "zones_all.csv", all_rows)
    presence = domain.presence_tally(card_rows, "hit_alarm")
    return dict(files=report_files, cards=card_rows, presence=presence, summary_csv=str(summary_path), names=names)


def _synth_paths(cache, split, limit, seed):
    paths = sorted(Path(cache).glob(f"{split}/seq*.npz"))
    if limit and len(paths) > limit:
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(paths), size=limit, replace=False)
        paths = [paths[i] for i in sorted(pick)]
    return paths


def _load_X(path):
    z = np.load(path)
    return np.array(z["X"], dtype=np.float32)


def eval_synth(a):
    cache = Path(a.synth_cache)
    train_paths = _synth_paths(cache, "train", a.synth_train_files, a.seed)
    if not train_paths:
        raise SystemExit(f"no train npz in {cache}")
    event(f"synth train on {len(train_paths)} v8 train files, no labels, ≤{a.synth_minutes} min")
    fields = [_load_X(p) for p in train_paths]
    t0 = time.time()
    model, hist = train_forecaster(
        fields, minutes=a.synth_minutes, seed=a.seed + 17, width=a.width, bottleneck=a.bottleneck,
        blind_t=a.blind_t, blind_x=a.blind_x, batch=a.batch, tile_t=a.tile_t, tile_x=a.tile_x,
        lr=a.lr, tag="[synth] ")
    train_sec = time.time() - t0
    save_model(model, Path(a.out) / "model_synth_v8.pt", train_files=[p.stem for p in train_paths], train_sec=train_sec)
    del fields
    return _eval_synth_passes(model, cache, a, train_sec, hist, len(train_paths))


def _eval_synth_passes(model, cache, a, train_sec, hist, n_train):
    rng = np.random.default_rng(a.seed + 3)
    hold = np.zeros(a.reservoir, np.float32)
    n_seen = 0
    for i, p in enumerate(sorted(Path(cache).glob("val/seq*.npz"))):
        z = np.load(p)
        X = np.array(z["X"], dtype=np.float32)
        valid = np.asarray(z["valid"], bool) & (z["ev"] == 0) & (z["nu"] == 0)
        score, _h = score_field(model, X)
        n_seen = domain.reservoir_update(hold, n_seen, score[valid], rng)
        if (i + 1) % 10 == 0:
            print(f"synth val {i+1}", flush=True)
    thr = metrics.far_threshold(hold[:n_seen], a.far) if n_seen else 1e9
    tot = np.zeros(3)
    tot_local = np.zeros(3)
    far_n = far_d = 0
    per = {}
    for i, p in enumerate(sorted(Path(cache).glob("test/seq*.npz"))):
        z = np.load(p)
        X = np.array(z["X"], dtype=np.float32)
        ev = z["ev"]
        nu = z["nu"]
        valid = np.asarray(z["valid"], bool)
        score, _h = score_field(model, X)
        pred = (score >= thr) & valid
        c = metrics.confusion(pred, ev > 0, valid)
        tot += c
        per[p.stem] = c
        bg = valid & (ev == 0) & (nu == 0)
        far_n += int((pred & bg).sum())
        far_d += int(bg.sum())
        thr_l = domain.flag_threshold(score, valid, a.far)
        tot_local += metrics.confusion((score >= thr_l) & valid, ev > 0, valid)
        if (i + 1) % 10 == 0:
            print(f"synth test {i+1}", flush=True)
    final = [h for h in hist if h.get("final")]
    return dict(
        train_files=n_train, train_sec=round(train_sec, 1), threshold_val_bg=float(thr),
        val_bg_samples=int(n_seen),
        test_iou_at_val_far=dict(**metrics.prf(*tot), far=float(far_n / max(far_d, 1)),
                                 iou_ci95=metrics.bootstrap_ci(per, sorted(per))),
        test_iou_per_file_quantile=metrics.prf(*tot_local),
        probe_final=final[-1] if final else None,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--real-root", default="/workspace/repo/src/MGTS_data")
    ap.add_argument("--cards", nargs="*", default=["data/operator_cards/cards_batch1.jsonl", "data/operator_cards/cards_batch2.jsonl"])
    ap.add_argument("--out", default="artifacts/unsup")
    ap.add_argument("--eval", default="artifacts/eval/unsup_real.json")
    ap.add_argument("--fig", default="docs/figures")
    ap.add_argument("--prelim", default="artifacts/eval/real_v8_prelim.json")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--max-minutes", type=float, default=8.0)
    ap.add_argument("--far", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=8)
    ap.add_argument("--bottleneck", type=int, default=BOTTLENECK)
    ap.add_argument("--blind-t", type=int, default=BLIND_T)
    ap.add_argument("--blind-x", type=int, default=BLIND_X)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--tile-t", type=int, default=80)
    ap.add_argument("--tile-x", type=int, default=160)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--holdouts", default="1,2,3,4,5,6")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--model-version", default="unsup-blindspot-0.1")
    ap.add_argument("--skip-real", action="store_true")
    ap.add_argument("--synth", action="store_true")
    ap.add_argument("--synth-cache", default="artifacts/cache/v8")
    ap.add_argument("--synth-minutes", type=float, default=8.0)
    ap.add_argument("--synth-train-files", type=int, default=16)
    ap.add_argument("--reservoir", type=int, default=1_000_000)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    event(f"start LOFO blind {a.blind_t}x{a.blind_x} bottleneck {a.bottleneck} threads {a.threads}")
    cards = load_cards(a.cards) if a.cards else []
    report = dict(
        method="blindspot_forecaster", blind_t=a.blind_t, blind_x=a.blind_x, width=a.width,
        bottleneck=a.bottleneck, far=a.far, threads=a.threads, max_minutes_per_model=a.max_minutes,
        threshold_rule="квантиль робастного |остатка| по валидным клеткам своего файла",
        target="канал 0 make_channels (ΔT_time/σ), остаток obs−pred",
        blind="центр ±blind_x//2 не читается ни в какой момент; будущее не читается; прошлое соседей — с лагом blind_t",
        cards_used_for="evaluation_only",
    )
    if not a.skip_real:
        real_part = run_real(a, cards)
        report.update(files=real_part["files"], cards=real_part["cards"], presence=real_part["presence"],
                      summary_csv=real_part["summary_csv"])
        report["supervised_real"] = supervised_card_replay(a.prelim, cards)
        report["supervised_synth_published"] = reference_synth_iou()
        Path(a.eval).parent.mkdir(parents=True, exist_ok=True)
        json.dump(report, open(a.eval, "w"), indent=1, ensure_ascii=False)
        event(f"real eval written {a.eval} presence {report['presence']}")
    if a.synth:
        try:
            report["synth_v8"] = eval_synth(a)
            json.dump(report, open(a.eval, "w"), indent=1, ensure_ascii=False)
            event(f"synth IoU {report['synth_v8']['test_iou_at_val_far']}")
        except Exception as exc:
            report["synth_error"] = str(exc)
            json.dump(report, open(a.eval, "w"), indent=1, ensure_ascii=False)
            event(f"synth failed: {exc}")
            raise
    print("UNSUP_DONE", a.eval, flush=True)


if __name__ == "__main__":
    main()
