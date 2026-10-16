"""Проверка GEN2 без четырёх файлов, по которым выбирался чекпойнт.

Первые 4 последовательности v8 val — прокси. Здесь val IoU и вложенные k=5/k=2
считаются на остальных 41. Порог holdout — FAR 1e-3 по их фону.
Бутстреп дельты — 1000 перестановок последовательностей, micro-IoU.
Test, v10 и v11 идут с порогом полного v8 val и в выбор чекпойнта не входили.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_k] = "2"

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)

from dts import data, metrics as M, models
from s3_gen2_train import BEST, predict_batched

WINNER = "ens_u2ft_tcn_lgbm"
WINNER_VAL = 0.38671235821914546
BOOT_N = 1000
BOOT_SEED = 20261016
OUT = Path("artifacts/eval/s3_gen2_check.json")
PREV = Path("artifacts/eval/s3_gen2_v8.json")
ENS = "ens_u2ft_tcn_lgbm"


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S", time.gmtime()) + " " + msg, flush=True)


def plain(obj):
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return plain(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def dump(obj) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(plain(obj), indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, OUT)


def blend_of(name, s, ws):
    if name == "unet_gen2":
        return s
    if name == "ens_blend":
        return (0.5 * (np.asarray(s) + ws)).astype(np.float32)
    if name == "ens_mean4":
        return ((4.0 * ws + np.asarray(s)) / 5.0).astype(np.float32)
    return ws


def bg_of(z):
    return np.asarray(z["valid"], bool) & (np.asarray(z["ev"]) == 0) & (np.asarray(z["nu"]) == 0)


def iou_of(tp, fp, fn):
    return float(tp) / max(float(tp) + float(fp) + float(fn), 1.0)


def pack(rows):
    tp = int(sum(r[0] for r in rows))
    fp = int(sum(r[1] for r in rows))
    fn = int(sum(r[2] for r in rows))
    return dict(tp=tp, fp=fp, fn=fn, iou=iou_of(tp, fp, fn), n_seq=len(rows))


def bootstrap(blend_rows, win_rows, n=BOOT_N, seed=BOOT_SEED):
    b = np.asarray(blend_rows, np.float64)
    w = np.asarray(win_rows, np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(b), size=(n, len(b)))
    sb = b[idx].sum(axis=1)
    sw = w[idx].sum(axis=1)
    db = sb[:, 0] / np.maximum(sb.sum(axis=1), 1.0)
    dw = sw[:, 0] / np.maximum(sw.sum(axis=1), 1.0)
    delta = db - dw
    lo, mid, hi = np.percentile(delta, [2.5, 50, 97.5])
    point = pack(blend_rows)["iou"] - pack(win_rows)["iou"]
    return dict(
        n=int(n), seed=int(seed), unit="sequence",
        point=float(point), mean=float(delta.mean()),
        p2_5=float(lo), p50=float(mid), p97_5=float(hi),
        frac_positive=float(np.mean(delta > 0)),
        covers_zero=bool(lo <= 0.0 <= hi),
    )


def predict_split(model, ver, split):
    files = data.list_cache(f"artifacts/cache/{ver}", split)
    ens = Path("artifacts/pred") / ver / ENS / split
    out = []
    t0 = time.time()
    for i, p in enumerate(files, 1):
        with np.load(p) as z:
            X = np.asarray(z["X"], np.float32)
            ev = np.asarray(z["ev"])
            valid = np.asarray(z["valid"], bool)
            nu = np.asarray(z["nu"])
            x = np.asarray(z["x"], np.float64) if ver == "v8" and split == "val" else None
            t = np.asarray(z["t"], np.float64) if x is not None else None
        prob = predict_batched(model, X)
        s = (1.0 - prob[0]).astype(np.float32)
        with np.load(ens / p.name) as pred:
            ws = np.asarray(pred["s"], np.float32)
        bg = valid & (ev == 0) & (nu == 0)
        out.append(dict(path=p, s=s, ws=ws, ev=ev, valid=valid, nu=nu, bg=bg, x=x, t=t))
        del X, prob
        if i == 1 or i % 15 == 0 or i == len(files):
            log(f"pred {ver} {split} {i}/{len(files)} {time.time() - t0:.0f}s")
    return out


def fit_thr(items, name, score_of):
    parts = [np.asarray(score_of(it)[it["bg"]], np.float32) for it in items]
    bg = np.concatenate(parts)
    thr = float(M.far_threshold(bg, 1e-3))
    log(f"порог {name} {thr:.6f} n_bg {bg.size}")
    del parts, bg
    return thr


def score_items(items, thr):
    rows = {n: [] for n in thr}
    for it in items:
        truth = it["ev"] > 0
        valid = it["valid"]
        s, ws = it["s"], it["ws"]
        maps = {
            "unet_gen2": s,
            "ens_blend": blend_of("ens_blend", s, ws),
            "ens_mean4": blend_of("ens_mean4", s, ws),
            WINNER: ws,
        }
        for n, arr in maps.items():
            rows[n].append(M.confusion((arr >= thr[n]) & valid, truth, valid))
    return {n: pack(rows[n]) for n in thr}, rows


def nested_holdout(items):
    import s3_sel2_cv as s2
    import s3_sel_core as sel
    import s3_sel_eval as ev
    import pandas as pd

    labels = pd.read_csv("data/synth_meta/v8/labels.csv")
    seqs = pd.read_csv("data/synth_meta/v8/sequences.csv").set_index("seq_id")
    buckets = {n: [] for n in ("ens_blend", "ens_mean4", WINNER)}
    for it in items:
        p = it["path"]
        ev_idx, nu_idx, bg_idx, n_bg = ev._indices(it["ev"], it["valid"], it["nu"], p.stem)
        cooling, _rows = ev._events_for(p.stem, it["ev"], it["x"], it["t"], labels, seqs, ev_idx)

        def frame(arr, stem=p.stem, cooling=cooling, n_bg=n_bg, ev_idx=ev_idx, nu_idx=nu_idx, bg_idx=bg_idx):
            e, n, b = s2._cols(arr, ev_idx, nu_idx, bg_idx)
            return dict(ev_m=e[None], nu_m=n[None], bg_m=b[None], n_bg=n_bg, cooling=cooling, stem=stem)

        s, ws = it["s"], it["ws"]
        buckets["ens_blend"].append(frame(blend_of("ens_blend", s, ws)))
        buckets["ens_mean4"].append(frame(blend_of("ens_mean4", s, ws)))
        buckets[WINNER].append(frame(ws))
    order = np.argsort([fr["stem"] for fr in buckets[WINNER]])
    frames = {n: [buckets[n][i] for i in order] for n in buckets}
    winner = frames[WINNER]
    out = {"stems": [fr["stem"] for fr in winner], "procedure": "s3_sel2_cv sequence_folds i%k on 41, holdout FAR 1e-3, micro"}
    for name in ("ens_blend", "ens_mean4"):
        out[name] = {}
        for k in (2, 5):
            folds = sel.sequence_folds(len(winner), k)
            parts_w, parts_b, rows = [], [], []
            for r, idx in enumerate(folds):
                hw = s2.eval_one(s2.take(winner, idx))
                hb = s2.eval_one(s2.take(frames[name], idx))
                parts_w.append(hw)
                parts_b.append(hb)
                rows.append(dict(
                    fold=r, n_seq=len(idx),
                    winner=float(hw["iou"]), selected=float(hb["iou"]),
                    delta=float(hb["iou"] - hw["iou"]),
                ))
            iw = float(sel.micro_iou(parts_w))
            ib = float(sel.micro_iou(parts_b))
            out[name][str(k)] = dict(
                winner_iou=iw, selected_iou=ib, delta=float(ib - iw), folds=rows,
            )
            log(f"nested41 {name} k={k} {ib:.4f} winner {iw:.4f} delta {ib - iw:+.4f}")
    return out


def drop_heavy(items):
    for it in items:
        it.pop("s", None)
        it.pop("ws", None)
        it.pop("ev", None)
        it.pop("valid", None)
        it.pop("nu", None)
        it.pop("bg", None)
        it.pop("x", None)
        it.pop("t", None)


def main():
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    prev = json.loads(PREV.read_text())
    frozen = {n: float(prev["val"][n]["threshold"]) for n in ("unet_gen2", "ens_blend", "ens_mean4", WINNER)}
    published = {
        "val": float(prev["val"]["ens_blend"]["iou"]),
        "test": float(prev["test"]["ens_blend"]["iou"]),
        "v10": float(prev["v10_test"]["ens_blend"]["iou"]),
        "v11": float(prev["v11_test"]["ens_blend"]["iou"]),
    }
    files = data.list_cache("artifacts/cache/v8", "val")
    if len(files) != 45:
        raise SystemExit(f"v8 val {len(files)} != 45")
    proxy = files[:4]
    log("proxy " + " ".join(p.stem for p in proxy))
    model = models.build("unet", 16)
    model.load_state_dict(torch.load(BEST, map_location="cpu", weights_only=True))
    model.eval()
    t0 = time.time()
    val_items = predict_split(model, "v8", "val")
    by_stem = {it["path"].stem: it for it in val_items}
    hold = [by_stem[p.stem] for p in files[4:]]
    proxy_items = [by_stem[p.stem] for p in proxy]
    if len(hold) != 41:
        raise SystemExit(f"holdout {len(hold)}")
    # Контроль протокола: полный val победителя при замороженном пороге.
    full_rows = []
    for it in val_items:
        full_rows.append(M.confusion((it["ws"] >= frozen[WINNER]) & it["valid"], it["ev"] > 0, it["valid"]))
    win_full = pack(full_rows)["iou"]
    if abs(win_full - WINNER_VAL) > 0.005:
        raise SystemExit(f"победитель на 45 файлах {win_full} != {WINNER_VAL}")
    log(f"winner full val {win_full:.6f} match")
    hold_thr = {}
    for n, fn in (
        ("unet_gen2", lambda it: it["s"]),
        ("ens_blend", lambda it: blend_of("ens_blend", it["s"], it["ws"])),
        ("ens_mean4", lambda it: blend_of("ens_mean4", it["s"], it["ws"])),
        (WINNER, lambda it: it["ws"]),
    ):
        hold_thr[n] = fit_thr(hold, n + " hold41", fn)
    hold_refit, _ = score_items(hold, hold_thr)
    hold_frozen, rows_frozen = score_items(hold, frozen)
    proxy_frozen, _ = score_items(proxy_items, frozen)
    log(f"hold41 refit blend {hold_refit['ens_blend']['iou']:.4f} winner {hold_refit[WINNER]['iou']:.4f}")
    log(f"hold41 frozen blend {hold_frozen['ens_blend']['iou']:.4f} winner {hold_frozen[WINNER]['iou']:.4f}")
    nest = nested_holdout(hold)
    # Бутстреп val — 41 файл. Frozen: порог полного v8 val. Refit: порог только по этим 41.
    b_refit = [M.confusion(
        (blend_of("ens_blend", it["s"], it["ws"]) >= hold_thr["ens_blend"]) & it["valid"],
        it["ev"] > 0, it["valid"]) for it in hold]
    w_refit = [M.confusion(
        (it["ws"] >= hold_thr[WINNER]) & it["valid"], it["ev"] > 0, it["valid"]) for it in hold]
    b_fr = rows_frozen["ens_blend"]
    w_fr = rows_frozen[WINNER]
    boot = {
        "val_hold41_refit": bootstrap(b_refit, w_refit),
        "val_hold41_frozen_thr": bootstrap(b_fr, w_fr),
    }
    drop_heavy(val_items)
    del hold, proxy_items, by_stem
    gc.collect()
    for ver, split, key in (("v8", "test", "test"), ("v10", "test", "v10"), ("v11", "test", "v11")):
        items = predict_split(model, ver, split)
        summary, rows = score_items(items, frozen)
        boot[key] = bootstrap(rows["ens_blend"], rows[WINNER])
        boot[key]["iou"] = {n: summary[n]["iou"] for n in summary}
        log(f"{key} blend {summary['ens_blend']['iou']:.4f} winner {summary[WINNER]['iou']:.4f} "
            f"delta {boot[key]['point']:+.4f} CI [{boot[key]['p2_5']:+.4f}, {boot[key]['p97_5']:+.4f}]")
        drop_heavy(items)
        del items, rows
        gc.collect()
    report = dict(
        name="GEN2_CHECK",
        proxy_files=[p.stem for p in proxy],
        holdout_n=41,
        proxy_note="Чекпойнт выбран по прокси IoU ens_blend на этих 4 файлах. Они исключены из val и из вложенных фолдов.",
        winner_full_val_iou=win_full,
        winner_match=abs(win_full - WINNER_VAL) < 1e-9,
        frozen_thresholds=frozen,
        holdout_thresholds=hold_thr,
        published_ens_blend=published,
        holdout_refit=hold_refit,
        holdout_at_frozen_threshold=hold_frozen,
        proxy_files_at_frozen_threshold=proxy_frozen,
        nested41=nest,
        bootstrap=boot,
        bootstrap_definition="1000 resamples of sequences with replacement, micro IoU delta ens_blend - winner, percentile 2.5 and 97.5",
        real_used=False,
        seconds=round(time.time() - t0, 1),
    )
    dump(report)
    log(f"записан {OUT}")


if __name__ == "__main__":
    main()
