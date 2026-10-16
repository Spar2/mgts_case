"""pixel_lgbm на признаках с DOM-шумом и ансамбль с DOM U-Net.

Аугментация только на train: dts.aug_real.apply_tile, спектр фона из
artifacts/real_noise_psd.npz. Реальные CSV не читаются. Порог FAR 1e-3
по фону v8 val. Выбор — максимум val IoU среди рецептов с новым LGBM.
Test порог не двигает. artifacts/stage1/pixel_lgbm.joblib не перезаписывается.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

_thr = "2"
if "--threads" in sys.argv:
    _thr = sys.argv[sys.argv.index("--threads") + 1]
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = _thr

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dts import aug_real, classic, data, metrics as M, models

WINNER_VAL = 0.38671235821914546
WINNER_TEST = 0.4052762065230473
EVENT_LOG = Path(os.environ.get("DOMLGBM_EVENT_LOG", "/workspace/run_logs/_events.log"))
TMP = Path("/tmp/s3_domlgbm_maps")
MEMBERS = ("unet_w24", "unet_v810ft", "tcn", "pixel_lgbm", "ens_u2ft_tcn_lgbm")
# В выбор входят только рецепты с новым LGBM. winner и blend_pub — контроль.
SELECT = ("lgbm_dom", "ens_lgbm", "ens_dom4", "ens_plus", "ens_blend")


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S', time.gmtime())} {msg}", flush=True)


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


def subsample(y, rng, n_pos=1500, n_neg=1500):
    pos = np.flatnonzero(y > 0)
    neg = np.flatnonzero(y == 0)
    parts = []
    if pos.size:
        parts.append(rng.choice(pos, min(pos.size, n_pos), replace=False))
    if neg.size:
        parts.append(rng.choice(neg, min(neg.size, n_neg), replace=False))
    if not parts:
        return np.zeros(0, np.int64)
    return np.concatenate(parts).astype(np.int64)


def train_lgbm(threads: int, seed: int):
    bank = aug_real.load_bank()
    shaper, fit = aug_real.fit_shaper(bank, steps=12, seed=seed, tile=(160, 256), max_lag=16)
    spec = aug_real.AugSpec(std_lo=0.25, std_hi=0.70, p_apply=1.0, sign_p=0.0)
    log(f"шейпер accepted_fit={fit.get('accepted_fit')} corr={fit.get('best_acf_corr')}")
    rng = np.random.default_rng(seed)
    aug_rng = np.random.default_rng(seed + 23)
    xs, ys = [], []
    files = data.list_cache("artifacts/cache/v8", "train")
    t0 = time.time()
    for i, p in enumerate(files, 1):
        with np.load(p) as z:
            X = np.asarray(z["X"], np.float32)
            sig = np.asarray(z["sigma"], np.float32)
            y = data.target(z["ev"], z["valid"]).ravel()
        sel = subsample(y, rng)
        if sel.size == 0:
            continue
        clean = classic.pixel_features(X)
        aug, _ = aug_real.apply_tile(X, np.zeros(X.shape[1:], np.int8), sig, aug_rng, shaper, spec)
        noisy = classic.pixel_features(aug)
        xs.append(clean[sel])
        xs.append(noisy[sel])
        ys.append(y[sel])
        ys.append(y[sel])
        del X, aug, clean, noisy
        if i == 1 or i % 30 == 0 or i == len(files):
            log(f"train features {i}/{len(files)} {time.time() - t0:.0f}s")
    Xtr = np.concatenate(xs)
    ytr = np.concatenate(ys).astype(np.int32)
    del xs, ys
    log(f"fit rows {len(ytr)} pos {(ytr > 0).sum()} neg {(ytr == 0).sum()}")
    import lightgbm as lgb
    gbm = lgb.LGBMClassifier(
        n_estimators=150, num_leaves=31, learning_rate=0.1,
        random_state=seed, verbose=-1, n_jobs=threads)
    gbm.fit(Xtr, ytr)
    del Xtr, ytr
    return gbm, fit, spec.as_dict()


def predict_lgbm(gbm, X):
    pr = gbm.predict_proba(classic.pixel_features(X)).reshape(X.shape[1], X.shape[2], 3)
    # Столбцы predict_proba идут в порядке classes_, не обязательно 0,1,2.
    full = np.zeros_like(pr)
    for i, c in enumerate(gbm.classes_):
        full[..., int(c)] = pr[..., i]
    return (1.0 - full[..., 0]).astype(np.float32), (full[..., 1] - full[..., 2]).astype(np.float32)


def predict_dom(model, X):
    pr = models.predict_full(model, X)
    return (1.0 - pr[0]).astype(np.float32), (pr[1] - pr[2]).astype(np.float32)


def save_pair(split, stem, tag, s, h):
    d = TMP / split
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"{stem}_{tag}_s.npy", s)
    np.save(d / f"{stem}_{tag}_h.npy", h.astype(np.float16))


def load_pair(split, stem, tag):
    d = TMP / split
    s = np.load(d / f"{stem}_{tag}_s.npy")
    h = np.load(d / f"{stem}_{tag}_h.npy").astype(np.float32)
    return s, h


def load_member(split, stem, name):
    with np.load(Path("artifacts/pred/v8") / name / split / f"{stem}.npz") as z:
        return np.asarray(z["s"], np.float32), np.asarray(z["h"], np.float32)


def combine(split, stem):
    w24 = load_member(split, stem, "unet_w24")
    v8 = load_member(split, stem, "unet_v810ft")
    tcn = load_member(split, stem, "tcn")
    old = load_member(split, stem, "pixel_lgbm")
    ens = load_member(split, stem, "ens_u2ft_tcn_lgbm")
    lgb = load_pair(split, stem, "lgbm")
    dom = load_pair(split, stem, "dom")

    def avg(pairs):
        s = sum(p[0] for p in pairs) / float(len(pairs))
        h = sum(p[1] for p in pairs) / float(len(pairs))
        return s.astype(np.float32), h.astype(np.float32)

    ens_lgbm = avg([w24, v8, tcn, lgb])
    return {
        "winner": ens,
        "lgbm_dom": lgb,
        "pixel_lgbm": old,
        "unet_dom": dom,
        "ens_lgbm": ens_lgbm,
        "ens_dom4": avg([w24, dom, tcn, lgb]),
        "ens_plus": avg([w24, v8, dom, tcn, lgb]),
        "ens_blend": avg([ens_lgbm, dom]),
        "ens_blend_pub": avg([ens, dom]),
    }


def dump_split(gbm, model, split: str):
    t0 = time.time()
    files = data.list_cache("artifacts/cache/v8", split)
    for i, p in enumerate(files, 1):
        with np.load(p) as z:
            X = np.asarray(z["X"], np.float32)
        ls, lh = predict_lgbm(gbm, X)
        ds, dh = predict_dom(model, X)
        save_pair(split, p.stem, "lgbm", ls, lh)
        save_pair(split, p.stem, "dom", ds, dh)
        del X, ls, lh, ds, dh
        if i == 1 or i % 10 == 0 or i == len(files):
            log(f"pred {split} {i}/{len(files)} {time.time() - t0:.0f}s")


def far_of(split: str, name: str) -> tuple[float, int]:
    parts = []
    for p in data.list_cache("artifacts/cache/v8", split):
        with np.load(p) as z:
            bg = np.asarray(z["valid"], bool) & (np.asarray(z["ev"]) == 0) & (np.asarray(z["nu"]) == 0)
        s, _ = combine(split, p.stem)[name]
        parts.append(np.ascontiguousarray(s[bg]))
        del s
    bg = np.concatenate(parts)
    return M.far_threshold(bg, 1e-3), int(bg.size)


def agg_events(rows):
    if not rows:
        return {"n": 0, "recall": None, "pixel_recall": None}
    tp = sum(r[1] for r in rows)
    fn = sum(r[2] for r in rows)
    return {
        "n": int(len(rows)),
        "recall": float(sum(r[0] for r in rows) / len(rows)),
        "pixel_recall": float(tp / max(tp + fn, 1)),
    }


def evaluate(split: str, names, thr: dict, labels: pd.DataFrame) -> dict:
    tot = {n: np.zeros(3, np.int64) for n in names}
    hot = {n: np.zeros(3, np.int64) for n in names}
    cold = {n: np.zeros(3, np.int64) for n in names}
    far_n = {n: 0 for n in names}
    far_d = {n: 0 for n in names}
    per = {n: {} for n in names}
    bins = {n: {"<=2C": [], "2-5C": [], ">5C": []} for n in names}
    for p in data.list_cache("artifacts/cache/v8", split):
        with np.load(p) as z:
            ev = np.asarray(z["ev"])
            valid = np.asarray(z["valid"], bool)
            nu = np.asarray(z["nu"])
            x = np.asarray(z["x"])
            t = np.asarray(z["t"])
        bg = valid & (ev == 0) & (nu == 0)
        truth = ev > 0
        hot_t = (ev >= 1) & (ev <= 4)
        cold_t = ev >= 5
        packed = combine(split, p.stem)
        events = []
        boxes = M.label_boxes(labels, p.stem)
        for r in boxes.itertuples():
            if r.kind != "event":
                continue
            px = M.event_pixels(r, x, t, ev)
            n = int(px.sum())
            if n == 0:
                continue
            events.append((M.amp_bin(float(r.amp_obs_C)), px, n))
        for n in names:
            s, h = packed[n]
            pred = (s >= thr[n]) & valid
            tot[n] += np.asarray(M.confusion(pred, truth, valid), np.int64)
            per[n][p.stem] = M.confusion(pred, truth, valid)
            hot[n] += np.asarray(M.confusion(pred & (h > 0), hot_t, valid), np.int64)
            cold[n] += np.asarray(M.confusion(pred & (h <= 0), cold_t, valid), np.int64)
            far_n[n] += int((pred & bg).sum())
            far_d[n] += int(bg.sum())
            pr = pred
            for amp, px, npx in events:
                etp = int((pr & px).sum())
                bins[n][amp].append((etp / npx >= 0.3, etp, npx - etp))
        del packed
    report = {}
    for n in names:
        row = {
            "threshold": float(thr[n]),
            "far": float(far_n[n] / max(far_d[n], 1)),
            "n_bg": int(far_d[n]),
            **M.prf(*tot[n]),
            "hot": M.prf(*hot[n]),
            "cold": M.prf(*cold[n]),
            "by_amp": {k: agg_events(v) for k, v in bins[n].items()},
        }
        weak = row["by_amp"]["<=2C"]
        row["weak_n"] = weak["n"]
        row["weak_recall"] = weak["recall"]
        row["weak_pixel_recall"] = weak["pixel_recall"]
        row["iou_ci95"] = M.bootstrap_ci(per[n], sorted(per[n]), n=200, seed=0)
        report[n] = row
        wr = weak["recall"]
        wr_s = "None" if wr is None else f"{wr:.4f}"
        log(f"{split} {n} IoU {row['iou']:.4f} F1 {row['f1']:.4f} FAR {row['far']:.5f} weak {wr_s}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts/eval/s3_domlgbm_v8.json")
    ap.add_argument("--skip-test", action="store_true")
    a = ap.parse_args()
    if a.threads > 2:
        raise SystemExit("threads must be <= 2")
    torch.set_num_threads(a.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True)
    t0 = time.time()
    try:
        gbm, fit, spec = train_lgbm(a.threads, a.seed)
        model = models.DTSUNet(w=16)
        model.load_state_dict(torch.load("artifacts/eval/s3_dom_unet.pt", map_location="cpu", weights_only=True))
        model.eval()
        dump_split(gbm, model, "val")
        labels = pd.read_csv("data/synth_meta/v8/labels.csv")
        names = list(SELECT) + ["winner", "ens_blend_pub", "pixel_lgbm", "unet_dom"]
        thr = {}
        n_bg = None
        for n in names:
            thr[n], n_bg = far_of("val", n)
            log(f"порог {n} {thr[n]:.6f}")
        val = evaluate("val", names, thr, labels)
        d_iou = abs(val["winner"]["iou"] - WINNER_VAL)
        log(f"winner check iou_delta={d_iou:.3e}")
        if d_iou > 1e-4:
            log("WARNING winner val IoU расходится с публикацией")
        chosen = max(SELECT, key=lambda n: val[n]["iou"])
        log(f"chosen {chosen} val IoU {val[chosen]['iou']:.6f}")
        test = {}
        if not a.skip_test:
            dump_split(gbm, model, "test")
            test = evaluate("test", names, thr, labels)
        beats = bool(val[chosen]["iou"] > WINNER_VAL)
        report = {
            "task": "s3_domlgbm",
            "replaced_release": False,
            "threads": a.threads,
            "seed": a.seed,
            "train": {
                "split": "v8 train",
                "rows": "1500 event + 1500 background pixels per sequence, clean features and one DOM view",
                "aug": spec,
                "shaper": {k: fit[k] for k in fit if k in ("accepted_fit", "best_acf_corr", "delta_abs_mean")},
                "lgbm": "n_estimators 150, num_leaves 31, learning_rate 0.1",
                "infer": "clean channels, no aug",
                "real_csv": False,
            },
            "dom_unet": "artifacts/eval/s3_dom_unet.pt width 16, not retrained here",
            "candidates": {
                "lgbm_dom": "new pixel_lgbm",
                "ens_lgbm": "mean(unet_w24, unet_v810ft, tcn, lgbm_dom)",
                "ens_dom4": "mean(unet_w24, unet_dom, tcn, lgbm_dom)",
                "ens_plus": "mean(unet_w24, unet_v810ft, unet_dom, tcn, lgbm_dom)",
                "ens_blend": "mean(ens_lgbm, unet_dom)",
            },
            "controls": {
                "winner": "published ens_u2ft_tcn_lgbm map",
                "ens_blend_pub": "mean(published ensemble, unet_dom)",
                "pixel_lgbm": "published pixel_lgbm map",
                "unet_dom": "DOM U-Net alone",
            },
            "select": "max val IoU at FAR 1e-3 among candidates; test frozen",
            "published_winner_val_iou": WINNER_VAL,
            "published_winner_test_iou": WINNER_TEST,
            "winner_val_iou_delta": d_iou,
            "n_bg_val": n_bg,
            "val": val,
            "test": test,
            "chosen": chosen,
            "beats_winner": beats,
            "sec": round(time.time() - t0, 1),
        }
        path = Path(a.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(plain(report), indent=1, ensure_ascii=False) + "\n")
        log(f"wrote {path}")
        cv, ct = val[chosen], test.get(chosen, {})
        wv, wt = val["winner"], test.get("winner", {})
        line = (
            "DOMLGBM_DONE "
            f"chosen={chosen} beats={'yes' if beats else 'no'} "
            f"val_iou={cv['iou']:.6f} val_iou_winner={wv['iou']:.6f} "
            f"val_weak={cv['weak_recall']} val_weak_winner={wv['weak_recall']} "
            f"test_iou={ct.get('iou', float('nan'))} test_iou_winner={wt.get('iou', float('nan'))} "
            f"test_weak={ct.get('weak_recall', 'nan')} "
            f"lgbm_dom_val={val['lgbm_dom']['iou']:.6f} "
            f"ens_dom4_val={val['ens_dom4']['iou']:.6f} "
            f"ens_plus_val={val['ens_plus']['iou']:.6f} "
            f"ens_blend_val={val['ens_blend']['iou']:.6f} "
            f"ens_blend_pub_val={val['ens_blend_pub']['iou']:.6f} "
            f"thr={cv['threshold']:.6f} sec={report['sec']}"
        )
        print(line, flush=True)
        with EVENT_LOG.open("a") as f:
            f.write(line + "\n")
    finally:
        shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
