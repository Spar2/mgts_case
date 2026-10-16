"""Копия классификатора этапа 2 с классом Static_Hotspot.

Окна строит dts.static_synth (только импорт). Кэш, dataset_v8.npz,
models_release/ и artifacts/stage2/ не переписываются. Реальные трассы не читаются.
GroupKFold по seq_id. Test в отбор не входит. Строка BEATS_WINNER не пишется:
8-классовый macro-F1 — другой набор меток, это приоритет ревью карточек.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_thr = "2"
if "--threads" in sys.argv:
    _thr = sys.argv[sys.argv.index("--threads") + 1]
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = _thr

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dts import events as E
from dts import models
from dts import static_synth as SS

SEED_BASE = 20261015
STATIC_NAME = "Static_Hotspot"
CLASSES = list(E.CLASSES) + [STATIC_NAME]
K = len(CLASSES)
I_LOCAL = CLASSES.index("Local_Heating")
I_TRACE = CLASSES.index("Contact_Trace")
I_STATIC = CLASSES.index(STATIC_NAME)
I_DUR = E.SCALAR_NAMES.index("log_dur_h")
I_WIDTH = E.SCALAR_NAMES.index("log_width_m")
I_AMP = E.SCALAR_NAMES.index("amp_med")
I_SPACE = E.SCALAR_NAMES.index("space_peak")
I_T0 = E.SCALAR_NAMES.index("t_frac_start")
I_T1 = E.SCALAR_NAMES.index("t_frac_end")
EVENT_LOG = Path(os.environ.get("ST2STATIC_EVENT_LOG", "/workspace/grok_logs/_events.log"))
COLLAPSE_LABELS = ["Local_Heating", "Contact_Trace", "Static_Hotspot", "other"]


def generator_check():
    """Нулевое поле: ширина, амплитуда, горб есть на первом кадре."""
    rng = np.random.default_rng(0)
    sp, _ = SS.hot_spec(rng, 800)
    if not sp or sp.get("role") != "hot":
        raise RuntimeError("hot_spec did not return a hot event")
    if not (SS.HOT_WIDTH[0] - 1e-6 <= sp["width_m"] <= SS.HOT_WIDTH[1] + 1e-6):
        raise RuntimeError(f"width {sp['width_m']} outside {SS.HOT_WIDTH}")
    if not (SS.HOT_AMP[0] - 1e-6 <= sp["amp"] <= SS.HOT_AMP[1] + 1e-6):
        raise RuntimeError(f"amp {sp['amp']} outside {SS.HOT_AMP}")
    field, pos, hard = SS.apply_profiles(np.zeros((40, 800), np.float32), [sp])
    added = field[:, int(sp["center"])]
    ratio = float(added.min() / sp["amp"])
    if ratio < 0.8:
        raise RuntimeError(f"in-window turn-on: peak min/amp={ratio}")
    if not bool(pos[int(sp["center"])]) or bool(hard.any()):
        raise RuntimeError("hot mask mismatch")
    return {
        "width_m": float(sp["width_m"]),
        "amp": float(sp["amp"]),
        "kind": sp["kind"],
        "peak_min_over_amp": ratio,
        "drift": float(sp["drift"]),
    }


def event_columns(ev: np.ndarray) -> np.ndarray:
    return (np.asarray(ev) > 0).mean(axis=0) > 0.02


def specs_for(split: str, stem: str, n_px: int, avoid: np.ndarray):
    if split == "train":
        rng = np.random.default_rng(SS.seq_seed(SEED_BASE, stem))
        return SS.sample_train_specs(rng, n_px, SS.DX, avoid)
    if split == "val":
        return SS.fixed_hot_specs(n_px, SS.seq_seed(SEED_BASE, stem), n_events=2, avoid=avoid)
    if split == "test":
        return SS.fixed_hot_specs(n_px, SS.seq_seed(SEED_BASE + 17, stem), n_events=2, avoid=avoid)
    raise RuntimeError(f"unknown split {split}")


def paste_channel1(X: np.ndarray, sigma: np.ndarray, specs: list) -> np.ndarray:
    """Горб в градусах на остатке канала 1. Канал 0 не меняется. X — уже копия."""
    sig = np.maximum(np.asarray(sigma, np.float32), 0.03)
    R = X[1] * sig[None, :]
    R2, _, _ = SS.apply_profiles(R, specs)
    X[1] = np.clip(R2 / sig[None, :], -12.0, 12.0)
    return X


def added_peak_ratio(X: np.ndarray, sigma: np.ndarray, sp: dict, specs: list) -> float:
    sig = np.maximum(np.asarray(sigma, np.float32), 0.03)
    R = X[1] * sig[None, :]
    R2, _, _ = SS.apply_profiles(R, specs)
    added = R2[:, int(sp["center"])] - R[:, int(sp["center"])]
    return float(added.min() / max(float(sp["amp"]), 1e-6))


def window_from_spec(X, sigma, x, t, sp):
    c = int(sp["center"])
    half = 0.5 * float(sp["width_m"])
    xa = max(float(x[c]) - half, float(x[0]))
    xb = min(float(x[c]) + half, float(x[-1]))
    if xb <= xa:
        return None
    ta, tb = float(t[0]), float(t[-1])
    img, feat = E.crop_and_features(X, sigma, x, t, xa, xb, ta, tb)
    return img, feat, xa, xb, ta, tb


def load_labels(path: Path):
    """Те же ts/te, что у metrics.label_boxes, один раз на всю таблицу."""
    import pandas as pd
    d = pd.read_csv(path)
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    d["ts"] = (pd.to_datetime(d.t_start, utc=True, format="mixed") - epoch).dt.total_seconds()
    d["te"] = (pd.to_datetime(d.t_end, utc=True, format="mixed") - epoch).dt.total_seconds()
    return {str(s): g for s, g in d.groupby("seq_id", sort=False)}


def collect(cache: Path, labels, img0, scal0, y0, g0, s0):
    """Исходные строки npz остаются первыми. Статичные окна дописываются в конец."""
    files = sorted(cache.glob("*/seq*.npz"))
    if not files:
        raise RuntimeError(f"no cache files in {cache}")
    orig_y, orig_g, orig_s = [], [], []
    orig_ctx, static_ctx = [], []
    simg, sscal, sy, sg, ss = [], [], [], [], []
    n_by = {"train": 0, "val": 0, "test": 0}
    n_neg = 0
    amps, widths = [], []
    first = None
    first_ratio = None
    t_build = time.time()
    for i, p in enumerate(files):
        split = p.parent.name
        with np.load(p) as z:
            x = np.array(z["x"], dtype=np.float64, copy=True)
            t = np.array(z["t"], dtype=np.float64, copy=True)
            sigma = np.array(z["sigma"], dtype=np.float32, copy=True)
            ev = np.array(z["ev"], copy=True)
            X = np.array(z["X"], dtype=np.float32, copy=True)
        B = labels.get(p.stem)
        boxes = []
        if B is not None:
            for r in B.itertuples():
                if r.label not in E.SYNTH_TO_CLASS:
                    continue
                xa = max(float(r.dist_start_m), float(x[0]))
                xb = min(float(r.dist_end_m), float(x[-1]))
                if xb < xa:
                    continue
                boxes.append((xa, xb, float(r.ts), float(r.te)))
                orig_y.append(E.CLASSES.index(E.SYNTH_TO_CLASS[r.label]))
                orig_g.append(p.stem)
                orig_s.append(split)
        avoid = event_columns(ev)
        specs = specs_for(split, p.stem, int(x.shape[0]), avoid)
        hots = [sp for sp in specs if sp.get("role") == "hot"]
        n_neg += sum(1 for sp in specs if sp.get("role") != "hot")
        if specs:
            if first_ratio is None and hots:
                first_ratio = added_peak_ratio(X, sigma, hots[0], specs)
                ch0 = np.array(X[0], copy=True)
                X = paste_channel1(X, sigma, specs)
                if not np.array_equal(X[0], ch0):
                    raise RuntimeError(f"channel 0 changed on {p.stem}")
                del ch0
            else:
                X = paste_channel1(X, sigma, specs)
        static_boxes = []
        for sp in hots:
            got = window_from_spec(X, sigma, x, t, sp)
            if got is None:
                continue
            im, feat, xa, xb, ta, tb = got
            simg.append(im)
            sscal.append(feat)
            sy.append(I_STATIC)
            sg.append(p.stem)
            ss.append(split)
            static_boxes.append((xa, xb, ta, tb))
            n_by[split] = n_by.get(split, 0) + 1
            amps.append(float(sp["amp"]))
            widths.append(float(sp["width_m"]))
            if first is None:
                first = {
                    "seq_id": p.stem,
                    "split": split,
                    "amp": float(sp["amp"]),
                    "width_m": float(sp["width_m"]),
                    "kind": sp.get("kind"),
                    "log_dur_h": float(feat[I_DUR]),
                    "log_width_m": float(feat[I_WIDTH]),
                    "amp_med": float(feat[I_AMP]),
                    "space_peak": float(feat[I_SPACE]),
                    "t_frac_start": float(feat[I_T0]),
                    "t_frac_end": float(feat[I_T1]),
                    "mean_abs_ch0": float(np.abs(im[0]).mean()),
                    "mean_abs_ch1": float(np.abs(im[1]).mean()),
                    "peak_min_over_amp": first_ratio,
                }
        both = boxes + static_boxes
        if both:
            xa, xb, ta, tb = (np.array(v, np.float64) for v in zip(*both))
            ctx = E.context_features(xa, xb, ta, tb)
            if boxes:
                orig_ctx.append(ctx[: len(boxes)])
            if static_boxes:
                static_ctx.append(ctx[len(boxes) :])
        del X
        if (i + 1) % 50 == 0 or i + 1 == len(files):
            print("build", i + 1, "/", len(files), "static", sum(n_by.values()), "sec", round(time.time() - t_build, 1), flush=True)
    orig_y = np.asarray(orig_y, np.int64)
    orig_g = np.asarray(orig_g)
    orig_s = np.asarray(orig_s)
    if len(orig_y) != len(y0) or not np.array_equal(orig_y, np.asarray(y0)):
        raise RuntimeError(f"original label alignment failed: walk {len(orig_y)} vs npz {len(y0)}")
    if not np.array_equal(orig_g.astype(str), np.asarray(g0).astype(str)):
        raise RuntimeError("original seq_id alignment failed")
    if not np.array_equal(orig_s.astype(str), np.asarray(s0).astype(str)):
        raise RuntimeError("original split alignment failed")
    if not simg:
        raise RuntimeError("no static windows")
    ctx = np.concatenate(orig_ctx + static_ctx).astype(np.float32)
    img = np.concatenate([np.asarray(img0, np.float32), np.stack(simg).astype(np.float32)], 0)
    scal = np.concatenate([np.asarray(scal0, np.float32), np.stack(sscal).astype(np.float32)], 0)
    y = np.concatenate([np.asarray(y0, np.int64), np.asarray(sy, np.int64)])
    groups = np.concatenate([np.asarray(g0).astype(str), np.asarray(sg).astype(str)])
    split = np.concatenate([np.asarray(s0).astype(str), np.asarray(ss).astype(str)])
    if len(ctx) != len(y) or len(img) != len(y):
        raise RuntimeError(f"row mismatch img {len(img)} ctx {len(ctx)} y {len(y)}")
    gate_static_windows(scal[len(y0) :], first, first_ratio)
    info = {
        "n_files": len(files),
        "n_original": int(len(y0)),
        "n_static": int(len(sy)),
        "n_static_by_split": n_by,
        "n_distractor_profiles": int(n_neg),
        "amp_median": float(np.median(amps)) if amps else None,
        "width_m_median": float(np.median(widths)) if widths else None,
        "amp_min": float(np.min(amps)) if amps else None,
        "amp_max": float(np.max(amps)) if amps else None,
        "first_window": first,
        "peak_min_over_amp": first_ratio,
        "feature_median": feature_median(scal, y),
    }
    return img, scal, y, groups, split, ctx, info


def gate_static_windows(scal_s, first, ratio):
    if ratio is None or ratio < 0.8:
        raise RuntimeError(f"pasted peak is not on from the first frame: min/amp={ratio}")
    sp = scal_s[:, I_SPACE]
    am = np.abs(scal_s[:, I_AMP])
    if float(np.median(sp)) < 1.0 or float(np.median(sp)) <= float(np.median(am)):
        raise RuntimeError(f"static space_peak median {np.median(sp)} vs |amp_med| {np.median(am)}")
    if float(np.median(scal_s[:, I_T0])) > 0.05 or float(np.median(scal_s[:, I_T1])) < 0.95:
        raise RuntimeError("static window is not the whole record in time")
    lo, hi = float(np.log1p(1.5)), float(np.log1p(8.0))
    med_w = float(np.median(scal_s[:, I_WIDTH]))
    if not (lo <= med_w <= hi):
        raise RuntimeError(f"static log_width_m median {med_w} outside [{lo}, {hi}]")
    if first is None:
        raise RuntimeError("no first static window")
    print(
        "static_gate",
        "n", len(scal_s),
        "space_peak_med", round(float(np.median(sp)), 3),
        "amp_med_med", round(float(np.median(scal_s[:, I_AMP])), 3),
        "log_dur_med", round(float(np.median(scal_s[:, I_DUR])), 3),
        "log_width_med", round(med_w, 3),
        "ch1", round(first["mean_abs_ch1"], 3),
        "ch0", round(first["mean_abs_ch0"], 3),
        flush=True,
    )


def feature_median(scal, y):
    names = {
        "log_dur_h": I_DUR,
        "log_width_m": I_WIDTH,
        "amp_med": I_AMP,
        "space_peak": I_SPACE,
        "t_frac_start": I_T0,
        "t_frac_end": I_T1,
    }
    out = {}
    for idx, name in ((I_LOCAL, "Local_Heating"), (I_TRACE, "Contact_Trace"), (I_STATIC, STATIC_NAME)):
        m = y == idx
        out[name] = {k: float(np.median(scal[m, j])) for k, j in names.items()} if m.any() else {}
        out[name]["n"] = int(m.sum())
    return out


def train_eventnet(img, scal, y, idx, seed, epochs, tag):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = models.EventNet(K, scal.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
    counts = np.bincount(y[idx], minlength=K)
    cw = torch.tensor(len(idx) / (K * np.maximum(counts, 1)), dtype=torch.float32).clamp(max=10)
    mu = scal[idx].mean(0)
    sd = scal[idx].std(0) + 1e-6
    Xi = torch.from_numpy(np.ascontiguousarray(img[idx]))
    Si = torch.from_numpy(((scal[idx] - mu) / sd).astype(np.float32))
    Yi = torch.from_numpy(y[idx].astype(np.int64))
    net.train()
    n = len(idx)
    t0 = time.time()
    for ep in range(epochs):
        perm = rng.permutation(n)
        for b in range(0, n, 64):
            j = perm[b:b + 64]
            xb = Xi[j]
            if rng.random() < 0.5:
                xb = torch.flip(xb, [3])
            loss = F.cross_entropy(net(xb, Si[j]), Yi[j], weight=cw, label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep == 0 or ep + 1 == epochs:
            print(tag, "epoch", ep + 1, "sec", round(time.time() - t0, 1), "n", n, flush=True)
    net.eval()
    return net, mu.astype(np.float32), sd.astype(np.float32)


@torch.no_grad()
def predict_eventnet(net, img, scal, idx, mu, sd):
    S = torch.from_numpy(((scal[idx] - mu) / sd).astype(np.float32))
    xt = torch.from_numpy(np.ascontiguousarray(img[idx]))
    lo = net(xt, S)
    lo = 0.5 * (lo + net(torch.flip(xt, [3]), S))
    return torch.softmax(lo, 1).numpy().astype(np.float32)


def train_lgbm(X, y, idx, seed, threads):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31, class_weight="balanced",
        random_state=seed, verbose=-1, n_jobs=threads)
    m.fit(X[idx], y[idx])
    return m


def proba_lgbm(model, X, idx):
    raw = model.predict_proba(X[idx])
    out = np.zeros((len(idx), K), np.float32)
    for i, c in enumerate(model.classes_):
        out[:, int(c)] = raw[:, i]
    return out


def pack(yt, yp):
    labels = list(range(K))
    per = f1_score(yt, yp, average=None, labels=labels, zero_division=0)
    cm = confusion_matrix(yt, yp, labels=labels)
    mapping = np.full(K, 3, np.int64)
    mapping[I_LOCAL] = 0
    mapping[I_TRACE] = 1
    mapping[I_STATIC] = 2
    cm3 = confusion_matrix(mapping[yt], mapping[yp], labels=[0, 1, 2, 3])
    return {
        "macro_f1": float(f1_score(yt, yp, average="macro", labels=labels, zero_division=0)),
        "accuracy": float((yt == yp).mean()) if len(yt) else 0.0,
        "per_class": {CLASSES[i]: float(per[i]) for i in labels},
        "support": np.bincount(yt, minlength=K).astype(int).tolist(),
        "confusion": cm.astype(int).tolist(),
        "focus": {
            "static_as_local": int(cm[I_STATIC, I_LOCAL]),
            "static_as_trace": int(cm[I_STATIC, I_TRACE]),
            "local_as_static": int(cm[I_LOCAL, I_STATIC]),
            "trace_as_static": int(cm[I_TRACE, I_STATIC]),
            "static_correct": int(cm[I_STATIC, I_STATIC]),
            "n_static": int(cm[I_STATIC].sum()),
            "n_local": int(cm[I_LOCAL].sum()),
            "n_trace": int(cm[I_TRACE].sum()),
        },
        "confusion_local_trace_static_other": {
            "labels": COLLAPSE_LABELS,
            "matrix": cm3.astype(int).tolist(),
        },
    }


def refuse_release_path(path: Path):
    parts = set(path.resolve().parts)
    if "models_release" in parts or ("stage2" in parts and "artifacts" in parts):
        raise SystemExit(f"refusing to write {path}")


def append_done(line: str):
    text = EVENT_LOG.read_text() if EVENT_LOG.exists() else ""
    if "ST2STATIC_DONE" in text:
        raise SystemExit("ST2STATIC_DONE already in the event log")
    with EVENT_LOG.open("a") as f:
        f.write(line + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset", default="artifacts/stage2/dataset_v8.npz")
    ap.add_argument("--cache", default="artifacts/cache/v8")
    ap.add_argument("--labels", default="data/synth_meta/v8/labels.csv")
    ap.add_argument("--out", default="artifacts/eval/s3_st2static.json")
    a = ap.parse_args()
    if a.threads > 2:
        raise SystemExit("threads must be <= 2")
    for bad in ("MGTS", "1-6.csv", "real_zones", "real_noise"):
        blob = " ".join((a.dataset, a.cache, a.labels, a.out))
        if bad in blob:
            raise SystemExit(f"refusing real-trace path ({bad})")
    out = Path(a.out)
    refuse_release_path(out)
    torch.set_num_threads(a.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    t0 = time.time()
    gen = generator_check()
    print("generator_check", gen, flush=True)
    labels = load_labels(Path(a.labels))
    with np.load(a.dataset) as D:
        img0 = np.array(D["img"], dtype=np.float32, copy=True)
        scal0 = np.array(D["scal"], dtype=np.float32, copy=True)
        y0 = np.array(D["y"], dtype=np.int64, copy=True)
        g0 = np.array(D["groups"], copy=True)
        s0 = np.array(D["split"], copy=True)
    print("npz", img0.shape, "counts", np.bincount(y0, minlength=7).tolist(), flush=True)
    img, scal, y, groups, split, ctx, info = collect(Path(a.cache), labels, img0, scal0, y0, g0, s0)
    del img0, scal0
    info["generator_check"] = gen
    info["classes"] = CLASSES
    print("rows", len(y), "static", info["n_static_by_split"], "class", np.bincount(y, minlength=K).tolist(), flush=True)
    dev = np.where(split != "test")[0]
    test = np.where(split == "test")[0]
    if set(groups[dev].tolist()) & set(groups[test].tolist()):
        raise RuntimeError("seq_id shared between dev and test")
    Xlgb = E.lgbm_matrix(scal, img, ctx, "full")
    oof_net = np.zeros((len(y), K), np.float32)
    oof_lgb = np.zeros((len(y), K), np.float32)
    fold_pairs = []
    for f, (tr, va) in enumerate(GroupKFold(5).split(dev, groups=groups[dev])):
        tr, va = dev[tr], dev[va]
        if set(groups[tr].tolist()) & set(groups[va].tolist()):
            raise RuntimeError(f"fold {f} leaks seq_id")
        fold_pairs.append((tr, va))
        net, mu, sd = train_eventnet(img, scal, y, tr, a.seed + f, a.epochs, f"fold{f}")
        oof_net[va] = predict_eventnet(net, img, scal, va, mu, sd)
        del net
        lgb = train_lgbm(Xlgb, y, tr, a.seed + f, a.threads)
        oof_lgb[va] = proba_lgbm(lgb, Xlgb, va)
        del lgb
        pred = (0.5 * (oof_net[va] + oof_lgb[va])).argmax(1)
        per = f1_score(y[va], pred, average=None, labels=list(range(K)), zero_division=0)
        print(
            "fold", f, "done", round(time.time() - t0, 1),
            "macro", round(float(f1_score(y[va], pred, average="macro", labels=list(range(K)), zero_division=0)), 4),
            "f1_static", round(float(per[I_STATIC]), 4),
            flush=True,
        )
    covered = np.concatenate([va for _, va in fold_pairs])
    if len(covered) != len(dev) or len(np.unique(covered)) != len(dev):
        raise RuntimeError("OOF does not cover dev exactly once")
    blend = 0.5 * oof_net + 0.5 * oof_lgb
    groupkfold = {
        "eventnet": pack(y[dev], oof_net[dev].argmax(1)),
        "lgbm_full": pack(y[dev], oof_lgb[dev].argmax(1)),
        "equal_weight_raw": pack(y[dev], blend[dev].argmax(1)),
    }
    for name, P in (("eventnet", oof_net), ("lgbm_full", oof_lgb), ("equal_weight_raw", blend)):
        groupkfold[name]["per_fold"] = []
        for f, (_, va) in enumerate(fold_pairs):
            yp = P[va].argmax(1)
            groupkfold[name]["per_fold"].append({
                "fold": f,
                "n": int(len(va)),
                "macro_f1": float(f1_score(y[va], yp, average="macro", labels=list(range(K)), zero_division=0)),
                "f1_static": float(f1_score(y[va], yp, average=None, labels=list(range(K)), zero_division=0)[I_STATIC]),
            })
    print("oof blend", round(groupkfold["equal_weight_raw"]["macro_f1"], 4),
          "f1_static", round(groupkfold["equal_weight_raw"]["per_class"][STATIC_NAME], 4), flush=True)
    net, mu, sd = train_eventnet(img, scal, y, dev, a.seed, a.epochs, "test_refit")
    pt = predict_eventnet(net, img, scal, test, mu, sd)
    del net
    lgb = train_lgbm(Xlgb, y, dev, a.seed, a.threads)
    pl = proba_lgbm(lgb, Xlgb, test)
    del lgb
    pb = 0.5 * pt + 0.5 * pl
    test_rep = {
        "used_for_selection": False,
        "eventnet": pack(y[test], pt.argmax(1)),
        "lgbm_full": pack(y[test], pl.argmax(1)),
        "equal_weight_raw": pack(y[test], pb.argmax(1)),
    }
    sec = round(time.time() - t0, 1)
    report = {
        "task": "stage2_static_hotspot",
        "model": "equal_weight_raw",
        "classes": CLASSES,
        "epochs": a.epochs,
        "threads": a.threads,
        "seed": a.seed,
        "seed_base_train_val": SEED_BASE,
        "seed_base_test": SEED_BASE + 17,
        "recipe": {
            "eventnet": "AdamW 2e-3 wd 1e-3, CE label_smoothing 0.05, class weights clamp 10, batch 64, x-flip 0.5, predict average with flip",
            "scalar_norm": "mean and std inside the training side of each fold; test refit uses dev only",
            "lgbm": "full = scalars + image_bank + shape + context, 300 trees, lr 0.05, num_leaves 31, class_weight balanced",
            "blend": "0.5 EventNet + 0.5 LGBM, no temperature",
            "context": "recomputed per sequence with the new boxes included, so original neighbor counts change",
            "paste": "channel 1 residual in degrees, clip ±12 sigma, channel 0 unchanged, no smooth trend, no x-flip of the field",
            "distractors": "train sample_train_specs neg roles are pasted and unlabeled; val/test fixed_hot_specs are two hots",
        },
        "real_traces_used": False,
        "replaced_release": False,
        "dataset_rewritten": False,
        "selection": "GroupKFold OOF on split!=test. Test refit is a report only. 8-class macro-F1 is not the published 7-class 0.9545 and does not score the detector.",
        "windows": info,
        "n_dev": int(len(dev)),
        "n_test": int(len(test)),
        "class_counts": np.bincount(y, minlength=K).astype(int).tolist(),
        "groupkfold": groupkfold,
        "test": test_rep,
        "sec": sec,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print("wrote", out, flush=True)
    g = groupkfold["equal_weight_raw"]
    tg = test_rep["equal_weight_raw"]
    line = (
        "ST2STATIC_DONE "
        f"model=equal_weight_raw macro_f1={g['macro_f1']:.4f} "
        f"f1_static={g['per_class'][STATIC_NAME]:.4f} "
        f"f1_local={g['per_class']['Local_Heating']:.4f} "
        f"f1_trace={g['per_class']['Contact_Trace']:.4f} "
        f"static_as_local={g['focus']['static_as_local']} "
        f"static_as_trace={g['focus']['static_as_trace']} "
        f"local_as_static={g['focus']['local_as_static']} "
        f"trace_as_static={g['focus']['trace_as_static']} "
        f"test_macro={tg['macro_f1']:.4f} sec={sec}"
    )
    print(line, flush=True)
    append_done(line)


if __name__ == "__main__":
    main()
