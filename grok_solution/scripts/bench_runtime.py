"""GroupKFold этапа 2 с runtime-признаками и без них.

Сравниваются нейросеть EventNet (кроп + скаляры), та же сеть с доп. входом
runtime_features, и вариант, где к ней добавлена 1D-свёртка по ряду температуры.
LightGBM (full) и смесь 0.5/0.5 — тот же протокол, что scripts/train_stage2.py.
Реальные CSV только описываются: на них ничего не обучается и флаг по ним не выбирается.

Пример:
  python scripts/bench_runtime.py --epochs 40 --threads 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Потоки до импорта numpy/torch. На общей машине держим мало ядер.
def _argv_value(flag, default):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


_THREADS = _argv_value("--threads", "2")
os.environ["OMP_NUM_THREADS"] = _THREADS
os.environ["MKL_NUM_THREADS"] = _THREADS

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from dts import events as E
from dts import metrics as M
from dts import runtime as R

# Опубликованный GroupKFold на v8 (artifacts/eval/stage2_v8.json), справочно.
PUBLISHED = {
    "eventnet_macro_f1": 0.9040738405440083,
    "eventnet_hvac_f1": 0.9806451612903225,
    "lgbm_full_macro_f1": 0.9473,
    "equal_weight_macro_f1": 0.9546,
}


def collect(cache, labels_path):
    """Ряды и runtime-признаки в том же порядке, что dataset_v8 / train_stage2."""
    labels = pd.read_csv(labels_path)
    feats, traces, ys, groups, splits = [], [], [], [], []
    ctx_rows = []
    files = sorted(Path(cache).glob("*/seq*.npz"))
    t0 = time.time()
    for i, p in enumerate(files):
        z = np.load(p)
        x = np.asarray(z["x"], np.float64)
        t = np.asarray(z["t"], np.float64)
        ch = np.asarray(z["X"][0], np.float32)
        valid = np.asarray(z["X"][2], np.float32)
        B = M.label_boxes(labels, p.stem)
        kept_x0, kept_x1, kept_t0, kept_t1 = [], [], [], []
        local_f, local_tr, local_y = [], [], []
        for r in B.itertuples():
            if r.label not in E.SYNTH_TO_CLASS:
                continue
            xa = max(float(r.dist_start_m), float(x[0]))
            xb = min(float(r.dist_end_m), float(x[-1]))
            if xb < xa:
                continue
            series = R.zone_mean_series(ch, valid, x, xa, xb)
            feat = R.runtime_features(series, t)
            local_f.append(R.feature_vector(feat))
            local_tr.append(R.resample_trace(series, t, R.SERIES_LEN))
            local_y.append(E.CLASSES.index(E.SYNTH_TO_CLASS[r.label]))
            kept_x0.append(xa)
            kept_x1.append(xb)
            kept_t0.append(float(r.ts))
            kept_t1.append(float(r.te))
        if not local_y:
            continue
        ctx_rows.append(E.context_features(kept_x0, kept_x1, kept_t0, kept_t1))
        feats.extend(local_f)
        traces.extend(local_tr)
        ys.extend(local_y)
        groups.extend([p.stem] * len(local_y))
        splits.extend([p.parent.name] * len(local_y))
        if i % 50 == 0:
            print("extract", i, len(files), len(ys), round(time.time() - t0, 1), flush=True)
    runtime = np.nan_to_num(np.stack(feats).astype(np.float32))
    series = np.stack(traces).astype(np.float32)
    ctx = np.concatenate(ctx_rows).astype(np.float32)
    y = np.asarray(ys, np.int64)
    groups = np.asarray(groups)
    split = np.asarray(splits)
    print("extract done", len(y), round(time.time() - t0, 1), flush=True)
    return runtime, series, ctx, y, groups, split


def score(yt, yp, prob, grp, classes):
    labels = list(range(len(classes)))
    per = f1_score(yt, yp, average=None, labels=labels, zero_division=0)
    cm = confusion_matrix(yt, yp, labels=labels)
    out = {
        "macro_f1": float(f1_score(yt, yp, average="macro", labels=labels, zero_division=0)),
        "per_class": {classes[k]: float(per[k]) for k in labels},
        "coarse_macro_f1": float(f1_score(
            [E.COARSE[classes[i]] for i in yt],
            [E.COARSE[classes[i]] for i in yp],
            average="macro", zero_division=0)),
        "accuracy": float((yt == yp).mean()) if len(yt) else 0.0,
        "confusion": cm.tolist(),
        "support": np.bincount(yt, minlength=len(classes)).tolist(),
    }
    hvac = classes.index("HVAC_Diurnal")
    out["f1_Periodic_Thermal"] = out["per_class"]["HVAC_Diurnal"]
    out["hvac_as"] = {
        classes[j]: int(cm[hvac, j]) for j in labels
    }
    out["predicted_hvac_from"] = {
        classes[i]: int(cm[i, hvac]) for i in labels
    }
    if prob is not None:
        out["ece"] = float(E.expected_calibration_error(prob, yt))
    if grp is not None and len(yt):
        rng = np.random.default_rng(0)
        ug = np.unique(grp)
        gi = {g: np.where(grp == g)[0] for g in ug}
        vals = []
        for _ in range(200):
            take = np.concatenate([gi[g] for g in rng.choice(ug, len(ug))])
            vals.append(f1_score(yt[take], yp[take], average="macro", labels=labels, zero_division=0))
        out["macro_f1_ci95"] = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
    return out


def train_net(img, scal, runtime, series, y, idx, use_runtime, use_series, seed, epochs, n_classes):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = R.make_runtime_net(
        n_classes, scal.shape[1], n_runtime=runtime.shape[1],
        use_runtime=use_runtime, use_series=use_series, series_len=series.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
    counts = np.bincount(y[idx], minlength=n_classes)
    cw = torch.tensor(len(idx) / (n_classes * np.maximum(counts, 1)), dtype=torch.float32).clamp(max=10)
    mu_s = scal[idx].mean(0)
    sd_s = scal[idx].std(0) + 1e-6
    base = ((scal[idx] - mu_s) / sd_s).astype(np.float32)
    if use_runtime:
        mu_r = runtime[idx].mean(0)
        sd_r = runtime[idx].std(0) + 1e-6
        extra = ((runtime[idx] - mu_r) / sd_r).astype(np.float32)
        Si = torch.from_numpy(np.concatenate([base, extra], 1))
    else:
        mu_r = sd_r = None
        Si = torch.from_numpy(base)
    Xi = torch.from_numpy(np.ascontiguousarray(img[idx]))
    Yi = torch.from_numpy(y[idx].astype(np.int64))
    Ser = torch.from_numpy(np.ascontiguousarray(series[idx])) if use_series else None
    net.train()
    n = len(idx)
    t_ep0 = time.time()
    for ep in range(epochs):
        perm = rng.permutation(n)
        for b in range(0, n, 64):
            j = perm[b:b + 64]
            xb = Xi[j]
            if rng.random() < 0.5:
                xb = torch.flip(xb, [3])
            ser_b = Ser[j] if Ser is not None else None
            loss = F.cross_entropy(net(xb, Si[j], ser_b), Yi[j], weight=cw, label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep == 0:
            print("epoch0_sec", round(time.time() - t_ep0, 2), "n", n, flush=True)
    net.eval()
    return net, (mu_s.astype(np.float32), sd_s.astype(np.float32),
                 None if mu_r is None else mu_r.astype(np.float32),
                 None if sd_r is None else sd_r.astype(np.float32))


@torch.no_grad()
def predict_net(net, img, scal, runtime, series, idx, norm, use_runtime, use_series):
    mu_s, sd_s, mu_r, sd_r = norm
    base = ((scal[idx] - mu_s) / sd_s).astype(np.float32)
    if use_runtime:
        extra = ((runtime[idx] - mu_r) / sd_r).astype(np.float32)
        S = torch.from_numpy(np.concatenate([base, extra], 1))
    else:
        S = torch.from_numpy(base)
    xt = torch.from_numpy(np.ascontiguousarray(img[idx]))
    ser = torch.from_numpy(np.ascontiguousarray(series[idx])) if use_series else None
    lo = net(xt, S, ser)
    lo = 0.5 * (lo + net(torch.flip(xt, [3]), S, ser))
    return torch.softmax(lo, 1).numpy().astype(np.float32)


def oof_net(img, scal, runtime, series, y, folds, use_runtime, use_series, seed, epochs, n_classes, tag):
    P = np.zeros((len(y), n_classes), np.float32)
    t0 = time.time()
    per_fold = []
    for f, (tr, va) in enumerate(folds):
        net, norm = train_net(img, scal, runtime, series, y, tr, use_runtime, use_series, seed + f, epochs, n_classes)
        P[va] = predict_net(net, img, scal, runtime, series, va, norm, use_runtime, use_series)
        per_fold.append(float(f1_score(y[va], P[va].argmax(1), average="macro", labels=list(range(n_classes)))))
        print(tag, "fold", f, "macro", round(per_fold[-1], 4), "sec", round(time.time() - t0, 1), flush=True)
        del net
    return P, per_fold


def fit_lgbm(scal, bank, shape, ctx, runtime, y, idx, with_runtime, seed, threads):
    import lightgbm as lgb
    parts = [scal[idx], bank[idx], shape[idx], ctx[idx]]
    if with_runtime:
        parts.append(runtime[idx])
    X = np.nan_to_num(np.concatenate(parts, 1).astype(np.float32))
    m = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31, class_weight="balanced",
        random_state=seed, verbose=-1, n_jobs=threads)
    m.fit(X, y[idx])
    return m


def proba_lgbm(model, scal, bank, shape, ctx, runtime, idx, with_runtime, n_classes):
    parts = [scal[idx], bank[idx], shape[idx], ctx[idx]]
    if with_runtime:
        parts.append(runtime[idx])
    X = np.nan_to_num(np.concatenate(parts, 1).astype(np.float32))
    raw = model.predict_proba(X)
    out = np.zeros((len(idx), n_classes), np.float32)
    for i, c in enumerate(model.classes_):
        out[:, int(c)] = raw[:, i]
    return out


def real_traces(real_dir, max_rows=2500):
    """Медианная температура по профилю. Только описание, без меток."""
    import json as _json
    rows = []
    paths = sorted(Path(real_dir).glob("*.csv"))
    for path in paths:
        t0 = time.time()
        n_lines = 0
        with open(path, "rb") as f:
            for _ in f:
                n_lines += 1
        n_data = max(n_lines - 1, 0)
        stride = max(1, int(np.ceil(n_data / max_rows))) if n_data else 1
        times, meds = [], []
        seen = 0
        for chunk in pd.read_csv(path, usecols=["timestamp", "points"], chunksize=200):
            for ts, pts in zip(chunk["timestamp"].to_numpy(), chunk["points"].to_numpy()):
                take = (seen % stride) == 0
                seen += 1
                if not take:
                    continue
                arr = np.asarray(_json.loads(pts), np.float64)
                if arr.ndim != 2 or arr.shape[1] < 2:
                    continue
                v = arr[:, 1]
                v = v[np.isfinite(v) & (v > -299.0)]
                if len(v) == 0:
                    continue
                times.append(ts)
                meds.append(float(np.median(v)))
        if len(meds) < 8:
            rows.append({"file": path.name, "error": "too_few_rows", "n": len(meds)})
            continue
        ts = pd.to_datetime(pd.Series(times), utc=True, format="mixed")
        t = (ts - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()
        order = np.argsort(t, kind="mergesort")
        t = t[order]
        med = np.asarray(meds, np.float64)[order]
        feat = R.runtime_features(med, t)
        keep = {k: feat[k] for k in ("active_frac", "period_s", "duty", "k", "T_inf", "resid_mad", "active_sign", "n_runs")}
        rows.append({
            "file": path.name,
            "n_rows_file": int(n_data),
            "n_used": int(len(med)),
            "stride": int(stride),
            "duration_h": float((t[-1] - t[0]) / 3600.0),
            "seconds": round(time.time() - t0, 1),
            "features": keep,
            "note": "evaluation only; no labels; not used to train, calibrate, or choose the flag",
        })
        print("real", path.name, "n", len(med), "period_h", round(keep["period_s"] / 3600.0, 3),
              "active", round(keep["active_frac"], 3), "sec", rows[-1]["seconds"], flush=True)
    return rows


def _pack(name, P, y, idx, grp, classes, per_fold=None):
    sc = score(y[idx], P[idx].argmax(1), P[idx], grp[idx], classes)
    if per_fold is not None:
        sc["per_fold"] = per_fold
    sc["name"] = name
    return sc


def median_by_class(runtime, y, classes):
    table = {}
    for k, name in enumerate(classes):
        m = y == k
        if not np.any(m):
            continue
        med = np.median(runtime[m], axis=0)
        table[name] = {R.FEATURE_NAMES[i]: float(med[i]) for i in range(len(R.FEATURE_NAMES))}
        table[name]["n"] = int(m.sum())
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="artifacts/cache/v8")
    ap.add_argument("--labels", default="data/synth_meta/v8/labels.csv")
    ap.add_argument("--dataset", default="artifacts/stage2/dataset_v8.npz")
    ap.add_argument("--out", default="artifacts/eval/runtime.json")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--real-dir", default="/workspace/mgts/MGTS_data")
    ap.add_argument("--skip-real", action="store_true")
    ap.add_argument("--skip-series", action="store_true", help="не учить 1D-ветку, только скаляры runtime")
    ap.add_argument("--skip-test", action="store_true", help="не дообучать на всём dev ради test")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.environ["OMP_NUM_THREADS"] = str(a.threads)

    D = np.load(a.dataset)
    img = D["img"]
    scal = D["scal"].astype(np.float32)
    runtime, series, ctx, y, groups, split = collect(a.cache, a.labels)
    if len(y) != len(D["y"]) or not np.array_equal(y, D["y"]):
        raise SystemExit(f"alignment y {y.shape} vs dataset {D['y'].shape}")
    if not np.array_equal(groups.astype(str), D["groups"].astype(str)):
        raise SystemExit("alignment: seq_id order differs from dataset")
    if not np.array_equal(split.astype(str), D["split"].astype(str)):
        raise SystemExit("alignment: split order differs from dataset")
    print("aligned", img.shape, runtime.shape, "ctx", ctx.shape, flush=True)

    classes = list(E.CLASSES)
    K = len(classes)
    dev = np.where(split != "test")[0]
    test = np.where(split == "test")[0]
    folds = []
    for tr, va in GroupKFold(5).split(dev, groups=groups[dev]):
        tr, va = dev[tr], dev[va]
        assert not set(groups[tr]) & set(groups[va])
        folds.append((tr, va))

    bank = E.image_bank(img)
    shape = E.shape_features(img)
    specs = [
        ("eventnet", False, False),
        ("eventnet_runtime", True, False),
    ]
    if not a.skip_series:
        specs.append(("eventnet_runtime_series", True, True))

    oof = {}
    fold_macro = {}
    for name, use_rt, use_ser in specs:
        P, pf = oof_net(img, scal, runtime, series, y, folds, use_rt, use_ser, a.seed, a.epochs, K, name)
        oof[name] = P
        fold_macro[name] = pf

    lgbm_specs = [("lgbm_full", False), ("lgbm_runtime", True)]
    for name, use_rt in lgbm_specs:
        P = np.zeros((len(y), K), np.float32)
        pf = []
        t0 = time.time()
        for f, (tr, va) in enumerate(folds):
            model = fit_lgbm(scal, bank, shape, ctx, runtime, y, tr, use_rt, a.seed + f, a.threads)
            P[va] = proba_lgbm(model, scal, bank, shape, ctx, runtime, va, use_rt, K)
            pf.append(float(f1_score(y[va], P[va].argmax(1), average="macro", labels=list(range(K)))))
            print(name, "fold", f, "macro", round(pf[-1], 4), "sec", round(time.time() - t0, 1), flush=True)
        oof[name] = P
        fold_macro[name] = pf

    oof["blend"] = 0.5 * oof["eventnet"] + 0.5 * oof["lgbm_full"]
    oof["blend_runtime"] = 0.5 * oof["eventnet_runtime"] + 0.5 * oof["lgbm_runtime"]
    if "eventnet_runtime_series" in oof:
        oof["blend_runtime_series"] = 0.5 * oof["eventnet_runtime_series"] + 0.5 * oof["lgbm_runtime"]

    groupkfold = {}
    for name, P in oof.items():
        groupkfold[name] = _pack(name, P, y, dev, groups, classes, fold_macro.get(name))
        print("GKF", name, round(groupkfold[name]["macro_f1"], 4),
              "HVAC", round(groupkfold[name]["per_class"]["HVAC_Diurnal"], 4), flush=True)

    # Test: модели заново на всём dev. В выбор флага эти числа не входят.
    test_rep = {}
    test_probs = {}
    if a.skip_test:
        print("skip test refit", flush=True)
        specs_test = []
        lgbm_test = []
    else:
        specs_test = specs
        lgbm_test = lgbm_specs
    for name, use_rt, use_ser in specs_test:
        net, norm = train_net(img, scal, runtime, series, y, dev, use_rt, use_ser, a.seed, a.epochs, K)
        test_probs[name] = predict_net(net, img, scal, runtime, series, test, norm, use_rt, use_ser)
        test_rep[name] = score(y[test], test_probs[name].argmax(1), test_probs[name], groups[test], classes)
        del net
        print("TEST", name, round(test_rep[name]["macro_f1"], 4),
              "HVAC", round(test_rep[name]["per_class"]["HVAC_Diurnal"], 4), flush=True)
    for name, use_rt in lgbm_test:
        model = fit_lgbm(scal, bank, shape, ctx, runtime, y, dev, use_rt, a.seed, a.threads)
        test_probs[name] = proba_lgbm(model, scal, bank, shape, ctx, runtime, test, use_rt, K)
        test_rep[name] = score(y[test], test_probs[name].argmax(1), test_probs[name], groups[test], classes)
        print("TEST", name, round(test_rep[name]["macro_f1"], 4),
              "HVAC", round(test_rep[name]["per_class"]["HVAC_Diurnal"], 4), flush=True)
    if "eventnet" in test_probs and "lgbm_full" in test_probs:
        test_probs["blend"] = 0.5 * test_probs["eventnet"] + 0.5 * test_probs["lgbm_full"]
        test_probs["blend_runtime"] = 0.5 * test_probs["eventnet_runtime"] + 0.5 * test_probs["lgbm_runtime"]
        if "eventnet_runtime_series" in test_probs:
            test_probs["blend_runtime_series"] = 0.5 * test_probs["eventnet_runtime_series"] + 0.5 * test_probs["lgbm_runtime"]
    for name, P in test_probs.items():
        if name not in test_rep:
            test_rep[name] = score(y[test], P.argmax(1), P, groups[test], classes)
            print("TEST", name, round(test_rep[name]["macro_f1"], 4),
                  "HVAC", round(test_rep[name]["per_class"]["HVAC_Diurnal"], 4), flush=True)

    base = groupkfold["eventnet"]
    candidates = ["eventnet_runtime", "blend_runtime", "lgbm_runtime"]
    if "eventnet_runtime_series" in groupkfold:
        candidates += ["eventnet_runtime_series", "blend_runtime_series"]

    def accepts(name):
        sc = groupkfold[name]
        ref = base if name.startswith("eventnet") else (
            groupkfold["blend"] if name.startswith("blend") else groupkfold["lgbm_full"])
        macro_ok = sc["macro_f1"] >= ref["macro_f1"] - 1e-3
        hvac_up = sc["per_class"]["HVAC_Diurnal"] > ref["per_class"]["HVAC_Diurnal"] + 1e-3
        return bool(macro_ok and hvac_up), ref["name"] if "name" in ref else ""

    decisions = {}
    for name in candidates:
        ok, _ref = accepts(name)
        ref_name = "eventnet" if name.startswith("eventnet") else ("blend" if name.startswith("blend") else "lgbm_full")
        decisions[name] = {
            "accept": ok,
            "compared_to": ref_name,
            "delta_macro_f1": groupkfold[name]["macro_f1"] - groupkfold[ref_name]["macro_f1"],
            "delta_hvac_f1": groupkfold[name]["per_class"]["HVAC_Diurnal"] - groupkfold[ref_name]["per_class"]["HVAC_Diurnal"],
        }
    # Включать флаг, только если нейросеть (EventNet с доп. входом) принимает критерий.
    neural_names = [n for n in ("eventnet_runtime", "eventnet_runtime_series") if n in decisions and decisions[n]["accept"]]
    if neural_names:
        recommend = max(neural_names, key=lambda n: (
            groupkfold[n]["per_class"]["HVAC_Diurnal"], groupkfold[n]["macro_f1"]))
        recommend_enable = True
    else:
        recommend = "off"
        recommend_enable = False

    report = {
        "task": "D_runtime",
        "cache": a.cache,
        "labels": a.labels,
        "dataset": a.dataset,
        "n": int(len(y)),
        "n_dev": int(len(dev)),
        "n_test": int(len(test)),
        "classes": classes,
        "taxonomy": {
            "Periodic_Thermal": "HVAC_Diurnal",
            "HVAC_Cycle": None,
            "Sensor_Fault": "Sensor_Artifact",
            "note": (
                "В dts нет отдельного класса HVAC_Cycle. Правила haklolarp зовут суточный цикл "
                "Periodic_Thermal и отображают его в HVAC_Diurnal (синтетическая метка diurnal_section). "
                "F1 Periodic_Thermal ниже — это F1 HVAC_Diurnal. HVAC_Cycle отдельной меткой не размечен, "
                "число для него не подставляется."
            ),
        },
        "features": list(R.FEATURE_NAMES),
        "series_input": (
            "Пространственное среднее канала dT/σ по интервалу дистанции зоны, по всей записи. "
            "Сырой T из кэша удалён slim_train_cache; канал 0 — температурная аномалия в единицах σ. "
            "k от аффинного масштаба температуры не зависит."
        ),
        "protocol": {
            "cv": "GroupKFold(5) by seq_id on split!=test",
            "epochs": a.epochs,
            "seed": a.seed,
            "threads": a.threads,
            "batch": 64,
            "optimizer": "AdamW lr=2e-3 weight_decay=1e-3",
            "loss": "cross_entropy class_weight label_smoothing=0.05",
            "augmentation": "flip x with p=0.5, inference averages flip",
            "lgbm": "n_estimators=300 lr=0.05 num_leaves=31 class_weight=balanced full=scalar+image+shape+context",
            "blend": "0.5 EventNet + 0.5 LGBM, no temperature",
            "decision": "GroupKFold dev only. Accept if macro-F1 drop <= 0.001 and HVAC_Diurnal F1 gain > 0.001 versus the paired baseline. Test is reported after refit on dev and does not choose the flag. Real CSV is not used.",
            "published_groupkfold_v8": PUBLISHED,
            "default_flag": False,
        },
        "feature_median_by_class_all_splits": median_by_class(runtime, y, classes),
        "groupkfold": groupkfold,
        "decisions": decisions,
        "recommend_enable": recommend_enable,
        "recommend_variant": recommend,
        "test": test_rep,
        "real_eval_only": [],
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out, "w"), ensure_ascii=False, indent=1)
    print("wrote", out, "recommend", recommend_enable, recommend, flush=True)

    ev = os.environ.get("HVAC_EVENT_LOG", "/workspace/grok_logs/_events.log")
    try:
        line = (
            "HVAC_DONE "
            f"eventnet_macro={base['macro_f1']:.4f} "
            f"runtime_macro={groupkfold['eventnet_runtime']['macro_f1']:.4f} "
            f"series_macro={groupkfold.get('eventnet_runtime_series', {}).get('macro_f1', float('nan')):.4f} "
            f"hvac_base={base['per_class']['HVAC_Diurnal']:.4f} "
            f"hvac_runtime={groupkfold['eventnet_runtime']['per_class']['HVAC_Diurnal']:.4f} "
            f"hvac_series={groupkfold.get('eventnet_runtime_series', {}).get('per_class', {}).get('HVAC_Diurnal', float('nan')):.4f} "
            f"blend_macro={groupkfold['blend']['macro_f1']:.4f} "
            f"blend_runtime_macro={groupkfold['blend_runtime']['macro_f1']:.4f} "
            f"lgbm_macro={groupkfold['lgbm_full']['macro_f1']:.4f} "
            f"lgbm_runtime_macro={groupkfold['lgbm_runtime']['macro_f1']:.4f} "
            f"recommend={'on' if recommend_enable else 'off'} variant={recommend}\n"
        )
        Path(ev).parent.mkdir(parents=True, exist_ok=True)
        with open(ev, "a", encoding="utf-8") as f:
            f.write(line)
        print(line.strip(), flush=True)
    except OSError as exc:
        print("event log skipped", exc, flush=True)

    if not a.skip_real and Path(a.real_dir).is_dir():
        try:
            report["real_eval_only"] = real_traces(a.real_dir)
        except Exception as exc:
            report["real_eval_only"] = [{
                "error": str(exc),
                "note": "real evaluation failed; synth decision is unchanged",
            }]
            print("real failed", exc, flush=True)
        json.dump(report, open(out, "w"), ensure_ascii=False, indent=1)
        print("updated", out, "real", len(report["real_eval_only"]), flush=True)
    return report


if __name__ == "__main__":
    main()
