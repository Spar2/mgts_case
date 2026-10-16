"""Ансамбль сидов EventNet и стэкинг с LightGBM. Короткое обучение, без выпуска.

Внешний GroupKFold(5) по seq_id (эпизод = одна запись) на split != test.
В каждом фолде три EventNet с разными сидами и один LGBM full.
Мета-логит учится только на OOF дополнения фолда и применяется к его val:
базовые модели этот val не видели, мета-модель тоже. Test не выбирает рецепт.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dts import events as E
from dts import models

# Сводка: 12 эпох, один сид, смесь 0.5 (artifacts/eval/runtime.json).
# Опубликованная смесь 40 эпох — equal_weight_raw 0.9545, в текстах 0.955.
REF_BLEND_12 = 0.9524
REF_BLEND_40 = 0.9545
EVENT_LOG = Path(os.environ.get("ST2SEED_EVENT_LOG", "/workspace/grok_logs/_events.log"))


def score(yt, yp, classes):
    labels = list(range(len(classes)))
    per = f1_score(yt, yp, average=None, labels=labels, zero_division=0)
    return {
        "macro_f1": float(f1_score(yt, yp, average="macro", labels=labels, zero_division=0)),
        "accuracy": float((yt == yp).mean()),
        "per_class": {classes[i]: float(per[i]) for i in labels},
    }


def train_eventnet(img, scal, y, idx, seed, epochs, n_classes):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = models.EventNet(n_classes, scal.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
    counts = np.bincount(y[idx], minlength=n_classes)
    cw = torch.tensor(len(idx) / (n_classes * np.maximum(counts, 1)), dtype=torch.float32).clamp(max=10)
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
        if ep == 0:
            print("epoch0_sec", round(time.time() - t0, 2), "n", n, "seed", seed, flush=True)
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


def proba_lgbm(model, X, idx, n_classes):
    raw = model.predict_proba(X[idx])
    out = np.zeros((len(idx), n_classes), np.float32)
    for i, c in enumerate(model.classes_):
        out[:, int(c)] = raw[:, i]
    return out


def fit_logreg_oof(features, y, folds, n_classes):
    """Вложенный стэк: на каждом фолде логит учится на чужих эпизодах."""
    P = np.zeros((features.shape[0], n_classes), np.float32)
    for tr, va in folds:
        clf = LogisticRegression(max_iter=500)
        clf.fit(features[tr], y[tr])
        raw = clf.predict_proba(features[va])
        for i, c in enumerate(clf.classes_):
            P[va, int(c)] = raw[:, i]
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--dataset", default="artifacts/stage2/dataset_v8.npz")
    ap.add_argument("--cache", default="artifacts/cache/v8")
    ap.add_argument("--labels", default="data/synth_meta/v8/labels.csv")
    ap.add_argument("--out", default="artifacts/eval/s3_st2seed.json")
    a = ap.parse_args()
    if a.threads > 2:
        raise SystemExit("threads must be <= 2")
    torch.set_num_threads(a.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    D = np.load(a.dataset)
    img = D["img"]
    scal = D["scal"]
    y = D["y"].astype(np.int64)
    groups = D["groups"].astype(str)
    split = D["split"].astype(str)
    classes = list(E.CLASSES)
    K = len(classes)
    dev = np.where(split != "test")[0]
    print("dataset", img.shape, "dev", len(dev), "epochs", a.epochs, "seeds", a.seeds, flush=True)

    t_ctx = time.time()
    ctx = E.dataset_context(a.cache, a.labels, y, groups)
    Xlgb = E.lgbm_matrix(scal, img, ctx, "full")
    print("lgbm matrix", Xlgb.shape, "sec", round(time.time() - t_ctx, 1), flush=True)

    folds = []
    for tr, va in GroupKFold(5).split(dev, groups=groups[dev]):
        tr, va = dev[tr], dev[va]
        assert not set(groups[tr]) & set(groups[va])
        folds.append((tr, va))

    n = len(y)
    seeds = [a.seed0 + 100 * s for s in range(a.seeds)]
    oof_seeds = [np.zeros((n, K), np.float32) for _ in seeds]
    oof_lgb = np.zeros((n, K), np.float32)
    t0 = time.time()
    for f, (tr, va) in enumerate(folds):
        for s, seed in enumerate(seeds):
            net, mu, sd = train_eventnet(img, scal, y, tr, seed + f, a.epochs, K)
            oof_seeds[s][va] = predict_eventnet(net, img, scal, va, mu, sd)
            mf = f1_score(y[va], oof_seeds[s][va].argmax(1), average="macro", labels=list(range(K)), zero_division=0)
            print(f"fold {f} seed {seed} macro {mf:.4f} sec {time.time() - t0:.0f}", flush=True)
            del net
        lgb = train_lgbm(Xlgb, y, tr, a.seed0 + f, a.threads)
        oof_lgb[va] = proba_lgbm(lgb, Xlgb, va, K)
        mf = f1_score(y[va], oof_lgb[va].argmax(1), average="macro", labels=list(range(K)), zero_division=0)
        print(f"fold {f} lgbm macro {mf:.4f} sec {time.time() - t0:.0f}", flush=True)
        del lgb

    dev_y = y[dev]
    dev_g = groups[dev]
    ens = np.mean(np.stack(oof_seeds, 0), axis=0).astype(np.float32)
    blend_seed0 = (0.5 * oof_seeds[0] + 0.5 * oof_lgb).astype(np.float32)
    blend_ens = (0.5 * ens + 0.5 * oof_lgb).astype(np.float32)
    # Признаки стэка — только OOF. Фолды те же, мета-модель не учится на своём val.
    stack_ens = fit_logreg_oof(np.concatenate([ens, oof_lgb], 1), y, folds, K)
    stack_seeds = fit_logreg_oof(np.concatenate(oof_seeds + [oof_lgb], 1), y, folds, K)

    named = {
        "lgbm_full": oof_lgb,
        "seed_mean": ens,
        "blend_seed0": blend_seed0,
        "blend_ens": blend_ens,
        "stack_ens": stack_ens,
        "stack_seeds": stack_seeds,
    }
    for s, P in enumerate(oof_seeds):
        named[f"seed{s}"] = P

    report_g = {}
    for name, P in named.items():
        sc = score(dev_y, P[dev].argmax(1), classes)
        sc["per_fold"] = [
            float(f1_score(y[va], P[va].argmax(1), average="macro", labels=list(range(K)), zero_division=0))
            for _, va in folds
        ]
        report_g[name] = sc
        print(
            f"{name} macro {sc['macro_f1']:.4f} hvac {sc['per_class']['HVAC_Diurnal']:.4f}",
            flush=True,
        )

    def delta(name):
        m = report_g[name]["macro_f1"]
        return {
            "macro_f1": m,
            "hvac_f1": report_g[name]["per_class"]["HVAC_Diurnal"],
            "minus_blend_12ep": m - REF_BLEND_12,
            "minus_blend_40ep": m - REF_BLEND_40,
        }

    compared = ["blend_seed0", "blend_ens", "stack_ens", "stack_seeds", "seed_mean"]
    best_name = max(compared, key=lambda k: report_g[k]["macro_f1"])
    out = {
        "task": "s3_st2seed",
        "trained_release": False,
        "epochs": a.epochs,
        "seeds": seeds,
        "threads": a.threads,
        "n_dev": int(len(dev)),
        "n_test": int((split == "test").sum()),
        "classes": classes,
        "cv": "GroupKFold(5) by seq_id on split!=test; episode = one recording",
        "norm": "scalar mean/std fit on the outer-train fold only",
        "eventnet": "AdamW 2e-3, wd 1e-3, batch 64, label_smoothing 0.05, class weights, x-flip 0.5, TTA flip at predict",
        "lgbm": "full, n_estimators 300, lr 0.05, num_leaves 31, class_weight balanced",
        "blend": "0.5 seed probability + 0.5 LGBM, no temperature",
        "stack": "LogisticRegression on OOF probabilities, fit on the complement of each outer fold (nested by seq_id). stack_ens uses mean(seeds)+LGBM; stack_seeds uses the three seeds and LGBM.",
        "reference": {
            "blend_12ep_one_seed": REF_BLEND_12,
            "blend_40ep_published": REF_BLEND_40,
            "published_name": "equal_weight_raw",
            "source_12ep": "artifacts/eval/runtime.json groupkfold.blend",
            "source_40ep": "artifacts/eval/stage2_v8.json groupkfold.equal_weight_raw 0.9545 (texts round to 0.955)",
        },
        "groupkfold": report_g,
        "compare": {k: delta(k) for k in compared},
        "best_of_new": best_name,
        "beats_12ep_blend": bool(report_g[best_name]["macro_f1"] > REF_BLEND_12),
        "beats_published_blend": bool(report_g[best_name]["macro_f1"] > REF_BLEND_40),
        "sec": round(time.time() - t0, 1),
    }
    path = Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print("wrote", path, flush=True)

    b = report_g["blend_ens"]
    s = report_g["stack_ens"]
    ss = report_g["stack_seeds"]
    line = (
        "ST2SEED_DONE "
        f"epochs={a.epochs} seeds={a.seeds} "
        f"seed_mean={report_g['seed_mean']['macro_f1']:.4f} "
        f"lgbm={report_g['lgbm_full']['macro_f1']:.4f} "
        f"blend_seed0={report_g['blend_seed0']['macro_f1']:.4f} "
        f"blend_ens={b['macro_f1']:.4f} "
        f"stack_ens={s['macro_f1']:.4f} "
        f"stack_seeds={ss['macro_f1']:.4f} "
        f"hvac_blend_ens={b['per_class']['HVAC_Diurnal']:.4f} "
        f"ref12={REF_BLEND_12:.4f} ref40={REF_BLEND_40:.4f} "
        f"best={best_name} beats12={'yes' if out['beats_12ep_blend'] else 'no'} "
        f"beats40={'yes' if out['beats_published_blend'] else 'no'} "
        f"sec={out['sec']}"
    )
    print(line, flush=True)
    with EVENT_LOG.open("a") as f:
        f.write(line + "\n")


if __name__ == "__main__":
    main()
