"""Stage 2: классификация типа события. GroupKFold(5) по seq_id на train+val, финальная проверка на test.

Модели: EventNet, CNN только по изображению, LightGBM (legacy и full = форма вырезки + контекст соседей),
rules. После OOF считаются температура, калиброванный вес, логистический стэк и заранее заданная
смесь 0.5/0.5 без температуры. В чекпоинт пишется рецепт с лучшим nested macro-F1 (test не участвует).
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, confusion_matrix
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dts import events as E, models, metrics as M, baseline

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="artifacts/cache/v8")
ap.add_argument("--labels", default="data/synth_meta/v8/labels.csv")
ap.add_argument("--out", default="artifacts/stage2"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--threads", type=int, default=7)
ap.add_argument("--report", default=None, help="json отчёта; по умолчанию artifacts/eval/stage2_<ver>.json")
ap.add_argument("--lgbm-kind", default="full", choices=["legacy", "shape", "context", "full"])
a = ap.parse_args()
torch.set_num_threads(a.threads); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
ver = Path(a.cache).name


def resolve_labels(path, ver):
    p = Path(path)
    if p.exists():
        return p
    alt = Path("data/synth_meta") / ver / "labels.csv"
    if alt.exists():
        print("labels fallback", alt, flush=True)
        return alt
    raise SystemExit(f"labels not found: {path}")


labels_path = resolve_labels(a.labels, ver)
labels = pd.read_csv(labels_path)
ds_path = out / f"dataset_{ver}.npz"

if not ds_path.exists():
    imgs, scal, ys, groups, splits, rules_lab, meta = [], [], [], [], [], [], []
    for p in sorted(Path(a.cache).glob("*/seq*.npz")):
        z = np.load(p); X = z["X"]; x = z["x"]; t = z["t"]; sig = z["sigma"]
        B = M.label_boxes(labels, p.stem); zones = []
        for r in B.itertuples():
            if r.label not in E.SYNTH_TO_CLASS: continue
            xa, xb = max(r.dist_start_m, x[0]), min(r.dist_end_m, x[-1])
            if xb < xa: continue
            im, f = E.crop_and_features(X, sig, x, t, xa, xb, r.ts, r.te)
            imgs.append(im); scal.append(f); ys.append(E.CLASSES.index(E.SYNTH_TO_CLASS[r.label]))
            groups.append(p.stem); splits.append(p.parent.name); zones.append(E.rules_zone(f, xa, xb, r.ts, r.te))
            meta.append(dict(seq_id=p.stem, label=r.label, amp=float(r.amp_obs_C) if r.amp_obs_C == r.amp_obs_C else 0.0))
        if zones:
            rl = baseline.rules_labels(zones, float(x[-1] - x[0]))
            rules_lab += [E.CLASSES.index(E.RULES_TO_CLASS[l]) for l in rl]
    np.savez(ds_path, img=np.stack(imgs), scal=np.stack(scal), y=np.array(ys), groups=np.array(groups), split=np.array(splits),
             rules=np.array(rules_lab), meta=np.array([json.dumps(m) for m in meta]))
D = np.load(ds_path)
img, scal, y, groups, split, rules = D["img"], D["scal"], D["y"], D["groups"], D["split"], D["rules"]
print("dataset", img.shape, np.bincount(y), flush=True)
ctx = E.dataset_context(a.cache, labels_path, y, groups)
K = len(E.CLASSES)
mu, sd = scal.mean(0), scal.std(0) + 1e-6


def train_cnn(idx, use_scalar=True, seed=0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = models.EventNet(K, scal.shape[1]); opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-3)
    cw = torch.tensor(len(idx) / (K * np.maximum(np.bincount(y[idx], minlength=K), 1)), dtype=torch.float32).clamp(max=10)
    Xi = torch.from_numpy(img[idx]); Si = torch.from_numpy(((scal[idx] - mu) / sd).astype(np.float32)); Yi = torch.from_numpy(y[idx])
    if not use_scalar: Si = Si * 0
    for ep in range(a.epochs):
        net.train(); perm = rng.permutation(len(idx))
        for b in range(0, len(idx), 64):
            j = perm[b:b + 64]; xb = Xi[j]
            if rng.random() < 0.5: xb = torch.flip(xb, [3])
            loss = F.cross_entropy(net(xb, Si[j]), Yi[j], weight=cw, label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval(); return net


def pred_cnn(net, idx, use_scalar=True):
    S = torch.from_numpy(((scal[idx] - mu) / sd).astype(np.float32))
    if not use_scalar: S = S * 0
    with torch.no_grad():
        xt = torch.from_numpy(img[idx])
        lo = net(xt, S); lo = 0.5 * (lo + net(torch.flip(xt, [3]), S))
    return torch.softmax(lo, 1).numpy()


def train_lgb(idx, seed, kind):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, class_weight="balanced",
                           random_state=seed, verbose=-1, n_jobs=a.threads)
    m.fit(E.lgbm_matrix(scal[idx], img[idx], ctx[idx], kind), y[idx]); return m


def proba_lgb(m, idx, kind):
    raw = m.predict_proba(E.lgbm_matrix(scal[idx], img[idx], ctx[idx], kind))
    out = np.zeros((len(idx), K), np.float32)
    for i, c in enumerate(m.classes_):
        out[:, int(c)] = raw[:, i]
    return out


def ece(prob, yt, bins=10):
    return E.expected_calibration_error(prob, yt, bins)


def score(yt, yp, prob=None, grp=None):
    r = dict(macro_f1=float(f1_score(yt, yp, average="macro", labels=range(K))),
             per_class={E.CLASSES[k]: float(v) for k, v in enumerate(f1_score(yt, yp, average=None, labels=range(K)))},
             coarse_macro_f1=float(f1_score([E.COARSE[E.CLASSES[i]] for i in yt], [E.COARSE[E.CLASSES[i]] for i in yp], average="macro")),
             accuracy=float((yt == yp).mean()), confusion=confusion_matrix(yt, yp, labels=range(K)).tolist())
    if prob is not None: r["ece"] = ece(prob, yt)
    if grp is not None:
        rng = np.random.default_rng(0); ug = np.unique(grp); gi = {g: np.where(grp == g)[0] for g in ug}; v = []
        for _ in range(300):
            s = np.concatenate([gi[g] for g in rng.choice(ug, len(ug))]); v.append(f1_score(yt[s], yp[s], average="macro", labels=range(K)))
        r["macro_f1_ci95"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return r


def temper_fit(prob, idx):
    return E.fit_temperature(np.log(np.clip(prob[idx], 1e-8, 1.0)), y[idx])


dev = np.where(split != "test")[0]; test = np.where(split == "test")[0]
names = ["cnn_eventnet", "cnn_image_only", "lgbm_features", "lgbm_" + a.lgbm_kind]
oof = {m: np.zeros((len(y), K)) for m in names}
fold_pairs = []
t0 = time.time()
for f, (tr, va) in enumerate(GroupKFold(5).split(dev, groups=groups[dev])):
    tr, va = dev[tr], dev[va]
    assert not set(groups[tr]) & set(groups[va])
    fold_pairs.append((tr, va))
    oof["cnn_eventnet"][va] = pred_cnn(train_cnn(tr, True, a.seed + f), va, True)
    oof["cnn_image_only"][va] = pred_cnn(train_cnn(tr, False, a.seed + f), va, False)
    oof["lgbm_features"][va] = proba_lgb(train_lgb(tr, a.seed + f, "legacy"), va, "legacy")
    if a.lgbm_kind != "legacy":
        oof["lgbm_" + a.lgbm_kind][va] = proba_lgb(train_lgb(tr, a.seed + f, a.lgbm_kind), va, a.lgbm_kind)
    else:
        oof["lgbm_" + a.lgbm_kind] = oof["lgbm_features"]
    print("fold", f, round(time.time() - t0), flush=True)

# Вложенные температура и стэкинг: T и вес фолда учатся без его val.
from sklearn.linear_model import LogisticRegression
cnn_p, lgb_p = oof["cnn_eventnet"], oof["lgbm_" + a.lgbm_kind]
blend = np.zeros_like(cnn_p); lr_oof = np.zeros_like(cnn_p); weights = []
for tr, va in fold_pairs:
    Tc, Tl = temper_fit(cnn_p, tr), temper_fit(lgb_p, tr)
    pc_tr, pl_tr = E.temper_proba(cnn_p[tr], Tc), E.temper_proba(lgb_p[tr], Tl)
    pc_va, pl_va = E.temper_proba(cnn_p[va], Tc), E.temper_proba(lgb_p[va], Tl)
    best_w, best_s = 0.5, -1.0
    for w in np.linspace(0, 1, 11):
        s = f1_score(y[tr], (w * pc_tr + (1 - w) * pl_tr).argmax(1), average="macro", labels=range(K))
        if s > best_s:
            best_s, best_w = s, float(w)
    weights.append(best_w)
    blend[va] = best_w * pc_va + (1 - best_w) * pl_va
    clf = LogisticRegression(max_iter=500)
    clf.fit(np.concatenate([pc_tr, pl_tr], 1), y[tr])
    raw = clf.predict_proba(np.concatenate([pc_va, pl_va], 1))
    for i, c in enumerate(clf.classes_):
        lr_oof[va, int(c)] = raw[:, i]
oof["stack_blend"] = blend
oof["stack_logreg"] = lr_oof
# Вес 0.5 задан до просмотра фолдов. Температура здесь не применяется: она не меняет argmax
# одной модели и на смеси в отборе проигрывала сырым вероятностям.
oof["equal_weight_raw"] = 0.5 * cnn_p + 0.5 * lgb_p

rep = dict(version=ver, classes=E.CLASSES, n_dev=int(len(dev)), n_test=int(len(test)),
           class_counts=np.bincount(y, minlength=K).tolist(), lgbm_kind=a.lgbm_kind,
           labels=str(labels_path), groupkfold={}, test={})
for m, P in oof.items():
    rep["groupkfold"][m] = score(y[dev], P[dev].argmax(1), P[dev], groups[dev])
    rep["groupkfold"][m]["per_fold"] = [float(f1_score(y[va], P[va].argmax(1), average="macro", labels=range(K))) for _, va in fold_pairs]
rep["groupkfold"]["rules_haklolarp"] = score(y[dev], rules[dev], None, groups[dev])
rep["groupkfold"]["stack_blend"]["weights"] = weights

# Рецепт для чекпоинта — по nested GroupKFold, не по test.
# equal_weight_raw стоит среди кандидатов: на v8 он выше и калиброванной смеси, и logreg.
candidates = ["cnn_eventnet", "lgbm_" + a.lgbm_kind, "equal_weight_raw", "stack_blend", "stack_logreg"]
winner = max(candidates, key=lambda m: rep["groupkfold"][m]["macro_f1"])
stack_coef = stack_intercept = None
if winner == "equal_weight_raw":
    T_cnn, T_lgb, cnn_weight = 1.0, 1.0, 0.5
elif winner == "stack_blend":
    T_cnn, T_lgb = temper_fit(cnn_p, dev), temper_fit(lgb_p, dev)
    cnn_weight = float(np.median(weights))
elif winner == "cnn_eventnet":
    T_cnn, T_lgb, cnn_weight = temper_fit(cnn_p, dev), 1.0, 1.0
elif winner == "stack_logreg":
    T_cnn, T_lgb = temper_fit(cnn_p, dev), temper_fit(lgb_p, dev)
    cnn_weight = 0.5
    pc, pl = E.temper_proba(cnn_p[dev], T_cnn), E.temper_proba(lgb_p[dev], T_lgb)
    final_lr = LogisticRegression(max_iter=500)
    final_lr.fit(np.concatenate([pc, pl], 1), y[dev])
    stack_coef = np.zeros((K, 2 * K), np.float32); stack_intercept = np.zeros(K, np.float32)
    for i, c in enumerate(final_lr.classes_):
        stack_coef[int(c)] = final_lr.coef_[i]
        stack_intercept[int(c)] = final_lr.intercept_[i]
else:
    T_cnn, T_lgb, cnn_weight = 1.0, temper_fit(lgb_p, dev), 0.0
rep["deployed"] = dict(winner=winner, temperature=T_cnn, lgbm_temperature=T_lgb, cnn_weight=cnn_weight,
                       stack="logreg" if stack_coef is not None else "weight")

net = train_cnn(dev, True, a.seed)
ck = dict(state=net.state_dict(), mu=mu, sd=sd, classes=E.CLASSES, temperature=T_cnn,
          lgbm_temperature=T_lgb, cnn_weight=cnn_weight)
if stack_coef is not None:
    ck["stack_coef"] = stack_coef; ck["stack_intercept"] = stack_intercept
torch.save(ck, out / "eventnet.pt")
Pt = E.temper_proba(pred_cnn(net, test, True), T_cnn)
rep["test"]["cnn_eventnet"] = score(y[test], Pt.argmax(1), Pt, groups[test])
lg = train_lgb(dev, a.seed, a.lgbm_kind)
Pl = E.temper_proba(proba_lgb(lg, test, a.lgbm_kind), T_lgb)
rep["test"]["lgbm_" + a.lgbm_kind] = score(y[test], Pl.argmax(1), Pl, groups[test])
if stack_coef is not None:
    dec = np.concatenate([Pt, Pl], 1) @ stack_coef.T + stack_intercept
    Pe = np.exp(dec - dec.max(1, keepdims=True)); Pe = Pe / Pe.sum(1, keepdims=True)
else:
    Pe = cnn_weight * Pt + (1.0 - cnn_weight) * Pl
rep["test"]["deployed"] = score(y[test], Pe.argmax(1), Pe, groups[test])
rep["test"]["rules_haklolarp"] = score(y[test], rules[test], None, groups[test])
import joblib
joblib.dump({"model": lg, "kind": a.lgbm_kind}, out / "lgbm.joblib")
report = Path(a.report) if a.report else Path("artifacts/eval") / f"stage2_{ver}.json"
report.parent.mkdir(parents=True, exist_ok=True)
json.dump(rep, open(report, "w"), indent=1)
print("wrote", report, "deployed", winner, flush=True)
for part in ("groupkfold", "test"):
    for m, r in rep[part].items():
        if isinstance(r, dict) and "macro_f1" in r:
            print(part, m, round(r["macro_f1"], 3), r.get("macro_f1_ci95"), "ece", r.get("ece"))
