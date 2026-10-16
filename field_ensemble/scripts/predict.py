"""Карты score/hotness для метода на сплитах кэша -> artifacts/pred/<ver>/<method>/<split>/<seq>.npz"""
import argparse, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dts import data, models, classic

ap = argparse.ArgumentParser()
ap.add_argument("--method", required=True); ap.add_argument("--cache", default="artifacts/cache/v8")
ap.add_argument("--ckpt", default=None); ap.add_argument("--width", type=int, default=None); ap.add_argument("--name", default=None)
ap.add_argument("--splits", default="val,test"); ap.add_argument("--threads", type=int, default=4); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
torch.set_num_threads(a.threads)
name = a.name or a.method
out = Path("artifacts/pred") / Path(a.cache).name / name
net = None; gbm = None; ssl_model = None
if a.method in ("unet", "unet_ms", "tcn", "dilated"):
    net = models.build(a.method, a.width); net.load_state_dict(torch.load(a.ckpt or f"artifacts/stage1/{a.method}.pt")); net.eval()
elif a.method == "ssl_ae":
    from experiments.ssl.ssl import load_model, score_field
    ssl_model = load_model(a.ckpt or "artifacts/stage1/ssl_ae.pt")
elif a.method == "pixel_lgbm":
    import lightgbm as lgb, joblib
    mp = Path("artifacts/stage1/pixel_lgbm.joblib")
    if mp.exists():
        gbm = joblib.load(mp)
    else:
        rng = np.random.default_rng(a.seed); Xs, ys = [], []
        for p in data.list_cache("artifacts/cache/v8", "train"):
            z = np.load(p); y = data.target(z["ev"], z["valid"]).ravel(); F = classic.pixel_features(z["X"].astype(np.float32))
            pos = np.where(y > 0)[0]; neg = np.where(y == 0)[0]
            sel = np.r_[rng.choice(pos, min(len(pos), 1500), replace=False) if len(pos) else [], rng.choice(neg, 1500, replace=False)].astype(int)
            Xs.append(F[sel]); ys.append(y[sel])
        gbm = lgb.LGBMClassifier(n_estimators=150, num_leaves=31, learning_rate=0.1, random_state=a.seed, verbose=-1, n_jobs=a.threads)
        gbm.fit(np.concatenate(Xs), np.concatenate(ys)); mp.parent.mkdir(parents=True, exist_ok=True); joblib.dump(gbm, mp)
t0 = time.time()
for split in a.splits.split(","):
    for p in data.list_cache(a.cache, split):
        dst = out / split / p.name
        if dst.exists(): continue
        X = np.load(p)["X"].astype(np.float32)
        if net is not None:
            pr = models.predict_full(net, X); s, h = 1 - pr[0], pr[1] - pr[2]
        elif a.method == "matched_filter":
            s, h = classic.matched_filter(X)
        elif a.method == "pixel_lgbm":
            pr = gbm.predict_proba(classic.pixel_features(X)).reshape(X.shape[1], X.shape[2], 3)
            s, h = 1 - pr[..., 0], pr[..., 1] - pr[..., 2]
        elif a.method == "ssl_ae":
            s, h = score_field(ssl_model, X)
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dst, s=s.astype(np.float32), h=h.astype(np.float16))
    print(name, split, "done", round(time.time() - t0), flush=True)
