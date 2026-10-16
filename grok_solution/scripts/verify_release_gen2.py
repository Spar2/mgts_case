"""Сверка: dts.infer.Analyzer(gen2='on') на файлах v8 val против сохранённого ens_blend
(artifacts/pred/v8/<winner>/val + GEN2 U-Net), порог из models_release_gen2. Без real данных."""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dts import infer, data, models
n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
files = data.list_cache("artifacts/cache/v8", "val")[:n]
A = infer.Analyzer(op="far_1e-3", zones="cc", gen2="on")
A._arm_gen2()
thr = float(A.gen2_spec["threshold"])
tp = fp = fn = 0; tpr = fpr = fnr = 0; dmax = 0.0
for p in files:
    z = np.load(p)
    X = np.asarray(z["X"], np.float32); valid = z["valid"].astype(bool); ev = z["ev"]
    w = np.load(Path("artifacts/pred/v8/ens_u2ft_tcn_lgbm/val") / p.name)
    ws = np.asarray(w["s"], np.float32)
    prob = models.predict_full(A.gen2_net, X)
    ref = 0.5 * (ws + (1 - prob[0]))
    out = A.run(z["T"], valid, z["x"], z["t"], postproc=False, gen2="on")
    s = out["score"]
    if s.shape != ref.shape:
        print(p.stem, "shape", s.shape, ref.shape); continue
    dmax = max(dmax, float(np.nanmax(np.abs(s - ref)[valid])))
    truth = (ev > 0) & valid
    for sc, acc in ((s, 0), (ref, 1)):
        m = (sc >= thr) & valid
        a = int((m & truth).sum()), int((m & ~truth).sum()), int((~m & truth).sum())
        if acc == 0: tp += a[0]; fp += a[1]; fn += a[2]
        else: tpr += a[0]; fpr += a[1]; fnr += a[2]
    print(p.stem, "maxdiff", dmax, flush=True)
iou = tp / max(tp + fp + fn, 1); iou_r = tpr / max(tpr + fpr + fnr, 1)
print(json.dumps(dict(n=len(files), thr=thr, max_abs_score_diff=dmax, iou_infer=iou, iou_ref=iou_r)))
