"""Ансамбль карт: среднее score/hotness нескольких методов -> artifacts/pred/<ver>/<name>."""
import argparse
from pathlib import Path
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--members", required=True); ap.add_argument("--name", required=True)
ap.add_argument("--ver", default="v8"); ap.add_argument("--splits", default="val,test"); ap.add_argument("--weights", default=None)
a = ap.parse_args()
mem = a.members.split(","); w = [float(v) for v in a.weights.split(",")] if a.weights else [1.0] * len(mem)
root = Path("artifacts/pred") / a.ver
for sp in a.splits.split(","):
    for f in sorted((root / mem[0] / sp).glob("*.npz")):
        s = sum(wi * np.load(root / m / sp / f.name)["s"].astype(np.float32) for m, wi in zip(mem, w)) / sum(w)
        h = sum(wi * np.load(root / m / sp / f.name)["h"].astype(np.float32) for m, wi in zip(mem, w)) / sum(w)
        d = root / a.name / sp; d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / f.name, s=s.astype(np.float32), h=h.astype(np.float16))
print("ok", a.name)
