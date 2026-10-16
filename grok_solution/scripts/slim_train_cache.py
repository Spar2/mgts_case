"""Ужать train-кэш (диск): убрать T/base_z/base_zone (для обучения не нужны; X есть и в mmap)."""
import sys, os, numpy as np
from pathlib import Path
for d in sys.argv[1:]:
    for p in sorted(Path(d).glob("*.npz")):
        z = np.load(p)
        if "T" not in z.files: continue
        keep = {k: z[k] for k in z.files if k not in ("T", "base_z", "base_zone")}
        tmp = p.with_suffix(".tmp.npz"); np.savez_compressed(tmp, **keep); os.replace(tmp, p)
print("ok")
