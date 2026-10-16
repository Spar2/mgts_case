"""Prepare design geometry from the already published, bounded demo package."""
from pathlib import Path
import hashlib
import json
import numpy as np
from matplotlib import colormaps, colors

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / "services/web-api/public/demo-v1/package.json"
p = json.loads(SOURCE.read_text(encoding="utf-8"))
raw = np.asarray(p["raw_c"], dtype=float)
processed = np.asarray(p["processed_c"], dtype=float)
valid = np.asarray(p["invalid_reason"]) == 0
lo, hi = float(raw[valid].min()), float(raw[valid].max())
cmap = colormaps["cividis"].resampled(32)
paths = [""] * 32
times = np.asarray(p["time_s"], dtype=float)
edges = np.r_[times[0] - (times[1]-times[0])/2,
              (times[:-1]+times[1:])/2,
              times[-1] + (times[-1]-times[-2])/2]
ys = (edges-edges[0]) / (edges[-1]-edges[0]) * 200
for r in range(raw.shape[0]):
    for c in range(raw.shape[1]):
        if not valid[r,c]:
            continue
        bucket = min(31, int((raw[r,c]-lo)/(hi-lo)*32))
        x0, x1 = c/86*800, (c+1)/86*800
        paths[bucket] += f"M{x0:.3f},{ys[r]:.3f}H{x1:.3f}V{ys[r+1]:.3f}H{x0:.3f}Z"
row = 25
geometry = {
    "source": p["source"], "source_sha256": p["source_sha256"],
    "package_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    "candidate": p["candidate"], "row_bounds": p["row_bounds"],
    "column_bounds": p["column_bounds"], "shape": list(raw.shape),
    "x_m": p["x_m"], "time_s": p["time_s"], "timestamps": p["timestamps"],
    "timezone": p["timezone"], "lo": lo, "hi": hi,
    "row": row, "row_timestamp": p["timestamps"][row],
    "raw_profile": raw[row].tolist(), "processed_profile": processed[row].tolist(),
    "raw_c": p["raw_c"], "processed_c": p["processed_c"],
    "invalid_reason": p["invalid_reason"],
    "provenance": p["provenance"],
    "paths": [{"color": colors.to_hex(cmap(i)), "geometry": path} for i,path in enumerate(paths)],
}
(HERE/"visual-data.json").write_text(json.dumps(geometry, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k:geometry[k] for k in ["shape","lo","hi","row_timestamp","source_sha256"]}, ensure_ascii=False))
