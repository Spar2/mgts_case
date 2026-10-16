"""Compare actual real-measurement evidence, spatial scores and temporal residuals."""
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT/"artifacts/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fiber.io import Record
from fiber.real_data import median
from fiber.grid import segments

out = ROOT/"docs/images/ml-test-review"
out.mkdir(parents=True, exist_ok=True)
examples = [("4", "Changing hot region", [94., 98.]),
            ("5", "Persistent warm region", [299., 303.]),
            ("6", "Warm region with observation gaps", [409., 412.])]
for name, title, bounds in examples:
    rec = Record.load(ROOT/"artifacts/records"/name)
    live = np.where(rec.valid, rec.raw_c, np.nan)
    cols = (rec.x_m >= bounds[0]) & (rec.x_m <= bounds[1])
    flank = ((rec.x_m >= bounds[0]-25) & (rec.x_m <= bounds[0]-8)) | ((rec.x_m >= bounds[1]+8) & (rec.x_m <= bounds[1]+25))
    core, reference = median(live[:, cols], axis=1), median(live[:, flank], axis=1)
    z = np.load(ROOT/"artifacts/anomaly"/name/"z_spatial.npy")
    residual = np.load(ROOT/"artifacts/ml-test-audit"/(name+"_cv_residual.npy"))
    cv_mask = np.load(ROOT/"artifacts/ml-test-audit"/(name+"_cv_mask.npy"))
    hours = (rec.time_s-rec.time_s[0])/3600
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True, layout="constrained")
    for segment in segments(rec.time_s):
        axes[0].plot(hours[segment], core[segment], color="#a54226", label="Raw core" if segment.start == 0 else None)
        axes[0].plot(hours[segment], reference[segment], color="#176c83", label="Raw flanks" if segment.start == 0 else None)
        axes[1].plot(hours[segment], median(z[:, cols], axis=1)[segment], color="#176c83")
        axes[2].plot(hours[segment], median(residual[:, cols], axis=1)[segment], color="#7459a8")
    axes[0].legend()
    axes[0].set_ylabel("Raw temperature, °C")
    axes[1].axhline(3.5, color="#b77f38", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Spatial Z, core median")
    axes[2].axhline(0, color="#87919a", linewidth=1)
    axes[2].set_ylabel("Temporal residual, °C")
    axes[2].set_xlabel("Elapsed wall-clock hours (gaps remain blank)")
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle(f"CSV {name}: {title}, {bounds[0]:.0f}–{bounds[1]:.0f} m\nSpatial contrast and temporal deviation are different measurements; no expert truth", fontsize=11)
    fig.savefig(out/(name+"_spatial_vs_temporal.png"), dpi=150)
    plt.close(fig)
    print(out/(name+"_spatial_vs_temporal.png"))
