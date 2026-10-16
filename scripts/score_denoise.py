"""Compare denoisers on a planted spike and on the 5.csv warm window."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fiber.anomaly.detect import spatial_z
from fiber.denoise import METHODS, apply_method
from fiber.denoise.score import band_contrast, highpass_mad, synthetic_denoise_case, synthetic_wide_case
from fiber.io import Record
from fiber.metrics import spike_retention


CLASSICAL = [
    "median_x3",
    "wiener_x",
    "wavelet_x",
    "wavelet_bayes",
    "nlm_x",
    "nlm_wide",
    "bilateral_x",
    "tv_x",
]

# Frozen protocol. Same window and threshold as the registered 5-early-warm witness.
WITNESS_ROWS = slice(20, 202)
WITNESS_X = (299.0, 303.0)
WITNESS_Z = 3.5


def _synthetic_row(name: str, case: dict) -> dict:
    t0 = time.perf_counter()
    out = apply_method(name, case["noisy"], case["valid"])
    row = case["clean"].shape[0] // 2
    keep = spike_retention(
        np.where(case["valid"][row], case["clean"][row], np.nan),
        np.where(case["valid"][row], out[row], np.nan),
        peak_index=case["peak_index"],
        half_width=2,
    )
    noise = out[case["valid"]] - case["clean"][case["valid"]]
    dead = case["dead_from"]
    return {
        "family": METHODS.get(name, {}).get("family", "research"),
        "spike_retention": round(float(keep), 4),
        "error_mad_c": round(float(np.median(np.abs(noise - np.median(noise))) * 1.4826), 4),
        "dead_finite": int(np.isfinite(out[:, dead:]).sum()),
        "elapsed_s": round(time.perf_counter() - t0, 3),
    }


def _witness_detection(field: np.ndarray, valid: np.ndarray, x_m: np.ndarray) -> dict:
    z = spatial_z(field, valid, x_m)[0]
    cols = (x_m >= WITNESS_X[0]) & (x_m <= WITNESS_X[1])
    window = z[WITNESS_ROWS][:, cols]
    ok = valid[WITNESS_ROWS][:, cols] & np.isfinite(window)
    if not ok.any():
        return {"witness_hot_frac": None, "witness_median_z": None}
    picked = window[ok]
    return {
        "witness_hot_frac": round(float((picked >= WITNESS_Z).mean()), 4),
        "witness_median_z": round(float(np.median(picked)), 3),
    }


def _real_row(name: str, raw: np.ndarray, valid: np.ndarray, x_m: np.ndarray, raw_contrast: float) -> dict:
    t0 = time.perf_counter()
    out = apply_method(name, raw, valid)
    contrast = band_contrast(out, valid, x_m)
    row = {
        "family": METHODS.get(name, {}).get("family", "research"),
        "hf_mad_c": round(highpass_mad(out, valid), 4),
        "witness_contrast_c": round(float(contrast), 4),
        "contrast_retention": round(float(contrast / raw_contrast), 4) if raw_contrast else None,
        "dead_alarms": int(np.isfinite(out[:, x_m >= 1700]).sum()),
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }
    row.update(_witness_detection(out, valid, x_m))
    return row


def main() -> None:
    names = list(CLASSICAL)
    unet = ROOT / "models" / "weights" / "unet2d_n2self.pt"
    if unet.exists():
        names.append("unet2d_n2self")
    print("synthetic 12x80")
    synthetic = {}
    small = synthetic_denoise_case()
    wide = synthetic_wide_case()
    synthetic_wide = {}
    for name in names:
        synthetic[name] = _synthetic_row(name, small)
        synthetic_wide[name] = _synthetic_row(name, wide)
        row = synthetic[name]
        wide_row = synthetic_wide[name]
        print(
            f"  {name:16} small keep={row['spike_retention']:.3f} err={row['error_mad_c']:.3f} "
            f"wide keep={wide_row['spike_retention']:.3f} err={wide_row['error_mad_c']:.3f}"
        )
    record = Record.load(ROOT / "artifacts" / "records" / "5")
    raw_contrast = band_contrast(record.raw_c, record.valid, record.x_m)
    print(f"real 5.csv raw contrast {raw_contrast:.3f} C")
    real = {}
    for name in names:
        real[name] = _real_row(name, record.raw_c, record.valid, record.x_m, raw_contrast)
        row = real[name]
        print(
            f"  {name:16} hf={row['hf_mad_c']:.3f} contrast={row['witness_contrast_c']:.3f} "
            f"keep={row['contrast_retention']:.3f} hot={row['witness_hot_frac']} "
            f"z={row['witness_median_z']} {row['elapsed_s']}s"
        )
    report = {
        "synthetic": synthetic,
        "synthetic_wide": synthetic_wide,
        "synthetic_note": "12x80 is smaller than the U-Net crop, so that network is returned unchanged there. synthetic_wide is 48x160.",
        "record_5": real,
        "raw_contrast_c": raw_contrast,
        "raw_hf_mad_c": round(highpass_mad(record.raw_c, record.valid), 4),
        "witness": {
            "id": "5-early-warm-core",
            "rows": [WITNESS_ROWS.start, WITNESS_ROWS.stop - 1],
            "x_m": list(WITNESS_X),
            "z_threshold": WITNESS_Z,
            "note": "threshold frozen; this column does not retune the detector",
        },
    }
    dest = ROOT / "artifacts" / "denoise"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "scorecard.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", dest / "scorecard.json")


if __name__ == "__main__":
    main()
