"""Denoise and quality metrics. No expert labels required."""

from __future__ import annotations

import numpy as np


def residual_stats(residual: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    r = np.asarray(residual, dtype=np.float64)
    v = np.asarray(valid, dtype=bool) & np.isfinite(r)
    if not v.any():
        return {"n": 0, "mad": float("nan"), "std": float("nan"), "p95": float("nan")}
    x = r[v]
    med = np.median(x)
    mad = float(np.median(np.abs(x - med)) * 1.4826)
    return {
        "n": int(v.sum()),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "mad": mad,
        "p95": float(np.percentile(np.abs(x), 95)),
        "max_abs": float(np.max(np.abs(x))),
    }


def spike_retention(clean: np.ndarray, denoised: np.ndarray, peak_index: int, half_width: int = 2) -> float:
    """Fraction of an injected peak height that survived denoising."""
    a = max(0, peak_index - half_width)
    b = min(clean.size, peak_index + half_width + 1)
    baseline = np.nanmedian(np.r_[clean[:a], clean[b:]])
    true_h = float(np.nanmax(clean[a:b]) - baseline)
    pred_h = float(np.nanmax(denoised[a:b]) - baseline)
    if true_h <= 1e-9:
        return float("nan")
    return pred_h / true_h
