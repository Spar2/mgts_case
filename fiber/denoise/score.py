"""Denoise scores that do not treat a smoother filter as automatically better."""

from __future__ import annotations

import numpy as np

from fiber.metrics import spike_retention


def highpass_mad(field: np.ndarray, valid: np.ndarray) -> float:
    """MAD of neighbouring differences along the fibre. Smaller means smoother."""
    left = np.asarray(field, dtype=np.float64)[:, :-1]
    right = np.asarray(field, dtype=np.float64)[:, 1:]
    ok = np.asarray(valid, dtype=bool)[:, :-1] & np.asarray(valid, dtype=bool)[:, 1:]
    ok = ok & np.isfinite(left) & np.isfinite(right)
    if not ok.any():
        return float("nan")
    delta = left[ok] - right[ok]
    return float(np.median(np.abs(delta - np.median(delta))) * 1.4826)


def band_contrast(
    field: np.ndarray,
    valid: np.ndarray,
    x_m: np.ndarray,
    core: tuple[float, float] = (297.0, 303.0),
    near: tuple[tuple[float, float], tuple[float, float]] = ((280.0, 290.0), (310.0, 320.0)),
) -> float:
    """Median temperature in the core minus the median of the two side bands."""
    x = np.asarray(x_m, dtype=np.float64)
    values = np.where(valid, field, np.nan).astype(np.float64)

    def _med(mask: np.ndarray) -> float:
        picked = values[:, mask]
        if picked.size == 0 or not np.isfinite(picked).any():
            return float("nan")
        return float(np.nanmedian(picked))

    core_m = (x >= core[0]) & (x <= core[1])
    side = ((x >= near[0][0]) & (x <= near[0][1])) | ((x >= near[1][0]) & (x <= near[1][1]))
    return _med(core_m) - _med(side)


def synthetic_denoise_case(
    seed: int = 0,
    shape: tuple[int, int] = (12, 80),
    peak_col: int = 40,
    dead_from: int = 70,
    noise_c: float = 0.35,
) -> dict:
    """Smooth cable, a 3-sample +6 C spike, and a dead tail.

    The default field is smaller than the U-Net crop. `synthetic_wide_case`
    is the one that actually exercises that network.
    """
    rng = np.random.default_rng(seed)
    t_count, x_count = shape
    clean = np.full((t_count, x_count), 22.0, dtype=np.float32)
    clean += (0.4 * np.sin(np.linspace(0, 3 * np.pi, x_count))).astype(np.float32)
    clean[:, peak_col:peak_col + 3] += 6.0
    noisy = clean + rng.normal(0, noise_c, size=clean.shape).astype(np.float32)
    valid = np.ones(clean.shape, dtype=bool)
    valid[:, dead_from:] = False
    noisy[:, dead_from:] = -300.0
    return {"clean": clean, "noisy": noisy, "valid": valid, "peak_index": peak_col + 1, "dead_from": dead_from}


def synthetic_wide_case(seed: int = 0) -> dict:
    """48 x 160 field, large enough for the 32 x 128 U-Net crop."""
    return synthetic_denoise_case(seed=seed, shape=(48, 160), peak_col=80, dead_from=140)
