"""Hardware artifacts and validity masks for DTS temperature fields."""

from __future__ import annotations

import numpy as np

# Case files use -300 as a missing-measurement sentinel.
# Values at or below this threshold are never treated as temperature.
SENTINEL_MAX = -200.0
SENTINEL_VALUE = -300.0


def quality_mask(raw_c: np.ndarray, point_counts: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Split a native field into valid temperatures and artifact reasons.

    `raw_c` is [T, X]. Optional `point_counts` marks declared tails: cells
    beyond the declared length of a profile are absent, not cold.
    """
    raw = np.asarray(raw_c)
    if raw.ndim != 2:
        raise ValueError("raw_c must be [time, distance]")
    finite = np.isfinite(raw)
    sentinel = finite & (raw <= SENTINEL_MAX)
    tail = np.zeros(raw.shape, dtype=bool)
    if point_counts is not None:
        counts = np.asarray(point_counts)
        if counts.shape != (raw.shape[0],):
            raise ValueError("point_counts must be one integer per profile")
        idx = np.arange(raw.shape[1])
        tail = idx[None, :] >= counts[:, None]
    valid = finite & ~sentinel & ~tail
    reasons = np.zeros(raw.shape, dtype=np.uint8)
    reasons[~finite] = 1
    reasons[sentinel] = 2
    reasons[tail] = 4
    reasons[valid] = 0
    return {
        "valid": valid,
        "is_dead_zone": ~valid,
        "is_sentinel": sentinel,
        "is_tail": tail,
        "is_nonfinite": ~finite,
        "reason_code": reasons,
    }


def masked_values(raw_c: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Copy of temperatures with invalid cells set to NaN (float64)."""
    out = np.asarray(raw_c, dtype=np.float64).copy()
    out[~np.asarray(valid, dtype=bool)] = np.nan
    return out
