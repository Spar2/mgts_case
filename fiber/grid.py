"""Physical sample support; gaps are boundaries, never observation time."""
from __future__ import annotations

import numpy as np


def validate_axis(values: np.ndarray, name: str) -> np.ndarray:
    axis = np.asarray(values, dtype=np.float64)
    if axis.ndim != 1 or not axis.size or not np.isfinite(axis).all() or np.any(np.diff(axis) <= 0):
        raise ValueError(f"{name} must be finite and strictly increasing")
    return axis


def segments(time_s: np.ndarray, max_gap_s: float = 300.0) -> list[slice]:
    times = validate_axis(time_s, "time_s")
    if not np.isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError("max_gap_s must be positive")
    cuts = np.r_[0, np.flatnonzero(np.diff(times) > max_gap_s) + 1, len(times)]
    return [slice(int(a), int(b)) for a, b in zip(cuts[:-1], cuts[1:], strict=True)]


def sample_widths(axis: np.ndarray, *, max_gap: float | None = None) -> np.ndarray:
    """Midpoint cell widths; extrapolate half a local step at each endpoint.

    A lone sample has unit support for scoring, an explicit convention rather
    than a measured duration. With gaps, adjacent large intervals are replaced
    by the median observed step; they never inflate event support.
    """
    values = validate_axis(axis, "axis")
    if len(values) == 1:
        return np.ones(1)
    delta = np.diff(values)
    if max_gap is not None:
        if max_gap <= 0:
            raise ValueError("max_gap must be positive")
        observed = delta[delta <= max_gap]
        typical = float(np.median(observed)) if len(observed) else 1.0
        delta = np.where(delta > max_gap, typical, delta)
    return (np.r_[delta[0], delta] + np.r_[delta, delta[-1]]) / 2
