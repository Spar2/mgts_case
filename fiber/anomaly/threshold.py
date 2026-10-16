"""Research thresholds. The operational |Z| >= 3.5 cut is not changed here.

Quiet-window percentile and a peaks-over-threshold fit (scipy genpareto).
ads-evt is an optional extra; this POT does not require it.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import genpareto


def percentile_threshold(values: np.ndarray, risk: float) -> float:
    if not 0.0 < risk < 1.0:
        raise ValueError("risk must lie in (0, 1)")
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if data.size == 0:
        raise ValueError("threshold needs finite values")
    return float(np.quantile(data, 1.0 - risk))


def pot_threshold(values: np.ndarray, risk: float, quantile: float = 0.98) -> float:
    """Threshold whose GPD tail probability matches `risk`."""
    if not 0.0 < risk < 1.0:
        raise ValueError("risk must lie in (0, 1)")
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if data.size < 32:
        return percentile_threshold(data, risk)
    initial = float(np.quantile(data, quantile))
    excess = data[data > initial] - initial
    tail = float(excess.size / data.size)
    if excess.size < 20 or tail <= 0.0 or risk >= tail:
        return percentile_threshold(data, risk)
    shape, _loc, scale = genpareto.fit(excess, floc=0.0)
    if not np.isfinite(scale) or scale <= 0:
        return percentile_threshold(data, risk)
    survival = risk / tail
    if not 0.0 < survival < 1.0:
        return percentile_threshold(data, risk)
    height = float(genpareto.ppf(1.0 - survival, shape, loc=0.0, scale=scale))
    if not np.isfinite(height) or height < 0:
        return percentile_threshold(data, risk)
    return float(initial + height)


def exceedance_rate(values: np.ndarray, threshold: float) -> float:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return float("nan")
    return float(np.mean(data > threshold))


def holdout_report(values: np.ndarray, risk: float, seed: int = 0) -> dict:
    """Fit on one half of a series and measure the exceedance rate on the other half."""
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if data.size < 8:
        raise ValueError("holdout needs at least 8 finite values")
    rng = np.random.default_rng(seed)
    order = rng.permutation(data.size)
    cut = data.size // 2
    train = data[order[:cut]]
    test = data[order[cut:]]
    percent = percentile_threshold(train, risk)
    pot = pot_threshold(train, risk)
    return {
        "risk": float(risk),
        "percentile_threshold": percent,
        "pot_threshold": pot,
        "percentile_exceedance": exceedance_rate(test, percent),
        "pot_exceedance": exceedance_rate(test, pot),
        "operational_threshold_unchanged": 3.5,
    }
