"""Same detector, several clean channels. Thresholds stay frozen."""

from __future__ import annotations

import numpy as np

from fiber.anomaly import spatial_z
from fiber.benchmark.events import alarm_fraction, evaluate_events
from fiber.denoise import apply_method

Z_THRESHOLD = 3.5
CHANNEL_METHODS = ("identity", "median_x3", "bilateral_x", "nlm_wide", "tv_x")
# Research denoisers. They do not enter CHANNEL_METHODS and do not move Z_THRESHOLD.
RESEARCH_CHANNEL_METHODS = ("n2n_tsr", "blindspot_x", "zsn2n")


def _scores(block: dict) -> dict:
    out = {}
    for key, value in block.items():
        out[key] = round(value, 4) if isinstance(value, float) else value
    return out


def detect_channel(
    method: str,
    field: np.ndarray,
    valid: np.ndarray,
    x_m: np.ndarray,
    device: str | None = None,
) -> np.ndarray:
    """Denoise, then spatial Z. `identity` is the raw field."""
    clean = apply_method(method, field, valid, device=device)
    return spatial_z(clean, valid, x_m)[0]


def score_local_channel(method: str, scene: dict, device: str | None = None) -> dict:
    z = detect_channel(method, scene["clean"], scene["valid"], scene["x_m"], device=device)
    hot = evaluate_events(scene["hot"], z >= Z_THRESHOLD, scene["time_s"], scene["x_m"], scene["valid"])
    cold = evaluate_events(scene["cold"], z <= -Z_THRESHOLD, scene["time_s"], scene["x_m"], scene["valid"])
    return {
        "method": method,
        "hot": _scores(hot),
        "cold": _scores(cold),
        "alarm_fraction": round(alarm_fraction(np.abs(z) >= Z_THRESHOLD, scene["valid"]), 4),
        "dead_tail_alarms": int(np.count_nonzero(np.abs(z[:, 180:]) >= Z_THRESHOLD)),
    }


def score_half_cable_channel(method: str, scene: dict, device: str | None = None) -> dict:
    z = detect_channel(method, scene["clean"], scene["valid"], scene["x_m"], device=device)
    return {
        "method": method,
        "alarm_fraction_inside": round(alarm_fraction(np.abs(z) >= Z_THRESHOLD, scene["broad"]), 4),
        "dead_or_invalid_alarms": int(np.count_nonzero((np.abs(z) >= Z_THRESHOLD) & ~scene["valid"])),
    }
