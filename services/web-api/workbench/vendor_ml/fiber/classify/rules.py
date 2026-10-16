"""Frozen shape rules. They describe geometry, not a physical cause.

Cuts are part of the method card. They are not tuned on the witness windows.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

LABELS = (
    "Background",
    "Local_Heating",
    "Local_Cooling",
    "Periodic_Thermal",
    "Broad_Shift",
    "Sensor_Fault",
)

# A glitch is narrower than a metre-scale patch and jumps by many degrees.
FAULT_MAX_WIDTH_M = 1.5
FAULT_MIN_PEAK_C = 12.0
# Wider than a room-local patch, or a quarter of the measured cable.
BROAD_WIDTH_M = 40.0
BROAD_CABLE_FRACTION = 0.25
BROAD_PROFILE_DELTA_C = 1.5
PERIOD_BAND_M = 8.0
PERIOD_MIN_COUNT = 3
PERIOD_CV_MAX = 0.35


def _epoch(zone: dict) -> float:
    if "t_start_s" in zone:
        return float(zone["t_start_s"])
    dt = datetime.fromisoformat(str(zone["t_start"]))
    # Never let the machine's timezone affect a naive wall-clock sequence.
    return dt.timestamp() if dt.tzinfo is not None else (dt - datetime(1970, 1, 1)).total_seconds()


def _center_m(zone: dict) -> float:
    return 0.5 * (float(zone["start_m"]) + float(zone["end_m"]))


def _peak_contrast(zone: dict) -> float:
    if zone.get("peak_contrast_c") is not None:
        return abs(float(zone["peak_contrast_c"]))
    return abs(float(zone.get("median_contrast_c") or 0.0))


def _periodic_indices(zones: list[dict]) -> set[int]:
    if len(zones) < PERIOD_MIN_COUNT:
        return set()
    order = sorted(range(len(zones)), key=lambda i: (zones[i]["polarity"], _center_m(zones[i]), _epoch(zones[i])))
    clusters: list[list[int]] = []
    for index in order:
        polarity = zones[index]["polarity"]
        center = _center_m(zones[index])
        placed = False
        for cluster in clusters:
            same = (zones[cluster[0]]["polarity"] == polarity and
                    zones[cluster[0]].get('time_segment') == zones[index].get('time_segment'))
            anchor = float(np.median([_center_m(zones[j]) for j in cluster]))
            if same and abs(center - anchor) <= PERIOD_BAND_M:
                cluster.append(index)
                placed = True
                break
        if not placed:
            clusters.append([index])
    chosen: set[int] = set()
    for cluster in clusters:
        if len(cluster) < PERIOD_MIN_COUNT:
            continue
        times = np.array([_epoch(zones[i]) for i in cluster], dtype=np.float64)
        gaps = np.diff(np.sort(times))
        if gaps.size < PERIOD_MIN_COUNT - 1 or np.any(gaps <= 0):
            continue
        cv = float(np.std(gaps) / np.mean(gaps))
        if cv <= PERIOD_CV_MAX:
            chosen.update(cluster)
    return chosen


def _one(zone: dict, cable_length_m: float, *, periodic: bool) -> str:
    width = float(zone["width_m"])
    peak = _peak_contrast(zone)
    if width <= FAULT_MAX_WIDTH_M and peak >= FAULT_MIN_PEAK_C:
        return "Sensor_Fault"
    broad_limit = max(BROAD_WIDTH_M, BROAD_CABLE_FRACTION * float(cable_length_m))
    if width >= broad_limit:
        return "Broad_Shift"
    if periodic:
        return "Periodic_Thermal"
    if zone["polarity"] == "hot":
        return "Local_Heating"
    if zone["polarity"] == "cold":
        return "Local_Cooling"
    return "Background"


def classify_zones(zones: list[dict], cable_length_m: float) -> list[str]:
    """Label each extracted zone. `cable_length_m` is the measured span, not the dead tail."""
    if cable_length_m <= 0:
        raise ValueError("cable_length_m must be positive")
    periodic = _periodic_indices(zones)
    return [_one(zone, cable_length_m, periodic=i in periodic) for i, zone in enumerate(zones)]


def profile_shift(
    clean: np.ndarray,
    valid: np.ndarray,
    x_m: np.ndarray,
    *,
    min_delta_c: float = BROAD_PROFILE_DELTA_C,
) -> dict:
    """A half-cable step versus a smooth route gradient.

    Column medians are fit by a straight line and by a single step. The step
    wins only when it explains the profile much better and the two levels
    differ by at least `min_delta_c`. A local patch does not move those levels.
    Halo detectors miss the step except at its edge.
    """
    values = np.where(np.asarray(valid, dtype=bool), np.asarray(clean, dtype=np.float64), np.nan)
    columns = np.flatnonzero(np.isfinite(values).any(axis=0))
    empty = {"label": "Background", "delta_c": 0.0, "raw_delta_c": 0.0, "side": "none"}
    if columns.size < 9:
        return empty
    profile = np.array([np.nanmedian(values[:, col]) for col in columns], dtype=np.float64)
    x = np.asarray(x_m, dtype=np.float64)[columns]
    slope, intercept = np.polyfit(x, profile, 1)
    line_rss = float(np.sum((profile - (slope * x + intercept)) ** 2))
    n = int(profile.size)
    best_rss = None
    best_delta = 0.0
    for cut in range(n // 5, n - n // 5):
        left = float(np.median(profile[:cut]))
        right = float(np.median(profile[cut:]))
        rss = float(np.sum((profile[:cut] - left) ** 2) + np.sum((profile[cut:] - right) ** 2))
        if best_rss is None or rss < best_rss:
            best_rss = rss
            best_delta = left - right
    third = n // 3
    raw_delta = float(np.median(profile[:third]) - np.median(profile[-third:]))
    step_rss = float(best_rss if best_rss is not None else line_rss)
    step_wins = line_rss > max(step_rss, 1e-9) * 4.0 and abs(best_delta) >= min_delta_c
    if not step_wins or not np.isfinite(best_delta):
        return {
            "label": "Background",
            "delta_c": round(float(best_delta), 3),
            "raw_delta_c": round(raw_delta, 3),
            "side": "none",
        }
    side = "left_warmer" if best_delta > 0 else "right_warmer"
    return {
        "label": "Broad_Shift",
        "delta_c": round(float(best_delta), 3),
        "raw_delta_c": round(raw_delta, 3),
        "side": side,
    }
