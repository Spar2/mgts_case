"""Attach spatial and temporal evidence to zones the spatial alarm already kept.

The temporal view does not add or remove zones and does not set anomaly status.
"""

from __future__ import annotations

import numpy as np

from fiber.anomaly.detect import temporal_view
from fiber.classify.rules import periodicity_cv


def _signed_contrast(zone: dict) -> float:
    sign = 1.0 if zone.get("polarity") == "hot" else -1.0
    median = zone.get("median_contrast_c")
    if median is not None and np.isfinite(median) and float(median) * sign > 0:
        return float(median)
    peak = zone.get("peak_contrast_c")
    if peak is not None and np.isfinite(peak) and float(peak) != 0.0:
        return float(sign * abs(float(peak)))
    if median is not None and np.isfinite(median) and float(median) != 0.0:
        return float(sign * abs(float(median)))
    raise ValueError("zone has no signed contrast")


def _positive(zone: dict, key: str) -> float:
    value = float(zone[key])
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{key} must be finite and positive")
    return value


def _box(zone: dict, x_m: np.ndarray, time_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x_m, dtype=np.float64)
    times = np.asarray(time_s, dtype=np.float64)
    cols = np.flatnonzero((x >= float(zone["start_m"]) - 5e-4) & (x <= float(zone["end_m"]) + 5e-4))
    rows = np.flatnonzero(
        (times >= float(zone["t_start_s"]) - 1e-3) & (times <= float(zone["t_end_s"]) + 1e-3)
    )
    return rows, cols


def _alarm(zone: dict, z_box: np.ndarray, live: np.ndarray, thresh: float) -> np.ndarray:
    if zone.get("polarity") == "hot":
        return live & (z_box >= thresh)
    if zone.get("polarity") == "cold":
        return live & (z_box <= -thresh)
    return live & (np.abs(z_box) >= thresh)


def annotate_zone_fields(
    zones: list[dict],
    *,
    temporal: np.ndarray,
    temporal_ok: np.ndarray,
    z: np.ndarray,
    spatial_ok: np.ndarray,
    x_m: np.ndarray,
    time_s: np.ndarray,
    thresh: float,
) -> list[dict]:
    """Add the five measure fields and both evidence views. Same zones out."""
    n_in = len(zones)
    cvs = periodicity_cv(zones)
    for zone, cv in zip(zones, cvs, strict=True):
        contrast = _signed_contrast(zone)
        width_m = _positive(zone, "width_m")
        observed_duration_s = _positive(zone, "observed_duration_s")
        support = int(zone["n_cells"])
        if support <= 0:
            raise ValueError("support must count alarm cells")
        zone["contrast"] = contrast
        zone["width_m"] = width_m
        zone["observed_duration_s"] = observed_duration_s
        zone["support"] = support
        zone["periodicity_CV"] = None if cv is None else float(cv)
        rows, cols = _box(zone, x_m, time_s)
        scored = np.empty(0, dtype=np.float64)
        if rows.size and cols.size:
            selector = np.ix_(rows, cols)
            alarm = _alarm(zone, np.asarray(z)[selector], np.asarray(spatial_ok, dtype=bool)[selector], thresh)
            use = alarm & np.asarray(temporal_ok, dtype=bool)[selector]
            if np.any(use):
                scored = np.asarray(temporal, dtype=np.float64)[selector][use]
                scored = scored[np.isfinite(scored)]
        if scored.size == 0:
            temporal_evidence = {
                "view": "temporal",
                "source": "temporal_innovation",
                "available": False,
                "score_state": "unavailable",
                "median_z": None,
                "n_scored": 0,
            }
        else:
            temporal_evidence = {
                "view": "temporal",
                "source": "temporal_innovation",
                "available": True,
                "score_state": "scored",
                "median_z": round(float(np.median(scored)), 6),
                "n_scored": int(scored.size),
            }
        zone["evidence"] = {
            "spatial": {
                "view": "spatial",
                "detector": "spatial_z_halo",
                "available": True,
                "contrast_c": contrast,
                "support": support,
            },
            "temporal": temporal_evidence,
        }
    if len(zones) != n_in:
        raise RuntimeError("evidence annotation changed the zone count")
    return zones


def attach_dual_view(
    zones: list[dict],
    clean: np.ndarray,
    valid: np.ndarray,
    x_m: np.ndarray,
    time_s: np.ndarray,
    z: np.ndarray,
    spatial_ok: np.ndarray,
    *,
    thresh: float,
) -> list[dict]:
    """Read the causal temporal residual onto zones the spatial alarm kept."""
    if not zones:
        return zones
    residual, observed = temporal_view(clean, valid, time_s)
    return annotate_zone_fields(
        zones,
        temporal=residual,
        temporal_ok=observed,
        z=z,
        spatial_ok=spatial_ok,
        x_m=x_m,
        time_s=time_s,
        thresh=thresh,
    )
