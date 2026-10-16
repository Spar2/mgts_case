"""Refine a median_x3 zone mask on a finer residual. Report only.

The operational zones stay in extract_zones. This module does not add a
component and does not leave the dilation of the zone it starts from.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

MODES = ("half_max", "region_grow")
GROW_K = 2.0


def _dx(x_m: np.ndarray) -> float:
    x = np.asarray(x_m, dtype=np.float64)
    if x.size < 2:
        return 1.0
    return float(max(np.median(np.diff(x)), 1e-6))


def dilate_mask(
    mask: np.ndarray,
    valid: np.ndarray,
    x_m: np.ndarray,
    *,
    dilate_m: float = 4.0,
    dilate_profiles: int = 2,
) -> np.ndarray:
    """Rectangle dilation, then only valid cells."""
    component = np.asarray(mask, dtype=bool)
    live = np.asarray(valid, dtype=bool)
    if component.shape != live.shape or component.ndim != 2:
        raise ValueError("mask and valid must share a [time, distance] shape")
    rad_x = max(1, int(round(float(dilate_m) / _dx(x_m))))
    rad_t = int(dilate_profiles)
    if rad_t < 0:
        raise ValueError("dilate_profiles must be non-negative")
    structure = np.ones((2 * rad_t + 1, 2 * rad_x + 1), dtype=bool)
    return ndimage.binary_dilation(component, structure=structure) & live


def zones_to_mask(
    zones: list[dict],
    shape: tuple[int, int],
    x_m: np.ndarray,
    time_s: np.ndarray,
    *,
    polarity: str | None = None,
) -> np.ndarray:
    """Rasterize zone bounding boxes. Zones do not carry their exact cells."""
    mask = np.zeros(shape, dtype=bool)
    x = np.asarray(x_m, dtype=np.float64)
    times = np.asarray(time_s, dtype=np.float64)
    if mask.shape != (times.size, x.size):
        raise ValueError("shape must be [len(time_s), len(x_m)]")
    for zone in zones:
        if polarity is not None and zone.get("polarity") != polarity:
            continue
        rows = (times >= float(zone["t_start_s"]) - 1e-6) & (times <= float(zone["t_end_s"]) + 1e-6)
        cols = (x >= float(zone["start_m"]) - 1e-6) & (x <= float(zone["end_m"]) + 1e-6)
        if rows.any() and cols.any():
            mask[np.ix_(rows, cols)] = True
    return mask


def _residual(fine_field: np.ndarray, background: np.ndarray) -> np.ndarray:
    resid = np.asarray(fine_field, dtype=np.float64) - np.asarray(background, dtype=np.float64)
    return resid


def _sigma(resid: np.ndarray, valid: np.ndarray) -> float:
    """1.4826 * MAD of the fine residual. No floor: a quiet field stays quiet."""
    live = np.asarray(valid, dtype=bool) & np.isfinite(resid)
    data = resid[live]
    if data.size == 0:
        return 0.0
    center = float(np.median(data))
    mad = float(np.median(np.abs(data - center)))
    if not np.isfinite(mad):
        return 0.0
    return 1.4826 * mad


def _peak_index(component: np.ndarray, signed: np.ndarray) -> tuple[int, int] | None:
    values = np.where(component & np.isfinite(signed), signed, -np.inf)
    if not np.isfinite(values).any() or not np.any(values > -np.inf):
        return None
    flat = int(np.argmax(values))
    return tuple(int(v) for v in np.unravel_index(flat, values.shape))


def _edges(mask: np.ndarray, x_m: np.ndarray, time_s: np.ndarray) -> dict | None:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    x = np.asarray(x_m, dtype=np.float64)
    times = np.asarray(time_s, dtype=np.float64)
    return {
        "start_m": float(x[cols[0]]),
        "end_m": float(x[cols[-1]]),
        "t_start_s": float(times[rows[0]]),
        "t_end_s": float(times[rows[-1]]),
    }


def _contrast(delta: np.ndarray, mask: np.ndarray, polarity: str) -> tuple[float | None, float | None]:
    live = mask & np.isfinite(delta)
    if not live.any():
        return None, None
    values = delta[live]
    if polarity == "hot":
        peak = float(np.max(values))
    else:
        peak = float(np.min(values))
    return peak, float(np.median(values))


def _refine_component(
    component: np.ndarray,
    resid: np.ndarray,
    valid: np.ndarray,
    x_m: np.ndarray,
    *,
    polarity: str,
    mode: str,
    sigma: float,
    dilate_m: float,
    dilate_profiles: int,
) -> np.ndarray:
    signed = resid if polarity == "hot" else -resid
    peak_at = _peak_index(component, signed)
    empty = np.zeros(component.shape, dtype=bool)
    if peak_at is None:
        return empty
    # The peak stays inside the original component, so a neighbour cannot steal it.
    if not component[peak_at]:
        return empty
    peak_value = float(signed[peak_at])
    if not np.isfinite(peak_value) or peak_value <= 0:
        return empty
    dilated = dilate_mask(component, valid, x_m, dilate_m=dilate_m, dilate_profiles=dilate_profiles)
    finite = np.isfinite(signed)
    if mode == "half_max":
        keep = dilated & finite & (signed >= 0.5 * peak_value) & (signed > 0)
    elif mode == "region_grow":
        level = GROW_K * sigma
        keep = dilated & finite & (signed >= level) & (signed > 0)
    else:
        raise ValueError(f"unknown refine mode {mode!r}")
    if not keep[peak_at]:
        return empty
    labeled, _count = ndimage.label(keep)
    label = int(labeled[peak_at])
    if label == 0:
        return empty
    return labeled == label


def refine_zones(
    zones_mask_hot: np.ndarray,
    zones_mask_cold: np.ndarray,
    fine_field: np.ndarray,
    background: np.ndarray,
    valid: np.ndarray,
    x_m: np.ndarray,
    time_s: np.ndarray,
    *,
    dilate_m: float = 4.0,
    dilate_profiles: int = 2,
    mode: str = "half_max",
    coarse_delta: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Return refined hot/cold masks and one record per input component.

    An empty input returns empty bool masks and does not raise. Invalid cells
    and non-finite residuals stay out of the mask.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    hot_in = np.asarray(zones_mask_hot, dtype=bool)
    cold_in = np.asarray(zones_mask_cold, dtype=bool)
    live = np.asarray(valid, dtype=bool)
    resid = _residual(fine_field, background)
    if hot_in.shape != cold_in.shape or hot_in.shape != resid.shape or hot_in.shape != live.shape:
        raise ValueError("masks, fine field, background and valid must share a shape")
    if hot_in.shape != (len(time_s), len(x_m)):
        raise ValueError("axes must match the field")
    hot_out = np.zeros(hot_in.shape, dtype=bool)
    cold_out = np.zeros(hot_in.shape, dtype=bool)
    if not hot_in.any() and not cold_in.any():
        return hot_out, cold_out, []
    sigma = _sigma(resid, live)
    baseline_delta = resid if coarse_delta is None else np.asarray(coarse_delta, dtype=np.float64)
    if baseline_delta.shape != resid.shape:
        raise ValueError("coarse_delta shape must agree with the fine field")
    records: list[dict] = []
    for polarity, source, dest in (("hot", hot_in, hot_out), ("cold", cold_in, cold_out)):
        labeled, count = ndimage.label(source)
        for index in range(1, count + 1):
            component = labeled == index
            refined = _refine_component(
                component, resid, live, x_m,
                polarity=polarity, mode=mode, sigma=sigma,
                dilate_m=dilate_m, dilate_profiles=dilate_profiles,
            )
            dest |= refined
            base_edges = _edges(component & live, x_m, time_s)
            if base_edges is None:
                continue
            base_peak, base_median = _contrast(baseline_delta, component & live, polarity)
            # A vanished component keeps the baseline box, so the zone count stays
            # and the contrast sign does not flip.
            if refined.any():
                edges = _edges(refined, x_m, time_s) or base_edges
                peak, median = _contrast(resid, refined, polarity)
            else:
                edges = base_edges
                peak, median = base_peak, base_median
            records.append({
                "polarity": polarity,
                "mode": mode,
                "refined_empty": not bool(refined.any()),
                "start_m": edges["start_m"],
                "end_m": edges["end_m"],
                "t_start_s": edges["t_start_s"],
                "t_end_s": edges["t_end_s"],
                "peak_delta_c": peak,
                "median_delta_c": median,
                "baseline_start_m": base_edges["start_m"],
                "baseline_end_m": base_edges["end_m"],
                "baseline_t_start_s": base_edges["t_start_s"],
                "baseline_t_end_s": base_edges["t_end_s"],
                "baseline_peak_delta_c": base_peak,
                "baseline_median_delta_c": base_median,
            })
    # Bool masks have no NaN. Invalid cells were excluded in dilate_mask.
    return hot_out, cold_out, records


def truth_extent_errors(record: dict, truth_delta: np.ndarray, x_m: np.ndarray, time_s: np.ndarray) -> dict | None:
    """Peak, median and edge errors of one refined zone against the planted polarity."""
    delta = np.asarray(truth_delta, dtype=np.float64)
    polarity = record["polarity"]
    truth = delta > 0 if polarity == "hot" else delta < 0
    edges = _edges(truth, x_m, time_s)
    if edges is None:
        return None
    values = delta[truth]
    truth_peak = float(np.max(values) if polarity == "hot" else np.min(values))
    truth_median = float(np.median(values))

    def _side(start_key: str, end_key: str, t0_key: str, t1_key: str, peak_key: str, med_key: str) -> dict:
        peak = record.get(peak_key)
        med = record.get(med_key)
        return {
            "peak_error_c": None if peak is None else abs(float(peak) - truth_peak),
            "median_error_c": None if med is None else abs(float(med) - truth_median),
            "x_error_m": 0.5 * (
                abs(float(record[start_key]) - edges["start_m"]) + abs(float(record[end_key]) - edges["end_m"])
            ),
            "t_error_s": 0.5 * (
                abs(float(record[t0_key]) - edges["t_start_s"]) + abs(float(record[t1_key]) - edges["t_end_s"])
            ),
        }

    return {
        "polarity": polarity,
        "refined": _side("start_m", "end_m", "t_start_s", "t_end_s", "peak_delta_c", "median_delta_c"),
        "baseline": _side(
            "baseline_start_m", "baseline_end_m", "baseline_t_start_s", "baseline_t_end_s",
            "baseline_peak_delta_c", "baseline_median_delta_c",
        ),
    }
