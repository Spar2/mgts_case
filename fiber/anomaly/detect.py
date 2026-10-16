"""Spatial contrast, causal temporal Z and CUSUM on median-denoised DTS.

Invalid cells stay at score 0. A slow room-wide drift is absorbed by the
wide spatial background, so HVAC-like changes do not become local alarms.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _bin_index(x_m: np.ndarray, bin_m: float) -> np.ndarray:
    x = np.asarray(x_m, dtype=np.float64)
    if x.size < 2 or np.any(np.diff(x) <= 0):
        raise ValueError("distance axis must be strictly increasing")
    with np.errstate(over='ignore',invalid='ignore'):
        raw_bins = np.floor((x - x[0]) / bin_m)
    if not np.isfinite(raw_bins).all() or raw_bins[-1] >= 100_000:
        raise ValueError('halo grid exceeds resource budget')
    return raw_bins.astype(np.int64)


def spatial_background(
    clean: np.ndarray,
    valid: np.ndarray,
    x_m: np.ndarray,
    *,
    inner_m: float = 8.0,
    outer_m: float = 40.0,
    bin_m: float = 2.0,
) -> np.ndarray:
    """Background is the median temperature in a halo around each point.

    The inner radius is excluded, so a local hotspot a few metres wide does
    not become its own background. A room-scale drift still sits in the halo.
    """
    field = np.asarray(clean, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(field)
    if field.shape != mask.shape or field.ndim != 2:
        raise ValueError("clean and valid must be [time, distance]")
    if not (0 < inner_m < outer_m):
        raise ValueError("need 0 < inner_m < outer_m")
    if not np.isfinite(bin_m) or bin_m <= 0 or field.shape[1] != len(x_m):
        raise ValueError("bin_m must be positive and x_m must match the field")
    bins = _bin_index(x_m, bin_m)
    n_bins = int(bins[-1]) + 1
    t_count = field.shape[0]
    if n_bins > 100_000 or t_count * n_bins > 25_000_000:
        raise ValueError('halo grid exceeds resource budget; process bounded time segments')
    sums = np.zeros((t_count, n_bins), dtype=np.float64)
    counts = np.zeros((t_count, n_bins), dtype=np.float64)
    flat_t = np.repeat(np.arange(t_count), field.shape[1])
    flat_b = np.tile(bins, t_count)
    np.add.at(sums, (flat_t, flat_b), np.where(mask, field, 0.0).ravel())
    np.add.at(counts, (flat_t, flat_b), mask.ravel().astype(np.float64))
    binned = np.full((t_count, n_bins), np.nan, dtype=np.float64)
    ok_bins = counts > 0
    binned[ok_bins] = sums[ok_bins] / counts[ok_bins]

    inner_bins = max(1, int(round(inner_m / bin_m)))
    outer_bins = max(inner_bins + 1, int(round(outer_m / bin_m)))
    # A full [T,bins,halo] copy exceeds 1 GB for file 5 before nanmedian's
    # scratch space. Rows are independent: bound temporary work, preserving
    # exactly the same binned halo and numeric results.
    window_width = 2 * outer_bins + 1
    chunk_rows = max(1, 2_000_000 // max(1, n_bins * window_width))
    bg_bins = np.full(binned.shape, np.nan)
    for start in range(0, t_count, chunk_rows):
        stop = min(t_count, start + chunk_rows)
        pad = np.pad(binned[start:stop], ((0, 0), (outer_bins, outer_bins)), constant_values=np.nan)
        windows = sliding_window_view(pad, window_width, axis=1).copy()
        windows[:, :, outer_bins - inner_bins:outer_bins + inner_bins + 1] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            bg_bins[start:stop] = np.nanmedian(windows, axis=-1)
    bg = bg_bins[:, bins]
    bg[~mask] = np.nan
    return bg


def spatial_z(
    clean: np.ndarray,
    valid: np.ndarray,
    x_m: np.ndarray,
    *,
    inner_m: float = 8.0,
    outer_m: float = 40.0,
    bin_m: float = 2.0,
    floor_c: float = 0.30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (z, background_c, scale_c). Z is 0 on invalid cells.

    Scale is the per-profile MAD of the spatial residual, floored so
    sub-degree jitter does not become a huge Z.
    """
    mask = np.asarray(valid, dtype=bool)
    if not np.isfinite(floor_c) or floor_c <= 0:
        raise ValueError("floor_c must be positive")
    bg = spatial_background(clean, mask, x_m, inner_m=inner_m, outer_m=outer_m, bin_m=bin_m)
    resid = np.asarray(clean, dtype=np.float64) - bg
    resid[~mask] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        scale = 1.4826 * np.nanmedian(np.abs(resid), axis=1)
    scale = np.where(np.isfinite(scale), np.maximum(scale, floor_c), floor_c)
    z = np.zeros(resid.shape, dtype=np.float32)
    ok = mask & np.isfinite(resid)
    denom = np.broadcast_to(scale[:, None], resid.shape)
    z[ok] = (resid[ok] / denom[ok]).astype(np.float32)
    return z, bg.astype(np.float32), scale.astype(np.float32)


def spatial_availability(clean: np.ndarray, valid: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Availability of the spatial score, distinct from temperature validity."""
    if np.shape(clean) != np.shape(valid) or np.shape(clean) != np.shape(background):
        raise ValueError("score availability arrays must share a shape")
    return np.asarray(valid, dtype=bool) & np.isfinite(clean) & np.isfinite(background)


def detection_status(n_zones: int, valid: np.ndarray, available: np.ndarray) -> dict:
    mask = np.asarray(valid, dtype=bool)
    live = np.asarray(available, dtype=bool)
    if mask.shape != live.shape or np.any(live & ~mask):
        raise ValueError("availability must be a subset of validity")
    n_valid = int(mask.sum())
    n_available = int(live.sum())
    complete = bool(n_valid and n_available == n_valid)
    return {
        "anomaly_status": "Anomaly" if n_zones else ("Normal" if complete else "Unknown"),
        "assessment": "complete" if complete else ("partial" if n_available else "unavailable"),
        "n_valid": n_valid,
        "n_score_available": n_available,
        "score_coverage": n_available / n_valid if n_valid else None,
        "n_score_unavailable": n_valid - n_available,
    }


def temporal_view(
    clean: np.ndarray,
    valid: np.ndarray,
    time_s: np.ndarray,
    *,
    window: int = 8,
    min_past: int = 3,
    max_gap_s: float = 300.0,
    floor_c: float = 0.30,
) -> tuple[np.ndarray, np.ndarray]:
    """Causal residual and the cells where a past window actually existed.

    `observed` is false at the start of a time segment. The residual there
    stays 0 because no score was formed; that zero is not a measurement.
    """
    field = np.asarray(clean, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    times = np.asarray(time_s, dtype=np.float64)
    if window < min_past:
        raise ValueError("window must cover min_past")
    if field.ndim != 2 or field.shape != mask.shape or len(times) != field.shape[0]:
        raise ValueError("temporal arrays must agree")
    gaps = np.diff(times, prepend=times[0])
    gaps[0] = 0.0
    segment = np.cumsum(gaps > max_gap_s)
    residual = np.zeros(field.shape, dtype=np.float32)
    observed = np.zeros(field.shape, dtype=bool)
    for t in range(1, field.shape[0]):
        start = max(0, t - window)
        same = segment[start:t] == segment[t]
        if not same.any():
            continue
        start = start + int(np.flatnonzero(same)[0])
        if t - start < min_past:
            continue
        window_v = np.where(mask[start:t], field[start:t], np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(window_v, axis=0)
            mad = 1.4826 * np.nanmedian(np.abs(window_v - med), axis=0)
        mad = np.where(np.isfinite(mad), np.maximum(mad, floor_c), np.nan)
        ok = mask[t] & np.isfinite(med) & np.isfinite(mad) & np.isfinite(field[t])
        residual[t, ok] = ((field[t, ok] - med[ok]) / mad[ok]).astype(np.float32)
        observed[t] = ok
    return residual, observed


def temporal_z(
    clean: np.ndarray,
    valid: np.ndarray,
    time_s: np.ndarray,
    *,
    window: int = 8,
    min_past: int = 3,
    max_gap_s: float = 300.0,
    floor_c: float = 0.30,
) -> np.ndarray:
    """Causal Z against the previous profiles in the same time segment."""
    residual, _observed = temporal_view(
        clean, valid, time_s,
        window=window, min_past=min_past, max_gap_s=max_gap_s, floor_c=floor_c,
    )
    return residual


def temporal_innovation(
    clean: np.ndarray,
    valid: np.ndarray,
    time_s: np.ndarray,
    *,
    window: int = 8,
    min_past: int = 3,
    max_gap_s: float = 300.0,
    floor_c: float = 0.30,
) -> np.ndarray:
    """Historical residual beside spatial_z_halo. Not an operational alarm.

    At each time the score uses only earlier profiles of that column inside
    the same time segment. It is the causal ``temporal_z`` residual: current
    temperature minus the past median, scaled by a floored MAD. ``analyze_field``
    does not read this array when it assigns anomaly status.
    """
    return temporal_z(
        clean,
        valid,
        time_s,
        window=window,
        min_past=min_past,
        max_gap_s=max_gap_s,
        floor_c=floor_c,
    )


def cusum_alarm(
    residual_c: np.ndarray,
    valid: np.ndarray,
    time_s: np.ndarray,
    *,
    drift_c: float = 0.5,
    threshold_c: float = 4.0,
    max_gap_s: float = 300.0,
) -> np.ndarray:
    """Two-sided CUSUM on the spatial residual. 1 = hot, 2 = cold, 0 = quiet."""
    resid = np.asarray(residual_c, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    times = np.asarray(time_s, dtype=np.float64)
    alarm = np.zeros(resid.shape, dtype=np.uint8)
    hot = np.zeros(resid.shape[1], dtype=np.float64)
    cold = np.zeros(resid.shape[1], dtype=np.float64)
    prev = float(times[0])
    for t in range(resid.shape[0]):
        if float(times[t]) - prev > max_gap_s:
            hot[:] = 0.0
            cold[:] = 0.0
        prev = float(times[t])
        row = np.where(mask[t] & np.isfinite(resid[t]), resid[t], 0.0)
        hot = np.maximum(0.0, hot + row - drift_c)
        cold = np.minimum(0.0, cold + row + drift_c)
        hot[~mask[t]] = 0.0
        cold[~mask[t]] = 0.0
        alarm[t, hot >= threshold_c] = 1
        alarm[t, cold <= -threshold_c] = 2
    return alarm
