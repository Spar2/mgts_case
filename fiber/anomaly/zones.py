"""Turn a Z-score field into hot/cold zones along the fibre."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from fiber.grid import sample_widths, segments, validate_axis


def extract_zones(
    z: np.ndarray,
    valid: np.ndarray,
    x_m: np.ndarray,
    time_iso: list[str],
    time_s: np.ndarray,
    *,
    thresh: float = 3.5,
    min_width_m: float = 2.0,
    min_profiles: int = 3,
    min_cells: int = 12,
    gap_m: float = 4.0,
    contrast_c: np.ndarray | None = None,
    max_gap_s: float = 300.0,
) -> list[dict]:
    """Connected components of |Z| above `thresh`, hot and cold separately.

    Gaps up to `gap_m` along the fibre and two profiles in time are closed
    before labeling, so one physical hotspot is not split into fragments.
    Cell counts still use the original mask, not the closed pixels.
    """
    score = np.asarray(z, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(score)
    x = np.asarray(x_m, dtype=np.float64)
    times = np.asarray(time_s, dtype=np.float64)
    validate_axis(x, "x_m")
    validate_axis(times, "time_s")
    if score.shape != mask.shape or score.shape != (len(times), len(x)) or len(time_iso) != len(times):
        raise ValueError("zone arrays and axes must agree")
    if contrast_c is not None and np.shape(contrast_c) != score.shape:
        raise ValueError("contrast_c shape must agree with z")
    time_parts = segments(times, max_gap_s)
    dt = sample_widths(times, max_gap=max_gap_s)
    dx = float(np.median(np.diff(x))) if x.size > 1 else 1.0
    space_k = max(1, int(round(gap_m / max(dx, 1e-6))))
    if space_k % 2 == 0:
        space_k += 1
    structure = np.ones((3, space_k), dtype=bool)
    zones: list[dict] = []
    sides = (("hot", score >= thresh), ("cold", score <= -thresh))
    for polarity, side in sides:
        support = side & mask
        closed = np.zeros_like(support)
        # Closing can repair a brief below-threshold dip, but cannot invent
        # continuity across a missing row/coordinate or a time discontinuity.
        for part in time_parts:
            padded = np.pad(support[part], ((3, 3), (space_k, space_k)))
            filled = ndimage.binary_closing(padded, structure=structure)
            closed[part] = filled[3:-3, space_k:-space_k] & mask[part]
        # Components at neighbouring array rows across a time gap must also
        # be separate even when both boundary rows contain an active event.
        labeled = np.zeros(score.shape, dtype=np.int32)
        n = 0
        for part in time_parts:
            chunk, count = ndimage.label(closed[part])
            chunk[chunk > 0] += n
            labeled[part] = chunk
            n += count
        if n == 0:
            continue
        for index, sl in enumerate(ndimage.find_objects(labeled), start=1):
            if sl is None:
                continue
            t_slice, x_slice = sl
            component = labeled[sl] == index
            support_sl = support[sl] & component
            n_cells = int(support_sl.sum())
            rows = np.flatnonzero(support_sl.any(axis=1))
            cols = np.flatnonzero(support_sl.any(axis=0))
            if rows.size == 0 or cols.size == 0:
                continue
            width_m = float((cols[-1] - cols[0] + 1) * dx)
            n_profiles = int(rows.size)
            if n_profiles < min_profiles or width_m < min_width_m or n_cells < min_cells:
                continue
            values = score[sl][support_sl]
            t0 = int(t_slice.start + rows[0])
            t1 = int(t_slice.start + rows[-1])
            x0 = int(x_slice.start + cols[0])
            x1 = int(x_slice.start + cols[-1])
            peak_flat = int(np.argmax(np.abs(values)))
            zone = {
                "polarity": polarity,
                "start_m": round(float(x[x0]), 3),
                "end_m": round(float(x[x1]), 3),
                "width_m": round(width_m, 3),
                "t_start": time_iso[t0],
                "t_end": time_iso[t1],
                "t_start_s": float(times[t0]),
                "t_end_s": float(times[t1]),
                "duration_s": round(float(times[t1] - times[t0]), 1),
                "observed_duration_s": round(float(dt[t_slice][rows].sum()), 1),
                "time_support_convention": "midpoint_cells_no_gap_time",
                "max_gap_s": max_gap_s,
                "time_segment": next(i for i, part in enumerate(time_parts) if part.start <= t0 < part.stop),
                "n_profiles": n_profiles,
                "n_cells": n_cells,
                "peak_z": round(float(values[peak_flat]), 3),
                "median_z": round(float(np.median(values)), 3),
            }
            if contrast_c is not None:
                contrast = np.asarray(contrast_c, dtype=np.float64)[sl][support_sl]
                zone["median_contrast_c"] = round(float(np.nanmedian(contrast)), 3)
                zone["peak_contrast_c"] = round(float(np.nanmax(np.abs(contrast)) * np.sign(np.nanmedian(contrast))), 3)
            zones.append(zone)
    zones.sort(key=lambda item: abs(item["peak_z"]), reverse=True)
    return zones
