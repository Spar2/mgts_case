"""Research context beside a marked rectangle or a zone.

The numbers are read from a field ``analyze_field`` already scored. The
channel does not add or remove zones, does not set ``anomaly_status``, and
does not change ``expert_confirmed``. The wide-channel soft flag is a read
of the halo-inside fraction and the far-cable gap. It is not a class.
"""

from __future__ import annotations

import math

import numpy as np

# Fixed before any card is read. Published next-0003 has a halo-inside
# fraction about 0.786 and a far-cable gap of several °C, so both cuts sit
# on the flagged side of that card. 2.0 °C is the round gap already used
# when a far-cable difference is treated as noticeable. 0.70 is below that
# published share and above a ring that mostly sits outside the rectangle.
WIDE_CHANNEL_HALO_INSIDE_MIN = 0.70
WIDE_CHANNEL_FAR_GAP_C = 2.0
WIDE_CHANNEL_CANDIDATE = "Broad_Shift"


def _median(clean: np.ndarray, available: np.ndarray, rows: np.ndarray, cols: np.ndarray):
    if rows.size == 0 or cols.size == 0:
        return None
    view = np.ix_(rows, cols)
    live = available[view] & np.isfinite(clean[view])
    if not np.any(live):
        return None
    return float(np.median(clean[view][live]))


def _side(present: bool, median_c: float | None, n_columns: int) -> dict:
    if not present:
        if median_c is not None or n_columns != 0:
            raise ValueError("an absent side has no median and no columns")
        return {"present": False, "median_c": None, "n_columns": 0}
    return {"present": True, "median_c": median_c, "n_columns": int(n_columns)}


def _meter_columns(x_m: np.ndarray, left: float, right: float) -> tuple[int, int]:
    c0 = int(np.searchsorted(x_m, float(left) - 1e-3, side="left"))
    c1 = int(np.searchsorted(x_m, float(right) + 1e-3, side="right"))
    c0 = max(0, min(c0, len(x_m) - 1))
    c1 = max(c0 + 1, min(c1, len(x_m)))
    return c0, c1


def _time_rows(time_s: np.ndarray, t_start: float, t_end: float) -> tuple[int, int]:
    t0 = int(np.searchsorted(time_s, float(t_start), side="left"))
    t1 = int(np.searchsorted(time_s, float(t_end), side="right"))
    t1 = max(t0 + 1, min(t1, len(time_s)))
    return t0, t1


def wide_channel_from_measurements(
    halo_inside_fraction: float,
    rectangle_median_c: float | None,
    far_cable_median_c: float | None,
) -> dict:
    """Soft flag «широкий канал». Both cutoffs are the module constants.

    True only when the 8–40 m ring lies mostly inside the rectangle and the
    rectangle median differs from the cable farther than 40 m. A missing far
    median cannot fire and is not replaced by the rectangle median. The
    candidate name is not an operational class and does not set
    ``anomaly_status``.
    """
    gap = None
    medians_live = (
        rectangle_median_c is not None
        and far_cable_median_c is not None
        and math.isfinite(rectangle_median_c)
        and math.isfinite(far_cable_median_c)
    )
    share_live = math.isfinite(halo_inside_fraction)
    if medians_live and share_live:
        gap = float(rectangle_median_c) - float(far_cable_median_c)
    share_high = share_live and float(halo_inside_fraction) >= WIDE_CHANNEL_HALO_INSIDE_MIN
    gap_noticeable = gap is not None and abs(gap) >= WIDE_CHANNEL_FAR_GAP_C
    flagged = bool(share_high and gap_noticeable)
    return {
        "wide_channel": flagged,
        "wide_channel_candidate": WIDE_CHANNEL_CANDIDATE if flagged else None,
        "far_gap_c": None if gap is None else float(gap),
        "wide_channel_halo_inside_min": float(WIDE_CHANNEL_HALO_INSIDE_MIN),
        "wide_channel_far_gap_min_c": float(WIDE_CHANNEL_FAR_GAP_C),
    }


def _boxes_overlap(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1] and left[2] < right[3] and right[2] < left[3]


def _zone_box(zone: dict, x_m: np.ndarray, time_s: np.ndarray) -> tuple[int, int, int, int]:
    t0, t1 = _time_rows(time_s, zone["t_start_s"], zone["t_end_s"])
    c0, c1 = _meter_columns(x_m, zone["start_m"], zone["end_m"])
    return (t0, t1, c0, c1)


def _zone_public(zone: dict) -> dict:
    return {
        "shape_hypothesis": zone.get("shape_hypothesis"),
        "contrast": float(zone["contrast"]),
        "start_m": float(zone["start_m"]),
        "end_m": float(zone["end_m"]),
        "t_start_s": float(zone["t_start_s"]),
        "t_end_s": float(zone["t_end_s"]),
        "polarity": zone.get("polarity"),
        "expert_confirmed": bool(zone.get("expert_confirmed")),
    }


def context_evidence(
    field: dict,
    x_m,
    time_s,
    *,
    rows_half_open: tuple[int, int] | None = None,
    x_span: tuple[float, float] | None = None,
    zone: dict | None = None,
    inner_m: float = 8.0,
    outer_m: float = 40.0,
) -> dict:
    """Median, far cable, halo-inside fraction, flank zones, and the soft flag.

    Pass a marked rectangle as ``rows_half_open`` and ``x_span``, or pass one
    zone already returned by ``analyze_field``. The ring is the columns
    8–40 m from the box center. The far cable is the same time rows more than
    40 m outside the box. A side with no such columns is absent. Flank zones
    meet the adjacent cable within 40 m and do not enter the box. The
    wide-channel flag reads those measurements and does not change the field.
    """
    if not (np.isfinite(inner_m) and np.isfinite(outer_m) and 0 < inner_m < outer_m):
        raise ValueError("need 0 < inner_m < outer_m")
    if (zone is None) == (rows_half_open is None or x_span is None):
        raise ValueError("pass a rectangle or a zone, not both and not neither")
    clean = np.asarray(field["clean"], dtype=np.float64)
    available = np.asarray(field["score_available"], dtype=bool)
    zones = list(field["zones"])
    x = np.asarray(x_m, dtype=np.float64)
    times = np.asarray(time_s, dtype=np.float64)
    if clean.shape != available.shape or clean.ndim != 2:
        raise ValueError("clean and score_available must share a [time, distance] shape")
    if clean.shape != (len(times), len(x)):
        raise ValueError("x_m and time_s must match the scored field")
    if zone is not None:
        t0, t1, c0, c1 = _zone_box(zone, x, times)
        x0 = float(zone["start_m"])
        x1 = float(zone["end_m"])
        rows_half_open = (t0, t1)
        x_span = (x0, x1)
    else:
        t0, t1 = int(rows_half_open[0]), int(rows_half_open[1])
        x0, x1 = float(x_span[0]), float(x_span[1])
        if not (0 <= t0 < t1 <= clean.shape[0]) or not (x0 < x1):
            raise ValueError("rectangle rows and meters must be a non-empty span")
        c0, c1 = _meter_columns(x, x0, x1)
    rect = (t0, t1, c0, c1)
    rows = np.arange(t0, t1)
    rect_cols = np.arange(c0, c1)
    rectangle_median_c = _median(clean, available, rows, rect_cols)

    far_left_cols = np.flatnonzero(x < x0 - outer_m)
    far_right_cols = np.flatnonzero(x > x1 + outer_m)
    far_left_median = _median(clean, available, rows, far_left_cols)
    far_right_median = _median(clean, available, rows, far_right_cols)
    far_cols = np.concatenate([far_left_cols, far_right_cols]) if far_left_cols.size or far_right_cols.size else np.array([], dtype=int)
    far_cable_median_c = _median(clean, available, rows, far_cols)
    if far_left_cols.size == 0 and far_left_median is not None:
        raise ValueError("absent far-left side was given a median")
    if far_right_cols.size == 0 and far_right_median is not None:
        raise ValueError("absent far-right side was given a median")
    if far_cable_median_c is not None and rectangle_median_c is not None:
        if far_left_cols.size == 0 and far_right_cols.size == 0:
            raise ValueError("far cable median was filled without a far side")

    center = 0.5 * (x0 + x1)
    halo = (np.abs(x - center) >= inner_m) & (np.abs(x - center) <= outer_m)
    halo_n = int(halo.sum())
    if halo_n == 0:
        raise ValueError("the 8–40 m ring has no columns on this cable")
    inside = (x >= x0) & (x <= x1)
    halo_inside = int(np.count_nonzero(halo & inside))
    halo_inside_fraction = float(halo_inside / halo_n)

    left_cols = np.flatnonzero((x >= x0 - outer_m) & (x < x[c0]))
    right_edge = x[c1 - 1]
    right_cols = np.flatnonzero((x > right_edge) & (x <= x1 + outer_m))
    left_box = (t0, t1, int(left_cols[0]), int(left_cols[-1]) + 1) if left_cols.size else None
    right_box = (t0, t1, int(right_cols[0]), int(right_cols[-1]) + 1) if right_cols.size else None
    flank_zones = []
    in_box_zones = []
    for item in zones:
        box = _zone_box(item, x, times)
        enters = _boxes_overlap(rect, box)
        on_flank = (left_box is not None and _boxes_overlap(left_box, box)) or (
            right_box is not None and _boxes_overlap(right_box, box)
        )
        if enters:
            in_box_zones.append(_zone_public(item))
        elif on_flank:
            flank_zones.append(_zone_public(item))
    order = lambda item: (item["start_m"], item["end_m"], item["t_start_s"], item["contrast"])
    flank_zones.sort(key=order)
    in_box_zones.sort(key=order)
    channel = wide_channel_from_measurements(
        halo_inside_fraction, rectangle_median_c, far_cable_median_c,
    )
    return {
        "rows_half_open": [int(rows_half_open[0]), int(rows_half_open[1])],
        "x_m": [x0, x1],
        "inner_m": float(inner_m),
        "outer_m": float(outer_m),
        "rectangle_median_c": rectangle_median_c,
        "far_cable_median_c": far_cable_median_c,
        "far_left": _side(far_left_cols.size > 0, far_left_median, int(far_left_cols.size)),
        "far_right": _side(far_right_cols.size > 0, far_right_median, int(far_right_cols.size)),
        "halo_inside_fraction": halo_inside_fraction,
        "halo_columns": halo_n,
        "halo_columns_inside": halo_inside,
        "flank_zones": flank_zones,
        "in_box_zones": in_box_zones,
        **channel,
    }
