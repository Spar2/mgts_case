"""Synthetic cables whose temperature leaves a cold baseline, on a metre grid.

A column's cold baseline is the median of the samples at or below that
column's own median. A long hot stretch then stays warm in the residual,
and a short hump does too. Windows are 28 profiles by 64 metres.
"""

from __future__ import annotations

import numpy as np

from fiber.classify.rules import LABELS
from fiber.classify.scan import LABEL_INDEX, TIME_STRIDE
from fiber.models.patch_classifier import TIME_CROP

METRE_CROP = 64
RESIDUAL_T = 240
RESIDUAL_X = 680
RESIDUAL_DX_M = 1.0
SPACE_STRIDE_M = 16
TRAIN_SEEDS = tuple(range(3_000_000, 3_000_024))
EVAL_SEEDS = tuple(range(3_100_000, 3_100_004))


def cold_baseline(field: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Median of the cooler half of each column. Empty columns stay NaN."""
    data = np.where(valid, np.asarray(field, dtype=np.float64), np.nan)
    ok = np.isfinite(data).any(axis=0)
    center = np.full(data.shape[1], np.nan, dtype=np.float64)
    base = np.full(data.shape[1], np.nan, dtype=np.float64)
    if ok.any():
        center[ok] = np.nanmedian(data[:, ok], axis=0)
        below = np.where(np.isfinite(data[:, ok]) & (data[:, ok] <= center[ok]), data[:, ok], np.nan)
        base[ok] = np.nanmedian(below, axis=0)
    return np.where(np.isfinite(base), base, center)


def residual_field(field: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Warmth above the cold baseline. Empty cells stay NaN."""
    base = cold_baseline(field, valid)
    out = np.asarray(field, dtype=np.float64) - base
    return np.where(valid & np.isfinite(out), out, np.nan)


def residual_pair(field: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Two pictures: against the column median, and against the cold baseline."""
    data = np.where(valid, np.asarray(field, dtype=np.float64), np.nan)
    ok = np.isfinite(data).any(axis=0)
    center = np.full(data.shape[1], np.nan, dtype=np.float64)
    base = cold_baseline(field, valid)
    if ok.any():
        center[ok] = np.nanmedian(data[:, ok], axis=0)
    from_median = np.where(valid, data - center, np.nan)
    from_cold = np.where(valid, data - base, np.nan)
    return np.stack((from_median, from_cold), axis=0)


def _metre_index(x_m: np.ndarray, x0: int) -> np.ndarray:
    start = float(x_m[int(x0)])
    targets = start + np.arange(METRE_CROP, dtype=np.float64)
    index = np.searchsorted(x_m, targets, side="left")
    return np.clip(index, 0, len(x_m) - 1), targets


def meter_crops(field: np.ndarray, valid: np.ndarray, x_m: np.ndarray, origins: np.ndarray) -> np.ndarray:
    """One 2×28×64 image per origin. Missing metres stay 0."""
    pair = residual_pair(field, valid)
    origins = np.asarray(origins, dtype=np.int64)
    out = np.zeros((len(origins), 2, TIME_CROP, METRE_CROP), dtype=np.float32)
    x_m = np.asarray(x_m, dtype=np.float64)
    for row, (t0, x0) in enumerate(origins):
        index, targets = _metre_index(x_m, int(x0))
        near = np.abs(x_m[index] - targets) <= 1.5
        for channel in range(2):
            slab = pair[channel, int(t0):int(t0) + TIME_CROP][:, index]
            mask = valid[int(t0):int(t0) + TIME_CROP][:, index] & near
            good = mask & np.isfinite(slab)
            piece = np.zeros(slab.shape, dtype=np.float32)
            piece[good] = slab[good].astype(np.float32)
            out[row, channel] = piece
    return out


def _stamp(field, labels, t0, t1, x0, x1, value, class_id) -> None:
    field[t0:t1, x0:x1] = field[t0:t1, x0:x1] + np.float32(value)
    if class_id:
        labels[t0:t1, x0:x1] = class_id


def plant_residual(seed: int) -> dict:
    """One cable. Events sit more than 64 m apart, so one window holds one of them."""
    rng = np.random.default_rng(int(seed))
    x_m = np.arange(RESIDUAL_X, dtype=np.float64) * RESIDUAL_DX_M
    profile = (2.2 * np.sin(2.0 * np.pi * x_m / RESIDUAL_X)).astype(np.float32)
    noise = rng.normal(0.0, 0.35, size=(RESIDUAL_T, RESIDUAL_X)).astype(np.float32)
    field = profile + noise
    labels = np.zeros((RESIDUAL_T, RESIDUAL_X), dtype=np.int16)
    heat = LABEL_INDEX["Local_Heating"]
    cool = LABEL_INDEX["Local_Cooling"]
    broad = LABEL_INDEX["Broad_Shift"]
    period = LABEL_INDEX["Periodic_Thermal"]
    fault = LABEL_INDEX["Sensor_Fault"]
    _stamp(field, labels, 40, 100, 40, 54, float(rng.uniform(3.5, 7.0)), heat)
    _stamp(field, labels, 80, 92, 140, 154, float(rng.uniform(-8.0, -4.0)), cool)
    for row in (20, 80, 140, 190):
        _stamp(field, labels, row, row + 4, 220, 228, float(rng.uniform(2.5, 4.5)), period)
    _stamp(field, labels, 30, 100, 310, 380, float(rng.uniform(2.5, 4.5)), broad)
    _stamp(field, labels, 20, 90, 460, 484, float(rng.uniform(3.0, 6.0)), heat)
    _stamp(field, labels, 120, RESIDUAL_T, 560, 568, float(rng.uniform(4.0, 8.0)), heat)
    field[30, 630] = field[30, 630] + np.float32(16.0)
    labels[30, 630] = fault
    valid = np.ones(field.shape, dtype=bool)
    valid[:, 640:] = False
    labels[:, 640:] = 0
    return {
        "field": field,
        "valid": valid,
        "labels": labels,
        "x_m": x_m,
        "kind": "residual",
    }


def residual_windows(labels: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Windows of 28 by 64 m. The class is the planted label that owns the window."""
    n_time, n_space = labels.shape
    if n_time < TIME_CROP or n_space < METRE_CROP:
        return np.zeros((0, 2), dtype=np.int64), []
    time_starts = list(range(0, n_time - TIME_CROP + 1, TIME_STRIDE))
    space_starts = list(range(0, n_space - METRE_CROP + 1, SPACE_STRIDE_M))
    origins = []
    names = []
    for t0 in time_starts:
        for x0 in space_starts:
            lab = labels[t0:t0 + TIME_CROP, x0:x0 + METRE_CROP]
            mask = valid[t0:t0 + TIME_CROP, x0:x0 + METRE_CROP]
            if int(mask.sum()) * 2 < TIME_CROP * METRE_CROP:
                continue
            counts = {
                name: int(np.count_nonzero((lab == LABEL_INDEX[name]) & mask))
                for name in LABELS
                if name != "Background"
            }
            if counts["Sensor_Fault"] >= 1:
                name = "Sensor_Fault"
            else:
                name = max(counts, key=counts.get)
                if counts[name] < 8:
                    name = "Background"
            origins.append((t0, x0))
            names.append(name)
    if not origins:
        return np.zeros((0, 2), dtype=np.int64), []
    return np.asarray(origins, dtype=np.int64), names
