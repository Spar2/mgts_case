"""Temporal excess over each column's own median.

Global subsamples of records 1..6, not the witness boxes, put the spatial
peak p90 at 2.79 °C while the temporal excess p95 is 0.93 °C and the p99 is
2.89 °C. Most of a spatial peak is a standing profile. Heating in this
generator is a time-limited stamp whose excess lands in that observed tail.
Witness coordinates are not an input.
"""

from __future__ import annotations

import numpy as np

from fiber.classify.context import (
    CONTEXT_PEAK_FRACTION,
    CONTEXT_SIDE,
    FEATURE_NAMES as CONTEXT_FEATURES,
    SPIKE_MAD_FLOOR,
    context_table,
)
from fiber.classify.plant import LABEL_INDEX, SoftmaxSceneModel, _stamp
from fiber.classify.rules import LABELS
from fiber.models.patch_classifier import SPACE_CROP, TIME_CROP
from fiber.quality import SENTINEL_VALUE

# Published record means. They were not measured on witness boxes.
EXCESS_LEVEL_C = 22.746278285980225
EXCESS_INVALID_SHARE = 0.047769979141731485
# Spatial peak p90 on 15000 scored windows, subsample seed 0.
EXCESS_STANDING_AMP_C = 2.79
# Excess of an on-block equals the stamp while the block stays under half
# the trace. The draw sits between the global excess p95 and p99.
EXCESS_HEAT_C = (1.6, 2.4)
EXCESS_COOL_C = (1.6, 2.4)
EXCESS_BROAD_C = (1.6, 2.4)
EXCESS_PERIOD_C = (4.5, 6.5)
EXCESS_FAULT_C = (14.0, 18.0)
EXCESS_RIDGE_C = (2.0, 2.79)
# 28-row median of the 0.18 °C texture noise is about 0.04 °C.
# Support below this floor is the standing profile, not a stamp.
EXCESS_FLOOR_C = 0.25
# One cable-scale bow. Several shorter periods made spatial runs look like
# Periodic_Thermal on held-out synthetic cables, so they are not the profile.
EXCESS_KINDS = (
    {"noise": 0.12, "periods": (480,)},
    {"noise": 0.15, "periods": (480,)},
    {"noise": 0.18, "periods": (480,)},
)
EXCESS_T = 140
EXCESS_X = 480
# Rows [28, 84) cover three full 28-row crops on the stride-14 grid.
EXCESS_ON = (28, 84)
EXCESS_TRAIN_SEEDS = tuple(range(1_300_000, 1_300_024))
EXCESS_STEPS = 500
EXCESS_PER_EVENT = 128
EXCESS_FEATURES = CONTEXT_FEATURES + ("excess", "excess_support", "row_share")


def long_run_median(field: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Median temperature of each column. Invalid samples stay out."""
    mask = np.asarray(valid, dtype=bool)
    masked = np.where(mask, np.asarray(field, dtype=np.float64), np.nan)
    live = mask.any(axis=0)
    out = np.full(masked.shape[1], np.nan, dtype=np.float64)
    if live.any():
        out[live] = np.nanmedian(masked[:, live], axis=0)
    return out


def _window_extra(field: np.ndarray, valid: np.ndarray, t0: int, x0: int, baseline: np.ndarray) -> tuple[float, float, float]:
    """Signed excess, how many columns carry it, and the share of hot rows."""
    slab = np.asarray(field[t0:t0 + TIME_CROP, x0:x0 + SPACE_CROP], dtype=np.float64)
    mask = np.asarray(valid[t0:t0 + TIME_CROP, x0:x0 + SPACE_CROP], dtype=bool)
    count = mask.sum(axis=0)
    live = count > 0
    excess = 0.0
    support = 0.0
    if live.any():
        kept = np.where(mask[:, live], slab[:, live], np.nan)
        column = np.nanmedian(kept, axis=0)
        delta = column - np.asarray(baseline[x0:x0 + SPACE_CROP], dtype=np.float64)[live]
        finite = np.isfinite(delta)
        if finite.any():
            values = delta[finite]
            excess = float(values[int(np.argmax(np.abs(values)))])
            if abs(excess) >= EXCESS_FLOOR_C:
                support = float(np.count_nonzero(np.abs(values) >= CONTEXT_PEAK_FRACTION * abs(excess)))
    if not mask.any():
        return excess, support, 0.0
    center = float(np.median(slab[mask]))
    dev = np.where(mask, slab - center, np.nan)
    peak_abs = float(np.nanmax(np.abs(dev)))
    if peak_abs < 1e-8:
        return excess, support, 0.0
    row_peak = np.nanmax(np.abs(dev), axis=1)
    row_share = float(np.mean(np.nan_to_num(row_peak, nan=0.0) >= CONTEXT_PEAK_FRACTION * peak_abs))
    return excess, support, row_share


def excess_table(field: np.ndarray, valid: np.ndarray, origins: np.ndarray) -> np.ndarray:
    """Context features, temporal excess, its column count, and row share.

    `excess` is the crop's column-median deviation from that column's
    all-time median, at the column where the absolute deviation is largest.
    An empty origin list has shape (0, 8).
    """
    origins = np.asarray(origins, dtype=np.int64)
    context = context_table(field, valid, origins)
    extra = np.zeros((len(origins), 3), dtype=np.float64)
    if len(origins) == 0:
        return np.column_stack((context, extra))
    baseline = long_run_median(field, valid)
    for index, (t0, x0) in enumerate(origins):
        extra[index] = _window_extra(field, valid, int(t0), int(x0), baseline)
    return np.column_stack((context, extra))


def plant_excess(seed: int = 0) -> dict:
    """Standing profile plus time-limited events. Labels are the painted cells.

    A full-time ridge shares the spatial amplitude of a heat stamp and is
    left unlabeled, so a spatial peak alone is not the heating teacher.
    Event bands stay 40 columns apart.
    """
    rng = np.random.default_rng(int(seed))
    kind = EXCESS_KINDS[int(seed) % len(EXCESS_KINDS)]
    n_dead = int(round(EXCESS_INVALID_SHARE * EXCESS_X))
    n_live = EXCESS_X - n_dead
    x_m = np.arange(EXCESS_X, dtype=np.float64)
    time_s = np.arange(EXCESS_T, dtype=np.float64) * 60.0
    field = np.full((EXCESS_T, EXCESS_X), SENTINEL_VALUE, dtype=np.float32)
    valid = np.zeros(field.shape, dtype=bool)
    valid[:, :n_live] = True
    x = np.arange(n_live, dtype=np.float64)
    wave = np.zeros(n_live, dtype=np.float64)
    for period in kind["periods"]:
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        weight = abs(float(rng.normal())) + 0.2
        wave += weight * np.sin(2.0 * np.pi * x / float(period) + phase)
    wave -= wave.mean()
    peak = float(np.max(np.abs(wave)))
    if peak > 0.0:
        wave *= EXCESS_STANDING_AMP_C / peak
    field[:, :n_live] = (EXCESS_LEVEL_C + wave).astype(np.float32)
    field[:, :n_live] += rng.normal(0.0, float(kind["noise"]), size=(EXCESS_T, n_live)).astype(np.float32)
    labels = np.zeros(field.shape, dtype=np.int8)
    event_id = np.zeros(field.shape, dtype=np.int16)

    ridge_w = int(rng.integers(4, 11))
    ridge_amp = float(rng.uniform(*EXCESS_RIDGE_C))
    field[:, 430:430 + ridge_w] += np.float32(ridge_amp)

    on0, on1 = EXCESS_ON
    heat_x = 16 + int(rng.integers(0, 16))
    heat_w = int(rng.integers(6, 13))
    _stamp(
        field, labels, event_id, on0, on1, heat_x, heat_x + heat_w,
        float(rng.uniform(*EXCESS_HEAT_C)), LABEL_INDEX["Local_Heating"], 1,
    )
    cool_x = 100 + int(rng.integers(0, 16))
    cool_w = int(rng.integers(6, 13))
    _stamp(
        field, labels, event_id, on0, on1, cool_x, cool_x + cool_w,
        -float(rng.uniform(*EXCESS_COOL_C)), LABEL_INDEX["Local_Cooling"], 2,
    )
    period_x = 190 + int(rng.integers(0, 16))
    period_amp = float(rng.uniform(*EXCESS_PERIOD_C))
    for row in (6, 50, 110):
        _stamp(
            field, labels, event_id, row, row + 4, period_x, period_x + 6,
            period_amp, LABEL_INDEX["Periodic_Thermal"], 3,
        )
    fault_x = 270 + int(rng.integers(0, 11))
    _stamp(
        field, labels, event_id, 20, 21, fault_x, fault_x + 1,
        float(rng.uniform(*EXCESS_FAULT_C)), LABEL_INDEX["Sensor_Fault"], 4,
    )
    broad_amp = float(rng.uniform(*EXCESS_BROAD_C))
    _stamp(
        field, labels, event_id, on0, on1, 340, 390,
        broad_amp, LABEL_INDEX["Broad_Shift"], 5,
    )
    labels[~valid] = 0
    event_id[~valid] = 0
    shift = np.float32(EXCESS_LEVEL_C - float(np.median(field[valid])))
    field[valid] = field[valid] + shift
    events = (
        {"id": 1, "label": "Local_Heating"},
        {"id": 2, "label": "Local_Cooling"},
        {"id": 3, "label": "Periodic_Thermal"},
        {"id": 4, "label": "Sensor_Fault"},
        {"id": 5, "label": "Broad_Shift"},
    )
    return {
        "kind": "excess",
        "texture_index": int(seed) % len(EXCESS_KINDS),
        "label_source": "generator",
        "field": field,
        "valid": valid,
        "x_m": x_m,
        "time_s": time_s,
        "labels": labels,
        "event_id": event_id,
        "events": events,
    }


def dump_excess_model(model: SoftmaxSceneModel, extra: dict | None = None) -> dict:
    if model.weight is None or model.mu is None or model.sig is None or model.bias is None:
        raise RuntimeError("model is not fit")
    payload = {
        "family": "softmax_excess",
        "features": list(EXCESS_FEATURES),
        "context_side": CONTEXT_SIDE,
        "peak_fraction": CONTEXT_PEAK_FRACTION,
        "spike_mad_floor": SPIKE_MAD_FLOOR,
        "labels": list(LABELS),
        "mu": model.mu.tolist(),
        "sig": model.sig.tolist(),
        "weight": model.weight.tolist(),
        "bias": model.bias.tolist(),
    }
    if extra:
        payload.update(extra)
    return payload


def load_excess_model(payload: dict) -> SoftmaxSceneModel:
    if list(payload["features"]) != list(EXCESS_FEATURES):
        raise ValueError("excess feature list does not match this code")
    if int(payload["context_side"]) != CONTEXT_SIDE or float(payload["peak_fraction"]) != CONTEXT_PEAK_FRACTION:
        raise ValueError("context constants do not match this code")
    model = SoftmaxSceneModel()
    model.mu = np.asarray(payload["mu"], dtype=np.float64)
    model.sig = np.asarray(payload["sig"], dtype=np.float64)
    model.weight = np.asarray(payload["weight"], dtype=np.float64)
    model.bias = np.asarray(payload["bias"], dtype=np.float64)
    return model
