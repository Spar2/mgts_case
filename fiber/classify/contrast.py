"""Shoulder contrast of a temporal excess.

On 12000 scored windows from records 1..6, seed 4, and no witness boxes, the
28-row excess has absolute p90 = 1.735 °C and p95 = 2.796 °C. The 40-column
shoulder has p90 = 0.479 °C, p95 = 0.789 °C and p99 = 2.164 °C. Absolute
contrast has p50 = 0.370 °C and p90 = 1.132 °C. Duty at 1 °C inside the crop
has p50 = 0.25 and p90 = 0.893, and the 1 °C width has p90 = 2.843 m,
p95 = 7 m and p99 = 31.468 m. Heating here is a local excess in that tail.
A shift the neighbors share stays background except at its edge. Witness
coordinates are not an input.
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
from fiber.classify.excess import long_run_median
from fiber.classify.plant import LABEL_INDEX, SoftmaxSceneModel, _stamp
from fiber.classify.rules import LABELS
from fiber.models.patch_classifier import SPACE_CROP, TIME_CROP
from fiber.quality import SENTINEL_VALUE

# Published record means. They were not measured on witness boxes.
CONTRAST_LEVEL_C = 22.746278285980225
CONTRAST_INVALID_SHARE = 0.047769979141731485
# Spatial peak p90 on the earlier 15000-window subsample.
CONTRAST_STANDING_AMP_C = 2.79
# Absolute 28-row excess, p90 to p95 of this shoulder measurement.
CONTRAST_HEAT_C = (1.74, 2.80)
CONTRAST_COOL_C = (1.74, 2.80)
# Overlap of that excess tail with the shoulder tail, which ends at p99.
CONTRAST_BROAD_C = (1.74, 2.16)
# Spike p50 to p90 from the global context subsample. Four rows stay under
# the 7-row duty body, so the 28-row median does not move.
CONTRAST_PERIOD_C = (4.5, 6.5)
CONTRAST_PERIOD_ROWS = 4
CONTRAST_PERIOD_COUNT = 3
# Above the global spike p99 of 14.2 °C.
CONTRAST_FAULT_C = (14.0, 18.0)
# Duty p50 at 1 °C is 0.25 of a 28-row crop.
CONTRAST_DECOY_ROWS = 7
# Fills every 28-row crop that sits inside it. 56/140 is under half the
# trace, so the column median stays on the unheated samples. Duty p95 is 1.
CONTRAST_ON = (28, 84)
CONTRAST_T = 140
CONTRAST_X = 640
# 1 °C width p90 to p95, in metres. The generator plants one column per metre.
CONTRAST_HEAT_WIDTH_M = (3, 7)
# Above the 1 °C width p99, and wider than the crop plus both locked sides.
CONTRAST_BROAD_WIDTH_M = 160
CONTRAST_DUTY_C = 1.0
# Shoulder p90, and absolute-contrast p50. They split background rows only.
CONTRAST_SHOULDER_P90 = 0.479
CONTRAST_ABS_P50 = 0.370
CONTRAST_TRAIN_SEEDS = tuple(range(1_500_000, 1_500_024))
CONTRAST_STEPS = 500
CONTRAST_PER_EVENT = 128
CONTRAST_FEATURES = CONTEXT_FEATURES + ("excess", "shoulder", "contrast", "duty")


def _median_finite(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return float(np.median(finite))


def _column_excess(residual: np.ndarray, t0: int) -> np.ndarray:
    slab = residual[t0:t0 + TIME_CROP]
    out = np.full(slab.shape[1], np.nan, dtype=np.float64)
    live = np.isfinite(slab).any(axis=0)
    if live.any():
        out[live] = np.nanmedian(slab[:, live], axis=0)
    return out


def _window_extra(excess: np.ndarray, residual: np.ndarray, t0: int, x0: int) -> tuple[float, float, float, float]:
    """Signed excess, shoulder level, their difference, and duty at 1 °C."""
    n_space = int(excess.shape[0])
    center = excess[x0:x0 + SPACE_CROP]
    finite = np.isfinite(center)
    if not finite.any():
        return 0.0, 0.0, 0.0, 0.0
    values = center[finite]
    pick = int(np.argmax(np.abs(values)))
    peak = float(values[pick])
    local = int(np.flatnonzero(finite)[pick])
    left = excess[max(0, x0 - CONTEXT_SIDE):x0]
    right = excess[x0 + SPACE_CROP:min(n_space, x0 + SPACE_CROP + CONTEXT_SIDE)]
    wings = np.concatenate((left, right)) if left.size or right.size else np.zeros(0, dtype=np.float64)
    shoulder = _median_finite(wings)
    if shoulder is None:
        shoulder = 0.0
    column = x0 + local
    series = residual[t0:t0 + TIME_CROP, column]
    series = series[np.isfinite(series)]
    duty = float(np.mean(np.abs(series) >= CONTRAST_DUTY_C)) if series.size else 0.0
    return peak, float(shoulder), peak - float(shoulder), duty


def contrast_table(field: np.ndarray, valid: np.ndarray, origins: np.ndarray) -> np.ndarray:
    """Context features plus excess, shoulder, contrast, and duty.

    An empty origin list has shape (0, 9).
    """
    origins = np.asarray(origins, dtype=np.int64)
    context = context_table(field, valid, origins)
    extra = np.zeros((len(origins), 4), dtype=np.float64)
    if len(origins) == 0:
        return np.column_stack((context, extra))
    baseline = long_run_median(field, valid)
    residual = np.where(
        np.asarray(valid, dtype=bool) & np.isfinite(baseline),
        np.asarray(field, dtype=np.float64) - baseline,
        np.nan,
    )
    cache: dict[int, np.ndarray] = {}
    for index, (t0, x0) in enumerate(origins):
        t0 = int(t0)
        if t0 not in cache:
            cache[t0] = _column_excess(residual, t0)
        extra[index] = _window_extra(cache[t0], residual, t0, int(x0))
    return np.column_stack((context, extra))


def plant_contrast(seed: int = 0) -> dict:
    """Standing bow, a shared shift, and one local excess per signed class.

    Labels are the painted cells. A 7-row burst is not an event. The shared
    shift is wider than both locked shoulders, so an interior window keeps
    the label Background. Event bands stay 40 columns apart.
    """
    rng = np.random.default_rng(int(seed))
    n_dead = int(round(CONTRAST_INVALID_SHARE * CONTRAST_X))
    n_live = CONTRAST_X - n_dead
    x_m = np.arange(CONTRAST_X, dtype=np.float64)
    time_s = np.arange(CONTRAST_T, dtype=np.float64) * 60.0
    field = np.full((CONTRAST_T, CONTRAST_X), SENTINEL_VALUE, dtype=np.float32)
    valid = np.zeros(field.shape, dtype=bool)
    valid[:, :n_live] = True
    x = np.arange(n_live, dtype=np.float64)
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    wave = np.sin(2.0 * np.pi * x / float(CONTRAST_X) + phase)
    wave -= wave.mean()
    peak = float(np.max(np.abs(wave)))
    if peak > 0.0:
        wave *= CONTRAST_STANDING_AMP_C / peak
    noise = 0.12 + 0.03 * (int(seed) % 3)
    field[:, :n_live] = (CONTRAST_LEVEL_C + wave).astype(np.float32)
    field[:, :n_live] += rng.normal(0.0, noise, size=(CONTRAST_T, n_live)).astype(np.float32)
    labels = np.zeros(field.shape, dtype=np.int8)
    event_id = np.zeros(field.shape, dtype=np.int16)

    for start in (0, 100):
        column = 560 + int(rng.integers(0, 8))
        field[start:start + CONTRAST_DECOY_ROWS, column] += np.float32(rng.uniform(*CONTRAST_HEAT_C))

    on0, on1 = CONTRAST_ON
    heat_x = 16 + int(rng.integers(0, 9))
    heat_w = int(rng.integers(CONTRAST_HEAT_WIDTH_M[0], CONTRAST_HEAT_WIDTH_M[1] + 1))
    _stamp(
        field, labels, event_id, on0, on1, heat_x, heat_x + heat_w,
        float(rng.uniform(*CONTRAST_HEAT_C)), LABEL_INDEX["Local_Heating"], 1,
    )
    cool_x = 100 + int(rng.integers(0, 9))
    cool_w = int(rng.integers(CONTRAST_HEAT_WIDTH_M[0], CONTRAST_HEAT_WIDTH_M[1] + 1))
    _stamp(
        field, labels, event_id, on0, on1, cool_x, cool_x + cool_w,
        -float(rng.uniform(*CONTRAST_COOL_C)), LABEL_INDEX["Local_Cooling"], 2,
    )
    period_x = 200 + int(rng.integers(0, 9))
    period_amp = float(rng.uniform(*CONTRAST_PERIOD_C))
    for row in (8, 48, 110):
        _stamp(
            field, labels, event_id, row, row + CONTRAST_PERIOD_ROWS, period_x, period_x + 8,
            period_amp, LABEL_INDEX["Periodic_Thermal"], 3,
        )
    fault_x = 300 + int(rng.integers(0, 9))
    _stamp(
        field, labels, event_id, 30, 31, fault_x, fault_x + 1,
        float(rng.uniform(*CONTRAST_FAULT_C)), LABEL_INDEX["Sensor_Fault"], 4,
    )
    _stamp(
        field, labels, event_id, on0, on1, 360, 360 + CONTRAST_BROAD_WIDTH_M,
        float(rng.uniform(*CONTRAST_BROAD_C)), LABEL_INDEX["Broad_Shift"], 5,
    )
    labels[~valid] = 0
    event_id[~valid] = 0
    shift = np.float32(CONTRAST_LEVEL_C - float(np.median(field[valid])))
    field[valid] = field[valid] + shift
    events = (
        {"id": 1, "label": "Local_Heating"},
        {"id": 2, "label": "Local_Cooling"},
        {"id": 3, "label": "Periodic_Thermal"},
        {"id": 4, "label": "Sensor_Fault"},
        {"id": 5, "label": "Broad_Shift"},
    )
    return {
        "kind": "contrast",
        "label_source": "generator",
        "field": field,
        "valid": valid,
        "x_m": x_m,
        "time_s": time_s,
        "labels": labels,
        "event_id": event_id,
        "events": events,
    }


def dump_contrast_model(model: SoftmaxSceneModel, extra: dict | None = None) -> dict:
    if model.weight is None or model.mu is None or model.sig is None or model.bias is None:
        raise RuntimeError("model is not fit")
    payload = {
        "family": "softmax_contrast",
        "features": list(CONTRAST_FEATURES),
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


def load_contrast_model(payload: dict) -> SoftmaxSceneModel:
    if list(payload["features"]) != list(CONTRAST_FEATURES):
        raise ValueError("contrast feature list does not match this code")
    if int(payload["context_side"]) != CONTEXT_SIDE or float(payload["peak_fraction"]) != CONTEXT_PEAK_FRACTION:
        raise ValueError("context constants do not match this code")
    model = SoftmaxSceneModel()
    model.mu = np.asarray(payload["mu"], dtype=np.float64)
    model.sig = np.asarray(payload["sig"], dtype=np.float64)
    model.weight = np.asarray(payload["weight"], dtype=np.float64)
    model.bias = np.asarray(payload["bias"], dtype=np.float64)
    return model
