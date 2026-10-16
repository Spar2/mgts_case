"""Absolute excess against the column's own MAD.

On 12000 scored windows from records 1..6, seed 7, and no witness boxes,
column MAD has p10 = 0.21 °C and p50 = 0.55 °C. The joint tail where |z|
is at least 3.6 and the absolute excess is at least 1.74 °C has excess
p10 = 2.19 °C and p50 = 3.88 °C. Columns at the MAD floor stay near
0.24–0.37 °C. Heating is that joint tail. A stable column is not.
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
from fiber.classify.excess import long_run_median
from fiber.classify.plant import LABEL_INDEX, SoftmaxSceneModel, _stamp
from fiber.classify.rules import LABELS
from fiber.models.patch_classifier import SPACE_CROP, TIME_CROP
from fiber.quality import SENTINEL_VALUE

# Published record means. They were not measured on witness boxes.
STABILITY_LEVEL_C = 22.746278285980225
STABILITY_INVALID_SHARE = 0.047769979141731485
# Context spatial-peak p50. A standing bow of this size is ordinary.
STABILITY_STANDING_C = 0.48
# MAD of a normal column, and of a column at the measured p10.
STABILITY_MAD_P50 = 0.55
STABILITY_MAD_P10 = 0.21
# 0.55 / 0.6745 and 0.21 / 0.6745.
STABILITY_NOISE_C = 0.82
STABILITY_STABLE_NOISE_C = 0.31
# Excess p10 and p50 inside the joint tail |z| >= 3.6 and |excess| >= 1.74.
STABILITY_HEAT_C = (2.19, 3.88)
STABILITY_COOL_C = (2.19, 3.88)
# Excess p50 to p90 of columns whose MAD is at or below 0.25 °C.
STABILITY_STABLE_C = (0.24, 0.37)
# Absolute-excess p50 up to a level under the joint-tail p10.
STABILITY_BROAD_C = (0.59, 1.15)
STABILITY_PERIOD_C = (4.5, 6.5)
STABILITY_PERIOD_ROWS = 4
STABILITY_FAULT_C = (14.0, 18.0)
# Three full 28-row crops on the stride-14 grid.
STABILITY_ON = (28, 84)
STABILITY_HEAT_X = (30, 36)
STABILITY_COOL_X = (100, 106)
STABILITY_PERIOD_X = (180, 188)
STABILITY_FAULT_X = 250
STABILITY_BROAD_X = (320, 376)
STABILITY_STABLE_X = (420, 480)
STABILITY_BUMP_X = (448, 454)
STABILITY_RIDGE_X = (560, 566)
STABILITY_T = 168
STABILITY_X = 640
STABILITY_TRAIN_SEEDS = tuple(range(1_800_000, 1_800_024))
STABILITY_STEPS = 500
STABILITY_PER_EVENT = 128
# Background sample: a stable column, then an ordinary one.
STABILITY_MAD_SPLIT = 0.30
STABILITY_STABLE_MAX = 1.0
STABILITY_FEATURES = CONTEXT_FEATURES + ("excess", "mad")


def _column_mad(residual: np.ndarray) -> np.ndarray:
    n_space = residual.shape[1]
    mad = np.full(n_space, np.nan, dtype=np.float64)
    for column in range(n_space):
        series = residual[:, column]
        series = series[np.isfinite(series)]
        if series.size < TIME_CROP:
            continue
        center = float(np.median(series))
        mad[column] = float(np.median(np.abs(series - center)))
    return mad


def _column_excess(residual: np.ndarray, t0: int) -> np.ndarray:
    slab = residual[t0:t0 + TIME_CROP]
    out = np.full(slab.shape[1], np.nan, dtype=np.float64)
    live = np.isfinite(slab).any(axis=0)
    if live.any():
        out[live] = np.nanmedian(slab[:, live], axis=0)
    return out


def _window_extra(excess: np.ndarray, mad: np.ndarray, x0: int) -> tuple[float, float]:
    """Signed excess and the MAD of the column that carries it."""
    center = excess[x0:x0 + SPACE_CROP]
    scale = mad[x0:x0 + SPACE_CROP]
    finite = np.isfinite(center) & np.isfinite(scale)
    if not finite.any():
        return 0.0, 0.0
    values = center[finite]
    pick = int(np.argmax(np.abs(values)))
    column = int(np.flatnonzero(finite)[pick])
    return float(values[pick]), float(scale[column])


def stability_table(field: np.ndarray, valid: np.ndarray, origins: np.ndarray) -> np.ndarray:
    """Context features plus temporal excess and the column MAD.

    An empty origin list has shape (0, 7).
    """
    origins = np.asarray(origins, dtype=np.int64)
    context = context_table(field, valid, origins)
    extra = np.zeros((len(origins), 2), dtype=np.float64)
    if len(origins) == 0:
        return np.column_stack((context, extra))
    baseline = long_run_median(field, valid)
    residual = np.where(
        np.asarray(valid, dtype=bool) & np.isfinite(baseline),
        np.asarray(field, dtype=np.float64) - baseline,
        np.nan,
    )
    mad = _column_mad(residual)
    cache: dict[int, np.ndarray] = {}
    for index, (t0, x0) in enumerate(origins):
        t0 = int(t0)
        if t0 not in cache:
            cache[t0] = _column_excess(residual, t0)
        extra[index] = _window_extra(cache[t0], mad, int(x0))
    return np.column_stack((context, extra))


def _stamp_level(field, profile, labels, event_id, t0, t1, x0, x1, value, class_id, eid) -> None:
    """Replace the standing level with one flat offset."""
    field[t0:t1, x0:x1] = profile[x0:x1] + np.float32(value)
    if class_id:
        labels[t0:t1, x0:x1] = class_id
        event_id[t0:t1, x0:x1] = eid


def plant_stability(seed: int = 0) -> dict:
    """Joint-tail heat on an ordinary column, and a small bump on a stable one.

    The stable bump is not an event. A full-time ridge uses the heat
    amplitude and is not an event, because the column median removes it.
    Labels are the painted cells. Event bands stay 40 columns apart.
    """
    rng = np.random.default_rng(int(seed))
    n_dead = int(round(STABILITY_INVALID_SHARE * STABILITY_X))
    n_live = STABILITY_X - n_dead
    x_m = np.arange(STABILITY_X, dtype=np.float64)
    time_s = np.arange(STABILITY_T, dtype=np.float64) * 60.0
    field = np.full((STABILITY_T, STABILITY_X), SENTINEL_VALUE, dtype=np.float32)
    valid = np.zeros(field.shape, dtype=bool)
    valid[:, :n_live] = True
    x = np.arange(n_live, dtype=np.float64)
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    wave = np.sin(2.0 * np.pi * x / float(STABILITY_X) + phase)
    wave -= wave.mean()
    peak = float(np.max(np.abs(wave)))
    if peak > 0.0:
        wave *= STABILITY_STANDING_C / peak
    profile = np.full(STABILITY_X, SENTINEL_VALUE, dtype=np.float32)
    profile[:n_live] = (STABILITY_LEVEL_C + wave).astype(np.float32)
    field[:, :n_live] = profile[:n_live]
    field[:, :n_live] += rng.normal(0.0, STABILITY_NOISE_C, size=(STABILITY_T, n_live)).astype(np.float32)
    lo, hi = STABILITY_STABLE_X
    field[:, lo:hi] = profile[lo:hi] + rng.normal(
        0.0, STABILITY_STABLE_NOISE_C, size=(STABILITY_T, hi - lo)
    ).astype(np.float32)
    labels = np.zeros(field.shape, dtype=np.int8)
    event_id = np.zeros(field.shape, dtype=np.int16)

    heat = float(rng.uniform(*STABILITY_HEAT_C))
    cool = float(rng.uniform(*STABILITY_COOL_C))
    broad = float(rng.uniform(*STABILITY_BROAD_C))
    bump = float(rng.uniform(*STABILITY_STABLE_C))
    on0, on1 = STABILITY_ON
    _stamp_level(
        field, profile, labels, event_id, on0, on1, *STABILITY_HEAT_X,
        heat, LABEL_INDEX["Local_Heating"], 1,
    )
    _stamp_level(
        field, profile, labels, event_id, on0, on1, *STABILITY_COOL_X,
        -cool, LABEL_INDEX["Local_Cooling"], 2,
    )
    _stamp_level(
        field, profile, labels, event_id, on0, on1, *STABILITY_BROAD_X,
        broad, LABEL_INDEX["Broad_Shift"], 5,
    )
    _stamp_level(
        field, profile, labels, event_id, on0, on1, *STABILITY_BUMP_X,
        bump, 0, 0,
    )
    ridge0, ridge1 = STABILITY_RIDGE_X
    field[:, ridge0:ridge1] = profile[ridge0:ridge1] + np.float32(heat)

    period_amp = float(rng.uniform(*STABILITY_PERIOD_C))
    for row in (8, 48, 96, 140):
        _stamp(
            field, labels, event_id, row, row + STABILITY_PERIOD_ROWS, *STABILITY_PERIOD_X,
            period_amp, LABEL_INDEX["Periodic_Thermal"], 3,
        )
    _stamp(
        field, labels, event_id, 30, 31, STABILITY_FAULT_X, STABILITY_FAULT_X + 1,
        float(rng.uniform(*STABILITY_FAULT_C)), LABEL_INDEX["Sensor_Fault"], 4,
    )
    labels[~valid] = 0
    event_id[~valid] = 0
    shift = np.float32(STABILITY_LEVEL_C - float(np.median(field[valid])))
    field[valid] = field[valid] + shift
    events = (
        {"id": 1, "label": "Local_Heating"},
        {"id": 2, "label": "Local_Cooling"},
        {"id": 3, "label": "Periodic_Thermal"},
        {"id": 4, "label": "Sensor_Fault"},
        {"id": 5, "label": "Broad_Shift"},
    )
    return {
        "kind": "stability",
        "label_source": "generator",
        "field": field,
        "valid": valid,
        "x_m": x_m,
        "time_s": time_s,
        "labels": labels,
        "event_id": event_id,
        "events": events,
    }


def dump_stability_model(model: SoftmaxSceneModel, extra: dict | None = None) -> dict:
    if model.weight is None or model.mu is None or model.sig is None or model.bias is None:
        raise RuntimeError("model is not fit")
    payload = {
        "family": "softmax_stability",
        "features": list(STABILITY_FEATURES),
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


def load_stability_model(payload: dict) -> SoftmaxSceneModel:
    if list(payload["features"]) != list(STABILITY_FEATURES):
        raise ValueError("stability feature list does not match this code")
    if int(payload["context_side"]) != CONTEXT_SIDE or float(payload["peak_fraction"]) != CONTEXT_PEAK_FRACTION:
        raise ValueError("context constants do not match this code")
    model = SoftmaxSceneModel()
    model.mu = np.asarray(payload["mu"], dtype=np.float64)
    model.sig = np.asarray(payload["sig"], dtype=np.float64)
    model.weight = np.asarray(payload["weight"], dtype=np.float64)
    model.bias = np.asarray(payload["bias"], dtype=np.float64)
    return model
