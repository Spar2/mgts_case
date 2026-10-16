"""Lag-1 persistence of the peak column.

On 12000 scored windows from records 1..6, seed 8, and no witness boxes,
the lag-1 correlation of a column has p50 = 0.58. Columns with MAD at or
below 0.25 °C and absolute excess below 1 °C sit at ac1 p50 = 0.20.
The joint tail, absolute excess at least 1.74 °C and |z| at least 3.6,
starts at ac1 p10 = 0.81. Heating is that persistent tail. A stable
low-excess column is background. Witness coordinates are not an input.
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

PERSIST_LEVEL_C = 22.746278285980225
PERSIST_INVALID_SHARE = 0.047769979141731485
PERSIST_STANDING_C = 0.48
# Overall ac1 p50. An AR(1) of this memory lands near that value.
PERSIST_PHI = 0.55
PERSIST_INNOVATION_C = 0.55
# Stable-column MAD p10. White noise here stays a low-excess background.
PERSIST_MAD_P10 = 0.21
PERSIST_STABLE_NOISE_C = 0.31
PERSIST_AC1_STABLE_P50 = 0.20
PERSIST_AC1_JOINT_P10 = 0.81
# Excess p10 and p50 of the joint tail.
PERSIST_HEAT_C = (2.19, 3.88)
PERSIST_COOL_C = (2.19, 3.88)
PERSIST_STABLE_C = (0.24, 0.37)
PERSIST_BROAD_C = (0.59, 1.15)
PERSIST_PERIOD_C = (4.5, 6.5)
PERSIST_PERIOD_ROWS = 4
PERSIST_FAULT_C = (14.0, 18.0)
PERSIST_ON = (28, 84)
PERSIST_HEAT_X = (30, 36)
PERSIST_COOL_X = (100, 106)
PERSIST_PERIOD_X = (180, 188)
PERSIST_FAULT_X = 250
PERSIST_BROAD_X = (320, 376)
PERSIST_STABLE_X = (420, 480)
PERSIST_BUMP_X = (448, 454)
PERSIST_BUMP_ROWS = 7
PERSIST_RIDGE_X = (560, 566)
PERSIST_T = 168
PERSIST_X = 640
PERSIST_TRAIN_SEEDS = tuple(range(1_900_000, 1_900_024))
PERSIST_STEPS = 500
PERSIST_PER_EVENT = 128
PERSIST_AC1_SPLIT = 0.45
PERSIST_STABLE_MAX = 1.0
PERSIST_FEATURES = CONTEXT_FEATURES + ("excess", "ac1")


def _ac1(series: np.ndarray) -> float:
    values = np.asarray(series, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 4:
        return 0.0
    centered = values - np.mean(values)
    denom = float(np.dot(centered, centered))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(centered[:-1], centered[1:]) / denom)


def _column_ac1(residual: np.ndarray) -> np.ndarray:
    out = np.zeros(residual.shape[1], dtype=np.float64)
    for column in range(residual.shape[1]):
        series = residual[:, column]
        if np.isfinite(series).sum() < TIME_CROP:
            continue
        out[column] = _ac1(series)
    return out


def _column_excess(residual: np.ndarray, t0: int) -> np.ndarray:
    slab = residual[t0:t0 + TIME_CROP]
    out = np.full(slab.shape[1], np.nan, dtype=np.float64)
    live = np.isfinite(slab).any(axis=0)
    if live.any():
        out[live] = np.nanmedian(slab[:, live], axis=0)
    return out


def _window_extra(excess: np.ndarray, ac1: np.ndarray, x0: int) -> tuple[float, float]:
    """Signed excess and the lag-1 correlation of the column that carries it."""
    center = excess[x0:x0 + SPACE_CROP]
    finite = np.isfinite(center)
    if not finite.any():
        return 0.0, 0.0
    values = center[finite]
    pick = int(np.argmax(np.abs(values)))
    column = x0 + int(np.flatnonzero(finite)[pick])
    return float(values[pick]), float(ac1[column])


def persist_table(field: np.ndarray, valid: np.ndarray, origins: np.ndarray) -> np.ndarray:
    """Context features plus temporal excess and column lag-1.

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
    ac1 = _column_ac1(residual)
    cache: dict[int, np.ndarray] = {}
    for index, (t0, x0) in enumerate(origins):
        t0 = int(t0)
        if t0 not in cache:
            cache[t0] = _column_excess(residual, t0)
        extra[index] = _window_extra(cache[t0], ac1, int(x0))
    return np.column_stack((context, extra))


def _stamp_level(field, profile, labels, event_id, t0, t1, x0, x1, value, class_id, eid) -> None:
    field[t0:t1, x0:x1] = profile[x0:x1] + np.float32(value)
    if class_id:
        labels[t0:t1, x0:x1] = class_id
        event_id[t0:t1, x0:x1] = eid


def _ar_field(rng: np.random.Generator, n_time: int, n_live: int) -> np.ndarray:
    innov = rng.normal(0.0, PERSIST_INNOVATION_C, size=(n_time, n_live))
    out = np.empty_like(innov)
    out[0] = innov[0]
    phi = PERSIST_PHI
    for row in range(1, n_time):
        out[row] = phi * out[row - 1] + innov[row]
    return out.astype(np.float32)


def plant_persist(seed: int = 0) -> dict:
    """Persistent heat on a smooth column, and a stable column left as background.

    The stable band is white noise plus a 7-row bump. It is not an event.
    A full-time ridge uses the heat amplitude and is not an event, because
    the column median removes it. Event bands stay 40 columns apart.
    """
    rng = np.random.default_rng(int(seed))
    n_dead = int(round(PERSIST_INVALID_SHARE * PERSIST_X))
    n_live = PERSIST_X - n_dead
    x_m = np.arange(PERSIST_X, dtype=np.float64)
    time_s = np.arange(PERSIST_T, dtype=np.float64) * 60.0
    field = np.full((PERSIST_T, PERSIST_X), SENTINEL_VALUE, dtype=np.float32)
    valid = np.zeros(field.shape, dtype=bool)
    valid[:, :n_live] = True
    x = np.arange(n_live, dtype=np.float64)
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    wave = np.sin(2.0 * np.pi * x / float(PERSIST_X) + phase)
    wave -= wave.mean()
    peak = float(np.max(np.abs(wave)))
    if peak > 0.0:
        wave *= PERSIST_STANDING_C / peak
    profile = np.full(PERSIST_X, SENTINEL_VALUE, dtype=np.float32)
    profile[:n_live] = (PERSIST_LEVEL_C + wave).astype(np.float32)
    field[:, :n_live] = profile[:n_live] + _ar_field(rng, PERSIST_T, n_live)
    lo, hi = PERSIST_STABLE_X
    field[:, lo:hi] = profile[lo:hi] + rng.normal(
        0.0, PERSIST_STABLE_NOISE_C, size=(PERSIST_T, hi - lo)
    ).astype(np.float32)
    labels = np.zeros(field.shape, dtype=np.int8)
    event_id = np.zeros(field.shape, dtype=np.int16)

    heat = float(rng.uniform(*PERSIST_HEAT_C))
    cool = float(rng.uniform(*PERSIST_COOL_C))
    broad = float(rng.uniform(*PERSIST_BROAD_C))
    bump = float(rng.uniform(*PERSIST_STABLE_C))
    on0, on1 = PERSIST_ON
    _stamp_level(
        field, profile, labels, event_id, on0, on1, *PERSIST_HEAT_X,
        heat, LABEL_INDEX["Local_Heating"], 1,
    )
    _stamp_level(
        field, profile, labels, event_id, on0, on1, *PERSIST_COOL_X,
        -cool, LABEL_INDEX["Local_Cooling"], 2,
    )
    _stamp_level(
        field, profile, labels, event_id, on0, on1, *PERSIST_BROAD_X,
        broad, LABEL_INDEX["Broad_Shift"], 5,
    )
    bump0, bump1 = PERSIST_BUMP_X
    field[on0:on0 + PERSIST_BUMP_ROWS, bump0:bump1] = profile[bump0:bump1] + np.float32(bump)
    ridge0, ridge1 = PERSIST_RIDGE_X
    field[:, ridge0:ridge1] = profile[ridge0:ridge1] + np.float32(heat)

    period_amp = float(rng.uniform(*PERSIST_PERIOD_C))
    for row in (8, 48, 96, 140):
        _stamp(
            field, labels, event_id, row, row + PERSIST_PERIOD_ROWS, *PERSIST_PERIOD_X,
            period_amp, LABEL_INDEX["Periodic_Thermal"], 3,
        )
    _stamp(
        field, labels, event_id, 30, 31, PERSIST_FAULT_X, PERSIST_FAULT_X + 1,
        float(rng.uniform(*PERSIST_FAULT_C)), LABEL_INDEX["Sensor_Fault"], 4,
    )
    labels[~valid] = 0
    event_id[~valid] = 0
    shift = np.float32(PERSIST_LEVEL_C - float(np.median(field[valid])))
    field[valid] = field[valid] + shift
    events = (
        {"id": 1, "label": "Local_Heating"},
        {"id": 2, "label": "Local_Cooling"},
        {"id": 3, "label": "Periodic_Thermal"},
        {"id": 4, "label": "Sensor_Fault"},
        {"id": 5, "label": "Broad_Shift"},
    )
    return {
        "kind": "persist",
        "label_source": "generator",
        "field": field,
        "valid": valid,
        "x_m": x_m,
        "time_s": time_s,
        "labels": labels,
        "event_id": event_id,
        "events": events,
    }


def dump_persist_model(model: SoftmaxSceneModel, extra: dict | None = None) -> dict:
    if model.weight is None or model.mu is None or model.sig is None or model.bias is None:
        raise RuntimeError("model is not fit")
    payload = {
        "family": "softmax_persist",
        "features": list(PERSIST_FEATURES),
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


def load_persist_model(payload: dict) -> SoftmaxSceneModel:
    if list(payload["features"]) != list(PERSIST_FEATURES):
        raise ValueError("persist feature list does not match this code")
    if int(payload["context_side"]) != CONTEXT_SIDE or float(payload["peak_fraction"]) != CONTEXT_PEAK_FRACTION:
        raise ValueError("context constants do not match this code")
    model = SoftmaxSceneModel()
    model.mu = np.asarray(payload["mu"], dtype=np.float64)
    model.sig = np.asarray(payload["sig"], dtype=np.float64)
    model.weight = np.asarray(payload["weight"], dtype=np.float64)
    model.bias = np.asarray(payload["bias"], dtype=np.float64)
    return model
