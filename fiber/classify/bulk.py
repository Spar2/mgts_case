"""Warm-tail gate. The bulk of a column stays background.

On 12000 scored windows from records 1..6, seed 9, and no witness boxes,
a window is in the warm tail only when its excess is at least 2.19 °C,
its full-column lag-1 is at least 0.81, and its |z| is at least 3.6.
That tail is 325 windows, share 0.0271. Inside it, excess p10/p50 is
2.78/4.50 °C, lag-1 p10/p50 is 0.93/0.97, and |z| p10/p50 is 4.39/6.67.
A spatial peak of at least 2.79 °C that misses this tail is 1632 windows,
share 0.136, with excess p50 0.86 °C. The complement, share 0.959, is the
bulk. Heating is the warm tail. The bulk is background. Witness coordinates
are not an input.
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

BULK_LEVEL_C = 22.746278285980225
BULK_INVALID_SHARE = 0.047769979141731485
# abs peak p90 on the same 12000 windows. A full-time spike of this size
# is a standing profile: the column median removes it.
BULK_STANDING_C = 3.85
BULK_PHI = 0.55
# 0.55 leaves a 56-row block near |z| 3.2, short of the measured cut.
# 0.30 puts that block at |z| about 6, inside the measured tail.
BULK_INNOVATION_C = 0.30
BULK_MAD_P10 = 0.21
BULK_STABLE_NOISE_C = 0.31
BULK_MAD_FLOOR = 0.05
# Cuts that defined the measured warm tail.
BULK_WARM_EXCESS = 2.19
BULK_WARM_AC1 = 0.81
BULK_WARM_Z = 3.6
# Excess p10 and p50 inside that tail.
BULK_HEAT_C = (2.78, 4.50)
BULK_COOL_C = (2.78, 4.50)
BULK_STABLE_C = (0.24, 0.37)
# Below the excess cut, so a wide step stays in the bulk.
BULK_BROAD_C = (0.59, 1.15)
BULK_PERIOD_C = (4.5, 6.5)
BULK_PERIOD_ROWS = 4
BULK_FAULT_C = (14.0, 18.0)
BULK_ON = (28, 84)
BULK_HEAT_X = (30, 36)
BULK_COOL_X = (100, 106)
BULK_PERIOD_X = (180, 188)
BULK_FAULT_X = 250
BULK_BROAD_X = (320, 376)
BULK_STABLE_X = (420, 480)
BULK_BUMP_X = (448, 454)
BULK_BUMP_ROWS = 7
BULK_SPIKE_X = (508, 514)
BULK_RIDGE_X = (560, 566)
BULK_T = 168
BULK_X = 640
BULK_TRAIN_SEEDS = tuple(range(2_000_000, 2_000_024))
BULK_STEPS = 500
BULK_PER_EVENT = 128
BULK_AC1_SPLIT = 0.45
BULK_STABLE_MAX = 1.0
BULK_DECOY_PEAK = 2.79
BULK_FEATURES = CONTEXT_FEATURES + ("excess", "ac1", "z")


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


def _column_stats(residual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ac1 = np.zeros(residual.shape[1], dtype=np.float64)
    mad = np.full(residual.shape[1], np.nan, dtype=np.float64)
    for column in range(residual.shape[1]):
        series = residual[:, column]
        series = series[np.isfinite(series)]
        if series.size < TIME_CROP:
            continue
        center = float(np.median(series))
        mad[column] = float(np.median(np.abs(series - center)))
        ac1[column] = _ac1(series)
    return ac1, mad


def _column_excess(residual: np.ndarray, t0: int) -> np.ndarray:
    slab = residual[t0:t0 + TIME_CROP]
    out = np.full(slab.shape[1], np.nan, dtype=np.float64)
    live = np.isfinite(slab).any(axis=0)
    if live.any():
        out[live] = np.nanmedian(slab[:, live], axis=0)
    return out


def _window_extra(excess: np.ndarray, ac1: np.ndarray, mad: np.ndarray, x0: int) -> tuple[float, float, float]:
    """Signed excess, lag-1, and |z| of the column that carries the excess."""
    center = excess[x0:x0 + SPACE_CROP]
    finite = np.isfinite(center)
    if not finite.any():
        return 0.0, 0.0, 0.0
    values = center[finite]
    pick = int(np.argmax(np.abs(values)))
    column = x0 + int(np.flatnonzero(finite)[pick])
    scale = mad[column]
    signed = float(values[pick])
    if not np.isfinite(scale):
        zee = 0.0
    else:
        zee = abs(signed) / max(float(scale), BULK_MAD_FLOOR)
    return signed, float(ac1[column]), float(zee)


def bulk_table(field: np.ndarray, valid: np.ndarray, origins: np.ndarray) -> np.ndarray:
    """Context features plus excess, lag-1, and |z|.

    An empty origin list has shape (0, 8).
    """
    origins = np.asarray(origins, dtype=np.int64)
    context = context_table(field, valid, origins)
    extra = np.zeros((len(origins), 3), dtype=np.float64)
    if len(origins) == 0:
        return np.column_stack((context, extra))
    baseline = long_run_median(field, valid)
    residual = np.where(
        np.asarray(valid, dtype=bool) & np.isfinite(baseline),
        np.asarray(field, dtype=np.float64) - baseline,
        np.nan,
    )
    ac1, mad = _column_stats(residual)
    cache: dict[int, np.ndarray] = {}
    for index, (t0, x0) in enumerate(origins):
        t0 = int(t0)
        if t0 not in cache:
            cache[t0] = _column_excess(residual, t0)
        extra[index] = _window_extra(cache[t0], ac1, mad, int(x0))
    return np.column_stack((context, extra))


def warm_tail_mask(features: np.ndarray) -> np.ndarray:
    """True only on the measured warm tail. An empty table returns no rows."""
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if matrix.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    excess = matrix[:, BULK_FEATURES.index("excess")]
    ac1 = matrix[:, BULK_FEATURES.index("ac1")]
    zee = matrix[:, BULK_FEATURES.index("z")]
    return (excess >= BULK_WARM_EXCESS) & (ac1 >= BULK_WARM_AC1) & (zee >= BULK_WARM_Z)


def bulk_names(model: SoftmaxSceneModel, features: np.ndarray) -> list[str]:
    """Model names on the warm tail. Every other window is Background."""
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if matrix.shape[0] == 0:
        return []
    raw = model.predict(matrix)
    mask = warm_tail_mask(matrix)
    return [name if flag else "Background" for name, flag in zip(raw, mask, strict=True)]


def _stamp_level(field, profile, labels, event_id, t0, t1, x0, x1, value, class_id, eid) -> None:
    field[t0:t1, x0:x1] = profile[x0:x1] + np.float32(value)
    if class_id:
        labels[t0:t1, x0:x1] = class_id
        event_id[t0:t1, x0:x1] = eid


def _ar_field(rng: np.random.Generator, n_time: int, n_live: int) -> np.ndarray:
    innov = rng.normal(0.0, BULK_INNOVATION_C, size=(n_time, n_live))
    out = np.empty_like(innov)
    out[0] = innov[0]
    phi = BULK_PHI
    for row in range(1, n_time):
        out[row] = phi * out[row - 1] + innov[row]
    return out.astype(np.float32)


def plant_bulk(seed: int = 0) -> dict:
    """A warm block on a smooth column. A standing spike stays in the bulk.

    The spike is added for the whole trace, so the column median removes it
    and the warm-tail cuts stay closed. The stable band is white noise plus
    a 7-row bump. A full-time ridge uses the heat amplitude and is not an
    event. Event bands stay 40 columns apart.
    """
    rng = np.random.default_rng(int(seed))
    n_dead = int(round(BULK_INVALID_SHARE * BULK_X))
    n_live = BULK_X - n_dead
    x_m = np.arange(BULK_X, dtype=np.float64)
    time_s = np.arange(BULK_T, dtype=np.float64) * 60.0
    field = np.full((BULK_T, BULK_X), SENTINEL_VALUE, dtype=np.float32)
    valid = np.zeros(field.shape, dtype=bool)
    valid[:, :n_live] = True
    x = np.arange(n_live, dtype=np.float64)
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    wave = np.sin(2.0 * np.pi * x / float(BULK_X) + phase)
    wave -= wave.mean()
    peak = float(np.max(np.abs(wave)))
    if peak > 0.0:
        wave *= 0.48 / peak
    profile = np.full(BULK_X, SENTINEL_VALUE, dtype=np.float32)
    profile[:n_live] = (BULK_LEVEL_C + wave).astype(np.float32)
    field[:, :n_live] = profile[:n_live] + _ar_field(rng, BULK_T, n_live)
    lo, hi = BULK_STABLE_X
    field[:, lo:hi] = profile[lo:hi] + rng.normal(
        0.0, BULK_STABLE_NOISE_C, size=(BULK_T, hi - lo)
    ).astype(np.float32)
    labels = np.zeros(field.shape, dtype=np.int8)
    event_id = np.zeros(field.shape, dtype=np.int16)

    heat = float(rng.uniform(*BULK_HEAT_C))
    cool = float(rng.uniform(*BULK_COOL_C))
    broad = float(rng.uniform(*BULK_BROAD_C))
    bump = float(rng.uniform(*BULK_STABLE_C))
    on0, on1 = BULK_ON
    _stamp_level(
        field, profile, labels, event_id, on0, on1, *BULK_HEAT_X,
        heat, LABEL_INDEX["Local_Heating"], 1,
    )
    _stamp_level(
        field, profile, labels, event_id, on0, on1, *BULK_COOL_X,
        -cool, LABEL_INDEX["Local_Cooling"], 2,
    )
    _stamp_level(
        field, profile, labels, event_id, on0, on1, *BULK_BROAD_X,
        broad, LABEL_INDEX["Broad_Shift"], 5,
    )
    bump0, bump1 = BULK_BUMP_X
    field[on0:on0 + BULK_BUMP_ROWS, bump0:bump1] = profile[bump0:bump1] + np.float32(bump)
    spike0, spike1 = BULK_SPIKE_X
    field[:, spike0:spike1] = field[:, spike0:spike1] + np.float32(BULK_STANDING_C)
    ridge0, ridge1 = BULK_RIDGE_X
    field[:, ridge0:ridge1] = profile[ridge0:ridge1] + np.float32(heat)

    period_amp = float(rng.uniform(*BULK_PERIOD_C))
    for row in (8, 48, 96, 140):
        _stamp(
            field, labels, event_id, row, row + BULK_PERIOD_ROWS, *BULK_PERIOD_X,
            period_amp, LABEL_INDEX["Periodic_Thermal"], 3,
        )
    _stamp(
        field, labels, event_id, 30, 31, BULK_FAULT_X, BULK_FAULT_X + 1,
        float(rng.uniform(*BULK_FAULT_C)), LABEL_INDEX["Sensor_Fault"], 4,
    )
    labels[~valid] = 0
    event_id[~valid] = 0
    shift = np.float32(BULK_LEVEL_C - float(np.median(field[valid])))
    field[valid] = field[valid] + shift
    events = (
        {"id": 1, "label": "Local_Heating"},
        {"id": 2, "label": "Local_Cooling"},
        {"id": 3, "label": "Periodic_Thermal"},
        {"id": 4, "label": "Sensor_Fault"},
        {"id": 5, "label": "Broad_Shift"},
    )
    return {
        "kind": "bulk",
        "label_source": "generator",
        "field": field,
        "valid": valid,
        "x_m": x_m,
        "time_s": time_s,
        "labels": labels,
        "event_id": event_id,
        "events": events,
    }


def dump_bulk_model(model: SoftmaxSceneModel, extra: dict | None = None) -> dict:
    if model.weight is None or model.mu is None or model.sig is None or model.bias is None:
        raise RuntimeError("model is not fit")
    payload = {
        "family": "softmax_bulk",
        "features": list(BULK_FEATURES),
        "context_side": CONTEXT_SIDE,
        "peak_fraction": CONTEXT_PEAK_FRACTION,
        "spike_mad_floor": SPIKE_MAD_FLOOR,
        "warm_excess": BULK_WARM_EXCESS,
        "warm_ac1": BULK_WARM_AC1,
        "warm_z": BULK_WARM_Z,
        "labels": list(LABELS),
        "mu": model.mu.tolist(),
        "sig": model.sig.tolist(),
        "weight": model.weight.tolist(),
        "bias": model.bias.tolist(),
    }
    if extra:
        payload.update(extra)
    return payload


def load_bulk_model(payload: dict) -> SoftmaxSceneModel:
    if list(payload["features"]) != list(BULK_FEATURES):
        raise ValueError("bulk feature list does not match this code")
    if int(payload["context_side"]) != CONTEXT_SIDE or float(payload["peak_fraction"]) != CONTEXT_PEAK_FRACTION:
        raise ValueError("context constants do not match this code")
    if float(payload["warm_excess"]) != BULK_WARM_EXCESS or float(payload["warm_ac1"]) != BULK_WARM_AC1:
        raise ValueError("warm-tail cuts do not match this code")
    if float(payload["warm_z"]) != BULK_WARM_Z:
        raise ValueError("warm-tail cuts do not match this code")
    model = SoftmaxSceneModel()
    model.mu = np.asarray(payload["mu"], dtype=np.float64)
    model.sig = np.asarray(payload["sig"], dtype=np.float64)
    model.weight = np.asarray(payload["weight"], dtype=np.float64)
    model.bias = np.asarray(payload["bias"], dtype=np.float64)
    return model
