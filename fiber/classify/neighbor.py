"""Neighbor shape of a temporal excess.

On 11997 scored windows from records 1..6, seed 5, and no witness boxes, the
28-row excess has absolute p90 = 1.743 °C and p95 = 2.810 °C. Among windows
already past 2 °C the peak sits 0.252 °C above its two neighbors, a ratio
p50 of 0.068, and the cable-wide median excess has p50 = 0.159 °C. Across
all windows that ratio has p90 = 0.818, and the cable-wide median has
p90 = 0.444 °C and p99 = 0.896 °C. Heating here is the smooth strong peak.
A jagged bulk peak and a drift the whole cable shares stay background.
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
NEIGHBOR_LEVEL_C = 22.746278285980225
NEIGHBOR_INVALID_SHARE = 0.047769979141731485
# Spatial peak p90 on the earlier context subsample.
NEIGHBOR_STANDING_AMP_C = 2.79
# Absolute 28-row excess, p90 to p95.
NEIGHBOR_HEAT_C = (1.74, 2.81)
NEIGHBOR_COOL_C = (1.74, 2.81)
# Sigma whose one-column neighbor ratio is the strong-peak p50, 0.068.
NEIGHBOR_HEAT_SIGMA_M = 2.66
# Body of the absolute excess, where the jagged ratio p90 actually occurs.
NEIGHBOR_DECOY_C = (0.59, 1.74)
# More than half of a 28-row crop, so the median equals the stamp.
NEIGHBOR_DECOY_ROWS = 20
# Cable-wide median excess, p90 to p99. The whole live cable shares it.
NEIGHBOR_COMMON_C = (0.44, 0.90)
# Above the strong-peak width p95 of 53 m, and wider than one crop.
NEIGHBOR_BROAD_C = (1.74, 2.81)
NEIGHBOR_BROAD_WIDTH_M = 56
# Spike p50 to p90, and above the spike p99 of 14.2 °C.
NEIGHBOR_PERIOD_C = (4.5, 6.5)
NEIGHBOR_PERIOD_ROWS = 4
NEIGHBOR_FAULT_C = (14.0, 18.0)
# Just above the absolute-excess p10 of 0.237 °C.
NEIGHBOR_FLOOR_C = 0.25
# Overall stick-ratio p90, and a ratio under the strong-peak p99 of 0.257.
NEIGHBOR_JAGGED_RATIO = 0.818
NEIGHBOR_FLAT_RATIO = 0.20
NEIGHBOR_ON = (28, 84)
NEIGHBOR_T = 140
NEIGHBOR_X = 640
NEIGHBOR_TRAIN_SEEDS = tuple(range(1_600_000, 1_600_024))
NEIGHBOR_STEPS = 500
NEIGHBOR_PER_EVENT = 128
NEIGHBOR_FEATURES = CONTEXT_FEATURES + ("excess", "stick", "ratio", "common")


def _column_excess(residual: np.ndarray, t0: int) -> np.ndarray:
    slab = residual[t0:t0 + TIME_CROP]
    out = np.full(slab.shape[1], np.nan, dtype=np.float64)
    live = np.isfinite(slab).any(axis=0)
    if live.any():
        out[live] = np.nanmedian(slab[:, live], axis=0)
    return out


def _window_extra(excess: np.ndarray, x0: int, common: float) -> tuple[float, float, float, float]:
    """Signed excess, degrees above the neighbor pair, their ratio, and the cable mode."""
    n_space = int(excess.shape[0])
    center = excess[x0:x0 + SPACE_CROP]
    finite = np.isfinite(center)
    if not finite.any():
        return 0.0, 0.0, 0.0, common
    values = center[finite]
    pick = int(np.argmax(np.abs(values)))
    peak = float(values[pick])
    column = x0 + int(np.flatnonzero(finite)[pick])
    if column <= 0 or column + 1 >= n_space or abs(peak) < NEIGHBOR_FLOOR_C:
        return peak, 0.0, 0.0, common
    neighbors = excess[column - 1:column + 2]
    if not np.isfinite(neighbors).all():
        return peak, 0.0, 0.0, common
    stick = float(np.sign(peak) * (neighbors[1] - 0.5 * (neighbors[0] + neighbors[2])))
    return peak, stick, stick / abs(peak), common


def neighbor_table(field: np.ndarray, valid: np.ndarray, origins: np.ndarray) -> np.ndarray:
    """Context features plus excess, neighbor stick, ratio, and the cable mode.

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
    cache: dict[int, tuple[np.ndarray, float]] = {}
    for index, (t0, x0) in enumerate(origins):
        t0 = int(t0)
        if t0 not in cache:
            profile = _column_excess(residual, t0)
            finite = profile[np.isfinite(profile)]
            mode = float(np.median(finite)) if finite.size else 0.0
            cache[t0] = (profile, mode)
        profile, mode = cache[t0]
        extra[index] = _window_extra(profile, int(x0), mode)
    return np.column_stack((context, extra))


def _paint_gaussian(field, labels, event_id, t0, t1, center, amplitude, class_id, eid) -> None:
    sigma = NEIGHBOR_HEAT_SIGMA_M
    half = int(np.ceil(3.0 * sigma))
    for offset in range(-half, half + 1):
        column = int(center) + offset
        if column < 0 or column >= field.shape[1]:
            continue
        gain = float(np.exp(-0.5 * (offset / sigma) ** 2))
        field[t0:t1, column] += np.float32(amplitude * gain)
        if gain >= 0.5:
            labels[t0:t1, column] = class_id
            event_id[t0:t1, column] = eid


def plant_neighbor(seed: int = 0) -> dict:
    """Standing bow, one smooth bump per signed class, and two unlabeled shapes.

    The unlabeled shapes are a jagged column in the bulk of the excess and a
    drift shared by every live column. Labels are the painted cells. Event
    bands stay 40 columns apart.
    """
    rng = np.random.default_rng(int(seed))
    n_dead = int(round(NEIGHBOR_INVALID_SHARE * NEIGHBOR_X))
    n_live = NEIGHBOR_X - n_dead
    x_m = np.arange(NEIGHBOR_X, dtype=np.float64)
    time_s = np.arange(NEIGHBOR_T, dtype=np.float64) * 60.0
    field = np.full((NEIGHBOR_T, NEIGHBOR_X), SENTINEL_VALUE, dtype=np.float32)
    valid = np.zeros(field.shape, dtype=bool)
    valid[:, :n_live] = True
    x = np.arange(n_live, dtype=np.float64)
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    wave = np.sin(2.0 * np.pi * x / float(NEIGHBOR_X) + phase)
    wave -= wave.mean()
    peak = float(np.max(np.abs(wave)))
    if peak > 0.0:
        wave *= NEIGHBOR_STANDING_AMP_C / peak
    noise = 0.12 + 0.03 * (int(seed) % 3)
    field[:, :n_live] = (NEIGHBOR_LEVEL_C + wave).astype(np.float32)
    field[:, :n_live] += rng.normal(0.0, noise, size=(NEIGHBOR_T, n_live)).astype(np.float32)
    labels = np.zeros(field.shape, dtype=np.int8)
    event_id = np.zeros(field.shape, dtype=np.int16)

    on0, on1 = NEIGHBOR_ON
    field[on0:on1, :n_live] += np.float32(rng.uniform(*NEIGHBOR_COMMON_C))
    decoy_end = NEIGHBOR_T - 7
    field[decoy_end - NEIGHBOR_DECOY_ROWS:decoy_end, 560] += np.float32(rng.uniform(*NEIGHBOR_DECOY_C))

    _paint_gaussian(
        field, labels, event_id, on0, on1, 30,
        float(rng.uniform(*NEIGHBOR_HEAT_C)), LABEL_INDEX["Local_Heating"], 1,
    )
    _paint_gaussian(
        field, labels, event_id, on0, on1, 140,
        -float(rng.uniform(*NEIGHBOR_COOL_C)), LABEL_INDEX["Local_Cooling"], 2,
    )
    period_x = 240
    period_amp = float(rng.uniform(*NEIGHBOR_PERIOD_C))
    for row in (8, 48, 110):
        _stamp(
            field, labels, event_id, row, row + NEIGHBOR_PERIOD_ROWS, period_x, period_x + 8,
            period_amp, LABEL_INDEX["Periodic_Thermal"], 3,
        )
    _stamp(
        field, labels, event_id, 30, 31, 320, 321,
        float(rng.uniform(*NEIGHBOR_FAULT_C)), LABEL_INDEX["Sensor_Fault"], 4,
    )
    _stamp(
        field, labels, event_id, on0, on1, 400, 400 + NEIGHBOR_BROAD_WIDTH_M,
        float(rng.uniform(*NEIGHBOR_BROAD_C)), LABEL_INDEX["Broad_Shift"], 5,
    )
    labels[~valid] = 0
    event_id[~valid] = 0
    shift = np.float32(NEIGHBOR_LEVEL_C - float(np.median(field[valid])))
    field[valid] = field[valid] + shift
    events = (
        {"id": 1, "label": "Local_Heating"},
        {"id": 2, "label": "Local_Cooling"},
        {"id": 3, "label": "Periodic_Thermal"},
        {"id": 4, "label": "Sensor_Fault"},
        {"id": 5, "label": "Broad_Shift"},
    )
    return {
        "kind": "neighbor",
        "label_source": "generator",
        "field": field,
        "valid": valid,
        "x_m": x_m,
        "time_s": time_s,
        "labels": labels,
        "event_id": event_id,
        "events": events,
    }


def dump_neighbor_model(model: SoftmaxSceneModel, extra: dict | None = None) -> dict:
    if model.weight is None or model.mu is None or model.sig is None or model.bias is None:
        raise RuntimeError("model is not fit")
    payload = {
        "family": "softmax_neighbor",
        "features": list(NEIGHBOR_FEATURES),
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


def load_neighbor_model(payload: dict) -> SoftmaxSceneModel:
    if list(payload["features"]) != list(NEIGHBOR_FEATURES):
        raise ValueError("neighbor feature list does not match this code")
    if int(payload["context_side"]) != CONTEXT_SIDE or float(payload["peak_fraction"]) != CONTEXT_PEAK_FRACTION:
        raise ValueError("context constants do not match this code")
    model = SoftmaxSceneModel()
    model.mu = np.asarray(payload["mu"], dtype=np.float64)
    model.sig = np.asarray(payload["sig"], dtype=np.float64)
    model.weight = np.asarray(payload["weight"], dtype=np.float64)
    model.bias = np.asarray(payload["bias"], dtype=np.float64)
    return model
