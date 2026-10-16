"""Long plateaus against the short excursions that dominate the cable.

On records 1..6, without the witness boxes, a run at least 1 °C above the
column median has length p50 = 1 row and p99 = 7 rows. At 2 °C the p99 is
19 rows, and only 0.44% of runs last 112 rows. Contiguous width at 2 °C has
p95 = 2.3 m and p99 = 5.3 m. Heating here is a 112-row plateau a few metres
wide. Bursts of 7 and 19 rows stay background. Witness coordinates are not
an input.
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
PLATEAU_LEVEL_C = 22.746278285980225
PLATEAU_INVALID_SHARE = 0.047769979141731485
# Spatial peak p90 on 15000 scored windows.
PLATEAU_STANDING_AMP_C = 2.79
# Stamp between the 2 °C duration level and the global excess p99 of 2.89 °C.
PLATEAU_HEAT_C = (2.0, 2.9)
PLATEAU_COOL_C = (2.0, 2.9)
PLATEAU_BROAD_C = (2.0, 2.9)
PLATEAU_PERIOD_C = (4.5, 6.5)
PLATEAU_FAULT_C = (14.0, 18.0)
# 1 °C run p99, and 2 °C run p99. Both stay unlabeled.
PLATEAU_DECOY_ROWS = (7, 19)
# 112-row block: the measured tail, 0.44% of 2 °C runs.
PLATEAU_ON = (56, 168)
PLATEAU_T = 280
PLATEAU_X = 480
# Between the 2 °C width p95 (2.3 m) and p99 (5.3 m).
PLATEAU_HEAT_WIDTH_M = (4, 6)
# Above the 2 °C width p99.
PLATEAU_BROAD_WIDTH_M = 30
# 28-row median of 0.18 °C noise is about 0.04 °C.
PLATEAU_FLOOR_C = 0.25
PLATEAU_TRAIN_SEEDS = tuple(range(1_400_000, 1_400_024))
PLATEAU_STEPS = 500
PLATEAU_PER_EVENT = 128
# Painted periodic bursts last 4 rows, below the 7-row p99, and there are three of them.
PLATEAU_REPEAT_ROWS = 4
PLATEAU_PERIOD_COUNT = 3
# One crop is 28 rows, above the 2 °C run p99 of 19.
PLATEAU_FEATURES = CONTEXT_FEATURES + ("excess", "hold_rows", "hold_m", "long_hold", "repeats", "many_repeats")


def _run_count(mask: np.ndarray, minimum: int) -> int:
    if mask.size == 0 or not bool(mask.any()):
        return 0
    padded = np.concatenate(([False], mask, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return int(np.count_nonzero((ends - starts) >= int(minimum)))


def _run_through(mask: np.ndarray, index: int) -> int:
    if index < 0 or index >= int(mask.size) or not bool(mask[index]):
        return 0
    left = index
    while left > 0 and bool(mask[left - 1]):
        left -= 1
    right = index + 1
    limit = int(mask.size)
    while right < limit and bool(mask[right]):
        right += 1
    return int(right - left)


def _window_extra(
    field: np.ndarray,
    valid: np.ndarray,
    t0: int,
    x0: int,
    baseline: np.ndarray,
    x_m: np.ndarray,
) -> tuple[float, float, float, float, float, float]:
    """Excess, length, width, long-run flag, long-run count, and the repeat flag."""
    slab = np.asarray(field[t0:t0 + TIME_CROP, x0:x0 + SPACE_CROP], dtype=np.float64)
    mask = np.asarray(valid[t0:t0 + TIME_CROP, x0:x0 + SPACE_CROP], dtype=bool)
    count = mask.sum(axis=0)
    live = count > 0
    if not live.any():
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    kept = np.where(mask[:, live], slab[:, live], np.nan)
    column = np.nanmedian(kept, axis=0)
    base = np.asarray(baseline[x0:x0 + SPACE_CROP], dtype=np.float64)[live]
    delta = column - base
    finite = np.isfinite(delta)
    excess = float(delta[finite][int(np.argmax(np.abs(delta[finite])))]) if finite.any() else 0.0
    cropped = np.where(mask, slab - np.asarray(baseline[x0:x0 + SPACE_CROP], dtype=np.float64), np.nan)
    if not np.isfinite(cropped).any():
        return excess, 0.0, 0.0, 0.0, 0.0, 0.0
    flat = np.nanargmax(np.abs(cropped))
    row_offset, local_index = divmod(int(flat), SPACE_CROP)
    peak = float(cropped[row_offset, local_index])
    if abs(peak) < PLATEAU_FLOOR_C:
        return excess, 0.0, 0.0, 0.0, 0.0, 0.0
    column_index = x0 + local_index
    series = np.asarray(field[:, column_index], dtype=np.float64) - float(baseline[column_index])
    series_ok = np.asarray(valid[:, column_index], dtype=bool) & np.isfinite(series)
    thresh = CONTEXT_PEAK_FRACTION * abs(peak)
    same = series_ok & (np.sign(series) == np.sign(peak)) & (np.abs(series) >= thresh)
    hold_rows = float(_run_through(same, t0 + row_offset))
    repeats = float(_run_count(same, PLATEAU_REPEAT_ROWS))
    many_repeats = 1.0 if repeats >= PLATEAU_PERIOD_COUNT else 0.0
    row = np.asarray(field[t0 + row_offset], dtype=np.float64) - baseline
    row_ok = np.asarray(valid[t0 + row_offset], dtype=bool) & np.isfinite(row)
    across = row_ok & (np.sign(row) == np.sign(peak)) & (np.abs(row) >= thresh)
    columns = _run_through(across, column_index)
    dx = float(np.median(np.diff(x_m))) if len(x_m) > 1 else 1.0
    long_hold = 1.0 if hold_rows >= TIME_CROP else 0.0
    return excess, hold_rows, float(columns) * dx, long_hold, repeats, many_repeats


def plateau_table(field: np.ndarray, valid: np.ndarray, origins: np.ndarray, x_m: np.ndarray) -> np.ndarray:
    """Context features plus excess, hold, width, and the repeat flags.

    An empty origin list has shape (0, 11).
    """
    origins = np.asarray(origins, dtype=np.int64)
    context = context_table(field, valid, origins)
    extra = np.zeros((len(origins), 6), dtype=np.float64)
    if len(origins) == 0:
        return np.column_stack((context, extra))
    baseline = long_run_median(field, valid)
    meters = np.asarray(x_m, dtype=np.float64)
    for index, (t0, x0) in enumerate(origins):
        extra[index] = _window_extra(field, valid, int(t0), int(x0), baseline, meters)
    return np.column_stack((context, extra))


def plant_plateau(seed: int = 0) -> dict:
    """Standing bow, short unlabeled bursts, and one long plateau per class.

    Labels are the painted cells. A 7-row or 19-row burst is not an event.
    Event bands stay 40 columns apart.
    """
    rng = np.random.default_rng(int(seed))
    n_dead = int(round(PLATEAU_INVALID_SHARE * PLATEAU_X))
    n_live = PLATEAU_X - n_dead
    x_m = np.arange(PLATEAU_X, dtype=np.float64)
    time_s = np.arange(PLATEAU_T, dtype=np.float64) * 60.0
    field = np.full((PLATEAU_T, PLATEAU_X), SENTINEL_VALUE, dtype=np.float32)
    valid = np.zeros(field.shape, dtype=bool)
    valid[:, :n_live] = True
    x = np.arange(n_live, dtype=np.float64)
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    wave = np.sin(2.0 * np.pi * x / float(PLATEAU_X) + phase)
    wave -= wave.mean()
    peak = float(np.max(np.abs(wave)))
    if peak > 0.0:
        wave *= PLATEAU_STANDING_AMP_C / peak
    noise = 0.12 + 0.03 * (int(seed) % 3)
    field[:, :n_live] = (PLATEAU_LEVEL_C + wave).astype(np.float32)
    field[:, :n_live] += rng.normal(0.0, noise, size=(PLATEAU_T, n_live)).astype(np.float32)
    labels = np.zeros(field.shape, dtype=np.int8)
    event_id = np.zeros(field.shape, dtype=np.int16)

    for start in (12, 100, 210):
        for width_rows in PLATEAU_DECOY_ROWS:
            column = 400 + int(rng.integers(0, 8))
            amp = float(rng.uniform(*PLATEAU_HEAT_C))
            field[start:start + width_rows, column] += np.float32(amp)

    on0, on1 = PLATEAU_ON
    heat_x = 16 + int(rng.integers(0, 10))
    heat_w = int(rng.integers(PLATEAU_HEAT_WIDTH_M[0], PLATEAU_HEAT_WIDTH_M[1] + 1))
    _stamp(
        field, labels, event_id, on0, on1, heat_x, heat_x + heat_w,
        float(rng.uniform(*PLATEAU_HEAT_C)), LABEL_INDEX["Local_Heating"], 1,
    )
    cool_x = 80 + int(rng.integers(0, 10))
    cool_w = int(rng.integers(PLATEAU_HEAT_WIDTH_M[0], PLATEAU_HEAT_WIDTH_M[1] + 1))
    _stamp(
        field, labels, event_id, on0, on1, cool_x, cool_x + cool_w,
        -float(rng.uniform(*PLATEAU_COOL_C)), LABEL_INDEX["Local_Cooling"], 2,
    )
    period_x = 160 + int(rng.integers(0, 10))
    period_amp = float(rng.uniform(*PLATEAU_PERIOD_C))
    for row in (8, 96, 220):
        _stamp(
            field, labels, event_id, row, row + 4, period_x, period_x + 6,
            period_amp, LABEL_INDEX["Periodic_Thermal"], 3,
        )
    fault_x = 240 + int(rng.integers(0, 8))
    _stamp(
        field, labels, event_id, 30, 31, fault_x, fault_x + 1,
        float(rng.uniform(*PLATEAU_FAULT_C)), LABEL_INDEX["Sensor_Fault"], 4,
    )
    _stamp(
        field, labels, event_id, on0, on1, 300, 300 + PLATEAU_BROAD_WIDTH_M,
        float(rng.uniform(*PLATEAU_BROAD_C)), LABEL_INDEX["Broad_Shift"], 5,
    )
    labels[~valid] = 0
    event_id[~valid] = 0
    shift = np.float32(PLATEAU_LEVEL_C - float(np.median(field[valid])))
    field[valid] = field[valid] + shift
    events = (
        {"id": 1, "label": "Local_Heating"},
        {"id": 2, "label": "Local_Cooling"},
        {"id": 3, "label": "Periodic_Thermal"},
        {"id": 4, "label": "Sensor_Fault"},
        {"id": 5, "label": "Broad_Shift"},
    )
    return {
        "kind": "plateau",
        "label_source": "generator",
        "field": field,
        "valid": valid,
        "x_m": x_m,
        "time_s": time_s,
        "labels": labels,
        "event_id": event_id,
        "events": events,
    }


def dump_plateau_model(model: SoftmaxSceneModel, extra: dict | None = None) -> dict:
    if model.weight is None or model.mu is None or model.sig is None or model.bias is None:
        raise RuntimeError("model is not fit")
    payload = {
        "family": "softmax_plateau",
        "features": list(PLATEAU_FEATURES),
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


def load_plateau_model(payload: dict) -> SoftmaxSceneModel:
    if list(payload["features"]) != list(PLATEAU_FEATURES):
        raise ValueError("plateau feature list does not match this code")
    if int(payload["context_side"]) != CONTEXT_SIDE or float(payload["peak_fraction"]) != CONTEXT_PEAK_FRACTION:
        raise ValueError("context constants do not match this code")
    model = SoftmaxSceneModel()
    model.mu = np.asarray(payload["mu"], dtype=np.float64)
    model.sig = np.asarray(payload["sig"], dtype=np.float64)
    model.weight = np.asarray(payload["weight"], dtype=np.float64)
    model.bias = np.asarray(payload["bias"], dtype=np.float64)
    return model
