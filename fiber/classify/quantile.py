"""Where a crop sits inside its own column.

On 12000 scored windows from records 1..6, seed 6, and no witness boxes, the
MAD of a column around its own median has p50 = 0.550 °C. The absolute
excess divided by that MAD has p95 = 3.58 and p99 = 9.99. The share of crop
rows above the column's own p95 has p90 = 0.25 and p99 = 0.57, and only
1.5% of windows are above half the crop. Heating here is a short block in
that standardized tail. A burst at the p90 occupancy stays background.
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
QUANTILE_LEVEL_C = 22.746278285980225
QUANTILE_INVALID_SHARE = 0.047769979141731485
# Spatial peak p90 on the earlier context subsample.
QUANTILE_STANDING_AMP_C = 2.79
# Noise whose MAD is the measured window-MAD p50. 0.550 / 0.6745 ≈ 0.82.
QUANTILE_NOISE_C = 0.82
QUANTILE_MAD_C = 0.55
# Absolute z p95 to a level under the p99 of 9.99.
QUANTILE_HEAT_Z = (3.6, 6.0)
QUANTILE_COOL_Z = (3.6, 6.0)
# Body of |z|, whose p50 is 1.19. A wide shift stays here.
QUANTILE_BROAD_Z = (0.8, 1.5)
# round(0.57 * 28). The p99 occupancy of a column's own p95.
QUANTILE_HEAT_ROWS = 16
# round(0.25 * 28). The p90 occupancy. Unlabeled.
QUANTILE_DECOY_ROWS = 7
# Narrower than one crop. Two shoulder rows at this width stay under eight cells,
# so only the window that holds all sixteen rows is labeled.
QUANTILE_HEAT_WIDTH = 3
# Wider than one crop, so an interior window is background.
QUANTILE_BROAD_WIDTH = 56
# Spike p50 to p90, and above the spike p99 of 14.2 °C.
QUANTILE_PERIOD_C = (4.5, 6.5)
QUANTILE_PERIOD_ROWS = 4
QUANTILE_FAULT_C = (14.0, 18.0)
QUANTILE_BLOCKS = (0, 44, 88, 132)
QUANTILE_T = 168
QUANTILE_X = 640
QUANTILE_TRAIN_SEEDS = tuple(range(1_700_000, 1_700_024))
QUANTILE_STEPS = 500
QUANTILE_PER_EVENT = 128
QUANTILE_FEATURES = CONTEXT_FEATURES + ("z", "rank", "tail_rows")


def _column_stats(residual: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    n_space = residual.shape[1]
    mad = np.full(n_space, np.nan, dtype=np.float64)
    high = np.full(n_space, np.nan, dtype=np.float64)
    low = np.full(n_space, np.nan, dtype=np.float64)
    order: list = [None] * n_space
    for column in range(n_space):
        series = residual[:, column]
        series = series[np.isfinite(series)]
        if series.size < TIME_CROP:
            continue
        center = float(np.median(series))
        mad[column] = float(np.median(np.abs(series - center)))
        high[column] = float(np.quantile(series, 0.95))
        low[column] = float(np.quantile(series, 0.05))
        order[column] = np.sort(series)
    return mad, high, low, order


def _column_excess(residual: np.ndarray, t0: int) -> np.ndarray:
    slab = residual[t0:t0 + TIME_CROP]
    out = np.full(slab.shape[1], np.nan, dtype=np.float64)
    live = np.isfinite(slab).any(axis=0)
    if live.any():
        out[live] = np.nanmedian(slab[:, live], axis=0)
    return out


def _window_extra(
    excess: np.ndarray,
    residual: np.ndarray,
    t0: int,
    x0: int,
    mad: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    order: list,
) -> tuple[float, float, float]:
    """Standardized excess, its rank inside the column, and the tail occupancy."""
    center = excess[x0:x0 + SPACE_CROP]
    finite = np.isfinite(center)
    if not finite.any():
        return 0.0, 0.0, 0.0
    values = center[finite]
    pick = int(np.argmax(np.abs(values)))
    peak = float(values[pick])
    column = x0 + int(np.flatnonzero(finite)[pick])
    samples = order[column]
    scale = mad[column]
    if samples is None or not np.isfinite(scale):
        return 0.0, 0.0, 0.0
    z = peak / max(float(scale), SPIKE_MAD_FLOOR)
    if peak >= 0.0:
        rank = float(np.searchsorted(samples, peak, side="right") / samples.size)
        gate = float(high[column])
        series = residual[t0:t0 + TIME_CROP, column]
        series = series[np.isfinite(series)]
        tail = float(np.mean(series >= gate)) if series.size else 0.0
    else:
        rank = float(np.searchsorted(samples, peak, side="left") / samples.size - 1.0)
        gate = float(low[column])
        series = residual[t0:t0 + TIME_CROP, column]
        series = series[np.isfinite(series)]
        tail = float(np.mean(series <= gate)) if series.size else 0.0
    return z, rank, tail


def quantile_table(field: np.ndarray, valid: np.ndarray, origins: np.ndarray) -> np.ndarray:
    """Context features plus standardized excess, column rank, and tail occupancy.

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
    mad, high, low, order = _column_stats(residual)
    cache: dict[int, np.ndarray] = {}
    for index, (t0, x0) in enumerate(origins):
        t0 = int(t0)
        if t0 not in cache:
            cache[t0] = _column_excess(residual, t0)
        extra[index] = _window_extra(cache[t0], residual, t0, int(x0), mad, high, low, order)
    return np.column_stack((context, extra))


def _stamp_level(field, profile, labels, event_id, t0, t1, x0, x1, value, class_id, eid) -> None:
    """Replace the standing level with one flat offset. Noise does not break the tie."""
    field[t0:t1, x0:x1] = profile[x0:x1] + np.float32(value)
    labels[t0:t1, x0:x1] = class_id
    event_id[t0:t1, x0:x1] = eid


def plant_quantile(seed: int = 0) -> dict:
    """Standing bow plus a flat block in the standardized tail.

    The block is sixteen rows, so the crop median sits on that level and the
    column rank is in the tail. A two-row shoulder is not labeled. A 7-row
    burst is not an event. The wide block stays at the body of |z|. Labels
    are the painted cells. Event bands stay 40 columns apart.
    """
    rng = np.random.default_rng(int(seed))
    n_dead = int(round(QUANTILE_INVALID_SHARE * QUANTILE_X))
    n_live = QUANTILE_X - n_dead
    x_m = np.arange(QUANTILE_X, dtype=np.float64)
    time_s = np.arange(QUANTILE_T, dtype=np.float64) * 60.0
    field = np.full((QUANTILE_T, QUANTILE_X), SENTINEL_VALUE, dtype=np.float32)
    valid = np.zeros(field.shape, dtype=bool)
    valid[:, :n_live] = True
    x = np.arange(n_live, dtype=np.float64)
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    wave = np.sin(2.0 * np.pi * x / float(QUANTILE_X) + phase)
    wave -= wave.mean()
    peak = float(np.max(np.abs(wave)))
    if peak > 0.0:
        wave *= QUANTILE_STANDING_AMP_C / peak
    profile = np.full(QUANTILE_X, SENTINEL_VALUE, dtype=np.float32)
    profile[:n_live] = (QUANTILE_LEVEL_C + wave).astype(np.float32)
    field[:, :n_live] = profile[:n_live]
    field[:, :n_live] += rng.normal(0.0, QUANTILE_NOISE_C, size=(QUANTILE_T, n_live)).astype(np.float32)
    labels = np.zeros(field.shape, dtype=np.int8)
    event_id = np.zeros(field.shape, dtype=np.int16)

    heat = float(rng.uniform(*QUANTILE_HEAT_Z)) * QUANTILE_MAD_C
    cool = float(rng.uniform(*QUANTILE_COOL_Z)) * QUANTILE_MAD_C
    broad = float(rng.uniform(*QUANTILE_BROAD_Z)) * QUANTILE_MAD_C
    # Only the first block. Later starts overlap the stride by 8–12 rows and
    # would label a window whose median never reaches the level.
    start = QUANTILE_BLOCKS[0]
    end = start + QUANTILE_HEAT_ROWS
    for origin in (30, 80):
        _stamp_level(
            field, profile, labels, event_id, start, end, origin, origin + QUANTILE_HEAT_WIDTH,
            heat, LABEL_INDEX["Local_Heating"], 1,
        )
    for origin in (140, 190):
        _stamp_level(
            field, profile, labels, event_id, start, end, origin, origin + QUANTILE_HEAT_WIDTH,
            -cool, LABEL_INDEX["Local_Cooling"], 2,
        )
    for start in QUANTILE_BLOCKS:
        end = start + QUANTILE_HEAT_ROWS
        _stamp(
            field, labels, event_id, start, end, 400, 400 + QUANTILE_BROAD_WIDTH,
            broad, LABEL_INDEX["Broad_Shift"], 5,
        )
        decoy_at = start + QUANTILE_HEAT_ROWS + 4
        field[decoy_at:decoy_at + QUANTILE_DECOY_ROWS, 560] += np.float32(heat)

    period_amp = float(rng.uniform(*QUANTILE_PERIOD_C))
    for row in (8, 36, 70, 110):
        _stamp(
            field, labels, event_id, row, row + QUANTILE_PERIOD_ROWS, 240, 248,
            period_amp, LABEL_INDEX["Periodic_Thermal"], 3,
        )
    _stamp(
        field, labels, event_id, 30, 31, 320, 321,
        float(rng.uniform(*QUANTILE_FAULT_C)), LABEL_INDEX["Sensor_Fault"], 4,
    )
    labels[~valid] = 0
    event_id[~valid] = 0
    shift = np.float32(QUANTILE_LEVEL_C - float(np.median(field[valid])))
    field[valid] = field[valid] + shift
    events = (
        {"id": 1, "label": "Local_Heating"},
        {"id": 2, "label": "Local_Cooling"},
        {"id": 3, "label": "Periodic_Thermal"},
        {"id": 4, "label": "Sensor_Fault"},
        {"id": 5, "label": "Broad_Shift"},
    )
    return {
        "kind": "quantile",
        "label_source": "generator",
        "field": field,
        "valid": valid,
        "x_m": x_m,
        "time_s": time_s,
        "labels": labels,
        "event_id": event_id,
        "events": events,
    }


def dump_quantile_model(model: SoftmaxSceneModel, extra: dict | None = None) -> dict:
    if model.weight is None or model.mu is None or model.sig is None or model.bias is None:
        raise RuntimeError("model is not fit")
    payload = {
        "family": "softmax_quantile",
        "features": list(QUANTILE_FEATURES),
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


def load_quantile_model(payload: dict) -> SoftmaxSceneModel:
    if list(payload["features"]) != list(QUANTILE_FEATURES):
        raise ValueError("quantile feature list does not match this code")
    if int(payload["context_side"]) != CONTEXT_SIDE or float(payload["peak_fraction"]) != CONTEXT_PEAK_FRACTION:
        raise ValueError("context constants do not match this code")
    model = SoftmaxSceneModel()
    model.mu = np.asarray(payload["mu"], dtype=np.float64)
    model.sig = np.asarray(payload["sig"], dtype=np.float64)
    model.weight = np.asarray(payload["weight"], dtype=np.float64)
    model.bias = np.asarray(payload["bias"], dtype=np.float64)
    return model
