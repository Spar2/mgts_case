"""Two heating arms. The median-excess tail is empty on a noisy cable.

Seed 10, 2000 windows a record, no witness boxes. Record 5 has lag-1
p99 = 0.68 and warm-tail share 0. The same tail needs lag-1 >= 0.81.
Record 6 has |z| p99 = 2.87. Its long-hot columns, hot share >= 0.3,
have lag-1 p10 = 0.83 and cold-baseline excess p50 = 2.16 °C.
Seed 11, 1500 windows a record. Record 4 MAD p90 = 0.47 and p50 = 0.20,
and 0.726 of its windows have MAD <= 0.25. Record 5 MAD p10 = 0.58, so
every sampled window has MAD >= 0.50, and |peak| p10 = 0.53. Record 4's
MAD >= 0.50 columns have lag-1 p10 = 0.97, so a low-lag arm does not
open there. Heating is the warm tail, or a noisy low-lag peak, or a
smooth column whose cold baseline still sees the warm mode. Everything
else is background. Witness coordinates are not an input.
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

REGIME_LEVEL_C = 22.746278285980225
REGIME_INVALID_SHARE = 0.047769979141731485
REGIME_STANDING_C = 3.85
REGIME_PHI = 0.55
REGIME_INNOVATION_C = 0.30
REGIME_NOISY_C = 0.93
REGIME_MAD_FLOOR = 0.05
# Warm tail. Record 5's lag-1 never reaches it.
REGIME_WARM_EXCESS = 2.19
REGIME_WARM_AC1 = 0.81
REGIME_WARM_Z = 3.6
# Noisy arm. MAD cut is the gap between record 4 p90 and record 5 p10.
REGIME_NOISY_MAD = 0.50
REGIME_NOISY_AC1 = 0.32
REGIME_NOISY_PEAK = 0.53
# Long smooth arm. Cold excess p50 of record 6's long-hot columns.
REGIME_LONG_AC1 = 0.81
REGIME_LONG_COLD = 2.16
REGIME_HEAT_C = (2.78, 4.50)
REGIME_LONG_C = (2.80, 4.50)
REGIME_LONG_ROWS = 100
REGIME_BUMP_C = (1.20, 2.20)
REGIME_COOL_C = (2.78, 4.50)
REGIME_BROAD_C = (0.59, 1.15)
REGIME_PERIOD_C = (4.5, 6.5)
REGIME_PERIOD_ROWS = 4
REGIME_FAULT_C = (14.0, 18.0)
REGIME_ON = (28, 84)
REGIME_HEAT_X = (30, 36)
REGIME_LONG_X = (100, 106)
REGIME_COOL_X = (170, 176)
REGIME_PERIOD_X = (240, 248)
REGIME_NOISY_X = (300, 400)
REGIME_BUMP_X = (330, 336)
REGIME_FAULT_X = 420
REGIME_BROAD_X = (470, 520)
REGIME_SPIKE_X = (560, 566)
REGIME_T = 168
REGIME_X = 640
REGIME_TRAIN_SEEDS = tuple(range(2_100_000, 2_100_024))
REGIME_STEPS = 500
REGIME_PER_EVENT = 128
REGIME_FEATURES = CONTEXT_FEATURES + (
    "excess",
    "ac1",
    "z",
    "mad_peak",
    "ac1_peak",
    "cold",
    "ac1_cold",
)


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


def _column_stats(field: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Median, MAD, cold baseline, and lag-1 of each column."""
    n_space = field.shape[1]
    med = np.full(n_space, np.nan, dtype=np.float64)
    mad = np.full(n_space, np.nan, dtype=np.float64)
    cold = np.full(n_space, np.nan, dtype=np.float64)
    ac1 = np.zeros(n_space, dtype=np.float64)
    for column in range(n_space):
        series = field[:, column]
        series = series[valid[:, column] & np.isfinite(series)]
        if series.size < TIME_CROP:
            continue
        center = float(np.median(series))
        med[column] = center
        mad[column] = float(np.median(np.abs(series - center)))
        below = series[series <= center]
        cold[column] = float(np.median(below)) if below.size else center
        ac1[column] = _ac1(series - center)
    return med, mad, cold, ac1


def _window_medians(field: np.ndarray, valid: np.ndarray, t0: int) -> np.ndarray:
    slab = np.asarray(field[t0:t0 + TIME_CROP], dtype=np.float64)
    mask = np.asarray(valid[t0:t0 + TIME_CROP], dtype=bool)
    out = np.full(slab.shape[1], np.nan, dtype=np.float64)
    live = mask.any(axis=0)
    if live.any():
        out[live] = np.nanmedian(np.where(mask[:, live], slab[:, live], np.nan), axis=0)
    return out


def _crop_features(
    window: np.ndarray,
    med: np.ndarray,
    mad: np.ndarray,
    cold: np.ndarray,
    ac1: np.ndarray,
    x0: int,
) -> tuple[float, float, float, float, float, float, float]:
    """Excess arm, peak-column scale, and cold-baseline arm inside one crop."""
    sl = slice(x0, x0 + SPACE_CROP)
    sample = window[sl]
    base = med[sl]
    scale = mad[sl]
    finite = np.isfinite(sample) & np.isfinite(base) & np.isfinite(scale)
    excess = ac_ex = zee = 0.0
    mad_peak = ac_peak = cold_ex = ac_cold = 0.0
    if not finite.any():
        return excess, ac_ex, zee, mad_peak, ac_peak, cold_ex, ac_cold
    delta = sample[finite] - base[finite]
    pick = int(np.argmax(np.abs(delta)))
    column = x0 + int(np.flatnonzero(finite)[pick])
    excess = float(delta[pick])
    ac_ex = float(ac1[column])
    zee = abs(excess) / max(float(mad[column]), REGIME_MAD_FLOOR)
    center = float(np.nanmedian(sample[finite]))
    shape = sample[finite] - center
    peak_at = int(np.nanargmax(np.abs(shape)))
    peak_column = x0 + int(np.flatnonzero(finite)[peak_at])
    mad_peak = float(mad[peak_column]) if np.isfinite(mad[peak_column]) else 0.0
    ac_peak = float(ac1[peak_column])
    cold_delta = sample - cold[sl]
    cold_finite = np.isfinite(cold_delta)
    if cold_finite.any():
        values = cold_delta[cold_finite]
        cold_pick = int(np.argmax(values))
        cold_column = x0 + int(np.flatnonzero(cold_finite)[cold_pick])
        cold_ex = float(values[cold_pick])
        ac_cold = float(ac1[cold_column])
    return excess, ac_ex, zee, mad_peak, ac_peak, cold_ex, ac_cold


def regime_table(field: np.ndarray, valid: np.ndarray, origins: np.ndarray) -> np.ndarray:
    """Context features plus the three heating arms.

    An empty origin list has shape (0, 12).
    """
    origins = np.asarray(origins, dtype=np.int64)
    context = context_table(field, valid, origins)
    extra = np.zeros((len(origins), 7), dtype=np.float64)
    if len(origins) == 0:
        return np.column_stack((context, extra))
    valid = np.asarray(valid, dtype=bool)
    med, mad, cold, ac1 = _column_stats(np.asarray(field, dtype=np.float64), valid)
    cache: dict[int, np.ndarray] = {}
    for index, (t0, x0) in enumerate(origins):
        t0 = int(t0)
        if t0 not in cache:
            cache[t0] = _window_medians(field, valid, t0)
        extra[index] = _crop_features(cache[t0], med, mad, cold, ac1, int(x0))
    return np.column_stack((context, extra))


def _columns(features: np.ndarray) -> tuple[np.ndarray, ...]:
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    names = {name: matrix[:, REGIME_FEATURES.index(name)] for name in (
        "peak", "excess", "ac1", "z", "mad_peak", "ac1_peak", "cold", "ac1_cold",
    )}
    return (
        names["peak"],
        names["excess"],
        names["ac1"],
        names["z"],
        names["mad_peak"],
        names["ac1_peak"],
        names["cold"],
        names["ac1_cold"],
    )


def regime_mask(features: np.ndarray) -> np.ndarray:
    """True on the warm tail, the noisy peak, or the long smooth mode."""
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if matrix.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    peak, excess, ac1, zee, mad_peak, ac1_peak, cold, ac1_cold = _columns(matrix)
    warm = (excess >= REGIME_WARM_EXCESS) & (ac1 >= REGIME_WARM_AC1) & (zee >= REGIME_WARM_Z)
    noisy = (mad_peak >= REGIME_NOISY_MAD) & (ac1_peak <= REGIME_NOISY_AC1) & (np.abs(peak) >= REGIME_NOISY_PEAK)
    long = (ac1_cold >= REGIME_LONG_AC1) & (cold >= REGIME_LONG_COLD)
    return warm | noisy | long


def regime_names(model: SoftmaxSceneModel, features: np.ndarray) -> list[str]:
    """Local_Heating on the measured regime. Every other window is Background."""
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if matrix.shape[0] == 0:
        return []
    if model.weight is None:
        raise RuntimeError("model is not fit")
    mask = regime_mask(matrix)
    return ["Local_Heating" if flag else "Background" for flag in mask]


def _stamp_level(field, profile, labels, event_id, t0, t1, x0, x1, value, class_id, eid) -> None:
    field[t0:t1, x0:x1] = profile[x0:x1] + np.float32(value)
    if class_id:
        labels[t0:t1, x0:x1] = class_id
        event_id[t0:t1, x0:x1] = eid


def _ar_field(rng: np.random.Generator, n_time: int, n_live: int) -> np.ndarray:
    innov = rng.normal(0.0, REGIME_INNOVATION_C, size=(n_time, n_live))
    out = np.empty_like(innov)
    out[0] = innov[0]
    phi = REGIME_PHI
    for row in range(1, n_time):
        out[row] = phi * out[row - 1] + innov[row]
    return out.astype(np.float32)


def plant_regime(seed: int = 0) -> dict:
    """A short warm tail, a long mode, and a noisy spatial peak.

    The long mode is added to the AR column, so the cold baseline stays
    under the median. The noisy band shares one white-noise series, so a
    crop without the bump has no spatial peak. The standing spike is a
    full-time shift on the smooth field, so no arm opens. Event bands stay
    40 columns apart.
    """
    rng = np.random.default_rng(int(seed))
    n_dead = int(round(REGIME_INVALID_SHARE * REGIME_X))
    n_live = REGIME_X - n_dead
    x_m = np.arange(REGIME_X, dtype=np.float64)
    time_s = np.arange(REGIME_T, dtype=np.float64) * 60.0
    field = np.full((REGIME_T, REGIME_X), SENTINEL_VALUE, dtype=np.float32)
    valid = np.zeros(field.shape, dtype=bool)
    valid[:, :n_live] = True
    x = np.arange(n_live, dtype=np.float64)
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    wave = np.sin(2.0 * np.pi * x / float(REGIME_X) + phase)
    wave -= wave.mean()
    wave_peak = float(np.max(np.abs(wave)))
    if wave_peak > 0.0:
        wave *= 0.48 / wave_peak
    profile = np.full(REGIME_X, SENTINEL_VALUE, dtype=np.float32)
    profile[:n_live] = (REGIME_LEVEL_C + wave).astype(np.float32)
    field[:, :n_live] = profile[:n_live] + _ar_field(rng, REGIME_T, n_live)
    labels = np.zeros(field.shape, dtype=np.int8)
    event_id = np.zeros(field.shape, dtype=np.int16)

    heat = float(rng.uniform(*REGIME_HEAT_C))
    long_amp = float(rng.uniform(*REGIME_LONG_C))
    cool = float(rng.uniform(*REGIME_COOL_C))
    broad = float(rng.uniform(*REGIME_BROAD_C))
    bump = float(rng.uniform(*REGIME_BUMP_C))
    on0, on1 = REGIME_ON
    _stamp_level(
        field, profile, labels, event_id, on0, on1, *REGIME_HEAT_X,
        heat, LABEL_INDEX["Local_Heating"], 1,
    )
    long0, long1 = REGIME_LONG_X
    field[0:REGIME_LONG_ROWS, long0:long1] = (
        field[0:REGIME_LONG_ROWS, long0:long1] + np.float32(long_amp)
    )
    labels[0:REGIME_LONG_ROWS, long0:long1] = LABEL_INDEX["Local_Heating"]
    event_id[0:REGIME_LONG_ROWS, long0:long1] = 6
    _stamp_level(
        field, profile, labels, event_id, on0, on1, *REGIME_COOL_X,
        -cool, LABEL_INDEX["Local_Cooling"], 2,
    )
    _stamp_level(
        field, profile, labels, event_id, on0, on1, *REGIME_BROAD_X,
        broad, LABEL_INDEX["Broad_Shift"], 5,
    )
    noisy0, noisy1 = REGIME_NOISY_X
    shared = rng.normal(0.0, REGIME_NOISY_C, size=(REGIME_T, 1)).astype(np.float32)
    field[:, noisy0:noisy1] = profile[noisy0:noisy1] + shared
    bump0, bump1 = REGIME_BUMP_X
    field[:, bump0:bump1] = field[:, bump0:bump1] + np.float32(bump)
    labels[:, bump0:bump1] = LABEL_INDEX["Local_Heating"]
    event_id[:, bump0:bump1] = 7
    spike0, spike1 = REGIME_SPIKE_X
    field[:, spike0:spike1] = field[:, spike0:spike1] + np.float32(REGIME_STANDING_C)

    period_amp = float(rng.uniform(*REGIME_PERIOD_C))
    for row in (8, 48, 96, 140):
        _stamp(
            field, labels, event_id, row, row + REGIME_PERIOD_ROWS, *REGIME_PERIOD_X,
            period_amp, LABEL_INDEX["Periodic_Thermal"], 3,
        )
    _stamp(
        field, labels, event_id, 30, 31, REGIME_FAULT_X, REGIME_FAULT_X + 1,
        float(rng.uniform(*REGIME_FAULT_C)), LABEL_INDEX["Sensor_Fault"], 4,
    )
    labels[~valid] = 0
    event_id[~valid] = 0
    shift = np.float32(REGIME_LEVEL_C - float(np.median(field[valid])))
    field[valid] = field[valid] + shift
    events = (
        {"id": 1, "label": "Local_Heating"},
        {"id": 6, "label": "Local_Heating"},
        {"id": 7, "label": "Local_Heating"},
        {"id": 2, "label": "Local_Cooling"},
        {"id": 3, "label": "Periodic_Thermal"},
        {"id": 4, "label": "Sensor_Fault"},
        {"id": 5, "label": "Broad_Shift"},
    )
    return {
        "kind": "regime",
        "label_source": "generator",
        "field": field,
        "valid": valid,
        "x_m": x_m,
        "time_s": time_s,
        "labels": labels,
        "event_id": event_id,
        "events": events,
    }


def dump_regime_model(model: SoftmaxSceneModel, extra: dict | None = None) -> dict:
    if model.weight is None or model.mu is None or model.sig is None or model.bias is None:
        raise RuntimeError("model is not fit")
    payload = {
        "family": "softmax_regime",
        "features": list(REGIME_FEATURES),
        "context_side": CONTEXT_SIDE,
        "peak_fraction": CONTEXT_PEAK_FRACTION,
        "spike_mad_floor": SPIKE_MAD_FLOOR,
        "warm_excess": REGIME_WARM_EXCESS,
        "warm_ac1": REGIME_WARM_AC1,
        "warm_z": REGIME_WARM_Z,
        "noisy_mad": REGIME_NOISY_MAD,
        "noisy_ac1": REGIME_NOISY_AC1,
        "noisy_peak": REGIME_NOISY_PEAK,
        "long_ac1": REGIME_LONG_AC1,
        "long_cold": REGIME_LONG_COLD,
        "labels": list(LABELS),
        "mu": model.mu.tolist(),
        "sig": model.sig.tolist(),
        "weight": model.weight.tolist(),
        "bias": model.bias.tolist(),
    }
    if extra:
        payload.update(extra)
    return payload


def load_regime_model(payload: dict) -> SoftmaxSceneModel:
    if list(payload["features"]) != list(REGIME_FEATURES):
        raise ValueError("regime feature list does not match this code")
    if int(payload["context_side"]) != CONTEXT_SIDE or float(payload["peak_fraction"]) != CONTEXT_PEAK_FRACTION:
        raise ValueError("context constants do not match this code")
    expected = {
        "warm_excess": REGIME_WARM_EXCESS,
        "warm_ac1": REGIME_WARM_AC1,
        "warm_z": REGIME_WARM_Z,
        "noisy_mad": REGIME_NOISY_MAD,
        "noisy_ac1": REGIME_NOISY_AC1,
        "noisy_peak": REGIME_NOISY_PEAK,
        "long_ac1": REGIME_LONG_AC1,
        "long_cold": REGIME_LONG_COLD,
    }
    for key, value in expected.items():
        if float(payload[key]) != value:
            raise ValueError("regime cuts do not match this code")
    model = SoftmaxSceneModel()
    model.mu = np.asarray(payload["mu"], dtype=np.float64)
    model.sig = np.asarray(payload["sig"], dtype=np.float64)
    model.weight = np.asarray(payload["weight"], dtype=np.float64)
    model.bias = np.asarray(payload["bias"], dtype=np.float64)
    return model
