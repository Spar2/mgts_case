"""Synthetic DTS field matched to the six records, not to witness boxes.

The three constants are the mean of per-record summaries measured before
this generator was scored on witnesses. A flat planted scene sits near 22 °C
with almost no along-cable spread and a 10% dead tail.
"""

from __future__ import annotations

import numpy as np

from fiber.classify.plant import LABEL_INDEX, _stamp
from fiber.quality import SENTINEL_VALUE

# Mean of the six records: median valid temperature, std of column medians,
# and the share of cells that are invalid or sentinel.
ROOM_LEVEL_C = 22.746278285980225
ROOM_ALONG_C = 2.3394429475256957
ROOM_INVALID_SHARE = 0.047769979141731485
ROOM_T = 96
ROOM_X = 480
ROOM_NOISE_C = 0.15
ROOM_PERIODS = (40, 80, 160, 320)
ROOM_TRAIN_SEEDS = tuple(range(1_100_000, 1_100_024))
ROOM_STEPS = 500
ROOM_PER_EVENT = 128
SUMMARY_KEYS = ("level_c", "along_cable_c", "invalid_share")


def field_summary(field: np.ndarray, valid: np.ndarray) -> dict:
    """Temperature level, along-cable variation, and invalid/sentinel share."""
    values = np.asarray(field, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    live = values[mask]
    columns = []
    for index in range(values.shape[1]):
        cells = values[:, index][mask[:, index]]
        if cells.size:
            columns.append(float(np.median(cells)))
    spread = np.asarray(columns, dtype=np.float64)
    return {
        "level_c": float(np.median(live)) if live.size else 0.0,
        "along_cable_c": float(np.std(spread)) if spread.size else 0.0,
        "invalid_share": float(np.mean(~mask)),
    }


def records_target(records) -> dict:
    """Equal-weight mean of `field_summary` over records. No window coordinates."""
    parts = [field_summary(record.raw_c, record.valid) for record in records]
    return {
        key: float(np.mean([part[key] for part in parts]))
        for key in SUMMARY_KEYS
    }


def summary_gaps(summary: dict, target: dict) -> dict:
    return {key: abs(float(summary[key]) - float(target[key])) for key in SUMMARY_KEYS}


def _along_cable_wave(n_live: int, rng: np.random.Generator) -> np.ndarray:
    x = np.arange(n_live, dtype=np.float64)
    wave = np.zeros(n_live, dtype=np.float64)
    for period in ROOM_PERIODS:
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        weight = float(rng.normal())
        wave += weight * np.sin(2.0 * np.pi * x / period + phase)
    wave -= wave.mean()
    spread = float(np.std(wave))
    if spread > 0.0:
        wave *= ROOM_ALONG_C / spread
    return wave


def plant_room(seed: int = 0) -> dict:
    """One cable with the record-level temperature, spread, and dead share.

    Class ids are the painted events. They are not a shape-rule reading.
    """
    rng = np.random.default_rng(int(seed))
    n_dead = int(round(ROOM_INVALID_SHARE * ROOM_X))
    n_live = ROOM_X - n_dead
    x_m = np.arange(ROOM_X, dtype=np.float64)
    time_s = np.arange(ROOM_T, dtype=np.float64) * 60.0
    field = np.full((ROOM_T, ROOM_X), SENTINEL_VALUE, dtype=np.float32)
    valid = np.zeros(field.shape, dtype=bool)
    valid[:, :n_live] = True
    wave = _along_cable_wave(n_live, rng)
    noise = rng.normal(0.0, ROOM_NOISE_C, size=(ROOM_T, n_live)).astype(np.float32)
    field[:, :n_live] = (ROOM_LEVEL_C + wave).astype(np.float32) + noise
    labels = np.zeros(field.shape, dtype=np.int8)
    event_id = np.zeros(field.shape, dtype=np.int16)

    heat_x = 24 + int(rng.integers(0, 8))
    heat_w = int(rng.integers(5, 17))
    heat_id = LABEL_INDEX["Local_Heating"]
    _stamp(field, labels, event_id, 0, ROOM_T, heat_x, heat_x + heat_w,
           float(rng.uniform(5.0, 7.0)), heat_id, 1)

    cool_x = 100 + int(rng.integers(0, 8))
    cool_w = int(rng.integers(5, 17))
    cool_id = LABEL_INDEX["Local_Cooling"]
    _stamp(field, labels, event_id, 0, ROOM_T, cool_x, cool_x + cool_w,
           -float(rng.uniform(4.5, 6.5)), cool_id, 2)

    period_x = 180 + int(rng.integers(0, 8))
    period_id = LABEL_INDEX["Periodic_Thermal"]
    period_amp = float(rng.uniform(4.5, 6.5))
    for row in (6, 36, 66):
        _stamp(field, labels, event_id, row, row + 4, period_x, period_x + 6,
               period_amp, period_id, 3)

    fault_x = 250 + int(rng.integers(0, 8))
    fault_t = 40 + int(rng.integers(0, 20))
    fault_id = LABEL_INDEX["Sensor_Fault"]
    field[fault_t, fault_x] += float(rng.uniform(14.0, 20.0))
    labels[fault_t, fault_x] = fault_id
    event_id[fault_t, fault_x] = 4

    broad_id = LABEL_INDEX["Broad_Shift"]
    field[:, 320:370] += float(rng.uniform(3.2, 4.5))
    labels[:, 320:370] = broad_id
    event_id[:, 320:370] = 5

    labels[~valid] = 0
    event_id[~valid] = 0
    shift = np.float32(ROOM_LEVEL_C - float(np.median(field[valid])))
    field[valid] = field[valid] + shift
    events = (
        {"id": 1, "label": "Local_Heating"},
        {"id": 2, "label": "Local_Cooling"},
        {"id": 3, "label": "Periodic_Thermal"},
        {"id": 4, "label": "Sensor_Fault"},
        {"id": 5, "label": "Broad_Shift"},
    )
    return {
        "kind": "room",
        "label_source": "generator",
        "field": field,
        "valid": valid,
        "x_m": x_m,
        "time_s": time_s,
        "labels": labels,
        "event_id": event_id,
        "events": events,
    }
