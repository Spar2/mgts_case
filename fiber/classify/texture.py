"""Synthetic cable texture matched to bulk window features, not to witness boxes.

The three textures were fixed from global percentiles of the five context
features on the six records, and from synthetic windows that do not touch a
painted event. Witness coordinates were not an input.
"""

from __future__ import annotations

import numpy as np

from fiber.classify.plant import LABEL_INDEX, _stamp
from fiber.quality import SENTINEL_VALUE

# Level and invalid share are the same record means already published in room.py.
# They were measured without witness coordinates.
TEXTURE_LEVEL_C = 22.746278285980225
TEXTURE_INVALID_SHARE = 0.047769979141731485
TEXTURE_T = 96
TEXTURE_X = 480
TEXTURE_KINDS = (
    {"amp": 0.35, "noise": 0.18, "periods": (24, 48, 96)},
    {"amp": 0.45, "noise": 0.15, "periods": (16, 32, 64)},
    {"amp": 0.30, "noise": 0.10, "periods": (8, 16, 32)},
)
TEXTURE_HEAT_C = (4.5, 6.5)
TEXTURE_COOL_C = (4.0, 6.0)
TEXTURE_PERIOD_C = (4.5, 6.5)
TEXTURE_BROAD_C = (3.2, 4.5)
TEXTURE_FAULT_C = (14.0, 18.0)
TEXTURE_TRAIN_SEEDS = tuple(range(1_200_000, 1_200_024))
TEXTURE_STEPS = 500
TEXTURE_PER_EVENT = 128


def plant_texture(seed: int = 0) -> dict:
    """One cable. Class ids are the painted events, not a shape-rule reading.

    Seed picks one of the three frozen textures. Event bands stay 40 columns
    apart so a center crop cannot hold two events.
    """
    rng = np.random.default_rng(int(seed))
    kind = TEXTURE_KINDS[int(seed) % len(TEXTURE_KINDS)]
    n_dead = int(round(TEXTURE_INVALID_SHARE * TEXTURE_X))
    n_live = TEXTURE_X - n_dead
    x_m = np.arange(TEXTURE_X, dtype=np.float64)
    time_s = np.arange(TEXTURE_T, dtype=np.float64) * 60.0
    field = np.full((TEXTURE_T, TEXTURE_X), SENTINEL_VALUE, dtype=np.float32)
    valid = np.zeros(field.shape, dtype=bool)
    valid[:, :n_live] = True
    x = np.arange(n_live, dtype=np.float64)
    wave = np.zeros(n_live, dtype=np.float64)
    for period in kind["periods"]:
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        weight = abs(float(rng.normal())) + 0.2
        wave += weight * np.sin(2.0 * np.pi * x / float(period) + phase)
    wave -= wave.mean()
    spread = float(np.std(wave))
    if spread > 0.0:
        wave *= float(kind["amp"]) / spread
    field[:, :n_live] = (TEXTURE_LEVEL_C + wave).astype(np.float32)
    field[:, :n_live] += rng.normal(0.0, float(kind["noise"]), size=(TEXTURE_T, n_live)).astype(np.float32)
    labels = np.zeros(field.shape, dtype=np.int8)
    event_id = np.zeros(field.shape, dtype=np.int16)

    heat_x = 16 + int(rng.integers(0, 25))
    heat_w = int(rng.integers(6, 13))
    _stamp(
        field, labels, event_id, 0, TEXTURE_T, heat_x, heat_x + heat_w,
        float(rng.uniform(*TEXTURE_HEAT_C)), LABEL_INDEX["Local_Heating"], 1,
    )
    cool_x = 100 + int(rng.integers(0, 21))
    cool_w = int(rng.integers(6, 13))
    _stamp(
        field, labels, event_id, 0, TEXTURE_T, cool_x, cool_x + cool_w,
        -float(rng.uniform(*TEXTURE_COOL_C)), LABEL_INDEX["Local_Cooling"], 2,
    )
    period_x = 190 + int(rng.integers(0, 21))
    period_amp = float(rng.uniform(*TEXTURE_PERIOD_C))
    for row in (6, 36, 66):
        _stamp(
            field, labels, event_id, row, row + 4, period_x, period_x + 6,
            period_amp, LABEL_INDEX["Periodic_Thermal"], 3,
        )
    fault_x = 270 + int(rng.integers(0, 11))
    fault_t = 40 + int(rng.integers(0, 20))
    _stamp(
        field, labels, event_id, fault_t, fault_t + 1, fault_x, fault_x + 1,
        float(rng.uniform(*TEXTURE_FAULT_C)), LABEL_INDEX["Sensor_Fault"], 4,
    )
    broad_x = 340
    _stamp(
        field, labels, event_id, 0, TEXTURE_T, broad_x, broad_x + 50,
        float(rng.uniform(*TEXTURE_BROAD_C)), LABEL_INDEX["Broad_Shift"], 5,
    )
    labels[~valid] = 0
    event_id[~valid] = 0
    shift = np.float32(TEXTURE_LEVEL_C - float(np.median(field[valid])))
    field[valid] = field[valid] + shift
    events = (
        {"id": 1, "label": "Local_Heating"},
        {"id": 2, "label": "Local_Cooling"},
        {"id": 3, "label": "Periodic_Thermal"},
        {"id": 4, "label": "Sensor_Fault"},
        {"id": 5, "label": "Broad_Shift"},
    )
    return {
        "kind": "texture",
        "texture_index": int(seed) % len(TEXTURE_KINDS),
        "label_source": "generator",
        "field": field,
        "valid": valid,
        "x_m": x_m,
        "time_s": time_s,
        "labels": labels,
        "event_id": event_id,
        "events": events,
    }
