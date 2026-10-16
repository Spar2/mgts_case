"""Planted temperature scenes with a known mask. Not a model of a real room."""

from __future__ import annotations

import numpy as np


def local_scene(seed: int = 0) -> dict:
    """Quiet cable, one hot blob, one cold blob, dead tail."""
    rng = np.random.default_rng(seed)
    x_m = np.arange(200, dtype=np.float64)
    time_s = np.arange(40, dtype=np.float64) * 60.0
    field = np.full((40, 200), 21.0, dtype=np.float32)
    field += rng.normal(0.0, 0.12, size=field.shape).astype(np.float32)
    hot = np.zeros(field.shape, dtype=bool)
    cold = np.zeros(field.shape, dtype=bool)
    hot[10:31, 80:87] = True
    cold[12:29, 140:147] = True
    field[hot] += 6.0
    field[cold] -= 5.0
    valid = np.ones(field.shape, dtype=bool)
    valid[:, 180:] = False
    field[:, 180:] = -300.0
    return {
        "name": "local_hot_cold",
        "clean": field,
        "valid": valid,
        "x_m": x_m,
        "time_s": time_s,
        "hot": hot,
        "cold": cold,
    }


def half_cable_shift(seed: int = 1) -> dict:
    """Left half is warmer for the whole record. Not a local hotspot."""
    rng = np.random.default_rng(seed)
    x_m = np.arange(160, dtype=np.float64)
    time_s = np.arange(24, dtype=np.float64) * 60.0
    field = np.full((24, 160), 20.0, dtype=np.float32)
    field += rng.normal(0.0, 0.1, size=field.shape).astype(np.float32)
    field[:, :80] += 4.0
    valid = np.ones(field.shape, dtype=bool)
    broad = np.zeros(field.shape, dtype=bool)
    broad[:, :80] = True
    return {
        "name": "half_cable_warm",
        "clean": field,
        "valid": valid,
        "x_m": x_m,
        "time_s": time_s,
        "broad": broad,
    }


def _base_field(seed: int, noise: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x_m = np.arange(200, dtype=np.float64)
    time_s = np.arange(40, dtype=np.float64) * 60.0
    field = np.full((40, 200), 21.0, dtype=np.float32)
    if noise > 0:
        field += rng.normal(0.0, noise, size=field.shape).astype(np.float32)
    return field, x_m, time_s, rng


def _ramp_weight(col: int, left: int, right: int, ramp: int) -> float:
    if col < left:
        return (col - (left - ramp)) / ramp
    if col > right:
        return (right + ramp - col) / ramp
    return 1.0


def _plant_block(delta: np.ndarray, rows: slice, center: int, width: int, ramp: int, amp: float) -> None:
    if width < 1 or ramp < 0:
        raise ValueError("width must be positive and ramp must be non-negative")
    left = center - width // 2
    right = left + width - 1
    start = left - ramp
    stop = right + ramp
    if start < 0 or stop >= delta.shape[1]:
        raise ValueError("planted block does not fit on the cable")
    for col in range(start, stop + 1):
        weight = _ramp_weight(col, left, right, ramp) if ramp else 1.0
        if weight == 0.0:
            continue
        delta[rows, col] = np.float32(amp * weight)


def background_scene(seed: int = 0) -> dict:
    """Same quiet cable as local_scene, with no planted hot or cold event."""
    field, x_m, time_s, _rng = _base_field(seed, 0.12)
    valid = np.ones(field.shape, dtype=bool)
    valid[:, 180:] = False
    field[:, 180:] = -300.0
    delta = np.zeros(field.shape, dtype=np.float32)
    return {
        "name": "background",
        "clean": field,
        "valid": valid,
        "x_m": x_m,
        "time_s": time_s,
        "hot": np.zeros(field.shape, dtype=bool),
        "cold": np.zeros(field.shape, dtype=bool),
        "truth_delta_c": delta,
        "amp_hot": 0.0,
        "amp_cold": 0.0,
        "width": 0,
        "ramp": 0,
        "noise": 0.12,
    }


def local_scene_var(
    seed: int,
    *,
    amp_hot: float = 6.0,
    amp_cold: float = -5.0,
    width: int = 7,
    noise: float = 0.12,
    ramp: int = 0,
) -> dict:
    """local_scene geometry with a known planted delta.

    `width` is the flat top in samples. `ramp` is the linear front on each
    side, in samples. `truth_delta_c` is that plant, before noise and before
    the dead tail is written.
    """
    field, x_m, time_s, _rng = _base_field(seed, noise)
    delta = np.zeros(field.shape, dtype=np.float32)
    if amp_hot != 0.0:
        _plant_block(delta, slice(10, 31), 83, int(width), int(ramp), float(amp_hot))
    if amp_cold != 0.0:
        _plant_block(delta, slice(12, 29), 143, int(width), int(ramp), float(amp_cold))
    field += delta
    valid = np.ones(field.shape, dtype=bool)
    valid[:, 180:] = False
    field[:, 180:] = -300.0
    delta[:, 180:] = 0.0
    return {
        "name": "local_var",
        "clean": field,
        "valid": valid,
        "x_m": x_m,
        "time_s": time_s,
        "hot": delta > 0,
        "cold": delta < 0,
        "truth_delta_c": delta,
        "amp_hot": float(amp_hot),
        "amp_cold": float(amp_cold),
        "width": int(width),
        "ramp": int(ramp),
        "noise": float(noise),
    }
