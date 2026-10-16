"""Поле DTS и протокол Analyzer. Реализации лежат в ml_common.adapters."""
from __future__ import annotations

import os
from typing import Mapping, Protocol, runtime_checkable

import numpy as np

THREAD_CAP = 4


def thread_cap() -> int:
    """Не больше 4 ядер: на боксе мало RAM, соседние воркстримы тоже считают."""
    return max(1, min(THREAD_CAP, os.cpu_count() or 1))


def limit_threads() -> int:
    """Выставить потоки BLAS/OpenMP до тяжёлых импортов, насколько это ещё можно."""
    n = str(thread_cap())
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, n)
    return int(n)


def normalize_field(field: Mapping) -> dict:
    """Привести вход адаптера к одному виду: raw[time, x], оси, маска, подписи."""
    raw = np.asarray(field["raw"], dtype=np.float32)
    x_m = np.asarray(field["x_m"], dtype=np.float64).reshape(-1)
    time_s = np.asarray(field["time_s"], dtype=np.float64).reshape(-1)
    if raw.ndim != 2 or raw.shape != (time_s.size, x_m.size) or x_m.size < 2 or time_s.size < 2:
        raise ValueError("raw must have shape [len(time_s), len(x_m)] with at least two samples on each axis")
    if not np.all(np.diff(x_m) > 0) or not np.all(np.diff(time_s) > 0):
        raise ValueError("x_m and time_s must be strictly increasing")
    valid = field.get("valid")
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != raw.shape:
            raise ValueError("valid shape must agree with raw")
    iso = field.get("time_iso")
    if iso is not None:
        iso = [str(item) for item in iso]
        if len(iso) != time_s.size:
            raise ValueError("time_iso must agree with time_s")
    return {
        "raw": raw,
        "x_m": x_m,
        "time_s": time_s,
        "valid": valid,
        "time_iso": iso,
        "dts_id": str(field.get("dts_id") or "unknown"),
        "channel_id": str(field.get("channel_id") or "1"),
    }


def grid_steps(x_m: np.ndarray, time_s: np.ndarray) -> tuple[float, float]:
    dx = float(np.median(np.diff(x_m)))
    dt = float(np.median(np.diff(time_s)))
    if not np.isfinite(dx) or dx <= 0 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("grid steps must be positive")
    return dx, dt


@runtime_checkable
class Analyzer(Protocol):
    """predict(field) -> документ dts-events/1."""

    method: str

    def predict(self, field: Mapping) -> dict:
        """Вернуть events.json v1 по одному полю температуры."""
