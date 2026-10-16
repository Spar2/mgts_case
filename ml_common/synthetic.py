"""Короткое синтетическое поле с известными рамками. Реальные MGTS CSV сюда не входят."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from fiber.io import load_csv


def planted_field(
    *,
    n_time: int = 48,
    n_space: int = 320,
    dx_m: float = 0.25,
    dt_s: float = 120.0,
    seed: int = 0,
) -> tuple[dict, list[dict]]:
    """Фон 22 °C и два пятна: локальный нагрев и локальное охлаждение.

    Размер подобран под пороги fiber (ширина, число профилей, число ячеек)
    и под морфологию larpathon (не короче нескольких шагов, уже 15 м).
    """
    rng = np.random.default_rng(seed)
    raw = np.full((n_time, n_space), 22.0, dtype=np.float32)
    raw += rng.normal(0.0, 0.05, size=raw.shape).astype(np.float32)
    x_m = (np.arange(n_space, dtype=np.float64) * dx_m)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    stamps = [start + timedelta(seconds=dt_s * i) for i in range(n_time)]
    time_s = np.array([item.timestamp() for item in stamps], dtype=np.float64)
    time_iso = [item.isoformat() for item in stamps]

    def paint(x0: float, x1: float, t0: int, t1: int, temp: float) -> dict:
        cols = np.flatnonzero((x_m >= x0) & (x_m <= x1))
        raw[t0:t1 + 1, cols] = np.float32(temp)
        return {
            "name": "hot" if temp > 22 else "cool",
            "x_from_m": float(x_m[cols[0]]),
            "x_to_m": float(x_m[cols[-1]]),
            "t_from_s": float(time_s[t0]),
            "t_to_s": float(time_s[t1]),
            "t_first": time_iso[t0],
            "t_last": time_iso[t1],
        }

    truth = [
        paint(20.0, 26.0, 12, 30, 40.0),
        paint(50.0, 57.0, 8, 28, 12.0),
    ]
    field = {
        "raw": raw,
        "x_m": x_m,
        "time_s": time_s,
        "valid": np.isfinite(raw),
        "time_iso": time_iso,
        "dts_id": "synthetic",
        "channel_id": "1",
    }
    return field, truth


def field_from_csv(path) -> dict:
    """MGTS CSV (timestamp + points), тот же разбор, что у fiber.io.load_csv."""
    record = load_csv(path)
    return {
        "raw": record.raw_c,
        "x_m": record.x_m,
        "time_s": record.time_s,
        "valid": record.valid,
        "time_iso": record.time_iso,
        "dts_id": record.dts_id or "unknown",
        "channel_id": str(record.channel_id),
    }
