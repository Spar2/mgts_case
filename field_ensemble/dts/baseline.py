"""Обёртка эталона haklolarp (median_x3 + spatial_z_halo, |Z|≥3.5 + extract_zones + rules).
Используется ТОЛЬКО как база для сравнения. Путь к пакету fiber: env HAKLOLARP_SRC."""
from __future__ import annotations
import os, sys
import numpy as np

SRC = os.environ.get("HAKLOLARP_SRC", "/workspace/repo/src")


def available() -> bool:
    try:
        _import(); return True
    except Exception:
        return False


def _import():
    if SRC not in sys.path:
        sys.path.insert(0, SRC)
    from fiber.pipeline import analyze_field  # noqa
    from fiber.classify.rules import classify_zones  # noqa
    return analyze_field, classify_zones


def run(raw, x, t, valid=None):
    analyze_field, _ = _import()
    o = analyze_field(raw, x, t, valid=valid)
    z = np.nan_to_num(np.asarray(o["z"], np.float32))
    zone = np.zeros(raw.shape, np.uint8)
    for zz in o["zones"]:
        ix = (x >= zz["start_m"]) & (x <= zz["end_m"])
        it = (t >= zz["t_start_s"]) & (t <= zz["t_end_s"])
        box = np.ix_(it, ix)
        sub = np.abs(z[box]) >= 3.5
        cur = zone[box]; cur[sub] = 1 if zz["polarity"] == "hot" else 2; zone[box] = cur
    return dict(z=z, zone=zone, zones=o["zones"], available=np.asarray(o["score_available"], bool))


def rules_labels(zones, cable_length_m):
    _, classify_zones = _import()
    return classify_zones(zones, cable_length_m)
