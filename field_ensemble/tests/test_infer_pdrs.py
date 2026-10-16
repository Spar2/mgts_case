"""Флаг зон инференса: по умолчанию тот же вывод, что events.zones_from_prob.

Гибрид PDRS проверяется на синтетической карте, без Analyzer и без реальных CSV.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dts import events, infer

ROOT = Path(__file__).resolve().parents[1]
NEED = ("id", "t0", "t1", "x0", "x1", "start_m", "end_m", "t_start_s", "t_end_s", "n_px", "polarity", "score")


def _field():
    """Два горячих очага через столбец ниже порога и один холодный. Замыкание их склеивает, PDRS — нет."""
    t_n, w_n = 36, 48
    s = np.full((t_n, w_n), 0.15, np.float32)
    h = np.zeros((t_n, w_n), np.float32)
    valid = np.ones((t_n, w_n), bool)
    s[6:18, 4:16] = 0.96
    h[6:18, 4:16] = 0.8
    s[6:18, 18:30] = 0.96
    h[6:18, 18:30] = 0.8
    s[22:34, 8:24] = 0.96
    h[22:34, 8:24] = -0.7
    x = np.arange(w_n, dtype=np.float64) * 0.5
    t = np.arange(t_n, dtype=np.float64) * 60.0
    return s, h, valid, x, t


def test_default_matches_zones_from_prob():
    s, h, valid, x, t = _field()
    thr = 0.5
    cc = events.zones_from_prob(s, h, valid, x, t, thr)
    assert infer.pipeline_zones(s, h, valid, x, t, thr) == cc
    assert infer.pipeline_zones(s, h, valid, x, t, thr, zones="cc") == cc
    assert len(cc) == 2
    assert {z["polarity"] for z in cc} == {"hot", "cold"}


def test_hybrid_keeps_pipeline_schema_and_both_polarities():
    s, h, valid, x, t = _field()
    thr = 0.8608087301264272
    zones = infer.pipeline_zones(s, h, valid, x, t, thr, zones="pdrs_hybrid")
    assert len(zones) >= 2
    assert [z["id"] for z in zones] == list(range(len(zones)))
    pols = set()
    for z in zones:
        assert set(NEED) <= set(z)
        assert z["start_m"] == float(x[z["x0"]])
        assert z["end_m"] == float(x[z["x1"] - 1])
        assert z["t_start_s"] == float(t[z["t0"]])
        assert z["t_end_s"] == float(t[z["t1"] - 1])
        assert z["n_px"] >= 12
        pols.add(z["polarity"])
    assert pols == {"hot", "cold"}
    cc = events.zones_from_prob(s, h, valid, x, t, thr)
    assert len(zones) > len(cc)


def test_unknown_zones_rejected():
    s, h, valid, x, t = _field()
    try:
        infer.pipeline_zones(s, h, valid, x, t, 0.5, zones="pdrs")
    except ValueError as exc:
        assert "cc" in str(exc) and "pdrs_hybrid" in str(exc)
    else:
        raise AssertionError("ожидался ValueError")


def test_hybrid_spec_matches_frozen_json():
    spec = infer.hybrid_spec()
    raw = json.loads((ROOT / "artifacts/eval/pdrs_hybrid_best_params.json").read_text())
    if isinstance(raw.get("best_params"), dict):
        raw = raw["best_params"]
    assert spec["mode"] == raw["mode"] == "seed"
    assert spec["floor"] == raw["floor"] == 0.7
    assert spec["r_saddle"] == raw["r_saddle"] == 0.3
    assert spec["region_median"] == raw["region_median"] == 0.5
    assert spec["fallback"] is False
