"""PDRS на карте вероятности ансамбля: контракт, детерминизм, синтетические зоны.

Выход extract_zones_pdrs сравнивается с extract_zones_cc (порог + связные компоненты).
Синтетика строится здесь же, без кэшей и без реальных файлов.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dts.pdrs import extract_zones_cc, extract_zones_pdrs

CC_KEYS = ("zone", "polarity", "n_px", "t0", "t1", "x0", "x1")


def _gauss(yy, xx, t0, x0, st, sx):
    return np.exp(-((yy - t0) ** 2 / (2 * st ** 2) + (xx - x0) ** 2 / (2 * sx ** 2)))


def _pix_iou(pred, gt):
    pred = np.asarray(pred, bool)
    gt = np.asarray(gt, bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter) / float(union) if union else 0.0


def _assert_contract(lab, zones, shape, min_cells):
    """Общий контракт с extract_zones_cc: карта меток и словарь зоны."""
    assert isinstance(lab, np.ndarray)
    assert lab.shape == shape
    assert np.issubdtype(lab.dtype, np.integer)
    assert int(lab.min()) == 0
    ids = [z["zone"] for z in zones]
    assert ids == list(range(1, len(zones) + 1))
    assert {int(v) for v in np.unique(lab)} == ({0} | set(ids))
    for z in zones:
        for key in CC_KEYS:
            assert key in z
        assert type(z["zone"]) is int
        assert z["polarity"] in ("hot", "cold")
        assert type(z["n_px"]) is int and z["n_px"] >= min_cells
        assert type(z["t0"]) is int and type(z["t1"]) is int
        assert type(z["x0"]) is int and type(z["x1"]) is int
        assert 0 <= z["t0"] < z["t1"] <= shape[0]
        assert 0 <= z["x0"] < z["x1"] <= shape[1]
        owned = lab == z["zone"]
        assert int(owned.sum()) == z["n_px"]
        assert int(owned[z["t0"]:z["t1"], z["x0"]:z["x1"]].sum()) == z["n_px"]


def _two_peaks(dist, r_saddle, region_median=0.4):
    """Две гауссовы вершины одной полярности. Больше dist — ниже седло между ними."""
    height, width = 60, 100
    yy, xx = np.mgrid[0:height, 0:width]
    sx = 5.0
    c1, c2 = 35, 35 + dist
    field = (_gauss(yy, xx, 30, c1, 6, sx) + _gauss(yy, xx, 30, c2, 6, sx)).astype(np.float32)
    field /= field.max()
    sign = np.ones_like(field)
    valid = np.ones(field.shape, bool)
    lab, zones = extract_zones_pdrs(
        field, valid, sign,
        seed=0.7, floor=0.25, r_saddle=r_saddle, region_median=region_median,
        min_cells=10, peak_size=(3, 5),
    )
    return lab, zones, c1, c2


def test_output_format_matches_connected_components():
    height, width = 80, 70
    yy, xx = np.mgrid[0:height, 0:width]
    hot = _gauss(yy, xx, 25, 18, 6, 4)
    cold = _gauss(yy, xx, 55, 50, 6, 4)
    score = np.maximum(hot, cold).astype(np.float32)
    sign = np.zeros_like(score)
    sign[hot >= 0.35] = 1.0
    sign[cold >= 0.35] = -1.0
    valid = np.ones(score.shape, bool)
    time_s = np.arange(height, dtype=np.float64) * 120.0
    dist_m = np.arange(width, dtype=np.float64) * 0.25

    lab_cc, zones_cc = extract_zones_cc(score, valid, sign, thr=0.7, min_cells=12)
    lab, zones = extract_zones_pdrs(
        score, valid, sign, time_s, dist_m,
        seed=0.7, floor=0.35, r_saddle=0.6, region_median=0.45, min_cells=12,
    )
    _assert_contract(lab_cc, zones_cc, score.shape, min_cells=12)
    _assert_contract(lab, zones, score.shape, min_cells=12)
    assert zones_cc and zones
    assert set(zones_cc[0]) <= set(zones[0])
    for z in zones:
        assert z["t_start_s"] == float(time_s[z["t0"]])
        assert z["t_end_s"] == float(time_s[z["t1"] - 1])
        assert z["x_start_m"] == float(dist_m[z["x0"]])
        assert z["x_end_m"] == float(dist_m[z["x1"] - 1])
        assert isinstance(z["score"], float)

    # без t/x физические края не обещаются; полярность при h is None — hot, как у CC
    lab_h, zones_h = extract_zones_pdrs(score, valid, None, seed=0.7, floor=0.35, region_median=0.45)
    assert zones_h
    assert all(z["polarity"] == "hot" for z in zones_h)
    assert "t_start_s" not in zones_h[0] and "x_start_m" not in zones_h[0]
    _assert_contract(lab_h, zones_h, score.shape, min_cells=12)


def test_deterministic_and_does_not_mutate_inputs():
    height, width = 80, 70
    yy, xx = np.mgrid[0:height, 0:width]
    hot = _gauss(yy, xx, 25, 18, 6, 4)
    cold = _gauss(yy, xx, 55, 50, 6, 4)
    score = np.maximum(hot, cold).astype(np.float32)
    sign = np.where(hot >= cold, 1.0, -1.0).astype(np.float32)
    valid = np.ones(score.shape, bool)
    time_s = np.arange(height, dtype=np.float64) * 120.0
    dist_m = np.arange(width, dtype=np.float64) * 0.25
    frozen = (score.copy(), sign.copy(), valid.copy(), time_s.copy(), dist_m.copy())
    kwargs = dict(seed=0.7, floor=0.35, r_saddle=0.6, region_median=0.45, min_cells=12)
    lab_a, zones_a = extract_zones_pdrs(score, valid, sign, time_s, dist_m, **kwargs)
    lab_b, zones_b = extract_zones_pdrs(score, valid, sign, time_s, dist_m, **kwargs)
    assert np.array_equal(lab_a, lab_b)
    assert zones_a == zones_b
    assert np.array_equal(score, frozen[0]) and np.array_equal(sign, frozen[1])
    assert np.array_equal(valid, frozen[2]) and np.array_equal(time_s, frozen[3])
    assert np.array_equal(dist_m, frozen[4])
    assert zones_a  # путь слияния и полярности реально выполнен


def test_empty_map_has_no_zones():
    score = np.zeros((24, 18), np.float32)
    valid = np.ones(score.shape, bool)
    valid[0, :] = False
    lab, zones = extract_zones_pdrs(score, valid, np.ones_like(score))
    assert zones == []
    assert lab.shape == score.shape and int(lab.max()) == 0
    lab2, zones2 = extract_zones_cc(score, valid, thr=0.5)
    assert zones2 == [] and lab2.shape == score.shape


def test_hot_and_cold_gaussians_found():
    height, width = 80, 70
    yy, xx = np.mgrid[0:height, 0:width]
    floor = 0.35
    hot = _gauss(yy, xx, 25, 18, 6, 4)
    cold = _gauss(yy, xx, 55, 50, 6, 4)
    score = np.maximum(hot, cold).astype(np.float32)
    sign = np.zeros_like(score)
    sign[hot >= cold] = 1.0
    sign[hot < cold] = -1.0
    valid = np.ones(score.shape, bool)
    time_s = np.arange(height, dtype=np.float64) * 120.0
    dist_m = np.arange(width, dtype=np.float64) * 0.25
    lab, zones = extract_zones_pdrs(
        score, valid, sign, time_s, dist_m,
        seed=0.7, floor=floor, r_saddle=0.6, region_median=0.45, min_cells=12,
    )
    _assert_contract(lab, zones, score.shape, min_cells=12)
    gt_hot = hot >= floor
    gt_cold = cold >= floor
    pred_hot = np.zeros(score.shape, bool)
    pred_cold = np.zeros(score.shape, bool)
    for z in zones:
        mask = lab == z["zone"]
        if z["polarity"] == "hot":
            pred_hot |= mask
        else:
            pred_cold |= mask
    assert sum(z["polarity"] == "hot" for z in zones) == 1
    assert sum(z["polarity"] == "cold" for z in zones) == 1
    assert _pix_iou(pred_hot, gt_hot) >= 0.5
    assert _pix_iou(pred_cold, gt_cold) >= 0.5
    # чужая полярность не закрашивает ядро соседней зоны
    assert _pix_iou(pred_hot, gt_cold) == 0.0
    assert _pix_iou(pred_cold, gt_hot) == 0.0


def test_high_saddle_merges_low_saddle_does_not():
    # Одинаковые пороги. Седло измеряно по границе watershed: d=14 → ~0.74, d=18 → ~0.40.
    # При r_saddle=0.6 высокое седло сливает две вершины, низкое оставляет две зоны.
    # Пол 0.25 ниже обоих седел, так что зоны соседние и решение принимает именно седло.
    lab_hi, zones_hi, c1, c2 = _two_peaks(dist=14, r_saddle=0.6)
    assert len(zones_hi) == 1
    assert bool(lab_hi[30, c1] == zones_hi[0]["zone"])
    assert bool(lab_hi[30, c2] == zones_hi[0]["zone"])

    lab_lo, zones_lo, c1, c2 = _two_peaks(dist=18, r_saddle=0.6)
    assert len(zones_lo) == 2
    covered = {(bool(lab_lo[30, c1] == z["zone"]), bool(lab_lo[30, c2] == z["zone"])) for z in zones_lo}
    assert covered == {(True, False), (False, True)}

    # Контроль критерия: то же низкое седло сливается, если порог седла опустить ниже него,
    # а высокое распадается, если порог поднять выше него.
    lab_lo_loose, zones_lo_loose, c1, c2 = _two_peaks(dist=18, r_saddle=0.3)
    assert len(zones_lo_loose) == 1
    assert bool(lab_lo_loose[30, c1] == zones_lo_loose[0]["zone"])
    assert bool(lab_lo_loose[30, c2] == zones_lo_loose[0]["zone"])
    _, zones_hi_strict, _, _ = _two_peaks(dist=14, r_saddle=0.9)
    assert len(zones_hi_strict) == 2


def test_hybrid_refines_core_and_keeps_weak_cc_zone():
    """CC-зона ниже ядра PDRS остаётся при fallback; сильное ядро сегментируется."""
    from dts.pdrs import extract_zones_hybrid

    height, width = 48, 70
    score = np.zeros((height, width), np.float32)
    yy, xx = np.mgrid[0:height, 0:width]
    # ядро s>=0.95 шире min_cells; плато отдельно, между порогом CC и ядром
    score += _gauss(yy, xx, 16, 18, 8, 6).astype(np.float32)
    score[30:42, 50:62] = 0.88
    sign = np.ones_like(score)
    valid = np.ones(score.shape, bool)
    thr, core = 0.86, 0.95
    kw = dict(thr=thr, seed=core, floor=core, r_saddle=0.3, region_median=0.5, min_cells=12)
    lab_fb, zones_fb = extract_zones_hybrid(score, valid, sign, mode="refine", fallback=True, **kw)
    lab_drop, zones_drop = extract_zones_hybrid(score, valid, sign, mode="refine", fallback=False, **kw)
    _assert_contract(lab_fb, zones_fb, score.shape, min_cells=12)
    _assert_contract(lab_drop, zones_drop, score.shape, min_cells=12)
    assert len(zones_fb) == 2
    assert len(zones_drop) == 1
    assert lab_fb[34, 54] > 0  # слабое плато сохранено
    assert lab_drop[34, 54] == 0
    assert lab_fb[16, 18] > 0 and lab_drop[16, 18] > 0
    # seed-режим совпадает с PDRS при том же seed
    from dts.pdrs import extract_zones_pdrs
    lab_h, zones_h = extract_zones_hybrid(
        score, valid, sign, mode="seed", seed=thr, floor=0.7, r_saddle=0.3, region_median=0.5, min_cells=12,
    )
    lab_p, zones_p = extract_zones_pdrs(
        score, valid, sign, seed=thr, floor=0.7, r_saddle=0.3, region_median=0.5, min_cells=12,
    )
    assert np.array_equal(lab_h, lab_p) and zones_h == zones_p
    lab_h2, zones_h2 = extract_zones_hybrid(
        score, valid, sign, mode="refine", fallback=True, **kw,
    )
    assert np.array_equal(lab_fb, lab_h2) and zones_fb == zones_h2


def test_time_gap_is_not_bridged():
    height, width = 40, 30
    score = np.zeros((height, width), np.float32)
    score[8:24, 10:20] = 0.95
    score[12, 15] = 1.0
    score[20, 15] = 1.0
    sign = np.ones_like(score)
    valid = np.ones(score.shape, bool)
    time_s = np.arange(height, dtype=np.float64) * 120.0
    time_s[16:] += 10_000.0  # разрыв между строками 15 и 16
    kwargs = dict(seed=0.8, floor=0.5, r_saddle=0.3, region_median=0.5, min_cells=8, peak_size=(3, 3))
    lab, zones = extract_zones_pdrs(score, valid, sign, time_s, **kwargs)
    before = {int(v) for v in np.unique(lab[:16])} - {0}
    after = {int(v) for v in np.unique(lab[16:])} - {0}
    assert before and after
    assert before.isdisjoint(after)
    assert len(zones) >= 2
    # без разрыва тот же прямоугольник — одна зона
    time_flat = np.arange(height, dtype=np.float64) * 120.0
    _, zones_flat = extract_zones_pdrs(score, valid, sign, time_flat, **kwargs)
    assert len(zones_flat) == 1
