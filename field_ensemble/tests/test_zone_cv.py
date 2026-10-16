"""Аннотация зон: контур, вытянутость, дрейф Хафа, период, ленивый cv2."""
import ast
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from dts import export, zone_cv


def _grid(n_t, n_x, dx=0.25, dt=120.0):
    x = np.arange(n_x, dtype=np.float64) * dx
    t = 1_700_000_000 + np.arange(n_t, dtype=np.float64) * dt
    valid = np.ones((n_t, n_x), bool)
    return x, t, valid


def _zone(mask):
    rows = np.flatnonzero(mask.any(1))
    cols = np.flatnonzero(mask.any(0))
    return dict(
        id=0, t0=int(rows[0]), t1=int(rows[-1]) + 1, x0=int(cols[0]), x1=int(cols[-1]) + 1,
        n_px=int(mask.sum()), start_m=0.0, end_m=1.0, t_start_s=0.0, t_end_s=1.0,
    )


def test_cv2_is_not_imported_at_module_level():
    tree = ast.parse(Path(zone_cv.__file__).read_text())
    for node in tree.body:
        if isinstance(node, ast.Import):
            mods = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods = [node.module or ""]
        else:
            continue
        assert not any(mod == "cv2" or mod.startswith("cv2.") for mod in mods)


def test_solid_blob_is_one_component_and_rounder_than_a_bar():
    x, t, valid = _grid(40, 60)
    square = np.zeros(valid.shape, bool)
    square[10:26, 20:36] = True
    bar = np.zeros(valid.shape, bool)
    bar[18:21, 5:50] = True
    dT = np.zeros(valid.shape, np.float32)
    z_sq = _zone(square)
    z_bar = _zone(bar)
    ann_sq = zone_cv.annotate(square, dT, valid, x, t, [z_sq])[0]
    ann_bar = zone_cv.annotate(bar, dT, valid, x, t, [z_bar])[0]
    assert ann_sq["cv_n_components"] == 1
    assert ann_sq["cv_area_px"] == int(square.sum())
    assert ann_sq["cv_solidity"] > 0.9
    assert ann_sq["contour"] is not None and len(ann_sq["contour"]) >= 4
    assert ann_bar["cv_eccentricity"] > ann_sq["cv_eccentricity"]
    assert ann_bar["cv_bbox_width_m"] > ann_sq["cv_bbox_width_m"]
    assert z_sq["cv_area_px"] == ann_sq["cv_area_px"]
    assert ann_sq["cv_period_h"] is None


def test_bar_orientations_differ_by_about_ninety_degrees():
    x, t, valid = _grid(50, 50)
    horizontal = np.zeros(valid.shape, bool)
    vertical = np.zeros(valid.shape, bool)
    horizontal[22:26, 5:45] = True
    vertical[5:45, 22:26] = True
    dT = np.zeros(valid.shape, np.float32)
    h = zone_cv.annotate(horizontal, dT, valid, x, t, [_zone(horizontal)])[0]
    v = zone_cv.annotate(vertical, dT, valid, x, t, [_zone(vertical)])[0]
    gap = abs(h["cv_orientation_deg"] - v["cv_orientation_deg"]) % 180.0
    gap = min(gap, 180.0 - gap)
    assert 60.0 <= gap <= 120.0


def test_diagonal_front_speed_is_positive():
    n_t, n_x = 80, 100
    x, t, valid = _grid(n_t, n_x)
    mask = np.zeros(valid.shape, bool)
    for row in range(12, 68):
        col = 8 + (row - 12)
        mask[row, col:col + 3] = True
    ann = zone_cv.annotate(mask, np.zeros(valid.shape, np.float32), valid, x, t, [_zone(mask)])[0]
    # 1 столбец (0.25 м) на 1 строку (120 с) = 7.5 м/ч
    assert ann["front_line"] is not None
    assert ann["cv_drift_m_per_h"] == pytest.approx(7.5, rel=0.35, abs=2.0)
    assert ann["cv_drift_m_per_h"] > 0


def test_two_blobs_and_a_sine_period():
    n_t, n_x = 48, 20
    x, t, valid = _grid(n_t, n_x)
    mask = np.zeros(valid.shape, bool)
    mask[4:10, 2:6] = True
    mask[20:28, 12:18] = True
    period = 12
    dT = np.sin(2 * np.pi * np.arange(n_t) / period).astype(np.float32)[:, None] * np.ones((1, n_x), np.float32)
    zone = dict(id=3, t0=0, t1=n_t, x0=0, x1=n_x, n_px=int(mask.sum()),
                start_m=0.0, end_m=float(x[-1]), t_start_s=float(t[0]), t_end_s=float(t[-1]))
    ann = zone_cv.annotate(mask, dT, valid, x, t, [zone])[0]
    assert ann["cv_n_components"] == 2
    assert ann["cv_period_h"] == pytest.approx(period * 120.0 / 3600.0, rel=0.15)
    assert zone["cv_period_h"] == ann["cv_period_h"]


def test_l_shape_is_less_solid_and_export_keeps_old_columns_first():
    x, t, valid = _grid(30, 30)
    mask = np.zeros(valid.shape, bool)
    mask[5:20, 5:9] = True
    mask[16:20, 5:22] = True
    solid = np.zeros(valid.shape, bool)
    solid[5:20, 5:20] = True
    dT = np.zeros(valid.shape, np.float32)
    ann_l = zone_cv.annotate(mask, dT, valid, x, t, [_zone(mask)])[0]
    ann_s = zone_cv.annotate(solid, dT, valid, x, t, [_zone(solid)])[0]
    assert ann_l["cv_solidity"] < ann_s["cv_solidity"]
    assert ann_l["cv_solidity"] < 0.85
    bare = dict(id=1, t_start_s=1_700_000_000, t_end_s=1_700_000_120, start_m=0.0, end_m=1.0,
                polarity="hot", n_px=4, score=1.0)
    row = next(export.zone_rows([bare], "1.csv", "abc"))
    assert list(row.keys()) == export.FIELDS
    assert row["cv_area_px"] == ""
    assert row["cv_drift_m_per_h"] == ""
    assert export.FIELDS[:15] == [
        "source", "source_sha256", "zone_id", "t_start_utc", "t_end_utc", "x_start_m", "x_end_m",
        "polarity", "class", "class_prob", "is_alarm", "score", "peak_dT_C", "n_px", "model_version",
    ]


def test_overlay_draws_on_the_delta_map():
    x, t, valid = _grid(24, 30)
    mask = np.zeros(valid.shape, bool)
    mask[6:16, 8:18] = True
    dT = mask.astype(np.float32) * 2
    ann = zone_cv.annotate(mask, dT, valid, x, t, [_zone(mask)])
    fig, ax = plt.subplots()
    zone_cv.show_dT(ax, dT, valid, x, t)
    before = len(ax.lines)
    zone_cv.overlay(fig, ann, x, t, ax=ax)
    assert len(ax.lines) > before
    plt.close(fig)


def test_empty_and_outside_zone_do_not_crash():
    x, t, valid = _grid(12, 8)
    score = np.zeros(valid.shape, np.float32)
    assert zone_cv.annotate(score, score, valid, x, t, []) == []
    zone = dict(id=0, t0=100, t1=120, x0=100, x1=110, n_px=5, start_m=0, end_m=1, t_start_s=0, t_end_s=1)
    ann = zone_cv.annotate(score, score, valid, x, t, [zone])[0]
    assert ann["cv_area_px"] == 0 and ann["contour"] is None
