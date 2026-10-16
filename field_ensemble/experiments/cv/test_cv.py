"""Классический CV: ленивый cv2, контраст горячего пятна, отсев спекла, постобработка U-Net."""
import ast
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dts import cv_common
from experiments.cv import cv_pipe, cv_ref


def _field(seed=0, amp=4.0, spike=False):
    rng = np.random.default_rng(seed)
    temp = 20 + rng.normal(0, 0.2, (96, 180)).astype(np.float32)
    temp += np.linspace(0, 1.2, 96)[:, None]
    temp += np.linspace(-1.0, 1.0, 180)[None, :]
    valid = np.ones(temp.shape, bool)
    valid[:, :4] = False
    if amp:
        temp[40:64, 80:104] += np.float32(amp)
    if spike:
        temp[12, 30] += np.float32(15)
    x = np.arange(180, dtype=np.float64) * 0.25
    t = 1_700_000_000 + np.arange(96, dtype=np.float64) * 120.0
    sigma = np.full(180, 0.25, np.float32)
    return temp, valid, x, t, sigma


def test_cv2_is_not_imported_at_module_level():
    root = Path(__file__).resolve().parents[2]
    files = (
        root / "dts" / "cv_common.py",
        root / "experiments" / "cv" / "cv_ref.py",
        root / "experiments" / "cv" / "cv_pipe.py",
    )
    for path in files:
        name = path.name
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node, ast.Import):
                mods = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods = [node.module or ""]
            else:
                continue
            assert not any(mod == "cv2" or mod.startswith("cv2.") for mod in mods), name


def test_row_median_matches_numpy_within_a_bin():
    rng = np.random.default_rng(1)
    row = rng.normal(22, 0.8, (6, 61)).astype(np.float32)
    got = cv_common.median_axis1(row, 9)
    for i in range(6):
        for j in (8, 20, 30, 45):
            ref = float(np.median(row[i, j - 4:j + 5]))
            assert abs(float(got[i, j]) - ref) < 0.2


def test_close_kernel_is_spatial_not_temporal():
    gap_x = np.zeros((5, 7), bool)
    gap_x[2, 1] = True
    gap_x[2, 3] = True
    closed = cv_common.morph(gap_x, "close", (3, 1))
    assert closed[2, 2]
    gap_t = np.zeros((5, 7), bool)
    gap_t[1, 3] = True
    gap_t[3, 3] = True
    closed_t = cv_common.morph(gap_t, "close", (3, 1))
    assert not closed_t[2, 3]


def test_reference_hot_blob_above_background_and_invalid_zero():
    temp, valid, x, t, sigma = _field()
    score, hot = cv_ref.score_sequence(temp, valid, x, t, sigma)
    assert score.shape == temp.shape and score.dtype == np.float32
    assert hot.dtype == np.float32 and np.isfinite(score).all()
    assert score[~valid].max() == 0
    assert hot[~valid].max() == 0
    assert float(score[50, 90]) > float(np.median(score[valid]))
    assert float(hot[50, 90]) > 0


def test_reference_cold_blob_is_negative_hotness():
    temp, valid, x, t, sigma = _field(amp=-4.0)
    _score, hot = cv_ref.score_sequence(temp, valid, x, t, sigma)
    assert float(hot[50, 90]) < 0


def test_pipe_ranks_blob_over_spike_and_zeros_invalid():
    temp, valid, x, t, sigma = _field(spike=True)
    score, hot = cv_pipe.score_sequence(temp, valid, x, t, sigma)
    assert score.shape == temp.shape and np.isfinite(score).all()
    assert score[~valid].max() == 0
    assert float(score[52, 92]) > float(score[12, 30])
    assert float(hot[52, 92]) > 0


def test_unet_post_cuts_speckle_and_keeps_supported_blob():
    temp, valid, x, t, sigma = _field()
    score = np.full(temp.shape, 0.1, np.float32)
    hot = np.zeros(temp.shape, np.float32)
    score[12, 30] = 0.99
    hot[12, 30] = 1
    score[40:64, 80:104] = 0.97
    hot[40:64, 80:104] = 1
    out, hot_out = cv_pipe.postprocess_unet(score, hot, temp, valid, x, t, sigma)
    assert float(out[12, 30]) <= cv_pipe.UNET_FLOOR + 1e-6
    assert float(out[52, 92]) > 0.9
    assert hot_out.shape == score.shape
    assert out[~valid].max() == 0


def test_flat_and_empty_valid_do_not_crash():
    temp = np.full((36, 48), 21.0, np.float32)
    x = np.arange(48) * 0.25
    t = np.arange(36) * 120.0
    sigma = np.full(48, 0.3, np.float32)
    valid = np.ones(temp.shape, bool)
    for fn in (cv_ref.score_sequence, cv_pipe.score_sequence):
        score, hot = fn(temp, valid, x, t, sigma)
        assert np.isfinite(score).all() and np.isfinite(hot).all()
    none = np.zeros(temp.shape, bool)
    score, hot = cv_pipe.score_sequence(temp, none, x, t, sigma)
    assert score.max() == 0 and hot.max() == 0


def test_drift_line_is_marked_by_hough():
    score = np.zeros((80, 120), np.float32)
    valid = np.ones(score.shape, bool)
    for i in range(8, 72):
        col = 15 + i // 2
        score[i, col:col + 2] = 5
    mask = cv_pipe._drift_mask(score, valid, 0.25, 120.0)
    assert int(mask.sum()) > 20
