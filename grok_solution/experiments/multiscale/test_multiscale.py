"""Прореживание, слияние и гистограммный порог. Сети и реальные CSV не читаются."""
import numpy as np
from dts import features, metrics
from experiments.multiscale.multiscale import (
    add_hist, block_mean, fuse, hist_ge, hist_threshold, upsample_axis, NB,
)


def test_block_mean_and_upsample_cover_blocks():
    a = np.arange(5 * 4, dtype=np.float32).reshape(5, 4)
    valid = np.ones_like(a, bool)
    valid[0, 1] = False
    coarse, ok = block_mean(a, valid, 0, 2)
    assert coarse.shape == (3, 4)
    assert ok.shape == coarse.shape
    # полный блок 0: строки 0 и 1, столбец 0 — оба валидны
    assert coarse[0, 0] == np.float32((a[0, 0] + a[1, 0]) / 2)
    # столбец 1: строка 0 невалидна, остаётся строка 1
    assert coarse[0, 1] == a[1, 1]
    # хвост — одна строка
    assert coarse[2, 0] == a[4, 0]
    up = upsample_axis(coarse, 5, 0, 2)
    assert up.shape == a.shape
    assert np.all(up[0] == coarse[0]) and np.all(up[1] == coarse[0])
    assert np.all(up[4] == coarse[2])


def test_fuse_max_mean_and_noise_weights():
    a = (np.array([[0.2, 0.8]], np.float32), np.array([[1.0, -1.0]], np.float32))
    b = (np.array([[0.6, 0.4]], np.float32), np.array([[-2.0, 2.0]], np.float32))
    mx, mh = fuse([a, b], [1.0, 4.0], "max")
    assert np.allclose(mx, [[0.6, 0.8]])
    # hotness берётся у масштаба, который дал max score
    assert np.allclose(mh, [[-2.0, -1.0]])
    av, ah = fuse([a, b], [1.0, 4.0], "mean")
    assert np.allclose(av, [[0.4, 0.6]])
    nv, nh = fuse([a, b], [1.0, 4.0], "noise")
    assert np.allclose(nv, [[(0.2 + 4 * 0.6) / 5, (0.8 + 4 * 0.4) / 5]])
    assert np.allclose(nh, [[(1 + 4 * -2) / 5, (-1 + 4 * 2) / 5]])


def test_time_average_lowers_sigma_by_about_sqrt_n():
    rng = np.random.default_rng(0)
    T = rng.normal(0.0, 1.0, size=(800, 20)).astype(np.float32)
    valid = np.ones(T.shape, bool)
    s1 = features.robust_sigma_x(T, valid, smooth_px=1)
    T2, v2 = block_mean(T, valid, 0, 4)
    s2 = features.robust_sigma_x(T2, v2, smooth_px=1)
    ratio = float(np.median(s1) / np.median(s2))
    assert 1.6 < ratio < 2.4


def test_hist_threshold_matches_order_statistic_within_a_bin():
    rng = np.random.default_rng(1)
    scores = rng.random(5000).astype(np.float64)
    hist = np.zeros(NB, np.int64)
    add_hist(hist, scores)
    far = 1e-3
    exact = metrics.far_threshold(scores, far)
    approx = hist_threshold(hist, far)
    assert abs(approx - exact) < 2.0 / (NB - 1) + 1e-6
    thr = 0.9
    exact_ge = int((scores >= thr).sum())
    assert abs(hist_ge(hist, thr) - exact_ge) <= int(hist.max())
