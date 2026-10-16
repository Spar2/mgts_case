"""Базовая линия оставляет широкое плато, голова сегментации не учится, дырка маски не протекает."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dts import models
from experiments.realnet.realnet import (
    HOLE_T, HOLE_X, RealNet, _hole_starts, adapt_batchnorm, combine_scores,
    design_matrix, fit_baseline, holdout_ids, huber_weighted, make_channels,
    phase_batch, random_holes, segmentation_head_tensors, train_mask_head,
    valid_far_threshold, weight_grid,
)


def _series(n_t, n_x, span_h, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, span_h * 3600.0, n_t)
    x = np.arange(n_x, dtype=np.float64) * 0.25
    return t, x, rng


def test_short_record_keeps_wide_transient_and_removes_level():
    t, x, rng = _series(180, 500, span_h=10.0)
    level = 21.0 + 0.4 * np.sin(x / 30.0)
    trend = 0.15 * ((t - t.mean()) / (t[-1] - t[0]))
    T = level[None, :] + trend[:, None] + rng.normal(0, 0.05, (len(t), len(x)))
    T = T.astype(np.float32)
    # 100 м × ~1.3 ч: шире медианы 40 м и короче записи
    T[70:100, 80:480] += 3.0
    valid = np.ones(T.shape, bool)
    valid[:, :5] = False
    T[:, :5] = -300
    _B, R, info = fit_baseline(T, valid, t, x)
    assert info["n_harm"] == 0
    assert abs(float(np.median(R[10:40, 100:200]))) < 0.25
    assert float(np.median(R[75:95, 200:360])) > 2.0
    # пропуски не тянут остаток соседей
    T2 = T.copy()
    T2[:, :5] = 80
    _B2, R2, _ = fit_baseline(T2, valid, t, x)
    assert np.allclose(R[:, 20:], R2[:, 20:], atol=1e-4)


def test_diurnal_is_absorbed_and_bump_is_not():
    t, x, rng = _series(240, 160, span_h=72.0, seed=1)
    diurnal = 1.2 * np.sin(2 * np.pi * t / 86400.0) + 0.4 * np.cos(4 * np.pi * t / 86400.0)
    prof = 0.2 * np.cos(x / 10.0)
    T = (18.0 + diurnal[:, None] + prof[None, :] + rng.normal(0, 0.04, (len(t), len(x)))).astype(np.float32)
    T[100:115, 20:40] += 2.5
    valid = np.ones(T.shape, bool)
    _B, R, info = fit_baseline(T, valid, t, x)
    assert info["n_harm"] == 2
    quiet = np.abs(R[:80]).mean()
    assert quiet < 0.2
    assert float(np.median(R[102:112, 24:36])) > 1.5


def test_channel0_keeps_plateau_that_spatial_median_erases():
    t, x, rng = _series(80, 600, span_h=8.0, seed=2)
    T = np.full((len(t), len(x)), 19.0, np.float32) + rng.normal(0, 0.05, (len(t), len(x))).astype(np.float32)
    # плато 100 м внутри, без краёв ближе 40 м к границе окна
    T[30:55, 80:480] += 2.0
    valid = np.ones(T.shape, bool)
    X, info = make_channels(T, valid, t, x, dx=0.25)
    interior = (slice(36, 50), slice(200, 360))
    assert float(np.median(np.abs(X[0][interior]))) > 2.0
    assert float(np.median(np.abs(X[1][interior]))) < 0.75
    assert X.shape[0] == 3
    assert info["residual"].shape == T.shape


def test_weight_grid_is_a_simplex_and_prefers_the_head_on_ties():
    grid = weight_grid()
    assert (1.0, 0.0, 0.0) in grid
    assert (0.0, 0.0, 1.0) in grid
    assert all(abs(sum(w) - 1.0) < 1e-6 for w in grid)
    assert grid[0][0] >= grid[-1][0]
    score = combine_scores(np.ones((2, 2)), np.zeros((2, 2)), np.full((2, 2), 0.5), (0.5, 0.0, 0.5))
    assert score.shape == (2, 2)
    assert float(score.mean()) == pytest.approx(0.75)


def test_holdout_never_includes_the_scored_file():
    names = ["1", "2", "3", "4", "5", "6"]
    for held in names:
        ids = holdout_ids(names, held)
        assert held not in ids
        assert len(ids) == 5


def test_hole_phases_cover_every_pixel_and_hide_it():
    for n in (16, 17, 32, 40, 64, 100, 201, 256):
        seen = np.zeros(n, bool)
        for off in (0, HOLE_T):
            for s in _hole_starts(n, off, HOLE_T, 16):
                seen[s:s + HOLE_T] = True
        if n >= HOLE_T:
            assert seen[:n].all(), n
    x = torch.zeros(1, 3, 48, 64)
    x[:, 0] = 1
    batch, mask = phase_batch(x)
    assert batch.shape[0] == 4
    covered = mask[:, 0].sum(0) > 0
    assert bool(covered.all())
    # в своей фазе дырка занулена, снаружи значение живо
    assert torch.all(batch[:, 0][mask[:, 0] > 0] == 0)
    assert torch.all(batch[:, 0][mask[:, 0] == 0] == 1)


def test_mask_forward_does_not_read_inside_the_hole():
    torch.manual_seed(0)
    net = RealNet(models.DTSUNet(w=4))
    # ненулевая голова, иначе прогноз тождественно ноль и тест ничего не проверяет
    with torch.no_grad():
        for mod in net.mask_head.modules():
            if isinstance(mod, nn.Conv2d):
                mod.weight.normal_(0, 0.02)
                if mod.bias is not None:
                    mod.bias.zero_()
    net.eval()
    x = torch.randn(1, 3, 48, 80)
    t0, w0 = 16, 24
    a = apply_masked(x, t0, w0)
    spiked = x.clone()
    spiked[:, :2, t0:t0 + HOLE_T, w0:w0 + HOLE_X] = 30
    b = apply_masked(spiked, t0, w0)
    ya, yb = net.mask_pred(a), net.mask_pred(b)
    assert torch.allclose(ya[:, :, t0:t0 + HOLE_T, w0:w0 + HOLE_X], yb[:, :, t0:t0 + HOLE_T, w0:w0 + HOLE_X])
    outside = x.clone()
    outside[:, 0, 4, 8] = 15
    yc = net.mask_pred(apply_masked(outside, t0, w0))
    assert not torch.allclose(ya, yc)


def apply_masked(x, t0, w0):
    y = x.clone()
    y[:, :3, t0:t0 + HOLE_T, w0:w0 + HOLE_X] = 0
    return y


def test_training_does_not_move_the_segmentation_head():
    torch.manual_seed(1)
    net = RealNet(models.DTSUNet(w=4))
    w, b = segmentation_head_tensors(net.unet)
    tiles = []
    bgs = []
    rng = np.random.default_rng(0)
    for _ in range(4):
        img = rng.normal(0, 0.2, (3, 32, 64)).astype(np.float32)
        img[0] += 2.0
        img[2] = 1
        tiles.append(img)
        bgs.append(np.ones((32, 64), np.float32))
    # сдвиг BN, затем короткая маска
    adapt_batchnorm(net.unet, tiles, batch=2)
    hist = train_mask_head(net, tiles, bgs, minutes=10, max_steps=3, batch=2, seed=0)
    w2, b2 = segmentation_head_tensors(net.unet)
    assert torch.equal(w, w2) and torch.equal(b, b2)
    assert hist[-1]["step"] == 3
    for p in net.unet.parameters():
        assert p.requires_grad is False


def test_mask_head_learns_to_fill_a_constant():
    torch.manual_seed(2)
    net = RealNet(models.DTSUNet(w=4))
    rng = np.random.default_rng(3)
    tiles, bgs = [], []
    for _ in range(6):
        img = np.zeros((3, 32, 64), np.float32)
        img[0] = 3.0 + rng.normal(0, 0.05, (32, 64))
        img[2] = 1
        tiles.append(img.astype(np.float32))
        bgs.append(np.ones((32, 64), np.float32))
    train_mask_head(net, tiles, bgs, minutes=10, max_steps=40, batch=4, seed=2, lr=0.1)
    net.eval()
    x = torch.zeros(1, 3, 32, 64)
    x[:, 0] = 3.0
    x[:, 2] = 1
    masked, m = random_holes(x, np.random.default_rng(4), n_holes=2)
    pred = net.mask_pred(masked)
    hole = m[0, 0] > 0
    assert float(pred[0, 0][hole].detach().mean()) > 1.5


def test_huber_ignores_zero_weight():
    pred = torch.ones(1, 1, 4, 4)
    target = torch.zeros(1, 1, 4, 4)
    assert float(huber_weighted(pred, target, torch.zeros_like(pred))) == 0.0
    assert float(huber_weighted(pred, target, torch.ones_like(pred))) > 0.0


def test_design_has_expected_rank_pieces():
    t = np.linspace(0, 48 * 3600, 100)
    Phi = design_matrix(t, 2)
    assert Phi.shape == (100, 6)
    assert np.linalg.matrix_rank(Phi) == 6


def test_valid_far_threshold_uses_only_valid_pixels():
    from dts import metrics
    rng = np.random.default_rng(0)
    score = rng.random((200, 50)).astype(np.float32)
    valid = np.ones(score.shape, bool)
    valid[:, :3] = False
    score[:, :3] = 100.0
    thr = valid_far_threshold(score, valid, 1e-3)
    # тот же квантиль, что у протокола FAR; невалидные сотни в него не входят
    assert thr == metrics.far_threshold(score[valid], 1e-3)
    assert thr < 2.0
    assert float(((score >= thr) & valid).mean()) < 2e-3
