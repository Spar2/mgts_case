"""Слепой автоэнкодер: дырка 8×16 не протекает, маска и скор согласованы, тайлы без меток."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.ssl.ssl import (
    BLOCK_T, BLOCK_X, BlindSpotAE, FieldTiles, anomaly_maps, blind_box,
    coverage_spans, load_model, masked_loss, reconstruct, sample_block_mask, save_model,
)


def _awaken(model):
    """Ненулевые положительные веса: GELU открыт, выход не тождественный ноль."""
    with torch.no_grad():
        for mod in model.modules():
            if isinstance(mod, torch.nn.Conv2d):
                mod.weight.fill_(0.05)
                if mod.bias is not None:
                    mod.bias.fill_(0.02)


def test_blind_box_is_8_by_16():
    r0, r1, c0, c1 = blind_box(20, 30)
    assert (r1 - r0, c1 - c0) == (BLOCK_T, BLOCK_X)
    assert r0 <= 20 < r1 and c0 <= 30 < c1


def test_center_block_does_not_change_prediction():
    torch.manual_seed(0)
    model = BlindSpotAE()
    _awaken(model)
    model.eval()
    x = torch.randn(1, 3, 48, 80)
    i, j = 24, 40
    y1 = model(x)
    r0, r1, c0, c1 = blind_box(i, j)
    x2 = x.clone()
    x2[:, :2, r0:r1, c0:c1] = torch.randn_like(x2[:, :2, r0:r1, c0:c1]) * 5
    y2 = model(x2)
    assert torch.allclose(y1[:, :, i, j], y2[:, :, i, j], atol=1e-5)
    # контекст сразу за верхней границей дырки обязан влиять
    x3 = x.clone()
    x3[:, 0, r0 - 1, j] += 10
    y3 = model(x3)
    assert (y1[:, :, i, j] - y3[:, :, i, j]).abs().sum() > 0


def test_channel_valid_may_be_seen_temperature_may_not():
    """Градиент каналов 0–1 внутри дырки нулевой; снаружи, у границы — нет."""
    torch.manual_seed(1)
    model = BlindSpotAE(width=8, dilations=(1, 2), k=3, side_k=3)
    _awaken(model)
    x = torch.randn(1, 3, 40, 64, requires_grad=True)
    i, j = 20, 32
    model(x)[0, :, i, j].sum().backward()
    r0, r1, c0, c1 = blind_box(i, j)
    assert torch.count_nonzero(x.grad[:, :2, r0:r1, c0:c1]) == 0
    assert x.grad[0, 0, r0 - 1, j].abs() > 0


def test_tiled_reconstruct_matches_full_forward():
    torch.manual_seed(2)
    model = BlindSpotAE(width=4, dilations=(1, 2), k=3, side_k=3)
    _awaken(model)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3, 70, 90)).astype(np.float32)
    X[2] = 1
    full = model(torch.from_numpy(X)[None])[0].detach().numpy()
    tiled = reconstruct(model, X, tile=(36, 48), batch=3, max_full_pixels=0)
    assert tiled.shape == (2, 70, 90)
    assert np.allclose(full, tiled, atol=1e-4)


def test_default_arch_tile_matches_full():
    torch.manual_seed(3)
    model = BlindSpotAE()
    _awaken(model)
    rng = np.random.default_rng(1)
    X = rng.normal(size=(3, 100, 140)).astype(np.float32)
    X[2] = 1
    full = model(torch.from_numpy(X)[None])[0].detach().numpy()
    tiled = reconstruct(model, X, tile=(80, 100), batch=2, max_full_pixels=0)
    assert np.allclose(full, tiled, atol=1e-4)


def test_block_mask_shape_and_block_size():
    rng = np.random.default_rng(0)
    m = sample_block_mask(4, 96, 160, rng, p=1.0)
    assert m.shape == (4, 1, 96, 160)
    # при p=1 блоки не слипаются: шаг сетки больше размера блока
    assert m.mean() < 0.3
    # найдём хоть один целый блок 8×16 из единиц
    found = False
    for t in range(0, 90):
        for c in range(0, 140):
            if m[0, 0, t:t + BLOCK_T, c:c + BLOCK_X].sum() == BLOCK_T * BLOCK_X:
                found = True
                break
        if found:
            break
    assert found


def test_loss_ignores_unmasked_and_invalid():
    pred = torch.zeros(2, 2, 16, 32)
    target = torch.ones(2, 2, 16, 32)
    mask = torch.zeros(2, 1, 16, 32)
    valid = torch.ones(2, 1, 16, 32)
    assert float(masked_loss(pred, target, mask, valid)) == 0.0
    mask[:, :, 0:8, 0:16] = 1
    valid[:, :, 0:8, 0:16] = 0
    assert float(masked_loss(pred, target, mask, valid)) == 0.0
    valid[:] = 1
    loss = masked_loss(pred, target, mask, valid)
    assert loss.ndim == 0 and float(loss) > 0
    pred2 = pred.clone()
    pred2[:, :, 8:, :] = 5  # вне маски
    assert torch.allclose(loss, masked_loss(pred2, target, mask, valid))


def test_score_sign_and_smoothing():
    X = np.zeros((3, 40, 60), np.float32)
    X[2] = 1
    pred = np.zeros((2, 40, 60), np.float32)
    X[0, 15:25, 20:40] = 6
    s, h = anomaly_maps(X, pred, sigma=(1.0, 2.0))
    assert s.dtype == np.float32 and h.dtype == np.float16
    assert s.shape == h.shape == (40, 60)
    assert np.isfinite(s).all() and (s >= 0).all()
    assert h[20, 30] > 0 and s[20, 30] > s[2, 2]
    X[0, 15:25, 20:40] = -6
    s2, h2 = anomaly_maps(X, pred, sigma=(1.0, 2.0))
    assert h2[20, 30] < 0 and s2[20, 30] > s2[2, 2]
    X[2] = 0
    s3, h3 = anomaly_maps(X, pred, sigma=(1.0, 2.0))
    assert s3.sum() == 0 and h3.sum() == 0


def test_coverage_spans_partition():
    rng = np.random.default_rng(0)
    for _ in range(40):
        length = int(rng.integers(1, 300))
        tile = int(rng.integers(1, 120))
        before = int(rng.integers(0, 30))
        after = int(rng.integers(0, 30))
        spans = coverage_spans(length, tile, before, after)
        covered = np.zeros(length, np.int32)
        for s, a, b in spans:
            assert 0 <= s <= a <= b <= s + tile or (s == 0 and b <= max(tile, length))
            assert b <= length and a >= 0
            covered[a:b] += 1
        assert covered.min() == 1 and covered.max() == 1


def test_field_tiles_do_not_read_labels(tmp_path):
    d = tmp_path / "train"
    d.mkdir()
    x = np.zeros((3, 30, 40), np.float16)
    x[0, 10, 12] = 4
    x[2] = 1
    np.save(d / "seq0007_X.npy", x)
    np.save(d / "seq0007_y.npy", np.full((30, 40), 9, np.int8))
    tiles = FieldTiles(tmp_path, seed=0)
    assert len(tiles.paths) == 1
    batch = tiles.sample(6, 16, 16)
    assert batch.shape == (6, 3, 16, 16) and batch.dtype == np.float32
    assert np.isclose(batch[:, 2].mean(), 1)


def test_checkpoint_roundtrip_and_score_dtype(tmp_path):
    torch.manual_seed(0)
    model = BlindSpotAE(width=4, dilations=(1,), k=3, side_k=3)
    model.score_sigma = (1.5, 3.0)
    path = tmp_path / "ssl_ae.pt"
    save_model(model, path, seed=0, steps=1, seconds=0.1)
    loaded = load_model(path)
    assert loaded.score_sigma == (1.5, 3.0)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3, 24, 36)).astype(np.float32)
    X[2] = 1
    with torch.no_grad():
        a = model(torch.from_numpy(X)[None])
        b = loaded(torch.from_numpy(X)[None])
    assert torch.allclose(a, b)
    from experiments.ssl.ssl import score_field
    s, h = score_field(loaded, X)
    assert s.dtype == np.float32 and h.dtype == np.float16 and s.shape == (24, 36)


def test_mask_loss_drops_on_fixed_batch():
    torch.manual_seed(0)
    model = BlindSpotAE(width=8, dilations=(1, 2), k=3, side_k=3)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    rng = np.random.default_rng(0)
    # гладкий фон предсказуем из соседей за пределами блока 8×16
    yy, xx = np.mgrid[0:40, 0:64]
    field = (2.0 * np.sin(xx / 8.0) + 1.2 * np.cos(yy / 7.0)).astype(np.float32)
    x = np.zeros((4, 3, 40, 64), np.float32)
    x[:, 0] = field
    x[:, 1] = 0.5 * field
    x[:, 2] = 1
    xb = torch.from_numpy(x)
    mask = torch.from_numpy(sample_block_mask(4, 40, 64, rng, p=0.5))
    losses = []
    for _ in range(25):
        model.train()
        inp = xb.clone()
        inp[:, :2] *= 1 - mask
        opt.zero_grad()
        loss = masked_loss(model(inp), xb[:, :2], mask, xb[:, 2:3])
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0] * 0.7


def test_same_seed_same_step():
    def once(seed):
        torch.manual_seed(seed)
        model = BlindSpotAE(width=4, dilations=(1,), k=3, side_k=3)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        g = torch.Generator().manual_seed(seed)
        xb = torch.randn(2, 3, 24, 32, generator=g)
        xb[:, 2] = 1
        mask = torch.zeros(2, 1, 24, 32)
        mask[:, :, 4:12, 4:20] = 1
        inp = xb.clone()
        inp[:, :2] *= 1 - mask
        loss = masked_loss(model(inp), xb[:, :2], mask, xb[:, 2:3])
        loss.backward()
        opt.step()
        return float(loss.detach()), model.head[-1].weight.detach().clone()
    a, wa = once(5)
    b, wb = once(5)
    c, wc = once(6)
    assert a == b and torch.equal(wa, wb)
    assert c != a or not torch.equal(wc, wa)
