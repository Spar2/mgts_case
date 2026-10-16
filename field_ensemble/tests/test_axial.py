"""Осевое внимание в бутылочном слое U-Net: тождество при нулевой проекции и смесь по оси."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dts import models
from dts.axial import AxialBottleneck, AxialUNet, AxisAttention, load_unet_weights


def test_time_attention_uniform_is_column_mean():
    att = AxisAttention(1, heads=1, axis="time", max_len=8)
    with torch.no_grad():
        att.qkv.weight.zero_()
        att.qkv.weight[2, 0, 0, 0] = 1.0  # V = x, Q = K = 0 -> равные логиты
        att.proj.weight.fill_(1.0)
        att.rel.zero_()
    x = torch.zeros(1, 1, 4, 2)
    x[0, 0, :, 0] = torch.tensor([1.0, 3.0, 5.0, 7.0])
    y = att(x)
    assert torch.allclose(y[0, 0, :, 0], torch.full((4,), 4.0))
    assert torch.allclose(y[0, 0, :, 1], torch.zeros(4))


def test_space_attention_uniform_is_row_mean():
    att = AxisAttention(1, heads=1, axis="x", max_len=8)
    with torch.no_grad():
        att.qkv.weight.zero_()
        att.qkv.weight[2, 0, 0, 0] = 1.0
        att.proj.weight.fill_(1.0)
        att.rel.zero_()
    x = torch.zeros(1, 1, 2, 4)
    x[0, 0, 0, :] = torch.tensor([1.0, 3.0, 5.0, 7.0])
    y = att(x)
    assert torch.allclose(y[0, 0, 0, :], torch.full((4,), 4.0))
    assert torch.allclose(y[0, 0, 1, :], torch.zeros(4))


def test_relative_bias_shifts_along_time():
    att = AxisAttention(1, heads=1, axis="time", max_len=8)
    with torch.no_grad():
        att.qkv.weight.zero_()
        att.qkv.weight[2, 0, 0, 0] = 1.0
        att.proj.weight.fill_(1.0)
        att.rel.fill_(-20.0)
        att.rel[0, att.max_len] = 20.0  # сдвиг k - q = +1
    x = torch.tensor([1.0, 10.0, 100.0, 1000.0]).view(1, 1, 4, 1)
    y = att(x)
    assert torch.allclose(y[0, 0, 0, 0], torch.tensor(10.0), atol=1e-4)
    assert torch.allclose(y[0, 0, 1, 0], torch.tensor(100.0), atol=1e-4)
    assert torch.allclose(y[0, 0, 2, 0], torch.tensor(1000.0), atol=1e-4)


def test_bottleneck_zero_proj_is_identity():
    block = AxialBottleneck(8, heads=2, mlp_ratio=1, max_time=16, max_x=16)
    x = torch.randn(2, 8, 16, 12)
    y = block(x)
    assert torch.allclose(y, x, atol=1e-6)


def _pair(seed=0):
    torch.manual_seed(seed)
    unet = models.DTSUNet(w=16)
    ax = AxialUNet(w=16)
    missing = ax.load_state_dict(unet.state_dict(), strict=False).missing_keys
    assert missing and all(k.startswith("axial.") for k in missing)
    return unet, ax


def test_axial_matches_unet_when_projection_is_zero():
    unet, ax = _pair()
    x = torch.randn(2, 3, 32, 48)
    x[:, 2] = 1
    unet.eval()
    ax.eval()
    with torch.no_grad():
        assert torch.allclose(ax(x), unet(x), atol=1e-5)
    unet.train()
    ax.train()
    assert torch.allclose(ax(x), unet(x), atol=1e-5)


def test_opened_projection_changes_logits():
    ckpt = Path(__file__).resolve().parents[1] / "artifacts/stage1/unet.pt"
    ax = AxialUNet()
    if ckpt.exists():
        load_unet_weights(ax, ckpt)
    else:
        _unet, ax = _pair(1)
    x = torch.randn(1, 3, 32, 48)
    x[:, 2] = 1
    ax.eval()
    with torch.no_grad():
        y_id = ax(x).clone()
        ax.axial.attn_t.proj.weight.normal_(0, 0.02)
        y_open = ax(x)
    assert (y_open - y_id).abs().max() > 1e-3


def test_hist_threshold_tracks_exact_order_statistic():
    import numpy as np
    from dts import metrics as M
    from dts.axial import far_threshold_hist
    rng = np.random.default_rng(0)
    scores = rng.random(200_000).astype(np.float64)
    exact = M.far_threshold(scores, 1e-3)
    approx = far_threshold_hist(scores, 1e-3, nbin=100_000)
    assert abs(exact - approx) < 2e-5


def test_invalid_mask_stays_finite_and_identity():
    unet, ax = _pair(2)
    x = torch.randn(1, 3, 32, 40)
    x[:, 2] = 0
    ax.eval()
    unet.eval()
    with torch.no_grad():
        y = ax(x)
    assert torch.isfinite(y).all()
    with torch.no_grad():
        assert torch.allclose(y, unet(x), atol=1e-5)
    with torch.no_grad():
        ax.axial.attn_t.proj.weight.normal_(0, 0.05)
        ax.axial.attn_x.proj.weight.normal_(0, 0.05)
        ax.axial.ff[2].weight.normal_(0, 0.05)
    y2 = ax(x)
    assert torch.isfinite(y2).all()


def test_proj_gets_grad_at_zero_init():
    _, ax = _pair(3)
    x = torch.randn(1, 3, 32, 40)
    x[:, 2] = 1
    y = torch.zeros(1, 32, 40, dtype=torch.long)
    y[0, 8:20, 10:30] = 1
    loss = torch.nn.functional.cross_entropy(ax(x), y)
    loss.backward()
    assert ax.axial.attn_t.proj.weight.grad.abs().sum() > 0
    assert ax.axial.ff[2].weight.grad.abs().sum() > 0
    # при нулевой проекции градиент QKV ещё нулевой: его открывает следующий шаг
    assert ax.axial.attn_t.qkv.weight.grad is None or ax.axial.attn_t.qkv.weight.grad.abs().sum() == 0
    opt = torch.optim.SGD(ax.parameters(), lr=1e-2)
    opt.step()
    opt.zero_grad()
    torch.nn.functional.cross_entropy(ax(x), y).backward()
    assert ax.axial.attn_t.qkv.weight.grad.abs().sum() > 0


def test_load_released_unet_checkpoint():
    ckpt = Path(__file__).resolve().parents[1] / "artifacts/stage1/unet.pt"
    if not ckpt.exists():
        return
    unet = models.DTSUNet(w=16)
    unet.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False))
    ax = AxialUNet(w=16)
    missing = load_unet_weights(ax, ckpt)
    assert all(k.startswith("axial.") for k in missing)
    x = torch.randn(1, 3, 64, 80)
    x[:, 2] = 1
    unet.eval()
    ax.eval()
    with torch.no_grad():
        err = (ax(x) - unet(x)).abs().max().item()
    assert err < 1e-4
