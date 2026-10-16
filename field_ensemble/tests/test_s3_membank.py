"""Банк нормы: FIFO, корежет, голова не сводится к расстоянию, энкодер заморожен."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dts.membank import (  # noqa: E402
    AGG_KERNEL,
    BANK_CAP,
    CORESET,
    EMBED,
    K_NN,
    MemBankNet,
    ScoreHead,
    blend_with_winner,
    fifo_append,
    greedy_coreset,
    l2_normalize,
    membank_loss,
    pure_background,
    score_field,
)


def test_fifo_drops_oldest():
    old = torch.arange(5, dtype=torch.float32)[:, None]
    new = torch.arange(5, 8, dtype=torch.float32)[:, None]
    out = fifo_append(old, new, cap=6)
    assert out.shape == (6, 1)
    assert int(out[0, 0]) == 2
    assert int(out[-1, 0]) == 7
    empty = fifo_append(torch.zeros(0, 3), torch.ones(2, 3), cap=4)
    assert empty.shape == (2, 3)
    assert fifo_append(old, torch.zeros(0, 1), cap=4).shape == (5, 1)


def test_coreset_is_deterministic_and_spread():
    x = torch.randn(40, 6)
    a = greedy_coreset(x, 7, seed=3)
    b = greedy_coreset(x, 7, seed=3)
    c = greedy_coreset(x, 7, seed=4)
    assert a.shape == (7, 6)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    for row in a:
        assert torch.any(torch.all(torch.isclose(x, row), dim=1))
    pts = torch.zeros(20, 2)
    pts[0] = torch.tensor([10.0, 0.0])
    pts[1] = torch.tensor([0.0, 10.0])
    picked = greedy_coreset(pts, 3, seed=1)
    span = torch.cdist(picked, picked).max()
    assert float(span) > 5.0
    small = greedy_coreset(x[:3], 8, seed=0)
    assert small.shape == (3, 6)


def test_pure_background_block():
    lab = torch.zeros(4, 6, dtype=torch.int16)
    lab[0, 0] = 1
    lab[2, 4] = -1
    m = pure_background(lab, 2)
    assert m.shape == (1, 2, 3)
    assert bool(m[0, 0, 0]) is False
    assert bool(m[0, 0, 1]) is True
    assert bool(m[0, 1, 2]) is False
    assert bool(m[0, 1, 0]) is True


def test_agg_starts_as_identity_and_keeps_shape():
    net = MemBankNet()
    assert net.agg.kernel_size == AGG_KERNEL
    x = torch.randn(2, EMBED, 8, 10)
    y = net.agg(x)
    assert y.shape == x.shape
    assert torch.allclose(y, x, atol=1e-5)


def test_head_reads_query_and_stats():
    d = 4
    head = ScoreHead(d)
    with torch.no_grad():
        head.mlp[0].weight.zero_()
        head.mlp[2].weight.zero_()
        head.mlp[2].bias.zero_()
        head.bilinear.weight.zero_()
        head.mlp[0].weight[0, 0] = 1.0
        head.mlp[2].weight[0, 0] = 1.0
    nearest = torch.zeros(1, d, 2, 2)
    stats = torch.zeros(1, 3, 2, 2)
    q1 = torch.zeros(1, d, 2, 2)
    q2 = q1.clone()
    q2[0, 0] = 1.5
    a, _h = head(q1, nearest, stats)
    b, _h = head(q2, nearest, stats)
    assert float((a - b).abs().sum().detach()) > 0.1
    with torch.no_grad():
        head.mlp[0].weight.zero_()
        head.mlp[0].weight[0, 2 * d] = 1.0
    stats2 = torch.ones(1, 3, 2, 2)
    c, _h = head(q1, nearest, stats2)
    z, _h = head(q1, nearest, stats)
    assert float((c - z).abs().sum().detach()) > 0.1


def test_head_learns_off_bank():
    torch.manual_seed(0)
    d = 4
    head = ScoreHead(d)
    opt = torch.optim.Adam(head.parameters(), lr=0.08)
    proto = l2_normalize(torch.tensor([[1.0, 0.0, 0.0, 0.0]]), 0).view(1, d)
    near = proto.view(1, d, 1, 1)
    for _ in range(60):
        bg = l2_normalize(proto + 0.02 * torch.randn(8, d), 1)
        an = l2_normalize(torch.tensor([[0.0, 1.0, 0.0, 0.0]]) + 0.02 * torch.randn(8, d), 1)
        q = torch.cat([bg, an], 0).view(16, d, 1, 1)
        nearest = near.expand(16, d, 1, 1).contiguous()
        diff = q - nearest
        dist = torch.sqrt((diff * diff).sum(1, keepdim=True).clamp(min=1e-12))
        stats = dist.expand(16, 3, 1, 1).contiguous()
        logit, hot = head(q, nearest, stats)
        lab = torch.zeros(16, 1, 1, dtype=torch.int16)
        lab[8:] = 1
        loss = membank_loss(logit, hot, lab, pos_weight=1.0, hot_weight=0.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        qbg = proto.view(1, d, 1, 1)
        qan = l2_normalize(torch.tensor([[0.0, 1.0, 0.0, 0.0]]), 0).view(1, d, 1, 1)
        nearest = proto.view(1, d, 1, 1)
        def stats_of(q):
            dist = torch.sqrt(((q - nearest) ** 2).sum(1, keepdim=True).clamp(min=1e-12))
            return dist.expand(1, 3, 1, 1).contiguous()
        lb, _ = head(qbg, nearest, stats_of(qbg))
        la, _ = head(qan, nearest, stats_of(qan))
    assert float(la) > float(lb) + 0.5


def test_encoder_frozen_and_bank_not_a_parameter():
    torch.manual_seed(1)
    net = MemBankNet()
    bank = torch.randn(8, EMBED)
    net.set_bank(bank)
    assert int(net.bank_count) == 8
    assert torch.allclose(net.active_bank().norm(dim=1), torch.ones(8), atol=1e-4)
    before = {k: v.detach().clone() for k, v in net.encoder.state_dict().items()}
    bank_before = net.bank.detach().clone()
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=1e-3)
    x = torch.randn(2, 3, 32, 64)
    lab = torch.zeros(2, 32, 64, dtype=torch.int16)
    lab[:, 8:16, 10:20] = 1
    net.train()
    opt.zero_grad()
    logit, hot, extra = net(x, perturb=False)
    assert extra is None
    assert logit.shape == (2, 1, 32, 64)
    loss = membank_loss(logit, hot, lab)
    loss.backward()
    opt.step()
    for k, v in net.encoder.state_dict().items():
        assert torch.equal(v, before[k])
    assert torch.equal(net.bank, bank_before)
    assert all(not p.requires_grad for p in net.encoder.parameters())
    assert net.adapter.weight.grad is not None
    assert float(net.adapter.weight.grad.abs().sum()) > 0
    assert float(net.head.bilinear.weight.grad.abs().sum()) > 0


def test_loss_ignores_invalid_and_extra_adds_targets():
    logit = torch.zeros(1, 1, 4, 4, requires_grad=True)
    hot = torch.zeros(1, 1, 4, 4, requires_grad=True)
    lab = torch.full((1, 4, 4), -1)
    loss = membank_loss(logit, hot, lab)
    assert float(loss.detach()) == 0.0
    lab2 = torch.zeros(1, 4, 4, dtype=torch.int16)
    extra = torch.zeros(1, 4, 4)
    extra[:, :, 0] = 1
    loss2 = membank_loss(logit, hot, lab2, extra=extra, pos_weight=1.0)
    assert float(loss2.detach()) > 0


def test_set_bank_caps_coreset():
    net = MemBankNet()
    proto = torch.randn(CORESET + 50, EMBED)
    net.set_bank(proto)
    assert int(net.bank_count) == CORESET
    raw = fifo_append(torch.randn(100, EMBED), torch.randn(BANK_CAP, EMBED), BANK_CAP)
    assert raw.shape[0] == BANK_CAP


def test_score_field_shape_and_eval_perturb_off():
    torch.manual_seed(2)
    net = MemBankNet()
    net.set_bank(torch.randn(K_NN + 2, EMBED))
    x = np.random.randn(3, 48, 70).astype(np.float32)
    net.eval()
    s1, h1 = score_field(net, x, tile=(32, 40), overlap=8, batch=2)
    s2, h2 = score_field(net, x, tile=(32, 40), overlap=8, batch=2)
    assert s1.shape == (48, 70)
    assert h1.shape == (48, 70)
    assert np.isfinite(s1).all() and np.isfinite(h1).all()
    assert np.allclose(s1, s2) and np.allclose(h1, h2)
    assert float(s1.min()) >= 0.0 and float(s1.max()) <= 1.0
    xt = torch.from_numpy(x[None])
    net.train()
    torch.manual_seed(0)
    _lp, _hp, extra_tr = net(xt, perturb=True)
    assert extra_tr is not None and extra_tr.shape == (1, 48, 70)
    net.eval()
    with torch.no_grad():
        a, _ha, ea = net(xt, perturb=True)
        b, _hb, eb = net(xt, perturb=False)
    assert ea is None and eb is None
    assert torch.allclose(a, b)


def test_blend_weight_is_fixed():
    s = np.array([0.0, 1.0], np.float32)
    w = np.array([1.0, 0.0], np.float32)
    out = blend_with_winner(s, w)
    assert np.allclose(out, np.array([0.8, 0.2], np.float32))


def test_source_does_not_name_real_traces():
    text = (ROOT / "dts" / "membank.py").read_text(encoding="utf-8")
    for bad in ("MGTS", "real_noise", "1-6.csv", "models_release"):
        assert bad not in text


def test_load_winner_freezes_e1():
    path = ROOT / "artifacts" / "stage1" / "unet_v810ft.pt"
    sd = torch.load(path, map_location="cpu", weights_only=True)
    net = MemBankNet()
    net.encoder.load_winner(str(path))
    assert torch.equal(net.encoder.e1[0].weight, sd["e1.0.weight"])
    assert all(not p.requires_grad for p in net.encoder.parameters())
    net.set_bank(torch.randn(6, EMBED))
    with torch.no_grad():
        logit, hot, extra = net(torch.zeros(1, 3, 32, 64), perturb=False)
    assert logit.shape[-2:] == (32, 64)
    assert torch.isfinite(logit).all() and extra is None
    _ = hot
