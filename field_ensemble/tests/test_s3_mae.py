"""MAE: маска не пропускает значение клетки, ширина <= 16, правила приёмки. Без больших кэшей."""
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dts.mae_anom import (  # noqa: E402
    BAND, CANDIDATES, NESTED_BAR, NESTED_GAIN, PROTOCOL_TEXT, WINNER_VAL,
    MaskedReconstructor, accept_decision, apply_cdf, blend_scores, channel_width,
    choose_candidate, infer_masks, interior_mask, mae_loss, reconstruct, residuals,
    sample_train_mask, spatial_band_mask, training_loss,
)


def test_width_cap_and_forward_shape():
    torch.set_num_threads(2)
    try:
        MaskedReconstructor(32)
        raise AssertionError("ширина 32 должна отвергаться")
    except ValueError:
        pass
    model = MaskedReconstructor(16)
    assert model.width == 16
    assert channel_width(model) <= 16
    model.eval()
    ch = torch.zeros(2, 2, 32, 64)
    valid = torch.ones(2, 1, 32, 64)
    mask = torch.zeros(2, 1, 32, 64)
    y = model(ch, valid, mask)
    assert tuple(y.shape) == (2, 2, 32, 64)
    assert torch.isfinite(y).all()
    # нулевая голова на нулевом входе — нулевой ответ
    assert float(y.detach().abs().max()) == 0.0


def test_masked_cell_does_not_change_output():
    torch.manual_seed(0)
    model = MaskedReconstructor(8)
    nn.init.normal_(model.head.weight, std=0.08)
    nn.init.zeros_(model.head.bias)
    model.eval()
    ch = torch.randn(1, 2, 32, 64)
    valid = torch.ones(1, 1, 32, 64)
    mask = torch.zeros(1, 1, 32, 64)
    mask[..., 16:] = 1
    y1 = model(ch, valid, mask)
    ch_hidden = ch.clone()
    ch_hidden[..., 16:] += 5
    y2 = model(ch_hidden, valid, mask)
    assert torch.allclose(y1, y2, atol=1e-5)
    ch_vis = ch.clone()
    ch_vis[..., :16] += 5
    y3 = model(ch_vis, valid, mask)
    assert not torch.allclose(y1, y3, atol=1e-4)


def test_reconstruct_ignores_hidden_column():
    torch.manual_seed(1)
    model = MaskedReconstructor(8)
    nn.init.normal_(model.head.weight, std=0.05)
    nn.init.zeros_(model.head.bias)
    model.eval()
    rng = np.random.default_rng(0)
    ch = rng.normal(0, 0.4, size=(2, 32, 64)).astype(np.float32)
    valid = np.ones((32, 64), np.float32)
    hat1 = reconstruct(model, ch, valid, tile=(32, 64), overlap=0, band=8, radius=1, tile_batch=1)
    ch2 = ch.copy()
    ch2[:, :, 20] += 3.0
    hat2 = reconstruct(model, ch2, valid, tile=(32, 64), overlap=0, band=8, radius=1, tile_batch=1)
    assert hat1.shape == (2, 32, 64)
    assert np.allclose(hat1[:, :, 20], hat2[:, :, 20], atol=1e-4)


def test_band_masks_cover_each_column_twice():
    masks = infer_masks(20, 64, band=8)
    assert len(masks) == 4
    total = np.sum(np.stack(masks, 0), 0)
    assert np.allclose(total, 2.0)
    one = spatial_band_mask(4, 64, 8, 0, 0)
    assert abs(float(one.mean()) - 0.5) < 1e-6
    inn = interior_mask(one, radius=1)
    assert float(inn.sum()) > 0
    assert np.all(inn <= one + 1e-6)
    # край полосы рядом с открытыми колонками не внутренний, середина — да
    assert float(inn[:, 7].sum()) == 0.0
    assert float(inn[:, 3].sum()) > 0.0


def test_loss_drops_on_one_batch():
    torch.manual_seed(2)
    model = MaskedReconstructor(8)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    rng = np.random.default_rng(1)
    ch = torch.from_numpy(rng.normal(0, 1, size=(4, 2, 32, 64)).astype(np.float32))
    # гладкая по x компонента, чтобы дыру можно было заполнить
    ch[:, 1] = torch.nn.functional.avg_pool2d(ch[:, 1:2], (1, 5), stride=1, padding=(0, 2))[:, 0]
    valid = torch.ones(4, 1, 32, 64)
    mask = torch.from_numpy(np.stack([sample_train_mask(32, 64, rng) for _ in range(4)])[:, None])
    model.train()
    first = None
    last = None
    for _ in range(30):
        opt.zero_grad(set_to_none=True)
        loss = training_loss(model, ch, valid, mask, sigma=0.0)
        loss.backward()
        opt.step()
        last = float(loss.detach())
        if first is None:
            first = last
    assert np.isfinite(last)
    assert last < first - 0.03
    # потеря смотрит только на скрытые клетки: смена цели там меняет loss, открытая цель — нет
    with torch.no_grad():
        hat = model(ch, valid, mask)
    base = float(mae_loss(hat, ch, mask, valid))
    tgt = ch.clone()
    tgt = tgt * (1 - mask) + (tgt + 1.0) * mask
    changed = float(mae_loss(hat, tgt, mask, valid))
    assert changed > base + 1e-4


def test_zero_head_member_maps_match_residuals():
    from dts.mae_anom import head_is_zero, member_maps
    model = MaskedReconstructor(16)
    assert head_is_zero(model)
    rng = np.random.default_rng(0)
    ch = rng.normal(size=(2, 32, 64)).astype(np.float32) * 20
    ch[0, 0, 1] = np.nan
    valid = np.ones((32, 64), np.float32)
    valid[:, 0] = 0
    knots = np.linspace(0, 12, 33)
    kw = dict(tile=(32, 64), overlap=0, band=8, radius=1, tile_batch=1)
    s, h = member_maps(model, ch, valid, knots, **kw)
    r0, r1 = residuals(model, ch, valid, **kw)
    assert np.allclose(s, apply_cdf(np.abs(r1), knots))
    expect_h = np.sign(r0).astype(np.float32) * (valid > 0.5)
    assert np.allclose(h, expect_h)


def test_mae_flag_default_off_and_blend_weight():
    import inspect
    from dts import infer
    assert inspect.signature(infer.Analyzer.__init__).parameters["mae"].default == "off"
    assert infer._MAE == ("off", "on")
    try:
        infer.Analyzer(mae="branch")
    except ValueError as exc:
        assert "off" in str(exc) and "on" in str(exc)
    else:
        raise AssertionError("ожидался ValueError")
    score = np.array([0.2, 0.8], np.float32)
    hot = np.array([0.5, -0.5], np.float32)
    s_mae = np.array([1.0, 0.0], np.float32)
    h_mae = np.array([1.0, -1.0], np.float32)
    got_s, got_h = infer.blend_with_mae(score, hot, s_mae, h_mae, 4)
    assert np.allclose(got_s, (4 * score + s_mae) / 5)
    assert np.allclose(got_h, (4 * hot + h_mae) / 5)
    spec = infer.mae_spec()
    assert abs(spec["threshold"] - 0.7626513242731557) < 1e-12
    assert abs(spec["mae_weight"] - 0.2) < 1e-12
    assert Path(spec["ckpt"]).is_file()


def test_cdf_blend_and_accept_rules():
    knots = np.linspace(0, 10, 11)
    s = apply_cdf(np.array([0.0, 5.0, 10.0, 20.0], np.float32), knots)
    assert abs(float(s[0])) < 1e-6
    assert float(s[1]) < float(s[2]) <= 1.0
    assert abs(float(s[3]) - 1.0) < 1e-6
    mae = np.array([0.2, 0.8], np.float32)
    win = np.array([0.4, 0.4], np.float32)
    assert np.allclose(blend_scores(mae, win, "ens_mean4"), (4 * win + mae) / 5)
    assert np.allclose(blend_scores(mae, win, "ens_max"), np.maximum(mae, win))
    ious = {k: 0.1 for k in CANDIDATES}
    ious["mae"] = 0.5
    assert choose_candidate(ious) == "mae"
    ious = {k: 0.2 for k in CANDIDATES}
    assert choose_candidate(ious) == "ens_mean4"
    good = accept_decision(0.40, 0.3868, 0.01, 0.40, 0.41, 0.40, 0.41, 0.40, True)
    assert good["accept"] and good["beats"]
    short = accept_decision(WINNER_VAL + 1e-5, WINNER_VAL - 1e-4, 0.01, 0.40, 0.5, 0.4, 0.5, 0.4, True)
    assert short["accept"] and not short["beats"]
    low_gain = accept_decision(0.40, 0.38, 0.002, 0.40, 0.5, 0.4, 0.5, 0.4, True)
    assert not low_gain["accept"] and not low_gain["nested_ok"]
    tied_bar = accept_decision(0.40, 0.38, 0.01, NESTED_BAR, 0.5, 0.4, 0.5, 0.4, True)
    assert not tied_bar["accept"]
    reg = accept_decision(0.40, 0.38, 0.01, 0.40, 0.30, 0.40, 0.50, 0.40, True)
    assert not reg["v10_ok"] and not reg["accept"]
    assert NESTED_GAIN == 0.003
    assert abs(NESTED_BAR - 0.3897) < 1e-9


def test_scripts_and_doc_contract():
    train = (ROOT / "scripts/s3_mae_train.py").read_text(encoding="utf-8")
    ev = (ROOT / "scripts/s3_mae_eval.py").read_text(encoding="utf-8")
    doc = (ROOT / "docs/S3_MAE.md").read_text(encoding="utf-8")
    assert "--resume" in train and "last.pt" in train and "DONE" in train
    assert "run_resumable" in train
    assert "load_csv" not in train and "MGTS" not in train and "aug_real" not in train
    assert "flowprior" not in train and "VelocityNet" not in train
    assert "s3_sel2_cv" in ev and "0.38671235821914546" in ev
    assert "load_csv" in ev and "used_for_selection" in ev
    assert "operator_cards" not in ev and "cards_batch" not in ev
    assert "/tmp/s3_mae_maps" in ev
    assert "np.savez" not in ev
    assert PROTOCOL_TEXT in doc
    assert "dT_space" in doc and "синтет" in doc
    assert "не rectified flow" in doc
    assert "0.3897" in doc
    assert "ACCEPTED (rule fix)" in doc
    assert "+0.034, −0.022, −0.002, +0.015, +0.002" in doc
    assert BAND == 8
