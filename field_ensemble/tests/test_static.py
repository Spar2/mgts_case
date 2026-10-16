"""Статичные узкие горячие точки: синтетика, канал без z-score, слияние."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dts.features import make_channels
from dts.metrics import far_threshold
from dts.grid import DT_S
from dts.static_net import (
    FEAT_SCALE_C, GATE_CIN, LONG_BINS, LONG_ROWS, N_FEAT, N_FEAT6, WIN_2H_ROWS, WIN_30MIN_ROWS,
    StaticGate, StaticNet, StaticNet6, add_score_hist, fuse, fuse_gated, gate_channels, gate_loss,
    receptive_field_px, residual_from_temperature, score_column_stats, temperature_features,
    temporal_features, threshold_from_hist,
)
from dts.static_synth import (
    DX, EDGE_RANGE, ENV_LO, HOT_AMP, HOT_WIDTH, S2_AMP, S2_WIDTH, SPOT_M, SPOT_RANGE,
    apply_baseline, apply_profiles, apply_row_gaps, fixed_hot_specs, fixed_scene, fwhm_m,
    hot_spec, row_gap_mask, sample_baseline, sample_train_specs, static2_train_specs, static2_view,
    time_multiplier, unit_profile,
)


def test_hot_spec_ranges_and_full_window_no_onset():
    rng = np.random.default_rng(0)
    R = np.zeros((36, 480), np.float32)
    valid_cols = np.zeros(480, bool)
    valid_cols[100:140] = True  # занято чужим событием — центр сюда не ставим
    specs = []
    avoid = valid_cols.copy()
    for _ in range(6):
        sp, avoid = hot_spec(rng, 480, DX, avoid)
        assert sp is not None
        specs.append(sp)
    for sp in specs:
        assert HOT_WIDTH[0] <= sp["width_m"] <= HOT_WIDTH[1]
        assert HOT_AMP[0] <= sp["amp"] <= HOT_AMP[1]
        assert abs(sp["drift"]) <= 0.20 + 1e-9
        assert SPOT_RANGE[0] <= sp["spot_m"] <= SPOT_RANGE[1]
        assert EDGE_RANGE[0] <= sp["edge_m"] <= EDGE_RANGE[1]
        assert 1 <= len(sp["fluct"]) <= 2
        assert sp["role"] == "hot"
        assert sp["kind"] == "segment"
        y = unit_profile(480, sp)
        fw = fwhm_m(y, DX)
        # пятно не даёт профилю схлопнуться в точку и не раздувает его дальше участка с краями
        assert fw + 1e-6 >= 0.75 * SPOT_M
        assert fw <= sp["width_m"] + sp["spot_m"] + sp["edge_m"] + 1.0
        assert float(np.max(np.abs(np.diff(y.astype(np.float64))))) < 0.55
        assert float(y.max()) == np.float32(1.0) or abs(float(y.max()) - 1.0) < 1e-5
        mult = time_multiplier(36, sp)
        assert float(mult.min()) >= ENV_LO - 1e-6
        assert float(mult.max()) <= 1.22 + 1e-6
        assert not valid_cols[sp["center"]]
    out, pos, hard = apply_profiles(R, specs)
    assert hard.sum() == 0
    assert pos.sum() > 0
    delta = out - R
    on = np.flatnonzero(pos)
    # событие есть на первом и последнем кадре и нигде не гаснет
    assert np.all(delta[:, on] > 0)
    col_min = delta[:, on].min(axis=0)
    col_max = delta[:, on].max(axis=0)
    # зажатый дрейф не выглядит как включение
    assert np.all(col_min > 0.5 * col_max)
    for sp in specs:
        added = delta[:, sp["center"]]
        assert float(added.min() / sp["amp"]) >= 0.8 - 1e-5


def test_spot_limits_unresolved_spike_and_amp_is_celsius():
    # Источник уже пятна прибора после свёртки не остаётся уже ~1 м.
    y = unit_profile(400, dict(center=200, width_m=0.25, amp=8.0, kind="gaussian", dx=DX))
    assert fwhm_m(y, DX) + 1e-6 >= SPOT_M * 0.75
    assert abs(float(y.max()) - 1.0) < 1e-5
    rng = np.random.default_rng(3)
    for _ in range(40):
        specs = sample_train_specs(rng, 1200)
        for sp in specs:
            if sp["role"] == "neg" and sp["kind"] == "gaussian":
                assert sp["width_m"] + 1e-6 >= SPOT_M
                assert sp["width_m"] < 2.0
            if sp["kind"] == "dip":
                assert HOT_WIDTH[0] <= sp["width_m"] <= HOT_WIDTH[1]
                assert sp["amp"] < 0
                assert sp["role"] == "neg"
            if sp["kind"] == "step":
                assert sp["mask"] == "edge"
                assert sp["role"] == "neg"
            if sp["role"] == "neg":
                assert sp["kind"] != "segment"
    # Пик — градусы избытка, не σ и не функция координаты вдоль кабеля.
    spec = dict(center=100, width_m=4.0, amp=12.0, drift=0.0, kind="segment", role="hot", dx=DX)
    spec_far = dict(spec)
    spec_far["center"] = 700
    a, _, _ = apply_profiles(np.zeros((6, 900), np.float32), [spec])
    b, _, _ = apply_profiles(np.zeros((6, 900), np.float32), [spec_far])
    assert abs(float(a[:, 100].mean()) - 12.0) < 0.05
    assert abs(float(b[:, 700].mean()) - 12.0) < 0.05
    # apply_profiles не подмешивает базовую линию: вдали от участка поле нулевое
    assert abs(float(a[:, 10].mean())) < 1e-4


def test_fixed_specs_are_deterministic_and_train_has_distractors():
    a = fixed_hot_specs(800, seed=20261015, n_events=2)
    b = fixed_hot_specs(800, seed=20261015, n_events=2)
    c = fixed_hot_specs(800, seed=20261016, n_events=2)
    assert a == b and a != c
    assert len(a) == 2
    rng = np.random.default_rng(1)
    saw_neg = False
    for _ in range(30):
        specs = sample_train_specs(rng, 900)
        if any(sp["role"] == "neg" for sp in specs):
            saw_neg = True
            break
    assert saw_neg
    specs = sample_train_specs(np.random.default_rng(2), 900)
    _, pos, hard = apply_profiles(np.zeros((8, 900), np.float32), specs)
    assert not np.any(pos & hard)


def test_scene_is_deterministic_and_gaps_do_not_zero_temperature():
    a = fixed_scene(900, seed=20261015, n_events=2)
    b = fixed_scene(900, seed=20261015, n_events=2)
    c = fixed_scene(900, seed=20261016, n_events=2)
    assert a == b and a != c
    hots = [sp for sp in a["specs"] if sp["role"] == "hot"]
    negs = [sp for sp in a["specs"] if sp["role"] != "hot"]
    assert hots == fixed_hot_specs(900, seed=20261015, n_events=2)
    assert len(hots) == 2
    assert 0 <= len(negs) <= 3
    base = sample_baseline(np.random.default_rng(4))
    field = apply_baseline(np.zeros((6, 800), np.float32), base)
    assert float(np.max(np.abs(field))) > 0.1
    assert float(np.max(np.abs(np.diff(field[0].astype(np.float64))))) < 1.0
    # пропуски снимают часть строк и оставляют не меньше 60 %
    gap = dict(frac=0.25, n_blocks=2, scatter=0.02, seed=7)
    mask = row_gap_mask(200, gap)
    assert mask.any() and not mask.all()
    assert float(mask.mean()) >= 0.60 - 1e-9
    # один и тот же gap даёт одну и ту же маску; температура не обнуляется
    assert np.array_equal(mask, row_gap_mask(200, gap))
    temp = np.full((200, 4), 7.5, np.float32)
    valid = np.ones_like(temp, bool)
    gvalid = apply_row_gaps(valid, gap)
    assert gvalid.sum() < valid.sum()
    assert float(gvalid.mean()) >= 0.60 - 1e-9
    assert np.all(temp == 7.5)
    # ступенька помечает переход, а не половину кабеля
    step = dict(center=400, width_m=10.0, amp=6.0, drift=0.0, kind="step", role="neg",
                dx=DX, spot_m=1.0, edge_m=0.0, fluct=[], mask="edge")
    _, pos, hard = apply_profiles(np.zeros((8, 1600), np.float32), [step])
    assert pos.sum() == 0
    assert 0 < hard.sum() * DX < 30.0


def test_time_channel_removes_static_bump_space_channel_keeps_it():
    rng = np.random.default_rng(0)
    rows, cols = 80, 480
    noise = rng.normal(0, 0.2, size=(rows, cols)).astype(np.float32)
    trend = np.linspace(0, 2.0, cols).astype(np.float32)
    T = 21.0 + noise + trend[None, :]
    valid = np.ones((rows, cols), bool)
    sp = dict(center=240, width_m=4.0, amp=15.0, drift=0.0, kind="gaussian", role="hot", dx=DX)
    y = unit_profile(cols, sp)
    T2 = T + sp["amp"] * y[None, :]
    ch, aux = make_channels(T2, valid, dx=DX)
    core = y >= 0.5
    c0 = float(np.abs(ch[0][:, core]).mean())
    c1 = float(np.abs(ch[1][:, core]).mean())
    # канал 0 вычел временной профиль вместе со статичным горбом;
    # канал 1 — T минус гладкая пространственная опора, горб на месте (обрезка ±12 σ)
    assert c1 > 4.0 * max(c0, 1e-3)
    assert c1 > 3.0
    assert float(np.max(np.abs(aux["profile"][core]))) > 5.0


def test_features_are_celsius_not_zscore_and_trend_is_removed():
    rows, cols = 48, 800
    x = np.arange(cols) * DX
    sine = (4.0 * np.sin(2 * np.pi * x / 200.0)).astype(np.float32)
    sp = dict(center=400, width_m=4.0, amp=10.0, drift=0.0, kind="gaussian", role="hot", dx=DX)
    y = unit_profile(cols, sp)
    T = 18.0 + sine[None, :] + 10.0 * y[None, :]
    T = np.broadcast_to(T, (rows, cols)).copy()
    T += 0.05
    valid = np.ones_like(T, bool)
    resid, v = residual_from_temperature(T, valid, max_rows=rows)
    feats, good = temporal_features(resid, v)
    assert feats.shape[0] == N_FEAT
    assert good[400]
    # 10 °C / 10 = ~1, а не 10/σ. Гладкая синусоида 200 м в остаток не входит.
    assert 0.6 < float(feats[0, 400]) < 1.6
    far = np.r_[80:140, 660:720]
    assert float(np.max(np.abs(feats[0, far]))) < 0.35


def test_reference_pool_wider_than_hotspot_and_starts_at_zero():
    net = StaticNet()
    # 80 px * 0.25 м = 20 м, шире горячей точки 8 м
    assert net.ref.pool * DX >= 16.0
    x = torch.randn(2, N_FEAT, 400)
    with torch.no_grad():
        ref = net.ref(x)
        logit = net(x)
    assert ref.shape == x.shape
    assert torch.allclose(ref, torch.zeros_like(ref), atol=1e-5)
    assert logit.shape == (2, 400)
    assert torch.isfinite(logit).all()


def test_fuse_never_lowers_ensemble_and_gate_blocks_weak_static():
    score = np.array([[0.20, 0.90], [0.10, 0.40]], np.float32)
    p = np.array([0.30, 0.99], np.float32)
    fused = fuse(score, p, mode="max", scale=1.0, gate=0.0)
    assert np.all(fused >= score - 1e-6)
    assert abs(float(fused[0, 1]) - 0.99) < 1e-5
    assert abs(float(fused[1, 0]) - 0.30) < 1e-5
    gated = fuse(score, p, mode="max", scale=1.0, gate=0.95)
    assert abs(float(gated[1, 0]) - 0.10) < 1e-5
    assert abs(float(gated[0, 1]) - 0.99) < 1e-5
    ident = fuse(score, p, mode="identity")
    assert np.allclose(ident, score)
    blend = fuse(score, p, mode="blend", scale=1.0)
    assert np.allclose(blend, fused)
    # scale < 1 не опускает ансамбль
    soft = fuse(score, p, mode="blend", scale=0.5)
    assert np.all(soft + 1e-6 >= score)
    alone = fuse(score, p, mode="branch")
    assert alone.shape == score.shape
    assert abs(float(alone[0, 0]) - 0.30) < 1e-5
    assert abs(float(alone[1, 1]) - 0.99) < 1e-5


def test_gate_starts_closed_and_never_lowers_winner():
    gate = StaticGate(GATE_CIN)
    x = torch.randn(2, GATE_CIN, 96)
    with torch.no_grad():
        g = gate(x)
    assert g.shape == (2, 96)
    assert float(g.mean()) < 0.05
    score = np.array([[0.20, 0.90], [0.40, 0.10]], np.float32)
    closed = fuse_gated(score, np.array([0.99, 0.99], np.float32), np.zeros(2, np.float32))
    assert np.allclose(closed, score)
    opened = fuse_gated(score, np.array([0.30, 0.95], np.float32), np.ones(2, np.float32))
    assert np.all(opened + 1e-6 >= score)
    assert abs(float(opened[0, 1]) - 0.95) < 1e-5
    rows, cols = 32, 80
    s = np.full((rows, cols), 0.15, np.float32)
    s[:, 10] = 0.97
    valid = np.ones_like(s, bool)
    med, p90, mx = score_column_stats(s, valid, max_rows=rows)
    assert abs(float(mx[10]) - 0.97) < 1e-5
    assert float(med[10]) > 0.9
    assert float(mx[20]) < 0.2
    feats = np.zeros((N_FEAT + LONG_BINS, cols), np.float32)
    feats[0, 4] = 1.2
    ch, mean = gate_channels(s, valid, np.full(cols, 0.8, np.float32), np.full(cols, 0.4, np.float32), feats)
    assert ch.shape == (GATE_CIN, cols)
    assert abs(float(mean[0]) - 0.6) < 1e-5
    assert abs(float(ch[0, 10]) - float(med[10])) < 1e-5
    # пространственный канал — уже градусы/10, не z-score столбца s
    assert abs(float(ch[7, 4]) - 1.2) < 1e-5
    y = torch.zeros(2, 16)
    y[:, 3:6] = 1
    gg = torch.full_like(y, 0.2, requires_grad=True)
    pp = torch.full_like(y, 0.95)
    ww = torch.ones_like(y)
    loss, parts = gate_loss(gg, y, ww, pp)
    loss.backward()
    assert torch.isfinite(loss)


def test_long_bins_cover_the_whole_window():
    rows, cols = 240, 160
    R = np.zeros((rows, cols), np.float32)
    valid = np.ones_like(R, bool)
    R[:, 80] = 10.0
    R[:30, 40] = 10.0
    short = temporal_features(R, valid)
    assert short[0].shape[0] == N_FEAT
    feats, good = temporal_features(R, valid, max_rows=LONG_ROWS, n_bins=LONG_BINS)
    assert feats.shape[0] == N_FEAT + LONG_BINS
    assert good[80] and good[40]
    assert float(feats[N_FEAT:, 80].min()) > 0.7
    assert float(feats[N_FEAT, 40]) > 0.5
    assert float(feats[-1, 40]) < 0.15


def test_hist_threshold_matches_exact_far():
    rng = np.random.default_rng(0)
    scores = rng.random(200_000).astype(np.float64)
    hist = np.zeros(100_000, np.int64)
    add_score_hist(hist, scores)
    exact = far_threshold(scores, 1e-3)
    approx = threshold_from_hist(hist, 1e-3)
    assert abs(exact - approx) < 2e-5


def test_static2_sampler_has_no_spot_and_does_not_call_static5(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("генератор STATIC5 вызван из профиля STATIC2")

    for name in ("hot_spec", "neighbor_spec", "sample_train_specs", "fixed_scene", "sample_baseline", "sample_gap"):
        monkeypatch.setattr(f"dts.static_synth.{name}", boom)
    rng = np.random.default_rng(1)
    specs = static2_train_specs(rng, 1600)
    assert specs
    hots = [s for s in specs if s["role"] == "hot"]
    assert hots
    for sp in specs:
        assert sp["kind"] in ("gaussian", "raised_cosine", "broad")
        assert sp["spot_m"] == 0.0
        assert sp["fluct"] == []
        assert sp["kind"] != "segment"
    for sp in hots:
        assert S2_WIDTH[0] - 1e-6 <= sp["width_m"] <= S2_WIDTH[1] + 1e-6
        assert S2_AMP[0] - 1e-6 <= sp["amp"] <= S2_AMP[1] + 1e-6
        assert abs(sp["drift"]) <= 0.15 + 1e-9
    # опубликованный вид без spot_m не должен получить пятно 1 м
    raw = dict(center=100, width_m=4.0, amp=10.0, drift=0.0, kind="gaussian", role="hot", dx=DX)
    y = unit_profile(200, static2_view(raw))
    sigma = 4.0 / 2.355
    x = (np.arange(200) - 100) * DX
    ref = np.exp(-0.5 * (x / sigma) ** 2).astype(np.float32)
    assert np.allclose(y, ref, atol=1e-5)
    # живые константы STATIC5 на месте
    assert HOT_WIDTH == (2.0, 8.0)
    assert abs(SPOT_M - 1.0) < 1e-9


def test_static6_receptive_field_and_multiscale_means():
    assert WIN_30MIN_ROWS * DT_S == 1800
    assert WIN_2H_ROWS * DT_S == 7200
    assert receptive_field_px([(3, 1, 1)]) == 3
    assert receptive_field_px([(3, 2, 1), (3, 2, 1)]) == 1 + 4 + 4
    net = StaticNet6()
    assert receptive_field_px(net.ref.layers) * DX >= 200.0
    assert receptive_field_px(net.trunk_layers) * DX >= 200.0
    x = torch.randn(2, net.cin, 320)
    with torch.no_grad():
        ref = net.ref(x)
        logit = net(x)
    assert ref.shape == x.shape
    assert torch.allclose(ref, torch.zeros_like(ref), atol=1e-5)
    assert logit.shape == (2, 320)
    assert torch.isfinite(logit).all()

    rows, cols = 240, 400
    valid = np.ones((rows, cols), bool)
    T = np.zeros((rows, cols), np.float32)
    T[:, 80] = 10.0
    T[0, 200] = 10.0
    feats, good = temperature_features(T, valid, max_rows=rows)
    assert feats.shape[0] == N_FEAT6
    assert good[80] and good[200]
    assert float(feats[0, 80]) > 0.9
    assert float(feats[1, 80]) > 0.9 and float(feats[2, 80]) > 0.9
    assert float(feats[3, 80]) > 0.9 and float(feats[4, 80]) > 0.9
    assert float(feats[0, 200]) < 0.02
    assert float(feats[1, 200]) > 3.0 * float(feats[0, 200])
    assert float(feats[2, 200]) < 0.02

    short = np.full((10, 32), 7.0, np.float32)
    short[:, 4] = 17.0
    fs, gs = temperature_features(short, np.ones_like(short, bool), max_rows=10)
    assert gs[4]
    # окно длиннее ряда: мин и макс совпадают со средним тайла; скаляр 7 °C снят
    assert abs(float(fs[0, 4]) - 1.0) < 0.05
    assert abs(float(fs[1, 4] - fs[2, 4])) < 1e-5
    assert abs(float(fs[0, 0])) < 0.05

    ramp = np.linspace(0.0, 20.0, cols).astype(np.float32)
    Tr = np.broadcast_to(ramp, (rows, cols)).copy()
    fr, gr = temperature_features(Tr, valid, max_rows=rows)
    span = float(fr[0, gr].max() - fr[0, gr].min()) * FEAT_SCALE_C
    assert span > 15.0
    resid, _ = residual_from_temperature(Tr, valid, max_rows=rows)
    assert float(np.max(np.abs(resid[:, 80:320]))) < 0.5


def test_static6_resume_state_roundtrip(tmp_path):
    import random
    torch.manual_seed(0)
    net = StaticNet6()
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20, eta_min=2e-3 * 0.15)
    x = torch.randn(2, net.cin, 48)
    y = torch.zeros(2, 48)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(net(x), y)
    loss.backward()
    opt.step()
    sched.step()
    gen = np.random.default_rng(0)
    seen = float(gen.random())
    blob = dict(
        state={k: v.detach().clone() for k, v in net.state_dict().items()},
        optim=opt.state_dict(),
        sched=sched.state_dict(),
        step=7,
        best=dict(score=0.25, step=7),
        rng_gen=gen.bit_generator.state,
    )
    path = tmp_path / "last.pt"
    torch.save(blob, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    net2 = StaticNet6()
    opt2 = torch.optim.AdamW(net2.parameters(), lr=2e-3)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=20, eta_min=2e-3 * 0.15)
    net2.load_state_dict(loaded["state"])
    opt2.load_state_dict(loaded["optim"])
    sched2.load_state_dict(loaded["sched"])
    gen2 = np.random.default_rng(1)
    gen2.bit_generator.state = loaded["rng_gen"]
    twin = np.random.default_rng(0)
    assert float(twin.random()) == seen
    for a, b in zip(net.parameters(), net2.parameters()):
        assert torch.allclose(a, b)
    assert int(loaded["step"]) == 7
    assert abs(float(sched2.get_last_lr()[0]) - float(sched.get_last_lr()[0])) < 1e-12
    assert float(gen2.random()) == float(twin.random())
    random.seed(0)


def test_positive_mask_ignores_broad_plateau():
    specs = [
        dict(center=200, width_m=4.0, amp=12.0, drift=0.0, kind="gaussian", role="hot", dx=DX),
        dict(center=500, width_m=40.0, amp=12.0, drift=0.0, kind="broad", role="neg", dx=DX),
    ]
    R = np.zeros((10, 800), np.float32)
    _, pos, hard = apply_profiles(R, specs)
    assert 2.0 <= pos.sum() * DX <= 6.5
    assert hard.sum() * DX > 20.0
    assert not np.any(pos & hard)
