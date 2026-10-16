"""CATCC: аугментации без масштаба амплитуды, окна, контраст, бюджет меток."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dts import events, semi_contrastive as C


def test_enabled_is_bool():
    assert isinstance(C.ENABLED, bool)
    assert C.decision_enabled(0.51, 0.50, 0.40)
    assert not C.decision_enabled(0.50, 0.50, 0.40)
    assert not C.decision_enabled(0.45, 0.40, 0.45)


def test_real_paths_rejected():
    with pytest.raises(RuntimeError):
        C.assert_not_real("/workspace/mgts/MGTS_data/1.csv")
    with pytest.raises(RuntimeError):
        C.assert_not_real("/workspace/dts_solution/artifacts/eval/real_zones.csv")
    C.assert_not_real("/workspace/dts_solution/artifacts/cache/v8/train/seq0000.npz")


def test_mask_keeps_values_and_crop_of_constant_does_not_scale():
    base = torch.arange(1, 65, dtype=torch.float32).view(1, 1, 64).repeat(2, 3, 1)
    masked = C.mask_time(base, 0.4, np.random.default_rng(0))
    kept = masked[0, 0] != 0
    assert bool(kept.any()) and bool((~kept).any())
    assert torch.equal(masked[0, 0][kept], base[0, 0][kept])
    constant = torch.full((4, 3, 64), 3.0)
    views = C.make_views(constant, constant.clone(), np.random.default_rng(1))
    for view in views:
        arr = view.numpy()
        assert np.logical_or(np.isclose(arr, 0.0), np.isclose(arr, 3.0)).all()
    resized = C.crop_resize(constant[:2], np.array([4, 10]), 40)
    assert torch.allclose(resized, torch.full_like(resized, 3.0))


def test_overlapping_crops_and_deterministic_views():
    rng = np.random.default_rng(2)
    need = None
    for _ in range(25):
        (sa, la), (sb, lb) = C.paired_crop_batch(3, 64, 0.85, 0.65, 0.5, rng)
        need = int(np.ceil(0.5 * min(la, lb)))
        for i in range(3):
            assert C.overlap_len(int(sa[i]), la, int(sb[i]), lb) >= need
    series = torch.arange(1, 65, dtype=torch.float32).view(1, 1, 64).repeat(4, 3, 1)
    left = C.make_views(series, series.clone(), np.random.default_rng(4))
    right = C.make_views(series, series.clone(), np.random.default_rng(4))
    for a, b in zip(left, right):
        assert torch.equal(a, b)


def test_budget_is_stratified_and_folds_do_not_leak():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2])
    idx = C.budget_indices(y, 0.25, seed=3)
    assert set(y[idx].tolist()) == {0, 1, 2}
    assert np.array_equal(idx, C.budget_indices(y, 0.25, seed=3))
    groups = np.array([f"e{i}" for i in range(10) for _ in range(3)])
    mask = np.ones(len(groups), dtype=bool)
    mask[-3:] = False
    folds = C.episode_folds(groups, n_splits=3, mask=mask)
    seen = []
    for tr, va in folds:
        assert not (set(groups[tr]) & set(groups[va]))
        seen.append(va)
    assert set(np.concatenate(seen).tolist()) == set(np.flatnonzero(mask).tolist())


def test_pseudo_threshold_is_inclusive():
    prob = np.array([[0.90, 0.10], [0.899, 0.101], [0.20, 0.80], [0.05, 0.95]])
    pred, keep, conf = C.filter_pseudo(prob, 0.9)
    assert keep.tolist() == [True, False, False, True]
    assert pred[keep].tolist() == [0, 1]
    assert conf[0] == pytest.approx(0.9)


def test_series_keeps_sign_of_hot_and_cold():
    x = np.arange(16) * 0.25
    t = np.arange(20) * 100.0
    hot = np.zeros((3, 20, 16), np.float32)
    hot[0] = 5
    hot[2] = 1
    temporal, spatial = C.series_pair(hot, x, t, float(x[1]), float(x[-2]), float(t[1]), float(t[-2]))
    assert temporal.shape == (3, C.LENGTH) and spatial.shape == (3, C.LENGTH)
    assert temporal[0].min() > 0 and spatial[0].min() > 0
    assert C.amplitude_from_series(temporal)[0] == pytest.approx(5 / C.AMP_SCALE)
    cold = hot.copy()
    cold[0] = -4
    temporal_c, _ = C.series_pair(cold, x, t, float(x[1]), float(x[-2]), float(t[1]), float(t[-2]))
    assert temporal_c[0].max() < 0


def test_collect_synth_windows_and_lgbm_width(tmp_path):
    import pandas as pd
    cache = tmp_path / "artifacts" / "cache" / "v8" / "train"
    meta = tmp_path / "data" / "synth_meta" / "v8"
    cache.mkdir(parents=True)
    meta.mkdir(parents=True)
    t_len, w_len = 36, 24
    field = np.zeros((3, t_len, w_len), np.float32)
    field[2] = 1
    field[0, 4:30, 2:12] = 6
    field[0, 4:30, 14:22] = -5
    x = np.arange(w_len) * 0.25
    base = 1_700_000_000.0
    t = base + np.arange(t_len) * 120.0
    np.savez(cache / "seq0007.npz", X=field, x=x, t=t, sigma=np.ones(w_len, np.float32))

    def iso(sec):
        return pd.to_datetime(sec, unit="s", utc=True).strftime("%Y-%m-%d %H:%M:%S%z")

    pd.DataFrame([
        dict(seq_id="seq0007", t_start=iso(t[6]), t_end=iso(t[26]), dist_start_m=float(x[3]), dist_end_m=float(x[10]), label="hotspot"),
        dict(seq_id="seq0007", t_start=iso(t[6]), t_end=iso(t[26]), dist_start_m=float(x[15]), dist_end_m=float(x[20]), label="coldspot"),
    ]).to_csv(meta / "labels.csv", index=False)
    ds = C.collect_synth_windows(tmp_path, versions=("v8",), splits=("train",))
    assert ds["temporal"].shape == (2, 3, C.LENGTH)
    assert ds["spatial"].shape[0] == 2
    assert ds["img"].shape[1:] == (3,) + events.CROP
    assert ds["scal"].shape[1] == len(events.SCALAR_NAMES)
    assert set(ds["y"].tolist()) == {events.CLASSES.index("Local_Heating"), events.CLASSES.index("Local_Cooling")}
    assert ds["groups"].tolist() == ["seq0007", "seq0007"]
    hot = ds["y"] == events.CLASSES.index("Local_Heating")
    assert ds["amp"][hot, 0].mean() > ds["amp"][~hot, 0].mean()
    feat = C.lgbm_feature_matrix(ds["img"], ds["scal"], ds["ctx"])
    assert feat.shape == (2, len(events.SCALAR_NAMES) + events.IMAGE_BANK_DIM + len(events.SHAPE_NAMES) + len(events.CONTEXT_NAMES))
    path = tmp_path / "w.npz"
    C.save_windows(path, ds)
    back = C.load_windows(path)
    assert back["y"].tolist() == ds["y"].tolist()
    assert np.allclose(back["temporal"], ds["temporal"], atol=2e-3)


def test_ssl_backward_finetune_and_baselines():
    C.set_seed(0)
    model = C.CATCC(width=32)
    other = C.CATCC(width=32)
    C.set_seed(0)
    again = C.CATCC(width=32)
    data = torch.randn(8, 3, 64)
    space = torch.randn(8, 3, 64)
    model.eval()
    again.eval()
    assert torch.allclose(model.embed(data, space), again.embed(data, space))
    other.eval()
    # другая инициализация (seed уже сдвинут) не обязана совпасть; форма обязана
    assert other.embed(data[:2], space[:2]).shape == (2, other.embed_dim)
    model.train()
    tw, sw, ts, ss = C.make_views(data, space, np.random.default_rng(0))
    loss, _, _ = model.ssl_loss(tw, sw, ts, ss)
    loss.backward()
    assert torch.isfinite(loss).item()

    temporal = data.detach().numpy().copy()
    spatial = space.detach().numpy().copy()
    y = np.arange(8) % 4
    temporal[:, 0, :] += y[:, None]
    amp = np.stack([
        temporal[:, 0].mean(1), temporal[:, 0].max(1), temporal[:, 0].min(1), np.abs(temporal[:, 0]).max(1)
    ], 1).astype(np.float32) / C.AMP_SCALE
    C.pretrain_encoder(model, temporal, spatial, np.arange(8), epochs=1, batch_size=8, seed=0)
    head = C.Head(model.embed_dim, C.AMP_DIM, 7)
    prob, stats = C.finetune_predict(
        model, head, temporal, spatial, amp, y,
        np.array([0, 1, 2, 3]), np.array([4, 5]), np.array([6, 7]),
        epochs_sup=2, epochs_pseudo=1, batch_size=4, seed=0, n_classes=7)
    assert prob.shape == (2, 7)
    assert np.isfinite(prob).all()
    assert np.allclose(prob.sum(1), 1, atol=1e-4)
    assert stats["threshold"] == 0.9
    assert stats["n_pseudo"] <= stats["n_unlabeled"]

    rng = np.random.default_rng(0)
    n, k = 18, 4
    img = rng.normal(size=(n, 3, 32, 64)).astype(np.float32)
    scal = rng.normal(size=(n, len(events.SCALAR_NAMES))).astype(np.float32)
    ctx = rng.normal(size=(n, len(events.CONTEXT_NAMES))).astype(np.float32)
    labels = np.arange(n) % k
    feat = C.lgbm_feature_matrix(img, scal, ctx)
    labeled, predict = np.arange(12), np.arange(12, 18)
    p_net = C.train_eventnet_classifier(img, scal, labels, labeled, labeled, predict, epochs=1, seed=0, n_classes=k)
    p_lgb = C.train_lgbm_classifier(feat, labels, labeled, predict, seed=0, n_classes=k, n_estimators=8)
    assert p_net.shape == (6, k) and p_lgb.shape == (6, k)
    assert np.allclose(p_net.sum(1), 1, atol=1e-4)
    assert np.allclose(p_lgb.sum(1), 1, atol=1e-4)
    scored = C.score_probs(labels[predict], p_lgb, class_names=[f"c{i}" for i in range(k)])
    assert 0.0 <= scored["macro_f1"] <= 1.0
    assert 0.0 <= scored["reject_rate"] <= 1.0


def test_pseudo_used_only_if_fraction_reaches_min():
    weak = np.full((10, 3), 0.05)
    # Ниже порога: только строка 0 дотягивает до conf ≥ 0.9, доля 0.1 < min_frac.
    weak[:, 1] = 0.4
    weak[0, 0] = 0.95
    weak[0, 1] = 0.02
    _pred, keep, _conf, frac, used = C.resolve_pseudo(weak, 0.9, 0.2)
    assert frac == pytest.approx(0.1)
    assert used is False
    assert not keep.any()
    strong = weak.copy()
    strong[:3, :] = 0.02
    strong[:3, 0] = 0.95
    pred, keep, _conf, frac, used = C.resolve_pseudo(strong, 0.9, 0.2)
    assert frac == pytest.approx(0.3)
    assert used is True
    assert int(keep.sum()) == 3
    assert np.array_equal(pred[keep], np.zeros(3, np.int64))
    edge = np.full((5, 2), 0.1)
    edge[:, 0] = 0.9
    _pred, keep, _conf, frac, used = C.resolve_pseudo(edge, 0.9, 0.2)
    assert frac == pytest.approx(1.0) and used and keep.all()


def test_collapse_stats_and_linear_probe():
    dead = np.zeros((24, 8), np.float32)
    stats = C.embedding_collapse_stats(dead)
    assert stats["collapsed"] and stats["mean_std"] == 0.0
    rng = np.random.default_rng(0)
    live = rng.normal(size=(24, 8))
    assert not C.embedding_collapse_stats(live)["collapsed"]
    a = rng.normal(0.0, 0.15, size=(40, 4))
    b = rng.normal(3.0, 0.15, size=(40, 4))
    features = np.vstack([a, b]).astype(np.float32)
    labels = np.array([0] * 40 + [1] * 40)
    labeled = np.concatenate([np.arange(0, 40, 4), np.arange(40, 80, 4)])
    unlabeled = np.setdiff1d(np.arange(80), labeled)
    prob, head = C.fit_frozen_head(
        "linear", features, labels, labeled, unlabeled, np.arange(80),
        seed=0, n_classes=2, standardize=True)
    assert prob.shape == (80, 2)
    assert np.isfinite(prob).all()
    assert np.allclose(prob.sum(1), 1, atol=1e-4)
    assert head["kind"] == "linear"
