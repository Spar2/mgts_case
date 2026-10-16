import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dts import grid, features, metrics, events


def test_grid_mean_downsample_and_gap():
    x = np.arange(0, 10, 0.1); t = np.r_[np.arange(0, 1200, 30.0), np.arange(3000, 4200, 30.0)]
    a = np.ones((len(t), len(x)), np.float32); v = np.ones_like(a, bool)
    p = grid.plan(x, t)
    o, w = grid.resample(a, v, p, "mean")
    assert o.shape == (len(p["t"]), len(p["x"]))
    assert np.allclose(o[w], 1)
    gap_rows = (p["t"] > 1500) & (p["t"] < 2700)
    assert not w[gap_rows].any()          # разрыв > 300 c не заполняется


def test_grid_nearest_labels_upsample():
    x = np.arange(0, 20, 1.0); t = np.arange(0, 1200, 120.0)
    lab = np.zeros((len(t), len(x)), np.uint8); lab[:, 5] = 2
    o, _ = grid.resample(lab, np.ones_like(lab, bool), grid.plan(x, t), "nearest")
    assert set(np.unique(o)) == {0, 2}


def test_channels_remove_common_mode_and_static():
    rng = np.random.default_rng(0)
    T = 20 + rng.normal(0, .2, (200, 400)).astype(np.float32)
    T += np.linspace(0, 3, 200)[:, None]          # синфазный дрейф
    T += np.linspace(0, 5, 400)[None, :]          # статический профиль
    ch, aux = features.make_channels(T, np.ones_like(T, bool))
    assert ch.shape == (3, 200, 400)
    assert abs(np.median(ch[0])) < 0.2 and np.percentile(np.abs(ch[0]), 99) < 4.5


def test_hot_spot_visible_in_channels():
    rng = np.random.default_rng(1)
    T = 20 + rng.normal(0, .2, (200, 400)).astype(np.float32); T[80:140, 200:220] += 2.0
    ch, _ = features.make_channels(T, np.ones_like(T, bool))
    assert ch[0, 100, 210] > 5 and ch[1, 100, 210] > 5


def test_far_threshold():
    s = np.random.default_rng(0).normal(size=100000)
    th = metrics.far_threshold(s, 1e-2)
    assert abs((s >= th).mean() - 1e-2) < 2e-3


def test_crop_features_shape_and_sign():
    rng = np.random.default_rng(2)
    T = 20 + rng.normal(0, .2, (200, 400)).astype(np.float32); T[80:140, 200:220] -= 3
    ch, aux = features.make_channels(T, np.ones_like(T, bool))
    x = np.arange(400) * .25; t = np.arange(200) * 120.0
    img, f = events.crop_and_features(ch, aux["sigma_x"], x, t, 50, 55, 80 * 120, 140 * 120)
    assert img.shape == (3,) + events.CROP
    assert f[events.SCALAR_NAMES.index("sign")] == -1
    assert len(f) == len(events.SCALAR_NAMES)


def test_zones_from_prob():
    s = np.zeros((50, 60)); s[10:20, 10:30] = .9; h = np.ones_like(s)
    z = events.zones_from_prob(s, h, np.ones_like(s, bool), np.arange(60) * .25, np.arange(50) * 120., .5)
    assert len(z) == 1 and z[0]["polarity"] == "hot"


def test_unet_shapes_and_determinism():
    torch = pytest.importorskip("torch")
    from dts.models import DTSUNet, predict_full
    torch.manual_seed(0); m = DTSUNet(w=8)
    X = np.random.default_rng(0).normal(size=(3, 70, 150)).astype(np.float32)
    p1 = predict_full(m, X, tile=(64, 128), overlap=8); p2 = predict_full(m, X, tile=(64, 128), overlap=8)
    assert p1.shape == (3, 70, 150) and np.allclose(p1.sum(0), 1, atol=1e-4) and np.array_equal(p1, p2)
