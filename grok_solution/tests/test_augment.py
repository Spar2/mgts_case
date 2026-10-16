"""Цветной шум в T до make_channels и симметрия hot/cold."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dts import augment, data, features


def _lag1(a, axis):
    if axis == 0:
        x, y = a[:-1].ravel(), a[1:].ravel()
    else:
        x, y = a[:, :-1].ravel(), a[:, 1:].ravel()
    x = x - x.mean()
    y = y - y.mean()
    return float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum()))


def test_colored_noise_std_and_correlation():
    rng = np.random.default_rng(0)
    valid = np.ones((400, 280), bool)
    valid[:, -5:] = False
    zt = augment.colored_noise(rng, valid.shape, 0.62, 0.0, 0.28, valid)
    zx = augment.colored_noise(np.random.default_rng(1), valid.shape, 0.0, 0.71, 0.28, valid)
    assert abs(float(zt[valid].std()) - 0.28) < 1e-4
    assert abs(float(zx[valid].std()) - 0.28) < 1e-4
    assert np.all(zt[:, -5:] == 0) and np.all(zx[:, -5:] == 0)
    assert 0.52 < _lag1(zt[:, :-5], 0) < 0.72
    assert abs(_lag1(zt[:, :-5], 1)) < 0.08
    assert 0.62 < _lag1(zx[:, :-5], 1) < 0.80
    assert abs(_lag1(zx[:, :-5], 0)) < 0.08


def test_amplitude_jitters_around_sigma():
    rng = np.random.default_rng(2)
    draws = [augment.draw_amplitude(rng, 0.283, 0.25) for _ in range(40)]
    assert min(draws) >= 0.283 * 0.75 - 1e-9
    assert max(draws) <= 0.283 * 1.25 + 1e-9
    assert min(draws) < 0.283 < max(draws)
    assert augment.draw_amplitude(rng, 0.283, 0.0) == 0.283
    assert augment.draw_amplitude(rng, 0.0, 0.25) == 0.0


def test_noise_is_applied_before_make_channels():
    rng = np.random.default_rng(3)
    T = rng.normal(15.0, 0.2, (48, 64)).astype(np.float32)
    valid = np.ones(T.shape, bool)
    valid[:, :2] = False
    T[:, :2] = -300
    ch, Tn, amp = augment.featurize_noisy(T, valid, rng, phi_t=0.36, phi_x=0.74, sigma_c=0.2, jitter=0.0)
    assert amp == 0.2
    ref, _ = features.make_channels(Tn, valid)
    assert np.allclose(ch, ref, atol=1e-5)
    # синфазный по x шум почти полностью снимается признаками, а не остаётся в канале
    common = np.linspace(-0.4, 0.5, T.shape[0], dtype=np.float32)[:, None] * np.ones((1, T.shape[1]), np.float32)
    delta = augment.feature_delta(np.zeros_like(T), common, valid)
    assert float(np.max(np.abs(delta[0]))) < 0.05
    raw = common / 0.3
    assert float(np.max(np.abs(raw))) > 1.0


def test_zero_noise_keeps_cached_tile_and_sign_flips_labels():
    rng = np.random.default_rng(4)
    T = np.zeros((36, 40), np.float32)
    T[8:18, 10:22] = 3.0
    valid = np.ones(T.shape, bool)
    ch, aux = features.make_channels(T, valid)
    y = np.zeros(T.shape, np.int64)
    y[8:18, 10:22] = 1
    y[0, 0] = -1
    y[20:28, 4:10] = 2
    spec = augment.Spec(0.0, 0.36, 0.74, 0.25, 0.0)
    out, yo = augment.apply_tile(ch, y, aux["sigma_x"], rng, spec)
    assert np.allclose(out, ch)
    assert np.array_equal(yo, y)
    spec.sign_p = 1
    out, yo = augment.apply_tile(ch, y, aux["sigma_x"], np.random.default_rng(5), spec)
    assert np.allclose(out[0], -ch[0]) and np.allclose(out[1], -ch[1])
    assert np.allclose(out[2], ch[2])
    assert yo[10, 12] == 2 and yo[24, 6] == 1 and yo[0, 0] == -1 and yo[2, 2] == 0
    Tinv = augment.invert_around_background(T, valid)
    assert Tinv[12, 16] < 0
    assert abs(float(Tinv[2, 2])) < 0.2


def test_full_tile_delta_matches_make_channels():
    """Приращение тайла = make_channels(R+шум) − make_channels(R), не прибавка шума к каналу."""
    rng = np.random.default_rng(6)
    T = rng.normal(0.0, 0.05, (40, 48)).astype(np.float32)
    valid = np.ones(T.shape, bool)
    ch, aux = features.make_channels(T, valid)
    R = ch[0] * aux["sigma_x"][None, :]
    spec = augment.Spec(0.12, 0.45, 0.55, 0.0, 0.0)
    y = np.zeros(T.shape, np.int64)
    out, _ = augment.apply_tile(ch, y, aux["sigma_x"], np.random.default_rng(7), spec)
    noise = augment.colored_noise(np.random.default_rng(7), R.shape, spec.phi_t, spec.phi_x, spec.sigma_c, valid)
    delta = augment.feature_delta(R, noise, valid)
    expect = ch.copy()
    expect[0] = np.clip(ch[0] + delta[0], -features.CLIP, features.CLIP)
    expect[1] = np.clip(ch[1] + delta[1], -features.CLIP, features.CLIP)
    assert np.allclose(out, expect, atol=1e-5)
    added = ch.copy()
    added[0] = ch[0] + noise / aux["sigma_x"][None, :]
    # общий уровень шума, который make_channels снимает, не должен совпасть с сырой прибавкой
    assert float(np.mean(np.abs(out[0] - ch[0]))) + 1e-3 < float(np.mean(np.abs(added[0] - ch[0])))


def test_lit_recipe_and_train_defaults():
    text = (
        "| real, все файлы | 10687672 | 0.2830 | 0.4213 |\n"
        "Рецепт noise-matched аугментации для агента ae:\n"
        "окрасить поле AR(1): φ_t = 0.357 на шаге 120 с и φ_x = 0.736 на шаге 0.25 м.\n"
    )
    assert augment.lit_temperature_recipe(text) == {"sigma_c": 0.283, "phi_t": 0.357, "phi_x": 0.736}
    assert augment.lit_temperature_recipe("нет рецепта") is None
    live = Path("/workspace/dts_solution/docs/EXPERIMENTS_LATE.md").read_text(encoding="utf-8")
    got = augment.lit_temperature_recipe(live)
    assert got == {"sigma_c": 0.283, "phi_t": 0.357, "phi_x": 0.736}
    src = (Path(__file__).resolve().parents[1] / "scripts" / "train_stage1.py").read_text()
    assert 'ap.add_argument("--temp-noise", type=float, default=0.0' in src
    assert 'ap.add_argument("--sign-flip", type=float, default=0.0' in src
    assert "if a.temp_noise != 0 or a.sign_flip > 0:" in src
    assert "return ce + dice" in src


def test_sampler_calls_aug_before_flip():
    y = np.zeros((24, 28), np.int8)
    y[6:14, 6:16] = 1
    X = np.zeros((3, 24, 28), np.float32)
    X[0, 6:14, 6:16] = 5
    X[2] = 1
    sigma = np.full(28, 0.25, np.float32)
    it = dict(X=X, y=y, sigma=sigma)
    spec = augment.Spec(0.0, 0.36, 0.74, 0.0, 1.0)

    def aug(xt, yt, item, t0, w0, spec=spec):
        col = item["sigma"][w0:w0 + xt.shape[-1]]
        return augment.apply_tile(xt, yt, col, np.random.default_rng(0), spec)

    S = data.TileSampler([it], tile=(24, 28), seed=1, p_event=0.0, aug=aug)
    xs, ys = S.sample(6)
    assert (ys == 2).any() and not (ys == 1).any()
    assert float(xs[:, 0].min()) < 0
    plain = data.TileSampler([it], tile=(24, 28), seed=1, p_event=0.0)
    xp, yp = plain.sample(6)
    assert (yp == 1).any() and float(xp[:, 0].max()) > 0
