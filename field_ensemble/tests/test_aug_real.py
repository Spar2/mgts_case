"""Спектральная аугментация фона: детерминизм, АКФ, неизменность маски."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dts import aug_real, features


def _shaper(phi_t=0.62, phi_x=0.45):
    bank = aug_real.synthetic_bank(phi_t, phi_x)
    net, info = aug_real.fit_shaper(bank, steps=0, seed=0, tile=(96, 96), max_lag=12)
    return bank, net, info


def test_embed_matches_ar1_and_is_deterministic():
    bank, net, info = _shaper()
    assert info["accepted_fit"] is False
    rng = np.random.default_rng(0)
    a = net.sample(rng, (512, 320), 0, std=1.0)
    b = net.sample(np.random.default_rng(0), (512, 320), 0, std=1.0)
    assert np.allclose(a, b)
    assert abs(float(a.std()) - 1.0) < 1e-4
    m = aug_real.acf_match(a, bank.acf_t[0], bank.acf_x[0], max_lag=20)
    assert abs(m["phi_t"] - 0.62) < 0.05
    assert abs(m["phi_x"] - 0.45) < 0.05
    assert m["corr_t"] > 0.95
    assert m["corr_x"] > 0.95
    # второй источник банка слабее по φ
    c = net.sample(np.random.default_rng(1), (512, 320), 1, std=1.0)
    m2 = aug_real.acf_match(c, bank.acf_t[1], bank.acf_x[1], max_lag=16)
    assert abs(m2["phi_t"] - 0.31) < 0.05


def test_zero_std_and_p_apply_keep_tile_sign_flips_labels():
    _, net, _ = _shaper()
    rng = np.random.default_rng(4)
    T = np.zeros((36, 40), np.float32)
    T[8:18, 10:22] = 2.5
    valid = np.ones(T.shape, bool)
    ch, aux = features.make_channels(T, valid)
    y = np.zeros(T.shape, np.int64)
    y[8:18, 10:22] = 1
    y[20:28, 4:10] = 2
    y[0, 0] = -1
    spec = aug_real.AugSpec(0.0, 0.0, 1.0, 0.0)
    out, yo = aug_real.apply_tile(ch, y, aux["sigma_x"], rng, net, spec)
    assert np.allclose(out, ch)
    assert np.array_equal(yo, y)
    spec = aug_real.AugSpec(0.4, 0.4, 0.0, 0.0)
    out, yo = aug_real.apply_tile(ch, y, aux["sigma_x"], np.random.default_rng(5), net, spec)
    assert np.allclose(out, ch)
    spec = aug_real.AugSpec(0.0, 0.0, 1.0, 1.0)
    out, yo = aug_real.apply_tile(ch, y, aux["sigma_x"], np.random.default_rng(6), net, spec)
    assert np.allclose(out[0], -ch[0]) and np.allclose(out[1], -ch[1])
    assert np.allclose(out[2], ch[2])
    assert yo[10, 12] == 2 and yo[22, 6] == 1 and yo[0, 0] == -1


def test_noise_respects_valid_and_changes_channels():
    _, net, _ = _shaper(0.5, 0.5)
    rng = np.random.default_rng(7)
    T = np.full((48, 64), 18.0, np.float32)
    valid = np.ones(T.shape, bool)
    valid[:, :3] = False
    ch, aux = features.make_channels(T, valid)
    y = np.zeros(T.shape, np.int64)
    y[:, :3] = -1
    spec = aug_real.AugSpec(0.8, 0.8, 1.0, 0.0)
    out, yo = aug_real.apply_tile(ch, y, aux["sigma_x"], rng, net, spec)
    assert np.array_equal(yo, y)
    assert np.allclose(out[2], ch[2])
    assert np.all(out[0, :, :3] == 0) and np.all(out[1, :, :3] == 0)
    assert float(np.std(out[0, :, 3:])) > float(np.std(ch[0, :, 3:])) + 0.05
    assert np.isfinite(out).all()


def test_real_psd_has_long_memory_not_ar1():
    bank = aug_real.load_bank()
    assert bank.names[-1] == "pooled"
    phi_t, phi_x = bank.phi(-1)
    assert abs(phi_t - 0.3200639082834765) < 1e-9
    assert abs(phi_x - 0.5563940358344389) < 1e-9
    # лаг 8 у измеренной АКФ далеко от AR(1) с тем же φ
    ar_lag8 = phi_t ** 8
    assert bank.acf_t[-1, 8] > 0.15
    assert bank.acf_t[-1, 8] > ar_lag8 * 50
    net, info = aug_real.fit_shaper(bank, steps=0, seed=1, tile=(128, 192), max_lag=16)
    field = net.sample(np.random.default_rng(2), (640, 400), bank.n_src - 1, std=1.0)
    m = aug_real.acf_match(field, bank.acf_t[-1], bank.acf_x[-1], max_lag=24)
    assert m["corr_t"] > 0.9
    assert m["corr_x"] > 0.9
    assert m["lag8_t"] > 0.12
    assert abs(m["phi_t"] - phi_t) < 0.06
    assert abs(m["phi_x"] - phi_x) < 0.06
    assert info["base_acf_corr"] > 0.85


def test_fit_does_not_read_real_traces_and_stays_finite():
    bank = aug_real.synthetic_bank(0.7, 0.35, n_lag=20)
    net, info = aug_real.fit_shaper(bank, steps=4, seed=3, tile=(80, 96), max_lag=8)
    field = net.sample(np.random.default_rng(0), (200, 180), 0, std=0.5)
    assert np.isfinite(field).all()
    assert abs(float(field.std()) - 0.5) < 1e-3
    assert "accepted_fit" in info
    gap = aug_real.spectrum_gap(field, np.ones(field.shape, bool), bank, src=0, max_lag=8)
    assert gap["corr_t"] > 0.8
