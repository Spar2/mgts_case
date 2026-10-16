import numpy as np
import pytest

from fiber.denoise import apply_method, available_methods, residual
from fiber.metrics import residual_stats, spike_retention
from fiber.quality import quality_mask


def _spike_field():
    rng = np.random.default_rng(0)
    t, x = 12, 64
    base = 22.0 + 0.3 * np.sin(np.linspace(0, 4 * np.pi, x))
    field = np.broadcast_to(base, (t, x)).copy()
    field += rng.normal(0, 0.15, size=field.shape)
    field[:, 30:33] += 6.0
    field[3, 50] = -300.0
    valid = quality_mask(field)["valid"]
    return field.astype(np.float32), valid


def test_methods_registered():
    names = available_methods()
    for required in ("median_x3", "wavelet_x", "wavelet_bayes", "nlm_x", "nlm_wide", "wiener_x", "bilateral_x", "tv_x", "median_xt"):
        assert required in names


def test_invalid_stays_nan():
    raw, valid = _spike_field()
    for name in ("median_x3", "savgol_x", "gaussian_x"):
        out = apply_method(name, raw, valid)
        assert np.isnan(out[~valid]).all()
        assert np.isfinite(out[valid]).all()


def test_median_reduces_noise_keeps_spike():
    raw, valid = _spike_field()
    out = apply_method("median_x3", raw, valid)
    noise = residual(raw, out, valid)
    stats = residual_stats(noise, valid)
    assert stats["mad"] > 0.02
    row = 5
    keep = spike_retention(
        np.where(valid[row], raw[row], np.nan),
        np.where(valid[row], out[row], np.nan),
        peak_index=31,
        half_width=2,
    )
    assert keep > 0.7


def test_bayes_and_nlm_keep_most_of_the_spike():
    raw, valid = _spike_field()
    for name in ("wavelet_bayes", "nlm_x"):
        out = apply_method(name, raw, valid)
        keep = spike_retention(
            np.where(valid[5], raw[5], np.nan),
            np.where(valid[5], out[5], np.nan),
            peak_index=31,
            half_width=2,
        )
        assert keep > 0.6
        assert np.isnan(out[~valid]).all()


def test_identity_zero_residual():
    raw, valid = _spike_field()
    out = apply_method("identity", raw, valid)
    noise = residual(raw, out, valid)
    assert np.nanmax(np.abs(noise)) == 0


def test_checkerboard_hides_each_pixel_once():
    from fiber.models.unet2d_denoise import _checkerboard_inputs

    _inputs, keep = _checkerboard_inputs(np.zeros((8, 10), dtype=np.float32))
    assert keep.shape == (4, 8, 10)
    assert np.all(keep.sum(axis=0) == 1)


def test_unet_predicts_absolute_temperature():
    torch = pytest.importorskip("torch")
    from fiber.models.unet2d_denoise import UNet2DDenoiser

    net = UNet2DDenoiser()
    out = net(torch.zeros(2, 1, 32, 128))
    assert tuple(out.shape) == (2, 1, 32, 128)


@pytest.mark.parametrize("name", ["wavelet_x", "wavelet_bayes", "nlm_x", "nlm_wide", "wiener_x", "bilateral_x", "tv_x", "median_xt"])
def test_modern_methods_run(name):
    raw, valid = _spike_field()
    out = apply_method(name, raw, valid)
    assert out.shape == raw.shape
    assert np.isfinite(out[valid]).all()
