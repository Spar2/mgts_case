"""Период на синусоиде, пассивная релаксация и флаг runtime у этапа 2."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dts import runtime as R


def _hvac(period, duty, dt, n_cycles, heat=True, seed=0):
    """Кусочная релаксация: включено — к T_on, выключено — к T_off."""
    t = np.arange(0.0, n_cycles * period, dt)
    ton, toff = (26.0, 20.0) if heat else (16.0, 22.0)
    k_on, k_off = 1.0 / 400.0, 1.0 / 1400.0
    T = np.empty(len(t), np.float64)
    T[0] = toff
    on = (t % period) < duty * period
    for i in range(1, len(t)):
        dti = t[i] - t[i - 1]
        if on[i]:
            T[i] = T[i - 1] + (-k_on * (T[i - 1] - ton)) * dti
        else:
            T[i] = T[i - 1] + (-k_off * (T[i - 1] - toff)) * dti
    T += np.random.default_rng(seed).normal(0.0, 0.01, len(t))
    return t, T, on


def test_sine_period_found():
    for period, dt in ((3600.0, 60.0), (900.0, 15.0)):
        t = np.arange(0.0, 8.0 * period, dt)
        series = 18.5 + 1.7 * np.sin(2.0 * np.pi * t / period)
        feat = R.runtime_features(series, t)
        assert feat["period_s"] == pytest.approx(period, rel=0.1), feat


def test_sine_period_is_deterministic_and_finite():
    period = 2400.0
    t = np.arange(0.0, 6.0 * period, 40.0)
    series = np.sin(2.0 * np.pi * t / period)
    dirty_s = series.copy()
    dirty_t = t.copy()
    dirty_s[10] = np.nan
    dirty_t = dirty_t[::-1].copy()
    dirty_s = dirty_s[::-1].copy()
    feat = R.runtime_features(series, t)
    feat2 = R.runtime_features(series, t)
    feat3 = R.runtime_features(dirty_s, dirty_t)
    assert feat == feat2
    assert feat3["period_s"] == pytest.approx(period, rel=0.1)
    for key in ("active_frac", "period_s", "duty", "k", "T_inf", "resid_mad", "active_sign"):
        assert np.isfinite(feat[key])


def test_passive_decay_is_mostly_inactive_and_k_matches():
    k_true = 1.0 / 1800.0
    t_inf = 21.0
    t = np.arange(0.0, 8 * 3600.0, 30.0)
    series = t_inf + 5.0 * np.exp(-k_true * t)
    series = series + np.random.default_rng(2).normal(0.0, 0.005, len(t))
    feat = R.runtime_features(series, t)
    assert feat["active_frac"] < 0.15
    assert feat["k"] == pytest.approx(k_true, rel=0.35)
    assert feat["T_inf"] == pytest.approx(t_inf, abs=0.5)


def test_hvac_cycles_period_duty_and_sign():
    period, duty = 3600.0, 0.25
    t, series, on = _hvac(period, duty, 30.0, n_cycles=8, heat=True, seed=1)
    feat = R.runtime_features(series, t)
    assert feat["period_s"] == pytest.approx(period, rel=0.1), feat
    assert feat["active_sign"] > 0
    assert feat["active_frac"] > 0.05
    assert feat["duty"] == pytest.approx(float(on.mean()), abs=0.15)
    t_c, series_c, _ = _hvac(period, duty, 30.0, n_cycles=8, heat=False, seed=3)
    cool = R.runtime_features(series_c, t_c)
    assert cool["period_s"] == pytest.approx(period, rel=0.1)
    assert cool["active_sign"] < 0


def test_short_and_constant_series_are_finite():
    feat = R.runtime_features([20.0, 20.1, 20.0], [0.0, 10.0, 20.0])
    assert feat["period_s"] == 0.0
    assert all(np.isfinite(feat[k]) for k in ("k", "T_inf", "active_frac"))
    flat = R.runtime_features(np.full(40, 19.0), np.arange(40) * 60.0)
    assert flat["active_frac"] == 0.0
    assert flat["period_s"] == 0.0


def test_stack_scalars_flag_off_is_the_same_object():
    scal = np.arange(12, dtype=np.float32).reshape(3, 4)
    extra = np.ones((3, len(R.FEATURE_NAMES)), np.float32)
    assert R.stack_scalars(scal, extra, enabled=False) is scal
    wide = R.stack_scalars(scal, extra, enabled=True)
    assert wide.shape == (3, 4 + len(R.FEATURE_NAMES))
    assert np.allclose(wide[:, :4], scal)
    vec = R.feature_vector(
        {"active_frac": 0.2, "period_s": 3600.0, "duty": 0.25, "k": 1.0 / 3600.0,
         "T_inf": 21.0, "resid_mad": 0.01, "active_sign": 1.0}
    )
    assert vec.shape == (len(R.FEATURE_NAMES),)
    assert vec[1] == pytest.approx(np.log1p(60.0))
    assert vec[3] == pytest.approx(1.0)


def test_runtime_net_flag_off_matches_eventnet_width():
    torch = pytest.importorskip("torch")
    from dts.models import EventNet

    torch.manual_seed(0)
    base = EventNet(7, 19)
    net = R.make_runtime_net(7, 19, n_runtime=len(R.FEATURE_NAMES), use_runtime=False, use_series=False)
    net.backbone.load_state_dict(base.state_dict())
    base.eval()
    net.eval()
    img = torch.randn(2, 3, 32, 64)
    scal = torch.randn(2, 19)
    with torch.no_grad():
        assert torch.allclose(base(img, scal), net(img, scal))
    wide = R.make_runtime_net(7, 19, n_runtime=len(R.FEATURE_NAMES), use_runtime=True, use_series=True)
    assert wide.n_in == 19 + len(R.FEATURE_NAMES)
    series = torch.randn(2, R.SERIES_LEN)
    scal_w = torch.randn(2, wide.n_in)
    with torch.no_grad():
        out = wide(img, scal_w, series)
    assert out.shape == (2, 7)
