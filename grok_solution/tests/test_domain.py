"""Пофайловый порог, АКФ и пересечение карточки. Без МГТС и без весов."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dts import domain, events


def test_flag_threshold_hits_target_rate():
    rng = np.random.default_rng(0)
    score = rng.random((200, 300)).astype(np.float32)
    valid = np.ones(score.shape, bool)
    valid[:, :10] = False
    for far in (1e-3, 3e-3, 1e-2):
        thr = domain.flag_threshold(score, valid, far)
        got = domain.flag_fraction(score, valid, thr)
        assert abs(got - far) < 0.002


def test_acf_ar1_lag1():
    rng = np.random.default_rng(1)
    phi = 0.8
    x = np.zeros(4000)
    x[0] = rng.normal()
    for i in range(1, len(x)):
        x[i] = phi * x[i - 1] + rng.normal() * np.sqrt(1 - phi ** 2)
    a = domain.acf_1d(x, 5)
    assert abs(a[0] - 1) < 1e-6
    assert abs(a[1] - phi) < 0.05


def test_mean_acf_skips_short_gaps():
    field = np.random.default_rng(2).normal(size=(80, 4))
    valid = np.ones(field.shape, bool)
    valid[30:50, :] = False
    ac = domain.mean_acf(field, valid, axis=0, max_lag=4, n_traces=4, rng=np.random.default_rng(0), min_len=20)
    assert ac.shape == (5,) and np.isfinite(ac[0])


def test_zone_overlaps_card_box():
    zones = [dict(start_m=10, end_m=12, t_start_s=100, t_end_s=200, alarm=True, n_px=5, class_="Local_Heating")]
    zones[0]["class"] = "Local_Heating"
    assert domain.overlapping_zones(zones, 11, 15, 150, 180, True)
    assert not domain.overlapping_zones(zones, 20, 22, 150, 180, True)
    zones[0]["alarm"] = False
    assert not domain.overlapping_zones(zones, 11, 15, 150, 180, True)
    assert domain.overlapping_zones(zones, 11, 15, 150, 180, False)


def test_presence_tally_ignores_unclear():
    rows = [
        dict(presence="yes", status="ok", hit_alarm=True, hit_base=True),
        dict(presence="yes", status="ok", hit_alarm=False, hit_base=True),
        dict(presence="no", status="ok", hit_alarm=False, hit_base=False),
        dict(presence="unclear", status="ok", hit_alarm=True, hit_base=True),
        dict(presence="yes", status="box_outside_record", hit_alarm=False, hit_base=False),
    ]
    t = domain.presence_tally(rows, "hit_alarm")
    assert t["yes_hit_alarm"] == 1 and t["yes_n"] == 2
    assert t["no_hit_alarm"] == 0 and t["no_n"] == 1


def test_rethreshold_changes_zone_count():
    score = np.zeros((40, 30), np.float32)
    score[10:18, 8:14] = 0.9
    hot = np.ones_like(score)
    valid = np.ones(score.shape, bool)
    x = np.arange(30) * 0.25
    t = np.arange(40) * 120.0
    dT = np.ones_like(score)
    X = np.zeros((3, 40, 30), np.float32)
    sigma = np.ones(30, np.float32)

    class Fake:
        def classify(self, X, sigma, x, t, zones):
            return [np.full(len(events.CLASSES), 1.0 / len(events.CLASSES)) for _ in zones]

    out = dict(grid=dict(x=x, t=t), X=X, valid=valid, score=score, hotness=hot, sigma=sigma, dT=dT)
    hi = domain.rethreshold(Fake(), out, 0.5)
    lo = domain.rethreshold(Fake(), out, 0.99)
    assert len(hi["zones"]) == 1 and hi["zones"][0]["alarm"]
    assert len(lo["zones"]) == 0


def test_sigma_scale_and_recipe_finite():
    real = np.array([0.2, 0.4, 0.4])
    synth = np.array([0.1, 0.1, 0.2])
    assert abs(domain.sigma_scale_factor(real, synth) - 4.0) < 1e-9
    recipe = domain.augmentation_recipe(
        dict(sigma_p50=0.4, ch0_mad=1.4, acf_t_lag1=0.6, acf_x_lag1=0.85),
        dict(sigma_p50=0.2, ch0_mad=1.0, acf_t_lag1=0.1, acf_x_lag1=0.4),
    )
    assert recipe["physical_sigma_ratio_real_over_synth"] == 2.0
    assert recipe["extra_channel_noise_std"] > 0
    assert "AR(1)" in recipe["apply_before_make_channels"]
