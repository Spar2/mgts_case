"""Threshold calibration uses the shipped functions, not a second implementation."""

import numpy as np
import pytest

from fiber.anomaly.detect import spatial_z
from fiber.benchmark.synthetic import local_scene_var
from fiber.benchmark.threshold_cal import (
    FAIL_REAL,
    FAIL_WEAK,
    SCALES,
    assert_disjoint,
    dead_alarms,
    decide_channel,
    denoise_field,
    far,
    far_ratio,
    fit_median_profile,
    iou_curve,
    parse_seed_spec,
    threshold_for_far,
    z_field,
)


def test_gaussian_threshold_stays_near_the_requested_far():
    rng = np.random.default_rng(1)
    score = rng.normal(0.0, 1.0, size=(200, 100)).astype(np.float32)
    valid = np.ones(score.shape, dtype=bool)
    truth = np.zeros(score.shape, dtype=bool)
    thr = threshold_for_far(np.abs(score), 0.05)
    achieved = far(score, valid, truth, thr)
    assert abs(achieved - 0.05) <= 0.30 * 0.05


def test_flat_far_stays_on_3_5():
    quiet = np.full(1000, 0.5, dtype=np.float32)
    assert threshold_for_far(quiet, 0.0) == 3.5


def test_zero_target_far_ratio_is_inside_the_band():
    assert far_ratio(0.0, 0.0) == 1.0
    assert not np.isfinite(far_ratio(0.01, 0.0))


def test_median_profile_threshold_stays_near_3_5():
    threshold, dead = fit_median_profile([0])
    assert abs(threshold - 3.5) <= 0.1
    assert dead == 0


def test_scales_do_not_alarm_the_dead_tail():
    scene = local_scene_var(0, amp_hot=6.0, amp_cold=-5.0, width=7, noise=0.12, ramp=0)
    clean = denoise_field("median_x3", scene["clean"], scene["valid"])
    for scale in SCALES:
        score = z_field(clean, scene["valid"], scene["x_m"], scale=scale, raw=scene["clean"])
        assert dead_alarms(score, scene["valid"], 3.5) == 0
        if scale == "mad_profile":
            profile, _bg, _mad = spatial_z(clean, scene["valid"], scene["x_m"])
            assert np.array_equal(score, profile)


def test_learned_scales_do_not_alarm_the_dead_tail():
    pytest.importorskip("torch")
    scene = local_scene_var(1, amp_hot=4.0, amp_cold=-4.0, width=7, noise=0.12, ramp=0)
    for method in ("n2n_tsr", "zsn2n"):
        clean = denoise_field(method, scene["clean"], scene["valid"], device="cpu")
        for scale in SCALES:
            score = z_field(clean, scene["valid"], scene["x_m"], scale=scale, raw=scene["clean"])
            assert dead_alarms(score, scene["valid"], 3.5) == 0


def test_seed_lists_are_disjoint():
    calib = parse_seed_spec("0-19")
    holdout = parse_seed_spec("100-139")
    assert calib == list(range(20))
    assert holdout == list(range(100, 140))
    assert_disjoint(calib, holdout)
    with pytest.raises(AssertionError):
        assert_disjoint([0, 1], [1, 2])


def test_iou_curve_uses_support_iou():
    score = np.zeros((8, 12), dtype=np.float32)
    score[:, 3:7] = 6.0
    hot = score > 0
    cold = np.zeros_like(hot)
    valid = np.ones(score.shape, dtype=bool)
    time_s = np.arange(8, dtype=np.float64) * 60.0
    x_m = np.arange(12, dtype=np.float64)
    curve = iou_curve(hot, cold, score, valid, time_s, x_m, thresholds=np.array([2.0, 3.5]))
    assert curve[0]["hot_iou"] == pytest.approx(1.0)
    assert curve[0]["far"] == 0.0
    assert "cold_iou" in curve[1]


def _channel(**overrides):
    metrics = {
        "hot_iou": 0.90,
        "cold_iou": 0.88,
        "median_hot_iou": 0.90,
        "median_cold_iou": 0.88,
        "far_holdout": 0.01,
        "median_far_holdout": 0.01,
        "weak_hot_iou": 0.50,
        "median_weak_hot_iou": 0.40,
        "weak_cold_iou": 0.42,
        "median_weak_cold_iou": 0.42,
        "weak_hot_recall": 0.40,
        "median_weak_hot_recall": 0.40,
        "weak_cold_recall": 0.40,
        "median_weak_cold_recall": 0.40,
        "target_far": 0.01,
        "threshold_std": 0.0,
        "real_scored": True,
        "unstable_real": False,
        "half_alarm_fraction": 0.10,
        "median_half_alarm_fraction": 0.12,
        "dead_alarms": 0,
        "witness_passes": 8,
        "median_witness_passes": 8,
        "witness_sha_mismatch": False,
    }
    metrics.update(overrides)
    return metrics


def test_decide_channel_names_the_failed_clause():
    assert decide_channel(_channel())["status"] == "кандидат на продвижение"
    missing = decide_channel(_channel(real_scored=False))
    assert missing["status"] == "не принят"
    assert FAIL_REAL in missing["failed"]
    weak = decide_channel(_channel(weak_hot_iou=0.40))
    assert weak["status"] == "не принят"
    assert FAIL_WEAK in weak["failed"]
