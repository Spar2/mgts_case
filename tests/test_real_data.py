"""Scientific failure modes: abstention, temporal leakage and missing data."""
import numpy as np
import pytest

from fiber.real_data import (correlation, crop_partition, diagnose_record,
                             episodes, temporal_partition, weak_evidence)


def test_missing_reference_never_becomes_background():
    x, t = np.arange(5.), np.arange(20.)*60
    raw = np.full((20, 5), 22.)
    e, _ = weak_evidence(raw, np.ones_like(raw, bool), x, t, [1, 3])
    assert e["weak_presence_label"] is None
    assert e["reference_columns"] == 0
    assert e["expert_label"] is None


def test_unobserved_rows_break_runs_and_do_not_prove_periodicity():
    t = np.arange(12.)*60
    active = np.ones(12, bool)
    observed = active.copy()
    observed[5] = False
    assert episodes(active, observed, t) == [(0, 5), (6, 12)]
    t[6:] += 1000
    assert episodes(active, active, t) == [(0, 6), (6, 12)]


def test_real_rules_keep_presence_separate_from_cause():
    x, t = np.arange(100.), np.arange(20.)*60
    raw = np.full((20, 100), 22.)
    raw[5:15, 48:53] += 3
    e, _ = weak_evidence(raw, np.ones_like(raw, bool), x, t, [48, 52])
    assert e["weak_presence_label"] == 1
    assert e["shape_hypotheses"] == ["local_heating"]
    assert e["physical_cause"] is None and not e["expert_confirmed"]
    # A broad shift equally affecting core and reference is NOT ruled out by 0.
    e, _ = weak_evidence(raw*0+30, np.ones_like(raw, bool), x, t, [48, 52])
    assert e["weak_presence_label"] == 0
    assert "not all anomalies" in e["label_scope"]


def test_guard_and_crop_assignment_cannot_leak_across_split():
    t = np.arange(200.)*60
    split, _ = temporal_partition(t, guard_s=1800)
    for a, b in ((0, 1), (1, 2)):
        assert t[split == b].min()-t[split == a].max() >= 1800
    assert crop_partition(split[100:140]) == -1
    assert crop_partition(split[:10]) == 0
    with pytest.raises(ValueError):
        temporal_partition(t, guard_s=-1)


def test_gap_increments_are_excluded_and_residual_is_not_noise_truth():
    x = np.arange(10.)
    t = np.array([0., 60., 1000., 1060.])
    raw = np.array([np.full(10, k) for k in [22, 22, 100, 100]])
    e = diagnose_record(raw, raw, np.ones_like(raw, bool), x, t)
    assert e["temporal_increment_abs_c"]["q99"] == 0
    assert not e["independent_noise_assumption_verified"]
    assert e["residual_adjacent_time"]["eligible_pairs"] == 20
    assert e["residual_adjacent_time"]["pearson"] is None


def test_correlation_reports_sample_size_and_ignores_missing_pairs():
    a = np.arange(20., dtype=float)
    b = a*2
    a[2] = np.nan
    e = correlation(a, b, limit=5)
    assert e["eligible_pairs"] == 19
    assert e["sampled_pairs"] <= 5
    assert e["pearson"] == pytest.approx(1)


def test_short_high_cadence_burst_is_not_a_three_minute_episode():
    x, t = np.arange(100.), np.arange(30.)*5
    raw = np.full((30, 100), 22.)
    raw[5:15, 48:53] += 3
    e, _ = weak_evidence(raw, np.ones_like(raw, bool), x, t, [48, 52])
    assert e["rule_votes"]["sustained_hot"] is None
    assert e["weak_presence_label"] is None


def test_reconstruction_blind_spot_and_target_exclusion():
    from scripts.benchmark_real_reconstruction import neighbour_indices, samples
    from types import SimpleNamespace
    x = np.linspace(0, 100, 501)
    ids = neighbour_indices(x, 250)
    assert all(abs(x[i]-x[250]) >= 2-1e-8 for i in ids)
    assert 250 not in ids
    rec = SimpleNamespace(x_m=x, time_s=np.arange(200.)*60,
                          n_profiles=200, raw_c=np.full((200, 501), 22.),
                          valid=np.ones((200, 501), bool))
    features, target, codes, _, _ = samples(rec)
    assert set(codes) == {0, 1}
    # Alter every target column: predictor footprints are disjoint at 17m spacing.
    anchors, next_m = [], 8.
    for col, distance in enumerate(x):
        if distance >= next_m:
            anchors.append(col); next_m = float(distance)+17
    rec.raw_c[:, anchors] += 7
    changed_features, changed_target, _, _, _ = samples(rec)
    np.testing.assert_array_equal(features, changed_features)
    np.testing.assert_allclose(changed_target-target, 7)
