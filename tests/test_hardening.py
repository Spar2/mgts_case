"""Regression cases from the independent audit, not model-score snapshots."""
import csv
import json

import numpy as np
import pytest

from fiber.io import Record, load_csv
from fiber.anomaly import extract_zones, spatial_z
from fiber.anomaly.detect import detection_status, spatial_availability
from fiber.benchmark.events import evaluate_events
from fiber.denoise.dsp import median_x


def _csv(tmp_path, second_points=None, declared=None, timestamps=None):
    path = tmp_path / 'record.csv'
    rows = [[[0, 20], [1, 21], [2, 22], [3, 23]], second_points or [[0, 11], [2, 31], [3, 41]]]
    stamps = timestamps or ['2025-10-16 12:00:06', '2025-10-16 12:01:06']
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['timestamp', 'dts_id', 'channel_id', 'points_size', 'points'])
        writer.writeheader()
        for i, points in enumerate(rows):
            writer.writerow(dict(timestamp=stamps[i], dts_id='a', channel_id='1', points_size=declared or len(points), points=json.dumps(points)))
    return path


def test_missing_internal_coordinate_never_borrows_temperature(tmp_path):
    rec = load_csv(_csv(tmp_path))
    assert rec.valid[1].tolist() == [True, False, True, True]
    assert np.isnan(rec.raw_c[1, 1])
    assert rec.raw_c[1, 3] == 41
    assert rec.quality_codes[1].tolist() == [0, 4, 0, 0]
    rec.save(tmp_path / 'saved')
    restored = Record.load(tmp_path / 'saved')
    np.testing.assert_array_equal(restored.quality_codes, rec.quality_codes)
    assert '+00:00' not in restored.time_iso[0]
    assert restored.time_s[1] - restored.time_s[0] == 60


@pytest.mark.parametrize('points', [[[0,20],[0,21]], [[2,20],[1,21]], [[0,20],[9,21]]])
def test_ambiguous_or_changed_grid_is_rejected(tmp_path, points):
    with pytest.raises(ValueError):
        load_csv(_csv(tmp_path, points))


def test_declared_size_mismatch_rejected(tmp_path):
    with pytest.raises(ValueError, match='points_size'):
        load_csv(_csv(tmp_path, declared=999))


def test_unknown_timezone_cannot_be_combined_with_absolute_time(tmp_path):
    with pytest.raises(ValueError, match='timezones'):
        load_csv(_csv(tmp_path, timestamps=['2025-10-16 12:00:06', '2025-10-16 12:01:06+00:00']))


@pytest.mark.parametrize('gap,missing', [(3600, False), (0, True), (3600, True)])
def test_zones_do_not_infer_continuity_through_absence(gap, missing):
    z = np.zeros((17,100))
    z[1:16,40:45] = 5
    valid = np.ones_like(z, dtype=bool)
    if missing:
        valid[8] = False
    times = np.arange(17) * 60.0
    times[8:] += gap
    zones = extract_zones(z, valid, np.arange(100), [str(t) for t in times], times, min_width_m=3, min_profiles=6, min_cells=24)
    assert len(zones) == 2
    assert all(zone['duration_s'] < 500 for zone in zones)


def test_first_profile_is_a_real_event_even_when_missed():
    truth = np.zeros((4,5), bool)
    truth[0,1:4] = True
    score = evaluate_events(truth, np.zeros_like(truth), np.arange(4)*60, np.arange(5))
    assert score['truth_events'] == 1
    assert score['event_recall'] == 0
    assert score['event_f1'] == 0


def test_event_scoring_splits_time_discontinuities():
    truth = np.ones((6,3), bool)
    prediction = truth.copy()
    prediction[3:] = False
    score = evaluate_events(truth, prediction, np.array([0,60,120,3720,3780,3840]), np.arange(3))
    assert score['truth_events'] == 2
    assert score['matched_events'] == 1
    assert score['event_recall'] == 0.5


def test_missing_halo_is_unknown_not_normal():
    field = np.zeros((2,100))
    field[:,50] = 35
    valid = np.zeros_like(field, bool)
    valid[:,50] = True
    _, bg, _ = spatial_z(field, valid, np.arange(100))
    available = spatial_availability(field, valid, bg)
    status = detection_status(0, valid, available)
    assert status['anomaly_status'] == 'Unknown'
    assert status['n_valid'] == 2 and status['n_score_available'] == 0


def test_large_median_does_not_cross_invalid_spatial_barrier():
    raw = np.array([[0, -300, 100, 100]], float)
    valid = raw > -200
    result = median_x(raw, valid, k=5)
    assert result[0,0] == 0
    assert np.isnan(result[0,1])
    assert result[0,2] == 100
