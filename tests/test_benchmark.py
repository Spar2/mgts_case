import numpy as np

from fiber.benchmark import alarm_fraction, evaluate_events
from fiber.benchmark.synthetic import half_cable_shift, local_scene
from fiber.anomaly import spatial_z


def _grid(t=8, x=10):
    return np.arange(t, dtype=np.float64) * 60.0, np.arange(x, dtype=np.float64)


def test_perfect_match_scores_one():
    truth = np.zeros((8, 10), dtype=bool)
    truth[2:6, 3:6] = True
    time_s, x_m = _grid()
    score = evaluate_events(truth, truth, time_s, x_m)
    assert score["matched_events"] == 1
    assert score["event_f1"] == 1.0
    assert score["predicted_invalid_cells"] == 0


def test_wrong_place_is_a_miss():
    truth = np.zeros((8, 10), dtype=bool)
    pred = np.zeros_like(truth)
    truth[2:6, 1:3] = True
    pred[2:6, 7:9] = True
    time_s, x_m = _grid()
    score = evaluate_events(truth, pred, time_s, x_m)
    assert score["matched_events"] == 0
    assert score["event_recall"] == 0.0
    assert score["event_precision"] == 0.0


def test_one_cell_does_not_cover_a_long_event():
    truth = np.zeros((8, 12), dtype=bool)
    pred = np.zeros_like(truth)
    truth[2:7, 2:8] = True
    pred[4, 4] = True
    time_s, x_m = _grid(x=12)
    score = evaluate_events(truth, pred, time_s, x_m, min_iou=0.2)
    assert score["matched_events"] == 0


def test_invalid_prediction_is_counted_and_excluded():
    truth = np.zeros((6, 4), dtype=bool)
    pred = np.zeros_like(truth)
    valid = np.ones_like(truth)
    valid[:, 3] = False
    pred[:, 3] = True
    time_s, x_m = _grid(t=6, x=4)
    score = evaluate_events(truth, pred, time_s, x_m, valid)
    assert score["predicted_invalid_cells"] == 6
    assert score["predicted_events"] == 0


def test_spatial_z_recovers_planted_blobs_and_not_the_dead_tail():
    scene = local_scene()
    z, _, _ = spatial_z(scene["clean"], scene["valid"], scene["x_m"])
    pred = np.abs(z) >= 3.5
    assert not pred[:, 180:].any()
    hot = evaluate_events(scene["hot"], (z >= 3.5), scene["time_s"], scene["x_m"], scene["valid"])
    cold = evaluate_events(scene["cold"], (z <= -3.5), scene["time_s"], scene["x_m"], scene["valid"])
    assert hot["event_recall"] == 1.0
    assert cold["event_recall"] == 1.0
    assert hot["predicted_invalid_cells"] == 0
    assert alarm_fraction(pred, scene["valid"]) < 0.2


def test_half_cable_shift_is_not_mostly_local():
    scene = half_cable_shift()
    z, _, _ = spatial_z(scene["clean"], scene["valid"], scene["x_m"])
    local = np.abs(z) >= 3.5
    assert alarm_fraction(local, scene["broad"]) < 0.25
