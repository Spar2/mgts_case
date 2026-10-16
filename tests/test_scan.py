import inspect
from pathlib import Path

import numpy as np
import pytest

from fiber.classify.plant import PERSISTENT_WIDTH, PROFILE_AMP_C, PROFILE_DRIFT_C, plant_long
from fiber.classify.scan import (
    HOT_PASS_FRACTION,
    SPACE_STRIDE,
    TIME_STRIDE,
    choose_weights,
    decide_witnesses,
    deviation_crops,
    labeled_windows,
    locator_table,
    overlapping_indices,
    pool_search,
    score_search,
    scored_origins,
    stops,
    summarize_witness,
    window_label,
    window_origins,
    window_valid_counts,
)
from fiber.models.patch_classifier import SPACE_CROP, TIME_CROP


def _blank(rows=28, cols=40):
    labels = np.zeros((rows, cols), dtype=np.int8)
    valid = np.ones((rows, cols), dtype=bool)
    return labels, valid


def test_grid_covers_every_cell_including_the_tail():
    n_time, n_space = 96, 360
    origins = window_origins(n_time, n_space)
    covered = np.zeros((n_time, n_space), dtype=bool)
    for t0, x0 in origins:
        covered[t0:t0 + TIME_CROP, x0:x0 + SPACE_CROP] = True
    assert covered.all()
    assert TIME_STRIDE < TIME_CROP
    assert SPACE_STRIDE < SPACE_CROP
    # 96 - 28 is not a multiple of 14, so the tail start has to be added.
    assert int(stops(n_time, TIME_CROP, TIME_STRIDE)[-1]) == n_time - TIME_CROP


def test_window_label_follows_the_cell_counts():
    labels, valid = _blank()
    assert window_label(labels, valid, 0, 0) == "Background"

    labels[0, :7] = 1
    assert window_label(labels, valid, 0, 0) == "Background"
    labels[0, 7] = 1
    assert window_label(labels, valid, 0, 0) == "Local_Heating"

    labels[:] = 0
    labels[0, 0] = 5
    assert window_label(labels, valid, 0, 0) == "Sensor_Fault"

    labels[:] = 0
    labels[:, :8] = 4
    assert window_label(labels, valid, 0, 0) == "Broad_Shift"
    labels[:] = 4
    assert window_label(labels, valid, 0, 0) == "Background"

    valid[:] = False
    valid[:, :19] = True
    labels[:] = 0
    assert window_label(labels, valid, 0, 0) is None


def test_valid_count_matches_the_fraction_gate():
    valid = np.zeros((28, 80), dtype=bool)
    valid[:, :40] = True
    origins = np.array([[0, 0], [0, 20], [0, 40]], dtype=np.int64)
    counts = window_valid_counts(valid, origins)
    assert list(counts) == [28 * 40, 28 * 20, 0]
    kept = scored_origins(valid)
    assert any(np.array_equal(row, [0, 0]) for row in kept)
    assert any(np.array_equal(row, [0, 20]) for row in kept)
    assert not any(int(row[1]) == 40 for row in kept)


def test_long_events_stay_out_of_one_crop_and_each_has_a_locator():
    for seed in range(6):
        scene = plant_long(seed)
        assert scene["labels"].shape == (96, 360)
        assert not scene["labels"][~scene["valid"]].any()
        assert not scene["event_id"][~scene["valid"]].any()
        columns: list[set[int]] = []
        for x in range(scene["event_id"].shape[1]):
            columns.append({int(item) for item in np.unique(scene["event_id"][:, x]) if int(item) != 0})
        for x0 in range(0, scene["event_id"].shape[1] - SPACE_CROP + 1):
            union: set[int] = set()
            for x in range(x0, x0 + SPACE_CROP):
                union |= columns[x]
            assert len(union) <= 1
        origins, truth = labeled_windows(scene["labels"], scene["valid"])
        locators = locator_table(scene["event_id"], origins, truth, scene["events"])
        assert len(scene["events"]) == 5
        for event in scene["events"]:
            assert locators[event["id"]], event["label"]
        broad = [name for name in truth if name == "Broad_Shift"]
        assert broad
        # Interior of the fixed 258:308 segment is one grid window, 260:300.
        interior = window_label(scene["labels"], scene["valid"], 0, 260)
        assert interior == "Background"


def test_room_profile_keeps_the_same_events_and_lifts_background():
    spans = []
    for seed in range(4):
        flat = plant_long(seed)
        prof = plant_long(seed, profile=True)
        assert prof["kind"] == "profile"
        assert np.array_equal(flat["labels"], prof["labels"])
        assert np.array_equal(flat["event_id"], prof["event_id"])
        delta = prof["field"].astype(np.float64) - flat["field"].astype(np.float64)
        assert float(np.max(np.abs(delta))) <= PROFILE_AMP_C[1] + PROFILE_DRIFT_C + 1e-3
        assert window_label(prof["labels"], prof["valid"], 0, 100) == "Background"
        origins, truth = labeled_windows(prof["labels"], prof["valid"])
        crops = deviation_crops(prof["field"], prof["valid"], origins)
        for crop, name in zip(crops, truth, strict=True):
            if name != "Background":
                continue
            column = np.median(crop, axis=0)
            spans.append(float(column.max() - column.min()))
    assert float(np.median(spans)) >= 0.5


def test_persistent_streak_is_local_and_does_not_meet_the_next_event():
    from fiber.models.patch_classifier import SPACE_CROP as crop

    for seed in range(4):
        scene = plant_long(seed, profile=True, persistent=True)
        assert scene["kind"] == "persistent"
        for eid, low, high in ((1, PERSISTENT_WIDTH[0], PERSISTENT_WIDTH[1] - 1), (2, PERSISTENT_WIDTH[0], PERSISTENT_WIDTH[1] - 1)):
            cols = np.flatnonzero((scene["event_id"] == eid).any(axis=0))
            assert low <= len(cols) <= high
            assert np.array_equal(scene["event_id"][:, cols], np.full((scene["event_id"].shape[0], len(cols)), eid))
        assert window_label(scene["labels"], scene["valid"], 0, 0) == "Local_Heating"
        # Cool ends by column 107 and the next event starts at 148, so this
        # window sits in the gap on every seed. Column 100 can still hold cooling.
        assert window_label(scene["labels"], scene["valid"], 0, 108) == "Background"
        assert window_label(scene["labels"], scene["valid"], 0, 260) == "Background"
        columns = []
        for x in range(scene["event_id"].shape[1]):
            columns.append({int(item) for item in np.unique(scene["event_id"][:, x]) if int(item) != 0})
        for x0 in range(0, scene["event_id"].shape[1] - crop + 1):
            union: set[int] = set()
            for x in range(x0, x0 + crop):
                union |= columns[x]
            assert len(union) <= 1


def test_search_score_counts_a_hit_and_a_false_alarm():
    events = ({"id": 1, "label": "Local_Heating"}, {"id": 2, "label": "Local_Cooling"})
    truth = ["Local_Heating", "Background", "Local_Cooling", "Background"]
    predicted = ["Local_Heating", "Local_Heating", "Background", "Background"]
    locators = {1: [0], 2: [2]}
    score = score_search(truth, predicted, events, locators)
    assert score["event_recall"] == 0.5
    assert score["false_alarm_rate"] == 0.5
    assert score["n_false_alarms"] == 1
    assert score["per_class_event_recall"]["Local_Heating"] == 1.0
    assert score["per_class_event_recall"]["Local_Cooling"] == 0.0
    assert score["n_agreements"] == 2
    assert score["window_agreement"] == 0.5


def test_window_agreement_counts_windows_and_pools_by_count():
    """One correct locator is not agreement. The pool adds counts, it does not average rates."""
    events = ({"id": 1, "label": "Local_Heating"},)
    truth = ["Local_Heating", "Local_Heating", "Background"]
    predicted = ["Local_Heating", "Background", "Background"]
    first = score_search(truth, predicted, events, {1: [0, 1]})
    assert first["event_recall"] == 1.0
    assert first["n_false_alarms"] == 0
    assert first["n_agreements"] == 2
    assert first["window_agreement"] == 2 / 3
    second = score_search(
        ["Background", "Background"],
        ["Background", "Local_Heating"],
        (),
        {},
    )
    assert second["n_false_alarms"] == 1
    assert second["n_agreements"] == 1
    pooled = pool_search([first, second])
    assert pooled["n_agreements"] == 3
    assert pooled["n_windows"] == 5
    assert pooled["window_agreement"] == 3 / 5
    assert pooled["n_false_alarms"] == 1
    assert pooled["event_recall"] == 1.0


def test_witness_rule_is_frozen():
    """Direct call on three tiny name lists. Fraction 0.3 and expert_confirmed stay false."""
    assert HOT_PASS_FRACTION == 0.3
    hot = ["Local_Heating"] * 3 + ["Background"] * 7
    quiet = ["Background", "Background"]
    invalid: list[str] = []
    hot_row = summarize_witness("positive_hot", hot)
    quiet_row = summarize_witness("negative_quiet", quiet)
    invalid_row = summarize_witness("negative_invalid", invalid)
    assert hot_row["pass"]
    assert hot_row["local_heating_fraction"] == 0.3
    assert hot_row["expert_confirmed"] is False
    assert not summarize_witness("positive_hot", ["Local_Heating"] * 2 + ["Background"] * 8)["pass"]
    assert not summarize_witness("positive_hot", [])["pass"]
    assert quiet_row["pass"]
    assert quiet_row["expert_confirmed"] is False
    assert not summarize_witness("negative_quiet", ["Background", "Local_Heating"])["pass"]
    assert not summarize_witness("negative_quiet", [])["pass"]
    assert invalid_row["pass"]
    assert invalid_row["n_windows"] == 0
    assert invalid_row["expert_confirmed"] is False
    assert not summarize_witness("negative_invalid", ["Background"])["pass"]


def test_context_and_agree_scans_call_the_same_witness_routine():
    """Both scan entry points call decide_witnesses, and that calls summarize_witness."""
    root = Path(__file__).resolve().parents[1]
    for script in ("scan_context.py", "scan_agree.py"):
        text = (root / "scripts" / script).read_text(encoding="utf-8")
        assert "decide_witnesses(" in text
    assert "summarize_witness(" in inspect.getsource(decide_witnesses)
    x_m = np.arange(80, dtype=np.float64)
    origins = np.array([[0, 0], [0, 10], [0, 40]], dtype=np.int64)
    specs = [{
        "id": "tiny-hot",
        "role": "positive_hot",
        "rows": [0, 5],
        "x_m": [0.0, 20.0],
    }]
    names = ["Local_Heating", "Background", "Local_Heating"]
    hit = overlapping_indices(origins, x_m, 0, 5, 0.0, 20.0)
    direct = summarize_witness("positive_hot", [names[int(index)] for index in hit])
    rows = decide_witnesses(specs, origins, x_m, names, "4")
    assert len(rows) == 1
    assert rows[0]["pass"] is direct["pass"]
    assert rows[0]["local_heating_fraction"] == direct["local_heating_fraction"]
    assert rows[0]["expert_confirmed"] is False
    assert rows[0]["id"] == "tiny-hot"
    assert rows[0]["record"] == "4"


def test_overlap_uses_half_open_rows_and_inclusive_metres():
    x_m = np.arange(100, dtype=np.float64)
    origins = np.array([[0, 10], [0, 0], [30, 10]], dtype=np.int64)
    hit = overlapping_indices(origins, x_m, 0, 5, 45.0, 46.0)
    assert list(hit) == [0]


def test_choose_weights_uses_only_the_planted_scores():
    scanner = {"event_recall": 0.8, "false_alarm_rate": 0.2}
    naming = {"event_recall": 0.6, "false_alarm_rate": 0.0}
    assert choose_weights(scanner, naming) == "scanner"
    assert choose_weights(scanner, {"event_recall": 0.9, "false_alarm_rate": 0.5}) == "naming"
    tied = {"event_recall": 0.8, "false_alarm_rate": 0.4}
    assert choose_weights(scanner, tied) == "scanner"
    assert choose_weights({"event_recall": 0.8, "false_alarm_rate": 0.4}, scanner) == "naming"
    assert choose_weights(scanner, {"event_recall": 0.8, "false_alarm_rate": 0.2}) == "scanner"
    assert choose_weights(scanner, None) == "scanner"


def test_random_net_scores_one_label_per_window():
    torch = pytest.importorskip("torch")
    from fiber.models.patch_classifier import PatchClassifier

    scene = plant_long(0)
    origins, truth = labeled_windows(scene["labels"], scene["valid"])
    crops = deviation_crops(scene["field"], scene["valid"], origins)
    net = PatchClassifier().eval()
    with torch.inference_mode():
        logits = net(torch.from_numpy(crops)[:, None])
    assert tuple(logits.shape) == (len(truth), 6)
    assert len(truth) > 0
