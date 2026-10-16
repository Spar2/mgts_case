"""The room generator stays closer to the six records than a flat planted scene."""

from pathlib import Path

import numpy as np
import pytest

from fiber.classify.plant import LABEL_INDEX, plant_scene, rule_scene_label
from fiber.classify.room import SUMMARY_KEYS, field_summary, plant_room, records_target, summary_gaps
from fiber.io import Record

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "artifacts" / "records"


def _load_records(root: Path = RECORDS):
    if not all((root / name / "raw_c.npy").is_file() for name in "123456"):
        pytest.skip("local DTS records are absent")
    return [Record.load(root / name) for name in "123456"]


def test_missing_records_skip(tmp_path):
    with pytest.raises(pytest.skip.Exception):
        _load_records(tmp_path)


def test_room_field_is_closer_than_the_flat_scene():
    records = _load_records()
    target = records_target(records)
    for seed in (3, 11, 29):
        room = plant_room(seed)
        flat = plant_scene("background", seed)
        room_gap = summary_gaps(field_summary(room["field"], room["valid"]), target)
        flat_gap = summary_gaps(field_summary(flat["field"], flat["valid"]), target)
        for key in SUMMARY_KEYS:
            assert room_gap[key] < flat_gap[key]


def test_painted_labels_are_the_generator_class_not_the_shape_rule():
    scene = plant_room(7)
    assert scene["label_source"] == "generator"
    for event in scene["events"]:
        mask = scene["event_id"] == event["id"]
        assert mask.any()
        assert np.all(scene["labels"][mask] == LABEL_INDEX[event["label"]])
    assert not scene["labels"][~scene["valid"]].any()
    fault = LABEL_INDEX["Sensor_Fault"]
    ys, xs = np.nonzero(scene["labels"] == fault)
    assert len(ys) == 1
    isolated = {
        "field": np.full(scene["field"].shape, float(np.median(scene["field"][scene["valid"]])), dtype=np.float32),
        "valid": scene["valid"].copy(),
        "x_m": scene["x_m"],
        "time_s": scene["time_s"],
    }
    isolated["field"][ys[0], xs[0]] = scene["field"][ys[0], xs[0]]
    assert rule_scene_label(isolated) == "Background"
    assert scene["labels"][ys[0], xs[0]] == fault
