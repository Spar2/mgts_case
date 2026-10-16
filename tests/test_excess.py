"""Temporal-excess cables keep a standing ridge out of the heating class."""

from pathlib import Path

import numpy as np

from fiber.classify.excess import (
    EXCESS_FEATURES,
    EXCESS_HEAT_C,
    EXCESS_KINDS,
    EXCESS_ON,
    EXCESS_STANDING_AMP_C,
    EXCESS_TRAIN_SEEDS,
    excess_table,
    plant_excess,
)
from fiber.classify.plant import LABEL_INDEX
from fiber.classify.scan import labeled_windows
from fiber.models.patch_classifier import SPACE_CROP


def test_excess_recipe_is_frozen():
    assert EXCESS_FEATURES[-3:] == ("excess", "excess_support", "row_share")
    assert len(EXCESS_FEATURES) == 8
    assert EXCESS_STANDING_AMP_C == 2.79
    assert EXCESS_HEAT_C == (1.6, 2.4)
    assert EXCESS_ON == (28, 84)
    assert len(EXCESS_KINDS) == 3
    assert EXCESS_KINDS[0]["periods"] == (480,)
    assert EXCESS_KINDS[1]["periods"] == (480,)
    assert EXCESS_KINDS[2]["periods"] == (480,)
    assert EXCESS_TRAIN_SEEDS[0] == 1_300_000
    assert len(EXCESS_TRAIN_SEEDS) == 24
    assert 1_200_000 not in EXCESS_TRAIN_SEEDS
    assert 1_100_000 not in EXCESS_TRAIN_SEEDS
    source = Path("fiber/classify/excess.py").read_text(encoding="utf-8")
    assert "witnesses.json" not in source
    assert "4-early-quiet" not in source
    train = Path("fiber/classify/excess_train.py").read_text(encoding="utf-8")
    assert "decide_witnesses(" in train


def test_painted_excess_labels_are_the_generator_class():
    scene = plant_excess(1_300_007)
    assert scene["label_source"] == "generator"
    assert scene["texture_index"] == 1_300_007 % 3
    for event in scene["events"]:
        mask = scene["event_id"] == event["id"]
        assert mask.any()
        assert np.all(scene["labels"][mask] == LABEL_INDEX[event["label"]])
    assert not scene["labels"][~scene["valid"]].any()
    again = plant_excess(1_300_007)
    assert np.array_equal(scene["field"], again["field"])
    assert np.array_equal(scene["labels"], again["labels"])


def test_standing_ridge_has_no_excess_and_heat_does():
    scene = plant_excess(1_300_003)
    origins, truth = labeled_windows(scene["labels"], scene["valid"])
    feats = excess_table(scene["field"], scene["valid"], origins)
    excess = feats[:, EXCESS_FEATURES.index("excess")]
    heat = np.array([name == "Local_Heating" for name in truth])
    ridge = (origins[:, 1] >= 400) & (origins[:, 1] <= 420)
    assert heat.any()
    assert ridge.any()
    assert float(np.median(excess[heat])) > 1.2
    assert float(np.max(np.abs(excess[ridge]))) < 0.25


def test_one_crop_holds_at_most_one_excess_event():
    for seed in (1_300_000, 1_300_001, 1_300_002):
        scene = plant_excess(seed)
        columns = []
        for x in range(scene["event_id"].shape[1]):
            columns.append({int(item) for item in np.unique(scene["event_id"][:, x]) if int(item) != 0})
        for x0 in range(0, scene["event_id"].shape[1] - SPACE_CROP + 1):
            union: set[int] = set()
            for x in range(x0, x0 + SPACE_CROP):
                union |= columns[x]
            assert len(union) <= 1
