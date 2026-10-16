"""A stable column stays off the joint heating tail."""

from pathlib import Path

import numpy as np

from fiber.classify.plant import LABEL_INDEX
from fiber.classify.scan import labeled_windows
from fiber.classify.stability import (
    STABILITY_BROAD_C,
    STABILITY_FEATURES,
    STABILITY_HEAT_C,
    STABILITY_MAD_P10,
    STABILITY_MAD_P50,
    STABILITY_STABLE_C,
    STABILITY_TRAIN_SEEDS,
    plant_stability,
    stability_table,
)
from fiber.classify.stability_train import _balanced
from fiber.models.patch_classifier import SPACE_CROP


def test_stability_recipe_is_frozen():
    assert STABILITY_FEATURES[-2:] == ("excess", "mad")
    assert len(STABILITY_FEATURES) == 7
    assert STABILITY_HEAT_C == (2.19, 3.88)
    assert STABILITY_BROAD_C == (0.59, 1.15)
    assert STABILITY_STABLE_C == (0.24, 0.37)
    assert STABILITY_MAD_P50 == 0.55
    assert STABILITY_MAD_P10 == 0.21
    assert STABILITY_TRAIN_SEEDS[0] == 1_800_000
    assert len(STABILITY_TRAIN_SEEDS) == 24
    for blocked in (1_100_000, 1_200_000, 1_300_000, 1_400_000, 1_500_000, 1_600_000, 1_700_000):
        assert blocked not in STABILITY_TRAIN_SEEDS
    source = Path("fiber/classify/stability.py").read_text(encoding="utf-8")
    assert "witnesses.json" not in source
    assert "4-early-quiet" not in source
    train = Path("fiber/classify/stability_train.py").read_text(encoding="utf-8")
    assert "decide_witnesses(" in train


def test_painted_stability_labels_are_the_generator_class():
    scene = plant_stability(1_800_007)
    assert scene["label_source"] == "generator"
    for event in scene["events"]:
        mask = scene["event_id"] == event["id"]
        assert mask.any()
        assert np.all(scene["labels"][mask] == LABEL_INDEX[event["label"]])
    assert not scene["labels"][~scene["valid"]].any()
    again = plant_stability(1_800_007)
    assert np.array_equal(scene["field"], again["field"])
    assert np.array_equal(scene["labels"], again["labels"])


def test_heat_is_the_joint_tail_and_a_stable_bump_is_not():
    scene = plant_stability(1_800_003)
    origins, truth = labeled_windows(scene["labels"], scene["valid"])
    feats = stability_table(scene["field"], scene["valid"], origins)
    excess = feats[:, STABILITY_FEATURES.index("excess")]
    mad = feats[:, STABILITY_FEATURES.index("mad")]
    heat = np.array([name == "Local_Heating" for name in truth])
    assert heat.any()
    assert float(np.median(excess[heat])) > 1.6
    assert float(np.median(mad[heat])) > 0.4
    stable = (origins[:, 1] >= 420) & (origins[:, 1] <= 440)
    assert stable.any()
    assert float(np.median(np.abs(excess[stable]))) < 0.8
    assert float(np.median(mad[stable])) < 0.35
    assert all(truth[int(index)] == "Background" for index in np.flatnonzero(stable))


def test_one_crop_holds_at_most_one_stability_event():
    for seed in (1_800_000, 1_800_001, 1_800_002):
        scene = plant_stability(seed)
        columns = []
        for x in range(scene["event_id"].shape[1]):
            columns.append({int(item) for item in np.unique(scene["event_id"][:, x]) if int(item) != 0})
        for x0 in range(0, scene["event_id"].shape[1] - SPACE_CROP + 1):
            union: set[int] = set()
            for x in range(x0, x0 + SPACE_CROP):
                union |= columns[x]
            assert len(union) <= 1


def test_stability_class_buckets_fill():
    features, target = _balanced()
    assert features.shape == (6 * 128, len(STABILITY_FEATURES))
    assert int(target.shape[0]) == 6 * 128
    assert set(int(item) for item in target) == set(range(6))
    mad = features[target == 0, STABILITY_FEATURES.index("mad")]
    excess = features[target == 0, STABILITY_FEATURES.index("excess")]
    assert float(np.min(mad)) < 0.35
    assert float(np.median(np.abs(excess[mad <= 0.30]))) < 1.0
