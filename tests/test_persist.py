"""A stable low-excess column stays off the persistent heating tail."""

from pathlib import Path

import numpy as np

from fiber.classify.persist import (
    PERSIST_AC1_JOINT_P10,
    PERSIST_AC1_STABLE_P50,
    PERSIST_FEATURES,
    PERSIST_HEAT_C,
    PERSIST_TRAIN_SEEDS,
    persist_table,
    plant_persist,
)
from fiber.classify.persist_train import _balanced
from fiber.classify.plant import LABEL_INDEX
from fiber.classify.scan import labeled_windows
from fiber.models.patch_classifier import SPACE_CROP


def test_persist_recipe_is_frozen():
    assert PERSIST_FEATURES[-2:] == ("excess", "ac1")
    assert len(PERSIST_FEATURES) == 7
    assert PERSIST_HEAT_C == (2.19, 3.88)
    assert PERSIST_AC1_STABLE_P50 == 0.20
    assert PERSIST_AC1_JOINT_P10 == 0.81
    assert PERSIST_TRAIN_SEEDS[0] == 1_900_000
    assert len(PERSIST_TRAIN_SEEDS) == 24
    for blocked in (1_100_000, 1_200_000, 1_300_000, 1_400_000, 1_500_000, 1_600_000, 1_700_000, 1_800_000):
        assert blocked not in PERSIST_TRAIN_SEEDS
    source = Path("fiber/classify/persist.py").read_text(encoding="utf-8")
    assert "witnesses.json" not in source
    assert "4-early-quiet" not in source
    train = Path("fiber/classify/persist_train.py").read_text(encoding="utf-8")
    assert "decide_witnesses(" in train


def test_painted_persist_labels_are_the_generator_class():
    scene = plant_persist(1_900_007)
    assert scene["label_source"] == "generator"
    for event in scene["events"]:
        mask = scene["event_id"] == event["id"]
        assert mask.any()
        assert np.all(scene["labels"][mask] == LABEL_INDEX[event["label"]])
    assert not scene["labels"][~scene["valid"]].any()
    again = plant_persist(1_900_007)
    assert np.array_equal(scene["field"], again["field"])
    assert np.array_equal(scene["labels"], again["labels"])


def test_heat_is_persistent_and_a_stable_column_stays_background():
    scene = plant_persist(1_900_003)
    origins, truth = labeled_windows(scene["labels"], scene["valid"])
    feats = persist_table(scene["field"], scene["valid"], origins)
    excess = feats[:, PERSIST_FEATURES.index("excess")]
    ac1 = feats[:, PERSIST_FEATURES.index("ac1")]
    heat = np.array([name == "Local_Heating" for name in truth])
    assert heat.any()
    assert float(np.median(excess[heat])) > 1.5
    assert float(np.median(ac1[heat])) > 0.8
    stable = (origins[:, 1] >= 420) & (origins[:, 1] <= 440)
    assert stable.any()
    assert float(np.median(np.abs(excess[stable]))) < 0.8
    assert float(np.median(ac1[stable])) < 0.5
    assert all(truth[int(index)] == "Background" for index in np.flatnonzero(stable))


def test_one_crop_holds_at_most_one_persist_event():
    for seed in (1_900_000, 1_900_001, 1_900_002):
        scene = plant_persist(seed)
        columns = []
        for x in range(scene["event_id"].shape[1]):
            columns.append({int(item) for item in np.unique(scene["event_id"][:, x]) if int(item) != 0})
        for x0 in range(0, scene["event_id"].shape[1] - SPACE_CROP + 1):
            union: set[int] = set()
            for x in range(x0, x0 + SPACE_CROP):
                union |= columns[x]
            assert len(union) <= 1


def test_persist_class_buckets_fill():
    features, target = _balanced()
    assert features.shape == (6 * 128, len(PERSIST_FEATURES))
    assert int(target.shape[0]) == 6 * 128
    assert set(int(item) for item in target) == set(range(6))
    background = target == 0
    ac1 = features[background, PERSIST_FEATURES.index("ac1")]
    excess = features[background, PERSIST_FEATURES.index("excess")]
    low = ac1 <= 0.45
    assert int(low.sum()) >= 64
    assert float(np.median(np.abs(excess[low]))) < 1.0
