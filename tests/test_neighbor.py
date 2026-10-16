"""A smooth strong peak stays off a jagged bulk spike."""

from pathlib import Path

import numpy as np

from fiber.classify.neighbor import (
    NEIGHBOR_BROAD_WIDTH_M,
    NEIGHBOR_DECOY_C,
    NEIGHBOR_FEATURES,
    NEIGHBOR_FLAT_RATIO,
    NEIGHBOR_HEAT_C,
    NEIGHBOR_HEAT_SIGMA_M,
    NEIGHBOR_JAGGED_RATIO,
    NEIGHBOR_ON,
    NEIGHBOR_TRAIN_SEEDS,
    neighbor_table,
    plant_neighbor,
)
from fiber.classify.neighbor_train import _balanced
from fiber.classify.plant import LABEL_INDEX
from fiber.classify.scan import labeled_windows
from fiber.models.patch_classifier import SPACE_CROP


def test_neighbor_recipe_is_frozen():
    assert NEIGHBOR_FEATURES[-4:] == ("excess", "stick", "ratio", "common")
    assert len(NEIGHBOR_FEATURES) == 9
    assert NEIGHBOR_HEAT_C == (1.74, 2.81)
    assert NEIGHBOR_DECOY_C == (0.59, 1.74)
    assert NEIGHBOR_HEAT_SIGMA_M == 2.66
    assert NEIGHBOR_JAGGED_RATIO == 0.818
    assert NEIGHBOR_FLAT_RATIO == 0.20
    assert NEIGHBOR_ON == (28, 84)
    assert NEIGHBOR_BROAD_WIDTH_M == 56
    assert NEIGHBOR_TRAIN_SEEDS[0] == 1_600_000
    assert len(NEIGHBOR_TRAIN_SEEDS) == 24
    for blocked in (1_100_000, 1_200_000, 1_300_000, 1_400_000, 1_500_000):
        assert blocked not in NEIGHBOR_TRAIN_SEEDS
    source = Path("fiber/classify/neighbor.py").read_text(encoding="utf-8")
    assert "witnesses.json" not in source
    assert "4-early-quiet" not in source
    train = Path("fiber/classify/neighbor_train.py").read_text(encoding="utf-8")
    assert "decide_witnesses(" in train


def test_painted_neighbor_labels_are_the_generator_class():
    scene = plant_neighbor(1_600_007)
    assert scene["label_source"] == "generator"
    for event in scene["events"]:
        mask = scene["event_id"] == event["id"]
        assert mask.any()
        assert np.all(scene["labels"][mask] == LABEL_INDEX[event["label"]])
    assert not scene["labels"][~scene["valid"]].any()
    again = plant_neighbor(1_600_007)
    assert np.array_equal(scene["field"], again["field"])
    assert np.array_equal(scene["labels"], again["labels"])


def test_smooth_peak_is_not_a_jagged_spike():
    scene = plant_neighbor(1_600_003)
    origins, truth = labeled_windows(scene["labels"], scene["valid"])
    feats = neighbor_table(scene["field"], scene["valid"], origins)
    excess = feats[:, NEIGHBOR_FEATURES.index("excess")]
    ratio = feats[:, NEIGHBOR_FEATURES.index("ratio")]
    stick = feats[:, NEIGHBOR_FEATURES.index("stick")]
    heat = np.array([name == "Local_Heating" for name in truth])
    assert heat.any()
    assert float(np.median(excess[heat])) > 1.5
    assert float(np.median(ratio[heat])) < NEIGHBOR_FLAT_RATIO
    assert float(np.median(stick[heat])) > 0.05
    flat = (
        (origins[:, 0] >= NEIGHBOR_ON[0])
        & (origins[:, 0] <= NEIGHBOR_ON[0] + 28)
        & (origins[:, 1] >= 400)
        & (origins[:, 1] <= 410)
    )
    assert int(flat.sum()) >= 3
    assert float(np.median(ratio[flat])) < NEIGHBOR_FLAT_RATIO
    assert float(np.median(np.abs(excess[flat]))) >= NEIGHBOR_HEAT_C[0]
    assert all(truth[int(index)] == "Background" for index in np.flatnonzero(flat))
    decoy = (origins[:, 0] >= 112) & (origins[:, 1] >= 530) & (origins[:, 1] <= 560)
    assert decoy.any()
    assert float(np.median(ratio[decoy])) >= NEIGHBOR_JAGGED_RATIO
    assert all(truth[int(index)] == "Background" for index in np.flatnonzero(decoy))


def test_one_crop_holds_at_most_one_neighbor_event():
    for seed in (1_600_000, 1_600_001, 1_600_002):
        scene = plant_neighbor(seed)
        columns = []
        for x in range(scene["event_id"].shape[1]):
            columns.append({int(item) for item in np.unique(scene["event_id"][:, x]) if int(item) != 0})
        for x0 in range(0, scene["event_id"].shape[1] - SPACE_CROP + 1):
            union: set[int] = set()
            for x in range(x0, x0 + SPACE_CROP):
                union |= columns[x]
            assert len(union) <= 1


def test_neighbor_background_split_fills():
    features, target = _balanced()
    assert features.shape == (6 * 128, len(NEIGHBOR_FEATURES))
    assert int(target.shape[0]) == 6 * 128
    assert set(int(item) for item in target) == set(range(6))
