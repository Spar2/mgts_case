"""A standardized column tail stays off a short burst."""

from pathlib import Path

import numpy as np

from fiber.classify.plant import LABEL_INDEX
from fiber.classify.quantile import (
    QUANTILE_BLOCKS,
    QUANTILE_BROAD_Z,
    QUANTILE_DECOY_ROWS,
    QUANTILE_FEATURES,
    QUANTILE_HEAT_ROWS,
    QUANTILE_HEAT_Z,
    QUANTILE_MAD_C,
    QUANTILE_TRAIN_SEEDS,
    plant_quantile,
    quantile_table,
)
from fiber.classify.quantile_train import _balanced
from fiber.classify.scan import labeled_windows
from fiber.models.patch_classifier import SPACE_CROP


def test_quantile_recipe_is_frozen():
    assert QUANTILE_FEATURES[-3:] == ("z", "rank", "tail_rows")
    assert len(QUANTILE_FEATURES) == 8
    assert QUANTILE_HEAT_Z == (3.6, 6.0)
    assert QUANTILE_BROAD_Z == (0.8, 1.5)
    assert QUANTILE_HEAT_ROWS == 16
    assert QUANTILE_DECOY_ROWS == 7
    assert QUANTILE_MAD_C == 0.55
    assert QUANTILE_BLOCKS == (0, 44, 88, 132)
    assert QUANTILE_TRAIN_SEEDS[0] == 1_700_000
    assert len(QUANTILE_TRAIN_SEEDS) == 24
    for blocked in (1_100_000, 1_200_000, 1_300_000, 1_400_000, 1_500_000, 1_600_000):
        assert blocked not in QUANTILE_TRAIN_SEEDS
    source = Path("fiber/classify/quantile.py").read_text(encoding="utf-8")
    assert "witnesses.json" not in source
    assert "4-early-quiet" not in source
    train = Path("fiber/classify/quantile_train.py").read_text(encoding="utf-8")
    assert "decide_witnesses(" in train


def test_painted_quantile_labels_are_the_generator_class():
    scene = plant_quantile(1_700_007)
    assert scene["label_source"] == "generator"
    for event in scene["events"]:
        mask = scene["event_id"] == event["id"]
        assert mask.any()
        assert np.all(scene["labels"][mask] == LABEL_INDEX[event["label"]])
    assert not scene["labels"][~scene["valid"]].any()
    again = plant_quantile(1_700_007)
    assert np.array_equal(scene["field"], again["field"])
    assert np.array_equal(scene["labels"], again["labels"])


def test_heat_block_is_in_the_column_tail_and_a_short_burst_is_not():
    scene = plant_quantile(1_700_003)
    origins, truth = labeled_windows(scene["labels"], scene["valid"])
    feats = quantile_table(scene["field"], scene["valid"], origins)
    z = feats[:, QUANTILE_FEATURES.index("z")]
    rank = feats[:, QUANTILE_FEATURES.index("rank")]
    tail = feats[:, QUANTILE_FEATURES.index("tail_rows")]
    heat = np.array([name == "Local_Heating" for name in truth])
    assert heat.any()
    assert float(np.median(z[heat])) > 3.0
    assert float(np.median(rank[heat])) > 0.9
    assert float(np.median(tail[heat])) > 0.4
    decoy = (origins[:, 1] >= 530) & (origins[:, 1] <= 560)
    assert decoy.any()
    assert float(np.median(np.abs(z[decoy]))) < 2.0
    assert all(truth[int(index)] == "Background" for index in np.flatnonzero(decoy))


def test_one_crop_holds_at_most_one_quantile_event():
    for seed in (1_700_000, 1_700_001, 1_700_002):
        scene = plant_quantile(seed)
        columns = []
        for x in range(scene["event_id"].shape[1]):
            columns.append({int(item) for item in np.unique(scene["event_id"][:, x]) if int(item) != 0})
        for x0 in range(0, scene["event_id"].shape[1] - SPACE_CROP + 1):
            union: set[int] = set()
            for x in range(x0, x0 + SPACE_CROP):
                union |= columns[x]
            assert len(union) <= 1


def test_quantile_class_buckets_fill():
    features, target = _balanced()
    assert features.shape == (6 * 128, len(QUANTILE_FEATURES))
    assert int(target.shape[0]) == 6 * 128
    assert set(int(item) for item in target) == set(range(6))
