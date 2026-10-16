"""Local excess stays off the shift that the shoulders share."""

from pathlib import Path

import numpy as np

from fiber.classify.contrast import (
    CONTRAST_ABS_P50,
    CONTRAST_BROAD_C,
    CONTRAST_DECOY_ROWS,
    CONTRAST_FEATURES,
    CONTRAST_HEAT_C,
    CONTRAST_HEAT_WIDTH_M,
    CONTRAST_ON,
    CONTRAST_SHOULDER_P90,
    CONTRAST_TRAIN_SEEDS,
    contrast_table,
    plant_contrast,
)
from fiber.classify.contrast_train import _balanced
from fiber.classify.plant import LABEL_INDEX
from fiber.classify.scan import labeled_windows
from fiber.models.patch_classifier import SPACE_CROP, TIME_CROP


def test_contrast_recipe_is_frozen():
    assert CONTRAST_FEATURES[-4:] == ("excess", "shoulder", "contrast", "duty")
    assert len(CONTRAST_FEATURES) == 9
    assert CONTRAST_HEAT_C == (1.74, 2.80)
    assert CONTRAST_BROAD_C == (1.74, 2.16)
    assert CONTRAST_DECOY_ROWS == 7
    assert CONTRAST_ON == (28, 84)
    assert CONTRAST_HEAT_WIDTH_M == (3, 7)
    assert CONTRAST_SHOULDER_P90 == 0.479
    assert CONTRAST_ABS_P50 == 0.370
    assert CONTRAST_TRAIN_SEEDS[0] == 1_500_000
    assert len(CONTRAST_TRAIN_SEEDS) == 24
    for blocked in (1_100_000, 1_200_000, 1_300_000, 1_400_000):
        assert blocked not in CONTRAST_TRAIN_SEEDS
    source = Path("fiber/classify/contrast.py").read_text(encoding="utf-8")
    assert "witnesses.json" not in source
    assert "4-early-quiet" not in source
    train = Path("fiber/classify/contrast_train.py").read_text(encoding="utf-8")
    assert "decide_witnesses(" in train


def test_painted_contrast_labels_are_the_generator_class():
    scene = plant_contrast(1_500_007)
    assert scene["label_source"] == "generator"
    for event in scene["events"]:
        mask = scene["event_id"] == event["id"]
        assert mask.any()
        assert np.all(scene["labels"][mask] == LABEL_INDEX[event["label"]])
    assert not scene["labels"][~scene["valid"]].any()
    again = plant_contrast(1_500_007)
    assert np.array_equal(scene["field"], again["field"])
    assert np.array_equal(scene["labels"], again["labels"])


def test_local_excess_is_isolated_and_a_short_burst_is_not():
    scene = plant_contrast(1_500_003)
    origins, truth = labeled_windows(scene["labels"], scene["valid"])
    feats = contrast_table(scene["field"], scene["valid"], origins)
    excess = feats[:, CONTRAST_FEATURES.index("excess")]
    shoulder = feats[:, CONTRAST_FEATURES.index("shoulder")]
    contrast = feats[:, CONTRAST_FEATURES.index("contrast")]
    duty = feats[:, CONTRAST_FEATURES.index("duty")]
    heat = np.array([name == "Local_Heating" for name in truth])
    assert heat.any()
    assert float(np.median(excess[heat])) > 1.5
    assert float(np.median(contrast[heat])) > 1.2
    assert float(np.median(duty[heat])) >= 0.89
    full = (
        (origins[:, 1] >= 400)
        & (origins[:, 1] <= 440)
        & (duty >= 0.89)
    )
    assert int(full.sum()) >= 3
    assert float(np.median(np.abs(contrast[full]))) < CONTRAST_ABS_P50
    assert float(np.median(np.abs(shoulder[full]))) >= 1.5
    assert all(truth[int(index)] == "Background" for index in np.flatnonzero(full))
    decoy = (origins[:, 1] >= 540) & (origins[:, 1] <= 600)
    assert decoy.any()
    assert float(np.median(np.abs(excess[decoy]))) < 0.5
    assert float(np.max(duty[decoy])) <= (CONTRAST_DECOY_ROWS + 1) / TIME_CROP


def test_one_crop_holds_at_most_one_contrast_event():
    for seed in (1_500_000, 1_500_001, 1_500_002):
        scene = plant_contrast(seed)
        columns = []
        for x in range(scene["event_id"].shape[1]):
            columns.append({int(item) for item in np.unique(scene["event_id"][:, x]) if int(item) != 0})
        for x0 in range(0, scene["event_id"].shape[1] - SPACE_CROP + 1):
            union: set[int] = set()
            for x in range(x0, x0 + SPACE_CROP):
                union |= columns[x]
            assert len(union) <= 1


def test_contrast_background_split_fills():
    features, target = _balanced()
    assert features.shape == (6 * 128, len(CONTRAST_FEATURES))
    assert int(target.shape[0]) == 6 * 128
    assert set(int(item) for item in target) == set(range(6))
