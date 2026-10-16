"""Long plateaus keep a short burst out of the heating class."""

from pathlib import Path

import numpy as np

from fiber.classify.plant import LABEL_INDEX
from fiber.classify.plateau import (
    PLATEAU_DECOY_ROWS,
    PLATEAU_FEATURES,
    PLATEAU_HEAT_C,
    PLATEAU_HEAT_WIDTH_M,
    PLATEAU_ON,
    PLATEAU_TRAIN_SEEDS,
    plant_plateau,
    plateau_table,
)
from fiber.classify.scan import labeled_windows
from fiber.models.patch_classifier import SPACE_CROP


def test_plateau_recipe_is_frozen():
    assert PLATEAU_FEATURES[-6:] == ("excess", "hold_rows", "hold_m", "long_hold", "repeats", "many_repeats")
    assert len(PLATEAU_FEATURES) == 11
    assert PLATEAU_HEAT_C == (2.0, 2.9)
    assert PLATEAU_DECOY_ROWS == (7, 19)
    assert PLATEAU_ON == (56, 168)
    assert PLATEAU_HEAT_WIDTH_M == (4, 6)
    assert PLATEAU_TRAIN_SEEDS[0] == 1_400_000
    assert len(PLATEAU_TRAIN_SEEDS) == 24
    assert 1_300_000 not in PLATEAU_TRAIN_SEEDS
    assert 1_200_000 not in PLATEAU_TRAIN_SEEDS
    assert 1_100_000 not in PLATEAU_TRAIN_SEEDS
    source = Path("fiber/classify/plateau.py").read_text(encoding="utf-8")
    assert "witnesses.json" not in source
    assert "4-early-quiet" not in source
    train = Path("fiber/classify/plateau_train.py").read_text(encoding="utf-8")
    assert "decide_witnesses(" in train


def test_painted_plateau_labels_are_the_generator_class():
    scene = plant_plateau(1_400_007)
    assert scene["label_source"] == "generator"
    for event in scene["events"]:
        mask = scene["event_id"] == event["id"]
        assert mask.any()
        assert np.all(scene["labels"][mask] == LABEL_INDEX[event["label"]])
    assert not scene["labels"][~scene["valid"]].any()
    again = plant_plateau(1_400_007)
    assert np.array_equal(scene["field"], again["field"])
    assert np.array_equal(scene["labels"], again["labels"])


def test_long_plateau_holds_and_short_burst_does_not():
    scene = plant_plateau(1_400_003)
    origins, truth = labeled_windows(scene["labels"], scene["valid"])
    feats = plateau_table(scene["field"], scene["valid"], origins, scene["x_m"])
    excess = feats[:, PLATEAU_FEATURES.index("excess")]
    hold = feats[:, PLATEAU_FEATURES.index("hold_rows")]
    flag = feats[:, PLATEAU_FEATURES.index("long_hold")]
    heat = np.array([name == "Local_Heating" for name in truth])
    assert heat.any()
    assert float(np.median(excess[heat])) > 1.5
    assert float(np.median(hold[heat])) >= 56
    assert float(np.min(flag[heat])) == 1.0
    periodic = np.array([name == "Periodic_Thermal" for name in truth])
    repeats = feats[:, PLATEAU_FEATURES.index("repeats")]
    assert periodic.any()
    assert float(np.min(repeats[periodic])) >= 3.0
    decoy = (origins[:, 1] >= 390) & (origins[:, 1] <= 420)
    assert decoy.any()
    assert float(np.max(hold[decoy])) <= PLATEAU_DECOY_ROWS[-1]
    assert float(np.max(flag[decoy])) == 0.0


def test_one_crop_holds_at_most_one_plateau_event():
    for seed in (1_400_000, 1_400_001, 1_400_002):
        scene = plant_plateau(seed)
        columns = []
        for x in range(scene["event_id"].shape[1]):
            columns.append({int(item) for item in np.unique(scene["event_id"][:, x]) if int(item) != 0})
        for x0 in range(0, scene["event_id"].shape[1] - SPACE_CROP + 1):
            union: set[int] = set()
            for x in range(x0, x0 + SPACE_CROP):
                union |= columns[x]
            assert len(union) <= 1
