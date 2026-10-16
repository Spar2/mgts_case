"""The warm tail is heating. A standing peak in the bulk stays background."""

from pathlib import Path

import numpy as np

from fiber.classify.bulk import (
    BULK_DECOY_PEAK,
    BULK_FEATURES,
    BULK_HEAT_C,
    BULK_STEPS,
    BULK_TRAIN_SEEDS,
    BULK_WARM_AC1,
    BULK_WARM_EXCESS,
    BULK_WARM_Z,
    bulk_names,
    bulk_table,
    plant_bulk,
    warm_tail_mask,
)
from fiber.classify.bulk_train import _balanced
from fiber.classify.plant import LABEL_INDEX, SoftmaxSceneModel
from fiber.classify.scan import labeled_windows
from fiber.models.patch_classifier import SPACE_CROP


def test_bulk_recipe_is_frozen():
    assert BULK_FEATURES[-3:] == ("excess", "ac1", "z")
    assert len(BULK_FEATURES) == 8
    assert BULK_HEAT_C == (2.78, 4.50)
    assert BULK_WARM_EXCESS == 2.19
    assert BULK_WARM_AC1 == 0.81
    assert BULK_WARM_Z == 3.6
    assert BULK_DECOY_PEAK == 2.79
    assert BULK_TRAIN_SEEDS[0] == 2_000_000
    assert len(BULK_TRAIN_SEEDS) == 24
    for blocked in (
        1_100_000, 1_200_000, 1_300_000, 1_400_000, 1_500_000,
        1_600_000, 1_700_000, 1_800_000, 1_900_000,
    ):
        assert blocked not in BULK_TRAIN_SEEDS
    source = Path("fiber/classify/bulk.py").read_text(encoding="utf-8")
    assert "witnesses.json" not in source
    assert "4-early-quiet" not in source
    train = Path("fiber/classify/bulk_train.py").read_text(encoding="utf-8")
    assert "decide_witnesses(" in train
    assert "bulk_names(" in train


def test_off_tail_names_are_background():
    class _Hot:
        def predict(self, features):
            return ["Local_Heating"] * len(features)

    table = np.zeros((2, len(BULK_FEATURES)), dtype=np.float64)
    excess = BULK_FEATURES.index("excess")
    ac1 = BULK_FEATURES.index("ac1")
    zee = BULK_FEATURES.index("z")
    table[0, excess] = 0.4
    table[0, ac1] = 0.95
    table[0, zee] = 8.0
    table[1, excess] = 3.0
    table[1, ac1] = 0.95
    table[1, zee] = 8.0
    assert bulk_names(_Hot(), table) == ["Background", "Local_Heating"]
    assert warm_tail_mask(np.zeros((0, len(BULK_FEATURES)))).size == 0


def test_painted_bulk_labels_are_the_generator_class():
    scene = plant_bulk(2_000_007)
    assert scene["label_source"] == "generator"
    for event in scene["events"]:
        mask = scene["event_id"] == event["id"]
        assert mask.any()
        assert np.all(scene["labels"][mask] == LABEL_INDEX[event["label"]])
    assert not scene["labels"][~scene["valid"]].any()
    again = plant_bulk(2_000_007)
    assert np.array_equal(scene["field"], again["field"])
    assert np.array_equal(scene["labels"], again["labels"])


def test_heat_is_in_the_warm_tail_and_a_standing_peak_stays_in_the_bulk():
    scene = plant_bulk(2_000_003)
    origins, truth = labeled_windows(scene["labels"], scene["valid"])
    feats = bulk_table(scene["field"], scene["valid"], origins)
    excess = feats[:, BULK_FEATURES.index("excess")]
    ac1 = feats[:, BULK_FEATURES.index("ac1")]
    zee = feats[:, BULK_FEATURES.index("z")]
    peak = feats[:, BULK_FEATURES.index("peak")]
    tail = warm_tail_mask(feats)
    heat = np.array([name == "Local_Heating" for name in truth])
    broad = np.array([name == "Broad_Shift" for name in truth])
    assert heat.any()
    assert float(np.median(excess[heat])) > BULK_WARM_EXCESS
    assert float(np.median(ac1[heat])) > BULK_WARM_AC1
    assert float(np.median(zee[heat])) > BULK_WARM_Z
    assert bool(np.median(tail[heat]) == 1.0)
    assert broad.any()
    assert not bool(tail[broad].any())
    stable = (origins[:, 1] >= 420) & (origins[:, 1] <= 440)
    assert stable.any()
    assert not bool(tail[stable].any())
    assert float(np.median(np.abs(excess[stable]))) < 0.8
    assert all(truth[int(index)] == "Background" for index in np.flatnonzero(stable))
    spike = (origins[:, 1] >= 500) & (origins[:, 1] <= 510)
    assert spike.any()
    assert not bool(tail[spike].any())
    assert float(np.median(np.abs(peak[spike]))) > 2.0
    assert float(np.median(np.abs(excess[spike]))) < 1.0
    assert all(truth[int(index)] == "Background" for index in np.flatnonzero(spike))


def test_one_crop_holds_at_most_one_bulk_event():
    for seed in (2_000_000, 2_000_001, 2_000_002):
        scene = plant_bulk(seed)
        columns = []
        for x in range(scene["event_id"].shape[1]):
            columns.append({int(item) for item in np.unique(scene["event_id"][:, x]) if int(item) != 0})
        for x0 in range(0, scene["event_id"].shape[1] - SPACE_CROP + 1):
            union: set[int] = set()
            for x in range(x0, x0 + SPACE_CROP):
                union |= columns[x]
            assert len(union) <= 1


def test_bulk_class_buckets_fill_and_the_model_keeps_the_tail():
    features, target = _balanced()
    assert features.shape == (6 * 128, len(BULK_FEATURES))
    assert int(target.shape[0]) == 6 * 128
    assert set(int(item) for item in target) == set(range(6))
    model = SoftmaxSceneModel()
    model.fit(features, target, steps=BULK_STEPS)
    scene = plant_bulk(2_001_000)
    origins, truth = labeled_windows(scene["labels"], scene["valid"])
    feats = bulk_table(scene["field"], scene["valid"], origins)
    names = bulk_names(model, feats)
    heat = [index for index, name in enumerate(truth) if name == "Local_Heating" and bool(warm_tail_mask(feats[index]))]
    assert heat
    assert all(names[index] == "Local_Heating" for index in heat)
    stable = np.flatnonzero((origins[:, 1] >= 420) & (origins[:, 1] <= 440))
    spike = np.flatnonzero((origins[:, 1] >= 500) & (origins[:, 1] <= 510))
    assert stable.size and spike.size
    assert all(names[int(index)] == "Background" for index in stable)
    assert all(names[int(index)] == "Background" for index in spike)
