"""Three measured arms. A standing spike and a flat noisy crop stay background."""

from pathlib import Path

import numpy as np
import pytest

from fiber.classify.plant import LABEL_INDEX, SoftmaxSceneModel
from fiber.classify.regime import (
    REGIME_BUMP_X,
    REGIME_FEATURES,
    REGIME_HEAT_C,
    REGIME_HEAT_X,
    REGIME_LONG_AC1,
    REGIME_LONG_C,
    REGIME_LONG_COLD,
    REGIME_LONG_ROWS,
    REGIME_LONG_X,
    REGIME_NOISY_AC1,
    REGIME_NOISY_C,
    REGIME_NOISY_MAD,
    REGIME_NOISY_PEAK,
    REGIME_NOISY_X,
    REGIME_ON,
    REGIME_SPIKE_X,
    REGIME_STEPS,
    REGIME_TRAIN_SEEDS,
    REGIME_WARM_AC1,
    REGIME_WARM_EXCESS,
    REGIME_WARM_Z,
    dump_regime_model,
    load_regime_model,
    plant_regime,
    regime_mask,
    regime_names,
    regime_table,
)
from fiber.classify.regime_train import _balanced
from fiber.classify.scan import labeled_windows
from fiber.models.patch_classifier import SPACE_CROP, TIME_CROP


def _col(feats: np.ndarray, name: str) -> np.ndarray:
    return feats[:, REGIME_FEATURES.index(name)]


def _scene(seed: int):
    scene = plant_regime(seed)
    origins, truth = labeled_windows(scene["labels"], scene["valid"])
    feats = regime_table(scene["field"], scene["valid"], origins)
    return scene, origins, truth, feats


def _fitted() -> SoftmaxSceneModel:
    model = SoftmaxSceneModel()
    model.fit(
        np.zeros((6, len(REGIME_FEATURES)), dtype=np.float64),
        np.arange(6, dtype=np.int64),
        steps=1,
    )
    return model


def _inside(origins: np.ndarray, truth: list[str]) -> dict[str, np.ndarray]:
    space = origins[:, 1]
    time = origins[:, 0]
    heat = (
        (space <= REGIME_HEAT_X[0])
        & (space + SPACE_CROP >= REGIME_HEAT_X[1])
        & (time >= REGIME_ON[0])
        & (time + TIME_CROP <= REGIME_ON[1])
    )
    long = (
        (space <= REGIME_LONG_X[0])
        & (space + SPACE_CROP >= REGIME_LONG_X[1])
        & (time + TIME_CROP <= REGIME_LONG_ROWS)
    )
    bump = (space >= REGIME_BUMP_X[0] - 10) & (space <= REGIME_BUMP_X[0])
    flat = (space >= REGIME_BUMP_X[1] + 14) & (space + SPACE_CROP <= REGIME_NOISY_X[1])
    spike = (space >= REGIME_SPIKE_X[0] - 30) & (space <= REGIME_SPIKE_X[0])
    names = np.asarray(truth)
    return {
        "heat": heat & (names == "Local_Heating"),
        "long": long & (names == "Local_Heating"),
        "bump": bump & (names == "Local_Heating"),
        "flat": flat & (names == "Background"),
        "spike": spike & (names == "Background"),
    }


def test_regime_recipe_is_frozen():
    assert REGIME_FEATURES[-7:] == ("excess", "ac1", "z", "mad_peak", "ac1_peak", "cold", "ac1_cold")
    assert len(REGIME_FEATURES) == 12
    assert REGIME_HEAT_C == (2.78, 4.50)
    assert REGIME_LONG_C == (2.80, 4.50)
    assert REGIME_NOISY_C == 0.93
    assert REGIME_WARM_EXCESS == 2.19
    assert REGIME_WARM_AC1 == 0.81
    assert REGIME_WARM_Z == 3.6
    assert REGIME_NOISY_MAD == 0.50
    assert REGIME_NOISY_AC1 == 0.32
    assert REGIME_NOISY_PEAK == 0.53
    assert REGIME_LONG_AC1 == 0.81
    assert REGIME_LONG_COLD == 2.16
    assert REGIME_NOISY_X == (300, 400)
    assert REGIME_TRAIN_SEEDS[0] == 2_100_000
    assert len(REGIME_TRAIN_SEEDS) == 24
    for blocked in (
        1_100_000, 1_200_000, 1_300_000, 1_400_000, 1_500_000,
        1_600_000, 1_700_000, 1_800_000, 1_900_000, 2_000_000,
    ):
        assert blocked not in REGIME_TRAIN_SEEDS
    assert 2_101_000 not in REGIME_TRAIN_SEEDS
    source = Path("fiber/classify/regime.py").read_text(encoding="utf-8")
    assert "witnesses.json" not in source
    assert "4-early-quiet" not in source
    train = Path("fiber/classify/regime_train.py").read_text(encoding="utf-8")
    assert "decide_witnesses(" in train
    assert "regime_names(" in train


def test_regime_mask_matches_three_rows_and_rejects_an_unfitted_model():
    width = len(REGIME_FEATURES)
    table = np.zeros((5, width), dtype=np.float64)
    table[0, REGIME_FEATURES.index("excess")] = REGIME_WARM_EXCESS
    table[0, REGIME_FEATURES.index("ac1")] = REGIME_WARM_AC1
    table[0, REGIME_FEATURES.index("z")] = REGIME_WARM_Z
    table[1, REGIME_FEATURES.index("peak")] = REGIME_NOISY_PEAK
    table[1, REGIME_FEATURES.index("mad_peak")] = REGIME_NOISY_MAD
    table[1, REGIME_FEATURES.index("ac1_peak")] = REGIME_NOISY_AC1
    table[2, REGIME_FEATURES.index("cold")] = REGIME_LONG_COLD
    table[2, REGIME_FEATURES.index("ac1_cold")] = REGIME_LONG_AC1
    table[3, REGIME_FEATURES.index("peak")] = 3.0
    table[3, REGIME_FEATURES.index("mad_peak")] = 0.2
    table[3, REGIME_FEATURES.index("ac1_peak")] = 0.95
    table[3, REGIME_FEATURES.index("excess")] = 0.4
    table[3, REGIME_FEATURES.index("ac1")] = 0.95
    table[3, REGIME_FEATURES.index("z")] = 1.0
    table[3, REGIME_FEATURES.index("cold")] = 0.4
    table[3, REGIME_FEATURES.index("ac1_cold")] = 0.95
    table[4, REGIME_FEATURES.index("excess")] = REGIME_WARM_EXCESS - 0.01
    table[4, REGIME_FEATURES.index("ac1")] = REGIME_WARM_AC1
    table[4, REGIME_FEATURES.index("z")] = REGIME_WARM_Z
    assert regime_mask(table).tolist() == [True, True, True, False, False]
    assert regime_mask(np.zeros((0, width))).size == 0
    assert regime_names(_fitted(), table) == [
        "Local_Heating",
        "Local_Heating",
        "Local_Heating",
        "Background",
        "Background",
    ]
    with pytest.raises(RuntimeError, match="not fit"):
        regime_names(SoftmaxSceneModel(), table[:1])
    scene = plant_regime(2_100_000)
    empty = regime_table(scene["field"], scene["valid"], np.zeros((0, 2), dtype=np.int64))
    assert empty.shape == (0, 12)
    assert regime_names(_fitted(), empty) == []


def test_painted_regime_labels_are_the_generator_class():
    scene = plant_regime(2_100_004)
    assert scene["kind"] == "regime"
    assert scene["label_source"] == "generator"
    for event in scene["events"]:
        mask = scene["event_id"] == event["id"]
        assert mask.any()
        assert np.all(scene["labels"][mask] == LABEL_INDEX[event["label"]])
    assert not scene["labels"][~scene["valid"]].any()
    again = plant_regime(2_100_004)
    assert np.array_equal(scene["field"], again["field"])
    assert np.array_equal(scene["labels"], again["labels"])


def _assert_arms(seed: int, model: SoftmaxSceneModel) -> None:
    _scene_obj, origins, truth, feats = _scene(seed)
    groups = _inside(origins, truth)
    for name in ("heat", "long", "bump", "flat", "spike"):
        assert groups[name].any(), name
    mask = regime_mask(feats)
    excess = _col(feats, "excess")
    ac1 = _col(feats, "ac1")
    zee = _col(feats, "z")
    peak = _col(feats, "peak")
    mad_peak = _col(feats, "mad_peak")
    ac1_peak = _col(feats, "ac1_peak")
    cold = _col(feats, "cold")
    ac1_cold = _col(feats, "ac1_cold")
    heat = groups["heat"]
    long = groups["long"]
    bump = groups["bump"]
    assert bool(mask[heat].all())
    assert float(np.min(excess[heat])) >= REGIME_WARM_EXCESS
    assert float(np.min(ac1[heat])) >= REGIME_WARM_AC1
    assert float(np.min(zee[heat])) >= REGIME_WARM_Z
    warm = (excess >= REGIME_WARM_EXCESS) & (ac1 >= REGIME_WARM_AC1) & (zee >= REGIME_WARM_Z)
    assert bool(mask[long].all())
    assert not bool(warm[long].any())
    assert float(np.min(cold[long])) >= REGIME_LONG_COLD
    assert float(np.min(ac1_cold[long])) >= REGIME_LONG_AC1
    assert bool(mask[bump].all())
    assert float(np.min(mad_peak[bump])) >= REGIME_NOISY_MAD
    assert float(np.max(ac1_peak[bump])) <= REGIME_NOISY_AC1
    assert float(np.min(np.abs(peak[bump]))) >= REGIME_NOISY_PEAK
    assert not bool(mask[groups["flat"]].any())
    assert not bool(mask[groups["spike"]].any())
    names = regime_names(model, feats)
    assert all(names[int(index)] == "Background" for index in np.flatnonzero(groups["spike"]))
    assert all(names[int(index)] == "Background" for index in np.flatnonzero(groups["flat"]))
    assert all(names[int(index)] == "Local_Heating" for index in np.flatnonzero(heat))
    assert all(names[int(index)] == "Local_Heating" for index in np.flatnonzero(long))
    assert all(names[int(index)] == "Local_Heating" for index in np.flatnonzero(bump))


def test_planted_arms_clear_the_frozen_cuts():
    _assert_arms(2_100_003, _fitted())


def test_one_crop_holds_at_most_one_regime_event():
    for seed in (2_100_000, 2_100_001, 2_100_002):
        scene = plant_regime(seed)
        columns = []
        for x in range(scene["event_id"].shape[1]):
            columns.append({int(item) for item in np.unique(scene["event_id"][:, x]) if int(item) != 0})
        for x0 in range(0, scene["event_id"].shape[1] - SPACE_CROP + 1):
            union: set[int] = set()
            for x in range(x0, x0 + SPACE_CROP):
                union |= columns[x]
            assert len(union) <= 1


def test_regime_class_buckets_fill_and_the_model_keeps_the_mask():
    features, target = _balanced()
    assert features.shape == (6 * 128, len(REGIME_FEATURES))
    assert int(target.shape[0]) == 6 * 128
    assert set(int(item) for item in target) == set(range(6))
    model = SoftmaxSceneModel()
    model.fit(features, target, steps=REGIME_STEPS)
    blob = dump_regime_model(model, {"serve": "scorecard", "objective": "regime_witness"})
    scored = load_regime_model(blob)
    broken = dict(blob)
    broken["features"] = list(REGIME_FEATURES)[:-1]
    with pytest.raises(ValueError, match="feature list"):
        load_regime_model(broken)
    _assert_arms(2_101_000, scored)


def test_holdout_arms_fire_and_the_spike_stays_background():
    model = _fitted()
    for seed in range(2_101_000, 2_101_008):
        assert seed not in REGIME_TRAIN_SEEDS
        _assert_arms(seed, model)
