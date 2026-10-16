import numpy as np

from fiber.classify.context import (
    CONTEXT_PEAK_FRACTION,
    CONTEXT_SIDE,
    FEATURE_NAMES,
    WIDE_COLUMNS,
    context_table,
    wide_context_crop,
    wide_context_crops,
    window_context,
)
from fiber.classify.plant import PERSISTENT_WIDTH, plant_long
from fiber.classify.scan import labeled_windows
from fiber.models.patch_classifier import SPACE_CROP, TIME_CROP, crop_deviation


def test_context_constants_stay_put():
    assert CONTEXT_SIDE == 40
    assert CONTEXT_PEAK_FRACTION == 0.5
    assert FEATURE_NAMES == ("support", "peak", "n_half", "spike", "runs")


def test_wide_support_separates_a_streak_from_a_broad_edge():
    cap = PERSISTENT_WIDTH[1] - 1
    for seed in range(4):
        scene = plant_long(seed, profile=True, persistent=True)
        origins, truth = labeled_windows(scene["labels"], scene["valid"])
        feats = context_table(scene["field"], scene["valid"], origins)
        assert feats.shape == (len(truth), len(FEATURE_NAMES))
        seen = set()
        local = []
        broad = []
        for row, name in zip(feats, truth, strict=True):
            seen.add(name)
            if name == "Local_Heating":
                assert row[0] <= cap
                assert row[1] > 0
                local.append(float(row[0]))
            elif name == "Local_Cooling":
                assert row[0] <= cap
                assert row[1] < 0
                local.append(float(row[0]))
            elif name == "Broad_Shift":
                broad.append(float(row[0]))
            elif name == "Sensor_Fault":
                assert row[2] == 1
        assert {"Local_Heating", "Local_Cooling", "Broad_Shift", "Sensor_Fault"} <= seen
        assert local and broad
        assert max(local) < min(broad)


def test_table_matches_one_window_and_accepts_an_empty_list():
    scene = plant_long(0, profile=True, persistent=True)
    origins, _truth = labeled_windows(scene["labels"], scene["valid"])
    table = context_table(scene["field"], scene["valid"], origins[:5])
    for index, (t0, x0) in enumerate(origins[:5]):
        alone = window_context(scene["field"], scene["valid"], int(t0), int(x0))
        assert np.allclose(table[index], alone)
    assert context_table(scene["field"], scene["valid"], np.zeros((0, 2), dtype=np.int64)).shape == (0, 5)


def test_wide_crop_matches_the_center_and_drops_the_dead_tail():
    assert WIDE_COLUMNS == CONTEXT_SIDE + SPACE_CROP + CONTEXT_SIDE
    assert WIDE_COLUMNS == 120
    scene = plant_long(1, profile=True, persistent=True)
    origins, _truth = labeled_windows(scene["labels"], scene["valid"])
    wide = wide_context_crops(scene["field"], scene["valid"], origins)
    assert wide.shape == (len(origins), TIME_CROP, WIDE_COLUMNS)
    for (t0, x0), image in zip(origins, wide, strict=True):
        center = crop_deviation(scene["field"], scene["valid"], int(t0), int(x0))
        assert np.allclose(image[:, CONTEXT_SIDE:CONTEXT_SIDE + SPACE_CROP], center)
    on_edge = [index for index, (_t0, x0) in enumerate(origins) if int(x0) == 0]
    assert on_edge
    assert np.all(wide[on_edge[0], :, :CONTEXT_SIDE] == 0)
    # Dead tail starts at column 330. A crop at 320 has wings from 280, so
    # image column 50 is cable column 330.
    tail = wide_context_crop(scene["field"], scene["valid"], 0, 320)
    assert np.all(tail[:, 50:] == 0)
    assert wide_context_crops(scene["field"], scene["valid"], np.zeros((0, 2), dtype=np.int64)).shape == (0, TIME_CROP, WIDE_COLUMNS)


def test_patch_net_accepts_the_wide_crop():
    torch = __import__("pytest").importorskip("torch")
    from fiber.models.patch_classifier import PatchClassifier

    out = PatchClassifier()(torch.zeros(2, 1, TIME_CROP, WIDE_COLUMNS))
    assert tuple(out.shape) == (2, 6)
