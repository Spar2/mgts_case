"""The residual planter keeps a hot stretch warm against the cold baseline."""

import numpy as np

from fiber.classify.residual import (
    EVAL_SEEDS,
    TRAIN_SEEDS,
    meter_crops,
    plant_residual,
    residual_field,
    residual_windows,
)
from fiber.classify.rules import LABELS


def test_train_and_eval_seeds_do_not_overlap():
    assert set(TRAIN_SEEDS).isdisjoint(EVAL_SEEDS)


def test_hot_block_stays_warm_in_the_residual():
    scene = plant_residual(3_000_000)
    residual = residual_field(scene["field"], scene["valid"])
    hot = scene["labels"][:, 40:54] == LABELS.index("Local_Heating")
    quiet = scene["labels"][:, 40:54] == 0
    assert hot.any() and quiet.any()
    assert float(np.nanmedian(residual[:, 40:54][hot])) > 2.0
    assert abs(float(np.nanmedian(residual[:, 40:54][quiet]))) < 1.5


def test_each_class_appears_and_one_window_holds_one_event():
    scene = plant_residual(3_000_003)
    _origins, names = residual_windows(scene["labels"], scene["valid"])
    assert set(names) == set(LABELS)
    crops = meter_crops(scene["field"], scene["valid"], scene["x_m"], np.asarray([[40, 40]], dtype=np.int64))
    assert crops.shape == (1, 2, 28, 64)
    assert float(crops[0, 1].max()) > 2.0
