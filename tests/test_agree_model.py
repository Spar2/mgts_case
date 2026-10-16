import numpy as np

from fiber.classify.agree_model import (
    AGREE_EVAL_SEEDS,
    AGREE_FEATURE_NAMES,
    AGREE_LEARNING_RATE,
    AGREE_LEAVES,
    AGREE_MIN_CHILD_SAMPLES,
    AGREE_TRAIN_SEEDS,
    AGREE_TREES,
    agree_table,
)
from fiber.classify.context import CONTEXT_PEAK_FRACTION, CONTEXT_SIDE, FEATURE_NAMES
from fiber.classify.scan import HOT_PASS_FRACTION


def test_agree_split_and_tree_size_stay_put():
    assert AGREE_TREES == 300
    assert AGREE_LEAVES == 31
    assert AGREE_LEARNING_RATE == 0.08
    assert AGREE_MIN_CHILD_SAMPLES == 40
    assert AGREE_FEATURE_NAMES == FEATURE_NAMES + ("max_jump", "second_jump")
    assert AGREE_EVAL_SEEDS == tuple(range(700_000, 700_008))
    assert set(AGREE_TRAIN_SEEDS).isdisjoint(AGREE_EVAL_SEEDS)
    assert set(AGREE_TRAIN_SEEDS).isdisjoint(range(600_000, 600_004))
    assert set(AGREE_TRAIN_SEEDS).isdisjoint(range(100, 104))
    assert 100 not in AGREE_TRAIN_SEEDS
    assert 103 not in AGREE_TRAIN_SEEDS
    assert CONTEXT_SIDE == 40
    assert CONTEXT_PEAK_FRACTION == 0.5
    assert HOT_PASS_FRACTION == 0.3


def test_profile_jump_marks_one_step_and_ignores_a_flat_span():
    field = np.full((28, 200), 22.0, dtype=np.float64)
    valid = np.ones_like(field, dtype=bool)
    field[:, 100:] += 4.0
    origins = np.asarray([[0, 20], [0, 80]], dtype=np.int64)
    table = agree_table(field, valid, origins)
    empty = agree_table(field, valid, np.zeros((0, 2), dtype=np.int64))
    jump = AGREE_FEATURE_NAMES.index("max_jump")
    second = AGREE_FEATURE_NAMES.index("second_jump")
    assert table.shape == (2, len(AGREE_FEATURE_NAMES))
    assert empty.shape == (0, len(AGREE_FEATURE_NAMES))
    assert table[0, jump] < 0.5
    assert table[1, jump] > 3.0
    assert table[1, second] < 0.5
