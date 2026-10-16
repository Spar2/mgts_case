import numpy as np

from fiber.classify.lgb_context import LGB_LEARNING_RATE, LGB_LEAVES, LGB_MIN_CHILD_SAMPLES, LGB_TREES
from fiber.classify.rules import LABELS


def test_lgb_size_is_frozen():
    assert LGB_TREES == 200
    assert LGB_LEAVES == 15
    assert LGB_LEARNING_RATE == 0.05
    assert LGB_MIN_CHILD_SAMPLES == 20


def test_lgb_roundtrip_returns_known_names():
    pytest = __import__("pytest")
    pytest.importorskip("lightgbm")
    from fiber.classify.lgb_context import fit_context_lgb, predict_context_lgb

    rng = np.random.default_rng(0)
    rows = [rng.normal(class_id * 3.0, 0.05, size=(12, 5)) for class_id in range(len(LABELS))]
    target = np.repeat(np.arange(len(LABELS)), 12)
    model = fit_context_lgb(np.vstack(rows), target)
    names = predict_context_lgb(model.booster_, np.vstack(rows)[: len(LABELS)])
    assert len(names) == len(LABELS)
    assert set(names) <= set(LABELS)
