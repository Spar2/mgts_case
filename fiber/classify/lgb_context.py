"""LightGBM on the frozen context features.

Tree size is fixed before the held-out search. The features stay the five
numbers already used by the logistic model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fiber.classify.context import FEATURE_NAMES
from fiber.classify.rules import LABELS

LGB_TREES = 200
LGB_LEAVES = 15
LGB_LEARNING_RATE = 0.05
LGB_MIN_CHILD_SAMPLES = 20


def fit_context_lgb(features: np.ndarray, target: np.ndarray):
    import lightgbm as lgb

    matrix = np.asarray(features, dtype=np.float64)
    labels = np.asarray(target, dtype=np.int64)
    if matrix.shape[1] != len(FEATURE_NAMES):
        raise ValueError("context feature width does not match LightGBM")
    if set(int(item) for item in np.unique(labels)) != set(range(len(LABELS))):
        raise ValueError("LightGBM training rows must include every class")
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(LABELS),
        n_estimators=LGB_TREES,
        learning_rate=LGB_LEARNING_RATE,
        num_leaves=LGB_LEAVES,
        min_child_samples=LGB_MIN_CHILD_SAMPLES,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=0,
        n_jobs=1,
        verbosity=-1,
        deterministic=True,
    )
    model.fit(matrix, labels, feature_name=list(FEATURE_NAMES))
    return model


def save_context_lgb(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(path))


def load_context_lgb(path: Path):
    import lightgbm as lgb

    booster = lgb.Booster(model_file=str(path))
    if tuple(booster.feature_name()) != FEATURE_NAMES:
        raise ValueError("LightGBM feature list does not match this code")
    return booster


def predict_context_lgb(booster, features: np.ndarray) -> list[str]:
    prob = np.asarray(booster.predict(np.asarray(features, dtype=np.float64)))
    if prob.ndim != 2 or prob.shape[1] != len(LABELS):
        raise RuntimeError(f"expected class probabilities, got shape {prob.shape}")
    index = np.argmax(prob, axis=1)
    return [LABELS[int(item)] for item in index]
