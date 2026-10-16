"""LightGBM on the context features plus two profile jumps.

Tree size stays the one checked on design seeds 100-103. The five context
features stay the published list. max_jump and second_jump are the two largest
steps between neighbouring column medians, counted only where both columns
are finite. They were checked on seeds 100-103 after a five-feature model
had already been scored once on seeds 600000-600003. This model's held-out
seeds start at 700000 and do not include either of those sets.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fiber.classify.context import (
    FEATURE_NAMES,
    _column_medians,
    _features,
    _window_arrays,
)
from fiber.classify.rules import LABELS
from fiber.models.patch_classifier import TIME_CROP

AGREE_TREES = 300
AGREE_LEAVES = 31
AGREE_LEARNING_RATE = 0.08
AGREE_MIN_CHILD_SAMPLES = 40
AGREE_TRAIN_SEEDS = tuple(seed for seed in range(200) if seed not in range(100, 104))
AGREE_EVAL_SEEDS = tuple(range(700_000, 700_008))
AGREE_FEATURE_NAMES = FEATURE_NAMES + ("max_jump", "second_jump")


def _jump_pair(prof: np.ndarray) -> tuple[float, float]:
    """Largest and second-largest absolute step along one window profile."""
    if prof.size < 2:
        return 0.0, 0.0
    finite = np.isfinite(prof)
    both = finite[:-1] & finite[1:]
    if not bool(both.any()):
        return 0.0, 0.0
    diff = np.abs(np.diff(np.asarray(prof, dtype=np.float64)))
    diff[~both] = 0.0
    if diff.size == 1:
        return float(diff[0]), 0.0
    top = np.partition(diff, -2)[-2:]
    return float(top.max()), float(top.min())


def agree_table(field: np.ndarray, valid: np.ndarray, origins: np.ndarray) -> np.ndarray:
    """One row of `AGREE_FEATURE_NAMES` per origin. Empty input has shape (0, 7)."""
    origins = np.asarray(origins, dtype=np.int64)
    out = np.zeros((len(origins), len(AGREE_FEATURE_NAMES)), dtype=np.float64)
    if len(origins) == 0:
        return out
    field64 = np.asarray(field, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    order = np.argsort(origins[:, 0], kind="mergesort")
    cached_t = None
    col_med = None
    jump_at = len(FEATURE_NAMES)
    for index in order:
        t0 = int(origins[index, 0])
        x0 = int(origins[index, 1])
        if cached_t != t0:
            col_med = _column_medians(field64[t0:t0 + TIME_CROP], mask[t0:t0 + TIME_CROP])
            cached_t = t0
        prof, cell_dev = _window_arrays(field64, mask, t0, x0, col_med)
        out[index, :jump_at] = _features(prof, cell_dev)
        out[index, jump_at:] = _jump_pair(prof)
    return out


def fit_agree_model(features: np.ndarray, target: np.ndarray):
    import lightgbm as lgb

    matrix = np.asarray(features, dtype=np.float64)
    labels = np.asarray(target, dtype=np.int64)
    if matrix.shape[1] != len(AGREE_FEATURE_NAMES):
        raise ValueError("agree feature width does not match the agree model")
    if set(int(item) for item in np.unique(labels)) != set(range(len(LABELS))):
        raise ValueError("agree-model training rows must include every class")
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(LABELS),
        n_estimators=AGREE_TREES,
        learning_rate=AGREE_LEARNING_RATE,
        num_leaves=AGREE_LEAVES,
        min_child_samples=AGREE_MIN_CHILD_SAMPLES,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=0,
        n_jobs=1,
        verbosity=-1,
        deterministic=True,
    )
    model.fit(matrix, labels, feature_name=list(AGREE_FEATURE_NAMES))
    return model


def load_agree_model(path: Path):
    import lightgbm as lgb

    booster = lgb.Booster(model_file=str(path))
    if tuple(booster.feature_name()) != AGREE_FEATURE_NAMES:
        raise ValueError("agree-model feature list does not match this code")
    return booster
