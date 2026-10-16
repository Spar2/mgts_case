"""Smoke: a matrix-profile discord score on a small field."""

import numpy as np
import pytest


def test_discord_score_matches_the_field():
    pytest.importorskip("stumpy")
    from fiber.anomaly.matrix_profile import discord_score

    rng = np.random.default_rng(0)
    field = rng.normal(0.0, 0.2, size=(30, 4)).astype(np.float32)
    field[12:16, 1] += 5.0
    valid = np.ones(field.shape, dtype=bool)
    score = discord_score(field, valid, window=6)
    assert score.shape == field.shape
    assert np.all(score[~valid] == 0)
