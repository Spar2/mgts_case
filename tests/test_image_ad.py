"""Smoke: Anomalib Patchcore score on a small field."""

import numpy as np
import pytest


def test_image_anomaly_score_matches_the_field():
    pytest.importorskip("anomalib")
    pytest.importorskip("torch")
    from fiber.anomaly.image_ad import image_anomaly_score

    rng = np.random.default_rng(0)
    field = rng.normal(21.0, 0.15, size=(32, 32)).astype(np.float32)
    valid = np.ones(field.shape, dtype=bool)
    distance = np.arange(field.shape[1], dtype=np.float64)
    score = image_anomaly_score(field, valid, distance, tile=16, device="cpu")
    assert score.shape == field.shape
    assert np.all(score[~valid] == 0)


def test_minority_hotspot_is_scored_against_quiet_tiles():
    """A hotspot in less than half a tile must not sit in the memory.

    Median |Z| on that tile stays below the gate, which used to label it
    normal. The query was then the memory, and the distance was zero.
    """
    pytest.importorskip("anomalib")
    pytest.importorskip("torch")
    from fiber.anomaly.image_ad import image_anomaly_score

    rng = np.random.default_rng(0)
    field = np.full((32, 32), 21.0, np.float32)
    field += rng.normal(0.0, 0.1, size=field.shape).astype(np.float32)
    field[16:20, 16:20] += 8.0
    valid = np.ones(field.shape, dtype=bool)
    distance = np.arange(field.shape[1], dtype=np.float64)
    score = image_anomaly_score(field, valid, distance, tile=16, device="cpu")
    hot = np.abs(score[16:32, 16:32])
    quiet = np.abs(score[0:16, 0:16])
    assert np.isfinite(score).all()
    assert float(hot.max()) > 0.0
    assert float(quiet.max()) == 0.0
