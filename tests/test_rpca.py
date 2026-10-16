"""Robust PCA recovers a low-rank field and puts a planted spike in S."""

import numpy as np

from fiber.anomaly.rpca import decompose_rpca, rpca_score
from fiber.anomaly.zones import extract_zones


def test_low_rank_plus_spike():
    time = np.linspace(0.0, 1.0, 24)[:, None]
    distance = np.linspace(0.0, 1.0, 30)[None, :]
    low_rank = 20.0 + 2.0 * time + 0.5 * distance
    sparse = np.zeros_like(low_rank)
    sparse[6, 12] = 8.0
    observed = low_rank + sparse
    recovered, residual = decompose_rpca(observed, n_iter=100)
    relative = np.linalg.norm(recovered - low_rank) / np.linalg.norm(low_rank)
    assert relative < 0.1
    assert abs(residual[6, 12]) > 1.0
    assert abs(residual[6, 12]) == np.max(np.abs(residual))


def test_score_can_be_passed_to_extract_zones():
    field = np.full((16, 20), 21.0, dtype=np.float32)
    field[4:10, 8:12] += 6.0
    valid = np.ones(field.shape, dtype=bool)
    score, _low, _sparse = rpca_score(field, valid, n_iter=40)
    times = np.arange(field.shape[0], dtype=np.float64) * 60.0
    distance = np.arange(field.shape[1], dtype=np.float64)
    zones = extract_zones(
        score,
        valid,
        distance,
        [str(index) for index in range(field.shape[0])],
        times,
        thresh=3.5,
        min_width_m=3,
        min_profiles=6,
        min_cells=24,
    )
    assert isinstance(zones, list)
    assert score.shape == field.shape
    assert np.all(score[~valid] == 0)
