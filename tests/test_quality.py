import numpy as np

from fiber.quality import SENTINEL_VALUE, quality_mask


def test_sentinel_and_finite():
    raw = np.array([[25.0, SENTINEL_VALUE, np.nan, 18.5]], dtype=np.float32)
    mask = quality_mask(raw)
    assert mask["valid"].tolist() == [[True, False, False, True]]
    assert mask["is_sentinel"].tolist() == [[False, True, False, False]]
    assert mask["is_nonfinite"].tolist() == [[False, False, True, False]]


def test_declared_tail():
    raw = np.array([[20.0, 21.0, 22.0, SENTINEL_VALUE]], dtype=np.float32)
    mask = quality_mask(raw, point_counts=np.array([2], dtype=np.int32))
    assert mask["valid"].tolist() == [[True, True, False, False]]
    assert mask["is_tail"][0, 2]


def test_minus_200_is_dead():
    raw = np.array([[-200.0, -199.9]], dtype=np.float32)
    mask = quality_mask(raw)
    assert mask["valid"].tolist() == [[False, True]]
