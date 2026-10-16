"""Research denoisers on the shipped synthetic_wide_case."""

import numpy as np
import pytest

from fiber.denoise.score import synthetic_wide_case


def _dead_finite(out: np.ndarray, case: dict) -> int:
    return int(np.isfinite(out[:, case["dead_from"]:]).sum())


def test_n2n_tsr_keeps_dead_cells_and_is_deterministic():
    pytest.importorskip("torch")
    from fiber.models.n2n_tsr import denoise_n2n_tsr

    case = synthetic_wide_case(seed=0)
    first = denoise_n2n_tsr(case["noisy"], case["valid"], seed=3, steps=2, device="cpu", weights=None)
    second = denoise_n2n_tsr(case["noisy"], case["valid"], seed=3, steps=2, device="cpu", weights=None)
    assert first.shape == case["noisy"].shape
    assert _dead_finite(first, case) == 0
    assert np.array_equal(np.isnan(first), np.isnan(second))
    assert np.allclose(first[case["valid"]], second[case["valid"]], atol=1e-5)


def test_blindspot_x_keeps_dead_cells_and_is_deterministic():
    pytest.importorskip("torch")
    from fiber.models.blindspot_x import denoise_blindspot_x

    case = synthetic_wide_case(seed=0)
    first = denoise_blindspot_x(case["noisy"], case["valid"], seed=3, steps=0, device="cpu", weights=None)
    second = denoise_blindspot_x(case["noisy"], case["valid"], seed=3, steps=0, device="cpu", weights=None)
    assert first.shape == case["noisy"].shape
    assert _dead_finite(first, case) == 0
    assert np.isfinite(first[case["valid"]]).all()
    assert np.array_equal(np.isnan(first), np.isnan(second))
    assert np.allclose(first[case["valid"]], second[case["valid"]], atol=1e-5)


def test_zsn2n_loss_ignores_the_sentinel():
    pytest.importorskip("torch")
    from fiber.models.zsn2n import denoise_zsn2n

    case = synthetic_wide_case(seed=1)
    out = denoise_zsn2n(case["noisy"], case["valid"], seed=1, steps=2, device="cpu", weights=None)
    assert out.shape == case["noisy"].shape
    assert _dead_finite(out, case) == 0
    assert np.isfinite(out[case["valid"]]).all()
