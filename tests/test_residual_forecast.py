"""Smoke: Chronos-Bolt exceedance on a short field."""

import numpy as np
import pytest


def test_forecast_score_matches_the_field():
    pytest.importorskip("chronos")
    pytest.importorskip("torch")
    from fiber.anomaly.residual_forecast import forecast_score

    rng = np.random.default_rng(0)
    field = rng.normal(20.0, 0.1, size=(20, 2)).astype(np.float32)
    valid = np.ones(field.shape, dtype=bool)
    score = forecast_score(field, valid, context=8, device="cpu", column_stride=1)
    assert score.shape == field.shape
    assert np.all(score[~valid] == 0)


def test_each_time_uses_its_own_emitted_interval(monkeypatch):
    """A one-step interval must not be pasted onto every later time.

    The spike sits inside the nominal context. Its history has to end before
    the spike, and a flat future after a ramp must not share one interval.
    Bolt's trained grid replaces the requested 0.05/0.95 with 0.1/0.9.
    """
    pytest.importorskip("chronos")
    pytest.importorskip("torch")
    import chronos

    from fiber.anomaly.residual_forecast import forecast_score

    seen = []
    original = chronos.BaseChronosPipeline.from_pretrained

    def wrapped(name, *args, **kwargs):
        pipe = original(name, *args, **kwargs)
        real = pipe.predict_quantiles

        def spy(inputs, prediction_length=None, quantile_levels=None, **kw):
            seen.append(
                {
                    "prediction_length": prediction_length,
                    "quantile_levels": [float(level) for level in quantile_levels],
                    "inputs": inputs.detach().cpu().numpy().copy(),
                }
            )
            return real(
                inputs,
                prediction_length=prediction_length,
                quantile_levels=quantile_levels,
                **kw,
            )

        pipe.predict_quantiles = spy
        return pipe

    monkeypatch.setattr(chronos.BaseChronosPipeline, "from_pretrained", wrapped)

    field = np.full((18, 1), 20.0, np.float32)
    field[6, 0] = 45.0
    valid = np.ones(field.shape, dtype=bool)
    score = forecast_score(field, valid, context=12, device="cpu")
    assert score.shape == field.shape
    assert float(score[6, 0]) > 0.0
    assert np.all(score[:4, 0] == 0.0)
    histories = [item["inputs"] for item in seen if item["inputs"].shape[-1] == 6]
    assert histories
    assert all(float(np.max(np.abs(batch - 20.0))) < 1e-3 for batch in histories)
    assert all(item["quantile_levels"] == [0.1, 0.9] for item in seen)
    assert all(item["prediction_length"] == 1 for item in seen)
    low, high = forecast_score.emitted_quantile_levels
    assert low == pytest.approx(0.1)
    assert high == pytest.approx(0.9)

    seen.clear()
    ramp = np.full((16, 1), 33.0, np.float32)
    ramp[:8, 0] = 20.0 + np.arange(8, dtype=np.float32)
    ramp_score = forecast_score(ramp, np.ones_like(ramp, dtype=bool), context=8, device="cpu")
    future = ramp_score[8:, 0]
    assert not np.allclose(future, future[0])
