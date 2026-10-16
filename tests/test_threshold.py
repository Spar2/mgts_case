"""Percentile and peaks-over-threshold rates on standard normal noise."""

import numpy as np

from fiber.anomaly.threshold import holdout_report


def test_exceedance_stays_near_the_requested_risk():
    risk = 0.05
    values = np.random.default_rng(0).normal(0.0, 1.0, size=20000)
    report = holdout_report(values, risk, seed=1)
    for key in ("percentile_exceedance", "pot_exceedance"):
        rate = report[key]
        assert abs(rate - risk) <= 0.5 * risk
    assert report["operational_threshold_unchanged"] == 3.5
