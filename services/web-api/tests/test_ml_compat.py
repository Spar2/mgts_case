"""Compatibility uses the frozen CPU implementation, never weights or training."""

from fastapi.testclient import TestClient
from workbench.ml_compat import app
from workbench.analysis_compute import ML_COMMIT


def test_profile_preserves_mask_and_declares_real_implementation():
    with TestClient(app) as client:
        assert client.get("/health").json()["implementation_sha"] == ML_COMMIT
        result = client.post(
            "/denoise/profile",
            json={
                "raw": [20, 99, 20, -300, 22],
                "distances": [0, 1, 2, 3, 4],
                "method": "median_x3",
            },
        )
        assert result.status_code == 200
        assert result.json()["denoised"] == [59.5, 20.0, 59.5, None, 22.0]
        assert result.json()["training_origin"] == "untrained"
        assert (
            client.post(
                "/denoise/profile", json={"raw": [1, 2], "distances": [1, 1]}
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/denoise/profile",
                json={"raw": [1, 2], "distances": [0, 1], "method": "n2n_tsr"},
            ).status_code
            == 422
        )


def test_field_keeps_unknown_when_measurements_unavailable():
    with TestClient(app) as client:
        result = client.post(
            "/detect/field",
            json={
                "raw": [[None, None], [None, None]],
                "distances": [0, 1],
                "time_s": [0, 60],
            },
        )
        assert (
            result.status_code == 200 and result.json()["anomaly_status"] == "Unknown"
        )
        assert "clean" not in result.json() and "z" not in result.json()
