"""Pinned evidence, access isolation and read-only research API."""

import json

import pytest
from fastapi.testclient import TestClient
from test_platform import login
from test_platform import pg as platform_pg
from test_workbench import fixture as native_fixture
from workbench import experiments
from workbench.accounts import add_user
from workbench.api import create_app

pg = platform_pg


def test_pack_is_compact_and_keeps_metric_scope():
    experiments.read_pack.cache_clear()
    pack = experiments.read_pack()
    assert len(pack.calibration["rows"]) == 9
    assert pack.source_commit == "237168f4d4a3e11260c9f692880cc477cd7af779"
    assert pack.audit.refinement_operational is False
    assert pack.audit.neural_cuda_reproduced is False
    assert pack.operational_pair["threshold"] == 3.5
    assert "crop_fine" not in experiments.PACK.read_text(encoding="utf-8")
    assert experiments.PACK.stat().st_size < 100000


def test_tamper_is_rejected(tmp_path, monkeypatch):
    modified = tmp_path / "bad.json"
    modified.write_text(
        json.dumps({"operational_pair": {"threshold": 4.85}}), encoding="utf-8"
    )
    monkeypatch.setattr(experiments, "PACK", modified)
    experiments.read_pack.cache_clear()
    with pytest.raises(ValueError, match="integrity"):
        experiments.read_pack()
    experiments.read_pack.cache_clear()


def test_research_access_isolated_and_never_promotes(pg, tmp_path):
    root, queue, _ = native_fixture.__wrapped__(tmp_path)
    add_user(pg, "research-reader", "QA-only-password-123456", "Reader", "viewer")
    with pg.connect() as db:
        db.execute("INSERT INTO projects VALUES('other','Other')")
    add_user(pg, "other-reader", "QA-only-password-123456", "Other", "viewer", "other")
    app = create_app(root / "records", queue, root / "runtime")
    with TestClient(app) as client:
        assert client.get("/api/v2/ml-experiments").status_code == 401
        headers = login(client, "research-reader")
        response = client.get("/api/v2/ml-experiments")
        assert response.status_code == 200, response.text
        assert response.json()["operational_pair"]["threshold"] == 3.5
        assert (
            client.post(
                "/api/v2/ml-experiments", headers=headers, json={"threshold": 4.85}
            ).status_code
            == 405
        )
        client.headers["X-DTS-Project"] = "other"
        assert (
            client.post(
                "/api/v2/auth/login",
                json={
                    "username": "other-reader",
                    "password": "QA-only-password-123456",
                },
            ).status_code
            == 200
        )
        assert client.get("/api/v2/ml-experiments").status_code == 403
