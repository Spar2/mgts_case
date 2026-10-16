"""Read-only evidence and confirmed monitoring; no fabricated measurement claims."""

from fastapi.testclient import TestClient
from test_investigations import setup
from test_platform import login
from test_platform import pg as pg  # noqa: PLC0414 - pytest fixture re-export
from workbench import laboratory
from workbench.analysis_jobs import analysis_beat


def test_product_direct_links_serve_client_without_exposing_api(
    pg, tmp_path, monkeypatch
):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("QA client shell", encoding="utf-8")
    monkeypatch.setenv("DTS_WEB_DIST", str(dist))
    app, _ = setup(pg, tmp_path)
    with TestClient(app) as client:
        for route in ("/works", "/works/" + "a" * 32, "/laboratory", "/monitoring"):
            response = client.get(route)
            assert response.status_code == 200
            assert response.text == "QA client shell"
        assert client.get("/api/v2/investigations").status_code == 401


def test_laboratory_is_real_only_read_only_and_pinned(pg, tmp_path, monkeypatch):
    app, _ = setup(pg, tmp_path)
    with TestClient(app) as client:
        assert client.get("/api/v2/research-studies/real-next").status_code == 401
        login(client, "work-viewer")
        response = client.get("/api/v2/research-studies/real-next")
        assert response.status_code == 200, response.text
        data = response.json()["data"]
        assert len(data["report"]["windows"]) == 18
        assert data["classification-report"]["models_trained"] == 0
        assert data["report"]["unseen_accuracy"] is None
        assert data["report"]["operational_approved"] is False
        monkeypatch.setattr(laboratory, "MANIFEST_SHA", "f" * 64)
        assert client.get("/api/v2/research-studies/real-next").status_code == 503
        assert client.get("/api/v1/records").status_code == 200


def test_stream_profile_requires_confirmed_native_source_and_exact_version(
    pg, tmp_path
):
    app, body = setup(pg, tmp_path)
    with TestClient(app) as client:
        headers = login(client, "work-admin")
        platform = app.state.platform
        dsp = platform.blobs.put(
            {
                "method": "median_x3",
                "source_sha256": body["context"]["source_sha256"],
                "row": 0,
                "denoised": [20, None, 22],
            }
        )
        state = {
            "id": "stream",
            "project_id": "local",
            "record_id": "5",
            "source_sha256": body["context"]["source_sha256"],
            "snapshot_version": 2,
            "accepted": 1,
            "persisted": 0,
            "verified_persisted": 0,
            "analyzed": 0,
            "last_profile_dsp": dsp,
            "mode": "measurement_replay",
        }
        platform.put("stream", "stream", state)
        result = client.get("/api/v2/stream/stream/profile?snapshot_version=2")
        assert result.status_code == 200, result.text
        assert result.json()["raw_c"] == [20, None, 22] and result.json()[
            "processed_c"
        ] == [20, None, 22]
        assert (
            result.json()["processing_scope"] == "single_profile_dsp_not_field_analysis"
        )
        assert (
            client.get("/api/v2/stream/stream/profile?snapshot_version=1").status_code
            == 409
        )
        assert client.get("/api/v2/streams").json()["items"][0]["id"] == "stream"
        platform.put(
            "stream", "foreign", {**state, "id": "foreign", "project_id": "foreign"}
        )
        assert client.get("/api/v2/stream/foreign/profile").status_code == 403
        assert all(
            v["id"] != "foreign" for v in client.get("/api/v2/streams").json()["items"]
        )
        analysis_beat(platform.blobs.root.parent)
        request = {
            "region": body["context"]["region"],
            "source_sha256": body["context"]["source_sha256"],
            "stream_id": "stream",
        }
        assert (
            client.post(
                "/api/v2/analysis-jobs",
                json=request,
                headers={**headers, "Idempotency-Key": "confirmed-only"},
            ).status_code
            == 409
        )
        request["region"] = {**request["region"], "row_end": 1}
        accepted = client.post(
            "/api/v2/analysis-jobs",
            json=request,
            headers={**headers, "Idempotency-Key": "confirmed-only"},
        )
        assert accepted.status_code == 202, accepted.text
        platform.put("stream", "stream", {**state, "verified_persisted": -1})
        assert (
            client.post(
                "/api/v2/analysis-jobs",
                json=request,
                headers={**headers, "Idempotency-Key": "confirmed-only"},
            ).json()["id"]
            == accepted.json()["id"]
        )
        assert client.get("/api/v2/stream/stream/profile").status_code == 409
