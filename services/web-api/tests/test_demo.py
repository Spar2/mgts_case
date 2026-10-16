"""Rehearsal isolation, immutable provenance and exact scientific slices."""

import os
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from test_platform import login
from test_platform import pg as platform_pg
from test_workbench import body
from test_workbench import fixture as native_fixture
from workbench.accounts import add_user
from workbench.api import create_app


@pytest.fixture
def fixture(tmp_path):
    return native_fixture.__wrapped__(tmp_path)


@pytest.fixture
def pg(tmp_path, monkeypatch):
    yield from platform_pg.__wrapped__(tmp_path, monkeypatch)


def start(client):
    headers = {
        "X-Workbench-Token": client.get("/api/v1/session").json()["mutation_token"]
    }
    response = client.post(
        "/api/v1/demo/sessions",
        json={
            "package": "general",
            "candidate": "card",
            "client_key": "technical-test-000001",
        },
        headers=headers,
    )
    assert response.status_code == 201, response.text
    assert "HttpOnly" in response.headers["set-cookie"]
    assert "SameSite=strict" in response.headers["set-cookie"]
    return response.json(), headers


def test_rehearsal_protects_working_journal_and_restores(fixture):
    root, queue, record = fixture
    runtime = root / "demo-runtime"
    app = create_app(root / "records", queue, runtime)
    with TestClient(app) as client:
        initial = client.get("/api/v1/queue").json()["annotations"]
        session, headers = start(client)
        key = session["id"]
        duplicate, _ = start(client)
        assert duplicate["id"] == key
        # Even a manually submitted regular request with a valid token is denied.
        for route, payload in (
            ("/api/v1/reviews", body(queue)),
            ("/api/v1/denoise", {"record_id": "5", "row": 0}),
            ("/api/v1/bookmarks", {**session["region"], "title": "blocked"}),
        ):
            assert client.post(route, json=payload, headers=headers).status_code == 403
        response = client.post(
            f"/api/v1/demo/sessions/{key}/answer",
            json={
                "temperature_episode": "unknown",
                "notes": "QA ONLY — rehearsal, not human labels",
            },
            headers=headers,
        )
        assert response.status_code == 200, response.text
        assert client.get("/api/v1/queue").json()["annotations"] == initial
        assert app.state.workspace.list("run") == []
        evidence = client.get(
            f"/api/v1/demo/sessions/{key}/evidence?row=0&column=1"
        ).json()
        assert evidence["selection"]["temperature"] is None
        assert evidence["profile_invalid_reason"] == record.invalid_reason[0].tolist()
        assert evidence["selection"]["trend"] == [None, 23.0, None, 26.0]
        assert evidence["source_sha256"] == record.manifest["source_sha256"]
        report = client.post(
            f"/api/v1/demo/sessions/{key}/report?format=html", json={}, headers=headers
        )
        assert report.status_code == 200
        assert "РЕПЕТИЦИЯ" in report.text and "QA ONLY" in report.text
        assert "<script" not in report.text and "<link" not in report.text
        exported = client.post(
            f"/api/v1/demo/sessions/{key}/report?format=json", json={}, headers=headers
        ).json()
        assert exported["profile"]["invalid_reason"] == [0, 1, 0]
        assert exported["expert_answer"]["reviewer_type"] == "rehearsal"
    with TestClient(create_app(root / "records", queue, runtime)) as client:
        restored = client.get(f"/api/v1/demo/sessions/{key}").json()
        assert restored["expert_answer"]["notes"].startswith("QA ONLY")
        headers = {
            "X-Workbench-Token": client.get("/api/v1/session").json()["mutation_token"]
        }
        assert (
            client.post(
                f"/api/v1/demo/sessions/{key}/activate", json={}, headers=headers
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/api/v1/reviews", json=body(queue), headers=headers
            ).status_code
            == 403
        )
        assert (
            client.post("/api/v1/demo/leave", json={}, headers=headers).status_code
            == 200
        )
        assert (
            client.post(
                "/api/v1/reviews",
                json=body(queue),
                headers={**headers, "Idempotency-Key": "normal-after-rehearsal"},
            ).status_code
            == 200
        )


def test_prepared_profile_and_processing_do_not_create_working_runs(fixture):
    root, queue, record = fixture
    app = create_app(root / "records", queue, root / "runtime")
    with TestClient(app) as client:
        headers = {
            "X-Workbench-Token": client.get("/api/v1/session").json()["mutation_token"]
        }
        imported = client.post(
            "/api/v1/results",
            headers=headers,
            json={
                "schema_version": "dts-method-result-v1",
                "region": {
                    "record_id": "5",
                    "row_start": 0,
                    "row_end": 1,
                    "column_start": 0,
                    "column_end": 3,
                },
                "source_sha256": record.manifest["source_sha256"],
                "method": "contract-test",
                "method_version": "test-only",
                "parameters": {},
                "calculated_at": "2026-10-10",
                "training_origin": "untrained",
                "limitations": "QA contract fixture, no model quality claim",
                "result": {"denoised": [20.0, 0.0, 22.0]},
            },
        )
        assert imported.status_code == 201, imported.text
        # Real profiles include both denoised and residual arrays. The dedicated
        # bounded import may exceed 64 KiB; other mutations retain their limit.
        large = imported.request.read()
        import json

        payload = json.loads(large)
        payload["result"]["contract_fixture_padding"] = "x" * 70000
        large_import = client.post("/api/v1/results", json=payload, headers=headers)
        assert large_import.status_code == 201
        payload["result"]["contract_fixture_padding"] = "x" * (256 * 1024)
        assert (
            client.post("/api/v1/results", json=payload, headers=headers).status_code
            == 413
        )
        assert (
            client.post(
                "/api/v1/bookmarks", content=b"x" * 70000, headers=headers
            ).status_code
            == 413
        )
        app.state.workspace.delete("run", large_import.json()["id"])
        session, headers = start(client)
        key = session["id"]
        route = f"/api/v1/demo/sessions/{key}/process"
        assert (
            client.post(
                route, json={"row": 1, "run_id": imported.json()["id"]}, headers=headers
            ).status_code
            == 409
        )
        assert (
            client.post(
                route, json={"row": 0, "run_id": "missing"}, headers=headers
            ).status_code
            == 404
        )
        result = client.post(
            route, json={"row": 0, "run_id": imported.json()["id"]}, headers=headers
        )
        assert result.status_code == 200, result.text
        assert result.json()["result"]["denoised"] == [20.0, None, 22.0]
        assert len(app.state.workspace.list("run")) == 1

        async def unavailable(_):
            from fastapi import HTTPException

            raise HTTPException(503, "ML unavailable")

        app.state.clean_profile = unavailable
        assert client.post(route, json={"row": 0}, headers=headers).status_code == 503
        assert client.get(f"/api/v1/demo/sessions/{key}/evidence").status_code == 200
        assert len(app.state.workspace.list("run")) == 1
        session["source_sha256"] = "0" * 64
        app.state.workspace.put("demo_session", key, session)
        assert client.get(f"/api/v1/demo/sessions/{key}").status_code == 409


def test_rehearsal_access_and_stream_permissions(pg, fixture):
    root, queue, _ = fixture
    add_user(pg, "demo-viewer", "QA-only-password-123456", "Demo viewer", "viewer")
    add_user(pg, "demo-other", "QA-only-password-123456", "Other participant", "admin")
    app = create_app(root / "records", queue, Path(pg.blobs.root).parent)
    with TestClient(app) as client:
        login(client, "demo-viewer")
        session, headers = start(client)
        key = session["id"]
        assert (
            client.post(
                f"/api/v1/demo/sessions/{key}/stream", json={}, headers=headers
            ).status_code
            == 403
        )
        assert (
            client.post(
                f"/api/v1/demo/sessions/{key}/answer",
                json={"temperature_episode": "unknown", "notes": "QA ONLY"},
                headers=headers,
            ).status_code
            == 200
        )
        other_headers = login(client, "demo-other")
        assert client.get(f"/api/v1/demo/sessions/{key}").status_code == 403
        assert (
            client.post(
                f"/api/v1/demo/sessions/{key}/activate", json={}, headers=other_headers
            ).status_code
            == 403
        )


def test_demo_preflight_without_stream_and_invalid_selection(fixture):
    root, queue, _ = fixture
    with TestClient(create_app(root / "records", queue, root / "runtime")) as client:
        session, _ = start(client)
        key = session["id"]
        preflight = client.get(f"/api/v1/demo/sessions/{key}/preflight")
        assert preflight.status_code == 200, preflight.text
        assert preflight.json()["source"] and not preflight.json()["stream_permitted"]
        assert (
            client.get(f"/api/v1/demo/sessions/{key}/evidence?row=99").status_code
            == 422
        )
        assert (
            client.get(f"/api/v1/demo/sessions/{key}/evidence?column=99").status_code
            == 422
        )


def test_stream_counts_and_own_job_resume(pg, fixture):
    root, queue, _ = fixture
    add_user(pg, "demo-admin", "QA-only-password-123456", "Demo admin", "admin")
    with TestClient(
        create_app(root / "records", queue, Path(pg.blobs.root).parent)
    ) as client:
        login(client, "demo-admin")
        session, headers = start(client)
        route = f"/api/v1/demo/sessions/{session['id']}/stream"
        job = client.post(route, json={}, headers=headers).json()
        repeated = client.post(route, json={}, headers=headers).json()
        assert repeated["id"] == job["id"]
        pg.put(
            "stream",
            job["id"],
            {
                "id": job["id"],
                "project_id": "local",
                "accepted": 2,
                "persisted": 0,
                "analyzed": -1,
                "verified_persisted": -1,
                "dsp_profiles": 0,
            },
        )
        snapshot = client.get(route).json()["snapshot"]
        assert snapshot["accepted"] == 3 and snapshot["persisted"] == 1
        assert snapshot["analyzed"] == 0 and snapshot["total_profiles"] == 3
        assert snapshot["native_row_indices"]["accepted"] == 2
        assert (
            client.post(route + "/cancel", json={}, headers=headers).status_code == 200
        )
        resumed = client.post(route + "/resume", json={}, headers=headers)
        assert resumed.status_code == 200, resumed.text
        assert resumed.json()["id"] == job["id"]


def test_demo_exact_evidence_for_every_real_record(tmp_path):
    records, queue = (
        os.environ.get("DTS_REAL_RECORDS"),
        os.environ.get("DTS_REAL_QUEUE"),
    )
    if not records or not queue:
        pytest.skip("Requires the six real prepared records and their verified queue")
    app = create_app(records, queue, tmp_path / "real-demo")
    with TestClient(app) as client:
        headers = {
            "X-Workbench-Token": client.get("/api/v1/session").json()["mutation_token"]
        }
        candidates = client.get("/api/v1/candidates?package=general").json()
        seen = set()
        for candidate in candidates:
            record_id = candidate["region"]["record_id"]
            if record_id in seen:
                continue
            seen.add(record_id)
            response = client.post(
                "/api/v1/demo/sessions",
                headers=headers,
                json={
                    "package": "general",
                    "candidate": candidate["id"],
                    "client_key": "scientific-native-" + record_id,
                },
            )
            assert response.status_code == 201, response.text
            session = response.json()
            region = session["region"]
            row = region["row_start"]
            column = region["column_start"]
            data = client.get(
                f"/api/v1/demo/sessions/{session['id']}/evidence?row={row}&column={column}"
            ).json()
            native = app.state.store.get(record_id)
            assert data["source_sha256"] == native.manifest["source_sha256"]
            assert data["time_kind"] == native.manifest["time_kind"]
            np.testing.assert_array_equal(
                data["profile_invalid_reason"],
                native.invalid_reason[
                    row, region["column_start"] : region["column_end"]
                ],
            )
            expected = native.raw_c[row].copy()
            expected[native.invalid_reason[row] != 0] = np.nan
            np.testing.assert_array_equal(
                np.array(data["selection"]["profile"], dtype=float), expected
            )
            np.testing.assert_array_equal(data["selection"]["x_m"], native.x_m)
            points = dict(zip(data["selection"]["time_s"], data["selection"]["trend"]))
            for r in range(region["row_start"], region["row_end"]):
                assert points[float(native.elapsed_s[r])] == (
                    float(native.raw_c[r, column])
                    if native.invalid_reason[r, column] == 0
                    else None
                )
            assert data["selection"]["timestamp"] == native.profiles[row]["timestamp"]
        assert seen == set(app.state.store.records)
        assert len(seen) == 6
