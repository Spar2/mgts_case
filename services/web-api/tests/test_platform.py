"""Run against a dedicated temporary schema in a real local PostgreSQL."""

import os
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import quote

import psycopg
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_workbench import body
from test_workbench import fixture as native_fixture
from workbench.accounts import add_user
from workbench.api import create_app
from workbench.platform import Platform


@pytest.fixture
def pg(tmp_path, monkeypatch):
    url = os.environ.get("DTS_TEST_PG_URL")
    if not url:
        pytest.skip("DTS_TEST_PG_URL requires real isolated PostgreSQL")
    schema = "qa_" + uuid.uuid4().hex
    with psycopg.connect(url, autocommit=True) as db:
        db.execute("CREATE SCHEMA " + schema)
    isolated = (
        url
        + ("&" if "?" in url else "?")
        + "options="
        + quote("-c search_path=" + schema)
    )
    monkeypatch.setenv("DTS_DATABASE_URL", isolated)
    monkeypatch.setenv("DTS_RATE_NAMESPACE", "qa:access:" + schema + ":")
    platform = Platform(isolated, tmp_path)
    yield platform
    with psycopg.connect(url, autocommit=True) as db:
        db.execute("DROP SCHEMA " + schema + " CASCADE")


def login(client, name):
    response = client.post(
        "/api/v2/auth/login",
        json={"username": name, "password": "QA-only-password-123456"},
    )
    assert response.status_code == 200, response.text
    return {"X-Workbench-Token": client.get("/api/v1/session").json()["mutation_token"]}


def test_selected_report_and_passport_check_project_and_exact_run(pg, tmp_path):
    root, queue, record = native_fixture.__wrapped__(tmp_path)
    add_user(pg, "selected-admin", "QA-only-password-123456", "Selected admin", "admin")
    app = create_app(root / "records", queue, root / "runtime")
    region = {
        "record_id": "5",
        "row_start": 0,
        "row_end": 1,
        "column_start": 0,
        "column_end": 3,
    }
    run = {
        "id": "chosen",
        "status": "complete",
        "method": "prepared-fixture",
        "method_version": "contract-v1",
        "parameters": {"safe_fixture": True},
        "training_origin": "untrained",
        "limitations": "Contract only",
        "source_sha256": record.manifest["source_sha256"],
        "region": region,
        "result": {"denoised": [10, 11, None]},
    }
    with TestClient(app) as client:
        app.state.workspace.put("run", "chosen", run)
        app.state.workspace.put(
            "run",
            "other-project",
            {**run, "id": "other-project", "project_id": "other"},
        )
        headers = login(client, "selected-admin")
        result = client.post(
            "/api/v1/reports?profile=0&method=prepared-fixture&run_id=chosen",
            headers=headers,
            json=region,
        )
        assert result.status_code == 200
        assert result.json()["profile"]["processed_temperature_c"] == [10, None, None]
        assert (
            client.post(
                "/api/v1/reports?profile=0&run_id=other-project",
                headers=headers,
                json=region,
            ).status_code
            == 403
        )
        body = {
            "title": "Selected case",
            "region": region,
            "profile": 0,
            "method": "prepared-fixture",
            "source_sha256": record.manifest["source_sha256"],
            "scale": [0, 20],
            "run_id": "chosen",
        }
        passport = client.post("/api/v2/passports", headers=headers, json=body)
        assert passport.status_code == 201
        saved = passport.json()
        assert saved["run_id"] == "chosen" and saved["method_version"] == "contract-v1"
        assert [r["id"] for r in saved["run_references"]] == ["chosen"]
        assert (
            client.post(
                "/api/v2/passports", headers=headers, json={**body, "method": "wrong"}
            ).status_code
            == 409
        )
        assert (
            client.post(
                "/api/v2/passports",
                headers=headers,
                json={**body, "run_id": "other-project"},
            ).status_code
            == 403
        )


def test_import_rejects_duplicate_before_publishing(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from workbench.store import Store
    from workbench.worker import execute

    root, _, _ = native_fixture.__wrapped__(tmp_path)
    runtime = tmp_path / "import-runtime"
    store = Store(root / "records", runtime / "cache")
    imports = tmp_path / "imports"
    imports.mkdir()
    duplicate = next(iter(store.records)) + ".csv"
    (imports / duplicate).write_text(
        "invalid CSV must never be parsed", encoding="utf-8"
    )
    monkeypatch.setenv("DTS_IMPORT_ROOT", str(imports))
    platform = SimpleNamespace(blobs=SimpleNamespace(root=runtime / "blobs"))
    with pytest.raises(ValueError, match="Source name already exists"):
        execute(
            platform, store, {"kind": "import_csv", "payload": {"filename": duplicate}}
        )
    assert not list((runtime / "imported-records").glob("*/manifest.json"))
    assert len(Store(root / "records", runtime / "cache").records) == len(store.records)


def test_accounts_scope_csrf_reviews_and_logout(pg, tmp_path):
    root, queue, _ = native_fixture.__wrapped__(tmp_path)
    for name, role in [
        ("alice", "expert"),
        ("bob", "expert"),
        ("viewer", "viewer"),
        ("admin", "admin"),
    ]:
        add_user(pg, name, "QA-only-password-123456", name, role)
    with TestClient(
        create_app(root / "records", queue, tmp_path / "app-runtime")
    ) as client:
        assert client.get("/api/v1/records").status_code == 401
        headers = login(client, "alice")
        assert client.get("/api/v1/records").status_code == 200
        suffix = "?independent=true&reviewer=alice"
        assert client.get("/api/v1/queue/card/methods" + suffix).status_code == 403
        vote = {**body(queue), "reviewer": "alice"}
        assert (
            client.post(
                "/api/v1/reviews" + suffix,
                json=vote,
                headers={"Idempotency-Key": "qa-revision-one"},
            ).status_code
            == 403
        )
        headers["Idempotency-Key"] = "qa-revision-one"
        saved = client.post("/api/v1/reviews" + suffix, json=vote, headers=headers)
        assert saved.status_code == 200, saved.text
        assert saved.json()["revision"] == 1
        assert (
            client.post("/api/v1/reviews" + suffix, json=vote, headers=headers).json()[
                "revision"
            ]
            == 1
        )
        assert client.get("/api/v1/queue/card/methods" + suffix).status_code == 200
        assert (
            client.get("/api/v1/queue?independent=true&reviewer=bob").status_code == 403
        )
        client.post("/api/v2/auth/logout", json={}, headers=headers)
        assert client.get("/api/v1/records").status_code == 401
        login(client, "bob")
        assert (
            client.get(
                "/api/v1/queue/card/methods?independent=true&reviewer=bob"
            ).status_code
            == 403
        )
        viewer_headers = login(client, "viewer")
        assert (
            client.post(
                "/api/v1/jobs/detect",
                json={
                    "record_id": "5",
                    "row_start": 0,
                    "row_end": 2,
                    "column_start": 0,
                    "column_end": 2,
                },
                headers=viewer_headers,
            ).status_code
            == 403
        )


def test_worker_lease_idempotency_artifact_and_cancellation(pg):
    job = pg.enqueue("prepare_views", {"source": "sha"}, key="one-action")
    assert (
        pg.enqueue("prepare_views", {"source": "sha"}, key="one-action")["id"]
        == job["id"]
    )
    first = pg.claim("first", ["prepare_views"])
    assert first["attempts"] == 1
    assert pg.claim("other", ["prepare_views"]) is None
    with pg.connect() as db:
        db.execute(
            "UPDATE jobs SET lease_until=%s WHERE id=%s", (time.time() - 1, job["id"])
        )
    second = pg.claim("second", ["prepare_views"])
    assert second["attempts"] == 2
    pg.finish(job["id"], "first", result={"wrong": True})
    assert pg.job(job["id"])["status"] == "running"
    pg.cancel_job(job["id"])
    pg.finish(job["id"], "second", result={"must_be_discarded": True})
    final = pg.job(job["id"])
    assert final["status"] == "cancelled" and final["result"] is None
    value = pg.put(
        "run",
        "artifact",
        {"id": "artifact", "status": "complete", "result": {"denoised": [1, None, 3]}},
    )
    assert value["result"]["denoised"] == [1, None, 3]
    metadata = pg.list("run")[0]
    assert "result" not in metadata and metadata["result_ref"]["sha256"]
    assert pg.get("run", "artifact")["result"]["denoised"] == [1, None, 3]


def test_failed_job_can_finish_without_json_result(pg):
    job = pg.enqueue("prepare_views", {})
    pg.claim("worker", ["prepare_views"])
    pg.finish(job["id"], "worker", error="Explicit contract failure")
    assert pg.job(job["id"])["status"] == "failed"
    assert pg.job(job["id"])["result"] is None


def test_lease_fences_snapshots_and_atomic_run_publication(pg):
    from workbench.platform import LeaseLost, worker_lease

    job = pg.enqueue("prepare_views", {})
    pg.claim("old", ["prepare_views"])
    token = worker_lease.set({"id": job["id"], "owner": "old"})
    try:
        pg.put("stream", job["id"], {"id": job["id"], "accepted": 1})
        with pg.connect() as db:
            db.execute(
                "UPDATE jobs SET lease_until=%s WHERE id=%s",
                (time.time() - 1, job["id"]),
            )
        assert pg.heartbeat(job["id"], "old") is None
        with pytest.raises(LeaseLost):
            pg.put("stream", job["id"], {"id": job["id"], "accepted": 999})
        pg.claim("new", ["prepare_views"])
        with pytest.raises(LeaseLost):
            pg.put("stream", job["id"], {"id": job["id"], "accepted": 999})
        assert pg.get("stream", job["id"])["accepted"] == 1
    finally:
        worker_lease.reset(token)
    detect = pg.enqueue(
        "detect",
        {
            "record_id": "5",
            "row_start": 0,
            "row_end": 2,
            "column_start": 0,
            "column_end": 2,
            "source_sha256": "a" * 64,
        },
    )
    pg.claim("compute", ["detect"])
    ref = pg.blobs.put({"zones": [], "verification": "test-contract"})
    assert not pg.finish(detect["id"], "stale", result=ref)
    assert "result_ref" not in pg.get("run", detect["id"])
    assert pg.finish(detect["id"], "compute", result=ref)
    assert pg.get("run", detect["id"])["result"]["verification"] == "test-contract"


def test_resume_preserves_replay_identity_and_checks_scope(pg):
    from workbench.platform import JobConflict

    job = pg.enqueue("stream_replay", {"record_id": "5", "source_sha256": "a" * 64})
    pg.claim("stream", ["stream_replay"])
    pg.finish(job["id"], "stream", error="Storage interruption")
    with pytest.raises(PermissionError):
        pg.retry_job(job["id"], "other")
    resumed = pg.retry_job(job["id"], "local")
    assert resumed["id"] == job["id"] and resumed["payload"] == job["payload"]
    assert resumed["status"] == "queued" and not resumed["cancel_requested"]
    with pytest.raises(JobConflict):
        pg.retry_job(job["id"], "local")
    pg.claim("new-stream", ["stream_replay"])
    assert pg.job(job["id"])["attempts"] == 2


def test_account_duplicate_and_disable_revoke_all_sessions(pg, tmp_path):
    root, queue, _ = native_fixture.__wrapped__(tmp_path)
    add_user(pg, "admin", "QA-only-password-123456", "Admin", "admin")
    participant = add_user(pg, "expert", "QA-only-password-123456", "Expert", "expert")
    with (
        TestClient(
            create_app(root / "records", queue, tmp_path / "admin-runtime")
        ) as admin,
        TestClient(
            create_app(root / "records", queue, tmp_path / "expert-runtime")
        ) as expert,
    ):
        headers = login(admin, "admin")
        login(expert, "expert")
        assert expert.get("/api/v2/auth/me").status_code == 200
        duplicate = admin.post(
            "/api/v2/accounts",
            headers=headers,
            json={
                "username": " EXPERT ",
                "password": "QA-only-password-123456",
                "display_name": "Other",
                "role": "expert",
            },
        )
        assert duplicate.status_code == 409
        assert (
            admin.post(
                "/api/v2/accounts/" + participant["id"] + "/disable",
                headers=headers,
                json={},
            ).status_code
            == 200
        )
        assert expert.get("/api/v2/auth/me").status_code == 401
        listed = admin.get("/api/v2/accounts").json()
        assert not next(u for u in listed if u["id"] == participant["id"])["enabled"]
        assert (
            admin.post(
                "/api/v2/accounts/" + participant["id"] + "/enable",
                headers=headers,
                json={},
            ).status_code
            == 200
        )
        # Enabling must not resurrect revoked cookies.
        assert expert.get("/api/v2/auth/me").status_code == 401
        assert (
            expert.post(
                "/api/v2/auth/login",
                json={"username": "expert", "password": "QA-only-password-123456"},
            ).status_code
            == 200
        )


def test_maintenance_snapshot_integrity_and_legacy_identity(pg, tmp_path):
    import json
    import sqlite3

    from workbench.maintenance import backup, migrate

    legacy = tmp_path / "legacy"
    legacy.mkdir()
    with sqlite3.connect(legacy / "workspace.sqlite3") as db:
        db.execute("CREATE TABLE documents(kind,id,payload)")
        db.execute(
            "INSERT INTO documents VALUES(?,?,?)",
            (
                "run",
                "old",
                json.dumps(
                    {
                        "id": "old",
                        "status": "running",
                        "method": "reference",
                        "method_version": "legacy",
                        "limitations": "unverified",
                    }
                ),
            ),
        )
    migrate(pg, legacy)
    migrate(pg, legacy)
    assert len(pg.list("run")) == 1
    assert pg.list("run")[0]["status"] == "interrupted"
    assert len(pg.list("legacy_archive")) == 1
    assert pg.list("legacy_archive")[0]["identity_mapping"].startswith("unassigned")
    destination = tmp_path / "backup"
    backup(pg, destination)
    import hashlib

    manifest = json.loads((destination / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((destination / name).read_bytes()).hexdigest() == digest


def test_backup_restore_into_empty_schema(pg, tmp_path):
    from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

    from workbench.accounts import add_user
    from workbench.maintenance import backup, restore
    from workbench.platform import Platform

    add_user(
        pg, "restore-expert", "QA-only-password-123456", "Restore expert", "expert"
    )
    pg.put(
        "run",
        "restore-one",
        {
            "id": "restore-one",
            "status": "complete",
            "result": {"denoised": [1.25, None, 3.5]},
        },
    )
    snapshot = tmp_path / "snapshot"
    backup(pg, snapshot)
    schema = "restoreqa_" + uuid.uuid4().hex
    parts = urlsplit(pg.url)
    params = dict(parse_qsl(parts.query))
    params["options"] = "-c search_path=" + schema
    url = urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            parts.path,
            urlencode(params, quote_via=quote),
            parts.fragment,
        )
    )
    with psycopg.connect(pg.url, autocommit=True) as db:
        db.execute("CREATE SCHEMA " + schema)
    try:
        target = Platform(url, tmp_path / "restored")
        restore(target, snapshot)
        assert target.get("run", "restore-one")["result"]["denoised"] == [
            1.25,
            None,
            3.5,
        ]
        with target.connect() as db:
            assert (
                db.execute("SELECT username FROM users").fetchone()[0]
                == "restore-expert"
            )
            assert db.execute("SELECT count(*) FROM sessions").fetchone()[0] == 0
        with pytest.raises(ValueError):
            restore(target, snapshot)
    finally:
        with psycopg.connect(pg.url, autocommit=True) as db:
            db.execute("DROP SCHEMA " + schema + " CASCADE")


def test_project_permission_and_case_revision(pg, tmp_path):
    root, queue, _record = native_fixture.__wrapped__(tmp_path)
    add_user(pg, "admin", "QA-only-password-123456", "Admin", "admin")
    expert = add_user(pg, "expert", "QA-only-password-123456", "Expert", "expert")
    with pg.connect() as db:
        db.execute("INSERT INTO projects VALUES('other','Other project')")
    outsider = add_user(
        pg, "outside", "QA-only-password-123456", "Outside", "admin", "other"
    )
    with TestClient(create_app(root / "records", queue, tmp_path / "runtime")) as c:
        h = login(c, "admin")
        body = {
            "package": "general",
            "review_id": "card",
            "state": "assigned",
            "assigned_to": expert["id"],
            "expected_revision": 0,
        }
        first = c.post("/api/v2/cases", headers=h, json=body)
        assert first.status_code == 200, first.text
        assert c.post("/api/v2/cases", headers=h, json=body).status_code == 409
        assert (
            c.post(
                "/api/v2/cases",
                headers=h,
                json={**body, "expected_revision": 1, "assigned_to": outsider["id"]},
            ).status_code
            == 422
        )
        assert (
            c.post(
                "/api/v2/accounts/" + outsider["id"] + "/disable", headers=h, json={}
            ).status_code
            == 403
        )
        headers = login(c, "expert")
        assert c.post("/api/v2/cases", headers=headers, json=body).status_code == 403
        assert (
            c.get("/api/v2/cases?package=general").json()["items"][0]["assigned_to"]
            == expert["id"]
        )
        c.post(
            "/api/v2/auth/login",
            json={"username": "outside", "password": "QA-only-password-123456"},
        ).raise_for_status()
        other = {"X-DTS-Project": "other"}
        assert c.get("/api/v2/cases?package=general", headers=other).status_code == 403
        assert (
            c.get("/api/v2/disagreements?package=general", headers=other).status_code
            == 403
        )
        status = c.get("/api/v2/status", headers=other).json()
        assert status["sources"] == 0 and not status["capabilities"]["archive"]


def test_readiness_rejects_invalid_heartbeat_and_missing_stream(tmp_path):
    from workbench.health import Readiness, heartbeat_age

    (tmp_path / "worker-compute.heartbeat").write_text("partial write")
    assert heartbeat_age(tmp_path, "compute") is None
    readiness = Readiness(tmp_path)
    services = readiness.get()
    assert not services["compute_worker"] and not services["stream_worker"]
    assert not services["redis"] and not services["clickhouse"]
