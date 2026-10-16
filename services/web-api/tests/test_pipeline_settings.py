"""Pipeline setting documents, preflight and immutable job snapshots."""

import uuid
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

import psycopg
from fastapi.testclient import TestClient
from psycopg.types.json import Jsonb
from test_platform import login
from test_platform import pg as platform_pg
from test_workbench import fixture as native_fixture
from workbench.accounts import add_user
from workbench.analysis_compute import METHOD, ML_COMMIT
from workbench.analysis_jobs import analysis_beat
from workbench.api import create_app

pg = platform_pg


def document_text(platform, kind, key):
    with platform.connect() as db:
        row = db.execute(
            "SELECT payload::text FROM documents WHERE kind=%s AND id=%s",
            (kind, key),
        ).fetchone()
    assert row is not None
    return row[0]


def job_count(platform):
    with platform.connect() as db:
        return db.execute("SELECT count(*) FROM jobs").fetchone()[0]


def test_pipeline_settings_contract(pg, tmp_path):
    root, queue, native = native_fixture.__wrapped__(tmp_path)
    add_user(pg, "pipe-admin", "QA-only-password-123456", "Pipe admin", "admin")
    add_user(pg, "pipe-viewer", "QA-only-password-123456", "Pipe viewer", "viewer")
    sha = native.manifest["source_sha256"]
    region = {
        "record_id": "5",
        "row_start": 0,
        "row_end": 3,
        "column_start": 0,
        "column_end": 3,
    }
    window = {key: region[key] for key in ("row_start", "row_end", "column_start", "column_end")}
    pg.put(
        "run",
        "pipe-run",
        {
            "id": "pipe-run",
            "project_id": "local",
            "status": "complete",
            "source_sha256": sha,
            "region": region,
        },
    )
    pg.put(
        "run",
        "other-source",
        {
            "id": "other-source",
            "project_id": "local",
            "status": "complete",
            "source_sha256": "0" * 64,
            "region": region,
        },
    )
    app = create_app(root / "records", queue, root / "runtime")
    body = {
        "title": "Window 6",
        "record_id": "5",
        "source_sha256": sha,
        "window": window,
        "point": {"row": 0, "column": 0},
        "method": METHOD,
        "method_version": ML_COMMIT,
        "parameters": {},
        "run_ids": ["pipe-run"],
    }
    job_body = {"region": region, "source_sha256": sha}
    with TestClient(app) as client:
        assert not (app.state.platform.blobs.root.parent / "analysis-worker.json").exists()
        headers = {**login(client, "pipe-admin"), "Idempotency-Key": "pipe-create-1"}
        case = client.post(
            "/api/v2/cases",
            headers=headers,
            json={
                "title": "Existing evidence",
                "region": region,
                "profile": 0,
                "source_sha256": sha,
            },
        )
        assert case.status_code == 201, case.text
        case_id = case.json()["id"]
        case_bytes = document_text(pg, "evidence_case", case_id)
        assert (
            client.post(
                "/api/v2/pipeline-settings",
                headers=headers,
                json={**body, "point": {"row": 0, "column": 3}},
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/api/v2/pipeline-settings",
                headers=headers,
                json={**body, "method": "other"},
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/api/v2/pipeline-settings",
                headers=headers,
                json={**body, "parameters": {"gain": 1}},
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/api/v2/pipeline-settings",
                headers=headers,
                json={**body, "job_ids": ["client-set"]},
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/api/v2/pipeline-settings",
                headers={**headers, "Idempotency-Key": "bad-run-key"},
                json={**body, "run_ids": ["other-source"]},
            ).status_code
            == 409
        )
        assert (
            client.post(
                "/api/v2/pipeline-settings",
                headers={**headers, "Idempotency-Key": "missing-run"},
                json={**body, "run_ids": ["missing-run"]},
            ).status_code
            == 404
        )
        with ThreadPoolExecutor(max_workers=2) as pool:
            created = list(
                pool.map(
                    lambda _: client.post(
                        "/api/v2/pipeline-settings", json=body, headers=headers
                    ),
                    range(2),
                )
            )
        assert all(item.status_code == 201 for item in created), [item.text for item in created]
        assert len({item.json()["id"] for item in created}) == 1
        setting = created[0].json()
        repeated = client.post("/api/v2/pipeline-settings", json=body, headers=headers)
        assert repeated.status_code == 201
        assert repeated.json()["id"] == setting["id"]
        assert repeated.json() == setting
        assert setting["job_ids"] == [] and setting["revision"] == 1
        assert setting["project_id"] == "local"
        assert setting["author_name"] == "Pipe admin"
        assert setting["method"] == METHOD and setting["method_version"] == ML_COMMIT
        with pg.connect() as db:
            assert (
                db.execute(
                    "SELECT count(*) FROM documents WHERE kind='pipeline_setting'"
                ).fetchone()[0]
                == 1
            )
            assert (
                db.execute(
                    "SELECT count(*) FROM documents WHERE kind='investigation'"
                ).fetchone()[0]
                == 0
            )
        assert (
            client.post(
                "/api/v2/pipeline-settings",
                json={**body, "title": "Different"},
                headers=headers,
            ).status_code
            == 409
        )
        assert (
            client.post("/api/v2/pipeline-settings", json=body, headers=headers).json()[
                "id"
            ]
            == setting["id"]
        )
        print("ASSERT idempotent create one id")

        route = "/api/v2/pipeline-settings/" + setting["id"]
        renamed = client.patch(
            route,
            headers=headers,
            json={"expected_revision": 1, "title": "Renamed window"},
        )
        assert renamed.status_code == 200, renamed.text
        assert renamed.json()["revision"] == 2
        assert renamed.json()["job_ids"] == []
        before_stale = document_text(pg, "pipeline_setting", setting["id"])
        stale = client.patch(
            route,
            headers=headers,
            json={"expected_revision": 1, "title": "Must not win"},
        )
        assert stale.status_code == 409, stale.text
        current = stale.json()["detail"]["current"]
        assert current["revision"] == 2 and current["title"] == "Renamed window"
        assert document_text(pg, "pipeline_setting", setting["id"]) == before_stale
        assert client.get(route, headers=headers).json()["title"] == "Renamed window"
        assert (
            client.patch(
                route, headers=headers, json={"expected_revision": 2, "job_ids": []}
            ).status_code
            == 422
        )
        print("ASSERT stale revision 409")

        viewer = login(client, "pipe-viewer")
        assert client.get("/api/v2/pipeline-settings", headers=viewer).status_code == 200
        assert client.get(route, headers=viewer).status_code == 200
        assert (
            client.post(
                "/api/v2/pipeline-settings",
                headers={**viewer, "Idempotency-Key": "viewer-save"},
                json=body,
            ).status_code
            == 403
        )
        assert (
            client.patch(
                route,
                headers=viewer,
                json={"expected_revision": 2, "title": "Viewer edit"},
            ).status_code
            == 403
        )
        assert (
            client.post(
                "/api/v2/analysis-jobs",
                headers={**viewer, "Idempotency-Key": "viewer-submit"},
                json=job_body,
            ).status_code
            == 403
        )
        assert (
            client.post("/api/v2/analysis-preflight", headers=viewer, json=job_body).status_code
            == 403
        )
        assert document_text(pg, "pipeline_setting", setting["id"]) == before_stale
        print("ASSERT viewer cannot save or submit")

        headers = {**login(client, "pipe-admin"), "Idempotency-Key": "pipe-create-1"}
        foreign_id = "f" * 32
        foreign = {**setting, "id": foreign_id, "project_id": "other", "job_ids": []}
        with pg.connect() as db:
            db.execute(
                "INSERT INTO documents VALUES('pipeline_setting',%s,%s,%s)",
                (foreign_id, Jsonb(foreign), foreign["updated_at"]),
            )
        assert client.get("/api/v2/pipeline-settings/" + foreign_id).status_code == 403
        assert client.get("/api/v2/pipeline-settings/" + "e" * 32).status_code == 404
        assert client.post(
            "/api/v2/pipeline-settings",
            json=body,
            headers={**headers, "X-DTS-Project": "foreign"},
        ).status_code in (401, 403)
        assert client.get(
            "/api/v2/pipeline-settings", headers={**headers, "X-DTS-Project": "foreign"}
        ).status_code in (401, 403)
        listed = client.get("/api/v2/pipeline-settings", headers=headers).json()
        assert [item["id"] for item in listed] == [setting["id"]]
        print("ASSERT foreign project rejected")

        headers["Idempotency-Key"] = "pipe-submit-1"
        assert job_count(pg) == 0
        down = client.post("/api/v2/analysis-preflight", headers=headers, json=job_body)
        assert down.status_code == 200, down.text
        assert down.json()["accepted"] is False and down.json()["ready"] is False
        assert down.json()["warning"] == "Pinned analysis worker is unavailable"
        assert down.json()["method"] == METHOD and down.json()["method_version"] == ML_COMMIT
        assert (
            client.post(
                "/api/v2/analysis-jobs",
                headers=headers,
                json={**job_body, "settings_id": setting["id"], "settings_revision": 2},
            ).status_code
            == 503
        )
        assert job_count(pg) == 0
        assert document_text(pg, "pipeline_setting", setting["id"]) == before_stale
        assert (
            client.post(
                "/api/v2/analysis-preflight",
                headers=headers,
                json={**job_body, "settings_id": setting["id"]},
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/api/v2/analysis-preflight",
                headers=headers,
                json={
                    **job_body,
                    "settings_id": setting["id"],
                    "settings_revision": 1,
                },
            ).status_code
            == 409
        )
        mismatched = {
            **job_body,
            "settings_id": setting["id"],
            "settings_revision": 2,
            "region": {**region, "row_end": 2},
        }
        assert (
            client.post(
                "/api/v2/analysis-preflight", headers=headers, json=mismatched
            ).status_code
            == 409
        )
        analysis_beat(app.state.platform.blobs.root.parent)
        ready = client.post(
            "/api/v2/analysis-preflight",
            headers=headers,
            json={**job_body, "settings_id": setting["id"], "settings_revision": 2},
        )
        assert ready.status_code == 200, ready.text
        assert ready.json()["accepted"] is True and ready.json()["ready"] is True
        assert "warning" not in ready.json()
        assert ready.json()["max_cells"] == 1_000_000 and ready.json()["timeout_s"] == 180
        assert job_count(pg) == 0
        assert document_text(pg, "pipeline_setting", setting["id"]) == before_stale
        print("ASSERT preflight creates zero jobs")

        submitted = client.post(
            "/api/v2/analysis-jobs",
            headers=headers,
            json={**job_body, "settings_id": setting["id"], "settings_revision": 2},
        )
        assert submitted.status_code == 202, submitted.text
        job = submitted.json()
        assert job["settings_id"] == setting["id"]
        assert job["settings_revision"] == 2
        assert job["settings_revision_after"] == 3
        assert job["author_name"] == "Pipe admin"
        assert job_count(pg) == 1
        linked = client.get(route, headers=headers).json()
        assert linked["revision"] == 3 and linked["job_ids"] == [job["id"]]
        stored = app.state.platform.job(job["id"])["payload"]
        snapshot = {
            key: stored[key]
            for key in (
                "window",
                "source_sha256",
                "point",
                "method",
                "method_version",
                "parameters",
                "revision",
            )
        }
        assert snapshot == {
            "window": window,
            "source_sha256": sha,
            "point": {"row": 0, "column": 0},
            "method": METHOD,
            "method_version": ML_COMMIT,
            "parameters": {},
            "revision": 2,
        }
        (app.state.platform.blobs.root.parent / "analysis-worker.json").unlink()
        replay = client.post(
            "/api/v2/analysis-jobs",
            headers=headers,
            json={**job_body, "settings_id": setting["id"], "settings_revision": 2},
        )
        assert replay.status_code == 202 and replay.json()["id"] == job["id"]
        assert client.get(route, headers=headers).json()["job_ids"] == [job["id"]]
        changed = client.patch(
            route,
            headers=headers,
            json={
                "expected_revision": 3,
                "title": "Draft after submit",
                "point": {"row": 1, "column": 1},
            },
        )
        assert changed.status_code == 200, changed.text
        assert changed.json()["revision"] == 4
        assert changed.json()["point"] == {"row": 1, "column": 1}
        assert app.state.platform.job(job["id"])["payload"] == stored
        follow = client.get("/api/v2/analysis-jobs/" + job["id"], headers=headers).json()
        assert follow["settings_revision"] == 2 and follow["settings_revision_after"] == 3
        print("ASSERT job snapshot survives draft change")

        assert document_text(pg, "evidence_case", case_id) == case_bytes
        from workbench.maintenance import backup

        snapshot_dir = tmp_path / "snapshot"
        backup(app.state.platform, snapshot_dir)
        schema = "pipeqa_" + uuid.uuid4().hex
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
        from workbench.platform import Platform

        with psycopg.connect(pg.url, autocommit=True) as db:
            db.execute("CREATE SCHEMA " + schema)
        try:
            restored = Platform(url, tmp_path / "restored")
            from workbench.maintenance import restore

            restore(restored, snapshot_dir)
            assert document_text(restored, "evidence_case", case_id) == case_bytes
            assert document_text(
                restored, "pipeline_setting", setting["id"]
            ) == document_text(pg, "pipeline_setting", setting["id"])
            assert document_text(pg, "evidence_case", case_id) == case_bytes
        finally:
            with psycopg.connect(pg.url, autocommit=True) as db:
                db.execute("DROP SCHEMA " + schema + " CASCADE")
        print("ASSERT backup restore setting and case unchanged")
