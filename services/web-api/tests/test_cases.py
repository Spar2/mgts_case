"""Immutable case, blind access, revision conflicts and bounded offline ZIP."""

import copy
import io
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from test_platform import login
from test_platform import pg as pg  # noqa: PLC0414 - pytest fixture registration
from test_workbench import body
from test_workbench import fixture as native_fixture
from workbench.accounts import add_user
from workbench.api import create_app
from workbench.case_bundle import read_bundle
from workbench.cases import digest, encoded, make_bundle
from workbench.evidence_check import verify_evidence


def setup(pg, tmp_path):
    root, queue, record = native_fixture.__wrapped__(tmp_path)
    with pg.connect() as db:
        db.execute("INSERT INTO projects VALUES('other','Other project')")
    for name, role, project in (
        ("author", "expert", "local"),
        ("reader", "expert", "local"),
        ("viewer", "viewer", "local"),
        ("analyst", "analyst", "local"),
        ("foreign", "admin", "other"),
    ):
        add_user(pg, name, "QA-only-password-123456", name, role, project)
    app = create_app(root / "records", queue, root / "runtime")
    region = {
        "record_id": "5",
        "row_start": 0,
        "row_end": 3,
        "column_start": 0,
        "column_end": 3,
    }
    request = {
        "title": "Actual fixture window",
        "region": region,
        "profile": 0,
        "source_sha256": record.manifest["source_sha256"],
        "run_id": "selected",
    }
    run = {
        "id": "selected",
        "status": "complete",
        "region": {**region, "row_end": 1},
        "source_sha256": request["source_sha256"],
        "method": "reference",
        "method_version": "contract-only",
        "training_origin": "untrained",
        "limitations": "Not a model",
        "parameters": {},
        "result": {"denoised": [20.0, 123.0, None]},
        "internal_secret": "MUST-NOT-EXPORT",
    }
    return app, root, queue, record, request, run


def test_case_snapshot_revision_bundle_restart_and_independence(pg, tmp_path):
    app, root, queue, record, request, run = setup(pg, tmp_path)
    with TestClient(app) as client:
        app.state.workspace.put("run", "selected", run)
        headers = login(client, "author")
        vote = {**body(queue), "reviewer": "author"}
        suffix = "?package=general&independent=true&reviewer=author"
        answer = client.post(
            "/api/v1/reviews" + suffix,
            headers={**headers, "Idempotency-Key": "author-v1"},
            json=vote,
        )
        assert answer.status_code == 200, answer.text
        request["review"] = {
            "package": "general",
            "review_id": "card",
            "queue_sha256": vote["queue_sha256"],
            "expected_revision": 1,
        }
        preview = client.post("/api/v2/cases/preview", headers=headers, json=request)
        assert (
            preview.status_code == 200
            and preview.json()["expert_answer"]["revision"] == 1
        )
        result = client.post("/api/v2/cases", headers=headers, json=request)
        assert result.status_code == 201, result.text
        case = result.json()
        assert case["result"]["evidence"]["profile"]["processed_temperature_c"] == [
            20.0,
            None,
            None,
        ]
        assert "MUST-NOT-EXPORT" not in result.text
        path = "/api/v2/cases/" + case["id"]
        original = client.get(path + "/bundle").content
        assert make_bundle(case) == original
        assert (
            make_bundle(case) == original
        )  # rendering cannot mutate a retained snapshot
        archive_path = tmp_path / "case.zip"
        archive_path.write_bytes(original)
        evidence, saved_run, manifest = read_bundle(archive_path)
        assert evidence["processing_changes"]["paired_points"] == 1
        with zipfile.ZipFile(archive_path) as archived:
            assert "Что изменила обработка" in archived.read("evidence.html").decode()
        assert (
            verify_evidence(evidence, record, saved_run)["processing_run_id"]
            == "selected"
        )
        forged_changes = copy.deepcopy(evidence)
        forged_changes["processing_changes"]["changed_points"] += 1
        with pytest.raises(ValueError, match="Processing changes"):
            verify_evidence(forged_changes, record, saved_run)
        legacy = copy.deepcopy(case)
        legacy["result"]["evidence"].pop("processing_changes")
        legacy["snapshot_sha256"] = digest(encoded(legacy["result"]))
        legacy_bytes = make_bundle(legacy)
        assert make_bundle(legacy) == legacy_bytes
        with zipfile.ZipFile(io.BytesIO(legacy_bytes)) as archived:
            assert (
                "Что изменила обработка" not in archived.read("evidence.html").decode()
            )
        assert manifest["review"]["revision"] == 1
        assert evidence["preview"] and evidence["preview_valid"]
        with zipfile.ZipFile(io.BytesIO(original)) as archive:
            html = archive.read("evidence.html").decode()
            assert "<script" not in html and "MUST-NOT-EXPORT" not in html
            assert "evidence.json" in archive.namelist()
        assert (
            client.put(path, headers=headers, json={"title": "changed"}).status_code
            == 405
        )
        assert (
            client.post(
                "/api/v1/reviews" + suffix,
                headers={**headers, "Idempotency-Key": "author-v2"},
                json={**vote, "expected_revision": 1, "notes": "Changed answer"},
            ).status_code
            == 200
        )
        assert (
            client.post("/api/v2/cases", headers=headers, json=request).status_code
            == 409
        )
        app.state.workspace.put(
            "run", "selected", {**run, "result": {"denoised": [91.0, 92.0, 93.0]}}
        )
        assert not client.get(path).json()["research_available"]
        assert client.get(path + "/bundle").content == original
        reader = login(client, "reader")
        assert client.get(path).status_code == 403
        assert client.get(path + "/bundle").status_code == 403
        assert client.get("/api/v2/cases").json() == []
        reader_vote = {**vote, "reviewer": "reader"}
        assert (
            client.post(
                "/api/v1/reviews?package=general&independent=true&reviewer=reader",
                headers={**reader, "Idempotency-Key": "reader-v1"},
                json=reader_vote,
            ).status_code
            == 200
        )
        assert client.get(path).status_code == 200
        viewer = login(client, "viewer")
        assert client.get(path).status_code == 403
        raw_request = {**request, "run_id": None, "review": None}
        raw = client.post("/api/v2/cases", headers=viewer, json=raw_request)
        assert raw.status_code == 403
        raw = client.post('/api/v2/cases',headers=login(client,'analyst'),json=raw_request)
        assert raw.status_code == 201
        assert raw.json()["result"]["evidence"]["expert_answer"] is None
        client.headers["X-DTS-Project"] = "other"
        foreign_login = client.post(
            "/api/v2/auth/login",
            json={"username": "foreign", "password": "QA-only-password-123456"},
        )
        assert foreign_login.status_code == 200
        foreign = {"X-Workbench-Token": foreign_login.json()["csrf"]}
        assert client.get(path, headers={"X-DTS-Project": "other"}).status_code == 403
        assert (
            client.get(path + "/bundle", headers={"X-DTS-Project": "other"}).status_code
            == 403
        )
        assert (
            client.post(
                "/api/v2/cases",
                headers={**foreign, "X-DTS-Project": "other"},
                json=raw_request,
            ).status_code
            == 403
        )
    restarted = create_app(root / "records", queue, root / "runtime")
    with TestClient(restarted) as client:
        login(client, "author")
        assert client.get(path + "/bundle").content == original
        restarted.state.store.records["5"].manifest["source_sha256"] = "0" * 64
        detail = client.get(path).json()
        assert not detail["source_available"] and not detail["research_available"]
        assert client.get(path + "/bundle").content == original


def test_case_rejects_incompatible_and_unbound_context(pg, tmp_path):
    app, _, queue, _, request, run = setup(pg, tmp_path)
    with TestClient(app) as client:
        app.state.workspace.put("run", "selected", run)
        app.state.workspace.put(
            "run", "foreign-run", {**run, "id": "foreign-run", "project_id": "other"}
        )
        headers = login(client, "author")
        for changes, expected in (
            ({"source_sha256": "0" * 64}, 409),
            ({"profile": 1}, 409),
            ({"run_id": "absent"}, 404),
            ({"run_id": "foreign-run"}, 403),
            ({"profile": 99}, 422),
        ):
            response = client.post(
                "/api/v2/cases", headers=headers, json={**request, **changes}
            )
            assert response.status_code == expected, response.text
        bad_review = {
            "package": "general",
            "review_id": "card",
            "queue_sha256": body(queue)["queue_sha256"],
            "expected_revision": 0,
        }
        response = client.post(
            "/api/v2/cases",
            headers=headers,
            json={
                **request,
                "region": {**request["region"], "row_end": 2},
                "review": bad_review,
            },
        )
        assert response.status_code == 409
        assert (
            client.post("/api/v2/cases", json=request).status_code == 403
        )  # Missing CSRF
        pending = client.post(
            "/api/v2/cases", headers=headers, json={**request, "review": bad_review}
        )
        assert pending.status_code == 201 and pending.json()["review"] is None
        case = pending.json()
        app.state.workspace.delete("run", "selected")
        assert not client.get("/api/v2/cases/" + case["id"]).json()[
            "research_available"
        ]
        path = tmp_path / "case.zip"
        path.write_bytes(client.get("/api/v2/cases/" + case["id"] + "/bundle").content)
        with zipfile.ZipFile(path) as archive:
            files = {name: archive.read(name) for name in archive.namelist()}
        # The manifest remains unchanged while one evidence byte is modified.
        files["evidence.json"] += b" "
        with zipfile.ZipFile(path, "w") as archive:
            for name, payload in files.items():
                archive.writestr(name, payload)
        with pytest.raises(ValueError):
            read_bundle(path)
        files["../escape"] = b"No extraction allowed"
        with zipfile.ZipFile(path, "w") as archive:
            for name, payload in files.items():
                archive.writestr(name, payload)
        with pytest.raises(ValueError):
            read_bundle(path)


def test_case_idempotency_concurrent_replay_and_account_scope(pg, tmp_path):
    app, root, queue, _record, request, run = setup(pg, tmp_path)
    with TestClient(app) as client:
        app.state.workspace.put("run", "selected", run)
        headers = {**login(client, "author"), "Idempotency-Key": "fixed-attempt"}
        vote = {**body(queue), "reviewer": "author"}
        suffix = "?package=general&independent=true&reviewer=author"
        assert (
            client.post(
                "/api/v1/reviews" + suffix,
                headers={**headers, "Idempotency-Key": "review-first"},
                json=vote,
            ).status_code
            == 200
        )
        request["review"] = {
            "package": "general",
            "review_id": "card",
            "queue_sha256": vote["queue_sha256"],
            "expected_revision": 1,
        }

        def save(_):
            response = client.post("/api/v2/cases", headers=headers, json=request)
            assert response.status_code == 201, response.text
            return response.json()

        with ThreadPoolExecutor(max_workers=3) as pool:
            copies = list(pool.map(save, range(3)))
        original = copies[0]
        assert all(
            c["id"] == original["id"] and c["result"] == original["result"]
            for c in copies
        )
        assert len(client.get("/api/v2/cases").json()) == 1
        assert (
            client.post(
                "/api/v2/cases", headers=headers, json={**request, "notes": "different"}
            ).status_code
            == 409
        )
        assert (
            client.post(
                "/api/v1/reviews" + suffix,
                headers={**headers, "Idempotency-Key": "review-new"},
                json={**vote, "expected_revision": 1, "notes": "New answer"},
            ).status_code
            == 200
        )
        with app.state.workspace.connect() as db:
            db.execute("DELETE FROM documents WHERE kind='run' AND id='selected'")
        replay = save(0)
        assert (
            replay["result"] == original["result"] and not replay["research_available"]
        )
        viewer_headers = {**login(client, "viewer"), "Idempotency-Key": "fixed-attempt"}
        app.state.workspace.put("run", "selected", run)
        viewer = client.post(
            "/api/v2/cases", headers=viewer_headers, json={**request, "review": None}
        )
        assert viewer.status_code == 403
        other=client.post('/api/v2/cases',headers={**login(client,'analyst'),'Idempotency-Key':'fixed-attempt'},json={**request,'review':None})
        assert other.status_code == 201 and other.json()["id"] != original["id"]
        client.headers["X-DTS-Project"] = "other"
        foreign_login = client.post(
            "/api/v2/auth/login",
            json={"username": "foreign", "password": "QA-only-password-123456"},
        )
        foreign_headers = {
            "X-Workbench-Token": foreign_login.json()["csrf"],
            "Idempotency-Key": "fixed-attempt",
        }
        assert (
            client.post(
                "/api/v2/cases", headers=foreign_headers, json=request
            ).status_code
            == 403
        )
    app = create_app(root / "records", queue, root / "runtime")
    with TestClient(app) as client:
        headers = {**login(client, "author"), "Idempotency-Key": "fixed-attempt"}
        replay = client.post("/api/v2/cases", headers=headers, json=request)
        assert (
            replay.status_code == 201 and replay.json()["result"] == original["result"]
        )
        assert replay.json()["id"] == original["id"]
        assert (
            client.post(
                "/api/v2/cases",
                headers={**headers, "Idempotency-Key": "x" * 129},
                json=request,
            ).status_code
            == 422
        )
        with pg.connect() as db:
            db.execute("UPDATE users SET enabled=false WHERE username='author'")
        assert (
            client.post("/api/v2/cases", headers=headers, json=request).status_code
            == 401
        )
