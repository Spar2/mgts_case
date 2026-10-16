"""Actual child calculation, atomic publication and owned-process cancellation."""

import json
import sys
import threading
import time

import pytest
from fastapi.testclient import TestClient
from test_platform import login
from test_platform import pg as platform_pg
from test_workbench import fixture as native_fixture
from workbench.accounts import add_user
from workbench.analysis_compute import ML_COMMIT, PIN, verify_pin
from workbench.analysis_jobs import analysis_beat, execute_analysis, supervise
from workbench.api import create_app
from workbench.platform import LeaseLost, worker_lease

pg = platform_pg


def test_pinned_source_is_exact():
    verify_pin()
    assert json.loads((PIN / "source.json").read_text())["commit"] == ML_COMMIT


@pytest.mark.parametrize("publication_failure", [None, "cancel", "lease", "disabled"])
def test_native_job_atomic_result(pg, tmp_path, monkeypatch, publication_failure):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    root, queue, native = native_fixture.__wrapped__(tmp_path)
    add_user(pg, "compute-admin", "QA-only-password-123456", "Compute", "admin")
    app = create_app(root / "records", queue, root / "runtime")
    body = {
        "region": {
            "record_id": "5",
            "row_start": 0,
            "row_end": 3,
            "column_start": 0,
            "column_end": 3,
        },
        "source_sha256": native.manifest["source_sha256"],
    }
    with TestClient(app) as client:
        headers = {**login(client, "compute-admin"), "Idempotency-Key": "compute-test"}
        assert (
            client.post("/api/v2/analysis-jobs", json=body, headers=headers).status_code
            == 503
        )
        analysis_beat(app.state.platform.blobs.root.parent)
        response = client.post("/api/v2/analysis-jobs", json=body, headers=headers)
        assert response.status_code == 202, response.text
        job = app.state.platform.claim("test-owner", ["analysis_field"])
        assert job["id"] == response.json()["id"]
        assert (
            client.post("/api/v2/analysis-jobs", json=body, headers=headers).json()[
                "id"
            ]
            == job["id"]
        )
        assert (
            client.post(
                "/api/v2/analysis-jobs",
                json={**body, "source_sha256": "0" * 64},
                headers=headers,
            ).status_code
            == 409
        )
        assert (
            client.post(
                "/api/v2/analysis-jobs",
                json=body,
                headers={**headers, "Idempotency-Key": "another-attempt"},
            ).status_code
            == 429
        )
        from workbench import analysis_jobs

        original = analysis_jobs.import_package

        def intercept(*args, **kwargs):
            publish = kwargs["publish"]

            def wrapped(db, run):
                if publication_failure == "cancel":
                    app.state.platform.cancel_job(job["id"])
                elif publication_failure == "lease":
                    with app.state.platform.connect() as conn:
                        conn.execute(
                            "UPDATE jobs SET lease_until=0 WHERE id=%s", (job["id"],)
                        )
                elif publication_failure == "disabled":
                    with app.state.platform.connect() as conn:
                        conn.execute(
                            "UPDATE users SET enabled=false WHERE username='compute-admin'"
                        )
                publish(db, run)

            return original(*args, **{**kwargs, "publish": wrapped})

        monkeypatch.setattr(analysis_jobs, "import_package", intercept)
        token = worker_lease.set({"id": job["id"], "owner": "test-owner"})
        try:
            if publication_failure:
                with pytest.raises(
                    {
                        "cancel": InterruptedError,
                        "lease": LeaseLost,
                        "disabled": PermissionError,
                    }[publication_failure]
                ):
                    execute_analysis(app.state.platform, app.state.store, job)
                assert app.state.platform.list("analysis_run") == []
                assert not list((app.state.platform.blobs.root / "arrays").iterdir())
                assert app.state.platform.job(job["id"])["result"] is None
                if publication_failure == "lease":
                    monkeypatch.setattr(analysis_jobs, "import_package", original)
                    recovered = app.state.platform.claim(
                        "replacement-owner", ["analysis_field"]
                    )
                    assert recovered["id"] == job["id"] and recovered["attempts"] == 2
                    replacement = worker_lease.set(
                        {"id": job["id"], "owner": "replacement-owner"}
                    )
                    try:
                        restored = execute_analysis(
                            app.state.platform, app.state.store, recovered
                        )
                    finally:
                        worker_lease.reset(replacement)
                    assert app.state.platform.job(job["id"])["result"] == restored
                    assert len(app.state.platform.list("analysis_run")) == 1
                    assert not app.state.platform.finish(
                        job["id"], "test-owner", error="stale worker"
                    )
                return
            result = execute_analysis(app.state.platform, app.state.store, job)
        finally:
            worker_lease.reset(token)
        assert app.state.platform.job(job["id"])["status"] == "complete"
        # A worker dying after this commit cannot erase or overwrite its publication.
        assert not app.state.platform.finish(
            job["id"], "test-owner", error="late failure"
        )
        run_id = result["run_id"]
        state = client.get("/api/v2/analysis-jobs/" + job["id"]).json()
        assert state["run_id"] == run_id and state["stage"] == "complete"
        run = client.get("/api/v2/result-runs/" + run_id).json()
        assert (
            run["method_version"] == ML_COMMIT and run["training_origin"] == "untrained"
        )
        slices = client.get(
            f"/api/v2/result-runs/{run_id}/slices?row=0&column=0"
        ).json()
        assert slices["layers"]["raw"]["profile"] == [20, None, 22]
        assert len(slices["layers"]["processed_temperature_c"]["profile"]) == 3
        assert slices["layers"]["processed_temperature_c"]["profile"][1] is None
        (app.state.platform.blobs.root.parent / "analysis-worker.json").unlink()
        assert (
            client.post("/api/v2/analysis-jobs", json=body, headers=headers).json()[
                "id"
            ]
            == job["id"]
        )


@pytest.mark.parametrize("mode", ["timeout", "cancel", "lease", "shutdown"])
def test_supervisor_stops_owned_process_tree(pg, tmp_path, mode):
    job = pg.enqueue("analysis_field", {}, project="local")
    job = pg.claim("supervisor-owner", ["analysis_field"])
    script = tmp_path / "slow.py"
    marker = tmp_path / "escaped.txt"
    script.write_text(
        "import subprocess,sys,time\nsubprocess.Popen([sys.executable,'-c',\"import time;from pathlib import Path;time.sleep(1.5);Path(sys.argv[1]).write_text('escaped')\".replace('import time;','import time,sys;'),sys.argv[1]])\ntime.sleep(60)\n"
    )

    stop = threading.Event()

    def interrupt():
        time.sleep(0.3)
        if mode == "shutdown":
            stop.set()
        elif mode == "cancel":
            pg.cancel_job(job["id"])
        elif mode == "lease":
            with pg.connect() as db:
                db.execute("UPDATE jobs SET lease_until=0 WHERE id=%s", (job["id"],))

    thread = threading.Thread(target=interrupt)
    thread.start()
    token = worker_lease.set({"id": job["id"], "owner": "supervisor-owner"})
    try:
        with pytest.raises(
            {
                "timeout": TimeoutError,
                "cancel": InterruptedError,
                "lease": LeaseLost,
                "shutdown": InterruptedError,
            }[mode]
        ):
            supervise(
                pg,
                job,
                [sys.executable, str(script), str(marker)],
                time.monotonic() + (0.4 if mode == "timeout" else 5),
                stop=stop,
            )
    finally:
        worker_lease.reset(token)
        thread.join()
    time.sleep(1.6)
    assert not marker.exists(), "Child descendant survived cancellation"


def test_submission_permissions_and_parallel_repeat(pg, tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    root, queue, native = native_fixture.__wrapped__(tmp_path)
    add_user(pg, "job-admin", "QA-only-password-123456", "Admin", "admin")
    add_user(pg, "job-viewer", "QA-only-password-123456", "Viewer", "viewer")
    app = create_app(root / "records", queue, root / "runtime")
    body = {
        "region": {
            "record_id": "5",
            "row_start": 0,
            "row_end": 3,
            "column_start": 0,
            "column_end": 3,
        },
        "source_sha256": native.manifest["source_sha256"],
    }
    with TestClient(app) as client:
        assert client.get("/api/v2/analysis-capability").status_code == 401
        viewer = {**login(client, "job-viewer"), "Idempotency-Key": "permissions-test"}
        assert (
            client.post("/api/v2/analysis-jobs", json=body, headers=viewer).status_code
            == 403
        )
        client.post("/api/v2/auth/logout", headers=viewer)
        headers = {**login(client, "job-admin"), "Idempotency-Key": "parallel-repeat"}
        analysis_beat(app.state.platform.blobs.root.parent)
        assert (
            client.post(
                "/api/v2/analysis-jobs",
                json={**body, "method": "other"},
                headers=headers,
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/api/v2/analysis-jobs",
                json={**body, "source_sha256": "0" * 64},
                headers=headers,
            ).status_code
            == 409
        )
        assert (
            client.post(
                "/api/v2/analysis-jobs",
                json={**body, "region": {**body["region"], "row_end": 4}},
                headers=headers,
            ).status_code
            == 422
        )
        assert client.post(
            "/api/v2/analysis-jobs",
            json=body,
            headers={**headers, "X-DTS-Project": "foreign"},
        ).status_code in (401, 403)
        with ThreadPoolExecutor(max_workers=2) as pool:
            replies = list(
                pool.map(
                    lambda _: client.post(
                        "/api/v2/analysis-jobs", json=body, headers=headers
                    ),
                    range(2),
                )
            )
        assert all(r.status_code == 202 for r in replies)
        assert len({r.json()["id"] for r in replies}) == 1
        with app.state.platform.connect() as db:
            assert (
                db.execute(
                    "SELECT count(*) FROM jobs WHERE kind='analysis_field'"
                ).fetchone()[0]
                == 1
            )


def test_pipeline_guards_and_confirmed_cancel(pg, tmp_path):
    """Viewer, SHA, bounds, project, idempotency and server-confirmed cancel."""
    root, queue, native = native_fixture.__wrapped__(tmp_path)
    add_user(pg, "pipe-admin", "QA-only-password-123456", "Admin", "admin")
    add_user(pg, "pipe-viewer", "QA-only-password-123456", "Viewer", "viewer")
    app = create_app(root / "records", queue, root / "runtime")
    body = {
        "region": {
            "record_id": "5",
            "row_start": 0,
            "row_end": 3,
            "column_start": 0,
            "column_end": 3,
        },
        "source_sha256": native.manifest["source_sha256"],
    }
    with TestClient(app) as client:
        viewer = {**login(client, "pipe-viewer"), "Idempotency-Key": "pipeline-viewer"}
        assert (
            client.post("/api/v2/analysis-jobs", json=body, headers=viewer).status_code
            == 403
        )
        print("ASSERT viewer 403")
        client.post("/api/v2/auth/logout", headers=viewer)
        headers = {**login(client, "pipe-admin"), "Idempotency-Key": "pipeline-once"}
        analysis_beat(app.state.platform.blobs.root.parent)
        assert (
            client.post(
                "/api/v2/analysis-jobs",
                json={**body, "source_sha256": "0" * 64},
                headers=headers,
            ).status_code
            == 409
        )
        print("ASSERT wrong sha 409")
        assert (
            client.post(
                "/api/v2/analysis-jobs",
                json={**body, "region": {**body["region"], "row_end": 99}},
                headers=headers,
            ).status_code
            == 422
        )
        print("ASSERT out of range 422")
        assert client.post(
            "/api/v2/analysis-jobs",
            json=body,
            headers={**headers, "X-DTS-Project": "foreign"},
        ).status_code in (401, 403)
        print("ASSERT foreign project rejected")
        created = client.post("/api/v2/analysis-jobs", json=body, headers=headers)
        assert created.status_code == 202, created.text
        repeated = client.post("/api/v2/analysis-jobs", json=body, headers=headers)
        assert repeated.status_code == 202
        assert repeated.json()["id"] == created.json()["id"]
        print("ASSERT idempotent body one job")
        cancelled = client.post(
            "/api/v2/jobs/" + created.json()["id"] + "/cancel", headers=headers
        )
        assert cancelled.status_code == 200, cancelled.text
        confirmed = client.get(
            "/api/v2/analysis-jobs/" + created.json()["id"], headers=headers
        ).json()
        assert cancelled.json()["status"] == confirmed["status"] == "cancelled"
        assert confirmed["run_id"] is None
        assert confirmed["status"] != "complete"
        print("ASSERT cancel server status")
