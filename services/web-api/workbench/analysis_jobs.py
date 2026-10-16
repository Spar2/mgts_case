"""Durable native analysis jobs and guarded atomic publication."""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

from fastapi import Header, HTTPException
from psycopg.types.json import Jsonb
from pydantic import BaseModel, ConfigDict, Field

from .analysis_compute import METHOD, ML_COMMIT, verify_pin
from .analysis_results import import_package
from .heartbeat import publish
from .platform import LeaseLost, principal, worker_lease
from .workspace_routes import Region

TIMEOUT = 180


class AnalysisJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    region: Region
    source_sha256: str = Field(pattern="^[0-9a-f]{64}$")
    stream_id: str | None = Field(default=None, min_length=1, max_length=128)
    settings_id: str | None = Field(default=None, pattern="^[0-9a-f]{32}$")
    settings_revision: int | None = Field(default=None, ge=1)


class AnalysisCapability(BaseModel):
    ready: bool
    method: str
    method_version: str
    max_cells: int
    timeout_s: int


class AnalysisJobView(BaseModel):
    id: str
    status: str
    stage: str
    region: Region
    source_sha256: str
    method: str
    method_version: str
    created_at: float
    updated_at: float
    cancel_requested: bool
    run_id: str | None = None
    error: str | None = None
    settings_id: str | None = None
    settings_revision: int | None = None
    settings_revision_after: int | None = None
    author_name: str | None = None


class AnalysisPreflight(BaseModel):
    accepted: bool
    ready: bool
    method: str
    method_version: str
    max_cells: int
    timeout_s: int
    warning: str | None = None


def capability(runtime):
    try:
        value = json.loads(
            (Path(runtime) / "analysis-worker.json").read_text(encoding="utf-8")
        )
        return value["commit"] == ML_COMMIT and 0 <= time.time() - value["at"] <= 30
    except (OSError, ValueError, KeyError, TypeError):
        return False


def analysis_beat(runtime):
    path = Path(runtime) / "analysis-worker.json"
    publish(path, json.dumps({"commit": ML_COMMIT, "at": time.time()}))


def validate_region(store, body):
    native = store.get(body.region.record_id)
    if body.source_sha256 != native.manifest["source_sha256"]:
        raise HTTPException(409, "Source SHA changed; reopen the source")
    r = body.region
    if not (
        0 <= r.row_start < r.row_end <= native.raw_c.shape[0]
        and 0 <= r.column_start < r.column_end <= native.raw_c.shape[1]
        and r.column_end - r.column_start >= 2
    ):
        raise HTTPException(422, "Select native bounds with at least two columns")
    rows, columns = r.row_end - r.row_start, r.column_end - r.column_start
    span = float(native.x_m[r.column_end - 1] - native.x_m[r.column_start])
    if (
        rows * columns > 1_000_000
        or span > 200_000
        or rows * (span / 2 + 1) > 4_000_000
    ):
        raise HTTPException(413, "Window exceeds native/halo budget; reduce the window")


def same_submission(payload, body):
    incoming = body.model_dump(exclude_none=True)
    stored = {key: payload[key] for key in incoming if key in payload}
    return stored == incoming


def validate_submission(app, body):
    validate_region(app.state.store, body)
    platform = app.state.platform
    if body.stream_id:
        try:
            stream = platform.get("stream", body.stream_id)
        except KeyError:
            raise HTTPException(404, "Replay not found") from None
        if (
            stream["record_id"] != body.region.record_id
            or stream["source_sha256"] != body.source_sha256
            or body.region.row_end > stream.get("verified_persisted", -1) + 1
        ):
            raise HTTPException(409, "Analysis window exceeds confirmed replay data")
    if (body.settings_id is None) != (body.settings_revision is None):
        raise HTTPException(422, "Pass settings_id and settings_revision together")
    if body.settings_id is None:
        return None
    from .pipeline_settings import assert_submission_matches, read_setting

    setting = read_setting(platform, body.settings_id)
    assert_submission_matches(setting, body)
    return setting


def enqueue_with_setting(platform, body, key, project):
    """Job insert and setting link commit together; the payload keeps the snapshot."""
    from .pipeline_settings import (
        assert_submission_matches,
        replace_setting,
        setting_for_update,
    )

    identity = uuid.uuid4().hex
    now = time.time()
    with platform.connect() as db:
        db.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s,0))",
            ("jobs:" + project,),
        )
        previous = db.execute(
            "SELECT id,kind,payload,project_id FROM jobs WHERE dedup_key=%s",
            (key,),
        ).fetchone()
        if previous:
            if (
                previous[1] != "analysis_field"
                or previous[3] != project
                or not same_submission(previous[2], body)
            ):
                raise HTTPException(409, "Idempotency key has different content")
            return previous[0]
        user = principal.get()
        if user:
            active = db.execute(
                "SELECT count(*) FROM jobs WHERE project_id=%s AND status IN ('queued','running') AND submitted_by=%s",
                (project, user["id"]),
            ).fetchone()[0]
            if active:
                raise HTTPException(
                    429,
                    "У вас уже есть активная задача",
                    headers={"Retry-After": "10"},
                )
        if (
            db.execute(
                "SELECT count(*) FROM jobs WHERE project_id=%s AND status='queued'",
                (project,),
            ).fetchone()[0]
            >= 10
        ):
            raise HTTPException(
                429, "Очередь проекта заполнена", headers={"Retry-After": "10"}
            )
        setting = setting_for_update(db, body.settings_id)
        assert_submission_matches(setting, body)
        accepted = setting["revision"]
        payload = {
            **body.model_dump(exclude_none=True),
            "method": setting["method"],
            "method_version": setting["method_version"],
            "window": setting["window"],
            "source_sha256": setting["source_sha256"],
            "point": setting["point"],
            "parameters": setting["parameters"],
            "revision": accepted,
            "settings_id": setting["id"],
            "settings_revision": accepted,
            "settings_revision_after": accepted + 1,
            "author_name": user["display_name"] if user else None,
        }
        row = db.execute(
            "INSERT INTO jobs(id,project_id,kind,payload,dedup_key,created,updated,submitted_by) VALUES(%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(dedup_key) DO UPDATE SET dedup_key=excluded.dedup_key RETURNING id,kind,payload,project_id",
            (
                identity,
                project,
                "analysis_field",
                Jsonb(payload),
                key,
                now,
                now,
                user["id"] if user else None,
            ),
        ).fetchone()
        if row[1:] != ("analysis_field", payload, project):
            raise HTTPException(409, "Idempotency key has different content")
        if row[0] != identity:
            return row[0]
        replace_setting(
            db,
            {
                **setting,
                "job_ids": [*setting["job_ids"], row[0]],
                "revision": accepted + 1,
                "updated_at": now,
            },
        )
    return identity


def install_analysis_jobs(app):
    def access(write=False):
        user = principal.get()
        if not user:
            raise HTTPException(401, "Sign in first")
        if user["project_id"] != "local" or (
            write and user["role"] not in ("analyst", "admin")
        ):
            raise HTTPException(403, "Project role does not permit calculation")
        if not app.state.platform:
            raise HTTPException(503, "PostgreSQL and analysis worker are required")
        return user, app.state.platform

    def view(job):
        access()
        if (
            job["project_id"] != principal.get()["project_id"]
            or job["kind"] != "analysis_field"
        ):
            raise HTTPException(403, "Job belongs to another project or kind")
        try:
            stage = app.state.platform.get("analysis_stage", job["id"])["stage"]
        except KeyError:
            stage = "queued"
        if job["status"] in ("queued", "complete", "failed", "cancelled"):
            stage = job["status"]
        return {
            **{
                k: job[k]
                for k in (
                    "id",
                    "status",
                    "created_at",
                    "updated_at",
                    "cancel_requested",
                )
            },
            **job["payload"],
            "stage": stage,
            "run_id": (job.get("result") or {}).get("run_id"),
            "error": "Calculation did not finish; check worker readiness or choose a smaller window"
            if job["status"] == "failed"
            else None,
        }

    @app.get("/api/v2/analysis-capability", response_model=AnalysisCapability)
    def ready():
        access()
        return {
            "ready": capability(app.state.platform.blobs.root.parent),
            "method": METHOD,
            "method_version": ML_COMMIT,
            "max_cells": 1_000_000,
            "timeout_s": TIMEOUT,
        }

    @app.post("/api/v2/analysis-jobs", response_model=AnalysisJobView, status_code=202)
    def submit(
        body: AnalysisJobRequest,
        idempotency_key: str = Header(min_length=8, max_length=128),
    ):
        user, platform = access(True)
        key = f"analysis:{user['project_id']}:{user['id']}:{idempotency_key}"
        # A committed repeat remains available even when the worker/source is offline.
        with platform.connect() as db:
            previous = db.execute(
                "SELECT id,payload FROM jobs WHERE dedup_key=%s", (key,)
            ).fetchone()
        if previous:
            if not same_submission(previous[1], body):
                raise HTTPException(409, "Idempotency key has different content")
            return view(platform.job(previous[0]))
        validate_submission(app, body)
        if not capability(platform.blobs.root.parent):
            raise HTTPException(
                503,
                "Pinned analysis worker is unavailable",
                headers={"Retry-After": "10"},
            )
        if body.settings_id is None:
            payload = {
                **body.model_dump(exclude_none=True),
                "method": METHOD,
                "method_version": ML_COMMIT,
            }
            return view(
                platform.enqueue(
                    "analysis_field", payload, key=key, project=user["project_id"]
                )
            )
        return view(
            platform.job(
                enqueue_with_setting(platform, body, key, user["project_id"])
            )
        )

    @app.post(
        "/api/v2/analysis-preflight",
        response_model=AnalysisPreflight,
        response_model_exclude_none=True,
    )
    def analysis_preflight(body: AnalysisJobRequest):
        _, platform = access(True)
        validate_submission(app, body)
        ready = capability(platform.blobs.root.parent)
        return {
            "accepted": ready,
            "ready": ready,
            "method": METHOD,
            "method_version": ML_COMMIT,
            "max_cells": 1_000_000,
            "timeout_s": TIMEOUT,
            "warning": None if ready else "Pinned analysis worker is unavailable",
        }

    @app.get("/api/v2/analysis-jobs/{key}", response_model=AnalysisJobView)
    def read(key: str):
        _, platform = access()
        return view(platform.job(key))


def guard(platform, job, deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Analysis exceeded 180 seconds")
    lease = worker_lease.get()
    if not lease:
        raise LeaseLost("Analysis requires an owned lease")
    with platform.connect() as db:
        row = db.execute(
            "SELECT cancel FROM jobs WHERE id=%s AND owner=%s AND status='running' AND lease_until>%s",
            (job["id"], lease["owner"], time.time()),
        ).fetchone()
    if not row:
        raise LeaseLost("Analysis lease lost")
    if row[0]:
        raise InterruptedError("Analysis cancelled")


def stop_child(process):
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=10)


def supervise(platform, job, command, deadline, stop=None):
    options = (
        {
            "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.CREATE_NO_WINDOW
        }
        if os.name == "nt"
        else {"start_new_session": True}
    )
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
        **options,
    )
    try:
        while process.poll() is None:
            if stop is not None and stop.is_set():
                raise InterruptedError("Worker is stopping")
            guard(platform, job, deadline)
            time.sleep(0.1)
        guard(platform, job, deadline)
        if process.returncode:
            raise RuntimeError("Pinned calculation failed")
    finally:
        stop_child(process)


def execute_analysis(platform, store, job, stop=None):
    if (
        job["payload"]["method_version"] != ML_COMMIT
        or job["payload"]["method"] != METHOD
    ):
        raise ValueError("Unsupported pinned method")
    verify_pin()
    validate_region(
        store,
        AnalysisJobRequest.model_validate(
            {k: job["payload"][k] for k in ("region", "source_sha256")}
        ),
    )
    runtime = platform.blobs.root.parent
    deadline = time.monotonic() + TIMEOUT
    platform.put(
        "analysis_stage",
        job["id"],
        {"stage": "computing", "project_id": job["project_id"]},
    )
    with tempfile.TemporaryDirectory(prefix="analysis-job-", dir=runtime) as directory:
        root = Path(directory)
        request = root / "request.json"
        request.write_text(
            json.dumps({**job["payload"], "records": str(store.root)}), encoding="utf-8"
        )
        output = root / "result"
        supervise(
            platform,
            job,
            [
                sys.executable,
                "-m",
                "workbench.analysis_compute",
                str(request),
                str(output),
            ],
            deadline,
            stop=stop,
        )
        platform.put(
            "analysis_stage",
            job["id"],
            {"stage": "validating", "project_id": job["project_id"]},
        )
        lease = worker_lease.get()

        def publish(db, run):
            guard(platform, job, deadline)
            row = db.execute(
                "SELECT cancel,submitted_by FROM jobs WHERE id=%s AND owner=%s AND status='running' AND lease_until>%s FOR UPDATE",
                (job["id"], lease["owner"], time.time()),
            ).fetchone()
            if not row:
                raise LeaseLost("Analysis lease lost before publication")
            if row[0]:
                raise InterruptedError("Analysis cancelled before publication")
            if not db.execute(
                "SELECT 1 FROM memberships m JOIN users u ON u.id=m.user_id WHERE m.user_id=%s AND m.project_id=%s AND u.enabled AND m.role IN ('analyst','admin')",
                (row[1], job["project_id"]),
            ).fetchone():
                raise PermissionError("Calculation author is no longer permitted")
            db.execute(
                "UPDATE jobs SET status='complete',result=%s,owner=NULL,updated=%s WHERE id=%s",
                (Jsonb({"run_id": run["id"]}), time.time(), job["id"]),
            )

        with platform.connect() as db:
            author = db.execute(
                "SELECT submitted_by FROM jobs WHERE id=%s", (job["id"],)
            ).fetchone()[0]
        run = import_package(
            platform, store, output, job["project_id"], author, publish=publish
        )
        return {"run_id": run["id"]}
