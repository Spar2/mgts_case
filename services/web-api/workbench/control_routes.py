"""Project control plane, durable computations and reproducible investigations."""

import asyncio
import json
import math
import os
import time
import uuid

from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse
from psycopg import OperationalError
from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

from .health import Readiness, heartbeat_age
from .platform import principal
from .workspace_routes import Region


class JobRequest(BaseModel):
    kind: str
    record_id: str | None = None
    filename: str | None = None
    delay_s: float = Field(default=0.05, ge=0.01, le=10)


class Passport(BaseModel):
    title: str = Field(min_length=1, max_length=150)
    notes: str = Field(default="", max_length=8000)
    region: Region
    profile: int = Field(ge=0)
    method: str
    source_sha256: str = Field(pattern="^[0-9a-f]{64}$")
    scale: tuple[float, float]
    proposal_id: str | None = Field(default=None, max_length=200)
    run_id: str | None = Field(default=None, min_length=1, max_length=128)


class CaseUpdate(BaseModel):
    package: str = "real-data-v2"
    review_id: str
    state: str = "new"
    assigned_to: str | None = None
    expected_revision: int = Field(default=0, ge=0)
    note: str = Field(default="", max_length=2000)


def install_control(app):
    readiness = Readiness(os.environ.get("DTS_RUNTIME", ".runtime"))

    def platform():
        if not app.state.platform:
            raise HTTPException(503, "Configure PostgreSQL and worker for this feature")
        return app.state.platform

    def require(*roles):
        user = principal.get()
        if not user or user["role"] not in roles:
            raise HTTPException(403, "Project role does not permit this action")
        return user

    def require_archive(user):
        if user["project_id"] != "local":
            raise HTTPException(403, "Native archive belongs to the local project")

    @app.get("/api/v2/health/live")
    def live():
        return {"alive": True}

    @app.get("/api/v2/health/ready")
    def ready():
        db = platform()
        with db.connect() as conn:
            conn.execute("SELECT 1")
        age = heartbeat_age(os.environ.get("DTS_RUNTIME", ".runtime"), "compute")
        if age is None or age > 30:
            raise HTTPException(503, "Worker heartbeat unavailable")
        return {
            "database": True,
            "worker_age_s": age,
            "records": len(app.state.store.records),
        }

    @app.get("/api/v2/status")
    def status():
        db = platform()
        user = principal.get()
        with db.connect() as conn:
            jobs = [
                dict(zip(("status", "count"), r))
                for r in conn.execute(
                    "SELECT status,count(*) FROM jobs WHERE project_id=%s GROUP BY status",
                    (user["project_id"],),
                ).fetchall()
            ]
        services = readiness.get()
        return {
            "jobs": jobs,
            "sources": len(app.state.store.records)
            if user["project_id"] == "local"
            else 0,
            "mode": "local_authenticated",
            "capabilities": {
                "archive": user["project_id"] == "local",
                "stream_replay": user["project_id"] == "local"
                and all(
                    services[k]
                    for k in ("stream_worker", "ingestion", "redis", "clickhouse")
                ),
                "prepare_views": services["compute_worker"],
                "equipment": False,
            },
            "data_path": "real CSV → backend → Redis → ClickHouse → confirmed snapshot → web",
            "trained_classifier_operational": False,
            "services": services,
        }

    @app.get("/api/v2/service-capabilities")
    def service_capabilities():
        user = require("viewer", "expert", "analyst", "admin")
        require_archive(user)
        from .analysis_compute import ML_COMMIT
        from .analysis_jobs import capability

        services = readiness.get()
        return {
            "legacy_profile": {
                "available": services["legacy_ml"],
                "method_version": services["legacy_ml_version"],
                "methods": ["identity", "median_x3", "median_x5"],
                "reason": None
                if services["legacy_ml"]
                else "ML profile service unavailable; saved data remain accessible",
            },
            "analysis_field": {
                "available": capability(os.environ.get("DTS_RUNTIME", ".runtime")),
                "method_version": ML_COMMIT,
            },
            "replay": {
                "available": all(
                    services[k]
                    for k in ("stream_worker", "ingestion", "redis", "clickhouse")
                ),
                "mode": "measurement_replay",
            },
            "checked_at": services["checked_at"],
        }

    @app.get("/api/v2/review-experts")
    def review_experts():
        user = require("analyst", "admin")
        require_archive(user)
        with platform().connect() as db:
            return [
                {"id": row[0], "display_name": row[1]}
                for row in db.execute(
                    "SELECT u.id,u.display_name FROM users u JOIN memberships m ON m.user_id=u.id WHERE m.project_id=%s AND m.role IN ('expert','admin') AND u.enabled ORDER BY u.display_name,u.id",
                    (user["project_id"],),
                ).fetchall()
            ]

    @app.get("/api/v2/jobs")
    def jobs(limit: int = 50, offset: int = 0):
        user = principal.get()
        db = platform()
        if not 1 <= limit <= 100 or offset < 0:
            raise HTTPException(422, "Invalid pagination")
        with db.connect() as conn:
            ids = conn.execute(
                "SELECT id FROM jobs WHERE project_id=%s ORDER BY created DESC,id LIMIT %s OFFSET %s",
                (user["project_id"], limit, offset),
            ).fetchall()
        return {
            "items": [db.job(r[0]) for r in ids],
            "next_offset": offset + limit if len(ids) == limit else None,
        }

    @app.post("/api/v2/jobs", status_code=202)
    def submit(body: JobRequest, request: Request):
        user = require("analyst", "admin")
        if body.kind not in ("prepare_views", "stream_replay", "import_csv"):
            raise HTTPException(422, "Unsupported task")
        if user["project_id"] != "local":
            raise HTTPException(403, "Native archive belongs to the local project")
        if body.kind == "stream_replay":
            source = app.state.store.get(body.record_id)
        if body.kind == "import_csv" and (
            not body.filename or not os.environ.get("DTS_IMPORT_ROOT")
        ):
            raise HTTPException(422, "Configure an import directory first")
        key = request.headers.get("idempotency-key")
        result = platform().enqueue(
            body.kind,
            {
                **body.model_dump(),
                **(
                    {"source_sha256": source.manifest["source_sha256"]}
                    if body.kind == "stream_replay"
                    else {}
                ),
            },
            key=f"{user['id']}:{key}" if key else None,
            project=user["project_id"],
        )
        platform().audit("job.submitted", result["id"], user["project_id"])
        return result

    @app.post("/api/v2/jobs/{key}/cancel")
    def cancel(key: str):
        user = require("analyst", "admin")
        db = platform()
        if db.job(key)["project_id"] != user["project_id"]:
            raise HTTPException(403, "Wrong project")
        return db.cancel_job(key)

    @app.post("/api/v2/jobs/{key}/resume", status_code=202)
    def resume(key: str):
        user = require("analyst", "admin")
        return platform().retry_job(key, user["project_id"])

    @app.post("/api/v2/passports", status_code=201)
    def save_passport(body: Passport):
        user = require("analyst", "admin")
        if user["project_id"] != "local":
            raise HTTPException(403, "Native archive belongs to the local project")
        record = app.state.store.get(body.region.record_id)
        if record.manifest["source_sha256"] != body.source_sha256:
            raise HTTPException(409, "Source version changed")
        r = body.region
        if not (
            0 <= r.row_start <= body.profile < r.row_end <= record.raw_c.shape[0]
            and 0 <= r.column_start < r.column_end <= record.raw_c.shape[1]
        ):
            raise HTTPException(422, "Invalid native context")
        selected_run = None
        if body.run_id:
            from .workspace_routes import checked_profile_run

            selected_run, _ = checked_profile_run(
                platform(),
                body.run_id,
                r,
                body.profile,
                body.source_sha256,
                record.raw_c.shape[1],
            )
            if selected_run["method"] != body.method:
                raise HTTPException(409, "Passport method differs from selected run")
        key = uuid.uuid4().hex
        return platform().put(
            "passport",
            key,
            {
                "id": key,
                "project_id": user["project_id"],
                "account_id": user["id"],
                "version": 1,
                **body.model_dump(),
                "source_artifact_sha256": record.manifest["artifact_sha256"],
                "time_kind": record.manifest["time_kind"],
                "method_version": selected_run["method_version"]
                if selected_run
                else "unknown; selection is not evidence of an executed implementation",
                "parameters": selected_run.get("parameters", {})
                if selected_run
                else {
                    "window_points": 3
                    if body.method == "median_x3"
                    else 5
                    if body.method == "median_x5"
                    else None
                },
                "scale_policy": "fixed source percentile 1–99; original values preserved",
                "created_at": time.time(),
                "run_references": [
                    {
                        "id": r["id"],
                        "method": r.get("method"),
                        "method_version": r.get("method_version"),
                        "parameters": r.get("parameters"),
                        "result_ref": r.get("result_ref"),
                    }
                    for r in ([selected_run] if selected_run else [])
                ],
                "automatic_training_labels_created": 0,
            },
        )

    @app.get("/api/v2/passports/{key}")
    def passport(key: str):
        item = platform().get("passport", key)
        if item["project_id"] != principal.get()["project_id"]:
            raise HTTPException(403, "Wrong project")
        return item

    @app.get("/api/v2/experts")
    def experts():
        user = require("expert", "analyst", "admin")
        with platform().connect() as db:
            return [
                {"id": r[0], "display_name": r[1]}
                for r in db.execute(
                    "SELECT u.id,u.display_name FROM users u JOIN memberships m ON m.user_id=u.id WHERE m.project_id=%s AND m.role IN ('expert','admin') AND u.enabled ORDER BY u.display_name",
                    (user["project_id"],),
                ).fetchall()
            ]

    @app.get("/api/v2/case-assignments")
    def cases(package: str = "real-data-v2", limit: int = 50, offset: int = 0):
        require_archive(require("expert", "analyst", "admin"))
        if not 1 <= limit <= 100 or offset < 0:
            raise HTTPException(422, "Invalid pagination")
        from .review_packages import resolve_queue

        queue = resolve_queue(app, package)
        saved = {
            v["review_id"]: v
            for v in platform().list("case", 500)
            if v["queue_sha256"] == queue.sha
        }
        cards = list(queue.cards.values())
        return {
            "items": [
                {
                    "review_id": c["review_id"],
                    "source": c["source"],
                    "queue_sha256": queue.sha,
                    "state": "new",
                    "revision": 0,
                    "assigned_to": None,
                    **saved.get(c["review_id"], {}),
                }
                for c in cards[offset : offset + limit]
            ],
            "total": len(cards),
        }

    @app.post("/api/v2/case-assignments")
    def update_case(body: CaseUpdate):
        user = require("analyst", "admin")
        require_archive(user)
        from .review_packages import resolve_queue

        queue = resolve_queue(app, body.package)
        if body.review_id not in queue.cards or body.state not in (
            "new",
            "assigned",
            "discuss",
            "deferred",
            "closed",
        ):
            raise HTTPException(422, "Invalid case")
        db = platform()
        key = user["project_id"] + ":" + queue.sha + ":" + body.review_id
        with db.connect() as conn:
            conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s,0))", (key,))
            row = conn.execute(
                "SELECT payload FROM documents WHERE kind='case' AND id=%s", (key,)
            ).fetchone()
            old = row[0] if row else {"revision": 0}
            if old["revision"] != body.expected_revision:
                raise HTTPException(409, {"message": "Case changed", "current": old})
            if (
                body.assigned_to
                and not conn.execute(
                    "SELECT 1 FROM memberships m JOIN users u ON u.id=m.user_id WHERE user_id=%s AND project_id=%s AND role IN ('expert','admin') AND u.enabled",
                    (body.assigned_to, user["project_id"]),
                ).fetchone()
            ):
                raise HTTPException(422, "Assign to an enabled expert in this project")
            value = {
                **body.model_dump(),
                "id": key,
                "project_id": user["project_id"],
                "queue_sha256": queue.sha,
                "revision": old["revision"] + 1,
                "updated_at": time.time(),
                "actor": user["id"],
                "training_label": None,
            }
            conn.execute(
                "INSERT INTO documents VALUES('case',%s,%s,%s) ON CONFLICT(kind,id) DO UPDATE SET payload=excluded.payload,updated=excluded.updated",
                (key, Jsonb(value), value["updated_at"]),
            )
            conn.execute(
                "INSERT INTO audit(project_id,actor,action,resource,at) VALUES(%s,%s,'case.updated',%s,%s)",
                (user["project_id"], user["id"], key, value["updated_at"]),
            )
        return value

    app.state.case_assignment_list = cases
    app.state.case_assignment_update = update_case

    @app.get("/api/v2/audit")
    def audit(limit: int = 100):
        user = require("admin")
        with platform().connect() as db:
            return [
                dict(zip(("actor", "action", "resource", "at"), r))
                for r in db.execute(
                    "SELECT actor,action,resource,at FROM audit WHERE project_id=%s ORDER BY id DESC LIMIT %s",
                    (user["project_id"], max(1, min(limit, 100))),
                ).fetchall()
            ]

    @app.get("/api/v2/stream/{key}")
    def snapshot(key: str):
        item = platform().get("stream", key)
        if item["project_id"] != principal.get()["project_id"]:
            raise HTTPException(403, "Wrong project")
        return item

    @app.get("/api/v2/streams")
    def streams(limit: int = 50, offset: int = 0):
        user = require("viewer", "expert", "analyst", "admin")
        require_archive(user)
        if not 1 <= limit <= 100 or offset < 0:
            raise HTTPException(422, "Invalid pagination")
        keys = (
            "id",
            "record_id",
            "source_sha256",
            "mode",
            "status",
            "accepted",
            "persisted",
            "verified_persisted",
            "analyzed",
            "snapshot_version",
            "analyzed_at",
            "storage_poll_at",
            "backlog_profiles",
            "verified_cells",
            "updated_at",
        )
        items = platform().list("stream", limit, offset)
        return {
            "items": [{k: value.get(k) for k in keys} for value in items],
            "next_offset": offset + limit if len(items) == limit else None,
        }

    @app.get("/api/v2/stream/{key}/profile")
    def stream_profile(key: str, snapshot_version: int | None = None):
        user = require("viewer", "expert", "analyst", "admin")
        require_archive(user)
        db = platform()
        value = db.get("stream", key)
        if value["project_id"] != user["project_id"]:
            raise HTTPException(403, "Wrong project")
        if (
            snapshot_version is not None
            and snapshot_version != value["snapshot_version"]
        ):
            raise HTTPException(409, "Confirmed snapshot advanced; reload stream")
        native = app.state.store.get(value["record_id"])
        if native.manifest["source_sha256"] != value["source_sha256"]:
            raise HTTPException(409, "Source changed; no replacement")
        ref = value.get("last_profile_dsp")
        if not ref:
            raise HTTPException(409, "No confirmed DSP profile yet")
        dsp = db.blobs.get(ref)
        row = dsp["row"]
        if (
            isinstance(row, bool)
            or not isinstance(row, int)
            or not 0 <= row < native.raw_c.shape[0]
            or row != value.get("analyzed")
            or row > value.get("verified_persisted", -1)
            or dsp.get("method") != "median_x3"
            or dsp.get("source_sha256") != value["source_sha256"]
            or not isinstance(dsp.get("denoised"), list)
            or len(dsp["denoised"]) != len(native.x_m)
        ):
            raise HTTPException(409, "DSP snapshot differs from confirmed source")
        raw = native.raw_c[row]
        good = native.invalid_reason[row] == 0
        values = dsp["denoised"]
        if any(
            v is not None
            and (
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
            )
            for v in values
        ):
            raise HTTPException(409, "Invalid DSP values")
        return {
            "stream_id": key,
            "snapshot_version": value["snapshot_version"],
            "source_sha256": value["source_sha256"],
            "row": row,
            "elapsed_s": float(native.elapsed_s[row]),
            "timestamp": native.profiles[row]["timestamp"],
            "time_kind": native.manifest["time_kind"],
            "x_m": native.x_m.tolist(),
            "raw_c": [float(v) if flag else None for v, flag in zip(raw, good)],
            "processed_c": [v if flag else None for v, flag in zip(values, good)],
            "invalid_reason": native.invalid_reason[row].tolist(),
            "method": "median_x3",
            "processing_scope": "single_profile_dsp_not_field_analysis",
            "verified_at": value.get("analyzed_at"),
        }

    @app.get("/api/v2/stream/{key}/events")
    async def events(key: str, request: Request):
        user = principal.get()
        db = platform()
        await asyncio.to_thread(db.get, "stream", key)

        async def messages():
            prior = None
            while not await request.is_disconnected():
                from .accounts import session_is_current

                if not await session_is_current(db, request, user):
                    return
                try:
                    item = await asyncio.to_thread(db.get, "stream", key)
                except (KeyError, PermissionError, OperationalError):
                    return
                if item["project_id"] != user["project_id"]:
                    return
                state = {
                    k: item.get(k)
                    for k in (
                        "id",
                        "accepted",
                        "persisted",
                        "analyzed",
                        "status",
                        "record_id",
                        "snapshot_version",
                        "error",
                    )
                }
                encoded = json.dumps(state)
                if encoded != prior:
                    yield f"id: {item.get('snapshot_version', 0)}\nevent: snapshot\ndata: {encoded}\n\n"
                    prior = encoded
                else:
                    yield ": heartbeat\n\n"
                await asyncio.sleep(1)

        return StreamingResponse(
            messages(),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no"},
        )

    disagreement_cache = {}

    @app.get("/api/v2/disagreements")
    def disagreements(package: str = "general", limit: int = 50, offset: int = 0):
        require_archive(require("analyst", "admin"))
        if not 1 <= limit <= 100 or offset < 0:
            raise HTTPException(422, "Invalid pagination")
        from .review_packages import resolve_queue

        queue = resolve_queue(app, package)
        if queue.sha in disagreement_cache:
            cached = disagreement_cache[queue.sha]
            return {
                "items": cached[offset : offset + limit],
                "total": len(cached),
                "next_offset": offset + limit if offset + limit < len(cached) else None,
            }
        items = []
        for card in queue.cards.values():
            if len(card["methods"]) < 2:
                continue
            _, arrays = queue.load(card["review_id"])
            signed = arrays["signed"]
            valid = arrays["invalid_reason"] == 0
            differing = (signed.max(axis=0) != signed.min(axis=0)) & valid
            items.append(
                {
                    "review_id": card["review_id"],
                    "source": card["source"],
                    "disagreement_fraction": float(
                        differing.sum() / max(valid.sum(), 1)
                    ),
                    "methods": card["methods"],
                    "meaning": "method disagreement, not calibrated uncertainty",
                    "source_sha256": card["evidence"]["source_sha256"],
                }
            )
        items.sort(key=lambda v: (-v["disagreement_fraction"], v["review_id"]))
        disagreement_cache[queue.sha] = items
        return {
            "items": items[offset : offset + min(limit, 100)],
            "total": len(items),
            "next_offset": offset + limit if offset + limit < len(items) else None,
        }
