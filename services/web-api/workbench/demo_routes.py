"""Persistent rehearsal context; answers never enter any expert journal."""

import asyncio
import hashlib
import json
import os
import time
from types import SimpleNamespace
from typing import Literal

import httpx
from fastapi import HTTPException, Query, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from .health import Readiness
from .platform import principal
from .workspace import report_html

COOKIE = "dts_rehearsal"


class DemoCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    client_key: str = Field(pattern="^[a-zA-Z0-9-]{16,64}$")
    package: str = "real-data-v2"
    candidate: str


class DemoAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    temperature_episode: Literal["present", "absent", "ambiguous", "unknown"]
    notes: str = Field(min_length=1, max_length=8000)


class DemoProcessing(BaseModel):
    model_config = ConfigDict(extra="forbid")
    row: int = Field(ge=0)
    method: Literal["median_x3", "median_x5"] = "median_x3"
    run_id: str | None = Field(default=None, max_length=128)


def owner():
    user = principal.get()
    return user["id"] if user else "local-self-declared"


def get_session(app, key, validate=True):
    session = app.state.workspace.get("demo_session", key)
    if session["owner_id"] != owner():
        raise HTTPException(403, "Rehearsal belongs to another participant")
    if session["expires_at"] < time.time():
        raise HTTPException(410, "Rehearsal expired; start a new one")
    if validate:
        record = app.state.store.get(session["region"]["record_id"])
        from .review_packages import resolve_queue
        queue = resolve_queue(app,session["package"])
        if (
            record.manifest["source_sha256"] != session["source_sha256"]
            or queue.sha != session["queue_sha256"]
        ):
            raise HTTPException(409, "Rehearsal source or queue version changed")
    for kind, field in (
        ("demo_answer", "expert_answer"),
        ("demo_processing", "processing"),
    ):
        try:
            session[field] = app.state.workspace.get(kind, key)["value"]
        except KeyError:
            session[field] = None
    return session


def install_demo_routes(app, describe, bounds):
    from .workspace_routes import Region

    readiness = Readiness(os.environ.get("DTS_RUNTIME", ".runtime"))

    def activate(response, key):
        response.set_cookie(
            COOKIE,
            key,
            max_age=48 * 3600,
            httponly=True,
            samesite="strict",
            secure=os.environ.get("DTS_SECURE_COOKIES") == "true",
        )

    def require_active(request, key):
        if request.cookies.get(COOKIE) != key:
            raise HTTPException(409, "Activate this rehearsal before changing it")
        return get_session(app, key)

    def native(session, row, column):
        region = Region(**session["region"])
        row = region.row_start if row is None else row
        column = (
            (region.column_start + region.column_end - 1) // 2
            if column is None
            else column
        )
        if (
            not region.row_start <= row < region.row_end
            or not region.column_start <= column < region.column_end
        ):
            raise HTTPException(422, "Selection must belong to the rehearsal window")
        return region, row, column

    @app.get("/api/v1/demo/state")
    def state(request: Request):
        key = request.cookies.get(COOKIE)
        return {"active": bool(key), "session_id": key}

    @app.post("/api/v1/demo/leave")
    def leave(response: Response):
        response.delete_cookie(COOKIE)
        return {"active": False}

    @app.post("/api/v1/demo/sessions", status_code=201)
    def create(body: DemoCreate, response: Response):
        from .review_packages import resolve_queue
        queue = resolve_queue(app,body.package)
        card = queue.cards[body.candidate]
        key = hashlib.sha256((owner() + ":" + body.client_key).encode()).hexdigest()[
            :32
        ]
        try:
            session = get_session(app, key)
            if (
                session["package"] != body.package
                or session["candidate"] != body.candidate
            ):
                raise HTTPException(
                    409, "Creation key already belongs to another window"
                )
        except KeyError:
            region = Region(
                record_id=card["source"].removesuffix(".csv"),
                row_start=card["row_bounds"][0],
                row_end=card["row_bounds"][1],
                column_start=card["column_bounds"][0],
                column_end=card["column_bounds"][1],
            )
            record = bounds(region)
            session = app.state.workspace.put(
                "demo_session",
                key,
                {
                    "id": key,
                    "owner_id": owner(),
                    "project_id": "local",
                    "package": body.package,
                    "candidate": body.candidate,
                    "queue_sha256": queue.sha,
                    "source_sha256": record.manifest["source_sha256"],
                    "region": region.model_dump(),
                    "proposal_bounds": card.get("proposal_bounds"),
                    "scale": app.state.store.meta[region.record_id]["scale"],
                    "expires_at": time.time() + 48 * 3600,
                    "mode": "rehearsal",
                    "expert_answer": None,
                    "processing": None,
                    "stream_job_id": None,
                },
            )
        activate(response, key)
        return session

    @app.get("/api/v1/demo/sessions/{key}")
    def session(key: str):
        return get_session(app, key)

    @app.post("/api/v1/demo/sessions/{key}/activate")
    def restore(key: str, response: Response):
        session = get_session(app, key)
        activate(response, key)
        return session

    @app.get("/api/v1/demo/sessions/{key}/evidence")
    def evidence(
        key: str,
        row: int | None = Query(None, ge=0),
        column: int | None = Query(None, ge=0),
    ):
        session = get_session(app, key)
        region, row, column = native(session, row, column)
        result = describe(region, row)
        r = bounds(region)
        result["selection"] = app.state.store.slices(
            region.record_id, row, column, region.row_start, region.row_end
        )
        result["profile_invalid_reason"] = r.invalid_reason[
            row, region.column_start : region.column_end
        ].tolist()
        result["profile_segment_id"] = int(r.segment_id[row])
        result["rehearsal"] = True
        return result

    @app.post("/api/v1/demo/sessions/{key}/answer")
    def answer(key: str, body: DemoAnswer, request: Request):
        session = require_active(request, key)
        if not body.notes.strip():
            raise HTTPException(422, "Describe the observed signal")
        value = {
            **body.model_dump(),
            "reviewer": "Репетиция — не экспертная разметка",
            "reviewer_type": "rehearsal",
            "saved_at": time.time(),
            "source_sha256": session["source_sha256"],
            "region": session["region"],
        }
        app.state.workspace.put("demo_answer", key, {"value": value})
        return get_session(app, key)

    def compatible(session, run, row=None):
        if run.get("source_sha256") != session["source_sha256"] or run.get(
            "status"
        ) not in ("complete", "imported"):
            return False
        region = session["region"]
        rr = run.get("region", {})
        return rr == region or (
            rr.get("record_id") == region["record_id"]
            and rr.get("row_end", -1) == rr.get("row_start", -2) + 1
            and region["row_start"] <= rr.get("row_start", -1) < region["row_end"]
            and (row is None or rr["row_start"] == row)
            and rr.get("column_start", -1) <= region["column_start"]
            and rr.get("column_end", -1) >= region["column_end"]
        )

    @app.get("/api/v1/demo/sessions/{key}/results")
    def results(key: str):
        session = get_session(app, key)
        return [
            {k: v for k, v in run.items() if k not in ("result", "result_ref")}
            for run in app.state.workspace.list("run")
            if compatible(session, run)
        ]

    @app.post("/api/v1/demo/sessions/{key}/process")
    async def process(key: str, body: DemoProcessing, request: Request):
        session = await asyncio.to_thread(require_active, request, key)
        _, row, _ = native(session, body.row, None)
        if body.run_id:
            run = await asyncio.to_thread(app.state.workspace.get, "run", body.run_id)
            if not compatible(session, run, row) or not isinstance(
                run.get("result"), dict
            ):
                raise HTTPException(
                    409, "Result unavailable or belongs to another context"
                )
            result = run["result"]
            full = result.get("denoised")
            if full is not None:
                r = bounds(Region(**session["region"]))
                if (
                    run["region"]["row_end"] != row + 1
                    or run["region"]["row_start"] != row
                    or len(full) != r.raw_c.shape[1]
                ):
                    raise HTTPException(
                        422, "Processed profile must identify exactly one native row"
                    )
            metadata = {
                name: run.get(name)
                for name in (
                    "id",
                    "method",
                    "method_version",
                    "parameters",
                    "training_origin",
                    "limitations",
                    "calculated_at",
                )
            }
        else:
            result = await app.state.clean_profile(
                SimpleNamespace(
                    record_id=session["region"]["record_id"],
                    row=row,
                    method=body.method,
                )
            )
            metadata = {
                "method": body.method,
                "method_version": "ML API contract 0.1; implementation SHA not supplied",
                "training_origin": "untrained",
                "parameters": {"method": body.method},
                "limitations": "Очистка профиля; не классификация события",
            }
        # Preserve invalidity even when a prepared method supplies filled temperatures.
        result = dict(result)
        if isinstance(result.get("denoised"), list):
            raw = app.state.store.slices(session["region"]["record_id"], row, 0)[
                "profile"
            ]
            result["denoised"] = [
                v if raw[i] is not None else None
                for i, v in enumerate(result["denoised"])
            ]
        processing = {
            "row": row,
            "source_sha256": session["source_sha256"],
            **metadata,
            "result": result,
        }
        await asyncio.to_thread(
            app.state.workspace.put, "demo_processing", key, {"value": processing}
        )
        return processing

    @app.get("/api/v1/demo/sessions/{key}/preflight")
    async def preflight(key: str):
        session = await asyncio.to_thread(get_session, app, key)
        services = await asyncio.to_thread(readiness.get)
        ml = False
        try:
            async with httpx.AsyncClient(timeout=2, trust_env=False) as client:
                ml = (
                    await client.get(
                        os.environ.get("DTS_ML_URL", "http://127.0.0.1:8082").rstrip(
                            "/"
                        )
                        + "/health"
                    )
                ).is_success
        except httpx.RequestError:
            pass
        user = principal.get()
        return {
            "source": True,
            "queue": True,
            "session_persisted": True,
            "ml_available": ml,
            "prepared_results": len(await asyncio.to_thread(results, key)),
            "services": services,
            "stream_permitted": bool(
                user and user["role"] in ("analyst", "admin") and app.state.platform
            ),
            "rehearsal_guard": True,
            "source_sha256": session["source_sha256"],
        }

    @app.post("/api/v1/demo/sessions/{key}/report")
    def report(
        key: str,
        request: Request,
        row: int | None = Query(None, ge=0),
        format: Literal["html", "json"] = "html",
    ):
        session = get_session(app, key)
        region, row, _ = native(session, row, None)
        content = describe(region, row)
        r = bounds(region)
        content["profile"]["invalid_reason"] = r.invalid_reason[
            row, region.column_start : region.column_end
        ].tolist()
        content["profile"]["segment_id"] = int(r.segment_id[row])
        content.update(
            {
                "rehearsal": True,
                "expert_answer": session["expert_answer"],
                "queue_sha256": session["queue_sha256"],
                "rehearsal_session_id": key,
            }
        )
        processing = session.get("processing")
        if processing and processing["row"] == row:
            content["processing"] = processing
            clean = processing["result"].get("denoised")
            if isinstance(clean, list):
                content["profile"]["processed_temperature_c"] = clean[
                    region.column_start : region.column_end
                ]
        result = (
            report_html(content)
            if format == "html"
            else json.dumps(content, ensure_ascii=False, allow_nan=False)
        )
        return Response(
            result,
            media_type="text/html" if format == "html" else "application/json",
            headers={
                "Content-Disposition": f'attachment; filename="dts-rehearsal.{format}"'
            },
        )

    @app.post("/api/v1/demo/sessions/{key}/stream", status_code=202)
    def start_stream(key: str, request: Request):
        session = require_active(request, key)
        user = principal.get()
        if (
            not app.state.platform
            or not user
            or user["role"] not in ("analyst", "admin")
        ):
            raise HTTPException(
                403, "Stream rehearsal requires analyst or administrator"
            )
        if session.get("stream_job_id"):
            return app.state.platform.job(session["stream_job_id"])
        job = app.state.platform.enqueue(
            "stream_replay",
            {
                "record_id": session["region"]["record_id"],
                "source_sha256": session["source_sha256"],
                "delay_s": 0.1,
                "purpose": "demo_rehearsal",
                "demo_session_id": key,
            },
            key="demo-stream:" + key,
            project="local",
        )
        session["stream_job_id"] = job["id"]
        app.state.workspace.put("demo_session", key, session)
        return job

    @app.get("/api/v1/demo/sessions/{key}/stream")
    def stream_state(key: str):
        session = get_session(app, key)
        job_id = session.get("stream_job_id")
        if not job_id or not app.state.platform:
            return {"job": None, "snapshot": None}
        try:
            snapshot = app.state.platform.get("stream", job_id)
            # Native snapshots store inclusive row indices; the presentation shows counts.
            snapshot["native_row_indices"] = {
                name: snapshot.get(name, -1)
                for name in ("accepted", "persisted", "analyzed", "verified_persisted")
            }
            for name, value in snapshot["native_row_indices"].items():
                snapshot[name] = value + 1
            snapshot["total_profiles"] = app.state.store.get(
                session["region"]["record_id"]
            ).raw_c.shape[0]
        except KeyError:
            snapshot = None
        return {"job": app.state.platform.job(job_id), "snapshot": snapshot}

    @app.post("/api/v1/demo/sessions/{key}/stream/{action}")
    def stream_command(key: str, action: Literal["cancel", "resume"], request: Request):
        session = require_active(request, key)
        user = principal.get()
        if (
            not app.state.platform
            or not user
            or user["role"] not in ("analyst", "admin")
        ):
            raise HTTPException(
                403, "Stream rehearsal requires analyst or administrator"
            )
        job_id = session.get("stream_job_id")
        if not job_id:
            raise HTTPException(409, "Start a technical replay first")
        return (
            app.state.platform.cancel_job(job_id)
            if action == "cancel"
            else app.state.platform.retry_job(job_id, "local")
        )
