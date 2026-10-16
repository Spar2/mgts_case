"""Loopback-only workbench gateway; never exposes ingestion or storage credentials."""

import asyncio
import hashlib
import json
import os
import secrets
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from urllib.parse import quote

import httpx
import numpy as np
from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from psycopg import OperationalError
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.gzip import GZipMiddleware

from .access_policy import protect_response, public_asset, reject_untrusted
from .accounts import authenticate, install_accounts, session_is_current, touch_session
from .control_routes import install_control
from .platform import JobConflict, configured, isolated_anonymous_test, principal
from .public_demo import PUBLIC_PATHS, install_public, load_package
from .rate_limits import RateLimiter
from .review_packages import resolve_queue
from .store import Queue, Store, cell_edges, packet
from .vendor.reviews import ReviewConflict, ReviewJournal
from .workspace import Workspace, adapt_real_pack
from .workspace_routes import install_workspace_routes

BASE = next(
    (p for p in Path(__file__).resolve().parents if (p / "apps/web").is_dir()),
    Path.cwd(),
)


class Review(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    queue_sha256: str
    review_id: str
    expected_revision: int = Field(ge=0)
    reviewer: str = Field(min_length=1, max_length=100)
    reviewer_type: Literal["expert", "agent"]
    temperature_episode: Literal["present", "absent", "ambiguous", "unknown"]
    phenomenon: Literal["local_change", "common_trend", "dropout", "mixed", "unknown"]
    notes: str = Field(default='',max_length=8000)


class ReplayCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    record_id: str


class ReplayCommand(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    action: Literal["play", "pause", "seek", "speed"]
    row: int | None = Field(default=None, ge=0)
    speed: Literal[1, 10, 60] = 1


class Denoise(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    record_id: str
    row: int = Field(ge=0)
    method: Literal["median_x3", "median_x5", "identity"] = "median_x3"


class DenoiseResult(BaseModel):
    model_config = ConfigDict(extra="allow")
    run_id: str
    source_sha256: str
    row: int
    denoised: list[float | None]


class Detect(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    record_id: str
    row_start: int = Field(ge=0)
    row_end: int = Field(ge=1)
    column_start: int = Field(ge=0)
    column_end: int = Field(ge=1)


class ReplayState(BaseModel):
    id: str | None = None
    record_id: str
    row: int
    playing: bool
    speed: Literal[1, 10, 60]
    position_s: float


class ReplayCreated(ReplayState):
    id: str


class RecordMeta(BaseModel):
    id: str
    source: str
    source_sha256: str
    shape: tuple[int, int]
    x_bounds: tuple[float, float]
    time_bounds: tuple[float, float]
    scale: tuple[float, float]
    first_timestamp: str
    last_timestamp: str
    time_kind: str
    segments: int
    valid_fraction: float
    dts_id: str
    channel_id: str


class RecordDetail(RecordMeta):
    timestamps: list[str]
    time_s: list[float]
    x_m: list[float]
    segment_id: list[int]


class Slices(BaseModel):
    row: int
    column: int
    x_m: list[float]
    profile: list[float | None]
    time_s: list[float]
    trend: list[float | None]
    timestamp: str
    elapsed_s: float
    distance_m: float
    temperature: float | None
    invalid_reason: int
    source_sha256: str


class Card(BaseModel):
    review_id: str
    source: str
    source_sha256: str
    native_shape: tuple[int, int]
    x_bounds_m: tuple[float, float]
    time_bounds_s: tuple[float, float]
    timestamps: list[str]
    row_bounds: tuple[int, int]
    column_bounds: tuple[int, int]
    shortlist: bool = False


class Annotation(Review):
    model_config = ConfigDict(extra="allow")
    revision: int
    saved_at: str


class ReviewQueue(BaseModel):
    queue_sha256: str
    cards: list[Card]
    answers: dict[str, list[str]]
    annotations: dict[str, Annotation]


def create_app(records=None, queue=None, runtime=None):
    records = Path(records or os.environ.get("DTS_RECORDS", BASE / ".runtime/records"))
    queue = Path(queue or os.environ.get("DTS_QUEUE", BASE / ".runtime/queue"))
    runtime = Path(runtime or os.environ.get("DTS_RUNTIME", BASE / ".runtime"))

    @asynccontextmanager
    async def lifespan(app):
        if os.environ.get("DTS_ENV") == "production":
            if (
                os.environ.get("DTS_SECURE_COOKIES") != "true"
                or not os.environ.get("DTS_ALLOWED_HOSTS")
                or not os.environ.get("DTS_ALLOWED_ORIGINS")
            ):
                raise ValueError(
                    "Production requires Secure cookies and explicit Host/Origin"
                )
            if any(
                not origin.startswith("https://")
                for origin in os.environ["DTS_ALLOWED_ORIGINS"].split(",")
            ):
                raise ValueError("Production origins must use HTTPS")
        app.state.store = Store(records, runtime / "tiles")
        app.state.queue = Queue(queue, app.state.store)
        app.state.journal = ReviewJournal(
            queue / "queue.json", runtime / "reviews" / f"{app.state.queue.sha}.sqlite3"
        )
        app.state.packages = {"general": app.state.queue}
        app.state.package_info = {
            "general": {
                "id": "general",
                "title": "Локальная очередь",
                "cards": len(app.state.queue.cards),
                "queue_sha256": app.state.queue.sha,
            }
        }
        real_pack = os.environ.get("DTS_RESEARCH_PACK")
        if real_pack:
            target = runtime / "packages" / "real-data-v2"
            info = adapt_real_pack(
                real_pack,
                app.state.store,
                target,
                app.state.queue.data["allowed_answers"],
            )
            research = Queue(target, app.state.store, max_cells=1_000_000)
            app.state.packages["real-data-v2"] = research
            app.state.package_info["real-data-v2"] = {
                "id": "real-data-v2",
                "title": "Реальные данные — вопросы Захару",
                "queue_sha256": research.sha,
                **info,
            }
        try:
            app.state.platform = configured(runtime)
        except ValueError:
            app.state.platform = None
        app.state.anonymous_test = isolated_anonymous_test()
        app.state.limiter = RateLimiter()
        app.state.public_package = load_package()
        app.state.workspace = app.state.platform or Workspace(
            runtime / "workspace.sqlite3"
        )
        app.state.token = secrets.token_urlsafe(32)
        app.state.replays = {}
        app.state.jobs = {}
        app.state.lock = threading.RLock()
        app.state.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="dts-analysis"
        )
        yield
        app.state.executor.shutdown(wait=False, cancel_futures=True)
        if hasattr(app.state, "telemetry"):
            app.state.telemetry.shutdown()

    app = FastAPI(title="DTS Workbench", version="0.1.0", lifespan=lifespan)
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    install_public(app)

    @app.middleware("http")
    async def local_only(request, call_next):
        rejected = reject_untrusted(request)
        if rejected is not None:
            return rejected
        platform = getattr(app.state, "platform", None)
        anonymous_test = getattr(app.state, "anonymous_test", False)
        public = (
            (anonymous_test or platform is None)
            and request.url.path in PUBLIC_PATHS
        ) or request.url.path in (
            "/api/v1/health",
            "/api/v2/health/live",
            "/api/v2/auth/login",
            "/api/v2/auth/activate",
            "/api/v2/auth/recover",
        )
        page_entry = request.url.path in ("/", "/login", "/activate", "/recover")
        static_asset = public_asset(request.url.path)
        try:
            user = (
                await asyncio.to_thread(authenticate, platform, request)
                if platform and not public and not page_entry and not static_asset
                else None
            )
        except OperationalError:
            return protect_response(JSONResponse(
                {"detail": "Metadata storage temporarily unavailable"}, status_code=503,
            ))
        scope = principal.set(user)

        async def guarded():
            if (
                (not anonymous_test or platform or getattr(app.state, "public_package", None))
                and request.url.path.startswith("/api/")
                and not public
                and not user
            ):
                headers = {}
                if request.url.path == "/api/v2/auth/guard":
                    original = request.headers.get("x-original-uri", "/works")
                    if len(original) > 4096 or not original.startswith("/") or original.startswith("//") or "\\" in original:
                        original = "/works"
                    headers["X-Login-Redirect"] = "/login?next=" + quote(original, safe="")
                return JSONResponse(
                    {
                        "detail": "Sign in first"
                        if platform
                        else "Account storage unavailable"
                    },
                    status_code=401 if platform else 503,
                    headers=headers,
                )
            if (
                not anonymous_test
                and not user
                and not request.url.path.startswith("/api/")
                and not page_entry
                and not static_asset
            ):
                target = request.url.path
                if request.url.query:
                    target += "?" + request.url.query
                return RedirectResponse("/login?next=" + quote(target, safe=""), status_code=303)
            if (
                user
                and request.url.path.startswith("/api/v1/")
                and user["project_id"] != "local"
            ):
                return JSONResponse(
                    {"detail": "Archive endpoints belong to the local project"},
                    status_code=403,
                )
            if user and request.method not in ("GET", "HEAD", "OPTIONS"):
                path = request.url.path
                roles = (
                    ("expert", "admin")
                    if path.endswith("/reviews")
                    else ("analyst", "admin")
                    if any(p in path for p in ("/denoise", "/jobs/", "/results", "/bookmarks"))
                    else None
                )
                if roles and user["role"] not in roles:
                    return JSONResponse(
                        {"detail": "Project role does not permit this action"},
                        status_code=403,
                    )
            if request.cookies.get("dts_rehearsal") and request.method not in (
                "GET",
                "HEAD",
                "OPTIONS",
            ):
                path = request.url.path
                if not (
                    public
                    or path.startswith("/api/v1/demo/")
                    or path
                    in (
                        "/api/v1/regions/evidence",
                        "/api/v2/auth/login",
                        "/api/v2/auth/logout",
                    )
                ):
                    return JSONResponse(
                        {
                            "detail": "Репетиция активна: рабочие записи защищены. Завершите репетицию перед рабочим действием."
                        },
                        status_code=403,
                    )
            if request.method not in ("GET", "HEAD", "OPTIONS"):
                if not public and not secrets.compare_digest(
                    request.headers.get("x-workbench-token", "").encode("utf-8"),
                    (user["csrf"] if user else getattr(app.state, "token", "!")).encode("utf-8"),
                ):
                    return JSONResponse(
                        {"detail": "Local mutation token required"}, status_code=403
                    )
                received = 0
                chunks = []
                body_limit = (
                    256 * 1024 if request.url.path == "/api/v1/results" else 65536
                )
                async for chunk in request.stream():
                    received += len(chunk)
                    if received > body_limit:
                        return JSONResponse(
                            {"detail": f"Body exceeds {body_limit // 1024} KiB"},
                            status_code=413,
                        )
                    chunks.append(chunk)
                request._body = b"".join(chunks)
                if user:
                    try:
                        await asyncio.to_thread(touch_session, platform, request)
                    except OperationalError:
                        return JSONResponse(
                            {"detail": "Metadata storage temporarily unavailable"},
                            status_code=503,
                        )
            return await call_next(request)

        try:
            response = await guarded()
            return protect_response(response, static_asset and request.url.path.startswith("/assets/"))

        finally:
            principal.reset(scope)

    @app.exception_handler(KeyError)
    async def missing(request, exc):
        return JSONResponse({"detail": "Resource not found"}, status_code=404)

    @app.exception_handler(OperationalError)
    async def storage_unavailable(request, exc):
        return JSONResponse(
            {"detail": "Metadata storage temporarily unavailable"}, status_code=503
        )

    @app.exception_handler(JobConflict)
    async def job_conflict(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(PermissionError)
    async def forbidden(request, exc):
        return JSONResponse({"detail": "Wrong project"}, status_code=403)

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.get("/api/v1/health")
    def health():
        return {
            "status": "ok",
            **({"records": len(app.state.store.records)} if app.state.anonymous_test else {}),
            "local_only": True,
            "qa_mode": os.environ.get("DTS_QA_MODE") == "isolated",
            "auth_enabled": not app.state.anonymous_test or app.state.platform is not None
            or app.state.public_package is not None,
            "access_mode": "isolated-test" if app.state.anonymous_test else "closed-team",
            "build_sha": os.environ.get("DTS_BUILD_SHA", "unknown"),
        }

    @app.get("/api/v2/auth/guard", include_in_schema=False)
    def auth_guard():
        if not principal.get():
            raise HTTPException(401, "Sign in first")
        return Response(status_code=204)

    @app.get("/api/v1/session")
    def session():
        user = principal.get()
        return {
            "mutation_token": user["csrf"] if user else app.state.token,
            "identity": user["display_name"]
            if user
            else "self-declared local reviewer; not authenticated",
        }

    @app.get("/api/v1/records", response_model=list[RecordMeta])
    def catalog():
        app.state.store.refresh_imports()
        return list(app.state.store.meta.values())

    @app.get("/api/v1/records/{key}", response_model=RecordDetail)
    def metadata(key: str):
        r = app.state.store.get(key)
        return {
            **app.state.store.meta[key],
            "timestamps": [p["timestamp"] for p in r.profiles],
            "time_s": r.elapsed_s.tolist(),
            "x_m": r.x_m.tolist(),
            "segment_id": r.segment_id.tolist(),
        }

    @app.get("/api/v1/records/{key}/tiles")
    def tiles(
        key: str,
        request: Request,
        x0: float,
        x1: float,
        t0: float,
        t1: float,
        format_version: int = Query(1, ge=1, le=2),
        width: int = Query(128, ge=1, le=256),
        height: int = Query(128, ge=1, le=256),
        view_version: str | None = None,
    ):
        if view_version and view_version != app.state.store.meta[key]["source_sha256"]:
            raise HTTPException(409, "Версия источника изменилась. Обновите запись.")
        data = app.state.store.tile(key, x0, x1, t0, t1, width, height, format_version)
        etag = '"' + hashlib.sha256(data).hexdigest() + '"'
        if request.headers.get("if-none-match") == etag:
            return Response(status_code=304, headers={"ETag": etag})
        return Response(
            data, media_type="application/octet-stream", headers={"ETag": etag}
        )

    @app.get("/api/v1/records/{key}/slices", response_model=Slices)
    def slices(
        key: str,
        row: int = Query(0, ge=0),
        column: int = Query(0, ge=0),
        row_start: int = Query(0, ge=0),
        row_end: int | None = Query(None, ge=1),
        source_sha256: str | None = None,
    ):
        if (
            source_sha256
            and source_sha256 != app.state.store.meta[key]["source_sha256"]
        ):
            raise HTTPException(409, "Версия источника изменилась. Обновите запись.")
        return app.state.store.slices(key, row, column, row_start, row_end)

    @app.get("/api/v1/queue", response_model=ReviewQueue)
    def review_queue(
        package: str = "general", reviewer: str = "", independent: bool = False
    ):
        q, journal = review_context(package, reviewer, independent)
        # Do not send selection rationale/method identities before a blind response.
        return {
            "queue_sha256": q.sha,
            "cards": [
                {
                    "review_id": c["review_id"],
                    "source": c["source"],
                    "source_sha256": c["evidence"]["source_sha256"],
                    "native_shape": c["native_shape"],
                    "x_bounds_m": c["x_bounds_m"],
                    "time_bounds_s": c["time_bounds_s"],
                    "timestamps": c["timestamps"],
                    "row_bounds": c["row_bounds"],
                    "column_bounds": c["column_bounds"],
                    "shortlist": c.get("shortlist", False),
                }
                for c in q.cards.values()
            ],
            "answers": q.data["allowed_answers"],
            "annotations": journal.annotations() if journal else {},
        }

    @app.get("/api/v1/queue/{key}/window")
    def window(key: str, package: str = "general", view_version: str | None = None):
        q = resolve_queue(app,package)
        if view_version and view_version != q.sha:
            raise HTTPException(409, "Версия очереди изменилась. Обновите очередь.")
        c, d = q.load(key)
        if max(d["raw_c"].shape) > 256:
            x0, x1 = c["x_bounds_m"]
            t0, t1 = c["time_bounds_s"]
            return Response(
                app.state.store.tile(
                    c["source"].removesuffix(".csv"),
                    x0,
                    x1,
                    t0,
                    t1 if t1 > t0 else t0 + 1,
                    128,
                    128,
                ),
                media_type="application/octet-stream",
            )
        return Response(
            packet(
                d["raw_c"],
                d["invalid_reason"],
                (d["invalid_reason"] == 0).astype("u4"),
                x_cells=cell_edges(d["x_m"]),
                time_cells=cell_edges(d["time_s"], 300),
                aggregation="native_observations",
                source_sha256=c["evidence"]["source_sha256"],
                scale=app.state.store.meta[c["source"].removesuffix(".csv")]["scale"],
            ),
            media_type="application/octet-stream",
        )

    @app.get("/api/v1/queue/{key}/methods")
    def methods(
        key: str,
        package: str = "general",
        reviewer: str = "",
        independent: bool = False,
    ):
        q, journal = review_context(package, reviewer, independent)
        if not journal or key not in journal.annotations():
            raise HTTPException(
                403, "Submit an independent response before revealing methods"
            )
        c, d = q.load(key)
        return {
            "methods": c["methods"],
            "counts": [int(np.count_nonzero(v)) for v in d["binary"]],
            "result_run_ids": c.get('result_run_ids',[]),
            "policy": "Research hypotheses; origin belongs to each run. Accident accuracy remains unverified.",
        }

    @app.get("/api/v1/queue/{key}/methods/{method}/window")
    def method_window(
        key: str,
        method: str,
        package: str = "general",
        reviewer: str = "",
        independent: bool = False,
        view_version: str | None = None,
    ):
        q, journal = review_context(package, reviewer, independent)
        if view_version and view_version != q.sha:
            raise HTTPException(409, "Версия очереди изменилась. Обновите очередь.")
        if not journal or key not in journal.annotations():
            raise HTTPException(403, "Submit an independent response first")
        c, d = q.load(key)
        if method not in c["methods"]:
            raise HTTPException(404, "Unknown method")
        i = c["methods"].index(method)
        return Response(
            packet(
                d["signed"][i],
                d["invalid_reason"],
                (d["invalid_reason"] == 0).astype("u4"),
                x_cells=cell_edges(d["x_m"]),
                time_cells=cell_edges(d["time_s"], 300),
                aggregation="native_method_hypothesis",
                source_sha256=c["evidence"]["source_sha256"],
                scale=[-1, 1],
                palette="diverging",
                title="Направление гипотезы: −1 / 0 / +1",
                unit="знак",
            ),
            media_type="application/octet-stream",
        )

    @app.post("/api/v1/reviews", response_model=Annotation)
    def save_review(
        body: Review,
        idempotency_key: str = Header(min_length=8, max_length=128),
        package: str = "general",
        reviewer: str = "",
        independent: bool = False,
    ):
        if app.state.platform and not independent:
            raise HTTPException(
                403, "Legacy journal is read-only; use authenticated independent review"
            )
        if independent and (
            reviewer.strip() != body.reviewer.strip() or body.reviewer_type != "expert"
        ):
            raise HTTPException(422, "Selected reviewer must match response author")
        _, journal = review_context(package, reviewer, independent)
        if not journal:
            raise HTTPException(422, "Select a local reviewer first")
        try:
            return journal.save(body.model_dump(), idempotency_key=idempotency_key)
        except ReviewConflict as exc:
            return JSONResponse(
                {
                    "detail": str(exc),
                    "current": journal.annotations().get(body.review_id),
                },
                status_code=409,
            )

    @app.get("/api/v1/export")
    def export(package: str = "general", reviewer: str = "", independent: bool = False):
        _, journal = review_context(package, reviewer, independent)
        if not journal:
            raise HTTPException(422, "Select a local reviewer first")
        return JSONResponse(
            journal.export(),
            headers={
                "Content-Disposition": 'attachment; filename="dts-review-packet.json"'
            },
        )

    def review_context(package, reviewer, independent):
        q = resolve_queue(app,package)
        user = principal.get()
        if user:
            if not independent:
                if user["role"] != "admin":
                    raise HTTPException(
                        403, "Legacy journal is an administrator archive"
                    )
            else:
                if reviewer.strip() != user["display_name"]:
                    raise HTTPException(
                        403, "Reviewer must match authenticated account"
                    )
                from .pg_reviews import PGReviewJournal

                return q, PGReviewJournal(app.state.platform, q, user)
        if not independent and package == "general":
            return q, app.state.journal
        if not reviewer.strip() or len(reviewer) > 100:
            return q, None
        digest = hashlib.sha256(reviewer.strip().encode()).hexdigest()
        journal = ReviewJournal(
            q.root / "queue.json", runtime / "reviews" / f"{q.sha}-{digest}.sqlite3"
        )
        return q, journal

    def replay_state(key):
        with app.state.lock:
            s = app.state.replays[key]
            user = principal.get()
            if app.state.platform and (not user or s.get("account_id") != user["id"] or s.get("project_id") != user["project_id"]):
                raise HTTPException(403, "Replay belongs to another account")
            r = app.state.store.get(s["record_id"])
            now = time.monotonic()
            if s["playing"]:
                position = min(
                    float(r.elapsed_s[-1]),
                    s["anchor_position"] + (now - s["anchor_clock"]) * s["speed"],
                )
                s["row"] = max(
                    0, int(np.searchsorted(r.elapsed_s, position, side="right") - 1)
                )
                if position >= r.elapsed_s[-1]:
                    s["playing"] = False
                s["position_s"] = position
            s["touched"] = now
            return {
                k: s[k] for k in ("record_id", "row", "playing", "speed", "position_s")
            }

    @app.post("/api/v1/replays", response_model=ReplayCreated)
    def replay_create(body: ReplayCreate):
        app.state.store.get(body.record_id)
        with app.state.lock:
            now = time.monotonic()
            for k in list(app.state.replays):
                if now - app.state.replays[k]["touched"] > 1800:
                    del app.state.replays[k]
            if len(app.state.replays) >= 8:
                raise HTTPException(
                    429, "Eight replay sessions maximum; close an unused session"
                )
            key = uuid.uuid4().hex
            app.state.replays[key] = {
                "account_id": principal.get()["id"] if principal.get() else None,
                "project_id": principal.get()["project_id"] if principal.get() else None,
                "record_id": body.record_id,
                "row": 0,
                "playing": False,
                "speed": 1,
                "position_s": 0.0,
                "anchor_position": 0.0,
                "anchor_clock": now,
                "touched": now,
            }
            return {"id": key, **replay_state(key)}

    @app.get("/api/v1/replays/{key}", response_model=ReplayState)
    def replay_get(key: str):
        return replay_state(key)

    @app.delete("/api/v1/replays/{key}")
    def replay_delete(key: str):
        with app.state.lock:
            if key in app.state.replays:
                replay_state(key)
            app.state.replays.pop(key, None)
        return {"closed": True}

    @app.post("/api/v1/replays/{key}/commands", response_model=ReplayState)
    def replay_command(key: str, body: ReplayCommand):
        with app.state.lock:
            replay_state(key)
            s = app.state.replays[key]
            r = app.state.store.get(s["record_id"])
            if body.action == "seek":
                if body.row is None or body.row >= len(r.elapsed_s):
                    raise HTTPException(422, "Valid native row required")
                s["row"] = body.row
                s["position_s"] = float(r.elapsed_s[body.row])
            elif body.action == "speed":
                s["speed"] = body.speed
            else:
                s["playing"] = body.action == "play"
            s["anchor_position"] = s["position_s"]
            s["anchor_clock"] = time.monotonic()
            return replay_state(key)

    @app.get("/api/v1/replays/{key}/events")
    async def replay_events(key: str, request: Request):
        replay_state(key)
        user = principal.get()

        async def stream():
            previous = None
            seq = 0
            while not await request.is_disconnected():
                if app.state.platform and not await session_is_current(app.state.platform, request, user):
                    return
                try:
                    state = replay_state(key)
                except KeyError:
                    break
                signature = (state["row"], state["playing"], state["speed"])
                if signature != previous:
                    seq += 1
                    previous = signature
                    yield f"id: {seq}\nevent: cursor\ndata: {json.dumps(state)}\n\n"
                else:
                    yield ": heartbeat\n\n"
                await asyncio.sleep(0.5)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no"},
        )

    ml_url = os.environ.get("DTS_ML_URL", "http://127.0.0.1:8082").rstrip("/")

    @app.post("/api/v1/denoise", response_model=DenoiseResult)
    async def denoise(body: Denoise):
        result = await clean_profile(body)
        key = uuid.uuid4().hex
        app.state.workspace.put(
            "run",
            key,
            {
                "id": key,
                "status": "complete",
                "method": body.method,
                "method_version": result.get('implementation_sha') if isinstance(result.get('implementation_sha'),str) and len(result['implementation_sha'])==40 and all(c in '0123456789abcdef' for c in result['implementation_sha']) else "ML API contract 0.1; implementation SHA not supplied",
                "parameters": {"method": body.method},
                "training_origin": "untrained",
                "source_sha256": result["source_sha256"],
                "region": {
                    "record_id": body.record_id,
                    "row_start": body.row,
                    "row_end": body.row + 1,
                    "column_start": 0,
                    "column_end": len(result["denoised"]),
                },
                "limitations": "Profile denoising, not anomaly classification",
                "result": result,
            },
        )
        return {**result, "run_id": key}

    async def clean_profile(body):
        s = app.state.store.slices(body.record_id, body.row, 0)
        if any(v is not None and v <= -200 for v in s["profile"]):
            raise HTTPException(
                422,
                "This ML contract masks temperatures <= -200 C; use native data or a compatible method",
            )
        # ML profile expects sentinel floats, not null; restore mask as sentinel only in request.
        payload = {
            "method": body.method,
            "raw": [v if v is not None else -300.0 for v in s["profile"]],
            "distances": s["x_m"],
        }
        try:
            async with httpx.AsyncClient(timeout=30,trust_env=False) as client:
                response = await client.post(ml_url + "/denoise/profile", json=payload)
            if response.is_error:
                raise HTTPException(
                    response.status_code, "ML rejected profile: " + response.text[:500]
                )
            result = {
                **response.json(),
                "source_sha256": s["source_sha256"],
                "row": body.row,
                "quality_contract": "ML <= -200 masked; original mask retained separately",
            }
            values = result.get("denoised")
            if (
                not isinstance(values, list)
                or len(values) != len(s["profile"])
                or any(
                    v is not None
                    and (
                        isinstance(v, bool)
                        or not isinstance(v, (int, float))
                        or not np.isfinite(v)
                    )
                    for v in values
                )
            ):
                raise HTTPException(
                    502, "ML profile violates the finite native-shape contract"
                )
            result["denoised"] = [
                v if native is not None else None
                for v, native in zip(values, s["profile"])
            ]
            return result
        except httpx.RequestError:
            raise HTTPException(
                503, "ML service is unavailable. Raw data remains available."
            )

    app.state.clean_profile = clean_profile

    @app.post("/api/v1/jobs/detect", status_code=202)
    def detect(
        body: Detect,
        idempotency_key: str | None = Header(None, min_length=8, max_length=128),
    ):
        r = app.state.store.get(body.record_id)
        rows, cols = body.row_end - body.row_start, body.column_end - body.column_start
        if (
            rows < 1
            or cols < 2
            or body.row_end > r.raw_c.shape[0]
            or body.column_end > r.raw_c.shape[1]
            or rows * cols > 1_000_000
        ):
            raise HTTPException(
                413, "Select a bounded native window, at most 1,000,000 cells"
            )
        if app.state.platform:
            user = principal.get()
            queued = app.state.platform.enqueue(
                "detect",
                {**body.model_dump(), "source_sha256": r.manifest["source_sha256"]},
                key=f"{user['id']}:{idempotency_key}" if idempotency_key else None,
                project=user["project_id"],
            )
            key = queued["id"]
            try:
                return app.state.workspace.get("run", key)
            except KeyError:
                return app.state.workspace.put(
                    "run",
                    key,
                    {
                        "id": key,
                        "status": queued["status"],
                        "progress": 0,
                        "region": body.model_dump(),
                        "source_sha256": r.manifest["source_sha256"],
                        "method": "reference_detector",
                        "method_version": "ML API contract 0.1; implementation SHA not supplied",
                        "parameters": {},
                        "training_origin": "untrained",
                        "limitations": "Reference hypotheses; real accuracy unknown",
                    },
                )
        with app.state.lock:
            if any(
                j["status"] in ("queued", "running", "cancel_requested")
                for j in app.state.jobs.values()
            ):
                raise HTTPException(429, "One analysis at a time")
            if len(app.state.jobs) >= 16:
                del app.state.jobs[next(iter(app.state.jobs))]
            key = uuid.uuid4().hex
            app.state.jobs[key] = {
                "id": key,
                "status": "queued",
                "progress": 0,
                "source_sha256": r.manifest["source_sha256"],
                "result_bytes": 0,
                "region": body.model_dump(),
                "method": "reference_detector",
                "method_version": "ML API contract 0.1; implementation SHA not supplied",
                "parameters": {},
                "training_origin": "untrained",
                "limitations": "Offline reference hypotheses; real accuracy unknown",
            }
            app.state.workspace.put("run", key, app.state.jobs[key])

        def work():
            job = app.state.jobs[key]
            with app.state.lock:
                if job["status"] == "cancel_requested":
                    job.update(status="cancelled")
                    app.state.workspace.put("run", key, job)
                    return
                job.update(status="running", progress=None)
                app.state.workspace.put("run", key, job)
            try:
                a = r.raw_c[
                    body.row_start : body.row_end, body.column_start : body.column_end
                ]
                q = r.invalid_reason[
                    body.row_start : body.row_end, body.column_start : body.column_end
                ]
                if np.any((q == 0) & (a <= -200)):
                    raise ValueError(
                        "ML quality contract would discard a valid native value <= -200 C"
                    )
                payload = {
                    "raw": [
                        [float(v) if flag == 0 else None for v, flag in zip(row, mask)]
                        for row, mask in zip(a, q)
                    ],
                    "distances": r.x_m[body.column_start : body.column_end].tolist(),
                    "time_s": r.elapsed_s[body.row_start : body.row_end].tolist(),
                }
                with (
                    httpx.Client(timeout=180) as client,
                    client.stream(
                        "POST", ml_url + "/detect/field", json=payload
                    ) as response,
                ):
                    response.raise_for_status()
                    chunks, size = [], 0
                    for chunk in response.iter_bytes():
                        size += len(chunk)
                        if size > 8 * 1024 * 1024:
                            raise ValueError(
                                "ML result exceeds the 8 MiB per-job budget; select a smaller window"
                            )
                        chunks.append(chunk)
                result = json.loads(b"".join(chunks))
                if not isinstance(result, dict):
                    # Protocol failure, handled with other invalid ML payloads below.
                    raise ValueError("ML field response must be an object")  # noqa: TRY004
                with app.state.lock:
                    if job["status"] == "cancel_requested":
                        job.update(status="cancelled", progress=None)
                        app.state.workspace.put("run", key, job)
                        return
                    job.update(
                        status="complete",
                        progress=100,
                        result=result,
                        result_bytes=size,
                    )
                    app.state.workspace.put("run", key, job)
                    for old in list(app.state.jobs):
                        if (
                            sum(
                                j.get("result_bytes", 0)
                                for j in app.state.jobs.values()
                            )
                            <= 16 * 1024 * 1024
                        ):
                            break
                        if old != key and app.state.jobs[old]["status"] not in (
                            "queued",
                            "running",
                        ):
                            del app.state.jobs[old]
            except (httpx.HTTPError, ValueError) as exc:
                with app.state.lock:
                    job.update(
                        status="cancelled"
                        if job["status"] == "cancel_requested"
                        else "failed",
                        error=str(exc)[:500],
                    )
                    app.state.workspace.put("run", key, job)

        app.state.executor.submit(work)
        return app.state.jobs[key]

    @app.get("/api/v1/jobs/{key}")
    def job(key: str):
        if app.state.platform:
            state = app.state.platform.job(key)
            user = principal.get()
            if state["project_id"] != user["project_id"]:
                raise HTTPException(403, "Wrong project")
            metadata = app.state.workspace.get("run", key)
            return {
                **metadata,
                "status": state["status"],
                "error": state["error"],
                "progress": 100 if state["status"] == "complete" else None,
            }
        return app.state.jobs.get(key) or app.state.workspace.get("run", key)

    @app.post("/api/v1/jobs/{key}/cancel")
    def cancel_job(key: str):
        if app.state.platform:
            state = app.state.platform.job(key)
            if state["project_id"] != principal.get()["project_id"]:
                raise HTTPException(403, "Wrong project")
            return app.state.platform.cancel_job(key)
        with app.state.lock:
            job = app.state.jobs[key]
            if job["status"] in ("queued", "running"):
                job.update(status="cancel_requested")
                app.state.workspace.put("run", key, job)
            return job

    from .telemetry import install_telemetry

    app.state.telemetry = install_telemetry(app)
    install_accounts(app)
    install_control(app)
    install_workspace_routes(app, review_context)
    from .analysis_results import install_analysis

    install_analysis(app)
    from .analysis_jobs import install_analysis_jobs

    install_analysis_jobs(app)
    from .experiments import install_experiments

    install_experiments(app)
    from .investigations import install_investigations

    install_investigations(app)
    from .pipeline_settings import install_pipeline_settings

    install_pipeline_settings(app)
    from .pipeline_check import install_pipeline_check

    install_pipeline_check(app)
    from .laboratory import install_laboratory
    install_laboratory(app)
    if os.environ.get("DTS_WEB_DIST"):
        dist = Path(os.environ["DTS_WEB_DIST"]).resolve()
        if not (dist / "index.html").is_file():
            raise ValueError("Build frontend before configuring DTS_WEB_DIST")

        @app.get("/{path:path}", include_in_schema=False)
        def frontend(path: str):
            candidate = (dist / path).resolve()
            if not candidate.is_relative_to(dist):
                raise HTTPException(404)
            if candidate.is_file():
                return FileResponse(candidate)
            if path in (
                "",
                "research",
                "works",
                "laboratory",
                "monitoring",
                "review",
                "candidates",
                "runs",
                "analysis",
                "pipeline",
                "experiments",
                "cases",
                "demo",
                "system",
                "try",
                "request-access",
                "activate",
                "recover",
                "login",
                "account",
            ) or (
                (path.startswith("cases/") or path.startswith('works/'))
                and len(path.split('/',1)[1]) == 32
                and all(c in "0123456789abcdef" for c in path.split('/',1)[1])
            ):
                return FileResponse(dist / "index.html")
            raise HTTPException(404)

    return app


app = create_app()
