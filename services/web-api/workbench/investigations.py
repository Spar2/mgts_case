"""Mutable research context; source, independent reviews and evidence remain distinct."""

import hashlib
import json
import time
import uuid
import tempfile
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Header, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
from psycopg.types.json import Jsonb

from .platform import principal
from .workspace_routes import Region
import numpy as np


class InvestigationContext(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    region: Region
    source_sha256: str = Field(pattern="^[0-9a-f]{64}$")
    profile: int = Field(ge=0)
    column: int = Field(ge=0)
    run_ids: list[str] = Field(default_factory=list, max_length=2)
    layer: Literal[
        "raw",
        "processed_temperature_c",
        "background_temperature_c",
        "score",
        "difference",
    ] = "raw"


class InvestigationCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    title: str = Field(min_length=1, max_length=150)
    notes: str = Field(default="", max_length=8000)
    context: InvestigationContext
    origin: Literal["archive", "analysis", "review", "monitoring"] = "archive"
    stream_id: str | None = Field(default=None, max_length=128)
    case_ids: list[str] = Field(default_factory=list, max_length=50)


class InvestigationPatch(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    expected_revision: int = Field(ge=1)
    title: str | None = Field(default=None, min_length=1, max_length=150)
    notes: str | None = Field(default=None, max_length=8000)
    context: InvestigationContext | None = None
    case_ids: list[str] | None = Field(default=None, max_length=50)


class InvestigationView(InvestigationCreate):
    id: str
    project_id: str
    account_id: str
    author_name: str
    revision: int
    created_at: float
    updated_at: float
    source_available: bool = True
    dropped_run_ids: list[str] = Field(default_factory=list)


class InvestigationReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    expected_revision: int = Field(ge=1)
    assigned_to: str = Field(min_length=1, max_length=128)


class InvestigationCaseLink(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    case_id: str = Field(pattern="^[0-9a-f]{32}$")


class InvestigationReviewView(BaseModel):
    id: str
    investigation_id: str
    project_id: str
    package: str
    review_id: str
    queue_sha256: str
    source_sha256: str
    context: InvestigationContext
    assigned_to: str
    created_by: str
    created_at: float
    status: str
    answer_revision: int | None = None
    answer_author: str | None = None


class ReviewProgressRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    package: str = Field(min_length=1, max_length=128)
    review_id: str = Field(min_length=1, max_length=128)
    state: Literal["skipped"] = "skipped"


def install_investigations(app):
    def access(edit=False):
        user = principal.get()
        if not user or not app.state.platform:
            raise HTTPException(503, "Authenticated PostgreSQL workspace required")
        if user["project_id"] != "local":
            raise HTTPException(403, "Native archive belongs to local project")
        if edit and user["role"] not in ("analyst", "admin"):
            raise HTTPException(403, "Analyst or administrator required")
        return user, app.state.platform

    def document(db, key):
        row = db.execute(
            "SELECT payload FROM documents WHERE kind='investigation' AND id=%s", (key,)
        ).fetchone()
        if not row:
            raise HTTPException(404, "Work not found")
        if row[0]["project_id"] != principal.get()["project_id"]:
            raise HTTPException(403, "Wrong project")
        return row[0]

    def available(value):
        context = value["context"]
        try:
            return (
                app.state.store.get(context["region"]["record_id"]).manifest[
                    "source_sha256"
                ]
                == context["source_sha256"]
            )
        except KeyError:
            return False

    def view(value):
        return {**value, "source_available": available(value)}

    def checked_context(body, old=None, stream_id=None):
        try:
            native = app.state.store.get(body.region.record_id)
        except KeyError:
            raise HTTPException(404, "Source not found") from None
        if native.manifest["source_sha256"] != body.source_sha256:
            raise HTTPException(409, "Source SHA changed; no replacement selected")
        r = body.region
        if not (
            0 <= r.row_start <= body.profile < r.row_end <= native.raw_c.shape[0]
            and 0
            <= r.column_start
            <= body.column
            < r.column_end
            <= native.raw_c.shape[1]
            and r.column_end - r.column_start >= 2
        ):
            raise HTTPException(
                422, "Select native bounds, an included profile and point"
            )
        if len(set(body.run_ids)) != len(body.run_ids):
            raise HTTPException(422, "Distinct selected results required")
        if old and (
            r.record_id != old["region"]["record_id"]
            or body.source_sha256 != old["source_sha256"]
        ):
            raise HTTPException(409, "Work source is fixed; create another work")
        if stream_id:
            try:
                stream = app.state.platform.get("stream", stream_id)
            except KeyError:
                raise HTTPException(404, "Stream not found") from None
            if (
                stream["record_id"] != r.record_id
                or stream["source_sha256"] != body.source_sha256
                or r.row_end > stream.get("verified_persisted", -1) + 1
            ):
                raise HTTPException(409, "Window is not confirmed by this replay")
        retained = []
        dropped = []
        from .analysis_results import load

        old_runs = set(old["run_ids"]) if old else set()
        for key in body.run_ids:
            try:
                run, _ = load(app, key)
                bounds = Region.model_validate(run["region"])
                if (
                    run["source_sha256"] != body.source_sha256
                    or bounds.record_id != r.record_id
                    or not (
                        bounds.row_start <= r.row_start < r.row_end <= bounds.row_end
                        and bounds.column_start
                        <= r.column_start
                        < r.column_end
                        <= bounds.column_end
                    )
                ):
                    raise HTTPException(
                        409, "Selected result does not cover this source/window"
                    )
            except (KeyError, ValueError, HTTPException) as exc:
                if isinstance(exc, HTTPException) and exc.status_code in (401, 403):
                    raise
                if key in old_runs:
                    dropped.append(key)
                    continue
                raise HTTPException(
                    409, "Selected result is unavailable or incompatible"
                ) from None
            retained.append(key)
        if body.run_ids and body.run_ids[0] in dropped:
            dropped = list(body.run_ids)
            retained = []
        layer = body.layer
        if layer != "raw":
            if not retained:
                if dropped:
                    layer = "raw"
                else:
                    raise HTTPException(
                        409, "Select a result before selecting a processed layer"
                    )
            else:
                primary, _ = load(app, retained[0])
                layers = set(primary["layers"])
                if "processed_temperature_c" in layers:
                    layers.add("difference")
                if layer not in layers:
                    raise HTTPException(409, "Selected layer is unavailable")
        return {**body.model_dump(), "run_ids": retained, "layer": layer}, dropped

    def checked_cases(db, keys, context):
        if len(set(keys)) != len(keys):
            raise HTTPException(422, "Distinct cases required")
        for key in keys:
            row = db.execute(
                "SELECT payload FROM documents WHERE kind='evidence_case' AND id=%s",
                (key,),
            ).fetchone()
            if not row:
                raise HTTPException(404, "Case not found")
            case = row[0]
            if case["project_id"] != principal.get()["project_id"]:
                raise HTTPException(403, "Wrong case project")
            if (
                case["source_sha256"] != context["source_sha256"]
                or case["region"]["record_id"] != context["region"]["record_id"]
            ):
                raise HTTPException(409, "Case belongs to another source")

    def store(db, value, action, changes):
        db.execute(
            "INSERT INTO documents VALUES('investigation',%s,%s,%s) ON CONFLICT(kind,id) DO UPDATE SET payload=excluded.payload,updated=excluded.updated",
            (value["id"], Jsonb(value), value["updated_at"]),
        )
        user = principal.get()
        event = {
            "id": value["id"] + ":" + str(value["revision"]).zfill(8),
            "investigation_id": value["id"],
            "project_id": user["project_id"],
            "revision": value["revision"],
            "at": value["updated_at"],
            "actor": user["id"],
            "author_name": user["display_name"],
            "action": action,
            "changes": changes,
        }
        db.execute(
            "INSERT INTO documents VALUES('investigation_event',%s,%s,%s)",
            (event["id"], Jsonb(event), event["at"]),
        )
        db.execute(
            "INSERT INTO audit(project_id,actor,action,resource,at) VALUES(%s,%s,%s,%s,%s)",
            (
                user["project_id"],
                user["id"],
                "investigation." + action,
                value["id"],
                event["at"],
            ),
        )

    @app.get("/api/v2/investigations", response_model=list[InvestigationView])
    def catalog(limit: int = Query(50, ge=1, le=100), offset: int = Query(0, ge=0)):
        user, platform = access()
        return [view(v) for v in platform.list("investigation", limit, offset)]

    @app.post(
        "/api/v2/investigations", response_model=InvestigationView, status_code=201
    )
    def create(
        body: InvestigationCreate,
        idempotency_key: Annotated[str, Header(min_length=8, max_length=128)],
    ):
        user, platform = access(True)
        namespace = (
            user["project_id"] + ":" + user["id"] + ":investigation:" + idempotency_key
        )
        request_hash = hashlib.sha256(
            json.dumps(body.model_dump(), sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        with platform.connect() as db:
            db.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s,0))", (namespace,)
            )
            previous = db.execute(
                "SELECT payload FROM documents WHERE kind='investigation_request' AND id=%s",
                (namespace,),
            ).fetchone()
            if previous:
                if previous[0]["request_sha256"] != request_hash:
                    raise HTTPException(
                        409, "Idempotency key reused with different content"
                    )
                return previous[0]["response"]
            if not body.title.strip():
                raise HTTPException(422, "Work title required")
            if body.origin == "monitoring" and not body.stream_id:
                raise HTTPException(422, "Confirmed replay required")
            context, dropped = checked_context(body.context, stream_id=body.stream_id)
            checked_cases(db, body.case_ids, context)
            at = time.time()
            value = {
                **body.model_dump(),
                "title": body.title.strip(),
                "context": context,
                "id": uuid.uuid4().hex,
                "project_id": user["project_id"],
                "account_id": user["id"],
                "author_name": user["display_name"],
                "revision": 1,
                "created_at": at,
                "updated_at": at,
            }
            store(db, value, "created", {"context": context, "origin": body.origin})
            response = view(value)
            db.execute(
                "INSERT INTO documents VALUES('investigation_request',%s,%s,%s)",
                (
                    namespace,
                    Jsonb(
                        {
                            "project_id": user["project_id"],
                            "request_sha256": request_hash,
                            "response": response,
                        }
                    ),
                    at,
                ),
            )
            return response

    @app.get("/api/v2/investigations/{key}", response_model=InvestigationView)
    def detail(key: str):
        user, platform = access()
        with platform.connect() as db:
            return view(document(db, key))

    @app.patch("/api/v2/investigations/{key}", response_model=InvestigationView)
    def update(key: str, body: InvestigationPatch):
        user, platform = access(True)
        with platform.connect() as db:
            db.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s,0))",
                ("investigation:" + key,),
            )
            old = document(db, key)
            if body.expected_revision != old["revision"]:
                raise HTTPException(
                    409,
                    {
                        "message": "Work changed; reload before saving",
                        "current": view(old),
                    },
                )
            changes = body.model_dump(exclude_unset=True)
            changes.pop("expected_revision")
            dropped = []
            if any(v is None for v in changes.values()):
                raise HTTPException(422, "Omit unchanged fields instead of null")
            if "title" in changes:
                changes["title"] = changes["title"].strip()
                if not changes["title"]:
                    raise HTTPException(422, "Work title required")
            if body.context is not None:
                changes["context"], dropped = checked_context(
                    body.context, old["context"], old.get("stream_id")
                )
            context = changes.get("context", old["context"])
            if body.case_ids is not None:
                checked_cases(db, body.case_ids, context)
            value = {
                **old,
                **changes,
                "revision": old["revision"] + 1,
                "updated_at": time.time(),
            }
            store(db, value, "updated", changes)
            return {**view(value), "dropped_run_ids": dropped}

    @app.get("/api/v2/investigations/{key}/history")
    def history(
        key: str, limit: int = Query(50, ge=1, le=100), offset: int = Query(0, ge=0)
    ):
        user, platform = access()
        with platform.connect() as db:
            document(db, key)
            rows = db.execute(
                "SELECT payload FROM documents WHERE kind='investigation_event' AND payload->>'investigation_id'=%s AND payload->>'project_id'=%s ORDER BY (payload->>'revision')::integer DESC LIMIT %s OFFSET %s",
                (key, user["project_id"], limit, offset),
            ).fetchall()
        return {
            "items": [r[0] for r in rows],
            "next_offset": offset + limit if len(rows) == limit else None,
        }

    @app.post(
        "/api/v2/investigations/{key}/review-requests",
        response_model=InvestigationReviewView,
        status_code=201,
    )
    def request_review(
        key: str,
        body: InvestigationReviewRequest,
        idempotency_key: Annotated[str, Header(min_length=8, max_length=128)],
    ):
        user, platform = access(True)
        namespace = (
            user["project_id"] + ":" + user["id"] + ":work-review:" + idempotency_key
        )
        request_hash = hashlib.sha256(
            json.dumps({"work": key, **body.model_dump()}, sort_keys=True).encode()
        ).hexdigest()
        with platform.connect() as db:
            db.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s,0))", (namespace,)
            )
            old = db.execute(
                "SELECT payload FROM documents WHERE kind='investigation_request' AND id=%s",
                (namespace,),
            ).fetchone()
            if old:
                if old[0]["request_sha256"] != request_hash:
                    raise HTTPException(
                        409, "Idempotency key reused with different assignment"
                    )
                return old[0]["response"]
            db.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s,0))",
                ("investigation:" + key,),
            )
            work = document(db, key)
            if work["revision"] != body.expected_revision:
                raise HTTPException(409, "Work changed; review the current window")
            if not db.execute(
                "SELECT 1 FROM memberships m JOIN users u ON u.id=m.user_id WHERE m.project_id=%s AND m.user_id=%s AND m.role IN ('expert','admin') AND u.enabled",
                (user["project_id"], body.assigned_to),
            ).fetchone():
                raise HTTPException(422, "Choose an enabled expert in this project")
            context, _ = checked_context(
                InvestigationContext.model_validate(work["context"]),
                stream_id=work.get("stream_id"),
            )
            r = Region.model_validate(context["region"])
            native = app.state.store.get(r.record_id)
            rs, re, cs, ce = r.row_start, r.row_end, r.column_start, r.column_end
            if (re - rs) * (ce - cs) > 1_000_000:
                raise HTTPException(
                    413, "Reduce the review window; no hidden truncation"
                )
            from .review_packages import publish_queue
            from .vendor.data import sha256

            ident = uuid.uuid4().hex
            staging = platform.blobs.root / "review-staging"
            staging.mkdir(exist_ok=True)
            with tempfile.TemporaryDirectory(dir=staging) as temporary:
                folder = Path(temporary)
                artifact = folder / "card.npz"
                shape = (re - rs, ce - cs)
                np.savez_compressed(
                    artifact,
                    raw_c=native.raw_c[rs:re, cs:ce],
                    invalid_reason=native.invalid_reason[rs:re, cs:ce],
                    x_m=native.x_m[cs:ce],
                    time_s=native.elapsed_s[rs:re],
                    signed=np.zeros((0, *shape), np.int8),
                    binary=np.zeros((0, *shape), bool),
                )
                card = {
                    "review_id": ident,
                    "source": r.record_id + ".csv",
                    "artifact": artifact.name,
                    "artifact_sha256": sha256(artifact),
                    "evidence": {
                        "source_sha256": context["source_sha256"],
                        "origin": "immutable_work_window_no_labels",
                    },
                    "native_shape": list(shape),
                    "row_bounds": [rs, re],
                    "column_bounds": [cs, ce],
                    "x_bounds_m": [float(native.x_m[cs]), float(native.x_m[ce - 1])],
                    "time_bounds_s": [
                        float(native.elapsed_s[rs]),
                        float(native.elapsed_s[re - 1]),
                    ],
                    "timestamps": [p["timestamp"] for p in native.profiles[rs:re]],
                    "methods": [],
                    "result_run_ids": context["run_ids"],
                }
                queue = {
                    "cards": [card],
                    "allowed_answers": {
                        "temperature_episode": [
                            "present",
                            "absent",
                            "ambiguous",
                            "unknown",
                        ],
                        "phenomenon": [
                            "local_change",
                            "common_trend",
                            "dropout",
                            "mixed",
                            "unknown",
                        ],
                    },
                }
                (folder / "queue.json").write_text(
                    json.dumps(queue, ensure_ascii=False), encoding="utf-8"
                )
                (folder / "manifest.json").write_text(
                    json.dumps(
                        {"cards": 1, "queue_sha256": sha256(folder / "queue.json")}
                    ),
                    encoding="utf-8",
                )
                package = publish_queue(
                    platform,
                    app.state.store,
                    folder,
                    user["project_id"],
                    "Независимая проверка окна",
                    db,
                )
            at = time.time()
            request = {
                "id": ident,
                "investigation_id": key,
                "project_id": user["project_id"],
                "package": package["id"],
                "review_id": ident,
                "queue_sha256": package["queue_sha256"],
                "source_sha256": context["source_sha256"],
                "context": context,
                "assigned_to": body.assigned_to,
                "created_by": user["id"],
                "created_at": at,
                "status": "assigned",
            }
            db.execute(
                "INSERT INTO documents VALUES('investigation_review',%s,%s,%s)",
                (ident, Jsonb(request), at),
            )
            assignment_key = (
                user["project_id"] + ":" + package["queue_sha256"] + ":" + ident
            )
            assignment = {
                "id": assignment_key,
                "package": package["id"],
                "review_id": ident,
                "queue_sha256": package["queue_sha256"],
                "source_sha256": context["source_sha256"],
                "project_id": user["project_id"],
                "state": "assigned",
                "assigned_to": body.assigned_to,
                "revision": 1,
                "updated_at": at,
                "actor": user["id"],
                "training_label": None,
            }
            db.execute(
                "INSERT INTO documents VALUES('case',%s,%s,%s)",
                (assignment_key, Jsonb(assignment), at),
            )
            work = {**work, "revision": work["revision"] + 1, "updated_at": at}
            store(
                db,
                work,
                "review_requested",
                {
                    "request_id": ident,
                    "package": package["id"],
                    "assigned_to": body.assigned_to,
                    "context": context,
                },
            )
            db.execute(
                "INSERT INTO documents VALUES('investigation_request',%s,%s,%s)",
                (
                    namespace,
                    Jsonb(
                        {
                            "project_id": user["project_id"],
                            "request_sha256": request_hash,
                            "response": request,
                        }
                    ),
                    at,
                ),
            )
            return request

    @app.get(
        "/api/v2/investigations/{key}/review-requests",
        response_model=list[InvestigationReviewView],
    )
    def review_requests(key: str):
        user, platform = access()
        with platform.connect() as db:
            document(db, key)
            rows = db.execute(
                "SELECT payload FROM documents WHERE kind='investigation_review' AND payload->>'investigation_id'=%s AND payload->>'project_id'=%s ORDER BY updated DESC",
                (key, user["project_id"]),
            ).fetchall()
            result = []
            for row in rows:
                value = row[0]
                namespace = f"{user['project_id']}:{value['queue_sha256']}:{value['assigned_to']}"
                answer = db.execute(
                    "SELECT payload FROM reviews WHERE namespace=%s AND review_id=%s ORDER BY revision DESC LIMIT 1",
                    (namespace, value["review_id"]),
                ).fetchone()
                result.append(
                    {
                        **value,
                        "status": "answered" if answer else value["status"],
                        "answer_revision": answer[0]["revision"] if answer else None,
                        "answer_author": answer[0]["reviewer"] if answer else None,
                    }
                )
        return result

    @app.get("/api/v2/review-inbox")
    def inbox():
        user, platform = access()
        if user["role"] not in ("expert", "admin"):
            raise HTTPException(403, "Expert inbox required")
        with platform.connect() as db:
            rows = db.execute(
                "SELECT payload FROM documents WHERE kind='investigation_review' AND payload->>'project_id'=%s AND payload->>'assigned_to'=%s ORDER BY updated DESC LIMIT 100",
                (user["project_id"], user["id"]),
            ).fetchall()
            result = []
            for row in rows:
                value = row[0]
                namespace = f"{user['project_id']}:{value['queue_sha256']}:{user['id']}"
                answered = db.execute(
                    "SELECT 1 FROM reviews WHERE namespace=%s AND review_id=%s LIMIT 1",
                    (namespace, value["review_id"]),
                ).fetchone()
                result.append(
                    {
                        k: value[k]
                        for k in (
                            "id",
                            "investigation_id",
                            "package",
                            "review_id",
                            "source_sha256",
                            "created_at",
                        )
                    }
                    | {
                        "region": value["context"]["region"],
                        "status": "answered" if answered else "assigned",
                    }
                )
        return {"items": result}

    @app.get("/api/v2/review-progress")
    def review_progress(package: str):
        user, platform = access()
        from .review_packages import resolve_queue

        q = resolve_queue(app, package)
        prefix = f"{user['project_id']}:{q.sha}:{user['id']}:"
        with platform.connect() as db:
            rows = db.execute(
                "SELECT payload FROM documents WHERE kind='review_progress' AND payload->>'namespace'=%s",
                (prefix,),
            ).fetchall()
        return {"items": {row[0]["review_id"]: row[0]["state"] for row in rows}}

    @app.post("/api/v2/review-progress")
    def skip_review(body: ReviewProgressRequest):
        user, platform = access()
        if user["role"] not in ("expert", "admin"):
            raise HTTPException(403, "Expert role required")
        from .review_packages import resolve_queue

        q = resolve_queue(app, body.package)
        if body.review_id not in q.cards:
            raise HTTPException(404, "Review card not found")
        prefix = f"{user['project_id']}:{q.sha}:{user['id']}:"
        value = {
            "namespace": prefix,
            "review_id": body.review_id,
            "state": "skipped",
            "project_id": user["project_id"],
            "account_id": user["id"],
            "queue_sha256": q.sha,
            "training_target": None,
        }
        platform.put("review_progress", prefix + body.review_id, value)
        return value

    @app.post("/api/v2/investigations/{key}/cases", response_model=InvestigationView)
    def link_case(key: str, body: InvestigationCaseLink):
        user, platform = access()
        with platform.connect() as db:
            db.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s,0))",
                ("investigation:" + key,),
            )
            work = document(db, key)
            row = db.execute(
                "SELECT payload FROM documents WHERE kind='evidence_case' AND id=%s",
                (body.case_id,),
            ).fetchone()
            if not row:
                raise HTTPException(404, "Case not found")
            case = row[0]
            if user["role"] not in ("analyst", "admin") and not (
                user["role"] == "expert" and case["account_id"] == user["id"]
            ):
                raise HTTPException(
                    403, "Link your own expert case or use analyst role"
                )
            checked_cases(db, [body.case_id], work["context"])
            if body.case_id in work["case_ids"]:
                return view(work)
            if len(work["case_ids"]) >= 50:
                raise HTTPException(413, "Work case link budget reached")
            value = {
                **work,
                "case_ids": [*work["case_ids"], body.case_id],
                "revision": work["revision"] + 1,
                "updated_at": time.time(),
            }
            store(db, value, "case_linked", {"case_id": body.case_id})
            return view(value)

    app.state.investigation_access = access
    app.state.investigation_document = document
    app.state.investigation_store = store
