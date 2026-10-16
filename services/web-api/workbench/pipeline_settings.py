"""Saved pipeline windows. A separate document kind, never an investigation."""

import hashlib
import json
import time
import uuid
from typing import Annotated

from fastapi import Header, HTTPException, Query
from psycopg.types.json import Jsonb
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .analysis_compute import METHOD, ML_COMMIT
from .analysis_jobs import AnalysisJobRequest, validate_region
from .platform import principal
from .workspace_routes import Region

KIND = "pipeline_setting"
REQUEST_KIND = "pipeline_setting_request"


class Window(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    row_start: int = Field(ge=0)
    row_end: int = Field(ge=1)
    column_start: int = Field(ge=0)
    column_end: int = Field(ge=2)


class Point(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    row: int = Field(ge=0)
    column: int = Field(ge=0)


def _pinned_method(value):
    if value is not None and value != METHOD:
        raise ValueError("Only the pinned method is accepted")
    return value


def _pinned_version(value):
    if value is not None and value != ML_COMMIT:
        raise ValueError("Only the pinned method version is accepted")
    return value


def _empty_parameters(value):
    if value is not None and value != {}:
        raise ValueError("This method has no client parameters")
    return value


class PipelineSettingCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    title: str = Field(min_length=1, max_length=150)
    record_id: str = Field(min_length=1, max_length=128)
    source_sha256: str = Field(pattern="^[0-9a-f]{64}$")
    window: Window
    point: Point
    method: str = Field(min_length=1, max_length=100)
    method_version: str = Field(min_length=1, max_length=64)
    parameters: dict
    run_ids: list[str] = Field(default_factory=list, max_length=2)

    @field_validator("method")
    @classmethod
    def method_is_pinned(cls, value):
        return _pinned_method(value)

    @field_validator("method_version")
    @classmethod
    def version_is_pinned(cls, value):
        return _pinned_version(value)

    @field_validator("parameters")
    @classmethod
    def parameters_are_empty(cls, value):
        return _empty_parameters(value)

    @field_validator("run_ids")
    @classmethod
    def run_ids_are_short(cls, value):
        if any(not isinstance(key, str) or not 1 <= len(key) <= 128 for key in value):
            raise ValueError("Run id length must be 1..128")
        return value


class PipelineSettingPatch(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    expected_revision: int = Field(ge=1)
    title: str | None = Field(default=None, min_length=1, max_length=150)
    record_id: str | None = Field(default=None, min_length=1, max_length=128)
    source_sha256: str | None = Field(default=None, pattern="^[0-9a-f]{64}$")
    window: Window | None = None
    point: Point | None = None
    method: str | None = Field(default=None, min_length=1, max_length=100)
    method_version: str | None = Field(default=None, min_length=1, max_length=64)
    parameters: dict | None = None
    run_ids: list[str] | None = Field(default=None, max_length=2)

    @field_validator("method")
    @classmethod
    def method_is_pinned(cls, value):
        return _pinned_method(value)

    @field_validator("method_version")
    @classmethod
    def version_is_pinned(cls, value):
        return _pinned_version(value)

    @field_validator("parameters")
    @classmethod
    def parameters_are_empty(cls, value):
        return _empty_parameters(value)

    @field_validator("run_ids")
    @classmethod
    def run_ids_are_short(cls, value):
        if value is not None and any(
            not isinstance(key, str) or not 1 <= len(key) <= 128 for key in value
        ):
            raise ValueError("Run id length must be 1..128")
        return value


class PipelineSettingView(PipelineSettingCreate):
    id: str
    project_id: str
    account_id: str
    author_name: str
    revision: int
    created_at: float
    updated_at: float
    job_ids: list[str]


def read_setting(platform, key):
    with platform.connect() as db:
        return fetch_setting(db, key)


def fetch_setting(db, key):
    row = db.execute(
        "SELECT payload FROM documents WHERE kind=%s AND id=%s",
        (KIND, key),
    ).fetchone()
    if not row:
        raise HTTPException(404, "Pipeline setting not found")
    user = principal.get()
    if not user or row[0]["project_id"] != user["project_id"]:
        raise HTTPException(403, "Wrong project")
    return row[0]


def setting_for_update(db, key):
    db.execute(
        "SELECT pg_advisory_xact_lock(hashtextextended(%s,0))",
        ("pipeline_setting:" + key,),
    )
    row = db.execute(
        "SELECT payload FROM documents WHERE kind=%s AND id=%s FOR UPDATE",
        (KIND, key),
    ).fetchone()
    if not row:
        raise HTTPException(404, "Pipeline setting not found")
    user = principal.get()
    if not user or row[0]["project_id"] != user["project_id"]:
        raise HTTPException(403, "Wrong project")
    return row[0]


def replace_setting(db, value):
    db.execute(
        "UPDATE documents SET payload=%s, updated=%s WHERE kind=%s AND id=%s",
        (Jsonb(value), value["updated_at"], KIND, value["id"]),
    )


def assert_submission_matches(setting, body):
    if setting["revision"] != body.settings_revision:
        raise HTTPException(
            409,
            {
                "message": "Pipeline setting changed; reload before submitting",
                "current": setting,
            },
        )
    saved = {"record_id": setting["record_id"], **setting["window"]}
    if body.region.model_dump() != saved or body.source_sha256 != setting["source_sha256"]:
        raise HTTPException(
            409, "Analysis window does not match the saved pipeline setting"
        )
    if (
        setting["method"] != METHOD
        or setting["method_version"] != ML_COMMIT
        or setting["parameters"] != {}
    ):
        raise HTTPException(409, "Saved pipeline setting is not the pinned method")


def checked(app, value):
    window = dict(value["window"])
    point = dict(value["point"])
    if value["method"] != METHOD or value["method_version"] != ML_COMMIT:
        raise HTTPException(422, "Only the pinned method is accepted")
    if value["parameters"] != {}:
        raise HTTPException(422, "This method has no client parameters")
    try:
        region = Region.model_validate({"record_id": value["record_id"], **window})
    except ValidationError:
        raise HTTPException(422, "Select native bounds with at least two columns") from None
    try:
        validate_region(
            app.state.store,
            AnalysisJobRequest(region=region, source_sha256=value["source_sha256"]),
        )
    except KeyError:
        raise HTTPException(404, "Source not found") from None
    if not (
        window["row_start"] <= point["row"] < window["row_end"]
        and window["column_start"] <= point["column"] < window["column_end"]
    ):
        raise HTTPException(422, "Point must lie inside the half-open window")
    checked_runs(
        app.state.platform,
        value["run_ids"],
        value["record_id"],
        value["source_sha256"],
    )


def checked_runs(platform, run_ids, record_id, source_sha256):
    if len(run_ids) > 2:
        raise HTTPException(422, "At most two runs")
    user = principal.get()
    if not run_ids:
        return
    with platform.connect() as db:
        for key in run_ids:
            row = db.execute(
                "SELECT payload FROM documents WHERE id=%s AND kind IN ('analysis_run','run') ORDER BY kind LIMIT 1",
                (key,),
            ).fetchone()
            if not row:
                raise HTTPException(404, "Selected run not found")
            run = row[0]
            if run.get("project_id", "local") != user["project_id"]:
                raise HTTPException(403, "Selected run belongs to another project")
            region = run.get("region") or {}
            if (
                region.get("record_id") != record_id
                or run.get("source_sha256") != source_sha256
            ):
                raise HTTPException(409, "Selected run belongs to another source")


def install_pipeline_settings(app):
    def access(edit=False):
        user = principal.get()
        if not user or not app.state.platform:
            raise HTTPException(503, "Authenticated PostgreSQL workspace required")
        if user["project_id"] != "local":
            raise HTTPException(403, "Native archive belongs to local project")
        if edit and user["role"] not in ("analyst", "admin"):
            raise HTTPException(403, "Analyst or administrator required")
        return user, app.state.platform

    def insert_setting(db, value):
        db.execute(
            "INSERT INTO documents VALUES(%s,%s,%s,%s)",
            (KIND, value["id"], Jsonb(value), value["updated_at"]),
        )

    @app.get("/api/v2/pipeline-settings", response_model=list[PipelineSettingView])
    def list_pipeline_settings(
        limit: int = Query(50, ge=1, le=100), offset: int = Query(0, ge=0)
    ):
        _, platform = access()
        return platform.list(KIND, limit, offset)

    @app.post(
        "/api/v2/pipeline-settings",
        response_model=PipelineSettingView,
        status_code=201,
    )
    def create_pipeline_setting(
        body: PipelineSettingCreate,
        idempotency_key: Annotated[str, Header(min_length=8, max_length=128)],
    ):
        user, platform = access(True)
        namespace = (
            user["project_id"] + ":" + user["id"] + ":pipeline_setting:" + idempotency_key
        )
        request_hash = hashlib.sha256(
            json.dumps(body.model_dump(), sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        with platform.connect() as db:
            db.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s,0))", (namespace,)
            )
            previous = db.execute(
                "SELECT payload FROM documents WHERE kind=%s AND id=%s",
                (REQUEST_KIND, namespace),
            ).fetchone()
            if previous:
                if previous[0]["request_sha256"] != request_hash:
                    raise HTTPException(
                        409, "Idempotency key reused with different content"
                    )
                # Replay does not need the worker, the source, or a new id.
                return previous[0]["response"]
            if not body.title.strip():
                raise HTTPException(422, "Title required")
            checked(app, body.model_dump())
            at = time.time()
            value = {
                **body.model_dump(),
                "title": body.title.strip(),
                "id": uuid.uuid4().hex,
                "project_id": user["project_id"],
                "account_id": user["id"],
                "author_name": user["display_name"],
                "revision": 1,
                "created_at": at,
                "updated_at": at,
                "job_ids": [],
            }
            insert_setting(db, value)
            db.execute(
                "INSERT INTO documents VALUES(%s,%s,%s,%s)",
                (
                    REQUEST_KIND,
                    namespace,
                    Jsonb(
                        {
                            "project_id": user["project_id"],
                            "request_sha256": request_hash,
                            "response": value,
                        }
                    ),
                    at,
                ),
            )
            return value

    @app.get(
        "/api/v2/pipeline-settings/{key}", response_model=PipelineSettingView
    )
    def get_pipeline_setting(key: str):
        _, platform = access()
        return read_setting(platform, key)

    @app.patch(
        "/api/v2/pipeline-settings/{key}", response_model=PipelineSettingView
    )
    def patch_pipeline_setting(key: str, body: PipelineSettingPatch):
        _, platform = access(True)
        with platform.connect() as db:
            old = setting_for_update(db, key)
            if body.expected_revision != old["revision"]:
                raise HTTPException(
                    409,
                    {
                        "message": "Pipeline setting changed; reload before saving",
                        "current": old,
                    },
                )
            changes = body.model_dump(exclude_unset=True)
            changes.pop("expected_revision")
            if any(item is None for item in changes.values()):
                raise HTTPException(422, "Omit unchanged fields instead of null")
            if "title" in changes:
                changes["title"] = changes["title"].strip()
                if not changes["title"]:
                    raise HTTPException(422, "Title required")
            value = {
                **old,
                **changes,
                "revision": old["revision"] + 1,
                "updated_at": time.time(),
            }
            checked(app, value)
            replace_setting(db, value)
            return value
