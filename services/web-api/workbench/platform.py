"""Transactional control plane. Large immutable payloads live outside PostgreSQL."""

import hashlib
import json
import os
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

import psycopg
from fastapi import HTTPException
from psycopg.types.json import Jsonb

from .migrations import migrate

principal = ContextVar("dts_principal", default=None)
worker_lease = ContextVar("dts_worker_lease", default=None)

SCHEMA = """
CREATE TABLE IF NOT EXISTS platform_schema(version integer PRIMARY KEY);
INSERT INTO platform_schema VALUES(1) ON CONFLICT DO NOTHING;
CREATE TABLE IF NOT EXISTS documents(kind text, id text, payload jsonb NOT NULL, updated double precision NOT NULL, PRIMARY KEY(kind,id));
CREATE TABLE IF NOT EXISTS users(id text PRIMARY KEY, username text UNIQUE NOT NULL, display_name text NOT NULL, password_hash text NOT NULL, enabled boolean NOT NULL DEFAULT true);
CREATE TABLE IF NOT EXISTS projects(id text PRIMARY KEY, name text NOT NULL);
INSERT INTO projects VALUES('local','Локальный проект DTS') ON CONFLICT DO NOTHING;
CREATE TABLE IF NOT EXISTS memberships(project_id text REFERENCES projects, user_id text REFERENCES users, role text CHECK(role IN ('viewer','expert','analyst','admin')), PRIMARY KEY(project_id,user_id));
CREATE TABLE IF NOT EXISTS sessions(digest text PRIMARY KEY, user_id text REFERENCES users, csrf text NOT NULL, expires double precision NOT NULL);
CREATE TABLE IF NOT EXISTS jobs(id text PRIMARY KEY, project_id text NOT NULL, kind text NOT NULL, payload jsonb NOT NULL, status text NOT NULL DEFAULT 'queued', dedup_key text UNIQUE, attempts integer NOT NULL DEFAULT 0, max_attempts integer NOT NULL DEFAULT 3, owner text, lease_until double precision, cancel boolean NOT NULL DEFAULT false, result jsonb, error text, created double precision NOT NULL, updated double precision NOT NULL);
CREATE INDEX IF NOT EXISTS jobs_claim ON jobs(status,created);
CREATE TABLE IF NOT EXISTS reviews(namespace text, review_id text, revision integer, payload jsonb NOT NULL, PRIMARY KEY(namespace,review_id,revision));
CREATE TABLE IF NOT EXISTS review_requests(namespace text, key text, digest text NOT NULL, payload jsonb NOT NULL, PRIMARY KEY(namespace,key));
CREATE TABLE IF NOT EXISTS audit(id bigserial PRIMARY KEY, project_id text, actor text, action text NOT NULL, resource text, at double precision NOT NULL);
"""


class JobConflict(ValueError):
    pass


class LeaseLost(RuntimeError):
    """A superseded worker must stop publishing mutable state."""


class BlobStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, value):
        data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
        key = hashlib.sha256(data).hexdigest()
        path = self.root / (key + ".json")
        if not path.exists():
            temp = self.root / (key + "." + uuid.uuid4().hex + ".tmp")
            temp.write_bytes(data)
            os.replace(temp, path)
        return {"key": key, "sha256": key, "bytes": len(data)}

    def get(self, ref):
        key = ref["key"]
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("Invalid artifact identity")
        data = (self.root / (key + ".json")).read_bytes()
        if hashlib.sha256(data).hexdigest() != ref["sha256"]:
            raise ValueError("Artifact integrity mismatch")
        return json.loads(data)


class Platform:
    def __init__(self, url, runtime, initialize=True):
        self.url = url
        self.blobs = BlobStore(Path(runtime) / "artifacts")
        if initialize:
            with self.connect() as db:
                db.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(current_schema()||':dts_platform_schema',0))"
                )
                db.execute(SCHEMA)
                migrate(db)

    @contextmanager
    def connect(self):
        with psycopg.connect(self.url, connect_timeout=5) as db:
            yield db

    def audit(self, action, resource="", project="local", actor=None):
        user = principal.get()
        actor = actor or (user["id"] if user else "system")
        with self.connect() as db:
            db.execute(
                "INSERT INTO audit(project_id,actor,action,resource,at) VALUES(%s,%s,%s,%s,%s)",
                (project, actor, action, resource, time.time()),
            )

    def put(self, kind, key, value):
        user = principal.get()
        if user:
            if value.get("project_id", user["project_id"]) != user["project_id"]:
                raise PermissionError("Wrong project")
            value = {**value, "project_id": user["project_id"]}
        value = {**value, "updated_at": time.time()}
        stored = dict(value)
        if "result" in stored:
            result = stored.pop("result")
            stored["result_ref"] = self.blobs.put(result)
            zones = result.get("zones", []) if isinstance(result, dict) else []
            zones = zones if isinstance(zones, list) else []
            stored["has_result"] = True
            stored["result_summary"] = {
                "zones": zones[:100],
                "zone_count": len(zones)
                if isinstance(result, dict) and "zones" in result
                else None,
                "processed_points": len(result["denoised"])
                if isinstance(result, dict) and isinstance(result.get("denoised"), list)
                else None,
            }
        with self.connect() as db:
            lease = worker_lease.get()
            if (
                lease
                and not db.execute(
                    "SELECT 1 FROM jobs WHERE id=%s AND owner=%s AND status='running' AND lease_until>%s FOR SHARE",
                    (lease["id"], lease["owner"], time.time()),
                ).fetchone()
            ):
                raise LeaseLost("Worker no longer owns this job")
            project = user["project_id"] if user else None
            row = db.execute(
                "INSERT INTO documents VALUES(%s,%s,%s,%s) ON CONFLICT(kind,id) DO UPDATE SET payload=excluded.payload,updated=excluded.updated WHERE %s::text IS NULL OR COALESCE(documents.payload->>'project_id','local')=%s RETURNING id",
                (kind, key, Jsonb(stored), value["updated_at"], project, project),
            ).fetchone()
            if not row:
                raise PermissionError("Wrong project")
        return value

    def hydrate(self, value):
        if "result_ref" in value:
            value["result"] = self.blobs.get(value["result_ref"])
        return value

    def list(self, kind, limit=500, offset=0, hydrate=False):
        user = principal.get()
        project = user["project_id"] if user else None
        with self.connect() as db:
            rows = db.execute(
                "SELECT payload FROM documents WHERE kind=%s AND (%s::text IS NULL OR COALESCE(payload->>'project_id','local')=%s) ORDER BY updated DESC,id LIMIT %s OFFSET %s",
                (kind, project, project, min(limit, 500), offset),
            ).fetchall()
        rows = [
            r
            for r in rows
            if not user or r[0].get("project_id", "local") == user["project_id"]
        ]
        values = [self.hydrate(r[0]) if hydrate else r[0] for r in rows]
        if kind == "run":
            with self.connect() as db:
                statuses = {
                    r[0]: (r[1], r[2])
                    for r in db.execute(
                        "SELECT id,status,error FROM jobs WHERE id=ANY(%s)",
                        ([v["id"] for v in values],),
                    ).fetchall()
                }
            for value in values:
                if value["id"] in statuses:
                    value.update(
                        status=statuses[value["id"]][0], error=statuses[value["id"]][1]
                    )
        return values

    def get(self, kind, key):
        with self.connect() as db:
            row = db.execute(
                "SELECT payload FROM documents WHERE kind=%s AND id=%s", (kind, key)
            ).fetchone()
        if not row:
            raise KeyError(key)
        user = principal.get()
        if user and row[0].get("project_id", "local") != user["project_id"]:
            raise PermissionError("Wrong project")
        return self.hydrate(row[0])

    def delete(self, kind, key):
        self.get(kind, key)
        user = principal.get()
        project = user["project_id"] if user else None
        with self.connect() as db:
            row = db.execute(
                "DELETE FROM documents WHERE kind=%s AND id=%s AND (%s::text IS NULL OR COALESCE(payload->>'project_id','local')=%s) RETURNING id",
                (kind, key, project, project),
            ).fetchone()
            if not row:
                raise KeyError(key)

    def enqueue(self, kind, payload, key=None, project="local"):
        identity = uuid.uuid4().hex
        now = time.time()
        with self.connect() as db:
            db.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s,0))",
                ("jobs:" + project,),
            )
            previous = (
                db.execute(
                    "SELECT id,kind,payload,project_id FROM jobs WHERE dedup_key=%s",
                    (key,),
                ).fetchone()
                if key
                else None
            )
            if previous:
                if previous[1:] != (kind, payload, project):
                    raise JobConflict("Idempotency key has different content")
                return self.job(previous[0])
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
            row = db.execute(
                "INSERT INTO jobs(id,project_id,kind,payload,dedup_key,created,updated,submitted_by) VALUES(%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(dedup_key) DO UPDATE SET dedup_key=excluded.dedup_key RETURNING id,kind,payload,project_id",
                (
                    identity,
                    project,
                    kind,
                    Jsonb(payload),
                    key,
                    now,
                    now,
                    user["id"] if user else None,
                ),
            ).fetchone()
            if row[1:] != (kind, payload, project):
                raise JobConflict("Idempotency key has different content")
            if kind == "detect":
                metadata = {
                    "id": row[0],
                    "project_id": project,
                    "status": "queued",
                    "progress": 0,
                    "region": {
                        k: payload[k]
                        for k in (
                            "record_id",
                            "row_start",
                            "row_end",
                            "column_start",
                            "column_end",
                        )
                    },
                    "source_sha256": payload["source_sha256"],
                    "method": "reference_detector",
                    "method_version": "ML API contract 0.1; implementation SHA not supplied",
                    "parameters": {},
                    "training_origin": "untrained",
                    "limitations": "Reference hypotheses; real accuracy unknown",
                    "updated_at": now,
                }
                db.execute(
                    "INSERT INTO documents VALUES('run',%s,%s,%s) ON CONFLICT(kind,id) DO NOTHING",
                    (row[0], Jsonb(metadata), now),
                )
        return self.job(row[0])

    def job(self, key):
        with self.connect() as db:
            row = db.execute(
                "SELECT id,project_id,kind,payload,status,attempts,cancel,result,error,created,updated FROM jobs WHERE id=%s",
                (key,),
            ).fetchone()
        if not row:
            raise KeyError(key)
        return dict(
            zip(
                (
                    "id",
                    "project_id",
                    "kind",
                    "payload",
                    "status",
                    "attempts",
                    "cancel_requested",
                    "result",
                    "error",
                    "created_at",
                    "updated_at",
                ),
                row,
            )
        )

    def claim(self, owner, kinds=None):
        now = time.time()
        with self.connect() as db:
            db.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(current_schema()||':job_claim',0))"
            )
            db.execute(
                "UPDATE jobs SET status=CASE WHEN cancel THEN 'cancelled' WHEN attempts>=max_attempts THEN 'failed' ELSE 'queued' END,error='Worker lease expired',owner=NULL WHERE status='running' AND lease_until<%s",
                (now,),
            )
            db.execute(
                "UPDATE jobs SET status='cancelled',updated=%s WHERE status='queued' AND cancel",
                (now,),
            )
            row = db.execute(
                "SELECT j.id FROM jobs j WHERE j.status='queued' AND NOT j.cancel AND j.kind=ANY(%s) AND (SELECT count(*) FROM jobs r WHERE r.project_id=j.project_id AND r.status='running')<2 ORDER BY j.created FOR UPDATE SKIP LOCKED LIMIT 1",
                (kinds or ["detect", "prepare_views", "import_csv", "stream_replay"],),
            ).fetchone()
            if row:
                db.execute(
                    "UPDATE jobs SET status='running',owner=%s,lease_until=%s,attempts=attempts+1,updated=%s WHERE id=%s",
                    (owner, now + 30, now, row[0]),
                )
        return self.job(row[0]) if row else None

    def heartbeat(self, key, owner):
        with self.connect() as db:
            return db.execute(
                "UPDATE jobs SET lease_until=%s,updated=%s WHERE id=%s AND owner=%s AND status='running' AND lease_until>%s RETURNING cancel",
                (time.time() + 30, time.time(), key, owner, time.time()),
            ).fetchone()

    def finish(self, key, owner, result=None, error=None, retry=False):
        with self.connect() as db:
            completed = db.execute(
                "UPDATE jobs SET status=CASE WHEN cancel THEN 'cancelled' WHEN %s AND attempts<max_attempts THEN 'queued' WHEN %s THEN 'failed' ELSE 'complete' END,result=CASE WHEN cancel THEN NULL ELSE %s::jsonb END,error=%s,updated=%s,owner=NULL WHERE id=%s AND owner=%s AND status='running' AND lease_until>%s RETURNING kind,status",
                (
                    retry,
                    error is not None,
                    Jsonb(result) if result is not None else None,
                    error,
                    time.time(),
                    key,
                    owner,
                    time.time(),
                ),
            ).fetchone()
            if completed == ("detect", "complete"):
                # Commit the immutable result reference and job completion together.
                # No post-finish window permits an old worker to replace a newer run.
                db.execute(
                    "UPDATE documents SET payload=payload || %s,updated=%s WHERE kind='run' AND id=%s",
                    (
                        Jsonb(
                            {
                                "result_ref": result,
                                "has_result": True,
                                "status": "complete",
                                "progress": 100,
                            }
                        ),
                        time.time(),
                        key,
                    ),
                )
        return completed is not None

    def retry_job(self, key, project):
        with self.connect() as db:
            db.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s,0))",
                ("jobs:" + project,),
            )
            job = db.execute(
                "SELECT status,kind,project_id FROM jobs WHERE id=%s FOR UPDATE",
                (key,),
            ).fetchone()
            if not job:
                raise KeyError(key)
            if job[2] != project:
                raise PermissionError("Wrong project")
            if job[0] not in ("failed", "cancelled") or job[1] != "stream_replay":
                raise JobConflict(
                    "Only a failed or cancelled measurement replay can be resumed"
                )
            user = principal.get()
            if (
                user
                and db.execute(
                    "SELECT count(*) FROM jobs WHERE project_id=%s AND submitted_by=%s AND status IN ('queued','running')",
                    (project, user["id"]),
                ).fetchone()[0]
            ):
                raise HTTPException(
                    429, "У вас уже есть активная задача", headers={"Retry-After": "10"}
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
            db.execute(
                "UPDATE jobs SET status='queued',cancel=false,owner=NULL,lease_until=NULL,error=NULL,result=NULL,max_attempts=attempts+3,updated=%s,submitted_by=%s WHERE id=%s",
                (time.time(), user["id"] if user else None, key),
            )
            db.execute(
                "INSERT INTO audit(project_id,actor,action,resource,at) VALUES(%s,%s,'job.resumed',%s,%s)",
                (
                    project,
                    principal.get()["id"] if principal.get() else "system",
                    key,
                    time.time(),
                ),
            )
        return self.job(key)

    def cancel_job(self, key):
        with self.connect() as db:
            db.execute(
                "UPDATE jobs SET cancel=true,status=CASE WHEN status='queued' THEN 'cancelled' ELSE status END,updated=%s WHERE id=%s AND status IN ('queued','running')",
                (time.time(), key),
            )
        return self.job(key)


def isolated_anonymous_test():
    """The legacy, unauthenticated fixture is never a deployment default."""
    return (
        os.environ.get("DTS_ENV") == "test"
        and os.environ.get("DTS_TEST_ALLOW_ANONYMOUS") == "true"
    )


def configured(runtime):
    url = os.environ.get("DTS_DATABASE_URL")
    if not url and not isolated_anonymous_test():
        raise ValueError(
            "Team deployment requires PostgreSQL; account fallback is forbidden"
        )
    if not url:
        return None
    try:
        return Platform(url, runtime)
    except psycopg.OperationalError:
        # Preserve the account boundary during outages. Requests requiring
        # storage fail 503; no anonymous workspace fallback is permitted.
        return Platform(url, runtime, initialize=False)
