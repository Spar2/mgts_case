"""Offline maintenance. Explicit commands; never reattribute legacy votes to accounts."""

import argparse
import hashlib
import json
import os
import sqlite3
from pathlib import Path

from psycopg import sql
from psycopg.types.json import Jsonb

from .platform import Platform

TABLES = (
    "projects",
    "users",
    "memberships",
    "documents",
    "reviews",
    "review_requests",
    "audit",
    "jobs",
    "access_requests",
    "invitations",
    "password_resets",
)


def backup(platform, target):
    target = Path(target)
    target.mkdir(parents=True, exist_ok=False)
    files = {}
    with platform.connect() as db:
        db.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        for table in TABLES:
            result = db.execute("SELECT * FROM " + table)
            columns = [c.name for c in result.description]
            data = json.dumps(
                {"columns": columns, "rows": result.fetchall()},
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            name = table + ".json"
            (target / name).write_bytes(data)
            files[name] = hashlib.sha256(data).hexdigest()
    artifacts = target / "artifacts"
    artifacts.mkdir()
    # Blobs are immutable; only include references present in this snapshot.
    documents = json.loads((target / "documents.json").read_bytes())
    references = set()
    array_references = {}

    def visit(value):
        if isinstance(value, dict):
            if {"key", "sha256", "bytes"} <= value.keys():
                references.add(value["key"])
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(documents)
    kind_index = documents["columns"].index("kind")
    payload_index = documents["columns"].index("payload")
    for row in documents["rows"]:
        if row[kind_index] in ("analysis_run","review_package"):
            value = row[payload_index]
            identity = value["id"]
            if len(identity) != 64 or any(
                c not in "0123456789abcdef" for c in identity
            ):
                raise ValueError("Invalid analysis artifact identity")
            array_references[identity] = value["files"]
    visit(json.loads((target / "jobs.json").read_bytes()))
    for identity, entries in array_references.items():
        destination = target / "artifacts" / "arrays" / identity
        destination.mkdir(parents=True, exist_ok=True)
        for filename, checksum in entries.items():
            if Path(filename).name != filename:
                raise ValueError("Invalid analysis artifact path")
            data = (platform.blobs.root / "arrays" / identity / filename).read_bytes()
            if hashlib.sha256(data).hexdigest() != checksum:
                raise ValueError("Corrupt analysis artifact")
            name = "artifacts/arrays/" + identity + "/" + filename
            (target / name).write_bytes(data)
            files[name] = checksum
    for key in references:
        data = (platform.blobs.root / (key + ".json")).read_bytes()
        if hashlib.sha256(data).hexdigest() != key:
            raise ValueError("Corrupt artifact")
        name = "artifacts/" + key + ".json"
        (target / name).write_bytes(data)
        files[name] = key
    (target / "manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "files": files,
                "sessions_restored": False,
                "access_links_restored": "revoked",
                "jobs_restored": "completed preserved; active marked failed",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        "Snapshot verified and written. Keep it private: it contains account password hashes."
    )


def restore(platform, source):
    source = Path(source)
    manifest = json.loads((source / "manifest.json").read_bytes())
    if manifest["version"] != 1:
        raise ValueError("Unsupported snapshot")
    for name, digest in manifest["files"].items():
        path = (source / name).resolve()
        if (
            not path.is_relative_to(source.resolve())
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest
        ):
            raise ValueError("Snapshot integrity failure")
    with platform.connect() as db:
        for table in TABLES:
            if (
                table != "projects"
                and db.execute(
                    sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table))
                ).fetchone()[0]
            ):
                raise ValueError("Restore requires an empty target database")
        if db.execute("SELECT count(*) FROM sessions").fetchone()[0]:
            raise ValueError("Restore requires an empty target database")
        for table in TABLES:
            if not (source / (table + ".json")).exists():
                if table in ("access_requests", "invitations", "password_resets"):
                    continue  # Legacy v1 snapshots predate account lifecycle.
                raise ValueError("Incomplete snapshot")
            value = json.loads((source / (table + ".json")).read_bytes())
            columns = value["columns"]
            for row in value["rows"]:
                if table in ("invitations", "password_resets"):
                    row[columns.index("revoked")] = True
                if table == "jobs" and row[columns.index("status")] in (
                    "queued",
                    "running",
                ):
                    row[columns.index("status")] = "failed"
                    row[columns.index("owner")] = None
                    row[columns.index("error")] = (
                        "Restored interrupted task; resubmit explicitly"
                    )
                params = [Jsonb(v) if isinstance(v, (dict, list)) else v for v in row]
                db.execute(
                    sql.SQL(
                        "INSERT INTO {} ({}) VALUES ({}) ON CONFLICT DO NOTHING"
                    ).format(
                        sql.Identifier(table),
                        sql.SQL(",").join(map(sql.Identifier, columns)),
                        sql.SQL(",").join([sql.Placeholder()] * len(columns)),
                    ),
                    params,
                )
        db.execute(
            "SELECT setval(pg_get_serial_sequence('audit','id'),COALESCE((SELECT max(id) FROM audit),1),(SELECT count(*)>0 FROM audit))"
        )
        for name in manifest["files"]:
            if name.startswith("artifacts/"):
                if name.startswith("artifacts/arrays/"):
                    destination = platform.blobs.root / name.removeprefix("artifacts/")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes((source / name).read_bytes())
                    continue
                value = json.loads((source / name).read_bytes())
                ref = platform.blobs.put(value)
                if ref["sha256"] != manifest["files"][name]:
                    raise ValueError("Artifact serialization differs")
    print(
        "Restored. Sessions and in-flight jobs intentionally require fresh login/submission."
    )


def migrate(platform, source):
    source = Path(source).resolve()
    counts = {}
    for path in sorted(source.rglob("*.sqlite3")):
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
            tables = {
                r[0]
                for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            if "documents" in tables:
                for kind, key, payload in db.execute(
                    "SELECT kind,id,payload FROM documents"
                ):
                    value = json.loads(payload)
                    if kind == "run" and value.get("status") in (
                        "running",
                        "queued",
                        "cancel_requested",
                    ):
                        value.update(
                            status="interrupted",
                            error="Imported legacy in-flight task; resubmit explicitly",
                        )
                    identity = (
                        "legacy-"
                        + hashlib.sha256((str(path) + kind + key).encode()).hexdigest()
                    )
                    try:
                        platform.get(kind, identity)
                    except KeyError:
                        platform.put(
                            kind,
                            identity,
                            {
                                **value,
                                "id": identity,
                                "legacy_id": key,
                                "migration_origin": str(path),
                                "project_id": "local",
                            },
                        )
                    counts[kind] = counts.get(kind, 0) + 1
            # Archive the complete original SQLite DB as bytes in an immutable document.
            # No self-declared author is silently converted into an authenticated expert.
            import base64

            snapshot = db.serialize()
            digest = hashlib.sha256(snapshot).hexdigest()
            platform.put(
                "legacy_archive",
                digest,
                {
                    "id": digest,
                    "source_sha256": digest,
                    "filename": path.name,
                    "result": {"sqlite_base64": base64.b64encode(snapshot).decode()},
                    "identity_mapping": "unassigned; explicit human mapping required",
                    "project_id": "local",
                },
            )
    print(
        json.dumps(
            {"imported": counts, "legacy_reviews": "archived without reassignment"}
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["backup", "restore", "migrate-sqlite"])
    parser.add_argument("directory")
    args = parser.parse_args()
    platform = Platform(
        os.environ["DTS_DATABASE_URL"], os.environ.get("DTS_RUNTIME", ".runtime")
    )
    {"backup": backup, "restore": restore, "migrate-sqlite": migrate}[args.action](
        platform, args.directory
    )


if __name__ == "__main__":
    main()
