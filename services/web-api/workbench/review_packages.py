"""Immutable project review packages; no replacement of legacy queues."""

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path

import numpy as np
from fastapi import HTTPException
from psycopg.types.json import Jsonb

from .platform import principal
from .store import Queue
from .vendor.data import sha256


def resolve_queue(app, key):
    if key in ("general", "real-data-v2"):
        if key not in app.state.packages:
            raise HTTPException(404, "Review package not found")
        return app.state.packages[key]
    user = principal.get()
    if not user or not app.state.platform:
        raise HTTPException(401, "Sign in to read review package")
    try:
        metadata = app.state.platform.get("review_package", key)
    except KeyError:
        raise HTTPException(404, "Review package not found") from None
    root = app.state.platform.blobs.root / "arrays" / metadata["id"]
    if len(metadata["id"]) != 64 or any(
        c not in "0123456789abcdef" for c in metadata["id"]
    ):
        raise HTTPException(503, "Invalid saved package identity")
    try:
        for name, checksum in metadata["files"].items():
            if Path(name).name != name or sha256(root / name) != checksum:
                raise ValueError("Review package integrity differs")
        if key not in app.state.packages:
            app.state.packages[key] = Queue(root, app.state.store, max_cells=1_000_000)
        return app.state.packages[key]
    except (OSError, ValueError):
        raise HTTPException(
            503, "Saved review package unavailable; no replacement selected"
        ) from None


def package_views(app):
    items = list(app.state.package_info.values())
    if app.state.platform and principal.get():
        for value in app.state.platform.list("review_package"):
            items.append(
                {k: value[k] for k in ("id", "title", "cards", "queue_sha256")}
            )
    return items


def publish_queue(platform, store, source, project, title, db):
    """Verify source cells before publishing the immutable package in our transaction."""
    source = Path(source)
    queue = Queue(source, store, max_cells=1_000_000)
    if len(queue.cards) > 1000:
        raise HTTPException(413, "Review package too large")
    names = {"queue.json", "manifest.json"}
    for key, card in queue.cards.items():
        _, data = queue.load(key)
        native = store.get(card["source"].removesuffix(".csv"))
        rs, re = card["row_bounds"]
        cs, ce = card["column_bounds"]
        for field, expected in (
            ("raw_c", native.raw_c[rs:re, cs:ce]),
            ("invalid_reason", native.invalid_reason[rs:re, cs:ce]),
            ("x_m", native.x_m[cs:ce]),
            ("time_s", native.elapsed_s[rs:re]),
        ):
            if data[field].dtype != expected.dtype or not np.array_equal(
                data[field], expected, equal_nan=True
            ):
                raise ValueError("Review data differ from native observations")
        names.add(card["artifact"])
    if sum((source / name).stat().st_size for name in names) > 512 * 1024 * 1024:
        raise HTTPException(413, "Review package exceeds budget")
    files = {name: sha256(source / name) for name in sorted(names)}
    identity = hashlib.sha256(
        json.dumps({"project": project, "files": files}, sort_keys=True).encode()
    ).hexdigest()
    target = platform.blobs.root / "arrays" / identity
    target.parent.mkdir(exist_ok=True)
    db.execute(
        "SELECT pg_advisory_xact_lock(hashtextextended(%s,0))",
        ("review-package:" + identity,),
    )
    old = db.execute(
        "SELECT payload FROM documents WHERE kind='review_package' AND id=%s",
        (identity,),
    ).fetchone()
    if old:
        for name, checksum in old[0]["files"].items():
            if Path(name).name != name or sha256(target / name) != checksum:
                raise ValueError("Previously imported review artifact corrupt")
        return old[0]
    temporary = target.parent / ("review-pending-" + uuid.uuid4().hex)
    temporary.mkdir()
    try:
        for name, checksum in files.items():
            shutil.copyfile(source / name, temporary / name)
            if sha256(temporary / name) != checksum:
                raise ValueError("Review package changed during publication")
        if not target.exists():
            os.rename(temporary, target)
        else:
            for name, checksum in files.items():
                if sha256(target / name) != checksum:
                    raise ValueError("Existing package differs")
    finally:
        # Exactly our newly created staging directory, no external paths or recursive traversal.
        if temporary.exists():
            for file in temporary.iterdir():
                file.unlink()
            temporary.rmdir()
    value = {
        "id": identity,
        "project_id": project,
        "title": title,
        "cards": len(queue.cards),
        "queue_sha256": queue.sha,
        "files": files,
        "expert_labels_created": 0,
    }
    db.execute(
        "INSERT INTO documents VALUES('review_package',%s,%s,extract(epoch from now()))",
        (identity, Jsonb(value)),
    )
    return value
