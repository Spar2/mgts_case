"""Explicit review provenance, transactional revisions and no implicit targets.

Adapts the identity/label-scope separation from frozen Sol dts/reviews.py;
storage and revision conflicts are independently implemented here.
"""
import json
import hashlib
from contextlib import contextmanager
from pathlib import Path
import sqlite3
from datetime import datetime, timezone

from .data import sha256


class ReviewConflict(ValueError):
    pass


class ReviewJournal:
    def __init__(self, queue_path, database):
        self.queue_path, self.database = Path(queue_path), Path(database)
        self.queue_sha = sha256(self.queue_path)
        self.queue = json.loads(self.queue_path.read_text(encoding="utf-8"))
        self.cards = {card["review_id"]: card for card in self.queue["cards"]}
        if len(self.cards) != len(self.queue["cards"]):
            raise ValueError("duplicate queue identity")
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            row = conn.execute("SELECT value FROM meta WHERE key='queue_sha256'").fetchone()
            if row and row[0] != self.queue_sha:
                raise ValueError("review database belongs to a different immutable queue")
            conn.execute("INSERT OR IGNORE INTO meta VALUES ('queue_sha256', ?)", (self.queue_sha,))
            conn.execute("CREATE TABLE IF NOT EXISTS annotations (review_id TEXT NOT NULL, revision INTEGER NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(review_id, revision))")
            if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("review journal integrity failure")

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.database, timeout=10)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=FULL")
            with conn:
                yield conn
        finally:
            conn.close()

    def annotations(self):
        with self.connect() as conn:
            rows = conn.execute("SELECT a.payload FROM annotations a JOIN (SELECT review_id, MAX(revision) revision FROM annotations GROUP BY review_id) latest ON a.review_id=latest.review_id AND a.revision=latest.revision ORDER BY a.review_id").fetchall()
        return {item["review_id"]: item for (payload,) in rows for item in [json.loads(payload)]}

    def save(self, body, *, idempotency_key=None):
        allowed = {"queue_sha256", "review_id", "expected_revision", "reviewer", "reviewer_type", "temperature_episode", "phenomenon", "notes"}
        if not isinstance(body, dict) or set(body) != allowed or body["queue_sha256"] != self.queue_sha:
            raise ValueError("exact review fields and matching queue digest required")
        key, expected = body["review_id"], body["expected_revision"]
        if not isinstance(key, str) or key not in self.cards or isinstance(expected, bool) or not isinstance(expected, int) or expected < 0:
            raise ValueError("known review identity and nonnegative expected revision required")
        if body["reviewer_type"] not in ("expert", "agent"):
            raise ValueError("explicit expert or agent review type required")
        for field, limit in (("reviewer", 100), ("notes", 8000)):
            if not isinstance(body[field], str) or not body[field].strip() or len(body[field]) > limit:
                raise ValueError("reviewer and evidence notes must be nonempty and bounded")
        for field in ("temperature_episode", "phenomenon"):
            if body[field] not in self.queue["allowed_answers"][field]:
                raise ValueError("invalid review answer")
        card = self.cards[key]
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("CREATE TABLE IF NOT EXISTS request_keys (key TEXT PRIMARY KEY, digest TEXT NOT NULL, payload TEXT NOT NULL)")
            digest = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            if idempotency_key:
                prior = conn.execute("SELECT digest, payload FROM request_keys WHERE key=?", (idempotency_key,)).fetchone()
                if prior:
                    if prior[0] != digest:
                        raise ReviewConflict("idempotency key reused with different content")
                    return json.loads(prior[1])
            current = conn.execute("SELECT MAX(revision) FROM annotations WHERE review_id=?", (key,)).fetchone()[0] or 0
            if current != expected:
                raise ReviewConflict("review changed in another window; reload before saving")
            payload = {**body, "revision": current+1, "reviewer": body["reviewer"].strip(), "notes": body["notes"].strip(),
                       "saved_at": datetime.now(timezone.utc).isoformat(), "source": card["source"], "evidence": card["evidence"],
                       "window": {"row_bounds": card["row_bounds"], "column_bounds": card["column_bounds"]},
                       "label_scope": "displayed window only; no event boundaries or automatic training labels",
                       "physical_cause": "unknown", "operational_alarm": "unknown", "training_target": None}
            conn.execute("INSERT INTO annotations VALUES (?, ?, ?)", (key, current+1, json.dumps(payload, ensure_ascii=False, allow_nan=False)))
            if idempotency_key:
                conn.execute("INSERT INTO request_keys VALUES (?, ?, ?)", (idempotency_key, digest, json.dumps(payload, ensure_ascii=False, allow_nan=False)))
        return payload

    def export(self):
        return {"version": "general-review-packet-v1", "queue_sha256": self.queue_sha,
                "annotations": list(self.annotations().values()), "automatic_training_labels_created": 0,
                "policy": "Expert and agent responses retain reviewer_type. Window responses are not cell labels or physical causes."}
