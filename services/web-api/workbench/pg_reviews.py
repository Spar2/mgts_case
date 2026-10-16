"""Independent review revisions keyed by account, project and immutable queue."""

import hashlib
import json
from datetime import UTC, datetime

from psycopg.types.json import Jsonb

from .vendor.reviews import ReviewConflict


class PGReviewJournal:
    def __init__(self, platform, queue, user):
        self.platform, self.queue, self.user = platform, queue, user
        self.queue_sha = queue.sha
        self.namespace = f"{user['project_id']}:{queue.sha}:{user['id']}"

    def annotations(self):
        with self.platform.connect() as db:
            rows = db.execute(
                "SELECT DISTINCT ON(review_id) payload FROM reviews WHERE namespace=%s ORDER BY review_id,revision DESC",
                (self.namespace,),
            ).fetchall()
        return {row[0]["review_id"]: row[0] for row in rows}

    def save(self, body, *, idempotency_key=None):
        if (
            body["queue_sha256"] != self.queue_sha
            or body["review_id"] not in self.queue.cards
        ):
            raise ValueError("Immutable queue and known card required")
        if (
            body["reviewer"] != self.user["display_name"]
            or body["reviewer_type"] != "expert"
        ):
            raise ValueError("Authenticated author required")
        card = self.queue.cards[body["review_id"]]
        digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        with self.platform.connect() as db:
            # Serialize all saves for one journal, including idempotency keys across cards.
            db.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s,0))",
                (self.namespace,),
            )
            if idempotency_key:
                old = db.execute(
                    "SELECT digest,payload FROM review_requests WHERE namespace=%s AND key=%s",
                    (self.namespace, idempotency_key),
                ).fetchone()
                if old:
                    if old[0] != digest:
                        raise ReviewConflict("Idempotency key has different content")
                    return old[1]
            revision = db.execute(
                "SELECT COALESCE(MAX(revision),0) FROM reviews WHERE namespace=%s AND review_id=%s",
                (self.namespace, body["review_id"]),
            ).fetchone()[0]
            if revision != body["expected_revision"]:
                raise ReviewConflict("Review changed in another window")
            payload = {
                **body,
                "revision": revision + 1,
                "saved_at": datetime.now(UTC).isoformat(),
                "account_id": self.user["id"],
                "project_id": self.user["project_id"],
                "source": card["source"],
                "evidence": card["evidence"],
                "window": {
                    "row_bounds": card["row_bounds"],
                    "column_bounds": card["column_bounds"],
                },
                "label_scope": "displayed window only; no automatic training labels",
                "physical_cause": "unknown",
                "operational_alarm": "unknown",
                "training_target": None,
            }
            db.execute(
                "INSERT INTO reviews VALUES(%s,%s,%s,%s)",
                (self.namespace, body["review_id"], revision + 1, Jsonb(payload)),
            )
            if idempotency_key:
                db.execute(
                    "INSERT INTO review_requests VALUES(%s,%s,%s,%s)",
                    (self.namespace, idempotency_key, digest, Jsonb(payload)),
                )
            db.execute(
                "INSERT INTO audit(project_id,actor,action,resource,at) VALUES(%s,%s,'review.saved',%s,extract(epoch from now()))",
                (self.user["project_id"], self.user["id"], body["review_id"]),
            )
        return payload

    def export(self):
        return {
            "version": "general-review-packet-v1",
            "queue_sha256": self.queue_sha,
            "annotations": list(self.annotations().values()),
            "automatic_training_labels_created": 0,
            "policy": "Authenticated independent window responses; physical causes remain unknown",
        }
