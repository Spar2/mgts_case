"""Contract fixtures only: never model training or real expert material."""

import argparse
import csv
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services/web-api"))
from workbench.queue_builder import build_queue
from workbench.vendor.data import ingest, sha256
from workbench.vendor.reviews import ReviewJournal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(".runtime/ci-smoke"))
    root = parser.parse_args().output.resolve()
    workspace = Path(__file__).resolve().parents[1]
    if not root.is_relative_to(workspace / ".runtime") or root.exists():
        parser.error("Choose a new folder inside this repository .runtime")
    sources = root / "csv"
    sources.mkdir(parents=True)
    for i in range(1, 7):
        path = sources / f"{i}.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["dts_id", "channel_id", "timestamp", "points"]
            )
            writer.writeheader()
            for row in range(121):
                seconds = row * 10 + (3600 if i == 3 and row >= 60 else 0)
                points = [[x, 20 + row % 11 * 0.1 + x * 0.05] for x in (0, 1, 4, 11)]
                if row == 5:
                    points[1][1] = -300
                writer.writerow(
                    {
                        "dts_id": f"contract-fixture-{i}",
                        "channel_id": "1",
                        "timestamp": (
                            datetime(2025, 1, 1) + timedelta(seconds=seconds)  # noqa: DTZ001 — intentionally unknown timezone contract
                        ).strftime("%Y-%m-%d %H:%M:%S"),
                        "points": json.dumps(points),
                    }
                )
        ingest(path, root / "records")
    queue = build_queue(root / "records", root / "queue")
    digest = sha256(queue / "queue.json")
    journal = ReviewJournal(
        queue / "queue.json", root / "runtime/reviews" / f"{digest}.sqlite3"
    )
    card = next(iter(journal.cards))
    journal.save(
        {
            "queue_sha256": digest,
            "review_id": card,
            "expected_revision": 0,
            "reviewer": "QA ONLY — отдельный тестовый журнал",
            "reviewer_type": "expert",
            "temperature_episode": "unknown",
            "phenomenon": "unknown",
            "notes": "Technical CI marker for fictitious contract data. Never a real-data expert assessment or training label.",
        },
        idempotency_key="ci-marker",
    )
    print(root)


if __name__ == "__main__":
    main()
