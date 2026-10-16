"""Verify real native sources and select two reproducible observation contexts.

Reads sources/queue only. Never trains, labels events or substitutes a method result.
The output contains metadata and aggregate checks, not temperature matrices.
"""

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import urlencode

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "services/web-api"))
from workbench.store import Queue, Store


def inspect(records, queue_path):
    store = Store(records, BASE / ".runtime/defense-cache")
    if set(store.records) != {"1", "2", "3", "4", "5", "6"}:
        raise ValueError("Defense archive must contain all six expected sources")
    sources = []
    for key, record in sorted(store.records.items()):
        sampled = 0
        for row in {0, record.raw_c.shape[0] // 2, record.raw_c.shape[0] - 1}:
            native = store.slices(key, row, 0)
            for column, value in enumerate(native["profile"]):
                if record.invalid_reason[row, column] != 0:
                    if value is not None:
                        raise ValueError("Masked cell became a temperature")
                elif value != float(record.raw_c[row, column]):
                    raise ValueError("Numeric slice differs from native precision")
                sampled += 1
        codes, counts = np.unique(record.invalid_reason, return_counts=True)
        sources.append(
            {
                "record_id": key,
                "source_sha256": record.manifest["source_sha256"],
                "shape": list(record.raw_c.shape),
                "time_kind": record.manifest["time_kind"],
                "mask_counts": {str(int(k)): int(v) for k, v in zip(codes, counts)},
                "sampled_cells_exact": sampled,
            }
        )
    queue = Queue(queue_path, store)
    cards = list(queue.cards.values())
    primary = next(
        (
            c
            for c in cards
            if c["source"] == "5.csv" and c["row_bounds"][0] <= 536 < c["row_bounds"][1]
        ),
        next((c for c in cards if c["source"] == "5.csv"), cards[0]),
    )
    backup = next((c for c in cards if c["source"] != primary["source"]), None)
    if not backup:
        raise ValueError("Need a second source for the backup case")
    cases = []
    for role, card in [("primary", primary), ("backup", backup)]:
        _, data = queue.load(card["review_id"])
        r0, r1 = card["row_bounds"]
        c0, c1 = card["column_bounds"]
        key = card["source"].removesuffix(".csv")
        native = store.get(key)
        if not np.array_equal(
            data["raw_c"], native.raw_c[r0:r1, c0:c1], equal_nan=True
        ) or not np.array_equal(
            data["invalid_reason"], native.invalid_reason[r0:r1, c0:c1]
        ):
            raise ValueError("Case values or masks differ from the native source")
        if not np.array_equal(data["x_m"], native.x_m[c0:c1]) or not np.array_equal(
            data["time_s"], native.elapsed_s[r0:r1]
        ):
            raise ValueError("Case coordinates differ from the native source")
        query = {
            "record": key,
            "row": r0,
            "column": c0,
            "r0": r0,
            "r1": r1,
            "c0": c0,
            "c1": c1,
            "source_sha256": card["evidence"]["source_sha256"],
        }
        cases.append(
            {
                "role": role,
                "review_id": card["review_id"],
                "source_sha256": card["evidence"]["source_sha256"],
                "region": {
                    "record_id": key,
                    "row_start": r0,
                    "row_end": r1,
                    "column_start": c0,
                    "column_end": c1,
                },
                "research_path": "/research?" + urlencode(query),
                "review_path": "/review?package=general&card=" + card["review_id"],
                "valid_fraction": float(np.mean(data["invalid_reason"] == 0)),
                "interpretation": "Observed signal; physical cause and expert accuracy not established",
                "selected_run_id": None,
            }
        )
    return {
        "schema": "defense_readiness_v1",
        "sources": sources,
        "queue_sha256": queue.sha,
        "verified_queue_cards": len(cards),
        "cases": cases,
        "final_team_result": "not supplied; do not substitute median processing or research weights",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--queue", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error("Use a new output path; do not overwrite evidence")
    result = inspect(a.records, a.queue)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"Verified {len(result['sources'])} sources, {result['verified_queue_cards']} queue cards and two contexts. No model quality claim."
    )


if __name__ == "__main__":
    main()
