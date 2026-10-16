"""Immutable approved native window with precomputed real median API output."""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services/web-api"))
from workbench.store import Queue, Store


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--ml-url", default="http://127.0.0.1:8082")
    args = parser.parse_args()
    if args.target.exists():
        raise ValueError("Immutable target already exists")
    store = Store(
        args.records,
        Path(__file__).resolve().parents[1] / ".runtime/public-prepare-cache",
    )
    queue = Queue(args.queue, store, max_cells=1_000_000)
    card, data = queue.load("review-00018")
    if (
        card["source"] != "5.csv"
        or card["row_bounds"] != [1419, 1470]
        or card["column_bounds"] != [798, 884]
    ):
        raise ValueError("Published window changed; review required")
    raw, mask = data["raw_c"], data["invalid_reason"]
    values = [
        [float(v) if reason == 0 else None for v, reason in zip(row, reasons)]
        for row, reasons in zip(raw, mask)
    ]
    processed = []
    with httpx.Client(timeout=30) as client:
        for row, reasons in zip(raw, mask):
            response = client.post(
                args.ml_url.rstrip("/") + "/denoise/profile",
                json={
                    "method": "median_x3",
                    "raw": [
                        float(v) if r == 0 else -300.0 for v, r in zip(row, reasons)
                    ],
                    "distances": data["x_m"].tolist(),
                },
            )
            response.raise_for_status()
            output = response.json()["denoised"]
            if len(output) != len(row) or any(
                v is None or not np.isfinite(v)
                for v, r in zip(output, reasons)
                if r == 0
            ):
                raise ValueError("Invalid ML output")
            processed.append(
                [float(v) if r == 0 else None for v, r in zip(output, reasons)]
            )
    record = store.get("5")
    start, end = card["row_bounds"]
    payload = {
        "schema_version": "dts-public-demo-v1",
        "candidate": "review-00018",
        "source": "5.csv",
        "source_sha256": card["evidence"]["source_sha256"],
        "queue_sha256": queue.sha,
        "row_bounds": card["row_bounds"],
        "column_bounds": card["column_bounds"],
        "x_m": data["x_m"].tolist(),
        "time_s": data["time_s"].tolist(),
        "timestamps": [p["timestamp"] for p in record.profiles[start:end]],
        "timezone": None,
        "segment_id": record.segment_id[start:end].tolist(),
        "raw_c": values,
        "invalid_reason": mask.tolist(),
        "processed_c": processed,
        "provenance": {
            "method": "median_x3",
            "method_version": "ML API contract 0.1; implementation SHA not supplied",
            "training_origin": "untrained",
            "scope": "Published row only; no observations outside the window used",
            "calculated_at": datetime.now(timezone.utc).isoformat(),
            "limitations": "Denoising is not event classification or an expert label",
        },
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode()
    args.target.mkdir(parents=True)
    (args.target / "package.json").write_bytes(encoded)
    (args.target / "manifest.json").write_text(
        json.dumps(
            {"sha256": hashlib.sha256(encoded).hexdigest(), "bytes": len(encoded)}
        )
    )
    print(
        f"Published {raw.shape[0]} x {raw.shape[1]} native cells; {len(encoded)} bytes"
    )


if __name__ == "__main__":
    main()
