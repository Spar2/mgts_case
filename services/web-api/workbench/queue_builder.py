"""Reproducible coverage windows for a fresh clone, never generated labels.

Prefer an existing verified general review queue to retain its prepared methods.
This fallback deliberately includes no method outputs or inferred answers.
"""

import json
import os
import uuid
from pathlib import Path

import numpy as np

from .vendor.data import load_record, sha256


def build_queue(records, output):
    records, output = Path(records), Path(output)
    if output.exists():
        raise ValueError("Queue is immutable: choose a new output directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.with_name(".building-queue-" + uuid.uuid4().hex)
    stage.mkdir()
    cards, sources = [], []
    seen = set()
    for manifest in sorted(records.glob("*/manifest.json")):
        r = load_record(manifest.parent)
        m = r.manifest
        sources.append(
            {
                "source": m["source_name"],
                "source_sha256": m["source_sha256"],
                "time_kind": m["time_kind"],
            }
        )
        for tf in (0.2, 0.5, 0.8):
            center = int((len(r.elapsed_s) - 1) * tf)
            r0, r1 = max(0, center - 60), min(len(r.elapsed_s), center + 60)
            for xf in (0.2, 0.5, 0.8):
                x = r.x_m[0] + (r.x_m[-1] - r.x_m[0]) * xf
                column = int(np.searchsorted(r.x_m, x))
                c0, c1 = max(0, column - 30), min(len(r.x_m), column + 31)
                key = f"{r.record_id}-r{r0}-{r1}-x{c0}-{c1}"
                if key in seen:
                    continue
                seen.add(key)
                shape = (r1 - r0, c1 - c0)
                path = stage / (key + ".npz")
                np.savez_compressed(
                    path,
                    raw_c=r.raw_c[r0:r1, c0:c1],
                    invalid_reason=r.invalid_reason[r0:r1, c0:c1],
                    time_s=r.elapsed_s[r0:r1],
                    x_m=r.x_m[c0:c1],
                    signed=np.zeros((0, *shape), np.int8),
                    binary=np.zeros((0, *shape), bool),
                )
                cards.append(
                    {
                        "review_id": key,
                        "source": m["source_name"],
                        "record_id": r.record_id,
                        "row_bounds": [r0, r1],
                        "column_bounds": [c0, c1],
                        "native_shape": list(shape),
                        "x_bounds_m": [float(r.x_m[c0]), float(r.x_m[c1 - 1])],
                        "time_bounds_s": [
                            float(r.elapsed_s[r0]),
                            float(r.elapsed_s[r1 - 1]),
                        ],
                        "timestamps": [p["timestamp"] for p in r.profiles[r0:r1]],
                        "methods": [],
                        "artifact": path.name,
                        "artifact_sha256": sha256(path),
                        "evidence": {
                            "source_sha256": m["source_sha256"],
                            "selection": "fixed native coverage; not event labels",
                        },
                        "response": None,
                        "training_target": None,
                    }
                )
    if not cards:
        raise ValueError("No prepared records found")
    queue = {
        "version": "web-native-review-v1",
        "cards": cards,
        "records": sources,
        "allowed_answers": {
            "temperature_episode": ["present", "absent", "ambiguous", "unknown"],
            "phenomenon": [
                "local_change",
                "common_trend",
                "dropout",
                "mixed",
                "unknown",
            ],
        },
        "review_policy": "Coverage windows only; no method outputs, physical causes or automatic training targets.",
    }
    (stage / "queue.json").write_text(
        json.dumps(queue, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (stage / "manifest.json").write_text(
        json.dumps({"cards": len(cards), "queue_sha256": sha256(stage / "queue.json")}),
        encoding="utf-8",
    )
    os.replace(stage, output)
    return output
