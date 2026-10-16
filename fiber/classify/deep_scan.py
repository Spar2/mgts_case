"""Scan raw records with one deep classifier, only if planted search chose it.

The witness rule lives in fiber.classify.scan and is not adjusted here.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from fiber.classify.context import CONTEXT_PEAK_FRACTION, CONTEXT_SIDE, wide_context_crops
from fiber.classify.deep_search import KINDS, _predict
from fiber.classify.rules import LABELS
from fiber.classify.scan import (
    HOT_PASS_FRACTION,
    MIN_VALID_FRACTION,
    SPACE_STRIDE,
    TIME_STRIDE,
    overlapping_indices,
    scored_origins,
    summarize_witness,
    window_origins,
)
from fiber.io import Record
from fiber.models.patch_classifier import SPACE_CROP, TIME_CROP

ROOT = Path(__file__).resolve().parents[2]


def _counts(names: list[str]) -> dict[str, int]:
    return {label: sum(name == label for name in names) for label in LABELS}


def main_scan(kind: str) -> None:
    import argparse

    import torch

    spec = KINDS[kind]
    parser = argparse.ArgumentParser(description=f"Scan raw records with the {kind} model if it was chosen.")
    parser.add_argument("--report", type=Path, default=ROOT / "artifacts" / "classify" / spec["report"])
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "classify" / spec["report"].replace("_report.json", "_scan.json"))
    parser.add_argument("--batch", type=int, default=256)
    args = parser.parse_args()
    if not args.report.is_file():
        raise SystemExit(f"{args.report} is missing; train before scanning records")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if report.get("choice") != kind:
        raise SystemExit(f"choice is {report.get('choice')}; real traces stay unscanned")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _blob, device = spec["loader"](Path(report["chosen_weights"]), device)
    print(f"weights {report['chosen_weights']} choice {report['choice']}", flush=True)
    witness_payload = json.loads((ROOT / "fiber" / "benchmark" / "witnesses.json").read_text(encoding="utf-8"))
    by_record: dict[str, list[dict]] = {}
    for row in witness_payload["witnesses"]:
        by_record.setdefault(row["record"], []).append(row)
    records = ROOT / "artifacts" / "records"
    record_rows = {}
    witness_rows = []
    for name in ("1", "2", "3", "4", "5", "6"):
        directory = records / name
        if not (directory / "raw_c.npy").is_file():
            raise SystemExit(f"missing record {directory}")
        record = Record.load(directory)
        for row in by_record.get(name, []):
            if record.source_sha256 != row["source_sha256"]:
                raise SystemExit(f"{name} sha changed; witnesses do not match this record")
        started = time.perf_counter()
        origins = window_origins(record.raw_c.shape[0], record.raw_c.shape[1])
        kept = scored_origins(record.valid)
        crops = wide_context_crops(record.raw_c, record.valid, kept) if len(kept) else None
        names = _predict(model, crops, device, args.batch) if crops is not None and len(crops) else []
        elapsed = time.perf_counter() - started
        record_rows[name] = {
            "n_origins": int(len(origins)),
            "n_scored": int(len(kept)),
            "counts": _counts(names),
            "elapsed_s": round(elapsed, 2),
            "source_sha256": record.source_sha256,
        }
        print(f"record {name} scored {len(kept)} {record_rows[name]['counts']} {elapsed:.1f}s", flush=True)
        for row in by_record.get(name, []):
            hit = overlapping_indices(kept, record.x_m, row["rows"][0], row["rows"][1], row["x_m"][0], row["x_m"][1])
            summary = summarize_witness(row["role"], [names[int(index)] for index in hit])
            summary.update({"id": row["id"], "record": name, "x_m": row["x_m"], "rows": row["rows"]})
            witness_rows.append(summary)
            print(
                f"  {row['id']} pass={summary['pass']} n={summary['n_windows']} hot={summary['local_heating_fraction']}",
                flush=True,
            )
    payload = {
        "weights": str(report["chosen_weights"]),
        "choice": report["choice"],
        "objective": spec["objective"],
        "serve": "scorecard",
        "field": "raw",
        "protocol": {
            "time_crop": TIME_CROP,
            "space_crop": SPACE_CROP,
            "time_stride": TIME_STRIDE,
            "space_stride": SPACE_STRIDE,
            "min_valid_fraction": MIN_VALID_FRACTION,
            "positive_hot_min_fraction": HOT_PASS_FRACTION,
            "context_side": CONTEXT_SIDE,
            "peak_fraction": CONTEXT_PEAK_FRACTION,
            "argmax": True,
            "expert_confirmed": False,
        },
        "records": record_rows,
        "witnesses": witness_rows,
        "n_pass": int(sum(row["pass"] for row in witness_rows)),
        "n_witnesses": len(witness_rows),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", args.out, flush=True)
    print(f"witnesses {payload['n_pass']}/{payload['n_witnesses']}", flush=True)
