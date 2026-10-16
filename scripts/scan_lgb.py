"""Scan raw records with the context LightGBM, only if planted search chose it.

The witness rule lives in fiber.classify.scan and is not adjusted here.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fiber.classify.context import CONTEXT_PEAK_FRACTION, CONTEXT_SIDE, context_table
from fiber.classify.lgb_context import load_context_lgb, predict_context_lgb
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


def _counts(names: list[str]) -> dict[str, int]:
    return {label: sum(name == label for name in names) for label in LABELS}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "artifacts" / "classify" / "lgb_report.json")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "classify" / "lgb_scan.json")
    args = parser.parse_args()
    if not args.report.is_file():
        raise SystemExit(f"{args.report} is missing; train before scanning records")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if report.get("choice") != "lgb":
        raise SystemExit(f"choice is {report.get('choice')}; real traces stay unscanned")
    weights = Path(report["chosen_weights"])
    booster = load_context_lgb(weights)
    print(f"weights {weights} choice {report['choice']}", flush=True)

    witness_payload = json.loads((ROOT / "fiber" / "benchmark" / "witnesses.json").read_text(encoding="utf-8"))
    by_record: dict[str, list[dict]] = {}
    for spec in witness_payload["witnesses"]:
        by_record.setdefault(spec["record"], []).append(spec)

    records = ROOT / "artifacts" / "records"
    record_rows = {}
    witness_rows = []
    for name in ("1", "2", "3", "4", "5", "6"):
        directory = records / name
        if not (directory / "raw_c.npy").is_file():
            raise SystemExit(f"missing record {directory}")
        record = Record.load(directory)
        for spec in by_record.get(name, []):
            if record.source_sha256 != spec["source_sha256"]:
                raise SystemExit(f"{name} sha changed; witnesses do not match this record")
        started = time.perf_counter()
        origins = window_origins(record.raw_c.shape[0], record.raw_c.shape[1])
        kept = scored_origins(record.valid)
        names = predict_context_lgb(booster, context_table(record.raw_c, record.valid, kept)) if len(kept) else []
        elapsed = time.perf_counter() - started
        record_rows[name] = {
            "n_origins": int(len(origins)),
            "n_scored": int(len(kept)),
            "counts": _counts(names),
            "elapsed_s": round(elapsed, 2),
            "source_sha256": record.source_sha256,
        }
        print(f"record {name} scored {len(kept)} {record_rows[name]['counts']} {elapsed:.1f}s", flush=True)
        for spec in by_record.get(name, []):
            hit = overlapping_indices(kept, record.x_m, spec["rows"][0], spec["rows"][1], spec["x_m"][0], spec["x_m"][1])
            summary = summarize_witness(spec["role"], [names[int(index)] for index in hit])
            summary.update({
                "id": spec["id"],
                "record": name,
                "x_m": spec["x_m"],
                "rows": spec["rows"],
            })
            witness_rows.append(summary)
            print(f"  {spec['id']} pass={summary['pass']} n={summary['n_windows']} hot={summary['local_heating_fraction']}", flush=True)

    payload = {
        "weights": str(weights),
        "choice": report["choice"],
        "objective": "planted_context_search",
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
            "n_estimators": report.get("n_estimators"),
            "num_leaves": report.get("num_leaves"),
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


if __name__ == "__main__":
    main()
