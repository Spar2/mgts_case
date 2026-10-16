"""Scan raw records with the agree-model, only if planted search chose it.

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

from fiber.classify.agree_model import agree_table, load_agree_model
from fiber.classify.context import CONTEXT_PEAK_FRACTION, CONTEXT_SIDE
from fiber.classify.lgb_context import predict_context_lgb
from fiber.classify.rules import LABELS
from fiber.classify.scan import (
    HOT_PASS_FRACTION,
    MIN_VALID_FRACTION,
    SPACE_STRIDE,
    TIME_STRIDE,
    decide_witnesses,
    scored_origins,
    window_origins,
)
from fiber.io import Record
from fiber.models.patch_classifier import SPACE_CROP, TIME_CROP


def _counts(names: list[str]) -> dict[str, int]:
    return {label: sum(name == label for name in names) for label in LABELS}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "artifacts" / "classify" / "agree_report.json")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "classify" / "agree_scan.json")
    args = parser.parse_args()
    if not args.report.is_file():
        raise SystemExit(f"{args.report} is missing; train before scanning records")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if report.get("choice") != "agree":
        raise SystemExit(f"choice is {report.get('choice')}; real traces stay unscanned")
    weights = Path(report["chosen_weights"])
    booster = load_agree_model(weights)
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
        names = predict_context_lgb(booster, agree_table(record.raw_c, record.valid, kept)) if len(kept) else []
        elapsed = time.perf_counter() - started
        record_rows[name] = {
            "n_origins": int(len(origins)),
            "n_scored": int(len(kept)),
            "counts": _counts(names),
            "elapsed_s": round(elapsed, 2),
            "source_sha256": record.source_sha256,
        }
        print(f"record {name} scored {len(kept)} {record_rows[name]['counts']} {elapsed:.1f}s", flush=True)
        for summary in decide_witnesses(by_record.get(name, []), kept, record.x_m, names, name):
            witness_rows.append(summary)
            print(f"  {summary['id']} pass={summary['pass']} n={summary['n_windows']} hot={summary['local_heating_fraction']}", flush=True)

    payload = {
        "weights": str(weights),
        "choice": report["choice"],
        "objective": "planted_window_agreement",
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
