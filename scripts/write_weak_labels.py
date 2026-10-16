"""Write weak labels for the six real records.

Crops come from operational spatial_z zones, shape-rule votes, and quiet
28x40 windows. Holdout, guard, and mixed time splits are not written.
Fifteen Zakhar cards are high-weight seeds. A seed whose window is
holdout, guard, or mixed is not written. This does not train a model
and does not change analyze_field.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fiber.classify.rules import LABELS
from fiber.io import Record
from fiber.real_data import RULE_CONFIG
from fiber.weak_labels import (
    AUTOMATIC_WEIGHT,
    EXPERT_SEEDS,
    EXPERT_WEIGHT,
    crops_from_record,
    crops_to_weak_rows,
    summarize_weak_rows,
)

JSONL_PATH = ROOT / "docs" / "evidence" / "weak_labels.jsonl"
SUMMARY_PATH = ROOT / "docs" / "evidence" / "weak_labels_summary.json"


def main():
    rows = []
    offered_seeds = []
    record_splits = {}
    for name in map(str, range(1, 7)):
        directory = ROOT / "artifacts" / "records" / name
        if not directory.is_dir():
            raise SystemExit(f"missing record {directory}")
        print(f"Record {name}: load and detect", flush=True)
        record = Record.load(directory)
        crops, info = crops_from_record(record)
        offered_seeds.extend(crop["id"] for crop in crops if crop["origin"] == "expert_seed")
        kept = crops_to_weak_rows(crops)
        rows.extend(kept)
        record_splits[name] = {
            "cut_time_s": info["cut_time_s"],
            "guard_s": info["guard_s"],
            "counts": info["counts"],
            "codes": info["codes"],
        }
        origins = {}
        for row in kept:
            origins[row["origin"]] = origins.get(row["origin"], 0) + 1
        print(
            f"Record {name} finished: offered {len(crops)} kept {len(kept)} origins {origins}",
            flush=True,
        )
    summary = summarize_weak_rows(rows)
    counts_by_file = {name: int(summary["counts_by_file"].get(name, 0)) for name in map(str, range(1, 7))}
    counts_by_class = {name: int(summary["counts_by_class"].get(name, 0)) for name in LABELS}
    if sum(counts_by_file.values()) != summary["n_rows"] or sum(counts_by_class.values()) != summary["n_rows"]:
        raise SystemExit("file and class counts do not add up to the jsonl rows")
    if summary["holdout_used_for_generation"] or summary["guard_used_for_generation"]:
        raise SystemExit("holdout or guard row survived")
    if any(row["split_code"] not in (0, 1) for row in rows):
        raise SystemExit("split_code outside train/development")
    emitted_seeds = [row["id"] for row in rows if row["origin"] == "expert_seed"]
    excluded = [seed_id for seed_id in offered_seeds if seed_id not in emitted_seeds]
    present = [seed["id"] for seed in EXPERT_SEEDS if seed["id"] in emitted_seeds]
    automatic = [row for row in rows if row["origin"] != "expert_seed"]
    if automatic and any(row["weight"] != AUTOMATIC_WEIGHT or row["expert_confirmed"] for row in automatic):
        raise SystemExit("automatic row has expert weight or expert_confirmed")
    if any(row["weight"] != EXPERT_WEIGHT or not row["expert_confirmed"] for row in rows if row["id"] in present):
        raise SystemExit("expert seed lost its weight or confirmation")
    if automatic and present and min(row["weight"] for row in rows if row["id"] in present) <= max(
        row["weight"] for row in automatic
    ):
        raise SystemExit("expert seed is not strictly heavier than automatic rows")
    body = {
        "schema": "weak_labels_v1",
        "jsonl": "docs/evidence/weak_labels.jsonl",
        "n_rows": summary["n_rows"],
        "counts_by_file": counts_by_file,
        "counts_by_class": counts_by_class,
        "counts_by_origin": {key: int(summary["counts_by_origin"].get(key, 0)) for key in (
            "spatial_z", "rule", "quiet", "expert_seed")},
        "split_codes": summary["split_codes"],
        "holdout_used_for_generation": False,
        "guard_used_for_generation": False,
        "holdout_and_guard_were_not_used_for_generation": True,
        "mixed_splits_dropped": True,
        "splits_used": [
            {"split_code": 0, "name": "train"},
            {"split_code": 1, "name": "development"},
        ],
        "splits_not_used": [
            {"split_code": 2, "name": "internal_holdout"},
            {"split_code": -1, "name": "excluded_guard_or_mixed"},
        ],
        "split_definition": {
            "function": "fiber.real_data.temporal_partition",
            "crop_function": "fiber.real_data.crop_partition",
            "guard_s": RULE_CONFIG["split_guard_s"],
            "train_index_fraction": RULE_CONFIG["split_train_fraction"],
            "development_index_fraction": RULE_CONFIG["split_dev_fraction"],
        },
        "record_splits": record_splits,
        "automatic_weight": AUTOMATIC_WEIGHT,
        "expert_weight": EXPERT_WEIGHT,
        "expert_seed_ids": present,
        "expert_seeds_excluded": excluded,
        "expert_confirmed_only_on_seeds": True,
        "temporal_innovation_used_as_source": False,
        "operational_status_unchanged": True,
        "note": (
            "Weak labels are training seeds, not expert truth and not a reason "
            "to adopt a new detector. analyze_field stays median_x3 + spatial_z_halo at |Z| >= 3.5."
        ),
    }
    JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with JSONL_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    SUMMARY_PATH.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {JSONL_PATH} rows {body['n_rows']} "
        f"holdout_used_for_generation {body['holdout_used_for_generation']} "
        f"guard_used_for_generation {body['guard_used_for_generation']}",
        flush=True,
    )
    print(f"counts_by_file {counts_by_file}", flush=True)
    print(f"counts_by_class {counts_by_class}", flush=True)
    print(f"counts_by_origin {body['counts_by_origin']}", flush=True)
    print(f"expert_seed_ids {present} excluded {excluded}", flush=True)


if __name__ == "__main__":
    main()
