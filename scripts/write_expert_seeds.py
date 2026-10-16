"""Write expert cards and the rows generation keeps.

expert_card_report builds crops with expert_crops and emits them through
crops_to_weak_rows. A holdout, guard, or mixed window is listed and not
written. Unmapped gap answers are listed and not written. The previous
answers file is left as it is. The automatic weak_labels.jsonl is left as
it is. This does not train a model and does not change analyze_field.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fiber.weak_labels import (
    EXPERT_SEED_BY_ID,
    EXPERT_SEEDS,
    _ASK_DOWNLOAD,
    ask_seed_report,
    crops_to_weak_rows,
    current_expert_ids,
    expert_card_report,
    gap_seed_index,
)

PREVIOUS_ANSWERS_PATH = ROOT / "docs" / "evidence" / "zakhar_next_shortlist_answers.jsonl"
PREVIOUS_DOWNLOAD_PATH = Path(r"C:\Users\bobr\Downloads\zakhar_next_shortlist_answers.jsonl")
GAP_DOWNLOAD_PATH = Path(r"C:\Users\bobr\Downloads\zakhar_next_shortlist_answers (1).jsonl")
GAP_ANSWERS_PATH = ROOT / "docs" / "evidence" / "zakhar_gap_shortlist_answers.jsonl"
JSONL_PATH = ROOT / "docs" / "evidence" / "expert_seeds.jsonl"
SUMMARY_PATH = ROOT / "docs" / "evidence" / "expert_seeds_summary.json"
ASK_ANSWERS_PATH = ROOT / "docs" / "evidence" / "zakhar_ask_shortlist_answers.jsonl"
ASK_JSONL_PATH = ROOT / "docs" / "evidence" / "expert_seeds_ask.jsonl"
ASK_SUMMARY_PATH = ROOT / "docs" / "evidence" / "expert_seeds_ask_summary.json"
PROTECTED_ANSWER_PATHS = (PREVIOUS_ANSWERS_PATH, GAP_ANSWERS_PATH)


def write_ask_seeds(download_path=None):
    """Write weight-10 seeds for the ask-shortlist download.

    The eleven-card answers file and the gap answers file are not written.
    weak_labels.jsonl is not written. A second call emits the same ids.
    """
    source = Path(download_path) if download_path is not None else _ASK_DOWNLOAD
    if not source.is_file():
        raise SystemExit(f"missing answers {source}")
    payload = source.read_bytes()
    before = {path: path.read_bytes() for path in PROTECTED_ANSWER_PATHS}
    ASK_ANSWERS_PATH.write_bytes(payload)
    for path, blob in before.items():
        if path.read_bytes() != blob:
            raise SystemExit(f"protected answers file changed: {path.name}")
    answers = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
    times = {}
    for name in map(str, range(1, 7)):
        directory = ROOT / "artifacts" / "records" / name
        if not directory.is_dir():
            raise SystemExit(f"missing record {directory}")
        times[name] = (
            np.load(directory / "time_s.npy"),
            np.load(directory / "time_iso.npy", allow_pickle=True),
        )
    existing = current_expert_ids()
    rows, report = ask_seed_report(answers, times, existing)
    again, _again_report = ask_seed_report(answers, times, existing)
    if [row["id"] for row in again] != [row["id"] for row in rows]:
        raise SystemExit("second ask pass emitted different ids")
    if report["n_kept"] + report["n_excluded"] + report["n_unclear"] + report["n_empty_shape"] != report["n_answers"]:
        raise SystemExit("ask counts do not add up")
    if report["holdout_used_for_generation"] or report["guard_used_for_generation"]:
        raise SystemExit("holdout or guard ask row was emitted")
    if any(row["split_code"] not in (0, 1) for row in rows):
        raise SystemExit("ask row is outside train/development")
    if len(rows) != report["n_kept"] or [row["id"] for row in rows] != report["generated_ids"]:
        raise SystemExit("ask rows do not match the report")
    ASK_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ASK_JSONL_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    ASK_SUMMARY_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"Wrote {ASK_JSONL_PATH.name} rows {len(rows)} "
        f"answers {report['n_answers']} kept {report['n_kept']} "
        f"excluded {report['n_excluded']} unclear {report['n_unclear']} "
        f"empty_shape {report['n_empty_shape']}",
        flush=True,
    )
    return rows, report


def main():
    if not GAP_DOWNLOAD_PATH.is_file():
        raise SystemExit(f"missing answers {GAP_DOWNLOAD_PATH}")
    previous_bytes = PREVIOUS_ANSWERS_PATH.read_bytes()
    GAP_ANSWERS_PATH.write_bytes(GAP_DOWNLOAD_PATH.read_bytes())
    if PREVIOUS_ANSWERS_PATH.read_bytes() != previous_bytes:
        raise SystemExit("previous answers file was overwritten")
    if PREVIOUS_DOWNLOAD_PATH.is_file() and previous_bytes != PREVIOUS_DOWNLOAD_PATH.read_bytes():
        raise SystemExit("previous answers file no longer matches its download")
    gap_n = sum(1 for line in GAP_ANSWERS_PATH.read_text(encoding="utf-8").splitlines() if line.strip())
    if gap_n != 10:
        raise SystemExit(f"gap answers are not the ten-line file: {gap_n}")
    times = {}
    for name in map(str, range(1, 7)):
        directory = ROOT / "artifacts" / "records" / name
        if not directory.is_dir():
            raise SystemExit(f"missing record {directory}")
        times[name] = (
            np.load(directory / "time_s.npy"),
            np.load(directory / "time_iso.npy", allow_pickle=True),
        )
    rows, report = expert_card_report(times)
    if len(EXPERT_SEEDS) != 15 or report["n_cards"] != len(EXPERT_SEEDS) + gap_n:
        raise SystemExit("expert card count does not cover the previous seeds and the gap answers")
    if [card["id"] for card in report["cards"][:15]] != [seed["id"] for seed in EXPERT_SEEDS]:
        raise SystemExit("previous expert seeds moved")
    if report["n_kept"] + report["n_excluded"] + report["n_unmapped"] != report["n_cards"]:
        raise SystemExit("kept, excluded, and unmapped cards do not add up")
    if report["holdout_used_for_generation"] or report["guard_used_for_generation"]:
        raise SystemExit("holdout or guard row was emitted")
    if any(row["split_code"] not in (0, 1) for row in rows):
        raise SystemExit("generated expert row is outside train/development")
    if len(rows) != report["n_kept"] or [row["id"] for row in rows] != report["generated_ids"]:
        raise SystemExit("emitted rows do not match the report")
    # The same builder the record writer uses must accept these crops.
    index = {**EXPERT_SEED_BY_ID, **gap_seed_index(times)}
    again = crops_to_weak_rows([
        {
            "id": row["id"],
            "origin": "expert_seed",
            "record": row["record"],
            "class_name": row["class_name"],
            "rows_half_open": row["rows_half_open"],
            "x_m": row["x_m"],
            "time_s": row["time_s"],
            "time_iso": row["time_iso"],
            "split_codes": [row["split_code"]],
            "polarity": row["polarity"],
            "provenance": {},
        }
        for row in rows
    ], seed_index=index)
    if [row["id"] for row in again] != [row["id"] for row in rows]:
        raise SystemExit("crops_to_weak_rows dropped a kept expert row")
    JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with JSONL_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    SUMMARY_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {JSONL_PATH.name} rows {len(rows)} "
        f"cards {report['n_cards']} kept {report['n_kept']} "
        f"excluded {report['n_excluded']} unmapped {report['n_unmapped']}",
        flush=True,
    )
    for card in report["cards"]:
        print(
            f"{card['id']} record {card['record']} presence {card['presence']} "
            f"class {card['class_name']} weight {card['weight']} "
            f"split {card['split_code']} {card['status']}",
            flush=True,
        )


if __name__ == "__main__":
    write_ask_seeds()
    main()
