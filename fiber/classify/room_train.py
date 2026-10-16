"""Train one logistic model on room fields and score the eight witnesses once.

The recipe constants live in `fiber.classify.room` and are not arguments.
Weights are written only when the witness pass count is strictly above the
published context-logistic count of 7/8.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fiber.classify.context import context_table, dump_context_model
from fiber.classify.plant import LABEL_INDEX, SoftmaxSceneModel, plant_scene
from fiber.classify.room import (
    ROOM_PER_EVENT,
    ROOM_STEPS,
    ROOM_TRAIN_SEEDS,
    field_summary,
    plant_room,
    records_target,
    summary_gaps,
)
from fiber.classify.rules import LABELS
from fiber.classify.scan import (
    HOT_PASS_FRACTION,
    labeled_windows,
    overlapping_indices,
    scored_origins,
    summarize_witness,
)
from fiber.io import Record

ROOT = Path(__file__).resolve().parents[2]
PUBLISHED_WITNESS_PASSES = 7


def _balanced() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    buckets = {name: [] for name in LABELS}
    for seed in ROOM_TRAIN_SEEDS:
        scene = plant_room(int(seed))
        origins, truth = _windows(scene)
        feats = context_table(scene["field"], scene["valid"], origins)
        for row, name in zip(feats, truth, strict=True):
            buckets[name].append(row)
    counts = {name: len(rows) for name, rows in buckets.items()}
    cap = min(counts.values())
    if cap < ROOM_PER_EVENT:
        raise RuntimeError(f"room class buckets did not fill: {counts}")
    chosen_x = []
    chosen_y = []
    for name in LABELS:
        take = rng.choice(counts[name], size=ROOM_PER_EVENT, replace=False)
        for index in take:
            chosen_x.append(buckets[name][int(index)])
            chosen_y.append(LABEL_INDEX[name])
    return np.stack(chosen_x), np.asarray(chosen_y, dtype=np.int64)


def _windows(scene: dict):
    return labeled_windows(scene["labels"], scene["valid"])


def _score_witnesses(model: SoftmaxSceneModel) -> list[dict]:
    payload = json.loads((ROOT / "fiber" / "benchmark" / "witnesses.json").read_text(encoding="utf-8"))
    by_record: dict[str, list[dict]] = {}
    for spec in payload["witnesses"]:
        by_record.setdefault(spec["record"], []).append(spec)
    rows = []
    records = ROOT / "artifacts" / "records"
    for name in ("1", "2", "3", "4", "5", "6"):
        directory = records / name
        if not (directory / "raw_c.npy").is_file():
            raise SystemExit(f"missing record {directory}")
        record = Record.load(directory)
        for spec in by_record.get(name, []):
            if record.source_sha256 != spec["source_sha256"]:
                raise SystemExit(f"{name} sha changed; witnesses do not match this record")
        kept = scored_origins(record.valid)
        names = model.predict(context_table(record.raw_c, record.valid, kept)) if len(kept) else []
        print(f"record {name} scored {len(kept)}", flush=True)
        for spec in by_record.get(name, []):
            hit = overlapping_indices(
                kept, record.x_m, spec["rows"][0], spec["rows"][1], spec["x_m"][0], spec["x_m"][1],
            )
            summary = summarize_witness(spec["role"], [names[int(index)] for index in hit])
            summary.update({"id": spec["id"], "record": name})
            rows.append(summary)
            print(
                f"  {spec['id']} pass={summary['pass']} n={summary['n_windows']} "
                f"hot={summary['local_heating_fraction']}",
                flush=True,
            )
    return rows


def _similarity() -> dict:
    records = [Record.load(ROOT / "artifacts" / "records" / name) for name in "123456"]
    target = records_target(records)
    room = plant_room(ROOM_TRAIN_SEEDS[0])
    flat = plant_scene("background", seed=0)
    room_gap = summary_gaps(field_summary(room["field"], room["valid"]), target)
    flat_gap = summary_gaps(field_summary(flat["field"], flat["valid"]), target)
    return {
        "target": target,
        "room_gap": room_gap,
        "flat_gap": flat_gap,
        "room_summary": field_summary(room["field"], room["valid"]),
        "flat_summary": field_summary(flat["field"], flat["valid"]),
    }


def main() -> None:
    features, target = _balanced()
    model = SoftmaxSceneModel()
    model.fit(features, target, steps=ROOM_STEPS)
    similarity = _similarity()
    witnesses = _score_witnesses(model)
    if len(witnesses) != 8:
        raise SystemExit("witness protocol did not return eight windows")
    n_pass = int(sum(bool(row["pass"]) for row in witnesses))
    adopted = n_pass > PUBLISHED_WITNESS_PASSES
    weights = ROOT / "models" / "weights" / "room_context.json"
    report = {
        "adopted": adopted,
        "witness_pass_count": n_pass,
        "n_witnesses": 8,
        "published_context_logistic_passes": PUBLISHED_WITNESS_PASSES,
        "expert_confirmed": False,
        "positive_hot_min_fraction": HOT_PASS_FRACTION,
        "serve": "scorecard",
        "decides": "none",
        "objective": "room_witness",
        "train_seeds": list(ROOM_TRAIN_SEEDS),
        "steps": ROOM_STEPS,
        "per_event": ROOM_PER_EVENT,
        "similarity": similarity,
        "witnesses": witnesses,
        "weights": str(weights) if adopted else None,
        "note": "Witness pass count is not cable precision and not expert accuracy.",
    }
    if adopted:
        weights.parent.mkdir(parents=True, exist_ok=True)
        blob = dump_context_model(model, {
            "serve": "scorecard",
            "objective": "room_witness",
            "steps": ROOM_STEPS,
            "per_event": ROOM_PER_EVENT,
            "train_seeds": list(ROOM_TRAIN_SEEDS),
            "witness_pass_count": n_pass,
        })
        weights.write_text(json.dumps(blob, ensure_ascii=False, indent=2), encoding="utf-8")
        print("adopted", weights, flush=True)
    else:
        print("the run was not adopted", flush=True)
    dest = ROOT / "artifacts" / "classify" / "room_report.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"witnesses {n_pass}/8", flush=True)
    print("saved", dest, flush=True)
