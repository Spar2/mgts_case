"""Train one logistic model on the warm tail and score the witnesses once.

Constants live in `fiber.classify.bulk` and are not arguments. Off the warm
tail every window is Background. Weights are written only when the witness
pass count is 8/8.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fiber.classify.bulk import (
    BULK_AC1_SPLIT,
    BULK_DECOY_PEAK,
    BULK_FEATURES,
    BULK_PER_EVENT,
    BULK_STABLE_MAX,
    BULK_STEPS,
    BULK_TRAIN_SEEDS,
    bulk_names,
    bulk_table,
    dump_bulk_model,
    load_bulk_model,
    plant_bulk,
    warm_tail_mask,
)
from fiber.classify.plant import LABEL_INDEX, SoftmaxSceneModel
from fiber.classify.rules import LABELS
from fiber.classify.scan import HOT_PASS_FRACTION, decide_witnesses, labeled_windows, scored_origins
from fiber.io import Record

ROOT = Path(__file__).resolve().parents[2]
PUBLISHED_WITNESS_PASSES = 7


def _balanced() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    buckets = {name: [] for name in LABELS}
    for seed in BULK_TRAIN_SEEDS:
        scene = plant_bulk(int(seed))
        origins, truth = labeled_windows(scene["labels"], scene["valid"])
        feats = bulk_table(scene["field"], scene["valid"], origins)
        for row, name in zip(feats, truth, strict=True):
            buckets[name].append(row)
    counts = {name: len(rows) for name, rows in buckets.items()}
    cap = min(len(rows) for name, rows in buckets.items() if name != "Background")
    peak_index = BULK_FEATURES.index("peak")
    ac_index = BULK_FEATURES.index("ac1")
    excess_index = BULK_FEATURES.index("excess")
    stacked = np.stack(buckets["Background"]) if buckets["Background"] else np.zeros((0, len(BULK_FEATURES)))
    warm = warm_tail_mask(stacked) if len(stacked) else np.zeros(0, dtype=bool)
    standing = [
        row for row, flagged in zip(stacked, warm, strict=True)
        if not flagged and abs(float(row[peak_index])) >= BULK_DECOY_PEAK
    ]
    stable = [
        row for row, flagged in zip(stacked, warm, strict=True)
        if (not flagged)
        and float(row[ac_index]) <= BULK_AC1_SPLIT
        and abs(float(row[excess_index])) < BULK_STABLE_MAX
        and abs(float(row[peak_index])) < BULK_DECOY_PEAK
    ]
    if cap < BULK_PER_EVENT or len(standing) < 64 or len(stable) < 64:
        raise RuntimeError(
            f"bulk class buckets did not fill: {counts} standing {len(standing)} stable {len(stable)}"
        )
    chosen_x = []
    chosen_y = []
    for pool in (standing, stable):
        take = rng.choice(len(pool), size=BULK_PER_EVENT // 2, replace=False)
        for index in take:
            chosen_x.append(pool[int(index)])
            chosen_y.append(LABEL_INDEX["Background"])
    for name in LABELS:
        if name == "Background":
            continue
        take = rng.choice(counts[name], size=BULK_PER_EVENT, replace=False)
        for index in take:
            chosen_x.append(buckets[name][int(index)])
            chosen_y.append(LABEL_INDEX[name])
    return np.stack(chosen_x), np.asarray(chosen_y, dtype=np.int64)


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
        names = bulk_names(model, bulk_table(record.raw_c, record.valid, kept)) if len(kept) else []
        print(f"record {name} scored {len(kept)}", flush=True)
        for summary in decide_witnesses(by_record.get(name, []), kept, record.x_m, names, name):
            rows.append(summary)
            print(
                f"  {summary['id']} pass={summary['pass']} n={summary['n_windows']} "
                f"hot={summary['local_heating_fraction']}",
                flush=True,
            )
    return rows


def main() -> None:
    features, target = _balanced()
    model = SoftmaxSceneModel()
    model.fit(features, target, steps=BULK_STEPS)
    weights = ROOT / "models" / "weights" / "bulk_context.json"
    weights.parent.mkdir(parents=True, exist_ok=True)
    blob = dump_bulk_model(model, {
        "serve": "scorecard",
        "objective": "bulk_witness",
        "steps": BULK_STEPS,
        "per_event": BULK_PER_EVENT,
        "train_seeds": list(BULK_TRAIN_SEEDS),
    })
    weights.write_text(json.dumps(blob, ensure_ascii=False, indent=2), encoding="utf-8")
    scored = load_bulk_model(json.loads(weights.read_text(encoding="utf-8")))
    witnesses = _score_witnesses(scored)
    if len(witnesses) != 8:
        raise SystemExit("witness protocol did not return eight windows")
    n_pass = int(sum(bool(row["pass"]) for row in witnesses))
    adopted = n_pass == 8
    report = {
        "adopted": adopted,
        "witness_pass_count": n_pass,
        "n_witnesses": 8,
        "published_context_logistic_passes": PUBLISHED_WITNESS_PASSES,
        "expert_confirmed": False,
        "positive_hot_min_fraction": HOT_PASS_FRACTION,
        "serve": "scorecard",
        "decides": "none",
        "objective": "bulk_witness",
        "train_seeds": list(BULK_TRAIN_SEEDS),
        "steps": BULK_STEPS,
        "per_event": BULK_PER_EVENT,
        "witnesses": witnesses,
        "weights": str(weights) if adopted else None,
        "note": "Witness pass count is not cable precision and not expert accuracy.",
    }
    if adopted:
        blob["witness_pass_count"] = n_pass
        weights.write_text(json.dumps(blob, ensure_ascii=False, indent=2), encoding="utf-8")
        print("adopted", weights, flush=True)
    else:
        weights.unlink(missing_ok=True)
        print("the run was not adopted", flush=True)
    dest = ROOT / "artifacts" / "classify" / "bulk_report.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"witnesses {n_pass}/8", flush=True)
    print("saved", dest, flush=True)
