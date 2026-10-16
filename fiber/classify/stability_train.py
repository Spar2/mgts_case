"""Train one logistic model on column MAD and score the witnesses once.

Constants live in `fiber.classify.stability` and are not arguments. The score
uses `decide_witnesses`. Weights are written only when the witness pass
count is strictly above the published context-logistic count of 7/8.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fiber.classify.plant import LABEL_INDEX, SoftmaxSceneModel
from fiber.classify.rules import LABELS
from fiber.classify.scan import HOT_PASS_FRACTION, decide_witnesses, labeled_windows, scored_origins
from fiber.classify.stability import (
    STABILITY_FEATURES,
    STABILITY_MAD_SPLIT,
    STABILITY_PER_EVENT,
    STABILITY_STABLE_MAX,
    STABILITY_STEPS,
    STABILITY_TRAIN_SEEDS,
    dump_stability_model,
    load_stability_model,
    plant_stability,
    stability_table,
)
from fiber.io import Record

ROOT = Path(__file__).resolve().parents[2]
PUBLISHED_WITNESS_PASSES = 7


def _balanced() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    buckets = {name: [] for name in LABELS}
    for seed in STABILITY_TRAIN_SEEDS:
        scene = plant_stability(int(seed))
        origins, truth = labeled_windows(scene["labels"], scene["valid"])
        feats = stability_table(scene["field"], scene["valid"], origins)
        for row, name in zip(feats, truth, strict=True):
            buckets[name].append(row)
    counts = {name: len(rows) for name, rows in buckets.items()}
    cap = min(len(rows) for name, rows in buckets.items() if name != "Background")
    mad_index = STABILITY_FEATURES.index("mad")
    excess_index = STABILITY_FEATURES.index("excess")
    stable = [
        row for row in buckets["Background"]
        if float(row[mad_index]) <= STABILITY_MAD_SPLIT and abs(float(row[excess_index])) < STABILITY_STABLE_MAX
    ]
    ordinary = [
        row for row in buckets["Background"]
        if not (float(row[mad_index]) <= STABILITY_MAD_SPLIT and abs(float(row[excess_index])) < STABILITY_STABLE_MAX)
    ]
    if cap < STABILITY_PER_EVENT or len(stable) < 64 or len(ordinary) < 64:
        raise RuntimeError(
            f"stability class buckets did not fill: {counts} stable {len(stable)} ordinary {len(ordinary)}"
        )
    chosen_x = []
    chosen_y = []
    for pool in (stable, ordinary):
        take = rng.choice(len(pool), size=STABILITY_PER_EVENT // 2, replace=False)
        for index in take:
            chosen_x.append(pool[int(index)])
            chosen_y.append(LABEL_INDEX["Background"])
    for name in LABELS:
        if name == "Background":
            continue
        take = rng.choice(counts[name], size=STABILITY_PER_EVENT, replace=False)
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
        names = model.predict(stability_table(record.raw_c, record.valid, kept)) if len(kept) else []
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
    model.fit(features, target, steps=STABILITY_STEPS)
    weights = ROOT / "models" / "weights" / "stability_context.json"
    weights.parent.mkdir(parents=True, exist_ok=True)
    blob = dump_stability_model(model, {
        "serve": "scorecard",
        "objective": "stability_witness",
        "steps": STABILITY_STEPS,
        "per_event": STABILITY_PER_EVENT,
        "train_seeds": list(STABILITY_TRAIN_SEEDS),
    })
    weights.write_text(json.dumps(blob, ensure_ascii=False, indent=2), encoding="utf-8")
    scored = load_stability_model(json.loads(weights.read_text(encoding="utf-8")))
    witnesses = _score_witnesses(scored)
    if len(witnesses) != 8:
        raise SystemExit("witness protocol did not return eight windows")
    n_pass = int(sum(bool(row["pass"]) for row in witnesses))
    adopted = n_pass > PUBLISHED_WITNESS_PASSES
    report = {
        "adopted": adopted,
        "witness_pass_count": n_pass,
        "n_witnesses": 8,
        "published_context_logistic_passes": PUBLISHED_WITNESS_PASSES,
        "expert_confirmed": False,
        "positive_hot_min_fraction": HOT_PASS_FRACTION,
        "serve": "scorecard",
        "decides": "none",
        "objective": "stability_witness",
        "train_seeds": list(STABILITY_TRAIN_SEEDS),
        "steps": STABILITY_STEPS,
        "per_event": STABILITY_PER_EVENT,
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
    dest = ROOT / "artifacts" / "classify" / "stability_report.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"witnesses {n_pass}/8", flush=True)
    print("saved", dest, flush=True)
