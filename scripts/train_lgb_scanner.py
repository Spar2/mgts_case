"""Fit LightGBM on the frozen context features and compare it with the logistic.

Eval seeds are the same 500_000 block. The planted search score picks the
checkpoint before any new real record is scanned. A loss leaves the files
already scanned by the logistic model as they are.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fiber.classify.context import FEATURE_NAMES, context_table, load_context_model
from fiber.classify.lgb_context import (
    LGB_LEARNING_RATE,
    LGB_LEAVES,
    LGB_MIN_CHILD_SAMPLES,
    LGB_TREES,
    fit_context_lgb,
    predict_context_lgb,
    save_context_lgb,
)
from fiber.classify.plant import LABEL_INDEX, plant_long
from fiber.classify.rules import LABELS
from fiber.classify.scan import choose_better, labeled_windows, locator_table, pool_search, score_search

EVAL_SEEDS = tuple(range(500_000, 500_004))
TRAIN_SEEDS = tuple(range(24))


def _balanced(seeds: tuple[int, ...], rng: np.random.Generator, per_event: int):
    buckets = {name: [] for name in LABELS}
    for seed in seeds:
        scene = plant_long(int(seed), profile=True, persistent=True)
        origins, truth = labeled_windows(scene["labels"], scene["valid"])
        feats = context_table(scene["field"], scene["valid"], origins)
        for row, name in zip(feats, truth, strict=True):
            buckets[name].append(row)
    counts = {name: len(rows) for name, rows in buckets.items()}
    cap = min(counts.values())
    cap = min(cap, per_event)
    if cap < 8:
        raise RuntimeError(f"context buckets did not fill: {counts}")
    chosen_x = []
    chosen_y = []
    for name in LABELS:
        take = rng.choice(counts[name], size=cap, replace=False)
        for index in take:
            chosen_x.append(buckets[name][int(index)])
            chosen_y.append(LABEL_INDEX[name])
    return np.stack(chosen_x), np.asarray(chosen_y, dtype=np.int64), counts, cap


def _score(predict, scenes: list[dict]) -> dict:
    parts = []
    for scene in scenes:
        origins, truth = labeled_windows(scene["labels"], scene["valid"])
        feats = context_table(scene["field"], scene["valid"], origins)
        predicted = predict(feats)
        locators = locator_table(scene["event_id"], origins, truth, scene["events"])
        parts.append(score_search(truth, predicted, scene["events"], locators))
    return pool_search(parts)


def _round_score(score: dict) -> dict:
    out = dict(score)
    for key in ("event_recall", "false_alarm_rate"):
        if out[key] is not None:
            out[key] = round(float(out[key]), 4)
    out["per_class_event_recall"] = {
        name: (None if value is None else round(float(value), 4))
        for name, value in score["per_class_event_recall"].items()
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-event", type=int, default=128)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--baseline", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or (ROOT / "models" / "weights" / "context_lgb.txt")
    baseline_path = args.baseline or (ROOT / "models" / "weights" / "context_logistic.json")
    started = time.perf_counter()
    rng = np.random.default_rng(0)
    features, target, counts, cap = _balanced(TRAIN_SEEDS, rng, args.per_event)
    model = fit_context_lgb(features, target)
    train_names = predict_context_lgb(model.booster_, features)
    train_hit = sum(pred == LABELS[int(index)] for pred, index in zip(train_names, target, strict=True))
    scenes = [plant_long(seed, profile=True, persistent=True) for seed in EVAL_SEEDS]
    challenger_score = _round_score(_score(lambda feats: predict_context_lgb(model.booster_, feats), scenes))
    baseline_score = None
    if baseline_path.is_file():
        logistic = load_context_model(json.loads(baseline_path.read_text(encoding="utf-8")))
        baseline_score = _round_score(_score(logistic.predict, scenes))
    picked = choose_better(challenger_score, baseline_score)
    choice = "lgb" if picked == "challenger" else "logistic"
    chosen_path = out if picked == "challenger" else baseline_path
    elapsed = time.perf_counter() - started
    print("train", cap, counts, f"fit_acc={train_hit / len(target):.3f}", flush=True)
    print("lgb", challenger_score["event_recall"], challenger_score["false_alarm_rate"], challenger_score["per_class_event_recall"], flush=True)
    if baseline_score is not None:
        print("logistic", baseline_score["event_recall"], baseline_score["false_alarm_rate"], baseline_score["per_class_event_recall"], flush=True)
    print("choice", choice, flush=True)
    save_context_lgb(model, out)
    report = {
        "choice": choice,
        "chosen_weights": str(chosen_path),
        "challenger": "lgb",
        "baseline": "logistic" if baseline_score is not None else None,
        "challenger_weights": str(out),
        "baseline_weights": str(baseline_path) if baseline_score is not None else None,
        "challenger_score": challenger_score,
        "baseline_score": baseline_score,
        "train_counts": counts,
        "per_event": cap,
        "train_accuracy": round(train_hit / len(target), 4),
        "elapsed_s": round(elapsed, 2),
        "serve": "scorecard",
        "objective": "planted_context_search",
        "eval_seeds": list(EVAL_SEEDS),
        "train_seeds": list(TRAIN_SEEDS),
        "features": list(FEATURE_NAMES),
        "n_estimators": LGB_TREES,
        "num_leaves": LGB_LEAVES,
        "learning_rate": LGB_LEARNING_RATE,
        "min_child_samples": LGB_MIN_CHILD_SAMPLES,
        "note": "Choice uses planted event recall only. Real traces are not an input.",
    }
    dest = ROOT / "artifacts" / "classify"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "lgb_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", out, f"{elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
