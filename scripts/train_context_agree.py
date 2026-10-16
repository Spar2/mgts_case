"""Fit the agree-model and compare it with the published context logistic.

Held-out seeds start at 700_000. Seeds 600_000-600_003 were scored once with
the five-feature model and are not reused. The planted scores are written
before any real record is scanned. This script does not scan records.
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

from fiber.classify.agree_model import (
    AGREE_EVAL_SEEDS,
    AGREE_FEATURE_NAMES,
    AGREE_LEARNING_RATE,
    AGREE_LEAVES,
    AGREE_MIN_CHILD_SAMPLES,
    AGREE_TRAIN_SEEDS,
    AGREE_TREES,
    agree_table,
    fit_agree_model,
    load_agree_model,
)
from fiber.classify.context import context_table, load_context_model
from fiber.classify.lgb_context import predict_context_lgb, save_context_lgb
from fiber.classify.plant import LABEL_INDEX, plant_long
from fiber.classify.rules import LABELS
from fiber.classify.scan import choose_better, labeled_windows, locator_table, pool_search, score_search


def _collect(seeds: tuple[int, ...]):
    scenes = []
    features = []
    target = []
    for seed in seeds:
        scene = plant_long(int(seed), profile=True, persistent=True)
        scenes.append(scene)
        origins, truth = labeled_windows(scene["labels"], scene["valid"])
        feats = agree_table(scene["field"], scene["valid"], origins)
        for row, name in zip(feats, truth, strict=True):
            features.append(row)
            target.append(LABEL_INDEX[name])
    return scenes, np.stack(features), np.asarray(target, dtype=np.int64)


def _score(predict, scenes: list[dict], table) -> dict:
    parts = []
    for scene in scenes:
        origins, truth = labeled_windows(scene["labels"], scene["valid"])
        predicted = predict(table(scene["field"], scene["valid"], origins))
        locators = locator_table(scene["event_id"], origins, truth, scene["events"])
        parts.append(score_search(truth, predicted, scene["events"], locators))
    return pool_search(parts)


def _round_score(score: dict) -> dict:
    kept = {key: value for key, value in score.items() if key != "events"}
    for key in ("event_recall", "false_alarm_rate", "window_agreement"):
        if kept.get(key) is not None:
            kept[key] = round(float(kept[key]), 4)
    kept["per_class_event_recall"] = {
        name: (None if value is None else round(float(value), 4))
        for name, value in score["per_class_event_recall"].items()
    }
    return kept


def _print_score(name: str, score: dict) -> None:
    print(
        name,
        f"agreement={score['window_agreement']}",
        f"recall={score['event_recall']}",
        f"false_alarms={score['n_false_alarms']}/{score['n_background_windows']}",
        score["per_class_event_recall"],
        flush=True,
    )


def _rescore(report_path: Path) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    weights = Path(report["challenger_weights"])
    booster = load_agree_model(weights)
    scenes, _, _ = _collect(tuple(report["eval_seeds"]))
    score = _round_score(_score(lambda feats: predict_context_lgb(booster, feats), scenes, agree_table))
    published = report["challenger_score"]
    print(
        "rescore",
        f"agreement={score['window_agreement']}",
        f"recall={score['event_recall']}",
        f"false_alarms={score['n_false_alarms']}/{score['n_background_windows']}",
        flush=True,
    )
    if score["event_recall"] != published["event_recall"] or score["n_false_alarms"] != published["n_false_alarms"]:
        raise SystemExit("rescore does not match the saved report")
    if score["n_agreements"] != published["n_agreements"]:
        raise SystemExit("rescore window agreements do not match the saved report")
    print("rescore matches the report", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rescore", action="store_true")
    parser.add_argument("--report", type=Path, default=ROOT / "artifacts" / "classify" / "agree_report.json")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--baseline", type=Path, default=None)
    args = parser.parse_args()
    if args.rescore:
        _rescore(args.report)
        return
    if set(AGREE_TRAIN_SEEDS) & set(AGREE_EVAL_SEEDS):
        raise SystemExit("train seeds overlap the held-out cables")
    out = args.out or (ROOT / "models" / "weights" / "context_agree.txt")
    baseline_path = args.baseline or (ROOT / "models" / "weights" / "context_logistic.json")
    started = time.perf_counter()
    _train_scenes, features, target = _collect(AGREE_TRAIN_SEEDS)
    model = fit_agree_model(features, target)
    eval_scenes, _, _ = _collect(AGREE_EVAL_SEEDS)
    challenger_score = _round_score(
        _score(lambda feats: predict_context_lgb(model.booster_, feats), eval_scenes, agree_table)
    )
    baseline_score = None
    if baseline_path.is_file():
        logistic = load_context_model(json.loads(baseline_path.read_text(encoding="utf-8")))
        baseline_score = _round_score(_score(logistic.predict, eval_scenes, context_table))
    picked = choose_better(challenger_score, baseline_score)
    choice = "agree" if picked == "challenger" else "logistic"
    chosen_path = out if picked == "challenger" else baseline_path
    elapsed = time.perf_counter() - started
    _print_score("agree", challenger_score)
    if baseline_score is not None:
        _print_score("logistic", baseline_score)
    print("choice", choice, flush=True)
    if choice != "agree":
        print("real-record scan not started", flush=True)
    save_context_lgb(model, out)
    report = {
        "choice": choice,
        "chosen_weights": str(chosen_path),
        "challenger": "agree",
        "baseline": "logistic" if baseline_score is not None else None,
        "challenger_weights": str(out),
        "baseline_weights": str(baseline_path) if baseline_score is not None else None,
        "challenger_score": challenger_score,
        "baseline_score": baseline_score,
        "elapsed_s": round(elapsed, 2),
        "serve": "scorecard",
        "objective": "planted_window_agreement",
        "eval_seeds": list(AGREE_EVAL_SEEDS),
        "train_seeds": list(AGREE_TRAIN_SEEDS),
        "n_train_rows": int(len(target)),
        "features": list(AGREE_FEATURE_NAMES),
        "n_estimators": AGREE_TREES,
        "num_leaves": AGREE_LEAVES,
        "learning_rate": AGREE_LEARNING_RATE,
        "min_child_samples": AGREE_MIN_CHILD_SAMPLES,
        "note": "Planted-window agreement is not cable precision. Real traces are not an input.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", out, f"{elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
