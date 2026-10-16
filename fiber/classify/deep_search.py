"""Train and score the depth CNN, the column graph, and the column transformer.

The planted score is written before any real record is scanned. This module
does not scan records.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from fiber.classify.agree_model import agree_table, load_agree_model
from fiber.classify.context import wide_context_crops
from fiber.classify.deep_split import (
    DEEP_BATCH,
    DEEP_EPOCHS,
    DEEP_EVAL_SEEDS,
    DEEP_MAX_CABLES,
    DEEP_PER_EVENT,
    DEEP_PUBLISHED_EVAL,
    train_seed,
)
from fiber.classify.lgb_context import predict_context_lgb
from fiber.classify.plant import LABEL_INDEX, plant_long
from fiber.classify.rules import LABELS
from fiber.classify.scan import EVENT_LABELS, choose_better, labeled_windows, locator_table, pool_search, score_search
from fiber.models.column_graph import ColumnGraph, load_column_graph
from fiber.models.column_transformer import ColumnTransformer, load_column_transformer
from fiber.models.depth_cnn import DepthCNN, load_depth_cnn

ROOT = Path(__file__).resolve().parents[2]

KINDS = {
    "depth": {
        "factory": DepthCNN,
        "loader": load_depth_cnn,
        "weights": "depth_cnn.pt",
        "report": "depth_cnn_report.json",
        "objective": "planted_depth_search",
    },
    "graph": {
        "factory": ColumnGraph,
        "loader": load_column_graph,
        "weights": "column_graph.pt",
        "report": "column_graph_report.json",
        "objective": "planted_graph_search",
    },
    "transformer": {
        "factory": ColumnTransformer,
        "loader": load_column_transformer,
        "weights": "column_transformer.pt",
        "report": "column_transformer_report.json",
        "objective": "planted_transformer_search",
    },
}


def _planter(seed: int) -> dict:
    return plant_long(int(seed), profile=True, persistent=True)


def _gather(epoch: int, per_event: int, rng: np.random.Generator):
    buckets = {name: [] for name in LABELS}
    seeds = []
    produced = 0
    # One share per class. The 6-epoch run kept five background shares and
    # then labeled every window Background.
    need_background = per_event
    while any(len(buckets[name]) < per_event for name in EVENT_LABELS) or len(buckets["Background"]) < need_background:
        if produced >= DEEP_MAX_CABLES:
            raise RuntimeError("planted search windows did not fill the class buckets")
        seed = train_seed(epoch, produced)
        seeds.append(seed)
        scene = _planter(seed)
        origins, truth = labeled_windows(scene["labels"], scene["valid"])
        crops = wide_context_crops(scene["field"], scene["valid"], origins)
        for crop, name in zip(crops, truth, strict=True):
            buckets[name].append(crop)
        produced += 1
    chosen_x = []
    chosen_y = []
    for name in EVENT_LABELS:
        take = rng.choice(len(buckets[name]), size=per_event, replace=False)
        for index in take:
            chosen_x.append(buckets[name][int(index)])
            chosen_y.append(LABEL_INDEX[name])
    take = rng.choice(len(buckets["Background"]), size=need_background, replace=False)
    for index in take:
        chosen_x.append(buckets["Background"][int(index)])
        chosen_y.append(LABEL_INDEX["Background"])
    return np.stack(chosen_x), np.asarray(chosen_y, dtype=np.int64), seeds


def _predict(model, crops: np.ndarray, device: str, batch: int) -> list[str]:
    import torch

    names: list[str] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(crops), batch):
            chunk = torch.from_numpy(np.ascontiguousarray(crops[start:start + batch]))[:, None].to(device)
            pred = model(chunk).argmax(dim=1).detach().cpu().numpy()
            names.extend(LABELS[int(index)] for index in pred)
    return names


def _score_model(model, device: str, batch: int, scenes: list[dict]) -> dict:
    parts = []
    for scene in scenes:
        origins, truth = labeled_windows(scene["labels"], scene["valid"])
        predicted = _predict(model, wide_context_crops(scene["field"], scene["valid"], origins), device, batch)
        locators = locator_table(scene["event_id"], origins, truth, scene["events"])
        parts.append(score_search(truth, predicted, scene["events"], locators))
    return pool_search(parts)


def _score_agree(booster, scenes: list[dict]) -> dict:
    parts = []
    for scene in scenes:
        origins, truth = labeled_windows(scene["labels"], scene["valid"])
        predicted = predict_context_lgb(booster, agree_table(scene["field"], scene["valid"], origins))
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


def _finite(score: dict) -> None:
    if score.get("event_recall") is None or not math.isfinite(float(score["event_recall"])):
        raise SystemExit("planted event recall is not finite")
    if score.get("n_false_alarms") is None:
        raise SystemExit("planted false-alarm count is missing")


def _rescore(kind: str, report_path: Path, batch: int) -> None:
    import torch

    spec = KINDS[kind]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _blob, device = spec["loader"](Path(report["challenger_weights"]), device)
    scenes = [_planter(seed) for seed in report["eval_seeds"]]
    score = _round_score(_score_model(model, device, batch, scenes))
    published = report["challenger_score"]
    _print_score("rescore", score)
    if score["event_recall"] != published["event_recall"] or score["n_false_alarms"] != published["n_false_alarms"]:
        raise SystemExit("rescore does not match the saved report")
    print("rescore matches the report", flush=True)


def _fit(kind: str, device: str, batch: int):
    """Train one stack. Returns the model, history, seeds, step count, and seconds."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    spec = KINDS[kind]
    torch.manual_seed(0)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(0)
    model = spec["factory"]().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    history = []
    train_seeds: list[int] = []
    n_steps = 0
    started = time.perf_counter()
    for epoch in range(1, DEEP_EPOCHS + 1):
        rng = np.random.default_rng(epoch)
        patches, target, seeds = _gather(epoch, DEEP_PER_EVENT, rng)
        if set(seeds) & set(DEEP_EVAL_SEEDS) or set(seeds) & set(DEEP_PUBLISHED_EVAL):
            raise SystemExit("train seeds overlap a held-out cable")
        train_seeds.extend(seeds)
        order = rng.permutation(len(patches))
        tensor_x = torch.from_numpy(np.ascontiguousarray(patches[order]))[:, None]
        tensor_y = torch.from_numpy(target[order])
        counts = np.bincount(target, minlength=len(LABELS)).astype(np.float64)
        class_weight = torch.tensor(
            counts.sum() / (len(LABELS) * np.maximum(counts, 1.0)),
            dtype=torch.float32,
            device=device,
        )
        loader = DataLoader(TensorDataset(tensor_x, tensor_y), batch_size=batch, shuffle=False, drop_last=True)
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            loss = torch.nn.functional.cross_entropy(model(xb), yb, weight=class_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
            n_steps += 1
        mean_loss = float(np.mean(losses))
        history.append({"epoch": epoch, "loss": round(mean_loss, 4), "steps": len(losses)})
        print(f"epoch {epoch:02d} loss={mean_loss:.4f} steps={len(losses)}", flush=True)
    if n_steps <= 30:
        raise SystemExit("training did not exceed the published 30-step run")
    if set(train_seeds) & set(DEEP_EVAL_SEEDS) or set(train_seeds) & set(DEEP_PUBLISHED_EVAL):
        raise SystemExit("train seeds overlap a held-out cable")
    return model, history, train_seeds, n_steps, time.perf_counter() - started


def main_train(kind: str) -> None:
    spec = KINDS[kind]
    parser = argparse.ArgumentParser(description=f"Train the {kind} classifier and compare it with context_agree.")
    parser.add_argument("--rescore", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--report", type=Path, default=ROOT / "artifacts" / "classify" / spec["report"])
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--batch", type=int, default=DEEP_BATCH)
    args = parser.parse_args()
    if args.rescore:
        _rescore(kind, args.report, args.batch)
        return
    import torch

    if DEEP_EPOCHS < 2:
        raise SystemExit("training must run for more than one epoch")
    if set(DEEP_EVAL_SEEDS) & set(DEEP_PUBLISHED_EVAL):
        raise SystemExit("holdout reuses the published 800k cables")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = args.out or (ROOT / "models" / "weights" / spec["weights"])
    baseline_path = args.baseline or (ROOT / "models" / "weights" / "context_agree.txt")
    model, history, train_seeds, n_steps, elapsed = _fit(kind, device, args.batch)
    scenes = [_planter(seed) for seed in DEEP_EVAL_SEEDS]
    challenger_score = _round_score(_score_model(model, device, args.batch, scenes))
    _finite(challenger_score)
    baseline_score = None
    if baseline_path.is_file():
        booster = load_agree_model(baseline_path)
        baseline_score = _round_score(_score_agree(booster, scenes))
    picked = choose_better(challenger_score, baseline_score)
    choice = kind if picked == "challenger" else "agree"
    chosen_path = out if picked == "challenger" else baseline_path
    _print_score(kind, challenger_score)
    if baseline_score is not None:
        _print_score("agree", baseline_score)
    print("choice", choice, flush=True)
    print("steps", n_steps, flush=True)
    if choice != kind:
        print("real-record scan not started", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_class": len(LABELS),
            "labels": list(LABELS),
            "history": history,
            "n_optimizer_steps": n_steps,
            "epochs": DEEP_EPOCHS,
            "elapsed_s": elapsed,
            "device": device,
            "serve": "scorecard",
            "objective": spec["objective"],
            "eval_seeds": list(DEEP_EVAL_SEEDS),
            "train_seeds": train_seeds,
        },
        out,
    )
    report = {
        "choice": choice,
        "chosen_weights": str(chosen_path),
        "challenger": kind,
        "baseline": "agree" if baseline_score is not None else None,
        "challenger_weights": str(out),
        "baseline_weights": str(baseline_path) if baseline_score is not None else None,
        "challenger_score": challenger_score,
        "baseline_score": baseline_score,
        "history": history,
        "n_optimizer_steps": n_steps,
        "epochs": DEEP_EPOCHS,
        "elapsed_s": round(elapsed, 2),
        "device": device,
        "serve": "scorecard",
        "objective": spec["objective"],
        "eval_seeds": list(DEEP_EVAL_SEEDS),
        "train_seeds": train_seeds,
        "note": "Planted-window agreement is not cable precision. Real traces are not an input.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", out, f"{elapsed:.1f}s", flush=True)
