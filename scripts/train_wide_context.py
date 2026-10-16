"""Train the patch CNN on a 28×120 context crop and compare it with the logistic.

Eval seeds are the 500_000 block. The planted search score picks the
checkpoint before any new real record is scanned. A loss leaves the logistic
scan as it is. Epochs, width, and the network stay at the values below.
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

from fiber.classify.context import WIDE_COLUMNS, context_table, load_context_model, wide_context_crops
from fiber.classify.plant import LABEL_INDEX, plant_long
from fiber.classify.rules import LABELS
from fiber.classify.scan import EVENT_LABELS, choose_better, labeled_windows, locator_table, pool_search, score_search
from fiber.models.patch_classifier import TIME_CROP, PatchClassifier

EVAL_SEEDS = tuple(range(500_000, 500_004))


def _planter(seed: int) -> dict:
    return plant_long(seed, profile=True, persistent=True)


def _gather(epoch: int, per_event: int, rng: np.random.Generator):
    buckets = {name: [] for name in LABELS}
    produced = 0
    while any(len(buckets[name]) < per_event for name in EVENT_LABELS) or len(buckets["Background"]) < per_event * len(EVENT_LABELS):
        scene = _planter(epoch * 1000 + produced)
        origins, truth = labeled_windows(scene["labels"], scene["valid"])
        crops = wide_context_crops(scene["field"], scene["valid"], origins)
        for crop, name in zip(crops, truth, strict=True):
            buckets[name].append(crop)
        produced += 1
        if produced > 40:
            raise RuntimeError("planted search windows did not fill the class buckets")
    chosen_x = []
    chosen_y = []
    for name in EVENT_LABELS:
        take = rng.choice(len(buckets[name]), size=per_event, replace=False)
        for index in take:
            chosen_x.append(buckets[name][int(index)])
            chosen_y.append(LABEL_INDEX[name])
    take = rng.choice(len(buckets["Background"]), size=per_event * len(EVENT_LABELS), replace=False)
    for index in take:
        chosen_x.append(buckets["Background"][int(index)])
        chosen_y.append(LABEL_INDEX["Background"])
    return np.stack(chosen_x), np.asarray(chosen_y, dtype=np.int64), produced


def _predict_cnn(model, crops: np.ndarray, device: str, batch: int) -> list[str]:
    import torch

    names: list[str] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(crops), batch):
            chunk = torch.from_numpy(np.ascontiguousarray(crops[start:start + batch]))[:, None].to(device)
            pred = model(chunk).argmax(dim=1).detach().cpu().numpy()
            names.extend(LABELS[int(index)] for index in pred)
    return names


def _score_cnn(model, device: str, batch: int, scenes: list[dict]) -> dict:
    parts = []
    for scene in scenes:
        origins, truth = labeled_windows(scene["labels"], scene["valid"])
        crops = wide_context_crops(scene["field"], scene["valid"], origins)
        predicted = _predict_cnn(model, crops, device, batch)
        locators = locator_table(scene["event_id"], origins, truth, scene["events"])
        parts.append(score_search(truth, predicted, scene["events"], locators))
    return pool_search(parts)


def _score_logistic(model, scenes: list[dict]) -> dict:
    parts = []
    for scene in scenes:
        origins, truth = labeled_windows(scene["labels"], scene["valid"])
        predicted = model.predict(context_table(scene["field"], scene["valid"], origins))
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
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--per-event", type=int, default=48)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--baseline", type=Path, default=None)
    args = parser.parse_args()
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = args.out or (ROOT / "models" / "weights" / "wide_context.pt")
    baseline_path = args.baseline or (ROOT / "models" / "weights" / "context_logistic.json")
    model = PatchClassifier().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        rng = np.random.default_rng(epoch)
        patches, target, n_cables = _gather(epoch, args.per_event, rng)
        order = rng.permutation(len(patches))
        tensor_x = torch.from_numpy(patches[order])[:, None]
        tensor_y = torch.from_numpy(target[order])
        loader = DataLoader(TensorDataset(tensor_x, tensor_y), batch_size=args.batch, shuffle=True, drop_last=True)
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            loss = torch.nn.functional.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        mse = float(np.mean(losses))
        history.append({"epoch": epoch, "loss": round(mse, 4), "cables": n_cables})
        print(f"epoch {epoch:02d} loss={mse:.4f} cables={n_cables}", flush=True)

    scenes = [_planter(seed) for seed in EVAL_SEEDS]
    challenger_score = _round_score(_score_cnn(model, device, args.batch, scenes))
    baseline_score = None
    if baseline_path.is_file():
        logistic = load_context_model(json.loads(baseline_path.read_text(encoding="utf-8")))
        baseline_score = _round_score(_score_logistic(logistic, scenes))
    picked = choose_better(challenger_score, baseline_score)
    choice = "wide" if picked == "challenger" else "logistic"
    chosen_path = out if picked == "challenger" else baseline_path
    elapsed = time.perf_counter() - started
    print("wide", challenger_score["event_recall"], challenger_score["false_alarm_rate"], challenger_score["per_class_event_recall"], flush=True)
    if baseline_score is not None:
        print("logistic", baseline_score["event_recall"], baseline_score["false_alarm_rate"], baseline_score["per_class_event_recall"], flush=True)
    print("choice", choice, flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_class": len(LABELS),
            "labels": list(LABELS),
            "time_crop": TIME_CROP,
            "wide_columns": WIDE_COLUMNS,
            "history": history,
            "elapsed_s": elapsed,
            "device": device,
            "serve": "scorecard",
            "objective": "planted_wide_context_search",
            "eval_seeds": list(EVAL_SEEDS),
            "epochs": args.epochs,
            "per_event": args.per_event,
            "planted_search": challenger_score,
            "baseline_search": baseline_score,
            "choice": choice,
        },
        out,
    )
    report = {
        "choice": choice,
        "chosen_weights": str(chosen_path),
        "challenger": "wide",
        "baseline": "logistic" if baseline_score is not None else None,
        "challenger_weights": str(out),
        "baseline_weights": str(baseline_path) if baseline_score is not None else None,
        "challenger_score": challenger_score,
        "baseline_score": baseline_score,
        "history": history,
        "elapsed_s": round(elapsed, 2),
        "device": device,
        "serve": "scorecard",
        "objective": "planted_wide_context_search",
        "eval_seeds": list(EVAL_SEEDS),
        "epochs": args.epochs,
        "per_event": args.per_event,
        "wide_columns": WIDE_COLUMNS,
        "note": "Choice uses planted event recall only. Real traces are not an input.",
    }
    dest = ROOT / "artifacts" / "classify"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "wide_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", out, f"{elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
