"""Train a patch CNN to search a long cable. The naming weights stay untouched.

Eval seeds start at 200_000 and do not overlap training seeds.
The checkpoint chosen for real traces is the one with higher planted event
recall. That choice is written before any real record is scanned.
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

from fiber.classify.plant import PERSISTENT_WIDTH, PROFILE_AMP_C, PROFILE_PERIODS, plant_long
from fiber.classify.rules import LABELS
from fiber.classify.scan import (
    EVENT_LABELS,
    choose_better,
    choose_weights,
    deviation_crops,
    labeled_windows,
    locator_table,
    pool_search,
    score_search,
)
from fiber.models.patch_classifier import SPACE_CROP, TIME_CROP, PatchClassifier, load_patch_classifier

LABEL_INDEX = {name: index for index, name in enumerate(LABELS)}
EVAL_SEEDS = tuple(range(200_000, 200_004))


def _gather(epoch: int, per_event: int, rng: np.random.Generator, planter):
    buckets = {name: [] for name in LABELS}
    produced = 0
    while any(len(buckets[name]) < per_event for name in EVENT_LABELS) or len(buckets["Background"]) < per_event * len(EVENT_LABELS):
        scene = planter(epoch * 1000 + produced)
        origins, truth = labeled_windows(scene["labels"], scene["valid"])
        crops = deviation_crops(scene["field"], scene["valid"], origins)
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
        crops = deviation_crops(scene["field"], scene["valid"], origins)
        predicted = _predict(model, crops, device, batch)
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
    parser.add_argument("--profile", action="store_true", help="train on cables with a room-scale baseline")
    parser.add_argument("--persistent", action="store_true", help="long heating and cooling on the room-scale baseline")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--naming", type=Path, default=None)
    args = parser.parse_args()
    if args.persistent:
        planter = lambda seed: plant_long(seed, profile=True, persistent=True)
        eval_seeds = tuple(range(400_000, 400_004))
        out = args.out or (ROOT / "models" / "weights" / "patch_persistent.pt")
        baseline_path = args.naming or (ROOT / "models" / "weights" / "patch_profile.pt")
        report_name = "persistent_report.json"
        objective = "planted_persistent_search"
        challenger_name = "persistent"
        baseline_name = "profile"
    elif args.profile:
        planter = lambda seed: plant_long(seed, profile=True)
        eval_seeds = tuple(range(300_000, 300_004))
        out = args.out or (ROOT / "models" / "weights" / "patch_profile.pt")
        baseline_path = args.naming or (ROOT / "models" / "weights" / "patch_scanner.pt")
        report_name = "profile_report.json"
        objective = "planted_profile_search"
        challenger_name = "profile"
        baseline_name = "scanner"
    else:
        planter = plant_long
        eval_seeds = EVAL_SEEDS
        out = args.out or (ROOT / "models" / "weights" / "patch_scanner.pt")
        baseline_path = args.naming or (ROOT / "models" / "weights" / "patch_classifier.pt")
        report_name = "scanner_report.json"
        objective = "planted_long_search"
        challenger_name = "scanner"
        baseline_name = "naming"

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = PatchClassifier().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        rng = np.random.default_rng(epoch)
        patches, target, n_cables = _gather(epoch, args.per_event, rng, planter)
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

    scenes = [planter(seed) for seed in eval_seeds]
    challenger_score = _round_score(_score_model(model, device, args.batch, scenes))
    baseline_score = None
    if baseline_path.is_file():
        baseline_model, _blob, baseline_device = load_patch_classifier(baseline_path, device=device)
        baseline_score = _round_score(_score_model(baseline_model, baseline_device, args.batch, scenes))
    picked = choose_better(challenger_score, baseline_score)
    if args.persistent or args.profile:
        choice = challenger_name if picked == "challenger" else baseline_name
    else:
        choice = choose_weights(challenger_score, baseline_score)
    chosen_path = out if picked == "challenger" else baseline_path
    print(challenger_name, challenger_score["event_recall"], challenger_score["false_alarm_rate"], challenger_score["per_class_event_recall"], flush=True)
    if baseline_score is not None:
        print(baseline_name, baseline_score["event_recall"], baseline_score["false_alarm_rate"], baseline_score["per_class_event_recall"], flush=True)
    print("choice", choice, flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_class": len(LABELS),
            "labels": list(LABELS),
            "time_crop": TIME_CROP,
            "space_crop": SPACE_CROP,
            "history": history,
            "elapsed_s": time.perf_counter() - started,
            "device": device,
            "serve": "scorecard",
            "objective": objective,
            "eval_seeds": list(eval_seeds),
            "planted_search": challenger_score,
            "baseline_search": baseline_score,
            "choice": choice,
        },
        out,
    )
    report = {
        "choice": choice,
        "chosen_weights": str(chosen_path),
        "challenger": challenger_name,
        "baseline": baseline_name if baseline_score is not None else None,
        "challenger_weights": str(out),
        "baseline_weights": str(baseline_path) if baseline_score is not None else None,
        "challenger_score": challenger_score,
        "baseline_score": baseline_score,
        "history": history,
        "device": device,
        "serve": "scorecard",
        "objective": objective,
        "eval_seeds": list(eval_seeds),
        "profile_amp_c": list(PROFILE_AMP_C) if args.profile or args.persistent else None,
        "profile_periods": list(PROFILE_PERIODS) if args.profile or args.persistent else None,
        "persistent_width": list(PERSISTENT_WIDTH) if args.persistent else None,
        "note": "Choice uses planted event recall only. Real traces are not an input.",
    }
    # Keys the flat-cable report already published.
    if not args.profile and not args.persistent:
        report["scanner_weights"] = str(out)
        report["naming_weights"] = str(baseline_path) if baseline_score is not None else None
        report["scanner"] = challenger_score
        report["naming"] = baseline_score
    dest = ROOT / "artifacts" / "classify"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / report_name).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
