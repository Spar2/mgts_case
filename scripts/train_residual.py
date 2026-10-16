"""Train the residual net on drawn cables. The hand marks are not read here."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fiber.classify.residual import EVAL_SEEDS, TRAIN_SEEDS, meter_crops, plant_residual, residual_windows
from fiber.classify.rules import LABELS
from fiber.models.residual_net import ResidualNet

LABEL_INDEX = {name: index for index, name in enumerate(LABELS)}
EPOCHS = 12
PER_CLASS = 96
BATCH = 64


def _gather(seeds: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    buckets = {name: [] for name in LABELS}
    for seed in seeds:
        scene = plant_residual(int(seed))
        origins, truth = residual_windows(scene["labels"], scene["valid"])
        crops = meter_crops(scene["field"], scene["valid"], scene["x_m"], origins)
        for crop, name in zip(crops, truth, strict=True):
            buckets[name].append(crop)
    counts = {name: len(rows) for name, rows in buckets.items()}
    if any(counts[name] < PER_CLASS for name in LABELS):
        raise RuntimeError(f"class buckets did not fill: {counts}")
    rng = np.random.default_rng(0)
    chosen_x = []
    chosen_y = []
    for name in LABELS:
        take = rng.choice(counts[name], size=PER_CLASS, replace=False)
        for index in take:
            chosen_x.append(buckets[name][int(index)])
            chosen_y.append(LABEL_INDEX[name])
    return np.stack(chosen_x), np.asarray(chosen_y, dtype=np.int64)


def _predict(model, crops: np.ndarray, device: str) -> list[str]:
    import torch

    names: list[str] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(crops), BATCH):
            chunk = torch.from_numpy(np.ascontiguousarray(crops[start:start + BATCH])).to(device)
            pred = model(chunk).argmax(dim=1).detach().cpu().numpy()
            names.extend(LABELS[int(item)] for item in pred)
    return names


def _holdout(model, device: str) -> dict:
    totals = {name: 0 for name in LABELS}
    hits = {name: 0 for name in LABELS}
    for seed in EVAL_SEEDS:
        scene = plant_residual(int(seed))
        origins, truth = residual_windows(scene["labels"], scene["valid"])
        pred = _predict(model, meter_crops(scene["field"], scene["valid"], scene["x_m"], origins), device)
        for expected, got in zip(truth, pred, strict=True):
            totals[expected] += 1
            hits[expected] += int(expected == got)
    recall = {
        name: (None if totals[name] == 0 else round(hits[name] / totals[name], 3))
        for name in LABELS
    }
    return {"windows": totals, "recall": recall}


def main() -> None:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    features, target = _gather(TRAIN_SEEDS)
    model = ResidualNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    order = np.arange(len(target))
    rng = np.random.default_rng(1)
    model.train()
    last_loss = None
    for epoch in range(EPOCHS):
        rng.shuffle(order)
        total = 0.0
        seen = 0
        for start in range(0, len(order), BATCH):
            batch = order[start:start + BATCH]
            xb = torch.from_numpy(np.ascontiguousarray(features[batch])).to(device)
            yb = torch.from_numpy(target[batch]).to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(batch)
            seen += len(batch)
        last_loss = total / max(seen, 1)
        print(f"epoch {epoch + 1} loss {last_loss:.4f}", flush=True)
    holdout = _holdout(model, device)
    print("holdout", json.dumps(holdout, ensure_ascii=False), flush=True)
    dest = ROOT / "models" / "weights" / "residual_net.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_class": len(LABELS),
            "labels": list(LABELS),
            "train_seeds": list(TRAIN_SEEDS),
            "eval_seeds": list(EVAL_SEEDS),
            "epochs": EPOCHS,
            "holdout": holdout,
        },
        dest,
    )
    report = ROOT / "artifacts" / "classify" / "residual_report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps({"loss": last_loss, "device": device, "holdout": holdout}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("saved", dest, flush=True)


if __name__ == "__main__":
    main()
