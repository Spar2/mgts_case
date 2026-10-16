"""Train the patch CNN on planted classes. Labels come from the generator."""

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

from fiber.classify.plant import KIND_TO_LABEL, LABEL_INDEX, plant_scene, rule_scene_label
from fiber.models.patch_classifier import SPACE_CROP, TIME_CROP, PatchClassifier, crop_deviation


KINDS = tuple(KIND_TO_LABEL)


def _sample_origin(scene: dict, rng: np.random.Generator) -> tuple[int, int]:
    t_count, _x_count = scene["field"].shape
    max_t = t_count - TIME_CROP
    valid_cols = np.flatnonzero(scene["valid"].any(axis=0))
    max_x = int(valid_cols[-1]) - SPACE_CROP + 1
    kind = scene["kind"]
    if kind == "background":
        return int(rng.integers(0, max_t + 1)), int(rng.integers(0, max_x + 1))
    if kind == "broad_shift":
        labeled = np.flatnonzero(scene["labels"].any(axis=0))
        cut = int(labeled[-1]) + 1
        return int(rng.integers(0, max_t + 1)), int(np.clip(cut - SPACE_CROP // 2, 0, max_x))
    if kind == "periodic":
        labeled = np.flatnonzero(scene["labels"].any(axis=0))
        center = int(np.median(labeled))
        return int(rng.integers(0, min(6, max_t + 1))), int(np.clip(center - SPACE_CROP // 2, 0, max_x))
    rows, cols = np.nonzero(scene["labels"] > 0)
    row = int(np.median(rows))
    col = int(np.median(cols))
    return int(np.clip(row - TIME_CROP // 2, 0, max_t)), int(np.clip(col - SPACE_CROP // 2, 0, max_x))


def _crop(scene: dict, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    t0, x0 = _sample_origin(scene, rng)
    patch = crop_deviation(scene["field"], scene["valid"], t0, x0)
    return patch, LABEL_INDEX[scene["label"]]


def _per_class(pairs: list[tuple[str, str]]) -> dict:
    out = {}
    for name in KIND_TO_LABEL.values():
        total = sum(truth == name for _pred, truth in pairs)
        hit = sum(pred == truth for pred, truth in pairs if truth == name)
        out[name] = round(hit / total, 4) if total else None
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--per-class", type=int, default=48)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "weights" / "patch_classifier.pt")
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = PatchClassifier().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        rng = np.random.default_rng(epoch)
        patches = []
        target = []
        for kind_index, kind in enumerate(KINDS):
            for sample in range(args.per_class):
                scene = plant_scene(kind, seed=epoch * 1000 + kind_index * 100 + sample)
                patch, label = _crop(scene, rng)
                patches.append(patch)
                target.append(label)
        order = rng.permutation(len(patches))
        tensor_x = torch.from_numpy(np.stack(patches)[order])[:, None]
        tensor_y = torch.from_numpy(np.array(target, dtype=np.int64)[order])
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
        history.append({"epoch": epoch, "loss": mse})
        print(f"epoch {epoch:02d} loss={mse:.4f}", flush=True)
    model.eval()
    names = list(KIND_TO_LABEL.values())
    cnn_pairs = []
    rule_pairs = []
    confusion = np.zeros((len(names), len(names)), dtype=int)
    rng = np.random.default_rng(9)
    with torch.inference_mode():
        for kind_index, kind in enumerate(KINDS):
            for sample in range(30):
                scene = plant_scene(kind, seed=100_000 + kind_index * 100 + sample)
                patch, _label = _crop(scene, rng)
                pred = int(model(torch.from_numpy(patch)[None, None].to(device)).argmax(dim=1).cpu())
                cnn_pairs.append((names[pred], scene["label"]))
                confusion[LABEL_INDEX[scene["label"]], pred] += 1
                rule_pairs.append((rule_scene_label(scene), scene["label"]))
    print("confusion rows=truth cols=pred", names, flush=True)
    print(confusion, flush=True)
    cnn_recall = _per_class(cnn_pairs)
    rule_recall = _per_class(rule_pairs)
    print("cnn", cnn_recall, flush=True)
    print("rules", rule_recall, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_class": 6,
            "labels": list(KIND_TO_LABEL.values()),
            "time_crop": TIME_CROP,
            "space_crop": SPACE_CROP,
            "history": history,
            "elapsed_s": time.perf_counter() - started,
            "device": device,
            "serve": "scorecard",
            "objective": "planted_scene_class",
            "cnn_recall": cnn_recall,
            "rule_recall": rule_recall,
            "confusion_truth_rows": confusion.tolist(),
        },
        args.out,
    )
    report = {
        "labels": names,
        "cnn_recall": cnn_recall,
        "rule_recall_on_same_scenes": rule_recall,
        "confusion_truth_rows": confusion.tolist(),
        "history": history,
        "device": device,
        "serve": "scorecard",
        "note": "Truth is the generator. Shape rules are a baseline, not the teacher.",
    }
    dest = ROOT / "artifacts" / "classify"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "patch_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
