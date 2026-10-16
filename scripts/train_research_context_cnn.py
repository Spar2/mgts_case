"""Fit a research CNN on the residual crop plus three context numbers.

The context numbers are the rectangle median, the far-cable median, and the
halo-inside fraction from context_evidence. Fit seeds are the previous 15
expert seeds plus every weight-10 ask expert seed. Gap-shortlist cards are
not fit seeds. Automatic holdout and guard rows stay out. The previous 15
stay in even when their own window is split 2 or -1. Weights are
serve=research and are a separate file from the residual2d checkpoint. This
does not change analyze_field.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from fiber.io import Record
from fiber.pipeline import analyze_field
from fiber.classify.rules import LABELS
from fiber.research_cnn import (
    CONTEXT_ARCHITECTURE,
    CONTEXT_NAMES,
    CONTEXT_SCALE_C,
    FIT_SPLIT_CODES,
    FOCUS_EXPERT_WEIGHT,
    POLARITY_LOSS_SCALE,
    ResearchContextCNN,
    context_vector,
    fit_rows,
    loss_and_checkpoint_masks,
    read_context_crop,
    research_batch_loss,
)
from fiber.weak_labels import EXPERT_SEED_BY_ID, EXPERT_SEEDS, EXPERT_WEIGHT

JSONL = ROOT / "docs" / "evidence" / "weak_labels.jsonl"
ASK_JSONL = ROOT / "docs" / "evidence" / "expert_seeds_ask.jsonl"
WEIGHTS = ROOT / "models" / "weights" / "research_context_cnn.pt"
MANIFEST = ROOT / "docs" / "evidence" / "research_context_cnn_manifest.json"
REPORT = ROOT / "docs" / "evidence" / "research_context_cnn_report.json"
PUBLISHED = ROOT / "docs" / "evidence" / "research_cnn_report.json"
RESIDUAL_WEIGHTS = ROOT / "models" / "weights" / "research_residual_cnn.pt"
EPOCHS = 6
SAMPLES_PER_EPOCH = 4096
BATCH = 64


class _Crops(Dataset):
    def __init__(self, features, context, labels, weights, polarity):
        self.features = features
        self.context = context
        self.labels = labels
        self.weights = weights
        self.polarity = polarity

    def __len__(self):
        return int(self.labels.shape[0])

    def __getitem__(self, index):
        return (
            torch.from_numpy(self.features[index]),
            torch.from_numpy(self.context[index]),
            int(self.labels[index]),
            float(self.weights[index]),
            float(self.polarity[index]),
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _expert_rows():
    path = ROOT / "scripts" / "train_research_cnn.py"
    spec = importlib.util.spec_from_file_location("train_research_cnn_context_fit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._expert_rows()


def _ask_rows() -> list[dict]:
    """Weight-10 ask expert seeds. Unclear answers are not in this file."""
    if not ASK_JSONL.is_file():
        raise ValueError(f"missing ask expert seeds {ASK_JSONL}")
    rows = []
    seen = set()
    for line in ASK_JSONL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        seed_id = str(row["id"])
        if (
            seed_id in seen
            or seed_id in EXPERT_SEED_BY_ID
            or seed_id.startswith("gap-")
        ):
            raise ValueError(f"ask seed id is not a new expert seed: {seed_id}")
        if row.get("origin") != "expert_seed" or row.get("expert_confirmed") is not True:
            raise ValueError(f"ask seed {seed_id} is not an expert-confirmed seed")
        if float(row["weight"]) < float(EXPERT_WEIGHT):
            raise ValueError(f"ask seed {seed_id} is not high-weight")
        if row["class_name"] not in LABELS:
            raise ValueError(f"ask seed {seed_id} left the six class names")
        seen.add(seed_id)
        rows.append(row)
    if not rows:
        raise ValueError("ask expert seed file is empty")
    return rows


def build_fit_rows() -> list[dict]:
    """Automatic split 0/1 rows, the previous 15, and every ask expert seed.

    Gap cards stay out. An ask answer that is not in the weight-10 seed file
    stays out. Automatic split 2 and -1 stay out. The previous 15 stay in.
    """
    rows = [json.loads(line) for line in JSONL.read_text(encoding="utf-8").splitlines() if line.strip()]
    ask = _ask_rows()
    seed_ids = [seed["id"] for seed in EXPERT_SEEDS]
    ask_ids = [row["id"] for row in ask]
    blocked = set(seed_ids) | set(ask_ids)
    automatic = [
        row for row in rows
        if row.get("origin") != "expert_seed" and str(row["id"]) not in blocked
    ]
    fit = fit_rows(automatic + _expert_rows() + ask, extra_seeds={row["id"]: row for row in ask})
    expert = [row for row in fit if row["origin"] == "expert_seed"]
    if [row["id"] for row in expert] != seed_ids + ask_ids:
        raise ValueError("context fit seeds are not the previous 15 plus the ask seeds")
    if len(expert) != len(set(row["id"] for row in expert)):
        raise ValueError("context fit expert id is duplicated")
    if any(str(row["id"]).startswith("gap-") for row in fit):
        raise ValueError("gap card entered the context fit")
    if any(row["origin"] != "expert_seed" and int(row["split_code"]) not in FIT_SPLIT_CODES for row in fit):
        raise ValueError("automatic holdout or guard row survived into the context fit")
    return fit


def _stack(field, record, examples):
    features = []
    context = []
    kept = []
    for index, example in enumerate(examples):
        try:
            crop, vector, measured = read_context_crop(
                field,
                record.x_m,
                record.time_s,
                example["rows_half_open"],
                example["x_m"],
            )
        except ValueError as exc:
            if example["origin"] == "expert_seed":
                raise SystemExit(f"{example['id']} context read failed: {exc}") from exc
            continue
        if crop is None:
            if example["origin"] == "expert_seed":
                raise SystemExit(f"{example['id']} residual crop is empty")
            continue
        if measured["far_cable_median_c"] is None and np.isfinite(vector[1]):
            raise SystemExit(f"{example['id']} absent far side was filled")
        if measured["rectangle_median_c"] is None:
            raise SystemExit(f"{example['id']} rectangle median is missing")
        if vector[0] != np.float32(measured["rectangle_median_c"]):
            raise SystemExit(f"{example['id']} rectangle median drifted")
        features.append(crop)
        context.append(vector)
        kept.append(example)
        if (index + 1) % 4000 == 0:
            print(f"Record {record.name} context {index + 1}/{len(examples)}", flush=True)
    if not features:
        empty = np.zeros((0, 1, 32, 64), np.float32)
        return empty, np.zeros((0, 3), np.float32), []
    stacked = np.stack(features).astype(np.float32)[:, None, :, :]
    return stacked, np.stack(context).astype(np.float32), kept


def _weighted_mean(model, features, context, labels, weights, device):
    total = 0.0
    scale = 0.0
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(labels), 256):
            stop = start + 256
            batch = torch.from_numpy(features[start:stop]).to(device)
            ctx = torch.from_numpy(context[start:stop]).to(device)
            target = torch.from_numpy(labels[start:stop]).to(device)
            weight = torch.from_numpy(weights[start:stop]).to(device)
            loss = loss_fn(model(batch, ctx), target)
            total += float((loss * weight).sum())
            scale += float(weight.sum())
    return total / max(scale, 1e-6)


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    published_sha = _sha256(PUBLISHED)
    labels_sha = _sha256(JSONL)
    residual_sha = _sha256(RESIDUAL_WEIGHTS)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fit = build_fit_rows()
    previous_ids = [seed["id"] for seed in EXPERT_SEEDS]
    present = [row for row in fit if row["origin"] == "expert_seed"]
    seed_ids = [row["id"] for row in present]
    ask_ids = [seed_id for seed_id in seed_ids if seed_id not in EXPERT_SEED_BY_ID]
    if seed_ids[:len(previous_ids)] != previous_ids or len(previous_ids) != 15:
        raise SystemExit("previous 15 expert seeds left the context fit")
    if not ask_ids or any(seed_id.startswith("gap-") for seed_id in ask_ids):
        raise SystemExit("ask expert seeds did not extend the context fit")
    if any(str(row["id"]).startswith("gap-") for row in fit):
        raise SystemExit("gap card entered the context fit")
    print(
        f"architecture {CONTEXT_ARCHITECTURE} device {device} serve=research "
        f"context {' '.join(CONTEXT_NAMES)} scale_c {CONTEXT_SCALE_C}",
        flush=True,
    )
    print(
        f"fit_rows {len(fit)} expert_seeds {len(present)} "
        f"previous_expert_seeds {len(previous_ids)} ask_expert_seeds {len(ask_ids)} "
        f"gap_cards_in_fit 0",
        flush=True,
    )
    by_record = {}
    for example in fit:
        by_record.setdefault(str(example["record"]), []).append(example)
    features = []
    context = []
    kept_rows = []
    for name in map(str, range(1, 7)):
        print(f"Record {name}: analyze_field", flush=True)
        record = Record.load(ROOT / "artifacts" / "records" / name)
        found = analyze_field(
            record.raw_c,
            record.x_m,
            record.time_s,
            valid=record.valid,
            time_iso=record.time_iso,
        )
        manifest = found["manifest"]
        config = manifest["config"]
        if (
            manifest["detector"] != "spatial_z_halo"
            or manifest["clean"] != "median_x3"
            or float(config["threshold_z"]) != 3.5
            or float(config["inner_m"]) != 8.0
            or float(config["outer_m"]) != 40.0
            or manifest["trained_classifier_operational"] is not False
        ):
            raise SystemExit("analyze_field is not the frozen operational pair")
        status = found["anomaly_status"]
        n_zones = len(found["zones"])
        stacked, ctx, kept = _stack(found, record, by_record.get(name, []))
        if found["anomaly_status"] != status or len(found["zones"]) != n_zones:
            raise SystemExit(f"record {name} context read changed the operational field")
        if len(kept):
            features.append(stacked)
            context.append(ctx)
            kept_rows.extend(kept)
        print(
            f"Record {name} finished crops {len(kept)} "
            f"operational_check clean {manifest['clean']} detector {manifest['detector']} "
            f"threshold_z {config['threshold_z']}",
            flush=True,
        )
        del found, record
    kept_ids = {row["id"] for row in kept_rows}
    missing_seeds = [seed_id for seed_id in seed_ids if seed_id not in kept_ids]
    if missing_seeds:
        raise SystemExit(f"expert seed crop missing: {missing_seeds}")
    features = np.concatenate(features, axis=0)
    context = np.concatenate(context, axis=0)
    if context.shape[1] != 3:
        raise SystemExit("context input is not the three measured numbers")
    labels = np.array([row["class_index"] for row in kept_rows], dtype=np.int64)
    weights = np.array([row["loss_weight"] for row in kept_rows], dtype=np.float32)
    polarities = np.array([row["polarity_target"] for row in kept_rows], dtype=np.float32)
    train, dev = loss_and_checkpoint_masks(kept_rows)
    expert_index = np.array([row["origin"] == "expert_seed" for row in kept_rows], dtype=bool)
    if int(expert_index.sum()) != len(seed_ids):
        raise SystemExit("expert seed crop missing from the loss mask")
    if not train.any() or not dev.any():
        raise SystemExit("fit needs both train and development rows")
    if expert_index[dev].any():
        raise SystemExit("expert card entered the checkpoint set")
    counts = np.bincount(labels[train], minlength=6).astype(np.float64)
    sample_weight = np.zeros(int(train.sum()), dtype=np.float64)
    train_labels = labels[train]
    for index, class_index in enumerate(train_labels):
        sample_weight[index] = 1.0 / max(counts[int(class_index)], 1.0)
    sampler = WeightedRandomSampler(sample_weight, num_samples=SAMPLES_PER_EPOCH, replacement=True)
    loader = DataLoader(
        _Crops(features[train], context[train], labels[train], weights[train], polarities[train]),
        batch_size=BATCH,
        sampler=sampler,
        num_workers=0,
    )
    model = ResearchContextCNN(n_class=len(LABELS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    best_loss = None
    best_state = None
    steps = 0
    for epoch in range(EPOCHS):
        model.train()
        for batch, ctx, target, weight, polarity in loader:
            batch = batch.to(device)
            ctx = ctx.to(device)
            target = target.to(device)
            weight = weight.to(device)
            polarity = polarity.to(device)
            loss = research_batch_loss(model(batch, ctx), target, weight, polarity)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            steps += 1
        expert_loss = research_batch_loss(
            model(torch.from_numpy(features[expert_index]).to(device), torch.from_numpy(context[expert_index]).to(device)),
            torch.from_numpy(labels[expert_index]).to(device),
            torch.from_numpy(weights[expert_index]).to(device),
            torch.from_numpy(polarities[expert_index]).to(device),
        )
        optimizer.zero_grad(set_to_none=True)
        expert_loss.backward()
        optimizer.step()
        steps += 1
        score = _weighted_mean(model, features[dev], context[dev], labels[dev], weights[dev], device)
        print(
            f"epoch {epoch} development_loss {score:.6f} steps {steps} "
            f"expert_seeds_in_loss {int(expert_index.sum())} expert_loss {float(expert_loss.detach().cpu()):.6f}",
            flush=True,
        )
        if best_loss is None or score < best_loss:
            best_loss = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if steps < 2 or best_state is None:
        raise SystemExit("training did not take more than one step")
    model.load_state_dict(best_state)
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "state_dict": model.state_dict(),
        "serve": "research",
        "operational": False,
        "trained_classifier_operational": False,
        "architecture": CONTEXT_ARCHITECTURE,
        "n_class": len(LABELS),
        "class_names": list(LABELS),
        "in_shape": [1, 32, 64],
        "context_names": list(CONTEXT_NAMES),
        "context_scale_c": float(CONTEXT_SCALE_C),
        "residual": "median_x3 crop minus its valid median, divided by 5 C",
        "context": "rectangle median, far-cable median, halo-inside fraction",
        "holdout_used_for_fit": False,
        "guard_used_for_fit": False,
        "gap_cards_in_fit": False,
        "sets_anomaly_status": False,
        "steps": steps,
        "epochs": EPOCHS,
        "development_loss": best_loss,
    }
    torch.save(blob, WEIGHTS)
    manifest = {
        "schema": "research_context_cnn_v1",
        "serve": "research",
        "operational": False,
        "trained_classifier_operational": False,
        "architecture": CONTEXT_ARCHITECTURE,
        "weights": "models/weights/research_context_cnn.pt",
        "weights_sha256": _sha256(WEIGHTS),
        "weights_bytes": WEIGHTS.stat().st_size,
        "steps": steps,
        "epochs": EPOCHS,
        "development_loss": best_loss,
        "n_fit_rows": int(train.sum()),
        "n_checkpoint_rows": int(dev.sum()),
        "n_expert_seeds": len(seed_ids),
        "n_previous_expert_seeds": len(previous_ids),
        "n_ask_expert_seeds": len(ask_ids),
        "expert_ids": seed_ids,
        "ask_expert_ids": ask_ids,
        "gap_cards_in_fit": False,
        "holdout_used_for_fit": False,
        "guard_used_for_fit": False,
        "context_names": list(CONTEXT_NAMES),
        "context_scale_c": float(CONTEXT_SCALE_C),
        "focus_expert_weight": float(FOCUS_EXPERT_WEIGHT),
        "polarity_loss_scale": float(POLARITY_LOSS_SCALE),
        "sets_anomaly_status": False,
        "note": (
            "Research weights only. The forward takes the residual crop plus the rectangle "
            "median, the far-cable median, and the halo-inside fraction. Fit seeds are the "
            "previous 15 expert cards plus every ask expert seed. Gap-shortlist cards are "
            "not fit seeds. Recovery on those seeds is in-sample. analyze_field stays "
            "median_x3 and spatial_z_halo at |Z| >= 3.5."
        ),
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if _sha256(PUBLISHED) != published_sha or _sha256(JSONL) != labels_sha or _sha256(RESIDUAL_WEIGHTS) != residual_sha:
        raise SystemExit("training rewrote the published report, weak labels, or residual2d weights")
    print(f"serve=research operational False steps {steps} weights {WEIGHTS}", flush=True)
    print(f"weights_sha256 {manifest['weights_sha256']}", flush=True)
    print("operational_check clean median_x3 detector spatial_z_halo threshold_z 3.5", flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
