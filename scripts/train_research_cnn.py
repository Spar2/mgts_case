"""Fit the research residual 2D CNN on weak-label residual crops.

Automatic holdout and guard rows are not used to fit or to pick the
checkpoint. The previous EXPERT_SEEDS stay in the loss, including a
hard-negative Background seed and a seed whose window is holdout or mixed.
Gap-shortlist statistics cards are not fit seeds. The checkpoint is
automatic development rows only. Weights are serve=research. This does
not change analyze_field.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from fiber.denoise.dsp import median_x
from fiber.io import Record
from fiber.classify.rules import LABELS
from fiber.research_cnn import (
    COOLING_IDS,
    FIT_SPLIT_CODES,
    FOCUS_EXPERT_WEIGHT,
    HARD_NEGATIVE_IDS,
    POLARITY_LOSS_SCALE,
    ResearchResidualCNN,
    fit_rows,
    load_research_cnn,
    loss_and_checkpoint_masks,
    meter_columns,
    research_batch_loss,
    residual_crop,
)
from fiber.weak_labels import EXPERT_SEED_BY_ID, EXPERT_SEEDS, expert_card_report

JSONL = ROOT / "docs" / "evidence" / "weak_labels.jsonl"
WEIGHTS = ROOT / "models" / "weights" / "research_residual_cnn.pt"
MANIFEST = ROOT / "docs" / "evidence" / "research_cnn_manifest.json"
EPOCHS = 6
SAMPLES_PER_EPOCH = 4096
BATCH = 64


class _Crops(Dataset):
    def __init__(self, features, labels, weights, polarity):
        self.features = features
        self.labels = labels
        self.weights = weights
        self.polarity = polarity

    def __len__(self):
        return int(self.labels.shape[0])

    def __getitem__(self, index):
        return (
            torch.from_numpy(self.features[index]),
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


def _stack(record, examples):
    clean = median_x(record.raw_c, record.valid)
    features = []
    kept = []
    for example in examples:
        t0, t1 = example["rows_half_open"]
        c0, c1 = meter_columns(record.x_m, example["x_m"][0], example["x_m"][1])
        crop = residual_crop(clean, record.valid, t0, t1, c0, c1)
        if crop is None:
            continue
        features.append(crop)
        kept.append(example)
    if not features:
        return np.zeros((0, 1, 32, 64), np.float32), []
    stacked = np.stack(features).astype(np.float32)[:, None, :, :]
    return stacked, kept


def _weighted_mean(model, features, labels, weights, device):
    total = 0.0
    scale = 0.0
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(labels), 256):
            stop = start + 256
            batch = torch.from_numpy(features[start:stop]).to(device)
            target = torch.from_numpy(labels[start:stop]).to(device)
            weight = torch.from_numpy(weights[start:stop]).to(device)
            loss = loss_fn(model(batch), target)
            total += float((loss * weight).sum())
            scale += float(weight.sum())
    return total / max(scale, 1e-6)


def _expert_rows():
    times = {}
    for name in map(str, range(1, 7)):
        directory = ROOT / "artifacts" / "records" / name
        times[name] = (
            np.load(directory / "time_s.npy"),
            np.load(directory / "time_iso.npy", allow_pickle=True),
        )
    _emitted, report = expert_card_report(times)
    rows = []
    # Gap statistics share this report. Only the previous seed table is a fit seed.
    for card in report["cards"]:
        if card["id"] not in EXPERT_SEED_BY_ID:
            continue
        rows.append({
            "id": card["id"],
            "record": card["record"],
            "origin": "expert_seed",
            "class_name": card["class_name"],
            "split_code": int(card["split_code"]),
            "weight": float(card["weight"]),
            "expert_confirmed": True,
            "rows_half_open": list(card["rows_half_open"]),
            "x_m": list(card["x_m"]),
            "notes": card["notes"],
            "presence": card["presence"],
        })
    if [row["id"] for row in rows] != [seed["id"] for seed in EXPERT_SEEDS]:
        raise ValueError("research fit seeds are not the previous EXPERT_SEEDS")
    return rows


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = [json.loads(line) for line in JSONL.read_text(encoding="utf-8").splitlines() if line.strip()]
    seed_ids = [seed["id"] for seed in EXPERT_SEEDS]
    seed_id_set = set(seed_ids)
    automatic = [
        row for row in rows
        if row.get("origin") != "expert_seed" and str(row["id"]) not in seed_id_set
    ]
    holdout_excluded = sum(int(row["split_code"]) == 2 for row in automatic)
    guard_excluded = sum(int(row["split_code"]) == -1 for row in automatic)
    fit = fit_rows(automatic + _expert_rows())
    if any(row["origin"] != "expert_seed" and row["split_code"] not in FIT_SPLIT_CODES for row in fit):
        raise SystemExit("automatic holdout or guard row survived into the fit set")
    present = [row for row in fit if row["origin"] == "expert_seed"]
    if [row["id"] for row in present] != seed_ids and set(row["id"] for row in present) != seed_id_set:
        raise SystemExit(f"expert seeds missing from the fit set: {[row['id'] for row in present]}")
    if len(present) != len(seed_ids):
        raise SystemExit("expert seed count is not 15")
    if any(row["loss_weight"] < 10.0 or row["loss_weight"] <= 1.0 for row in present):
        raise SystemExit("expert seed is not heavier than an automatic row")
    by_seed = {row["id"]: row for row in present}
    ordinary_heating = [
        row["loss_weight"]
        for row in present
        if row["class_name"] == "Local_Heating" and row["presence"] == "yes"
    ]
    if not ordinary_heating:
        raise SystemExit("ordinary expert Local_Heating card missing")
    ordinary_ceiling = max(ordinary_heating)
    for seed_id in tuple(HARD_NEGATIVE_IDS) + tuple(COOLING_IDS):
        if by_seed[seed_id]["loss_weight"] <= ordinary_ceiling:
            raise SystemExit(f"{seed_id} is not heavier than ordinary expert Local_Heating")
    if FOCUS_EXPERT_WEIGHT <= ordinary_ceiling or POLARITY_LOSS_SCALE <= 1.0:
        raise SystemExit("focus weight or polarity scale is not stronger than the ordinary expert term")
    for seed_id in ("next-0005", "next-0006", "next-0007", "next-0010"):
        if by_seed[seed_id]["class_name"] != "Background" or by_seed[seed_id]["presence"] != "no":
            raise SystemExit(f"hard negative {seed_id} is not Background")
    if "в норме" not in by_seed["next-0006"]["notes"]:
        raise SystemExit("next-0006 lost the normal-cable note")
    for seed_id in ("review-00223", "next-0002"):
        if by_seed[seed_id]["class_name"] != "Local_Cooling":
            raise SystemExit(f"cooling seed {seed_id} left Local_Cooling")
    outside = [
        {
            "id": row["id"],
            "record": row["record"],
            "split_code": int(row["split_code"]),
            "class_name": row["class_name"],
            "presence": row["presence"],
        }
        for row in present
        if int(row["split_code"]) not in FIT_SPLIT_CODES
    ]
    if not outside:
        raise SystemExit("expected expert seeds on split 2 or -1")
    if any(EXPERT_SEED_BY_ID[item["id"]]["class_name"] != item["class_name"] for item in outside):
        raise SystemExit("outside expert seed class drifted")
    print(f"architecture residual2d device {device} serve=research", flush=True)
    print(
        f"automatic holdout and guard rows were excluded "
        f"holdout_rows_excluded {holdout_excluded} guard_rows_excluded {guard_excluded}",
        flush=True,
    )
    print(
        "expert_seeds_still_fit_on_split_2_or_minus_1 "
        + " ".join(f"{item['id']}:{item['split_code']}" for item in outside),
        flush=True,
    )
    by_record = {}
    for example in fit:
        by_record.setdefault(example["record"], []).append(example)
    features = []
    kept_rows = []
    for name in map(str, range(1, 7)):
        print(f"Record {name}: residual crops", flush=True)
        record = Record.load(ROOT / "artifacts" / "records" / name)
        stacked, kept = _stack(record, by_record.get(name, []))
        if len(kept):
            features.append(stacked)
            kept_rows.extend(kept)
        print(f"Record {name} finished crops {len(kept)}", flush=True)
    kept_ids = {row["id"] for row in kept_rows}
    missing_seeds = [seed_id for seed_id in seed_ids if seed_id not in kept_ids]
    if missing_seeds:
        raise SystemExit(f"expert seed crop missing: {missing_seeds}")
    features = np.concatenate(features, axis=0)
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
    if any(row["origin"] != "expert_seed" and int(row["split_code"]) not in FIT_SPLIT_CODES for row in kept_rows):
        raise SystemExit("automatic holdout or guard crop stayed in the stack")
    counts = np.bincount(labels[train], minlength=6).astype(np.float64)
    sample_weight = np.zeros(int(train.sum()), dtype=np.float64)
    train_labels = labels[train]
    for index, class_index in enumerate(train_labels):
        sample_weight[index] = 1.0 / max(counts[int(class_index)], 1.0)
    sampler = WeightedRandomSampler(sample_weight, num_samples=SAMPLES_PER_EPOCH, replacement=True)
    loader = DataLoader(
        _Crops(features[train], labels[train], weights[train], polarities[train]),
        batch_size=BATCH,
        sampler=sampler,
        num_workers=0,
    )
    continued = WEIGHTS.is_file()
    if continued:
        model, prior, _device = load_research_cnn(WEIGHTS, device=device)
        if prior.get("serve") != "research":
            raise SystemExit("refusing to continue from a non-research checkpoint")
        print("continued from research checkpoint", flush=True)
    else:
        model = ResearchResidualCNN(n_class=6).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    best_loss = None
    best_state = None
    steps = 0
    for epoch in range(EPOCHS):
        model.train()
        for batch, target, weight, polarity in loader:
            batch = batch.to(device)
            target = target.to(device)
            weight = weight.to(device)
            polarity = polarity.to(device)
            loss = research_batch_loss(model(batch), target, weight, polarity)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            steps += 1
        expert_batch = torch.from_numpy(features[expert_index]).to(device)
        expert_target = torch.from_numpy(labels[expert_index]).to(device)
        expert_weight = torch.from_numpy(weights[expert_index]).to(device)
        expert_polarity = torch.from_numpy(polarities[expert_index]).to(device)
        expert_loss = research_batch_loss(
            model(expert_batch), expert_target, expert_weight, expert_polarity,
        )
        optimizer.zero_grad(set_to_none=True)
        expert_loss.backward()
        optimizer.step()
        steps += 1
        score = _weighted_mean(model, features[dev], labels[dev], weights[dev], device)
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
        "architecture": "residual2d",
        "n_class": 6,
        "class_names": list(LABELS),
        "in_shape": [1, 32, 64],
        "residual": "median_x3 crop minus its valid median, divided by 5 C",
        "holdout_used_for_fit": False,
        "guard_used_for_fit": False,
        "automatic_holdout_used_for_fit": False,
        "automatic_guard_used_for_fit": False,
        "fit_split_codes": [0],
        "checkpoint_split_codes": [1],
        "expert_seeds_still_fit_on_split_2_or_minus_1": outside,
        "expert_cards_are_not_an_untouched_holdout": True,
        "continued_from_checkpoint": continued,
        "steps": steps,
        "epochs": EPOCHS,
        "development_loss": best_loss,
    }
    torch.save(blob, WEIGHTS)
    manifest = {
        "schema": "research_cnn_v1",
        "serve": "research",
        "operational": False,
        "trained_classifier_operational": False,
        "architecture": "residual2d",
        "weights": "models/weights/research_residual_cnn.pt",
        "weights_sha256": _sha256(WEIGHTS),
        "weights_bytes": WEIGHTS.stat().st_size,
        "steps": steps,
        "epochs": EPOCHS,
        "development_loss": best_loss,
        "n_fit_rows": int(train.sum()),
        "n_checkpoint_rows": int(dev.sum()),
        "n_expert_seeds": len(seed_ids),
        "holdout_rows_excluded": holdout_excluded,
        "guard_rows_excluded": guard_excluded,
        "holdout_used_for_fit": False,
        "guard_used_for_fit": False,
        "automatic_holdout_used_for_fit": False,
        "automatic_guard_used_for_fit": False,
        "fit_split_codes": [0],
        "checkpoint_split_codes": [1],
        "expert_ids": seed_ids,
        "expert_seeds_still_fit_on_split_2_or_minus_1": outside,
        "expert_cards_are_not_an_untouched_holdout": True,
        "hard_negative_ids": ["next-0005", "next-0006", "next-0007", "next-0010"],
        "hard_negative_class": "Background",
        "cooling_ids": ["review-00223", "next-0002"],
        "cooling_class": "Local_Cooling",
        "continued_from_checkpoint": continued,
        "expert_loss_weight": 10.0,
        "focus_expert_weight": float(FOCUS_EXPERT_WEIGHT),
        "polarity_loss_scale": float(POLARITY_LOSS_SCALE),
        "polarity_is_a_class": False,
        "automatic_loss_weight": 1.0,
        "note": (
            "Research weights only. Automatic holdout and guard rows were not used. "
            "Expert cards on split 2 or -1 were still fit as seeds and are not an untouched holdout. "
            "Cooling experts and presence=no experts use a heavier loss weight than an ordinary "
            "expert Local_Heating card, plus a stronger hot/cold polarity term. Polarity is not a class. "
            "Not an operational detector and not expert accuracy."
        ),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"serve=research steps {steps} weights {WEIGHTS}", flush=True)
    print(
        f"holdout_used_for_fit {manifest['holdout_used_for_fit']} "
        f"guard_used_for_fit {manifest['guard_used_for_fit']}",
        flush=True,
    )
    print("finished", flush=True)


if __name__ == "__main__":
    main()
