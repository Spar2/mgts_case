"""Train a patch denoiser and heat/cool head on dts_synth_v1.

Synthetic HDF5 is the only training source. The 17 hand marks in
artifacts/labels/hand_events.json are scored after training and are never
sampled into the training set. Nothing under haklolarp is written.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import median_filter
from torch.utils.data import DataLoader, TensorDataset

PATCH_TIME = 32
PATCH_METERS = 64.0
PATCH_DX_M = 0.5
PATCH_SPACE = int(PATCH_METERS / PATCH_DX_M)  # 128
MIN_CLASS_CELLS = 8
CLASS_NAMES = ("background", "heat", "cool")
HEAT_IDS = (1, 2, 3, 4)  # tracing, hotspot, section_heating, slow_warming
COOL_IDS = (5, 6)  # coldspot, cooling

SYNTH_ROOT = Path(r"C:\Users\bobr\haklolarp\artifacts\dts_synth_v1\h5")
HAND_PATH = Path(r"C:\Users\bobr\haklolarp\artifacts\labels\hand_events.json")
RECORD_ROOT = Path(r"C:\Users\bobr\haklolarp\artifacts\records")
OUT_DIR = Path(r"C:\Users\bobr\test2_larpathon")


def label_from_mask(mask: np.ndarray, min_cells: int = MIN_CLASS_CELLS) -> int:
    """Map a resampled event mask to background / heat / cool."""
    heat = int(np.isin(mask, HEAT_IDS).sum())
    cool = int(np.isin(mask, COOL_IDS).sum())
    if heat >= min_cells and heat >= cool:
        return 1
    if cool >= min_cells and cool > heat:
        return 2
    return 0


def nearest_indices(distance_m: np.ndarray, targets_m: np.ndarray) -> np.ndarray:
    index = np.searchsorted(distance_m, targets_m, side="left")
    index = np.clip(index, 0, distance_m.size - 1)
    left = np.clip(index - 1, 0, distance_m.size - 1)
    choose_left = np.abs(distance_m[left] - targets_m) <= np.abs(distance_m[index] - targets_m)
    return np.where(choose_left, left, index).astype(np.int64)


def _resample_temperature(values: np.ndarray, distance_m: np.ndarray, targets_m: np.ndarray) -> np.ndarray:
    out = np.empty((values.shape[0], targets_m.size), dtype=np.float32)
    for row_index, row in enumerate(values):
        good = np.isfinite(row) & np.isfinite(distance_m)
        if int(good.sum()) < 2:
            out[row_index] = np.nan
            continue
        out[row_index] = np.interp(
            targets_m, distance_m[good], row[good], left=np.nan, right=np.nan
        )
    return out


def crop_patch(
    temperature: np.ndarray,
    distance_m: np.ndarray,
    center_t: int,
    center_m: float,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray] | None:
    """Return a 32 x 128 patch in °C, optional nearest-neighbor mask, and finite mask."""
    time_count = temperature.shape[0]
    if time_count < PATCH_TIME or distance_m.size < 2:
        return None
    half_m = PATCH_METERS / 2.0
    low_m = float(distance_m[0]) + half_m
    high_m = float(distance_m[-1]) - half_m
    if high_m < low_m:
        return None
    center_m = float(np.clip(center_m, low_m, high_m))
    t0 = int(center_t) - PATCH_TIME // 2
    t0 = max(0, min(t0, time_count - PATCH_TIME))
    t1 = t0 + PATCH_TIME
    targets = np.linspace(center_m - PATCH_METERS / 2.0, center_m + PATCH_METERS / 2.0, PATCH_SPACE)
    window = temperature[t0:t1]
    resampled = _resample_temperature(window, distance_m, targets)
    finite = np.isfinite(resampled)
    if finite.mean() < 0.8:
        return None
    mask_out = None
    if mask is not None:
        columns = nearest_indices(distance_m, targets)
        mask_out = mask[t0:t1][:, columns]
        outside = (targets < float(distance_m[0])) | (targets > float(distance_m[-1]))
        mask_out[:, outside] = 0
    return resampled, mask_out, finite


def normalize_patch(temperature_c: np.ndarray) -> tuple[np.ndarray, float, float]:
    finite = temperature_c[np.isfinite(temperature_c)]
    center = float(np.median(finite))
    scale = float(1.4826 * np.median(np.abs(finite - center)) + 1e-3)
    normalized = (temperature_c - center) / scale
    normalized = np.where(np.isfinite(normalized), normalized, 0.0).astype(np.float32)
    return normalized, center, scale


class SynthPatchNet(nn.Module):
    """Convolutional denoiser plus a 3-way background/heat/cool head."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 2, stride=2),
        )
        self.head = nn.Linear(64, len(CLASS_NAMES))

    def forward(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(batch)
        recon = self.decoder(encoded)
        recon = recon[..., : batch.shape[-2], : batch.shape[-1]]
        logits = self.head(encoded.mean(dim=(2, 3)))
        return recon, logits


def _time_origin(center_t: int, time_count: int) -> int:
    t0 = int(center_t) - PATCH_TIME // 2
    return max(0, min(t0, time_count - PATCH_TIME))


def _collect_from_file(
    path: Path,
    rng: np.random.Generator,
    buckets: dict[int, list],
    per_class_cap: int,
    attempts_event: int,
    attempts_background: int,
) -> None:
    if all(len(buckets[name]) >= per_class_cap for name in buckets):
        return
    with h5py.File(path, "r") as handle:
        distance = np.asarray(handle["x"][:], dtype=np.float64)
        mask = np.asarray(handle["mask_event"][:])
        time_count = int(mask.shape[0])
        if time_count < PATCH_TIME or float(distance[-1] - distance[0]) < PATCH_METERS:
            return
        need_heat = len(buckets[1]) < per_class_cap
        need_cool = len(buckets[2]) < per_class_cap
        need_bg = len(buckets[0]) < per_class_cap
        has_heat = bool(np.isin(mask, HEAT_IDS).any()) if need_heat else False
        has_cool = bool(np.isin(mask, COOL_IDS).any()) if need_cool else False
        if not (need_bg or has_heat or has_cool):
            return
        noisy = np.asarray(handle["noisy"][:], dtype=np.float32)
        clean = np.asarray(handle["clean"][:], dtype=np.float32)
    taken = {0: 0, 1: 0, 2: 0}
    file_cap = 40
    for class_id, class_ids in ((1, HEAT_IDS), (2, COOL_IDS)):
        if len(buckets[class_id]) >= per_class_cap:
            continue
        event_t, event_s = np.where(np.isin(mask, class_ids))
        if event_t.size == 0:
            continue
        order = np.arange(event_t.size)
        rng.shuffle(order)
        for pick in order[:attempts_event]:
            if len(buckets[class_id]) >= per_class_cap or taken[class_id] >= file_cap:
                break
            center_m = float(distance[event_s[pick]])
            cropped = crop_patch(noisy, distance, int(event_t[pick]), center_m, mask)
            clean_crop = crop_patch(clean, distance, int(event_t[pick]), center_m, None)
            if cropped is None or clean_crop is None:
                continue
            temperature, mask_patch, _finite = cropped
            label = label_from_mask(mask_patch)
            if label != class_id or taken[label] >= file_cap:
                continue
            normalized, center, scale = normalize_patch(temperature)
            clean_in_noisy_units = (clean_crop[0] - center) / scale
            clean_in_noisy_units = np.where(
                np.isfinite(clean_in_noisy_units), clean_in_noisy_units, 0.0
            )
            buckets[label].append(
                (
                    normalized,
                    clean_in_noisy_units.astype(np.float32),
                    np.float32(center),
                    np.float32(scale),
                    label,
                )
            )
            taken[label] += 1
    for _ in range(attempts_background):
        if len(buckets[0]) >= per_class_cap or taken[0] >= file_cap:
            break
        low_m = float(distance[0]) + PATCH_METERS / 2.0
        high_m = float(distance[-1]) - PATCH_METERS / 2.0
        if high_m <= low_m:
            break
        center_t = int(rng.integers(0, time_count))
        center_m = float(rng.uniform(low_m, high_m))
        cropped = crop_patch(noisy, distance, center_t, center_m, mask)
        clean_crop = crop_patch(clean, distance, center_t, center_m, None)
        if cropped is None or clean_crop is None:
            continue
        temperature, mask_patch, _finite = cropped
        if label_from_mask(mask_patch, min_cells=1) != 0:
            continue
        normalized, center, scale = normalize_patch(temperature)
        clean_in_noisy_units = (clean_crop[0] - center) / scale
        clean_in_noisy_units = np.where(np.isfinite(clean_in_noisy_units), clean_in_noisy_units, 0.0)
        buckets[0].append(
            (normalized, clean_in_noisy_units.astype(np.float32), np.float32(center), np.float32(scale), 0)
        )
        taken[0] += 1


def build_split(
    split: str,
    per_class: int,
    seed: int,
    max_files: int,
) -> dict[str, np.ndarray]:
    files = sorted((SYNTH_ROOT / split).glob("*.h5"))
    rng = np.random.default_rng(seed)
    rng.shuffle(files)
    if max_files > 0:
        files = files[:max_files]
    buckets: dict[int, list] = {0: [], 1: [], 2: []}
    for index, path in enumerate(files, start=1):
        _collect_from_file(path, rng, buckets, per_class, attempts_event=120, attempts_background=80)
        if index % 20 == 0 or index == len(files):
            counts = {CLASS_NAMES[key]: len(buckets[key]) for key in buckets}
            print(f"{split} files {index}/{len(files)} {counts}", flush=True)
        if all(len(buckets[key]) >= per_class for key in buckets):
            break
    rows = []
    for key in (0, 1, 2):
        rows.extend(buckets[key][:per_class])
    if not rows:
        raise RuntimeError(f"no patches collected for {split}")
    rng.shuffle(rows)
    noisy = np.stack([row[0] for row in rows])[:, None, :, :]
    clean = np.stack([row[1] for row in rows])[:, None, :, :]
    center = np.array([row[2] for row in rows], dtype=np.float32)
    scale = np.array([row[3] for row in rows], dtype=np.float32)
    labels = np.array([row[4] for row in rows], dtype=np.int64)
    print(
        f"{split} patches {len(rows)} "
        + " ".join(f"{name}={int((labels == i).sum())}" for i, name in enumerate(CLASS_NAMES)),
        flush=True,
    )
    return {"noisy": noisy, "clean": clean, "center": center, "scale": scale, "y": labels}


def _loader(split_data: dict[str, np.ndarray], batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = TensorDataset(
        torch.from_numpy(split_data["noisy"]),
        torch.from_numpy(split_data["clean"]),
        torch.from_numpy(split_data["scale"]),
        torch.from_numpy(split_data["y"]),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def evaluate(model: SynthPatchNet, split_data: dict[str, np.ndarray], device: torch.device) -> dict:
    model.eval()
    loader = _loader(split_data, batch_size=128, shuffle=False, seed=0)
    correct = 0
    total = 0
    per_hit = np.zeros(3, dtype=np.int64)
    per_count = np.zeros(3, dtype=np.int64)
    abs_error_c = 0.0
    cells = 0
    with torch.no_grad():
        for noisy, clean, scale, labels in loader:
            noisy = noisy.to(device)
            clean = clean.to(device)
            scale = scale.to(device)
            labels = labels.to(device)
            recon, logits = model(noisy)
            pred = logits.argmax(dim=1)
            correct += int((pred == labels).sum().item())
            total += int(labels.numel())
            for class_id in range(3):
                chosen = labels == class_id
                per_count[class_id] += int(chosen.sum().item())
                per_hit[class_id] += int((pred[chosen] == class_id).sum().item())
            abs_error_c += float((torch.abs(recon - clean) * scale[:, None, None, None]).sum().item())
            cells += int(noisy.numel())
    recall = {
        CLASS_NAMES[i]: (float(per_hit[i] / per_count[i]) if per_count[i] else None) for i in range(3)
    }
    present = [recall[name] for name in CLASS_NAMES if recall[name] is not None]
    return {
        "accuracy": correct / max(total, 1),
        "macro_recall": float(np.mean(present)) if present else 0.0,
        "recall": recall,
        "mae_c": abs_error_c / max(cells, 1),
        "n": total,
    }


def train_model(
    train_data: dict[str, np.ndarray],
    val_data: dict[str, np.ndarray],
    epochs: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> tuple[SynthPatchNet, list[dict]]:
    torch.manual_seed(seed)
    model = SynthPatchNet().to(device)
    counts = np.array([(train_data["y"] == i).sum() for i in range(3)], dtype=np.float64)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    criterion_cls = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    criterion_rec = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    history = []
    best_state = None
    best_score = -1.0
    for epoch in range(epochs):
        model.train()
        loader = _loader(train_data, batch_size, shuffle=True, seed=seed + epoch)
        total_loss = 0.0
        batches = 0
        for noisy, clean, _scale, labels in loader:
            noisy = noisy.to(device)
            clean = clean.to(device)
            labels = labels.to(device)
            recon, logits = model(noisy)
            loss = criterion_cls(logits, labels) + 0.3 * criterion_rec(recon, clean)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
        val_metrics = evaluate(model, val_data, device)
        row = {"epoch": epoch + 1, "loss": total_loss / max(batches, 1), **val_metrics}
        history.append(row)
        print(
            f"epoch {epoch + 1}/{epochs} loss={row['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.3f} val_macro={val_metrics['macro_recall']:.3f} "
            f"mae={val_metrics['mae_c']:.3f}C recall={val_metrics['recall']}",
            flush=True,
        )
        if val_metrics["macro_recall"] > best_score:
            best_score = val_metrics["macro_recall"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def expected_hand_name(event: dict) -> str:
    name = event["suggested_class"]
    if name == "Local_Heating":
        return "heat"
    if name == "Local_Cooling":
        return "cool"
    if name == "Background":
        return "background"
    if name == "Broad_Shift":
        return "heat" if float(event.get("delta_c", 0.0)) >= 0 else "cool"
    return "background"


def _record_arrays(file_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    folder = RECORD_ROOT / file_id
    temperature = np.asarray(np.load(folder / "raw_c.npy"), dtype=np.float32)
    temperature = median_filter(temperature, size=(1, 3))
    # Real file 5 fills the unmeasured tail with -300. That is not a temperature.
    temperature = temperature.copy()
    temperature[(temperature < -40.0) | (temperature > 150.0)] = np.nan
    distance = np.asarray(np.load(folder / "x_m.npy"), dtype=np.float64)
    time_s = np.asarray(np.load(folder / "time_s.npy"), dtype=np.float64)
    return temperature, distance, time_s


def _hours_to_index(time_s: np.ndarray, hour: float) -> int:
    target = float(time_s[0]) + hour * 3600.0
    return int(np.clip(np.searchsorted(time_s, target), 0, time_s.size - 1))


@torch.no_grad()
def _mean_probs(
    model: SynthPatchNet,
    temperature: np.ndarray,
    distance: np.ndarray,
    centers: list[tuple[int, float]],
    device: torch.device,
) -> np.ndarray | None:
    patches = []
    for center_t, center_m in centers:
        cropped = crop_patch(temperature, distance, center_t, center_m, None)
        if cropped is None:
            continue
        normalized, _center, _scale = normalize_patch(cropped[0])
        patches.append(normalized)
    if not patches:
        return None
    batch = torch.from_numpy(np.stack(patches)[:, None, :, :]).to(device)
    model.eval()
    _recon, logits = model(batch)
    return torch.softmax(logits, dim=1).mean(dim=0).detach().cpu().numpy()


def _box_centers(time_s: np.ndarray, distance: np.ndarray, event: dict) -> list[tuple[int, float]]:
    t_start = _hours_to_index(time_s, float(event["hours_from_start"][0]))
    t_end = _hours_to_index(time_s, float(event["hours_from_start"][1]))
    if t_end < t_start:
        t_start, t_end = t_end, t_start
    m0, m1 = float(event["meters"][0]), float(event["meters"][1])
    if m1 < m0:
        m0, m1 = m1, m0
    if t_end - t_start < PATCH_TIME:
        time_centers = [(t_start + t_end) // 2]
    else:
        time_centers = list(range(t_start + PATCH_TIME // 2, t_end - PATCH_TIME // 2 + 1, PATCH_TIME // 2))
        if not time_centers:
            time_centers = [(t_start + t_end) // 2]
    if m1 - m0 <= PATCH_METERS:
        meter_centers = [(m0 + m1) / 2.0]
    else:
        meter_centers = list(np.arange(m0 + PATCH_METERS / 2.0, m1 - PATCH_METERS / 2.0 + 1e-6, PATCH_METERS / 2.0))
        if not meter_centers:
            meter_centers = [(m0 + m1) / 2.0]
    return [(int(t_value), float(m_value)) for t_value in time_centers for m_value in meter_centers]


def score_hand_events(model: SynthPatchNet, device: torch.device) -> dict:
    payload = json.loads(HAND_PATH.read_text(encoding="utf-8"))
    events = payload["events"]
    by_file: dict[str, list] = {}
    for event in events:
        by_file.setdefault(str(event["file"]), []).append(event)
    rows = []
    for event in events:
        temperature, distance, time_s = _record_arrays(str(event["file"]))
        probs = _mean_probs(model, temperature, distance, _box_centers(time_s, distance, event), device)
        if probs is None:
            predicted = "no_measurement"
            match = event["suggested_class"] == "Background"
        else:
            predicted = CLASS_NAMES[int(np.argmax(probs))]
            match = predicted == expected_hand_name(event)
        rows.append(
            {
                "file": str(event["file"]),
                "meters": event["meters"],
                "hours_from_start": event["hours_from_start"],
                "what": event.get("what"),
                "suggested_class": event["suggested_class"],
                "expected": expected_hand_name(event),
                "predicted": predicted,
                "probs": None if probs is None else [float(value) for value in probs],
                "match": match,
            }
        )
    quiet_hits = 0
    quiet_n = 0
    rng = np.random.default_rng(1)
    for file_id, file_events in by_file.items():
        temperature, distance, time_s = _record_arrays(file_id)
        hours = (time_s - time_s[0]) / 3600.0
        kept = 0
        for _attempt in range(40):
            if kept >= 4:
                break
            center_t = int(rng.integers(0, time_s.size))
            center_m = float(rng.uniform(float(distance[0]) + 40.0, float(distance[-1]) - 40.0))
            hour = float(hours[center_t])
            too_close = False
            for event in file_events:
                m0, m1 = float(event["meters"][0]), float(event["meters"][1])
                h0, h1 = float(event["hours_from_start"][0]), float(event["hours_from_start"][1])
                if (min(m0, m1) - 40.0) <= center_m <= (max(m0, m1) + 40.0) and (
                    min(h0, h1) - 0.5
                ) <= hour <= (max(h0, h1) + 0.5):
                    too_close = True
                    break
            if too_close:
                continue
            probs = _mean_probs(model, temperature, distance, [(center_t, center_m)], device)
            if probs is None:
                continue
            quiet_n += 1
            kept += 1
            if CLASS_NAMES[int(np.argmax(probs))] == "background":
                quiet_hits += 1
    matched = sum(1 for row in rows if row["match"])
    return {
        "events": rows,
        "matched": matched,
        "n_events": len(rows),
        "quiet_background": quiet_hits,
        "quiet_n": quiet_n,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the synth patch net on the GPU")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--per-class", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-files", type=int, default=0, help="Cap files per split; 0 means all")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--score-only", action="store_true", help="Score the held-out hand marks with saved weights")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device_name = args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda is not available")
    device = torch.device(device_name)
    if device_name == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"device {torch.cuda.get_device_name(0)}", flush=True)
    if args.score_only:
        weight_path = OUT_DIR / "weights" / "synth_patch_net.pt"
        checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
        model = SynthPatchNet().to(device)
        model.load_state_dict(checkpoint["state_dict"])
        hand = score_hand_events(model, device)
        print(
            f"hand match {hand['matched']}/{hand['n_events']} "
            f"quiet background {hand['quiet_background']}/{hand['quiet_n']}",
            flush=True,
        )
        for row in hand["events"]:
            print(
                f"  file {row['file']} {row['meters']} {row['suggested_class']} "
                f"expected={row['expected']} predicted={row['predicted']} match={row['match']}",
                flush=True,
            )
        report_path = OUT_DIR / "reports" / "gpu_train.json"
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
        report["hand"] = hand
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0
    print("hand labels are held out and are not read while patches are sampled", flush=True)
    train_data = build_split("train", args.per_class, args.seed, args.max_files)
    val_data = build_split("val", max(100, args.per_class // 4), args.seed + 1, args.max_files)
    model, history = train_model(
        train_data, val_data, args.epochs, args.batch_size, args.seed, device
    )
    best_val = evaluate(model, val_data, device)
    test_metrics = None
    if not args.skip_test:
        test_data = build_split("test", max(100, args.per_class // 4), args.seed + 2, args.max_files)
        test_metrics = evaluate(model, test_data, device)
        print(f"test acc={test_metrics['accuracy']:.3f} macro={test_metrics['macro_recall']:.3f} mae={test_metrics['mae_c']:.3f}C", flush=True)
    hand = score_hand_events(model, device)
    print(
        f"hand match {hand['matched']}/{hand['n_events']} "
        f"quiet background {hand['quiet_background']}/{hand['quiet_n']}",
        flush=True,
    )
    for row in hand["events"]:
        print(
            f"  file {row['file']} {row['meters']} {row['suggested_class']} "
            f"expected={row['expected']} predicted={row['predicted']} match={row['match']}",
            flush=True,
        )
    weight_path = OUT_DIR / "weights" / "synth_patch_net.pt"
    report_path = OUT_DIR / "reports" / "gpu_train.json"
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_names": CLASS_NAMES,
            "patch_time": PATCH_TIME,
            "patch_space": PATCH_SPACE,
            "patch_meters": PATCH_METERS,
            "patch_dx_m": PATCH_DX_M,
            "seed": args.seed,
            "val": best_val,
            "test": test_metrics,
        },
        weight_path,
    )
    report = {
        "device": torch.cuda.get_device_name(0) if device_name == "cuda" else "cpu",
        "hand_labels_used_in_training": False,
        "val": best_val,
        "test": test_metrics,
        "history": [
            {
                "epoch": row["epoch"],
                "loss": row["loss"],
                "accuracy": row["accuracy"],
                "macro_recall": row["macro_recall"],
                "mae_c": row["mae_c"],
                "recall": row["recall"],
            }
            for row in history
        ],
        "hand": hand,
        "weights": str(weight_path),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {weight_path}", flush=True)
    print(f"wrote {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
