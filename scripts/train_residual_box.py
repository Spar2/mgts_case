"""Train one residual net on dts_synth_v1 boxes.

Nuisance spans are Background. The loss sees every window of a box together.
The checkpoint is chosen on validation only. The test split is scored once
after the weights are frozen.

Published residual_net.pt and the already scored experimental residual_net.pt
are not opened for writing. Periodic_Thermal and Sensor_Fault have no pixels
in this set, so their class weights stay 0 and nothing is planted for them.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fiber.classify.dts_synth import iter_split
from fiber.classify.residual import METRE_CROP, SPACE_STRIDE_M, meter_crops
from fiber.classify.rules import BROAD_WIDTH_M, LABELS
from fiber.classify.scan import SPACE_STRIDE, TIME_STRIDE
from fiber.models.patch_classifier import SPACE_CROP, TIME_CROP

EXPERIMENTAL = ROOT / "models" / "weights" / "experimental"
OFFICIAL = EXPERIMENTAL / "residual_net_box.pt"
REPORT = ROOT / "artifacts" / "classify" / "residual_box_report.json"
BANNED = (
    ROOT / "models" / "weights" / "residual_net.pt",
    EXPERIMENTAL / "residual_net.pt",
)
RECIPE = "box-v1-nuisance-bg-cool3-bag-lse"
CLASS_WEIGHTS = (1.0, 1.0, 3.0, 0.0, 1.0, 0.0)
EVENT_NAMES = ("Local_Heating", "Local_Cooling", "Broad_Shift")
EVENT_CAP = 6
NUISANCE_CAP = 4
TRAIN_QUIET = 6
SCORE_QUIET = 12
BAGS_PER_STEP = 8
QUIET_FA_ANCHOR = 0.16
# Same column grid the already scored residual used in score_test.
SCORE_WINDOW_CAP = 36


def _unix(text: str) -> float:
    raw = text.strip()
    if raw.endswith("+00") and not raw.endswith("+0000"):
        raw = raw[:-3] + "+0000"
    return datetime.strptime(raw, "%Y-%m-%d %H:%M:%S%z").timestamp()


def _box_name(row: dict) -> str | None:
    label = row["label"]
    if label == "tracing":
        return None
    if label == "hotspot":
        return "Local_Heating"
    if label in ("coldspot", "cooling"):
        return "Local_Cooling"
    if label in ("section_heating", "slow_warming"):
        width = float(row["dist_end_m"]) - float(row["dist_start_m"])
        return "Broad_Shift" if width > BROAD_WIDTH_M else "Local_Heating"
    return None


def _cover(n_time: int, n_space: int, i0: int, i1: int, j0: int, j1: int) -> list[tuple[int, int]]:
    """Column origins. Matches score_test: stride 14 by 10, at most 36, plus a center."""
    if n_time < TIME_CROP or n_space < SPACE_CROP:
        return []
    t_lo = max(0, min(i0, n_time - TIME_CROP))
    t_hi = min(max(i1, 0), n_time - TIME_CROP)
    x_lo = max(0, min(j0, n_space - SPACE_CROP))
    x_hi = min(max(j1, 0), n_space - SPACE_CROP)
    if t_hi < t_lo:
        t_lo = t_hi
    if x_hi < x_lo:
        x_lo = x_hi
    t_starts = list(range(t_lo, t_hi + 1, TIME_STRIDE)) or [t_lo]
    x_starts = list(range(x_lo, x_hi + 1, SPACE_STRIDE)) or [x_lo]
    if len(t_starts) * len(x_starts) > SCORE_WINDOW_CAP:
        t_starts = t_starts[:: max(1, len(t_starts) // 6)]
        x_starts = x_starts[:: max(1, len(x_starts) // 6)]
    center = (
        int(np.clip((i0 + i1) // 2, 0, n_time - TIME_CROP)),
        int(np.clip((j0 + j1) // 2, 0, n_space - SPACE_CROP)),
    )
    origins = [(t0, x0) for t0 in t_starts for x0 in x_starts]
    if center not in origins:
        origins.append(center)
    return origins[:SCORE_WINDOW_CAP]


def _locate_score(record: dict, row: dict) -> tuple[int, int, int, int]:
    t_unix = record["t_unix"]
    x = record["x"]
    i0 = int(np.searchsorted(t_unix, _unix(row["t_start"])))
    i1 = int(np.searchsorted(t_unix, _unix(row["t_end"]), side="right") - 1)
    j0 = int(np.searchsorted(x, float(row["dist_start_m"]) - 1.0))
    j1 = int(np.searchsorted(x, float(row["dist_end_m"]) + 1.0, side="right") - 1)
    i0 = max(0, i0 - 2)
    i1 = min(len(t_unix) - 1, max(i0, i1 + 2))
    j0 = min(max(0, j0), len(x) - 1)
    j1 = min(max(j0, j1), len(x) - 1)
    return i0, i1, j0, j1


def _locate_train(record: dict, row: dict) -> tuple[int, int, int, int]:
    t_unix = record["t_unix"]
    x = record["x"]
    i0 = int(np.searchsorted(t_unix, _unix(row["t_start"])))
    i1 = int(np.searchsorted(t_unix, _unix(row["t_end"]), side="right") - 1)
    j0 = int(np.searchsorted(x, float(row["dist_start_m"])))
    j1 = int(np.searchsorted(x, float(row["dist_end_m"]), side="right") - 1)
    i0 = min(max(i0, 0), len(t_unix) - 1)
    i1 = min(max(i1, i0), len(t_unix) - 1)
    j0 = min(max(j0, 0), len(x) - 1)
    j1 = min(max(j1, j0), len(x) - 1)
    return i0, i1, j0, j1


def _kept_fraction(valid: np.ndarray, x: np.ndarray, t0: int, x0: int) -> float:
    """Share of the 28 by 64 m crop that meter_crops keeps. Missing metres count as empty."""
    start = float(x[int(x0)])
    targets = start + np.arange(METRE_CROP, dtype=np.float64)
    index = np.clip(np.searchsorted(x, targets, side="left"), 0, len(x) - 1)
    near = np.abs(x[index] - targets) <= 1.5
    slab = valid[int(t0):int(t0) + TIME_CROP][:, index]
    if slab.size == 0:
        return 0.0
    return float((slab & near).mean())


def _event_free(mask: np.ndarray, x: np.ndarray, t0: int, x0: int) -> bool:
    start = float(x[int(x0)])
    targets = start + np.arange(METRE_CROP, dtype=np.float64)
    index = np.clip(np.searchsorted(x, targets, side="left"), 0, len(x) - 1)
    near = np.abs(x[index] - targets) <= 1.5
    if not np.any(near):
        return False
    return int(mask[int(t0):int(t0) + TIME_CROP][:, index[near]].max()) == 0


def _thin(starts: list[int], keep: int) -> list[int]:
    if len(starts) <= keep:
        return starts
    idx = np.unique(np.round(np.linspace(0, len(starts) - 1, keep)).astype(int))
    return [starts[int(i)] for i in idx]


def _time_starts(n_time: int, i0: int, i1: int) -> list[int]:
    t_lo = max(0, i0 - TIME_CROP + 1)
    t_hi = min(i1, n_time - TIME_CROP)
    if t_hi < t_lo:
        return [int(np.clip((i0 + i1) // 2, 0, n_time - TIME_CROP))]
    return list(range(t_lo, t_hi + 1, TIME_STRIDE)) or [t_lo]


def _space_starts(x: np.ndarray, j0: int, j1: int) -> list[int]:
    x_start = float(x[j0])
    x_end = float(x[j1])
    idx = int(np.searchsorted(x, max(float(x[0]), x_start - (METRE_CROP - 1))))
    idx = min(max(idx, 0), len(x) - 1)
    starts: list[int] = []
    guard = 0
    while idx < len(x) and float(x[idx]) <= x_end and guard < 10000:
        start_m = float(x[idx])
        if start_m < x_end and start_m + METRE_CROP > x_start:
            starts.append(idx)
        new_idx = int(np.searchsorted(x, start_m + float(SPACE_STRIDE_M)))
        idx = new_idx if new_idx > idx else idx + 1
        guard += 1
    if not starts:
        starts = [min(max(j0, 0), len(x) - 1)]
    return starts


def _cap(origins: list[tuple[int, int]], center: tuple[int, int] | None, cap: int) -> list[tuple[int, int]]:
    if len(origins) <= cap:
        return origins
    idx = np.unique(np.round(np.linspace(0, len(origins) - 1, cap)).astype(int))
    chosen = [origins[int(i)] for i in idx]
    if center is not None and center not in chosen:
        chosen[-1] = center
    return chosen


def _train_origins(record: dict, i0: int, i1: int, j0: int, j1: int, cap: int) -> list[tuple[int, int]]:
    """64 m windows over the raw box. At most `cap`, and the center stays when it has data."""
    x = record["x"]
    valid = record["valid"]
    n_time = len(record["t_unix"])
    if n_time < TIME_CROP or len(x) < 2:
        return []
    t_starts = _time_starts(n_time, i0, i1)
    x_starts = _space_starts(x, j0, j1)
    if len(t_starts) * len(x_starts) > 48:
        t_starts = _thin(t_starts, 6)
        x_starts = _thin(x_starts, 8)
    center = (
        int(np.clip((i0 + i1) // 2, 0, n_time - TIME_CROP)),
        int(np.clip(np.searchsorted(x, 0.5 * (float(x[j0]) + float(x[j1])) - METRE_CROP / 2.0), 0, len(x) - 1)),
    )
    origins = [(t0, x0) for t0 in t_starts for x0 in x_starts]
    if center not in origins:
        origins.append(center)
    seen: set[tuple[int, int]] = set()
    unique: list[tuple[int, int]] = []
    for item in origins:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    kept = [item for item in unique if _kept_fraction(valid, x, item[0], item[1]) >= 0.5]
    forced = center if center in kept else None
    return _cap(kept, forced, cap)


def _train_quiet(record: dict, rng: np.random.Generator) -> list[tuple[int, int]]:
    mask = record["mask_event"]
    valid = record["valid"]
    x = record["x"]
    n_time, n_space = mask.shape
    found: list[tuple[int, int]] = []
    tries = 0
    if n_time < TIME_CROP or n_space < 2:
        return found
    while len(found) < TRAIN_QUIET and tries < TRAIN_QUIET * 40:
        tries += 1
        t0 = int(rng.integers(0, n_time - TIME_CROP + 1))
        x0 = int(rng.integers(0, n_space))
        if _kept_fraction(valid, x, t0, x0) < 0.5:
            continue
        if not _event_free(mask, x, t0, x0):
            continue
        found.append((t0, x0))
    return found


def _score_quiet(record: dict, rng: np.random.Generator) -> list[tuple[int, int]]:
    """Twelve 28 by 40 windows. Same draw as the scored residual: seed 2, valid fraction at least 0.5."""
    mask = record["mask_event"]
    valid = record["valid"]
    n_time, n_space = mask.shape
    found: list[tuple[int, int]] = []
    tries = 0
    while len(found) < SCORE_QUIET and tries < 200 and n_time >= TIME_CROP and n_space >= SPACE_CROP:
        tries += 1
        t0 = int(rng.integers(0, n_time - TIME_CROP + 1))
        x0 = int(rng.integers(0, n_space - SPACE_CROP + 1))
        if int(mask[t0:t0 + TIME_CROP, x0:x0 + SPACE_CROP].max()) != 0:
            continue
        if float(valid[t0:t0 + TIME_CROP, x0:x0 + SPACE_CROP].mean()) < 0.5:
            continue
        found.append((t0, x0))
    return found


def _cache_dir(root: Path, split: str, kind: str, limit: int | None) -> Path:
    tag = "all" if limit is None else f"lim{limit}"
    return root / "cache_residual_box" / f"{split}_{kind}_{tag}"


def _labels_by_seq(root: Path, split: str) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    with (root / "labels.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == split:
                grouped.setdefault(row["seq_id"], []).append(row)
    return grouped


def _write_ready(dest: Path, payload: dict) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "ready.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _ready(dest: Path) -> dict | None:
    path = dest / "ready.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("recipe") != RECIPE:
        return None
    return payload


def _clear(dest: Path) -> None:
    if not dest.exists():
        return
    for path in dest.glob("*"):
        if path.is_file():
            path.unlink()


def collect_train(root: Path, limit: int | None, rebuild: bool) -> Path:
    """One crop array. bag id -1 is a quiet window and stays out of the bag loss."""
    dest = _cache_dir(root, "train", "train", limit)
    if not rebuild and _ready(dest) is not None:
        print(f"cache train reuse {dest}", flush=True)
        return dest
    _clear(dest)
    dest.mkdir(parents=True, exist_ok=True)
    by_seq = _labels_by_seq(root, "train")
    rng = np.random.default_rng(0)
    pieces: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    bags: list[np.ndarray] = []
    bag_id = 0
    counts = {name: 0 for name in EVENT_NAMES}
    counts["Background"] = 0
    nuisance_bags = 0
    quiet_n = 0
    seen = 0
    for seq_id, record in iter_split(root, "train"):
        if limit is not None and seen >= limit:
            break
        seen += 1
        file_origins: list[tuple[int, int]] = []
        file_y: list[int] = []
        file_bag: list[int] = []
        for row in by_seq.get(seq_id, []):
            if row["kind"] == "event":
                name = _box_name(row)
                if name is None:
                    continue
                cap = EVENT_CAP
                target = LABELS.index(name)
            elif row["kind"] == "nuisance":
                name = "Background"
                cap = NUISANCE_CAP
                target = 0
            else:
                continue
            i0, i1, j0, j1 = _locate_train(record, row)
            origins = _train_origins(record, i0, i1, j0, j1, cap)
            if not origins:
                continue
            file_origins.extend(origins)
            file_y.extend([target] * len(origins))
            file_bag.extend([bag_id] * len(origins))
            bag_id += 1
            counts[name] += 1
            if row["kind"] == "nuisance":
                nuisance_bags += 1
        for origin in _train_quiet(record, rng):
            file_origins.append(origin)
            file_y.append(0)
            file_bag.append(-1)
            quiet_n += 1
        if file_origins:
            field = np.asarray(record["noisy"], dtype=np.float32)
            pieces.append(meter_crops(field, record["valid"], record["x"], np.asarray(file_origins, dtype=np.int64)))
            ys.append(np.asarray(file_y, dtype=np.int64))
            bags.append(np.asarray(file_bag, dtype=np.int64))
        print(f"cache train {seen} {seq_id} windows {len(file_origins)} bags {bag_id}", flush=True)
    if not pieces:
        raise RuntimeError("train cache has no windows")
    crops = np.concatenate(pieces, axis=0)
    y = np.concatenate(ys, axis=0)
    bag = np.concatenate(bags, axis=0)
    np.save(dest / "crops.npy", crops)
    np.save(dest / "y.npy", y)
    np.save(dest / "bag.npy", bag)
    payload = {
        "recipe": RECIPE,
        "split": "train",
        "sequences": seen,
        "windows": int(len(y)),
        "bags": int(bag_id),
        "nuisance_bags": int(nuisance_bags),
        "quiet_windows": int(quiet_n),
        "bag_counts": counts,
        "weights": list(CLASS_WEIGHTS),
    }
    _write_ready(dest, payload)
    print(f"cache train ready {payload}", flush=True)
    if limit is None and (nuisance_bags == 0 or counts["Local_Cooling"] == 0 or counts["Broad_Shift"] == 0):
        raise RuntimeError(f"full train cache is missing a required bag: {payload}")
    return dest


def collect_score(root: Path, split: str, limit: int | None, rebuild: bool) -> Path:
    """Validation or test boxes on the column grid, then metre crops. Quiet uses seed 2."""
    if split not in ("val", "test"):
        raise ValueError(split)
    dest = _cache_dir(root, split, "score", limit)
    if not rebuild and _ready(dest) is not None:
        print(f"cache {split} reuse {dest}", flush=True)
        return dest
    _clear(dest)
    dest.mkdir(parents=True, exist_ok=True)
    by_seq = _labels_by_seq(root, split)
    rng = np.random.default_rng(2)
    pieces: list[np.ndarray] = []
    jobs: list[dict] = []
    quiet_spans: list[dict] = []
    cursor = 0
    quiet_n = 0
    tracing = 0
    seen = 0
    box_counts = {name: 0 for name in EVENT_NAMES}
    nuisance_n = 0
    for seq_id, record in iter_split(root, split):
        if limit is not None and seen >= limit:
            break
        seen += 1
        field = np.asarray(record["noisy"], dtype=np.float32)
        valid = record["valid"]
        x = record["x"]
        n_time, n_space = field.shape
        file_origins: list[tuple[int, int]] = []
        file_jobs: list[dict] = []
        for row in by_seq.get(seq_id, []):
            if row["kind"] == "event" and row["label"] == "tracing":
                tracing += 1
                continue
            mapped = _box_name(row) if row["kind"] == "event" else None
            if row["kind"] == "event" and mapped is None:
                continue
            if row["kind"] not in ("event", "nuisance"):
                continue
            i0, i1, j0, j1 = _locate_score(record, row)
            origins = _cover(n_time, n_space, i0, i1, j0, j1)
            if not origins:
                continue
            file_jobs.append({
                "start": cursor + len(file_origins),
                "stop": cursor + len(file_origins) + len(origins),
                "kind": row["kind"],
                "mapped": mapped,
                "difficulty": row.get("difficulty") or "",
            })
            file_origins.extend(origins)
            if row["kind"] == "event":
                box_counts[mapped] += 1
            else:
                nuisance_n += 1
        quiet = _score_quiet(record, rng)
        box_windows = len(file_origins)
        file_origins.extend(quiet)
        if file_origins:
            crops = meter_crops(field, valid, x, np.asarray(file_origins, dtype=np.int64))
            pieces.append(crops)
            jobs.extend(file_jobs)
            if quiet:
                quiet_spans.append({"start": cursor + box_windows, "stop": cursor + len(file_origins)})
            cursor += len(file_origins)
            quiet_n += len(quiet)
        print(f"cache {split} {seen} {seq_id} boxes {len(file_jobs)} quiet {len(quiet)}", flush=True)
    crops = np.concatenate(pieces, axis=0) if pieces else np.zeros((0, 2, TIME_CROP, METRE_CROP), dtype=np.float32)
    _check_score_index(len(crops), jobs, quiet_spans)
    np.save(dest / "crops.npy", crops)
    (dest / "jobs.json").write_text(json.dumps(jobs), encoding="utf-8")
    (dest / "quiet.json").write_text(json.dumps(quiet_spans), encoding="utf-8")
    payload = {
        "recipe": RECIPE,
        "split": split,
        "sequences": seen,
        "windows": int(len(crops)),
        "quiet_windows": int(quiet_n),
        "tracing_boxes_excluded": int(tracing),
        "nuisance_boxes": int(nuisance_n),
        "box_counts": box_counts,
        "quiet_seed": 2,
    }
    _write_ready(dest, payload)
    print(f"cache {split} ready {payload}", flush=True)
    if limit is None and (quiet_n < 100 or box_counts["Local_Cooling"] == 0 or box_counts["Broad_Shift"] == 0):
        raise RuntimeError(f"full {split} score cache is missing a required box: {payload}")
    return dest


def _load_train(folder: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.load(folder / "crops.npy"), np.load(folder / "y.npy"), np.load(folder / "bag.npy")


def _check_score_index(n_rows: int, jobs: list[dict], quiet_spans: list[dict]) -> None:
    covered = np.zeros(n_rows, dtype=bool)
    for job in jobs:
        start, stop = int(job["start"]), int(job["stop"])
        if not (0 <= start < stop <= n_rows) or covered[start:stop].any():
            raise RuntimeError("score box windows overlap or fall outside the crop array")
        covered[start:stop] = True
    for span in quiet_spans:
        start, stop = int(span["start"]), int(span["stop"])
        if not (0 <= start < stop <= n_rows) or covered[start:stop].any():
            raise RuntimeError("quiet windows overlap a box or fall outside the crop array")
        covered[start:stop] = True
    if int(covered.sum()) != n_rows:
        raise RuntimeError("score cache has unlabeled windows")


def _load_score(folder: Path) -> tuple[np.ndarray, list[dict], list[dict], dict]:
    crops = np.load(folder / "crops.npy")
    jobs = json.loads((folder / "jobs.json").read_text(encoding="utf-8"))
    quiet_spans = json.loads((folder / "quiet.json").read_text(encoding="utf-8"))
    ready = json.loads((folder / "ready.json").read_text(encoding="utf-8"))
    return crops, jobs, quiet_spans, ready


def macro_recall(hits: dict) -> float:
    scores = []
    for name in EVENT_NAMES:
        stat = hits.get(name)
        if stat and stat["total"]:
            scores.append(stat["hit"] / stat["total"])
    return float(np.mean(scores)) if scores else 0.0


def selection_score(hits: dict, quiet_fa: float | None) -> float:
    penalty = 0.0 if quiet_fa is None else 0.5 * max(0.0, float(quiet_fa) - QUIET_FA_ANCHOR)
    return macro_recall(hits) - penalty


def judge(pred: np.ndarray, jobs: list[dict], quiet_spans: list[dict]) -> dict:
    hits: dict[str, dict[str, int]] = {}
    difficulty: dict[str, dict[str, int]] = {}
    nuisance_wrong = 0
    nuisance_n = 0
    for job in jobs:
        called = pred[int(job["start"]):int(job["stop"])]
        if job["kind"] == "event":
            index = LABELS.index(job["mapped"])
            hit = int(np.any(called == index))
            bucket = hits.setdefault(job["mapped"], {"hit": 0, "total": 0})
            bucket["total"] += 1
            bucket["hit"] += hit
            level = job.get("difficulty") or "unknown"
            level_bucket = difficulty.setdefault(level, {"hit": 0, "total": 0})
            level_bucket["total"] += 1
            level_bucket["hit"] += hit
        else:
            nuisance_n += 1
            nuisance_wrong += int(bool(len(called)) and bool(np.any(called != 0)))
    quiet_wrong = 0
    quiet_n = 0
    for span in quiet_spans:
        block = pred[int(span["start"]):int(span["stop"])]
        quiet_wrong += int(np.sum(block != 0))
        quiet_n += int(len(block))
    quiet_fa = (quiet_wrong / quiet_n) if quiet_n else None
    return {
        "box_hit": hits,
        "by_difficulty": difficulty,
        "quiet_false_alarm": quiet_fa,
        "quiet_windows": int(quiet_n),
        "nuisance_called_event": (nuisance_wrong / nuisance_n) if nuisance_n else None,
        "nuisance_boxes": int(nuisance_n),
        "macro_recall": macro_recall(hits),
        "selection_score": selection_score(hits, quiet_fa),
    }


def _predict(model, crops: np.ndarray, device: str) -> np.ndarray:
    import torch

    model.eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, len(crops), 64):
            xb = torch.from_numpy(np.ascontiguousarray(crops[start:start + 64])).to(device)
            pieces.append(model(xb).argmax(dim=1).cpu().numpy())
    return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.int64)


def _groups(bag: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    quiet = np.flatnonzero(bag < 0)
    ids = np.unique(bag[bag >= 0])
    return [np.flatnonzero(bag == int(item)) for item in ids], quiet


def _require_cuda() -> str:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("cuda is required")
    name = torch.cuda.get_device_name(0)
    print(f"device {name}", flush=True)
    return name


def _refuse(dest: Path) -> None:
    resolved = dest.resolve()
    for banned in BANNED:
        if resolved == banned.resolve():
            raise RuntimeError(f"refusing to overwrite {banned}")


def fit(train_dir: Path, val_dir: Path, epochs: int, dest: Path) -> dict:
    import torch
    from torch import nn

    from fiber.models.residual_net import ResidualNet

    device = "cuda"
    _require_cuda()
    crops, y, bag = _load_train(train_dir)
    groups, quiet = _groups(bag)
    if not groups:
        raise RuntimeError("train cache has no bags")
    for rows in groups:
        if len(rows) == 0 or len(set(int(item) for item in y[rows])) != 1:
            raise RuntimeError("a bag is empty or mixed")
    if len(quiet) and int(y[quiet].max()) != 0:
        raise RuntimeError("a quiet window is not Background")
    val_crops, jobs, quiet_spans, val_ready = _load_score(val_dir)
    quiet_rows = sum(int(span["stop"]) - int(span["start"]) for span in quiet_spans)
    if quiet_rows != int(val_ready["quiet_windows"]):
        raise RuntimeError("val quiet count does not match the spans")
    torch.manual_seed(0)
    model = ResidualNet(n_class=len(LABELS)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    weight = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32, device=device)
    rng = np.random.default_rng(0)
    best_score = -1e9
    best_state = None
    best_val = None
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(groups))
        q = np.array(quiet, copy=True)
        rng.shuffle(q)
        steps = int(np.ceil(len(order) / BAGS_PER_STEP))
        quiet_chunks = np.array_split(q, steps) if len(q) else [np.zeros(0, dtype=np.int64) for _ in range(steps)]
        losses = []
        for step in range(steps):
            chosen = order[step * BAGS_PER_STEP:(step + 1) * BAGS_PER_STEP]
            bag_rows = [groups[int(i)] for i in chosen]
            quiet_rows = quiet_chunks[step]
            parts = list(bag_rows)
            if len(quiet_rows):
                parts.append(quiet_rows)
            index = np.concatenate(parts)
            xb = torch.from_numpy(np.ascontiguousarray(crops[index])).to(device)
            yb = torch.from_numpy(y[index]).to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            window_loss = nn.functional.cross_entropy(logits, yb, weight=weight)
            offset = 0
            stacked = []
            targets = []
            for rows in bag_rows:
                stacked.append(torch.logsumexp(logits[offset:offset + len(rows)], dim=0))
                targets.append(yb[offset])
                offset += len(rows)
            bag_loss = nn.functional.cross_entropy(torch.stack(stacked), torch.stack(targets), weight=weight)
            loss = window_loss + bag_loss
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        if not np.isfinite(mean_loss):
            raise RuntimeError(f"loss is not finite at epoch {epoch}")
        pred = _predict(model, val_crops, device)
        scored = judge(pred, jobs, quiet_spans)
        if not np.isfinite(scored["selection_score"]):
            raise RuntimeError(f"selection score is not finite at epoch {epoch}")
        row = {
            "epoch": epoch,
            "loss": mean_loss,
            "val_macro_recall": scored["macro_recall"],
            "quiet_false_alarm": scored["quiet_false_alarm"],
            "selection_score": scored["selection_score"],
            "box_hit": scored["box_hit"],
            "nuisance_called_event": scored["nuisance_called_event"],
        }
        history.append(row)
        fa = scored["quiet_false_alarm"]
        fa_text = "none" if fa is None else f"{fa:.4f}"
        print(
            f"train residual_net_box epoch {epoch} loss {mean_loss:.4f} "
            f"val_recall {scored['macro_recall']:.4f} quiet_fa {fa_text} score {scored['selection_score']:.4f}",
            flush=True,
        )
        if scored["selection_score"] > best_score + 1e-4:
            best_score = float(scored["selection_score"])
            best_epoch = epoch
            best_val = scored
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 3:
                break
    if best_state is None or best_val is None:
        raise RuntimeError("residual_net_box produced no checkpoint")
    _refuse(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "n_class": len(LABELS)}, dest)
    return {
        "best_epoch": best_epoch,
        "epochs_ran": history[-1]["epoch"],
        "selection_score": best_score,
        "val": best_val,
        "history": history,
        "path": str(dest),
    }


def score_file(folder: Path, weight_path: Path) -> dict:
    import torch

    from fiber.models.residual_net import ResidualNet

    device = "cuda"
    _require_cuda()
    blob = torch.load(weight_path, map_location=device, weights_only=True)
    model = ResidualNet(n_class=int(blob["n_class"]))
    model.load_state_dict(blob["state_dict"])
    model.to(device)
    model.eval()
    crops, jobs, quiet_spans, ready = _load_score(folder)
    quiet_rows = sum(int(span["stop"]) - int(span["start"]) for span in quiet_spans)
    if quiet_rows != int(ready["quiet_windows"]):
        raise RuntimeError("quiet count does not match the spans")
    pred = _predict(model, crops, device)
    scored = judge(pred, jobs, quiet_spans)
    scored["tracing_boxes_excluded"] = int(ready["tracing_boxes_excluded"])
    scored["sequences"] = int(ready["sequences"])
    return scored


def _baseline() -> dict:
    return {
        "file": "models/weights/experimental/residual_net.pt",
        "box_hit": {"hit": 113, "total": 172},
        "Local_Heating": {"hit": 60, "total": 105},
        "Local_Cooling": {"hit": 17, "total": 25},
        "Broad_Shift": {"hit": 36, "total": 42},
        "quiet_false_alarm": 0.13703703703703704,
        "nuisance_called_event": 0.6290322580645161,
    }


def _report(device: str, trained: dict | None, test: dict | None, train_ready: dict | None, val_ready: dict | None) -> dict:
    return {
        "source": "dts_synth_v1",
        "device": device,
        "recipe": RECIPE,
        "weights": str(OFFICIAL),
        "loss": {
            "window": "weighted_cross_entropy",
            "bag": "cross_entropy of logsumexp over the box windows",
            "weights": list(CLASS_WEIGHTS),
            "weight_names": list(LABELS),
            "nuisance_target": "Background",
            "quiet_in_bag": False,
            "periodic_and_sensor_fault": "weight 0, no planted pixels",
        },
        "selection": "val box macro-recall - 0.5 * max(0, quiet_false_alarm - 0.16)",
        "optimizer": {"name": "AdamW", "lr": 1e-3, "weight_decay": 1e-4, "seed": 0},
        "train_cache": train_ready,
        "val_cache": val_ready,
        "trained": trained,
        "test": test,
        "compare_to_experimental_residual": _baseline(),
    }


def _tally(hits: dict) -> tuple[int, int]:
    return sum(int(item["hit"]) for item in hits.values()), sum(int(item["total"]) for item in hits.values())


def _print_score(title: str, scored: dict) -> None:
    hit, total = _tally(scored["box_hit"])
    cool = scored["box_hit"].get("Local_Cooling", {"hit": 0, "total": 0})
    broad = scored["box_hit"].get("Broad_Shift", {"hit": 0, "total": 0})
    fa = scored["quiet_false_alarm"]
    fa_text = "none" if fa is None else f"{fa:.4f}"
    print(
        f"{title} box {hit}/{total} cool {cool['hit']}/{cool['total']} "
        f"broad {broad['hit']}/{broad['total']} quiet_fa {fa_text} "
        f"nuisance {scored['nuisance_called_event']}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "artifacts" / "dts_synth_v1")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--stage", choices=("all", "train", "test"), default="all")
    args = parser.parse_args()
    print(f"pid {os.getpid()}", flush=True)
    assert list(CLASS_WEIGHTS) == [1.0, 1.0, 3.0, 0.0, 1.0, 0.0]
    assert LABELS.index("Local_Cooling") == 2 and LABELS.index("Broad_Shift") == 4
    device = _require_cuda()
    official = args.limit is None
    dest = OFFICIAL if official else Path(os.environ["TEMP"]) / "residual_net_box_smoke.pt"
    trained = None
    test = None
    train_ready = None
    val_ready = None
    if args.stage in ("all", "train"):
        train_dir = collect_train(args.root, args.limit, args.rebuild)
        val_dir = collect_score(args.root, "val", args.limit, args.rebuild)
        train_ready = _ready(train_dir)
        val_ready = _ready(val_dir)
        trained = fit(train_dir, val_dir, args.epochs, dest)
        _print_score("val", trained["val"])
    if args.stage in ("all", "test"):
        weight_path = OFFICIAL if args.stage == "test" or official else dest
        if args.stage == "test" and not weight_path.is_file():
            raise RuntimeError(f"missing weights {weight_path}")
        test_dir = collect_score(args.root, "test", args.limit, args.rebuild)
        test = score_file(test_dir, weight_path)
        _print_score("test", test)
    if official:
        if args.stage == "test" and REPORT.is_file():
            previous = json.loads(REPORT.read_text(encoding="utf-8"))
            trained = previous.get("trained")
            train_ready = previous.get("train_cache")
            val_ready = previous.get("val_cache")
            device = previous.get("device") or device
        payload = _report(device, trained, test, train_ready, val_ready)
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("TRAIN_DONE", flush=True)
    else:
        print("SMOKE_DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ABORT {exc}", flush=True)
        raise
