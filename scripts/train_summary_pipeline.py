"""Train the summary-pipeline models on dts_synth_v1 and score mixed alarms.

New weights go only under models/weights/experimental, and a smoke run writes
them under the temp directory instead. Published weights, residual_net_box,
and the three supervised denoisers are read, never replaced. median_x3 and
spatial_z_halo stay the operational pair. The test split does not pick epochs.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tempfile
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fiber.anomaly.detect import spatial_z
from fiber.anomaly.zones import extract_zones
from fiber.classify.dts_synth import _denoise_norm, _row_norm, iter_split, sample_windows
from fiber.classify.rules import BROAD_WIDTH_M, LABELS
from fiber.denoise.dsp import median_x, nlm_wide, wavelet_x
from fiber.models.patch_classifier import SPACE_CROP, TIME_CROP
from fiber.models.summary_nets import DilatedDenoiser, GradientFusion, SegUNet

EXPERIMENTAL = ROOT / "models" / "weights" / "experimental"
REPORT = ROOT / "artifacts" / "classify" / "summary_pipeline_report.json"
ALLOWED = {
    "dilated1d_supervised.pt",
    "vmd_dilated1d.pt",
    "gradient_fusion.pt",
    "seg_unet.pt",
    "zone_xgb.json",
    "gadf_efficientnet.pt",
    "yolo_waterfall.pt",
}
PROTECTED = [
    ROOT / "models" / "weights" / "residual_net.pt",
    EXPERIMENTAL / "residual_net.pt",
    EXPERIMENTAL / "residual_net_box.pt",
    EXPERIMENTAL / "conv1d_supervised.pt",
    EXPERIMENTAL / "unet2d_supervised.pt",
    EXPERIMENTAL / "conv_ae_bg.pt",
    ROOT / "models" / "weights" / "conv1d_n2self.pt",
    ROOT / "models" / "weights" / "unet2d_n2self.pt",
    ROOT / "models" / "weights" / "conv_ae_bg.pt",
]
CLASS_WEIGHT = np.asarray([1, 1, 3, 0, 1, 0], dtype=np.float32)
SEG_WEIGHT = np.asarray([1, 1, 3, 1], dtype=np.float32)
YOLO_NAMES = ("Local_Heating", "Local_Cooling", "Broad_Shift")
FEATURE_NAMES = (
    "mean_c",
    "time_range_c",
    "time_slope_c_per_s",
    "mean_abs_dt",
    "mean_abs_dx",
    "var_column_resid",
    "frac_abs_z_ge_3_5",
    "length_m",
    "duration_s",
)
COPIED_MEDIAN = {
    "box_hit": {"Local_Heating": [46, 105], "Local_Cooling": [6, 25], "Broad_Shift": [22, 42], "total": [74, 172]},
    "quiet_false_alarm": [84, 540],
    "quiet_rate": 84 / 540,
    "nuisance": [94, 124],
    "zones": 709,
    "alarm_cells": [6967370, 136082815],
    "by_difficulty": {"hard": [8, 27], "easy": [63, 123], "medium": [3, 22]},
    "source": "copied from the earlier median_x3 halo pass, not recomputed",
}
COPIED_UNET = {
    "box_hit": {"Local_Heating": [43, 105], "Local_Cooling": [6, 25], "Broad_Shift": [21, 42], "total": [70, 172]},
    "quiet_false_alarm": [81, 540],
    "quiet_rate": 81 / 540,
    "nuisance": [91, 124],
    "zones": 418,
    "alarm_cells": [8338154, 136082815],
    "by_difficulty": {"hard": [8, 27], "easy": [58, 123], "medium": [4, 22]},
    "source": "copied from artifacts/classify/unet_halo_report.json, not recomputed",
}
KNOWN_MAE = {
    "raw": 0.61355,
    "median_x3": 0.53705,
    "conv1d_supervised": 0.48021,
    "conv_ae_bg": 0.42519,
    "unet2d_supervised": 0.38790,
    "valid_cells": 136082815,
    "source": "copied from artifacts/classify/denoise_celsius_report.json",
}


class Score:
    def __init__(self):
        self.hits = {}
        self.difficulty = {}
        self.quiet_wrong = 0
        self.quiet_n = 0
        self.nuisance_wrong = 0
        self.nuisance_n = 0
        self.alarm_cells = 0
        self.alarm_total = 0
        self.zones = 0
        self.zone_files = 0

    def event(self, mapped: str, level: str, hit: bool) -> None:
        slot = self.hits.setdefault(mapped, {"hit": 0, "total": 0})
        slot["total"] += 1
        slot["hit"] += int(hit)
        bucket = self.difficulty.setdefault(level or "unknown", {"hit": 0, "total": 0})
        bucket["total"] += 1
        bucket["hit"] += int(hit)

    def nuisance(self, called: bool) -> None:
        self.nuisance_n += 1
        self.nuisance_wrong += int(called)

    def quiet(self, called: bool) -> None:
        self.quiet_n += 1
        self.quiet_wrong += int(called)

    def as_dict(self) -> dict:
        total_hit = int(sum(item["hit"] for item in self.hits.values()))
        total = int(sum(item["total"] for item in self.hits.values()))
        rate = (self.quiet_wrong / self.quiet_n) if self.quiet_n else None
        return {
            "box_hit": self.hits,
            "box_hit_total": [total_hit, total],
            "by_difficulty": self.difficulty,
            "quiet_false_alarm": [self.quiet_wrong, self.quiet_n],
            "quiet_rate": rate,
            "nuisance_called_event": [self.nuisance_wrong, self.nuisance_n],
            "alarm_cells": [self.alarm_cells, self.alarm_total] if self.alarm_total else None,
            "zones": self.zones if self.zone_files else None,
        }


def _mae_store() -> dict:
    return {"abs": 0.0, "n": 0, "bg_abs": 0.0, "bg_n": 0, "ev_abs": 0.0, "ev_n": 0}


def _add_mae(store: dict, pred: np.ndarray, clean: np.ndarray, valid: np.ndarray, mask: np.ndarray) -> None:
    ok = np.asarray(valid, dtype=bool) & np.isfinite(pred) & np.isfinite(clean)
    if not np.any(ok):
        return
    err = np.abs(np.asarray(pred, dtype=np.float64)[ok] - np.asarray(clean, dtype=np.float64)[ok])
    store["abs"] += float(err.sum())
    store["n"] += int(ok.sum())
    bg = ok & (np.asarray(mask) == 0)
    ev = ok & (np.asarray(mask) > 0)
    if np.any(bg):
        store["bg_abs"] += float(np.abs(pred[bg] - clean[bg]).sum())
        store["bg_n"] += int(bg.sum())
    if np.any(ev):
        store["ev_abs"] += float(np.abs(pred[ev] - clean[ev]).sum())
        store["ev_n"] += int(ev.sum())


def _mae_dict(store: dict, note: str | None = None) -> dict:
    def ratio(total: float, count: int):
        return (total / count) if count else None

    out = {
        "mae_c": ratio(store["abs"], store["n"]),
        "cells": store["n"],
        "background_mae_c": ratio(store["bg_abs"], store["bg_n"]),
        "background_cells": store["bg_n"],
        "event_mae_c": ratio(store["ev_abs"], store["ev_n"]),
        "event_cells": store["ev_n"],
    }
    if note:
        out["note"] = note
    return out


def _guard(path: Path) -> Path:
    if path.name not in ALLOWED:
        raise RuntimeError(f"refusing unexpected weight name {path.name}")
    resolved = path.resolve()
    for item in PROTECTED:
        if item.exists() and resolved == item.resolve():
            raise RuntimeError(f"refusing to overwrite {item}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


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


def _locate(t_unix: np.ndarray, x: np.ndarray, row: dict) -> tuple[int, int, int, int]:
    i0 = int(np.searchsorted(t_unix, _unix(row["t_start"])))
    i1 = int(np.searchsorted(t_unix, _unix(row["t_end"]), side="right") - 1)
    j0 = int(np.searchsorted(x, float(row["dist_start_m"]) - 1.0))
    j1 = int(np.searchsorted(x, float(row["dist_end_m"]) + 1.0, side="right") - 1)
    i0 = max(0, i0 - 2)
    i1 = min(len(t_unix) - 1, max(i0, i1 + 2))
    j0 = min(max(0, j0), len(x) - 1)
    j1 = min(max(j0, j1), len(x) - 1)
    return i0, i1, j0, j1


def _cover(nt: int, nx: int, i0: int, i1: int, j0: int, j1: int) -> list[tuple[int, int]]:
    if nt < TIME_CROP or nx < SPACE_CROP:
        return []
    t_lo = max(0, min(i0, nt - TIME_CROP))
    t_hi = min(max(i1, 0), nt - TIME_CROP)
    x_lo = max(0, min(j0, nx - SPACE_CROP))
    x_hi = min(max(j1, 0), nx - SPACE_CROP)
    if t_hi < t_lo:
        t_lo = t_hi
    if x_hi < x_lo:
        x_lo = x_hi
    t_starts = list(range(t_lo, t_hi + 1, 14)) or [t_lo]
    x_starts = list(range(x_lo, x_hi + 1, 10)) or [x_lo]
    if len(t_starts) * len(x_starts) > 36:
        t_starts = t_starts[:: max(1, len(t_starts) // 6)]
        x_starts = x_starts[:: max(1, len(x_starts) // 6)]
    center = (int(np.clip((i0 + i1) // 2, 0, nt - TIME_CROP)), int(np.clip((j0 + j1) // 2, 0, nx - SPACE_CROP)))
    origins = [(t0, x0) for t0 in t_starts for x0 in x_starts]
    if center not in origins:
        origins.append(center)
    return origins[:36]


def _load_labels(root: Path) -> dict[str, dict[str, list]]:
    rows = list(csv.DictReader((root / "labels.csv").open(newline="", encoding="utf-8")))
    grouped = {"train": {}, "val": {}, "test": {}}
    for row in rows:
        grouped[row["split"]].setdefault(row["seq_id"], []).append(row)
    return grouped


def _limited(root: Path, split: str, limit: int | None):
    for index, item in enumerate(iter_split(root, split)):
        if limit is not None and index >= limit:
            break
        yield item


def _increasing(values: np.ndarray) -> bool:
    if len(values) < 2:
        return False
    return bool(np.all(np.diff(np.asarray(values, dtype=np.float64)) > 0))


def _draw_quiet(record: dict, rng: np.random.Generator, count: int) -> list[tuple[int, int]]:
    mask = record["mask_event"]
    valid = record["valid"]
    nt, nx = mask.shape
    found = []
    tries = 0
    while len(found) < count and tries < 200 and nt >= TIME_CROP and nx >= SPACE_CROP:
        tries += 1
        t0 = int(rng.integers(0, nt - TIME_CROP + 1))
        x0 = int(rng.integers(0, nx - SPACE_CROP + 1))
        if int(mask[t0:t0 + TIME_CROP, x0:x0 + SPACE_CROP].max()) != 0:
            continue
        if float(valid[t0:t0 + TIME_CROP, x0:x0 + SPACE_CROP].mean()) < 0.5:
            continue
        found.append((t0, x0))
    return found


def _tile_starts(length: int, size: int, stride: int) -> list[int]:
    if length < size:
        return []
    starts = list(range(0, length - size + 1, stride))
    last = length - size
    if starts[-1] != last:
        starts.append(last)
    return starts


def _fusion_crop(noisy: np.ndarray, valid: np.ndarray, sigma: np.ndarray, t0: int, x0: int) -> np.ndarray:
    patch = np.asarray(noisy[t0:t0 + TIME_CROP, x0:x0 + SPACE_CROP], dtype=np.float32)
    ok = np.asarray(valid[t0:t0 + TIME_CROP, x0:x0 + SPACE_CROP], dtype=bool)
    sig = np.maximum(np.asarray(sigma[x0:x0 + SPACE_CROP], dtype=np.float32), 1e-3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        base = np.nanmedian(np.where(ok, patch, np.nan), axis=0)
    base = np.where(np.isfinite(base), base, 0.0).astype(np.float32)
    base_grid = np.broadcast_to(base, patch.shape)
    sig_grid = np.broadcast_to(sig, patch.shape)
    resid = np.zeros(patch.shape, dtype=np.float32)
    resid[ok] = ((patch[ok] - base_grid[ok]) / sig_grid[ok]).astype(np.float32)
    dt = np.zeros(patch.shape, dtype=np.float32)
    dx = np.zeros(patch.shape, dtype=np.float32)
    both_t = ok[1:] & ok[:-1]
    if np.any(both_t):
        dt[1:][both_t] = ((patch[1:][both_t] - patch[:-1][both_t]) / sig_grid[1:][both_t]).astype(np.float32)
    both_x = ok[:, 1:] & ok[:, :-1]
    if np.any(both_x):
        dx[:, 1:][both_x] = ((patch[:, 1:][both_x] - patch[:, :-1][both_x]) / sig_grid[:, 1:][both_x]).astype(np.float32)
    return np.stack([resid, dt, dx], axis=0)


def _seg_target(mask: np.ndarray, valid: np.ndarray, x: np.ndarray, t0: int, x0: int) -> np.ndarray:
    view = np.asarray(mask[t0:t0 + 32, x0:x0 + 128])
    ok = np.asarray(valid[t0:t0 + 32, x0:x0 + 128], dtype=bool)
    target = np.full(view.shape, 255, dtype=np.int64)
    target[ok & (view == 0)] = 0
    target[ok & (view == 2)] = 1
    target[ok & ((view == 5) | (view == 6))] = 2
    sec = (view == 3) | (view == 4)
    if np.any(sec):
        touched = ((mask[t0:t0 + 32] == 3) | (mask[t0:t0 + 32] == 4)).any(axis=0)
        col_cls = np.zeros(len(x), dtype=np.int64)
        cursor = 0
        while cursor < len(touched):
            if not touched[cursor]:
                cursor += 1
                continue
            end = cursor + 1
            while end < len(touched) and touched[end]:
                end += 1
            width = float(x[end - 1] - x[cursor])
            col_cls[cursor:end] = 3 if width > BROAD_WIDTH_M else 1
            cursor = end
        grid = np.broadcast_to(col_cls[x0:x0 + 128], view.shape)
        take = ok & sec
        target[take] = grid[take]
    target[view == 1] = 255
    target[~ok] = 255
    return target


def _row_pack(noisy_row: np.ndarray, valid: np.ndarray, sigma: np.ndarray):
    sig = np.maximum(np.asarray(sigma, dtype=np.float32), 1e-3)
    base = float(np.median(noisy_row[valid])) if np.any(valid) else 0.0
    xin = np.zeros(noisy_row.shape, dtype=np.float32)
    xin[valid] = ((noisy_row[valid] - base) / sig[valid]).astype(np.float32)
    return xin, base, sig


def _vmd_modes(row: np.ndarray) -> np.ndarray:
    from vmdpy import VMD

    signal = np.where(np.isfinite(row), row, 0.0).astype(np.float64)
    modes, _hat, _omega = VMD(signal, 2000, 0, 4, 0, 1, 1e-6)
    modes = np.asarray(modes, dtype=np.float32)
    out = np.zeros((4, row.shape[0]), dtype=np.float32)
    count = min(4, modes.shape[0])
    out[:count, : modes.shape[1]] = modes[:count]
    return out


def _gadf(series: np.ndarray) -> np.ndarray:
    finite = np.asarray(series, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    grid = np.zeros(64, dtype=np.float64)
    if finite.size == 1:
        grid[:] = finite[0]
    elif finite.size > 1:
        grid = np.interp(np.linspace(0.0, 1.0, 64), np.linspace(0.0, 1.0, finite.size), finite)
    lo = float(grid.min())
    hi = float(grid.max())
    if hi - lo < 1e-6:
        scaled = np.zeros(64, dtype=np.float64)
    else:
        scaled = np.clip(2.0 * (grid - lo) / (hi - lo) - 1.0, -1.0, 1.0)
    phi = np.arccos(scaled)
    return np.sin(phi[:, None] - phi[None, :]).astype(np.float32)


def _spatial_mean(field: np.ndarray, valid: np.ndarray, i0: int, i1: int, j0: int, j1: int) -> np.ndarray:
    patch = field[i0:i1 + 1, j0:j1 + 1]
    ok = valid[i0:i1 + 1, j0:j1 + 1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.where(ok, patch, np.nan), axis=1)


def _zone_features(field, valid, score, x, t_unix, i0, i1, j0, j1) -> np.ndarray:
    patch = np.asarray(field[i0:i1 + 1, j0:j1 + 1], dtype=np.float32)
    ok = np.asarray(valid[i0:i1 + 1, j0:j1 + 1], dtype=bool)
    out = np.zeros(9, dtype=np.float32)
    out[7] = float(x[j1] - x[j0])
    out[8] = float(t_unix[i1] - t_unix[i0])
    if not np.any(ok):
        return out
    out[0] = float(patch[ok].mean())
    series = _spatial_mean(field, valid, i0, i1, j0, j1)
    finite = series[np.isfinite(series)]
    if finite.size:
        out[1] = float(finite.max() - finite.min())
    times = np.asarray(t_unix[i0:i1 + 1], dtype=np.float64)
    good = np.isfinite(series)
    if int(good.sum()) >= 2 and float(times[good][-1] - times[good][0]) != 0.0:
        slope = np.polyfit(times[good] - times[good][0], series[good], 1)[0]
        out[2] = float(slope)
    both_t = ok[1:] & ok[:-1]
    if np.any(both_t):
        out[3] = float(np.abs(patch[1:][both_t] - patch[:-1][both_t]).mean())
    both_x = ok[:, 1:] & ok[:, :-1]
    if np.any(both_x):
        out[4] = float(np.abs(patch[:, 1:][both_x] - patch[:, :-1][both_x]).mean())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        base = np.nanmedian(np.where(ok, patch, np.nan), axis=0)
    base = np.where(np.isfinite(base), base, 0.0).astype(np.float32)
    resid = patch - np.broadcast_to(base, patch.shape)
    out[5] = float(np.var(resid[ok]))
    zz = np.asarray(score[i0:i1 + 1, j0:j1 + 1])
    out[6] = float(np.mean(np.abs(zz[ok]) >= 3.5))
    out[~np.isfinite(out)] = 0.0
    return out


def _fill_columns(field: np.ndarray, valid: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        base = np.nanmedian(np.where(valid, field, np.nan), axis=0)
    base = np.where(np.isfinite(base), base, 0.0).astype(np.float32)
    return np.where(valid, field, np.broadcast_to(base.astype(np.float32), field.shape)).astype(np.float32)


def _opencv_alarm(field: np.ndarray, valid: np.ndarray, x: np.ndarray):
    import cv2

    filled = _fill_columns(np.asarray(field, dtype=np.float32), valid)
    blurred = cv2.GaussianBlur(np.ascontiguousarray(filled), (3, 3), 0)
    sobel_src = np.nan_to_num(blurred, nan=0.0).astype(np.float32)
    cv2.Sobel(sobel_src, cv2.CV_32F, 1, 0, ksize=3)
    cv2.Sobel(sobel_src, cv2.CV_32F, 0, 1, ksize=3)
    blurred = blurred.astype(np.float32)
    blurred[~valid] = np.nan
    score, _bg, _scale = spatial_z(blurred, valid, x, inner_m=8.0, outer_m=40.0, bin_m=2.0, floor_c=0.30)
    raw_alarm = (np.abs(score) >= 3.5).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(cv2.morphologyEx(raw_alarm, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)
    _count, _labels, _stats, _cent = cv2.connectedComponentsWithStats(closed, connectivity=8)
    survive = closed > 0
    hot = survive & (score >= 3.5)
    cold = survive & (score <= -3.5)
    return score, hot, cold, int(_count - 1)


def _signed_hit(hot: np.ndarray, cold: np.ndarray, i0: int, i1: int, j0: int, j1: int, mapped: str) -> bool:
    sl = (slice(i0, i1 + 1), slice(j0, j1 + 1))
    if mapped == "Local_Cooling":
        return bool(np.any(cold[sl]))
    if mapped == "Local_Heating":
        return bool(np.any(hot[sl]))
    return bool(np.any(hot[sl] | cold[sl]))


def _kalman(field: np.ndarray, valid: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    sig = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-3)
    process = (0.25 * sig) ** 2
    measure = sig ** 2
    out = np.full(field.shape, np.nan, dtype=np.float64)
    state = np.full(field.shape[1], np.nan, dtype=np.float64)
    variance = np.zeros(field.shape[1], dtype=np.float64)
    alive = np.zeros(field.shape[1], dtype=bool)
    for index in range(field.shape[0]):
        row = np.asarray(field[index], dtype=np.float64)
        ok = np.asarray(valid[index], dtype=bool) & np.isfinite(row)
        fresh = ok & ~alive
        update = ok & alive
        variance[alive] += process[alive]
        if np.any(update):
            gain = variance[update] / (variance[update] + measure[update])
            state[update] = state[update] + gain * (row[update] - state[update])
            variance[update] = (1.0 - gain) * variance[update]
            out[index, update] = state[update]
        state[fresh] = row[fresh]
        variance[fresh] = measure[fresh]
        out[index, fresh] = row[fresh]
        alive[fresh] = True
    return out.astype(np.float32)


def _macro_f1(truth: np.ndarray, predicted: np.ndarray) -> float:
    if len(truth) == 0:
        return 0.0
    scores = []
    for label in sorted(set(int(item) for item in truth)):
        tp = int(np.sum((truth == label) & (predicted == label)))
        fp = int(np.sum((truth != label) & (predicted == label)))
        fn = int(np.sum((truth == label) & (predicted != label)))
        if tp + fp + fn == 0:
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return float(np.mean(scores)) if scores else 0.0


def _stack(rows: list[np.ndarray]) -> np.ndarray:
    if not rows:
        return np.zeros((0, 1), dtype=np.float32)
    return np.stack(rows).astype(np.float32)


def _save_state(model, dest: Path, extra: dict) -> None:
    import torch

    blob = {key: int(value) for key, value in extra.items()}
    blob["state_dict"] = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save(blob, _guard(dest))


def _fit_huber(model, train_x, train_y, train_m, val_x, val_y, val_m, dest: Path, epochs: int, extra: dict) -> dict:
    import torch
    from torch import nn

    if len(train_x) == 0:
        return {"skipped": "no training crops"}
    device = "cuda"
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best = None
    best_loss = 1e9
    stale = 0
    history = []
    rng = np.random.default_rng(0)

    def loss_of(pred, target, mask):
        err = nn.functional.smooth_l1_loss(pred, target, reduction="none")
        return (err * mask).sum() / mask.sum().clamp(min=1.0)

    def run_loss(x, y, mask):
        xb = torch.from_numpy(x).to(device)
        yb = torch.from_numpy(y).to(device)
        mb = torch.from_numpy(mask).to(device)
        pred = model(xb)
        return float(loss_of(pred, yb, mb))

    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train_x))
        losses = []
        step = min(32, len(train_x))
        for start in range(0, len(train_x), step):
            pick = order[start:start + step]
            xb = torch.from_numpy(train_x[pick]).to(device)
            yb = torch.from_numpy(train_y[pick]).to(device)
            mb = torch.from_numpy(train_m[pick]).to(device)
            opt.zero_grad()
            loss = loss_of(model(xb), yb, mb)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            val_loss = run_loss(val_x, val_y, val_m) if len(val_x) else float(np.mean(losses))
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_huber": val_loss})
        print(f"train {dest.stem} epoch {epoch} loss {history[-1]['loss']:.4f} val_huber {val_loss:.4f}", flush=True)
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 3:
                break
    if best is None:
        best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best)
    _save_state(model, dest, extra)
    return {"val_huber": best_loss, "epochs_ran": history[-1]["epoch"], "history": history, "path": str(dest)}


def _fit_ce(model, train_x, train_y, val_x, val_y, dest: Path, epochs: int, batch: int, weight: np.ndarray, n_class: int) -> dict:
    import torch
    from torch import nn

    if len(train_y) == 0:
        return {"skipped": "no training windows"}
    device = "cuda"
    model = model.to(device)
    class_weight = torch.tensor(weight, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best = None
    best_score = -1.0
    stale = 0
    history = []
    rng = np.random.default_rng(0)
    train_y = np.asarray(train_y, dtype=np.int64)
    val_y = np.asarray(val_y, dtype=np.int64)

    def predict(x):
        pieces = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(x), batch):
                xb = torch.from_numpy(x[start:start + batch]).to(device)
                pieces.append(model(xb).argmax(dim=1).cpu().numpy())
        return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.int64)

    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train_y))
        losses = []
        step = min(batch, len(train_y))
        for start in range(0, len(train_y), step):
            pick = order[start:start + step]
            xb = torch.from_numpy(train_x[pick]).to(device)
            yb = torch.from_numpy(train_y[pick]).to(device)
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(xb), yb, weight=class_weight)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        score = _macro_f1(val_y, predict(val_x)) if len(val_y) else 0.0
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_macro_f1": score})
        print(f"train {dest.stem} epoch {epoch} loss {history[-1]['loss']:.4f} val_f1 {score:.4f}", flush=True)
        if score > best_score + 1e-4:
            best_score = score
            best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 3:
                break
    if best is None:
        best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best)
    _save_state(model, dest, {"n_class": n_class})
    return {"val_macro_f1": best_score, "epochs_ran": history[-1]["epoch"], "history": history, "path": str(dest)}


def _fit_seg(model, train_x, train_y, val_x, val_y, dest: Path, epochs: int) -> dict:
    import torch
    from torch import nn

    if len(train_y) == 0:
        return {"skipped": "no segmentation crops"}
    device = "cuda"
    model = model.to(device)
    weight = torch.tensor(SEG_WEIGHT, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best = None
    best_score = -1.0
    stale = 0
    history = []
    rng = np.random.default_rng(0)

    def pixel_f1(x, y):
        truth = []
        pred = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(x), 8):
                xb = torch.from_numpy(x[start:start + 8]).to(device)
                out = model(xb).argmax(dim=1).cpu().numpy()
                target = y[start:start + 8]
                keep = target != 255
                if np.any(keep):
                    truth.append(target[keep])
                    pred.append(out[keep])
        if not truth:
            return 0.0
        return _macro_f1(np.concatenate(truth), np.concatenate(pred))

    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train_y))
        losses = []
        for start in range(0, len(train_y), 8):
            pick = order[start:start + 8]
            xb = torch.from_numpy(train_x[pick]).to(device)
            yb = torch.from_numpy(train_y[pick]).to(device)
            if int((yb != 255).sum()) == 0:
                continue
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(xb), yb, weight=weight, ignore_index=255)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        score = pixel_f1(val_x, val_y) if len(val_y) else 0.0
        history.append({"epoch": epoch, "loss": float(np.mean(losses)) if losses else None, "val_pixel_macro_f1": score})
        loss_text = "nan" if history[-1]["loss"] is None else f"{history[-1]['loss']:.4f}"
        print(f"train {dest.stem} epoch {epoch} loss {loss_text} val_pixel_f1 {score:.4f}", flush=True)
        if score > best_score + 1e-4:
            best_score = score
            best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 3:
                break
    if best is None:
        best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best)
    _save_state(model, dest, {"n_class": 4})
    return {"val_pixel_macro_f1": best_score, "epochs_ran": history[-1]["epoch"], "history": history, "path": str(dest)}


def _empty_bags() -> dict:
    return {
        "dil_x": [], "dil_y": [], "dil_m": [],
        "vmd_x": [], "vmd_y": [], "vmd_m": [], "vmd_base": [], "vmd_sig": [], "vmd_clean": [],
        "fus_x": [], "fus_y": [],
        "seg_x": [], "seg_y": [],
        "gadf_x": [], "gadf_y": [],
        "zone_x": [], "zone_y": [],
    }


def _collect(root: Path, split: str, limit: int | None, labels: dict, rng: np.random.Generator, yolo_root: Path | None, vmd_state: dict) -> dict:
    bags = _empty_bags()
    per_file_quiet = 8 if yolo_root is None or split != "test" else 0
    for seq_id, record in _limited(root, split, limit):
        if not _increasing(record["t_unix"]) or not _increasing(record["x"]):
            print(f"skip axis {split} {seq_id}", flush=True)
            continue
        noisy = np.asarray(record["noisy"], dtype=np.float32)
        clean = np.asarray(record["clean"], dtype=np.float32)
        valid = record["valid"]
        sigma = record["sigma_x"]
        mask = record["mask_event"]
        nt, nx = noisy.shape
        cleaned = median_x(noisy, valid, k=3)
        score, _bg, _scale = spatial_z(cleaned, valid, record["x"], inner_m=8.0, outer_m=40.0, bin_m=2.0, floor_c=0.30)
        _collect_dilated(bags, noisy, clean, valid, sigma, rng, vmd_state)
        _collect_fusion(bags, record, rng)
        _collect_seg(bags, noisy, clean, valid, sigma, mask, record["x"], rng)
        _collect_boxes(bags, labels.get(seq_id, []), cleaned, valid, score, record, rng, yolo_root, split, seq_id)
        if per_file_quiet:
            for t0, x0 in _draw_quiet(record, rng, 8):
                bags["zone_x"].append(_zone_features(cleaned, valid, score, record["x"], record["t_unix"], t0, t0 + TIME_CROP - 1, x0, x0 + SPACE_CROP - 1))
                bags["zone_y"].append(0)
                if len(bags["gadf_x"]) < 4000:
                    bags["gadf_x"].append(_gadf(_spatial_mean(noisy, valid, t0, t0 + TIME_CROP - 1, x0, x0 + SPACE_CROP - 1)))
                    bags["gadf_y"].append(0)
        print(
            f"cache {split} {seq_id} dil {len(bags['dil_x'])} vmd {len(bags['vmd_x'])} "
            f"fus {len(bags['fus_y'])} seg {len(bags['seg_y'])} gadf {len(bags['gadf_y'])} zone {len(bags['zone_y'])}",
            flush=True,
        )
        del noisy, clean, cleaned, score
    return bags


def _collect_dilated(bags, noisy, clean, valid, sigma, rng, vmd_state) -> None:
    nt, nx = noisy.shape
    if nx < 1024:
        return
    width = 1024
    got = 0
    tries = 0
    while got < 8 and tries < 80 and len(bags["dil_x"]) < 2000:
        tries += 1
        row = int(rng.integers(0, nt))
        x0 = 0 if nx <= width else int(rng.integers(0, nx - width + 1))
        ok = valid[row, x0:x0 + width]
        if float(ok.mean()) < 0.7:
            continue
        xin, _base, _sig = _row_pack(noisy[row, x0:x0 + width], ok, sigma[x0:x0 + width])
        _yt, target, mask = _row_norm(noisy[row, x0:x0 + width], clean[row, x0:x0 + width], ok, sigma[x0:x0 + width])
        bags["dil_x"].append(xin[None, :])
        bags["dil_y"].append(target[None, :])
        bags["dil_m"].append(mask[None, :])
        got += 1
        if len(bags["vmd_x"]) >= vmd_state["cap"]:
            continue
        if vmd_state["seconds"] is None or vmd_state["probes"] < 2:
            started = time.perf_counter()
            try:
                modes = _vmd_modes(xin)
            except Exception as exc:
                vmd_state["error"] = str(exc)
                vmd_state["cap"] = 0
                print(f"VMD failed: {exc}", flush=True)
                continue
            elapsed = time.perf_counter() - started
            vmd_state["probes"] += 1
            vmd_state["seconds"] = elapsed if vmd_state["seconds"] is None else (vmd_state["seconds"] + elapsed) / 2
            print(f"vmd probe {vmd_state['probes']} {elapsed:.3f}s", flush=True)
            if vmd_state["seconds"] > 1.0:
                vmd_state["cap"] = min(vmd_state["cap"], 128)
                vmd_state["slow"] = True
                print("vmd cache capped at 128", flush=True)
        else:
            if vmd_state.get("error"):
                continue
            try:
                modes = _vmd_modes(xin)
            except Exception as exc:
                vmd_state["error"] = str(exc)
                vmd_state["cap"] = 0
                continue
        bags["vmd_x"].append(modes)
        bags["vmd_y"].append(target[None, :])
        bags["vmd_m"].append(mask[None, :])
        bags["vmd_base"].append(_base)
        bags["vmd_sig"].append(_sig.astype(np.float32))
        bags["vmd_clean"].append(np.asarray(clean[row, x0:x0 + width], dtype=np.float32))


def _collect_fusion(bags, record, rng) -> None:
    grouped = sample_windows(record, rng, 8, "columns")
    noisy = record["noisy"]
    valid = record["valid"]
    sigma = record["sigma_x"]
    for name, origins in grouped.items():
        for t0, x0 in origins:
            bags["fus_x"].append(_fusion_crop(noisy, valid, sigma, int(t0), int(x0)))
            bags["fus_y"].append(LABELS.index(name))


def _collect_seg(bags, noisy, clean, valid, sigma, mask, x, rng) -> None:
    nt, nx = noisy.shape
    if nt < 32 or nx < 128 or len(bags["seg_x"]) >= 2000:
        return
    got = 0
    tries = 0
    while got < 8 and tries < 80:
        tries += 1
        t0 = int(rng.integers(0, nt - 31))
        x0 = int(rng.integers(0, nx - 127))
        sl = (slice(t0, t0 + 32), slice(x0, x0 + 128))
        if float(valid[sl].mean()) < 0.9:
            continue
        xin, _yin, _ok = _denoise_norm(noisy[sl], clean[sl], valid[sl], sigma[x0:x0 + 128])
        bags["seg_x"].append(xin[None, :])
        bags["seg_y"].append(_seg_target(mask, valid, x, t0, x0))
        got += 1


def _collect_boxes(bags, rows, cleaned, valid, score, record, rng, yolo_root, split, seq_id) -> None:
    noisy = record["noisy"]
    x = record["x"]
    t_unix = record["t_unix"]
    image_dir = None
    label_dir = None
    if yolo_root is not None and split in ("train", "val"):
        image_dir = yolo_root / "images" / split
        label_dir = yolo_root / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
    quiet_for_yolo = _draw_quiet(record, rng, 2) if image_dir is not None else []
    for t0, x0 in quiet_for_yolo:
        _write_yolo(image_dir, label_dir, f"{seq_id}_q_{t0}_{x0}", noisy, record["valid"], t0, t0 + 64, x0, x0 + 128, None, None)
    for row in rows:
        if row["kind"] == "event" and row["label"] == "tracing":
            continue
        mapped = _box_name(row) if row["kind"] == "event" else None
        if row["kind"] == "event" and mapped is None:
            continue
        if row["kind"] not in ("event", "nuisance"):
            continue
        i0, i1, j0, j1 = _locate(t_unix, x, row)
        label = 0 if mapped is None else LABELS.index(mapped)
        bags["zone_x"].append(_zone_features(cleaned, record["valid"], score, x, t_unix, i0, i1, j0, j1))
        bags["zone_y"].append(label)
        if mapped is not None and len(bags["gadf_x"]) < 4000:
            bags["gadf_x"].append(_gadf(_spatial_mean(noisy, record["valid"], i0, i1, j0, j1)))
            bags["gadf_y"].append(label)
        if image_dir is not None and mapped is not None:
            window = _yolo_window(noisy.shape[0], noisy.shape[1], i0, i1, j0, j1, rng, True)
            _write_yolo(image_dir, label_dir, f"{seq_id}_{row['event_id']}_{i0}", noisy, record["valid"], *window, (i0, i1, j0, j1), mapped)


def _yolo_window(nt, nx, i0, i1, j0, j1, rng, jitter):
    th = min(nt, max(32, i1 - i0 + 1))
    tw = min(nx, max(64, j1 - j0 + 1))
    center_t = (i0 + i1) // 2
    center_x = (j0 + j1) // 2
    if jitter and rng is not None:
        center_t += int(rng.integers(-4, 5))
        center_x += int(rng.integers(-8, 9))
    t0 = int(np.clip(center_t - th // 2, 0, max(0, nt - th)))
    x0 = int(np.clip(center_x - tw // 2, 0, max(0, nx - tw)))
    return t0, t0 + th, x0, x0 + tw


def _write_yolo(image_dir, label_dir, stem, noisy, valid, t0, t1, x0, x1, box, mapped) -> None:
    import cv2

    nt, nx = noisy.shape
    t0 = max(0, min(t0, nt - 1))
    x0 = max(0, min(x0, nx - 1))
    t1 = max(t0 + 1, min(t1, nt))
    x1 = max(x0 + 1, min(x1, nx))
    patch = np.asarray(noisy[t0:t1, x0:x1], dtype=np.float32)
    ok = np.asarray(valid[t0:t1, x0:x1], dtype=bool)
    image = np.zeros(patch.shape, dtype=np.uint8)
    if np.any(ok):
        lo, hi = np.percentile(patch[ok], [1, 99])
        if hi > lo:
            scaled = np.zeros(patch.shape, dtype=np.float32)
            scaled[ok] = np.clip((patch[ok] - lo) / (hi - lo), 0, 1)
            image = (scaled * 255).astype(np.uint8)
    cv2.imwrite(str(image_dir / f"{stem}.png"), image)
    lines = []
    if box is not None and mapped in YOLO_NAMES:
        bi0, bi1, bj0, bj1 = box
        th = max(1, t1 - t0)
        tw = max(1, x1 - x0)
        cx = ((bj0 + bj1) / 2 - x0) / tw
        cy = ((bi0 + bi1) / 2 - t0) / th
        bw = (bj1 - bj0 + 1) / tw
        bh = (bi1 - bi0 + 1) / th
        cx = float(np.clip(cx, 0, 1))
        cy = float(np.clip(cy, 0, 1))
        bw = float(np.clip(bw, 0, 1))
        bh = float(np.clip(bh, 0, 1))
        lines.append(f"{YOLO_NAMES.index(mapped)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    (label_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")


def _as_batch(rows: list[np.ndarray], shape_tail: tuple[int, ...]) -> np.ndarray:
    if not rows:
        return np.zeros((0, *shape_tail), dtype=np.float32)
    return np.stack(rows).astype(np.float32)


def _xgb_predict(model, features: np.ndarray) -> np.ndarray:
    import xgboost as xgb

    if len(features) == 0:
        return np.zeros(0, dtype=np.int64)
    probs = model.predict(xgb.DMatrix(np.asarray(features, dtype=np.float32)))
    probs = np.asarray(probs)
    if probs.ndim == 1:
        return probs.astype(np.int64)
    return probs.argmax(axis=1).astype(np.int64)


def _fit_xgb(train_x, train_y, val_x, val_y, dest: Path) -> dict:
    import xgboost as xgb

    if len(train_y) == 0:
        return {"skipped": "no zone rows"}
    weight = CLASS_WEIGHT[np.asarray(train_y, dtype=np.int64)]
    dtrain = xgb.DMatrix(np.asarray(train_x, dtype=np.float32), label=np.asarray(train_y, dtype=np.int64), weight=weight)
    dval = xgb.DMatrix(np.asarray(val_x, dtype=np.float32), label=np.asarray(val_y, dtype=np.int64)) if len(val_y) else None

    def boost(device: str):
        params = {
            "objective": "multi:softprob",
            "num_class": 6,
            "max_depth": 4,
            "eta": 0.08,
            "tree_method": "hist",
            "device": device,
            "eval_metric": "mlogloss",
            "seed": 0,
        }
        kwargs = {"num_boost_round": 300, "verbose_eval": False}
        if dval is not None:
            kwargs["evals"] = [(dval, "val")]
            kwargs["early_stopping_rounds"] = 25
        return xgb.train(params, dtrain, **kwargs)

    fit_note = "cuda"
    try:
        model = boost("cuda")
    except Exception as exc:
        fit_note = f"cpu after cuda fit failed: {exc}"
        model = boost("cpu")
    best_iter = getattr(model, "best_iteration", None)
    if dval is not None:
        pred = _xgb_predict(model, val_x)
        score = _macro_f1(np.asarray(val_y, dtype=np.int64), pred)
    else:
        score = 0.0
    model.save_model(str(_guard(dest)))
    print(f"train zone_xgb val_f1 {score:.4f} trees {best_iter} {fit_note}", flush=True)
    return {"val_macro_f1": score, "best_iteration": best_iter, "device_note": fit_note, "path": str(dest), "model": model}


def _shap_report(model, val_x: np.ndarray) -> dict:
    gain = {}
    try:
        raw = {str(key): float(value) for key, value in model.get_score(importance_type="gain").items()}
        gain = {}
        for key, value in raw.items():
            if key.startswith("f") and key[1:].isdigit() and int(key[1:]) < len(FEATURE_NAMES):
                gain[FEATURE_NAMES[int(key[1:])]] = value
            else:
                gain[key] = value
    except Exception as exc:
        gain = {"error": str(exc)}
    try:
        import shap

        take = min(256, len(val_x))
        if take == 0:
            return {"feature_gain": gain, "shap_error": "empty val"}
        values = shap.TreeExplainer(model).shap_values(val_x[:take])
        if isinstance(values, list):
            stacked = np.stack([np.abs(np.asarray(item)) for item in values], axis=0)
            mean_abs = stacked.mean(axis=(0, 1))
        else:
            array = np.abs(np.asarray(values))
            if array.ndim == 3:
                array = array.mean(axis=-1)
            mean_abs = array.mean(axis=0)
        return {
            "feature_gain": gain,
            "shap_mean_abs_val": {name: float(value) for name, value in zip(FEATURE_NAMES, mean_abs)},
            "shap_rows": take,
        }
    except Exception as exc:
        return {"feature_gain": gain, "shap_error": str(exc)}


def _clusters(train_x, train_y, test_x, test_y) -> dict:
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    report = {}
    if len(train_x) < 4 or len(test_x) == 0:
        return {"skipped": "not enough zone rows"}
    mu = train_x.mean(axis=0)
    sig = train_x.std(axis=0)
    sig = np.where(sig < 1e-6, 1.0, sig)
    train_z = (train_x - mu) / sig
    test_z = (test_x - mu) / sig
    kmeans = KMeans(n_clusters=4, random_state=0, n_init=10)
    kmeans.fit(train_z)
    pred = kmeans.predict(test_z)
    report["kmeans"] = {
        "agreement": _majority(test_y, pred),
        "ari": float(adjusted_rand_score(test_y, pred)),
        "noise_fraction": 0.0,
    }
    try:
        import umap
        import hdbscan

        neighbors = min(15, len(train_z) - 1)
        reducer = umap.UMAP(n_components=2, random_state=0, n_neighbors=neighbors)
        embedded = reducer.fit_transform(train_z)
        test_embedded = reducer.transform(test_z)
        size = min(15, max(2, len(train_z) // 20))
        cluster = hdbscan.HDBSCAN(min_cluster_size=size, prediction_data=True)
        cluster.fit(embedded)
        labels, _strength = hdbscan.approximate_predict(cluster, test_embedded)
        labels = np.asarray(labels)
        noise = float(np.mean(labels < 0))
        report["umap_hdbscan"] = {
            "agreement": _majority(test_y, labels),
            "ari": float(adjusted_rand_score(test_y, labels)),
            "noise_fraction": noise,
            "min_cluster_size": size,
        }
    except Exception as exc:
        report["umap_hdbscan"] = {"error": str(exc)}
    return report


def _majority(truth: np.ndarray, clusters: np.ndarray) -> float | None:
    known = np.asarray(clusters) >= 0
    if not np.any(known):
        return None
    pred = np.full(len(truth), -1, dtype=np.int64)
    for cid in np.unique(clusters[known]):
        sel = clusters == cid
        values, counts = np.unique(truth[sel], return_counts=True)
        pred[sel] = int(values[int(np.argmax(counts))])
    return float(np.mean(pred[known] == truth[known]))


def _ultralytics_cache() -> None:
    try:
        from ultralytics import settings

        dest = ROOT / "artifacts" / "dts_synth_v1" / "cache_summary"
        dest.mkdir(parents=True, exist_ok=True)
        settings.update({"weights_dir": str(dest)})
    except Exception:
        pass


def _train_yolo(yolo_root: Path, run_root: Path, dest: Path, epochs: int) -> dict:
    _ultralytics_cache()
    train_images = list((yolo_root / "images" / "train").glob("*.png")) if (yolo_root / "images" / "train").exists() else []
    val_images = list((yolo_root / "images" / "val").glob("*.png")) if (yolo_root / "images" / "val").exists() else []
    if not train_images:
        return {"skipped": "no yolo images"}
    val_key = "images/val" if val_images else "images/train"
    yaml = yolo_root / "waterfall.yaml"
    yaml.write_text(
        "\n".join([
            f"path: {yolo_root.as_posix()}",
            "train: images/train",
            f"val: {val_key}",
            "names:",
            "  0: Local_Heating",
            "  1: Local_Cooling",
            "  2: Broad_Shift",
            "",
        ]),
        encoding="utf-8",
    )
    try:
        from ultralytics import YOLO
    except Exception as exc:
        return {"error": str(exc)}
    cached_weights = ROOT / "artifacts" / "dts_synth_v1" / "cache_summary" / "yolov8n.pt"
    downloaded = ROOT / "yolov8n.pt"
    try:
        if cached_weights.exists():
            model = YOLO(str(cached_weights))
            source = str(cached_weights)
        elif downloaded.exists():
            model = YOLO(str(downloaded))
            source = str(downloaded)
        else:
            model = YOLO("yolov8n.pt")
            source = "yolov8n.pt"
    except Exception as exc:
        print(f"yolo weights download failed: {exc}", flush=True)
        model = YOLO("yolov8n.yaml")
        source = "yolov8n.yaml"
    model.train(
        data=str(yaml),
        epochs=epochs,
        imgsz=256,
        workers=0,
        device=0,
        project=str(run_root),
        name="waterfall",
        plots=False,
        exist_ok=True,
        mosaic=0.0,
        mixup=0.0,
        fliplr=0.0,
        flipud=0.0,
        scale=0.0,
        translate=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        verbose=False,
        batch=8,
    )
    best = run_root / "waterfall" / "weights" / "best.pt"
    if not best.exists():
        candidates = list(run_root.rglob("best.pt"))
        if not candidates:
            return {"error": "yolo best.pt missing", "source": source}
        best = candidates[0]
    shutil.copyfile(best, _guard(dest))
    print(f"train yolo_waterfall copied {best}", flush=True)
    return {"path": str(dest), "source": source, "epochs": epochs}


def _load_torch(path: Path, factory):
    import torch

    blob = torch.load(path, map_location="cuda", weights_only=True)
    model = factory(blob)
    model.load_state_dict(blob["state_dict"])
    model.to("cuda")
    model.eval()
    return model


def _reconstruct_unet(noisy, valid, sigma, model) -> np.ndarray:
    import torch

    nt, nx = noisy.shape
    acc = np.zeros((nt, nx), dtype=np.float64)
    weight = np.zeros((nt, nx), dtype=np.float64)
    coords = [(t0, x0) for t0 in _tile_starts(nt, 32, 16) for x0 in _tile_starts(nx, 128, 64)]
    out = np.full((nt, nx), np.nan, dtype=np.float32)
    if not coords:
        return out
    with torch.no_grad():
        for start in range(0, len(coords), 16):
            batch = []
            meta = []
            for t0, x0 in coords[start:start + 16]:
                patch = noisy[t0:t0 + 32, x0:x0 + 128]
                ok = valid[t0:t0 + 32, x0:x0 + 128]
                sig = np.maximum(np.asarray(sigma[x0:x0 + 128], dtype=np.float32), 1e-3)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    base = np.nanmedian(np.where(ok, patch, np.nan), axis=0)
                base = np.where(np.isfinite(base), base, 0.0).astype(np.float32)
                xin = np.zeros_like(patch, dtype=np.float32)
                base_grid = np.broadcast_to(base, patch.shape)
                sig_grid = np.broadcast_to(sig, patch.shape)
                xin[ok] = ((patch[ok] - base_grid[ok]) / sig_grid[ok]).astype(np.float32)
                batch.append(xin)
                meta.append((t0, x0, base, sig, ok))
            pred = model(torch.from_numpy(np.stack(batch)[:, None]).to("cuda")).squeeze(1).cpu().numpy()
            for tile, (t0, x0, base, sig, ok) in zip(pred, meta):
                celsius = tile * sig[None, :] + base[None, :]
                acc[t0:t0 + 32, x0:x0 + 128][ok] += celsius[ok]
                weight[t0:t0 + 32, x0:x0 + 128][ok] += 1
    mask = weight > 0
    out[mask] = (acc[mask] / weight[mask]).astype(np.float32)
    return out


def _reconstruct_dilated(noisy, valid, sigma, model) -> np.ndarray:
    import torch

    nt, nx = noisy.shape
    acc = np.zeros((nt, nx), dtype=np.float64)
    weight = np.zeros((nt, nx), dtype=np.float64)
    width = 1024 if nx >= 1024 else nx
    stride = 512 if nx >= 1024 else max(width, 1)
    starts = _tile_starts(nx, width, stride) or [0]
    out = np.full((nt, nx), np.nan, dtype=np.float32)
    batch = []
    meta = []

    def flush():
        if not batch:
            return
        with torch.no_grad():
            pred = model(torch.from_numpy(np.stack(batch)).to("cuda")).squeeze(1).cpu().numpy()
        for tile, (row, x0, base, sig, ok) in zip(pred, meta):
            celsius = tile * sig + base
            acc[row, x0:x0 + width][ok] += celsius[ok]
            weight[row, x0:x0 + width][ok] += 1
        batch.clear()
        meta.clear()

    for row in range(nt):
        for x0 in starts:
            ok = valid[row, x0:x0 + width]
            if float(ok.mean()) < 0.5 or int(ok.sum()) < 8:
                continue
            xin, base, sig = _row_pack(noisy[row, x0:x0 + width], ok, sigma[x0:x0 + width])
            batch.append(xin[None, :])
            meta.append((row, x0, base, sig, ok))
            if len(batch) == 32:
                flush()
    flush()
    mask = weight > 0
    out[mask] = (acc[mask] / weight[mask]).astype(np.float32)
    return out


def _opencv_nlm(noisy, valid) -> np.ndarray:
    import cv2

    values = noisy[valid]
    out = np.full(noisy.shape, np.nan, dtype=np.float32)
    if values.size == 0:
        return out
    lo, hi = np.percentile(values, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.where(valid, noisy, np.nan).astype(np.float32)
    scaled = _fill_columns(np.clip((noisy - lo) / (hi - lo), 0, 1) * 255, valid)
    den = cv2.fastNlMeansDenoising(np.ascontiguousarray(scaled.astype(np.uint8)), None, 10, 7, 21)
    restored = den.astype(np.float32) / 255.0 * (hi - lo) + lo
    out[valid] = restored[valid]
    return out


def _predict_windows(model, crops: list[np.ndarray], batch: int) -> np.ndarray:
    import torch

    if not crops:
        return np.zeros(0, dtype=np.int64)
    pieces = []
    array = np.stack(crops).astype(np.float32)
    with torch.no_grad():
        for start in range(0, len(array), batch):
            pred = model(torch.from_numpy(array[start:start + batch]).to("cuda")).argmax(dim=1).cpu().numpy()
            pieces.append(pred)
    return np.concatenate(pieces)


def _seg_hits(model, noisy, valid, sigma, mask, boxes, quiet) -> tuple[list[bool], list[bool]]:
    import torch

    hits = []
    quiet_flags = []
    with torch.no_grad():
        for i0, i1, j0, j1, mapped in boxes:
            hits.append(_seg_box(model, noisy, valid, sigma, i0, i1, j0, j1, mapped))
        for t0, x0 in quiet:
            called = _seg_box(model, noisy, valid, sigma, t0, t0 + TIME_CROP - 1, x0, x0 + SPACE_CROP - 1, "Local_Heating")
            window = _seg_any_event(model, noisy, valid, sigma, t0, x0)
            quiet_flags.append(window)
            del called
    return hits, quiet_flags


def _seg_tiles(nt, nx, i0, i1, j0, j1):
    if nt < 32 or nx < 128:
        return []
    t_starts = [item for item in _tile_starts(nt, 32, 16) if item <= i1 and item + 32 > i0]
    x_starts = [item for item in _tile_starts(nx, 128, 64) if item <= j1 and item + 128 > j0]
    if not t_starts:
        t_starts = [int(np.clip(i0, 0, nt - 32))]
    if not x_starts:
        x_starts = [int(np.clip(j0, 0, nx - 128))]
    return [(t0, x0) for t0 in t_starts[:4] for x0 in x_starts[:4]]


def _seg_box(model, noisy, valid, sigma, i0, i1, j0, j1, mapped) -> bool:
    import torch

    wanted = {"Local_Heating": 1, "Local_Cooling": 2, "Broad_Shift": 3}.get(mapped, 1)
    found = False
    for t0, x0 in _seg_tiles(noisy.shape[0], noisy.shape[1], i0, i1, j0, j1):
        xin, _yin, _ok = _denoise_norm(noisy[t0:t0 + 32, x0:x0 + 128], noisy[t0:t0 + 32, x0:x0 + 128], valid[t0:t0 + 32, x0:x0 + 128], sigma[x0:x0 + 128])
        pred = model(torch.from_numpy(xin[None, None]).to("cuda")).argmax(dim=1)[0].cpu().numpy()
        r0 = max(0, i0 - t0)
        r1 = min(32, i1 - t0 + 1)
        c0 = max(0, j0 - x0)
        c1 = min(128, j1 - x0 + 1)
        if r1 > r0 and c1 > c0 and np.any(pred[r0:r1, c0:c1] == wanted):
            found = True
            break
    return found


def _seg_any_event(model, noisy, valid, sigma, t0, x0) -> bool:
    import torch

    if noisy.shape[0] < 32 or noisy.shape[1] < 128:
        return False
    tt = int(np.clip(t0, 0, noisy.shape[0] - 32))
    xx = int(np.clip(x0, 0, noisy.shape[1] - 128))
    xin, _yin, _ok = _denoise_norm(
        noisy[tt:tt + 32, xx:xx + 128], noisy[tt:tt + 32, xx:xx + 128], valid[tt:tt + 32, xx:xx + 128], sigma[xx:xx + 128],
    )
    pred = model(torch.from_numpy(xin[None, None]).to("cuda")).argmax(dim=1)[0].cpu().numpy()
    r0 = max(0, t0 - tt)
    c0 = max(0, x0 - xx)
    window = pred[r0:r0 + TIME_CROP, c0:c0 + SPACE_CROP]
    return bool(np.any(window > 0))


def _ae_threshold(root: Path, limit: int | None, model) -> float:
    import torch

    errors = []
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for _seq_id, record in _limited(root, "train", limit):
            noisy = record["noisy"]
            valid = record["valid"]
            sigma = record["sigma_x"]
            mask = record["mask_event"]
            nt, nx = noisy.shape
            if nt < 8 or nx < 64:
                continue
            got = 0
            tries = 0
            while got < 4 and tries < 40 and len(errors) < 200000:
                tries += 1
                t0 = int(rng.integers(0, nt - 7))
                x0 = int(rng.integers(0, nx - 63))
                sl = (slice(t0, t0 + 8), slice(x0, x0 + 64))
                if float(valid[sl].mean()) < 0.9 or int(mask[sl].max()) != 0:
                    continue
                xin, _yin, ok = _denoise_norm(noisy[sl], noisy[sl], valid[sl], sigma[x0:x0 + 64])
                pred = model(torch.from_numpy(xin[None, None]).to("cuda"))[0, 0, :8, :64].cpu().numpy()
                err = np.abs(pred - xin)[ok.astype(bool)]
                if err.size:
                    errors.append(err.astype(np.float32))
                got += 1
    if not errors:
        return float("inf")
    return float(np.percentile(np.concatenate(errors), 95))


def _ae_region(model, noisy, valid, sigma, i0, i1, j0, j1, threshold: float) -> bool:
    import torch

    nt, nx = noisy.shape
    for t0 in _tile_starts(nt, 8, 8) or [0]:
        if t0 > i1 or t0 + 8 <= i0:
            continue
        for x0 in _tile_starts(nx, 64, 64) or [0]:
            if x0 > j1 or x0 + 64 <= j0:
                continue
            xin, _yin, ok = _denoise_norm(noisy[t0:t0 + 8, x0:x0 + 64], noisy[t0:t0 + 8, x0:x0 + 64], valid[t0:t0 + 8, x0:x0 + 64], sigma[x0:x0 + 64])
            with torch.no_grad():
                pred = model(torch.from_numpy(xin[None, None]).to("cuda"))[0, 0, :8, :64].cpu().numpy()
            err = np.abs(pred - xin)
            r0 = max(0, i0 - t0)
            r1 = min(8, i1 - t0 + 1)
            c0 = max(0, j0 - x0)
            c1 = min(64, j1 - x0 + 1)
            view = ok[r0:r1, c0:c1].astype(bool) & (err[r0:r1, c0:c1] > threshold)
            if np.any(view):
                return True
    return False


def _score_file_boxes(score: Score, rows, t_unix, x, nt, nx, decide) -> None:
    for row in rows:
        if row["kind"] == "event" and row["label"] == "tracing":
            continue
        mapped = _box_name(row) if row["kind"] == "event" else None
        if row["kind"] == "event" and mapped is None:
            continue
        if row["kind"] not in ("event", "nuisance"):
            continue
        i0, i1, j0, j1 = _locate(t_unix, x, row)
        called = decide(i0, i1, j0, j1, mapped)
        if row["kind"] == "event":
            score.event(mapped, row.get("difficulty", ""), called)
        else:
            score.nuisance(called)


def _vmd_crop_mae(model, bags, store: dict) -> None:
    import torch

    if not bags["vmd_x"]:
        return
    with torch.no_grad():
        for start in range(0, len(bags["vmd_x"]), 16):
            stop = start + 16
            pred = model(torch.from_numpy(np.stack(bags["vmd_x"][start:stop])).to("cuda")).squeeze(1).cpu().numpy()
            for tile, base, sig, clean, mask in zip(
                pred, bags["vmd_base"][start:stop], bags["vmd_sig"][start:stop], bags["vmd_clean"][start:stop], bags["vmd_m"][start:stop],
            ):
                celsius = tile * sig + base
                ok = mask[0] > 0.5
                if not np.any(ok):
                    continue
                err = np.abs(celsius[ok] - clean[ok])
                store["abs"] += float(err.sum())
                store["n"] += int(ok.sum())


def _prepare_arrays(bags: dict) -> dict:
    out = {}
    out["dil_x"] = _as_batch(bags["dil_x"], (1, 1024)) if bags["dil_x"] and bags["dil_x"][0].shape[-1] == 1024 else _stack(bags["dil_x"])
    out["dil_y"] = _stack(bags["dil_y"])
    out["dil_m"] = _stack(bags["dil_m"])
    out["vmd_x"] = _stack(bags["vmd_x"]) if bags["vmd_x"] else np.zeros((0, 4, 1024), np.float32)
    out["vmd_y"] = _stack(bags["vmd_y"]) if bags["vmd_y"] else np.zeros((0, 1, 1024), np.float32)
    out["vmd_m"] = _stack(bags["vmd_m"]) if bags["vmd_m"] else np.zeros((0, 1, 1024), np.float32)
    out["fus_x"] = _stack(bags["fus_x"]) if bags["fus_x"] else np.zeros((0, 3, TIME_CROP, SPACE_CROP), np.float32)
    out["fus_y"] = np.asarray(bags["fus_y"], dtype=np.int64)
    out["seg_x"] = _stack(bags["seg_x"]) if bags["seg_x"] else np.zeros((0, 1, 32, 128), np.float32)
    out["seg_y"] = np.stack(bags["seg_y"]).astype(np.int64) if bags["seg_y"] else np.zeros((0, 32, 128), np.int64)
    out["gadf_x"] = _as_batch([item[None, :] for item in bags["gadf_x"]], (1, 64, 64)) if bags["gadf_x"] else np.zeros((0, 1, 64, 64), np.float32)
    out["gadf_y"] = np.asarray(bags["gadf_y"], dtype=np.int64)
    out["zone_x"] = np.stack(bags["zone_x"]).astype(np.float32) if bags["zone_x"] else np.zeros((0, 9), np.float32)
    out["zone_y"] = np.asarray(bags["zone_y"], dtype=np.int64)
    return out


def _pad_width(array: np.ndarray, width: int) -> np.ndarray:
    if array.size == 0 or array.shape[-1] == width:
        return array
    out = np.zeros(array.shape[:-1] + (width,), dtype=array.dtype)
    out[..., : array.shape[-1]] = array
    return out


def train_models(root: Path, limit: int | None, epochs: int | None, weight_dir: Path, cache: Path) -> dict:
    labels = _load_labels(root)
    vmd_state = {"cap": 128 if limit else 2000, "seconds": None, "probes": 0, "slow": False, "error": None}
    if limit:
        vmd_state["cap"] = 32
    train_bags = _collect(root, "train", limit, labels["train"], np.random.default_rng(0), cache / "yolo", vmd_state)
    val_bags = _collect(root, "val", limit, labels["val"], np.random.default_rng(1), cache / "yolo", vmd_state)
    train = _prepare_arrays(train_bags)
    val = _prepare_arrays(val_bags)
    if train["dil_x"].size and train["dil_x"].shape[-1] != 1024:
        train["dil_x"] = _pad_width(train["dil_x"], 1024)
        train["dil_y"] = _pad_width(train["dil_y"], 1024)
        train["dil_m"] = _pad_width(train["dil_m"], 1024)
        val["dil_x"] = _pad_width(val["dil_x"], 1024)
        val["dil_y"] = _pad_width(val["dil_y"], 1024)
        val["dil_m"] = _pad_width(val["dil_m"], 1024)
    dilated_epochs = 8 if epochs is None else epochs
    fusion_epochs = 12 if epochs is None else epochs
    report = {"vmd": {key: value for key, value in vmd_state.items() if key != "error"} | {"error": vmd_state["error"]}}
    report["dilated1d_supervised"] = _fit_huber(
        DilatedDenoiser(1), train["dil_x"], train["dil_y"], train["dil_m"], val["dil_x"], val["dil_y"], val["dil_m"],
        weight_dir / "dilated1d_supervised.pt", dilated_epochs, {"in_channels": 1, "hidden": 32},
    )
    if vmd_state["error"] or len(train_bags["vmd_x"]) == 0:
        report["vmd_dilated1d"] = {"skipped": vmd_state["error"] or "no vmd crops"}
    else:
        report["vmd_dilated1d"] = _fit_huber(
            DilatedDenoiser(4), train["vmd_x"], train["vmd_y"], train["vmd_m"], val["vmd_x"], val["vmd_y"], val["vmd_m"],
            weight_dir / "vmd_dilated1d.pt", dilated_epochs, {"in_channels": 4, "hidden": 32},
        )
    report["gradient_fusion"] = _fit_ce(
        GradientFusion(), train["fus_x"], train["fus_y"], val["fus_x"], val["fus_y"],
        weight_dir / "gradient_fusion.pt", fusion_epochs, 64, CLASS_WEIGHT, 6,
    )
    report["seg_unet"] = _fit_seg(
        SegUNet(), train["seg_x"], train["seg_y"], val["seg_x"], val["seg_y"], weight_dir / "seg_unet.pt", dilated_epochs,
    )
    try:
        import timm

        gadf = timm.create_model("efficientnet_b0", pretrained=False, in_chans=1, num_classes=6)
        report["gadf_efficientnet"] = _fit_ce(
            gadf, train["gadf_x"], train["gadf_y"], val["gadf_x"], val["gadf_y"],
            weight_dir / "gadf_efficientnet.pt", dilated_epochs, 16, CLASS_WEIGHT, 6,
        )
    except Exception as exc:
        report["gadf_efficientnet"] = {"error": str(exc)}
    try:
        xgb_report = _fit_xgb(train["zone_x"], train["zone_y"], val["zone_x"], val["zone_y"], weight_dir / "zone_xgb.json")
        model = xgb_report.pop("model", None)
        if model is not None:
            xgb_report["shap"] = _shap_report(model, val["zone_x"])
        report["zone_xgb"] = xgb_report
    except Exception as exc:
        report["zone_xgb"] = {"error": str(exc)}
    np.savez(cache / "zones_train.npz", x=train["zone_x"], y=train["zone_y"])
    np.savez(cache / "zones_val.npz", x=val["zone_x"], y=val["zone_y"])
    report["yolo_waterfall"] = _train_yolo(cache / "yolo", cache / "yolo_runs", weight_dir / "yolo_waterfall.pt", dilated_epochs)
    report["vmd_test_bags"] = {
        "x": train_bags["vmd_x"][:0],
    }
    report["_vmd_state"] = vmd_state
    report["_train_zone"] = train["zone_x"]
    report["_train_zone_y"] = train["zone_y"]
    return report


def _beats(pack: dict) -> bool:
    total = pack.get("box_hit_total") or [0, 0]
    rate = pack.get("quiet_rate")
    return bool(total[1] == 172 and total[0] > 74 and rate is not None and rate <= 0.16)


def score_test(root: Path, limit: int | None, weight_dir: Path, cache: Path, trained: dict) -> dict:
    labels = _load_labels(root)
    vmd_state = trained.get("_vmd_state", {"seconds": None, "slow": False, "error": None, "cap": 0})
    mae = {name: _mae_store() for name in ("dilated1d", "vmd_dilated1d", "wavelet_x", "nlm_wide", "opencv_nlm", "unet_sanity", "kalman")}
    packs = {name: Score() for name in (
        "median_opencv", "unet_opencv", "unet_kalman_halo", "conv_ae_bg", "gradient_fusion", "seg_unet",
        "gadf_efficientnet", "zone_xgb", "yolo_waterfall",
    )}
    notes = {"nlm_wide": "every 10th row", "opencv_nlm": "uint8 round trip, h=10, template 7, search 21"}
    errors = {}
    tracing = 0
    skipped = []
    rng = np.random.default_rng(2)
    vmd_crops = 0
    vmd_crop_cap = 128 if vmd_state.get("slow") or (vmd_state.get("seconds") or 0) > 1 else 360

    def try_load(name, factory):
        path = weight_dir / name
        if not path.exists():
            return None
        try:
            return _load_torch(path, factory)
        except Exception as exc:
            errors[name] = str(exc)
            return None

    dilated = try_load("dilated1d_supervised.pt", lambda blob: DilatedDenoiser(int(blob["in_channels"]), int(blob["hidden"])))
    vmd_net = try_load("vmd_dilated1d.pt", lambda blob: DilatedDenoiser(int(blob["in_channels"]), int(blob["hidden"])))
    fusion = try_load("gradient_fusion.pt", lambda _blob: GradientFusion())
    seg = try_load("seg_unet.pt", lambda _blob: SegUNet(4))
    gadf = None
    if (weight_dir / "gadf_efficientnet.pt").exists():
        try:
            import timm
            gadf = try_load("gadf_efficientnet.pt", lambda _blob: timm.create_model("efficientnet_b0", pretrained=False, in_chans=1, num_classes=6))
        except Exception as exc:
            errors["gadf_efficientnet"] = str(exc)
    xgb_model = None
    if (weight_dir / "zone_xgb.json").exists():
        try:
            import xgboost as xgb
            xgb_model = xgb.Booster()
            xgb_model.load_model(str(weight_dir / "zone_xgb.json"))
        except Exception as exc:
            errors["zone_xgb"] = str(exc)
    yolo = None
    if (weight_dir / "yolo_waterfall.pt").exists():
        try:
            from ultralytics import YOLO
            yolo = YOLO(str(weight_dir / "yolo_waterfall.pt"))
        except Exception as exc:
            errors["yolo_waterfall"] = str(exc)
    ae = None
    ae_threshold = None
    ae_path = EXPERIMENTAL / "conv_ae_bg.pt"
    if ae_path.exists():
        try:
            from fiber.models.conv_ae import ConvBackgroundAE
            ae = _load_torch(ae_path, lambda _blob: ConvBackgroundAE())
            ae_threshold = _ae_threshold(root, limit, ae)
        except Exception as exc:
            errors["conv_ae_bg"] = str(exc)
    unet = None
    unet_path = EXPERIMENTAL / "unet2d_supervised.pt"
    if unet_path.exists():
        try:
            from fiber.models.unet2d_denoise import UNet2DDenoiser
            unet = _load_torch(unet_path, lambda _blob: UNet2DDenoiser())
        except Exception as exc:
            errors["unet2d_supervised"] = str(exc)

    test_zone_x = []
    test_zone_y = []
    for seq_id, record in _limited(root, "test", limit):
        if not _increasing(record["t_unix"]) or not _increasing(record["x"]):
            skipped.append(seq_id)
            continue
        noisy = np.asarray(record["noisy"], dtype=np.float32)
        clean = np.asarray(record["clean"], dtype=np.float32)
        valid = record["valid"]
        sigma = record["sigma_x"]
        mask = record["mask_event"]
        x = record["x"]
        t_unix = record["t_unix"]
        nt, nx = noisy.shape
        rows = labels["test"].get(seq_id, [])
        tracing += sum(1 for row in rows if row["kind"] == "event" and row["label"] == "tracing")
        quiet = _draw_quiet(record, rng, 12)
        cleaned = median_x(noisy, valid, k=3)
        try:
            hot_m, cold_m, zones_m = _opencv_parts(cleaned, valid, x)
            _score_masks(packs["median_opencv"], rows, t_unix, x, quiet, hot_m, cold_m, valid)
            packs["median_opencv"].zones += zones_m
            packs["median_opencv"].zone_files += 1
            packs["median_opencv"].alarm_cells += int(np.sum(hot_m | cold_m))
            packs["median_opencv"].alarm_total += int(valid.sum())
        except Exception as exc:
            errors["median_opencv"] = str(exc)
        unet_field = None
        if unet is not None:
            try:
                unet_field = _reconstruct_unet(noisy, valid, sigma, unet)
                _add_mae(mae["unet_sanity"], unet_field, clean, valid, mask)
                hot_u, cold_u, zones_u = _opencv_parts(np.where(np.isfinite(unet_field), unet_field, noisy), valid & np.isfinite(unet_field), x)
                live = valid & np.isfinite(unet_field)
                _score_masks(packs["unet_opencv"], rows, t_unix, x, quiet, hot_u, cold_u, live)
                packs["unet_opencv"].zones += zones_u
                packs["unet_opencv"].zone_files += 1
                packs["unet_opencv"].alarm_cells += int(np.sum((hot_u | cold_u) & live))
                packs["unet_opencv"].alarm_total += int(live.sum())
                kalman = _kalman(unet_field, np.isfinite(unet_field), sigma)
                _add_mae(mae["kalman"], kalman, clean, valid & np.isfinite(kalman), mask)
                kz, _bg, _scale = spatial_z(np.where(np.isfinite(kalman), kalman, 0), np.isfinite(kalman), x, inner_m=8, outer_m=40, bin_m=2, floor_c=0.30)
                khot = np.isfinite(kalman) & (kz >= 3.5)
                kcold = np.isfinite(kalman) & (kz <= -3.5)
                _score_masks(packs["unet_kalman_halo"], rows, t_unix, x, quiet, khot, kcold, np.isfinite(kalman))
                zone_n = _zone_count(kz, np.isfinite(kalman), x, t_unix)
                if zone_n is not None:
                    packs["unet_kalman_halo"].zones += zone_n
                    packs["unet_kalman_halo"].zone_files += 1
                packs["unet_kalman_halo"].alarm_cells += int(np.sum(khot | kcold))
                packs["unet_kalman_halo"].alarm_total += int(np.isfinite(kalman).sum())
            except Exception as exc:
                errors["unet_mix"] = str(exc)
        if dilated is not None:
            try:
                pred = _reconstruct_dilated(noisy, valid, sigma, dilated)
                _add_mae(mae["dilated1d"], pred, clean, valid, mask)
            except Exception as exc:
                errors["dilated1d"] = str(exc)
        if vmd_net is not None and vmd_state.get("error") is None and vmd_crops < vmd_crop_cap:
            try:
                vmd_crops += _score_vmd_rows(
                    vmd_net, noisy, clean, valid, mask, sigma, mae["vmd_dilated1d"], min(8, vmd_crop_cap - vmd_crops),
                )
            except Exception as exc:
                errors["vmd_dilated1d"] = str(exc)
        try:
            _add_mae(mae["wavelet_x"], wavelet_x(noisy, valid), clean, valid, mask)
        except Exception as exc:
            errors["wavelet_x"] = str(exc)
        try:
            rows_idx = np.arange(0, nt, 10)
            pred = np.full_like(noisy, np.nan)
            pred[rows_idx] = nlm_wide(noisy[rows_idx], valid[rows_idx])
            _add_mae(mae["nlm_wide"], pred, clean, valid, mask)
        except Exception as exc:
            errors["nlm_wide"] = str(exc)
        try:
            _add_mae(mae["opencv_nlm"], _opencv_nlm(noisy, valid), clean, valid, mask)
        except Exception as exc:
            errors["opencv_nlm"] = str(exc)
            notes["opencv_nlm"] = f"failed: {exc}"
        if ae is not None and ae_threshold is not None:
            try:
                _score_ae(packs["conv_ae_bg"], ae, noisy, valid, sigma, rows, t_unix, x, quiet, ae_threshold)
            except Exception as exc:
                errors["conv_ae_bg"] = str(exc)
        if fusion is not None:
            try:
                _score_fusion(packs["gradient_fusion"], fusion, noisy, valid, sigma, rows, t_unix, x, quiet)
            except Exception as exc:
                errors["gradient_fusion"] = str(exc)
        if seg is not None:
            try:
                _score_seg(packs["seg_unet"], seg, noisy, valid, sigma, rows, t_unix, x, quiet)
            except Exception as exc:
                errors["seg_unet"] = str(exc)
        if gadf is not None:
            try:
                _score_gadf(packs["gadf_efficientnet"], gadf, noisy, valid, rows, t_unix, x, quiet)
            except Exception as exc:
                errors["gadf_efficientnet"] = str(exc)
        if xgb_model is not None:
            try:
                zone_score, _bg, _scale = spatial_z(cleaned, valid, x, inner_m=8, outer_m=40, bin_m=2, floor_c=0.30)
                _score_xgb(packs["zone_xgb"], xgb_model, cleaned, valid, zone_score, x, t_unix, rows, quiet, test_zone_x, test_zone_y)
            except Exception as exc:
                errors["zone_xgb"] = str(exc)
        if yolo is not None:
            try:
                _score_yolo(packs["yolo_waterfall"], yolo, noisy, valid, rows, t_unix, x, quiet, seq_id)
            except Exception as exc:
                errors["yolo_waterfall"] = str(exc)
        print(f"test {seq_id} quiet {len(quiet)}", flush=True)
    cluster = {}
    train_path = cache / "zones_train.npz"
    if train_path.exists() and test_zone_x:
        try:
            saved = np.load(train_path)
            cluster = _clusters(saved["x"], saved["y"], np.stack(test_zone_x), np.asarray(test_zone_y, dtype=np.int64))
        except Exception as exc:
            cluster = {"error": str(exc)}
    packs_out = {name: pack.as_dict() for name, pack in packs.items()}
    packs_out["median_x3_spatial_z_halo"] = COPIED_MEDIAN
    packs_out["unet2d_spatial_z_halo"] = COPIED_UNET
    better = [name for name, pack in packs_out.items() if name not in ("median_x3_spatial_z_halo", "unet2d_spatial_z_halo") and _beats(pack)]
    if (vmd_state.get("seconds") or 0) > 1 or vmd_state.get("slow"):
        notes["vmd_dilated1d"] = f"MAE on capped crops because one VMD probe averaged {vmd_state.get('seconds')} s. Not the full valid-cell count."
    elif vmd_net is not None:
        notes["vmd_dilated1d"] = "MAE on row crops, not claimed equal to the 136082815-cell table unless cells match."
    return {
        "mae_c": {name: _mae_dict(store, notes.get(name)) for name, store in mae.items()},
        "known_mae_c": KNOWN_MAE,
        "alarms": packs_out,
        "better_than_halo_on_boxes_and_quiet": better,
        "clusters": cluster,
        "errors": errors,
        "tracing_boxes_excluded": tracing,
        "skipped_time_order": skipped,
        "ae_threshold_normalized": ae_threshold,
        "operational_pair_unchanged": "median_x3 + spatial_z_halo at |Z| >= 3.5. A higher synth score does not switch it.",
    }


def _opencv_parts(field, valid, x):
    _score, hot, cold, zones = _opencv_alarm(field, valid, x)
    return hot, cold, zones


def _score_masks(pack: Score, rows, t_unix, x, quiet, hot, cold, valid) -> None:
    def decide(i0, i1, j0, j1, mapped):
        if mapped is None:
            sl = (slice(i0, i1 + 1), slice(j0, j1 + 1))
            return bool(np.any(hot[sl] | cold[sl]))
        return _signed_hit(hot, cold, i0, i1, j0, j1, mapped)

    _score_file_boxes(pack, rows, t_unix, x, valid.shape[0], valid.shape[1], decide)
    for t0, x0 in quiet:
        sl = (slice(t0, t0 + TIME_CROP), slice(x0, x0 + SPACE_CROP))
        pack.quiet(bool(np.any(hot[sl] | cold[sl])))


def _zone_count(score, valid, x, t_unix):
    times = np.asarray(t_unix, dtype=np.float64)
    iso = [str(index) for index in range(len(times))]
    try:
        zones = extract_zones(score, valid, x, iso, times, thresh=3.5, min_width_m=3, min_profiles=6, min_cells=24, max_gap_s=300)
    except ValueError:
        return None
    return len(zones)


def _score_vmd_rows(model, noisy, clean, valid, mask, sigma, store, max_crops: int) -> int:
    import torch

    nt, nx = noisy.shape
    if nx < 1024 or max_crops <= 0:
        return 0
    width = 1024
    x0 = (nx - width) // 2
    taken = 0
    with torch.no_grad():
        for row in range(0, nt, 40):
            if taken >= max_crops or taken >= 8:
                break
            ok = valid[row, x0:x0 + width]
            if float(ok.mean()) < 0.7:
                continue
            xin, base, sig = _row_pack(noisy[row, x0:x0 + width], ok, sigma[x0:x0 + width])
            modes = _vmd_modes(xin)
            pred = model(torch.from_numpy(modes[None]).to("cuda")).squeeze().cpu().numpy()
            celsius = pred * sig + base
            _add_mae(
                store,
                celsius[None, :],
                clean[row:row + 1, x0:x0 + width],
                ok[None, :],
                mask[row:row + 1, x0:x0 + width],
            )
            taken += 1
    return taken


def _score_ae(pack, model, noisy, valid, sigma, rows, t_unix, x, quiet, threshold) -> None:
    def decide(i0, i1, j0, j1, _mapped):
        return _ae_region(model, noisy, valid, sigma, i0, i1, j0, j1, threshold)

    _score_file_boxes(pack, rows, t_unix, x, noisy.shape[0], noisy.shape[1], decide)
    for t0, x0 in quiet:
        pack.quiet(_ae_region(model, noisy, valid, sigma, t0, t0 + TIME_CROP - 1, x0, x0 + SPACE_CROP - 1, threshold))


def _score_fusion(pack, model, noisy, valid, sigma, rows, t_unix, x, quiet) -> None:
    crops = []
    spans = []
    for row in rows:
        if row["kind"] == "event" and row["label"] == "tracing":
            continue
        mapped = _box_name(row) if row["kind"] == "event" else None
        if row["kind"] == "event" and mapped is None:
            continue
        if row["kind"] not in ("event", "nuisance"):
            continue
        i0, i1, j0, j1 = _locate(t_unix, x, row)
        origins = _cover(noisy.shape[0], noisy.shape[1], i0, i1, j0, j1)
        start = len(crops)
        for t0, x0 in origins:
            crops.append(_fusion_crop(noisy, valid, sigma, t0, x0))
        spans.append((row, mapped, start, len(crops)))
    qstart = len(crops)
    for t0, x0 in quiet:
        crops.append(_fusion_crop(noisy, valid, sigma, t0, x0))
    pred = _predict_windows(model, crops, 64)
    for row, mapped, start, stop in spans:
        called = pred[start:stop]
        if row["kind"] == "event":
            pack.event(mapped, row.get("difficulty", ""), bool(np.any(called == LABELS.index(mapped))))
        else:
            pack.nuisance(bool(np.any(called != 0)))
    if quiet:
        pack.quiet_n += len(quiet)
        pack.quiet_wrong += int(np.sum(pred[qstart:] != 0))


def _score_seg(pack, model, noisy, valid, sigma, rows, t_unix, x, quiet) -> None:
    for row in rows:
        if row["kind"] == "event" and row["label"] == "tracing":
            continue
        mapped = _box_name(row) if row["kind"] == "event" else None
        if row["kind"] == "event" and mapped is None:
            continue
        if row["kind"] not in ("event", "nuisance"):
            continue
        i0, i1, j0, j1 = _locate(t_unix, x, row)
        if row["kind"] == "event":
            pack.event(mapped, row.get("difficulty", ""), _seg_box(model, noisy, valid, sigma, i0, i1, j0, j1, mapped))
        else:
            pack.nuisance(_seg_any_region(model, noisy, valid, sigma, i0, i1, j0, j1))
    for t0, x0 in quiet:
        pack.quiet(_seg_any_event(model, noisy, valid, sigma, t0, x0))


def _seg_any_region(model, noisy, valid, sigma, i0, i1, j0, j1) -> bool:
    return any(
        _seg_box(model, noisy, valid, sigma, i0, i1, j0, j1, name)
        for name in YOLO_NAMES
    )


def _score_gadf(pack, model, noisy, valid, rows, t_unix, x, quiet) -> None:
    images = []
    metas = []
    for row in rows:
        if row["kind"] == "event" and row["label"] == "tracing":
            continue
        mapped = _box_name(row) if row["kind"] == "event" else None
        if row["kind"] == "event" and mapped is None:
            continue
        if row["kind"] not in ("event", "nuisance"):
            continue
        i0, i1, j0, j1 = _locate(t_unix, x, row)
        images.append(_gadf(_spatial_mean(noisy, valid, i0, i1, j0, j1))[None, :])
        metas.append((row, mapped))
    qstart = len(images)
    for t0, x0 in quiet:
        images.append(_gadf(_spatial_mean(noisy, valid, t0, t0 + TIME_CROP - 1, x0, x0 + SPACE_CROP - 1))[None, :])
    pred = _predict_windows(model, images, 16)
    for index, (row, mapped) in enumerate(metas):
        called = int(pred[index])
        if row["kind"] == "event":
            pack.event(mapped, row.get("difficulty", ""), called == LABELS.index(mapped))
        else:
            pack.nuisance(called != 0)
    if quiet:
        pack.quiet_n += len(quiet)
        pack.quiet_wrong += int(np.sum(pred[qstart:] != 0))


def _score_xgb(pack, model, field, valid, score, x, t_unix, rows, quiet, test_x, test_y) -> None:
    features = []
    metas = []
    for row in rows:
        if row["kind"] == "event" and row["label"] == "tracing":
            continue
        mapped = _box_name(row) if row["kind"] == "event" else None
        if row["kind"] == "event" and mapped is None:
            continue
        if row["kind"] not in ("event", "nuisance"):
            continue
        i0, i1, j0, j1 = _locate(t_unix, x, row)
        feat = _zone_features(field, valid, score, x, t_unix, i0, i1, j0, j1)
        features.append(feat)
        metas.append((row, mapped))
        test_x.append(feat)
        test_y.append(0 if mapped is None else LABELS.index(mapped))
    qstart = len(features)
    for t0, x0 in quiet:
        features.append(_zone_features(field, valid, score, x, t_unix, t0, t0 + TIME_CROP - 1, x0, x0 + SPACE_CROP - 1))
    if not features:
        return
    pred = _xgb_predict(model, np.stack(features))
    for index, (row, mapped) in enumerate(metas):
        called = int(pred[index])
        if row["kind"] == "event":
            pack.event(mapped, row.get("difficulty", ""), called == LABELS.index(mapped))
        else:
            pack.nuisance(called != 0)
    if quiet:
        pack.quiet_n += len(quiet)
        pack.quiet_wrong += int(np.sum(pred[qstart:] != 0))


def _score_yolo(pack, model, noisy, valid, rows, t_unix, x, quiet, seq_id) -> None:
    del seq_id

    def predict_crop(t0, t1, x0, x1):
        patch = np.asarray(noisy[t0:t1, x0:x1], dtype=np.float32)
        ok = np.asarray(valid[t0:t1, x0:x1], dtype=bool)
        image = np.zeros(patch.shape, dtype=np.uint8)
        if np.any(ok):
            lo, hi = np.percentile(patch[ok], [1, 99])
            if hi > lo:
                scaled = np.zeros(patch.shape, dtype=np.float32)
                scaled[ok] = np.clip((patch[ok] - lo) / (hi - lo), 0, 1)
                image = (scaled * 255).astype(np.uint8)
        result = model.predict(image, conf=0.25, verbose=False, device=0, imgsz=256)
        boxes = []
        if result and result[0].boxes is not None and len(result[0].boxes):
            xyxy = result[0].boxes.xyxy.cpu().numpy()
            cls = result[0].boxes.cls.cpu().numpy().astype(int)
            for rect, kind in zip(xyxy, cls):
                if kind < 0 or kind >= len(YOLO_NAMES):
                    continue
                boxes.append((YOLO_NAMES[kind], rect))
        return boxes

    def intersects(rect, i0, i1, j0, j1, t0, x0):
        x1, y1, x2, y2 = rect
        fi0 = t0 + y1
        fi1 = t0 + y2
        fj0 = x0 + x1
        fj1 = x0 + x2
        return not (fi1 < i0 or i1 < fi0 or fj1 < j0 or j1 < fj0)

    for row in rows:
        if row["kind"] == "event" and row["label"] == "tracing":
            continue
        mapped = _box_name(row) if row["kind"] == "event" else None
        if row["kind"] == "event" and mapped is None:
            continue
        if row["kind"] not in ("event", "nuisance"):
            continue
        i0, i1, j0, j1 = _locate(t_unix, x, row)
        t0, t1, x0, x1 = _yolo_window(noisy.shape[0], noisy.shape[1], i0, i1, j0, j1, None, False)
        found = predict_crop(t0, t1, x0, x1)
        if row["kind"] == "event":
            hit = any(name == mapped and intersects(rect, i0, i1, j0, j1, t0, x0) for name, rect in found)
            pack.event(mapped, row.get("difficulty", ""), hit)
        else:
            pack.nuisance(any(intersects(rect, i0, i1, j0, j1, t0, x0) for _name, rect in found))
    for index, (t0, x0) in enumerate(quiet):
        t1 = min(noisy.shape[0], t0 + TIME_CROP)
        x1 = min(noisy.shape[1], x0 + SPACE_CROP)
        found = predict_crop(t0, t1, x0, x1)
        pack.quiet(bool(found))


def _public_train(report: dict) -> dict:
    hidden = {"_vmd_state", "_train_zone", "_train_zone_y", "vmd_test_bags"}
    return {key: value for key, value in report.items() if key not in hidden}


def main() -> None:
    warnings.filterwarnings("ignore", message="All-NaN slice encountered")
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    import torch

    if not torch.cuda.is_available():
        print("ABORT cuda is not available", flush=True)
        raise SystemExit(2)
    print(torch.cuda.get_device_name(0), flush=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "artifacts" / "dts_synth_v1")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--stage", choices=("all", "train", "score"), default="all")
    args = parser.parse_args()
    smoke = args.limit is not None
    weight_dir = Path(tempfile.gettempdir()) / "summary_pipeline_smoke" if smoke else EXPERIMENTAL
    cache = args.root / "cache_summary" / ("smoke" if smoke else "full")
    cache.mkdir(parents=True, exist_ok=True)
    weight_dir.mkdir(parents=True, exist_ok=True)
    trained = {}
    if args.stage in ("all", "train"):
        trained = train_models(args.root, args.limit, args.epochs, weight_dir, cache)
        torch.save({"keep": 1}, cache / "_train_marker.pt")
    scored = {}
    if args.stage in ("all", "score"):
        if args.stage == "score":
            trained = {"_vmd_state": {"seconds": None, "slow": True, "error": None, "cap": 128}}
        scored = score_test(args.root, args.limit, weight_dir, cache, trained)
    payload = {
        "source": "dts_synth_v1",
        "device": torch.cuda.get_device_name(0),
        "smoke": smoke,
        "limit": args.limit,
        "epochs": args.epochs,
        "weights": str(weight_dir),
        "models": _public_train(trained),
        **scored,
        "not_trained": [
            "RADNet", "R-OTDR", "1D-DCRNN", "MSAFR-D", "two-branch GAN", "Stokes VMD",
            "Neighbor2Neighbor", "Noise2Noise", "PSSDN", "conduction Kalman", "thermal-resistance GNN", "RepOD",
        ],
    }
    if smoke:
        dest = cache / "smoke_report.json"
        dest.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print("SMOKE_DONE", flush=True)
    else:
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print("SCORE_DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"ABORT {exc}", flush=True)
        raise
