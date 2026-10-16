"""Matched-FAR thresholds for a denoised field. Report only.

analyze_field stays on median_x3 and spatial_z at |Z| >= 3.5. Nothing here
writes an operational channel, threshold, or zone field.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from fiber.anomaly.detect import spatial_availability, spatial_background, spatial_z
from fiber.anomaly.refine import (
    refine_zones,
    truth_extent_errors,
    zones_to_mask,
)
from fiber.anomaly.zones import extract_zones
from fiber.benchmark.events import alarm_fraction, evaluate_events
from fiber.benchmark.synthetic import background_scene, half_cable_shift, local_scene_var
from fiber.denoise import apply_method
from fiber.pipeline import PipelineConfig

ROOT = Path(__file__).resolve().parents[2]
AMP_GRID = (1.0, 2.0, 6.0)
WIDTH_GRID = (3, 7, 15)
RAMP_GRID = (0, 4)
NOISE = 0.12
FLOOR_C = 0.30
OPERATIONAL_Z = 3.5
ZSN2N_SEED = 0
ZSN2N_STEPS = 4
PREFER_THRESHOLD = 3.5
THRESHOLD_GRID = np.round(np.arange(0.5, 12.01, 0.05), 2)
IOU_THRESHOLDS = np.round(np.arange(2.0, 8.01, 0.25), 2)
SCALES = ("mad_profile", "mad_residual_global", "mad_raw_residual")
METHODS = ("median_x3", "n2n_tsr", "zsn2n")
THRESHOLD_RULE = "grid point closest to 3.5 among those with FAR closest to the target"
CANDIDATE = "кандидат на продвижение"
REJECTED = "не принят"
BASELINE = "baseline"
FAIL_REAL = "нет оценки на MGTS_data/1–6.csv"
FAIL_WITNESS = "нет оценки свидетелей"
FAIL_SHA = "sha256 свидетеля не совпал"
FAIL_UNSTABLE = "нестабилен на реальных данных"
FAIL_WEAK = "нет прироста на слабых событиях (амплитуда ≤ 2 °C) на 0.05 IoU или recall"
FAIL_HOT = "held-out hot IoU ниже median_x3 на 3.5 больше чем на 0.01"
FAIL_COLD = "held-out cold IoU ниже median_x3 на 3.5 больше чем на 0.01"
FAIL_FAR_CAP = "holdout FAR выше 1.1× FAR median_x3"
FAIL_RATIO = "holdout FAR/target вне [0.5, 2]"
FAIL_STD = "bootstrap std порога > 0.25"
FAIL_HALF = "half_cable_shift alarm fraction внутри широкой маски выше median_x3"
FAIL_DEAD = "dead alarms не 0"
FAIL_WITNESS_COUNT = "witness pass count ниже median_x3"
FAIL_H_HOT = "held-out hot IoU ниже median_x3 больше чем на 0.005"
FAIL_H_COLD = "held-out cold IoU ниже median_x3 больше чем на 0.005"
FAIL_H_RAMP = "IoU на ramp=0 упал"
FAIL_H_PEAK = "медианная ошибка пикового контраста не ниже на 20%"
FAIL_H_AMP = "группа амплитуды хуже больше чем на 5%"
FAIL_H_X = "медианная ошибка границы по x на ramp=0 хуже median_x3"
FAIL_H_STEP = "медианная ошибка границы по x на ramp=0 больше одного шага сетки"
FAIL_H_T = "ошибка границы по времени хуже median_x3"
FAIL_H_SHIFT = "медианный сдвиг границы > 2 м"
FAIL_H_SIGN = "зона сменила знак контраста"
REAL_FILES = tuple(f"{index}.csv" for index in range(1, 7))


def parse_seed_spec(spec: str) -> list[int]:
    seeds: list[int] = []
    for part in str(spec).split(","):
        piece = part.strip()
        if not piece:
            continue
        if "-" in piece:
            left, right = piece.split("-", 1)
            start, stop = int(left), int(right)
            if stop < start:
                raise ValueError(f"seed range {piece!r} is empty")
            seeds.extend(range(start, stop + 1))
        else:
            seeds.append(int(piece))
    if not seeds:
        raise ValueError("empty seed spec")
    ordered: list[int] = []
    seen: set[int] = set()
    for seed in seeds:
        if seed not in seen:
            ordered.append(seed)
            seen.add(seed)
    return ordered


def assert_disjoint(left: list[int], right: list[int]) -> None:
    overlap = sorted(set(left) & set(right))
    if overlap:
        raise AssertionError(f"seed lists overlap: {overlap}")


def variant_grid() -> list[dict]:
    rows = []
    for amp in AMP_GRID:
        for width in WIDTH_GRID:
            for ramp in RAMP_GRID:
                rows.append({
                    "amp_hot": float(amp),
                    "amp_cold": float(-amp),
                    "width": int(width),
                    "ramp": int(ramp),
                    "noise": NOISE,
                })
    return rows


def far(z: np.ndarray, valid: np.ndarray, truth_any: np.ndarray, thr: float) -> float:
    """Fraction of |Z| >= thr on valid cells that are not events."""
    score = np.asarray(z)
    live = np.asarray(valid, dtype=bool) & np.isfinite(score)
    background = live & ~np.asarray(truth_any, dtype=bool)
    if not background.any():
        return 0.0
    return float(np.mean(np.abs(score[background]) >= thr))


def dead_alarms(z: np.ndarray, valid: np.ndarray, thr: float) -> int:
    dead = ~np.asarray(valid, dtype=bool)
    if not dead.any():
        return 0
    score = np.asarray(z)
    return int(np.count_nonzero(np.abs(score[dead]) >= thr))


def threshold_for_far(abs_values: np.ndarray, target: float, *, prefer: float = PREFER_THRESHOLD) -> float:
    """Pick the grid point nearest `prefer` among those closest to `target` FAR.

    A quieter tail does not pull the threshold down: every grid point with the
    same FAR error is a tie, and the tie breaks toward 3.5.
    """
    values = np.asarray(abs_values, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    grid = THRESHOLD_GRID
    if values.size == 0:
        return float(prefer)
    ordered = np.sort(values)
    n = int(ordered.size)
    index = np.searchsorted(ordered, grid, side="left")
    rates = (n - index) / n
    error = np.abs(rates - float(target))
    tol = 0.5 / n
    best = np.flatnonzero(error <= error.min() + tol)
    distance = np.abs(grid[best] - float(prefer))
    nearest = best[np.flatnonzero(np.isclose(distance, distance.min()))]
    return float(grid[nearest].min())


def far_ratio(holdout_far: float, target_far: float) -> float:
    """0/0 is inside the band. A positive holdout over a zero target is not."""
    holdout = float(holdout_far)
    target = float(target_far)
    if target == 0.0 and holdout == 0.0:
        return 1.0
    if target == 0.0:
        return float("inf")
    return holdout / target


def _global_mad(resid: np.ndarray, mask: np.ndarray, floor: float) -> float:
    data = np.asarray(resid, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    data = data[np.isfinite(data)]
    if data.size == 0:
        return float(floor)
    center = float(np.median(data))
    mad = 1.4826 * float(np.median(np.abs(data - center)))
    if not np.isfinite(mad):
        return float(floor)
    return float(max(mad, floor))


def _scale_residual(resid: np.ndarray, valid: np.ndarray, scale: float) -> np.ndarray:
    z = np.zeros(resid.shape, dtype=np.float32)
    ok = np.asarray(valid, dtype=bool) & np.isfinite(resid)
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("scale must be positive")
    z[ok] = (resid[ok] / scale).astype(np.float32)
    return z


def z_field(
    clean: np.ndarray,
    valid: np.ndarray,
    x_m: np.ndarray,
    *,
    scale: str,
    raw: np.ndarray | None = None,
    raw_bg: np.ndarray | None = None,
    clean_bg: np.ndarray | None = None,
    inner_m: float = 8.0,
    outer_m: float = 40.0,
    bin_m: float = 2.0,
    floor_c: float = FLOOR_C,
) -> np.ndarray:
    """Three scales. mad_profile calls spatial_z and does not edit it."""
    if scale == "mad_profile":
        z, _bg, _profile = spatial_z(
            clean, valid, x_m, inner_m=inner_m, outer_m=outer_m, bin_m=bin_m, floor_c=floor_c,
        )
        return z
    if clean_bg is None:
        clean_bg = spatial_background(
            clean, valid, x_m, inner_m=inner_m, outer_m=outer_m, bin_m=bin_m,
        )
    resid = np.asarray(clean, dtype=np.float64) - np.asarray(clean_bg, dtype=np.float64)
    live = np.asarray(valid, dtype=bool) & np.isfinite(resid)
    if scale == "mad_residual_global":
        denom = _global_mad(resid, live, floor_c)
    elif scale == "mad_raw_residual":
        if raw_bg is None:
            if raw is None:
                raise ValueError("mad_raw_residual needs raw or raw_bg")
            raw_bg = spatial_background(
                raw, valid, x_m, inner_m=inner_m, outer_m=outer_m, bin_m=bin_m,
            )
        raw_resid = np.asarray(raw, dtype=np.float64) - np.asarray(raw_bg, dtype=np.float64)
        raw_live = np.asarray(valid, dtype=bool) & np.isfinite(raw_resid)
        denom = _global_mad(raw_resid, raw_live, floor_c)
    else:
        raise ValueError(f"unknown scale {scale!r}")
    return _scale_residual(resid, live, denom)


def denoise_real(method: str, raw: np.ndarray, valid: np.ndarray, *, device: str = "cpu") -> np.ndarray:
    """Real files are large. A CUDA out-of-memory retry stays on the already fitted threshold."""
    try:
        return denoise_field(method, raw, valid, device=device)
    except RuntimeError as exc:
        if device == "cpu" or "out of memory" not in str(exc).lower():
            raise
        import torch

        torch.cuda.empty_cache()
        print(f"  {method} cuda out of memory, retry on cpu", flush=True)
        return denoise_field(method, raw, valid, device="cpu")


def denoise_field(method: str, raw: np.ndarray, valid: np.ndarray, *, device: str = "cpu") -> np.ndarray:
    """n2n_tsr loads models/weights/n2n_tsr.pt. zsn2n fits the field at a fixed seed."""
    if method == "zsn2n":
        from fiber.models.zsn2n import denoise_zsn2n

        return denoise_zsn2n(
            raw, valid, seed=ZSN2N_SEED, steps=ZSN2N_STEPS, device=device, weights=None,
        )
    if method == "n2n_tsr":
        weights = ROOT / "models" / "weights" / "n2n_tsr.pt"
        return apply_method("n2n_tsr", raw, valid, device=device, weights=weights)
    if method == "median_x3":
        return apply_method("median_x3", raw, valid)
    raise KeyError(method)


def iou_curve(
    hot: np.ndarray,
    cold: np.ndarray,
    z: np.ndarray,
    valid: np.ndarray,
    time_s: np.ndarray,
    x_m: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> list[dict]:
    """support_iou from evaluate_events, plus FAR on the non-event cells of this field."""
    levels = IOU_THRESHOLDS if thresholds is None else np.asarray(thresholds, dtype=np.float64)
    truth = np.asarray(hot, dtype=bool) | np.asarray(cold, dtype=bool)
    live = np.asarray(valid, dtype=bool)
    rows = []
    for thr in levels:
        level = float(thr)
        hot_metrics = evaluate_events(hot, (z >= level) & live, time_s, x_m, live)
        cold_metrics = evaluate_events(cold, (z <= -level) & live, time_s, x_m, live)
        rows.append({
            "threshold": round(level, 2),
            "hot_iou": hot_metrics["support_iou"],
            "cold_iou": cold_metrics["support_iou"],
            "hot_recall": hot_metrics["event_recall"],
            "cold_recall": cold_metrics["event_recall"],
            "hot_precision": hot_metrics["event_precision"],
            "cold_precision": cold_metrics["event_precision"],
            "far": far(z, live, truth, level),
        })
    return rows


def _num(metrics: dict, key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def decide_channel(metrics: dict) -> dict:
    """Candidate only when every clause holds. Otherwise name the failed ones."""
    failed: list[str] = []
    hot = _num(metrics, "hot_iou")
    cold = _num(metrics, "cold_iou")
    median_hot = _num(metrics, "median_hot_iou")
    median_cold = _num(metrics, "median_cold_iou")
    if hot is None or median_hot is None or hot < median_hot - 0.01:
        failed.append(FAIL_HOT)
    if cold is None or median_cold is None or cold < median_cold - 0.01:
        failed.append(FAIL_COLD)
    hold = _num(metrics, "far_holdout")
    median_far = _num(metrics, "median_far_holdout")
    if hold is None or median_far is None or hold > 1.1 * median_far + 1e-12:
        failed.append(FAIL_FAR_CAP)
    gains = []
    for key, base in (
        ("weak_hot_iou", "median_weak_hot_iou"),
        ("weak_cold_iou", "median_weak_cold_iou"),
        ("weak_hot_recall", "median_weak_hot_recall"),
        ("weak_cold_recall", "median_weak_cold_recall"),
    ):
        left, right = _num(metrics, key), _num(metrics, base)
        if left is None or right is None:
            continue
        gains.append(left - right)
    if not any(gain >= 0.05 - 1e-9 for gain in gains):
        failed.append(FAIL_WEAK)
    target = _num(metrics, "target_far")
    if hold is None or target is None or not (0.5 <= far_ratio(hold, target) <= 2.0):
        failed.append(FAIL_RATIO)
    std = _num(metrics, "threshold_std")
    if std is None or std > 0.25:
        failed.append(FAIL_STD)
    if not metrics.get("real_scored"):
        failed.append(FAIL_REAL)
    elif metrics.get("unstable_real"):
        failed.append(FAIL_UNSTABLE)
    half = _num(metrics, "half_alarm_fraction")
    median_half = _num(metrics, "median_half_alarm_fraction")
    if half is None or median_half is None or half > median_half + 1e-12:
        failed.append(FAIL_HALF)
    dead = metrics.get("dead_alarms")
    if dead is None or int(dead) != 0:
        failed.append(FAIL_DEAD)
    if metrics.get("witness_sha_mismatch"):
        failed.append(FAIL_SHA)
    elif metrics.get("witness_passes") is None or metrics.get("median_witness_passes") is None:
        failed.append(FAIL_WITNESS)
    elif int(metrics["witness_passes"]) < int(metrics["median_witness_passes"]):
        failed.append(FAIL_WITNESS_COUNT)
    return {"status": CANDIDATE if not failed else REJECTED, "failed": failed}


def decide_hybrid(metrics: dict) -> dict:
    failed: list[str] = []
    hot = _num(metrics, "hot_iou")
    cold = _num(metrics, "cold_iou")
    median_hot = _num(metrics, "median_hot_iou")
    median_cold = _num(metrics, "median_cold_iou")
    if hot is None or median_hot is None or hot < median_hot - 0.005:
        failed.append(FAIL_H_HOT)
    if cold is None or median_cold is None or cold < median_cold - 0.005:
        failed.append(FAIL_H_COLD)
    ramp_hot = _num(metrics, "ramp0_hot_iou")
    ramp_cold = _num(metrics, "ramp0_cold_iou")
    ramp_base_hot = _num(metrics, "ramp0_median_hot_iou")
    ramp_base_cold = _num(metrics, "ramp0_median_cold_iou")
    if (
        ramp_hot is None or ramp_cold is None or ramp_base_hot is None or ramp_base_cold is None
        or ramp_hot + 1e-9 < ramp_base_hot or ramp_cold + 1e-9 < ramp_base_cold
    ):
        failed.append(FAIL_H_RAMP)
    peak = _num(metrics, "peak_error")
    base_peak = _num(metrics, "median_peak_error")
    # A zero baseline error cannot fall by another 20%.
    if peak is None or base_peak is None or base_peak <= 0 or peak > base_peak * 0.8 + 1e-9:
        failed.append(FAIL_H_PEAK)
    grouped = metrics.get("peak_error_by_amp") or {}
    baseline_grouped = metrics.get("median_peak_error_by_amp") or {}
    if not baseline_grouped:
        failed.append(FAIL_H_AMP)
    else:
        amp_bad = False
        for key, base_value in baseline_grouped.items():
            if key not in grouped or grouped[key] is None or base_value is None:
                amp_bad = True
                continue
            base_number = float(base_value)
            limit = base_number if base_number <= 0 else base_number * 1.05
            if float(grouped[key]) > limit + 1e-9:
                amp_bad = True
        if amp_bad:
            failed.append(FAIL_H_AMP)
    x_error = _num(metrics, "x_error_ramp0")
    median_x = _num(metrics, "median_x_error_ramp0")
    step = _num(metrics, "grid_step_m")
    if step is None:
        step = 1.0
    if x_error is None or median_x is None or x_error > median_x + 1e-9:
        failed.append(FAIL_H_X)
    if x_error is None or x_error > step + 1e-9:
        failed.append(FAIL_H_STEP)
    t_error = _num(metrics, "t_error")
    median_t = _num(metrics, "median_t_error")
    if t_error is None or median_t is None or t_error > median_t + 1e-9:
        failed.append(FAIL_H_T)
    if not metrics.get("real_scored"):
        failed.append(FAIL_REAL)
    else:
        shift = _num(metrics, "median_boundary_shift_m")
        if shift is None or shift > 2.0:
            failed.append(FAIL_H_SHIFT)
        flips = metrics.get("contrast_sign_flips")
        if flips is None or int(flips) != 0:
            failed.append(FAIL_H_SIGN)
    if metrics.get("witness_sha_mismatch"):
        failed.append(FAIL_SHA)
    elif metrics.get("witness_passes") is None or metrics.get("median_witness_passes") is None:
        failed.append(FAIL_WITNESS)
    elif int(metrics["witness_passes"]) < int(metrics["median_witness_passes"]):
        failed.append(FAIL_WITNESS_COUNT)
    return {"status": CANDIDATE if not failed else REJECTED, "failed": failed}


def bootstrap_threshold_std(
    per_seed_abs: dict[int, np.ndarray],
    target: float,
    *,
    n: int = 5,
    rng_seed: int = 0,
) -> float:
    seeds = list(per_seed_abs)
    if not seeds:
        return 0.0
    rng = np.random.default_rng(rng_seed)
    chosen = []
    for _ in range(n):
        draw = rng.integers(0, len(seeds), size=len(seeds))
        parts = [np.asarray(per_seed_abs[seeds[int(i)]], dtype=np.float64).ravel() for i in draw]
        parts = [part for part in parts if part.size]
        pooled = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64)
        chosen.append(threshold_for_far(pooled, target))
    return float(np.std(np.asarray(chosen, dtype=np.float64), ddof=0))


def fit_median_profile(seeds: list[int]) -> tuple[float, int]:
    """median_x3 + mad_profile aimed at the FAR of |Z| >= 3.5 on these seeds."""
    parts: list[np.ndarray] = []
    dead = 0
    for seed in seeds:
        for scene in _scenes_for_seed(int(seed)):
            clean = denoise_field("median_x3", scene["raw"], scene["valid"])
            score = z_field(clean, scene["valid"], scene["x_m"], scale="mad_profile")
            dead += dead_alarms(score, scene["valid"], OPERATIONAL_Z)
            background = scene["valid"] & ~scene["hot"] & ~scene["cold"]
            parts.append(np.abs(score[background]).astype(np.float32, copy=False))
    pooled = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    target = float(np.mean(pooled >= OPERATIONAL_Z)) if pooled.size else 0.0
    return threshold_for_far(pooled, target), int(dead)


def _scenes_for_seed(seed: int) -> list[dict]:
    scenes = [_scene_record(background_scene(seed), seed, "background")]
    for spec in variant_grid():
        scenes.append(_scene_record(local_scene_var(seed, **spec), seed, "event"))
    return scenes


def _scene_record(scene: dict, seed: int, kind: str) -> dict:
    return {
        "seed": int(seed),
        "kind": kind,
        "raw": np.asarray(scene["clean"], dtype=np.float32),
        "valid": np.asarray(scene["valid"], dtype=bool),
        "hot": np.asarray(scene["hot"], dtype=bool),
        "cold": np.asarray(scene["cold"], dtype=bool),
        "x_m": np.asarray(scene["x_m"], dtype=np.float64),
        "time_s": np.asarray(scene["time_s"], dtype=np.float64),
        "amp": float(scene.get("amp_hot", 0.0)),
        "width": int(scene.get("width", 0)),
        "ramp": int(scene.get("ramp", 0)),
    }


def _fit_hasher(seeds: list[int], variants: list[dict]) -> "hashlib._Hash":
    digest = hashlib.sha256()
    payload = {"calib_seeds": list(seeds), "variants": variants}
    digest.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
    return digest


def _hash_field(digest, field: np.ndarray) -> None:
    digest.update(np.ascontiguousarray(field, dtype=np.float32).tobytes())


def build_scenes(seeds: list[int], *, split: str, digest=None) -> list[dict]:
    scenes: list[dict] = []
    for seed in seeds:
        for scene in _scenes_for_seed(int(seed)):
            scene["split"] = split
            if digest is not None:
                _hash_field(digest, scene["raw"])
            scenes.append(scene)
    return scenes


def _concat_abs(parts: list[np.ndarray]) -> np.ndarray:
    usable = [np.asarray(part, dtype=np.float64).ravel() for part in parts if len(part)]
    if not usable:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(usable)


def stack_packs(packs: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if not packs:
        return None
    z = np.concatenate([pack["z"] for pack in packs], axis=0)
    hot = np.concatenate([pack["hot"] for pack in packs], axis=0)
    cold = np.concatenate([pack["cold"] for pack in packs], axis=0)
    valid = np.concatenate([pack["valid"] for pack in packs], axis=0)
    times = []
    cursor = 0.0
    for pack in packs:
        shifted = np.asarray(pack["time_s"], dtype=np.float64) + cursor
        times.append(shifted)
        cursor = float(shifted[-1]) + 10000.0
    return z, hot, cold, valid, np.concatenate(times), np.asarray(packs[0]["x_m"], dtype=np.float64)


def select_packs(packs: list[dict], *, amp: float | None = None, width: int | None = None,
                 ramp: int | None = None, amp_max: float | None = None) -> list[dict]:
    chosen = []
    for pack in packs:
        if amp is not None and abs(float(pack["amp"]) - amp) > 1e-6:
            continue
        if width is not None and int(pack["width"]) != int(width):
            continue
        if ramp is not None and int(pack["ramp"]) != int(ramp):
            continue
        if amp_max is not None and float(pack["amp"]) > amp_max + 1e-6:
            continue
        chosen.append(pack)
    return chosen


def _event_metrics(packs: list[dict], thr: float) -> dict:
    stacked = stack_packs(packs)
    empty = {"hot_iou": None, "cold_iou": None, "hot_recall": None, "cold_recall": None,
             "hot_precision": None, "cold_precision": None}
    if stacked is None:
        return empty
    z, hot, cold, valid, time_s, x_m = stacked
    level = float(thr)
    hot_metrics = evaluate_events(hot, (z >= level) & valid, time_s, x_m, valid)
    cold_metrics = evaluate_events(cold, (z <= -level) & valid, time_s, x_m, valid)
    return {
        "hot_iou": hot_metrics["support_iou"],
        "cold_iou": cold_metrics["support_iou"],
        "hot_recall": hot_metrics["event_recall"],
        "cold_recall": cold_metrics["event_recall"],
        "hot_precision": hot_metrics["event_precision"],
        "cold_precision": cold_metrics["event_precision"],
    }


def _curve(packs: list[dict], abs_bg: np.ndarray) -> list[dict]:
    stacked = stack_packs(packs)
    if stacked is None:
        return []
    z, hot, cold, valid, time_s, x_m = stacked
    rows = iou_curve(hot, cold, z, valid, time_s, x_m, IOU_THRESHOLDS)
    # FAR for the report includes background_scene, which is not in the event stack.
    for row, thr in zip(rows, IOU_THRESHOLDS, strict=True):
        row["far"] = float(np.mean(abs_bg >= float(thr))) if abs_bg.size else 0.0
    return rows


def _group_table(packs: list[dict], thr: float) -> dict:
    table = {"weak": _event_metrics(select_packs(packs, amp_max=2.0), thr)}
    table["amp"] = {str(amp): _event_metrics(select_packs(packs, amp=amp), thr) for amp in AMP_GRID}
    table["width"] = {str(width): _event_metrics(select_packs(packs, width=width), thr) for width in WIDTH_GRID}
    table["ramp"] = {str(ramp): _event_metrics(select_packs(packs, ramp=ramp), thr) for ramp in RAMP_GRID}
    return table


def _half_alarm(method: str, scale: str, thr: float, device: str) -> tuple[float, int]:
    scene = half_cable_shift(1)
    clean = denoise_field(method, scene["clean"], scene["valid"], device=device)
    score = z_field(clean, scene["valid"], scene["x_m"], scale=scale, raw=scene["clean"])
    inside = scene["broad"] & scene["valid"]
    fraction = alarm_fraction(np.abs(score) >= thr, inside)
    return fraction, dead_alarms(score, scene["valid"], thr)


def _collect_method(method: str, scenes: list[dict], scales: list[str], device: str) -> dict:
    collected = {
        scale: {"packs": {"calib": [], "holdout": []}, "bg": {"calib": {}, "holdout": {}}, "dead": 0}
        for scale in scales
    }
    for block in collected.values():
        for scene in scenes:
            block["bg"][scene["split"]].setdefault(scene["seed"], [])
    for index, scene in enumerate(scenes, start=1):
        if index == 1 or scene["kind"] == "background":
            print(f"  {method} seed {scene['seed']} {scene['split']}", flush=True)
        clean = denoise_field(method, scene["raw"], scene["valid"], device=device)
        profile, clean_bg, _scale = spatial_z(clean, scene["valid"], scene["x_m"])
        raw_bg = scene.get("raw_bg")
        if raw_bg is None:
            raw_bg = spatial_background(scene["raw"], scene["valid"], scene["x_m"])
            scene["raw_bg"] = raw_bg
        z_map = {"mad_profile": profile}
        if "mad_residual_global" in collected:
            z_map["mad_residual_global"] = z_field(
                clean, scene["valid"], scene["x_m"], scale="mad_residual_global", clean_bg=clean_bg,
            )
        if "mad_raw_residual" in collected:
            z_map["mad_raw_residual"] = z_field(
                clean, scene["valid"], scene["x_m"], scale="mad_raw_residual",
                raw=scene["raw"], raw_bg=raw_bg, clean_bg=clean_bg,
            )
        background = scene["valid"] & ~scene["hot"] & ~scene["cold"]
        for scale, score in z_map.items():
            block = collected[scale]
            block["dead"] += dead_alarms(score, scene["valid"], min(OPERATIONAL_Z, 0.5))
            block["bg"][scene["split"]][scene["seed"]].append(np.abs(score[background]).astype(np.float32, copy=False))
            if scene["kind"] != "event":
                continue
            block["packs"][scene["split"]].append({
                "z": np.asarray(score, dtype=np.float32),
                "hot": scene["hot"],
                "cold": scene["cold"],
                "valid": scene["valid"],
                "time_s": scene["time_s"],
                "x_m": scene["x_m"],
                "amp": scene["amp"],
                "width": scene["width"],
                "ramp": scene["ramp"],
            })
    return collected


def _bg_pool(per_seed: dict[int, list[np.ndarray]]) -> np.ndarray:
    parts = [_concat_abs(chunks) for chunks in per_seed.values()]
    return _concat_abs(parts)


def _seed_pools(per_seed: dict[int, list[np.ndarray]]) -> dict[int, np.ndarray]:
    return {seed: _concat_abs(chunks) for seed, chunks in per_seed.items()}


def choose_halo(scenes: list[dict], device: str) -> dict:
    """Best matched-FAR halo on calibration scenes. Does not replace 8/40 m."""
    calib = [scene for scene in scenes if scene["split"] == "calib"]
    pairs = []
    best = None
    for inner in (4.0, 8.0, 12.0):
        for outer in (24.0, 40.0, 60.0):
            abs_parts = []
            packs = []
            for scene in calib:
                clean = denoise_field("median_x3", scene["raw"], scene["valid"], device=device)
                score, _bg, _profile = spatial_z(
                    clean, scene["valid"], scene["x_m"], inner_m=inner, outer_m=outer,
                )
                background = scene["valid"] & ~scene["hot"] & ~scene["cold"]
                abs_parts.append(np.abs(score[background]))
                if scene["kind"] == "event":
                    packs.append({
                        "z": score, "hot": scene["hot"], "cold": scene["cold"], "valid": scene["valid"],
                        "time_s": scene["time_s"], "x_m": scene["x_m"],
                        "amp": scene["amp"], "width": scene["width"], "ramp": scene["ramp"],
                    })
            pooled = _concat_abs(abs_parts)
            target = float(np.mean(pooled >= OPERATIONAL_Z)) if pooled.size else 0.0
            thr = threshold_for_far(pooled, target)
            metrics = _event_metrics(packs, thr)
            score_sum = float(metrics["hot_iou"] or 0.0) + float(metrics["cold_iou"] or 0.0)
            row = {
                "inner_m": inner, "outer_m": outer, "threshold": thr,
                "hot_iou": metrics["hot_iou"], "cold_iou": metrics["cold_iou"], "score": score_sum,
            }
            pairs.append(row)
            if best is None or score_sum > best["score"]:
                best = row
    return {
        "status": "swept",
        "chosen": {"inner_m": best["inner_m"], "outer_m": best["outer_m"]},
        "pairs": pairs,
        "note": "operational halo 8/40 m is not replaced",
    }


def _row_from_collected(method: str, scale: str, block: dict, target_far: float, device: str,
                        reference: dict | None) -> dict:
    calib_seeds = _seed_pools(block["bg"]["calib"])
    hold_abs = _bg_pool(block["bg"]["holdout"])
    calib_abs = _bg_pool(block["bg"]["calib"])
    thr = threshold_for_far(calib_abs, target_far)
    hold_packs = block["packs"]["holdout"]
    at_thr = _event_metrics(hold_packs, thr)
    at_op = _event_metrics(hold_packs, OPERATIONAL_Z)
    weak = _event_metrics(select_packs(hold_packs, amp_max=2.0), thr)
    hold_far = float(np.mean(hold_abs >= thr)) if hold_abs.size else 0.0
    calib_far = float(np.mean(calib_abs >= thr)) if calib_abs.size else 0.0
    half, half_dead = _half_alarm(method, scale, thr, device)
    dead = int(block["dead"]) + int(half_dead)
    row = {
        "method": method,
        "scale": scale,
        "threshold": thr,
        "target_far": target_far,
        "far_calib": calib_far,
        "far_holdout": hold_far,
        "hot_iou": at_thr["hot_iou"],
        "cold_iou": at_thr["cold_iou"],
        "hot_recall": at_thr["hot_recall"],
        "cold_recall": at_thr["cold_recall"],
        "hot_iou_at_3_5": at_op["hot_iou"],
        "cold_iou_at_3_5": at_op["cold_iou"],
        "weak_hot_iou": weak["hot_iou"],
        "weak_cold_iou": weak["cold_iou"],
        "weak_hot_recall": weak["hot_recall"],
        "weak_cold_recall": weak["cold_recall"],
        "threshold_std": bootstrap_threshold_std(calib_seeds, target_far),
        "half_alarm_fraction": half,
        "dead_alarms": dead,
        "groups": _group_table(hold_packs, thr),
        "curve_holdout": _curve(hold_packs, hold_abs),
        "real_scored": False,
        "unstable_real": False,
        "witness_passes": None,
        "witness_sha_mismatch": False,
        "real_files": [],
    }
    if reference is None:
        row["decision"] = BASELINE
        row["failed"] = []
        row["role"] = BASELINE
    else:
        metrics = {**row, **_reference_view(reference)}
        verdict = decide_channel(metrics)
        row["decision"] = verdict["status"]
        row["failed"] = verdict["failed"]
        row["role"] = "candidate"
    return row


def _reference_view(reference: dict) -> dict:
    return {
        "median_hot_iou": reference["hot_iou_at_3_5"],
        "median_cold_iou": reference["cold_iou_at_3_5"],
        "median_far_holdout": reference["far_holdout_at_3_5"],
        "median_weak_hot_iou": reference["weak_hot_iou"],
        "median_weak_cold_iou": reference["weak_cold_iou"],
        "median_weak_hot_recall": reference["weak_hot_recall"],
        "median_weak_cold_recall": reference["weak_cold_recall"],
        "median_half_alarm_fraction": reference["half_alarm_fraction"],
        "median_witness_passes": reference.get("witness_passes"),
    }


def _capture_reference(block: dict, device: str) -> dict:
    hold_abs = _bg_pool(block["bg"]["holdout"])
    hold_packs = block["packs"]["holdout"]
    at_op = _event_metrics(hold_packs, OPERATIONAL_Z)
    weak = _event_metrics(select_packs(hold_packs, amp_max=2.0), OPERATIONAL_Z)
    half, _dead = _half_alarm("median_x3", "mad_profile", OPERATIONAL_Z, device)
    far_hold = float(np.mean(hold_abs >= OPERATIONAL_Z)) if hold_abs.size else 0.0
    return {
        "hot_iou_at_3_5": at_op["hot_iou"],
        "cold_iou_at_3_5": at_op["cold_iou"],
        "far_holdout_at_3_5": far_hold,
        "weak_hot_iou": weak["hot_iou"],
        "weak_cold_iou": weak["cold_iou"],
        "weak_hot_recall": weak["hot_recall"],
        "weak_cold_recall": weak["cold_recall"],
        "half_alarm_fraction": half,
        "witness_passes": None,
    }


def run_calibration(
    *,
    methods: list[str],
    scales: list[str],
    calib_seeds: list[int],
    holdout_seeds: list[int],
    device: str = "cpu",
    halo_grid: bool = False,
) -> dict:
    assert_disjoint(calib_seeds, holdout_seeds)
    variants = variant_grid()
    digest = _fit_hasher(calib_seeds, variants)
    print("build calib", flush=True)
    calib = build_scenes(calib_seeds, split="calib", digest=digest)
    print("build holdout", flush=True)
    holdout = build_scenes(holdout_seeds, split="holdout")
    scenes = calib + holdout
    ordered = ["median_x3"] + [method for method in methods if method != "median_x3"]
    if "mad_profile" not in scales:
        raise ValueError("mad_profile is required to name the median_x3 FAR")
    target_far = None
    reference = None
    rows = []
    wanted = set(methods)
    for method in ordered:
        print(f"method {method}", flush=True)
        collected = _collect_method(method, scenes, scales, device)
        if method == "median_x3":
            calib_abs = _bg_pool(collected["mad_profile"]["bg"]["calib"])
            target_far = float(np.mean(calib_abs >= OPERATIONAL_Z)) if calib_abs.size else 0.0
            reference_block = collected["mad_profile"]
            reference = _capture_reference(reference_block, device)
        if target_far is None or reference is None:
            raise RuntimeError("median_x3 mad_profile must be scored before other methods")
        for scale in scales:
            print(f"  fit {method} {scale}", flush=True)
            row = _row_from_collected(method, scale, collected[scale], target_far, device, None if method == "median_x3" and scale == "mad_profile" else reference)
            if method in wanted:
                rows.append(row)
        del collected
    report = {
        "manifest": {
            "calib_seeds": list(calib_seeds),
            "holdout_seeds": list(holdout_seeds),
            "fit_seeds": list(calib_seeds),
            "fit_input_sha256": digest.hexdigest(),
            "variants": variants,
            "real_files_in_fit": False,
        },
        "threshold_rule": THRESHOLD_RULE,
        "device": device,
        "zsn2n": {"seed": ZSN2N_SEED, "steps": ZSN2N_STEPS, "weights": None},
        "halo_grid": choose_halo(scenes, device) if halo_grid else "not_swept",
        "operational_halo_m": {"inner_m": 8.0, "outer_m": 40.0},
        "operational_pair_unchanged": True,
        "reference": reference,
        "target_far": target_far,
        "rows": rows,
    }
    return report


def _gates() -> dict:
    config = PipelineConfig()
    return {
        "thresh": config.threshold_z,
        "min_width_m": config.min_width_m,
        "min_profiles": config.min_profiles,
        "min_cells": config.min_cells,
        "max_gap_s": config.max_gap_s,
    }


def _extract(z, valid, clean, background, x_m, time_s, time_iso, thr: float) -> list[dict]:
    config = PipelineConfig()
    available = spatial_availability(clean, valid, background)
    return extract_zones(
        z, available, x_m, list(time_iso), time_s,
        thresh=thr,
        min_width_m=config.min_width_m,
        min_profiles=config.min_profiles,
        min_cells=config.min_cells,
        max_gap_s=config.max_gap_s,
        contrast_c=np.asarray(clean, dtype=np.float64) - np.asarray(background, dtype=np.float64),
    )


def _boxes_overlap(left: dict, right: dict) -> bool:
    if left.get("polarity") != right.get("polarity"):
        return False
    if left["end_m"] < right["start_m"] or right["end_m"] < left["start_m"]:
        return False
    return not (left["t_end_s"] < right["t_start_s"] or right["t_end_s"] < left["t_start_s"])


def _overlap(median_zones: list[dict], other_zones: list[dict]) -> float | None:
    if not median_zones:
        return None
    hits = 0
    for zone in median_zones:
        if any(_boxes_overlap(zone, other) for other in other_zones):
            hits += 1
    return hits / len(median_zones)


def _zone_ratio(count: int, median_count: int) -> float:
    if median_count == 0:
        return float("inf") if count > 0 else 1.0
    return count / median_count


def _unstable(file_rows: list[dict]) -> bool:
    bad = 0
    for row in file_rows:
        if not row.get("present"):
            continue
        ratio = _zone_ratio(int(row["zone_count"]), int(row["median_zone_count"]))
        if ratio > 2.0 or ratio < 0.5:
            bad += 1
    return bad >= 2


def _witness_specs() -> list[dict]:
    payload = json.loads((ROOT / "fiber" / "benchmark" / "witnesses.json").read_text(encoding="utf-8"))
    return list(payload["witnesses"])


def _window(x_m: np.ndarray, n_rows: int, spec: dict) -> tuple[slice, np.ndarray]:
    r0, r1 = spec["rows"]
    r1 = min(int(r1), n_rows - 1)
    cols = (np.asarray(x_m) >= spec["x_m"][0]) & (np.asarray(x_m) <= spec["x_m"][1])
    return slice(int(r0), r1 + 1), cols


def _pass_witness(role: str, hot_frac: float, cold_frac: float, n_valid: int, alarm_cells: int) -> bool:
    """Same fractions as scripts/score_channels.py. Not a new witness rule."""
    if role == "positive_hot":
        return hot_frac >= 0.3
    if role == "negative_quiet":
        return hot_frac <= 0.1 and cold_frac <= 0.1
    if role == "negative_invalid":
        return n_valid == 0 and alarm_cells == 0
    return False


def _witness_from_z(z, valid, x_m, thr: float, spec: dict) -> dict:
    sl, cols = _window(x_m, z.shape[0], spec)
    window_valid = np.asarray(valid, dtype=bool)[sl][:, cols]
    n_valid = int(window_valid.sum())
    score = np.asarray(z)[sl][:, cols]
    hot_frac = float(((score >= thr) & window_valid).sum() / n_valid) if n_valid else 0.0
    cold_frac = float(((score <= -thr) & window_valid).sum() / n_valid) if n_valid else 0.0
    alarm_cells = int(np.count_nonzero(np.abs(score) >= thr))
    return {
        "id": spec["id"],
        "role": spec["role"],
        "pass": _pass_witness(spec["role"], hot_frac, cold_frac, n_valid, alarm_cells),
        "sha_mismatch": False,
        "hot_fraction": hot_frac,
        "cold_fraction": cold_frac,
    }


def _scale_maps(clean, raw, valid, x_m, raw_bg) -> dict:
    profile, clean_bg, _scale = spatial_z(clean, valid, x_m)
    return {
        "mad_profile": (profile, clean_bg),
        "mad_residual_global": (z_field(clean, valid, x_m, scale="mad_residual_global", clean_bg=clean_bg), clean_bg),
        "mad_raw_residual": (
            z_field(clean, valid, x_m, scale="mad_raw_residual", raw=raw, raw_bg=raw_bg, clean_bg=clean_bg),
            clean_bg,
        ),
    }


def attach_real(report: dict, root: Path, *, device: str) -> dict:
    """Score MGTS files. Does not change the fitted threshold or the fit hash."""
    from fiber.io import load_csv

    root = Path(root)
    specs = _witness_specs()
    by_record: dict[str, list[dict]] = {}
    for spec in specs:
        by_record.setdefault(str(spec["record"]), []).append(spec)
    methods = []
    for row in report["rows"]:
        if row["method"] not in methods:
            methods.append(row["method"])
    scales = []
    for row in report["rows"]:
        if row["scale"] not in scales:
            scales.append(row["scale"])
    thresholds = {(row["method"], row["scale"]): float(row["threshold"]) for row in report["rows"]}
    file_rows = {(method, scale): [] for method in methods for scale in scales}
    witness_rows = {(method, scale): [] for method in methods for scale in scales}
    for name in REAL_FILES:
        path = root / name
        print(f"real {name}", flush=True)
        if not path.exists():
            for key in file_rows:
                file_rows[key].append({"file": name, "present": False, "error": "missing"})
            continue
        try:
            record = load_csv(path)
        except Exception as exc:
            for key in file_rows:
                file_rows[key].append({"file": name, "present": False, "error": f"{type(exc).__name__}: {exc}"})
            continue
        raw_bg = spatial_background(record.raw_c, record.valid, record.x_m)
        cleans = {}
        maps = {}
        for method in methods:
            print(f"  denoise {method}", flush=True)
            cleans[method] = denoise_real(method, record.raw_c, record.valid, device=device)
            maps[method] = _scale_maps(cleans[method], record.raw_c, record.valid, record.x_m, raw_bg)
        ref_z, ref_bg = maps["median_x3"]["mad_profile"]
        median_zones = _extract(
            ref_z, record.valid, cleans["median_x3"], ref_bg, record.x_m, record.time_s, record.time_iso, OPERATIONAL_Z,
        )
        stem = path.stem
        for method in methods:
            for scale in scales:
                score, background = maps[method][scale]
                thr = thresholds[(method, scale)]
                zones = _extract(
                    score, record.valid, cleans[method], background, record.x_m, record.time_s, record.time_iso, thr,
                )
                available = spatial_availability(cleans[method], record.valid, background)
                n_valid = int(np.asarray(record.valid).sum())
                coverage = float(available.sum() / n_valid) if n_valid else None
                file_rows[(method, scale)].append({
                    "file": name,
                    "present": True,
                    "error": None,
                    "zone_count": len(zones),
                    "median_zone_count": len(median_zones),
                    "alarm_fraction": alarm_fraction(np.abs(score) >= thr, record.valid),
                    "score_coverage": coverage,
                    "overlap_vs_median": _overlap(median_zones, zones),
                    "source_sha256": record.source_sha256,
                })
                for spec in by_record.get(stem, []):
                    if record.source_sha256 != spec["source_sha256"]:
                        witness_rows[(method, scale)].append({
                            "id": spec["id"], "role": spec["role"], "pass": None, "sha_mismatch": True,
                        })
                    else:
                        witness_rows[(method, scale)].append(_witness_from_z(score, record.valid, record.x_m, thr, spec))
    _apply_real_to_rows(report, file_rows, witness_rows)
    return report


def _apply_real_to_rows(report: dict, file_rows: dict, witness_rows: dict) -> None:
    reference = report["reference"]
    for row in report["rows"]:
        key = (row["method"], row["scale"])
        files = file_rows.get(key, [])
        row["real_files"] = files
        present = [item for item in files if item.get("present")]
        row["real_scored"] = len(present) == len(REAL_FILES) and len(files) == len(REAL_FILES)
        row["unstable_real"] = bool(row["real_scored"] and _unstable(files))
        witnesses = witness_rows.get(key, [])
        mismatch = any(item.get("sha_mismatch") for item in witnesses)
        row["witness_sha_mismatch"] = mismatch
        if mismatch or len(witnesses) != len(_witness_specs()):
            row["witness_passes"] = None
        else:
            row["witness_passes"] = int(sum(bool(item.get("pass")) for item in witnesses))
        row["witnesses"] = witnesses
        if row["role"] == BASELINE:
            reference["witness_passes"] = row["witness_passes"]
            row["decision"] = BASELINE
            row["failed"] = []
            continue
    for row in report["rows"]:
        if row["role"] == BASELINE:
            continue
        reference["witness_passes"] = next(
            (item["witness_passes"] for item in report["rows"] if item["role"] == BASELINE),
            reference.get("witness_passes"),
        )
        metrics = {**row, **_reference_view(reference)}
        verdict = decide_channel(metrics)
        row["decision"] = verdict["status"]
        row["failed"] = verdict["failed"]


def _median_finite(values: list[float | None]) -> float | None:
    data = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not data:
        return None
    return float(np.median(np.asarray(data, dtype=np.float64)))


def _amp_key(amp: float) -> str:
    return f"{float(amp):.1f}"


def _pool_masks(packs: list[dict], hot_key: str, cold_key: str) -> dict:
    if not packs:
        return {"hot_iou": None, "cold_iou": None}
    hot = np.concatenate([pack["hot"] for pack in packs], axis=0)
    cold = np.concatenate([pack["cold"] for pack in packs], axis=0)
    valid = np.concatenate([pack["valid"] for pack in packs], axis=0)
    pred_hot = np.concatenate([pack[hot_key] for pack in packs], axis=0)
    pred_cold = np.concatenate([pack[cold_key] for pack in packs], axis=0)
    times = []
    cursor = 0.0
    for pack in packs:
        shifted = np.asarray(pack["time_s"], dtype=np.float64) + cursor
        times.append(shifted)
        cursor = float(shifted[-1]) + 10000.0
    time_s = np.concatenate(times)
    x_m = np.asarray(packs[0]["x_m"], dtype=np.float64)
    hot_metrics = evaluate_events(hot, pred_hot, time_s, x_m, valid)
    cold_metrics = evaluate_events(cold, pred_cold, time_s, x_m, valid)
    return {"hot_iou": hot_metrics["support_iou"], "cold_iou": cold_metrics["support_iou"]}


def _error_groups(samples: list[dict]) -> dict:
    def pack(side: str, subset: list[dict]) -> dict:
        return {
            "peak_error_c": _median_finite([item[side]["peak_error_c"] for item in subset]),
            "median_error_c": _median_finite([item[side]["median_error_c"] for item in subset]),
            "x_error_m": _median_finite([item[side]["x_error_m"] for item in subset]),
            "t_error_s": _median_finite([item[side]["t_error_s"] for item in subset]),
        }

    def grouped(key: str, values) -> dict:
        out = {}
        for value in values:
            label = _amp_key(value) if key == "amp" else str(value)
            subset = [item for item in samples if item[key] == value or (key == "amp" and abs(item[key] - value) < 1e-6)]
            out[label] = {"refined": pack("refined", subset), "baseline": pack("baseline", subset)}
        return out

    return {
        "all": {"refined": pack("refined", samples), "baseline": pack("baseline", samples)},
        "by_amp": grouped("amp", AMP_GRID),
        "by_width": grouped("width", WIDTH_GRID),
        "by_ramp": grouped("ramp", RAMP_GRID),
    }


def _hybrid_metrics(mode: str, packs: list[dict], samples: list[dict]) -> dict:
    overall = _pool_masks(packs, "ref_hot", "ref_cold")
    baseline = _pool_masks(packs, "base_hot", "base_cold")
    ramp_packs = [pack for pack in packs if int(pack["ramp"]) == 0]
    ramp = _pool_masks(ramp_packs, "ref_hot", "ref_cold")
    ramp_base = _pool_masks(ramp_packs, "base_hot", "base_cold")
    groups = _error_groups(samples)
    ramp_samples = [item for item in samples if int(item["ramp"]) == 0]
    peak_by_amp = {}
    median_by_amp = {}
    for amp in AMP_GRID:
        key = _amp_key(amp)
        block = groups["by_amp"][key]
        peak_by_amp[key] = block["refined"]["peak_error_c"]
        median_by_amp[key] = block["baseline"]["peak_error_c"]
    return {
        "mode": mode,
        "hot_iou": overall["hot_iou"],
        "cold_iou": overall["cold_iou"],
        "median_hot_iou": baseline["hot_iou"],
        "median_cold_iou": baseline["cold_iou"],
        "ramp0_hot_iou": ramp["hot_iou"],
        "ramp0_cold_iou": ramp["cold_iou"],
        "ramp0_median_hot_iou": ramp_base["hot_iou"],
        "ramp0_median_cold_iou": ramp_base["cold_iou"],
        "peak_error": groups["all"]["refined"]["peak_error_c"],
        "median_peak_error": groups["all"]["baseline"]["peak_error_c"],
        "peak_error_by_amp": peak_by_amp,
        "median_peak_error_by_amp": median_by_amp,
        "x_error_ramp0": _median_finite([item["refined"]["x_error_m"] for item in ramp_samples]),
        "median_x_error_ramp0": _median_finite([item["baseline"]["x_error_m"] for item in ramp_samples]),
        "t_error": groups["all"]["refined"]["t_error_s"],
        "median_t_error": groups["all"]["baseline"]["t_error_s"],
        "grid_step_m": 1.0,
        "errors": groups,
        "real_scored": False,
        "median_boundary_shift_m": None,
        "contrast_sign_flips": None,
        "witness_passes": None,
        "median_witness_passes": None,
        "witness_sha_mismatch": False,
        "real": None,
    }


def run_hybrid(
    *,
    fine: str,
    modes: list[str],
    seeds: list[int],
    device: str = "cpu",
) -> dict:
    packs_by_mode = {mode: [] for mode in modes}
    samples_by_mode = {mode: [] for mode in modes}
    for seed in seeds:
        print(f"hybrid seed {seed}", flush=True)
        for spec in variant_grid():
            scene = local_scene_var(int(seed), **spec)
            raw = scene["clean"]
            valid = scene["valid"]
            median = denoise_field("median_x3", raw, valid, device=device)
            profile, median_bg, _scale = spatial_z(median, valid, scene["x_m"])
            iso = [str(float(t)) for t in scene["time_s"]]
            zones = _extract(profile, valid, median, median_bg, scene["x_m"], scene["time_s"], iso, OPERATIONAL_Z)
            base_hot = zones_to_mask(zones, raw.shape, scene["x_m"], scene["time_s"], polarity="hot")
            base_cold = zones_to_mask(zones, raw.shape, scene["x_m"], scene["time_s"], polarity="cold")
            fine_field = denoise_field(fine, raw, valid, device=device)
            fine_bg = spatial_background(fine_field, valid, scene["x_m"])
            coarse = np.asarray(median, dtype=np.float64) - np.asarray(median_bg, dtype=np.float64)
            for mode in modes:
                ref_hot = np.zeros(raw.shape, dtype=bool)
                ref_cold = np.zeros(raw.shape, dtype=bool)
                records = []
                for zone in zones:
                    one_hot = zones_to_mask([zone], raw.shape, scene["x_m"], scene["time_s"], polarity="hot")
                    one_cold = zones_to_mask([zone], raw.shape, scene["x_m"], scene["time_s"], polarity="cold")
                    hot_mask, cold_mask, found = refine_zones(
                        one_hot, one_cold, fine_field, fine_bg, valid, scene["x_m"], scene["time_s"],
                        mode=mode, coarse_delta=coarse,
                    )
                    ref_hot |= hot_mask
                    ref_cold |= cold_mask
                    records.extend(found)
                packs_by_mode[mode].append({
                    "hot": scene["hot"], "cold": scene["cold"], "valid": valid,
                    "time_s": scene["time_s"], "x_m": scene["x_m"],
                    "base_hot": base_hot, "base_cold": base_cold,
                    "ref_hot": ref_hot, "ref_cold": ref_cold,
                    "amp": float(spec["amp_hot"]), "width": int(spec["width"]), "ramp": int(spec["ramp"]),
                })
                for record in records:
                    errors = truth_extent_errors(record, scene["truth_delta_c"], scene["x_m"], scene["time_s"])
                    if errors is None:
                        continue
                    samples_by_mode[mode].append({
                        "amp": float(spec["amp_hot"]),
                        "width": int(spec["width"]),
                        "ramp": int(spec["ramp"]),
                        **errors,
                    })
    modes_out = {}
    for mode in modes:
        metrics = _hybrid_metrics(mode, packs_by_mode[mode], samples_by_mode[mode])
        verdict = decide_hybrid(metrics)
        metrics["decision"] = verdict["status"]
        metrics["failed"] = verdict["failed"]
        modes_out[mode] = metrics
    return {
        "manifest": {
            "seeds": list(seeds),
            "fine": fine,
            "modes": list(modes),
            "gates": _gates(),
            "variants": variant_grid(),
            "real_files_in_fit": False,
        },
        "device": device,
        "operational_pair_unchanged": True,
        "modes": modes_out,
    }


def _crop(field: np.ndarray, x_m, time_s, zone: dict, *, pad_m: float = 15.0) -> list | None:
    x = np.asarray(x_m, dtype=np.float64)
    times = np.asarray(time_s, dtype=np.float64)
    cols = np.flatnonzero((x >= zone["start_m"] - pad_m) & (x <= zone["end_m"] + pad_m))
    rows = np.flatnonzero((times >= zone["t_start_s"] - 1e-6) & (times <= zone["t_end_s"] + 1e-6))
    if rows.size == 0 or cols.size == 0:
        return None
    step_r = max(1, int(np.ceil(rows.size / 24)))
    step_c = max(1, int(np.ceil(cols.size / 40)))
    view = np.asarray(field, dtype=np.float64)[rows[::step_r][:, None], cols[::step_c]]
    cropped = []
    for row in view:
        cropped.append([None if not np.isfinite(value) else round(float(value), 2) for value in row])
    return cropped


def _offer_shift(bucket: list[dict], item: dict, limit: int = 6) -> None:
    if len(bucket) >= limit and item["shift_m"] <= bucket[-1]["shift_m"]:
        return
    bucket.append(item)
    bucket.sort(key=lambda row: row["shift_m"], reverse=True)
    del bucket[limit:]


def attach_hybrid_real(report: dict, root: Path, *, device: str) -> dict:
    from fiber.io import load_csv

    root = Path(root)
    fine = report["manifest"]["fine"]
    modes = list(report["manifest"]["modes"])
    specs = _witness_specs()
    by_record: dict[str, list[dict]] = {}
    for spec in specs:
        by_record.setdefault(str(spec["record"]), []).append(spec)
    shifts = {mode: [] for mode in modes}
    contrast_changes = {mode: [] for mode in modes}
    flips = {mode: 0 for mode in modes}
    files = {mode: [] for mode in modes}
    largest = {mode: [] for mode in modes}
    witness_rows = {mode: [] for mode in modes}
    median_witness = []
    for name in REAL_FILES:
        path = root / name
        print(f"hybrid real {name}", flush=True)
        if not path.exists():
            for mode in modes:
                files[mode].append({"file": name, "present": False, "error": "missing"})
            continue
        try:
            record = load_csv(path)
        except Exception as exc:
            for mode in modes:
                files[mode].append({"file": name, "present": False, "error": f"{type(exc).__name__}: {exc}"})
            continue
        median = denoise_real("median_x3", record.raw_c, record.valid, device=device)
        profile, median_bg, _scale = spatial_z(median, record.valid, record.x_m)
        zones = _extract(
            profile, record.valid, median, median_bg, record.x_m, record.time_s, record.time_iso, OPERATIONAL_Z,
        )
        fine_field = denoise_real(fine, record.raw_c, record.valid, device=device)
        fine_bg = spatial_background(fine_field, record.valid, record.x_m)
        coarse = np.asarray(median, dtype=np.float64) - np.asarray(median_bg, dtype=np.float64)
        fine_resid = np.asarray(fine_field, dtype=np.float64) - np.asarray(fine_bg, dtype=np.float64)
        stem = path.stem
        for spec in by_record.get(stem, []):
            if record.source_sha256 != spec["source_sha256"]:
                median_witness.append({"id": spec["id"], "pass": None, "sha_mismatch": True})
            else:
                median_witness.append(_witness_from_z(profile, record.valid, record.x_m, OPERATIONAL_Z, spec))
        for mode in modes:
            ref_hot = np.zeros(record.raw_c.shape, dtype=bool)
            ref_cold = np.zeros(record.raw_c.shape, dtype=bool)
            mode_shifts = []
            mode_contrast = []
            mode_flips = 0
            for zone in zones:
                one_hot = zones_to_mask([zone], record.raw_c.shape, record.x_m, record.time_s, polarity="hot")
                one_cold = zones_to_mask([zone], record.raw_c.shape, record.x_m, record.time_s, polarity="cold")
                hot_mask, cold_mask, found = refine_zones(
                    one_hot, one_cold, fine_field, fine_bg, record.valid, record.x_m, record.time_s,
                    mode=mode, coarse_delta=coarse,
                )
                ref_hot |= hot_mask
                ref_cold |= cold_mask
                for item in found:
                    shift = max(
                        abs(float(item["start_m"]) - float(item["baseline_start_m"])),
                        abs(float(item["end_m"]) - float(item["baseline_end_m"])),
                    )
                    peak = item["peak_delta_c"]
                    base = item["baseline_peak_delta_c"]
                    flipped = (
                        not item["refined_empty"] and peak is not None and base is not None
                        and float(peak) * float(base) < 0
                    )
                    if flipped:
                        mode_flips += 1
                    change = None if peak is None or base is None else abs(float(peak) - float(base))
                    mode_shifts.append(shift)
                    mode_contrast.append(change)
                    _offer_shift(largest[mode], {
                        "file": name,
                        "polarity": item["polarity"],
                        "shift_m": shift,
                        "peak_delta_c": peak,
                        "baseline_peak_delta_c": base,
                        "contrast_change_c": change,
                        "start_m": item["start_m"],
                        "end_m": item["end_m"],
                        "baseline_start_m": item["baseline_start_m"],
                        "baseline_end_m": item["baseline_end_m"],
                        "t_start_s": item["t_start_s"],
                        "t_end_s": item["t_end_s"],
                        "crop_fine": _crop(fine_resid, record.x_m, record.time_s, item),
                        "crop_median": _crop(coarse, record.x_m, record.time_s, item),
                    })
            shifts[mode].extend(mode_shifts)
            contrast_changes[mode].extend(mode_contrast)
            flips[mode] += mode_flips
            files[mode].append({
                "file": name,
                "present": True,
                "error": None,
                "zone_count": len(zones),
                "refined_zone_count": len(zones),
                "boundary_shift_m": {
                    "median": _median_finite(mode_shifts),
                    "max": None if not mode_shifts else float(np.max(mode_shifts)),
                },
                "contrast_change_c": {
                    "median": _median_finite(mode_contrast),
                    "max": None if not any(value is not None for value in mode_contrast) else float(max(value for value in mode_contrast if value is not None)),
                },
                "contrast_sign_flips": mode_flips,
                "source_sha256": record.source_sha256,
            })
            for spec in by_record.get(stem, []):
                if record.source_sha256 != spec["source_sha256"]:
                    witness_rows[mode].append({"id": spec["id"], "pass": None, "sha_mismatch": True})
                    continue
                sl, cols = _window(record.x_m, ref_hot.shape[0], spec)
                window_valid = np.asarray(record.valid)[sl][:, cols]
                n_valid = int(window_valid.sum())
                hot_frac = float((ref_hot[sl][:, cols] & window_valid).sum() / n_valid) if n_valid else 0.0
                cold_frac = float((ref_cold[sl][:, cols] & window_valid).sum() / n_valid) if n_valid else 0.0
                alarm_cells = int(np.count_nonzero(ref_hot[sl][:, cols] | ref_cold[sl][:, cols]))
                witness_rows[mode].append({
                    "id": spec["id"],
                    "role": spec["role"],
                    "pass": _pass_witness(spec["role"], hot_frac, cold_frac, n_valid, alarm_cells),
                    "sha_mismatch": False,
                })
    median_mismatch = any(item.get("sha_mismatch") for item in median_witness)
    median_passes = None
    if not median_mismatch and len(median_witness) == len(specs):
        median_passes = int(sum(bool(item.get("pass")) for item in median_witness))
    for mode in modes:
        block = report["modes"][mode]
        present = [item for item in files[mode] if item.get("present")]
        block["real_scored"] = len(present) == len(REAL_FILES)
        block["median_boundary_shift_m"] = _median_finite(shifts[mode]) if block["real_scored"] else None
        block["contrast_sign_flips"] = int(flips[mode]) if block["real_scored"] else None
        block["median_contrast_change_c"] = _median_finite(contrast_changes[mode]) if block["real_scored"] else None
        block["median_witness_passes"] = median_passes
        mismatch = any(item.get("sha_mismatch") for item in witness_rows[mode]) or median_mismatch
        block["witness_sha_mismatch"] = mismatch
        if mismatch or len(witness_rows[mode]) != len(specs):
            block["witness_passes"] = None
        else:
            block["witness_passes"] = int(sum(bool(item.get("pass")) for item in witness_rows[mode]))
        ranked = sorted(largest[mode], key=lambda item: item["shift_m"], reverse=True)[:6]
        block["real"] = {
            "files": files[mode],
            "boundary_shift_m": shifts[mode],
            "median_boundary_shift_m": block["median_boundary_shift_m"],
            "median_contrast_change_c": block["median_contrast_change_c"],
            "contrast_sign_flips": block["contrast_sign_flips"],
            "largest_shifts": ranked,
            "witnesses": witness_rows[mode],
            "median_witness_passes": median_passes,
        }
        # The full shift list can be large. Keep the summary and the six crops.
        block["real"]["boundary_shift_count"] = len(shifts[mode])
        block["real"].pop("boundary_shift_m", None)
        verdict = decide_hybrid(block)
        block["decision"] = verdict["status"]
        block["failed"] = verdict["failed"]
    return report


def jsonify(obj):
    if isinstance(obj, dict):
        return {str(key): jsonify(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonify(value) for value in obj]
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        if not np.isfinite(value):
            return None
        return round(value, 6)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def dump_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(jsonify(payload), ensure_ascii=False, indent=2, sort_keys=True)
    path.write_text(text + "\n", encoding="utf-8")


def summary_payload(report: dict) -> dict:
    if "rows" in report:
        return {
            "decisions": [
                {
                    "decision": row["decision"],
                    "failed": list(row["failed"]),
                    "method": row["method"],
                    "scale": row["scale"],
                    "threshold": row["threshold"],
                }
                for row in report["rows"]
            ]
        }
    return {
        "decisions": [
            {
                "decision": block["decision"],
                "failed": list(block["failed"]),
                "mode": mode,
            }
            for mode, block in report["modes"].items()
        ]
    }


def render_threshold_section(report: dict, hybrid: dict) -> str:
    lines = [
        "## Порог под денойзер и гибридное уточнение (после d288333)",
        "",
        "Рабочая пара не сменена: `median_x3` и `spatial_z` при `|Z| >= 3.5`.",
        "Порог ниже подобран под тот же FAR и в `analyze_field` не записан.",
        f"Правило порога: {report['threshold_rule']}.",
        f"Устройство прогона: `{report['device']}`.",
        "",
        "| метод | шкала | порог | hot IoU | cold IoU | FAR holdout | weak hot IoU | статус | условие |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in report["rows"]:
        failed = ", ".join(row["failed"]) if row["failed"] else "—"
        lines.append(
            "| `{method}` | `{scale}` | {threshold} | {hot} | {cold} | {far} | {weak} | {status} | {failed} |".format(
                method=row["method"],
                scale=row["scale"],
                threshold=_fmt(row["threshold"]),
                hot=_fmt(row["hot_iou"]),
                cold=_fmt(row["cold_iou"]),
                far=_fmt(row["far_holdout"]),
                weak=_fmt(row["weak_hot_iou"]),
                status=row["decision"],
                failed=failed,
            )
        )
    lines.extend([
        "",
        "Кривые: [iou_vs_threshold_hot.png](images/ssl/iou_vs_threshold_hot.png), "
        "[iou_vs_threshold_cold.png](images/ssl/iou_vs_threshold_cold.png), "
        "[far_vs_threshold.png](images/ssl/far_vs_threshold.png).",
        "",
        "### Реальные файлы",
        "",
    ])
    lines.extend(_real_lines(report))
    lines.extend(["", "### Гибридное уточнение", ""])
    lines.extend(_hybrid_lines(hybrid))
    lines.append("")
    lines.append(_decision_sentence(report, hybrid))
    lines.append("")
    return "\n".join(lines)


def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _real_lines(report: dict) -> list[str]:
    lines = [
        "| метод | шкала | файл | зоны | зоны median_x3 | доля тревоги | покрытие | пересечение |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    any_row = False
    for row in report["rows"]:
        for item in row.get("real_files") or []:
            any_row = True
            if not item.get("present"):
                lines.append(
                    f"| `{row['method']}` | `{row['scale']}` | {item.get('file')} | — | — | — | — | missing |"
                )
                continue
            lines.append(
                "| `{method}` | `{scale}` | {file} | {zones} | {median} | {alarm} | {cover} | {overlap} |".format(
                    method=row["method"],
                    scale=row["scale"],
                    file=item["file"],
                    zones=item["zone_count"],
                    median=item["median_zone_count"],
                    alarm=_fmt(item["alarm_fraction"]),
                    cover=_fmt(item["score_coverage"]),
                    overlap=_fmt(item["overlap_vs_median"]),
                )
            )
    if not any_row:
        lines.append("| — | — | — | — | — | — | — | нет прогона `--real-root` |")
    unstable = [f"`{row['method']}`/`{row['scale']}`" for row in report["rows"] if row.get("unstable_real")]
    if unstable:
        lines.append("")
        lines.append("Нестабилен на реальных данных: " + ", ".join(unstable) + ".")
    else:
        lines.append("")
        lines.append("Флаг «нестабилен на реальных данных» не стоит, либо реального прогона не было.")
    return lines


def _hybrid_lines(hybrid: dict) -> list[str]:
    lines = [
        "| режим | hot IoU | cold IoU | ошибка пика | ошибка x при ramp=0 | статус | условие |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for mode, block in hybrid.get("modes", {}).items():
        failed = ", ".join(block["failed"]) if block["failed"] else "—"
        lines.append(
            "| `{mode}` | {hot} | {cold} | {peak} | {edge} | {status} | {failed} |".format(
                mode=mode,
                hot=_fmt(block.get("hot_iou")),
                cold=_fmt(block.get("cold_iou")),
                peak=_fmt(block.get("peak_error")),
                edge=_fmt(block.get("x_error_ramp0")),
                status=block.get("decision"),
                failed=failed,
            )
        )
    lines.append("")
    lines.append(
        "Ошибки пика и границ по амплитуде, ширине и ramp лежат в `docs/evidence/hybrid_refine.json`. "
        "Картинки: `docs/images/ssl/refine_*.png`."
    )
    return lines


def _decision_sentence(report: dict, hybrid: dict) -> str:
    parts = []
    for row in report["rows"]:
        if row.get("role") == BASELINE or row["decision"] == BASELINE:
            continue
        if row["failed"]:
            parts.append(
                f"`{row['method']}` / `{row['scale']}` — {REJECTED}: {row['failed'][0]}"
            )
        else:
            parts.append(f"`{row['method']}` / `{row['scale']}` — {CANDIDATE}")
    for mode, block in hybrid.get("modes", {}).items():
        if block["failed"]:
            parts.append(f"гибрид `{mode}` — {REJECTED}: {block['failed'][0]}")
        else:
            parts.append(f"гибрид `{mode}` — {CANDIDATE}")
    if not parts:
        return "Кандидатов нет."
    return "Решение по правилу отчёта: " + "; ".join(parts) + "."


def render_decision_entry(report: dict, hybrid: dict) -> str:
    lines = [
        "## 2026-10-13 — порог под денойзер и гибридное уточнение",
        "",
        "- Рабочий канал не сменён. `analyze_field` по-прежнему чистит `median_x3` и ставит `spatial_z` при `|Z| >= 3.5`.",
        "- Порог под денойзер и уточнение краёв записаны только в отчёт. В пару они не входят.",
        f"- {_decision_sentence(report, hybrid)}",
        "- Отчёт: [docs/SSL_BENCHMARK.md](SSL_BENCHMARK.md), [docs/evidence/threshold_cal.json](evidence/threshold_cal.json), [docs/evidence/hybrid_refine.json](evidence/hybrid_refine.json).",
        "",
    ]
    return "\n".join(lines)


def splice_markdown(path: Path, heading: str, section: str) -> None:
    path = Path(path)
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    start = text.find(heading)
    if start == -1:
        if text and not text.endswith("\n"):
            text += "\n"
        text = text + "\n" + section
    else:
        text = text[:start] + section
    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text, encoding="utf-8")
