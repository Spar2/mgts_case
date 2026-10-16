"""Score the operational detector and a whole-profile baseline.

Synthetic scenes have a known mask. Real windows are agent witnesses:
a hit means the window was localised, not that precision on the cable is known.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fiber.anomaly import spatial_z
from fiber.benchmark import alarm_fraction, evaluate_events
from fiber.benchmark.cards import MODEL_CARDS
from fiber.benchmark.synthetic import half_cable_shift, local_scene
from fiber.io import Record


def profile_median_z(clean: np.ndarray, valid: np.ndarray, floor_c: float = 0.30) -> np.ndarray:
    field = np.where(valid, clean, np.nan).astype(np.float64)
    with np.errstate(all="ignore"):
        med = np.nanmedian(field, axis=1)
        mad = 1.4826 * np.nanmedian(np.abs(field - med[:, None]), axis=1)
    mad = np.where(np.isfinite(mad), np.maximum(mad, floor_c), floor_c)
    med = np.where(np.isfinite(med), med, 0.0)
    z = np.zeros(field.shape, dtype=np.float32)
    ok = valid & np.isfinite(field)
    denom = np.broadcast_to(mad[:, None], field.shape)
    center = np.broadcast_to(med[:, None], field.shape)
    z[ok] = ((field[ok] - center[ok]) / denom[ok]).astype(np.float32)
    return z


def _round(score: dict) -> dict:
    out = {}
    for key, value in score.items():
        out[key] = round(value, 4) if isinstance(value, float) else value
    return out


def score_synthetic() -> dict:
    scene = local_scene()
    started = time.perf_counter()
    z_halo, _, _ = spatial_z(scene["clean"], scene["valid"], scene["x_m"])
    z_base = profile_median_z(scene["clean"], scene["valid"])
    elapsed = time.perf_counter() - started
    report = {"scene": scene["name"], "elapsed_s": round(elapsed, 3), "models": {}}
    for name, z in (("spatial_z_halo", z_halo), ("profile_median_z", z_base)):
        hot = evaluate_events(scene["hot"], z >= 3.5, scene["time_s"], scene["x_m"], scene["valid"])
        cold = evaluate_events(scene["cold"], z <= -3.5, scene["time_s"], scene["x_m"], scene["valid"])
        report["models"][name] = {
            "hot": _round(hot),
            "cold": _round(cold),
            "alarm_fraction": round(alarm_fraction(np.abs(z) >= 3.5, scene["valid"]), 4),
            "dead_tail_alarms": int(np.count_nonzero(np.abs(z[:, 180:]) >= 3.5)),
        }
    broad = half_cable_shift()
    z_halo, _, _ = spatial_z(broad["clean"], broad["valid"], broad["x_m"])
    z_base = profile_median_z(broad["clean"], broad["valid"])
    report["half_cable_warm_alarm_fraction"] = {
        "spatial_z_halo": round(alarm_fraction(np.abs(z_halo) >= 3.5, broad["broad"]), 4),
        "profile_median_z": round(alarm_fraction(np.abs(z_base) >= 3.5, broad["broad"]), 4),
    }
    return report


def _window(x_m: np.ndarray, n_rows: int, spec: dict) -> tuple[slice, np.ndarray]:
    r0, r1 = spec["rows"]
    r1 = min(r1, n_rows)
    cols = (x_m >= spec["x_m"][0]) & (x_m <= spec["x_m"][1])
    return slice(r0, r1), cols


def score_witnesses(records: Path, denoise: Path) -> list[dict]:
    payload = json.loads((ROOT / "fiber" / "benchmark" / "witnesses.json").read_text(encoding="utf-8"))
    rows = []
    cache: dict[str, tuple] = {}
    for spec in payload["witnesses"]:
        name = spec["record"]
        record_dir = records / name
        clean_path = denoise / name / "median_x3.npy"
        if not record_dir.exists() or not clean_path.exists():
            rows.append({"id": spec["id"], "status": "missing_arrays"})
            continue
        if name not in cache:
            record = Record.load(record_dir)
            if record.source_sha256 != spec["source_sha256"]:
                raise ValueError(f"{name} sha changed; witnesses do not match this CSV")
            clean = np.load(clean_path)
            z = spatial_z(clean, record.valid, record.x_m)[0]
            cache[name] = (record, z)
        record, z = cache[name]
        sl, cols = _window(record.x_m, z.shape[0], spec)
        window_valid = record.valid[sl][:, cols]
        hot = (z[sl][:, cols] >= 3.5) & window_valid
        cold = (z[sl][:, cols] <= -3.5) & window_valid
        n_valid = int(window_valid.sum())
        hot_frac = float(hot.sum() / n_valid) if n_valid else 0.0
        cold_frac = float(cold.sum() / n_valid) if n_valid else 0.0
        role = spec["role"]
        if role == "positive_hot":
            passed = hot_frac >= 0.3
        elif role == "negative_quiet":
            passed = hot_frac <= 0.1 and cold_frac <= 0.1
        elif role == "negative_invalid":
            passed = n_valid == 0 and int(np.count_nonzero(np.abs(z[sl][:, cols]) >= 3.5)) == 0
        else:
            passed = False
        rows.append({
            "id": spec["id"],
            "record": name,
            "role": role,
            "valid_cells": n_valid,
            "hot_fraction": round(hot_frac, 4),
            "cold_fraction": round(cold_frac, 4),
            "pass": passed,
            "expert_confirmed": False,
        })
    return rows


def main() -> None:
    report = {
        "cards": MODEL_CARDS,
        "synthetic": score_synthetic(),
        "witnesses": score_witnesses(ROOT / "artifacts" / "records", ROOT / "artifacts" / "denoise"),
        "not_measured": [
            "No expert labels, so real-cable precision is unknown.",
            "Weak DSP targets from general-ml-analyze are not used as truth: their teacher abstains on the 5.csv warm region.",
            "Conv-AE and CUSUM are described on the card and are not in this decision table.",
        ],
    }
    out = ROOT / "artifacts" / "benchmark"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    syn = report["synthetic"]["models"]
    print("synthetic local_hot_cold")
    for name, block in syn.items():
        print(
            f"  {name}: hot F1={block['hot']['event_f1']} cold F1={block['cold']['event_f1']} "
            f"alarm={block['alarm_fraction']} dead={block['dead_tail_alarms']}"
        )
    print("half-cable alarm", report["synthetic"]["half_cable_warm_alarm_fraction"])
    print("witnesses")
    for row in report["witnesses"]:
        print(
            f"  {row['id']}: pass={row.get('pass')} hot={row.get('hot_fraction')} "
            f"cold={row.get('cold_fraction')} valid={row.get('valid_cells')}"
        )


if __name__ == "__main__":
    main()
