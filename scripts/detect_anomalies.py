"""Detect local anomalies on median_x3 fields."""

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

from fiber.anomaly import cusum_alarm, temporal_z
from fiber.io import Record
from fiber.pipeline import PipelineConfig, analyze_field


def witness_file5(z: np.ndarray, valid: np.ndarray, x_m: np.ndarray, thresh: float) -> dict:
    core = (x_m >= 297.0) & (x_m <= 303.0)
    dead = x_m >= 1700.0
    core_ok = valid[:, core]
    core_abs = np.abs(z[:, core])
    return {
        "core_valid_cells": int(core_ok.sum()),
        "core_frac_abs_z_ge_thresh": round(float(np.mean(core_abs[core_ok] >= thresh)), 4),
        "core_median_abs_z": round(float(np.median(core_abs[core_ok])), 3),
        "core_p90_abs_z": round(float(np.percentile(core_abs[core_ok], 90)), 3),
        "dead_max_abs_z": float(np.max(np.abs(z[:, dead]))) if dead.any() else None,
    }


def process(name: str, records: Path, denoise: Path, out: Path, thresh: float) -> dict:
    record = Record.load(records / name)
    clean = np.load(denoise / name / "median_x3.npy")
    if clean.shape != record.raw_c.shape:
        raise ValueError(f"{name}: median_x3 shape {clean.shape} != raw {record.raw_c.shape}")
    t0 = time.perf_counter()
    analysis = analyze_field(record.raw_c, record.x_m, record.time_s,
        valid=record.valid, time_iso=record.time_iso, config=PipelineConfig(threshold_z=thresh))
    if not np.array_equal(clean, analysis['clean'], equal_nan=True):
        raise ValueError(f'{name}: stale median_x3; rerun denoise_records.py')
    z, bg, scale = analysis['z'], analysis['background'], analysis['scale']
    available = analysis['score_available']
    z_time = temporal_z(clean, record.valid, record.time_s)
    residual = np.asarray(clean, dtype=np.float32) - bg
    residual[~record.valid] = 0.0
    alarm = cusum_alarm(residual, record.valid, record.time_s, drift_c=0.8, threshold_c=6.0)
    contrast = residual.copy()
    contrast[~record.valid] = np.nan
    zones = analysis['zones']
    dest = out / name
    dest.mkdir(parents=True, exist_ok=True)
    np.save(dest / "z_spatial.npy", z)
    np.save(dest / "score_available.npy", available)
    np.save(dest / "z_temporal.npy", z_time)
    np.save(dest / "cusum_alarm.npy", alarm)
    np.save(dest / "background.npy", bg)
    (dest / "zones.json").write_text(json.dumps(zones, ensure_ascii=False), encoding="utf-8")
    longest = sorted(zones, key=lambda zone: zone["duration_s"], reverse=True)[:5]
    summary = {
        "name": name,
        "dts_id": record.dts_id,
        "channel_id": record.channel_id,
        "shape": list(clean.shape),
        "method": "spatial_z_on_median_x3",
        "threshold": thresh,
        "background": "halo inner 8 m, outer 40 m, bin 2 m",
        "scale_c_median": round(float(np.median(scale)), 4),
        "elapsed_s": round(time.perf_counter() - t0, 2),
        **{key: analysis[key] for key in ('anomaly_status','assessment','n_valid',
            'n_score_available','score_coverage','n_score_unavailable','manifest')},
        "n_zones": len(zones),
        "n_hot": sum(zone["polarity"] == "hot" for zone in zones),
        "n_cold": sum(zone["polarity"] == "cold" for zone in zones),
        "cusum_hot_cells": int((alarm == 1).sum()),
        "cusum_cold_cells": int((alarm == 2).sum()),
        "temporal_p99_abs_z": round(float(np.percentile(np.abs(z_time[record.valid]), 99)), 3),
        "top_zones": zones[:8],
        "longest_zones": longest,
    }
    if name == "5":
        summary["witness"] = witness_file5(z, record.valid, record.x_m, thresh)
    (dest / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"{name}: {summary['anomaly_status']} zones={summary['n_zones']} "
        f"hot={summary['n_hot']} cold={summary['n_cold']} "
        f"scale={summary['scale_c_median']}C {summary['elapsed_s']}s"
    )
    if name == "5":
        w = summary["witness"]
        print(
            f"  witness 297-303m median|z|={w['core_median_abs_z']} "
            f"frac>={thresh}={w['core_frac_abs_z_ge_thresh']} dead_max|z|={w['dead_max_abs_z']}"
        )
    return {k: v for k, v in summary.items() if k != "top_zones"} | {"top_zones": zones[:5]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=ROOT / "artifacts" / "records")
    parser.add_argument("--denoise", type=Path, default=ROOT / "artifacts" / "denoise")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "anomaly")
    parser.add_argument("--thresh", type=float, default=3.5)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    index = json.loads((args.records / "index.json").read_text(encoding="utf-8"))
    reports = []
    for item in index:
        if args.only and item["name"] not in args.only:
            continue
        reports.append(process(item["name"], args.records, args.denoise, args.out, args.thresh))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "index.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
