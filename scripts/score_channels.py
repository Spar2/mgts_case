"""Spatial Z on several clean channels. Threshold and witness windows stay frozen.

A higher hot fraction inside an already-passing window is not a reason to
retune 3.5. The decision uses pass/fail, synthetic event F1, dead-tail
alarms and the whole-file alarm fraction.
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

from fiber.benchmark.channels import (
    CHANNEL_METHODS,
    Z_THRESHOLD,
    detect_channel,
    score_half_cable_channel,
    score_local_channel,
)
from fiber.benchmark.events import alarm_fraction
from fiber.benchmark.synthetic import half_cable_shift, local_scene
from fiber.io import Record


def _window(x_m: np.ndarray, n_rows: int, spec: dict) -> tuple[slice, np.ndarray]:
    r0, r1 = spec["rows"]
    r1 = min(r1, n_rows - 1)
    cols = (x_m >= spec["x_m"][0]) & (x_m <= spec["x_m"][1])
    return slice(r0, r1 + 1), cols


def _pass(role: str, hot_frac: float, cold_frac: float, n_valid: int, alarm_cells: int) -> bool:
    if role == "positive_hot":
        return hot_frac >= 0.3
    if role == "negative_quiet":
        return hot_frac <= 0.1 and cold_frac <= 0.1
    if role == "negative_invalid":
        return n_valid == 0 and alarm_cells == 0
    return False


def score_witness_channels(records: Path) -> dict:
    payload = json.loads((ROOT / "fiber" / "benchmark" / "witnesses.json").read_text(encoding="utf-8"))
    specs = payload["witnesses"]
    by_record: dict[str, list[dict]] = {}
    for spec in specs:
        by_record.setdefault(spec["record"], []).append(spec)
    loaded = {}
    for name in by_record:
        record = Record.load(records / name)
        for spec in by_record[name]:
            if record.source_sha256 != spec["source_sha256"]:
                raise ValueError(f"{name} sha changed; witnesses do not match this CSV")
        loaded[name] = record
    report: dict[str, dict] = {}
    for method in CHANNEL_METHODS:
        rows = []
        alarms = {}
        started = time.perf_counter()
        for name, record in loaded.items():
            z = detect_channel(method, record.raw_c, record.valid, record.x_m)
            alarms[name] = round(alarm_fraction(np.abs(z) >= Z_THRESHOLD, record.valid), 4)
            for spec in by_record[name]:
                sl, cols = _window(record.x_m, z.shape[0], spec)
                window_valid = record.valid[sl][:, cols]
                n_valid = int(window_valid.sum())
                hot_frac = float(((z[sl][:, cols] >= Z_THRESHOLD) & window_valid).sum() / n_valid) if n_valid else 0.0
                cold_frac = float(((z[sl][:, cols] <= -Z_THRESHOLD) & window_valid).sum() / n_valid) if n_valid else 0.0
                alarm_cells = int(np.count_nonzero(np.abs(z[sl][:, cols]) >= Z_THRESHOLD))
                rows.append({
                    "id": spec["id"],
                    "record": name,
                    "role": spec["role"],
                    "valid_cells": n_valid,
                    "hot_fraction": round(hot_frac, 4),
                    "cold_fraction": round(cold_frac, 4),
                    "pass": _pass(spec["role"], hot_frac, cold_frac, n_valid, alarm_cells),
                    "expert_confirmed": False,
                })
            print(f"  {method} record {name} alarm={alarms[name]}", flush=True)
        report[method] = {
            "elapsed_s": round(time.perf_counter() - started, 2),
            "alarm_fraction": alarms,
            "witnesses": rows,
            "n_pass": int(sum(row["pass"] for row in rows)),
            "n_witnesses": len(rows),
        }
    return report


def main() -> None:
    print("synthetic", flush=True)
    local = local_scene()
    broad = half_cable_shift()
    synthetic = {}
    for method in CHANNEL_METHODS:
        started = time.perf_counter()
        block = score_local_channel(method, local)
        half = score_half_cable_channel(method, broad)
        block["half_cable_alarm_fraction_inside"] = half["alarm_fraction_inside"]
        block["elapsed_s"] = round(time.perf_counter() - started, 3)
        synthetic[method] = block
        print(
            f"  {method}: hotF1={block['hot']['event_f1']} coldF1={block['cold']['event_f1']} "
            f"alarm={block['alarm_fraction']} dead={block['dead_tail_alarms']} "
            f"half={block['half_cable_alarm_fraction_inside']}",
            flush=True,
        )
    print("witnesses", flush=True)
    witnesses = score_witness_channels(ROOT / "artifacts" / "records")
    for method, block in witnesses.items():
        print(f"  {method}: {block['n_pass']}/{block['n_witnesses']}", flush=True)
        for row in block["witnesses"]:
            print(
                f"    {row['id']}: pass={row['pass']} hot={row['hot_fraction']} cold={row['cold_fraction']}",
                flush=True,
            )
    report = {
        "threshold": Z_THRESHOLD,
        "methods": list(CHANNEL_METHODS),
        "synthetic": synthetic,
        "witnesses": witnesses,
        "note": "Frozen protocol. Witness hits localise a window. They are not cable precision.",
    }
    dest = ROOT / "artifacts" / "benchmark"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "channels.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", dest / "channels.json", flush=True)


if __name__ == "__main__":
    main()
