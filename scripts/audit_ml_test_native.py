"""Run the untouched external loader/CV on all six source CSVs and compare native contracts."""
from pathlib import Path
import sys
import json
import time
import os
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT/"artifacts/matplotlib"))
sys.path.insert(0, str(ROOT))
from fiber.io import Record, file_sha256
sys.path.insert(0, str(ROOT.parent/"ml-test-review"))
from main import load_mgts_csv, numeric_timestamps
from src.detector_cv import run_cv_pipeline

OUT = ROOT/"artifacts/ml-test-audit"
results = []
witnesses = json.loads((ROOT/"fiber/benchmark/witnesses.json").read_text())["witnesses"]
for name in map(str, range(1, 7)):
    started = time.perf_counter()
    rec = Record.load(ROOT/"artifacts/records"/name)
    raw, x, clock = load_mgts_csv(ROOT.parent/(name+".csv"))
    times = numeric_timestamps(clock)
    equal_grid = np.array_equal(x, rec.x_m)
    valid = rec.valid
    equal_observed = equal_grid and np.array_equal(raw.astype(np.float32)[valid], rec.raw_c[valid])
    print(f"record={name} loader_shape={raw.shape} native_observed_equal={equal_observed}", flush=True)
    cv = run_cv_pipeline(raw, x, times)
    residual, mask = cv["residual_matrix"], cv["binary_mask"] != 0
    np.save(OUT/(name+"_cv_mask.npy"), mask)
    np.save(OUT/(name+"_cv_residual.npy"), residual)
    comparisons = []
    for witness in witnesses:
        if witness["record"] != name:
            continue
        a, b = witness["rows"]
        columns = (x >= witness["x_m"][0]) & (x <= witness["x_m"][1])
        selection = np.ix_(np.arange(a, b), np.flatnonzero(columns))
        active = mask[selection]
        r = residual[selection]
        eligible = valid[selection]
        comparisons.append({"id": witness["id"], "role": witness["role"],
                            "mask_fraction_all_cells": float(active.mean()),
                            "positive_residual_and_mask_fraction_valid": float((active & (r > 0) & eligible).sum()/eligible.sum()) if eligible.any() else None,
                            "negative_residual_and_mask_fraction_valid": float((active & (r < 0) & eligible).sum()/eligible.sum()) if eligible.any() else None,
                            "residual_median_c_valid": float(np.median(r[eligible])) if eligible.any() else None})
    finite_missing = (~valid) & np.isfinite(raw) & ~np.isfinite(rec.raw_c)
    stat = {"record": name, "source_sha256": file_sha256(ROOT.parent/(name+".csv")), "shape": list(raw.shape),
            "native_grid_equal": equal_grid, "observed_values_equal_after_float32_cast": equal_observed,
            "invalid_source_cells_seen_as_numeric": int((~valid & np.isfinite(raw)).sum()),
            "missing_native_cells_filled_by_external_loader": int(finite_missing.sum()),
            "cv_mask_on": int(mask.sum()), "cv_invalid_cells_on": int((mask & ~valid).sum()),
            "residual_abs_max_c": float(np.max(np.abs(residual))),
            "cnn_window_width_m": float(31*np.median(np.diff(x))),
            "cnn_window_typical_elapsed_s": float(31*np.median(np.diff(times))),
            "naive_timestamp_in_source": rec.tz_missing,
            "numeric_time_axis_equal_native": bool(np.array_equal(times, rec.time_s)),
            "witnesses": comparisons, "seconds_loader_cv": time.perf_counter()-started}
    results.append(stat)
    (OUT/"native_cv_audit.json").write_text(json.dumps({"records": results,
        "scope": "all native loader and CV; not CNN/IF consensus", "expert_confirmed": False}, indent=2), encoding="utf-8")
    print(json.dumps(stat), flush=True)
