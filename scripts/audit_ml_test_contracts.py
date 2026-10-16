"""Small contract counterexamples for the actual external functions.

No model is trained on these constructed fields; probes are not accuracy tests.
"""
import io
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT/"artifacts/matplotlib"))
sys.path.insert(0, str(ROOT.parent/"ml-test-review"))
from main import load_mgts_csv, numeric_timestamps
from src.detector_cv import run_cv_pipeline, _one_sign_mask
from src.stage2_clustering import classify_anomalies, _column_periodicity

def csv(rows):
    import csv as module
    handle = io.StringIO()
    writer = module.writer(handle)
    writer.writerow(["dts_id", "channel_id", "timestamp", "points", "points_size"])
    for device, channel, stamp, points in rows:
        writer.writerow([device, channel, stamp, json.dumps(points), len(points)])
    handle.seek(0)
    return handle

raw, _, _ = load_mgts_csv(csv([
    ("a", "1", "2026-01-01 00:00:00", [[0, 22], [1, 22], [2, 22]]),
    ("b", "2", "2026-01-01 00:01:00", [[0, 20], [2, 24]]),
]))
permanent = np.full((48, 64), 22.)
permanent[:, 25:30] += 8
cv = run_cv_pipeline(permanent, np.arange(64.), np.arange(48.)*60)
residual = np.full((4, 8), 4.)
mask = np.zeros_like(residual)
mask[1:3, 3:5] = 1
gap_catalog = classify_anomalies(residual, mask, [1700000000., 1700000060., 1700072060., 1700072120.], np.arange(8.))
relative_catalog = classify_anomalies(residual[:2], np.ones((2, 8)), [0., 60.], np.arange(8.))
first = np.zeros((80, 20), bool)
first[2:30, 10] = True
extra = first.copy()
extra[40:43, 10] = True; extra[50:53, 10] = True; extra[65:68, 10] = True
times = 1700000000+np.arange(80)*60
before = classify_anomalies(first.astype(float)*4, first, times, np.arange(20.))
after = classify_anomalies(extra.astype(float)*4, extra, times, np.arange(20.))
irregular = np.zeros(80, bool)
irregular[0:2] = True; irregular[9:11] = True; irregular[35:37] = True
small_event = np.zeros((64, 128))
small_event[12:20, 20:24] = 2.
with_distant_strong_event = small_event.copy()
with_distant_strong_event[36:44, 100:105] = 30.
small_before = _one_sign_mask(small_event)[12:20, 20:24]
small_after = _one_sign_mask(with_distant_strong_event)[12:20, 20:24]

try:
    import streamlit as st
    import inspect
    dataframe_alt_supported = "alt" in inspect.signature(st.dataframe).parameters
except ImportError:
    dataframe_alt_supported = None

result = {
    "constructed_contract_probes_not_training": True,
    "loader_filled_missing_meter_1_c": float(raw[1, 1]),
    "loader_accepted_mixed_devices_channels": True,
    "cv_permanent_8c_hot_strip_residual_abs_max": float(np.max(np.abs(cv["residual_matrix"]))),
    "cv_permanent_8c_hot_strip_mask_on": int(np.count_nonzero(cv["binary_mask"])),
    "stage2_component_across_20h_gap_count": len(gap_catalog),
    "stage2_component_across_20h_duration_min": float(gap_catalog.iloc[0]["duration_min"]),
    "numeric_relative_60s_reported_duration_min": float(relative_catalog.iloc[0]["duration_min"]),
    "same_early_component_periodicity_before": float(before.iloc[0]["periodicity"]),
    "same_early_component_periodicity_after_future_unrelated_runs": float(after.iloc[0]["periodicity"]),
    "same_early_component_type_before": str(before.iloc[0]["incident_type"]),
    "same_early_component_type_after": str(after.iloc[0]["incident_type"]),
    "irregular_3_runs_positive_periodicity": float(_column_periodicity(irregular)),
    "cv_2c_local_event_mask_cells_before_distant_30c_event": int(np.count_nonzero(small_before)),
    "cv_2c_local_event_mask_cells_after_distant_30c_event": int(np.count_nonzero(small_after)),
    "streamlit_dataframe_accepts_alt_argument": dataframe_alt_supported,
    "expert_truth_available": False,
}
(ROOT/"artifacts/ml-test-audit/contract_probes.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=True), flush=True)
