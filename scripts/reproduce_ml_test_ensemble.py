"""Instrument the untouched external ensemble on real CSVs, saving branch masks.

Wrappers record outputs/timings and save the locally trained CNN state.
No source algorithm, epoch count, forest setting or raw value is replaced.
"""
import argparse
from collections import Counter
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT/"artifacts/matplotlib"))
sys.path.insert(0, str(ROOT))
from fiber.io import Record, file_sha256
sys.path.insert(0, str(ROOT.parent/"ml-test-review"))
import main as external
import src.detector_cnn as cnn

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--records", nargs="+", default=["4"])
parser.add_argument("--threads", type=int, default=4)
args = parser.parse_args()
torch.set_num_threads(args.threads)
out = ROOT/"artifacts/ml-test-audit/reproduced"
out.mkdir(parents=True, exist_ok=True)
reported = json.loads((out.parent/"report_audit.json").read_text(encoding="utf-8"))
witnesses = json.loads((ROOT/"fiber/benchmark/witnesses.json").read_text())["witnesses"]
original_cv, original_cnn, original_ml = external.run_cv_pipeline, external.run_cnn_detector, external.run_ml_detector
original_train = cnn.train_autoencoder
versions = {key: importlib.metadata.version(key) for key in ("numpy", "scipy", "pandas", "torch", "scikit-learn", "catboost", "opencv-python-headless")}

for name in args.records:
    directory = out/name
    if (directory/"receipt.json").exists():
        print(f"record={name} already completed; preserved", flush=True)
        continue
    directory.mkdir(parents=True, exist_ok=True)
    rec = Record.load(ROOT/"artifacts/records"/name)
    raw, x, clocks = external.load_mgts_csv(ROOT.parent/(name+".csv"))
    captured, timings = {}, {}
    def record_cv(*a, **kw):
        started = time.perf_counter(); result = original_cv(*a, **kw)
        captured["cv"] = result; timings["cv_s"] = time.perf_counter()-started
        return result
    def record_cnn(*a, **kw):
        started = time.perf_counter(); result = original_cnn(*a, **kw)
        captured["cnn"] = result; timings["cnn_s"] = time.perf_counter()-started
        return result
    def record_ml(*a, **kw):
        started = time.perf_counter(); result = original_ml(*a, **kw)
        captured["ml"] = result; timings["ml_s"] = time.perf_counter()-started
        return result
    def save_trained_cnn(model, patches, *a, **kw):
        result = original_train(model, patches, *a, **kw)
        torch.save(model.state_dict(), directory/"cnn_weights.pt")
        return result
    external.run_cv_pipeline, external.run_cnn_detector, external.run_ml_detector = record_cv, record_cnn, record_ml
    cnn.train_autoencoder = save_trained_cnn
    started = time.perf_counter()
    result = external.run_ensemble(raw, x, clocks)
    timings["ensemble_s"] = time.perf_counter()-started
    masks = {"cv": captured["cv"]["binary_mask"] != 0, "cnn": captured["cnn"][1],
             "ml": captured["ml"].binary_mask, "consensus": result.consensus_mask}
    for key, mask in masks.items():
        np.save(directory/(key+"_mask.npy"), mask)
    np.save(directory/"residual.npy", result.residual)
    np.save(directory/"cnn_score.npy", captured["cnn"][0])
    np.save(directory/"ml_score.npy", captured["ml"].score_map)
    result.catalog.to_csv(directory/"catalog.csv", index=False)
    report_row = next(r for r in reported["records"] if r["file"] == "MGTS_data/"+name+".csv")
    witness_results = []
    own_z = np.load(ROOT/"artifacts/anomaly"/name/"z.npy", mmap_mode="r") if (ROOT/"artifacts/anomaly"/name/"z.npy").exists() else None
    for witness in witnesses:
        if witness["record"] != name:
            continue
        a, b = witness["rows"]
        cols = np.flatnonzero((x >= witness["x_m"][0]) & (x <= witness["x_m"][1]))
        selection = np.ix_(np.arange(a, b), cols)
        valid = rec.valid[selection]
        residual = result.residual[selection]
        fractions = {}
        for key, mask in masks.items():
            hot = mask[selection] & (residual > 0) & valid
            cold = mask[selection] & (residual < 0) & valid
            fractions[key] = {"all_cells_on": float(mask[selection].mean()),
                "positive_fraction_valid": float(hot.sum()/valid.sum()) if valid.any() else None,
                "negative_fraction_valid": float(cold.sum()/valid.sum()) if valid.any() else None}
        witness_results.append({"id": witness["id"], "role": witness["role"], "branches": fractions,
            "expert_confirmed": False, "scope": "development partial core; different residual semantics"})
    receipt = {"record": name, "source_sha256": file_sha256(ROOT.parent/(name+".csv")),
               "source_commit": reported["source_commit"], "device": "cuda" if torch.cuda.is_available() else "cpu",
               "threads": args.threads, "versions": versions, "timings": timings,
               "mask_counts": {k: int(v.sum()) for k, v in masks.items()},
               "invalid_cells_on": {k: int((v & ~rec.valid).sum()) for k, v in masks.items()},
               "mask_sha256": external.mask_sha256(result.consensus_mask),
               "matches_shipped_report_sha256": external.mask_sha256(result.consensus_mask) == report_row["consensus_sha256"],
               "catalog_rows": len(result.catalog), "classes": dict(Counter(result.catalog["incident_type"])),
               "cnn_weights_sha256": file_sha256(directory/"cnn_weights.pt"),
               "witnesses": witness_results, "synthetic_training_used": False, "expert_truth_available": False}
    (directory/"receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=True), flush=True)
