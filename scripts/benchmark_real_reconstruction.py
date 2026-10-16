"""Real-only, spatially masked reconstruction experiment on chronological splits.

This predicts noisy held-out observations, NOT clean truth. Correlated DTS
noise can invalidate a denoising interpretation. No synthetic scenes, anomaly
labels, future profiles or operational model replacement are used.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fiber.io import Record, file_sha256
from fiber.real_data import temporal_partition

OFFSETS_M = [-8., -4., -2., 2., 4., 8.]


def neighbour_indices(x, col):
    # Far side rounding ensures no predictor lies INSIDE the two-metre blind spot.
    target = x[col]
    ids = [int(np.searchsorted(x, target+d, side="right")-1) if d < 0
           else int(np.searchsorted(x, target+d)) for d in OFFSETS_M]
    if min(ids) < 0 or max(ids) >= len(x) or len(set(ids)) != 6:
        return None
    if np.any(np.abs(x[ids]-target) < 2-1e-8):
        raise ValueError("predictor leaked inside spatial blind spot")
    if np.any(np.abs(x[ids]-target-np.array(OFFSETS_M)) > 1.5*np.median(np.diff(x))):
        return None
    return ids


def samples(rec):
    split, manifest = temporal_partition(rec.time_s)
    # Physical anchor spacing >=16m makes neighbouring predictor footprints disjoint.
    anchors = []
    next_m = float(rec.x_m[0])+8
    for col, distance in enumerate(rec.x_m):
        if distance >= next_m:
            anchors.append(col); next_m = float(distance)+17
    features, targets, codes = [], [], []
    observed_offsets = []
    for col in anchors:
        ids = neighbour_indices(rec.x_m, col)
        if ids is None:
            continue
        for row in range(0, rec.n_profiles, 5):
            # Avoid crossing a missing spatial island. Only the target VALUE is hidden.
            if split[row] not in (0, 1) or not rec.valid[row, min(ids):max(ids)+1].all():
                continue
            neighbourhood = rec.raw_c[row, ids].astype(float)
            target = float(rec.raw_c[row, col])
            if not np.isfinite(neighbourhood).all() or not np.isfinite(target):
                continue
            features.append(neighbourhood); targets.append(target); codes.append(int(split[row]))
            observed_offsets.append((rec.x_m[ids]-rec.x_m[col]).tolist())
    return np.array(features).reshape(-1, 6), np.array(targets), np.array(codes), manifest, observed_offsets


def score(y, prediction):
    errors = prediction-y
    return {"n": len(y), "mse_noisy_target_c2": float(np.mean(errors**2)),
            "mae_noisy_target_c": float(np.mean(np.abs(errors))),
            "abs_error_q95_c": float(np.quantile(np.abs(errors), .95)),
            "bias_c": float(np.mean(errors))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT/"artifacts/real-reconstruction-v1")
    args = parser.parse_args()
    if (args.out/"report.json").exists():
        raise SystemExit("Use a new --out: completed experiments are immutable.")
    args.out.mkdir(parents=True, exist_ok=True)
    all_x, all_y, all_codes, all_record = [], [], [], []
    sources = {}
    for name in map(str, range(1, 7)):
        rec = Record.load(ROOT/"artifacts/records"/name)
        if file_sha256(Path(rec.source_csv)) != rec.source_sha256:
            raise ValueError("source CSV changed")
        x, y, codes, split, offsets = samples(rec)
        all_x.append(x); all_y.append(y); all_codes.append(codes)
        all_record.extend([name]*len(y))
        sources[name] = {"source_sha256": rec.source_sha256, "split": split,
                         "n_train": int((codes == 0).sum()), "n_development": int((codes == 1).sum()),
                         "actual_predictor_offset_range_m": [np.min(offsets, axis=0).tolist(), np.max(offsets, axis=0).tolist()] if offsets else None}
    x, y, codes = np.concatenate(all_x), np.concatenate(all_y), np.concatenate(all_codes)
    records = np.array(all_record)
    train, dev = codes == 0, codes == 1
    if train.sum() < 20 or dev.sum() < 20:
        raise ValueError("not enough observed real training/development anchors")
    # Offset equivariance: learn deviations from the neighbour mean.
    base = x.mean(axis=1)
    centered = x-base[:, None]
    mu, sigma = centered[train].mean(axis=0), centered[train].std(axis=0)
    sigma = np.maximum(sigma, 1e-6)
    design = np.column_stack([np.ones(len(x)), (centered-mu)/sigma])
    target = y-base
    candidates = []
    weights = []
    for alpha in (.1, 1., 10., 100.):
        penalty = np.eye(design.shape[1])*alpha
        penalty[0, 0] = 0
        weight = np.linalg.solve(design[train].T@design[train]+penalty, design[train].T@target[train])
        prediction = base+design@weight
        candidates.append({"alpha": alpha, "train": score(y[train], prediction[train]), "development": score(y[dev], prediction[dev])})
        weights.append(weight)
    best_index = min(range(len(candidates)), key=lambda i: candidates[i]["development"]["mse_noisy_target_c2"])
    selected = candidates[best_index]
    prediction = base+design@weights[best_index]
    baselines = {"neighbour_mean": base, "neighbour_median": np.median(x, axis=1),
                 "nearest_symmetric_pair": (x[:, 2]+x[:, 3])/2}
    per_record = {}
    for name in sources:
        mask = dev & (records == name)
        if mask.any():
            per_record[name] = {"ridge": score(y[mask], prediction[mask]),
                                **{key: score(y[mask], value[mask]) for key, value in baselines.items()}}
    baseline_scores = {key: score(y[dev], value[dev]) for key, value in baselines.items()}
    # Gate is deliberately NOT MSE alone: clean-truth/fidelity evidence is missing.
    model = {"schema": "real_masked_ridge_v1", "serve": "research_offline",
             "synthetic_training_used": False, "expert_confirmed": False,
             "noise_independence_verified": False, "operational_approved": False,
             "offsets_m": OFFSETS_M, "rounding": "away_from_target",
             "blind_radius_m": 2., "alpha": selected["alpha"],
             "mu": mu.tolist(), "sigma": sigma.tolist(), "weight": weights[best_index].tolist(),
             "prediction_formula": "mean(neighbours)+weight[0]+dot((neighbours-mean(neighbours)-mu)/sigma, weight[1:])",
             "sources": {name: entry["source_sha256"] for name, entry in sources.items()},
             "train_samples": int(train.sum()), "selection_samples": int(dev.sum())}
    model_path = args.out/"real_masked_ridge.json"
    model_path.write_text(json.dumps(model, indent=2, allow_nan=False), encoding="utf-8")
    report = {"protocol": "real_only_masked_reconstruction_v1", "sources": sources,
              "candidates": candidates, "selected_alpha": selected["alpha"], "baselines_development": baseline_scores,
              "per_record_development": per_record, "model_sha256": file_sha256(model_path),
              "script_sha256": file_sha256(Path(__file__)), "split_code_sha256": file_sha256(ROOT/"fiber/real_data.py"),
              "synthetic_training_used": False, "n_train": int(train.sum()), "n_development": int(dev.sum()),
              "holdout_used_for_fitting_or_selection": False,
              "clean_truth_available": False, "noise_independence_verified": False,
              "operational_approved": False,
              "limits": ["Noisy-target reconstruction loss is not clean-signal error.",
                         "Spatially correlated noise may reward retaining noise.",
                         "All six records already observed; this is development, not unseen validation.",
                         "Selection uses pooled MSE, which can overweight long fibres.",
                         "Two-metre masking can blur narrow events; event fidelity not certified.",
                         "No future profiles or target value in features; raw neighbouring measurements only."]}
    (args.out/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"n_train": int(train.sum()), "n_development": int(dev.sum()), "selected": selected,
                      "baselines": baseline_scores, "operational_approved": False}), flush=True)


if __name__ == "__main__":
    main()
