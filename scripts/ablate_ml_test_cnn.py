"""Compare the learned constant decoder with an untrained per-patch median.

This is an ablation of complexity, not an expert-accuracy measurement.
No model is fitted and no thresholds are retuned.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent/"ml-test-review"))
from src.detector_cnn import extract_patches, _accumulate, robust_scale, adaptive_threshold

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--records", nargs="+", default=["4"])
args = parser.parse_args()
torch.set_num_threads(2)
out = ROOT/"artifacts/ml-test-audit/reproduced"
for name in args.records:
    directory = out/name
    residual = np.load(directory/"residual.npy")
    normalized, scale = robust_scale(residual)
    spec = extract_patches(normalized)
    patch_array = spec.patches.numpy()
    reconstruction = np.median(patch_array, axis=(-1, -2), keepdims=True)
    error = torch.from_numpy(np.abs(patch_array-reconstruction))
    total, count = _accumulate(error, spec)
    t, d = spec.original_hw
    score = ((total/count)[0, 0, :t, :d].numpy()*scale).astype(float)
    threshold = adaptive_threshold(score)
    baseline_mask = score >= threshold
    cnn_score = np.load(directory/"cnn_score.npy")
    cnn_mask = np.load(directory/"cnn_mask.npy")
    cv_mask = np.load(directory/"cv_mask.npy")
    ml_mask = np.load(directory/"ml_mask.npy")
    consensus = baseline_mask.astype(np.uint8)+cv_mask.astype(np.uint8)+ml_mask.astype(np.uint8) >= 2
    original = np.load(directory/"consensus_mask.npy")
    indices = np.arange(0, score.size, max(1, score.size//200000))
    result = {"record": name, "median_decoder_training_steps": 0,
              "score_pearson": float(np.corrcoef(score.ravel()[indices], cnn_score.ravel()[indices])[0, 1]),
              "mean_abs_score_difference_c": float(np.mean(np.abs(score-cnn_score))),
              "mean_reconstruction_abs_error_c": {"cnn": float(cnn_score.mean()), "patch_median": float(score.mean())},
              "branch_mask_jaccard": float((baseline_mask & cnn_mask).sum()/(baseline_mask | cnn_mask).sum()),
              "consensus_mask_jaccard": float((consensus & original).sum()/(consensus | original).sum()),
              "consensus_cell_disagreement": int((consensus != original).sum()),
              "baseline_cells_on": int(baseline_mask.sum()),
              "accuracy_improvement_established": False, "expert_truth_available": False}
    np.save(directory/"patch_median_mask.npy", baseline_mask)
    (directory/"cnn_ablation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)
