"""Probe the actual CNN architecture without training any synthetic model."""
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent/"ml-test-review"))
from src.detector_cnn import ConvAutoencoder, adaptive_threshold

torch.set_num_threads(2)
torch.manual_seed(44)
model = ConvAutoencoder()
# Avoid the trivially zero output-layer initialization: use nonzero arbitrary
# weights to test architectural expressiveness rather than a trained score.
with torch.no_grad():
    for parameter in model.parameters():
        parameter.normal_(0, .1)
    output = model(torch.randn(3, 1, 32, 32)).numpy()
spread = [float(np.ptp(field)) for field in output[:, 0]]
score = np.zeros((32, 32))
threshold = adaptive_threshold(score)
result = {"architecture_probe_no_training": True, "output_spatial_temporal_peak_to_peak_per_patch": spread,
          "decoder_is_constant_within_each_patch": bool(max(spread) < 1e-5),
          "zero_score_threshold": threshold, "zero_score_mask_fraction": float((score >= threshold).mean()),
          "synthetic_training_performed": False, "torch_version": torch.__version__,
          "note": "Random inputs/weights are contract probes; not learning or accuracy evidence."}
out = ROOT/"artifacts/ml-test-audit/cnn_architecture_probe.json"
out.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result), flush=True)
