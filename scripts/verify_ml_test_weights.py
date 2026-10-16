"""Reload each audit CNN checkpoint and reproduce its full saved score map.

Checks artifact usability, not event accuracy. Does not train a model.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "ml-test-review"))
from src.detector_cnn import ConvAutoencoder, extract_patches, robust_scale, stitch_reconstruction_error

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--records", nargs="+", default=list(map(str, range(1, 7))))
args = parser.parse_args()
torch.set_num_threads(4)
for name in args.records:
    directory = ROOT / "artifacts/ml-test-audit/reproduced" / name
    receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["device"] == "cpu", "strict score check uses the same CPU device as this audit"
    state = torch.load(directory / "cnn_weights.pt", weights_only=True, map_location="cpu")
    assert all(torch.isfinite(value).all() for value in state.values())
    model = ConvAutoencoder()
    model.load_state_dict(state, strict=True)
    normalized, scale = robust_scale(np.load(directory / "residual.npy"))
    actual = stitch_reconstruction_error(model, extract_patches(normalized)) * scale
    expected = np.load(directory / "cnn_score.npy")
    np.testing.assert_array_equal(actual, expected)
    result = {"record": name, "weights_only_load": True, "finite_parameters": True,
              "state_parameters": sum(value.numel() for value in state.values()),
              "full_saved_cnn_score_reproduced_exactly": True,
              "max_abs_score_difference_c": float(np.max(np.abs(actual-expected))),
              "no_training_performed": True, "expert_accuracy_claim": False}
    (directory / "weights_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)
