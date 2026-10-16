"""Load the research weights and run one forward pass on a real residual crop."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fiber.denoise.dsp import median_x
from fiber.io import Record
from fiber.research_cnn import LABELS, load_research_cnn, meter_columns, residual_crop
from fiber.weak_labels import EXPERT_SEEDS

MANIFEST = ROOT / "docs" / "evidence" / "research_cnn_manifest.json"


def main():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    model, blob, device = load_research_cnn(ROOT / manifest["weights"], device="cpu")
    seed = EXPERT_SEEDS[0]
    record = Record.load(ROOT / "artifacts" / "records" / seed["record"])
    clean = median_x(record.raw_c, record.valid)
    t0, t1 = seed["rows_half_open"]
    c0, c1 = meter_columns(record.x_m, seed["x_m"][0], seed["x_m"][1])
    crop = residual_crop(clean, record.valid, t0, t1, c0, c1)
    if crop is None:
        raise SystemExit("real residual crop was empty")
    import torch
    with torch.inference_mode():
        scores = model(torch.from_numpy(crop[None, None, :, :])).cpu().numpy()[0]
    finite = bool(np.isfinite(scores).all())
    print(f"serve={blob['serve']} architecture {blob['architecture']} device {device}", flush=True)
    print(f"crop {seed['id']} finite {finite} scores {scores.tolist()}", flush=True)
    print("classes " + " ".join(f"{name}={float(score):.6f}" for name, score in zip(LABELS, scores, strict=True)), flush=True)
    if blob["serve"] != "research" or not finite:
        raise SystemExit("forward failed")
    print("finished", flush=True)


if __name__ == "__main__":
    main()
