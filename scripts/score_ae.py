"""Score records with the trained background autoencoder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fiber.io import Record
from fiber.models.conv_ae import reconstruction_z


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=ROOT / "artifacts" / "records")
    parser.add_argument("--denoise", type=Path, default=ROOT / "artifacts" / "denoise")
    parser.add_argument("--anomaly", type=Path, default=ROOT / "artifacts" / "anomaly")
    parser.add_argument("--weights", type=Path, default=ROOT / "models" / "weights" / "conv_ae_bg.pt")
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    index = json.loads((args.records / "index.json").read_text(encoding="utf-8"))
    for item in index:
        name = item["name"]
        if args.only and name not in args.only:
            continue
        record = Record.load(args.records / name)
        clean = np.load(args.denoise / name / "median_x3.npy")
        z = reconstruction_z(clean, record.valid, args.weights)
        dest = args.anomaly / name
        np.save(dest / "z_ae.npy", z)
        core = (record.x_m >= 297) & (record.x_m <= 303) if name == "5" else None
        stats = {
            "name": name,
            "median_abs_z": round(float(np.median(np.abs(z[record.valid]))), 3),
            "p95_abs_z": round(float(np.percentile(np.abs(z[record.valid]), 95)), 3),
        }
        if core is not None and core.any():
            stats["core_median_abs_z"] = round(float(np.median(np.abs(z[:, core][record.valid[:, core]]))), 3)
            stats["dead_max_abs_z"] = float(np.max(np.abs(z[:, record.x_m >= 1700])))
        print(name, stats)
        (dest / "ae_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
