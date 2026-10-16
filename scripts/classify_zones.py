"""Apply the frozen shape rules to zones already extracted by spatial Z."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fiber.classify.rules import classify_zones, profile_shift
from fiber.io import Record


def main() -> None:
    anomaly = ROOT / "artifacts" / "anomaly"
    records = ROOT / "artifacts" / "records"
    denoise = ROOT / "artifacts" / "denoise"
    summary = {}
    for zone_path in sorted(anomaly.glob("*/zones.json")):
        name = zone_path.parent.name
        zones = json.loads(zone_path.read_text(encoding="utf-8"))
        record = Record.load(records / name)
        valid_x = record.x_m[record.valid.any(axis=0)]
        cable_m = float(valid_x[-1] - valid_x[0]) if valid_x.size else 1.0
        labels = classify_zones(zones, cable_m)
        clean_path = denoise / name / "median_x3.npy"
        shift = {"label": "missing_clean"}
        if clean_path.exists():
            clean = np.load(clean_path)
            shift = profile_shift(clean, record.valid, record.x_m)
        counts = dict(Counter(labels))
        summary[name] = {
            "cable_length_m": round(cable_m, 3),
            "n_zones": len(zones),
            "zone_labels": counts,
            "profile_shift": shift,
        }
        print(f"{name}: zones={len(zones)} {counts} profile={shift['label']} delta={shift.get('delta_c')}")
    dest = anomaly / "classes.json"
    dest.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", dest)


if __name__ == "__main__":
    main()
