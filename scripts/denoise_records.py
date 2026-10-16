"""Run the denoise zoo on transformed records and store residuals."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fiber.denoise import METHODS, apply_method, residual
from fiber.io import Record
from fiber.metrics import residual_stats


DEFAULT_METHODS = [
    "median_x3",
    "median_xt",
    "savgol_x",
    "bilateral_x",
    "wavelet_x",
    "tv_x",
]


def process_record(record_dir: Path, out_root: Path, methods: list[str]) -> dict:
    record = Record.load(record_dir)
    summary = {
        "name": record.name,
        "shape": list(record.raw_c.shape),
        "n_valid": record.n_valid,
        "n_sentinel": record.n_sentinel,
        "methods": {},
    }
    dest = out_root / record.name
    dest.mkdir(parents=True, exist_ok=True)
    for name in methods:
        t0 = time.perf_counter()
        denoised = apply_method(name, record.raw_c, record.valid)
        noise = residual(record.raw_c, denoised, record.valid)
        elapsed = time.perf_counter() - t0
        np.save(dest / f"{name}.npy", denoised)
        np.save(dest / f"{name}_residual.npy", noise)
        stats = residual_stats(noise, record.valid)
        stats["elapsed_s"] = round(elapsed, 3)
        stats["family"] = METHODS.get(name, {}).get("family", "research")
        summary["methods"][name] = stats
        print(f"  {record.name:4} {name:16} mad={stats['mad']:.4f} p95={stats['p95']:.4f} {elapsed:.2f}s")
    (dest / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=ROOT / "artifacts" / "records")
    parser.add_argument("--out-root", type=Path, default=ROOT / "artifacts" / "denoise")
    parser.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    parser.add_argument("--only", nargs="*", default=None, help="record names, e.g. 5 6")
    args = parser.parse_args()
    names = args.only
    index = json.loads((args.records / "index.json").read_text(encoding="utf-8"))
    reports = []
    for item in index:
        if names and item["name"] not in names:
            continue
        print(f"record {item['name']}")
        reports.append(process_record(args.records / item["name"], args.out_root, args.methods))
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "index.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8",
    )


if __name__ == "__main__":
    main()
