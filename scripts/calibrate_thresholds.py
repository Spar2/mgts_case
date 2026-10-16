"""Fit matched-FAR thresholds. Does not change analyze_field."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fiber.benchmark.threshold_cal import (  # noqa: E402
    attach_real,
    dump_json,
    jsonify,
    parse_seed_spec,
    run_calibration,
    summary_payload,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate denoiser thresholds at a matched false-alarm rate")
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--scales", nargs="+", required=True)
    parser.add_argument("--calib-seeds", required=True)
    parser.add_argument("--holdout-seeds", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", required=True)
    parser.add_argument("--real-root", default=None)
    parser.add_argument("--halo-grid", action="store_true")
    args = parser.parse_args()
    report = run_calibration(
        methods=list(args.methods),
        scales=list(args.scales),
        calib_seeds=parse_seed_spec(args.calib_seeds),
        holdout_seeds=parse_seed_spec(args.holdout_seeds),
        device=args.device,
        halo_grid=bool(args.halo_grid),
    )
    if args.real_root:
        attach_real(report, Path(args.real_root), device=args.device)
    out = Path(args.out)
    dump_json(out, report)
    print("SUMMARY " + json.dumps(jsonify(summary_payload(report)), ensure_ascii=False, sort_keys=True), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
