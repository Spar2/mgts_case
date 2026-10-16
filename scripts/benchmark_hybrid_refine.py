"""Score half_max and region_grow on top of median_x3 zones. Report only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fiber.benchmark.threshold_cal import (  # noqa: E402
    attach_hybrid_real,
    dump_json,
    jsonify,
    parse_seed_spec,
    run_hybrid,
    summary_payload,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine median_x3 zones on an n2n_tsr residual")
    parser.add_argument("--fine", required=True)
    parser.add_argument("--modes", nargs="+", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", required=True)
    parser.add_argument("--real-root", default=None)
    args = parser.parse_args()
    report = run_hybrid(
        fine=args.fine,
        modes=list(args.modes),
        seeds=parse_seed_spec(args.seeds),
        device=args.device,
    )
    if args.real_root:
        attach_hybrid_real(report, Path(args.real_root), device=args.device)
    out = Path(args.out)
    dump_json(out, report)
    print("SUMMARY " + json.dumps(jsonify(summary_payload(report)), ensure_ascii=False, sort_keys=True), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
