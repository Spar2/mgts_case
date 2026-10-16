"""Run the explicit ML reference pipeline on a real native window and package it."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "services" / "web-api"))
from workbench.analysis_results import validate
from workbench.store import Store

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--ml-checkout", type=Path, required=True)
parser.add_argument("--records", type=Path, required=True)
parser.add_argument("--record", required=True)
parser.add_argument("--rows", nargs=2, type=int, required=True)
parser.add_argument("--columns", nargs=2, type=int, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
commit = subprocess.check_output(
    ["git", "-C", str(args.ml_checkout), "rev-parse", "HEAD"], text=True
).strip()
if subprocess.check_output(
    ["git", "-C", str(args.ml_checkout), "status", "--porcelain", "--", "fiber"],
    text=True,
).strip():
    parser.error("ML fiber source must be committed")
sys.path.insert(0, str(args.ml_checkout.resolve()))
from fiber.pipeline import analyze_field

store = Store(args.records, BASE / ".runtime" / "ml-field-cache")
native = store.get(args.record)
rs, re = args.rows
cs, ce = args.columns
if (
    not (
        0 <= rs < re <= native.raw_c.shape[0] and 0 <= cs < ce <= native.raw_c.shape[1]
    )
    or (re - rs) * (ce - cs) > 1_000_000
):
    parser.error("Use a native window of at most 1,000,000 cells")
x, t = native.x_m[cs:ce], native.elapsed_s[rs:re]
good = native.invalid_reason[rs:re, cs:ce] == 0
raw = native.raw_c[rs:re, cs:ce]
if np.any(good & (raw <= -200)):
    parser.error("ML quality convention would discard a valid native measurement")
result = analyze_field(
    raw, x, t, valid=good, time_iso=[p["timestamp"] for p in native.profiles[rs:re]]
)
from datetime import datetime, timezone

from workbench.analysis_compute import package_result

package_result(
    native,
    {
        "record_id": args.record,
        "row_start": rs,
        "row_end": re,
        "column_start": cs,
        "column_end": ce,
    },
    result,
    args.output,
    commit,
    datetime.now(timezone.utc).isoformat(),
)
manifest = json.loads((args.output / "manifest.json").read_text(encoding="utf-8"))
validate(args.output, store)
print(
    json.dumps(
        {
            "verified": True,
            "ml_commit": commit,
            "zone_count": len(
                json.loads((args.output / "zones.json").read_text(encoding="utf-8"))
            ),
            "method": manifest["method"],
        }
    )
)
