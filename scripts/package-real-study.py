"""Copy only reviewed public aggregate reports, never private arrays or journals."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--ml-checkout", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
names = [
    "report.json",
    "protocol.json",
    "temporal-report.json",
    "classification-report.json",
]
if args.output.exists():
    parser.error("Choose a new immutable destination")
args.output.mkdir(parents=True)
files = {}
for name in names:
    data = subprocess.check_output(
        ["git", "show", "HEAD:docs/evidence/ml-next/" + name], cwd=args.ml_checkout
    )
    payload = json.loads(data)
    if payload.get("operational_approved") is True:
        raise ValueError("Expected research-only evidence")
    (args.output / name).write_bytes(data)
    files[name] = hashlib.sha256(data).hexdigest()
commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=args.ml_checkout, text=True
).strip()
(args.output / "manifest.json").write_text(
    json.dumps(
        {
            "schema_version": "dts-real-study-pack-v1",
            "source_commit": commit,
            "files": files,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
    newline="\n",
)
print("Immutable public aggregate study packaged:", commit)
