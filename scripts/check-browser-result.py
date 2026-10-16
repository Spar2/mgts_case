"""Playwright CLI may print an error and exit zero; require its actual JSON result."""

import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("log", type=Path)
parser.add_argument("--require", nargs="+", required=True)
args = parser.parse_args()
lines = args.log.read_text(encoding="utf-8").splitlines()
if any(line.startswith("### Error") for line in lines):
    parser.exit(1, "Browser scenario reported an error; inspect safe QA diagnostics.\n")
for line in lines:
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        continue
    if isinstance(value, dict) and all(value.get(key) is True for key in args.require):
        print("Browser result confirmed")
        break
else:
    parser.exit(1, "Browser scenario did not return the required success result.\n")
