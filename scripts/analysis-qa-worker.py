"""Run a real pinned worker only against an existing isolated browser QA schema."""

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import psycopg

BASE = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--state", type=Path, required=True)
parser.add_argument("--records", type=Path, required=True)
args = parser.parse_args()
statefile = args.state.resolve()
if not statefile.is_relative_to(BASE / ".runtime"):
    parser.error("Private state must belong to this checkout")
state = json.loads(statefile.read_text(encoding="utf-8"))
if not state["schema"].startswith("qa_browser_"):
    parser.error("Isolated schema required")
with psycopg.connect(state["url"]) as db:
    if db.execute("SELECT current_schema()").fetchone()[0] != state["schema"]:
        parser.error("Existing isolated schema required")
environment = {
    **os.environ,
    "DTS_DATABASE_URL": state["url"],
    "DTS_RUNTIME": str(statefile.parent / "runtime"),
    "DTS_RECORDS": str(args.records.resolve()),
    "DTS_JOB_KINDS": "analysis_field",
    "PYTHONPATH": str(BASE / "services/web-api"),
}
process = subprocess.Popen(
    [sys.executable, "-m", "workbench.worker"],
    env=environment,
    cwd=BASE,
    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
)
(statefile.parent / "worker.pid").write_text(str(process.pid), encoding="utf-8")
for sig in (signal.SIGTERM, signal.SIGINT):
    signal.signal(sig, lambda *_: process.terminate())
try:
    sys.exit(process.wait())
finally:
    if process.poll() is None:
        process.terminate()
        process.wait(timeout=15)
