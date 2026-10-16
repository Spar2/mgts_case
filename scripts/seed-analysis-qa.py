"""Prepare a transmission-only package in an explicitly isolated QA schema."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "services/web-api"))
from workbench.analysis_results import import_package, sha
from workbench.platform import Platform
from workbench.store import Store

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--state", type=Path, required=True)
parser.add_argument("--records", type=Path, required=True)
args = parser.parse_args()
if not args.state.resolve().is_relative_to(BASE / ".runtime"):
    parser.error("Private state must stay under .runtime")
state = json.loads(args.state.read_text(encoding="utf-8"))
if not state["schema"].startswith("qa_browser_"):
    parser.error("Isolated QA schema required")
platform = Platform(state["url"], args.state.parent / "runtime", initialize=False)
store = Store(args.records, args.state.parent / "seed-cache")
key = next(iter(store.records))
native = store.get(key)
rows = min(8, native.raw_c.shape[0])
columns = min(16, native.raw_c.shape[1])
directory = args.state.parent / "analysis-contract-fixture"
directory.mkdir(exist_ok=False)
raw = native.raw_c[:rows, :columns]
valid = native.invalid_reason[:rows, :columns] == 0
np.save(directory / "x_m.npy", native.x_m[:columns])
np.save(directory / "time_s.npy", native.elapsed_s[:rows])
np.save(directory / "processed_temperature_c.npy", raw)
np.save(directory / "processed_temperature_c_valid.npy", valid)
score = np.where(valid, raw, 0).astype(np.float64)
np.save(directory / "score.npy", score)
np.save(directory / "score_valid.npy", valid)
(directory / "zones.json").write_text(
    json.dumps(
        [
            {
                "id": "qa-viewport",
                "row_start": 0,
                "row_end": rows,
                "column_start": 0,
                "column_end": columns,
                "shape_hypothesis": "QA viewport, not detector output",
            }
        ]
    ),
    encoding="utf-8",
)
m = {
    "schema_version": "dts-analysis-result-v2",
    "region": {
        "record_id": key,
        "row_start": 0,
        "row_end": rows,
        "column_start": 0,
        "column_end": columns,
    },
    "source_sha256": native.manifest["source_sha256"],
    "method": "qa-transmission-only",
    "method_version": "contract-v2",
    "parameters": {"identity": True},
    "calculated_at": "2026-10-13T00:00:00+03:00",
    "training_origin": "untrained",
    "research_status": "reference",
    "limitations": "Isolated UI contract fixture; identity temperatures; no training, labels or detector quality claim",
    "score": {
        "name": "QA numeric layer",
        "unit": "technical",
        "meaning": "Temperature values used only to test numeric layer transmission, not anomaly scores",
        "kind": "signed",
    },
    "files": {p.name: sha(p) for p in directory.iterdir()},
}
(directory / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
with platform.connect() as db:
    author = db.execute("SELECT id FROM users WHERE username='qa-admin'").fetchone()[0]
run = import_package(platform, store, directory, "local", author)
print(
    json.dumps(
        {
            "run_id": run["id"],
            "scope": "isolated transmission fixture, no model training",
        }
    )
)
