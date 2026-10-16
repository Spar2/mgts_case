"""Inspect all reported event bounds against the native timestamps of six CSVs."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fiber.io import Record, parse_timestamp, _time_coordinate

source = ROOT.parent/"ml-test-review"
results = []
record = None
for line in (source/"REPORT.md").read_text(encoding="utf-8").splitlines():
    if line.startswith("## File"):
        name = line.split("/")[-1].split(".")[0]
        record = Record.load(ROOT/"artifacts/records"/name)
        gaps = np.flatnonzero(np.diff(record.time_s) > 300)
        stat = {"record": name, "gaps_gt300s": len(gaps), "catalog_rows": 0,
                "event_bbox_crosses_gap": 0, "timestamp_bounds_not_in_source": 0,
                "invalid_cells_in_native": int((~record.valid).sum()), "examples_crossing_gap": []}
        results.append(stat)
    elif record is not None and line.startswith("| ") and not line.startswith(("| incident_type", "| ---")):
        cells = [s.strip() for s in line.strip("| ").split("|")]
        stat["catalog_rows"] += 1
        start, end = [_time_coordinate(parse_timestamp(cells[k])[0]) for k in (4, 5)]
        a, b = np.searchsorted(record.time_s, [start, end])
        if a >= len(record.time_s) or b >= len(record.time_s) or record.time_s[a] != start or record.time_s[b] != end:
            stat["timestamp_bounds_not_in_source"] += 1
        crossed = gaps[(gaps >= a) & (gaps < b)]
        if len(crossed):
            stat["event_bbox_crosses_gap"] += 1
            if len(stat["examples_crossing_gap"]) < 3:
                stat["examples_crossing_gap"].append({"rows_inclusive": [int(a), int(b)], "time_start": cells[4], "time_end": cells[5],
                    "max_gap_s": float(max(np.diff(record.time_s)[crossed]))})
out = ROOT/"artifacts/ml-test-audit/report_gap_audit.json"
out.write_text(json.dumps({"records": results, "criterion_s": 300, "scope": "reported bounds, not independently reproduced mask"}, indent=2), encoding="utf-8")
print(json.dumps(results), flush=True)
