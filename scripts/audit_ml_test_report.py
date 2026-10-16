"""Audit every table row of the frozen PiorlZ/ml-test REPORT, without trusting labels."""
from pathlib import Path
from collections import Counter
import json
import re
import hashlib
import math
import subprocess
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent/"ml-test-review"
OUT = ROOT/"artifacts/ml-test-audit"
OUT.mkdir(parents=True, exist_ok=True)
report_path = SOURCE/"REPORT.md"
current = None
records = []
cluster_values = {}
for line in report_path.read_text(encoding="utf-8").splitlines():
    if line.startswith("## File"):
        current = {"file": re.search(r"`([^`]+)`", line).group(1), "catalog_rows": 0,
                   "classes": Counter(), "zero_duration": 0, "zero_width": 0,
                   "period_positive": 0, "long_periodic": 0, "period_positive_single_row": 0,
                   "nonfinite_numeric_rows": 0, "negative_duration": 0,
                   "peak_abs_max_c": 0., "duration_max_min": 0., "periodicity_max": 0.,
                   "examples": []}
        records.append(current)
        cluster_values[current["file"]] = {}
    elif current is not None and line.startswith("- shape:"):
        current["shape"] = list(map(int, re.findall(r"\d+", line)))
    elif current is not None and line.startswith("- consensus_on:"):
        current["consensus_on"] = int(line.split(":", 1)[1])
    elif current is not None and line.startswith("- consensus_sha256:"):
        current["consensus_sha256"] = line.split(":", 1)[1].strip()
    elif current is not None and line.startswith("| ") and not line.startswith(("| incident_type", "| ---")):
        c = [cell.strip() for cell in line.strip("| ").split("|")]
        if len(c) != 12:
            raise ValueError("malformed report table")
        duration, width, mean, peak, elongation, periodicity = map(float, c[6:])
        cluster_values[current["file"]].setdefault(c[1], []).append((duration, width, mean, peak, periodicity))
        current["nonfinite_numeric_rows"] += not all(math.isfinite(v) for v in (duration, width, mean, peak, elongation, periodicity))
        current["catalog_rows"] += 1
        current["classes"][c[0]] += 1
        current["zero_duration"] += duration == 0
        current["zero_width"] += width == 0
        current["period_positive"] += periodicity > 0
        current["long_periodic"] += duration >= 20 and "периодическая" in c[0]
        current["period_positive_single_row"] += duration == 0 and periodicity > 0
        current["negative_duration"] += duration < 0
        current["peak_abs_max_c"] = max(current["peak_abs_max_c"], abs(peak))
        current["duration_max_min"] = max(current["duration_max_min"], duration)
        current["periodicity_max"] = max(current["periodicity_max"], periodicity)
        if len(current["examples"]) < 3:
            current["examples"].append(dict(zip(("incident_type", "cluster_label", "meter_start", "meter_end",
                                                "time_start", "time_end", "duration_min", "width_m",
                                                "mean_delta_t", "peak_delta_t", "elongation", "periodicity"), c, strict=True)))
for record in records:
    record["classes"] = dict(record["classes"])
    record["consensus_cell_fraction"] = record["consensus_on"]/(record["shape"][0]*record["shape"][1])
    record["cluster_summaries"] = {
        label: {"rows": len(values), **{key: median(value[index] for value in values)
            for index, key in enumerate(("median_duration_min", "median_width_m",
                                        "median_mean_delta_c", "median_peak_delta_c", "median_periodicity"))}}
        for label, values in sorted(cluster_values[record["file"]].items())}
git_prefix = ["git", "-c", f"safe.directory={SOURCE.as_posix()}", "-C", str(SOURCE)]
report_blob = subprocess.check_output(git_prefix + ["show", "HEAD:REPORT.md"])
tracked_files = subprocess.check_output(git_prefix + ["ls-files", "-z"]).decode().split("\0")
summary = {"source_commit": "33e12ac19b7b6318dc48b5f3e8c1239aaaf9244e",
           "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
           "report_bytes": report_path.stat().st_size,
           "report_git_blob_sha256": hashlib.sha256(report_blob).hexdigest(),
           "report_git_blob_bytes": len(report_blob),
           "checkout_newlines": "Windows CRLF; Git blob uses LF",
           "records": records, "total_catalog_rows": sum(r["catalog_rows"] for r in records),
           "expert_truth_in_report": False,
           "source_file_sha256": {name: hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()
                                  for name in sorted(tracked_files) if name}}
(OUT/"report_audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"total_catalog_rows": summary["total_catalog_rows"], "records": records}, ensure_ascii=False), flush=True)
