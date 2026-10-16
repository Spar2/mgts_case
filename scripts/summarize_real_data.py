"""Aggregate real-only evidence and detector coverage questions without truth claims."""
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
pack = ROOT/"artifacts/real-data-v2"
rows = [json.loads(line) for line in (pack/"weak_labels.jsonl").read_text(encoding="utf-8").splitlines()]
proposals = [r for r in rows if r["origin"] == "detector_proposal"]
questions = []
for row in rows:
    if row["origin"] == "detector_proposal" or row["weak_evidence"]["weak_presence_label"] != 1:
        continue
    overlap = any(z["record"] == row["record"] and
        max(z["rows_half_open"][0], row["rows_half_open"][0]) < min(z["rows_half_open"][1], row["rows_half_open"][1]) and
        max(z["x_m_inclusive"][0], row["x_m_inclusive"][0]) <= min(z["x_m_inclusive"][1], row["x_m_inclusive"][1]) for z in proposals)
    if not overlap:
        questions.append({"id": row["id"], "record": row["record"], "split_code": row["split_code"],
                          "question": "Raw local-contrast rule positive, no proposal bounding-box overlap; persistent spatial structure is one possible explanation.",
                          "confirmed_detector_miss": False})
report = json.loads((ROOT/"artifacts/real-reconstruction-v2/report.json").read_text())
audits = json.loads((pack/"noise_audit.json").read_text(encoding="utf-8"))
manifest = json.loads((pack/"manifest.json").read_text())
summary = {"schema": "real_data_summary_v1", "manifest": manifest,
           "review_reason_counts": dict(Counter(reason for row in rows for reason in row["weak_evidence"]["review_reasons"])),
           "n_positive_independent_windows_without_proposal_overlap": len(questions),
           "coverage_questions": questions,
           "by_origin": {origin: dict(Counter(str(r["weak_evidence"]["weak_presence_label"]) for r in rows if r["origin"] == origin)) for origin in manifest["origin_counts"]},
           "noise": [{"record": a["record"], "increment_variability_proxy_c": a["increment_variability_proxy_c"],
                      "residual_adjacent_space": a["residual_adjacent_space"], "residual_adjacent_time": a["residual_adjacent_time"]} for a in audits["records"]],
           "reconstruction": {"n_train": report["n_train"], "n_development": report["n_development"],
               "selected_alpha": report["selected_alpha"], "baselines": report["baselines_development"],
               "ridge": next(c["development"] for c in report["candidates"] if c["alpha"] == report["selected_alpha"]),
               "macro_record_mse": {key: float(np.mean([r[key]["mse_noisy_target_c2"] for r in report["per_record_development"].values()])) for key in ("ridge", "neighbour_median")},
               "operational_approved": False},
           "expert_labels_created": 0, "unseen_test_accuracy_estimated": False}
out = ROOT/"docs/evidence"
out.mkdir(parents=True, exist_ok=True)
(out/"real_data_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
(out/"real_reconstruction_report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
(out/"real_noise_audit.json").write_text(json.dumps(audits, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
(out/"real_data_verification.json").write_text((pack/"verification.json").read_text(), encoding="utf-8")
(pack/"coverage_questions.json").write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"coverage_questions": len(questions), "weak_labels": manifest["weak_presence_counts"], "reconstruction": summary["reconstruction"]}), flush=True)
