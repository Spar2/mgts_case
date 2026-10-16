"""Verify saved real-data reproductions and export small, portable audit evidence.

This does not fit a model or measure expert accuracy. Incomplete runs are
explicitly listed; never infer success from the existence of a partial array.
"""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys
import subprocess
import platform
import importlib.metadata
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fiber.io import Record, file_sha256
sys.path.insert(0, str(ROOT.parent / "ml-test-review"))
from src.detector_cnn import robust_scale, adaptive_threshold

AUDIT = ROOT / "artifacts/ml-test-audit"
OUT = ROOT / "docs/evidence"
OUT.mkdir(parents=True, exist_ok=True)
report = json.loads((AUDIT / "report_audit.json").read_text(encoding="utf-8"))
source = ROOT.parent / "ml-test-review"
head = subprocess.check_output(["git", "-c", f"safe.directory={source.as_posix()}",
    "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
assert head == report["source_commit"], "source checkout changed during audit"
for filename, digest in report["source_file_sha256"].items():
    assert file_sha256(source / filename) == digest, f"source file changed: {filename}"
witnesses = json.loads((ROOT / "fiber/benchmark/witnesses.json").read_text())["witnesses"]
summary = {"source_commit": report["source_commit"], "completed_records": [],
           "pending_records": [], "records": [], "development_witnesses": [],
           "expert_truth_available": False, "synthetic_training_used": False,
           "scope": "Retrospective real-data reproduction; agreement is not accuracy.",
           "audit_runtime": {"python": platform.python_version(), "platform": platform.platform()}}
summary["audit_runtime"]["additional_dependencies"] = {
    package: importlib.metadata.version(package) for package in ("matplotlib", "streamlit", "pytest")}

for name in map(str, range(1, 7)):
    directory = AUDIT / "reproduced" / name
    if not (directory / "receipt.json").exists():
        summary["pending_records"].append(name)
        continue
    receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
    record = Record.load(ROOT / "artifacts/records" / name)
    assert receipt["source_commit"] == report["source_commit"]
    assert file_sha256(ROOT.parent / f"{name}.csv") == receipt["source_sha256"] == record.source_sha256
    masks = {key: np.load(directory / f"{key}_mask.npy")
             for key in ("cv", "cnn", "ml", "consensus")}
    for key, mask in masks.items():
        assert mask.shape == record.valid.shape and mask.dtype == np.bool_
        assert int(mask.sum()) == receipt["mask_counts"][key]
        assert int((mask & ~record.valid).sum()) == receipt["invalid_cells_on"][key]
    digest = hashlib.sha256(masks["consensus"].astype(np.uint8).tobytes()).hexdigest()
    assert digest == receipt["mask_sha256"]
    assert file_sha256(directory / "cnn_weights.pt") == receipt["cnn_weights_sha256"]
    with (directory / "catalog.csv").open(encoding="utf-8", newline="") as handle:
        catalog = list(csv.DictReader(handle))
    assert len(catalog) == receipt["catalog_rows"]
    assert dict(Counter(row["incident_type"] for row in catalog)) == receipt["classes"]
    original = next(row for row in report["records"] if row["file"] == f"MGTS_data/{name}.csv")
    result = dict(receipt)
    result["reported_counts"] = {"consensus": original["consensus_on"], "catalog_rows": original["catalog_rows"]}
    result["cnn_weights_bytes"] = (directory / "cnn_weights.pt").stat().st_size
    residual = np.load(directory / "residual.npy")
    _, scale = robust_scale(residual)
    result["cnn_inference_contract"] = {
        "input": "external CV residual, not raw temperature or spatial Z",
        "residual_center_c": float(np.median(residual)), "residual_scale_c": scale,
        "patch_shape": [32, 32], "stride": [16, 16],
        "score_threshold_c": adaptive_threshold(np.load(directory / "cnn_score.npy")),
        "threshold_scope": "computed retrospectively on this file; not calibrated on independent truth",
        "source_module_checkout_sha256": report["source_file_sha256"]["src/detector_cnn.py"],
        "valid_mask_used_in_original_training": False,
        "original_author_weights": False,
        "saved_isolation_forest_weights": False}
    result["saved_outputs_verified"] = True
    result["pairwise_valid_mask_jaccard"] = {}
    for first, second in (("cv", "cnn"), ("cv", "ml"), ("cnn", "ml")):
        union = (masks[first] | masks[second]) & record.valid
        result["pairwise_valid_mask_jaccard"][f"{first}/{second}"] = (
            float(((masks[first] & masks[second]) & record.valid).sum() / union.sum())
            if union.any() else None)
    ablation = directory / "cnn_ablation.json"
    result["patch_median_ablation"] = json.loads(ablation.read_text()) if ablation.exists() else None
    verification = directory / "weights_verification.json"
    result["weights_verification"] = json.loads(verification.read_text()) if verification.exists() else None
    summary["records"].append(result)
    summary["completed_records"].append(name)
    z = np.load(ROOT / "artifacts/anomaly" / name / "z_spatial.npy", mmap_mode="r")
    for witness in witnesses:
        if witness["record"] != name:
            continue
        start, stop = witness["rows"]
        columns = np.flatnonzero((record.x_m >= witness["x_m"][0]) & (record.x_m <= witness["x_m"][1]))
        index = np.ix_(np.arange(start, stop), columns)
        valid = record.valid[index]
        own = z[index]
        external = next(item for item in receipt["witnesses"] if item["id"] == witness["id"])
        summary["development_witnesses"].append({"id": witness["id"], "record": name,
            "role": witness["role"], "expert_confirmed": False,
            "native_valid_cells": int(valid.sum()),
            "spatial_z_hot_fraction_valid": float(((own >= 3.5) & valid).sum()/valid.sum()) if valid.any() else None,
            "spatial_z_cold_fraction_valid": float(((own <= -3.5) & valid).sum()/valid.sum()) if valid.any() else None,
            "spatial_z_finite_fraction_all": float(np.isfinite(own).mean()),
            "external": external["branches"],
            "scope": "Partial development core, different score semantics; not precision/recall."})

summary["total_consensus_on_invalid_source"] = sum(row["invalid_cells_on"]["consensus"] for row in summary["records"])
summary["total_reproduced_catalog_rows"] = sum(row["catalog_rows"] for row in summary["records"])
summary["all_six_complete"] = not summary["pending_records"]
suite = ET.parse(AUDIT / "external-tests.xml").getroot().find("testsuite")
summary["upstream_unit_tests"] = {key: suite.attrib[key] for key in ("tests", "failures", "errors", "skipped", "time")}
summary["upstream_unit_tests"]["cases"] = [case.attrib["classname"] + "." + case.attrib["name"]
                                               for case in suite.findall("testcase")]
(OUT / "ml_test_reproduction.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
for source, target in (("report_audit", "report"), ("report_gap_audit", "gaps"),
                       ("native_cv_audit", "native_cv"), ("contract_probes", "contracts"),
                       ("cnn_architecture_probe", "cnn_architecture")):
    value = json.loads((AUDIT / f"{source}.json").read_text(encoding="utf-8"))
    (OUT / f"ml_test_{target}.json").write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
print(json.dumps({key: summary[key] for key in ("completed_records", "pending_records",
    "total_consensus_on_invalid_source", "total_reproduced_catalog_rows")}, ensure_ascii=True))

def number(value):
    return f"{value:,}".replace(",", " ")

tables = [f"Полный ансамбль выполнен на **{len(summary['completed_records'])}/6 CSV**.", "",
    "| CSV | Consensus в REPORT | Consensus повторно | Компонент повторно | На отсутствующих измерениях | Время CPU, с | Jaccard consensus при замене CNN медианой |",
    "|---|---:|---:|---:|---:|---:|---:|"]
for row in summary["records"]:
    ablation = row["patch_median_ablation"]
    agreement = f"{ablation['consensus_mask_jaccard']:.3f}" if ablation else "ещё не выполнено"
    tables.append(f"| {row['record']} | {number(row['reported_counts']['consensus'])} | {number(row['mask_counts']['consensus'])} | {number(row['catalog_rows'])} | {number(row['invalid_cells_on']['consensus'])} | {row['timings']['ensemble_s']:.1f} | {agreement} |")
tables += ["", "Jaccard — число клеток, включённых в обе маски, делённое на число клеток, включённых хотя бы в одну. Он показывает сходство, не правильность событий. Время включает fit каждой модели заново на файле; это CPU с четырьмя Torch threads в нашем окружении. Прогоны CSV 1 и 5 выполнялись параллельно; времена служат ориентиром затрат, а не нормированным performance benchmark.", "",
    "| Development core | Spatial Z ≥3.5, доля valid | Consensus, доля всех клеток | Consensus с положительным residual, доля valid | Consensus с отрицательным residual, доля valid |",
    "|---|---:|---:|---:|---:|"]
def fraction(value):
    return f"{value:.3f}" if value is not None else "—"
for row in summary["development_witnesses"]:
    external = row["external"]["consensus"]
    tables.append(f"| {row['id']} | {fraction(row['spatial_z_hot_fraction_valid'])} | {fraction(external['all_cells_on'])} | {fraction(external['positive_fraction_valid'])} | {fraction(external['negative_fraction_valid'])} |")
tables += ["", "Это частичные области, выбранные агентом ранее, без подтверждения экспертом. Различаются точка отсчёта и знак score; таблица объясняет различия, а не объявляет победителя по precision/recall.", ""]
review = ROOT / "docs/ML_TEST_REVIEW.md"
text = review.read_text(encoding="utf-8")
begin, end = "<!-- BEGIN VERIFIED REPRODUCTION -->", "<!-- END VERIFIED REPRODUCTION -->"
assert begin in text and end in text
prefix, remainder = text.split(begin, 1)
_, suffix = remainder.split(end, 1)
review.write_text(prefix + begin + "\n\n" + "\n".join(tables) + "\n" + end + suffix, encoding="utf-8", newline="\n")
