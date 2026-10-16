"""The inventory scorer lists every weight file and repeats its own recovery."""
import hashlib
import inspect
import json
import re
from pathlib import Path

from fiber.model_inventory import decision_section, inventory_models
from fiber.weak_labels import EXPERT_SEEDS

ROOT = Path(__file__).resolve().parents[1]
OLD_DECISIONS_SHA256 = "cb9114eda455ad10c587ff10c1de08e2727a1dce7c4b56195e3959e0afafe6e5"
WEAK_LABELS_SHA256 = "48acffcc46679f465cbf6ba167043b02246666afc13c7491a1b3be7239846c57"
RESIDUAL_REPORT_SHA256 = "2c05c26d555fc52f4d372af5c818ad11697c88aa8b27f906836b5cf29f2a93fc"
CONTEXT_WEIGHTS_SHA256 = "0eb6c52b132b4838f85190cc03567c0a2fcbf48c8ef4710a38ea68ee07f0eea5"
RESIDUAL_WEIGHTS_SHA256 = "601a2b5633a986738f9df4d60a33b08a1424286571a0e558a88aa68346ae59b5"
RUNNABLE = {
    "models/weights/research_residual_cnn.pt",
    "models/weights/research_context_cnn.pt",
    "models/weights/experimental/physical_shape_v1.json",
    "models/weights/experimental/gadf_efficientnet.pt",
    "models/weights/experimental/zone_xgb.json",
}
FAMILIES = {
    "research_residual_cnn",
    "research_context_cnn",
    "column_graph",
    "column_transformer",
    "context_logistic",
    "context_lgb",
    "wide_context",
    "patch_scanner",
    "patch_profile",
    "patch_persistent",
    "classify_zones",
    "ml_test_cnn",
    "real_masked_ridge",
}
OUTSIDE_WEIGHTS = (
    "research/ml-test-review/weights/1.pt",
    "research/ml-test-review/weights/2.pt",
    "research/ml-test-review/weights/3.pt",
    "research/ml-test-review/weights/4.pt",
    "research/ml-test-review/weights/5.pt",
    "research/ml-test-review/weights/6.pt",
    "research/real-data-v2/real_masked_ridge.json",
)
HEADING = "## 2026-10-16 — инвентарь research-моделей"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_inventory_lists_every_model_and_repeats_recovery():
    first = inventory_models(ROOT)
    second = inventory_models(ROOT)
    assert first == second
    assert first["schema"] == "model_inventory_v1"
    assert first["serve"] == "research"
    assert first["operational"] is False
    assert first["trained_classifier_operational"] is False
    assert first["sets_anomaly_status"] is False
    assert first["detector"] == "spatial_z_halo"
    assert first["clean"] == "median_x3"
    assert first["threshold_z"] == 3.5
    assert len(first["previous_ids"]) == 15
    assert len(first["ask_ids"]) == 110
    assert len(set(first["previous_ids"])) == 15
    assert len(set(first["ask_ids"])) == 110
    assert set(first["previous_ids"]).isdisjoint(first["ask_ids"])

    next_0003 = next(seed for seed in EXPERT_SEEDS if seed["id"] == "next-0003")
    assert next_0003["class_name"] == "Local_Heating"
    assert "next-0003" in first["previous_ids"]

    unclear = []
    answer_path = ROOT / "docs" / "evidence" / "zakhar_ask_shortlist_answers.jsonl"
    for line in answer_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("presence") == "unclear":
            unclear.append(str(row["id"]))
    assert len(unclear) == 40
    assert set(unclear) == set(first["unclear_ids"])
    scored = set(first["previous_ids"]) | set(first["ask_ids"])
    assert scored.isdisjoint(unclear)
    assert all(not seed_id.startswith("gap-") for seed_id in scored)

    weight_files = [
        path for path in (ROOT / "models" / "weights").rglob("*")
        if path.is_file() and path.suffix.lower() in {".pt", ".json", ".txt"}
    ]
    outside = [ROOT / relative for relative in OUTSIDE_WEIGHTS]
    assert all(path.is_file() for path in outside)
    rows = first["rows"]
    listed = [row["weight_path"] for row in rows if row["weight_path"] is not None]
    assert listed == sorted(path.relative_to(ROOT).as_posix() for path in [*weight_files, *outside])
    assert len(listed) == len(set(listed)) == len(weight_files) + len(outside)
    code_only = [row for row in rows if row["weight_path"] is None]
    assert [row["family"] for row in code_only] == ["classify_zones"]
    assert code_only[0]["weights_sha256"] is None
    assert code_only[0]["model"] == "classify_zones"
    assert FAMILIES <= {row["family"] for row in rows}

    by_sha = {}
    for row in rows:
        if row["weight_path"] is None:
            continue
        digest = _sha256(ROOT / row["weight_path"])
        assert row["weights_sha256"] == digest
        by_sha.setdefault(digest, []).append(row)
    for group in by_sha.values():
        if len(group) < 2:
            continue
        labels = {(row["previous_match_label"], row["ask_match_label"], row["reason"]) for row in group}
        assert len(labels) == 1
    by_path = {row["weight_path"]: row for row in rows}
    assert by_path["models/weights/research_context_cnn.pt"]["weights_sha256"] == CONTEXT_WEIGHTS_SHA256
    assert by_path["models/weights/research_residual_cnn.pt"]["weights_sha256"] == RESIDUAL_WEIGHTS_SHA256
    assert {row["weight_path"] for row in rows if row["runnable"]} == RUNNABLE
    for relative in OUTSIDE_WEIGHTS:
        outside_row = by_path[relative]
        assert outside_row["runnable"] is False
        assert outside_row["weights_sha256"]
        assert "не одно из шести имён класса" in outside_row["reason"]

    for row in rows:
        if row["runnable"]:
            assert row["reason"] is None
            assert row["previous_n_cards"] == 15
            assert row["ask_n_cards"] == 110
            assert row["previous_match_label"] == f"{row['previous_match_count']}/15"
            assert row["ask_match_label"] == f"{row['ask_match_count']}/110"
            assert 0 <= row["previous_match_count"] <= 15
            assert 0 <= row["ask_match_count"] <= 110
        else:
            assert row["reason"]
            assert row["previous_match_count"] is None
            assert row["previous_n_cards"] is None
            assert row["previous_match_label"] is None
            assert row["ask_match_count"] is None
            assert row["ask_n_cards"] is None
            assert row["ask_match_label"] is None

    stored = json.loads((ROOT / "docs" / "evidence" / "model_inventory.json").read_text(encoding="utf-8"))
    assert stored == first

    decisions = (ROOT / "docs" / "DECISIONS.md").read_text(encoding="utf-8")
    marker = "\n" + HEADING
    assert decisions.count(HEADING) == 1
    prefix, section = decisions.split(marker, 1)
    assert hashlib.sha256(prefix.encode("utf-8")).hexdigest() == OLD_DECISIONS_SHA256
    assert "## 2026-10-15 — research context CNN на старых и ask seeds" in prefix
    assert section.strip() == decision_section(first).split("\n", 1)[1].strip()
    assert "serve=research" in section
    assert "operational false" in section
    assert "median_x3" in section
    assert "spatial_z_halo" in section
    assert "|Z| >= 3.5" in section
    assert "trained_classifier_operational" in section
    assert "`next-0003` остаётся экспертом `Local_Heating`" in section
    assert "presence=unclear" in section
    for row in rows:
        if row["weights_sha256"]:
            assert row["weights_sha256"] in section
            assert f"`{row['weight_path']}`" in section
        else:
            assert "файла весов нет" in section
        if row["runnable"]:
            assert f"Прежние 15: `{row['previous_match_label']}`" in section
            assert f"Ask seeds: `{row['ask_match_label']}`" in section
        else:
            assert row["reason"] in section
    for line in section.splitlines():
        if "Дроби нет" in line:
            assert re.search(r"\d+/\d+", line) is None

    from fastapi.testclient import TestClient

    from fiber.viewer import expert_windows
    from fiber.viewer_app import app

    display_source = inspect.getsource(expert_windows)
    assert "analyze_field(" not in display_source
    assert ".pt" not in display_source
    assert "load_research" not in display_source
    example = TestClient(app).get("/view/example").json()
    assert example["manifest"]["clean"] == "median_x3"
    assert example["manifest"]["detector"] == "spatial_z_halo"
    assert example["manifest"]["config"]["threshold_z"] == 3.5
    assert example["manifest"]["trained_classifier_operational"] is False
    assert _sha256(ROOT / "docs" / "evidence" / "weak_labels.jsonl") == WEAK_LABELS_SHA256
    assert _sha256(ROOT / "docs" / "evidence" / "research_cnn_report.json") == RESIDUAL_REPORT_SHA256
