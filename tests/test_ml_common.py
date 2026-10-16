"""Контракт events.json v1, fiber/larpathon на синтетике и неизменный ответ /detect/field."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml_common.events import SCHEMA_PATH, validate_events
from ml_common.eval import score_document
from ml_common.synthetic import planted_field


def _minimal(**overrides):
    document = {
        "schema": "dts-events/1",
        "model_key": "fiber@abc123abc123",
        "method": "fiber",
        "dts_id": "synthetic",
        "channel_id": "1",
        "grid": {"dx_m": 0.25, "dt_s": 120.0},
        "threshold": 3.5,
        "code_sha": "abc1234def5678",
        "zones": [{
            "t_first": "2026-01-01T00:00:00+00:00",
            "t_last": "2026-01-01T00:02:00+00:00",
            "x_from_m": 1.0,
            "x_to_m": 2.0,
            "score": 4.0,
            "stage2_type": "Local_Heating",
            "type_proba": {},
            "amplitude_c": 1.5,
        }],
        "runtime": {"cpus": 4, "seconds": 0.1, "score_kind": "peak_abs_z", "threshold_kind": "spatial_z"},
    }
    document.update(overrides)
    return document


def test_schema_file_matches_plan_keys():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema["properties"]["schema"]["const"] == "dts-events/1"
    assert set(schema["required"]) == {
        "schema", "model_key", "method", "dts_id", "channel_id",
        "grid", "threshold", "code_sha", "zones", "runtime",
    }
    zone = schema["$defs"]["zone"]["required"]
    assert zone == [
        "t_first", "t_last", "x_from_m", "x_to_m", "score",
        "stage2_type", "type_proba", "amplitude_c",
    ]


def test_schema_accepts_minimal_and_rejects_nan():
    pytest.importorskip("jsonschema")
    parsed = validate_events(_minimal())
    assert parsed["method"] == "fiber"
    broken = _minimal()
    broken["zones"][0]["score"] = float("nan")
    with pytest.raises(ValueError):
        validate_events(broken)
    missing = _minimal()
    del missing["threshold"]
    with pytest.raises(ValueError):
        validate_events(missing)


def test_fiber_synthetic_hits_both_boxes():
    from ml_common.adapters.fiber import FiberAnalyzer

    field, truth = planted_field()
    document = FiberAnalyzer().predict(field)
    assert document["schema"] == "dts-events/1"
    assert document["method"] == "fiber"
    metrics = score_document(document, truth)
    assert metrics["hits"][0]["n_zones"] >= 1
    assert metrics["hits"][1]["n_zones"] >= 1
    assert any(zone["stage2_type"] == "Local_Heating" for zone in document["zones"])


def test_larpathon_synthetic_is_schema_valid():
    pytest.importorskip("torch")
    from ml_common.adapters.larpathon import LarpathonAnalyzer

    field, truth = planted_field()
    # 95 совпадает с синтетическим тестом larpathon; боевой 99.2 на коротком поле пустой.
    document = LarpathonAnalyzer(use_ae=False, percentile=95).predict(field)
    assert document["method"] == "larpathon"
    assert document["runtime"]["cpus"] <= 4
    assert document["threshold"] == pytest.approx(95.0)
    metrics = score_document(document, truth)
    assert metrics["hits"][0]["n_zones"] >= 1


def test_cli_predict_synthetic(tmp_path: Path):
    from ml_common.cli import main

    dest = tmp_path / "events.json"
    assert main(["predict", "--model", "fiber", "--synthetic", "--out", str(dest)]) == 0
    document = json.loads(dest.read_text(encoding="utf-8"))
    assert document["schema"] == "dts-events/1"
    assert document["dts_id"] == "synthetic"
    assert document["zones"]


def test_api_default_stays_fiber_shape():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from fiber.api import app

    client = TestClient(app)
    hot = [[22.0] * 100 for _ in range(12)]
    for row in hot:
        for col in range(45, 51):
            row[col] = 28.0
    body = {"raw": hot, "distances": list(range(100)), "time_s": [i * 60.0 for i in range(12)]}
    plain = client.post("/detect/field", json=body)
    named = client.post("/detect/field?model=fiber", json=body)
    assert plain.status_code == 200 and named.status_code == 200
    assert plain.json()["anomaly_status"] == "Anomaly"
    assert plain.json() == named.json()
    assert "manifest" in plain.json()
    unknown = client.post("/detect/field?model=no-such", json=body)
    assert unknown.status_code == 422
