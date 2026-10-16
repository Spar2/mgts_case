"""Offline handoff must detect changed sources, masks, profiles and processing."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from workbench.evidence_check import verify_evidence


def case():
    native = SimpleNamespace(
        manifest={
            "source_name": "5.csv",
            "source_sha256": "a" * 64,
            "time_kind": "unknown",
        },
        raw_c=np.array([[10.0, -300.0, 12.0], [13.0, 14.0, 15.0]]),
        invalid_reason=np.array([[0, 1, 0], [0, 0, 0]]),
        x_m=np.array([0.0, 1.5, 4.0]),
        elapsed_s=np.array([0.0, 900.0]),
        profiles=[{"timestamp": "10:00"}, {"timestamp": "10:15"}],
    )
    region = {
        "record_id": "5",
        "row_start": 0,
        "row_end": 2,
        "column_start": 0,
        "column_end": 3,
    }
    report = {
        "schema_version": "dts-region-evidence-v1",
        "region": region,
        "source_sha256": "a" * 64,
        "time_kind": "unknown",
        "native_shape": [2, 3],
        "mask_counts": {"0": 5, "1": 1},
        "valid_fraction": 5 / 6,
        "x_bounds_m": [0.0, 4.0],
        "time_bounds_s": [0.0, 900.0],
        "timestamps": ["10:00", "10:15"],
        "gaps_over_300s": [900.0],
        "temperature_c": {"min": 10.0, "median": 13.0, "max": 15.0},
        "profile": {
            "row": 0,
            "x_m": [0.0, 1.5, 4.0],
            "temperature_c": [10.0, None, 12.0],
        },
        "preview": [[10.0, 0.0, 12.0], [13.0, 14.0, 15.0]],
        "preview_valid": [[True, False, True], [True, True, True]],
        "preview_x_m": [0.0, 1.5, 4.0],
        "preview_time_s": [0.0, 900.0],
        "runs": [],
        "profile_runs": [],
    }
    run = {
        "id": "selected",
        "status": "complete",
        "source_sha256": "a" * 64,
        "region": {**region, "row_end": 1},
        "method": "actual-method",
        "result": {"denoised": [11.0, 99.0, None]},
    }
    return native, report, run


def processed_case():
    native, report, run = case()
    report["profile"].update(
        processing_run_id="selected", processed_temperature_c=[11.0, None, None]
    )
    metadata = {k: v for k, v in run.items() if k != "result"}
    report["runs"] = [metadata]
    report["profile_runs"] = [deepcopy(metadata)]
    return native, report, run


def test_raw_and_selected_evidence():
    native, report, _ = case()
    assert verify_evidence(report, native)["processing_run_id"] is None
    native, report, run = processed_case()
    result = verify_evidence(report, native, run)
    assert result["processing_run_id"] == "selected"
    assert result["quality_assessed"] is False
    assert result["expert_answer_verified"] is False


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r.update(source_sha256="b" * 64),
        lambda r: r["profile"]["x_m"].__setitem__(1, 2.0),
        lambda r: r["profile"]["temperature_c"].__setitem__(1, -300.0),
        lambda r: r["profile"]["temperature_c"].__setitem__(0, True),
        lambda r: r["preview_time_s"].__setitem__(1, 1.0),
        lambda r: r["preview_valid"][0].__setitem__(1, True),
        lambda r: r["profile"].update(row=-1),
        lambda r: r["profile"]["processed_temperature_c"].__setitem__(2, 12.0),
        lambda r: r["runs"][0].update(method="substituted"),
    ],
)
def test_tampered_evidence_rejected(change):
    native, report, run = processed_case()
    change(report)
    with pytest.raises(ValueError):
        verify_evidence(report, native, run)


def test_missing_other_or_newer_run_rejected():
    native, report, run = processed_case()
    for candidate in (None, {**run, "id": "newer"}, {**run, "source_sha256": "b" * 64}):
        with pytest.raises(ValueError):
            verify_evidence(report, native, candidate)
