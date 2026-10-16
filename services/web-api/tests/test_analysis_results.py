"""Real native axes, package integrity, masks and durable case integration."""

import copy
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from test_platform import login
from test_platform import pg as platform_pg
from test_workbench import fixture as native_fixture
from workbench.accounts import add_user
from workbench.analysis_results import import_package, sha, validate
from workbench.api import create_app
from workbench.case_bundle import read_bundle
from workbench.cases import make_bundle
from workbench.evidence_check import verify_evidence
from workbench.store import Store

pg = platform_pg


@pytest.mark.parametrize(
    "failure",
    [
        "shape",
        "mask",
        "nan",
        "infinity",
        "dtype",
        "score",
        "missing",
        "path",
        "checksum",
    ],
)
def test_reject_incompatible_arrays(tmp_path, failure):
    root, _, record = native_fixture.__wrapped__(tmp_path)
    store = Store(root / "records", root / "cache")
    directory, metadata = package(tmp_path, record)
    name = "processed_temperature_c.npy"
    if failure == "shape":
        np.save(directory / name, np.ones((2, 3)))
    elif failure == "mask":
        name = "processed_temperature_c_valid.npy"
        np.save(directory / name, np.ones((3, 3), dtype=np.uint8))
    elif failure in {"nan", "infinity"}:
        a = np.ones((3, 3))
        a[0, 0] = np.nan if failure == "nan" else np.inf
        np.save(directory / name, a)
    elif failure == "dtype":
        np.save(directory / name, np.ones((3, 3), dtype=np.int64))
    elif failure == "score":
        metadata["score"] = None
    elif failure == "missing":
        del metadata["files"]["score_valid.npy"]
    elif failure == "path":
        metadata["files"]["../secret.npy"] = "0" * 64
    elif failure == "checksum":
        metadata["files"][name] = "0" * 64
    if failure not in {"score", "missing", "path", "checksum"}:
        metadata["files"][name] = sha(directory / name)
    (directory / "manifest.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError):
        validate(directory, store)


def package(tmp_path, record):
    root = tmp_path / "package"
    root.mkdir()
    np.save(root / "x_m.npy", record.x_m)
    np.save(root / "time_s.npy", record.elapsed_s)
    np.save(
        root / "processed_temperature_c.npy",
        np.asarray(record.raw_c, dtype=np.float64) + 1,
    )
    np.save(root / "processed_temperature_c_valid.npy", record.invalid_reason == 0)
    np.save(
        root / "score.npy",
        np.array([[1.0, 900.0, -4.0], [2.0, 3.0, -2.0], [0.0, 1.0, -3.0]]),
    )
    np.save(root / "score_valid.npy", record.invalid_reason == 0)
    (root / "zones.json").write_text(
        json.dumps(
            [
                {
                    "id": "zone-1",
                    "row_start": 0,
                    "row_end": 2,
                    "column_start": 0,
                    "column_end": 3,
                }
            ]
        ),
        encoding="utf-8",
    )
    metadata = {
        "schema_version": "dts-analysis-result-v2",
        "region": {
            "record_id": "5",
            "row_start": 0,
            "row_end": 3,
            "column_start": 0,
            "column_end": 3,
        },
        "source_sha256": record.manifest["source_sha256"],
        "method": "contract-fixture",
        "method_version": "unit-test",
        "parameters": {},
        "calculated_at": "2026-10-13T12:00:00+03:00",
        "training_origin": "untrained",
        "research_status": "reference",
        "limitations": "Technical fixture, not a trained model",
        "score": {
            "name": "Spatial Z",
            "unit": "Z",
            "meaning": "Signed deviation",
            "kind": "signed",
            "threshold": 3.5,
            "comparison": "abs_ge",
        },
    }
    metadata["files"] = {p.name: sha(p) for p in root.iterdir()}
    (root / "manifest.json").write_text(json.dumps(metadata), encoding="utf-8")
    return root, metadata


def test_package_validation_axes_masks_and_sha(tmp_path):
    root, _, record = native_fixture.__wrapped__(tmp_path)
    store = Store(root / "records", root / "cache")
    directory, metadata = package(tmp_path, record)
    assert validate(directory, store)[1] == ["processed_temperature_c", "score"]
    original = copy.deepcopy(metadata)
    metadata["source_sha256"] = "0" * 64
    (directory / "manifest.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA"):
        validate(directory, store)
    metadata = original
    np.save(directory / "x_m.npy", record.x_m[::-1])
    metadata["files"]["x_m.npy"] = sha(directory / "x_m.npy")
    (directory / "manifest.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="Axes"):
        validate(directory, store)


def test_import_slices_tiles_case_and_isolation(pg, tmp_path):
    root, queue, record = native_fixture.__wrapped__(tmp_path)
    add_user(pg, "analysis-author", "QA-only-password-123456", "Author", "analyst")
    app = create_app(root / "records", queue, root / "runtime")
    directory, metadata = package(tmp_path, record)
    with pg.connect() as db:
        user_id = db.execute(
            "SELECT id FROM users WHERE username='analysis-author'"
        ).fetchone()[0]
    with TestClient(app) as client:
        run = import_package(
            app.state.platform, app.state.store, directory, "local", user_id
        )
        assert (
            import_package(
                app.state.platform, app.state.store, directory, "local", user_id
            )["id"]
            == run["id"]
        )
        assert client.get("/api/v2/result-runs").status_code == 401
        headers = login(client, "analysis-author")
        assert len(client.get("/api/v2/result-runs").json()) == 1
        geometry = client.get(f"/api/v2/result-runs/{run['id']}/zones").json()
        assert len(geometry) == 1 and geometry[0]["id"] == "zone-1"
        assert geometry[0]["source_sha256"] == metadata["source_sha256"]
        assert geometry[0]["t1"] < 100
        token_body = dict(run)
        token_body["zones"] = [
            {
                "id": "cross-gap",
                "row_start": 0,
                "row_end": 3,
                "column_start": 0,
                "column_end": 3,
            }
        ]
        app.state.platform.put("analysis_run", run["id"], token_body)
        segments = client.get(f"/api/v2/result-runs/{run['id']}/zones").json()
        assert len(segments) == 2 and segments[0]["t1"] < segments[1]["t0"]
        app.state.platform.put("analysis_run", run["id"], run)
        slices = client.get(
            f"/api/v2/result-runs/{run['id']}/slices?row=0&column=0&zone_id=zone-1"
        ).json()
        assert slices["layers"]["processed_temperature_c"]["profile"] == [21, None, 23]
        assert slices["layers"]["score"]["profile"] == [1, None, -4]
        assert slices["layers"]["raw"]["trend"] == [20, 21, None, 25]
        assert slices["layers"]["raw"]["zone"]["median"] == [21, 23, None, None]
        cropped = client.get(
            f"/api/v2/result-runs/{run['id']}/slices?row=1&column=1&row_start=1&row_end=2&column_start=1&column_end=3"
        ).json()
        assert cropped["x_m"] == [1, 4]
        assert cropped["layers"]["raw"]["profile"] == [23, 24]
        assert cropped["layers"]["raw"]["trend"] == [23]
        assert (
            client.get(
                f"/api/v2/result-runs/{run['id']}/slices?row=4&column=0"
            ).status_code
            == 422
        )
        assert (
            client.get(
                f"/api/v2/result-runs/{run['id']}/tiles?layer=score&width=2&height=2"
            ).status_code
            == 200
        )
        assert (
            client.get(
                f"/api/v2/result-runs/{run['id']}/tiles?layer=missing"
            ).status_code
            == 409
        )
        body = {
            "title": "Native analysis",
            "region": metadata["region"],
            "profile": 0,
            "source_sha256": metadata["source_sha256"],
            "run_id": run["id"],
            "analysis_column": 1,
        }
        response = client.post(
            "/api/v2/cases",
            json=body,
            headers={**headers, "Idempotency-Key": "analysis-case"},
        )
        assert response.status_code == 201, response.text
        case = response.json()
        assert case["result"]["evidence"]["analysis"]["raw_trend"] == [
            None,
            23,
            None,
            26,
        ]
        assert case["result"]["evidence"]["analysis"]["layers"]["score"]["profile"] == [
            1,
            None,
            -4,
        ]
        before = make_bundle(case)
        assert make_bundle(case) == before
        assert (
            verify_evidence(case["result"]["evidence"], record, case["result"]["run"])[
                "quality_assessed"
            ]
            is False
        )
        assert client.get("/api/v2/cases/" + case["id"]).json()["research_available"]
        custom = {
            **body,
            "region": {
                **metadata["region"],
                "row_start": 1,
                "row_end": 2,
                "column_start": 1,
            },
            "profile": 1,
            "analysis_layer": "score",
        }
        saved_custom = client.post(
            "/api/v2/cases",
            json=custom,
            headers={**headers, "Idempotency-Key": "custom-window"},
        )
        assert saved_custom.status_code == 201, saved_custom.text
        custom_case = saved_custom.json()
        view = custom_case["result"]["evidence"]["analysis_view"]
        assert view["region"] == custom["region"] and view["layer"] == "score"
        assert view["row"] == 1 and view["column"] == 1 and view["zone_id"] is None
        bundle = make_bundle(custom_case)
        assert (
            verify_evidence(
                custom_case["result"]["evidence"], record, custom_case["result"]["run"]
            )["quality_assessed"]
            is False
        )
        for key, value in (
            ("layer", "unavailable"),
            ("column", 3),
            ("region", metadata["region"]),
            ("source_sha256", "0" * 64),
        ):
            broken = copy.deepcopy(custom_case["result"]["evidence"])
            broken["analysis_view"][key] = value
            with pytest.raises(ValueError, match="Analysis"):
                verify_evidence(broken, record, custom_case["result"]["run"])
        assert (
            client.post(
                "/api/v2/cases",
                json={**custom, "analysis_layer": "missing"},
                headers=headers,
            ).status_code
            == 409
        )
        assert (
            client.post(
                "/api/v2/cases",
                json={**custom, "analysis_zone_id": "zone-1"},
                headers=headers,
            ).status_code
            == 409
        )
        assert (
            client.post(
                "/api/v2/cases",
                json=custom,
                headers={**headers, "Idempotency-Key": "custom-window"},
            ).json()["id"]
            == custom_case["id"]
        )
        assert (
            make_bundle(client.get("/api/v2/cases/" + custom_case["id"]).json())
            == bundle
        )
        assert make_bundle(client.get("/api/v2/cases/" + case["id"]).json()) == before
        client.headers["X-DTS-Project"] = "foreign"
        assert client.get(f"/api/v2/result-runs/{run['id']}").status_code in (401, 403)


def test_score_only_case_keeps_no_processing_and_bundle_binding(pg, tmp_path):
    root, queue, record = native_fixture.__wrapped__(tmp_path)
    add_user(pg, "score-author", "QA-only-password-123456", "Author", "analyst")
    directory, metadata = package(tmp_path, record)
    for name in ("processed_temperature_c.npy", "processed_temperature_c_valid.npy"):
        metadata["files"].pop(name)
    (directory / "manifest.json").write_text(json.dumps(metadata), encoding="utf-8")
    app = create_app(root / "records", queue, root / "runtime")
    with TestClient(app) as client:
        with pg.connect() as db:
            author = db.execute(
                "SELECT id FROM users WHERE username='score-author'"
            ).fetchone()[0]
        run = import_package(
            app.state.platform, app.state.store, directory, "local", author
        )
        headers = login(client, "score-author")
        body = {
            "title": "Score only",
            "region": metadata["region"],
            "profile": 0,
            "source_sha256": metadata["source_sha256"],
            "run_id": run["id"],
        }
        response = client.post("/api/v2/cases", json=body, headers=headers)
        assert response.status_code == 201, response.text
        case = response.json()
        assert "processed_temperature_c" not in case["result"]["evidence"]["profile"]
        assert "processing_changes" not in case["result"]["evidence"]
        path = tmp_path / "case.zip"
        path.write_bytes(make_bundle(case))
        report, saved, _ = read_bundle(path)
        assert verify_evidence(report, record, saved)["quality_assessed"] is False


def test_pipeline_slice_matches_saved_run(pg, tmp_path):
    """Difference is processed minus source, and a masked cell stays empty."""
    root, queue, record = native_fixture.__wrapped__(tmp_path)
    add_user(pg, "slice-author", "QA-only-password-123456", "Author", "analyst")
    directory, metadata = package(tmp_path, record)
    app = create_app(root / "records", queue, root / "runtime")
    with TestClient(app) as client:
        with pg.connect() as db:
            author = db.execute(
                "SELECT id FROM users WHERE username='slice-author'"
            ).fetchone()[0]
        run = import_package(
            app.state.platform, app.state.store, directory, "local", author
        )
        headers = login(client, "slice-author")
        slices = client.get(
            f"/api/v2/result-runs/{run['id']}/slices?row=0&column=0", headers=headers
        ).json()
        assert slices["run_id"] == run["id"]
        assert slices["source_sha256"] == metadata["source_sha256"]
        raw = slices["layers"]["raw"]["profile"]
        processed = slices["layers"]["processed_temperature_c"]["profile"]
        difference = slices["layers"]["difference"]["profile"]
        assert any(value is None for value in difference)
        for before, after, delta in zip(raw, processed, difference, strict=True):
            if before is None or after is None:
                assert delta is None
            else:
                assert delta == pytest.approx(after - before)
        print("ASSERT slice run_id and difference")
