"""A persisted historical result must never become a current-source profile."""

from fastapi.testclient import TestClient
from test_workbench import fixture as native_fixture
from workbench.api import create_app


def test_report_rejects_stale_source_and_unfinished_processing(tmp_path):
    root, queue, record = native_fixture.__wrapped__(tmp_path)
    app = create_app(root / "records", queue, root / "runtime")
    region = {
        "record_id": "5",
        "row_start": 0,
        "row_end": 1,
        "column_start": 0,
        "column_end": 3,
    }
    with TestClient(app) as client:
        headers = {
            "X-Workbench-Token": client.get("/api/v1/session").json()["mutation_token"]
        }
        current = client.post(
            "/api/v1/results",
            headers=headers,
            json={
                "schema_version": "dts-method-result-v1",
                "region": region,
                "source_sha256": record.manifest["source_sha256"],
                "method": "median_x3",
                "method_version": "contract-only",
                "parameters": {},
                "calculated_at": "2026-10-11",
                "training_origin": "untrained",
                "limitations": "Contract fixture, no model training",
                "result": {"denoised": [10, 11, 12]},
            },
        )
        assert current.status_code == 201
        current = current.json()
        # Simulate historical workspace retained after changing the native source.
        app.state.workspace.put(
            "run",
            "stale",
            {
                **current,
                "id": "stale",
                "source_sha256": "0" * 64,
                "result": {"denoised": [91, 92, 93]},
            },
        )
        app.state.workspace.put(
            "run",
            "unfinished",
            {
                **current,
                "id": "unfinished",
                "status": "error",
                "result": {"denoised": [81, 82, 83]},
            },
        )
        report = client.post(
            f"/api/v1/reports?profile=0&run_id={current['id']}",
            headers=headers,
            json=region,
        )
        assert report.status_code == 200
        content = report.json()
        assert content["profile"]["processing_run_id"] == current["id"]
        assert content["profile"]["processed_temperature_c"] == [10, None, 12]
        for group in ("runs", "profile_runs"):
            assert {r["id"] for r in content[group]} == {current["id"]}
        raw = client.post(
            "/api/v1/reports?profile=0&method=median_x3", headers=headers, json=region
        ).json()
        assert raw["profile"].get("processing_run_id") is None
        assert not raw["profile_runs"]
        for run_id, expected in [("stale", 409), ("unfinished", 409), ("missing", 404)]:
            assert (
                client.post(
                    f"/api/v1/reports?profile=0&run_id={run_id}",
                    headers=headers,
                    json=region,
                ).status_code
                == expected
            )
        # A later result of the same method cannot silently replace the selection.
        app.state.workspace.put(
            "run",
            "newer",
            {**current, "id": "newer", "result": {"denoised": [20, 21, 22]}},
        )
        again = client.post(
            f"/api/v1/reports?profile=0&run_id={current['id']}",
            headers=headers,
            json=region,
        ).json()
        assert again["profile"]["processed_temperature_c"] == [10, None, 12]
        html = client.post(
            f"/api/v1/reports?profile=0&run_id={current['id']}&format=html",
            headers=headers,
            json=region,
        )
        assert current["id"] in html.text
        for key, changes in {
            "wrong-row": {"region": {**region, "row_start": 1, "row_end": 2}},
            "bad-shape": {"result": {"denoised": [1, 2]}},
            "non-numeric": {"result": {"denoised": [True, 2, 3]}},
            "no-coverage": {"region": {**region, "column_start": 1}},
        }.items():
            app.state.workspace.put("run", key, {**current, "id": key, **changes})
            assert (
                client.post(
                    f"/api/v1/reports?profile=0&run_id={key}",
                    headers=headers,
                    json=region,
                ).status_code
                == 409
            )
