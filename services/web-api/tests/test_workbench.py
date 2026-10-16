import csv
import json
import os
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workbench.api import create_app
from workbench.store import Store, cell_edges
from workbench.vendor.data import ingest, sha256
from workbench.vendor.reviews import ReviewJournal


@pytest.fixture
def fixture(tmp_path):
    source = tmp_path / "5.csv"
    with source.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["dts_id", "channel_id", "timestamp", "points"]
        )
        writer.writeheader()
        for t, temperatures in [
            ("00:00:00", [20, -300, 22]),
            ("00:00:10", [21, 23, 24]),
            ("01:00:00", [25, 26, 27]),
        ]:
            writer.writerow(
                {
                    "dts_id": "sensor",
                    "channel_id": "1",
                    "timestamp": "2025-10-16 " + t,
                    "points": json.dumps(list(zip([0, 1, 4], temperatures))),
                }
            )
    record = ingest(source, tmp_path / "records")
    queue = tmp_path / "queue"
    queue.mkdir()
    signed = np.zeros((1, 3, 3), np.int8)
    signed[:, record.invalid_reason != 0] = 2
    np.savez_compressed(
        queue / "card.npz",
        raw_c=record.raw_c,
        invalid_reason=record.invalid_reason,
        time_s=record.elapsed_s,
        x_m=record.x_m,
        signed=signed,
        binary=np.zeros((1, 3, 3), bool),
    )
    card = {
        "review_id": "card",
        "source": "5.csv",
        "artifact": "card.npz",
        "artifact_sha256": sha256(queue / "card.npz"),
        "evidence": {"source_sha256": record.manifest["source_sha256"]},
        "native_shape": [3, 3],
        "row_bounds": [0, 3],
        "column_bounds": [0, 3],
        "x_bounds_m": [0, 4],
        "time_bounds_s": [0, 3600],
        "timestamps": [p["timestamp"] for p in record.profiles],
        "methods": ["native_flank"],
    }
    q = {
        "cards": [card],
        "allowed_answers": {
            "temperature_episode": ["present", "absent", "ambiguous", "unknown"],
            "phenomenon": [
                "local_change",
                "common_trend",
                "dropout",
                "mixed",
                "unknown",
            ],
        },
    }
    (queue / "queue.json").write_text(json.dumps(q), encoding="utf-8")
    (queue / "manifest.json").write_text(
        json.dumps({"queue_sha256": sha256(queue / "queue.json"), "cards": 1}),
        encoding="utf-8",
    )
    return tmp_path, queue, record


def body(queue):
    return {
        "queue_sha256": sha256(queue / "queue.json"),
        "review_id": "card",
        "expected_revision": 0,
        "reviewer": "Test expert",
        "reviewer_type": "expert",
        "temperature_episode": "ambiguous",
        "phenomenon": "unknown",
        "notes": "Evidence; not a training label",
    }


def unpack(data):
    size = struct.unpack("<I", data[:4])[0]
    meta = json.loads(data[4 : 4 + size])
    offset = (4 + size + 3) // 4 * 4
    n = np.prod(meta["shape"])
    values = np.frombuffer(data, dtype="<f4", count=n, offset=offset).reshape(
        meta["shape"]
    )
    mask = np.frombuffer(data, dtype="u1", count=n, offset=offset + n * 4).reshape(
        meta["shape"]
    )
    counts = np.frombuffer(
        data, dtype="<u4", count=n, offset=offset + (n * 5 + 3) // 4 * 4
    ).reshape(meta["shape"])
    return meta, values, mask, counts


def test_native_precision_mask_and_blind_contract(fixture):
    root, q, r = fixture
    with TestClient(create_app(root / "records", q, root / "runtime")) as c:
        catalog = c.get("/api/v1/records").json()
        assert catalog[0]["time_kind"] == "local_unzoned"
        assert c.get("/api/v1/queue/card/methods").status_code == 403
        queue = c.get("/api/v1/queue").json()
        assert (
            "methods" not in queue["cards"][0] and "evidence" not in queue["cards"][0]
        )
        meta, values, mask, counts = unpack(c.get("/api/v1/queue/card/window").content)
        np.testing.assert_allclose(values, r.raw_c.astype("f4"))
        np.testing.assert_equal(mask, r.invalid_reason)
        assert meta["aggregation"] == "native_observations" and counts[0, 1] == 0
        slices = c.get("/api/v1/records/5/slices?row=0&column=1").json()
        assert slices["temperature"] is None and slices["profile"][1] is None
        assert None in c.get("/api/v1/records/5/slices?row=0&column=0").json()["trend"]
        assert c.get("/api/v1/records/5/slices?row=3").status_code == 422
        assert (
            c.get(
                "/api/v1/records/5/tiles?x0=0&x1=4&t0=0&t1=3600&width=257"
            ).status_code
            == 422
        )


def test_physical_aggregation_and_gaps(fixture):
    root, _q, r = fixture
    store = Store(root / "records", root / "tiles")
    data = store.tile("5", 0, 4, 0, 10, 2, 1)
    meta, values, _mask, counts = unpack(data)
    assert meta["x_edges"] == [0, 2, 4]
    assert (
        values[0, 0] == 21 and counts[0, 0] == 3
    )  # median(20,21,23), sentinel excluded
    assert values[0, 1] == 23 and counts[0, 1] == 2
    assert data == store.tile("5", 0, 4, 0, 10, 2, 1)
    _, _v, m, n = unpack(store.tile("5", 0, 4, 0, 3600, 2, 8))
    assert np.all(m == 8) and np.all(n == 0)  # coarse bins crossing gap suppressed
    cells = cell_edges(r.elapsed_s, 300)
    assert cells[1][1] < cells[2][0]


def test_review_restart_conflict_and_atomic_idempotency(fixture):
    root, q, _r = fixture
    runtime = root / "runtime"
    with TestClient(create_app(root / "records", q, runtime)) as c:
        token = c.get("/api/v1/session").json()["mutation_token"]
        headers = {"X-Workbench-Token": token, "Idempotency-Key": "request-0001"}
        assert c.post("/api/v1/reviews", json=body(q)).status_code == 403
        first = c.post("/api/v1/reviews", json=body(q), headers=headers)
        assert first.status_code == 200
        assert (
            c.post("/api/v1/reviews", json=body(q), headers=headers).json()
            == first.json()
        )
        assert (
            c.post(
                "/api/v1/reviews",
                json=body(q),
                headers={**headers, "Idempotency-Key": "request-0002"},
            ).status_code
            == 409
        )
        assert c.get("/api/v1/queue/card/methods").status_code == 200
        export = c.get("/api/v1/export").json()
        assert (
            export["automatic_training_labels_created"] == 0
            and export["annotations"][0]["training_target"] is None
        )
    with TestClient(create_app(root / "records", q, runtime)) as c:
        token = c.get("/api/v1/session").json()["mutation_token"]
        headers = {"X-Workbench-Token": token, "Idempotency-Key": "request-0001"}
        assert (
            c.post("/api/v1/reviews", json=body(q), headers=headers).json()
            == first.json()
        )
        changed = {**body(q), "notes": "Changed"}
        assert (
            c.post("/api/v1/reviews", json=changed, headers=headers).status_code == 409
        )
    journal = ReviewJournal(q / "queue.json", root / "parallel.sqlite3")
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(
                lambda _: journal.save(body(q), idempotency_key="same-request"),
                range(4),
            )
        )
    assert {v["revision"] for v in results} == {1} and len(journal.annotations()) == 1


def test_host_origin_size_and_replay(fixture):
    root, q, _r = fixture
    with TestClient(create_app(root / "records", q, root / "runtime")) as c:
        assert (
            c.get("/api/v1/session", headers={"Host": "evil.example"}).status_code
            == 403
        )
        token = c.get("/api/v1/session").json()["mutation_token"]
        headers = {"X-Workbench-Token": token}
        assert (
            c.post(
                "/api/v1/replays",
                json={"record_id": "5"},
                headers={**headers, "Origin": "https://evil.example"},
            ).status_code
            == 403
        )
        assert (
            c.post("/api/v1/reviews", content="x" * 65537, headers=headers).status_code
            == 413
        )
        replay = c.post(
            "/api/v1/replays", json={"record_id": "5"}, headers=headers
        ).json()
        key = replay["id"]
        state = c.post(
            f"/api/v1/replays/{key}/commands",
            json={"action": "seek", "row": 2},
            headers=headers,
        ).json()
        assert state["row"] == 2 and state["position_s"] == 3600
        assert (
            c.post(
                f"/api/v1/replays/{key}/commands",
                json={"action": "seek", "row": 3},
                headers=headers,
            ).status_code
            == 422
        )
        assert (
            c.post(
                "/api/v1/jobs/detect",
                json={
                    "record_id": "5",
                    "row_start": 0,
                    "row_end": 99,
                    "column_start": 0,
                    "column_end": 3,
                },
                headers=headers,
            ).status_code
            == 413
        )
        assert c.delete(f"/api/v1/replays/{key}", headers=headers).status_code == 200
        assert c.get(f"/api/v1/replays/{key}").status_code == 404


def test_fresh_clone_queue_is_native_unlabelled_and_immutable(fixture):
    from workbench.queue_builder import build_queue
    from workbench.store import Queue

    root, _old, r = fixture
    target = build_queue(root / "records", root / "fresh-queue")
    store = Store(root / "records", root / "fresh-cache")
    queue = Queue(target, store)
    assert (
        len(queue.cards) == 1
    )  # Tiny source: all coverage windows have identical bounds.
    for key in queue.cards:
        card, data = queue.load(key)
        assert (
            card["methods"] == []
            and card["response"] is None
            and card["training_target"] is None
        )
        np.testing.assert_equal(data["raw_c"], r.raw_c)
        assert data["signed"].shape[0] == 0
    with pytest.raises(ValueError, match="immutable"):
        build_queue(root / "records", target)


def test_close_range_uses_native_cells_and_preserves_gaps(fixture):
    root, _q, r = fixture
    store = Store(root / "records", root / "cache")
    meta, values, mask, counts = unpack(store.tile("5", 0, 4, 0, 3600, 256, 256))
    assert meta["aggregation"] == "native_observations" and meta["shape"] == [3, 3]
    np.testing.assert_equal(mask, r.invalid_reason)
    np.testing.assert_equal(counts, (r.invalid_reason == 0).astype(np.uint32))
    assert np.isnan(values[0, 1]) and values[1, 1] == 23
    assert meta["time_cells"][1][1] < meta["time_cells"][2][0]
    assert meta["x_cells"] == cell_edges(r.x_m)


def test_ml_result_budget_fails_explicitly_without_retaining_large_result(
    fixture, monkeypatch
):
    import workbench.api as module

    root, q, _r = fixture

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def raise_for_status(self):
            pass

        def iter_bytes(self):
            chunk = b"x" * (4 * 1024 * 1024)
            yield chunk
            yield chunk
            yield chunk

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def stream(self, method, url, **kwargs):
            assert (
                kwargs["json"]["raw"][0][1] is None
            )  # Sentinel never reaches ML as temperature.
            return FakeResponse()

    monkeypatch.setattr(module.httpx, "Client", FakeClient)
    with TestClient(create_app(root / "records", q, root / "runtime")) as c:
        token = c.get("/api/v1/session").json()["mutation_token"]
        state = c.post(
            "/api/v1/jobs/detect",
            headers={"X-Workbench-Token": token},
            json={
                "record_id": "5",
                "row_start": 0,
                "row_end": 2,
                "column_start": 0,
                "column_end": 3,
            },
        ).json()
        for _ in range(100):
            state = c.get("/api/v1/jobs/" + state["id"]).json()
            if state["status"] in ("failed", "complete"):
                break
            time.sleep(0.01)
        assert state["status"] == "failed" and "8 MiB" in state["error"]
        assert state["result_bytes"] == 0 and "result" not in state


@pytest.mark.skipif(
    not os.environ.get("DTS_REAL_RECORDS"),
    reason="Set DTS_REAL_RECORDS for real six-record validation",
)
def test_all_real_records(tmp_path):
    store = Store(os.environ["DTS_REAL_RECORDS"], tmp_path / "tiles")
    assert len(store.records) == 6
    for key, r in store.records.items():
        for row, column in [
            (0, 0),
            (len(r.elapsed_s) // 2, len(r.x_m) // 2),
            (len(r.elapsed_s) - 1, len(r.x_m) - 1),
        ]:
            result = store.slices(key, row, column)
            assert result["invalid_reason"] == int(r.invalid_reason[row, column])
            assert result["temperature"] == (
                float(r.raw_c[row, column])
                if r.invalid_reason[row, column] == 0
                else None
            )
        meta = store.meta[key]
        _, values, mask, counts = unpack(
            store.tile(key, *meta["x_bounds"], *meta["time_bounds"], 32, 32)
        )
        assert (
            np.isfinite(values[mask == 0]).all()
            and not (values[mask == 0] == -300).any()
        )
        assert (counts[mask != 0] == 0).all()


def test_independent_review_does_not_reveal_another_subject(fixture, tmp_path):
    root, queue, _ = fixture
    with TestClient(create_app(root / "records", queue, tmp_path / "independent")) as c:
        payload = body(queue)
        payload["reviewer"] = "Alice"
        headers = {
            "X-Workbench-Token": c.get("/api/v1/session").json()["mutation_token"],
            "Idempotency-Key": "independent-alice-1",
        }
        own = "?independent=true&reviewer=Alice"
        other = "?independent=true&reviewer=Bob"
        assert (
            c.post("/api/v1/reviews" + own, json=payload, headers=headers).status_code
            == 200
        )
        assert c.get("/api/v1/queue/card/methods" + own).status_code == 200
        assert c.get("/api/v1/queue/card/methods" + other).status_code == 403
        assert c.get("/api/v1/queue" + other).json()["annotations"] == {}
        assert c.get("/api/v1/queue").json()["annotations"] == {}
        payload["reviewer"] = "Bob"
        headers["Idempotency-Key"] = "independent-bob-1"
        assert (
            c.post("/api/v1/reviews" + other, json=payload, headers=headers).json()[
                "revision"
            ]
            == 1
        )
        assert (
            c.get("/api/v1/export" + own).json()["annotations"][0]["reviewer"]
            == "Alice"
        )


def test_workspace_restart_report_and_result_sha(fixture, tmp_path):
    root, queue, _ = fixture
    records = root / "records"
    runtime = tmp_path / "workspace"
    region = {
        "record_id": "5",
        "row_start": 0,
        "row_end": 3,
        "column_start": 0,
        "column_end": 3,
    }
    with TestClient(create_app(records, queue, runtime)) as c:
        headers = {
            "X-Workbench-Token": c.get("/api/v1/session").json()["mutation_token"]
        }
        evidence = c.post(
            "/api/v1/regions/evidence", json=region, headers=headers
        ).json()
        assert evidence["temperature_c"]["min"] == 20
        assert evidence["mask_counts"] == {"0": 8, "1": 1}
        assert evidence["gaps_over_300s"] == [3590]
        bookmark = c.post(
            "/api/v1/bookmarks",
            json={**region, "title": "<script>unsafe</script>"},
            headers=headers,
        ).json()
        result = {
            "schema_version": "dts-method-result-v1",
            "region": region,
            "source_sha256": "0" * 64,
            "method": "Zakhar model",
            "method_version": "commit-123",
            "parameters": {},
            "calculated_at": "2026-10-09T00:00:00Z",
            "training_origin": "real",
            "limitations": "Not calibrated",
            "result": {"candidates": []},
        }
        assert (
            c.post("/api/v1/results", json=result, headers=headers).status_code == 422
        )
        result["source_sha256"] = evidence["source_sha256"]
        assert (
            c.post("/api/v1/results", json=result, headers=headers).status_code == 201
        )
        report = c.post("/api/v1/reports?format=html", json=region, headers=headers)
        assert (
            report.status_code == 200
            and "<svg" in report.text
            and "<script>" not in report.text
        )
        assert "3590" in report.text
        single = c.post(
            "/api/v1/reports?format=html",
            json={
                **region,
                "row_start": 0,
                "row_end": 1,
                "column_start": 0,
                "column_end": 2,
            },
            headers=headers,
        )
        assert single.status_code == 200 and "<circle" in single.text
        assert c.get("/api/v1/packages").status_code == 200
    with TestClient(create_app(records, queue, runtime)) as c:
        assert c.get("/api/v1/bookmarks").json()[0]["id"] == bookmark["id"]
        assert c.get("/api/v1/runs").json()[0]["method"] == "Zakhar model"


def test_research_adapter_verifies_cells_and_never_imports_votes(fixture, tmp_path):
    from workbench.workspace import adapt_real_pack

    root, queue, record = fixture
    pack = tmp_path / "research"
    (pack / "patches").mkdir(parents=True)
    patch = pack / "patches" / "sample.npz"

    def write_patch(change=False):
        raw = np.array(record.raw_c, dtype="f4")
        if change:
            raw[0, 0] += 1
        np.savez_compressed(
            patch,
            raw_c=raw,
            valid=record.invalid_reason == 0,
            x_m=record.x_m,
            time_s=record.elapsed_s,
            source_rows=np.arange(3),
            source_columns=np.arange(3),
            median_x3_c=raw,
        )
        (pack / "weak_labels.jsonl").write_text(
            json.dumps(
                {
                    "id": "sample",
                    "record": "5",
                    "source_sha256": record.manifest["source_sha256"],
                    "patch_sha256": sha256(patch),
                    "weak_presence_label": 1,
                }
            ),
            encoding="utf-8",
        )

    write_patch()
    (pack / "manifest.json").write_text(
        json.dumps({"schema": "real_data_pack_v1", "n_crops": 1}), encoding="utf-8"
    )
    (pack / "shortlist.json").write_text(
        json.dumps({"ids": ["sample"]}), encoding="utf-8"
    )
    store = Store(root / "records", tmp_path / "cache")
    answers = json.loads((queue / "queue.json").read_text())["allowed_answers"]
    result = adapt_real_pack(pack, store, tmp_path / "adapted", answers)
    assert result["verified_native_cells"] == 9
    generated = json.loads((tmp_path / "adapted/queue.json").read_text())
    assert "weak_presence_label" not in generated["cards"][0]
    with np.load(tmp_path / "adapted/sample.npz") as normalized:
        assert normalized["raw_c"].dtype == record.raw_c.dtype
        assert np.array_equal(normalized["invalid_reason"], record.invalid_reason)
    write_patch(True)
    with pytest.raises(ValueError, match="differs from native"):
        adapt_real_pack(pack, store, tmp_path / "bad-adapted", answers)


def test_workspace_restart_marks_unfinished_jobs_without_losing_results(tmp_path):
    from workbench.workspace import Workspace

    path = tmp_path / "workspace.sqlite3"
    workspace = Workspace(path)
    for status in ("queued", "running", "cancel_requested"):
        workspace.put("run", status, {"id": status, "status": status})
    workspace.put(
        "run", "done", {"id": "done", "status": "complete", "result": {"zones": []}}
    )
    restarted = Workspace(path)
    for status in ("queued", "running", "cancel_requested"):
        assert restarted.get("run", status)["status"] == "interrupted"
    assert restarted.get("run", "done")["result"] == {"zones": []}


def test_native_roi_trend_does_not_shift_after_gap(fixture):
    root, _queue, _record = fixture
    store = Store(root / "records", root / "tiles")
    result = store.slices("5", 2, 0, row_start=2, row_end=3)
    assert result["trend"] == [25.0]
    assert result["time_s"] == [3600.0]
    spanning = store.slices("5", 2, 0, row_start=1, row_end=3)
    assert spanning["trend"] == [21.0, None, 25.0]
    assert spanning["time_s"] == [10.0, 1805.0, 3600.0]


def test_v2_display_gap_is_distinct_from_native_quality_and_etag(fixture):
    root, queue, _record = fixture
    with TestClient(create_app(root / "records", queue, root / "runtime")) as client:
        params = {
            "x0": 0,
            "x1": 4,
            "t0": 0,
            "t1": 3600,
            "width": 2,
            "height": 2,
            "format_version": 2,
        }
        first = client.get("/api/v1/records/5/tiles", params=params)
        assert first.status_code == 200
        head, _values, native, counts = unpack(first.content)
        n = int(np.prod(head["shape"]))
        display = np.frombuffer(
            first.content, dtype="u1", count=n, offset=len(first.content) - n
        ).reshape(head["shape"])
        assert head["version"] == 2 and np.any(display == 2)
        assert not np.any(
            native == 8
        )  # This source contains only sentinel/native-valid cells.
        assert (counts[display != 0] == 0).all()
        cached = client.get(
            "/api/v1/records/5/tiles",
            params=params,
            headers={"If-None-Match": first.headers["etag"]},
        )
        assert cached.status_code == 304 and not cached.content


def test_stale_source_and_queue_versions_are_rejected(fixture):
    root, queue, _ = fixture
    with TestClient(create_app(root / "records", queue, root / "runtime")) as client:
        source = client.get("/api/v1/records/5").json()["source_sha256"]
        params = {
            "x0": 0,
            "x1": 4,
            "t0": 0,
            "t1": 3600,
            "width": 2,
            "height": 2,
            "view_version": "0" * 64,
        }
        assert client.get("/api/v1/records/5/tiles", params=params).status_code == 409
        params["view_version"] = source
        assert client.get("/api/v1/records/5/tiles", params=params).status_code == 200
        assert (
            client.get("/api/v1/records/5/slices?source_sha256=" + "0" * 64).status_code
            == 409
        )
        q = client.get("/api/v1/queue").json()
        card = q["cards"][0]
        assert card["source_sha256"] == source
        url = "/api/v1/queue/" + card["review_id"] + "/window"
        assert client.get(url, params={"view_version": "0" * 64}).status_code == 409
        assert (
            client.get(url, params={"view_version": q["queue_sha256"]}).status_code
            == 200
        )
