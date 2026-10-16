"""Real measurement replay through the existing Redis/ClickHouse ingestion stack."""

import json
import os
import time
import warnings
from contextlib import suppress
from urllib.parse import quote

import httpx
import numpy as np

# Narrow the immutable sorting key before loading large Array columns under FINAL.
# Predicates on non-key temperature/quality columns must never move here.
READ_SETTINGS = "SETTINGS max_threads=1,max_block_size=64,preferred_block_size_bytes=1048576"


def persisted(client, key):
    query = f"SELECT arraySort(groupUniqArray(row_idx)) AS rows FROM haklolarp.profiles_v2 FINAL PREWHERE replay_id='{key}' {READ_SETTINGS} FORMAT JSONEachRow"
    response = client.get(
        os.environ.get("DTS_CH_URL", "http://127.0.0.1:8123")
        + "/?query="
        + quote(query)
    )
    response.raise_for_status()
    rows = response.json()["rows"]
    cursor = -1
    for index in rows:
        if index != cursor + 1:
            break
        cursor = index
    return cursor


def verify_and_analyze(client, platform, record, state):
    """Advance only after every stored native cell matches its immutable source."""
    cursor = persisted(client, state["id"])
    state["persisted"] = cursor
    state["storage_poll_at"] = time.time()
    state["backlog_profiles"] = max(0, state["accepted"] - cursor)
    for start in range(state["analyzed"] + 1, cursor + 1, 64):
        end = min(cursor + 1, start + 64)
        query = (
            "SELECT row_idx,source_sha256,arrayMap(x->reinterpretAsUInt64(x),x_m) AS x_bits,arrayMap(x->reinterpretAsUInt64(ifNull(x,0)),temp_c) AS temp_bits,arrayMap(x->isNull(x),temp_c) AS nulls,invalid_reason,elapsed_s,segment_id,ts_raw,time_kind "
            f"FROM haklolarp.profiles_v2 FINAL PREWHERE replay_id='{state['id']}' AND row_idx>={start} AND row_idx<{end} ORDER BY row_idx {READ_SETTINGS} FORMAT JSONEachRow"
        )
        response = client.get(
            os.environ.get("DTS_CH_URL", "http://127.0.0.1:8123")
            + "/?query="
            + quote(query)
        )
        response.raise_for_status()
        rows = [json.loads(line) for line in response.text.splitlines() if line]
        if len(rows) != end - start:
            raise ValueError("Persisted native range is incomplete")
        for expected, stored in zip(range(start, end), rows):
            native = record.raw_c[expected]
            actual = np.array(
                [int(v) for v in stored["temp_bits"]], dtype=np.uint64
            ).view(np.float64)
            actual[np.array(stored["nulls"], dtype=bool)] = np.nan
            if (
                int(stored["row_idx"]) != expected
                or stored["source_sha256"] != state["source_sha256"]
                or not np.array_equal(
                    np.array([int(v) for v in stored["x_bits"]], dtype=np.uint64),
                    record.x_m.view(np.uint64),
                )
                or not np.array_equal(
                    stored["invalid_reason"], record.invalid_reason[expected]
                )
                or not np.array_equal(actual, native, equal_nan=True)
                or float(stored["elapsed_s"]) != float(record.elapsed_s[expected])
                or int(stored["segment_id"]) != int(record.segment_id[expected])
                or stored["ts_raw"] != record.profiles[expected]["timestamp"]
                or stored["time_kind"] not in ("zoned", "local_unzoned")
                or (stored["time_kind"] == "local_unzoned")
                != (record.manifest["time_kind"] == "local_unzoned")
            ):
                raise ValueError("Stored measurement differs from native source")
            values = np.where(record.invalid_reason[expected] == 0, native, np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                clean = np.nanmedian(
                    np.lib.stride_tricks.sliding_window_view(
                        np.pad(values, (1, 1), mode="edge"), 3
                    ),
                    axis=-1,
                )
            clean[record.invalid_reason[expected] != 0] = np.nan
            state["analyzed"] = expected
            state["verified_persisted"] = expected
            state["verified_cells"] = state.get("verified_cells", 0) + len(native)
            state["dsp_profiles"] = state.get("dsp_profiles", 0) + 1
        state["last_profile_dsp"] = platform.blobs.put(
            {
                "method": "median_x3",
                "method_version": "1",
                "training_origin": "untrained",
                "row": end - 1,
                "source_sha256": state["source_sha256"],
                "denoised": [float(v) if np.isfinite(v) else None for v in clean],
            }
        )
    state["analyzed_at"] = time.time()
    state["snapshot_version"] += 1
    platform.put("stream", state["id"], state)


def replay(platform, store, job):
    body = job["payload"]
    record = store.get(body["record_id"])
    if (
        body.get("source_sha256", record.manifest["source_sha256"])
        != record.manifest["source_sha256"]
    ):
        raise ValueError("Submitted source version is unavailable")
    key = job["id"]
    if len(key) != 32 or any(c not in "0123456789abcdef" for c in key):
        raise ValueError("Invalid replay identity")
    meta = store.meta[body["record_id"]]
    base = os.environ.get("DTS_INGEST_URL", "http://127.0.0.1:8000")
    try:
        state = platform.get("stream", key)
    except KeyError:
        state = {
            "id": key,
            "record_id": body["record_id"],
            "source_sha256": meta["source_sha256"],
            "project_id": job["project_id"],
            "accepted": -1,
            "persisted": -1,
            "analyzed": -1,
            "status": "starting",
            "mode": "measurement_replay",
            "snapshot_version": 0,
            "trained_classifier_operational": False,
        }
    state["status"] = "starting"
    platform.put("stream", key, state)
    with httpx.Client(timeout=30) as client:
        manifest = client.get(base + "/api/manifest")
        manifest.raise_for_status()
        token = manifest.json()["mutation_token"]
        headers = {"X-DTS-Token": token}
        response = client.post(
            base + "/api/session",
            headers=headers,
            json={
                "mode": "measurement_replay",
                "replay_id": key,
                "source_sha256": meta["source_sha256"],
                "schema_version": 2,
            },
        )
        response.raise_for_status()
        sid = response.json()["session_id"]
        try:
            cursor = persisted(client, key)
            # An INSERT may have succeeded before the last snapshot was committed.
            state["accepted"] = max(state["accepted"], cursor)
            verify_and_analyze(client, platform, record, state)
            start = cursor + 1
            for row in range(start, len(record.profiles)):
                if platform.job(key)["cancel_requested"]:
                    state["status"] = "cancelled"
                    break
                raw = record.raw_c[row]
                mask = record.invalid_reason[row]
                profile = {
                    "dts_id": meta["dts_id"],
                    "channel_id": meta["channel_id"],
                    "timestamp": record.profiles[row]["timestamp"],
                    "points": [
                        [float(x), float(v) if np.isfinite(v) else None]
                        for x, v in zip(record.x_m, raw)
                    ],
                }
                submitted = time.monotonic()
                response = client.post(
                    base + "/api/infer",
                    headers=headers,
                    json={
                        "session_id": sid,
                        "profile": profile,
                        "source_row": row,
                        "source_segment": int(record.segment_id[row]),
                        "elapsed_s": float(record.elapsed_s[row]),
                        "native_invalid_reason": mask.tolist(),
                    },
                )
                deadline = time.monotonic() + 60
                while response.status_code == 503 and time.monotonic() < deadline:
                    if platform.job(key)["cancel_requested"]:
                        state["status"] = "cancelled"
                        break
                    time.sleep(0.5)
                    response = client.post(
                        base + "/api/infer",
                        headers=headers,
                        json={
                            "session_id": sid,
                            "profile": profile,
                            "source_row": row,
                            "source_segment": int(record.segment_id[row]),
                            "elapsed_s": float(record.elapsed_s[row]),
                            "native_invalid_reason": mask.tolist(),
                        },
                    )
                if state["status"] == "cancelled":
                    break
                response.raise_for_status()
                state.update(
                    accepted=row,
                    status="running",
                    accepted_at=time.time(),
                    last_accept_ms=(time.monotonic() - submitted) * 1000,
                )
                if row % 10 == 0 or row == len(record.profiles) - 1:
                    verify_and_analyze(client, platform, record, state)
                time.sleep(body.get("delay_s", 0.05))
            if state["status"] != "cancelled":
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    verify_and_analyze(client, platform, record, state)
                    if state["persisted"] == state["accepted"]:
                        break
                    if platform.job(key)["cancel_requested"]:
                        state["status"] = "cancelled"
                        break
                    time.sleep(0.5)
                if state["status"] != "cancelled":
                    state["status"] = (
                        "complete"
                        if state["persisted"] == state["accepted"]
                        else "storage_lag"
                    )
        except Exception:
            state["status"] = "interrupted"
            raise
        finally:
            # Closing a transport session must not hide the original failure.
            with suppress(httpx.HTTPError):
                client.post(
                    base + "/api/session/close",
                    headers=headers,
                    json={"session_id": sid},
                )
            state["snapshot_version"] += 1
            platform.put("stream", key, state)
    return {k: v for k, v in state.items() if k != "last_profile_dsp"}
