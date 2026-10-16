"""Loopback acceptance: cancel and resume a real archive with its original ID.

Requires the local platform and private .runtime/local-admin.json. No votes,
training, or external calls. Adds a technical replay to the local job history.
"""

import json
import time
from pathlib import Path

import httpx


def main():
    with httpx.Client(
        base_url="http://127.0.0.1:8081", timeout=30, trust_env=False
    ) as client:
        client.post(
            "/api/v2/auth/login",
            json=json.loads(
                Path(".runtime/local-admin.json").read_text(encoding="utf-8")
            ),
        ).raise_for_status()
        headers = {
            "X-Workbench-Token": client.get("/api/v1/session").json()["mutation_token"]
        }
        client.get("/api/v2/health/ready").raise_for_status()
        response = client.post(
            "/api/v2/jobs",
            headers=headers,
            json={"kind": "stream_replay", "record_id": "4", "delay_s": 0.01},
        )
        response.raise_for_status()
        job = response.json()
        key = job["id"]
        try:
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                snapshot = client.get("/api/v2/stream/" + key)
                if snapshot.is_success and snapshot.json()["accepted"] >= 10:
                    break
                time.sleep(0.25)
            else:
                raise RuntimeError("No accepted profiles before timeout")
            client.post(
                "/api/v2/jobs/" + key + "/cancel", headers=headers, json={}
            ).raise_for_status()
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                jobs = client.get("/api/v2/jobs?limit=100").json()["items"]
                current = next(j for j in jobs if j["id"] == key)
                if current["status"] == "cancelled":
                    break
                time.sleep(0.25)
            else:
                raise RuntimeError("Cancellation did not finish")
            before = client.get("/api/v2/stream/" + key).json()
            response = client.post(
                "/api/v2/jobs/" + key + "/resume", headers=headers, json={}
            )
            response.raise_for_status()
            assert response.json()["id"] == key
            deadline = time.monotonic() + 240
            while time.monotonic() < deadline:
                after = client.get("/api/v2/stream/" + key).json()
                if after["status"] == "complete":
                    current = next(
                        j
                        for j in client.get("/api/v2/jobs?limit=100").json()["items"]
                        if j["id"] == key
                    )
                    if current["status"] == "complete":
                        break
                if after["status"] == "interrupted":
                    raise RuntimeError("Resumed replay interrupted")
                time.sleep(0.5)
            else:
                raise RuntimeError("Resumed replay did not finish")
            records = client.get("/api/v1/records").json()
            record = next(r for r in records if r["id"] == "4")
            last = record["shape"][0] - 1
            assert (
                after["accepted"]
                == after["persisted"]
                == after["analyzed"]
                == after["verified_persisted"]
                == last
            )
            assert after["dsp_profiles"] == record["shape"][0]
            assert after["verified_cells"] == record["shape"][0] * record["shape"][1]
            report = {
                "same_replay_id": True,
                "cancelled_after_profiles": before["accepted"] + 1,
                "resumed_profiles": after["dsp_profiles"],
                "verified_cells": after["verified_cells"],
                "accepted": after["accepted"] + 1,
                "persisted": after["persisted"] + 1,
                "analyzed": after["analyzed"] + 1,
                "source_sha256": after["source_sha256"],
                "conditions": "Local Docker stack; real 4.csv; explicit cancellation and resume, no expert votes",
            }
            Path(".runtime/stream-resume-check.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )
            print(json.dumps(report))
        finally:
            # A failing acceptance run must not leave an uncontrolled replay running.
            client.post("/api/v2/jobs/" + key + "/cancel", headers=headers, json={})


if __name__ == "__main__":
    main()
