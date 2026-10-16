"""Local real-data acceptance of the rehearsal-specific stream endpoints.

Creates its own technical replay, never stops shared services or sends answers.
Requires the running 8081 platform and private local-admin.json.
"""

import json
import time
import uuid
from pathlib import Path

import httpx


def main():
    job_id = None
    finished = False
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
        candidate = next(
            c
            for c in client.get(
                "/api/v1/candidates?package=real-data-v2&shortlist=true"
            ).json()
            if c["region"]["record_id"] == "4"
        )
        session = client.post(
            "/api/v1/demo/sessions",
            json={
                "client_key": str(uuid.uuid4()),
                "package": "real-data-v2",
                "candidate": candidate["id"],
            },
            headers=headers,
        )
        session.raise_for_status()
        route = "/api/v1/demo/sessions/" + session.json()["id"] + "/stream"
        try:
            started = client.post(route, json={}, headers=headers)
            started.raise_for_status()
            job_id = started.json()["id"]
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                state = client.get(route).json()
                if state["snapshot"] and state["snapshot"]["accepted"] >= 3:
                    break
                if state["job"]["status"] in ("failed", "cancelled", "complete"):
                    raise RuntimeError(
                        "Stream did not enter an interruptible live state"
                    )
                time.sleep(0.3)
            else:
                raise TimeoutError("No accepted real profiles")
            accepted_at_cancel = state["snapshot"]["accepted"]
            client.post(route + "/cancel", json={}, headers=headers).raise_for_status()
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                state = client.get(route).json()
                if state["job"]["status"] == "cancelled":
                    break
                time.sleep(0.3)
            else:
                raise TimeoutError("Cancellation not confirmed")
            resumed = client.post(route + "/resume", json={}, headers=headers)
            resumed.raise_for_status()
            assert resumed.json()["id"] == job_id
            print(
                f"Real archive interrupted after {accepted_at_cancel} profiles; same technical job resumed",
                flush=True,
            )
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                response = client.get(route)
                response.raise_for_status()
                state = response.json()
                if (
                    state["job"]["status"] == "complete"
                    and state["snapshot"]["status"] == "complete"
                ):
                    break
                if state["job"]["status"] in ("failed", "cancelled"):
                    raise RuntimeError("Resumed technical replay failed")
                time.sleep(1)
            else:
                raise TimeoutError("Archive not completed within acceptance interval")
            snapshot = state["snapshot"]
            for name in (
                "accepted",
                "persisted",
                "analyzed",
                "verified_persisted",
                "dsp_profiles",
            ):
                assert snapshot[name] == snapshot["total_profiles"], name
            evidence = {
                "measurement_date": "2026-10-10",
                "source": "4.csv",
                "source_sha256": candidate["source_sha256"],
                "accepted_at_cancel": accepted_at_cancel,
                "same_job_resumed": True,
                "rehearsal_endpoints": True,
                **{
                    name: snapshot[name]
                    for name in (
                        "accepted",
                        "persisted",
                        "analyzed",
                        "verified_persisted",
                        "verified_cells",
                        "total_profiles",
                        "dsp_profiles",
                    )
                },
            }
            Path(".runtime/demo-goal-20261010/stream-acceptance.json").write_text(
                json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
            )
            finished = True
            print(json.dumps(evidence), flush=True)
        finally:
            if job_id and not finished:
                client.post(route + "/cancel", json={}, headers=headers)
            client.post("/api/v1/demo/leave", json={}, headers=headers)


if __name__ == "__main__":
    main()
