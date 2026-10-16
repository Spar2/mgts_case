"""Mixed load on an explicitly disposable stack, including persisted QA revisions."""
import argparse
import json
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--credentials", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base", default="https://localhost:8443")
    parser.add_argument("--seconds", type=int, default=1800)
    parser.add_argument("--disposable-stack", action="store_true", required=True)
    args = parser.parse_args()
    users = json.loads(args.credentials.read_text())
    clients = [httpx.Client(base_url=args.base, verify=False, timeout=30, trust_env=False) for _ in range(10)]
    csrf, cards, revisions = [], [], [0] * 5
    for i, user in enumerate(users[:5]):
        client = clients[i + 5]
        response = client.post("/api/v2/auth/login", json=user)
        response.raise_for_status()
        csrf.append(response.json()["csrf"])
        queue_response = client.get("/api/v1/queue", params={"independent": "true", "reviewer": user["username"]})
        queue_response.raise_for_status()
        queue = queue_response.json()
        card = queue["cards"][i]
        cards.append((card, queue["queue_sha256"]))
    records = clients[5].get("/api/v1/records").json()
    for record in records:
        params = dict(zip(("x0", "x1", "t0", "t1"), [*record["x_bounds"], *record["time_bounds"]]), width=128, height=128, format_version=2)
        clients[5].get(f"/api/v1/records/{record['id']}/tiles", params=params).raise_for_status()
    elapsed_reads, errors, samples, writes = [], [], [], 0
    start, cycle = time.monotonic(), 0
    with ThreadPoolExecutor(10) as pool:
        while time.monotonic() - start < args.seconds:
            def read(i):
                client = clients[i]
                begin = time.perf_counter()
                if i < 5:
                    response = client.get("/api/public/v1/package")
                else:
                    record = records[(cycle + i) % len(records)]
                    params = dict(zip(("x0", "x1", "t0", "t1"), [*record["x_bounds"], *record["time_bounds"]]), width=128, height=128, format_version=2)
                    response = client.get(f"/api/v1/records/{record['id']}/tiles", params=params)
                return i, response.status_code, (time.perf_counter() - begin) * 1000
            for i, status, elapsed in pool.map(read, range(10)):
                elapsed_reads.append(elapsed)
                if status != 200:
                    errors.append({"cycle": cycle, "client": i, "status": status})
            if cycle % 30 == 0:
                for i in range(5):
                    card, sha = cards[i]
                    vote = {"queue_sha256": sha, "review_id": card["review_id"], "expected_revision": revisions[i], "reviewer": users[i]["username"], "reviewer_type": "expert", "temperature_episode": "unknown", "phenomenon": "unknown", "notes": "DISPOSABLE LOAD TEST ONLY: not scientific annotation; never export to training"}
                    headers = {"X-Workbench-Token": csrf[i], "Idempotency-Key": f"mixed-load-{i}-{cycle}"}
                    response = clients[i + 5].post("/api/v1/reviews", params={"independent": "true", "reviewer": users[i]["username"]}, json=vote, headers=headers)
                    if response.status_code != 200:
                        errors.append({"write": i, "cycle": cycle, "status": response.status_code})
                    else:
                        revisions[i] = response.json()["revision"]
                        writes += 1
                        repeated = clients[i + 5].post("/api/v1/reviews", params={"independent": "true", "reviewer": users[i]["username"]}, json=vote, headers=headers)
                        if repeated.status_code != 200 or repeated.json()["revision"] != revisions[i]:
                            errors.append({"idempotency": i, "cycle": cycle})
                result = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.Name}}|{{.MemUsage}}", "dts-production-qa-api", "dts-production-qa-web", "dts-production-qa-worker-1"], capture_output=True, text=True)
                samples.append({"elapsed_s": round(time.monotonic() - start, 1), "memory": result.stdout.splitlines()})
                print(json.dumps({"elapsed_s": round(time.monotonic() - start), "reads": len(elapsed_reads), "writes": writes, "errors": len(errors)}), flush=True)
            cycle += 1
            time.sleep(min(2, max(0, args.seconds - (time.monotonic() - start))))
    ordered = sorted(elapsed_reads)
    report = {"seconds": time.monotonic() - start, "users": {"public": 5, "members": 5}, "reads": len(ordered), "writes": writes, "final_revisions": revisions, "errors": errors, "warm_read_p95_ms": ordered[int(.95 * (len(ordered) - 1))], "median_ms": statistics.median(ordered), "samples": samples, "conditions": "Local Docker Desktop, HTTPS Nginx, real six records warmed; httpx 10 threads, 2s think time; disposable PG and volumes; no physical phone or external network simulated"}
    args.output.write_text(json.dumps(report, indent=2))
    for client in clients:
        client.close()
    print(json.dumps({k: v for k, v in report.items() if k not in ("samples", "conditions")}), flush=True)


if __name__ == "__main__":
    main()
