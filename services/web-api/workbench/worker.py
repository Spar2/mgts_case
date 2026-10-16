"""Leased worker: computations execute outside API and database transactions."""

import json
import os
import signal
import socket
import threading
import time
import uuid
from pathlib import Path

import httpx
import numpy as np

from .heartbeat import publish
from .platform import Platform, worker_lease
from .store import Store, cache_guard

STOP = threading.Event()


def beat(runtime):
    path = runtime / (
        "worker-" + os.environ.get("DTS_WORKER_ROLE", "compute") + ".heartbeat"
    )
    publish(path, str(time.time()))


def runtime_path(platform):
    return platform.blobs.root.parent


def execute(platform, store, job):
    body = job["payload"]
    if job["kind"] == "analysis_field":
        from .analysis_jobs import execute_analysis

        return execute_analysis(platform, store, job, stop=STOP)
    if job["kind"] == "prepare_views":
        prepared = 0
        for key, meta in store.meta.items():
            for size in (64, 128, 256):
                if platform.job(job["id"])["cancel_requested"]:
                    return {"prepared": prepared}
                store.tile(
                    key,
                    *meta["x_bounds"],
                    *meta["time_bounds"],
                    size,
                    size,
                    format_version=2,
                )
                prepared += 1
        package = runtime_path(platform) / "packages" / "real-data-v2"
        if (package / "queue.json").exists():
            from .store import Queue

            queue = Queue(package, store, max_cells=1_000_000)
            receipt = json.loads(
                (package / "verification.json").read_text(encoding="utf-8")
            )
            shortlist = receipt.get("info", {}).get("shortlist", [])
            for key in shortlist:
                if platform.job(job["id"])["cancel_requested"]:
                    break
                card = queue.cards[key]
                store.tile(
                    card["source"].removesuffix(".csv"),
                    *card["x_bounds_m"],
                    *card["time_bounds_s"],
                    128,
                    128,
                    format_version=2,
                )
                prepared += 1
        return {"prepared": prepared}
    if job["kind"] == "stream_replay":
        from .streaming import replay

        return replay(platform, store, job)
    if job["kind"] == "import_csv":
        from .vendor.data import ingest

        source = Path(os.environ["DTS_IMPORT_ROOT"]).resolve()
        path = (source / body["filename"]).resolve()
        if not path.is_relative_to(source) or path.suffix.lower() != ".csv":
            raise ValueError("CSV must belong to configured import directory")
        destination = runtime_path(platform) / "imported-records"
        destination.mkdir(parents=True, exist_ok=True)
        # Serialize imports across workers, then refresh before checking names.
        # Publishing a duplicate manifest would make subsequent catalogs fail.
        with cache_guard(destination):
            store.refresh_imports()
            if path.name.removesuffix(".csv") in store.records:
                raise ValueError(
                    "Source name already exists; use a distinct CSV filename"
                )
            record = ingest(path, destination)
        return {
            "source_sha256": record.manifest["source_sha256"],
            "reload_catalog": True,
        }
    if job["kind"] != "detect":
        raise ValueError("Unsupported job kind")
    r = store.get(body["record_id"])
    if r.manifest["source_sha256"] != body["source_sha256"]:
        raise ValueError("Source version changed")
    rs, re, cs, ce = (
        body[k] for k in ("row_start", "row_end", "column_start", "column_end")
    )
    if not (
        0 <= rs < re <= r.raw_c.shape[0]
        and 0 <= cs < ce <= r.raw_c.shape[1]
        and (re - rs) * (ce - cs) <= 1_000_000
    ):
        raise ValueError("Invalid bounded native region")
    raw = r.raw_c[rs:re, cs:ce]
    mask = r.invalid_reason[rs:re, cs:ce]
    if np.any((mask == 0) & (raw <= -200)):
        raise ValueError("ML contract would mask valid native measurements")
    payload = {
        "raw": np.where(mask == 0, raw, None).tolist(),
        "distances": r.x_m[cs:ce].tolist(),
        "time_s": r.elapsed_s[rs:re].tolist(),
    }
    with (
        httpx.Client(timeout=180) as client,
        client.stream(
            "POST",
            os.environ.get("DTS_ML_URL", "http://127.0.0.1:8082") + "/detect/field",
            json=payload,
        ) as response,
    ):
        response.raise_for_status()
        chunks = []
        size = 0
        for chunk in response.iter_bytes():
            size += len(chunk)
            if size > 8 * 1024 * 1024:
                raise ValueError("ML payload exceeds 8 MiB")
            chunks.append(chunk)
    result = json.loads(b"".join(chunks))
    json.dumps(result, allow_nan=False)
    if not isinstance(result, dict):
        raise ValueError("ML result must be an object")  # noqa: TRY004
    return platform.blobs.put(result)


def main():
    runtime = Path(os.environ.get("DTS_RUNTIME", ".runtime"))
    platform = Platform(os.environ["DTS_DATABASE_URL"], runtime)
    store = Store(os.environ["DTS_RECORDS"], runtime / "tiles")
    kinds = os.environ.get(
        "DTS_JOB_KINDS", "detect,prepare_views,import_csv,stream_replay,analysis_field"
    ).split(",")
    analysis_enabled = "analysis_field" in kinds
    if analysis_enabled:
        import scipy

        from .analysis_compute import verify_pin
        from .analysis_jobs import analysis_beat

        verify_pin()
        if scipy.__version__ != "1.18.1" or np.__version__ != "2.4.3":
            raise RuntimeError("Pinned analysis dependencies differ")
    worker_id = f"{socket.gethostname()}-{os.getpid()}"
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: STOP.set())
    while not STOP.is_set():
        if analysis_enabled:
            analysis_beat(runtime)
        owner = worker_id + "-" + uuid.uuid4().hex
        job = platform.claim(
            owner,
            kinds,
        )
        if not job:
            beat(runtime)
            STOP.wait(0.5)
            continue
        done = threading.Event()

        def heartbeat(job=job, done=done, owner=owner):
            while not done.wait(5):
                platform.heartbeat(job["id"], owner)
                beat(runtime)
                if analysis_enabled:
                    analysis_beat(runtime)

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        scope = worker_lease.set({"id": job["id"], "owner": owner})
        try:
            store.refresh_imports()
            result = execute(platform, store, job)
            platform.finish(job["id"], owner, result=result)
        except Exception as exc:  # noqa: BLE001 - isolate failed leased jobs
            platform.finish(
                job["id"],
                owner,
                error=type(exc).__name__ + ": " + str(exc)[:200],
                retry=isinstance(exc, httpx.RequestError),
            )
        finally:
            worker_lease.reset(scope)
            done.set()
            thread.join(timeout=1)
            beat(runtime)
        # Catalog changes are observed after preparation, not during native reads.
        if job["kind"] == "import_csv":
            store = Store(os.environ["DTS_RECORDS"], runtime / "tiles")


if __name__ == "__main__":
    main()
