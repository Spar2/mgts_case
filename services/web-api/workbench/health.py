"""Bounded, cached readiness probes. Never expose service URLs or credentials."""

import os
import threading
import time
from pathlib import Path

import httpx


def heartbeat_age(runtime, role):
    try:
        value = float((Path(runtime) / ("worker-" + role + ".heartbeat")).read_text())
        age = time.time() - value
        return age if 0 <= age < 3600 else None
    except (OSError, ValueError):
        return None


class Readiness:
    def __init__(self, runtime):
        self.runtime = runtime
        self.lock = threading.Lock()
        self.checked = 0
        self.snapshot = {}

    def get(self):
        with self.lock:
            if time.monotonic() - self.checked < 5:
                return self.snapshot
            compute = heartbeat_age(self.runtime, "compute")
            stream = heartbeat_age(self.runtime, "stream")
            snapshot = {
                "compute_worker": compute is not None and compute <= 30,
                "stream_worker": stream is not None and stream <= 30,
                "ingestion": False,
                "redis": False,
                "clickhouse": False,
                "legacy_ml": False,
                "legacy_ml_version": None,
                "backlog_profiles": None,
                "checked_at": time.time(),
            }
            with httpx.Client(timeout=.75,trust_env=False) as client:
                try:
                    response=client.get(os.environ.get('DTS_ML_URL','http://127.0.0.1:8082').rstrip('/')+'/health')
                    if response.is_success:
                        value=response.json()
                        snapshot['legacy_ml']=isinstance(value,dict) and value.get('status')=='ok' and 'median_x3' in value.get('denoise_methods',[])
                        version=value.get('implementation_sha') if snapshot['legacy_ml'] else None
                        snapshot['legacy_ml_version']=version if isinstance(version,str) and len(version)==40 and all(c in '0123456789abcdef' for c in version) else None
                except (httpx.HTTPError,ValueError,TypeError):pass
            if snapshot["stream_worker"]:
                with httpx.Client(timeout=2, trust_env=False) as client:
                    try:
                        response = client.get(
                            os.environ.get("DTS_INGEST_URL", "http://127.0.0.1:8000")
                            + "/readyz"
                        )
                        snapshot["ingestion"] = response.is_success
                        if response.is_success:
                            value = response.json()
                            snapshot["redis"] = value.get("redis") is True
                            snapshot["backlog_profiles"] = value.get("backlog_profiles")
                    except (httpx.HTTPError, ValueError):
                        pass
                    try:
                        response = client.get(
                            os.environ.get("DTS_CH_URL", "http://127.0.0.1:8123"),
                            params={"query": "SELECT 1"},
                        )
                        snapshot["clickhouse"] = (
                            response.is_success and response.text.strip() == "1"
                        )
                    except httpx.HTTPError:
                        pass
            self.snapshot = snapshot
            self.checked = time.monotonic()
            return snapshot
