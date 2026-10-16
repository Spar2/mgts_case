"""Safe structured request telemetry; no cookies, bodies or query strings."""

import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path

from fastapi import HTTPException
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from .platform import principal

logger = logging.getLogger("dts.requests")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)


def install_telemetry(app):
    counters = {}
    lock = threading.Lock()

    @app.get("/api/v2/metrics")
    def metrics():
        user = principal.get()
        if not user or user["role"] != "admin":
            raise HTTPException(403, "Administrator required")
        with lock:
            requests = [dict(route=k[0], status=k[1], **v) for k, v in counters.items()]
        backup_age = None
        try:
            marker = json.loads(
                (
                    Path(os.environ.get("DTS_RUNTIME", ".runtime"))
                    / "backup-latest.json"
                ).read_text()
            )
            backup_age = max(0, time.time() - marker["completed_at"])
        except (OSError, ValueError, KeyError):
            pass
        with app.state.platform.connect() as db:
            jobs = [
                {"status": r[0], "count": r[1]}
                for r in db.execute(
                    "SELECT status,count(*) FROM jobs WHERE project_id=%s GROUP BY status",
                    (user["project_id"],),
                )
            ]
        return {
            "requests": requests,
            "jobs": jobs,
            "backup_age_s": backup_age,
            "latency_buckets_ms": [10, 50, 100, 500, 1000, 5000],
            "scope": "Request counters cover this API process; queues and backup marker are shared",
        }

    provider = TracerProvider(resource=Resource.create({"service.name": "dts-web-api"}))
    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    tracer = provider.get_tracer("workbench")

    @app.middleware("http")
    async def measured(request, call_next):
        start = time.perf_counter()
        identity = uuid.uuid4().hex
        with tracer.start_as_current_span(request.method + " request") as span:
            response = await call_next(request)
            route = getattr(request.scope.get("route"), "path", "unmatched")
            span.set_attribute("http.request.method", request.method)
            span.set_attribute("http.route", route)
            span.set_attribute("http.response.status_code", response.status_code)
            elapsed = (time.perf_counter() - start) * 1000
            with lock:
                entry = counters.setdefault(
                    (route, response.status_code),
                    {"count": 0, "duration_sum_ms": 0, "buckets": [0] * 6},
                )
                entry["count"] += 1
                entry["duration_sum_ms"] += elapsed
                for i, limit in enumerate((10, 50, 100, 500, 1000, 5000)):
                    if elapsed <= limit:
                        entry["buckets"][i] += 1
            logger.info(
                json.dumps(
                    {
                        "request_id": identity,
                        "method": request.method,
                        "route": route,
                        "status": response.status_code,
                        "duration_ms": round(elapsed, 3),
                    }
                )
            )
            response.headers["X-Request-ID"] = identity
            response.headers["Server-Timing"] = f"api;dur={elapsed:.3f}"
            return response

    return provider
