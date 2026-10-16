"""Сборка и проверка документа dts-events/1."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from ml_common.protocol import grid_steps, thread_cap

SCHEMA_PATH = Path(__file__).resolve().parent / "schema" / "events_v1.schema.json"
REPO_ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def code_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    sha = out.strip()
    return sha if len(sha) >= 7 else "unknown"


@lru_cache(maxsize=1)
def _validator():
    import jsonschema

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return jsonschema.Draft202012Validator(schema)


def validate_events(document: dict) -> dict:
    """Проверить документ схемой. NaN/Inf отвергаются. Возвращается тот же dict."""
    encoded = json.dumps(document, ensure_ascii=False, allow_nan=False)
    parsed = json.loads(encoded)
    errors = sorted(_validator().iter_errors(parsed), key=lambda err: list(err.path))
    if errors:
        first = errors[0]
        where = "/".join(str(part) for part in first.path) or "<root>"
        raise ValueError(f"events.json v1: {where}: {first.message}")
    return parsed


def iso_for(ts: float, time_s, time_iso) -> str:
    """Ближайшая подпись времени. Без подписей time_s читается как unix-секунды."""
    if time_iso is not None and time_s is not None and len(time_iso) == len(time_s):
        import numpy as np

        index = int(np.argmin(np.abs(np.asarray(time_s, dtype=np.float64) - float(ts))))
        return str(time_iso[index])
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()


def build_events(
    *,
    method: str,
    model_key: str,
    field: dict,
    zones: list[dict],
    threshold: float,
    seconds: float,
    score_kind: str,
    threshold_kind: str,
    device: str = "cpu",
) -> dict:
    dx, dt = grid_steps(field["x_m"], field["time_s"])
    document = {
        "schema": "dts-events/1",
        "model_key": model_key,
        "method": method,
        "dts_id": field["dts_id"],
        "channel_id": field["channel_id"],
        "grid": {"dx_m": round(dx, 6), "dt_s": round(dt, 6)},
        "threshold": float(threshold),
        "code_sha": code_sha(),
        "zones": zones,
        "runtime": {
            "cpus": thread_cap(),
            "seconds": round(float(seconds), 6),
            "device": device,
            "threads": thread_cap(),
            "score_kind": score_kind,
            "threshold_kind": threshold_kind,
        },
    }
    return validate_events(document)
