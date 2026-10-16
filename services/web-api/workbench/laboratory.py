"""Read-only real studies, explicitly distinct from historical synthetic benchmarks."""

import hashlib
import json
from pathlib import Path

from fastapi import HTTPException

from .platform import principal

ROOT = Path(__file__).resolve().parents[1] / "research/ml-next"
MANIFEST_SHA = "072a2772f079f7efa40ab553b3f2cc4dcf7ff80eb5e7bc2f2d40fb726d7d21e9"


def real_study():
    encoded = (ROOT / "manifest.json").read_bytes()
    if hashlib.sha256(encoded).hexdigest() != MANIFEST_SHA:
        raise ValueError("Research manifest differs")
    manifest = json.loads(encoded)
    data = {}
    for name, checksum in manifest["files"].items():
        if name not in (
            "report.json",
            "protocol.json",
            "temporal-report.json",
            "classification-report.json",
        ):
            raise ValueError("Unexpected aggregate artifact")
        encoded = (ROOT / name).read_bytes()
        if (
            len(encoded) > 1024 * 1024
            or hashlib.sha256(encoded).hexdigest() != checksum
        ):
            raise ValueError("Research artifact differs")
        data[name.removesuffix(".json")] = json.loads(encoded)
    return {
        "schema_version": "dts-real-study-v1",
        "source_commit": manifest["source_commit"],
        "manifest_sha256": MANIFEST_SHA,
        "study_origin": "real_only_previously_seen",
        "files": manifest["files"],
        "data": data,
    }


def install_laboratory(app):
    def access():
        user = principal.get()
        if not user:
            raise HTTPException(401, "Sign in to read laboratory")
        if user["project_id"] != "local":
            raise HTTPException(403, "Study sources belong to local project")

    @app.get("/api/v2/research-studies")
    def catalog():
        access()
        return {
            "items": [
                {
                    "id": "real-next",
                    "title": "Real-only цикл: очистка, обнаружение, классификация",
                    "origin": "real_only_previously_seen",
                    "operational_approved": False,
                },
                {
                    "id": "ml-237168f",
                    "title": "Исторические пороги и refinement",
                    "origin": "synthetic_calibration_plus_real_diagnostics",
                    "operational_approved": False,
                },
            ]
        }

    @app.get("/api/v2/research-studies/{key}")
    def detail(key: str):
        access()
        if key != "real-next":
            raise HTTPException(404, "Use historical ml-experiments endpoint")
        try:
            return real_study()
        except (OSError, ValueError, KeyError):
            raise HTTPException(
                503, "Research report unavailable; saved results are unchanged"
            ) from None
