"""Read-only, pinned research evidence. No thresholds are applied to runtime."""

import hashlib
from functools import lru_cache
from pathlib import Path

from fastapi import HTTPException
from pydantic import BaseModel

from .platform import principal

PACK_SHA256 = "cd8daf572ceb4859094f720bef0b0bbb740649a774b5b27fe2c9baea29b3d5f8"
PACK = Path(__file__).resolve().parents[1] / "research/ml-237168f/experiments.json"


class ExperimentSource(BaseModel):
    path: str
    sha256: str


class ExperimentAudit(BaseModel):
    baseline_cpu_reproduced: bool
    neural_cuda_reproduced: bool
    refinement_operational: bool
    witness_origin: str
    limitations: list[str]


class ExperimentView(BaseModel):
    schema_version: str
    source_commit: str
    sources: list[ExperimentSource]
    operational_pair: dict
    calibration: dict
    hybrid: dict
    audit: ExperimentAudit


@lru_cache(maxsize=1)
def read_pack():
    if PACK.stat().st_size > 1024 * 1024:
        raise ValueError("Research package exceeds budget")
    data = PACK.read_bytes()
    if hashlib.sha256(data).hexdigest() != PACK_SHA256:
        raise ValueError("Research evidence integrity mismatch")
    return ExperimentView.model_validate_json(data)


def install_experiments(app):
    @app.get("/api/v2/ml-experiments", response_model=ExperimentView)
    def evidence():
        user = principal.get()
        if not user:
            raise HTTPException(401, "Sign in to read research evidence")
        if user["project_id"] != "local":
            raise HTTPException(403, "Research sources belong to local project")
        try:
            return read_pack()
        except (OSError, ValueError):
            raise HTTPException(
                503,
                "Research evidence is unavailable; operational analysis is unchanged",
            ) from None
