"""Managed CPU compatibility API; exact pinned DSP, no training or CSV access."""

import sys
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .analysis_compute import ML_COMMIT, PIN, verify_pin

verify_pin()
sys.path.insert(0, str(PIN))
from fiber.denoise import apply_method, residual
from fiber.grid import validate_axis
from fiber.quality import quality_mask, SENTINEL_MAX

app = FastAPI(title="DTS pinned CPU compatibility", version="0.1.0")
METHODS = ("identity", "median_x3", "median_x5")


class Profile(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    method: Literal["identity", "median_x3", "median_x5"] = "median_x3"
    raw: list[float] = Field(min_length=1, max_length=20000)
    distances: list[float] = Field(min_length=1, max_length=20000)


class FieldInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    raw: list[list[float | None]] = Field(min_length=1, max_length=10000)
    distances: list[float] = Field(min_length=2, max_length=20000)
    time_s: list[float] = Field(min_length=1, max_length=10000)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "haklolarp-ml",
        "implementation_sha": ML_COMMIT,
        "device": "cpu",
        "training_origin": "untrained",
        "denoise_methods": list(METHODS),
        "sentinel_max": SENTINEL_MAX,
    }


@app.get("/methods")
def methods():
    return {
        name: {
            "family": "standard",
            "implementation_sha": ML_COMMIT,
            "training_origin": "untrained",
        }
        for name in METHODS
    }


@app.post("/denoise/profile")
def denoise(body: Profile):
    if len(body.raw) != len(body.distances):
        raise HTTPException(422, "Native axis and profile differ")
    try:
        validate_axis(body.distances, "distances")
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    raw = np.asarray(body.raw, dtype=np.float32)[None, :]
    valid = quality_mask(raw)["valid"]
    clean = apply_method(body.method, raw, valid)
    noise = residual(raw, clean, valid)
    return {
        "method": body.method,
        "implementation_sha": ML_COMMIT,
        "training_origin": "untrained",
        "n_valid": int(valid.sum()),
        "denoised": [float(v) if np.isfinite(v) else None for v in clean[0]],
        "residual": [float(v) if np.isfinite(v) else None for v in noise[0]],
    }


@app.post("/detect/field")
def detect(body: FieldInput):
    from fiber.pipeline import analyze_field

    span = max(body.distances) - min(body.distances)
    if (
        sum(map(len, body.raw)) > 1_000_000
        or span > 200_000
        or len(body.time_s) * (span / 2 + 1) > 4_000_000
    ):
        raise HTTPException(413, "Reduce native/halo window")
    try:
        result = analyze_field(body.raw, body.distances, body.time_s)
    except (TypeError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from None
    return {
        k: v
        for k, v in result.items()
        if k not in {"clean", "z", "background", "scale", "score_available"}
    }
