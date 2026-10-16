"""Bounded CPU denoise and reference detection API (port 8082)."""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fiber.denoise import METHODS, available_methods
from fiber.quality import SENTINEL_MAX

app = FastAPI(title="haklolarp-ml", version="0.1.0")


class DenoiseRequest(BaseModel):
    method: str = Field(default="median_x3")
    raw: list[float] = Field(min_length=1, max_length=20000)
    distances: list[float] = Field(min_length=1, max_length=20000)


class FieldRequest(BaseModel):
    raw: list[list[float | None]] = Field(min_length=1, max_length=10000)
    distances: list[float] = Field(min_length=2, max_length=20000)
    time_s: list[float] = Field(min_length=1, max_length=10000)


@app.post('/detect/field')
def detect_field(req: FieldRequest, model: str | None = None) -> dict:
    """Bounded CPU reference analysis; time_s is an elapsed/wall-clock axis.

    No timestamp timezone is inferred and no learned class probability is
    claimed. Dense fields stay off the response; zones carry hypotheses.

    Query `model` defaults to the historical fiber pipeline and the same
    response shape. `ens_gen2` and `larpathon` return a dts-events/1 document.
    """
    chosen = (model or 'fiber').strip() or 'fiber'
    if sum(len(row) for row in req.raw) > 1_000_000:
        raise HTTPException(413, 'field exceeds 1,000,000 cells; use offline pipeline')
    if req.distances and req.time_s:
        span = max(req.distances) - min(req.distances)
        if span > 200_000 or len(req.time_s) * (span / 2 + 1) > 4_000_000:
            raise HTTPException(413, 'halo grid exceeds API resource budget; use offline pipeline')
    if chosen == 'fiber':
        from fiber.pipeline import analyze_field
        try:
            result = analyze_field(req.raw, req.distances, req.time_s)
        except (ValueError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from exc
        return {key: value for key, value in result.items()
                if key not in {'clean','z','background','scale','score_available'}}
    if chosen not in {'ens_gen2', 'larpathon'}:
        raise HTTPException(422, 'unknown model; expected fiber, ens_gen2, or larpathon')
    import numpy as np

    from ml_common.service import predict_field
    try:
        return predict_field(chosen, {
            'raw': np.asarray(req.raw, dtype=np.float32),
            'x_m': req.distances,
            'time_s': req.time_s,
            'dts_id': 'api',
            'channel_id': '1',
        })
    except FileNotFoundError as exc:
        raise HTTPException(503, f'weights unavailable: {exc}') from exc
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "haklolarp-ml",
        "port": int(os.environ.get("APP_PORT", "8082")),
        "sentinel_max": SENTINEL_MAX,
        "denoise_methods": available_methods(),
    }


@app.get("/methods")
def methods() -> dict:
    return {
        name: {"family": spec["family"], "doc": spec["doc"]}
        for name, spec in METHODS.items()
    }


@app.post("/denoise/profile")
def denoise_profile(req: DenoiseRequest) -> dict:
    import numpy as np

    from fiber.denoise import apply_method, residual
    from fiber.quality import quality_mask

    raw = np.asarray(req.raw, dtype=np.float32)[None, :]
    if raw.shape[1] != len(req.distances):
        raise HTTPException(422, 'raw and distances length mismatch')
    if req.method not in METHODS:
        raise HTTPException(422, 'unknown denoise method')
    from fiber.grid import validate_axis
    try:
        validate_axis(req.distances, 'distances')
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    valid = quality_mask(raw)["valid"]
    denoised = apply_method(req.method, raw, valid)
    noise = residual(raw, denoised, valid)
    return {
        "method": req.method,
        "n_valid": int(valid.sum()),
        "denoised": [None if not np.isfinite(v) else float(v) for v in denoised[0]],
        "residual": [None if not np.isfinite(v) else float(v) for v in noise[0]],
    }
