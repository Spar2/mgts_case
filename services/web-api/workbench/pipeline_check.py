"""Compile-only syntax check for pipeline stage sources. Never executes them."""

from typing import Literal

from fastapi import HTTPException
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .platform import principal

STAGE_ORDER = ("preprocess", "clean", "anomaly", "classify", "vision")
StageId = Literal["preprocess", "clean", "anomaly", "classify", "vision"]


class PipelineStage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: StageId
    source: str = Field(max_length=20000)


class PipelineCheckRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    stages: list[PipelineStage] = Field(max_length=5)

    @field_validator("stages")
    @classmethod
    def ids_at_most_once(cls, stages):
        ids = [stage.id for stage in stages]
        if len(ids) != len(set(ids)):
            raise ValueError("Each stage id at most once")
        return stages


class PipelineCheckOk(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: Literal[True]


class PipelineCheckFailure(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: Literal[False]
    stage_id: StageId
    line: int
    message: str


def compile_stages(stages: list[PipelineStage]) -> PipelineCheckOk | PipelineCheckFailure:
    sources = {stage.id: stage.source for stage in stages}
    for stage_id in STAGE_ORDER:
        source = sources.get(stage_id)
        if source is None or not source.strip():
            continue
        try:
            compile(source, "<pipeline>", "exec")
        except SyntaxError as exc:
            return PipelineCheckFailure(
                ok=False,
                stage_id=stage_id,
                line=1 if exc.lineno is None else exc.lineno,
                message=exc.msg if isinstance(exc.msg, str) else "invalid syntax",
            )
    return PipelineCheckOk(ok=True)


def install_pipeline_check(app):
    @app.exception_handler(RequestValidationError)
    async def hide_pipeline_source(request, exc):
        if request.url.path != "/api/v2/pipeline-check":
            return await request_validation_exception_handler(request, exc)
        # Drop input and ctx so a rejected body cannot echo stage source.
        return JSONResponse(
            {
                "detail": [
                    {
                        "type": item["type"],
                        "loc": list(item["loc"]),
                        "msg": item["msg"],
                    }
                    for item in exc.errors()
                ]
            },
            status_code=422,
        )

    @app.post("/api/v2/pipeline-check")
    def pipeline_check(body: PipelineCheckRequest) -> PipelineCheckOk | PipelineCheckFailure:
        user = principal.get()
        if not user:
            raise HTTPException(401, "Sign in first")
        return compile_stages(body.stages)
