"""Research context, result provenance and standalone evidence exports."""

import json
import uuid
from typing import Literal

import numpy as np
from fastapi import HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .platform import principal
from .store import cell_edges
from .workspace import report_html


class Region(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    record_id: str
    row_start: int = Field(ge=0)
    row_end: int = Field(ge=1)
    column_start: int = Field(ge=0)
    column_end: int = Field(ge=2)


def checked_profile_run(workspace, run_id, region, row, sha, width):
    try:
        selected = workspace.get("run", run_id)
    except KeyError:
        raise HTTPException(404, "Selected run not found") from None
    except PermissionError:
        raise
    except (OSError, ValueError):
        raise HTTPException(409, "Selected result is unavailable or corrupt") from None
    # PermissionError remains the existing project-access 403.
    try:
        selected_region = Region.model_validate(selected.get("region"))
    except ValidationError:
        raise HTTPException(409, "Selected run has invalid coordinates") from None
    if not (
        selected.get("source_sha256") == sha
        and selected.get("status") in ("complete", "imported")
        and selected_region.record_id == region.record_id
        and selected_region.row_start == row
        and selected_region.row_end == row + 1
        and selected_region.column_start <= region.column_start
        and region.column_end <= selected_region.column_end <= width
    ):
        raise HTTPException(409, "Selected run is incompatible with this profile")
    result = selected.get("result")
    denoised = result.get("denoised") if isinstance(result, dict) else None
    if not (
        isinstance(denoised, list)
        and len(denoised) == width
        and all(
            v is None
            or (
                not isinstance(v, bool)
                and isinstance(v, (int, float))
                and np.isfinite(v)
            )
            for v in denoised
        )
    ):
        raise HTTPException(409, "Selected result has no finite full native profile")
    return selected, denoised


class Bookmark(Region):
    title: str = Field(min_length=1, max_length=150)
    notes: str = Field(default="", max_length=8000)
    proposal_id: str | None = None


class ResultImport(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal["dts-method-result-v1"]
    region: Region
    source_sha256: str = Field(pattern="^[0-9a-f]{64}$")
    method: str = Field(min_length=1, max_length=100)
    method_version: str = Field(min_length=1, max_length=100)
    parameters: dict
    calculated_at: str = Field(min_length=1, max_length=100)
    training_origin: Literal["real", "synthetic", "mixed", "untrained", "unknown"]
    limitations: str = Field(min_length=1, max_length=4000)
    result: dict


class PackageView(BaseModel):
    id: str
    title: str
    cards: int
    queue_sha256: str
    shortlist: list[str] = Field(default_factory=list)
    source_manifest_sha256: str | None = None
    verified_native_cells: int | None = None


class CandidateView(BaseModel):
    id: str
    package: str
    queue_sha256: str
    source_sha256: str
    region: Region
    timestamps: list[str]
    x_bounds_m: list[float]
    shortlist: bool
    label: str
    method_results: str
    proposal_bounds: dict | None = None


class ResultSummary(BaseModel):
    zones: list[dict] = Field(default_factory=list)
    zone_count: int | None = None
    processed_points: int | None = None


class RunView(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    status: str
    method: str
    method_version: str
    source_sha256: str
    region: Region
    limitations: str
    training_origin: str
    updated_at: float
    has_result: bool = False
    result_summary: ResultSummary | None = None
    error: str | None = None


class RunDetail(RunView):
    result: dict | None = None


class EvidenceView(BaseModel):
    model_config = ConfigDict(extra="allow")
    valid_fraction: float
    native_shape: list[int]
    temperature_c: dict[str, float | None]
    source_sha256: str
    time_kind: str
    gaps_over_300s: list[float]


class BookmarkView(Bookmark):
    id: str
    source_sha256: str
    physical_cause: Literal["unknown"]
    updated_at: float


class MethodView(BaseModel):
    id: str
    version: str
    training_origin: str
    availability: str
    capabilities: list[str]
    limitations: str


def install_workspace_routes(app, review_context):
    def bounds(region):
        record = app.state.store.get(region.record_id)
        if not (
            region.row_start < region.row_end <= record.raw_c.shape[0]
            and region.column_start < region.column_end <= record.raw_c.shape[1]
        ):
            raise HTTPException(422, "Region must be inside native axes")
        if (region.row_end - region.row_start) * (
            region.column_end - region.column_start
        ) > 1_000_000:
            raise HTTPException(413, "Region exceeds 1,000,000 native observations")
        return record

    def describe(region, profile=None):
        r = bounds(region)
        selected_row = region.row_start if profile is None else profile
        if not region.row_start <= selected_row < region.row_end:
            raise HTTPException(422, "Report profile must belong to selected region")
        a = r.raw_c[
            region.row_start : region.row_end, region.column_start : region.column_end
        ]
        q = r.invalid_reason[
            region.row_start : region.row_end, region.column_start : region.column_end
        ]
        valid = q == 0
        values = a[valid]
        rows = np.linspace(0, a.shape[0] - 1, min(a.shape[0], 64), dtype=int)
        cols = np.linspace(0, a.shape[1] - 1, min(a.shape[1], 128), dtype=int)
        preview_valid = valid[np.ix_(rows, cols)]
        times = r.elapsed_s[region.row_start : region.row_end]
        result = {
            "schema_version": "dts-region-evidence-v1",
            "region": region.model_dump(),
            "source_sha256": r.manifest["source_sha256"],
            "source": app.state.store.meta[region.record_id]["source"],
            "time_kind": r.manifest["time_kind"],
            "scale": app.state.store.meta[region.record_id]["scale"],
            "timestamps": [
                r.profiles[region.row_start]["timestamp"],
                r.profiles[region.row_end - 1]["timestamp"],
            ],
            "x_bounds_m": [
                float(r.x_m[region.column_start]),
                float(r.x_m[region.column_end - 1]),
            ],
            "time_bounds_s": [float(times[0]), float(times[-1])],
            "valid_fraction": float(valid.mean()),
            "native_shape": list(a.shape),
            "temperature_c": {
                "min": float(values.min()) if values.size else None,
                "median": float(np.median(values)) if values.size else None,
                "max": float(values.max()) if values.size else None,
            },
            "mask_counts": {
                str(int(code)): int(np.count_nonzero(q == code))
                for code in np.unique(q)
            },
            "gaps_over_300s": [float(v) for v in np.diff(times) if v > 300],
            "preview": np.where(preview_valid, a[np.ix_(rows, cols)], 0).tolist(),
            "preview_valid": preview_valid.tolist(),
            "preview_time_s": times[rows].tolist(),
            "preview_x_m": r.x_m[region.column_start : region.column_end][
                cols
            ].tolist(),
            "preview_aggregation": "index_subsample; not an event boundary or a complete field",
            "physical_cause": "unknown",
            "expert_answer": None,
        }
        result["preview_time_cells"] = np.asarray(cell_edges(r.elapsed_s, 300))[
            region.row_start : region.row_end
        ][rows].tolist()
        result["preview_x_cells"] = np.asarray(cell_edges(r.x_m))[
            region.column_start : region.column_end
        ][cols].tolist()
        result["profile"] = {
            "row": selected_row,
            "x_m": r.x_m[region.column_start : region.column_end].tolist(),
            "temperature_c": [
                float(v) if flag == 0 else None
                for v, flag in zip(
                    a[selected_row - region.row_start],
                    q[selected_row - region.row_start],
                )
            ],
        }
        return result

    from .cases import install_cases

    install_cases(app, describe)

    @app.get("/api/v1/packages", response_model=list[PackageView])
    def packages() -> list[dict]:
        from .review_packages import package_views
        return package_views(app)

    @app.get("/api/v1/candidates", response_model=list[CandidateView])
    def candidates(package: str = "general", shortlist: bool = False) -> list[dict]:
        from .review_packages import resolve_queue
        q = resolve_queue(app,package)
        return [
            {
                "id": card["review_id"],
                "package": package,
                "queue_sha256": q.sha,
                "source_sha256": card["evidence"]["source_sha256"],
                "region": {
                    "record_id": card["source"].removesuffix(".csv"),
                    "row_start": card["row_bounds"][0],
                    "row_end": card["row_bounds"][1],
                    "column_start": card["column_bounds"][0],
                    "column_end": card["column_bounds"][1],
                },
                "timestamps": card["timestamps"],
                "x_bounds_m": card["x_bounds_m"],
                "shortlist": card.get("shortlist", False),
                "label": "Research window; not a confirmed event",
                "method_results": "offline hypotheses",
                "proposal_bounds": card.get("proposal_bounds"),
            }
            for card in q.cards.values()
            if not shortlist or card.get("shortlist", False)
        ]

    @app.post("/api/v1/regions/evidence", response_model=EvidenceView)
    def evidence(body: Region) -> dict:
        return describe(body)

    @app.get("/api/v1/bookmarks", response_model=list[BookmarkView])
    def bookmarks() -> list[dict]:
        return app.state.workspace.list("bookmark")

    @app.post("/api/v1/bookmarks", status_code=201, response_model=BookmarkView)
    def bookmark(body: Bookmark) -> dict:
        record = bounds(body)
        user = principal.get()
        key = uuid.uuid4().hex
        return app.state.workspace.put(
            "bookmark",
            key,
            {
                "id": key,
                **({"account_id": user["id"]} if user else {}),
                **body.model_dump(),
                "source_sha256": record.manifest["source_sha256"],
                "physical_cause": "unknown",
            },
        )

    @app.delete("/api/v1/bookmarks/{key}")
    def delete_bookmark(key: str) -> dict:
        app.state.workspace.delete("bookmark", key)
        return {"deleted": key}

    @app.get("/api/v1/runs", response_model=list[RunView])
    def runs(
        limit: int = Query(50, ge=1, le=100), offset: int = Query(0, ge=0)
    ) -> list[dict]:
        response = []
        items = (
            app.state.workspace.list("run", limit=limit, offset=offset)
            if app.state.platform
            else app.state.workspace.list("run")[offset : offset + limit]
        )
        for item in items:
            result = item.pop("result", None)
            item["has_result"] = result is not None or "result_ref" in item
            if isinstance(result, dict):
                zones = result.get("zones", [])
                item["result_summary"] = {
                    "zones": zones[:100] if isinstance(zones, list) else [],
                    "zone_count": len(zones)
                    if "zones" in result and isinstance(zones, list)
                    else None,
                    "processed_points": len(result["denoised"])
                    if isinstance(result.get("denoised"), list)
                    else None,
                }
            response.append(item)
        return response

    @app.get("/api/v1/runs/{key}", response_model=RunDetail)
    def run_result(key: str) -> dict:
        return app.state.workspace.get("run", key)

    @app.get("/api/v1/methods", response_model=list[MethodView])
    def methods() -> list[dict]:
        registry = [
            {
                "id": "median_x3",
                "training_origin": "untrained",
                "version": "ml-api-v0.1",
                "capabilities": ["profile"],
                "availability": "requires ML API",
                "limitations": "Denoising is not event classification",
            },
            {
                "id": "reference_detector",
                "training_origin": "untrained",
                "version": "ml-api-v0.1",
                "capabilities": ["bounded_field"],
                "availability": "requires ML API",
                "limitations": "Reference hypotheses; real accuracy unknown",
            },
        ]
        seen = {m["id"] + ":" + m["version"] for m in registry}
        for run in app.state.workspace.list("run"):
            key = run["method"] + ":" + run["method_version"]
            if key not in seen:
                registry.append(
                    {
                        "id": run["method"],
                        "version": run["method_version"],
                        "training_origin": run["training_origin"],
                        "availability": "prepared result; metadata declared by its author"
                        if run["status"] == "imported"
                        else run["status"],
                        "capabilities": ["prepared_result"],
                        "limitations": run["limitations"],
                    }
                )
                seen.add(key)
        return registry

    @app.post("/api/v1/results", status_code=201)
    def import_result(body: ResultImport) -> dict:
        record = bounds(body.region)
        if body.source_sha256 != record.manifest["source_sha256"]:
            raise HTTPException(422, "Result source SHA differs from native record")
        try:
            json.dumps(body.result, allow_nan=False)
        except ValueError as exc:
            raise HTTPException(422, "Nonfinite result values") from exc
        if "denoised" in body.result:
            values = body.result["denoised"]
            if (
                not isinstance(values, list)
                or body.region.row_end != body.region.row_start + 1
                or len(values) != record.raw_c.shape[1]
                or any(
                    v is not None
                    and (
                        isinstance(v, bool)
                        or not isinstance(v, (int, float))
                        or not np.isfinite(v)
                    )
                    for v in values
                )
            ):
                raise HTTPException(
                    422, "denoised must be a finite full native profile"
                )
        key = uuid.uuid4().hex
        return app.state.workspace.put(
            "run", key, {"id": key, "status": "imported", **body.model_dump()}
        )

    @app.post("/api/v1/reports")
    def report(
        body: Region,
        format: Literal["json", "html"] = "json",
        profile: int | None = Query(None, ge=0),
        package: str = "general",
        review_id: str = "",
        reviewer: str = "",
        method: str | None = Query(None, max_length=100),
        run_id: str | None = Query(None, min_length=1, max_length=128),
    ):
        content = describe(body, profile)
        # A method name is not an immutable result identity. No id means raw-only.
        content["runs"] = []
        content["profile_runs"] = []
        if run_id is not None:
            selected, denoised = checked_profile_run(
                app.state.workspace,
                run_id,
                body,
                content["profile"]["row"],
                content["source_sha256"],
                bounds(body).raw_c.shape[1],
            )
            content["profile"]["processed_temperature_c"] = [
                v if raw is not None else None
                for v, raw in zip(
                    denoised[body.column_start : body.column_end],
                    content["profile"]["temperature_c"],
                )
            ]
            content["profile"]["processing_run_id"] = run_id
            metadata = {
                k: v for k, v in selected.items() if k not in ("result", "result_ref")
            }
            content["profile_runs"] = [metadata]
            content["runs"] = [metadata]
        q, journal = (
            review_context(package, reviewer, True) if review_id else (None, None)
        )
        if q and journal and review_id in q.cards:
            card = q.cards[review_id]
            original = {
                "record_id": card["source"].removesuffix(".csv"),
                "row_start": card["row_bounds"][0],
                "row_end": card["row_bounds"][1],
                "column_start": card["column_bounds"][0],
                "column_end": card["column_bounds"][1],
            }
            if original == body.model_dump():
                content["expert_answer"] = journal.annotations().get(review_id)
        if format == "html":
            return Response(
                report_html(content),
                media_type="text/html",
                headers={
                    "Content-Disposition": 'attachment; filename="dts-evidence.html"'
                },
            )
        return Response(
            json.dumps(content, ensure_ascii=False, allow_nan=False),
            media_type="application/json",
            headers={"Content-Disposition": 'attachment; filename="dts-evidence.json"'},
        )

    from .demo_routes import install_demo_routes

    install_demo_routes(app, describe, bounds)
