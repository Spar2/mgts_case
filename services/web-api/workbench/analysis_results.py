"""Native-grid analysis packages. Arrays are immutable, never hydrated into JSON."""

import hashlib
import html
import json
import os
import shutil
import uuid
from itertools import pairwise
from pathlib import Path

import numpy as np
from fastapi import HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field

from .platform import principal
from .store import cache_guard, cell_edges, packet
from .workspace_routes import Region

LAYERS = ("processed_temperature_c", "background_temperature_c", "score")
LIMIT = 512 * 1024 * 1024


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: str
    region: Region
    source_sha256: str = Field(pattern="^[0-9a-f]{64}$")
    method: str = Field(min_length=1, max_length=100)
    method_version: str = Field(min_length=1, max_length=100)
    parameters: dict
    calculated_at: str
    training_origin: str
    limitations: str = Field(min_length=1, max_length=4000)
    research_status: str
    score: dict | None = None
    files: dict[str, str]


class AnalysisRunView(BaseModel):
    id: str
    region: Region
    source_sha256: str
    method: str
    method_version: str
    parameters: dict = Field(default_factory=dict)
    calculated_at: str | None = None
    training_origin: str
    research_status: str
    limitations: str
    layers: list[str]
    zones: list[dict]
    score: dict | None = None
    scales: dict = Field(default_factory=dict)
    schema_version: str | None = None
    legacy_profile: bool = False
    time_kind: str = "unknown"
    timestamps: list[str] = Field(default_factory=list)


class AnalysisSlicesView(BaseModel):
    run_id: str
    source_sha256: str
    x_m: list[float]
    time_s: list[float]
    native_time_s: list[float]
    row: int
    column: int
    layers: dict[str, dict]
    source_invalid_reason_profile: list[int]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(root, store):
    root = Path(root).resolve()
    if (root / "manifest.json").stat().st_size > 256 * 1024:
        raise ValueError("Manifest exceeds 256 KiB")
    m = Manifest.model_validate_json((root / "manifest.json").read_bytes())
    if m.schema_version != "dts-analysis-result-v2":
        raise ValueError("Unsupported analysis schema")
    if m.training_origin not in {"real", "synthetic", "mixed", "untrained", "unknown"}:
        raise ValueError("Unknown training origin")
    if m.research_status not in {"reference", "research", "candidate", "not_accepted"}:
        raise ValueError("Unknown research status")
    allowed = {"x_m.npy", "time_s.npy", "zones.json"}
    allowed |= {name + suffix for name in LAYERS for suffix in (".npy", "_valid.npy")}
    if not {"x_m.npy", "time_s.npy"} <= m.files.keys() or not m.files.keys() <= allowed:
        raise ValueError("Unknown or missing package files")
    total = 0
    for name, checksum in m.files.items():
        path = root / name
        if path.is_symlink() or not path.is_file() or path.parent.resolve() != root:
            raise ValueError("Package files must be regular local files")
        total += path.stat().st_size
        if total > LIMIT or sha(path) != checksum:
            raise ValueError("Package size or SHA mismatch")
    r = m.region
    record = store.get(r.record_id)
    rs, re, cs, ce = r.row_start, r.row_end, r.column_start, r.column_end
    if record.manifest["source_sha256"] != m.source_sha256:
        raise ValueError("Source SHA mismatch")
    if not (
        0 <= rs < re <= record.raw_c.shape[0] and 0 <= cs < ce <= record.raw_c.shape[1]
    ):
        raise ValueError("Region outside native axes")
    shape = (re - rs, ce - cs)
    arrays = {}
    for name in m.files:
        if name.endswith(".npy"):
            a = np.load(root / name, mmap_mode="r", allow_pickle=False)
            if a.dtype.kind not in "fb" or a.dtype.hasobject:
                raise ValueError("Only float and boolean arrays are allowed")
            arrays[name] = a
    for name, native in (
        ("x_m.npy", record.x_m[cs:ce]),
        ("time_s.npy", record.elapsed_s[rs:re]),
    ):
        a = arrays[name]
        if a.dtype != np.dtype("float64") or not np.array_equal(a, native):
            raise ValueError("Axes must exactly match native Float64 coordinates")
    present = []
    for name in LAYERS:
        values, mask = name + ".npy", name + "_valid.npy"
        if (values in arrays) != (mask in arrays):
            raise ValueError("Each layer requires its availability mask")
        if values not in arrays:
            continue
        a, valid = arrays[values], arrays[mask]
        if (
            a.shape != shape
            or valid.shape != shape
            or a.dtype not in (np.dtype("float32"), np.dtype("float64"))
            or valid.dtype != np.dtype("bool")
        ):
            raise ValueError("Layer dtype or shape differs from native region")
        for start in range(0, shape[0], 256):
            block, good = a[start : start + 256], valid[start : start + 256]
            if np.isinf(block).any() or not np.isfinite(block[good]).all():
                raise ValueError("Infinity or unavailable finite values")
        present.append(name)
    if "score" in present:
        if (
            not m.score
            or not all(m.score.get(k) for k in ("name", "unit", "meaning"))
            or m.score.get("kind") not in {"signed", "nonnegative"}
        ):
            raise ValueError("Score requires name, unit, meaning and kind")
        threshold = m.score.get("threshold")
        if threshold is not None and (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not np.isfinite(threshold)
            or threshold < 0
            or m.score.get("comparison") not in {"ge", "abs_ge"}
        ):
            raise ValueError("Invalid score threshold")
        if m.score["kind"] == "nonnegative" and np.any(
            arrays["score.npy"][arrays["score_valid.npy"]] < 0
        ):
            raise ValueError("Negative nonnegative score")
    zones = (
        json.loads((root / "zones.json").read_bytes())
        if "zones.json" in m.files
        else []
    )
    if not isinstance(zones, list) or len(zones) > 10000:
        raise ValueError("Invalid zones")
    ids = set()
    for zone in zones:
        z = Region.model_validate(
            {
                "record_id": r.record_id,
                **{
                    k: zone[k]
                    for k in ("row_start", "row_end", "column_start", "column_end")
                },
            }
        )
        if (
            not isinstance(zone.get("id"), str)
            or zone["id"] in ids
            or not (
                rs <= z.row_start < z.row_end <= re
                and cs <= z.column_start < z.column_end <= ce
            )
        ):
            raise ValueError("Zone outside result or duplicate ID")
        ids.add(zone["id"])
    if not present and not zones:
        raise ValueError("Package has no result")
    return m, present, zones


def import_package(platform, store, root, project, author, publish=None):
    m, layers, zones = validate(root, store)
    identity = hashlib.sha256(
        json.dumps(
            {"manifest": m.model_dump(), "project": project, "author": author},
            sort_keys=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    target = platform.blobs.root / "arrays" / identity
    target.parent.mkdir(exist_ok=True)
    with platform.connect() as db:
        db.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s,0))",
            ("analysis:" + identity,),
        )
        exists = db.execute(
            "SELECT payload FROM documents WHERE kind='analysis_run' AND id=%s",
            (identity,),
        ).fetchone()
        if exists:
            for name, checksum in exists[0]["files"].items():
                if sha(target / name) != checksum:
                    raise ValueError("Previously imported artifact is corrupt")
            if publish is not None:
                publish(db, exists[0])
            return exists[0]
        if not db.execute(
            "SELECT 1 FROM memberships WHERE project_id=%s AND user_id=%s",
            (project, author),
        ).fetchone():
            raise ValueError("Author must belong to project")
        if project != "local":
            raise ValueError("Native archive belongs to local project")
        created_target = False
        temporary = target.parent / ("tmp-" + uuid.uuid4().hex)
        temporary.mkdir()
        try:
            for name, checksum in m.files.items():
                shutil.copyfile(Path(root) / name, temporary / name)
                if sha(temporary / name) != checksum:
                    raise ValueError("Package changed during import")
            if not target.exists():
                os.rename(temporary, target)
                created_target = True
            else:
                for name, checksum in m.files.items():
                    if sha(target / name) != checksum:
                        raise ValueError("Existing analysis artifact differs")
        finally:
            if temporary.exists():
                for file in temporary.iterdir():
                    file.unlink()
                temporary.rmdir()
        scales = {}
        r = m.region
        native = store.get(r.record_id)
        good = (
            native.invalid_reason[
                r.row_start : r.row_end, r.column_start : r.column_end
            ]
            == 0
        )
        for name in layers:
            a = np.load(target / (name + ".npy"), mmap_mode="r", allow_pickle=False)
            flags_source = np.load(
                target / (name + "_valid.npy"), mmap_mode="r", allow_pickle=False
            )
            flags = flags_source & good
            flags_source._mmap.close()
            maximum = max(float(np.max(np.abs(a[flags]))) if flags.any() else 0, 0.01)
            scales[name] = [
                0 if name == "score" and m.score["kind"] == "nonnegative" else -maximum,
                maximum,
            ]
            if name == "processed_temperature_c":
                raw = native.raw_c[
                    r.row_start : r.row_end, r.column_start : r.column_end
                ]
                maximum = max(
                    float(np.max(np.abs(a[flags].astype(np.float64) - raw[flags])))
                    if flags.any()
                    else 0,
                    0.01,
                )
                scales["difference"] = [-maximum, maximum]
            a._mmap.close()
        value = {
            **m.model_dump(),
            "id": identity,
            "project_id": project,
            "account_id": author,
            "layers": layers,
            "zones": zones,
            "scales": scales,
            "status": "imported",
            "time_kind": native.manifest["time_kind"],
            "timestamps": [
                native.profiles[r.row_start]["timestamp"],
                native.profiles[r.row_end - 1]["timestamp"],
            ],
        }
        from psycopg.types.json import Jsonb

        db.execute(
            "INSERT INTO documents VALUES('analysis_run',%s,%s,extract(epoch from now()))",
            (identity, Jsonb(value)),
        )
        if publish is not None:
            try:
                publish(db, value)
            except Exception:  # Atomic publication failed: remove only our new files under the held import lock.
                if created_target:
                    if (
                        target.resolve().parent
                        != (platform.blobs.root / "arrays").resolve()
                    ):
                        raise ValueError("Artifact cleanup path differs")
                    for name in m.files:
                        (target / name).unlink()
                    target.rmdir()
                raise
    return value


def load(app, key):
    if not principal.get() or not app.state.platform:
        raise HTTPException(401, "Sign in to read analysis results")
    if len(key) != 64:
        run = app.state.workspace.get("run", key)
        values = run.get("result", {}).get("denoised")
        r = Region.model_validate(run["region"])
        native = app.state.store.get(r.record_id)
        if (
            run.get("status") not in {"complete", "imported"}
            or r.row_end != r.row_start + 1
            or not isinstance(values, list)
            or len(values) != native.raw_c.shape[1]
        ):
            raise HTTPException(409, "Legacy run has no full native profile")
        return {
            **run,
            "layers": ["processed_temperature_c"],
            "zones": [],
            "research_status": "reference"
            if run.get("training_origin") == "untrained"
            else "research",
            "legacy_profile": True,
            "time_kind": native.manifest["time_kind"],
            "timestamps": [native.profiles[r.row_start]["timestamp"]],
        }, None
    run = app.state.platform.get("analysis_run", key)
    native = app.state.store.get(run["region"]["record_id"])
    if native.manifest["source_sha256"] != run["source_sha256"]:
        raise HTTPException(409, "Analysis source SHA changed")
    root = app.state.platform.blobs.root / "arrays" / run["id"]
    verified = getattr(app.state, "analysis_verified", {})
    try:
        fingerprint = [
            (name, (root / name).stat().st_mtime_ns, (root / name).stat().st_size)
            for name in run["files"]
        ]
    except OSError:
        raise HTTPException(409, "Analysis artifact is unavailable") from None
    if verified.get(key) != fingerprint:
        for name, checksum in run["files"].items():
            if sha(root / name) != checksum:
                raise HTTPException(409, "Analysis artifact integrity mismatch")
        verified[key] = fingerprint
        app.state.analysis_verified = verified
    return run, root


def snapshot_analysis(app, key, region, row, column=None, zone_id=None):
    run, root = load(app, key)
    r = Region.model_validate(run["region"])
    if not (
        r.record_id == region.record_id
        and r.row_start <= region.row_start <= row < region.row_end <= r.row_end
        and r.column_start <= region.column_start < region.column_end <= r.column_end
    ):
        raise HTTPException(409, "Analysis result does not cover the case")
    rs, re = region.row_start - r.row_start, region.row_end - r.row_start
    cs, ce = region.column_start - r.column_start, region.column_end - r.column_start
    column = region.column_start if column is None else column
    if not region.column_start <= column < region.column_end:
        raise HTTPException(409, "Selected point is outside case")
    slices = app.state.analysis_slices(key, row, column, zone_id)
    native = app.state.store.get(region.record_id)
    t0, t1 = native.elapsed_s[region.row_start], native.elapsed_s[region.row_end - 1]
    temporal_indices = [i for i, t in enumerate(slices["time_s"]) if t0 <= t <= t1]
    rows = np.linspace(rs, re - 1, min(re - rs, 64), dtype=int)
    cols = np.linspace(cs, ce - 1, min(ce - cs, 128), dtype=int)
    saved = {
        "schema_version": "dts-analysis-snapshot-v1",
        "run_id": key,
        "region": region.model_dump(),
        "row": row,
        "column": column,
        "zone_id": zone_id,
        "time_s": [slices["time_s"][i] for i in temporal_indices],
        "distance_m": float(native.x_m[column]),
        "source_invalid_reason_profile": native.invalid_reason[
            row, region.column_start : region.column_end
        ].tolist(),
        "raw_trend": [slices["layers"]["raw"]["trend"][i] for i in temporal_indices],
        "raw_coverage": [
            slices["layers"]["raw"]["coverage"][i] for i in temporal_indices
        ],
        "aggregation": "index_subsample_native_not_complete_field",
        "preview_rows": (rows + r.row_start).tolist(),
        "preview_columns": (cols + r.column_start).tolist(),
        "score": run.get("score"),
        "scales": run.get("scales", {}),
        "layers": {},
    }
    for name in [
        *run["layers"],
        *(["difference"] if "processed_temperature_c" in run["layers"] else []),
    ]:
        a, good, _, _ = layer_arrays(app, run, root, name)
        values, flags = a[np.ix_(rows, cols)], good[np.ix_(rows, cols)]
        saved["layers"][name] = {
            "profile": nullable(
                a[row - r.row_start, cs:ce], good[row - r.row_start, cs:ce]
            ),
            "preview": np.where(flags, values, 0).tolist(),
            "preview_valid": flags.tolist(),
            "trend": [slices["layers"][name]["trend"][i] for i in temporal_indices],
            "coverage": [
                slices["layers"][name]["coverage"][i] for i in temporal_indices
            ],
        }
        if "zone" in slices["layers"][name]:
            saved["layers"][name]["zone"] = {
                k: [v[i] for i in temporal_indices]
                for k, v in slices["layers"][name]["zone"].items()
                if isinstance(v, list)
            }
    return {**run, "result": saved}, saved


def layer_arrays(app, run, root, layer):
    r = Region.model_validate(run["region"])
    native = app.state.store.get(r.record_id)
    if native.manifest["source_sha256"] != run["source_sha256"]:
        raise HTTPException(409, "Source SHA changed")
    raw = native.raw_c[r.row_start : r.row_end, r.column_start : r.column_end]
    good = (
        native.invalid_reason[r.row_start : r.row_end, r.column_start : r.column_end]
        == 0
    )
    if layer == "raw":
        return raw, good, native, r
    name = "processed_temperature_c" if layer == "difference" else layer
    if name not in run["layers"]:
        raise HTTPException(409, "Requested layer is unavailable")
    if run.get("legacy_profile"):
        values = run["result"]["denoised"][r.column_start : r.column_end]
        a = np.asarray(
            [np.nan if value is None else value for value in values], dtype=np.float64
        )[None, :]
        valid = good & np.isfinite(a)
    else:
        a = np.load(root / (name + ".npy"), mmap_mode="r", allow_pickle=False)
        valid = good & np.load(
            root / (name + "_valid.npy"), mmap_mode="r", allow_pickle=False
        )
    if layer == "difference":
        a = np.asarray(a, dtype=np.float64) - raw
    return a, valid, native, r


def nullable(a, valid):
    return [float(v) if flag else None for v, flag in zip(a, valid, strict=True)]


def analysis_html(evidence):
    saved = evidence.get("analysis")
    if not saved:
        return ""
    sections = [
        "<h2>Сохранённые слои расчёта</h2><p>Обзор — подвыборка исходных индексов, не полное поле. Пропуски сохранены. Score не является вероятностью аварии.</p>"
    ]
    xcells, tcells = evidence["preview_x_cells"], evidence["preview_time_cells"]
    xmin, xmax = min(c[0] for c in xcells), max(c[1] for c in xcells)
    tmin, tmax = min(c[0] for c in tcells), max(c[1] for c in tcells)
    for name, item in saved["layers"].items():
        unit = saved.get("score", {}).get("unit", "score") if name == "score" else "°C"
        scale = (
            saved.get("scales", {}).get(name, evidence["scale"])
            if name in {"score", "difference"}
            else evidence["scale"]
        )
        lo, hi = scale
        cells = []
        for iy, row in enumerate(item["preview"]):
            for ix, value in enumerate(row):
                if not item["preview_valid"][iy][ix]:
                    continue
                fraction = float(np.clip((value - lo) / max(hi - lo, 0.01), 0, 1))
                color = f"rgb({int(30 + 210 * fraction)},{int(50 + 170 * fraction)},{int(100 - 70 * fraction)})"
                x0, x1 = xcells[ix]
                t0, t1 = tcells[iy]
                cells.append(
                    f'<rect x="{(x0 - xmin) / max(xmax - xmin, 0.001) * 1000:.3f}" y="{(t0 - tmin) / max(tmax - tmin, 0.001) * 320:.3f}" width="{(x1 - x0) / max(xmax - xmin, 0.001) * 1000:.3f}" height="{(t1 - t0) / max(tmax - tmin, 0.001) * 320:.3f}" fill="{color}"/>'
                )
        table = "".join(
            f"<tr><td>{x}</td><td>{value if value is not None else 'Нет данных'}</td></tr>"
            for x, value in zip(
                evidence["profile"]["x_m"], item["profile"], strict=True
            )
        )
        sections.append(
            f'<h3>{html.escape(name)} · {html.escape(unit)}</h3><p>Шкала {lo} → {hi} {html.escape(unit)}; x: расстояние, м; y: время от начала, с.</p><svg viewBox="0 0 1000 320" role="img" aria-label="{html.escape(name)}">{"".join(cells)}</svg><details><summary>Точный сохранённый профиль</summary><table><tr><th>Расстояние, м</th><th>{html.escape(unit)}</th></tr>{table}</table></details>'
        )
    return "".join(sections)


def install_analysis(app):
    @app.get("/api/v2/result-runs", response_model=list[AnalysisRunView])
    def catalog(
        record_id: str | None = None,
        method: str | None = None,
        method_version: str | None = None,
        research_status: str | None = None,
    ):
        if not principal.get() or not app.state.platform:
            raise HTTPException(401, "Sign in to read analysis results")
        items = app.state.platform.list("analysis_run")
        for legacy in app.state.workspace.list("run"):
            try:
                adapted, _ = load(app, legacy["id"])
            except (KeyError, ValueError, HTTPException):
                continue
            adapted.pop("result", None)
            items.append(adapted)
        return [
            v
            for v in items
            if (not record_id or v["region"]["record_id"] == record_id)
            and (not method or v["method"] == method)
            and (not method_version or v["method_version"] == method_version)
            and (not research_status or v["research_status"] == research_status)
        ]

    @app.get("/api/v2/result-runs/{key}", response_model=AnalysisRunView)
    def detail(key: str):
        return load(app, key)[0]

    @app.get("/api/v2/result-runs/{key}/zones")
    def physical_zones(key: str):
        run, _ = load(app, key)
        native = app.state.store.get(run["region"]["record_id"])
        xcells = np.asarray(cell_edges(native.x_m))
        tcells = np.asarray(cell_edges(native.elapsed_s, 300))
        result = []
        for zone in run["zones"]:
            rs, re, cs, ce = (
                zone[k] for k in ("row_start", "row_end", "column_start", "column_end")
            )
            # A proposal that spans a time discontinuity gets separate display boxes.
            breaks = np.flatnonzero(np.diff(native.elapsed_s[rs:re]) > 300) + rs + 1
            boundaries = [rs, *breaks.tolist(), re]
            for index, (start, end) in enumerate(pairwise(boundaries)):
                result.append(
                    {
                        "id": zone["id"],
                        "segment": index,
                        "x0": float(xcells[cs, 0]),
                        "x1": float(xcells[ce - 1, 1]),
                        "t0": float(tcells[start, 0]),
                        "t1": float(tcells[end - 1, 1]),
                        "label": zone.get("shape_hypothesis", "Гипотеза алгоритма"),
                        "source_sha256": run["source_sha256"],
                        "run_id": key,
                    }
                )
        return result

    @app.get("/api/v2/result-runs/{key}/external-reviews")
    def external_reviews(key: str):
        run, _ = load(app, key)
        r = run["region"]
        return [
            value
            for value in app.state.platform.list("external_review")
            if value["source_sha256"] == run["source_sha256"]
            and value["region"]["row_start"] < r["row_end"]
            and r["row_start"] < value["region"]["row_end"]
            and value["region"]["column_start"] < r["column_end"]
            and r["column_start"] < value["region"]["column_end"]
        ]

    @app.get("/api/v2/result-runs/{key}/slices", response_model=AnalysisSlicesView)
    def slices(
        key: str,
        row: int,
        column: int,
        zone_id: str | None = None,
        row_start: int | None = None,
        row_end: int | None = None,
        column_start: int | None = None,
        column_end: int | None = None,
    ):
        run, root = load(app, key)
        original = Region.model_validate(run["region"])
        r = Region(
            record_id=original.record_id,
            row_start=original.row_start if row_start is None else row_start,
            row_end=original.row_end if row_end is None else row_end,
            column_start=original.column_start
            if column_start is None
            else column_start,
            column_end=original.column_end if column_end is None else column_end,
        )
        if not (
            original.row_start <= r.row_start < r.row_end <= original.row_end
            and original.column_start
            <= r.column_start
            < r.column_end
            <= original.column_end
        ):
            raise HTTPException(422, "Slice region outside result")
        if not (
            r.row_start <= row < r.row_end and r.column_start <= column < r.column_end
        ):
            raise HTTPException(422, "Slice outside result coverage")
        record = app.state.store.get(r.record_id)
        i, j = row - r.row_start, column - r.column_start
        result = {
            "run_id": key,
            "source_sha256": run["source_sha256"],
            "x_m": record.x_m[r.column_start : r.column_end].tolist(),
            "time_s": record.elapsed_s[r.row_start : r.row_end].tolist(),
            "row": row,
            "column": column,
            "layers": {},
            "source_invalid_reason_profile": record.invalid_reason[
                row, r.column_start : r.column_end
            ].tolist(),
        }
        zone = next((z for z in run["zones"] if z["id"] == zone_id), None)
        if zone_id and zone is None:
            raise HTTPException(422, "Unknown zone")
        if zone and not (
            zone["row_start"] < r.row_end
            and r.row_start < zone["row_end"]
            and zone["column_start"] < r.column_end
            and r.column_start < zone["column_end"]
        ):
            raise HTTPException(422, "Zone does not intersect requested window")
        for name in [
            "raw",
            *run["layers"],
            *(["difference"] if "processed_temperature_c" in run["layers"] else []),
        ]:
            a, valid, _, _ = layer_arrays(app, run, root, name)
            rs, re = r.row_start - original.row_start, r.row_end - original.row_start
            cs, ce = (
                r.column_start - original.column_start,
                r.column_end - original.column_start,
            )
            a, valid = a[rs:re, cs:ce], valid[rs:re, cs:ce]
            item = {
                "profile": nullable(a[i], valid[i]),
                "trend": nullable(a[:, j], valid[:, j]),
                "coverage": valid.mean(axis=1).tolist(),
            }
            if zone:
                start, end = (
                    max(0, zone["column_start"] - r.column_start),
                    min(
                        r.column_end - r.column_start,
                        zone["column_end"] - r.column_start,
                    ),
                )
                selected = np.where(valid[:, start:end], a[:, start:end], np.nan)
                count = valid[:, start:end].sum(axis=1)
                count[
                    (np.arange(r.row_start, r.row_end) < zone["row_start"])
                    | (np.arange(r.row_start, r.row_end) >= zone["row_end"])
                ] = 0
                quantiles = np.full((3, len(selected)), np.nan)
                for index in np.flatnonzero(count):
                    quantiles[:, index] = np.nanpercentile(
                        selected[index], [10, 50, 90]
                    )
                item["zone"] = {
                    "p10": nullable(quantiles[0], count > 0),
                    "median": nullable(quantiles[1], count > 0),
                    "p90": nullable(quantiles[2], count > 0),
                    "valid_count": count.tolist(),
                    "aggregation": "spatial_percentile_valid_native",
                    "zone_id": zone_id,
                }
            result["layers"][name] = item
        gaps = np.flatnonzero(np.diff(result["time_s"]) > 300)
        result["native_time_s"] = list(result["time_s"])
        for index in reversed(gaps.tolist()):
            result["time_s"].insert(
                index + 1, (result["time_s"][index] + result["time_s"][index + 1]) / 2
            )
            for item in result["layers"].values():
                item["trend"].insert(index + 1, None)
                item["coverage"].insert(index + 1, None)
                if "zone" in item:
                    for name in ("p10", "median", "p90", "valid_count"):
                        item["zone"][name].insert(index + 1, None)
        return result

    app.state.analysis_slices = slices

    @app.get("/api/v2/result-runs/{key}/tiles")
    def tiles(
        key: str,
        layer: str = "raw",
        width: int = Query(256, ge=1, le=256),
        height: int = Query(256, ge=1, le=256),
        x0: float | None = None,
        x1: float | None = None,
        t0: float | None = None,
        t1: float | None = None,
    ):
        run, root = load(app, key)
        identity = json.dumps(
            [
                "analysis-tiles-v2",
                key,
                run.get("updated_at"),
                layer,
                width,
                height,
                x0,
                x1,
                t0,
                t1,
            ],
            allow_nan=False,
        )
        cache_key = hashlib.sha256(identity.encode()).hexdigest()
        cached = app.state.store.cache / (cache_key + ".bin")
        with cache_guard(app.state.store.cache):
            if cached.exists():
                return Response(
                    cached.read_bytes(), media_type="application/octet-stream"
                )
        a, valid, record, r = layer_arrays(app, run, root, layer)
        x, t = (
            record.x_m[r.column_start : r.column_end],
            record.elapsed_s[r.row_start : r.row_end],
        )
        xmin, xmax = (
            (float(x[0]) if x0 is None else x0),
            (float(x[-1]) if x1 is None else x1),
        )
        tmin, tmax = (
            (float(t[0]) if t0 is None else t0),
            (float(t[-1]) if t1 is None else t1),
        )
        if not all(np.isfinite([xmin, xmax, tmin, tmax])) or not (
            x[0] <= xmin <= xmax <= x[-1] and t[0] <= tmin <= tmax <= t[-1]
        ):
            raise HTTPException(422, "Tile outside result coverage")
        xi, ti = (
            np.flatnonzero((x >= xmin) & (x <= xmax)),
            np.flatnonzero((t >= tmin) & (t <= tmax)),
        )
        if not len(xi) or not len(ti):
            raise HTTPException(422, "No observations in requested region")
        native_mode = len(xi) <= width and len(ti) <= height
        scale = list(app.state.store.meta[r.record_id]["scale"])
        if layer in {"difference", "score"}:
            if run.get("legacy_profile"):
                maximum = max(
                    float(np.max(np.abs(a[valid]))) if valid.any() else 0, 0.01
                )
                scale = [-maximum, maximum]
            else:
                scale = run["scales"][layer]
        metadata = {
            "title": run["score"]["name"]
            if layer == "score"
            else {
                "raw": "Исходное температурное поле",
                "processed_temperature_c": "Обработанное температурное поле",
                "background_temperature_c": "Оценка фона",
                "difference": "Разность: обработка − исходник",
            }.get(layer, layer),
            "source_sha256": run["source_sha256"],
            "run_id": key,
            "layer": layer,
            "scale": scale,
            "unit": run["score"]["unit"] if layer == "score" else "°C",
            "palette": "diverging"
            if scale[0] < 0 and layer in {"difference", "score"}
            else "cividis",
        }
        if native_mode:
            values, flags = a[np.ix_(ti, xi)], valid[np.ix_(ti, xi)]
            metadata.update(
                x_cells=np.asarray(cell_edges(record.x_m))[
                    r.column_start + xi
                ].tolist(),
                time_cells=np.asarray(cell_edges(record.elapsed_s, 300))[
                    r.row_start + ti
                ].tolist(),
                aggregation="native_observations",
            )
            counts = flags.astype(np.uint32)
        else:
            xe = np.linspace(xmin, xmax, min(width, len(xi)) + 1)
            if xmin == xmax:
                edge = cell_edges(record.x_m)[r.column_start + xi[0]]
                xe = np.asarray(edge, dtype=np.float64)
            te = np.linspace(
                tmin, tmax if tmax > tmin else tmin + 1, min(height, len(ti)) + 1
            )
            values = np.full((len(te) - 1, len(xe) - 1), np.nan)
            counts = np.zeros(values.shape, dtype=np.uint32)
            gaps = np.flatnonzero(np.diff(t) > 300)
            for iy in range(values.shape[0]):
                if np.any((te[iy] < t[gaps + 1]) & (te[iy + 1] > t[gaps])):
                    continue
                rows = ti[
                    (t[ti] >= te[iy])
                    & (
                        t[ti] <= te[iy + 1]
                        if iy == values.shape[0] - 1
                        else t[ti] < te[iy + 1]
                    )
                ]
                if len(rows) > 1 and np.any(np.diff(t[rows]) > 300):
                    continue
                for ix in range(values.shape[1]):
                    cols = xi[
                        (x[xi] >= xe[ix])
                        & (
                            x[xi] <= xe[ix + 1]
                            if ix == values.shape[1] - 1
                            else x[xi] < xe[ix + 1]
                        )
                    ]
                    v, good = a[np.ix_(rows, cols)], valid[np.ix_(rows, cols)]
                    v = v[good]
                    counts[iy, ix] = len(v)
                    if len(v):
                        values[iy, ix] = (
                            v[np.argmax(np.abs(v))]
                            if layer == "score"
                            else np.median(v)
                        )
            flags = counts > 0
            metadata.update(
                x_edges=xe.tolist(),
                time_edges=te.tolist(),
                aggregation="signed_max_absolute_valid"
                if layer == "score"
                else "median_valid_native",
            )
        if native_mode:
            source_mask = record.invalid_reason[
                np.ix_(r.row_start + ti, r.column_start + xi)
            ]
            mask = np.where(source_mask != 0, source_mask, np.where(flags, 0, 8))
        else:
            mask = np.where(flags, 0, 8)
        payload = packet(values, mask, counts, **metadata)
        with cache_guard(app.state.store.cache):
            temporary = cached.with_suffix("." + uuid.uuid4().hex + ".tmp")
            temporary.write_bytes(payload)
            os.replace(temporary, cached)
            files = sorted(
                app.state.store.cache.glob("*.bin"), key=lambda p: p.stat().st_mtime
            )
            size = sum(p.stat().st_size for p in files)
            for path in files:
                if size <= 256 * 1024 * 1024:
                    break
                size -= path.stat().st_size
                path.unlink()
        return Response(payload, media_type="application/octet-stream")
