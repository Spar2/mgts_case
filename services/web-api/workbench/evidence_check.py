"""Offline consistency checks; not a signature or an assessment of model quality."""

import json
import math

import numpy as np

from .profile_changes import profile_changes


def _equal(actual, expected, name):
    # Python treats True as 1: reject booleans in numeric evidence explicitly.
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"{name}: shape differs")
        for a, e in zip(actual, expected):
            _equal(a, e, name)
    elif expected is None:
        if actual is not None:
            raise ValueError(f"{name}: missing value became a number")
    elif isinstance(expected, bool):
        if type(actual) is not bool or actual != expected:
            raise ValueError(f"{name}: mask differs")
    elif isinstance(actual, bool) or not isinstance(actual, (float, int)):
        raise ValueError(f"{name}: invalid number")
    elif not math.isfinite(actual) or actual != expected:
        raise ValueError(f"{name}: value differs")


def verify_evidence(report, native, run=None):
    """Check an ordinary research JSON against verified native arrays and a run."""
    if report.get("schema_version") != "dts-region-evidence-v1":
        raise ValueError("Only ordinary research evidence v1 is supported")
    region = report["region"]
    source = native.manifest
    if (
        report.get("source_sha256") != source["source_sha256"]
        or region["record_id"] != source["source_name"].removesuffix(".csv")
        or report.get("time_kind") != source["time_kind"]
    ):
        raise ValueError("Source identity or time interpretation differs")
    r0, r1 = region["row_start"], region["row_end"]
    c0, c1 = region["column_start"], region["column_end"]
    row = report["profile"]["row"]
    if any(type(v) is not int for v in (r0, r1, c0, c1, row)) or not (
        0 <= r0 <= row < r1 <= native.raw_c.shape[0]
        and 0 <= c0 < c1 <= native.raw_c.shape[1]
    ):
        raise ValueError("Evidence coordinates are outside native axes")
    view = report.get("analysis_view")
    if view is not None:
        layers = set(report.get("analysis", {}).get("layers", {})) | {"raw"}
        if report["profile"].get("processed_temperature_c") is not None:
            layers.update({"processed_temperature_c", "difference"})
        if (
            view.get("schema_version") != "dts-analysis-view-v1"
            or not run
            or view.get("run_id") != run.get("id")
            or view.get("source_sha256") != report["source_sha256"]
            or view.get("region") != region
            or type(view.get("row")) is not int
            or view["row"] != row
            or type(view.get("column")) is not int
            or not c0 <= view["column"] < c1
            or view.get("layer") not in layers
        ):
            raise ValueError("Analysis view differs from saved evidence")
        analysis = report.get("analysis")
        if analysis and (
            view["column"] != analysis["column"]
            or view.get("zone_id") != analysis.get("zone_id")
        ):
            raise ValueError("Analysis point or zone differs from snapshot")
    raw = native.raw_c[r0:r1, c0:c1]
    mask = native.invalid_reason[r0:r1, c0:c1]
    valid = mask == 0
    _equal(report["native_shape"], list(raw.shape), "native shape")
    counts = {str(int(k)): int(np.count_nonzero(mask == k)) for k in np.unique(mask)}
    if report["mask_counts"] != counts:
        raise ValueError("Quality mask counts differ")
    _equal(report["valid_fraction"], float(valid.mean()), "valid fraction")
    _equal(
        report["x_bounds_m"],
        [float(native.x_m[c0]), float(native.x_m[c1 - 1])],
        "distance bounds",
    )
    _equal(
        report["time_bounds_s"],
        [float(native.elapsed_s[r0]), float(native.elapsed_s[r1 - 1])],
        "time bounds",
    )
    if report["timestamps"] != [
        native.profiles[r0]["timestamp"],
        native.profiles[r1 - 1]["timestamp"],
    ]:
        raise ValueError("Source timestamps differ")
    _equal(
        report["gaps_over_300s"],
        [float(v) for v in np.diff(native.elapsed_s[r0:r1]) if v > 300],
        "time gaps",
    )
    valid_values = raw[valid]
    for name, value in (
        ("min", float(valid_values.min()) if valid_values.size else None),
        ("median", float(np.median(valid_values)) if valid_values.size else None),
        ("max", float(valid_values.max()) if valid_values.size else None),
    ):
        _equal(report["temperature_c"][name], value, "temperature summary")
    profile = report["profile"]
    _equal(profile["x_m"], native.x_m[c0:c1].tolist(), "profile coordinates")
    expected = [
        float(v) if q == 0 else None for v, q in zip(raw[row - r0], mask[row - r0])
    ]
    _equal(profile["temperature_c"], expected, "raw profile")
    rows = np.linspace(0, raw.shape[0] - 1, min(raw.shape[0], 64), dtype=int)
    cols = np.linspace(0, raw.shape[1] - 1, min(raw.shape[1], 128), dtype=int)
    pv = valid[np.ix_(rows, cols)]
    _equal(report["preview_valid"], pv.tolist(), "preview mask")
    _equal(
        report["preview"], np.where(pv, raw[np.ix_(rows, cols)], 0).tolist(), "preview"
    )
    _equal(
        report["preview_x_m"], native.x_m[c0:c1][cols].tolist(), "preview coordinates"
    )
    _equal(
        report["preview_time_s"], native.elapsed_s[r0:r1][rows].tolist(), "preview time"
    )
    run_id = profile.get("processing_run_id")
    if report.get("analysis") is not None:
        analysis = report["analysis"]
        if (
            run is None
            or run.get("id") != analysis.get("run_id")
            or run.get("source_sha256") != source["source_sha256"]
            or run.get("result") != analysis
        ):
            raise ValueError("Analysis snapshot differs from selected run")
        ar = run["region"]
        if not (
            ar["record_id"] == region["record_id"]
            and ar["row_start"] <= r0 < r1 <= ar["row_end"]
            and ar["column_start"] <= c0 < c1 <= ar["column_end"]
        ):
            raise ValueError("Analysis does not cover evidence")
        metadata = {k: v for k, v in run.items() if k != "result"}
        if report.get("runs") != [metadata] or report.get("profile_runs") != [metadata]:
            raise ValueError("Analysis provenance differs")
        _equal(analysis["preview_rows"], (rows + r0).tolist(), "analysis rows")
        _equal(analysis["preview_columns"], (cols + c0).tolist(), "analysis columns")
        for layer in analysis["layers"].values():
            if len(layer["profile"]) != c1 - c0:
                raise ValueError("Analysis profile shape differs")
            flags = np.asarray(layer["preview_valid"], dtype=bool)
            if flags.shape != pv.shape or np.any(flags & ~pv):
                raise ValueError("Analysis changed a native missing value")
            for index, value in enumerate(layer["profile"]):
                if expected[index] is None and value is not None:
                    raise ValueError("Analysis filled a native missing value")
                if value is not None and (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    raise ValueError("Invalid analysis profile value")
        processed = analysis["layers"].get("processed_temperature_c")
        if processed:
            _equal(
                profile.get("processed_temperature_c"),
                processed["profile"],
                "analysis processed profile",
            )
    elif run_id is None:
        if (
            "processed_temperature_c" in profile
            or report.get("runs")
            or report.get("profile_runs")
            or run is not None
        ):
            raise ValueError("Raw-only evidence contains an unbound processing result")
    else:
        if run is None:
            raise ValueError("Selected processing requires its saved run JSON (--run)")
        from types import SimpleNamespace

        from fastapi import HTTPException

        from .workspace_routes import Region, checked_profile_run

        if run.get("id") != run_id:
            raise ValueError("Selected run ID differs")
        try:
            _, values = checked_profile_run(
                SimpleNamespace(get=lambda *_: run),
                run_id,
                Region.model_validate(region),
                row,
                source["source_sha256"],
                native.raw_c.shape[1],
            )
        except HTTPException as error:
            raise ValueError("Selected run is incompatible with evidence") from error
        metadata = {k: v for k, v in run.items() if k not in ("result", "result_ref")}
        if report.get("runs") != [metadata] or report.get("profile_runs") != [metadata]:
            raise ValueError("Processing provenance differs from the saved run")
        _equal(
            profile["processed_temperature_c"],
            [v if q is not None else None for v, q in zip(values[c0:c1], expected)],
            "processed profile",
        )
    if "processing_changes" in report and json.dumps(
        report["processing_changes"], sort_keys=True, allow_nan=False
    ) != json.dumps(profile_changes(profile), sort_keys=True, allow_nan=False):
        raise ValueError("Processing changes differ from the saved profile")
    return {
        "source_sha256": source["source_sha256"],
        "record_id": region["record_id"],
        "profile_row": row,
        "profile_points": c1 - c0,
        "preview_cells": int(pv.size),
        "processing_run_id": run_id,
        "quality_assessed": False,
        "expert_answer_verified": False,
    }
