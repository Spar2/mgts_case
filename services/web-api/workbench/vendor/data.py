"""Strict CSV ingestion into immutable, versioned, memory-mapped records.

Two streaming passes bound parsing memory to one profile. The cache is keyed
by source content and preprocessing contract, never by only file name/mtime.
No resampling, timezone guessing, interpolation or temperature cleaning occurs.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np

CONTRACT_VERSION = "dts-record-v1"
VALID, SENTINEL, NONFINITE, ABSENT_TAIL = 0, 1, 2, 4
csv.field_size_limit(min(2**31 - 1, sys.maxsize))


class DataError(ValueError):
    """An input cannot satisfy the record contract; no partial output is used."""


@dataclass(frozen=True)
class DataConfig:
    max_continuity_gap_s: float = 300.0
    coordinate_tolerance_m: float = 1e-8
    sentinel_temperature_c: float = -300.0

    def __post_init__(self):
        if not math.isfinite(self.max_continuity_gap_s) or self.max_continuity_gap_s <= 0:
            raise ValueError("max_continuity_gap_s must be finite and positive")
        if not math.isfinite(self.coordinate_tolerance_m) or self.coordinate_tolerance_m < 0:
            raise ValueError("coordinate_tolerance_m must be finite and nonnegative")
        if not math.isfinite(self.sentinel_temperature_c):
            raise ValueError("sentinel_temperature_c must be finite")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def parse_time(value: str) -> datetime:
    # PostgreSQL exports use +03 as well as ISO +03:00. Keep source text separately.
    try:
        return datetime.fromisoformat(re.sub(r"([+-]\d{2})$", r"\1:00", value))
    except (TypeError, ValueError) as exc:
        raise DataError(f"invalid timestamp {value!r}") from exc


def _pairs(text: str) -> np.ndarray:
    value = json.loads(text)
    if not isinstance(value, list) or not value:
        raise DataError("points must be a nonempty JSON list of numeric pairs")
    # Explicitly reject numeric strings and booleans, which NumPy would coerce.
    for pair in value:
        if not isinstance(pair, list) or len(pair) != 2 or any(
            isinstance(item, bool) or not isinstance(item, (int, float)) for item in pair
        ):
            raise DataError("points must contain numeric [metres, temperature_C] pairs")
    return np.asarray(value, dtype=np.float64)


def read_profiles(path: Path) -> Iterator[tuple[int, dict, datetime, np.ndarray]]:
    required = {"dts_id", "channel_id", "timestamp", "points"}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or not required <= set(fields) or set(fields) - required - {"points_size"}:
            raise DataError(f"{path.name}: unexpected or duplicate CSV columns {fields}")
        row_idx = 0
        try:
            for row_idx, row in enumerate(reader):
                if None in row or any(value is None for value in row.values()):
                    raise DataError("wrong CSV field count")
                if not row["dts_id"].strip() or not row["channel_id"].strip():
                    raise DataError("empty instrument/channel identifier")
                pairs = _pairs(row["points"])
                x = pairs[:, 0]
                if not np.isfinite(x).all() or np.any(x < 0) or not np.all(np.diff(x) > 0):
                    raise DataError("coordinates must be finite, nonnegative and strictly increasing")
                if "points_size" in row and int(row["points_size"]) != len(pairs):
                    raise DataError("points_size does not match JSON pair count")
                yield row_idx, row, parse_time(row["timestamp"]), pairs
        except (ValueError, TypeError, OverflowError, csv.Error) as exc:
            raise DataError(f"{path.name}: profile {row_idx} (0-based), CSV line {reader.line_num}: {exc}") from exc


@dataclass
class Record:
    path: Path
    manifest: dict
    x_m: np.ndarray
    raw_c: np.ndarray
    invalid_reason: np.ndarray
    elapsed_s: np.ndarray
    segment_id: np.ndarray
    profiles: list[dict]

    @property
    def valid(self) -> np.ndarray:
        return self.invalid_reason == VALID

    @property
    def record_id(self) -> str:
        return self.manifest["record_id"]


def load_record(directory: Path, *, verify: bool = True) -> Record:
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["contract_version"] != CONTRACT_VERSION:
        raise DataError(f"unsupported contract {manifest['contract_version']}")
    if verify:
        for name, digest in manifest["artifact_sha256"].items():
            if Path(name).name != name or sha256(directory / name) != digest:
                raise DataError(f"cache integrity failure: {name}")
    arrays = {name: np.load(directory / f"{name}.npy", mmap_mode="r", allow_pickle=False)
              for name in ("x_m", "raw_c", "invalid_reason", "elapsed_s", "segment_id")}
    profiles = json.loads((directory / "profiles.json").read_text(encoding="utf-8"))
    shape = tuple(manifest["shape"])
    if (arrays["raw_c"].shape != shape or arrays["invalid_reason"].shape != shape
            or arrays["x_m"].shape != (shape[1],) or arrays["elapsed_s"].shape != (shape[0],)
            or arrays["segment_id"].shape != (shape[0],) or len(profiles) != shape[0]):
        raise DataError("cache dimensions do not match manifest")
    return Record(directory, manifest, profiles=profiles, **arrays)


def ingest(path: Path, cache_root: Path, config: DataConfig = DataConfig()) -> Record:
    path, cache_root = Path(path).resolve(), Path(cache_root).resolve()
    before = path.stat()
    digest = sha256(path)
    identity = {"source_sha256": digest, "contract_version": CONTRACT_VERSION, "config": asdict(config)}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    destination = cache_root / key
    if destination.exists():
        return load_record(destination)  # corrupt cache fails loudly, never silently reused
    cache_root.mkdir(parents=True, exist_ok=True)
    profiles, axis, first, previous, sensor = [], None, None, None, None
    segment, counts, dt_counts = 0, Counter(), Counter()
    sentinel_cells = nonfinite_cells = cells = 0
    for index, row, moment, pairs in read_profiles(path):
        x, t = pairs[:, 0], pairs[:, 1]
        current_sensor = (row["dts_id"], row["channel_id"])
        if sensor is None:
            sensor, first = current_sensor, moment
            axis = x.copy()
        if current_sensor != sensor:
            raise DataError(f"{path.name}: profile {index}: instrument/channel changed; split records explicitly")
        if (moment.tzinfo is None) != (first.tzinfo is None):
            raise DataError(f"{path.name}: profile {index}: mixed timezone-aware and unzoned timestamps")
        shared = min(len(axis), len(x))
        if not np.allclose(axis[:shared], x[:shared], atol=config.coordinate_tolerance_m, rtol=0):
            raise DataError(f"{path.name}: profile {index}: changed coordinate grid; no implicit interpolation")
        if len(x) > len(axis):
            axis = x.copy()
        dt = None if previous is None else (moment - previous).total_seconds()
        if dt is not None and dt <= 0:
            raise DataError(f"{path.name}: profile {index}: timestamps not strictly increasing")
        if dt is not None:
            dt_counts[dt] += 1
            if dt > config.max_continuity_gap_s:
                segment += 1
        flags = []
        n_sentinel = int(np.count_nonzero(t == config.sentinel_temperature_c))
        n_nonfinite = int(np.count_nonzero(~np.isfinite(t)))
        if n_sentinel:
            flags.append("sentinel")
        if n_nonfinite:
            flags.append("nonfinite_temperature")
        if dt is not None and dt > config.max_continuity_gap_s:
            flags.append("gap_before")
        profiles.append({"row_idx": index, "timestamp": row["timestamp"],
                         "elapsed_s": (moment - first).total_seconds(), "dt_s": dt,
                         "segment_id": segment, "n_points": len(x), "flags": flags})
        counts[len(x)] += 1
        cells += len(x)
        sentinel_cells += n_sentinel
        nonfinite_cells += n_nonfinite
        previous = moment
    if not profiles:
        raise DataError(f"{path.name}: empty dataset")
    for p in profiles:
        if p["n_points"] < len(axis):
            p["flags"].append("absent_tail")
    # Staging directories remain recoverable if abruptly interrupted; never published as cache.
    stage = cache_root / f".building-{key}-{uuid.uuid4().hex}"
    stage.mkdir()
    shape = (len(profiles), len(axis))
    raw = reason = None
    try:
        raw = np.lib.format.open_memmap(stage / "raw_c.npy", mode="w+", dtype="float64", shape=shape)
        reason = np.lib.format.open_memmap(stage / "invalid_reason.npy", mode="w+", dtype="uint8", shape=shape)
        raw[:] = np.nan
        reason[:] = ABSENT_TAIL
        second_count = 0
        for index, row, _, pairs in read_profiles(path):
            if index >= len(profiles) or row["timestamp"] != profiles[index]["timestamp"]:
                raise DataError("source changed during ingestion")
            n = len(pairs)
            raw[index, :n] = pairs[:, 1]
            reason[index, :n] = VALID
            reason[index, :n][pairs[:, 1] == config.sentinel_temperature_c] = SENTINEL
            reason[index, :n][~np.isfinite(pairs[:, 1])] = NONFINITE
            second_count += 1
        raw.flush(); reason.flush()
        # Explicitly release handles before directory rename on Windows.
        raw._mmap.close(); reason._mmap.close()
        raw = reason = None
        if second_count != len(profiles) or sha256(path) != digest:
            raise DataError("source changed during ingestion")
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise DataError("source metadata changed during ingestion")
        for name, array in (("x_m", axis), ("elapsed_s", [p["elapsed_s"] for p in profiles]),
                            ("segment_id", np.array([p["segment_id"] for p in profiles], dtype=np.int32))):
            np.save(stage / f"{name}.npy", array, allow_pickle=False)
        write_json(stage / "profiles.json", profiles)
        dt_array = np.diff([p["elapsed_s"] for p in profiles])
        kept_dt = dt_array[dt_array <= config.max_continuity_gap_s]
        manifest = {**identity, "cache_key": key, "record_id": key[:16],
                    "source_name": path.name, "source_path": str(path), "source_bytes": before.st_size,
                    "shape": list(shape), "source_cells": cells,
                    "sentinel_cells": sentinel_cells, "nonfinite_cells": nonfinite_cells,
                    "absent_tail_cells": shape[0] * shape[1] - cells,
                    "dts_id": sensor[0], "channel_id": sensor[1],
                    "time_kind": "offset_aware" if first.tzinfo is not None else "local_unzoned",
                    "timestamps_preserved": True, "interpolation": "none",
                    "spatial_resolution_m": None, "median_sampling_step_m": float(np.median(np.diff(axis))) if len(axis) > 1 else None,
                    "point_counts": dict(sorted(counts.items())), "interval_counts_s": dict(sorted(dt_counts.items())),
                    "segments": segment + 1,
                    "span_s": profiles[-1]["elapsed_s"], "continuous_observation_span_s": float(kept_dt.sum()),
                    "observation_span_definition": "sum of adjacent time differences <= configured continuity gap; no extrapolation after last sample",
                    "invalid_reason_codes": {"0": "valid", "1": "sentinel", "2": "nonfinite", "4": "absent_tail"},
                    "raw_dtype": "float64", "artifact_sha256": {}}
        manifest["artifact_sha256"] = {p.name: sha256(p) for p in sorted(stage.iterdir()) if p.is_file()}
        write_json(stage / "manifest.json", manifest)
        os.replace(stage, destination)
    except BaseException:
        for array in (raw, reason):
            if array is not None:
                array._mmap.close()
        shutil.rmtree(stage)
        raise
    return load_record(destination)
