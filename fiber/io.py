"""Load MGTS DTS CSV files onto a native [time, distance] grid."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fiber.quality import quality_mask

try:
    import orjson
except ImportError:  # pragma: no cover
    orjson = None

csv.field_size_limit(2**31 - 1)
_TZ_TAIL = re.compile(r"([+-])(\d{2})$")


def _parse_points(text: str) -> np.ndarray:
    payload = orjson.loads(text) if orjson is not None else json.loads(text)
    arr = np.asarray(payload, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("points must be a list of [distance_m, temperature_c] pairs")
    return arr


def parse_timestamp(text: str) -> tuple[datetime, bool]:
    """Return (datetime, tz_was_missing); never invent a timezone."""
    s = str(text).strip().replace("T", " ")
    s = _TZ_TAIL.sub(lambda m: f"{m.group(1)}{m.group(2)}:00", s)
    dt = datetime.fromisoformat(s)
    missing = dt.tzinfo is None
    return (dt if missing else dt.astimezone(timezone.utc)), missing


def _time_coordinate(dt: datetime) -> float:
    # A naive coordinate is a wall-clock offset, NOT an absolute UTC instant.
    if dt.tzinfo is None:
        return (dt - datetime(1970, 1, 1)).total_seconds()
    return dt.timestamp()


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Record:
    name: str
    dts_id: str
    channel_id: str
    x_m: np.ndarray
    time_s: np.ndarray
    time_iso: list[str]
    raw_c: np.ndarray
    valid: np.ndarray
    point_counts: np.ndarray
    source_csv: str
    source_sha256: str
    tz_missing: bool
    n_profiles: int
    n_points: int
    n_valid: int
    n_sentinel: int
    n_tail: int
    dx_m: float
    quality_codes: np.ndarray | None = None

    def save(self, directory: Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "x_m.npy", self.x_m)
        np.save(directory / "time_s.npy", self.time_s)
        np.save(directory / "time_iso.npy", np.asarray(self.time_iso))
        np.save(directory / "raw_c.npy", self.raw_c)
        np.save(directory / "valid.npy", self.valid)
        np.save(directory / "point_counts.npy", self.point_counts)
        codes = self.quality_codes
        if codes is None:
            codes = quality_mask(self.raw_c, self.point_counts)["reason_code"]
        np.save(directory / "quality_codes.npy", codes)
        skip = {"x_m", "time_s", "time_iso", "raw_c", "valid", "point_counts", "quality_codes"}
        # asdict deep-copies numpy arrays before dropping them, multiplying
        # memory on native cable grids. Read only the scalar metadata fields.
        meta = {field.name: getattr(self, field.name) for field in fields(self) if field.name not in skip}
        meta["shape"] = list(self.raw_c.shape)
        meta["time_start"] = self.time_iso[0]
        meta["time_end"] = self.time_iso[-1]
        meta["time_basis"] = "wall_clock_timezone_unknown" if self.tz_missing else "utc_epoch"
        meta["quality_code_schema"] = {"0": "valid", "1": "nonfinite_measurement", "2": "sentinel", "4": "missing_coordinate"}
        meta["n_missing"] = int(np.count_nonzero(codes == 4))
        (directory / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        return directory

    @staticmethod
    def load(directory: Path) -> "Record":
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        return Record(
            name=meta["name"],
            dts_id=meta["dts_id"],
            channel_id=str(meta["channel_id"]),
            x_m=np.load(directory / "x_m.npy"),
            time_s=np.load(directory / "time_s.npy"),
            time_iso=np.load(directory / "time_iso.npy").tolist(),
            raw_c=np.load(directory / "raw_c.npy"),
            valid=np.load(directory / "valid.npy"),
            point_counts=np.load(directory / "point_counts.npy"),
            source_csv=meta["source_csv"],
            source_sha256=meta["source_sha256"],
            tz_missing=bool(meta["tz_missing"]),
            n_profiles=int(meta["n_profiles"]),
            n_points=int(meta["n_points"]),
            n_valid=int(meta["n_valid"]),
            n_sentinel=int(meta["n_sentinel"]),
            n_tail=int(meta["n_tail"]),
            dx_m=float(meta["dx_m"]),
            quality_codes=np.load(directory / "quality_codes.npy") if (directory / "quality_codes.npy").exists() else None,
        )


def load_csv(path: Path) -> Record:
    path = Path(path)
    xs_rows: list[np.ndarray] = []
    temps: list[np.ndarray] = []
    stamps: list[datetime] = []
    iso: list[str] = []
    dts_ids: list[str] = []
    channels: list[str] = []
    tz_missing = False
    timezone_kinds: set[bool] = set()

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "points" not in reader.fieldnames or "timestamp" not in reader.fieldnames:
            raise ValueError(f"{path} is missing required CSV columns")
        for row_number, row in enumerate(reader, start=2):
            pts = _parse_points(row["points"])
            if not len(pts) or not np.isfinite(pts[:, 0]).all() or np.any(np.diff(pts[:, 0]) <= 0):
                raise ValueError(f"{path}:{row_number}: distances must be finite and strictly increasing")
            if row.get("points_size") not in (None, ""):
                try:
                    declared = int(row["points_size"])
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"{path}:{row_number}: invalid points_size") from exc
                if declared != len(pts):
                    raise ValueError(f"{path}:{row_number}: points_size={declared} but got {len(pts)} pairs")
            xs_rows.append(pts[:, 0])
            temps.append(pts[:, 1])
            dt, missing = parse_timestamp(row["timestamp"])
            tz_missing = tz_missing or missing
            timezone_kinds.add(missing)
            stamps.append(dt)
            iso.append(dt.isoformat())
            dts_ids.append(str(row.get("dts_id") or "").strip())
            channels.append(str(row.get("channel_id") or "").strip())

    if not temps:
        raise ValueError(f"{path} has no profiles")
    if len(timezone_kinds) > 1:
        raise ValueError(f"{path}: cannot mix unknown and known timezones")
    if len(set(dts_ids)) != 1 or len(set(channels)) != 1:
        raise ValueError(f"{path}: a record must contain one device and channel")

    lengths = np.fromiter((v.size for v in temps), dtype=np.int32, count=len(temps))
    ref_i = int(np.argmax(lengths))
    x_m = xs_rows[ref_i].astype(np.float64, copy=True)
    n_x = int(x_m.size)
    n_t = len(temps)
    raw = np.full((n_t, n_x), np.nan, dtype=np.float32)
    observed = np.zeros((n_t, n_x), dtype=bool)

    for i, (xs, ys) in enumerate(zip(xs_rows, temps)):
        n = int(ys.size)
        if n <= n_x and np.allclose(xs, x_m[:n], atol=1e-6, rtol=0):
            raw[i, :n] = ys.astype(np.float32)
            observed[i, :n] = True
            continue
        # Match measured coordinates only. Unknown axes require an explicit
        # resampling contract; filling missing measurements would invent data.
        right = np.clip(np.searchsorted(x_m, xs), 0, n_x - 1)
        left = np.maximum(right - 1, 0)
        idx = np.where(np.abs(x_m[left] - xs) < np.abs(x_m[right] - xs), left, right)
        if not np.allclose(x_m[idx], xs, atol=1e-6, rtol=0) or len(np.unique(idx)) != n:
            raise ValueError(f"{path}: row {i + 2} coordinates do not match the native reference axis")
        raw[i, idx] = ys.astype(np.float32)
        observed[i, idx] = True

    order = np.argsort(np.array([_time_coordinate(t) for t in stamps], dtype=np.float64))
    if not np.all(order == np.arange(n_t)):
        raw = raw[order]
        observed = observed[order]
        lengths = lengths[order]
        stamps = [stamps[i] for i in order]
        iso = [iso[i] for i in order]
        dts_ids = [dts_ids[i] for i in order]
        channels = [channels[i] for i in order]

    time_s = np.array([_time_coordinate(t) for t in stamps], dtype=np.float64)
    if np.any(np.diff(time_s) <= 0):
        raise ValueError(f"{path}: timestamps are not strictly increasing after sort")

    masks = quality_mask(raw)
    masks["valid"] &= observed
    masks["reason_code"][~observed] = 4
    dx = float(np.median(np.diff(x_m))) if n_x > 1 else 0.0
    dts_id = dts_ids[0] if len(set(dts_ids)) == 1 else ",".join(sorted(set(dts_ids)))
    channel_id = channels[0] if len(set(channels)) == 1 else ",".join(sorted(set(channels)))
    return Record(
        name=path.stem,
        dts_id=dts_id,
        channel_id=channel_id,
        x_m=x_m,
        time_s=time_s,
        time_iso=iso,
        raw_c=raw,
        valid=masks["valid"],
        point_counts=lengths,
        source_csv=str(path.resolve()),
        source_sha256=file_sha256(path),
        tz_missing=tz_missing,
        n_profiles=n_t,
        n_points=n_x,
        n_valid=int(masks["valid"].sum()),
        n_sentinel=int(masks["is_sentinel"].sum()),
        n_tail=int((~observed).sum()),
        dx_m=dx,
        quality_codes=masks["reason_code"],
    )


def transform_all(raw_root: Path, out_root: Path) -> list[dict]:
    raw_root = Path(raw_root)
    out_root = Path(out_root)
    files = sorted(raw_root.glob("*.csv"), key=lambda p: p.name)
    if not files:
        raise FileNotFoundError(f"no CSV files in {raw_root}")
    reports = []
    for path in files:
        record = load_csv(path)
        dest = out_root / record.name
        record.save(dest)
        reports.append(json.loads((dest / "meta.json").read_text(encoding="utf-8")))
    (out_root / "index.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return reports
