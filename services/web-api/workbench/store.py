"""Read-only, verified native records and bounded physical-bin views."""

import hashlib
import json
import os
import struct
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from weakref import WeakValueDictionary

import numpy as np

from .vendor.data import load_record, sha256


@contextmanager
def cache_guard(root):
    """Inter-process publication/pruning lock; numerical computation stays parallel."""
    with (root / ".cache.lock").open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            handle.write(b"0")
            handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def packet(values, mask, counts, *, format_version=1, display_flags=None, **metadata):
    values = np.asarray(values, dtype="<f4")
    mask = np.asarray(mask, dtype="u1")
    counts = np.asarray(counts, dtype="<u4")
    head = json.dumps(
        dict(version=format_version, shape=list(values.shape), **metadata),
        allow_nan=False,
    ).encode()
    prefix = struct.pack("<I", len(head)) + head
    prefix += b"\0" * (-len(prefix) % 4)
    payload = values.tobytes() + mask.tobytes()
    payload += b"\0" * (-len(payload) % 4)
    return (
        prefix
        + payload
        + counts.tobytes()
        + (
            np.asarray(
                display_flags if display_flags is not None else np.zeros(values.shape),
                dtype="u1",
            ).tobytes()
            if format_version == 2
            else b""
        )
    )


class Store:
    def __init__(self, root, cache):
        self.root, self.cache = Path(root), Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.records = {}
        self.meta = {}
        self.lock = threading.RLock()
        self.tile_locks = WeakValueDictionary()
        imported = self.cache.parent / "imported-records"
        for manifest in sorted(
            [*self.root.glob("*/manifest.json"), *imported.glob("*/manifest.json")]
        ):
            record = load_record(manifest.parent, verify=True)
            key = record.manifest["source_name"].removesuffix(".csv")
            if key in self.records:
                raise ValueError("Duplicate source name in catalog")
            self.records[key] = record
            meta_path = self.cache / ("meta-" + sha256(manifest) + ".json")
            if meta_path.exists():
                cached = json.loads(meta_path.read_text(encoding="utf-8"))
                if cached.get("source_sha256") == record.manifest["source_sha256"]:
                    self.meta[key] = cached
                    continue
            valid = record.invalid_reason == 0
            temps = record.raw_c[valid]
            low, high = np.percentile(temps, [1, 99]) if len(temps) else (0, 1)
            if high <= low:
                high = low + 1
            self.meta[key] = {
                "id": key,
                "source": record.manifest["source_name"],
                "source_sha256": record.manifest["source_sha256"],
                "shape": list(record.raw_c.shape),
                "x_bounds": [float(record.x_m[0]), float(record.x_m[-1])],
                "time_bounds": [
                    float(record.elapsed_s[0]),
                    float(record.elapsed_s[-1]),
                ],
                "first_timestamp": record.profiles[0]["timestamp"],
                "last_timestamp": record.profiles[-1]["timestamp"],
                "time_kind": record.manifest["time_kind"],
                "segments": record.manifest["segments"],
                "valid_fraction": float(valid.mean()),
                "scale": [float(low), float(high)],
                "dts_id": record.manifest["dts_id"],
                "channel_id": record.manifest["channel_id"],
            }

            temporary = meta_path.with_suffix("." + uuid.uuid4().hex + ".tmp")
            temporary.write_text(
                json.dumps(self.meta[key], allow_nan=False), encoding="utf-8"
            )
            os.replace(temporary, meta_path)

    def refresh_imports(self):
        imported = self.cache.parent / "imported-records"
        present = {r.path.resolve() for r in self.records.values()}
        pending = [
            p
            for p in imported.glob("*/manifest.json")
            if p.parent.resolve() not in present
        ]
        if pending:
            with self.lock:
                replacement = Store(self.root, self.cache)
                self.records, self.meta = replacement.records, replacement.meta

    def get(self, key):
        if key not in self.records:
            self.refresh_imports()
        if key not in self.records:
            raise KeyError("Unknown record")
        return self.records[key]

    def tile(self, key, x0, x1, t0, t1, width, height, format_version=1):
        record = self.get(key)
        if not (np.isfinite([x0, x1, t0, t1]).all() and x0 < x1 and t0 < t1):
            raise ValueError("Finite increasing physical bounds required")
        meta = self.meta[key]
        if (
            x0 < meta["x_bounds"][0]
            or x1 > meta["x_bounds"][1]
            or t0 < 0
            or t1 > meta["time_bounds"][1]
        ):
            raise ValueError("Viewport outside native extent")
        identity = json.dumps(
            [3, format_version, meta["source_sha256"], x0, x1, t0, t1, width, height]
        )
        cache_key = hashlib.sha256(identity.encode()).hexdigest()
        path = self.cache / (cache_key + ".bin")
        # Lock publishing avoids Windows replacement races. Hash-addressed cache is bounded.
        with self.lock:
            tile_lock = self.tile_locks.setdefault(cache_key, threading.Lock())
        with tile_lock:
            with self.lock, cache_guard(self.cache):
                if path.exists():
                    return path.read_bytes()
            xs, ts = record.x_m, record.elapsed_s
            xe, te = np.linspace(x0, x1, width + 1), np.linspace(t0, t1, height + 1)
            xi0, xi1 = np.searchsorted(xs, [x0, x1], side="left")
            xi1 = min(len(xs), int(np.searchsorted(xs, x1, side="right")))
            ti0, ti1 = np.searchsorted(ts, [t0, t1], side="left")
            ti1 = min(len(ts), int(np.searchsorted(ts, t1, side="right")))
            # At close range render actual cells instead of manufacturing empty
            # physical bins between otherwise continuous native observations.
            if 0 < ti1 - ti0 <= height and 0 < xi1 - xi0 <= width:
                reasons = np.array(record.invalid_reason[ti0:ti1, xi0:xi1], copy=True)
                values = np.array(record.raw_c[ti0:ti1, xi0:xi1], dtype=np.float32)
                values[reasons != 0] = np.nan
                return packet(
                    values,
                    reasons,
                    (reasons == 0).astype(np.uint32),
                    format_version=format_version,
                    x_cells=cell_edges(xs)[xi0:xi1],
                    time_cells=cell_edges(ts, 300)[ti0:ti1],
                    aggregation="native_observations",
                    source_sha256=meta["source_sha256"],
                    scale=meta["scale"],
                )
            xbin = np.minimum(
                width - 1, np.searchsorted(xe, xs[xi0:xi1], side="right") - 1
            )
            tbin = np.minimum(
                height - 1, np.searchsorted(te, ts[ti0:ti1], side="right") - 1
            )
            values = np.full((height, width), np.nan, np.float32)
            mask = np.full((height, width), 8, np.uint8)
            counts = np.zeros((height, width), np.uint32)
            native_mask = np.zeros((height, width), np.uint8)
            display = np.ones((height, width), np.uint8)
            xgroups = [np.flatnonzero(xbin == i) + xi0 for i in range(width)]
            for y in range(height):
                rows = np.flatnonzero(tbin == y) + ti0
                if not len(rows):
                    continue
                for x, columns in enumerate(xgroups):
                    if not len(columns):
                        continue
                    raw = record.raw_c[np.ix_(rows, columns)]
                    reasons = record.invalid_reason[np.ix_(rows, columns)]
                    native_mask[y, x] = np.bitwise_or.reduce(reasons.ravel())
                    display[y, x] = 0
                    good = reasons == 0
                    counts[y, x] = np.count_nonzero(good)
                    if counts[y, x]:
                        values[y, x] = np.median(raw[good])
                        mask[y, x] = 0
                    else:
                        mask[y, x] = np.bitwise_or.reduce(reasons.ravel())
            # Never bridge continuity boundaries, even if an overview bin crosses a gap.
            gaps = np.flatnonzero(np.diff(ts) > 300)
            for gap in gaps:
                crossed = (te[:-1] < ts[gap + 1]) & (te[1:] > ts[gap])
                values[crossed] = np.nan
                mask[crossed] = 8
                display[crossed] = 2
                counts[crossed] = 0
            data = packet(
                values,
                mask if format_version == 1 else native_mask,
                counts,
                format_version=format_version,
                display_flags=display,
                display_flag_codes={
                    "0": "observed",
                    "1": "empty_bin",
                    "2": "continuity_gap",
                },
                x_edges=xe.tolist(),
                time_edges=te.tolist(),
                aggregation="median_valid_physical_bins_v1",
                source_sha256=meta["source_sha256"],
                scale=meta["scale"],
            )
            temporary = path.with_suffix("." + uuid.uuid4().hex + ".tmp")
            temporary.write_bytes(data)
            with self.lock, cache_guard(self.cache):
                os.replace(temporary, path)
                files = sorted(
                    self.cache.glob("*.bin"), key=lambda p: p.stat().st_mtime
                )
                size = sum(p.stat().st_size for p in files)
                for old in files:
                    if size <= 256 * 1024 * 1024:
                        break
                    size -= old.stat().st_size
                    old.unlink(missing_ok=True)
            return data

    def slices(self, key, row, column, row_start=0, row_end=None):
        r = self.get(key)
        if row >= r.raw_c.shape[0] or column >= r.raw_c.shape[1]:
            raise ValueError("Native index outside record")

        def clean(values, reasons):
            return [
                float(v) if reason == 0 else None for v, reason in zip(values, reasons)
            ]

        profile = clean(r.raw_c[row], r.invalid_reason[row])
        row_end = r.raw_c.shape[0] if row_end is None else row_end
        if not 0 <= row_start < row_end <= r.raw_c.shape[0]:
            raise ValueError("Trend bounds must belong to native rows")
        trend = clean(
            r.raw_c[row_start:row_end, column],
            r.invalid_reason[row_start:row_end, column],
        )
        # Add explicit nulls before a discontinuity so line renderers do not connect it.
        trend_x, trend_y = [], []
        times = r.elapsed_s[row_start:row_end]
        for i, (x, y) in enumerate(zip(times, trend)):
            if i and times[i] - times[i - 1] > 300:
                trend_x.append(float((times[i] + times[i - 1]) / 2))
                trend_y.append(None)
            trend_x.append(float(x))
            trend_y.append(y)
        return {
            "row": row,
            "column": column,
            "x_m": r.x_m.tolist(),
            "profile": profile,
            "time_s": trend_x,
            "trend": trend_y,
            "timestamp": r.profiles[row]["timestamp"],
            "elapsed_s": float(r.elapsed_s[row]),
            "distance_m": float(r.x_m[column]),
            "temperature": profile[column],
            "invalid_reason": int(r.invalid_reason[row, column]),
            "source_sha256": r.manifest["source_sha256"],
        }


def cell_edges(axis, gap_limit=None):
    axis = np.asarray(axis)
    delta = np.diff(axis)
    step = (
        float(np.median(delta[delta <= gap_limit]))
        if gap_limit and np.any(delta <= gap_limit)
        else float(np.median(delta))
        if len(delta)
        else 1.0
    )
    left = np.r_[axis[0] - step / 2, (axis[:-1] + axis[1:]) / 2]
    right = np.r_[(axis[:-1] + axis[1:]) / 2, axis[-1] + step / 2]
    if gap_limit:
        for i in np.flatnonzero(delta > gap_limit):
            right[i] = axis[i] + step / 2
            left[i + 1] = axis[i + 1] - step / 2
    return np.column_stack([left, right]).tolist()


class Queue:
    def __init__(self, root, store, max_cells=200000):
        self.max_cells = max_cells
        self.root = Path(root)
        path = self.root / "queue.json"
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if sha256(path) != manifest["queue_sha256"]:
            raise ValueError("Queue integrity mismatch")
        self.data = json.loads(path.read_text(encoding="utf-8"))
        self.sha = manifest["queue_sha256"]
        self.cards = {c["review_id"]: c for c in self.data["cards"]}
        if (
            len(self.cards) != len(self.data["cards"])
            or len(self.cards) != manifest["cards"]
        ):
            raise ValueError("Queue cardinality mismatch")
        for card in self.cards.values():
            record = store.get(card["source"].removesuffix(".csv"))
            if card["evidence"]["source_sha256"] != record.manifest["source_sha256"]:
                raise ValueError("Queue source differs from catalog")
            self.load(card["review_id"])

    def load(self, key):
        card = self.cards[key]
        name = card["artifact"]
        if Path(name).name != name or not name.endswith(".npz"):
            raise ValueError("Invalid artifact path")
        path = self.root / name
        if sha256(path) != card["artifact_sha256"]:
            raise ValueError("Card integrity mismatch")
        with np.load(path, allow_pickle=False) as f:
            data = {k: f[k] for k in f.files}
        shape = tuple(card["native_shape"])
        raw, reasons = data["raw_c"], data["invalid_reason"]
        if (
            raw.shape != shape
            or reasons.shape != shape
            or np.prod(shape) > self.max_cells
        ):
            raise ValueError("Invalid native shape")
        if (
            data["signed"].shape != (len(card["methods"]), *shape)
            or data["binary"].shape != data["signed"].shape
        ):
            raise ValueError("Invalid method shape")
        good = reasons == 0
        if (
            not np.isin(reasons, [0, 1, 2, 4]).all()
            or not np.isfinite(raw[good]).all()
            or (raw[good] == -300).any()
            or not np.isin(data["signed"], [-1, 0, 1, 2]).all()
            or data["binary"].dtype != np.bool_
            or (data["binary"] & ~good[None]).any()
            or (data["signed"][:, ~good] != 2).any()
        ):
            raise ValueError("Invalid native quality/method mask")
        for axis, n in [("x_m", shape[1]), ("time_s", shape[0])]:
            if (
                data[axis].shape != (n,)
                or not np.isfinite(data[axis]).all()
                or not (np.diff(data[axis]) > 0).all()
            ):
                raise ValueError("Invalid native axis")
        return card, data
