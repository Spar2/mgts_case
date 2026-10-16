"""Score a fast common-mode stage beside median_x3 on the synth test split.

The stage is not registered in METHODS and analyze_field is not called.
median_x3 + spatial_z_halo at |Z| >= 3.5 stays the operational pair.
A stored common_mode series is an oracle ceiling only: a cable does not have it.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import median_filter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fiber.anomaly.detect import spatial_z
from fiber.anomaly.zones import extract_zones
from fiber.classify.dts_synth import join_h5, read_h5
from fiber.classify.rules import BROAD_WIDTH_M
from fiber.denoise.dsp import median_x
from fiber.models.patch_classifier import SPACE_CROP, TIME_CROP

OFFICIAL = ROOT / "artifacts" / "classify" / "common_mode_report.json"
SMOKE = ROOT / "artifacts" / "classify" / "common_mode_smoke.json"
FORBIDDEN = {
    "summary_pipeline_report.json",
    "unet_halo_report.json",
    "residual_box_report.json",
    "denoise_celsius_report.json",
    "real_auto_agreement.json",
}
# Copied from the earlier median_x3 halo pass. This run must reproduce them
# before the fast column's box score is trusted.
SANITY = {
    "box_hit_total": [74, 172],
    "Local_Heating": [46, 105],
    "Local_Cooling": [6, 25],
    "Broad_Shift": [22, 42],
    "quiet_false_alarm": [84, 540],
    "nuisance_called_event": [94, 124],
    "zones": 709,
    "alarm_cells": [6967370, 136082815],
    "by_difficulty": {"hard": [8, 27], "easy": [63, 123], "medium": [3, 22]},
    "median_mae_c": 0.5370462151409129,
    "raw_mae_c": 0.6135548096280755,
    "cells": 136082815,
}
PAIR = (
    "median_x3 + spatial_z_halo at |Z| >= 3.5. "
    "This stage is not wired into analyze_field or METHODS. "
    "A higher synth score does not switch the pair."
)


class Score:
    def __init__(self):
        self.hits = {}
        self.difficulty = {}
        self.quiet_wrong = 0
        self.quiet_n = 0
        self.nuisance_wrong = 0
        self.nuisance_n = 0
        self.alarm_cells = 0
        self.alarm_total = 0
        self.zones = 0
        self.zone_files = 0

    def event(self, mapped: str, level: str, hit: bool) -> None:
        slot = self.hits.setdefault(mapped, {"hit": 0, "total": 0})
        slot["total"] += 1
        slot["hit"] += int(hit)
        bucket = self.difficulty.setdefault(level or "unknown", {"hit": 0, "total": 0})
        bucket["total"] += 1
        bucket["hit"] += int(hit)

    def nuisance(self, called: bool) -> None:
        self.nuisance_n += 1
        self.nuisance_wrong += int(called)

    def quiet(self, called: bool) -> None:
        self.quiet_n += 1
        self.quiet_wrong += int(called)

    def as_dict(self) -> dict:
        total_hit = int(sum(item["hit"] for item in self.hits.values()))
        total = int(sum(item["total"] for item in self.hits.values()))
        rate = (self.quiet_wrong / self.quiet_n) if self.quiet_n else None
        return {
            "box_hit": self.hits,
            "box_hit_total": [total_hit, total],
            "by_difficulty": self.difficulty,
            "quiet_false_alarm": [self.quiet_wrong, self.quiet_n],
            "quiet_rate": rate,
            "nuisance_called_event": [self.nuisance_wrong, self.nuisance_n],
            "alarm_cells": [self.alarm_cells, self.alarm_total] if self.alarm_total else None,
            "zones": self.zones if self.zone_files else None,
        }


def _mae_store() -> dict:
    return {"abs": 0.0, "n": 0, "bg_abs": 0.0, "bg_n": 0, "ev_abs": 0.0, "ev_n": 0}


def _add_mae(store: dict, pred: np.ndarray, clean: np.ndarray, valid: np.ndarray, mask: np.ndarray) -> None:
    ok = np.asarray(valid, dtype=bool) & np.isfinite(pred) & np.isfinite(clean)
    if not np.any(ok):
        return
    err = np.abs(np.asarray(pred, dtype=np.float64)[ok] - np.asarray(clean, dtype=np.float64)[ok])
    store["abs"] += float(err.sum())
    store["n"] += int(ok.sum())
    bg = ok & (np.asarray(mask) == 0)
    ev = ok & (np.asarray(mask) > 0)
    if np.any(bg):
        store["bg_abs"] += float(np.abs(pred[bg] - clean[bg]).sum())
        store["bg_n"] += int(bg.sum())
    if np.any(ev):
        store["ev_abs"] += float(np.abs(pred[ev] - clean[ev]).sum())
        store["ev_n"] += int(ev.sum())


def _mae_dict(store: dict, note: str | None = None) -> dict:
    def ratio(total: float, count: int):
        return (total / count) if count else None

    out = {
        "mae_c": ratio(store["abs"], store["n"]),
        "cells": store["n"],
        "background_mae_c": ratio(store["bg_abs"], store["bg_n"]),
        "background_cells": store["bg_n"],
        "event_mae_c": ratio(store["ev_abs"], store["ev_n"]),
        "event_cells": store["ev_n"],
    }
    if note:
        out["note"] = note
    return out


def _unix(text: str) -> float:
    raw = text.strip()
    if raw.endswith("+00") and not raw.endswith("+0000"):
        raw = raw[:-3] + "+0000"
    return datetime.strptime(raw, "%Y-%m-%d %H:%M:%S%z").timestamp()


def _box_name(row: dict) -> str | None:
    label = row["label"]
    if label == "tracing":
        return None
    if label == "hotspot":
        return "Local_Heating"
    if label in ("coldspot", "cooling"):
        return "Local_Cooling"
    if label in ("section_heating", "slow_warming"):
        width = float(row["dist_end_m"]) - float(row["dist_start_m"])
        return "Broad_Shift" if width > BROAD_WIDTH_M else "Local_Heating"
    return None


def _locate(t_unix: np.ndarray, x: np.ndarray, row: dict) -> tuple[int, int, int, int]:
    i0 = int(np.searchsorted(t_unix, _unix(row["t_start"])))
    i1 = int(np.searchsorted(t_unix, _unix(row["t_end"]), side="right") - 1)
    j0 = int(np.searchsorted(x, float(row["dist_start_m"]) - 1.0))
    j1 = int(np.searchsorted(x, float(row["dist_end_m"]) + 1.0, side="right") - 1)
    i0 = max(0, i0 - 2)
    i1 = min(len(t_unix) - 1, max(i0, i1 + 2))
    j0 = min(max(0, j0), len(x) - 1)
    j1 = min(max(j0, j1), len(x) - 1)
    return i0, i1, j0, j1


def _load_labels(root: Path) -> dict[str, list]:
    rows = list(csv.DictReader((root / "labels.csv").open(newline="", encoding="utf-8")))
    grouped: dict[str, list] = {}
    for row in rows:
        if row["split"] == "test":
            grouped.setdefault(row["seq_id"], []).append(row)
    return grouped


def _increasing(values: np.ndarray) -> bool:
    if len(values) < 2:
        return False
    return bool(np.all(np.diff(np.asarray(values, dtype=np.float64)) > 0))


def _draw_quiet(record: dict, rng: np.random.Generator, count: int) -> list[tuple[int, int]]:
    mask = record["mask_event"]
    valid = record["valid"]
    nt, nx = mask.shape
    found = []
    tries = 0
    while len(found) < count and tries < 200 and nt >= TIME_CROP and nx >= SPACE_CROP:
        tries += 1
        t0 = int(rng.integers(0, nt - TIME_CROP + 1))
        x0 = int(rng.integers(0, nx - SPACE_CROP + 1))
        if int(mask[t0:t0 + TIME_CROP, x0:x0 + SPACE_CROP].max()) != 0:
            continue
        if float(valid[t0:t0 + TIME_CROP, x0:x0 + SPACE_CROP].mean()) < 0.5:
            continue
        found.append((t0, x0))
    return found


def _score_file_boxes(score: Score, rows, t_unix, x, decide) -> None:
    for row in rows:
        if row["kind"] == "event" and row["label"] == "tracing":
            continue
        mapped = _box_name(row) if row["kind"] == "event" else None
        if row["kind"] == "event" and mapped is None:
            continue
        if row["kind"] not in ("event", "nuisance"):
            continue
        i0, i1, j0, j1 = _locate(t_unix, x, row)
        called = decide(i0, i1, j0, j1, mapped)
        if row["kind"] == "event":
            score.event(mapped, row.get("difficulty", ""), called)
        else:
            score.nuisance(called)


def _signed_hit(hot: np.ndarray, cold: np.ndarray, i0: int, i1: int, j0: int, j1: int, mapped: str) -> bool:
    sl = (slice(i0, i1 + 1), slice(j0, j1 + 1))
    if mapped == "Local_Cooling":
        return bool(np.any(cold[sl]))
    if mapped == "Local_Heating":
        return bool(np.any(hot[sl]))
    return bool(np.any(hot[sl] | cold[sl]))


def _zone_count(score, valid, x, t_unix):
    times = np.asarray(t_unix, dtype=np.float64)
    iso = [str(index) for index in range(len(times))]
    try:
        zones = extract_zones(
            score, valid, x, iso, times,
            thresh=3.5, min_width_m=3, min_profiles=6, min_cells=24, max_gap_s=300,
        )
    except ValueError:
        return None
    return len(zones)


def remove_common_mode(field: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Subtract the fast spatial-median series. Does not read a stored common_mode.

    The 31-trace split is the one that defined cm_sigma in calibrate.py.
    It is not a threshold chosen on the test boxes. A non-finite trace median
    becomes 0 before the slow filter, so one empty trace does not spread NaNs.
    """
    data = np.where(np.asarray(valid, dtype=bool) & np.isfinite(field), field, np.nan).astype(np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        profile = np.nanmedian(data, axis=0)
        profile = np.where(np.isfinite(profile), profile, 0.0)
        cm = np.nanmedian(data - profile, axis=1)
    cm = np.where(np.isfinite(cm), cm, 0.0)
    if len(cm) > 5:
        size = min(31, (len(cm) // 2) * 2 - 1)
        slow = median_filter(cm, size=size, mode="nearest")
        fast = cm - slow
    else:
        fast = np.zeros_like(cm)
    out = np.array(field, dtype=np.float32, copy=True)
    live = np.asarray(valid, dtype=bool) & np.isfinite(out)
    correction = np.broadcast_to(np.asarray(fast, dtype=np.float32)[:, None], out.shape)
    out[live] = out[live] - correction[live]
    return out, fast


def _self_check() -> float:
    rng = np.random.default_rng(0)
    field = rng.normal(20.0, 0.05, size=(80, 50))
    field[10:13, :] += 1.2
    valid = np.ones(field.shape, dtype=bool)
    valid[10, 0] = False
    corrected, fast = remove_common_mode(field, valid)
    jump = float(np.min(fast[10:13]))
    other = float(np.mean(np.abs(np.concatenate([fast[:10], fast[13:]]))))
    if jump < 0.8 or other > 0.3:
        raise SystemExit(f"FAILED self-check jump={jump:.4f} other={other:.4f}")
    x = np.arange(50, dtype=np.float64) * 2.0
    z_raw = spatial_z(median_x(field, valid, k=3), valid, x, inner_m=8, outer_m=40, bin_m=2, floor_c=0.30)[0]
    z_fix = spatial_z(median_x(corrected, valid, k=3), valid, x, inner_m=8, outer_m=40, bin_m=2, floor_c=0.30)[0]
    gap = float(np.max(np.abs(z_raw.astype(np.float64) - z_fix.astype(np.float64))))
    if gap > 1e-3:
        raise SystemExit(f"FAILED self-check max |dZ|={gap:.6g}")
    print(f"self-check ok jump {jump:.3f} other {other:.4f} max_abs_dz {gap:.3e}", flush=True)
    return gap


def _read_with_oracle(path: Path) -> dict:
    data = read_h5(path)
    with h5py.File(path, "r") as handle:
        data["common_mode"] = np.asarray(handle["common_mode"][()], dtype=np.float32)
    return data


def iter_records(root: Path, split: str):
    folder = Path(root) / "h5" / split
    halves: dict[str, dict[str, Path]] = {}
    wholes: list[Path] = []
    for path in sorted(folder.glob("*.h5")):
        stem = path.stem
        if stem.endswith("_a") or stem.endswith("_b"):
            halves.setdefault(stem[:-2], {})[stem[-1]] = path
        else:
            wholes.append(path)
    for path in wholes:
        yield path.stem, _read_with_oracle(path)
    for seq_id, parts in sorted(halves.items()):
        if "a" not in parts or "b" not in parts:
            raise FileNotFoundError(f"{seq_id} is missing one time half in {folder}")
        yield seq_id, join_h5(_read_with_oracle(parts["a"]), _read_with_oracle(parts["b"]))


def _alarm(pack: Score, rows, t_unix, x, z, valid, quiet, by_label: dict) -> float | None:
    hot = np.asarray(z) >= 3.5
    cold = np.asarray(z) <= -3.5

    def decide(i0, i1, j0, j1, mapped):
        if mapped is None:
            sl = (slice(i0, i1 + 1), slice(j0, j1 + 1))
            return bool(np.any(hot[sl] | cold[sl]))
        return _signed_hit(hot, cold, i0, i1, j0, j1, mapped)

    _score_file_boxes(pack, rows, t_unix, x, decide)
    for row in rows:
        if row["kind"] != "nuisance":
            continue
        i0, i1, j0, j1 = _locate(t_unix, x, row)
        sl = (slice(i0, i1 + 1), slice(j0, j1 + 1))
        called = bool(np.any(hot[sl] | cold[sl]))
        slot = by_label.setdefault(row["label"], {"hit": 0, "total": 0})
        slot["total"] += 1
        slot["hit"] += int(called)
    for t0, x0 in quiet:
        sl = (slice(t0, t0 + TIME_CROP), slice(x0, x0 + SPACE_CROP))
        pack.quiet(bool(np.any(hot[sl] | cold[sl])))
    pack.alarm_cells += int(np.count_nonzero(hot | cold))
    pack.alarm_total += int(np.asarray(valid, dtype=bool).sum())
    return _zone_count(z, valid, x, t_unix)


def _oracle_field(noisy: np.ndarray, valid: np.ndarray, common_mode: np.ndarray) -> np.ndarray:
    if common_mode.shape[0] != noisy.shape[0]:
        raise ValueError("stored common_mode length does not match the joined traces")
    out = np.array(noisy, dtype=np.float32, copy=True)
    live = np.asarray(valid, dtype=bool) & np.isfinite(out)
    correction = np.broadcast_to(np.asarray(common_mode, dtype=np.float32)[:, None], out.shape)
    out[live] = out[live] - correction[live]
    return out


def _pair(slot: dict | None) -> list[int] | None:
    if not slot:
        return None
    return [int(slot["hit"]), int(slot["total"])]


def _difficulty(pack: dict) -> dict:
    return {key: _pair(value) for key, value in pack["by_difficulty"].items()}


def _sanity(median_pack: dict, mae: dict) -> dict:
    got = {
        "box_hit_total": median_pack["box_hit_total"],
        "Local_Heating": _pair(median_pack["box_hit"].get("Local_Heating")),
        "Local_Cooling": _pair(median_pack["box_hit"].get("Local_Cooling")),
        "Broad_Shift": _pair(median_pack["box_hit"].get("Broad_Shift")),
        "quiet_false_alarm": median_pack["quiet_false_alarm"],
        "nuisance_called_event": median_pack["nuisance_called_event"],
        "zones": median_pack["zones"],
        "alarm_cells": median_pack["alarm_cells"],
        "by_difficulty": _difficulty(median_pack),
    }
    mismatches = {key: [got[key], SANITY[key]] for key in got if got[key] != SANITY[key]}
    raw = mae["raw"]["mae_c"]
    med = mae["median_x3"]["mae_c"]
    cells = mae["median_x3"]["cells"]
    if cells != SANITY["cells"] or raw is None or abs(raw - SANITY["raw_mae_c"]) > 1e-4:
        mismatches["raw_mae"] = [mae["raw"], SANITY["raw_mae_c"]]
    if cells != SANITY["cells"] or med is None or abs(med - SANITY["median_mae_c"]) > 1e-4:
        mismatches["median_mae"] = [mae["median_x3"], SANITY["median_mae_c"]]
    return {"ok": not mismatches, "mismatches": mismatches}


def _resolve_out(limit: int | None, out: Path | None) -> Path:
    if limit is None:
        path = OFFICIAL if out is None else out
    else:
        path = SMOKE if out is None else out
        if path.resolve() == OFFICIAL.resolve():
            raise SystemExit("FAILED smoke must not write the official report")
    if path.name in FORBIDDEN:
        raise SystemExit(f"FAILED refusing to write {path.name}")
    return path


def score(root: Path, limit: int | None, out: Path) -> dict:
    labels = _load_labels(root)
    packs = {"median_x3": Score(), "fast_common_mode_median_x3": Score()}
    by_label = {"median_x3": {}, "fast_common_mode_median_x3": {}}
    mae = {name: _mae_store() for name in ("raw", "median_x3", "fast_common_mode_median_x3", "oracle_median_x3")}
    skipped = []
    sequences = []
    max_abs_dz = 0.0
    fast_abs = 0.0
    fast_n = 0
    fast_max = 0.0
    rng = np.random.default_rng(2)
    zone_errors = []
    for index, (seq_id, record) in enumerate(iter_records(root, "test")):
        if limit is not None and index >= limit:
            break
        if not _increasing(record["t_unix"]) or not _increasing(record["x"]):
            skipped.append(seq_id)
            print(f"skip axis {seq_id}", flush=True)
            continue
        noisy = np.asarray(record["noisy"], dtype=np.float32)
        clean = np.asarray(record["clean"], dtype=np.float32)
        valid = record["valid"]
        mask = record["mask_event"]
        x = record["x"]
        t_unix = record["t_unix"]
        rows = labels.get(seq_id, [])
        quiet = _draw_quiet(record, rng, 12)
        corrected, fast = remove_common_mode(noisy, valid)
        med = median_x(noisy, valid, k=3)
        med_fast = median_x(corrected, valid, k=3)
        oracle = median_x(_oracle_field(noisy, valid, record["common_mode"]), valid, k=3)
        _add_mae(mae["raw"], noisy, clean, valid, mask)
        _add_mae(mae["median_x3"], med, clean, valid, mask)
        _add_mae(mae["fast_common_mode_median_x3"], med_fast, clean, valid, mask)
        _add_mae(mae["oracle_median_x3"], oracle, clean, valid, mask)
        z_med = spatial_z(med, valid, x, inner_m=8, outer_m=40, bin_m=2, floor_c=0.30)[0]
        z_fast = spatial_z(med_fast, valid, x, inner_m=8, outer_m=40, bin_m=2, floor_c=0.30)[0]
        gap = float(np.max(np.abs(z_med.astype(np.float64) - z_fast.astype(np.float64))))
        max_abs_dz = max(max_abs_dz, gap)
        fast_abs += float(np.abs(fast).sum())
        fast_n += int(fast.size)
        fast_max = max(fast_max, float(np.max(np.abs(fast))))
        for name, z in (("median_x3", z_med), ("fast_common_mode_median_x3", z_fast)):
            n_zones = _alarm(packs[name], rows, t_unix, x, z, valid, quiet, by_label[name])
            if n_zones is None:
                zone_errors.append(f"{name} {seq_id}")
            else:
                packs[name].zones += n_zones
                packs[name].zone_files += 1
        nt, nx = noisy.shape
        sequences.append({"seq_id": seq_id, "shape": [int(nt), int(nx)], "quiet": len(quiet), "max_abs_delta_z": gap})
        print(
            f"test {index + 1} {seq_id} {nt}x{nx} quiet {len(quiet)} max_abs_dz {gap:.3e}",
            flush=True,
        )
        del noisy, clean, corrected, med, med_fast, oracle, z_med, z_fast, record
    median_pack = packs["median_x3"].as_dict()
    fast_pack = packs["fast_common_mode_median_x3"].as_dict()
    mae_out = {
        "raw": _mae_dict(mae["raw"]),
        "median_x3": _mae_dict(mae["median_x3"]),
        "fast_common_mode_median_x3": _mae_dict(mae["fast_common_mode_median_x3"]),
        "oracle_median_x3": _mae_dict(
            mae["oracle_median_x3"],
            "noisy minus the stored common_mode, then median_x3. "
            "Connector steps remain. The stored series is not available on a cable and is not the method.",
        ),
    }
    sanity = {"skipped": True} if limit is not None else _sanity(median_pack, mae_out)
    same_boxes = (
        median_pack["box_hit_total"] == fast_pack["box_hit_total"]
        and median_pack["quiet_false_alarm"] == fast_pack["quiet_false_alarm"]
        and median_pack["nuisance_called_event"] == fast_pack["nuisance_called_event"]
        and median_pack["zones"] == fast_pack["zones"]
        and median_pack["alarm_cells"] == fast_pack["alarm_cells"]
    )
    report = {
        "source": "dts_synth_v1",
        "split": "test",
        "limit": limit,
        "operational_pair_unchanged": PAIR,
        "method": {
            "name": "fast_common_mode",
            "steps": (
                "base = nanmedian over time; cm = nanmedian over space of (field - base); "
                "slow = median_filter(cm, size=min(31, (n//2)*2-1), nearest) when n>5; "
                "fast = cm - slow; subtract fast from valid cells; then median_x k=3; "
                "then the same spatial_z_halo."
            ),
            "window_source": "artifacts/dts_synth_v1/generator/calibrate.py lines 44-48. Not tuned on the test boxes.",
            "persistent_jumps": "Jumps of 0.5-3 h can stay in the slow band and remain in the field.",
            "sigma_x": "not used",
            "stored_common_mode": "oracle column only",
        },
        "scorer_ok": sanity.get("ok"),
        "sanity": sanity,
        "max_abs_delta_z": max_abs_dz,
        "boxes_quiet_nuisance_zones_cells_identical": same_boxes,
        "fast_abs_mean_c": (fast_abs / fast_n) if fast_n else None,
        "fast_abs_max_c": fast_max if fast_n else None,
        "mae_c": mae_out,
        "alarms": {
            "median_x3": {**median_pack, "nuisance_by_label": by_label["median_x3"]},
            "fast_common_mode_median_x3": {**fast_pack, "nuisance_by_label": by_label["fast_common_mode_median_x3"]},
        },
        "zone_errors": zone_errors,
        "skipped_time_order": skipped,
        "sequences": sequences,
        "trust_fast_box_score": bool(limit is None and sanity.get("ok")),
    }
    if limit is None and not sanity.get("ok"):
        report["do_not_quote_fast_box_score"] = "reproduced median halo is not the known 74/172 protocol"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the fast common-mode stage on the synth test split.")
    parser.add_argument("--root", type=Path, default=ROOT / "artifacts" / "dts_synth_v1")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    _self_check()
    out = _resolve_out(args.limit, args.out)
    report = score(args.root, args.limit, out)
    med = report["alarms"]["median_x3"]["box_hit_total"]
    fast = report["alarms"]["fast_common_mode_median_x3"]["box_hit_total"]
    print(
        f"DONE scorer_ok {report['scorer_ok']} median {med} fast {fast} "
        f"max_abs_dz {report['max_abs_delta_z']:.3e} wrote {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
