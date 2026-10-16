"""Score temporal_innovation beside spatial_z_halo on the six real CSVs.

Operational status stays median_x3 + spatial_z_halo at |Z| >= 3.5.
temporal_innovation is the causal historical residual (temporal_z). It is
not adopted. Witness windows match scripts/score_channels.py. Zakhar card
boxes use the meters published in docs/DECISIONS.md and the half-open rows
of the cyan boxes those cards were drawn on.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fiber.anomaly.detect import spatial_z, temporal_innovation
from fiber.denoise.dsp import median_x
from fiber.io import Record
from fiber.pipeline import PipelineConfig

Z_THRESHOLD = 3.5
EVIDENCE = ROOT / "docs" / "evidence" / "temporal_innovation.json"
PLOT_DIR = ROOT / "docs" / "images" / "temporal-innovation"
CARDS = (
    {
        "id": "review-00034",
        "record": "1",
        "rows_half_open": (310, 335),
        "x_m": (517.77, 523.25),
        "presence": "yes",
        "polarity": "hot",
        "structure": "one_span",
        "recorded_shape": "Local_Heating",
    },
    {
        "id": "review-00098",
        "record": "1",
        "rows_half_open": (0, 298),
        "x_m": (837.36, 851.98),
        "presence": "yes",
        "polarity": "hot",
        "structure": "one_span",
        "recorded_shape": "Local_Heating",
    },
    {
        "id": "review-00187",
        "record": "1",
        "rows_half_open": (35, 43),
        "x_m": (230.36, 270.36),
        "presence": "yes",
        "polarity": "hot",
        "structure": "several_peaks",
        "recorded_shape": "несколько Local_Heating",
    },
    {
        "id": "review-00223",
        "record": "1",
        "rows_half_open": (234, 294),
        "x_m": (363.15, 403.15),
        "presence": "yes",
        "polarity": "cold",
        "structure": "one_span",
        "recorded_shape": "Local_Cooling",
    },
)


def _witness_window(x_m: np.ndarray, n_rows: int, spec: dict) -> tuple[slice, np.ndarray]:
    r0, r1 = spec["rows"]
    r1 = min(int(r1), n_rows - 1)
    cols = (np.asarray(x_m) >= spec["x_m"][0]) & (np.asarray(x_m) <= spec["x_m"][1])
    return slice(int(r0), r1 + 1), cols


def _pass_witness(role: str, hot_frac: float, cold_frac: float, n_valid: int, alarm_cells: int) -> bool:
    if role == "positive_hot":
        return hot_frac >= 0.3
    if role == "negative_quiet":
        return hot_frac <= 0.1 and cold_frac <= 0.1
    if role == "negative_invalid":
        return n_valid == 0 and alarm_cells == 0
    return False


def _agreement(spatial: np.ndarray, temporal: np.ndarray, valid: np.ndarray) -> dict:
    mask = np.asarray(valid, dtype=bool)
    spatial_alarm = (np.abs(spatial) >= Z_THRESHOLD) & mask
    temporal_alarm = (np.abs(temporal) >= Z_THRESHOLD) & mask
    both = int(np.count_nonzero(spatial_alarm & temporal_alarm))
    spatial_only = int(np.count_nonzero(spatial_alarm & ~temporal_alarm))
    temporal_only = int(np.count_nonzero(~spatial_alarm & temporal_alarm))
    neither = int(np.count_nonzero(mask & ~spatial_alarm & ~temporal_alarm))
    n_valid = int(mask.sum())
    return {
        "n_valid": n_valid,
        "both": both,
        "spatial_only": spatial_only,
        "temporal_only": temporal_only,
        "neither": neither,
        "spatial_alarm_fraction": (both + spatial_only) / n_valid if n_valid else None,
        "temporal_alarm_fraction": (both + temporal_only) / n_valid if n_valid else None,
    }


def _wide_runs(mask: np.ndarray, x_m: np.ndarray, min_width_m: float) -> int:
    """Runs at least as wide as the operational zone gate. Shorter speckle is not a second span."""
    if mask.size == 0 or not mask.any():
        return 0
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    x = np.asarray(x_m, dtype=np.float64)
    count = 0
    for start, end in zip(starts, ends):
        if float(x[end - 1] - x[start]) >= min_width_m:
            count += 1
    return count


def _peak_count(profile: np.ndarray, min_prominence: float) -> int:
    y = np.asarray(profile, dtype=np.float64)
    finite = np.isfinite(y)
    idx = np.flatnonzero(finite)
    if idx.size < 3:
        return 0
    count = 0
    for k in range(1, idx.size - 1):
        i = idx[k]
        left_i = idx[k - 1]
        right_i = idx[k + 1]
        if y[i] < y[left_i] or y[i] < y[right_i]:
            continue
        if y[i] == y[left_i] and y[i] == y[right_i]:
            continue
        left = y[idx[:k]]
        right = y[idx[k + 1 :]]
        higher_left = np.flatnonzero(left > y[i])
        higher_right = np.flatnonzero(right > y[i])
        left_slice = left[higher_left[-1] + 1 :] if higher_left.size else left
        right_slice = right[: higher_right[0]] if higher_right.size else right
        if left_slice.size == 0 or right_slice.size == 0:
            continue
        prominence = float(y[i] - max(np.min(left_slice), np.min(right_slice)))
        if prominence >= min_prominence:
            count += 1
    return count


def _card_row(z: np.ndarray, clean: np.ndarray, background: np.ndarray, valid: np.ndarray, x_m: np.ndarray, spec: dict) -> dict:
    r0, r1 = spec["rows_half_open"]
    sl = slice(int(r0), int(r1))
    cols = (np.asarray(x_m) >= spec["x_m"][0]) & (np.asarray(x_m) <= spec["x_m"][1])
    window_valid = valid[sl][:, cols]
    contrast = np.asarray(clean[sl][:, cols], dtype=np.float64) - np.asarray(background[sl][:, cols], dtype=np.float64)
    contrast = np.where(window_valid, contrast, np.nan)
    temperature = np.where(window_valid, np.asarray(clean[sl][:, cols], dtype=np.float64), np.nan)
    n_valid = int(window_valid.sum())
    window_z = z[sl][:, cols]
    hot_cells = int(np.count_nonzero((window_z >= Z_THRESHOLD) & window_valid))
    cold_cells = int(np.count_nonzero((window_z <= -Z_THRESHOLD) & window_valid))
    med_contrast = float(np.nanmedian(contrast)) if n_valid else None
    along = np.nanmedian(contrast, axis=0)
    temp_along = np.nanmedian(temperature, axis=0)
    sign = 1.0 if spec["polarity"] == "hot" else -1.0
    on_side = np.isfinite(along) & (sign * along >= 0.30)
    n_runs = _wide_runs(on_side, x_m[cols], PipelineConfig().min_width_m)
    n_peaks = _peak_count(temp_along, 1.0)
    polarity_ok = med_contrast is not None and sign * med_contrast >= 0.30
    if spec["structure"] == "one_span":
        structure_ok = n_runs == 1
    else:
        structure_ok = n_peaks >= 3
    presence_ok = spec["presence"] == "yes" and n_valid > 0
    return {
        "id": spec["id"],
        "record": spec["record"],
        "presence": spec["presence"],
        "recorded_shape": spec["recorded_shape"],
        "polarity": spec["polarity"],
        "structure": spec["structure"],
        "rows_half_open": [int(r0), int(r1)],
        "x_m": [spec["x_m"][0], spec["x_m"][1]],
        "n_valid": n_valid,
        "hot_cells": hot_cells,
        "cold_cells": cold_cells,
        "median_spatial_residual_c": None if med_contrast is None else round(med_contrast, 6),
        "residual_runs": n_runs,
        "temperature_peaks": n_peaks,
        "presence_ok": presence_ok,
        "polarity_ok": polarity_ok,
        "structure_ok": structure_ok,
        "pass": bool(presence_ok and polarity_ok and structure_ok),
        "expert_confirmed": True,
    }


def _plot(files: list[dict]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    names = [row["record"] for row in files]
    x = np.arange(len(names))
    width = 0.22
    fig, ax = plt.subplots(figsize=(9, 4.5))
    both = [row["agreement"]["both"] for row in files]
    spatial_only = [row["agreement"]["spatial_only"] for row in files]
    temporal_only = [row["agreement"]["temporal_only"] for row in files]
    ax.bar(x - width, both, width, label="both")
    ax.bar(x, spatial_only, width, label="spatial_only")
    ax.bar(x + width, temporal_only, width, label="temporal_only")
    ax.set_xticks(x, [f"{name}.csv" for name in names])
    ax.set_ylabel("valid cells")
    ax.set_title("temporal_innovation vs spatial_z_halo at |Z| >= 3.5")
    ax.legend()
    fig.tight_layout()
    agreement_path = PLOT_DIR / "agreement_by_file.png"
    fig.savefig(agreement_path, dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    spatial_frac = [row["agreement"]["spatial_alarm_fraction"] for row in files]
    temporal_frac = [row["agreement"]["temporal_alarm_fraction"] for row in files]
    ax.bar(x - width / 2, spatial_frac, width, label="spatial_z_halo")
    ax.bar(x + width / 2, temporal_frac, width, label="temporal_innovation")
    ax.set_xticks(x, [f"{name}.csv" for name in names])
    ax.set_ylabel("alarm fraction of valid cells")
    ax.set_title("Alarm fraction, same |Z| >= 3.5, temporal channel not adopted")
    ax.legend()
    fig.tight_layout()
    fraction_path = PLOT_DIR / "alarm_fraction_by_file.png"
    fig.savefig(fraction_path, dpi=120)
    plt.close(fig)
    return [
        "docs/images/temporal-innovation/agreement_by_file.png",
        "docs/images/temporal-innovation/alarm_fraction_by_file.png",
    ]


def score(records: Path) -> dict:
    cfg = PipelineConfig()
    witness_payload = json.loads((ROOT / "fiber" / "benchmark" / "witnesses.json").read_text(encoding="utf-8"))
    labels = {
        json.loads(line)["id"]: json.loads(line)
        for line in (ROOT / "docs" / "evidence" / "zakhar_shortlist_2026-10-10.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    by_record: dict[str, list[dict]] = {}
    for spec in witness_payload["witnesses"]:
        by_record.setdefault(spec["record"], []).append(spec)
    files = []
    witness_rows = []
    cached: dict[str, tuple] = {}
    for name in ("1", "2", "3", "4", "5", "6"):
        started = time.perf_counter()
        record = Record.load(records / name)
        clean = median_x(record.raw_c, record.valid)
        z, background, _scale = spatial_z(
            clean,
            record.valid,
            record.x_m,
            inner_m=cfg.inner_m,
            outer_m=cfg.outer_m,
            bin_m=cfg.bin_m,
            floor_c=cfg.floor_c,
        )
        historical = temporal_innovation(
            clean,
            record.valid,
            record.time_s,
            max_gap_s=cfg.max_gap_s,
            floor_c=cfg.floor_c,
        )
        agreement = _agreement(z, historical, record.valid)
        elapsed = round(time.perf_counter() - started, 2)
        files.append({
            "record": name,
            "source_sha256": record.source_sha256,
            "shape": [int(record.raw_c.shape[0]), int(record.raw_c.shape[1])],
            "elapsed_s": elapsed,
            "agreement": agreement,
        })
        print(
            f"file {name}.csv finished valid={agreement['n_valid']} "
            f"both={agreement['both']} spatial_only={agreement['spatial_only']} "
            f"temporal_only={agreement['temporal_only']} neither={agreement['neither']} "
            f"elapsed_s={elapsed}",
            flush=True,
        )
        for spec in by_record.get(name, []):
            if record.source_sha256 != spec["source_sha256"]:
                raise ValueError(f"{name} sha changed; witnesses do not match this CSV")
            sl, cols = _witness_window(record.x_m, z.shape[0], spec)
            window_valid = record.valid[sl][:, cols]
            n_valid = int(window_valid.sum())
            hot_frac = float(((z[sl][:, cols] >= Z_THRESHOLD) & window_valid).sum() / n_valid) if n_valid else 0.0
            cold_frac = float(((z[sl][:, cols] <= -Z_THRESHOLD) & window_valid).sum() / n_valid) if n_valid else 0.0
            alarm_cells = int(np.count_nonzero((np.abs(z[sl][:, cols]) >= Z_THRESHOLD) & window_valid))
            witness_rows.append({
                "id": spec["id"],
                "record": name,
                "role": spec["role"],
                "valid_cells": n_valid,
                "hot_fraction": round(hot_frac, 4),
                "cold_fraction": round(cold_frac, 4),
                "alarm_cells": alarm_cells,
                "pass": _pass_witness(spec["role"], hot_frac, cold_frac, n_valid, alarm_cells),
                "expert_confirmed": False,
            })
        if name == "1":
            cached["1"] = (z, clean, background, record.valid, record.x_m, record.source_sha256)
        del clean, z, background, historical, record

    cards = []
    z, clean, background, valid, x_m, sha = cached["1"]
    for spec in CARDS:
        label = labels[spec["id"]]
        if label["presence"] != spec["presence"] or label["shape"] != spec["recorded_shape"]:
            raise ValueError(f"{spec['id']} no longer matches the committed shortlist")
        if sha != label["source_sha256"]:
            raise ValueError("1.csv sha does not match the Zakhar shortlist")
        row = _card_row(z, clean, background, valid, x_m, spec)
        cards.append(row)
        print(
            f"card {row['id']} pass={row['pass']} residual={row['median_spatial_residual_c']} "
            f"runs={row['residual_runs']} peaks={row['temperature_peaks']} "
            f"hot_cells={row['hot_cells']} cold_cells={row['cold_cells']}",
            flush=True,
        )
    n_pass = int(sum(row["pass"] for row in witness_rows))
    plots = _plot(files)
    return {
        "schema": "temporal-innovation-v1",
        "operational": {
            "clean": "median_x3",
            "detector": "spatial_z_halo",
            "threshold_z": cfg.threshold_z,
            "inner_m": cfg.inner_m,
            "outer_m": cfg.outer_m,
            "bin_m": cfg.bin_m,
            "floor_c": cfg.floor_c,
            "min_width_m": cfg.min_width_m,
            "min_profiles": cfg.min_profiles,
            "min_cells": cfg.min_cells,
            "max_gap_s": cfg.max_gap_s,
            "adopted_temporal_innovation": False,
            "status_reads_temporal_innovation": False,
        },
        "channel": {
            "name": "temporal_innovation",
            "function": "fiber.anomaly.detect.temporal_innovation",
            "same_as": "temporal_z",
            "window": 8,
            "min_past": 3,
            "max_gap_s": cfg.max_gap_s,
            "floor_c": cfg.floor_c,
            "causal": True,
            "alarm_compare_abs_z": Z_THRESHOLD,
            "adopted": False,
        },
        "files": files,
        "witnesses": {
            "n_pass": n_pass,
            "n_witnesses": len(witness_rows),
            "expert_confirmed": False,
            "rule": "scripts/score_channels.py hot>=0.3, quiet hot and cold <=0.1, dead tail silent",
            "rows": witness_rows,
        },
        "zakhar_cards": {
            "n_pass": int(sum(row["pass"] for row in cards)),
            "n_cards": len(cards),
            "rows": cards,
        },
        "plots": plots,
        "note": "8/8 localises the development rectangles. It is not class accuracy and does not adopt temporal_innovation.",
    }


def main() -> None:
    report = score(ROOT / "artifacts" / "records")
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", EVIDENCE, flush=True)
    print(
        f"witnesses {report['witnesses']['n_pass']}/{report['witnesses']['n_witnesses']} "
        f"cards {report['zakhar_cards']['n_pass']}/{report['zakhar_cards']['n_cards']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
