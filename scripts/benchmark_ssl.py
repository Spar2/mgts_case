"""Score research denoisers and detectors against frozen median_x3 and spatial Z.

`--cpu` forces the CPU. Without it, a research network uses CUDA when the
machine has it. The operational pair is not changed.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fiber.anomaly.detect import spatial_z
from fiber.anomaly.rpca import rpca_score
from fiber.anomaly.threshold import holdout_report
from fiber.anomaly.zones import extract_zones
from fiber.benchmark.channels import (
    RESEARCH_CHANNEL_METHODS,
    Z_THRESHOLD,
    score_half_cable_channel,
    score_local_channel,
)
from fiber.benchmark.events import evaluate_events
from fiber.benchmark.synthetic import half_cable_shift, local_scene
from fiber.denoise import apply_method
from fiber.denoise.score import band_contrast, synthetic_denoise_case, synthetic_wide_case
from fiber.io import Record
from fiber.metrics import spike_retention

MAD_BAR = {"small": 0.233, "wide": 0.239}
SPIKE_MIN = 0.92
CONTRAST_FLOOR = 0.95
REPORT_JSON = ROOT / "docs" / "evidence" / "ssl_benchmark.json"
REPORT_MD = ROOT / "docs" / "SSL_BENCHMARK.md"
PAIR = "median_x3 + spatial_z_halo at |Z| >= 3.5"


def _round(value):
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return round(value, 6)
    return value


def _iou(block: dict):
    value = block.get("support_iou") if isinstance(block, dict) else None
    return None if value is None else _round(float(value))


def _ge(left, right) -> bool:
    if left is None or right is None:
        return False
    return float(left) >= float(right) - 1e-9


def _le(left, right) -> bool:
    if left is None or right is None:
        return False
    return float(left) <= float(right) + 1e-9


def _synthetic_row(name: str, case: dict, device: str | None) -> dict:
    out = apply_method(name, case["noisy"], case["valid"], device=device)
    row = case["clean"].shape[0] // 2
    keep = spike_retention(
        np.where(case["valid"][row], case["clean"][row], np.nan),
        np.where(case["valid"][row], out[row], np.nan),
        peak_index=case["peak_index"],
        half_width=2,
    )
    noise = out[case["valid"]] - case["clean"][case["valid"]]
    mad = float(np.median(np.abs(noise - np.median(noise))) * 1.4826)
    return {
        "error_mad_c": _round(mad),
        "spike_retention": _round(float(keep)),
        "dead_finite": int(np.isfinite(out[:, case["dead_from"]:]).sum()),
    }


def _band(name: str, record: Record, device: str | None) -> float | None:
    out = apply_method(name, record.raw_c, record.valid, device=device)
    value = band_contrast(out, record.valid, record.x_m)
    if not np.isfinite(value):
        return None
    return _round(float(value))


def denoiser_status(row: dict) -> str:
    if row.get("missing"):
        return "не принят"
    small = row.get("small") or {}
    wide = row.get("wide") or {}
    needed = (
        small.get("error_mad_c"),
        wide.get("error_mad_c"),
        small.get("spike_retention"),
        wide.get("spike_retention"),
        row.get("band_contrast_c"),
        row.get("median_band_contrast_c"),
        row.get("hot_support_iou"),
        row.get("cold_support_iou"),
        row.get("median_hot_support_iou"),
        row.get("median_cold_support_iou"),
        row.get("dead_tail_alarms"),
    )
    if any(item is None for item in needed):
        return "не принят"
    ok = (
        small["error_mad_c"] < MAD_BAR["small"]
        and wide["error_mad_c"] < MAD_BAR["wide"]
        and small["spike_retention"] >= SPIKE_MIN
        and wide["spike_retention"] >= SPIKE_MIN
        and row["band_contrast_c"] >= CONTRAST_FLOOR * row["median_band_contrast_c"]
        and _ge(row["hot_support_iou"], row["median_hot_support_iou"])
        and _ge(row["cold_support_iou"], row["median_cold_support_iou"])
        and int(row["dead_tail_alarms"]) == 0
        and int(small["dead_finite"]) == 0
        and int(wide["dead_finite"]) == 0
    )
    return "кандидат" if ok else "не принят"


def detector_status(row: dict) -> str:
    if row.get("missing"):
        return "не принят"
    needed = (
        row.get("local_hot_iou"),
        row.get("local_cold_iou"),
        row.get("half_iou"),
        row.get("alarm_fraction_inside"),
        row.get("baseline_local_hot_iou"),
        row.get("baseline_local_cold_iou"),
        row.get("baseline_half_iou"),
        row.get("baseline_alarm_fraction_inside"),
    )
    if any(item is None for item in needed):
        return "не принят"
    ok = (
        _ge(row["local_hot_iou"], row["baseline_local_hot_iou"])
        and _ge(row["local_cold_iou"], row["baseline_local_cold_iou"])
        and _ge(row["half_iou"], row["baseline_half_iou"])
        and _le(row["alarm_fraction_inside"], row["baseline_alarm_fraction_inside"])
    )
    return "кандидат" if ok else "не принят"


def _alarm_iou(truth: np.ndarray, alarm: np.ndarray, scene: dict):
    block = evaluate_events(truth, alarm, scene["time_s"], scene["x_m"], scene["valid"])
    value = block.get("support_iou")
    return None if value is None else _round(float(value))


def _zones(score: np.ndarray, valid: np.ndarray, x_m: np.ndarray, time_s: np.ndarray) -> int:
    iso = [str(index) for index in range(len(time_s))]
    found = extract_zones(
        score, valid, x_m, iso, time_s,
        thresh=Z_THRESHOLD, min_width_m=3, min_profiles=6, min_cells=24, max_gap_s=300,
    )
    return len(found)


def _detector_from_score(
    name: str,
    local_score: np.ndarray,
    half_score: np.ndarray,
    local: dict,
    half: dict,
    baseline: dict,
) -> dict:
    hot = local_score >= Z_THRESHOLD
    cold = local_score <= -Z_THRESHOLD
    half_alarm = np.abs(half_score) >= Z_THRESHOLD
    row = {
        "name": name,
        "local_hot_iou": _alarm_iou(local["hot"], hot, local),
        "local_cold_iou": _alarm_iou(local["cold"], cold, local),
        "half_iou": _alarm_iou(half["broad"], half_alarm, half),
        "alarm_fraction_inside": _round(float(half_alarm[half["broad"]].mean()) if half["broad"].any() else 0.0),
        "zones": _zones(local_score, local["valid"], local["x_m"], local["time_s"]),
        "baseline_local_hot_iou": baseline["local_hot_iou"],
        "baseline_local_cold_iou": baseline["local_cold_iou"],
        "baseline_half_iou": baseline["half_iou"],
        "baseline_alarm_fraction_inside": baseline["alarm_fraction_inside"],
        "missing": None,
    }
    row["status"] = detector_status(row)
    return row


def alarm_on_half(score: np.ndarray, half: dict) -> np.ndarray:
    if score.shape != half["clean"].shape:
        raise ValueError("detector score shape does not match the half-cable scene")
    return np.abs(score) >= Z_THRESHOLD


def _baseline_detector(local: dict, half: dict) -> dict:
    z_local = spatial_z(apply_method("median_x3", local["clean"], local["valid"]), local["valid"], local["x_m"])[0]
    z_half = spatial_z(apply_method("median_x3", half["clean"], half["valid"]), half["valid"], half["x_m"])[0]
    half_alarm = np.abs(z_half) >= Z_THRESHOLD
    inside = half["broad"]
    return {
        "name": "spatial_z_halo",
        "clean_channel": "median_x3",
        "threshold_z": Z_THRESHOLD,
        "local_hot_iou": _alarm_iou(local["hot"], z_local >= Z_THRESHOLD, local),
        "local_cold_iou": _alarm_iou(local["cold"], z_local <= -Z_THRESHOLD, local),
        "half_iou": _alarm_iou(half["broad"], half_alarm, half),
        "alarm_fraction_inside": _round(float(half_alarm[inside].mean()) if inside.any() else 0.0),
        "zones": _zones(z_local, local["valid"], local["x_m"], local["time_s"]),
        "status": "baseline",
        "missing": None,
    }


def _missing(name: str, message: str, baseline: dict) -> dict:
    return {
        "name": name,
        "missing": message,
        "status": "не принят",
        "baseline_local_hot_iou": baseline.get("local_hot_iou"),
        "baseline_local_cold_iou": baseline.get("local_cold_iou"),
        "baseline_half_iou": baseline.get("half_iou"),
        "baseline_alarm_fraction_inside": baseline.get("alarm_fraction_inside"),
    }


def _annotation_note() -> dict:
    candidates = [
        ROOT / "artifacts" / "annotation_queue.json",
        ROOT / "docs" / "evidence" / "annotation_queue.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else payload.get("rows", [])
        labeled = [row for row in rows if row.get("expert_label") not in (None, "")]
        if not labeled:
            return {"labels_present": False, "note": f"{path.name} has no expert_label"}
        return {"labels_present": True, "labeled": len(labeled), "note": "precision/recall not computed: queue schema has no matched alarm cells"}
    return {"labels_present": False, "note": "очередь разметки без экспертных меток; precision/recall против spatial Z не считается"}


def _markdown(report: dict) -> str:
    lines = [
        "# Research SSL benchmark",
        "",
        f"Рабочая пара не сменена: `{report['operational_pair']}`",
        "",
        "Денойзер — «кандидат», только если MAD на 12×80 ниже 0.233 °C и на 48×160 ниже 0.239 °C, удержание спайка на обоих полях не меньше 0.92, контраст на `5.csv` не ниже `median_x3` больше чем на 5%, hot/cold IoU при `|Z| >= 3.5` не хуже `median_x3`, и `dead_tail_alarms = 0`. Иначе «не принят».",
        "",
        "Детектор — «кандидат», только если IoU на `local_scene` и `half_cable_shift` не ниже `spatial_z_halo`, а `alarm_fraction_inside` на полукабельном сдвиге не выше. Иначе «не принят».",
        "",
        f"Устройство прогона: `{report['device']}`.",
        "",
        "## Денойз",
        "",
        "| метод | статус | MAD 12×80 | MAD 48×160 | keep 12×80 | keep 48×160 | контраст 5.csv | dead_tail |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in report["denoisers"].items():
        small = row.get("small") or {}
        wide = row.get("wide") or {}
        lines.append(
            f"| `{name}` | {row.get('status')} | {small.get('error_mad_c', '—')} | {wide.get('error_mad_c', '—')} | "
            f"{small.get('spike_retention', '—')} | {wide.get('spike_retention', '—')} | "
            f"{row.get('band_contrast_c', '—')} | {row.get('dead_tail_alarms', '—')} |"
        )
        if row.get("missing"):
            lines.append(f"|  | {row['missing']} | | | | | | |")
    lines.extend([
        "",
        "## Детекторы",
        "",
        "| метод | статус | hot IoU | cold IoU | half IoU | alarm_fraction_inside |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for name, row in report["detectors"].items():
        lines.append(
            f"| `{name}` | {row.get('status')} | {row.get('local_hot_iou', '—')} | {row.get('local_cold_iou', '—')} | "
            f"{row.get('half_iou', '—')} | {row.get('alarm_fraction_inside', '—')} |"
        )
        if row.get("missing"):
            lines.append(f"|  | {row['missing']} | | | | |")
    chronos = report["detectors"].get("chronos_bolt", {})
    levels = chronos.get("quantile_levels")
    if levels:
        lines.extend([
            "",
            f"Chronos-Bolt: запрошены квантили 0.05/0.95, в счёт пошли {levels[0]} и {levels[1]}.",
        ])
    note = report["annotation_queue"]["note"]
    lines.extend(["", "## Очередь разметки", "", note, ""])
    return "\n".join(lines)


def _score_denoiser(name: str, device: str | None, record: Record | None, median: dict, local: dict) -> dict:
    try:
        small = _synthetic_row(name, synthetic_denoise_case(seed=0), device)
        wide = _synthetic_row(name, synthetic_wide_case(seed=0), device)
        contrast = _band(name, record, device) if record is not None else None
        local_score = score_local_channel(name, local, device=device)
        row = {
            "name": name,
            "family": "research" if name != "median_x3" else "standard",
            "small": small,
            "wide": wide,
            "band_contrast_c": contrast,
            "median_band_contrast_c": median.get("band_contrast_c"),
            "hot_support_iou": _iou(local_score["hot"]),
            "cold_support_iou": _iou(local_score["cold"]),
            "median_hot_support_iou": median.get("hot_support_iou"),
            "median_cold_support_iou": median.get("cold_support_iou"),
            "dead_tail_alarms": int(local_score["dead_tail_alarms"]),
            "missing": None if record is not None else "5.csv record is missing; band_contrast was not invented",
        }
        if name == "blindspot_x" and not (ROOT / "models" / "weights" / "blindspot_x.pt").exists():
            row["note"] = "weights missing; seeded forward pass, not an MGTS fit"
        row["status"] = "baseline" if name == "median_x3" else denoiser_status(row)
        return row
    except Exception as exc:
        return {
            "name": name,
            "missing": f"{name}: {exc}",
            "status": "не принят",
            "trace": traceback.format_exc().splitlines()[-1],
        }


def run(device: str | None) -> dict:
    local = local_scene(seed=0)
    half = half_cable_shift(seed=1)
    record = None
    record_path = ROOT / "artifacts" / "records" / "5"
    if (record_path / "raw_c.npy").exists():
        record = Record.load(record_path)
    median_local = score_local_channel("median_x3", local, device=device)
    median = {
        "name": "median_x3",
        "small": _synthetic_row("median_x3", synthetic_denoise_case(seed=0), device),
        "wide": _synthetic_row("median_x3", synthetic_wide_case(seed=0), device),
        "band_contrast_c": _band("median_x3", record, device) if record is not None else None,
        "hot_support_iou": _iou(median_local["hot"]),
        "cold_support_iou": _iou(median_local["cold"]),
        "dead_tail_alarms": int(median_local["dead_tail_alarms"]),
        "missing": None if record is not None else "5.csv record is missing; band_contrast was not invented",
        "status": "baseline",
        "family": "standard",
    }
    denoisers = {"median_x3": median}
    for name in RESEARCH_CHANNEL_METHODS:
        print(f"denoise {name}", flush=True)
        denoisers[name] = _score_denoiser(name, device, record, median, local)
        print(f"  {denoisers[name].get('status')}", flush=True)
    baseline = _baseline_detector(local, half)
    detectors = {"spatial_z_halo": baseline}
    print("detect rpca", flush=True)
    try:
        local_score, _low, _sparse = rpca_score(local["clean"], local["valid"])
        half_score, _low_h, _sparse_h = rpca_score(half["clean"], half["valid"])
        detectors["rpca"] = _detector_from_score("rpca", local_score, half_score, local, half, baseline)
    except Exception as exc:
        detectors["rpca"] = _missing("rpca", f"rpca: {exc}", baseline)
    print("detect matrix_profile", flush=True)
    try:
        from fiber.anomaly.matrix_profile import discord_score

        local_mp = discord_score(local["clean"], local["valid"], window=8)
        half_mp = discord_score(half["clean"], half["valid"], window=8)
        detectors["matrix_profile"] = _detector_from_score("matrix_profile", local_mp, half_mp, local, half, baseline)
    except ImportError as exc:
        detectors["matrix_profile"] = _missing("matrix_profile", f"stumpy is not installed: {exc}", baseline)
    except Exception as exc:
        detectors["matrix_profile"] = _missing("matrix_profile", f"matrix_profile: {exc}", baseline)
    print("detect chronos_bolt", flush=True)
    try:
        from fiber.anomaly.residual_forecast import forecast_score

        local_fc = forecast_score(local["clean"], local["valid"], context=16, device=device, column_stride=4)
        half_fc = forecast_score(half["clean"], half["valid"], context=12, device=device, column_stride=4)
        detectors["chronos_bolt"] = _detector_from_score("chronos_bolt", local_fc, half_fc, local, half, baseline)
        levels = getattr(forecast_score, "emitted_quantile_levels", None)
        if levels is not None:
            detectors["chronos_bolt"]["requested_quantile_levels"] = [0.05, 0.95]
            detectors["chronos_bolt"]["quantile_levels"] = [float(levels[0]), float(levels[1])]
    except ImportError as exc:
        detectors["chronos_bolt"] = _missing("chronos_bolt", f"chronos-forecasting is not installed: {exc}", baseline)
    except Exception as exc:
        detectors["chronos_bolt"] = _missing("chronos_bolt", f"chronos_bolt: {exc}", baseline)
    print("detect patchcore", flush=True)
    try:
        from fiber.anomaly.image_ad import image_anomaly_score

        local_ad = image_anomaly_score(local["clean"], local["valid"], local["x_m"], tile=16, device=device)
        half_ad = image_anomaly_score(half["clean"], half["valid"], half["x_m"], tile=8, device=device)
        detectors["patchcore"] = _detector_from_score("patchcore", local_ad, half_ad, local, half, baseline)
    except ImportError as exc:
        detectors["patchcore"] = _missing("patchcore", f"anomalib is not installed: {exc}", baseline)
    except Exception as exc:
        detectors["patchcore"] = _missing("patchcore", f"patchcore: {exc}", baseline)
    print("threshold", flush=True)
    noise = np.random.default_rng(0).normal(0.0, 1.0, size=20000)
    threshold = holdout_report(noise, 0.05, seed=1)
    threshold["status"] = "не принят"
    threshold["note"] = "research report only; operational |Z| stays 3.5"
    report = {
        "operational_pair": PAIR,
        "operational_pair_unchanged": True,
        "witness_promotion": False,
        "device": device or "auto",
        "baselines": {"denoise": "median_x3", "detector": "spatial_z_halo", "threshold_z": Z_THRESHOLD},
        "mad_bars_c": MAD_BAR,
        "spike_retention_min": SPIKE_MIN,
        "denoisers": denoisers,
        "detectors": detectors,
        "threshold": threshold,
        "annotation_queue": _annotation_note(),
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT_MD.write_text(_markdown(report), encoding="utf-8")
    print(f"wrote {REPORT_JSON}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Score research SSL denoisers and unsupervised detectors.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even when CUDA is available.")
    args = parser.parse_args()
    device = "cpu" if args.cpu else None
    run(device)


if __name__ == "__main__":
    main()
