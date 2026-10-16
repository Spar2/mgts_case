"""Draw the threshold and refine figures and splice the report from the JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fiber.benchmark.threshold_cal import (  # noqa: E402
    render_decision_entry,
    render_threshold_section,
    splice_markdown,
)

CAL_PATH = ROOT / "docs" / "evidence" / "threshold_cal.json"
HYBRID_PATH = ROOT / "docs" / "evidence" / "hybrid_refine.json"
IMAGE_DIR = ROOT / "docs" / "images" / "ssl"


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _series(report: dict):
    for row in report["rows"]:
        curve = row.get("curve_holdout") or []
        if not curve:
            continue
        label = f"{row['method']} {row['scale']}"
        yield label, curve


def _plot_curves(report: dict) -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    specs = (
        ("hot_iou", "hot IoU", "iou_vs_threshold_hot.png"),
        ("cold_iou", "cold IoU", "iou_vs_threshold_cold.png"),
        ("far", "FAR", "far_vs_threshold.png"),
    )
    for key, ylabel, name in specs:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for label, curve in _series(report):
            xs = [point["threshold"] for point in curve]
            ys = [np.nan if point.get(key) is None else point[key] for point in curve]
            ax.plot(xs, ys, label=label)
        ax.axvline(3.5, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("threshold")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(IMAGE_DIR / name, dpi=120)
        plt.close(fig)


def _plot_refine(hybrid: dict) -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    modes = list(hybrid.get("modes", {}))
    if not modes:
        return
    amps = []
    for mode in modes:
        grouped = hybrid["modes"][mode].get("errors", {}).get("by_amp", {})
        amps = list(grouped)
        break
    if amps:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        index = np.arange(len(amps))
        width = 0.8 / max(1, len(modes) * 2)
        for offset, mode in enumerate(modes):
            grouped = hybrid["modes"][mode]["errors"]["by_amp"]
            refined = [np.nan if grouped[amp]["refined"]["peak_error_c"] is None else grouped[amp]["refined"]["peak_error_c"] for amp in amps]
            baseline = [np.nan if grouped[amp]["baseline"]["peak_error_c"] is None else grouped[amp]["baseline"]["peak_error_c"] for amp in amps]
            ax.bar(index + offset * 2 * width, baseline, width, label=f"{mode} median_x3")
            ax.bar(index + offset * 2 * width + width, refined, width, label=f"{mode} refined")
        ax.set_xticks(index + width)
        ax.set_xticklabels(amps)
        ax.set_xlabel("amplitude, °C")
        ax.set_ylabel("peak error, °C")
        ax.set_title("peak contrast error")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(IMAGE_DIR / "refine_peak_error_by_amp.png", dpi=120)
        plt.close(fig)
    ramps = []
    for mode in modes:
        ramps = list(hybrid["modes"][mode].get("errors", {}).get("by_ramp", {}))
        break
    if ramps:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        index = np.arange(len(ramps))
        width = 0.8 / max(1, len(modes) * 2)
        for offset, mode in enumerate(modes):
            grouped = hybrid["modes"][mode]["errors"]["by_ramp"]
            refined = [np.nan if grouped[ramp]["refined"]["x_error_m"] is None else grouped[ramp]["refined"]["x_error_m"] for ramp in ramps]
            baseline = [np.nan if grouped[ramp]["baseline"]["x_error_m"] is None else grouped[ramp]["baseline"]["x_error_m"] for ramp in ramps]
            ax.bar(index + offset * 2 * width, baseline, width, label=f"{mode} median_x3")
            ax.bar(index + offset * 2 * width + width, refined, width, label=f"{mode} refined")
        ax.set_xticks(index + width)
        ax.set_xticklabels(ramps)
        ax.set_xlabel("ramp, samples")
        ax.set_ylabel("x boundary error, m")
        ax.set_title("boundary error by ramp")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(IMAGE_DIR / "refine_boundary_by_ramp.png", dpi=120)
        plt.close(fig)
    crops = []
    for mode in modes:
        real = hybrid["modes"][mode].get("real") or {}
        for item in real.get("largest_shifts") or []:
            if item.get("crop_fine"):
                crops.append((mode, item))
    crops = crops[:6]
    if not crops:
        return
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    for ax, (mode, item) in zip(axes.ravel(), crops, strict=False):
        values = [[np.nan if value is None else float(value) for value in row] for row in item["crop_fine"]]
        ax.imshow(np.asarray(values, dtype=float), aspect="auto", cmap="coolwarm")
        ax.set_title(f"{mode} {item['file']} {item['shift_m']:.2f} m", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.ravel()[len(crops):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "refine_largest_shifts.png", dpi=120)
    plt.close(fig)


def main() -> None:
    report = _load(CAL_PATH)
    hybrid = _load(HYBRID_PATH)
    _plot_curves(report)
    _plot_refine(hybrid)
    section = render_threshold_section(report, hybrid)
    splice_markdown(ROOT / "docs" / "SSL_BENCHMARK.md", "## Порог под денойзер и гибридное уточнение (после d288333)", section)
    entry = render_decision_entry(report, hybrid)
    splice_markdown(ROOT / "docs" / "DECISIONS.md", "## 2026-10-13 — порог под денойзер и гибридное уточнение", entry)
    print("wrote curves and report", flush=True)


if __name__ == "__main__":
    main()
