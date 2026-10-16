"""Адаптер эталонного DSP: fiber.pipeline.analyze_field."""
from __future__ import annotations

import time

from fiber.pipeline import PipelineConfig, analyze_field

from ml_common.events import build_events, code_sha
from ml_common.protocol import normalize_field


class FiberAnalyzer:
    method = "fiber"

    def predict(self, field) -> dict:
        spec = normalize_field(field)
        started = time.perf_counter()
        result = analyze_field(
            spec["raw"],
            spec["x_m"],
            spec["time_s"],
            valid=spec["valid"],
            time_iso=spec["time_iso"],
        )
        zones = []
        for zone in result["zones"]:
            label = str(zone.get("shape_hypothesis") or "unknown")
            item = {
                "t_first": str(zone.get("t_start") or ""),
                "t_last": str(zone.get("t_end") or ""),
                "x_from_m": float(zone["start_m"]),
                "x_to_m": float(zone["end_m"]),
                "score": float(zone.get("peak_z", 0.0)),
                "stage2_type": label,
                "type_proba": {},
                "amplitude_c": float(zone.get("median_contrast_c", zone.get("peak_contrast_c", 0.0))),
            }
            if zone.get("polarity"):
                item["polarity"] = str(zone["polarity"])
            zones.append(item)
        zones.sort(key=lambda item: abs(item["score"]), reverse=True)
        threshold = float(result["manifest"]["config"].get("threshold_z", PipelineConfig().threshold_z))
        return build_events(
            method=self.method,
            model_key=f"fiber@{code_sha()[:12]}",
            field=spec,
            zones=zones,
            threshold=threshold,
            seconds=time.perf_counter() - started,
            score_kind="peak_abs_z",
            threshold_kind="spatial_z",
            device="cpu",
        )
