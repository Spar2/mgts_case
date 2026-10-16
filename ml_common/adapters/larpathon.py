"""Адаптер solutions/larpathon: статистический ансамбль, опционально автоэнкодер на этом же поле.

Автоэнкодер по умолчанию выключен. Он учится на переданном поле (так устроен main.py),
а не на сохранённом датасете. Реальные MGTS-файлы этим адаптером не обучаются:
для них остаётся отдельный офлайн-прогон с явным --ae, если это вообще понадобится.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from ml_common.events import build_events, code_sha
from ml_common.protocol import limit_threads, normalize_field

_ROOT = Path(__file__).resolve().parents[2] / "solutions" / "larpathon"


def _module():
    limit_threads()
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    import mgts_anomaly

    return mgts_anomaly


def _finite(raw, valid):
    import numpy as np

    matrix = np.array(raw, dtype=np.float64, copy=True)
    bad = ~np.isfinite(matrix)
    if valid is not None:
        bad |= ~np.asarray(valid, dtype=bool)
    if bad.any():
        good = matrix[~bad]
        fill = float(np.median(good)) if good.size else 0.0
        matrix[bad] = fill
    return matrix.astype(np.float32)


class LarpathonAnalyzer:
    method = "larpathon"

    def __init__(self, *, use_ae: bool = False, epochs: int = 1, seed: int = 0, percentile: float | None = None):
        self.use_ae = bool(use_ae)
        self.epochs = int(epochs)
        self.seed = int(seed)
        self.percentile = percentile

    def predict(self, field) -> dict:
        import numpy as np

        spec = normalize_field(field)
        lib = _module()
        matrix = _finite(spec["raw"], spec["valid"])
        started = time.perf_counter()
        limit_threads()
        try:
            import torch

            torch.set_num_threads(limit_threads())
        except Exception:
            pass
        denoised = lib.denoise_matrix(matrix)
        parts = lib.compute_score_parts(
            denoised,
            use_ae=self.use_ae,
            seed=self.seed,
            epochs=self.epochs,
            device="cpu",
        )
        score = parts["ensemble"]
        if not np.isfinite(score).all():
            raise RuntimeError("larpathon ensemble score is not finite")
        base = float(self.percentile if self.percentile is not None else lib.DETECTOR_PERCENTILE)
        applied = float(lib.detector_percentile_for_score(score, base))
        timestamps = spec["time_iso"] or [str(value) for value in spec["time_s"]]
        dt_min = float(np.median(np.diff(spec["time_s"])) / 60.0)
        found = lib.extract_events(
            score,
            matrix,
            dists=spec["x_m"],
            percentile=base,
            timestamps=timestamps,
            time_step_min=dt_min,
            use_cv2=False,
            stat_score=parts["statistical"],
            ae_score=parts["autoencoder"],
        )
        events = list(found)
        zones = []
        for event in events:
            start_m = float(event.get("distance_start_m", 0.0))
            end_m = float(event.get("distance_end_m", start_m + float(event.get("spatial_extent_m", 0.0))))
            t0, t1 = event["t_range"]
            t0, t1 = int(t0), int(t1)
            t0 = min(max(t0, 0), len(timestamps) - 1)
            t1 = min(max(t1, 0), len(timestamps) - 1)
            if t1 < t0:
                t0, t1 = t1, t0
            label = str(event.get("type_hypothesis") or "unknown")
            confidence = float(event.get("confidence", 0.0))
            item = {
                "t_first": timestamps[t0],
                "t_last": timestamps[t1],
                "x_from_m": start_m,
                "x_to_m": end_m,
                "score": confidence,
                "stage2_type": label,
                "type_proba": {label: confidence} if np.isfinite(confidence) else {},
                "amplitude_c": float(event.get("amplitude", 0.0)),
            }
            zones.append(item)
        zones.sort(key=lambda item: abs(item["score"]), reverse=True)
        return build_events(
            method=self.method,
            model_key=f"larpathon@{code_sha()[:12]}",
            field=spec,
            zones=zones,
            threshold=applied,
            seconds=time.perf_counter() - started,
            score_kind="ensemble_confidence",
            threshold_kind="score_percentile",
            device="cpu",
        )
