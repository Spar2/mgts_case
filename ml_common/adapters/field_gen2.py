"""Адаптер выпуска ens_gen2: dts.infer.Analyzer(gen2='on') / release_analyzer.

Веса только читаются из DTS_WEIGHTS_ROOT (по умолчанию /workspace/dts_solution):
models_release/ и models_release_gen2/. В git они не копируются.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from ml_common.events import build_events, iso_for
from ml_common.protocol import limit_threads, normalize_field

_FIELD = Path(__file__).resolve().parents[2] / "field_ensemble"
_CACHED: tuple | None = None


def weights_root() -> Path:
    return Path(os.environ.get("DTS_WEIGHTS_ROOT", "/workspace/dts_solution"))


def _require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_analyzer():
    """Собрать Analyzer выпуска. gen2_release_dir подменяется только на время конструктора."""
    global _CACHED
    if _CACHED is not None:
        return _CACHED
    limit_threads()
    root = weights_root()
    release = root / "models_release"
    gen2 = root / "models_release_gen2"
    thresholds = _require(release / "stage1_thresholds.json")
    _require(release / "eventnet.pt")
    _require(release / "lgbm.joblib")
    _require(gen2 / "s3_gen2_unet.pt")
    gen2_meta = json.loads(_require(gen2 / "stage1_thresholds.json").read_text(encoding="utf-8"))
    if str(_FIELD) not in sys.path:
        sys.path.insert(0, str(_FIELD))
    import torch

    torch.set_num_threads(limit_threads())
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    import dts.infer as infer

    previous = infer.gen2_release_dir
    infer.gen2_release_dir = lambda: gen2
    try:
        model = infer.Analyzer(
            stage2=release / "eventnet.pt",
            lgbm=release / "lgbm.joblib",
            thresholds=thresholds,
            op="far_1e-3",
            zones="cc",
            static="off",
            mae="off",
            gen2="on",
        )
    finally:
        infer.gen2_release_dir = previous
    sha = str(gen2_meta.get("ckpt_sha256") or "unhashed")
    _CACHED = (model, f"ens_gen2@{sha[:12]}", float(model.thr))
    return _CACHED


class FieldGen2Analyzer:
    method = "ens_gen2"

    def __init__(self, **_ignored):
        pass

    def predict(self, field) -> dict:
        import numpy as np

        spec = normalize_field(field)
        model, model_key, threshold = load_analyzer()
        valid = spec["valid"] if spec["valid"] is not None else np.isfinite(spec["raw"])
        started = time.perf_counter()
        out = model.run(spec["raw"], valid, spec["x_m"], spec["time_s"], gen2="on")
        zones = []
        for zone in out["zones"]:
            label = str(zone.get("class") or "unknown")
            proba = {str(key): float(value) for key, value in (zone.get("class_prob") or {}).items()}
            item = {
                "t_first": iso_for(zone["t_start_s"], spec["time_s"], spec["time_iso"]),
                "t_last": iso_for(zone["t_end_s"], spec["time_s"], spec["time_iso"]),
                "x_from_m": float(zone["start_m"]),
                "x_to_m": float(zone["end_m"]),
                "score": float(zone.get("score", 0.0)),
                "stage2_type": label,
                "type_proba": proba,
                "amplitude_c": float(zone.get("peak_dT_C", 0.0)),
            }
            if zone.get("polarity"):
                item["polarity"] = str(zone["polarity"])
            if "alarm" in zone:
                item["alarm"] = bool(zone["alarm"])
            zones.append(item)
        zones.sort(key=lambda item: item["score"], reverse=True)
        return build_events(
            method=self.method,
            model_key=model_key,
            field=spec,
            zones=zones,
            threshold=float(threshold),
            seconds=time.perf_counter() - started,
            score_kind="ens_blend_probability",
            threshold_kind="far_1e-3",
            device="cpu",
        )
