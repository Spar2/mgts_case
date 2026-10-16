"""Одинаковые метрики трёх адаптеров на одном синтетическом поле."""
from __future__ import annotations

from datetime import datetime

from ml_common.adapters import get_analyzer
from ml_common.synthetic import planted_field


def _stamp(text: str) -> float:
    return datetime.fromisoformat(text).timestamp()


def _overlap(zone: dict, box: dict) -> bool:
    zx0, zx1 = float(zone["x_from_m"]), float(zone["x_to_m"])
    if zx0 > zx1:
        zx0, zx1 = zx1, zx0
    tx0, tx1 = _stamp(zone["t_first"]), _stamp(zone["t_last"])
    if tx0 > tx1:
        tx0, tx1 = tx1, tx0
    x_hit = min(zx1, box["x_to_m"]) >= max(zx0, box["x_from_m"])
    t_hit = min(tx1, box["t_to_s"]) >= max(tx0, box["t_from_s"])
    return bool(x_hit and t_hit)


def score_document(document: dict, truth: list[dict]) -> dict:
    zones = document["zones"]
    hits = []
    for box in truth:
        matched = [zone["stage2_type"] for zone in zones if _overlap(zone, box)]
        hits.append({"name": box["name"], "n_zones": len(matched), "types": matched[:8]})
    covered = {(id(zone)) for zone in zones if any(_overlap(zone, box) for box in truth)}
    return {
        "n_zones": len(zones),
        "hits": hits,
        "n_zones_outside_truth": len(zones) - len(covered),
        "seconds": document["runtime"]["seconds"],
        "threshold": document["threshold"],
        "model_key": document["model_key"],
    }


def run_eval(models: list[str], *, use_ae: bool = False, epochs: int = 1) -> dict:
    field, truth = planted_field()
    report = {
        "schema": "ml-common-eval/1",
        "synthetic": {
            "dts_id": field["dts_id"],
            "shape": [int(field["raw"].shape[0]), int(field["raw"].shape[1])],
            "truth": [{key: box[key] for key in ("name", "x_from_m", "x_to_m", "t_first", "t_last")} for box in truth],
            "real_mgts_used": False,
            "larpathon_percentile": 95.0,
        },
        "models": {},
    }
    for name in models:
        # 95 — процентиль синтетического теста larpathon. Боевой дефолт модуля 99.2
        # рассчитан на длинную запись и на этом коротком поле гасит оба пятна.
        kwargs = {"use_ae": use_ae, "epochs": epochs, "percentile": 95.0} if name == "larpathon" else {}
        try:
            document = get_analyzer(name, **kwargs).predict(field)
            metrics = score_document(document, truth)
            report["models"][name] = {"ok": True, "metrics": metrics, "events": document}
        except Exception as exc:
            report["models"][name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return report
