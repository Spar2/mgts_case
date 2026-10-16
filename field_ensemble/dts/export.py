"""Экспорт зон в плоскую таблицу (одна строка = одна зона) — удобно грузить в БД (ClickHouse и т.п.)."""
from __future__ import annotations
import csv
from datetime import datetime, timezone

# cv_* дописываются в конец и могут быть пустыми: старые колонки не сдвигаются.
CV_FIELDS = ["cv_area_px", "cv_solidity", "cv_eccentricity", "cv_orientation_deg",
             "cv_bbox_width_m", "cv_bbox_duration_h", "cv_drift_m_per_h", "cv_n_components", "cv_period_h"]
FIELDS = ["source", "source_sha256", "zone_id", "t_start_utc", "t_end_utc", "x_start_m", "x_end_m", "polarity",
          "class", "class_prob", "is_alarm", "score", "peak_dT_C", "n_px", "model_version", *CV_FIELDS]


def _cv_cell(zone, key):
    if key not in zone or zone[key] is None or zone[key] == "":
        return ""
    value = zone[key]
    if isinstance(value, float):
        return round(value, 4)
    return value


def _iso(ts):
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()


def zone_rows(zones, source, sha256, model_version="field_ensemble-0.1"):
    for z in zones:
        yield dict(source=source, source_sha256=sha256, zone_id=f"{source}-{z['id']:04d}", t_start_utc=_iso(z["t_start_s"]),
                   t_end_utc=_iso(z["t_end_s"]), x_start_m=round(z["start_m"], 2), x_end_m=round(z["end_m"], 2), polarity=z["polarity"],
                   **{"class": z.get("class", "")}, class_prob=round(max(z.get("class_prob", {"": 0}).values()), 3),
                   is_alarm=int(bool(z.get("alarm", True))), score=round(z["score"], 4), peak_dT_C=round(z.get("peak_dT_C", 0.0), 2),
                   n_px=z["n_px"], model_version=model_version,
                   **{key: _cv_cell(z, key) for key in CV_FIELDS})


def write_csv(path, rows):
    rows = list(rows)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    return len(rows)
