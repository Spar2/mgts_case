"""Catalog operational zones on records 1–6.

Each row is one zone from analyze_field. Temporal evidence is attached to
that zone. It does not set anomaly_status and it does not add zones.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fiber.classify.rules import LABELS
from fiber.io import Record
from fiber.pipeline import analyze_field

EVIDENCE = ROOT / "docs" / "evidence" / "zone_catalog.json"
FIELDS = (
    "shape_hypothesis",
    "contrast",
    "width_m",
    "observed_duration_s",
    "support",
    "periodicity_CV",
)
ALLOWED = set(LABELS) | {"Unknown"}


def _plain(value):
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(type(value).__name__)


def _row_ok(row: dict) -> bool:
    if row.get("shape_hypothesis") not in ALLOWED:
        return False
    if any(key not in row for key in FIELDS):
        return False
    evidence = row.get("evidence") or {}
    spatial = evidence.get("spatial") or {}
    temporal = evidence.get("temporal") or {}
    if spatial.get("view") != "spatial" or temporal.get("view") != "temporal":
        return False
    cv = row["periodicity_CV"]
    if cv is not None and cv != cv:
        return False
    return True


def build_catalog(root: Path | None = None) -> dict:
    base = Path(root) if root is not None else ROOT
    files = {}
    for name in ("1", "2", "3", "4", "5", "6"):
        record = Record.load(base / "artifacts" / "records" / name)
        result = analyze_field(
            record.raw_c,
            record.x_m,
            record.time_s,
            valid=record.valid,
            time_iso=list(record.time_iso),
        )
        events = [_plain(zone) for zone in result["zones"]]
        if len(events) != len(result["zones"]) or any(not _row_ok(row) for row in events):
            raise RuntimeError(f"record {name} catalog rows do not match operational zones")
        files[name] = {
            "n_zones": len(result["zones"]),
            "n_rows": len(events),
            "anomaly_status": result["anomaly_status"],
            "events": events,
        }
        print(f"file {name} zones {len(result['zones'])} rows {len(events)}", flush=True)
        del result, record
    return {
        "schema": "zone_catalog_v1",
        "status_source": "spatial_z_halo",
        "clean": "median_x3",
        "detector": "spatial_z_halo",
        "threshold_z": 3.5,
        "temporal_sets_status": False,
        "measurement": True,
        "files": files,
    }


def main() -> None:
    catalog = build_catalog()
    payload = json.dumps(catalog, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    EVIDENCE.write_text(payload, encoding="utf-8", newline="\n")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    print("required_fields shape_hypothesis contrast width_m observed_duration_s support periodicity_CV evidence.spatial evidence.temporal")
    print("all_rows_complete true")
    print(f"sha256 {digest}")
    print("finished")


if __name__ == "__main__":
    main()
