"""Write research context evidence for the fifteen expert cards.

Each card is scored by analyze_field, then by context_evidence. The channel
does not change anomaly status, the zone list, or expert_confirmed. The
automatic detector stays median_x3 and spatial_z_halo at |Z| >= 3.5.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fiber.io import Record
from fiber.pipeline import analyze_field
from fiber.research_context import context_evidence
from fiber.weak_labels import EXPERT_SEEDS, map_expert_class

OUT = ROOT / "docs" / "evidence" / "context_evidence.json"


def _card(seed: dict, field: dict, x_m, time_s) -> dict:
    config = field["manifest"]["config"]
    if float(config["threshold_z"]) != 3.5 or field["manifest"]["trained_classifier_operational"] is not False:
        raise SystemExit("analyze_field is not the frozen operational pair")
    status = field["anomaly_status"]
    n_zones = len(field["zones"])
    confirmed = [bool(zone.get("expert_confirmed")) for zone in field["zones"]]
    measured = context_evidence(
        field,
        x_m,
        time_s,
        rows_half_open=tuple(seed["rows_half_open"]),
        x_span=tuple(seed["x_m"]),
        inner_m=float(config["inner_m"]),
        outer_m=float(config["outer_m"]),
    )
    if field["anomaly_status"] != status or len(field["zones"]) != n_zones:
        raise SystemExit(f"{seed['id']} context channel changed the operational field")
    if [bool(zone.get("expert_confirmed")) for zone in field["zones"]] != confirmed:
        raise SystemExit(f"{seed['id']} context channel changed expert_confirmed")
    expert = seed["class_name"]
    if expert != map_expert_class(seed["presence"], seed["recorded_shape"]):
        raise SystemExit(f"{seed['id']} class drifted from presence")
    if seed["presence"] == "no" and expert != "Background":
        raise SystemExit(f"{seed['id']} presence no must stay Background")
    if seed["id"] == "next-0003" and expert != "Local_Heating":
        raise SystemExit("next-0003 must stay Local_Heating")
    for side in (measured["far_left"], measured["far_right"]):
        if not side["present"] and side["median_c"] is not None:
            raise SystemExit(f"{seed['id']} absent far side was filled")
    return {
        "id": seed["id"],
        "record": seed["record"],
        "presence": seed["presence"],
        "expert_class": expert,
        "notes": seed["notes"],
        "anomaly_status": status,
        "n_zones": n_zones,
        "expert_confirmed": False,
        **measured,
    }


def main() -> None:
    by_record: dict[str, list[dict]] = {}
    for seed in EXPERT_SEEDS:
        by_record.setdefault(str(seed["record"]), []).append(seed)
    cards = []
    for name, seeds in by_record.items():
        record = Record.load(ROOT / "artifacts" / "records" / name)
        found = analyze_field(
            record.raw_c,
            record.x_m,
            record.time_s,
            valid=record.valid,
            time_iso=record.time_iso,
        )
        manifest = found["manifest"]
        print(
            f"RECORD {record.name} n_zones {len(found['zones'])} "
            f"detector {manifest['detector']} clean {manifest['clean']} "
            f"threshold_z {manifest['config']['threshold_z']} "
            f"trained_classifier_operational {manifest['trained_classifier_operational']}",
            flush=True,
        )
        for seed in seeds:
            card = _card(seed, found, record.x_m, record.time_s)
            cards.append(card)
            flanks = ", ".join(
                f"{zone['shape_hypothesis']} {zone['contrast']}" for zone in card["flank_zones"]
            ) or "none"
            inside = ", ".join(
                f"{zone['shape_hypothesis']} {zone['contrast']}" for zone in card["in_box_zones"]
            ) or "none"
            print(
                f"CARD {card['id']} presence {card['presence']} class {card['expert_class']} "
                f"rectangle_median_c {card['rectangle_median_c']} "
                f"far_cable_median_c {card['far_cable_median_c']} "
                f"far_left {card['far_left']['present']} {card['far_left']['median_c']} "
                f"far_right {card['far_right']['present']} {card['far_right']['median_c']} "
                f"halo_inside_fraction {card['halo_inside_fraction']} "
                f"in_box {inside} flank {flanks}",
                flush=True,
            )
        del found, record
    order = {seed["id"]: index for index, seed in enumerate(EXPERT_SEEDS)}
    cards.sort(key=lambda card: order[card["id"]])
    if [card["id"] for card in cards] != [seed["id"] for seed in EXPERT_SEEDS]:
        raise SystemExit("context evidence did not cover the fifteen expert cards")
    payload = {
        "schema": "context_evidence_v1",
        "serve": "research",
        "operational": False,
        "trained_classifier_operational": False,
        "detector": "spatial_z_halo",
        "clean": "median_x3",
        "threshold_z": 3.5,
        "inner_m": 8.0,
        "outer_m": 40.0,
        "sets_anomaly_status": False,
        "note": (
            "Research context on the 15 expert cards. The channel does not add or remove zones "
            "and does not set anomaly_status. The operational pair remains median_x3 and "
            "spatial_z_halo at |Z| >= 3.5. presence=no stays Background. This file is not "
            "operational accuracy."
        ),
        "cards": cards,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"serve=research cards {len(cards)} wrote {OUT}", flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
