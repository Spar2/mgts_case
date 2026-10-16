"""Write the wide-channel soft flag for next-0003 and wide gap answers.

The flag is research-only. Reading it does not change anomaly_status, the
zone list, or expert_confirmed. expert_class stays the presence/shape class.
A gap answer is included when its meter span is strictly wider than 40 m,
including an unmapped answer and a holdout or guard window.
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
from fiber.research_context import (
    WIDE_CHANNEL_CANDIDATE,
    WIDE_CHANNEL_FAR_GAP_C,
    WIDE_CHANNEL_HALO_INSIDE_MIN,
    context_evidence,
)
from fiber.weak_labels import (
    EXPERT_SEED_BY_ID,
    class_from_presence_and_shape,
    load_gap_answers,
    resolve_answer_rows,
)

OUT = ROOT / "docs" / "evidence" / "wide_channel.json"
# Fixed before the card loop. Wider than the 8–40 m ring's outer radius.
MIN_WIDTH_M = 40.0


def _zone_tuples(field: dict) -> list[tuple]:
    return [
        (
            float(zone["start_m"]),
            float(zone["end_m"]),
            float(zone["contrast"]),
            bool(zone.get("expert_confirmed")),
        )
        for zone in field["zones"]
    ]


def _jobs() -> list[dict]:
    seed = EXPERT_SEED_BY_ID["next-0003"]
    x0, x1 = (float(seed["x_m"][0]), float(seed["x_m"][1]))
    expert = class_from_presence_and_shape(seed["presence"], seed["recorded_shape"])
    if expert != seed["class_name"] or expert != "Local_Heating":
        raise SystemExit("next-0003 must stay Local_Heating")
    jobs = [{
        "id": seed["id"],
        "record": str(seed["record"]),
        "x_span": (x0, x1),
        "width_m": x1 - x0,
        "presence": seed["presence"],
        "shape": seed["recorded_shape"],
        "expert_class": expert,
        "rows_half_open": (int(seed["rows_half_open"][0]), int(seed["rows_half_open"][1])),
        "answer": None,
        "source": "expert_seed",
    }]
    if not jobs[0]["width_m"] > MIN_WIDTH_M:
        raise SystemExit("next-0003 is not wider than 40 m")
    for answer in load_gap_answers():
        meters = answer["meters"]
        x0, x1 = float(meters[0]), float(meters[1])
        width = x1 - x0
        if not width > MIN_WIDTH_M:
            continue
        presence = str(answer["presence"])
        shape = str(answer.get("shape") or "")
        expert = class_from_presence_and_shape(presence, shape)
        if expert == WIDE_CHANNEL_CANDIDATE:
            raise SystemExit(f"{answer['id']} expert class must not be the soft-flag name")
        jobs.append({
            "id": str(answer["id"]),
            "record": str(answer["record"]),
            "x_span": (x0, x1),
            "width_m": width,
            "presence": presence,
            "shape": shape,
            "expert_class": expert,
            "rows_half_open": None,
            "answer": answer,
            "source": "gap_shortlist",
        })
    gap_ids = [job["id"] for job in jobs if job["source"] == "gap_shortlist"]
    if not gap_ids:
        raise SystemExit("no gap answer is wider than 40 m")
    if len({job["id"] for job in jobs}) != len(jobs):
        raise SystemExit("wide-channel ids are not unique")
    return jobs


def _card(job: dict, field: dict, x_m, time_s, time_iso) -> dict:
    config = field["manifest"]["config"]
    manifest = field["manifest"]
    if (
        manifest["detector"] != "spatial_z_halo"
        or manifest["clean"] != "median_x3"
        or float(config["threshold_z"]) != 3.5
        or float(config["inner_m"]) != 8.0
        or float(config["outer_m"]) != MIN_WIDTH_M
        or manifest["trained_classifier_operational"] is not False
    ):
        raise SystemExit("analyze_field is not the frozen operational pair")
    status = field["anomaly_status"]
    zones = _zone_tuples(field)
    if job["answer"] is None:
        rows = job["rows_half_open"]
    else:
        rows = resolve_answer_rows(job["answer"], time_iso)
    measured = context_evidence(
        field,
        x_m,
        time_s,
        rows_half_open=tuple(rows),
        x_span=tuple(job["x_span"]),
        inner_m=float(config["inner_m"]),
        outer_m=float(config["outer_m"]),
    )
    if field["anomaly_status"] != status or _zone_tuples(field) != zones:
        raise SystemExit(f"{job['id']} wide-channel read changed the operational field")
    if any(confirmed for *_rest, confirmed in zones):
        raise SystemExit(f"{job['id']} zone expert_confirmed was set")
    if measured["wide_channel_halo_inside_min"] != WIDE_CHANNEL_HALO_INSIDE_MIN:
        raise SystemExit("halo-inside cutoff drifted")
    if measured["wide_channel_far_gap_min_c"] != WIDE_CHANNEL_FAR_GAP_C:
        raise SystemExit("far-gap cutoff drifted")
    for side in (measured["far_left"], measured["far_right"]):
        if not side["present"] and side["median_c"] is not None:
            raise SystemExit(f"{job['id']} absent far side was filled")
    if measured["far_cable_median_c"] is None and measured["wide_channel"] is True:
        raise SystemExit(f"{job['id']} flagged without a far-cable median")
    candidate = measured["wide_channel_candidate"]
    if measured["wide_channel"]:
        if candidate != WIDE_CHANNEL_CANDIDATE:
            raise SystemExit(f"{job['id']} flagged without the candidate name")
    elif candidate is not None:
        raise SystemExit(f"{job['id']} candidate was set without the flag")
    if job["expert_class"] == WIDE_CHANNEL_CANDIDATE:
        raise SystemExit(f"{job['id']} expert class was replaced by the flag")
    if job["id"] == "next-0003":
        if job["expert_class"] != "Local_Heating":
            raise SystemExit("next-0003 must stay Local_Heating")
        if measured["wide_channel"] is not True:
            raise SystemExit(
                "next-0003 was not flagged: "
                f"halo {measured['halo_inside_fraction']} gap {measured['far_gap_c']}"
            )
    return {
        "id": job["id"],
        "record": job["record"],
        "source": job["source"],
        "width_m": job["width_m"],
        "presence": job["presence"],
        "expert_class": job["expert_class"],
        "wide_channel": measured["wide_channel"],
        "wide_channel_candidate": candidate,
        "halo_inside_fraction": measured["halo_inside_fraction"],
        "rectangle_median_c": measured["rectangle_median_c"],
        "far_cable_median_c": measured["far_cable_median_c"],
        "far_gap_c": measured["far_gap_c"],
        "far_left": measured["far_left"],
        "far_right": measured["far_right"],
        "halo_columns": measured["halo_columns"],
        "halo_columns_inside": measured["halo_columns_inside"],
        "rows_half_open": measured["rows_half_open"],
        "x_m": measured["x_m"],
        "anomaly_status": status,
        "n_zones": len(zones),
        "expert_confirmed": False,
    }


def main() -> None:
    jobs = _jobs()
    by_record: dict[str, list[dict]] = {}
    for job in jobs:
        by_record.setdefault(job["record"], []).append(job)
    cards = []
    for name, group in by_record.items():
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
        for job in group:
            card = _card(job, found, record.x_m, record.time_s, record.time_iso)
            cards.append(card)
            print(
                f"CARD {card['id']} source {card['source']} width_m {card['width_m']} "
                f"expert_class {card['expert_class']} wide_channel {card['wide_channel']} "
                f"candidate {card['wide_channel_candidate']} "
                f"halo_inside_fraction {card['halo_inside_fraction']} "
                f"rectangle_median_c {card['rectangle_median_c']} "
                f"far_cable_median_c {card['far_cable_median_c']} "
                f"far_gap_c {card['far_gap_c']} "
                f"far_left {card['far_left']['present']} far_right {card['far_right']['present']} "
                f"anomaly_status {card['anomaly_status']} n_zones {card['n_zones']}",
                flush=True,
            )
        del found, record
    order = {job["id"]: index for index, job in enumerate(jobs)}
    cards.sort(key=lambda card: order[card["id"]])
    if [card["id"] for card in cards] != [job["id"] for job in jobs]:
        raise SystemExit("wide-channel evidence dropped a card")
    payload = {
        "schema": "wide_channel_v1",
        "serve": "research",
        "operational": False,
        "trained_classifier_operational": False,
        "detector": "spatial_z_halo",
        "clean": "median_x3",
        "threshold_z": 3.5,
        "inner_m": 8.0,
        "outer_m": 40.0,
        "sets_anomaly_status": False,
        "min_width_m": MIN_WIDTH_M,
        "wide_channel_halo_inside_min": WIDE_CHANNEL_HALO_INSIDE_MIN,
        "wide_channel_far_gap_min_c": WIDE_CHANNEL_FAR_GAP_C,
        "wide_channel_candidate": WIDE_CHANNEL_CANDIDATE,
        "note": (
            "Research soft flag for a wide channel on next-0003 and every gap-shortlist "
            "answer whose meter span is wider than 40 m. The flag is true only when the "
            "share of the 8–40 m ring inside the rectangle is at least "
            "wide_channel_halo_inside_min and the absolute gap between the rectangle "
            "median and the cable farther than 40 m is at least wide_channel_far_gap_min_c. "
            "A missing far side is not filled from the rectangle. The candidate does not "
            "replace expert_class and does not set anomaly_status or expert_confirmed. "
            "The operational pair remains median_x3 and spatial_z_halo at |Z| >= 3.5."
        ),
        "cards": cards,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    flagged = [card["id"] for card in cards if card["wide_channel"]]
    print(
        f"serve=research operational False sets_anomaly_status False "
        f"cards {len(cards)} flagged {len(flagged)} wrote {OUT}",
        flush=True,
    )
    print("finished", flush=True)


if __name__ == "__main__":
    main()
