"""Score the context research CNN. Does not fit.

Recovery is class match on every ask expert seed, and separately on each of
the previous 15. Gap cards are not scored. Unmapped gap answers are not a
match or a miss. The historical labels 10/15 and 15/22 stay the comparison.
The published 9/15 report is left as it is.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from fiber.classify.rules import LABELS
from fiber.io import Record
from fiber.pipeline import analyze_field
from fiber.research_cnn import (
    CONTEXT_ARCHITECTURE,
    CONTEXT_NAMES,
    HISTORICAL_CONTEXT_CARDS,
    HISTORICAL_CONTEXT_MATCH_LABEL,
    HISTORICAL_CONTEXT_MATCHES,
    HISTORICAL_CONTEXT_SHA256,
    PREVIOUS_FIT_SHA256,
    load_research_context_cnn,
    read_context_crop,
)
from fiber.weak_labels import (
    EXPERT_SEED_BY_ID,
    EXPERT_SEEDS,
    EXPERT_WEIGHT,
    class_from_presence_and_shape,
    load_gap_answers,
    map_expert_class,
    resolve_answer_rows,
)

WEIGHTS = ROOT / "models" / "weights" / "research_context_cnn.pt"
REPORT = ROOT / "docs" / "evidence" / "research_context_cnn_report.json"
PUBLISHED = ROOT / "docs" / "evidence" / "research_cnn_report.json"
ASK_JSONL = ROOT / "docs" / "evidence" / "expert_seeds_ask.jsonl"
JSONL = ROOT / "docs" / "evidence" / "weak_labels.jsonl"
RESIDUAL_WEIGHTS = ROOT / "models" / "weights" / "research_residual_cnn.pt"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _gap_lists():
    """Classed and unmapped gap ids. Neither list is a recovery card."""
    unmapped = []
    classed = []
    for answer in load_gap_answers():
        presence = str(answer["presence"])
        shape = str(answer.get("shape") or "")
        expert = class_from_presence_and_shape(presence, shape)
        if expert is None:
            unmapped.append(str(answer["id"]))
            continue
        if expert not in LABELS:
            raise SystemExit(f"{answer['id']} expert class is outside the six names")
        classed.append(str(answer["id"]))
    return unmapped, classed


def _ask_jobs():
    if not ASK_JSONL.is_file():
        raise SystemExit(f"missing ask expert seeds {ASK_JSONL}")
    jobs = []
    seen = set()
    for line in ASK_JSONL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        seed_id = str(row["id"])
        if seed_id in seen or seed_id in EXPERT_SEED_BY_ID or seed_id.startswith("gap-"):
            raise SystemExit(f"ask seed id is not a new expert seed: {seed_id}")
        if float(row["weight"]) < float(EXPERT_WEIGHT) or row.get("expert_confirmed") is not True:
            raise SystemExit(f"ask seed {seed_id} is not a weight-10 expert seed")
        if int(row["split_code"]) not in (0, 1):
            raise SystemExit(f"ask seed {seed_id} is outside train/development")
        expert = str(row["class_name"])
        if expert not in LABELS:
            raise SystemExit(f"ask seed {seed_id} left the six class names")
        mapped = map_expert_class(str(row["presence"]), str(row.get("recorded_shape") or ""))
        if mapped != expert:
            raise SystemExit(f"ask seed {seed_id} class drifted from presence and shape")
        seen.add(seed_id)
        jobs.append({
            "id": seed_id,
            "record": str(row["record"]),
            "rows": (int(row["rows_half_open"][0]), int(row["rows_half_open"][1])),
            "x_span": (float(row["x_m"][0]), float(row["x_m"][1])),
            "expert_class": expert,
            "presence": str(row["presence"]),
            "source": "ask_expert_seed",
            "used_as_fit_seed": True,
            "answer": None,
        })
    if not jobs:
        raise SystemExit("ask expert seed file is empty")
    return jobs


def _jobs():
    jobs = []
    for seed in EXPERT_SEEDS:
        if seed["id"] == "next-0003" and seed["class_name"] != "Local_Heating":
            raise SystemExit("next-0003 must stay Local_Heating")
        jobs.append({
            "id": seed["id"],
            "record": str(seed["record"]),
            "rows": (int(seed["rows_half_open"][0]), int(seed["rows_half_open"][1])),
            "x_span": (float(seed["x_m"][0]), float(seed["x_m"][1])),
            "expert_class": seed["class_name"],
            "presence": seed["presence"],
            "source": "expert_seed",
            "used_as_fit_seed": True,
            "answer": None,
        })
    jobs.extend(_ask_jobs())
    if len(jobs) <= len(EXPERT_SEEDS):
        raise SystemExit("ask expert seeds did not extend the comparison past 15")
    unmapped, classed = _gap_lists()
    if any(job["id"] in unmapped or job["id"] in classed or str(job["id"]).startswith("gap-") for job in jobs):
        raise SystemExit("a gap card entered the recovery cards")
    return jobs, unmapped, classed


def _predict(model, crop, vector, device):
    crops = torch.from_numpy(crop[None, None]).to(device)
    context = torch.from_numpy(vector[None].copy()).to(device)
    with torch.inference_mode():
        logits = model(crops, context)[0].detach().cpu().numpy()
    if logits.shape != (len(LABELS),):
        raise SystemExit("context forward did not return six class scores")
    return LABELS[int(np.argmax(logits))]


def main() -> None:
    if not WEIGHTS.is_file():
        raise SystemExit(f"missing context checkpoint {WEIGHTS}")
    published_sha = _sha256(PUBLISHED)
    labels_sha = _sha256(JSONL)
    residual_sha = _sha256(RESIDUAL_WEIGHTS)
    weights_sha_before = _sha256(WEIGHTS)
    published = json.loads(PUBLISHED.read_text(encoding="utf-8"))
    published_label = f"{int(published['match_count'])}/{int(published['n_cards'])}"
    previous_label = str(published["previous_match_label"])
    if published_label != "9/15" or previous_label != "10/15":
        raise SystemExit("published recovery is not 9/15 beside 10/15")
    if int(published["previous_match_count"]) != 10 or int(published["previous_n_cards"]) != 15:
        raise SystemExit("published previous fit is not 10/15")
    if str(published["previous_weights_sha256"]) != PREVIOUS_FIT_SHA256:
        raise SystemExit("published previous sha256 drifted")
    model, blob, device = load_research_context_cnn(WEIGHTS)
    if blob.get("serve") != "research" or blob.get("operational") is True:
        raise SystemExit("context checkpoint is not serve=research")
    if blob.get("sets_anomaly_status") is True:
        raise SystemExit("context checkpoint sets anomaly status")
    if blob.get("architecture") != CONTEXT_ARCHITECTURE:
        raise SystemExit("context checkpoint architecture drifted")
    if list(blob.get("class_names")) != list(LABELS):
        raise SystemExit("context checkpoint left the six class names")
    if weights_sha_before == HISTORICAL_CONTEXT_SHA256:
        raise SystemExit("context weights are still the historical 15/22 checkpoint")
    jobs, unmapped, classed_gap = _jobs()
    by_record = {}
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
        config = manifest["config"]
        if (
            manifest["detector"] != "spatial_z_halo"
            or manifest["clean"] != "median_x3"
            or float(config["threshold_z"]) != 3.5
            or manifest["trained_classifier_operational"] is not False
        ):
            raise SystemExit("analyze_field is not the frozen operational pair")
        status = found["anomaly_status"]
        n_zones = len(found["zones"])
        confirmed = [bool(zone.get("expert_confirmed")) for zone in found["zones"]]
        print(
            f"RECORD {record.name} n_zones {n_zones} "
            f"operational_check clean {manifest['clean']} detector {manifest['detector']} "
            f"threshold_z {config['threshold_z']} "
            f"trained_classifier_operational {manifest['trained_classifier_operational']}",
            flush=True,
        )
        for job in group:
            rows = job["rows"]
            if job["answer"] is not None:
                rows = resolve_answer_rows(job["answer"], record.time_iso)
            crop, vector, measured = read_context_crop(
                found, record.x_m, record.time_s, rows, job["x_span"],
            )
            if found["anomaly_status"] != status or len(found["zones"]) != n_zones:
                raise SystemExit(f"{job['id']} context read changed anomaly_status or zones")
            if [bool(zone.get("expert_confirmed")) for zone in found["zones"]] != confirmed:
                raise SystemExit(f"{job['id']} context read changed expert_confirmed")
            if crop is None:
                raise SystemExit(f"{job['id']} residual crop is empty")
            if measured["far_cable_median_c"] is None and np.isfinite(vector[1]):
                raise SystemExit(f"{job['id']} absent far side was filled with a median")
            if job["id"] == "next-0003" and job["expert_class"] != "Local_Heating":
                raise SystemExit("next-0003 must stay Local_Heating")
            network = _predict(model, crop, vector, device)
            if network not in LABELS:
                raise SystemExit(f"{job['id']} network class left the six names")
            card = {
                "id": job["id"],
                "record": job["record"],
                "source": job["source"],
                "presence": job["presence"],
                "expert_class": job["expert_class"],
                "network_class": network,
                "match": network == job["expert_class"],
                "used_as_fit_seed": job["used_as_fit_seed"],
                "rectangle_median_c": None if measured["rectangle_median_c"] is None else float(measured["rectangle_median_c"]),
                "far_cable_median_c": None if measured["far_cable_median_c"] is None else float(measured["far_cable_median_c"]),
                "halo_inside_fraction": float(measured["halo_inside_fraction"]),
                "rows_half_open": [int(rows[0]), int(rows[1])],
                "x_m": [float(job["x_span"][0]), float(job["x_span"][1])],
            }
            cards.append(card)
            print(
                f"CARD {card['id']} expert_class {card['expert_class']} "
                f"network_class {card['network_class']} match {str(card['match']).lower()} "
                f"source {card['source']} fit_seed {str(card['used_as_fit_seed']).lower()} "
                f"halo_inside_fraction {card['halo_inside_fraction']} "
                f"rectangle_median_c {card['rectangle_median_c']} "
                f"far_cable_median_c {card['far_cable_median_c']}",
                flush=True,
            )
        del found, record
    order = {job["id"]: index for index, job in enumerate(jobs)}
    cards.sort(key=lambda card: order[card["id"]])
    if [card["id"] for card in cards] != [job["id"] for job in jobs]:
        raise SystemExit("scoring dropped a compared card")
    if any(card["id"] in unmapped or card["id"] in classed_gap for card in cards):
        raise SystemExit("a gap id entered the match count")
    if any(str(card["id"]).startswith("gap-") for card in cards):
        raise SystemExit("a gap card entered the match count")
    if any(card["source"] == "gap_shortlist" for card in cards):
        raise SystemExit("a gap card was scored")
    seed_cards = [card for card in cards if card["source"] == "expert_seed"]
    ask_cards = [card for card in cards if card["source"] == "ask_expert_seed"]
    if [card["id"] for card in seed_cards] != [seed["id"] for seed in EXPERT_SEEDS]:
        raise SystemExit("previous 15 were not scored once")
    if len(ask_cards) + len(seed_cards) != len(cards):
        raise SystemExit("recovery cards are not the previous 15 plus the ask seeds")
    if len({card["id"] for card in cards}) != len(cards):
        raise SystemExit("a recovery card was scored twice")
    seed_matches = sum(bool(card["match"]) for card in seed_cards)
    ask_matches = sum(bool(card["match"]) for card in ask_cards)
    weights_sha = _sha256(WEIGHTS)
    if weights_sha != weights_sha_before:
        raise SystemExit("scoring changed the context weights")
    if weights_sha == HISTORICAL_CONTEXT_SHA256:
        raise SystemExit("context weights are still the historical 15/22 checkpoint")
    payload = {
        "schema": "research_context_cnn_report_v1",
        "serve": "research",
        "operational": False,
        "trained_classifier_operational": False,
        "sets_anomaly_status": False,
        "architecture": CONTEXT_ARCHITECTURE,
        "weights": "models/weights/research_context_cnn.pt",
        "weights_sha256": weights_sha,
        "context_names": list(CONTEXT_NAMES),
        "published_match_count": 9,
        "published_n_cards": 15,
        "published_match_label": "9/15",
        "previous_match_count": 10,
        "previous_n_cards": 15,
        "previous_match_label": "10/15",
        "previous_weights_sha256": PREVIOUS_FIT_SHA256,
        "historical_context_match_count": HISTORICAL_CONTEXT_MATCHES,
        "historical_context_n_cards": HISTORICAL_CONTEXT_CARDS,
        "historical_context_match_label": HISTORICAL_CONTEXT_MATCH_LABEL,
        "historical_context_weights_sha256": HISTORICAL_CONTEXT_SHA256,
        "context_match_count": HISTORICAL_CONTEXT_MATCHES,
        "context_n_cards": HISTORICAL_CONTEXT_CARDS,
        "context_match_label": HISTORICAL_CONTEXT_MATCH_LABEL,
        "expert_seed_match_count": seed_matches,
        "expert_seed_n_cards": len(seed_cards),
        "expert_seed_match_label": f"{seed_matches}/{len(seed_cards)}",
        "ask_seed_match_count": ask_matches,
        "ask_seed_n_cards": len(ask_cards),
        "ask_seed_match_label": f"{ask_matches}/{len(ask_cards)}",
        "ask_recovery_is_in_sample": True,
        "expert_cards_are_not_an_untouched_holdout": True,
        "gap_classed_n_cards": len(classed_gap),
        "gap_cards_scored": False,
        "unmapped_gap_ids": unmapped,
        "unmapped_in_match_count": False,
        "gap_cards_in_fit": False,
        "cards": cards,
        "note": (
            "Research comparison. Recovery is class match on every ask expert seed, and "
            "separately on each of the previous 15. Both counts are in-sample: those cards "
            "are fit seeds. A miss stays a miss. This is not an untouched holdout score and "
            "not operational accuracy. The two fractions sit beside the historical context "
            f"comparison {HISTORICAL_CONTEXT_MATCH_LABEL} and the historical expert comparison "
            "10/15. Gap cards are not fit seeds and are not in either count. Unmapped gap "
            "answers are not a match or a miss. next-0003 stays Local_Heating. serve=research. "
            "analyze_field stays median_x3 and spatial_z_halo at |Z| >= 3.5."
        ),
    }
    REPORT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if _sha256(PUBLISHED) != published_sha or _sha256(JSONL) != labels_sha or _sha256(RESIDUAL_WEIGHTS) != residual_sha:
        raise SystemExit("scoring rewrote the published report, weak labels, or residual2d weights")
    if _sha256(WEIGHTS) != weights_sha:
        raise SystemExit("scoring changed the context weights")
    again = json.loads(PUBLISHED.read_text(encoding="utf-8"))
    if int(again["match_count"]) != 9 or again["previous_match_label"] != "10/15":
        raise SystemExit("published 9/15 report changed")
    print("serve=research", flush=True)
    print("operational False", flush=True)
    print("sets_anomaly_status False", flush=True)
    print(f"architecture {CONTEXT_ARCHITECTURE}", flush=True)
    print("operational_check clean median_x3 detector spatial_z_halo threshold_z 3.5", flush=True)
    print(f"published_match_label {published_label}", flush=True)
    print(f"previous_match_label {previous_label}", flush=True)
    print(f"historical_context_match_label {HISTORICAL_CONTEXT_MATCH_LABEL}", flush=True)
    print(
        f"n_cards {len(cards)} ask_seed_match_label {ask_matches}/{len(ask_cards)} "
        f"expert_seed_match_label {seed_matches}/{len(seed_cards)}",
        flush=True,
    )
    print(f"weights_sha256 {weights_sha}", flush=True)
    print("unmapped_excluded " + " ".join(unmapped), flush=True)
    print(f"wrote {REPORT}", flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
