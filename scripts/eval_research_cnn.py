"""Score the frozen research CNN. Does not fit.

Prints card calls, baseline-zone overlap, the quiet-control mix, and the
holdout mix. Writes the report and the four side-by-side examples.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from fiber.denoise.dsp import median_x
from fiber.io import Record
from fiber.pipeline import analyze_field
from fiber.real_data import RULE_CONFIG, crop_partition, temporal_partition
from fiber.research_cnn import (
    EVAL_SPACE,
    EVAL_SPACE_STRIDE,
    EVAL_TIME,
    EVAL_TIME_STRIDE,
    LABELS,
    PREVIOUS_FIT_CARDS,
    PREVIOUS_FIT_MATCHES,
    PREVIOUS_FIT_SHA256,
    PREVIOUS_NETWORK_CLASS,
    baseline_overlap,
    class_mix,
    load_research_cnn,
    meter_columns,
    residual_crop,
    window_hits,
)
from fiber.weak_labels import EXPERT_SEEDS

MANIFEST = ROOT / "docs" / "evidence" / "research_cnn_manifest.json"
REPORT = ROOT / "docs" / "evidence" / "research_cnn_report.json"
IMAGE_DIR = ROOT / "docs" / "images" / "research-cnn"
WITNESS = "4-early-quiet-control"


def _starts(length, crop, stride):
    if length < crop or stride <= 0:
        return np.zeros(0, dtype=np.int64)
    starts = list(range(0, length - crop + 1, stride))
    last = length - crop
    if starts[-1] != last:
        starts.append(last)
    return np.asarray(starts, dtype=np.int64)


def _predict(model, features, device):
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(features), 256):
            batch = torch.from_numpy(features[start:start + 256]).to(device)
            chunks.append(model(batch).cpu())
    return torch.cat(chunks, dim=0).numpy()


def _quiet_box(record):
    witnesses = json.loads((ROOT / "fiber" / "benchmark" / "witnesses.json").read_text(encoding="utf-8"))
    spec = next(row for row in witnesses["witnesses"] if row["id"] == WITNESS)
    if str(record.name) != spec["record"]:
        return None
    r0, r1 = spec["rows"]
    r1 = min(int(r1), record.n_profiles - 1)
    cols = np.flatnonzero((record.x_m >= spec["x_m"][0]) & (record.x_m <= spec["x_m"][1]))
    if cols.size == 0:
        return None
    return (int(r0), int(r1) + 1, int(cols[0]), int(cols[-1]) + 1)


def _zone_boxes(zones, x_m, time_s):
    boxes = []
    names = []
    for zone in zones:
        t0 = int(np.searchsorted(time_s, zone["t_start_s"], side="left"))
        t1 = int(np.searchsorted(time_s, zone["t_end_s"], side="right"))
        t1 = max(t0 + 1, min(t1, len(time_s)))
        c0, c1 = meter_columns(x_m, zone["start_m"], zone["end_m"])
        boxes.append((t0, t1, c0, c1))
        names.append(zone.get("shape_hypothesis") or "unknown")
    if not boxes:
        return np.zeros((0, 4), dtype=np.int32), []
    return np.asarray(boxes, dtype=np.int32), names


def _overlapping_zones(card_box, zones, zone_boxes):
    overlapping = []
    for zone, box in zip(zones, zone_boxes, strict=True):
        time_hit = card_box[0] < int(box[1]) and int(box[0]) < card_box[1]
        space_hit = card_box[2] < int(box[3]) and int(box[2]) < card_box[3]
        if time_hit and space_hit:
            overlapping.append(zone)
    return overlapping


def _baseline_fields(seed, overlapping):
    listed = []
    contrasts = []
    for zone in overlapping:
        contrast = zone.get("contrast")
        if contrast is not None and np.isfinite(float(contrast)):
            contrast = float(contrast)
            contrasts.append(contrast)
        else:
            contrast = None
        median = zone.get("median_contrast_c")
        listed.append({
            "shape_hypothesis": zone.get("shape_hypothesis"),
            "contrast_c": contrast,
            "median_contrast_c": None if median is None else float(median),
            "start_m": float(zone["start_m"]),
            "end_m": float(zone["end_m"]),
            "polarity": zone.get("polarity"),
        })
    overlaps = bool(overlapping)
    headline = float(max(contrasts, key=lambda value: abs(value))) if contrasts else None
    names = []
    for item in listed:
        name = str(item["shape_hypothesis"])
        if name not in names:
            names.append(name)
    detail = ", ".join(
        f"{item['shape_hypothesis']} {item['contrast_c']} C" for item in listed
    )
    if seed["presence"] == "no" and overlaps:
        note = f"spatial_z kept a zone here ({detail}). The expert called this span normal."
    elif seed["presence"] == "no":
        note = "spatial_z kept no zone on this span. The expert called this span normal."
    elif overlaps:
        note = f"spatial_z kept a zone here ({detail})."
    else:
        note = "spatial_z kept no zone on this span."
    return {
        "baseline_overlaps": overlaps,
        "baseline_contrast_c": headline,
        "baseline_zones": listed,
        "baseline_zone": ", ".join(names) if names else "no spatial_z zone",
        "baseline_note": note,
    }


def _score_cards(model, device, clean, record, zones, zone_boxes, split, card_rows):
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    for seed in EXPERT_SEEDS:
        if seed["record"] != str(record.name):
            continue
        t0, t1 = seed["rows_half_open"]
        c0, c1 = meter_columns(record.x_m, seed["x_m"][0], seed["x_m"][1])
        crop = residual_crop(clean, record.valid, t0, t1, c0, c1)
        if crop is None:
            raise SystemExit(f"empty card crop {seed['id']}")
        logits = _predict(model, crop[None, None, :, :], device)[0]
        predicted = LABELS[int(logits.argmax())]
        code = int(crop_partition(split[t0:t1]))
        baseline = _baseline_fields(seed, _overlapping_zones((t0, t1, c0, c1), zones, zone_boxes))
        image = IMAGE_DIR / f"{seed['id']}.png"
        _draw(image, crop, logits, predicted, baseline["baseline_zone"])
        card_rows.append({
            "id": seed["id"],
            "record": seed["record"],
            "presence": seed["presence"],
            "recorded": seed["class_name"],
            "expert_class": seed["class_name"],
            "recorded_shape": seed["recorded_shape"],
            "notes": seed["notes"],
            "structure": seed["structure"],
            "polarity": seed["polarity"],
            "predicted": predicted,
            "network_class": predicted,
            "previous_network_class": PREVIOUS_NETWORK_CLASS[seed["id"]],
            "match": predicted == seed["class_name"],
            "split_code": code,
            "used_as_fit_seed": True,
            "untouched_holdout": False,
            "fit_despite_split": code not in (0, 1),
            "rows_half_open": [int(t0), int(t1)],
            "x_m": [float(seed["x_m"][0]), float(seed["x_m"][1])],
            **baseline,
            "example": f"docs/images/research-cnn/{seed['id']}.png",
        })


def _draw(path, crop, scores, predicted, baseline):
    order = np.argsort(scores)
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.3), layout="constrained")
    image = axes[0].imshow(crop, aspect="auto", cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    axes[0].set_title("residual crop")
    axes[0].set_xlabel("metre sample")
    axes[0].set_ylabel("time sample")
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
    axes[1].barh([LABELS[index] for index in order], scores[order], color="#176c83")
    axes[1].set_title(f"network: {predicted}")
    axes[1].set_xlabel("score")
    axes[2].axis("off")
    axes[2].set_title("baseline zone")
    axes[2].text(0.0, 0.5, baseline, va="center", ha="left", wrap=True, fontsize=11)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    weights = ROOT / manifest["weights"]
    device = "cpu"
    model, blob, device = load_research_cnn(weights, device=device)
    if blob.get("serve") != "research" or blob.get("holdout_used_for_fit") or blob.get("guard_used_for_fit"):
        raise SystemExit("weights are not a research fit")
    windows = []
    positive = []
    hits = []
    quiet_names = []
    holdout_names = []
    card_rows = []
    for name in map(str, range(1, 7)):
        print(f"Record {name}: score", flush=True)
        record = Record.load(ROOT / "artifacts" / "records" / name)
        clean = median_x(record.raw_c, record.valid)
        detected = analyze_field(
            record.raw_c, record.x_m, record.time_s, valid=record.valid, time_iso=record.time_iso,
        )
        zone_boxes, _zone_names = _zone_boxes(detected["zones"], record.x_m, record.time_s)
        split, _ = temporal_partition(record.time_s, RULE_CONFIG["split_guard_s"])
        _score_cards(model, device, clean, record, detected["zones"], zone_boxes, split, card_rows)
        t_starts = _starts(record.n_profiles, EVAL_TIME, EVAL_TIME_STRIDE)
        x_starts = _starts(len(record.x_m), EVAL_SPACE, EVAL_SPACE_STRIDE)
        crops = []
        boxes = []
        split_codes = []
        for t0 in t_starts:
            t0 = int(t0)
            t1 = t0 + EVAL_TIME
            code = int(crop_partition(split[t0:t1]))
            for x0 in x_starts:
                x0 = int(x0)
                crop = residual_crop(clean, record.valid, t0, t1, x0, x0 + EVAL_SPACE)
                if crop is None:
                    continue
                crops.append(crop)
                boxes.append((t0, t1, x0, x0 + EVAL_SPACE))
                split_codes.append(code)
        if not crops:
            print(f"Record {name} finished windows 0", flush=True)
            continue
        features = np.stack(crops).astype(np.float32)[:, None, :, :]
        logits = _predict(model, features, device)
        classes = logits.argmax(axis=1)
        box_array = np.asarray(boxes, dtype=np.int32)
        hit = window_hits(box_array, zone_boxes)
        is_positive = classes != LABELS.index("Background")
        windows.append(box_array)
        positive.append(is_positive)
        hits.append(hit)
        quiet = _quiet_box(record)
        if quiet is not None:
            quiet_hit = window_hits(box_array, np.asarray([quiet], dtype=np.int32))
            quiet_names.extend(LABELS[int(index)] for index in classes[quiet_hit])
        holdout_names.extend(
            LABELS[int(index)] for index, code in zip(classes, split_codes, strict=True) if code == 2
        )
        print(f"Record {name} finished windows {len(boxes)}", flush=True)
        del detected
    overlap = baseline_overlap(np.concatenate(hits), np.concatenate(positive))
    order = {seed["id"]: index for index, seed in enumerate(EXPERT_SEEDS)}
    cards = sorted(card_rows, key=lambda card: order[card["id"]])
    if [card["id"] for card in cards] != [seed["id"] for seed in EXPERT_SEEDS]:
        raise SystemExit(f"expected 15 card calls, got {[card['id'] for card in cards]}")
    if not np.isfinite(overlap["overlap_jaccard"]):
        raise SystemExit("baseline overlap is not finite")
    if manifest["weights_sha256"] == PREVIOUS_FIT_SHA256:
        raise SystemExit("new weights sha256 matches the previous 10/15 fit")
    by_card = {card["id"]: card for card in cards}
    named = ("review-00223", "next-0002", "next-0010", "next-0006")
    for seed_id in named:
        card = by_card[seed_id]
        if card["previous_network_class"] != PREVIOUS_NETWORK_CLASS[seed_id]:
            raise SystemExit(f"{seed_id} lost its previous network class")
        if "expert_class" not in card or "network_class" not in card:
            raise SystemExit(f"{seed_id} is missing a class field")
    if by_card["review-00223"]["expert_class"] != "Local_Cooling" or by_card["next-0002"]["expert_class"] != "Local_Cooling":
        raise SystemExit("cooling expert class drifted")
    if by_card["next-0010"]["presence"] != "no" or by_card["next-0010"]["expert_class"] != "Background":
        raise SystemExit("next-0010 left Background")
    if by_card["next-0006"]["presence"] != "no" or by_card["next-0006"]["expert_class"] != "Background":
        raise SystemExit("next-0006 left Background")
    if by_card["next-0003"]["expert_class"] != "Local_Heating":
        raise SystemExit("next-0003 left Local_Heating")
    match_count = int(sum(bool(card["match"]) for card in cards))
    comparison_cards = [
        {
            "id": seed_id,
            "expert_class": by_card[seed_id]["expert_class"],
            "previous_network_class": by_card[seed_id]["previous_network_class"],
            "network_class": by_card[seed_id]["network_class"],
            "presence": by_card[seed_id]["presence"],
        }
        for seed_id in named
    ]
    comparison_lines = [
        (
            f"{item['id']} expert {item['expert_class']} "
            f"previous {item['previous_network_class']} "
            f"new {item['network_class']} "
            f"previous_fit {PREVIOUS_FIT_MATCHES}/{PREVIOUS_FIT_CARDS} "
            f"sha256 {PREVIOUS_FIT_SHA256} "
            f"new_sha256 {manifest['weights_sha256']}"
        )
        for item in comparison_cards
    ]
    quiet_mix = class_mix(quiet_names)
    holdout_mix = class_mix(holdout_names)
    report = {
        "schema": "research_cnn_report_v1",
        "serve": "research",
        "operational": False,
        "architecture": manifest["architecture"],
        "weights": manifest["weights"],
        "weights_sha256": manifest["weights_sha256"],
        "weights_bytes": manifest["weights_bytes"],
        "steps": manifest["steps"],
        "holdout_used_for_fit": manifest["holdout_used_for_fit"],
        "guard_used_for_fit": manifest["guard_used_for_fit"],
        "fit_split_codes": manifest["fit_split_codes"],
        "checkpoint_split_codes": manifest["checkpoint_split_codes"],
        "development_loss": manifest.get("development_loss"),
        "cards": cards,
        "match_count": match_count,
        "n_cards": len(cards),
        "previous_match_count": PREVIOUS_FIT_MATCHES,
        "previous_n_cards": PREVIOUS_FIT_CARDS,
        "previous_match_label": f"{PREVIOUS_FIT_MATCHES}/{PREVIOUS_FIT_CARDS}",
        "previous_weights_sha256": PREVIOUS_FIT_SHA256,
        "new_weights_sha256": manifest["weights_sha256"],
        "comparison_cards": comparison_cards,
        "comparison_lines": comparison_lines,
        "not_operational_accuracy": True,
        "expert_seeds_still_fit_on_split_2_or_minus_1": manifest.get(
            "expert_seeds_still_fit_on_split_2_or_minus_1", []
        ),
        "expert_cards_are_not_an_untouched_holdout": True,
        "baseline_zone_overlap": round(overlap["overlap_jaccard"], 6),
        "positive_hit_fraction": round(overlap["positive_hit_fraction"], 6),
        "overlap_counts": {key: overlap[key] for key in (
            "n_windows", "n_positive", "n_zone_windows", "intersection", "union")},
        "quiet_control": {"id": WITNESS, **quiet_mix},
        "holdout": holdout_mix,
        "examples": [card["example"] for card in cards],
        "note": (
            "Measurements of a research network. Card recovery is not operational accuracy "
            "and not an untouched holdout score: expert cards on split 2 or -1 were fit as seeds. "
            "analyze_field stays median_x3 + spatial_z_halo at |Z| >= 3.5. "
            f"Previous fit was {PREVIOUS_FIT_MATCHES}/{PREVIOUS_FIT_CARDS} at sha256 {PREVIOUS_FIT_SHA256}. "
            f"This fit matches {match_count}/{len(cards)}. Expert classes are not relabeled."
        ),
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    payload = {
        "cards": [
            {
                "id": card["id"],
                "recorded": card["recorded"],
                "predicted": card["predicted"],
                "presence": card["presence"],
                "split_code": card["split_code"],
                "baseline_overlaps": card["baseline_overlaps"],
                "baseline_contrast_c": card["baseline_contrast_c"],
                "baseline_note": card["baseline_note"],
                "untouched_holdout": card["untouched_holdout"],
            }
            for card in cards
        ],
        "baseline_zone_overlap": report["baseline_zone_overlap"],
        "positive_hit_fraction": report["positive_hit_fraction"],
        "quiet_control": report["quiet_control"],
        "holdout": report["holdout"],
        "holdout_used_for_fit": False,
        "guard_used_for_fit": False,
        "development_loss": report["development_loss"],
        "serve": "research",
        "match_count": match_count,
        "previous_match_label": report["previous_match_label"],
        "new_weights_sha256": report["new_weights_sha256"],
        "comparison_lines": comparison_lines,
    }
    print(
        f"serve=research development_loss {report['development_loss']} "
        f"matches {match_count}/{len(cards)} previous {PREVIOUS_FIT_MATCHES}/{PREVIOUS_FIT_CARDS}",
        flush=True,
    )
    for line in comparison_lines:
        print("COMPARE " + line, flush=True)
    print("METRICS " + json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
