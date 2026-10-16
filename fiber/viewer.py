"""Heatmap and zones for the operational field path.

``view_field`` calls ``analyze_field``. It does not retune ``|Z|`` and it does
not read hybrid refine, ``temporal_innovation`` as a status, or the research
weights. ``expert_windows`` only displays stored research JSON: the residual
report, the context report, the spatial-versus-expert measurement, context
evidence, and the wide-channel file.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fiber.pipeline import analyze_field

_ROOT = Path(__file__).resolve().parents[1]


def example_hotspot():
    """Local hot columns on a flat cable. The built-in viewer example."""
    raw = np.full((12, 100), 22.0, dtype=np.float32)
    raw[:, 45:51] += 6.0
    x_m = np.arange(100, dtype=np.float64)
    time_s = np.arange(12, dtype=np.float64) * 60.0
    return raw, x_m, time_s


def example_cable_step():
    """One step on every measured column. The spatial halo stays quiet."""
    raw = np.full((16, 120), 20.0, dtype=np.float32)
    raw[8:, :] = 28.0
    x_m = np.arange(120, dtype=np.float64)
    time_s = np.arange(16, dtype=np.float64) * 60.0
    return raw, x_m, time_s


def _contrast_text(values: list[float]) -> str:
    if not values:
        return "baseline kept no zone"
    parts = [format(value, ".6f").rstrip("0").rstrip(".") for value in values]
    return "baseline contrast " + ", ".join(parts) + " C"


def _stored_number(value):
    """Keep a stored median. An absent far side stays absent."""
    if value is None:
        return None
    return float(value)


def expert_windows(root: Path | None = None) -> dict:
    """Research card windows beside the saved baseline measurement.

    The previous 15 keep the residual-report classes. Baseline contrasts for
    a card that was remeasured come from that measurement; the other residual
    cards use the contrasts stored on the residual report. Their rectangle
    median, far-cable median, and halo-inside fraction come from the
    context-evidence file. Scored ask expert seeds are appended in
    context-report order. Their expert class, context-network class, and the
    three context numbers come from that stored card. An ask window has no
    baseline contrast. An absent far median is not copied from the rectangle
    median. Wide-channel rows come from the wide-channel file. This does
    not load research weights and does not call ``analyze_field``.
    """
    base = _ROOT if root is None else Path(root)
    evidence = base / "docs" / "evidence"
    research = json.loads((evidence / "research_cnn_report.json").read_text(encoding="utf-8"))
    spatial = json.loads((evidence / "spatial_vs_expert.json").read_text(encoding="utf-8"))
    context = json.loads((evidence / "context_evidence.json").read_text(encoding="utf-8"))
    wide = json.loads((evidence / "wide_channel.json").read_text(encoding="utf-8"))
    context_net = json.loads((evidence / "research_context_cnn_report.json").read_text(encoding="utf-8"))
    if research.get("serve") != "research" or research.get("operational") is True:
        raise ValueError("research report is not a serve=research measurement")
    if context.get("serve") != "research" or context.get("operational") is True:
        raise ValueError("context evidence is not a serve=research measurement")
    if wide.get("serve") != "research" or wide.get("operational") is True:
        raise ValueError("wide-channel file is not a serve=research measurement")
    if context_net.get("serve") != "research" or context_net.get("operational") is True:
        raise ValueError("context report is not a serve=research measurement")
    spatial_by_id = {str(card["id"]): card for card in spatial["cards"]}
    context_by_id = {str(card["id"]): card for card in context["cards"]}
    windows = []
    for card in research["cards"]:
        expert = str(card["expert_class"])
        network = str(card["network_class"])
        confirmed = expert == network
        span = spatial_by_id.get(str(card["id"]))
        if span is not None:
            contrasts = [float(zone["contrast"]) for zone in span["in_box"]]
            source = "spatial_vs_expert"
        elif card.get("baseline_overlaps"):
            contrasts = [float(zone["contrast_c"]) for zone in card["baseline_zones"]]
            source = "research_cnn_report"
        else:
            contrasts = []
            source = "research_cnn_report"
        measured = context_by_id.get(str(card["id"]))
        if measured is None:
            raise ValueError(f"context evidence missing {card['id']}")
        rectangle = _stored_number(measured["rectangle_median_c"])
        far_cable = _stored_number(measured["far_cable_median_c"])
        windows.append({
            "id": str(card["id"]),
            "expert_class": expert,
            "network_class": network,
            "confirmed": confirmed,
            "verdict": "confirmed" if confirmed else "refuted",
            "baseline_overlaps": bool(contrasts),
            "baseline_contrasts_c": contrasts,
            "baseline_text": _contrast_text(contrasts),
            "baseline_source": source,
            "rectangle_median_c": rectangle,
            "far_cable_median_c": far_cable,
            "halo_inside_fraction": _stored_number(measured["halo_inside_fraction"]),
            "serve": "research",
            "operational": False,
        })
    shown = {row["id"] for row in windows}
    ask_seen: set[str] = set()
    for card in context_net["cards"]:
        if card.get("source") != "ask_expert_seed":
            continue
        card_id = str(card["id"])
        if card_id in ask_seen:
            raise ValueError(f"duplicate ask window {card_id}")
        ask_seen.add(card_id)
        if card_id in shown:
            continue
        expert = str(card["expert_class"])
        network = str(card["network_class"])
        confirmed = expert == network
        windows.append({
            "id": card_id,
            "expert_class": expert,
            "network_class": network,
            "confirmed": confirmed,
            "verdict": "confirmed" if confirmed else "refuted",
            "baseline_overlaps": False,
            "baseline_contrasts_c": [],
            "baseline_text": _contrast_text([]),
            "baseline_source": "research_context_cnn_report",
            "rectangle_median_c": _stored_number(card["rectangle_median_c"]),
            "far_cable_median_c": _stored_number(card["far_cable_median_c"]),
            "halo_inside_fraction": _stored_number(card["halo_inside_fraction"]),
            "serve": "research",
            "operational": False,
        })
    wide_rows = []
    for card in wide["cards"]:
        rectangle = _stored_number(card["rectangle_median_c"])
        far_cable = _stored_number(card["far_cable_median_c"])
        wide_rows.append({
            "id": str(card["id"]),
            "wide_channel": bool(card["wide_channel"]),
            "wide_channel_candidate": card["wide_channel_candidate"],
            "expert_class": card["expert_class"],
            "rectangle_median_c": rectangle,
            "far_cable_median_c": far_cable,
            "halo_inside_fraction": _stored_number(card["halo_inside_fraction"]),
            "far_gap_c": _stored_number(card["far_gap_c"]),
            "serve": "research",
            "operational": False,
        })
    return {
        "serve": "research",
        "operational": False,
        "trained_classifier_operational": False,
        "detector": "spatial_z_halo",
        "path": "research-display",
        "note": (
            "Card recovery is not operational accuracy. "
            "The operational detector stays median_x3 and spatial_z_halo at |Z| >= 3.5. "
            "Context medians and the wide-channel flag do not set anomaly_status."
        ),
        "windows": windows,
        "wide_channels": wide_rows,
    }


def view_field(raw, x_m, time_s) -> dict:
    """Operational view: cleaned heatmap plus the zones ``analyze_field`` kept."""
    result = analyze_field(raw, x_m, time_s)
    heat = np.asarray(result["clean"], dtype=np.float64)
    return {
        "path": "operational",
        "detector": "spatial_z_halo",
        "anomaly_status": result["anomaly_status"],
        "manifest": result["manifest"],
        "heatmap": np.round(heat, 4).tolist(),
        "x_m": [float(value) for value in np.asarray(x_m, dtype=np.float64)],
        "time_s": [float(value) for value in np.asarray(time_s, dtype=np.float64)],
        "zones": result["zones"],
        "n_zones": len(result["zones"]),
    }
