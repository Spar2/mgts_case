"""Sliding-window labels for a long trace.

The crop subtracts its own median, so a window that sits entirely inside a
uniform shift looks like background. Broad_Shift is reserved for a window
that straddles the cut. These cuts are the search protocol. They are not
retuned on witness windows.
"""

from __future__ import annotations

import numpy as np

from fiber.classify.rules import LABELS
from fiber.models.patch_classifier import SPACE_CROP, TIME_CROP, crop_deviation

LABEL_INDEX = {name: index for index, name in enumerate(LABELS)}
EVENT_LABELS = (
    "Local_Heating",
    "Local_Cooling",
    "Periodic_Thermal",
    "Broad_Shift",
    "Sensor_Fault",
)

TIME_STRIDE = 14
SPACE_STRIDE = 10
MIN_VALID_FRACTION = 0.5
MIN_CLASS_CELLS = 8
HOT_PASS_FRACTION = 0.3


def stops(length: int, crop: int, stride: int) -> np.ndarray:
    """Window starts. The last start is included so the tail cell is covered."""
    if length < crop or stride <= 0:
        return np.zeros(0, dtype=np.int64)
    starts = list(range(0, length - crop + 1, stride))
    last = length - crop
    if starts[-1] != last:
        starts.append(last)
    return np.asarray(starts, dtype=np.int64)


def window_origins(n_time: int, n_space: int) -> np.ndarray:
    time_starts = stops(n_time, TIME_CROP, TIME_STRIDE)
    space_starts = stops(n_space, SPACE_CROP, SPACE_STRIDE)
    if time_starts.size == 0 or space_starts.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    grid = np.stack(np.meshgrid(time_starts, space_starts, indexing="ij"), axis=-1)
    return np.ascontiguousarray(grid.reshape(-1, 2), dtype=np.int64)


def window_valid_counts(valid: np.ndarray, origins: np.ndarray) -> np.ndarray:
    mask = np.asarray(valid, dtype=np.int32)
    integral = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dtype=np.int32)
    integral[1:, 1:] = mask.cumsum(axis=0).cumsum(axis=1)
    t0 = origins[:, 0]
    x0 = origins[:, 1]
    t1 = t0 + TIME_CROP
    x1 = x0 + SPACE_CROP
    return integral[t1, x1] - integral[t0, x1] - integral[t1, x0] + integral[t0, x0]


def _enough_valid(count: int) -> bool:
    return int(count) * 2 >= TIME_CROP * SPACE_CROP


def window_label(labels: np.ndarray, valid: np.ndarray, t0: int, x0: int) -> str | None:
    """Class a search window should return, or None when the crop is mostly invalid.

    Sensor_Fault wins on a single valid cell. Broad_Shift wins only when both
    sides of the cut have at least MIN_CLASS_CELLS valid cells. Any other
    class needs MIN_CLASS_CELLS and then the larger count.
    """
    lab = np.asarray(labels[t0:t0 + TIME_CROP, x0:x0 + SPACE_CROP])
    mask = np.asarray(valid[t0:t0 + TIME_CROP, x0:x0 + SPACE_CROP], dtype=bool)
    if mask.shape != (TIME_CROP, SPACE_CROP) or not _enough_valid(int(mask.sum())):
        return None
    counts = {
        name: int(np.count_nonzero((lab == LABEL_INDEX[name]) & mask))
        for name in EVENT_LABELS
    }
    if counts["Sensor_Fault"] >= 1:
        return "Sensor_Fault"
    non_broad = int(mask.sum()) - counts["Broad_Shift"]
    if counts["Broad_Shift"] >= MIN_CLASS_CELLS and non_broad >= MIN_CLASS_CELLS:
        return "Broad_Shift"
    ranked = ("Local_Heating", "Local_Cooling", "Periodic_Thermal")
    best = max(ranked, key=lambda name: (counts[name], -ranked.index(name)))
    if counts[best] >= MIN_CLASS_CELLS:
        return best
    return "Background"


def labeled_windows(labels: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, list[str]]:
    origins = window_origins(labels.shape[0], labels.shape[1])
    if len(origins) == 0:
        return origins, []
    counts = window_valid_counts(valid, origins)
    keep = np.flatnonzero(counts * 2 >= TIME_CROP * SPACE_CROP)
    kept_origins = origins[keep]
    truth = [window_label(labels, valid, int(t0), int(x0)) for t0, x0 in kept_origins]
    if any(name is None for name in truth):
        raise RuntimeError("valid-count gate and window_label disagreed")
    return kept_origins, truth


def scored_origins(valid: np.ndarray) -> np.ndarray:
    origins = window_origins(valid.shape[0], valid.shape[1])
    if len(origins) == 0:
        return origins
    counts = window_valid_counts(valid, origins)
    return origins[counts * 2 >= TIME_CROP * SPACE_CROP]


def deviation_crops(field: np.ndarray, valid: np.ndarray, origins: np.ndarray) -> np.ndarray:
    origins = np.asarray(origins, dtype=np.int64)
    out = np.zeros((len(origins), TIME_CROP, SPACE_CROP), dtype=np.float32)
    for index, (t0, x0) in enumerate(origins):
        out[index] = crop_deviation(field, valid, int(t0), int(x0))
    return out


def locator_table(event_id: np.ndarray, origins: np.ndarray, truth: list[str], events) -> dict[int, list[int]]:
    """Windows whose truth equals the event and that touch its cells."""
    table: dict[int, list[int]] = {int(event["id"]): [] for event in events}
    labels = {int(event["id"]): event["label"] for event in events}
    for index, ((t0, x0), name) in enumerate(zip(origins, truth, strict=True)):
        crop = event_id[int(t0):int(t0) + TIME_CROP, int(x0):int(x0) + SPACE_CROP]
        present = {int(item) for item in np.unique(crop) if int(item) != 0}
        for eid in present:
            if name == labels.get(eid):
                table[eid].append(index)
    return table


def score_search(truth: list[str], predicted: list[str], events, locators: dict[int, list[int]]) -> dict:
    if len(truth) != len(predicted):
        raise ValueError("truth and predicted windows differ")
    rows = []
    for event in events:
        indexes = locators[int(event["id"])]
        hit = any(predicted[index] == event["label"] for index in indexes)
        rows.append({
            "id": int(event["id"]),
            "label": event["label"],
            "n_locators": len(indexes),
            "hit": bool(hit),
        })
    background = [pred for label, pred in zip(truth, predicted, strict=True) if label == "Background"]
    false_alarms = sum(pred != "Background" for pred in background)
    n_agree = sum(label == pred for label, pred in zip(truth, predicted, strict=True))
    per_class = {}
    for name in EVENT_LABELS:
        chosen = [row for row in rows if row["label"] == name]
        per_class[name] = (sum(row["hit"] for row in chosen) / len(chosen)) if chosen else None
    n_events = len(rows)
    n_windows = len(truth)
    return {
        "event_recall": (sum(row["hit"] for row in rows) / n_events) if n_events else None,
        "false_alarm_rate": (false_alarms / len(background)) if background else None,
        "window_agreement": (n_agree / n_windows) if n_windows else None,
        "per_class_event_recall": per_class,
        "n_events": n_events,
        "n_background_windows": len(background),
        "n_false_alarms": false_alarms,
        "n_agreements": n_agree,
        "n_windows": n_windows,
        "events": rows,
    }


def pool_search(parts: list[dict]) -> dict:
    rows = [row for part in parts for row in part["events"]]
    n_background = sum(int(part["n_background_windows"]) for part in parts)
    n_false = sum(int(part["n_false_alarms"]) for part in parts)
    n_agree = sum(int(part["n_agreements"]) for part in parts)
    n_windows = sum(int(part["n_windows"]) for part in parts)
    per_class = {}
    for name in EVENT_LABELS:
        chosen = [row for row in rows if row["label"] == name]
        per_class[name] = (sum(row["hit"] for row in chosen) / len(chosen)) if chosen else None
    n_events = len(rows)
    return {
        "event_recall": (sum(row["hit"] for row in rows) / n_events) if n_events else None,
        "false_alarm_rate": (n_false / n_background) if n_background else None,
        "window_agreement": (n_agree / n_windows) if n_windows else None,
        "per_class_event_recall": per_class,
        "n_events": n_events,
        "n_background_windows": n_background,
        "n_false_alarms": n_false,
        "n_agreements": n_agree,
        "n_windows": n_windows,
        "n_cables": len(parts),
    }


def choose_better(challenger: dict, baseline: dict | None) -> str:
    """Pick a checkpoint from planted search scores only.

    Higher event recall wins. A tie goes to the lower background false-alarm
    rate. A remaining tie, or a missing baseline, keeps the challenger.
    """
    if baseline is None or baseline.get("event_recall") is None:
        return "challenger"
    if challenger["event_recall"] > baseline["event_recall"]:
        return "challenger"
    if baseline["event_recall"] > challenger["event_recall"]:
        return "baseline"
    if challenger["false_alarm_rate"] < baseline["false_alarm_rate"]:
        return "challenger"
    if baseline["false_alarm_rate"] < challenger["false_alarm_rate"]:
        return "baseline"
    return "challenger"


def choose_weights(scanner: dict, naming: dict | None) -> str:
    """Scanner versus the naming network. Same rule as `choose_better`."""
    picked = choose_better(scanner, naming)
    return "scanner" if picked == "challenger" else "naming"


def overlapping_indices(
    origins: np.ndarray,
    x_m: np.ndarray,
    row0: int,
    row1: int,
    meter0: float,
    meter1: float,
) -> np.ndarray:
    """Overlap with half-open row interval [row0, row1), inclusive metres."""
    if len(origins) == 0:
        return np.zeros(0, dtype=np.int64)
    t0 = origins[:, 0]
    x0 = origins[:, 1]
    time_hit = (t0 < row1) & (t0 + TIME_CROP > row0)
    in_span = (x_m >= meter0) & (x_m <= meter1)
    prefix = np.zeros(len(x_m) + 1, dtype=np.int32)
    prefix[1:] = np.cumsum(in_span.astype(np.int32))
    space_hit = (prefix[np.minimum(x0 + SPACE_CROP, len(x_m))] - prefix[x0]) > 0
    return np.flatnonzero(time_hit & space_hit)


def summarize_witness(role: str, names: list[str]) -> dict:
    """Frozen witness rule for a sliding classifier. Argmax names, no probability cut.

    positive_hot passes when at least 30% of overlapping scored windows say
    Local_Heating. negative_quiet passes only when every such window says
    Background and there is at least one. negative_invalid passes only when
    the dead area leaves no scored window.
    """
    counts = {name: sum(item == name for item in names) for name in LABELS}
    n = len(names)
    if n == 0:
        heating = None
        passed = role == "negative_invalid"
    else:
        heating = counts["Local_Heating"] / n
        if role == "positive_hot":
            passed = heating >= HOT_PASS_FRACTION
        elif role == "negative_quiet":
            passed = counts["Background"] == n
        else:
            passed = False
    return {
        "role": role,
        "n_windows": n,
        "local_heating_fraction": heating,
        "counts": counts,
        "pass": bool(passed),
        "expert_confirmed": False,
    }


def decide_witnesses(
    specs: list[dict],
    origins: np.ndarray,
    x_m: np.ndarray,
    names: list[str],
    record: str,
) -> list[dict]:
    """One witness pass for a record: overlap, then `summarize_witness`.

    Scans call this instead of copying the box loop. The pass rule stays in
    `summarize_witness` and is not retuned here.
    """
    rows = []
    for spec in specs:
        hit = overlapping_indices(
            origins,
            x_m,
            int(spec["rows"][0]),
            int(spec["rows"][1]),
            float(spec["x_m"][0]),
            float(spec["x_m"][1]),
        )
        summary = summarize_witness(spec["role"], [names[int(index)] for index in hit])
        summary.update({
            "id": spec["id"],
            "record": record,
            "x_m": spec["x_m"],
            "rows": spec["rows"],
        })
        rows.append(summary)
    return rows
