"""Research shortlist of about 150 unlabeled cards.

Selection is pure: candidate rows in, a quota per bucket out. Answered
review-*, next-*, and gap-* rectangles, holdout split 2, and guard split -1
stay out. The writer measures rectangles once. It does not fill an expert
class and it does not retune |Z| or the published wide-channel flag.
"""

from __future__ import annotations

import copy
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np

from fiber.weak_labels import EXPERT_SEEDS, load_gap_answers

QUOTA_BUCKETS = (
    "wide",
    "hard negative",
    "cooling",
    "residual-versus-context",
)
BUCKETS = QUOTA_BUCKETS + ("local",)
QUOTAS = {
    "wide": {"floor": 40, "target": 45, "ceiling": 50},
    "hard negative": {"floor": 30, "target": 35, "ceiling": 40},
    "cooling": {"floor": 20, "target": 25, "ceiling": 30},
    "residual-versus-context": {"floor": 20, "target": 25, "ceiling": 30},
}
MIN_TOTAL = 140
MAX_TOTAL = 160
TARGET_TOTAL = 150
CONFIRMED_IDS = frozenset(seed["id"] for seed in EXPERT_SEEDS)
EXCLUDED_SPLIT_CODES = frozenset({2, -1})
ANSWERED_PREFIXES = ("review-", "next-", "gap-")
RECORDS = frozenset({"1", "2", "3", "4", "5", "6"})
THRESHOLD_Z = 3.5
INNER_M = 8
OUTER_M = 40
WIDTH_LIMIT_M = 40
WIDE_HALO_MIN = 0.6
WIDE_FAR_GAP_MIN_C = 1.5
HARD_ALARM_FRACTION_MAX = 0.05
HARD_CONTRAST_MIN_C = 1.5
# Far gap below this is the same temperature, not a cold window.
COOLING_FAR_GAP_MIN_C = 0.05
UNION_GAP_M = 8
UNION_MAX_WIDTH_M = 150
COLUMN_RUN_CAP = 104
_CITED_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_ROW_SUFFIX = re.compile(r"-t(\d+)-(\d+)$")
# On the previous 12-card shortlist, absent from the gap answer file.
_GAP_SHORTLIST_ONLY = (
    {
        "id": "gap-r6-x1116.354-x1127.774-t385-418",
        "record": "6",
        "meters": (1116.354, 1127.774),
        "rows": (385, 418),
    },
    {
        "id": "gap-r6-x30.199-x44.157-t0-705",
        "record": "6",
        "meters": (30.199, 44.157),
        "rows": (0, 705),
    },
)


class ShortlistSelection(list):
    """Cards from select_shortlist, plus pool and taken counts."""

    def __init__(self, cards, stats):
        super().__init__(cards)
        self.stats = stats


def _load_answered():
    boxes = []
    ids = []

    def add(card_id, record, meters, rows):
        ids.append(str(card_id))
        boxes.append((str(record), float(meters[0]), float(meters[1]), int(rows[0]), int(rows[1])))

    for seed in EXPERT_SEEDS:
        add(seed["id"], seed["record"], seed["x_m"], seed["rows_half_open"])
    for answer in load_gap_answers():
        match = _ROW_SUFFIX.search(str(answer["id"]))
        if match is None:
            raise ValueError(f"{answer['id']} has no row suffix")
        add(
            answer["id"],
            answer["record"],
            answer["meters"],
            (int(match.group(1)), int(match.group(2))),
        )
    known = set(ids)
    for extra in _GAP_SHORTLIST_ONLY:
        if extra["id"] in known:
            continue
        add(extra["id"], extra["record"], extra["meters"], extra["rows"])
    return tuple(boxes), frozenset(ids)


ANSWERED_BOXES, ANSWERED_IDS = _load_answered()


def select_shortlist(candidates):
    """Pick disjoint unlabeled cards by bucket quota.

    A bucket whose disjoint pool is below its floor contributes that whole
    pool and is marked short. Nothing answered, overlapping, or on holdout
    or guard is added to fill the gap. Two calls on the same rows return the
    same ids.
    """
    eligible = []
    seen = set()
    for row in candidates:
        if not _eligible(row):
            continue
        if row["id"] in seen:
            continue
        seen.add(row["id"])
        eligible.append(row)
    eligible.sort(key=lambda row: (-float(row.get("priority", 0.0)), str(row["id"])))
    pools = {bucket: [] for bucket in BUCKETS}
    for row in eligible:
        pools[row["bucket"]].append(row)

    picked = []
    blocked = []
    stats = {}
    for bucket in QUOTA_BUCKETS:
        quota = QUOTAS[bucket]
        pool = _disjoint_pool(pools[bucket], blocked)
        if len(pool) < quota["floor"]:
            chosen = pool
            shortfall = True
        else:
            chosen = pool[: quota["target"]]
            shortfall = False
        picked.extend(chosen)
        blocked.extend(_row_box(row) for row in chosen)
        stats[bucket] = {
            "eligible": len(pools[bucket]),
            "pool": len(pool),
            "taken": len(chosen),
            "floor": quota["floor"],
            "target": quota["target"],
            "ceiling": quota["ceiling"],
            "shortfall": shortfall,
        }

    local_pool = _disjoint_pool(pools["local"], blocked)
    room = MAX_TOTAL - len(picked)
    need = max(0, TARGET_TOTAL - len(picked))
    if room <= 0 or not local_pool:
        local_chosen = []
    else:
        take_n = min(len(local_pool), room, max(need, 1))
        local_chosen = _diverse_local(local_pool, take_n)
    picked.extend(local_chosen)
    stats["local"] = {
        "eligible": len(pools["local"]),
        "pool": len(local_pool),
        "taken": len(local_chosen),
        "floor": 1,
        "target": need,
        "ceiling": max(0, room),
        "shortfall": len(local_pool) < 1,
    }
    cards = [_card(row) for row in picked]
    return ShortlistSelection(cards, stats)


def build_candidates(catalog, temporal):
    """Measure records 1–6 once. Selection stays in select_shortlist."""
    sha = {str(block["record"]): block["source_sha256"] for block in temporal["files"]}
    models = _load_models()
    rows = []
    seen = set()
    for record in ("1", "2", "3", "4", "5", "6"):
        for skeleton in _measure_record(record, catalog, sha[record], models):
            if skeleton["id"] in seen:
                continue
            bucket = _primary_bucket(skeleton["measurements"])
            if bucket is None:
                continue
            seen.add(skeleton["id"])
            rows.append(_finish(skeleton, bucket, _priority(bucket, skeleton["measurements"])))
    counts = {}
    wide_measure = 0
    wide_disagree = 0
    for row in rows:
        counts[row["bucket"]] = counts.get(row["bucket"], 0) + 1
        measured = row["measurements"]
        halo = measured.get("halo_inside_fraction")
        far = measured.get("far_cable_median_c")
        gap = measured.get("abs_far_gap_c")
        if halo is not None and far is not None and gap is not None and float(halo) >= WIDE_HALO_MIN and float(gap) >= WIDE_FAR_GAP_MIN_C:
            wide_measure += 1
            if row["bucket"] == "residual-versus-context":
                wide_disagree += 1
    print(
        f"shortlist candidates {len(rows)} {counts} wide_measure {wide_measure} wide_disagree {wide_disagree}",
        file=sys.stderr,
        flush=True,
    )
    return rows


def build_shortlist(catalog, temporal, queue):
    """Select the research cards and record pool sizes beside the taken counts."""
    selected = select_shortlist(build_candidates(catalog, temporal))
    cards = sorted(selected, key=lambda card: card["id"])
    if not cards:
        raise RuntimeError("selector returned no cards")
    counts = {bucket: 0 for bucket in BUCKETS}
    for card in cards:
        counts[card["bucket"]] += 1
    for bucket in BUCKETS:
        if counts[bucket] != selected.stats[bucket]["taken"]:
            raise RuntimeError(f"{bucket} count {counts[bucket]} != taken {selected.stats[bucket]['taken']}")
    n_items = int(queue["n_items"])
    n_cards = len(cards)
    shortfalls = [
        f"{bucket}: пул {selected.stats[bucket]['pool']}, взято {selected.stats[bucket]['taken']}"
        for bucket in QUOTA_BUCKETS
        if selected.stats[bucket]["shortfall"]
    ]
    shortfall_text = (
        " Короткие пулы не дополнялись отвеченными или перекрывающимися окнами: " + "; ".join(shortfalls) + "."
        if shortfalls
        else " Каждый квотируемый пул не меньше своей нижней границы."
    )
    summary = (
        f"Отобрано {n_cards} карточек для research. "
        f"Широкий канал здесь: доля кольца не ниже {WIDE_HALO_MIN} и модуль разницы с far cable "
        f"не ниже {WIDE_FAR_GAP_MIN_C} °C. Это не опубликованный флаг 0.7 и 2.0 °C. "
        f"Исключены уже отвеченные id: {len(ANSWERED_IDS)}. "
        f"Очередь scripts/prepare_annotation_queue.py содержит {n_items} окон и сюда не копируется. "
        f"Порог |Z| остаётся {THRESHOLD_Z}, median_x3 и spatial_z_halo не менялись. "
        "expert_confirmed на карточках false, класс эксперта пуст. serve=research, operational false."
        + shortfall_text
    )
    return {
        "schema": "research_shortlist_v1",
        "audience": "Захар",
        "purpose": "Короткий research-список. Это не полная очередь разметки и не экспертные классы.",
        "serve": "research",
        "operational": False,
        "expert_confirmed": False,
        "n_cards": n_cards,
        "not_the_full_queue": True,
        "full_queue": {
            "script": "scripts/prepare_annotation_queue.py",
            "n_items": n_items,
            "n_detector_proposals": int(queue["n_detector_proposals"]),
            "n_random_review": int(queue["n_random_review"]),
        },
        "why_not_the_full_queue": summary,
        "excluded_answered_id_count": len(ANSWERED_IDS),
        "excluded_answered_ids": sorted(ANSWERED_IDS),
        "wide_halo_min": WIDE_HALO_MIN,
        "wide_far_gap_min_c": WIDE_FAR_GAP_MIN_C,
        "buckets": counts,
        "bucket_stats": selected.stats,
        "clean": "median_x3",
        "detector": "spatial_z_halo",
        "threshold_z": THRESHOLD_Z,
        "cards": cards,
    }


def write_shortlist(root=None) -> str:
    """Write shortlist.json. A second call with the same inputs writes the same bytes."""
    catalog, temporal, queue = load_shortlist_inputs(root)
    document = build_shortlist(catalog, temporal, queue)
    text = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    evidence_paths(root)["shortlist"].write_text(text, encoding="utf-8")
    return text


def evidence_paths(root=None):
    base = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    folder = base / "docs" / "evidence"
    return {
        "catalog": folder / "zone_catalog.json",
        "temporal": folder / "temporal_innovation.json",
        "queue": folder / "annotation_queue_manifest.json",
        "shortlist": base / "shortlist.json",
        "records": base / "artifacts" / "records",
    }


def load_shortlist_inputs(root=None):
    paths = evidence_paths(root)
    return (
        json.loads(paths["catalog"].read_text(encoding="utf-8")),
        json.loads(paths["temporal"].read_text(encoding="utf-8")),
        json.loads(paths["queue"].read_text(encoding="utf-8")),
    )


def _eligible(row):
    if not isinstance(row, dict):
        return False
    card_id = row.get("id")
    if not isinstance(card_id, str) or card_id in ANSWERED_IDS:
        return False
    if card_id.startswith(ANSWERED_PREFIXES):
        return False
    if row.get("expert_confirmed") is True:
        return False
    if row.get("bucket") not in BUCKETS:
        return False
    if str(row.get("record")) not in RECORDS:
        return False
    code = row.get("split_code")
    if code is not None and int(code) in EXCLUDED_SPLIT_CODES:
        return False
    if _row_box(row) is None or _overlaps_answered(row):
        return False
    reason = row.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return False
    return True


def _card(row):
    card = copy.deepcopy(row)
    card.pop("priority", None)
    card.pop("split_code", None)
    card.pop("flags", None)
    card["expert_confirmed"] = False
    card["expert_class"] = None
    return card


def _row_box(row):
    rows = row.get("rows_half_open")
    meters = row.get("meters")
    if rows is None or meters is None:
        return None
    if len(rows) != 2 or len(meters) != 2:
        return None
    return (str(row.get("record")), float(meters[0]), float(meters[1]), int(rows[0]), int(rows[1]))


def _hit(left, right):
    if left is None or right is None or left[0] != right[0]:
        return False
    return left[1] < right[2] and right[1] < left[2] and left[3] < right[4] and right[3] < left[4]


def _overlaps_answered(row):
    box = _row_box(row)
    return any(_hit(box, seed) for seed in ANSWERED_BOXES)


def _q(value, digits=3):
    number = float(f"{float(value):.{digits}f}")
    if number == 0:
        number = 0.0
    return number


def _n(measurements, key):
    return json.dumps(measurements[key], ensure_ascii=False)


def _grounded(reason, measurements):
    blob = json.dumps(measurements, ensure_ascii=False)
    missing = [
        token for token in _CITED_NUMBER.findall(reason)
        if re.search(rf"(?<![\d.]){re.escape(token)}(?![\d.])", blob) is None
    ]
    if missing:
        raise ValueError(f"reason cites {missing} missing from measurements: {reason}")
    return reason


def _common(record):
    return {
        "record": str(record),
        "clean": "median_x3",
        "detector": "spatial_z_halo",
        "threshold_z": THRESHOLD_Z,
        "inner_m": INNER_M,
        "outer_m": OUTER_M,
        "wide_halo_min": WIDE_HALO_MIN,
        "wide_far_gap_min_c": WIDE_FAR_GAP_MIN_C,
        "hard_alarm_fraction_max": HARD_ALARM_FRACTION_MAX,
        "hard_contrast_min_c": HARD_CONTRAST_MIN_C,
        "cooling_far_gap_min_c": COOLING_FAR_GAP_MIN_C,
    }


def _class_pair(measurements):
    from fiber.classify.rules import LABELS

    residual = measurements.get("residual_class")
    context = measurements.get("context_class")
    if residual in LABELS and context in LABELS and residual != context:
        return residual, context
    return None


def _primary_bucket(measurements):
    """One bucket. Disagreement, then wide, cooling, hard negative, then a local zone."""
    if _class_pair(measurements) is not None:
        return "residual-versus-context"
    halo = measurements.get("halo_inside_fraction")
    far = measurements.get("far_cable_median_c")
    gap = measurements.get("abs_far_gap_c")
    if (
        halo is not None and far is not None and gap is not None
        and float(halo) >= WIDE_HALO_MIN and float(gap) >= WIDE_FAR_GAP_MIN_C
    ):
        return "wide"
    if _is_cooling(measurements):
        return "cooling"
    inbox = int(measurements.get("in_box_count") or 0)
    alarm_fraction = measurements.get("alarm_fraction")
    flank = measurements.get("abs_flank_contrast_c")
    low_alarm = inbox == 0 or (alarm_fraction is not None and float(alarm_fraction) < HARD_ALARM_FRACTION_MAX)
    flank_contrast = flank is not None and float(flank) >= HARD_CONTRAST_MIN_C
    far_contrast = far is not None and gap is not None and float(gap) >= HARD_CONTRAST_MIN_C
    if low_alarm and (flank_contrast or far_contrast):
        return "hard negative"
    if measurements.get("origin") == "catalog_zone":
        return "local"
    return None


def _priority(bucket, measurements):
    gap = float(measurements["abs_far_gap_c"]) if measurements.get("abs_far_gap_c") is not None else 0.0
    halo = float(measurements["halo_inside_fraction"] or 0.0)
    flank = (
        float(measurements["abs_flank_contrast_c"])
        if measurements.get("abs_flank_contrast_c") is not None else 0.0
    )
    contrast = float(measurements["contrast_c"]) if measurements.get("contrast_c") is not None else 0.0
    if bucket == "wide":
        return halo * 1000.0 + gap
    if bucket == "hard negative":
        return max(flank, gap) * 10.0 + (1.0 if int(measurements.get("in_box_count") or 0) == 0 else 0.0)
    if bucket == "cooling":
        cold = 0.0
        if contrast < 0:
            cold = max(cold, -contrast)
        signed_gap = measurements.get("far_gap_c")
        if signed_gap is not None and float(signed_gap) < 0:
            cold = max(cold, -float(signed_gap))
        if measurements.get("shape_hypothesis") == "Local_Cooling":
            cold += 100.0
        return cold
    if bucket == "residual-versus-context":
        return 10.0 + gap
    return abs(contrast)


def _is_cooling(measurements):
    """Local_Cooling, or a window colder than its surroundings.

    ``residual_c`` is the mean of a crop minus its own median. That is skew,
    not a temperature below the flanks or the far cable, so it does not count.
    """
    if measurements.get("shape_hypothesis") == "Local_Cooling":
        return True
    contrast = measurements.get("contrast_c")
    if contrast is not None and float(contrast) < 0:
        return True
    gap = measurements.get("far_gap_c")
    far = measurements.get("far_cable_median_c")
    if far is None or gap is None or float(gap) >= 0:
        return False
    return float(measurements.get("abs_far_gap_c") or 0.0) >= COOLING_FAR_GAP_MIN_C


def _disjoint_pool(rows, blocked):
    kept = []
    boxes = list(blocked)
    for row in rows:
        box = _row_box(row)
        if any(_hit(box, other) for other in boxes):
            continue
        kept.append(row)
        boxes.append(box)
    return kept


def _width_bin(row):
    width = float((row.get("measurements") or {}).get("width_m") or 0.0)
    if width < 5:
        return 0
    if width < 10:
        return 1
    if width < 20:
        return 2
    if width < 40:
        return 3
    return 4


def _duration_bin(row):
    duration = float((row.get("measurements") or {}).get("duration_s") or 0.0)
    if duration < 60:
        return 0
    if duration < 300:
        return 1
    if duration < 900:
        return 2
    if duration < 3600:
        return 3
    return 4


def _diverse_local(pool, count):
    """Round-robin width and duration bins. The pool is already priority-sorted."""
    if count <= 0 or not pool:
        return []
    groups = {}
    for row in pool:
        groups.setdefault((_width_bin(row), _duration_bin(row)), []).append(row)
    chosen = []
    while len(chosen) < count:
        grew = False
        for key in sorted(groups):
            if not groups[key]:
                continue
            chosen.append(groups[key].pop(0))
            grew = True
            if len(chosen) >= count:
                break
        if not grew:
            break
    return chosen


def _shown(measurements, key):
    """JSON spelling of a present measurement. Absent values are omitted, not printed."""
    if measurements.get(key) is None:
        return None
    return _n(measurements, key)


def _reason(bucket, measurements):
    record = measurements["record"]
    shape = measurements.get("shape_hypothesis")
    shape_name = shape if isinstance(shape, str) and shape else None
    if bucket == "wide":
        text = (
            f"Кандидат широкого канала на файле {record}: доля кольца "
            f"{_n(measurements, 'halo_inside_fraction')} при пороге "
            f"{_n(measurements, 'wide_halo_min')}, модуль разницы с far cable "
            f"{_n(measurements, 'abs_far_gap_c')} °C при пороге "
            f"{_n(measurements, 'wide_far_gap_min_c')} °C. "
            f"Медиана прямоугольника {_n(measurements, 'rectangle_median_c')} °C, "
            f"медиана far cable {_n(measurements, 'far_cable_median_c')} °C, "
            f"разница {_n(measurements, 'far_gap_c')} °C. "
            "Отсутствующая медиана far cable сюда не попадает. Класс эксперта пуст."
        )
    elif bucket == "hard negative":
        parts = [
            f"зон spatial_z в ящике {_n(measurements, 'in_box_count')}",
            f"доля тревоги {_n(measurements, 'alarm_fraction')} при пороге "
            f"{_n(measurements, 'hard_alarm_fraction_max')}",
        ]
        flank = _shown(measurements, "abs_flank_contrast_c")
        if flank is not None:
            parts.append(f"модуль контраста к флангу {flank} °C")
        far_gap = _shown(measurements, "abs_far_gap_c")
        if far_gap is not None:
            parts.append(f"модуль разницы с far cable {far_gap} °C")
        parts.append(f"порог контраста {_n(measurements, 'hard_contrast_min_c')} °C")
        text = f"Трудный негатив на файле {record}: " + ", ".join(parts) + ". Класс эксперта пуст."
    elif bucket == "cooling":
        parts = []
        if shape_name is not None:
            parts.append(f"гипотеза {shape_name}")
        contrast = _shown(measurements, "contrast_c")
        if contrast is not None:
            parts.append(f"контраст {contrast} °C")
        signed_gap = _shown(measurements, "far_gap_c")
        if signed_gap is not None:
            parts.append(
                f"разница с far cable {signed_gap} °C при пороге "
                f"{_n(measurements, 'cooling_far_gap_min_c')} °C"
            )
        if not parts:
            raise ValueError("cooling reason needs a hypothesis, a contrast, or a far-cable gap")
        text = f"Остывание на файле {record}: " + ", ".join(parts) + "."
        tail = []
        rectangle = _shown(measurements, "rectangle_median_c")
        if rectangle is not None:
            tail.append(f"медиана прямоугольника {rectangle} °C")
        far = _shown(measurements, "far_cable_median_c")
        if far is not None:
            tail.append(f"медиана far cable {far} °C")
        if tail:
            tail[0] = tail[0][:1].upper() + tail[0][1:]
            text += " " + ", ".join(tail) + "."
        text += " Класс эксперта пуст."
    elif bucket == "residual-versus-context":
        pair = _class_pair(measurements)
        if pair is None:
            raise ValueError("disagreement card needs two different research class names")
        text = (
            f"Расхождение residual и context на файле {record}: residual {pair[0]}, "
            f"context {pair[1]}. Обе оценки serve=research. Класс эксперта пуст."
        )
    elif bucket == "local":
        text = (
            f"Локальная зона на файле {record}: ширина {_n(measurements, 'width_m')} м, "
            f"длительность {_n(measurements, 'duration_s')} с"
        )
        if shape_name is not None:
            text += f", гипотеза {shape_name}"
        text += ". Класс эксперта пуст."
    else:
        raise ValueError(f"unknown bucket {bucket}")
    return _grounded(text, measurements)


def _finish(skeleton, bucket, priority):
    measurements = skeleton["measurements"]
    row = {
        "id": skeleton["id"],
        "record": skeleton["record"],
        "bucket": bucket,
        "priority": priority,
        "meters": list(skeleton["meters"]),
        "time": list(skeleton["time"]),
        "rows_half_open": list(skeleton["rows_half_open"]),
        "split_code": int(skeleton["split_code"]),
        "source_sha256": skeleton["source_sha256"],
        "expert_confirmed": False,
        "expert_class": None,
        "residual_class": measurements.get("residual_class"),
        "context_class": measurements.get("context_class"),
        "reason": _reason(bucket, measurements),
        "measurements": measurements,
    }
    return row


def _measure_record(record, catalog, sha, models):
    from fiber.io import Record
    from fiber.pipeline import analyze_field
    from fiber.real_data import RULE_CONFIG, crop_partition, temporal_partition
    from fiber.research_context import context_evidence

    loaded = Record.load(evidence_paths()["records"] / record)
    field = analyze_field(
        loaded.raw_c, loaded.x_m, loaded.time_s, valid=loaded.valid, time_iso=loaded.time_iso,
    )
    labels, _meta = temporal_partition(loaded.time_s, RULE_CONFIG["split_guard_s"])
    events = list(catalog["files"][str(record)]["events"])
    zones = []
    matched = 0
    for event in events:
        zone = _match_zone(field["zones"], event)
        if zone is None:
            continue
        matched += 1
        skeleton = _zone_skeleton(
            record, zone, field, loaded, labels, crop_partition, context_evidence, sha,
        )
        if skeleton is not None:
            zones.append((zone, skeleton))
    if events and matched == 0:
        raise RuntimeError(f"record {record} catalog events did not match analyze_field zones")
    kept = []
    seen = set()

    def add(skeleton):
        if skeleton is None or skeleton["id"] in seen or _overlaps_answered(skeleton):
            return
        seen.add(skeleton["id"])
        kept.append(skeleton)

    for _zone, skeleton in zones:
        add(skeleton)
    for skeleton in _union_skeletons(
        record, [zone for zone, _skeleton in zones], field, loaded, labels, crop_partition,
        context_evidence, sha,
    ):
        add(skeleton)
    for skeleton in _column_skeletons(record, field, loaded, labels, context_evidence, sha):
        add(skeleton)
    for skeleton in _sliding_skeletons(record, field, loaded, labels, context_evidence, sha):
        add(skeleton)
    for skeleton in _flank_skeletons(
        record, zones, field, loaded, labels, crop_partition, context_evidence, sha,
    ):
        add(skeleton)
    _score_networks(field, loaded, kept, models)
    print(
        f"shortlist record {record}: zones {len(zones)} kept {len(kept)}",
        file=sys.stderr,
        flush=True,
    )
    return kept


def _match_zone(zones, event):
    for zone in zones:
        if zone["t_start"] != event["t_start"]:
            continue
        if abs(float(zone["start_m"]) - float(event["start_m"])) < 1e-3:
            return zone
    return None


def _time_rows(time_s, t_start, t_end):
    start = int(np.searchsorted(time_s, float(t_start), side="left"))
    stop = int(np.searchsorted(time_s, float(t_end), side="right"))
    stop = max(start + 1, min(stop, len(time_s)))
    return start, stop


def _meter_columns(x_m, left, right):
    start = int(np.searchsorted(x_m, float(left) - 1e-3, side="left"))
    stop = int(np.searchsorted(x_m, float(right) + 1e-3, side="right"))
    start = max(0, min(start, len(x_m) - 1))
    stop = max(start + 1, min(stop, len(x_m)))
    return start, stop


def _split_code(labels, crop_partition, start, stop):
    if stop <= start:
        return -1
    return int(crop_partition(labels[start:stop]))


def _median_z(field, x_m, start, stop, x0, x1):
    z = np.asarray(field["z"])
    available = np.asarray(field["score_available"])
    c0, c1 = _meter_columns(x_m, x0, x1)
    block = z[start:stop, c0:c1]
    live = available[start:stop, c0:c1] & np.isfinite(block)
    if not np.any(live):
        return None
    return float(np.median(block[live]))


def _skeleton_from_context(
    record, origin, field, loaded, labels, crop_partition, evidence, sha, *,
    shape, polarity, contrast, n_cells, n_pieces, operational, extra,
):
    start, stop = (int(evidence["rows_half_open"][0]), int(evidence["rows_half_open"][1]))
    code = _split_code(labels, crop_partition, start, stop)
    if code not in (0, 1) or stop - start < 6:
        return None
    x0 = float(evidence["x_m"][0])
    x1 = float(evidence["x_m"][1])
    width = float(x1 - x0)
    width_q = _q(width)
    meters = [_q(x0), _q(x1)]
    if not meters[0] < meters[1]:
        return None
    t_start = str(loaded.time_iso[start])
    t_end = str(loaded.time_iso[stop - 1])
    if not t_start < t_end:
        return None
    rectangle = evidence["rectangle_median_c"]
    far = evidence["far_cable_median_c"]
    if rectangle is None:
        return None
    gap = None if far is None else float(rectangle) - float(far)
    contrast_q = None if contrast is None else _q(contrast)
    gap_q = None if gap is None else _q(gap)
    spatial = _median_z(field, loaded.x_m, start, stop, x0, x1)
    if spatial is None:
        return None
    inbox = int(len(evidence["in_box_zones"]))
    scored, alarm = _cell_counts(field, loaded.x_m, start, stop, x0, x1)
    alarm_fraction = 0.0 if scored == 0 else _q(alarm / scored, 6)
    flank = _flank_contrast(evidence)
    residual = _residual_mean(field, loaded.x_m, start, stop, x0, x1)
    duration = _q(float(loaded.time_s[stop - 1]) - float(loaded.time_s[start]))
    measurements = _common(record)
    measurements.update({
        "origin": origin,
        "shape_hypothesis": shape,
        "polarity": polarity,
        "contrast_c": contrast_q,
        "abs_contrast_c": None if contrast_q is None else _q(abs(contrast_q)),
        "width_m": width_q,
        "duration_s": duration,
        "spatial_median_z": _q(spatial),
        "rectangle_median_c": _q(rectangle),
        "far_cable_median_c": None if far is None else _q(far),
        "far_gap_c": gap_q,
        "abs_far_gap_c": None if gap_q is None else _q(abs(gap_q)),
        "halo_inside_fraction": _q(evidence["halo_inside_fraction"], 6),
        "in_box_count": inbox,
        "alarm_cells": alarm,
        "scored_cells": scored,
        "alarm_fraction": alarm_fraction,
        "flank_contrast_c": None if flank is None else flank,
        "abs_flank_contrast_c": None if flank is None else _q(abs(flank)),
        "residual_c": None if residual is None else _q(residual),
        "residual_class": None,
        "context_class": None,
        "n_cells": int(n_cells),
        "n_pieces": int(n_pieces),
        "operational_zone": bool(operational),
        "max_piece_gap_m": int(UNION_GAP_M),
    })
    measurements.update(extra)
    card_id = f"ask-r{record}-x{meters[0]:.3f}-x{meters[1]:.3f}-t{start}-{stop}"
    return {
        "id": card_id,
        "record": str(record),
        "meters": meters,
        "time": [t_start, t_end],
        "rows_half_open": [start, stop],
        "split_code": code,
        "source_sha256": sha,
        "measurements": measurements,
    }


def _zone_skeleton(record, zone, field, loaded, labels, crop_partition, context_evidence, sha):
    try:
        evidence = context_evidence(
            field, loaded.x_m, loaded.time_s, zone=zone,
        )
    except ValueError:
        return None
    return _skeleton_from_context(
        record, "catalog_zone", field, loaded, labels, crop_partition, evidence, sha,
        shape=zone.get("shape_hypothesis"),
        polarity=zone.get("polarity"),
        contrast=float(zone["contrast"]),
        n_cells=int(zone.get("n_cells") or 0),
        n_pieces=1,
        operational=int(zone.get("n_cells") or 0) > 0,
        extra={},
    )


def _union_skeletons(record, zones, field, loaded, labels, crop_partition, context_evidence, sha):
    ordered = sorted(zones, key=lambda zone: (float(zone["start_m"]), float(zone["t_start_s"]), float(zone["end_m"])))
    used = set()
    skeletons = []
    for index, zone in enumerate(ordered):
        if index in used:
            continue
        chain = [zone]
        used.add(index)
        grew = True
        while grew:
            grew = False
            added = None
            added_gap = None
            for other_index, other in enumerate(ordered):
                if other_index in used:
                    continue
                gap = _union_gap(chain, other)
                if gap is None:
                    continue
                if added is None or gap < added_gap or (gap == added_gap and abs(float(other["contrast"])) > abs(float(added["contrast"]))):
                    added = other
                    added_gap = gap
                    added_index = other_index
            if added is not None:
                chain.append(added)
                used.add(added_index)
                grew = True
        if len(chain) < 2:
            continue
        skeleton = _union_context(
            record, chain, field, loaded, labels, crop_partition, context_evidence, sha,
        )
        if skeleton is not None:
            skeletons.append(skeleton)
    return skeletons


def _union_gap(chain, other):
    if any(zone.get("polarity") != other.get("polarity") for zone in chain):
        return None
    t0 = max([float(zone["t_start_s"]) for zone in chain] + [float(other["t_start_s"])])
    t1 = min([float(zone["t_end_s"]) for zone in chain] + [float(other["t_end_s"])])
    if not t0 < t1:
        return None
    x0 = min(float(zone["start_m"]) for zone in chain)
    x1 = max(float(zone["end_m"]) for zone in chain)
    if float(other["end_m"]) < x0:
        gap = x0 - float(other["end_m"])
    elif float(other["start_m"]) > x1:
        gap = float(other["start_m"]) - x1
    else:
        gap = 0.0
    if gap > UNION_GAP_M:
        return None
    nx0 = min(x0, float(other["start_m"]))
    nx1 = max(x1, float(other["end_m"]))
    if nx1 - nx0 > UNION_MAX_WIDTH_M:
        return None
    return gap


def _union_context(record, chain, field, loaded, labels, crop_partition, context_evidence, sha):
    t0 = max(float(zone["t_start_s"]) for zone in chain)
    t1 = min(float(zone["t_end_s"]) for zone in chain)
    start, stop = _time_rows(loaded.time_s, t0, t1)
    if stop - start < 6:
        return None
    x0 = min(float(zone["start_m"]) for zone in chain)
    x1 = max(float(zone["end_m"]) for zone in chain)
    if not x1 - x0 > WIDTH_LIMIT_M:
        return None
    try:
        evidence = context_evidence(
            field, loaded.x_m, loaded.time_s,
            rows_half_open=(start, stop), x_span=(x0, x1),
        )
    except ValueError:
        return None
    contrasts = [float(zone["contrast"]) for zone in chain]
    contrast = float(np.median(contrasts))
    shapes = {zone.get("shape_hypothesis") for zone in chain}
    shape = next(iter(shapes)) if len(shapes) == 1 else None
    polarities = {zone.get("polarity") for zone in chain}
    polarity = next(iter(polarities)) if len(polarities) == 1 else None
    return _skeleton_from_context(
        record, "zone_union", field, loaded, labels, crop_partition, evidence, sha,
        shape=shape,
        polarity=polarity,
        contrast=contrast,
        n_cells=int(sum(int(zone.get("n_cells") or 0) for zone in chain)),
        n_pieces=len(chain),
        operational=True,
        extra={"max_piece_gap_m": int(UNION_GAP_M)},
    )


def _column_skeletons(record, field, loaded, labels, context_evidence, sha):
    x_m = np.asarray(loaded.x_m, dtype=np.float64)
    specs = []
    for start, stop in _kept_spans(labels):
        medians = _column_medians(field, start, stop)
        finite = np.isfinite(medians)
        if int(finite.sum()) < 10:
            continue
        reference = float(np.median(medians[finite]))
        delta = medians - reference
        for threshold in (1.0, 0.5, 0.3):
            specs.extend(_column_run_specs(x_m, delta, finite, start, stop, threshold))
    wide_specs = sorted((spec for spec in specs if spec["width"] >= 50.0), key=_spec_key)
    narrow_specs = sorted((spec for spec in specs if spec["width"] < 50.0), key=_spec_key)
    skeletons = []
    boxes = []
    wide_kept = 0
    narrow_kept = 0
    for spec in wide_specs + narrow_specs:
        wide_spec = spec["width"] >= 50.0
        if wide_spec and wide_kept >= 80:
            continue
        if not wide_spec and narrow_kept >= 24:
            continue
        box = (spec["start"], spec["stop"], spec["x0"], spec["x1"])
        if any(_span_hit(box, other) for other in boxes):
            continue
        skeleton = _column_context(
            record, field, loaded, labels, context_evidence, sha,
            spec["c0"], spec["c1"], spec["start"], spec["stop"], spec["threshold"], spec["delta"],
        )
        if skeleton is None:
            continue
        skeletons.append(skeleton)
        boxes.append(box)
        if wide_spec:
            wide_kept += 1
        else:
            narrow_kept += 1
        if len(skeletons) >= COLUMN_RUN_CAP:
            break
    return skeletons


def _column_context(record, field, loaded, labels, context_evidence, sha, c0, c1, start, stop, threshold, column_delta):
    from fiber.real_data import crop_partition

    x_m = np.asarray(loaded.x_m, dtype=np.float64)
    x0 = float(x_m[c0])
    x1 = float(x_m[c1 - 1])
    try:
        evidence = context_evidence(
            field, loaded.x_m, loaded.time_s,
            rows_half_open=(start, stop), x_span=(x0, x1),
        )
    except ValueError:
        return None
    return _skeleton_from_context(
        record, "column_median_run", field, loaded, labels, crop_partition, evidence, sha,
        shape=None,
        polarity=None,
        contrast=None,
        n_cells=0,
        n_pieces=0,
        operational=len(evidence["in_box_zones"]) > 0,
        extra={
            "column_delta_c": _q(column_delta),
            "run_threshold_c": _q(threshold),
        },
    )


def _column_medians(field, start, stop):
    clean = np.asarray(field["clean"], dtype=np.float64)
    available = np.asarray(field["score_available"], dtype=bool)
    block = clean[start:stop]
    live = available[start:stop] & np.isfinite(block)
    masked = np.where(live, block, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        medians = np.nanmedian(masked, axis=0)
    medians[live.sum(axis=0) < 6] = np.nan
    return medians


def _longest_span(labels, code):
    best = None
    start = None
    series = labels.tolist()
    for index, value in enumerate(series):
        if int(value) == int(code):
            if start is None:
                start = index
            continue
        if start is not None:
            if best is None or index - start > best[1] - best[0]:
                best = (start, index)
            start = None
    if start is not None and (best is None or len(series) - start > best[1] - best[0]):
        best = (start, len(series))
    return best


def _cap_run(x_m, delta, c0, c1):
    while c1 - c0 > 2 and float(x_m[c1 - 1] - x_m[c0]) > UNION_MAX_WIDTH_M:
        if abs(float(delta[c0])) < abs(float(delta[c1 - 1])):
            c0 += 1
        else:
            c1 -= 1
    return c0, c1


def _kept_spans(labels):
    spans = []
    for code in (0, 1):
        span = _longest_span(labels, code)
        if span is not None and span[1] - span[0] >= 6:
            spans.append(span)
    return spans


def _spec_key(spec):
    return (-float(spec["abs_delta"]), -float(spec["width"]), int(spec["c0"]), int(spec["start"]))


def _span_hit(left, right):
    return left[0] < right[1] and right[0] < left[1] and left[2] < right[3] and right[2] < left[3]


def _column_run_specs(x_m, delta, finite, start, stop, threshold):
    specs = []
    cursor = 0
    while cursor < len(x_m):
        if not finite[cursor] or abs(float(delta[cursor])) < threshold:
            cursor += 1
            continue
        sign = np.sign(delta[cursor])
        end = cursor + 1
        while (
            end < len(x_m) and finite[end]
            and abs(float(delta[end])) >= threshold and np.sign(delta[end]) == sign
        ):
            end += 1
        c0, c1 = _cap_run(x_m, delta, cursor, end)
        width = float(x_m[c1 - 1] - x_m[c0])
        if width >= 3.0:
            column_delta = float(np.median(delta[c0:c1]))
            specs.append({
                "c0": int(c0),
                "c1": int(c1),
                "start": int(start),
                "stop": int(stop),
                "threshold": float(threshold),
                "delta": column_delta,
                "abs_delta": abs(column_delta),
                "width": width,
                "x0": float(x_m[c0]),
                "x1": float(x_m[c1 - 1]),
            })
        cursor = end
    return specs


def _time_chunks(start, stop, parts=4, minimum=30):
    length = int(stop) - int(start)
    if length < minimum:
        return []
    parts = min(parts, max(1, length // minimum))
    chunks = []
    for index in range(parts):
        left = int(start) + (length * index) // parts
        right = int(start) + (length * (index + 1)) // parts
        if right - left >= minimum:
            chunks.append((left, right))
    return chunks


def _sliding_skeletons(record, field, loaded, labels, context_evidence, sha):
    x_m = np.asarray(loaded.x_m, dtype=np.float64)
    last = float(x_m[-1])
    skeletons = []
    spans = []
    for start, stop in _kept_spans(labels):
        spans.extend(_time_chunks(start, stop))
    for start, stop in spans:
        for width in (60.0, 80.0, 110.0, 140.0):
            step = width + 15.0
            left = float(x_m[0])
            while left + width <= last + 1e-3:
                skeleton = _span_skeleton(
                    record, "sliding_window", field, loaded, labels, context_evidence, sha,
                    start, stop, left, left + width,
                )
                if skeleton is not None:
                    skeletons.append(skeleton)
                left += step
    return skeletons


def _flank_skeletons(record, zones, field, loaded, labels, crop_partition, context_evidence, sha):
    del crop_partition
    x_m = np.asarray(loaded.x_m, dtype=np.float64)
    x_min = float(x_m[0])
    x_max = float(x_m[-1])
    skeletons = []
    for zone, skeleton in zones:
        start, stop = skeleton["rows_half_open"]
        x0 = float(zone["start_m"])
        x1 = float(zone["end_m"])
        for left, right in ((x0 - OUTER_M, x0), (x1, x1 + OUTER_M)):
            left = max(float(left), x_min)
            right = min(float(right), x_max)
            if right - left < 3.0:
                continue
            built = _span_skeleton(
                record, "flank_window", field, loaded, labels, context_evidence, sha,
                int(start), int(stop), left, right,
            )
            if built is not None:
                skeletons.append(built)
    return skeletons


def _span_skeleton(record, origin, field, loaded, labels, context_evidence, sha, start, stop, x0, x1):
    from fiber.real_data import crop_partition

    try:
        evidence = context_evidence(
            field, loaded.x_m, loaded.time_s,
            rows_half_open=(int(start), int(stop)), x_span=(float(x0), float(x1)),
        )
    except ValueError:
        return None
    return _skeleton_from_context(
        record, origin, field, loaded, labels, crop_partition, evidence, sha,
        shape=None,
        polarity=None,
        contrast=None,
        n_cells=0,
        n_pieces=0,
        operational=len(evidence["in_box_zones"]) > 0,
        extra={},
    )


def _cell_counts(field, x_m, start, stop, x0, x1):
    z = np.asarray(field["z"])
    available = np.asarray(field["score_available"])
    c0, c1 = _meter_columns(x_m, x0, x1)
    block = z[start:stop, c0:c1]
    live = available[start:stop, c0:c1] & np.isfinite(block)
    scored = int(np.count_nonzero(live))
    alarm = int(np.count_nonzero(live & (np.abs(block) >= THRESHOLD_Z)))
    return scored, alarm


def _residual_mean(field, x_m, start, stop, x0, x1):
    clean = np.asarray(field["clean"], dtype=np.float64)
    available = np.asarray(field["score_available"])
    c0, c1 = _meter_columns(x_m, x0, x1)
    patch = clean[start:stop, c0:c1]
    mask = available[start:stop, c0:c1] & np.isfinite(patch)
    if int(np.count_nonzero(mask)) < 6:
        return None
    center = float(np.median(patch[mask]))
    return float(np.mean(patch[mask] - center))


def _flank_contrast(evidence):
    best = None
    for zone in evidence.get("flank_zones") or []:
        value = zone.get("contrast")
        if value is None:
            continue
        value = float(value)
        if best is None or abs(value) > abs(best):
            best = value
    return None if best is None else _q(best)


def _load_models():
    from fiber.research_cnn import load_research_cnn, load_research_context_cnn

    weights = evidence_paths()["shortlist"].parent / "models" / "weights"
    residual, _residual_blob, device = load_research_cnn(weights / "research_residual_cnn.pt")
    context, _context_blob, _device = load_research_context_cnn(
        weights / "research_context_cnn.pt", device=device,
    )
    return residual, context, device


def _score_networks(field, loaded, skeletons, models):
    from fiber.classify.rules import LABELS
    from fiber.research_cnn import context_vector, meter_columns, residual_crop

    residual_model, context_model, device = models
    crops = []
    vectors = []
    indexes = []
    x_m = np.asarray(loaded.x_m, dtype=np.float64)
    clean = np.asarray(field["clean"])
    valid = np.asarray(field["score_available"])
    for index, skeleton in enumerate(skeletons):
        start, stop = skeleton["rows_half_open"]
        x0, x1 = skeleton["meters"]
        c0, c1 = meter_columns(x_m, float(x0), float(x1))
        crop = residual_crop(clean, valid, int(start), int(stop), c0, c1)
        if crop is None:
            continue
        measured = skeleton["measurements"]
        vectors.append(context_vector(
            measured["rectangle_median_c"],
            measured["far_cable_median_c"],
            measured["halo_inside_fraction"],
        ))
        crops.append(np.asarray(crop, dtype=np.float32))
        indexes.append(index)
    if not crops:
        return
    residual_names = _forward_classes(residual_model, crops, vectors, device, contextual=False)
    context_names = _forward_classes(context_model, crops, vectors, device, contextual=True)
    for index, residual_name, context_name in zip(indexes, residual_names, context_names, strict=True):
        if residual_name not in LABELS or context_name not in LABELS:
            raise RuntimeError("research CNN returned a name outside the six classes")
        measured = skeletons[index]["measurements"]
        measured["residual_class"] = residual_name
        measured["context_class"] = context_name


def _forward_classes(model, crops, vectors, device, *, contextual):
    import torch
    from fiber.classify.rules import LABELS

    names = []
    step = 64
    for start in range(0, len(crops), step):
        batch = torch.from_numpy(np.ascontiguousarray(np.stack(crops[start:start + step])[:, None])).to(device)
        with torch.inference_mode():
            if contextual:
                context = torch.from_numpy(np.ascontiguousarray(np.stack(vectors[start:start + step]))).to(device)
                logits = model(batch, context)
            else:
                logits = model(batch)
        choice = logits.argmax(dim=1).detach().cpu().numpy()
        names.extend(LABELS[int(item)] for item in choice)
    return names
