"""Weak labels for real crops. Training seeds, not expert truth.

A row is kept only when its time rows share one temporal_partition code in
{0 train, 1 development}. Holdout (2), guard (-1), and a crop that mixes
those codes are dropped. Zakhar cards are high-weight seeds.
presence=no is Background. A note does not choose the class.
An unclear presence or an empty shape stays unmapped and is not emitted.
Nothing here changes analyze_field or the |Z| >= 3.5 alarm.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from fiber.classify.rules import LABELS, profile_shift
from fiber.classify.scan import (
    SPACE_CROP,
    SPACE_STRIDE,
    TIME_CROP,
    TIME_STRIDE,
    stops,
    window_valid_counts,
)
from fiber.pipeline import analyze_field
from fiber.real_data import RULE_CONFIG, crop_partition, temporal_partition, weak_evidence

AUTOMATIC_WEIGHT = 1.0
EXPERT_WEIGHT = 10.0
AUTOMATIC_ORIGINS = ("spatial_z", "rule", "quiet")
KEPT_SPLIT_CODES = (0, 1)

# Cyan boxes already recorded in docs/DECISIONS.md and the two answer files.
# Class comes from presence and the written shape. Notes stay notes.
# review-00187 stays one Local_Heating class: several peaks, not a new name.
# next-0003 stays Local_Heating. The wide-channel note is not Broad_Shift.
_FIRST_ANSWERS = "docs/evidence/zakhar_shortlist_2026-10-10.jsonl"
_NEXT_ANSWERS = "docs/evidence/zakhar_next_shortlist_answers.jsonl"
_GAP_ANSWERS = "docs/evidence/zakhar_gap_shortlist_answers.jsonl"
_GAP_ROW_SUFFIX = re.compile(r"-t(\d+)-(\d+)$")


# A comma-separated shape keeps every written name in recorded_shape and
# emits the first name that is one of the six classes. The same rule is used
# for every such row. A prose shape such as "несколько Local_Heating" is not
# a list of class names and stays on the older heating rule.
MULTI_LABEL_RULE = "first_named_class"


def _named_shape_classes(shape: str) -> list[str] | None:
    parts = [part.strip() for part in str(shape).split(",")]
    if not parts or any(part not in LABELS for part in parts):
        return None
    return parts


def map_expert_class(presence: str, shape: str) -> str:
    """Map presence and the written shape onto the six class names.

    presence no is Background, including a note that only sounds like cooling.
    The note is not an argument here. A single class name is that class.
    A comma-separated list of class names uses the first name. The written
    shape is stored separately and is not replaced by that first name.
    """
    if presence == "no":
        return "Background"
    if presence != "yes":
        raise ValueError(f"presence must be yes or no, got {presence}")
    named = _named_shape_classes(shape)
    if named is not None:
        return named[0]
    if "Local_Heating" in shape and "Local_Cooling" not in shape:
        return "Local_Heating"
    raise ValueError(f"unmapped expert shape {shape!r}")


def class_from_presence_and_shape(presence: str, shape: str) -> str | None:
    """Return a class, or None when the answer stays unmapped.

    Unmapped means presence is unclear, or presence is yes and the shape is
    empty. The note is not read. Any other pair still goes through
    map_expert_class, which raises when it cannot name one of the six classes.
    """
    if presence == "unclear" or (presence == "yes" and str(shape).strip() == ""):
        return None
    return map_expert_class(presence, shape)


def _gap_answers_path() -> Path:
    return Path(__file__).resolve().parents[1] / _GAP_ANSWERS


def load_gap_answers() -> list[dict]:
    """Load the gap-shortlist answers. This is not the previous eleven-card file."""
    path = _gap_answers_path()
    if not path.is_file():
        raise ValueError(f"missing gap answers {path}")
    answers = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not answers:
        raise ValueError(f"gap answers file is empty: {path}")
    return answers


def resolve_answer_rows(answer: dict, time_iso) -> tuple[int, int]:
    """Half-open rows from the id suffix, checked against the answer time."""
    match = _GAP_ROW_SUFFIX.search(str(answer["id"]))
    if match is None:
        raise ValueError(f"{answer['id']} has no t-row suffix")
    start, stop = int(match.group(1)), int(match.group(2))
    iso = np.asarray(time_iso)
    if not (0 <= start < stop <= len(iso)):
        raise ValueError(f"{answer['id']} rows {(start, stop)} outside the record")
    left, right = str(iso[start]), str(iso[stop - 1])
    want_left, want_right = str(answer["time"][0]), str(answer["time"][1])
    if left != want_left or right != want_right:
        raise ValueError(
            f"{answer['id']} rows {start}:{stop} are {left} .. {right}, "
            f"answer time is {want_left} .. {want_right}"
        )
    return start, stop


def _polarity_and_structure(class_name: str, shape: str) -> tuple[str | None, str | None]:
    if class_name == "Background":
        return None, None
    if class_name == "Local_Cooling":
        return "cold", "one_span"
    if class_name == "Local_Heating":
        structure = "several_peaks" if "несколько" in shape else "one_span"
        return "hot", structure
    if class_name in ("Periodic_Thermal", "Broad_Shift", "Sensor_Fault"):
        return None, None
    raise ValueError(f"no polarity for {class_name}")


def _expert_seed(
    seed_id: str,
    record: str,
    rows_half_open: tuple[int, int],
    x_m: tuple[float, float],
    presence: str,
    recorded_shape: str,
    notes: str,
    source: str,
    polarity: str | None,
    structure: str | None,
) -> dict:
    class_name = map_expert_class(presence, recorded_shape)
    if class_name not in LABELS:
        raise ValueError(f"class_name must be one of {LABELS}")
    if presence == "no" and (class_name != "Background" or polarity is not None):
        raise ValueError(f"{seed_id} presence no must stay Background without a cooling polarity")
    return {
        "id": seed_id,
        "record": str(record),
        "rows_half_open": (int(rows_half_open[0]), int(rows_half_open[1])),
        "x_m": (float(x_m[0]), float(x_m[1])),
        "presence": presence,
        "polarity": polarity,
        "structure": structure,
        "class_name": class_name,
        "recorded_shape": recorded_shape,
        "notes": notes,
        "source": source,
    }


EXPERT_SEEDS = (
    _expert_seed(
        "review-00034", "1", (310, 335), (517.77, 523.25),
        "yes", "Local_Heating",
        "Весь участок — одно нагревание. Середина около 520 м чуть менее тёплая и относится к тому же участку, не отдельное остывание. По медиане за окно: теплее 28 °C с 515.2 до 524.2 м; промежуток 519.1–521.6 м около 24.7 °C при кабеле около 20 °C и боках 31–33 °C. За строки ядра температура стоит на месте. Голубой прямоугольник 517.77–523.25 м обрезает левый край.",
        _FIRST_ANSWERS, "hot", "one_span",
    ),
    _expert_seed(
        "review-00098", "1", (0, 298), (837.36, 851.98),
        "yes", "Local_Heating",
        "Один горячий участок на всё окно, небольшие затухания внутри относятся к нему. По медиане: теплее 28 °C с 835.4 до 854.7 м, с провалами около 841 м (~27.6 °C) и 850 м (~26.7 °C); пики около 42–44 °C, кабель сбоку около 23 °C. За ~9 часов середина прямоугольника держится около 36 °C; падение до 29.8 °C только в последние ~7 минут. Другие жёлтые полосы левее 830 м к этому участку не относятся. Причина провалов по графику не видна.",
        _FIRST_ANSWERS, "hot", "one_span",
    ),
    _expert_seed(
        "review-00187", "1", (35, 43), (230.36, 270.36),
        "yes", "несколько Local_Heating",
        "Не один участок, а отдельные короткие нагревы. Пики медианы: 237.1 м (24.7 °C, выше 22 °C на 236.0–238.1), 244.9 м (25.2 °C, 243.4–246.7), 257.6 м (26.3 °C, 256.1–258.8), широкий 264.6–266.6 м (26.3 °C, выше 22 °C на 262.7–268.1). Ещё один пик 274.1 м (26.2 °C, 272.4–275.5) уже за правым краем голубого прямоугольника 230.4–270.4 м. Между ними провалы около 18–19 °C. Слабее остальных полоса около 254.4 м (23.2 °C). За ~28 минут полосы стоят на месте. Нижний график усредняет весь прямоугольник, поэтому контраст там только 1.6 °C.",
        _FIRST_ANSWERS, "hot", "several_peaks",
    ),
    _expert_seed(
        "review-00223", "1", (234, 294), (363.15, 403.15),
        "yes", "Local_Cooling",
        "Один холодный участок, без отдельных событий. Холоднее 19 °C: 338.3–357.2 м (медиана 17.4 °C, левее прямоугольника) и 378.2–397.2 м (18.2 °C). Между ними 363.6–375.8 м держится около 21 °C. Прямоугольник 363.1–403.1 м в целом только на 0.7 °C холоднее флангов. Жёлтые полосы около 418 м и дальше правее прямоугольника. За 30 минут холодный кусок 378–397 м стоит около 18 °C.",
        _FIRST_ANSWERS, "cold", "one_span",
    ),
    _expert_seed(
        "next-0001", "3", (705, 788), (372.028, 381.418),
        "yes", "Local_Heating, Local_Heating", "",
        _NEXT_ANSWERS, "hot", "several_peaks",
    ),
    _expert_seed(
        "next-0002", "5", (0, 2711), (234.0, 273.0),
        "yes", "Local_Cooling", "",
        _NEXT_ANSWERS, "cold", "one_span",
    ),
    _expert_seed(
        "next-0003", "3", (618, 788), (111.405, 177.386),
        "yes", "Local_Heating",
        "на самом деле широкий канал heating с кучей local heating, но как будто надо считать это все одним широким heating. не знаю. можно пометить как много local heating и по какому-то правилу объединить. надо думать",
        _NEXT_ANSWERS, "hot", "one_span",
    ),
    _expert_seed(
        "next-0004", "2", (225, 409), (46.948, 93.897),
        "yes", "Local_Heating",
        "небольшое поднятие температуры после охлаждения.",
        _NEXT_ANSWERS, "hot", "one_span",
    ),
    _expert_seed(
        "next-0005", "4", (50, 201), (80.0, 84.0),
        "no", "",
        "здесь типо кулинг относительно левого участка, но на общем фоне то что показано, скорее слева local_heating а тут просто кабель упал в свой обычный рабочий темп",
        _NEXT_ANSWERS, None, None,
    ),
    _expert_seed(
        "next-0006", "1", (449, 508), (944.162, 984.061),
        "no", "",
        "тут либо относительно левого и правого участка широкий cooling либо слева и справа был local_heating а между ними кабель был в норме или типо того. я не знаю нормы температур для кабеля к сожалению",
        _NEXT_ANSWERS, None, None,
    ),
    _expert_seed(
        "next-0007", "2", (65, 80), (501.966, 541.555),
        "no", "",
        "тоже ниче непонятно. справа в упоре был local cooling как будто",
        _NEXT_ANSWERS, None, None,
    ),
    _expert_seed(
        "next-0008", "6", (822, 1119), (586.727, 598.4),
        "yes", "Local_Heating",
        "началось правее, чуть урезать бы этот участок и тогда будет явный local_heating",
        _NEXT_ANSWERS, "hot", "one_span",
    ),
    _expert_seed(
        "next-0009", "4", (326, 708), (93.895, 120.541),
        "yes", "Local_Heating",
        "heating с понижениями температур либо не на всю длину(время) кабеля",
        _NEXT_ANSWERS, "hot", "one_span",
    ),
    _expert_seed(
        "next-0010", "2", (0, 409), (291.587, 317.979),
        "no", "",
        "непонятно просадки какие-то типо а потом справа кстати подъем. но относительно левой и правой части как будто тут просто cooling был хз",
        _NEXT_ANSWERS, None, None,
    ),
    _expert_seed(
        "next-0011", "1", (479, 615), (434.721, 455.228),
        "yes", "несколько участков, Local_Heating",
        "несколько участков local heating с просадками . в целом тут по графику даже левее видно два столба с подъемами температуры так же",
        _NEXT_ANSWERS, "hot", "several_peaks",
    ),
)
EXPERT_SEED_BY_ID = {seed["id"]: seed for seed in EXPERT_SEEDS}
_SPLIT_NAMES = {0: "train", 1: "development", 2: "internal_holdout", -1: "excluded_guard_or_mixed"}


def _prepare_record_times(times_by_record: dict) -> dict:
    prepared = {}
    for name, pair in times_by_record.items():
        time_s, time_iso = pair
        time_s = np.asarray(time_s, dtype=float)
        split, _info = temporal_partition(time_s, RULE_CONFIG["split_guard_s"])
        prepared[str(name)] = (time_s, time_iso, split)
    return prepared


def gap_answer_items(times_by_record: dict) -> list[dict]:
    """Resolve every gap answer. A seed is present only when a class maps.

    Rows come from the id suffix and must match that answer's time on the
    record. The note is not passed to the class rule. Unmapped answers stay
    in the list with seed None so the statistics can name them.
    """
    prepared = _prepare_record_times(times_by_record)
    items = []
    seen = set()
    for answer in load_gap_answers():
        seed_id = str(answer["id"])
        if seed_id in seen or seed_id in EXPERT_SEED_BY_ID:
            raise ValueError(f"duplicate expert id {seed_id}")
        seen.add(seed_id)
        if answer.get("reviewer") != "Захар" or answer.get("expert_confirmed") is not True:
            raise ValueError(f"{seed_id} is not a confirmed Zakhar answer")
        if answer.get("reviewer_type") != "human":
            raise ValueError(f"{seed_id} reviewer_type must be human")
        record = str(answer["record"])
        if not seed_id.startswith(f"gap-r{record}-"):
            raise ValueError(f"{seed_id} does not name record {record}")
        if record not in prepared:
            raise ValueError(f"missing record for {seed_id}")
        _time_s, time_iso, split = prepared[record]
        start, stop = resolve_answer_rows(answer, time_iso)
        code = int(crop_partition(np.asarray(split)[start:stop]))
        class_name = class_from_presence_and_shape(str(answer["presence"]), str(answer["shape"]))
        seed = None
        if class_name is not None:
            polarity, structure = _polarity_and_structure(class_name, str(answer["shape"]))
            seed = _expert_seed(
                seed_id,
                record,
                (start, stop),
                (float(answer["meters"][0]), float(answer["meters"][1])),
                str(answer["presence"]),
                str(answer["shape"]),
                str(answer["notes"]),
                _GAP_ANSWERS,
                polarity,
                structure,
            )
        items.append({
            "id": seed_id,
            "answer": answer,
            "rows": (start, stop),
            "split_code": code,
            "class_name": class_name,
            "seed": seed,
        })
    return items


def gap_seed_index(times_by_record: dict) -> dict:
    """Mapped gap seeds keyed by id. Unmapped answers are left out."""
    return {
        item["id"]: item["seed"]
        for item in gap_answer_items(times_by_record)
        if item["seed"] is not None
    }


_ASK_ANSWERS = "docs/evidence/zakhar_ask_shortlist_answers.jsonl"
_ASK_DOWNLOAD = Path(r"C:\Users\bobr\Downloads\zakhar_next_shortlist_answers (2).jsonl")


def current_expert_ids() -> frozenset:
    """Ids already used by the previous seeds or the gap answers."""
    return frozenset(EXPERT_SEED_BY_ID) | frozenset(str(answer["id"]) for answer in load_gap_answers())


def load_ask_answers(path=None) -> list[dict]:
    """Load the ask-shortlist answers. This is not the eleven-card or gap file."""
    source = Path(path) if path is not None else Path(__file__).resolve().parents[1] / _ASK_ANSWERS
    if not source.is_file():
        raise ValueError(f"missing ask answers {source}")
    answers = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not answers:
        raise ValueError(f"ask answers file is empty: {source}")
    return answers


def _ask_card(answer: dict, *, status: str, class_name, split_code, weight, reason: str) -> dict:
    return {
        "id": str(answer["id"]),
        "record": str(answer.get("record")),
        "presence": str(answer.get("presence")),
        "recorded_shape": str(answer.get("shape") or ""),
        "notes": str(answer.get("notes") or ""),
        "class_name": class_name,
        "status": status,
        "split_code": split_code,
        "weight": weight,
        "reason": reason,
        "expert_confirmed": status == "kept",
    }


def ask_seed_report(answers: list[dict], times_by_record: dict, existing_ids: frozenset) -> tuple[list[dict], dict]:
    """Map one answers file to weight-10 seeds. Split 2 and −1 are not emitted.

    Class comes only from presence and shape. An id already in existing_ids
    is excluded. presence=unclear is counted and is not a weight-10 row.
    presence=yes with an empty shape is counted separately and is not emitted.
    A row suffix that does not match the answer time is excluded.
    Two calls on the same answers and times return the same ids.
    """
    prepared = _prepare_record_times(times_by_record)
    cards = []
    candidates = []
    seen = set()
    for answer in answers:
        seed_id = str(answer["id"])
        presence = str(answer.get("presence"))
        shape = str(answer.get("shape") or "")
        if seed_id in seen or seed_id in existing_ids:
            cards.append(_ask_card(
                answer, status="excluded", class_name=None, split_code=None,
                weight=None, reason="already_present",
            ))
            seen.add(seed_id)
            continue
        seen.add(seed_id)
        if presence == "unclear":
            cards.append(_ask_card(
                answer, status="unclear", class_name=None, split_code=None,
                weight=None, reason="unclear",
            ))
            continue
        if presence == "yes" and shape.strip() == "":
            cards.append(_ask_card(
                answer, status="empty_shape", class_name=None, split_code=None,
                weight=None, reason="empty_shape",
            ))
            continue
        try:
            class_name = map_expert_class(presence, shape)
        except ValueError:
            cards.append(_ask_card(
                answer, status="excluded", class_name=None, split_code=None,
                weight=None, reason="unmapped_shape",
            ))
            continue
        record = str(answer.get("record"))
        if record not in prepared or not seed_id.startswith(f"ask-r{record}-"):
            cards.append(_ask_card(
                answer, status="excluded", class_name=class_name, split_code=None,
                weight=None, reason="record",
            ))
            continue
        _time_s, time_iso, split = prepared[record]
        try:
            start, stop = resolve_answer_rows(answer, time_iso)
        except ValueError:
            cards.append(_ask_card(
                answer, status="excluded", class_name=class_name, split_code=None,
                weight=None, reason="time",
            ))
            continue
        code = int(crop_partition(np.asarray(split)[start:stop]))
        polarity, structure = _polarity_and_structure(class_name, shape)
        seed = _expert_seed(
            seed_id,
            record,
            (start, stop),
            (float(answer["meters"][0]), float(answer["meters"][1])),
            presence,
            shape,
            str(answer.get("notes") or ""),
            _ASK_ANSWERS,
            polarity,
            structure,
        )
        candidates.append((answer, seed, code, len(cards)))
        cards.append(_ask_card(
            answer, status="excluded", class_name=class_name, split_code=code,
            weight=None, reason="split",
        ))
    index = {seed["id"]: seed for _answer, seed, _code, _slot in candidates}
    crops = []
    seeds = [seed for _answer, seed, _code, _slot in candidates]
    for name in sorted(prepared):
        time_s, time_iso, split = prepared[name]
        crops.extend(expert_crops(name, time_s, time_iso, split, seeds=seeds))
    rows = crops_to_weak_rows(crops, seed_index=index)
    order = {seed["id"]: position for position, (_answer, seed, _code, _slot) in enumerate(candidates)}
    rows.sort(key=lambda row: order[row["id"]])
    emitted = {row["id"]: row for row in rows}
    for answer, seed, code, slot in candidates:
        if seed["id"] not in emitted:
            if code in KEPT_SPLIT_CODES:
                raise ValueError(f"{seed['id']} split {code} was dropped")
            continue
        row = emitted[seed["id"]]
        if int(row["split_code"]) != code or code not in KEPT_SPLIT_CODES:
            raise ValueError(f"{seed['id']} emitted with split {row['split_code']}")
        cards[slot] = _ask_card(
            answer, status="kept", class_name=seed["class_name"], split_code=code,
            weight=EXPERT_WEIGHT, reason="kept",
        )
    if any(int(row["split_code"]) not in KEPT_SPLIT_CODES for row in rows):
        raise ValueError("ask seed row is outside train/development")
    if any(row["weight"] != EXPERT_WEIGHT or row["expert_confirmed"] is not True for row in rows):
        raise ValueError("ask seed row lost expert weight")
    n_kept = sum(card["status"] == "kept" for card in cards)
    n_excluded = sum(card["status"] == "excluded" for card in cards)
    n_unclear = sum(card["status"] == "unclear" for card in cards)
    n_empty = sum(card["status"] == "empty_shape" for card in cards)
    if n_kept + n_excluded + n_unclear + n_empty != len(answers):
        raise ValueError("ask answer counts do not add up")
    kept_ids = [card["id"] for card in cards if card["status"] == "kept"]
    if [row["id"] for row in rows] != kept_ids:
        raise ValueError("ask seeds emitted by crops_to_weak_rows do not match kept cards")
    report = {
        "schema": "expert_seeds_ask_v1",
        "source": _ASK_ANSWERS,
        "multi_label_rule": MULTI_LABEL_RULE,
        "n_answers": len(answers),
        "n_kept": n_kept,
        "n_excluded": n_excluded,
        "n_unclear": n_unclear,
        "n_empty_shape": n_empty,
        "excluded_reasons": dict(Counter(card["reason"] for card in cards if card["status"] == "excluded")),
        "cards": cards,
        "generated_ids": [row["id"] for row in rows],
        "holdout_used_for_generation": False,
        "guard_used_for_generation": False,
        "automatic_weight": AUTOMATIC_WEIGHT,
        "expert_weight": EXPERT_WEIGHT,
        "automatic_jsonl_regenerated": False,
        "operational_clean": "median_x3",
        "operational_detector": "spatial_z_halo",
    }
    return rows, report


def _rule_class(evidence: dict) -> str | None:
    """Map one decisive raw-contrast vote onto the six class names."""
    if evidence.get("weak_presence_label") is None or evidence.get("rule_conflict"):
        return None
    shapes = set(evidence.get("shape_hypotheses") or [])
    if "repeated_local_change" in shapes:
        return "Periodic_Thermal"
    if "local_heating" in shapes:
        return "Local_Heating"
    if "local_cooling" in shapes:
        return "Local_Cooling"
    return None


def _bounds_match(crop: dict, seed: dict) -> bool:
    rows = [int(v) for v in crop["rows_half_open"]]
    want = [int(seed["rows_half_open"][0]), int(seed["rows_half_open"][1])]
    if rows != want:
        return False
    x0, x1 = float(crop["x_m"][0]), float(crop["x_m"][1])
    return abs(x0 - float(seed["x_m"][0])) < 0.02 and abs(x1 - float(seed["x_m"][1])) < 0.02


def crops_to_weak_rows(crops: list[dict], seed_index: dict | None = None) -> list[dict]:
    """Turn crops into weak-label rows.

    Drops a crop whose `split_codes` are holdout, guard, or mixed.
    Expert seeds keep the recorded class, presence, polarity, and notes and get
    EXPERT_WEIGHT. Every other row is expert_confirmed false at AUTOMATIC_WEIGHT.
    A note is copied onto the row and is not a class name.
    seed_index defaults to the previous fifteen seeds. The gap report passes
    those seeds plus the mapped gap cards. Unmapped cards are not in it.
    """
    index = EXPERT_SEED_BY_ID if seed_index is None else seed_index
    rows = []
    for crop in crops:
        code = int(crop_partition(np.asarray(crop["split_codes"])))
        if code not in KEPT_SPLIT_CODES:
            continue
        origin = crop["origin"]
        if origin not in AUTOMATIC_ORIGINS and origin != "expert_seed":
            raise ValueError(f"unknown origin {origin}")
        crop_id = str(crop["id"])
        seed = index.get(crop_id)
        if origin == "expert_seed":
            if seed is None or str(crop["record"]) != seed["record"] or not _bounds_match(crop, seed):
                continue
            class_name = seed["class_name"]
            if class_name not in LABELS or class_name != map_expert_class(seed["presence"], seed["recorded_shape"]):
                raise ValueError(f"expert class drifted for {crop_id}")
            weight = EXPERT_WEIGHT
            expert_confirmed = True
            presence = seed["presence"]
            polarity = seed["polarity"]
            structure = seed["structure"]
            recorded_shape = seed["recorded_shape"]
            notes = seed["notes"]
            record = seed["record"]
            rows_half_open = [int(seed["rows_half_open"][0]), int(seed["rows_half_open"][1])]
            x_m = [float(seed["x_m"][0]), float(seed["x_m"][1])]
        else:
            if seed is not None:
                raise ValueError(f"expert id reserved for seeds: {crop_id}")
            class_name = crop["class_name"]
            if class_name not in LABELS:
                raise ValueError(f"class_name must be one of {LABELS}")
            weight = AUTOMATIC_WEIGHT
            expert_confirmed = False
            presence = None
            polarity = crop.get("polarity")
            structure = None
            recorded_shape = None
            notes = None
            record = str(crop["record"])
            rows_half_open = [int(crop["rows_half_open"][0]), int(crop["rows_half_open"][1])]
            x_m = [float(crop["x_m"][0]), float(crop["x_m"][1])]
        provenance = dict(crop.get("provenance") or {})
        provenance["source_kind"] = origin
        provenance["split_code_name"] = _SPLIT_NAMES[code]
        provenance["holdout_used_for_generation"] = False
        provenance["guard_used_for_generation"] = False
        rows.append({
            "id": crop_id,
            "record": record,
            "origin": origin,
            "class_name": class_name,
            "rows_half_open": rows_half_open,
            "x_m": x_m,
            "time_s": [float(crop["time_s"][0]), float(crop["time_s"][1])],
            "time_iso": [str(crop["time_iso"][0]), str(crop["time_iso"][1])],
            "split_code": code,
            "weight": weight,
            "expert_confirmed": expert_confirmed,
            "presence": presence,
            "polarity": polarity,
            "structure": structure,
            "recorded_shape": recorded_shape,
            "notes": notes,
            "provenance": provenance,
        })
    return rows


def summarize_weak_rows(rows: list[dict]) -> dict:
    """Counts and leak flags. Both count groups add up to the row count."""
    by_file = Counter(str(row["record"]) for row in rows)
    by_class = Counter(row["class_name"] for row in rows)
    by_origin = Counter(row["origin"] for row in rows)
    codes = {int(row["split_code"]) for row in rows}
    return {
        "n_rows": len(rows),
        "counts_by_file": dict(by_file),
        "counts_by_class": dict(by_class),
        "counts_by_origin": dict(by_origin),
        "split_codes": sorted(codes),
        "holdout_used_for_generation": 2 in codes,
        "guard_used_for_generation": -1 in codes,
    }


def _zone_span(zone: dict, x_m: np.ndarray, time_s: np.ndarray) -> tuple[int, int, int, int]:
    t0 = int(np.searchsorted(time_s, zone["t_start_s"], side="left"))
    t1 = int(np.searchsorted(time_s, zone["t_end_s"], side="right"))
    t0 = max(0, min(t0, len(time_s) - 1))
    t1 = max(t0 + 1, min(t1, len(time_s)))
    c0 = int(np.searchsorted(x_m, float(zone["start_m"]) - 1e-3, side="left"))
    c1 = int(np.searchsorted(x_m, float(zone["end_m"]) + 1e-3, side="right"))
    c0 = max(0, min(c0, len(x_m) - 1))
    c1 = max(c0 + 1, min(c1, len(x_m)))
    return t0, t1, c0, c1


def _quiet_pairs(t_starts: np.ndarray, x_starts: np.ndarray, boxes: list[tuple[int, int, int, int]]):
    n_time, n_space = len(t_starts), len(x_starts)
    if n_time == 0 or n_space == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    if not boxes:
        grid_t, grid_x = np.meshgrid(np.arange(n_time), np.arange(n_space), indexing="ij")
        return grid_t.ravel(), grid_x.ravel()
    t0 = np.array([box[0] for box in boxes], dtype=np.int32)
    t1 = np.array([box[1] for box in boxes], dtype=np.int32)
    x0 = np.array([box[2] for box in boxes], dtype=np.int32)
    x1 = np.array([box[3] for box in boxes], dtype=np.int32)
    space_hit = (x_starts[:, None] < x1) & (x_starts[:, None] + SPACE_CROP > x0)
    parts_t, parts_x = [], []
    for start in range(0, n_time, 64):
        sl = t_starts[start:start + 64]
        time_hit = (sl[:, None] < t1) & (sl[:, None] + TIME_CROP > t0)
        covered = time_hit.astype(np.int32) @ space_hit.astype(np.int32).T
        free = np.argwhere(covered == 0)
        if len(free):
            parts_t.append(free[:, 0] + start)
            parts_x.append(free[:, 1])
    if not parts_t:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(parts_t), np.concatenate(parts_x)


def crops_from_record(record) -> tuple[list[dict], dict]:
    """Build spatial_z, rule, and quiet crops, plus expert seeds for this record.

    Zones come from analyze_field (median_x3, spatial_z_halo, |Z| >= 3.5).
    Rule crops are weak_evidence votes on those zone cores and profile_shift
    Broad_Shift windows. Quiet crops are 28x40 scan windows that miss every
    alarm zone. temporal_innovation is not a source.
    """
    split, info = temporal_partition(record.time_s, RULE_CONFIG["split_guard_s"])
    detected = analyze_field(
        record.raw_c, record.x_m, record.time_s, valid=record.valid, time_iso=record.time_iso,
    )
    zones = detected["zones"]
    clean = detected["clean"]
    crops: list[dict] = []
    boxes: list[tuple[int, int, int, int]] = []
    for index, zone in enumerate(zones):
        t0, t1, c0, c1 = _zone_span(zone, record.x_m, record.time_s)
        boxes.append((t0, t1, c0, c1))
        label = zone.get("shape_hypothesis")
        if label not in LABELS:
            continue
        codes = split[t0:t1]
        if int(crop_partition(codes)) not in KEPT_SPLIT_CODES:
            continue
        base = {
            "record": str(record.name),
            "rows_half_open": [t0, t1],
            "x_m": [float(zone["start_m"]), float(zone["end_m"])],
            "time_s": [float(record.time_s[t0]), float(record.time_s[t1 - 1])],
            "time_iso": [record.time_iso[t0], record.time_iso[t1 - 1]],
            "split_codes": codes,
            "polarity": zone["polarity"],
        }
        crops.append({
            **base,
            "id": f"spatial-{record.name}-{index:05d}",
            "origin": "spatial_z",
            "class_name": label,
            "provenance": {
                "detector": "spatial_z_halo",
                "clean": "median_x3",
                "threshold_z": 3.5,
                "min_width_m": 3.0,
                "min_profiles": 6,
                "min_cells": 24,
                "max_gap_s": 300.0,
                "shape_rule": "classify_zones",
            },
        })
        xa = int(np.searchsorted(record.x_m, float(zone["start_m"]) - 40.0, side="left"))
        xb = int(np.searchsorted(record.x_m, float(zone["end_m"]) + 40.0, side="right"))
        xb = max(xa + 1, min(xb, len(record.x_m)))
        evidence, _ = weak_evidence(
            record.raw_c[t0:t1, xa:xb],
            record.valid[t0:t1, xa:xb],
            record.x_m[xa:xb],
            record.time_s[t0:t1],
            [float(zone["start_m"]), float(zone["end_m"])],
        )
        rule_name = _rule_class(evidence)
        if rule_name is None:
            continue
        crops.append({
            **base,
            "id": f"rule-vote-{record.name}-{index:05d}",
            "origin": "rule",
            "class_name": rule_name,
            "provenance": {
                "rule": "weak_evidence",
                "shape_hypotheses": list(evidence["shape_hypotheses"]),
            },
        })
    print(f"Record {record.name}: zones {len(zones)} crops after rules {len(crops)}", flush=True)
    measured = np.flatnonzero(np.asarray(record.valid).any(axis=0))
    t_starts = stops(record.n_profiles, TIME_CROP, TIME_STRIDE)
    if measured.size >= 2 and t_starts.size:
        cable = [float(record.x_m[measured[0]]), float(record.x_m[measured[-1]])]
        for t0 in t_starts:
            t0 = int(t0)
            t1 = t0 + TIME_CROP
            if int(crop_partition(split[t0:t1])) not in KEPT_SPLIT_CODES:
                continue
            shift = profile_shift(clean[t0:t1], record.valid[t0:t1], record.x_m)
            if shift["label"] != "Broad_Shift":
                continue
            crops.append({
                "id": f"rule-broad-{record.name}-{t0:05d}",
                "origin": "rule",
                "record": str(record.name),
                "class_name": "Broad_Shift",
                "rows_half_open": [t0, t1],
                "x_m": cable,
                "time_s": [float(record.time_s[t0]), float(record.time_s[t1 - 1])],
                "time_iso": [record.time_iso[t0], record.time_iso[t1 - 1]],
                "split_codes": split[t0:t1],
                "polarity": None,
                "provenance": {"rule": "profile_shift", "delta_c": shift["delta_c"], "side": shift["side"]},
            })
    print(f"Record {record.name}: crops after profile_shift {len(crops)}", flush=True)
    x_starts = stops(len(record.x_m), SPACE_CROP, SPACE_STRIDE)
    pair_t, pair_x = _quiet_pairs(t_starts, x_starts, boxes)
    if len(pair_t):
        origins = np.column_stack([t_starts[pair_t], x_starts[pair_x]]).astype(np.int64, copy=False)
        counts = window_valid_counts(record.valid, origins)
        keep = counts * 2 >= TIME_CROP * SPACE_CROP
        for t0, x0 in origins[keep]:
            t0, x0 = int(t0), int(x0)
            t1 = t0 + TIME_CROP
            if int(crop_partition(split[t0:t1])) not in KEPT_SPLIT_CODES:
                continue
            x1 = x0 + SPACE_CROP
            crops.append({
                "id": f"quiet-{record.name}-{t0:05d}-{x0:05d}",
                "origin": "quiet",
                "record": str(record.name),
                "class_name": "Background",
                "rows_half_open": [t0, t1],
                "x_m": [float(record.x_m[x0]), float(record.x_m[x1 - 1])],
                "time_s": [float(record.time_s[t0]), float(record.time_s[t1 - 1])],
                "time_iso": [record.time_iso[t0], record.time_iso[t1 - 1]],
                "split_codes": split[t0:t1],
                "polarity": None,
                "provenance": {
                    "window": f"{TIME_CROP}x{SPACE_CROP}",
                    "time_stride": TIME_STRIDE,
                    "space_stride": SPACE_STRIDE,
                    "overlaps_alarm_zone": False,
                },
            })
    crops.extend(expert_crops(str(record.name), record.time_s, record.time_iso, split))
    return crops, info


def expert_crops(record_name: str, time_s, time_iso, split, seeds=None) -> list[dict]:
    """Expert crops for one record, using the same rows the weak-label writer keeps.

    crops_from_record appends the previous fifteen seeds. crops_to_weak_rows
    still drops a crop whose split is holdout, guard, or mixed. Gap cards are
    passed in `seeds` by the expert-seed report and are not part of that fifteen.
    """
    if seeds is None:
        seeds = EXPERT_SEEDS
    time_s = np.asarray(time_s)
    n = len(time_s)
    crops = []
    for seed in seeds:
        if str(record_name) != seed["record"]:
            continue
        a, b = seed["rows_half_open"]
        if not (0 <= a < b <= n):
            continue
        crops.append({
            "id": seed["id"],
            "origin": "expert_seed",
            "record": seed["record"],
            "class_name": seed["class_name"],
            "rows_half_open": [a, b],
            "x_m": [float(seed["x_m"][0]), float(seed["x_m"][1])],
            "time_s": [float(time_s[a]), float(time_s[b - 1])],
            "time_iso": [str(time_iso[a]), str(time_iso[b - 1])],
            "split_codes": np.asarray(split)[a:b],
            "polarity": seed["polarity"],
            "provenance": {
                "shortlist": seed["source"],
                "recorded_shape": seed["recorded_shape"],
                "structure": seed["structure"],
                "notes": seed["notes"],
                "presence": seed["presence"],
            },
        })
    return crops


def _card_for_seed(seed: dict, code: int, row: dict | None, prepared: dict) -> dict:
    kept = row is not None
    if kept:
        if code not in KEPT_SPLIT_CODES or int(row["split_code"]) != code:
            raise ValueError(f"{seed['id']} kept with split {code}")
        if row["class_name"] != seed["class_name"] or row["notes"] != seed["notes"]:
            raise ValueError(f"{seed['id']} row drifted from the seed")
        if row["weight"] < 10 or row["weight"] <= AUTOMATIC_WEIGHT or row["expert_confirmed"] is not True:
            raise ValueError(f"{seed['id']} lost expert weight")
    elif code in KEPT_SPLIT_CODES:
        raise ValueError(f"{seed['id']} dropped with kept split {code}")
    a, b = seed["rows_half_open"]
    _time_s, time_iso, _split = prepared[seed["record"]]
    return {
        "id": seed["id"],
        "record": seed["record"],
        "presence": seed["presence"],
        "class_name": seed["class_name"],
        "weight": EXPERT_WEIGHT,
        "split_code": code,
        "status": "kept" if kept else "excluded",
        "expert_confirmed": True,
        "notes": seed["notes"],
        "recorded_shape": seed["recorded_shape"],
        "rows_half_open": [a, b],
        "x_m": [float(seed["x_m"][0]), float(seed["x_m"][1])],
        "time_iso": [str(time_iso[a]), str(time_iso[b - 1])],
    }


def expert_card_report(times_by_record: dict) -> tuple[list[dict], dict]:
    """Emit kept expert rows and one card for every previous and gap answer.

    times_by_record maps a record name to (time_s, time_iso).
    A holdout, guard, or mixed window is listed as excluded and is not emitted.
    An unmapped gap answer is listed and is not passed to crops_to_weak_rows.
    """
    prepared = _prepare_record_times(times_by_record)
    missing = sorted({seed["record"] for seed in EXPERT_SEEDS if seed["record"] not in prepared})
    if missing:
        raise ValueError(f"missing records for expert seeds: {missing}")
    gap_items = gap_answer_items(times_by_record)
    mapped = [item["seed"] for item in gap_items if item["seed"] is not None]
    index = {**EXPERT_SEED_BY_ID, **{seed["id"]: seed for seed in mapped}}
    crops = []
    for name in sorted(prepared):
        time_s, time_iso, split = prepared[name]
        crops.extend(expert_crops(name, time_s, time_iso, split))
    for name in sorted(prepared):
        time_s, time_iso, split = prepared[name]
        crops.extend(expert_crops(name, time_s, time_iso, split, seeds=mapped))
    window_code = {}
    for seed in list(EXPERT_SEEDS) + mapped:
        time_s, _time_iso, split = prepared[seed["record"]]
        a, b = seed["rows_half_open"]
        if not (0 <= a < b <= len(time_s)):
            raise ValueError(f"{seed['id']} rows outside the record")
        window_code[seed["id"]] = int(crop_partition(split[a:b]))
    rows = crops_to_weak_rows(crops, seed_index=index)
    by_id = {row["id"]: row for row in rows}
    cards = [_card_for_seed(seed, window_code[seed["id"]], by_id.get(seed["id"]), prepared) for seed in EXPERT_SEEDS]
    for item in gap_items:
        if item["seed"] is None:
            if item["id"] in by_id:
                raise ValueError(f"{item['id']} unmapped answer was emitted")
            answer = item["answer"]
            record = str(answer["record"])
            start, stop = item["rows"]
            _time_s, time_iso, split = prepared[record]
            code = int(crop_partition(np.asarray(split)[start:stop]))
            if code != item["split_code"]:
                raise ValueError(f"{item['id']} split changed between measurements")
            cards.append({
                "id": item["id"],
                "record": record,
                "presence": str(answer["presence"]),
                "class_name": None,
                "weight": EXPERT_WEIGHT,
                "split_code": code,
                "status": "unmapped",
                "expert_confirmed": True,
                "notes": str(answer["notes"]),
                "recorded_shape": str(answer["shape"]),
                "rows_half_open": [start, stop],
                "x_m": [float(answer["meters"][0]), float(answer["meters"][1])],
                "time_iso": [str(time_iso[start]), str(time_iso[stop - 1])],
            })
            continue
        if item["split_code"] != window_code[item["id"]]:
            raise ValueError(f"{item['id']} split changed between measurements")
        cards.append(_card_for_seed(item["seed"], window_code[item["id"]], by_id.get(item["id"]), prepared))
    summary = summarize_weak_rows(rows)
    if summary["holdout_used_for_generation"] or summary["guard_used_for_generation"]:
        raise ValueError("holdout or guard expert row was emitted")
    if any(int(row["split_code"]) not in KEPT_SPLIT_CODES for row in rows):
        raise ValueError("generated expert row is outside train/development")
    n_kept = sum(card["status"] == "kept" for card in cards)
    n_excluded = sum(card["status"] == "excluded" for card in cards)
    n_unmapped = sum(card["status"] == "unmapped" for card in cards)
    if n_kept + n_excluded + n_unmapped != len(cards) or len(cards) != len(EXPERT_SEEDS) + len(gap_items):
        raise ValueError("expert card counts do not add up")
    if n_kept != len(rows):
        raise ValueError("kept cards and emitted rows diverged")
    report = {
        "schema": "expert_seeds_v1",
        "n_cards": len(cards),
        "n_kept": n_kept,
        "n_excluded": n_excluded,
        "n_unmapped": n_unmapped,
        "cards": cards,
        "generated_ids": [row["id"] for row in rows],
        "split_codes_emitted": summary["split_codes"],
        "holdout_used_for_generation": False,
        "guard_used_for_generation": False,
        "automatic_weight": AUTOMATIC_WEIGHT,
        "expert_weight": EXPERT_WEIGHT,
        "automatic_jsonl_regenerated": False,
    }
    return rows, report
