"""The research shortlist selector keeps quotas and stays unlabeled.

Tests call select_shortlist on synthetic rows. They do not rescan the six CSVs.
"""
import json
import re
from collections import Counter
from pathlib import Path

from fiber.classify.rules import LABELS
from fiber.shortlist import (
    ANSWERED_BOXES,
    ANSWERED_IDS,
    BUCKETS,
    COOLING_FAR_GAP_MIN_C,
    CONFIRMED_IDS,
    MAX_TOTAL,
    MIN_TOTAL,
    QUOTAS,
    WIDE_FAR_GAP_MIN_C,
    WIDE_HALO_MIN,
    _hit,
    _is_cooling,
    _primary_bucket,
    _reason,
    evidence_paths,
    select_shortlist,
)
from fiber.weak_labels import EXPERT_SEEDS, load_gap_answers

_CITED_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_GAP_ONLY = {
    "gap-r6-x1116.354-x1127.774-t385-418",
    "gap-r6-x30.199-x44.157-t0-705",
}


def _row(card_id, bucket, index, *, priority=1, split_code=0, expert_confirmed=False, record=None, meters=None, rows=None, width=None, duration=None):
    if record is None:
        record = str((index % 6) + 1)
    if meters is None:
        meters = [float(index) * 20.0, float(index) * 20.0 + 8.0]
    if rows is None:
        rows = [20000, 20020]
    if width is None:
        width = 3.0 + (index % 7) * 4.0
    if duration is None:
        duration = 20.0 + (index % 5) * 80.0
    return {
        "id": card_id,
        "record": record,
        "bucket": bucket,
        "priority": priority,
        "meters": meters,
        "rows_half_open": rows,
        "time": ["2026-01-01T00:00:00+00:00", "2026-01-01T01:00:00+00:00"],
        "expert_confirmed": expert_confirmed,
        "expert_class": "should-not-survive",
        "split_code": split_code,
        "reason": f"candidate {card_id}",
        "measurements": {"width_m": width, "duration_s": duration, "record": record},
    }


def _measured(**overrides):
    measurements = {
        "record": "1",
        "origin": "catalog_zone",
        "shape_hypothesis": "Local_Heating",
        "contrast_c": 2.0,
        "residual_c": 0.4,
        "width_m": 8.0,
        "duration_s": 120.0,
        "halo_inside_fraction": 0.1,
        "rectangle_median_c": 20.0,
        "far_cable_median_c": 19.8,
        "far_gap_c": 0.2,
        "abs_far_gap_c": 0.2,
        "in_box_count": 1,
        "alarm_fraction": 0.4,
        "abs_flank_contrast_c": 0.2,
        "residual_class": "Local_Heating",
        "context_class": "Local_Heating",
        "wide_halo_min": WIDE_HALO_MIN,
        "wide_far_gap_min_c": WIDE_FAR_GAP_MIN_C,
        "hard_alarm_fraction_max": 0.05,
        "hard_contrast_min_c": 1.5,
        "cooling_far_gap_min_c": COOLING_FAR_GAP_MIN_C,
    }
    measurements.update(overrides)
    return measurements


def test_primary_bucket_uses_research_cuts_and_does_not_copy_an_absent_far_side():
    wide = _measured(
        halo_inside_fraction=0.6,
        far_cable_median_c=18.0,
        far_gap_c=2.0,
        abs_far_gap_c=1.5,
        origin="sliding_window",
    )
    assert _primary_bucket(wide) == "wide"
    absent = _measured(
        halo_inside_fraction=1.0,
        far_cable_median_c=None,
        far_gap_c=None,
        abs_far_gap_c=None,
        origin="sliding_window",
    )
    assert _primary_bucket(absent) != "wide"
    short_gap = _measured(
        halo_inside_fraction=0.6,
        abs_far_gap_c=1.4,
        origin="sliding_window",
    )
    assert _primary_bucket(short_gap) != "wide"
    disagree = _measured(
        residual_class="Local_Cooling",
        context_class="Local_Heating",
        halo_inside_fraction=0.9,
        abs_far_gap_c=3.0,
        far_cable_median_c=17.0,
    )
    assert _primary_bucket(disagree) == "residual-versus-context"
    cooling = _measured(shape_hypothesis="Local_Cooling", contrast_c=-1.2, residual_c=-0.4)
    assert _is_cooling(cooling) is True
    assert _primary_bucket(cooling) == "cooling"
    cold_window = _measured(
        origin="flank_window",
        shape_hypothesis=None,
        contrast_c=None,
        residual_c=0.4,
        far_gap_c=-0.2,
        abs_far_gap_c=0.2,
        in_box_count=0,
        alarm_fraction=0.0,
        abs_flank_contrast_c=2.0,
    )
    assert _is_cooling(cold_window) is True
    assert _primary_bucket(cold_window) == "cooling"
    negative_contrast = _measured(
        contrast_c=-0.4,
        residual_c=1.2,
        far_gap_c=0.8,
        abs_far_gap_c=0.8,
        halo_inside_fraction=0.1,
    )
    assert _is_cooling(negative_contrast) is True
    assert _primary_bucket(negative_contrast) == "cooling"
    hot_heating = _measured(
        shape_hypothesis="Local_Heating",
        contrast_c=23.329,
        residual_c=-1.769,
        far_gap_c=21.693,
        abs_far_gap_c=21.693,
        rectangle_median_c=43.281,
        far_cable_median_c=21.588,
        halo_inside_fraction=0.07607,
    )
    assert _is_cooling(hot_heating) is False
    assert _primary_bucket(hot_heating) == "local"
    hotter_than_far = _measured(
        origin="column_median_run",
        shape_hypothesis=None,
        contrast_c=None,
        residual_c=-3.339,
        far_gap_c=13.668,
        abs_far_gap_c=13.668,
        rectangle_median_c=35.442,
        far_cable_median_c=21.774,
        in_box_count=1,
        alarm_fraction=0.464632,
        abs_flank_contrast_c=23.329,
        halo_inside_fraction=0.1,
    )
    assert _is_cooling(hotter_than_far) is False
    assert _primary_bucket(hotter_than_far) is None
    same_temperature = _measured(
        origin="sliding_window",
        shape_hypothesis=None,
        contrast_c=None,
        residual_c=-1.046,
        far_gap_c=-0.004,
        abs_far_gap_c=0.004,
        rectangle_median_c=23.418,
        far_cable_median_c=23.421,
        halo_inside_fraction=1.0,
        in_box_count=1,
        alarm_fraction=0.088667,
        abs_flank_contrast_c=0.0,
    )
    assert _is_cooling(same_temperature) is False
    assert _primary_bucket(same_temperature) is None
    residual_skew = _measured(
        origin="sliding_window",
        shape_hypothesis=None,
        contrast_c=None,
        residual_c=-2.0,
        far_cable_median_c=None,
        far_gap_c=None,
        abs_far_gap_c=None,
        in_box_count=1,
        alarm_fraction=0.4,
        abs_flank_contrast_c=0.1,
    )
    assert _is_cooling(residual_skew) is False
    assert _primary_bucket(residual_skew) is None
    hard = _measured(
        origin="flank_window",
        shape_hypothesis=None,
        contrast_c=None,
        residual_c=0.0,
        in_box_count=0,
        alarm_fraction=0.2,
        abs_flank_contrast_c=1.5,
        far_cable_median_c=None,
        abs_far_gap_c=None,
    )
    assert _primary_bucket(hard) == "hard negative"
    low_alarm = _measured(
        origin="flank_window",
        shape_hypothesis=None,
        contrast_c=0.2,
        residual_c=0.1,
        in_box_count=2,
        alarm_fraction=0.04,
        abs_flank_contrast_c=0.2,
        far_cable_median_c=18.0,
        abs_far_gap_c=1.5,
    )
    assert _primary_bucket(low_alarm) == "hard negative"
    quiet = _measured(
        origin="flank_window",
        shape_hypothesis=None,
        contrast_c=None,
        residual_c=0.0,
        in_box_count=0,
        alarm_fraction=0.0,
        abs_flank_contrast_c=1.0,
        far_cable_median_c=19.0,
        abs_far_gap_c=1.0,
    )
    assert _primary_bucket(quiet) is None
    local = _measured()
    assert _primary_bucket(local) == "local"
    assert _reason("wide", wide)
    hard_reason = _reason("hard negative", hard)
    assert "null" not in hard_reason
    assert "1.5" in hard_reason
    cooling_reason = _reason("cooling", cooling)
    assert "Local_Cooling" in cooling_reason
    assert "контраст" in cooling_reason
    assert "null" not in cooling_reason
    assert "несколько гипотез" not in cooling_reason
    blank_cooling = _measured(
        origin="column_median_run",
        shape_hypothesis=None,
        contrast_c=None,
        residual_c=0.143,
        far_gap_c=-2.53,
        abs_far_gap_c=2.53,
        rectangle_median_c=16.454,
        far_cable_median_c=18.984,
        halo_inside_fraction=0.0,
        in_box_count=0,
        alarm_fraction=0.0,
    )
    blank_reason = _reason("cooling", blank_cooling)
    assert "несколько гипотез" not in blank_reason
    assert "null" not in blank_reason
    assert "контраст" not in blank_reason
    assert "-2.53" in blank_reason
    assert "Медиана прямоугольника 16.454" in blank_reason
    assert "медиана far cable 18.984" in blank_reason
    assert "far cable" in blank_reason
    assert "Local_Cooling" in _reason("residual-versus-context", disagree)
    assert "Local_Heating" in _reason("residual-versus-context", disagree)
    local_reason = _reason("local", local)
    assert "Local_Heating" in local_reason
    assert "null" not in local_reason
    unnamed = _reason("local", _measured(shape_hypothesis=None))
    assert "несколько гипотез" not in unnamed
    assert "null" not in unnamed
    assert "гипотеза" not in unnamed


def test_selector_fills_quotas_and_drops_answered_holdout_and_overlaps():
    gap_ids = {answer["id"] for answer in load_gap_answers()}
    assert len(CONFIRMED_IDS) == len(EXPERT_SEEDS) == 15
    assert ANSWERED_IDS == CONFIRMED_IDS | gap_ids | _GAP_ONLY
    assert CONFIRMED_IDS.isdisjoint(gap_ids)
    assert CONFIRMED_IDS.isdisjoint(_GAP_ONLY)
    candidates = []
    plan = (
        ("wide", 60, 0),
        ("hard negative", 50, 1000),
        ("cooling", 40, 2000),
        ("residual-versus-context", 40, 3000),
        ("local", 40, 4000),
    )
    for bucket, count, start in plan:
        for offset in range(count):
            index = start + offset
            candidates.append(_row(f"ask-{bucket}-{offset}", bucket, index, priority=100 - offset))
    for pile in range(5):
        candidates.append(_row(
            f"pile-{pile}", "wide", 9000 + pile, priority=5000 - pile, record="1",
            meters=[1.0, 9.0], rows=[20000, 20020],
        ))
    candidates.append(_row("holdout-row", "wide", 9100, priority=9000, split_code=2, record="3"))
    candidates.append(_row("guard-row", "hard negative", 9101, priority=9000, split_code=-1, record="4"))
    candidates.append(_row("review-00034", "hard negative", 9102, priority=9000, record="1"))
    candidates.append(_row("next-0006", "hard negative", 9103, priority=9000, record="1"))
    candidates.append(_row("gap-r1-extra", "cooling", 9104, priority=9000, record="2"))
    candidates.append(_row(
        "already-confirmed", "local", 9105, priority=9000, expert_confirmed=True,
    ))
    candidates.append(_row(
        "overlap-next-0005", "cooling", 9106, priority=9000, record="4",
        meters=[80.0, 84.0], rows=[50, 201],
    ))
    candidates.append(_row(
        "overlap-gap-r1", "wide", 9107, priority=9000, record="1",
        meters=[300.0, 320.0], rows=[10, 100],
    ))
    before = json.dumps(candidates, sort_keys=True)
    picked = select_shortlist(candidates)
    again = select_shortlist(candidates)
    assert json.dumps(candidates, sort_keys=True) == before
    assert [card["id"] for card in picked] == [card["id"] for card in again]
    assert picked.stats == again.stats
    assert MIN_TOTAL <= len(picked) <= MAX_TOTAL
    counts = Counter(card["bucket"] for card in picked)
    assert counts["wide"] == QUOTAS["wide"]["target"]
    assert counts["hard negative"] == QUOTAS["hard negative"]["target"]
    assert counts["cooling"] == QUOTAS["cooling"]["target"]
    assert counts["residual-versus-context"] == QUOTAS["residual-versus-context"]["target"]
    assert counts["local"] >= 1
    assert sum(counts.values()) == len(picked)
    ids = [card["id"] for card in picked]
    assert len(ids) == len(set(ids))
    assert "pile-0" in ids
    assert "pile-1" not in ids
    blocked = {
        "holdout-row", "guard-row", "review-00034", "next-0006", "gap-r1-extra",
        "already-confirmed", "overlap-next-0005", "overlap-gap-r1",
    }
    assert blocked.isdisjoint(ids)
    assert ANSWERED_IDS.isdisjoint(ids)
    local = [card for card in picked if card["bucket"] == "local"]
    assert len({card["measurements"]["width_m"] for card in local}) > 1
    assert len({card["measurements"]["duration_s"] for card in local}) > 1
    for card in picked:
        assert card["expert_confirmed"] is False
        assert card["expert_class"] is None
        assert card["reason"].strip()
        assert not card["id"].startswith(("review-", "next-", "gap-"))
        assert "priority" not in card
        assert "split_code" not in card
    for bucket, quota in QUOTAS.items():
        stat = picked.stats[bucket]
        assert stat["taken"] == counts[bucket]
        assert quota["floor"] <= stat["taken"] <= quota["ceiling"]
        assert stat["shortfall"] is False


def test_short_pool_is_taken_whole_and_not_padded():
    candidates = []
    for offset in range(5):
        candidates.append(_row(f"wide-{offset}", "wide", offset, priority=50 - offset))
    for offset in range(40):
        candidates.append(_row(f"hard-{offset}", "hard negative", 1000 + offset, priority=40))
    for offset in range(30):
        candidates.append(_row(f"cool-{offset}", "cooling", 2000 + offset, priority=30))
    for offset in range(30):
        candidates.append(_row(f"diff-{offset}", "residual-versus-context", 3000 + offset, priority=20))
    for offset in range(25):
        candidates.append(_row(f"local-{offset}", "local", 4000 + offset, priority=10))
    picked = select_shortlist(candidates)
    again = select_shortlist(candidates)
    assert [card["id"] for card in picked] == [card["id"] for card in again]
    wide = picked.stats["wide"]
    assert wide["pool"] == 5
    assert wide["taken"] == wide["pool"]
    assert wide["shortfall"] is True
    assert sum(card["bucket"] == "wide" for card in picked) == 5
    assert len(picked) < MIN_TOTAL


def _cited_numbers_are_measured(card):
    blob = json.dumps(card["measurements"], ensure_ascii=False)
    missing = [
        token for token in _CITED_NUMBER.findall(card["reason"])
        if re.search(rf"(?<![\d.]){re.escape(token)}(?![\d.])", blob) is None
    ]
    assert not missing, (card["id"], missing, card["reason"])


def _box(card):
    meters = card["meters"]
    rows = card["rows_half_open"]
    return (str(card["record"]), float(meters[0]), float(meters[1]), int(rows[0]), int(rows[1]))


def test_written_shortlist_matches_stored_measurements():
    path = evidence_paths()["shortlist"]
    document = json.loads(path.read_text(encoding="utf-8"))
    cards = document["cards"]
    assert document["schema"] == "research_shortlist_v1"
    assert document["serve"] == "research"
    assert document["operational"] is False
    assert document["expert_confirmed"] is False
    assert document["clean"] == "median_x3"
    assert document["detector"] == "spatial_z_halo"
    assert document["threshold_z"] == 3.5
    assert document["wide_halo_min"] == WIDE_HALO_MIN == 0.6
    assert document["wide_far_gap_min_c"] == WIDE_FAR_GAP_MIN_C == 1.5
    assert document["excluded_answered_id_count"] == len(ANSWERED_IDS)
    assert set(document["excluded_answered_ids"]) == set(ANSWERED_IDS)
    assert document["n_cards"] == len(cards)
    false_cooling = {
        "ask-r1-x434.619-x455.330-t0-244",
        "ask-r1-x425.279-x429.442-t0-344",
        "ask-r1-x812.893-x816.142-t0-344",
        "ask-r1-x889.746-x894.010-t399-463",
        "ask-r6-x0.000-x110.000-t915-969",
    }
    cooling_ids = {card["id"] for card in cards if card["bucket"] == "cooling"}
    assert false_cooling.isdisjoint(cooling_ids)
    counts = Counter(card["bucket"] for card in cards)
    assert set(document["buckets"]) == set(BUCKETS)
    assert sum(document["buckets"][bucket] for bucket in BUCKETS) == len(cards)
    pools_cover = True
    for bucket, quota in QUOTAS.items():
        stat = document["bucket_stats"][bucket]
        assert stat["taken"] == counts[bucket] == document["buckets"][bucket]
        assert stat["taken"] <= stat["pool"]
        if stat["pool"] < quota["floor"]:
            assert stat["taken"] == stat["pool"]
            assert stat["shortfall"] is True
            pools_cover = False
        else:
            assert quota["floor"] <= stat["taken"] <= quota["ceiling"]
            assert stat["shortfall"] is False
    if pools_cover and document["bucket_stats"]["local"]["pool"] >= 1:
        assert MIN_TOTAL <= len(cards) <= MAX_TOTAL
    boxes = []
    for card in cards:
        assert card["id"].startswith("ask-")
        assert not card["id"].startswith(("review-", "next-", "gap-"))
        assert card["id"] not in ANSWERED_IDS
        assert card["bucket"] in BUCKETS
        assert card["record"] in {"1", "2", "3", "4", "5", "6"}
        assert card["expert_confirmed"] is False
        assert card["expert_class"] is None
        assert isinstance(card["reason"], str) and card["reason"].strip()
        assert "null" not in card["reason"]
        assert "несколько гипотез" not in card["reason"]
        assert card["residual_class"] == card["measurements"]["residual_class"]
        assert card["context_class"] == card["measurements"]["context_class"]
        _cited_numbers_are_measured(card)
        box = _box(card)
        assert not any(_hit(box, answered) for answered in ANSWERED_BOXES)
        assert not any(_hit(box, other) for other in boxes)
        boxes.append(box)
        measured = card["measurements"]
        if card["bucket"] == "wide":
            assert measured["halo_inside_fraction"] >= 0.6
            assert measured["far_cable_median_c"] is not None
            assert measured["abs_far_gap_c"] >= 1.5
            assert measured["far_cable_median_c"] != measured["rectangle_median_c"] or measured["abs_far_gap_c"] == 0
        elif card["bucket"] == "hard negative":
            low = measured["in_box_count"] == 0 or measured["alarm_fraction"] < 0.05
            flank = measured["abs_flank_contrast_c"] is not None and measured["abs_flank_contrast_c"] >= 1.5
            far = measured["far_cable_median_c"] is not None and measured["abs_far_gap_c"] >= 1.5
            assert low and (flank or far)
        elif card["bucket"] == "cooling":
            assert _is_cooling(measured)
            positive_heating = (
                measured.get("shape_hypothesis") == "Local_Heating"
                and measured.get("contrast_c") is not None
                and float(measured["contrast_c"]) > 0
            )
            assert not positive_heating
            hotter_than_far = (
                measured.get("far_cable_median_c") is not None
                and measured.get("far_gap_c") is not None
                and float(measured["far_gap_c"]) > 0
            )
            if hotter_than_far:
                assert measured.get("shape_hypothesis") == "Local_Cooling" or (
                    measured.get("contrast_c") is not None and float(measured["contrast_c"]) < 0
                )
            if measured.get("shape_hypothesis"):
                assert measured["shape_hypothesis"] in card["reason"]
            if measured.get("contrast_c") is not None:
                assert "контраст" in card["reason"]
            else:
                assert "контраст" not in card["reason"]
            if measured.get("far_gap_c") is not None:
                assert "far cable" in card["reason"]
        elif card["bucket"] == "residual-versus-context":
            assert card["residual_class"] in LABELS
            assert card["context_class"] in LABELS
            assert card["residual_class"] != card["context_class"]
    local = [card for card in cards if card["bucket"] == "local"]
    if len(local) >= 2:
        assert len({card["measurements"]["width_m"] for card in local}) > 1
        assert len({card["measurements"]["duration_s"] for card in local}) > 1
    published = json.loads((Path(path).parent / "docs" / "evidence" / "wide_channel.json").read_text(encoding="utf-8"))
    assert published["wide_channel_halo_inside_min"] == 0.7
    assert published["wide_channel_far_gap_min_c"] == 2.0
