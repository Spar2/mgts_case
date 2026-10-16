"""Weak-label rows: three origins, expert seed weight, no holdout or guard."""
import inspect

import numpy as np

from fiber.pipeline import analyze_field
from fiber.weak_labels import (
    AUTOMATIC_WEIGHT,
    EXPERT_SEED_BY_ID,
    EXPERT_SEEDS,
    EXPERT_WEIGHT,
    crops_to_weak_rows,
    expert_card_report,
    map_expert_class,
    summarize_weak_rows,
)


def _expert_crop(seed_id, codes):
    """Crop whose bounds match the seed. The class on the crop is a decoy."""
    seed = EXPERT_SEED_BY_ID[seed_id]
    return _crop(
        "expert_seed",
        codes,
        crop_id=seed_id,
        record=seed["record"],
        class_name="Local_Cooling",
        polarity="cold",
        rows_half_open=list(seed["rows_half_open"]),
        x_m=list(seed["x_m"]),
    )


def _crop(origin, codes, *, crop_id, class_name="Local_Heating", record="1", **extra):
    body = {
        "id": crop_id,
        "origin": origin,
        "record": record,
        "class_name": class_name,
        "rows_half_open": [0, 4],
        "x_m": [10.0, 16.0],
        "time_s": [0.0, 180.0],
        "time_iso": ["t0", "t1"],
        "split_codes": list(codes),
        "polarity": "hot",
        "provenance": {"detector": "spatial_z_halo"} if origin == "spatial_z" else {"rule": "weak_evidence"},
    }
    body.update(extra)
    return body


def test_crop_to_row_keeps_origins_and_drops_holdout_guard_and_mixed():
    crops = [
        _crop("spatial_z", [0, 0, 0], crop_id="hot-zone"),
        _crop("rule", [1, 1], crop_id="rule-vote", class_name="Local_Cooling", polarity="cold"),
        _crop("quiet", [0], crop_id="quiet-window", class_name="Background", polarity=None),
        _crop("spatial_z", [2, 2, 2], crop_id="holdout-only"),
        _crop("quiet", [-1, -1], crop_id="guard-only", class_name="Background"),
        _crop("rule", [0, 1], crop_id="mixed-train-dev"),
        _crop("rule", [0, 2], crop_id="mixed-holdout"),
        _crop("spatial_z", [1, -1, 2], crop_id="mixed-guard"),
        _crop(
            "expert_seed",
            [0, 0],
            crop_id="review-00034",
            class_name="Local_Cooling",
            polarity="cold",
            rows_half_open=[310, 335],
            x_m=[517.77, 523.25],
        ),
        _crop(
            "expert_seed",
            [2, 2],
            crop_id="review-00098",
            rows_half_open=[0, 298],
            x_m=[837.36, 851.98],
        ),
        # Split codes here are the builder input. The record split is checked
        # on the committed report, which drops a mixed or holdout window.
        _expert_crop("next-0006", [0, 0]),
        _expert_crop("next-0004", [1, 1]),
    ]
    rows = crops_to_weak_rows(crops)
    by_id = {row["id"]: row for row in rows}
    assert set(by_id) == {
        "hot-zone", "rule-vote", "quiet-window", "review-00034", "next-0006", "next-0004",
    }
    assert "holdout-only" not in by_id and "guard-only" not in by_id
    assert {row["origin"] for row in rows} == {"spatial_z", "rule", "quiet", "expert_seed"}
    for row in rows:
        assert row["split_code"] in (0, 1)
        assert row["record"]
        assert row["rows_half_open"][1] > row["rows_half_open"][0]
        assert "x_m" in row and "time_s" in row
        assert row["provenance"]["source_kind"] == row["origin"]
        assert row["provenance"]["holdout_used_for_generation"] is False
        assert row["provenance"]["guard_used_for_generation"] is False
    hot = by_id["hot-zone"]
    rule = by_id["rule-vote"]
    quiet = by_id["quiet-window"]
    assert hot["class_name"] == "Local_Heating" and hot["origin"] == "spatial_z"
    assert rule["class_name"] == "Local_Cooling" and rule["provenance"]["rule"] == "weak_evidence"
    assert quiet["class_name"] == "Background" and quiet["origin"] == "quiet"
    assert hot["weight"] == rule["weight"] == quiet["weight"] == AUTOMATIC_WEIGHT
    assert hot["expert_confirmed"] is rule["expert_confirmed"] is quiet["expert_confirmed"] is False
    seed = by_id["review-00034"]
    assert seed["expert_confirmed"] is True
    assert seed["weight"] == EXPERT_WEIGHT
    assert seed["weight"] > hot["weight"]
    assert seed["weight"] >= 10
    assert seed["class_name"] == "Local_Heating"
    assert seed["presence"] == "yes"
    assert seed["polarity"] == "hot"
    assert seed["structure"] == "one_span"
    assert seed["recorded_shape"] == "Local_Heating"
    hard_negative = by_id["next-0006"]
    assert hard_negative["expert_confirmed"] is True
    assert hard_negative["weight"] >= 10
    assert hard_negative["weight"] > hot["weight"]
    assert hard_negative["presence"] == "no"
    assert hard_negative["class_name"] == "Background"
    assert hard_negative["polarity"] is None
    assert "в норме" in hard_negative["notes"]
    assert hard_negative["class_name"] != "Local_Cooling"
    heating = by_id["next-0004"]
    assert heating["expert_confirmed"] is True
    assert heating["weight"] >= 10
    assert heating["presence"] == "yes"
    assert heating["class_name"] == "Local_Heating"
    assert heating["split_code"] == 1
    assert "после охлаждения" in heating["notes"]
    summary = summarize_weak_rows(rows)
    assert summary["n_rows"] == 6
    assert summary["holdout_used_for_generation"] is False
    assert summary["guard_used_for_generation"] is False
    assert sum(summary["counts_by_file"].values()) == summary["n_rows"]
    assert sum(summary["counts_by_class"].values()) == summary["n_rows"]
    assert set(summary["split_codes"]) <= {0, 1}


def test_analyze_field_does_not_read_weak_labels(monkeypatch):
    source = inspect.getsource(analyze_field)
    assert "weak_label" not in source
    assert "temporal_innovation" not in source
    real_open = open

    def guard(path, *args, **kwargs):
        name = str(path).replace("\\", "/")
        if "weak_label" in name:
            raise AssertionError(name)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", guard)
    hot = np.full((12, 100), 22.0, dtype=np.float32)
    hot[:, 45:51] += 6.0
    found = analyze_field(hot, np.arange(100), np.arange(12) * 60.0)
    assert found["anomaly_status"] == "Anomaly"
    assert found["manifest"]["clean"] == "median_x3"
    assert found["manifest"]["detector"] == "spatial_z_halo"
    assert found["manifest"]["config"]["threshold_z"] == 3.5
    assert found["manifest"]["trained_classifier_operational"] is False
    assert all(zone["expert_confirmed"] is False for zone in found["zones"])


def test_committed_weak_labels_match_summary_and_keep_seeds():
    """Recount the committed jsonl. Holdout and guard codes stay absent."""
    import json
    from collections import Counter
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    summary = json.loads((root / "docs" / "evidence" / "weak_labels_summary.json").read_text(encoding="utf-8"))
    by_file, by_class, by_origin = Counter(), Counter(), Counter()
    codes = set()
    confirmed = []
    seeds = {}
    n = 0
    with (root / "docs" / "evidence" / "weak_labels.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            n += 1
            by_file[str(row["record"])] += 1
            by_class[row["class_name"]] += 1
            by_origin[row["origin"]] += 1
            codes.add(int(row["split_code"]))
            if row["expert_confirmed"]:
                confirmed.append(row["id"])
            if row["origin"] == "expert_seed":
                seeds[row["id"]] = row
                assert row["weight"] > AUTOMATIC_WEIGHT
            else:
                assert row["weight"] == AUTOMATIC_WEIGHT
                assert row["expert_confirmed"] is False
    assert n == summary["n_rows"]
    assert {name: by_file[name] for name in map(str, range(1, 7))} == summary["counts_by_file"]
    assert {name: by_class[name] for name in (
        "Background", "Local_Heating", "Local_Cooling", "Periodic_Thermal", "Broad_Shift", "Sensor_Fault",
    )} == summary["counts_by_class"]
    assert sum(summary["counts_by_file"].values()) == n
    assert sum(summary["counts_by_class"].values()) == n
    assert summary["holdout_used_for_generation"] is False
    assert summary["guard_used_for_generation"] is False
    assert codes <= {0, 1}
    assert {"spatial_z", "rule", "quiet"} <= set(by_origin)
    assert set(confirmed) == {
        "review-00034", "review-00098", "review-00187", "review-00223",
    }
    assert seeds["review-00034"]["class_name"] == "Local_Heating"
    assert seeds["review-00034"]["polarity"] == "hot" and seeds["review-00034"]["structure"] == "one_span"
    assert seeds["review-00098"]["class_name"] == "Local_Heating" and seeds["review-00098"]["structure"] == "one_span"
    assert seeds["review-00187"]["class_name"] == "Local_Heating"
    assert seeds["review-00187"]["structure"] == "several_peaks"
    assert seeds["review-00187"]["recorded_shape"] == "несколько Local_Heating"
    assert seeds["review-00223"]["class_name"] == "Local_Cooling" and seeds["review-00223"]["polarity"] == "cold"
    assert all(row["presence"] == "yes" and row["expert_confirmed"] is True for row in seeds.values())


_YES_CLASS = {
    "next-0001": "Local_Heating",
    "next-0002": "Local_Cooling",
    "next-0003": "Local_Heating",
    "next-0004": "Local_Heating",
    "next-0008": "Local_Heating",
    "next-0009": "Local_Heating",
    "next-0011": "Local_Heating",
}
_NO_IDS = ("next-0005", "next-0006", "next-0007", "next-0010")
_OLD_IDS = ("review-00034", "review-00098", "review-00187", "review-00223")


_GAP_UNMAPPED = (
    "gap-r5-x0.000-x127.000-t0-1537",
    "gap-r5-x155.000-x213.000-t0-1537",
    "gap-r6-x1116.100-x1129.296-t251-383",
)


def test_committed_expert_seeds_cover_fifteen_cards_and_drop_holdout():
    """Load the seed file and recompute it from record time arrays."""
    import json
    from pathlib import Path

    import numpy as np

    from fiber.weak_labels import class_from_presence_and_shape

    root = Path(__file__).resolve().parents[1]
    answers_path = root / "docs" / "evidence" / "zakhar_next_shortlist_answers.jsonl"
    download = Path(r"C:\Users\bobr\Downloads\zakhar_next_shortlist_answers.jsonl")
    gap_path = root / "docs" / "evidence" / "zakhar_gap_shortlist_answers.jsonl"
    gap_download = Path(r"C:\Users\bobr\Downloads\zakhar_next_shortlist_answers (1).jsonl")
    assert answers_path.read_bytes() == download.read_bytes()
    assert gap_path.read_bytes() == gap_download.read_bytes()
    assert answers_path.read_bytes() != gap_path.read_bytes()
    answers = [json.loads(line) for line in answers_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    gap_answers = [json.loads(line) for line in gap_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [row["id"] for row in answers] == [f"next-{index:04d}" for index in range(1, 12)]
    assert len(gap_answers) == 10
    assert all(row["reviewer"] == "Захар" and row["expert_confirmed"] is True for row in gap_answers)
    times = {}
    for name in map(str, range(1, 7)):
        directory = root / "artifacts" / "records" / name
        times[name] = (
            np.load(directory / "time_s.npy"),
            np.load(directory / "time_iso.npy", allow_pickle=True),
        )
    rows, report = expert_card_report(times)
    stored_rows = [
        json.loads(line)
        for line in (root / "docs" / "evidence" / "expert_seeds.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    stored = json.loads((root / "docs" / "evidence" / "expert_seeds_summary.json").read_text(encoding="utf-8"))
    assert [json.loads(json.dumps(row, ensure_ascii=False, allow_nan=False)) for row in rows] == stored_rows
    assert report == stored
    cards = stored["cards"]
    assert len(EXPERT_SEEDS) == 15
    assert [card["id"] for card in cards[:15]] == [seed["id"] for seed in EXPERT_SEEDS]
    assert [card["id"] for card in cards[15:]] == [row["id"] for row in gap_answers]
    assert stored["n_cards"] == len(cards) == len(EXPERT_SEEDS) + len(gap_answers)
    assert stored["n_kept"] + stored["n_excluded"] + stored["n_unmapped"] == stored["n_cards"]
    assert {row["id"] for row in gap_answers} <= {card["id"] for card in cards}
    assert stored["holdout_used_for_generation"] is False
    assert stored["guard_used_for_generation"] is False
    assert stored["expert_weight"] >= 10
    assert stored["expert_weight"] > stored["automatic_weight"] == AUTOMATIC_WEIGHT
    by_card = {card["id"]: card for card in cards}
    by_row = {row["id"]: row for row in stored_rows}
    assert set(stored["generated_ids"]) == set(by_row)
    for row in stored_rows:
        assert row["split_code"] in (0, 1)
        assert row["origin"] == "expert_seed"
        assert row["expert_confirmed"] is True
        assert row["weight"] >= 10
        assert row["weight"] > AUTOMATIC_WEIGHT
        assert row["provenance"]["holdout_used_for_generation"] is False
        assert row["provenance"]["guard_used_for_generation"] is False
    for card in cards:
        assert card["weight"] >= 10
        assert card["expert_confirmed"] is True
        if card["status"] == "kept":
            assert card["split_code"] in (0, 1)
            assert card["id"] in by_row
            assert by_row[card["id"]]["class_name"] == card["class_name"]
            assert by_row[card["id"]]["notes"] == card["notes"]
            assert by_row[card["id"]]["presence"] == card["presence"]
        elif card["status"] == "excluded":
            assert card["split_code"] in (2, -1)
            assert card["id"] not in by_row
        else:
            assert card["status"] == "unmapped"
            assert card["class_name"] is None
            assert card["id"] not in by_row
    for answer in answers:
        card = by_card[answer["id"]]
        assert card["record"] == str(answer["record"])
        assert card["presence"] == answer["presence"]
        assert card["notes"] == answer["notes"]
        assert card["recorded_shape"] == answer["shape"]
        assert card["time_iso"] == answer["time"]
        assert abs(card["x_m"][0] - float(answer["meters"][0])) < 0.02
        assert abs(card["x_m"][1] - float(answer["meters"][1])) < 0.02
        assert card["class_name"] == map_expert_class(answer["presence"], answer["shape"])
    for seed_id in _NO_IDS:
        assert by_card[seed_id]["presence"] == "no"
        assert by_card[seed_id]["class_name"] == "Background"
        assert by_card[seed_id]["class_name"] != "Local_Cooling"
    assert "в норме" in by_card["next-0006"]["notes"]
    assert "обычный рабочий темп" in by_card["next-0005"]["notes"]
    assert by_card["next-0003"]["class_name"] == "Local_Heating"
    assert "широким heating" in by_card["next-0003"]["notes"]
    assert by_card["next-0003"]["class_name"] != "Broad_Shift"
    for seed_id, class_name in _YES_CLASS.items():
        assert by_card[seed_id]["presence"] == "yes"
        assert by_card[seed_id]["class_name"] == class_name
    for seed_id in _OLD_IDS:
        assert by_card[seed_id]["status"] == "kept"
        assert by_card[seed_id]["weight"] >= 10
        assert by_row[seed_id]["expert_confirmed"] is True
    assert by_row["review-00034"]["class_name"] == "Local_Heating"
    assert by_row["review-00187"]["structure"] == "several_peaks"
    assert by_row["review-00223"]["class_name"] == "Local_Cooling"
    for answer in gap_answers:
        card = by_card[answer["id"]]
        assert card["record"] == str(answer["record"])
        assert card["presence"] == answer["presence"]
        assert card["notes"] == answer["notes"]
        assert card["recorded_shape"] == answer["shape"]
        assert card["time_iso"] == answer["time"]
        assert card["expert_confirmed"] is True
        assert abs(card["x_m"][0] - float(answer["meters"][0])) < 0.02
        assert abs(card["x_m"][1] - float(answer["meters"][1])) < 0.02
        mapped = class_from_presence_and_shape(answer["presence"], answer["shape"])
        assert card["class_name"] == mapped
        if mapped is None:
            assert card["status"] == "unmapped"
            assert card["id"] not in by_row
        elif card["status"] == "kept":
            assert card["split_code"] in (0, 1)
            assert by_row[card["id"]]["class_name"] == mapped
            assert by_row[card["id"]]["notes"] == answer["notes"]
            assert by_row[card["id"]]["weight"] == EXPERT_WEIGHT
        else:
            assert card["status"] == "excluded"
            assert card["split_code"] in (2, -1)
            assert card["id"] not in by_row
    cooling = by_card["gap-r3-x0.000-x72.325-t0-465"]
    assert cooling["class_name"] == "Local_Cooling"
    assert cooling["class_name"] != "Broad_Shift"
    assert "нагрев" in cooling["notes"]
    quiet = by_card["gap-r2-x202.258-x266.971-t0-410"]
    assert quiet["presence"] == "no"
    assert quiet["class_name"] == "Background"
    assert quiet["class_name"] != "Local_Cooling"
    assert by_card["gap-r2-x271.285-x276.107-t0-246"]["class_name"] == "Local_Heating"
    for seed_id in _GAP_UNMAPPED:
        card = by_card[seed_id]
        assert card["status"] == "unmapped"
        assert card["class_name"] is None
        assert seed_id not in by_row
        assert card["notes"]
    unclear = by_card["gap-r5-x0.000-x127.000-t0-1537"]
    assert unclear["presence"] == "unclear"
    assert "хитингом" in unclear["notes"]
    empty_cooling_note = by_card["gap-r5-x155.000-x213.000-t0-1537"]
    assert empty_cooling_note["presence"] == "yes" and empty_cooling_note["recorded_shape"] == ""
    assert "охлаждение" in empty_cooling_note["notes"]
    empty_heating_note = by_card["gap-r6-x1116.100-x1129.296-t251-383"]
    assert empty_heating_note["presence"] == "yes" and empty_heating_note["recorded_shape"] == ""
    assert "нагрев" in empty_heating_note["notes"]


def test_gap_builder_keeps_mapped_answers_and_drops_unmapped():
    """expert_card_report is the writer. These checks read that return value."""
    import json
    from pathlib import Path

    import pytest

    from fiber.weak_labels import class_from_presence_and_shape

    root = Path(__file__).resolve().parents[1]
    gap_path = root / "docs" / "evidence" / "zakhar_gap_shortlist_answers.jsonl"
    gap_download = Path(r"C:\Users\bobr\Downloads\zakhar_next_shortlist_answers (1).jsonl")
    previous = root / "docs" / "evidence" / "zakhar_next_shortlist_answers.jsonl"
    assert gap_path.read_bytes() == gap_download.read_bytes()
    assert previous.read_bytes() != gap_path.read_bytes()
    gap_answers = [json.loads(line) for line in gap_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(gap_answers) == 10
    times = {}
    for name in map(str, range(1, 7)):
        directory = root / "artifacts" / "records" / name
        times[name] = (
            np.load(directory / "time_s.npy"),
            np.load(directory / "time_iso.npy", allow_pickle=True),
        )
    rows, report = expert_card_report(times)
    stored = json.loads((root / "docs" / "evidence" / "expert_seeds_summary.json").read_text(encoding="utf-8"))
    assert report == stored
    assert report["holdout_used_for_generation"] is False
    assert report["guard_used_for_generation"] is False
    assert report["n_kept"] + report["n_excluded"] + report["n_unmapped"] == report["n_cards"]
    assert report["n_cards"] == len(EXPERT_SEEDS) + len(gap_answers)
    by_card = {card["id"]: card for card in report["cards"]}
    emitted = {row["id"]: row for row in rows}
    assert set(report["generated_ids"]) == set(emitted)
    for row in rows:
        assert row["split_code"] in (0, 1)
        assert row["weight"] == EXPERT_WEIGHT
        assert row["weight"] > AUTOMATIC_WEIGHT
        assert row["expert_confirmed"] is True
        assert row["origin"] == "expert_seed"
    assert by_card["next-0003"]["class_name"] == "Local_Heating"
    assert [seed["id"] for seed in EXPERT_SEEDS] == [card["id"] for card in report["cards"][:15]]
    for answer in gap_answers:
        assert answer["reviewer"] == "Захар" and answer["expert_confirmed"] is True
        card = by_card[answer["id"]]
        assert card["notes"] == answer["notes"]
        assert card["presence"] == answer["presence"]
        assert card["recorded_shape"] == answer["shape"]
        assert card["record"] == str(answer["record"])
        assert card["time_iso"] == answer["time"]
        assert card["weight"] == EXPERT_WEIGHT
        assert card["expert_confirmed"] is True
        mapped = class_from_presence_and_shape(answer["presence"], answer["shape"])
        assert card["class_name"] == mapped
        if mapped is None:
            with pytest.raises(ValueError):
                map_expert_class(answer["presence"], answer["shape"])
            assert card["status"] == "unmapped"
            assert answer["id"] not in emitted
        else:
            assert mapped == map_expert_class(answer["presence"], answer["shape"])
            assert mapped in {
                "Background", "Local_Heating", "Local_Cooling",
                "Periodic_Thermal", "Broad_Shift", "Sensor_Fault",
            }
            assert card["status"] in ("kept", "excluded")
            if int(card["split_code"]) in (0, 1):
                assert card["status"] == "kept"
                assert emitted[answer["id"]]["class_name"] == mapped
                assert emitted[answer["id"]]["notes"] == answer["notes"]
                assert emitted[answer["id"]]["weight"] == EXPERT_WEIGHT
                assert emitted[answer["id"]]["split_code"] in (0, 1)
            else:
                assert card["status"] == "excluded"
                assert int(card["split_code"]) in (2, -1)
                assert answer["id"] not in emitted
    cooling = by_card["gap-r3-x0.000-x72.325-t0-465"]
    assert cooling["class_name"] == "Local_Cooling"
    assert cooling["class_name"] != "Broad_Shift"
    quiet = by_card["gap-r2-x202.258-x266.971-t0-410"]
    assert quiet["presence"] == "no" and quiet["class_name"] == "Background"
    assert by_card["gap-r2-x271.285-x276.107-t0-246"]["class_name"] == "Local_Heating"
    for seed_id in _GAP_UNMAPPED:
        assert by_card[seed_id]["status"] == "unmapped"
        assert by_card[seed_id]["class_name"] is None
        assert seed_id not in emitted


def _contiguous_code(codes, code, length=6):
    codes = np.asarray(codes)
    for start in np.flatnonzero(codes == code):
        stop = int(start) + length
        if stop <= len(codes) and np.all(codes[int(start):stop] == code):
            return int(start), stop
    raise AssertionError(f"no contiguous split {code}")


def _synthetic_answer(iso, start, stop, presence, shape, notes="", seed_id=None):
    if seed_id is None:
        seed_id = f"ask-r1-x0.000-x1.000-t{start}-{stop}"
    return {
        "id": seed_id,
        "record": "1",
        "meters": [0.0, 1.0],
        "time": [str(iso[start]), str(iso[stop - 1])],
        "presence": presence,
        "shape": shape,
        "notes": notes,
        "reviewer": "Захар",
        "reviewer_type": "human",
        "expert_confirmed": True,
    }


def test_ask_mapper_drops_unclear_existing_ids_and_holdout():
    """The shipped mapper and ask_seed_report. No record CSV is loaded."""
    from fiber.real_data import crop_partition, temporal_partition
    from fiber.weak_labels import ask_seed_report, class_from_presence_and_shape

    assert map_expert_class("no", "") == "Background"
    assert map_expert_class("no", "Local_Cooling") == "Background"
    assert map_expert_class("yes", "Periodic_Thermal") == "Periodic_Thermal"
    assert map_expert_class("yes", "Periodic_Thermal, Local_Heating") == "Periodic_Thermal"
    assert map_expert_class("yes", "Local_Heating, Periodic_Thermal") == "Local_Heating"
    assert map_expert_class("yes", "Local_Heating, Local_Heating") == "Local_Heating"
    assert map_expert_class("yes", "несколько Local_Heating") == "Local_Heating"
    assert map_expert_class("yes", "несколько участков, Local_Heating") == "Local_Heating"
    assert class_from_presence_and_shape("unclear", "Broad_Shift") is None
    assert class_from_presence_and_shape("yes", "") is None
    noted = map_expert_class("yes", "Local_Heating")
    assert noted == "Local_Heating"
    assert noted != "Broad_Shift"

    time_s = np.arange(120, dtype=float) * 100.0
    labels, _info = temporal_partition(time_s, 1800.0)
    iso = np.array([f"t{index:04d}" for index in range(len(time_s))])
    train = _contiguous_code(labels, 0)
    holdout = _contiguous_code(labels, 2)
    mixed = (train[0], holdout[1])
    assert int(crop_partition(labels[mixed[0]:mixed[1]])) == -1
    answers = [
        _synthetic_answer(iso, *train, "no", "", notes="как будто Local_Cooling и Broad_Shift"),
        _synthetic_answer(iso, *train, "yes", "Periodic_Thermal", notes="заметка про Broad_Shift", seed_id=f"ask-r1-x1.000-x2.000-t{train[0]}-{train[1]}"),
        _synthetic_answer(iso, *train, "yes", "Periodic_Thermal, Local_Heating", seed_id=f"ask-r1-x2.000-x3.000-t{train[0]}-{train[1]}"),
        _synthetic_answer(iso, *train, "unclear", "", notes="Periodic_Thermal", seed_id=f"ask-r1-x3.000-x4.000-t{train[0]}-{train[1]}"),
        _synthetic_answer(iso, *train, "yes", "", notes="охлаждение", seed_id=f"ask-r1-x4.000-x5.000-t{train[0]}-{train[1]}"),
        _synthetic_answer(iso, *train, "yes", "Sensor_Fault", seed_id="next-0005"),
        _synthetic_answer(iso, *holdout, "yes", "Local_Heating", seed_id=f"ask-r1-x5.000-x6.000-t{holdout[0]}-{holdout[1]}"),
        _synthetic_answer(iso, *mixed, "yes", "Local_Cooling", seed_id=f"ask-r1-x6.000-x7.000-t{mixed[0]}-{mixed[1]}"),
    ]
    times = {"1": (time_s, iso)}
    rows, report = ask_seed_report(answers, times, frozenset({"next-0005"}))
    again, _again = ask_seed_report(answers, times, frozenset({"next-0005"}))
    assert [row["id"] for row in again] == [row["id"] for row in rows]
    by_card = {card["id"]: card for card in report["cards"]}
    emitted = {row["id"]: row for row in rows}
    quiet = by_card[answers[0]["id"]]
    assert quiet["status"] == "kept" and quiet["class_name"] == "Background"
    assert quiet["class_name"] != "Local_Cooling"
    assert "Local_Cooling" in quiet["notes"]
    assert emitted[quiet["id"]]["class_name"] == "Background"
    assert emitted[quiet["id"]]["weight"] == EXPERT_WEIGHT
    periodic = by_card[answers[1]["id"]]
    assert periodic["class_name"] == "Periodic_Thermal"
    assert periodic["class_name"] != "Broad_Shift"
    assert "Broad_Shift" in periodic["notes"]
    multi = by_card[answers[2]["id"]]
    assert multi["recorded_shape"] == "Periodic_Thermal, Local_Heating"
    assert multi["class_name"] == "Periodic_Thermal"
    unclear = by_card[answers[3]["id"]]
    assert unclear["status"] == "unclear"
    assert unclear["weight"] != EXPERT_WEIGHT
    assert answers[3]["id"] not in emitted
    empty = by_card[answers[4]["id"]]
    assert empty["status"] == "empty_shape"
    assert empty["class_name"] is None
    assert empty["weight"] != EXPERT_WEIGHT
    assert answers[4]["id"] not in emitted
    assert by_card["next-0005"]["status"] == "excluded"
    assert by_card["next-0005"]["reason"] == "already_present"
    assert "next-0005" not in emitted
    assert by_card[answers[6]["id"]]["status"] == "excluded"
    assert by_card[answers[6]["id"]]["split_code"] == 2
    assert answers[6]["id"] not in emitted
    assert by_card[answers[7]["id"]]["status"] == "excluded"
    assert by_card[answers[7]["id"]]["split_code"] == -1
    assert answers[7]["id"] not in emitted
    assert report["n_kept"] == len(rows) == 3
    assert report["n_kept"] + report["n_excluded"] + report["n_unclear"] + report["n_empty_shape"] == len(answers)
    assert report["holdout_used_for_generation"] is False
    assert report["guard_used_for_generation"] is False
    assert all(row["split_code"] in (0, 1) and row["expert_confirmed"] is True for row in rows)


def test_written_ask_seeds_match_the_shipped_report():
    """Read the ask seed file written from the download. Do not rescan temperatures."""
    import json
    from pathlib import Path

    from fiber.weak_labels import (
        ask_seed_report,
        class_from_presence_and_shape,
        current_expert_ids,
        load_ask_answers,
    )

    root = Path(__file__).resolve().parents[1]
    download = Path(r"C:\Users\bobr\Downloads\zakhar_next_shortlist_answers (2).jsonl")
    copied = root / "docs" / "evidence" / "zakhar_ask_shortlist_answers.jsonl"
    previous = root / "docs" / "evidence" / "zakhar_next_shortlist_answers.jsonl"
    gap = root / "docs" / "evidence" / "zakhar_gap_shortlist_answers.jsonl"
    assert copied.read_bytes() == download.read_bytes()
    assert copied.read_bytes() != previous.read_bytes()
    assert copied.read_bytes() != gap.read_bytes()
    answers = load_ask_answers(copied)
    assert len(answers) == sum(1 for line in download.read_text(encoding="utf-8").splitlines() if line.strip())
    times = {}
    for name in map(str, range(1, 7)):
        directory = root / "artifacts" / "records" / name
        times[name] = (
            np.load(directory / "time_s.npy"),
            np.load(directory / "time_iso.npy", allow_pickle=True),
        )
    existing = current_expert_ids()
    rows, report = ask_seed_report(answers, times, existing)
    again, _again = ask_seed_report(answers, times, existing)
    assert [row["id"] for row in again] == report["generated_ids"]
    stored_rows = [
        json.loads(line)
        for line in (root / "docs" / "evidence" / "expert_seeds_ask.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    stored = json.loads((root / "docs" / "evidence" / "expert_seeds_ask_summary.json").read_text(encoding="utf-8"))
    assert [json.loads(json.dumps(row, ensure_ascii=False, allow_nan=False)) for row in rows] == stored_rows
    assert report == stored
    assert stored["n_kept"] == len(stored_rows)
    assert stored["n_kept"] + stored["n_excluded"] + stored["n_unclear"] + stored["n_empty_shape"] == stored["n_answers"] == len(answers)
    assert stored["expert_weight"] == EXPERT_WEIGHT > stored["automatic_weight"] == AUTOMATIC_WEIGHT
    assert stored["holdout_used_for_generation"] is False
    assert stored["guard_used_for_generation"] is False
    assert stored["automatic_jsonl_regenerated"] is False
    assert stored["operational_clean"] == "median_x3"
    assert stored["operational_detector"] == "spatial_z_halo"
    assert stored["multi_label_rule"] == "first_named_class"
    by_card = {card["id"]: card for card in stored["cards"]}
    emitted = {row["id"]: row for row in stored_rows}
    assert set(stored["generated_ids"]) == set(emitted)
    assert existing.isdisjoint(emitted)
    for row in stored_rows:
        assert row["split_code"] in (0, 1)
        assert row["weight"] == EXPERT_WEIGHT
        assert row["expert_confirmed"] is True
        assert row["origin"] == "expert_seed"
        assert row["class_name"] == map_expert_class(row["presence"], row["recorded_shape"])
        assert row["recorded_shape"] == by_card[row["id"]]["recorded_shape"]
    for answer in answers:
        card = by_card[answer["id"]]
        assert card["presence"] == answer["presence"]
        assert card["recorded_shape"] == answer["shape"]
        assert card["notes"] == answer["notes"]
        mapped = class_from_presence_and_shape(answer["presence"], answer["shape"])
        if answer["presence"] == "unclear":
            assert card["status"] == "unclear"
            assert card["weight"] != EXPERT_WEIGHT
            assert answer["id"] not in emitted
        elif answer["presence"] == "yes" and answer["shape"].strip() == "":
            assert card["status"] == "empty_shape"
            assert card["weight"] != EXPERT_WEIGHT
            assert answer["id"] not in emitted
        elif card["status"] == "kept":
            assert mapped == card["class_name"] == emitted[answer["id"]]["class_name"]
            assert emitted[answer["id"]]["notes"] == answer["notes"]
            named = [part.strip() for part in answer["shape"].split(",")]
            if len(named) > 1 and all(part in {
                "Background", "Local_Heating", "Local_Cooling",
                "Periodic_Thermal", "Broad_Shift", "Sensor_Fault",
            } for part in named):
                assert card["class_name"] == named[0]
                assert card["recorded_shape"] == answer["shape"]
        else:
            assert card["status"] == "excluded"
            assert answer["id"] not in emitted
    heating_note = by_card["ask-r1-x687.208-x691.980-t0-292"]
    assert heating_note["presence"] == "yes"
    assert heating_note["recorded_shape"] == "Local_Heating"
    assert heating_note["class_name"] == "Local_Heating"
    assert "broad_Shift" in heating_note["notes"]
    quiet = by_card["ask-r1-x0.000-x110.000-t172-258"]
    assert quiet["presence"] == "no" and quiet["class_name"] == "Background"
    assert "Periodic_Thermal" in quiet["notes"] and "Local_Heating" in quiet["notes"]
    periodic = by_card["ask-r1-x285.000-x365.000-t399-431"]
    assert periodic["recorded_shape"] == "Periodic_Thermal"
    assert periodic["class_name"] == "Periodic_Thermal"
    first = by_card["ask-r1-x1125.000-x1185.000-t399-431"]
    assert first["recorded_shape"] == "Periodic_Thermal, Local_Heating"
    assert first["class_name"] == "Periodic_Thermal"
    other = by_card["ask-r5-x490.000-x504.000-t192-211"]
    assert other["recorded_shape"] == "Local_Heating, Periodic_Thermal"
    assert other["class_name"] == "Local_Heating"
    decisions = (root / "docs" / "DECISIONS.md").read_text(encoding="utf-8")
    section = decisions.split("## 2026-10-15 — ответы ask-shortlist как expert seeds", 1)[1]
    assert f"{stored['n_kept']} kept" in section
    assert f"{stored['n_excluded']} excluded" in section
    assert f"{stored['n_unclear']} unclear" in section
    assert f"{stored['n_empty_shape']} empty_shape" in section
    assert "Holdout и guard для генерации не использованы." in section
    assert "median_x3" in section and "spatial_z_halo" in section
    assert "## 2026-10-14 — одиннадцать ответов Захара как expert seeds" in decisions
