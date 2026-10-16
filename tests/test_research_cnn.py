"""Research CNN fit rows keep expert seeds, and analyze_field ignores the weights."""
import inspect

import numpy as np

from fiber.pipeline import analyze_field
from fiber.research_cnn import baseline_overlap, fit_rows, loss_and_checkpoint_masks
from fiber.weak_labels import EXPERT_SEED_BY_ID, EXPERT_WEIGHT


def test_fit_rows_weight_expert_seed_and_drop_holdout_and_guard():
    rows = [
        {
            "id": "quiet-auto", "record": "1", "origin": "quiet", "class_name": "Background",
            "split_code": 0, "weight": 1.0, "expert_confirmed": False,
            "rows_half_open": [0, 28], "x_m": [10.0, 14.0],
        },
        {
            "id": "review-00034", "record": "1", "origin": "expert_seed", "class_name": "Local_Heating",
            "split_code": 0, "weight": 10.0, "expert_confirmed": True,
            "rows_half_open": [310, 335], "x_m": [517.77, 523.25],
        },
        {
            "id": "holdout-row", "record": "2", "origin": "quiet", "class_name": "Background",
            "split_code": 2, "weight": 1.0, "expert_confirmed": False,
            "rows_half_open": [0, 28], "x_m": [1.0, 5.0],
        },
        {
            "id": "guard-row", "record": "3", "origin": "rule", "class_name": "Local_Cooling",
            "split_code": -1, "weight": 1.0, "expert_confirmed": False,
            "rows_half_open": [4, 20], "x_m": [2.0, 9.0],
        },
        {
            "id": "dev-row", "record": "4", "origin": "spatial_z", "class_name": "Local_Heating",
            "split_code": 1, "weight": 1.0, "expert_confirmed": False,
            "rows_half_open": [8, 40], "x_m": [3.0, 8.0],
        },
        {
            "id": "next-0006", "record": "9", "origin": "expert_seed", "class_name": "Local_Cooling",
            "split_code": -1, "weight": 1.0, "expert_confirmed": False,
            "rows_half_open": [0, 2], "x_m": [0.0, 1.0],
            "notes": "это не норма, это cooling",
            "presence": "yes",
        },
        {
            "id": "next-0002", "record": "9", "origin": "expert_seed", "class_name": "Background",
            "split_code": 2, "weight": EXPERT_WEIGHT, "expert_confirmed": True,
            "rows_half_open": [0, 2], "x_m": [0.0, 1.0],
        },
    ]
    kept = fit_rows(rows)
    by_id = {row["id"]: row for row in kept}
    assert set(by_id) == {"quiet-auto", "review-00034", "dev-row", "next-0006", "next-0002"}
    assert "holdout-row" not in by_id
    assert "guard-row" not in by_id
    assert by_id["review-00034"]["loss_weight"] > by_id["quiet-auto"]["loss_weight"]
    assert by_id["review-00034"]["loss_weight"] > by_id["dev-row"]["loss_weight"]
    assert by_id["review-00034"]["class_name"] == "Local_Heating"
    assert by_id["quiet-auto"]["loss_weight"] == 1.0
    assert by_id["quiet-auto"]["expert_confirmed"] is False
    hard_negative = by_id["next-0006"]
    cooling = by_id["next-0002"]
    assert hard_negative["class_name"] == "Background"
    assert hard_negative["presence"] == "no"
    assert hard_negative["loss_weight"] >= 10
    assert hard_negative["loss_weight"] > by_id["quiet-auto"]["loss_weight"]
    assert hard_negative["expert_confirmed"] is True
    assert hard_negative["split_code"] == -1
    assert "в норме" in hard_negative["notes"]
    assert hard_negative["rows_half_open"] == list(EXPERT_SEED_BY_ID["next-0006"]["rows_half_open"])
    assert cooling["class_name"] == "Local_Cooling"
    assert cooling["loss_weight"] >= 10
    assert cooling["split_code"] == 2
    assert cooling["expert_confirmed"] is True
    for row in kept:
        if row["origin"] == "expert_seed":
            assert row["loss_weight"] >= 10
        else:
            assert row["split_code"] in (0, 1)
            assert row["loss_weight"] == 1.0
    loss, checkpoint = loss_and_checkpoint_masks(kept)
    loss_ids = {kept[index]["id"] for index, flag in enumerate(loss) if flag}
    checkpoint_ids = {kept[index]["id"] for index, flag in enumerate(checkpoint) if flag}
    assert {"review-00034", "next-0006", "next-0002", "quiet-auto"} <= loss_ids
    assert "dev-row" in checkpoint_ids
    assert "next-0006" not in checkpoint_ids and "next-0002" not in checkpoint_ids
    overlap = baseline_overlap(
        np.array([True, False, True, False]),
        np.array([True, True, False, False]),
    )
    assert overlap["intersection"] == 1
    assert overlap["union"] == 3
    assert overlap["overlap_jaccard"] == overlap["intersection"] / overlap["union"]


def test_analyze_field_ignores_research_cnn(monkeypatch):
    source = inspect.getsource(analyze_field)
    assert "research_cnn" not in source
    assert "ResearchResidualCNN" not in source
    assert "research_residual_cnn" not in source
    real_open = open

    def guard(path, *args, **kwargs):
        name = str(path).replace("\\", "/")
        if "research_residual_cnn" in name or "research_cnn" in name:
            raise AssertionError(name)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", guard)
    hot = np.full((12, 100), 22.0, dtype=np.float32)
    hot[:, 45:51] += 6.0
    found = analyze_field(hot, np.arange(100, dtype=np.float64), np.arange(12, dtype=np.float64) * 60.0)
    assert found["anomaly_status"] == "Anomaly"
    assert found["manifest"]["clean"] == "median_x3"
    assert found["manifest"]["detector"] == "spatial_z_halo"
    assert found["manifest"]["config"]["threshold_z"] == 3.5
    assert found["manifest"]["trained_classifier_operational"] is False
    assert all(zone["expert_confirmed"] is False for zone in found["zones"])


def test_research_fit_rows_skip_gap_statistics_cards():
    """Gap cards stay in the seed statistics and out of the research fit."""
    import importlib.util
    from pathlib import Path

    from fiber.classify.rules import LABELS
    from fiber.weak_labels import EXPERT_SEEDS

    path = Path(__file__).resolve().parents[1] / "scripts" / "train_research_cnn.py"
    spec = importlib.util.spec_from_file_location("train_research_cnn_gap", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expert = module._expert_rows()
    assert [row["id"] for row in expert] == [seed["id"] for seed in EXPERT_SEEDS]
    assert all(not str(row["id"]).startswith("gap-") for row in expert)
    kept = fit_rows(expert)
    assert [row["id"] for row in kept] == [seed["id"] for seed in EXPERT_SEEDS]
    for row in kept:
        assert row["class_name"] in LABELS
        assert row["class_name"] is not None
        assert row["origin"] == "expert_seed"
        assert row["loss_weight"] >= EXPERT_WEIGHT
        assert row["loss_weight"] > 1.0


_COOLING_IDS = ("review-00223", "next-0002")
_HARD_NEGATIVE_IDS = ("next-0005", "next-0006", "next-0007", "next-0010")
_ALL_IDS = (
    "review-00034", "review-00098", "review-00187", "review-00223",
    "next-0001", "next-0002", "next-0003", "next-0004", "next-0005",
    "next-0006", "next-0007", "next-0008", "next-0009", "next-0010", "next-0011",
)


def test_committed_research_report_covers_fifteen_cards():
    """Load the report and manifest written by the shipped eval entry."""
    import json
    import math
    from pathlib import Path

    from fiber.classify.rules import LABELS
    from fiber.weak_labels import EXPERT_SEEDS, map_expert_class

    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / "docs" / "evidence" / "research_cnn_report.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "docs" / "evidence" / "research_cnn_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "docs" / "evidence" / "expert_seeds_summary.json").read_text(encoding="utf-8"))
    assert report["serve"] == "research"
    assert manifest["serve"] == "research"
    assert manifest["trained_classifier_operational"] is False
    assert report["holdout_used_for_fit"] is False
    assert report["guard_used_for_fit"] is False
    assert manifest["holdout_used_for_fit"] is False
    assert manifest["guard_used_for_fit"] is False
    assert manifest["automatic_holdout_used_for_fit"] is False
    assert manifest["automatic_guard_used_for_fit"] is False
    assert math.isfinite(float(manifest["development_loss"]))
    assert math.isfinite(float(report["baseline_zone_overlap"]))
    cards = report["cards"]
    assert [card["id"] for card in cards] == list(_ALL_IDS)
    assert [card["id"] for card in cards] == [seed["id"] for seed in EXPERT_SEEDS]
    by_card = {card["id"]: card for card in cards}
    outside = {
        card["id"] for card in summary["cards"] if int(card["split_code"]) not in (0, 1)
    }
    named = {item["id"] for item in manifest["expert_seeds_still_fit_on_split_2_or_minus_1"]}
    assert named == outside
    assert named == {card["id"] for card in cards if card["fit_despite_split"]}
    assert report["expert_cards_are_not_an_untouched_holdout"] is True
    for card in cards:
        assert card["network_class"] in LABELS
        assert card["predicted"] == card["network_class"]
        assert card["expert_class"] == card["recorded"]
        assert card["expert_class"] == map_expert_class(card["presence"], card["recorded_shape"])
        assert card["used_as_fit_seed"] is True
        assert card["untouched_holdout"] is False
        seed = EXPERT_SEED_BY_ID[card["id"]]
        assert card["record"] == seed["record"]
        assert card["rows_half_open"] == [int(seed["rows_half_open"][0]), int(seed["rows_half_open"][1])]
        assert abs(card["x_m"][0] - float(seed["x_m"][0])) < 0.02
        assert abs(card["x_m"][1] - float(seed["x_m"][1])) < 0.02
    for seed_id in _COOLING_IDS:
        assert by_card[seed_id]["expert_class"] == "Local_Cooling"
        assert by_card[seed_id]["presence"] == "yes"
    for seed_id in _HARD_NEGATIVE_IDS:
        card = by_card[seed_id]
        assert card["presence"] == "no"
        assert card["expert_class"] == "Background"
        assert isinstance(card["baseline_note"], str) and card["baseline_note"]
        assert "baseline_overlaps" in card
        if card["baseline_overlaps"]:
            assert math.isfinite(float(card["baseline_contrast_c"]))
            assert "normal" in card["baseline_note"]
        else:
            assert card["baseline_contrast_c"] is None
    assert "в норме" in by_card["next-0006"]["notes"]


def test_context_forward_reads_analyze_field_and_leaves_status():
    """The context forward takes the residual crop plus the three measured numbers."""
    from pathlib import Path

    import torch

    from fiber.classify.rules import LABELS
    from fiber.research_cnn import (
        CONTEXT_ARCHITECTURE,
        ResearchContextCNN,
        ResearchResidualCNN,
        context_model_input,
        context_vector,
        encode_context,
        load_research_cnn,
        meter_columns,
        residual_crop,
    )
    from fiber.research_context import context_evidence

    raw = np.full((12, 261), 20.0, dtype=np.float32)
    raw[:, 30:151] += 8.0
    x_m = np.arange(261, dtype=np.float64)
    time_s = np.arange(12, dtype=np.float64) * 60.0
    field = analyze_field(raw, x_m, time_s)
    status = field["anomaly_status"]
    zones = [
        (zone["start_m"], zone["end_m"], zone["expert_confirmed"])
        for zone in field["zones"]
    ]
    measured = context_evidence(
        field, x_m, time_s, rows_half_open=(0, 12), x_span=(30.0, 150.0),
    )
    assert field["anomaly_status"] == status
    assert [
        (zone["start_m"], zone["end_m"], zone["expert_confirmed"])
        for zone in field["zones"]
    ] == zones
    vector = context_vector(
        measured["rectangle_median_c"],
        measured["far_cable_median_c"],
        measured["halo_inside_fraction"],
    )
    assert tuple(vector.shape) == (3,)
    assert np.isfinite(vector).all()
    assert vector[0] == np.float32(measured["rectangle_median_c"])
    assert vector[1] == np.float32(measured["far_cable_median_c"])
    assert vector[2] == np.float32(measured["halo_inside_fraction"])
    c0, c1 = meter_columns(x_m, 30.0, 150.0)
    crop = residual_crop(field["clean"], field["score_available"], 0, 12, c0, c1)
    assert crop is not None
    assert crop.shape == (32, 64)
    model = ResearchContextCNN(n_class=len(LABELS))
    scores = model(torch.from_numpy(crop[None, None]), torch.from_numpy(vector[None]))
    assert tuple(scores.shape) == (1, len(LABELS))
    assert int(scores.shape[1]) == 6
    assert torch.isfinite(scores).all()
    assert field["anomaly_status"] == status

    whole = context_evidence(
        field, x_m, time_s, rows_half_open=(0, 12), x_span=(0.0, 260.0),
    )
    assert field["anomaly_status"] == status
    assert whole["far_cable_median_c"] is None
    assert whole["far_left"]["present"] is False
    assert whole["far_right"]["present"] is False
    assert whole["far_left"]["median_c"] is None
    assert whole["far_right"]["median_c"] is None
    missing = context_vector(
        whole["rectangle_median_c"],
        whole["far_cable_median_c"],
        whole["halo_inside_fraction"],
    )
    assert not np.isfinite(missing[1])
    assert missing[0] == np.float32(whole["rectangle_median_c"])
    assert missing[1] != missing[0]
    packed = context_model_input(missing)
    encoded = encode_context(torch.from_numpy(missing[None].copy())).detach().numpy()[0]
    assert np.allclose(packed, encoded)
    assert float(packed[1]) == 0.0
    assert float(packed[3]) == 0.0
    assert float(packed[1]) != float(missing[0])
    c0, c1 = meter_columns(x_m, 0.0, 260.0)
    whole_crop = residual_crop(field["clean"], field["score_available"], 0, 12, c0, c1)
    assert whole_crop is not None
    whole_scores = model(
        torch.from_numpy(whole_crop[None, None]),
        torch.from_numpy(missing[None].copy()),
    )
    assert tuple(whole_scores.shape) == (1, 6)
    assert torch.isfinite(whole_scores).all()
    assert field["anomaly_status"] == status

    old_scores = ResearchResidualCNN(n_class=6)(torch.from_numpy(crop[None, None]))
    assert tuple(old_scores.shape) == (1, 6)
    root = Path(__file__).resolve().parents[1]
    _loaded, blob, device = load_research_cnn(root / "models" / "weights" / "research_residual_cnn.pt")
    assert blob["serve"] == "research"
    assert blob["architecture"] == "residual2d"
    assert blob["architecture"] != CONTEXT_ARCHITECTURE
    assert blob.get("operational") is not True
    assert blob.get("trained_classifier_operational") is not True
    loaded_scores = _loaded(torch.from_numpy(crop[None, None]).to(device))
    assert tuple(loaded_scores.shape) == (1, 6)


def test_context_fit_keeps_previous_seeds_and_drops_gap_cards():
    """The context fit is the previous 15 plus every ask expert seed.

    Gap cards stay out. Automatic split 2 and -1 stay out. The previous 15
    stay in, including a window whose own split is 2 or -1.
    """
    import json
    import importlib.util
    from pathlib import Path

    from fiber.classify.rules import LABELS
    from fiber.weak_labels import EXPERT_SEEDS, EXPERT_WEIGHT

    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "train_research_context_cnn.py"
    spec = importlib.util.spec_from_file_location("train_research_context_cnn", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fit = module.build_fit_rows()
    expert = [row for row in fit if row["origin"] == "expert_seed"]
    ask = [
        json.loads(line)
        for line in (root / "docs" / "evidence" / "expert_seeds_ask.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = json.loads((root / "docs" / "evidence" / "expert_seeds_ask_summary.json").read_text(encoding="utf-8"))
    assert [row["id"] for row in expert] == [seed["id"] for seed in EXPERT_SEEDS] + [row["id"] for row in ask]
    assert len(expert) == len({row["id"] for row in expert})
    assert all(not str(row["id"]).startswith("gap-") for row in fit)
    assert all(row["origin"] == "expert_seed" or int(row["split_code"]) in (0, 1) for row in fit)
    outside = [
        row for row in expert
        if row["id"] in {seed["id"] for seed in EXPERT_SEEDS} and int(row["split_code"]) not in (0, 1)
    ]
    assert outside
    by_id = {row["id"]: row for row in expert}
    unclear = {card["id"] for card in summary["cards"] if card["status"] == "unclear"}
    assert unclear
    assert unclear.isdisjoint(by_id)
    for row in ask:
        kept = by_id[row["id"]]
        assert kept["class_name"] == row["class_name"]
        assert kept["class_name"] in LABELS
        assert kept["loss_weight"] >= EXPERT_WEIGHT > 1.0
        assert kept["expert_confirmed"] is True
        assert kept["presence"] == row["presence"]
        assert int(kept["split_code"]) in (0, 1)
    noted = by_id["ask-r1-x687.208-x691.980-t0-292"]
    assert noted["class_name"] == "Local_Heating"
    assert "broad_Shift" in noted["notes"]
    quiet = by_id["ask-r1-x0.000-x110.000-t172-258"]
    assert quiet["presence"] == "no" and quiet["class_name"] == "Background"
    loss, checkpoint = loss_and_checkpoint_masks(fit)
    for index, row in enumerate(fit):
        if row["origin"] == "expert_seed":
            assert loss[index]
            assert not checkpoint[index]
        elif int(row["split_code"]) == 2 or int(row["split_code"]) == -1:
            raise AssertionError(row["id"])
    assert module.WEIGHTS.name == "research_context_cnn.pt"
    assert module.REPORT.name != "research_cnn_report.json"


def test_context_checkpoint_scores_ask_seeds_and_previous_fifteen():
    """Load the shipped checkpoint and read the shipped context report."""
    import hashlib
    import json
    import importlib.util
    from pathlib import Path

    from fiber.classify.rules import LABELS
    from fiber.research_cnn import (
        HISTORICAL_CONTEXT_MATCH_LABEL,
        HISTORICAL_CONTEXT_SHA256,
        load_research_context_cnn,
    )
    from fiber.weak_labels import EXPERT_SEEDS

    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "train_research_context_cnn.py"
    spec = importlib.util.spec_from_file_location("train_research_context_cnn_score", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fit = module.build_fit_rows()
    expert_ids = [row["id"] for row in fit if row["origin"] == "expert_seed"]
    ask = [
        json.loads(line)
        for line in (root / "docs" / "evidence" / "expert_seeds_ask.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert expert_ids == [seed["id"] for seed in EXPERT_SEEDS] + [row["id"] for row in ask]
    assert len(expert_ids) == len(set(expert_ids))
    assert all(not str(row["id"]).startswith("gap-") for row in fit)
    assert all(row["origin"] == "expert_seed" or int(row["split_code"]) in (0, 1) for row in fit)

    weights = root / "models" / "weights" / "research_context_cnn.pt"
    digest = hashlib.sha256()
    with weights.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    file_sha = digest.hexdigest()
    _model, blob, _device = load_research_context_cnn(weights)
    assert blob["serve"] == "research"
    assert blob.get("operational") is not True
    assert blob.get("trained_classifier_operational") is not True
    assert blob.get("sets_anomaly_status") is not True
    assert list(blob["class_names"]) == list(LABELS)
    assert len(LABELS) == 6
    report = json.loads((root / "docs" / "evidence" / "research_context_cnn_report.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "docs" / "evidence" / "research_context_cnn_manifest.json").read_text(encoding="utf-8"))
    assert file_sha == report["weights_sha256"] == manifest["weights_sha256"]
    assert file_sha != HISTORICAL_CONTEXT_SHA256
    assert report["serve"] == "research"
    assert report["operational"] is False
    assert report["sets_anomaly_status"] is False
    assert report["gap_cards_in_fit"] is False
    assert report["gap_cards_scored"] is False
    assert report["ask_recovery_is_in_sample"] is True
    assert report["previous_match_label"] == "10/15"
    assert report["historical_context_match_label"] == HISTORICAL_CONTEXT_MATCH_LABEL == "15/22"
    assert report["context_match_label"] == "15/22"
    cards = report["cards"]
    assert [card["id"] for card in cards] == expert_ids
    assert len(cards) == len({card["id"] for card in cards})
    by_card = {card["id"]: card for card in cards}
    assert by_card["next-0003"]["expert_class"] == "Local_Heating"
    seed_cards = [card for card in cards if card["source"] == "expert_seed"]
    ask_cards = [card for card in cards if card["source"] == "ask_expert_seed"]
    assert [card["id"] for card in seed_cards] == [seed["id"] for seed in EXPERT_SEEDS]
    assert [card["id"] for card in ask_cards] == [row["id"] for row in ask]
    assert report["expert_seed_match_label"] == (
        f"{sum(bool(card['match']) for card in seed_cards)}/{len(seed_cards)}"
    )
    assert report["ask_seed_match_label"] == (
        f"{sum(bool(card['match']) for card in ask_cards)}/{len(ask_cards)}"
    )
    assert report["expert_seed_n_cards"] == 15
    assert report["ask_seed_n_cards"] == len(ask)
    for card in cards:
        assert card["network_class"] in LABELS
        assert card["expert_class"] in LABELS
        assert card["match"] is (card["network_class"] == card["expert_class"])
        assert card["used_as_fit_seed"] is True
        assert not str(card["id"]).startswith("gap-")
    decisions = (root / "docs" / "DECISIONS.md").read_text(encoding="utf-8")
    heading = "## 2026-10-15 — research context CNN на старых и ask seeds"
    assert heading in decisions
    section = decisions.split(heading, 1)[1]
    report_text = (root / "REPORT.md").read_text(encoding="utf-8")
    statement = report_text.split("## Контекстная research-сеть", 1)[1]
    statement = statement.split("\n## ", 1)[0]
    for prose in (section, statement):
        assert report["ask_seed_match_label"] in prose
        assert report["expert_seed_match_label"] in prose
        assert "10/15" in prose
        assert "15/22" in prose
        assert file_sha in prose
        assert "serve=research" in prose
        assert "operational false" in prose
        assert "median_x3" in prose
        assert "spatial_z" in prose
        assert "Порог spatial_z и чистка median_x3 не менялись." in prose
