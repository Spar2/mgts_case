"""The viewer calls the shipped operational view, and REPORT matches the catalog."""
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from fiber.anomaly.detect import temporal_innovation
from fiber.denoise.dsp import median_x
from fiber.pipeline import analyze_field
from fiber.viewer import example_cable_step, example_hotspot, view_field


def _section(report: str, heading: str) -> str:
    body = report.split(heading, 1)[1]
    end = body.find("\n## ")
    return body if end < 0 else body[:end]


def _contrast_token(value: float) -> str:
    return format(float(value), ".6f").rstrip("0").rstrip(".")


def _printed(line: str, pattern: str) -> tuple[float, int]:
    match = re.search(pattern, line, flags=re.IGNORECASE)
    assert match, (pattern, line)
    token = match.group(1)
    places = len(token.split(".")[1]) if "." in token else 0
    return float(token), places


def _same_at_precision(stored, printed: float, places: int) -> None:
    assert round(float(stored), places) == printed

ROOT = Path(__file__).resolve().parents[1]


def test_view_field_hotspot_and_whole_cable_step():
    raw, x_m, time_s = example_hotspot()
    view = view_field(raw, x_m, time_s)
    found = analyze_field(raw, x_m, time_s)
    heat = np.asarray(view["heatmap"], dtype=np.float64)
    assert heat.ndim == 2 and heat.size > 1 and np.isfinite(heat).all()
    assert np.unique(heat).size > 1
    assert view["path"] == "operational"
    assert view["anomaly_status"] == found["anomaly_status"] == "Anomaly"
    manifest = view["manifest"]
    assert manifest["clean"] == "median_x3"
    assert manifest["detector"] == "spatial_z_halo"
    assert manifest["config"]["threshold_z"] == 3.5
    assert manifest["trained_classifier_operational"] is False
    assert view["n_zones"] == len(found["zones"]) >= 1
    alarm = (found["z"] >= 3.5) & found["score_available"]
    assert int(alarm.sum()) == sum(zone["support"] for zone in view["zones"])
    for zone, ref in zip(view["zones"], found["zones"], strict=True):
        assert zone["expert_confirmed"] is False
        assert zone["shape_hypothesis"] == ref["shape_hypothesis"]
        assert zone["evidence"]["spatial"]["view"] == "spatial"
        assert zone["evidence"]["temporal"]["view"] == "temporal"
        assert math.isfinite(zone["contrast"]) and zone["contrast"] > 0
        assert math.isfinite(zone["width_m"]) and zone["width_m"] > 0
        assert math.isfinite(zone["observed_duration_s"]) and zone["observed_duration_s"] > 0
        assert zone["support"] == ref["n_cells"] == ref["support"]
        cols = np.flatnonzero((x_m >= zone["start_m"] - 5e-4) & (x_m <= zone["end_m"] + 5e-4))
        rows = np.flatnonzero((time_s >= zone["t_start_s"] - 1e-3) & (time_s <= zone["t_end_s"] + 1e-3))
        assert zone["support"] == int(alarm[np.ix_(rows, cols)].sum())

    step, sx, st = example_cable_step()
    quiet = view_field(step, sx, st)
    assert quiet["anomaly_status"] == "Normal"
    assert quiet["zones"] == []
    valid = np.ones(step.shape, dtype=bool)
    historical = temporal_innovation(median_x(step, valid), valid, st)
    assert not np.all(historical == 0)


def test_example_endpoint_returns_the_shipped_view():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from fiber.viewer_app import app

    raw, x_m, time_s = example_hotspot()
    expected = view_field(raw, x_m, time_s)
    response = TestClient(app).get("/view/example")
    assert response.status_code == 200
    body = response.json()
    assert body["heatmap"] == expected["heatmap"]
    assert body["anomaly_status"] == expected["anomaly_status"]
    assert body["n_zones"] == expected["n_zones"]
    for zone, ref in zip(body["zones"], expected["zones"], strict=True):
        assert zone["shape_hypothesis"] == ref["shape_hypothesis"]
        assert zone["evidence"]["spatial"]["view"] == "spatial"
        assert zone["evidence"]["temporal"]["view"] == "temporal"
    page = TestClient(app).get("/")
    assert page.status_code == 200
    assert "median_x3" in page.text
    assert "spatial_z_halo" in page.text
    assert body["manifest"]["clean"] == "median_x3"
    assert body["manifest"]["detector"] == "spatial_z_halo"
    assert body["manifest"]["config"]["threshold_z"] == 3.5
    assert body["manifest"]["trained_classifier_operational"] is False
    assert any(math.isfinite(zone["contrast"]) and zone["contrast"] > 0 for zone in body["zones"])
    assert "/view/windows" in page.text
    assert "next-0010" in page.text
    assert "baseline kept no zone" in page.text


def test_report_matches_zone_catalog():
    report = (ROOT / "REPORT.md").read_text(encoding="utf-8")
    catalog = json.loads((ROOT / "docs" / "evidence" / "zone_catalog.json").read_text(encoding="utf-8"))
    assert "## Задача" in report
    assert "## Методы" in report
    assert "## Найденные аномалии" in report
    assert "## Ограничения" in report
    assert "hybrid refine не принят" in report
    assert "temporal_innovation" in report
    assert "не ставит anomaly_status" in report
    assert "serve=research" in report
    assert "expert_confirmed" in report and "false" in report
    assert "operational" in report
    assert "median_x3" in report and "spatial_z_halo" in report
    assert "docker compose -f compose.viewer.yaml up --build -d" in report
    assert catalog["status_source"] == "spatial_z_halo"
    assert catalog["temporal_sets_status"] is False
    for name, block in catalog["files"].items():
        needle = f"| {name} | {block['n_zones']} |"
        assert needle in report
        assert block["n_zones"] == block["n_rows"]
        print(f"file {name} report_zones {block['n_zones']} catalog_n_zones {block['n_zones']} match true")
    print("hybrid_refine_not_adopted true")
    print("temporal_innovation_does_not_set_status true")
    print("research_serve research")
    print("viewer_path operational")

    research = json.loads((ROOT / "docs" / "evidence" / "research_cnn_report.json").read_text(encoding="utf-8"))
    spatial = json.loads((ROOT / "docs" / "evidence" / "spatial_vs_expert.json").read_text(encoding="utf-8"))
    cards = _section(report, "## 15 экспертных карточек")
    assert "serve=research" in cards
    assert "не операционный детектор" in cards
    assert len(research["cards"]) == 15
    cooling = [card for card in research["cards"] if card["expert_class"] == "Local_Cooling"]
    hard_negative = [card for card in research["cards"] if card["presence"] == "no"]
    assert len(cooling) == 2
    assert len(hard_negative) == 4
    for card in research["cards"]:
        line = next(line for line in cards.splitlines() if f"`{card['id']}`" in line)
        assert card["expert_class"] in line
        assert card["network_class"] in line
        print(f"card {card['id']} expert {card['expert_class']} network {card['network_class']}")
    contrast = _section(report, "## норма vs контраст")
    assert "Медиана температуры `presence=no` не переписывает." in contrast
    assert "median_x3" in contrast and "spatial_z_halo" in contrast and "|Z| >= 3.5" in contrast
    by_spatial = {card["id"]: card for card in spatial["cards"]}
    for card_id in ("next-0010", "next-0006", "next-0003", "next-0005", "next-0007"):
        line = next(line for line in contrast.splitlines() if f"`{card_id}`" in line)
        card = by_spatial[card_id]
        assert card["expert_class"] in line
        if card["n_in_box"] == 0:
            assert "зону не оставил" in line
        for zone in card["in_box"]:
            assert _contrast_token(zone["contrast"]) in line
        if card_id == "next-0006":
            assert "в норме" in line
            assert card["expert_class"] == "Background"
            for zone in card["left_flank_only"] + card["right_flank_only"]:
                assert _contrast_token(zone["contrast"]) in line
        if card_id == "next-0010":
            assert "presence=no" in line
            assert card["n_in_box"] > 0
        if card_id == "next-0003":
            assert card["expert_class"] == "Local_Heating"
            assert "Local_Heating" in line

    context = json.loads((ROOT / "docs" / "evidence" / "context_evidence.json").read_text(encoding="utf-8"))
    halo = _section(report, "## Ограничение halo на широком участке")
    assert "\n## " not in halo
    assert "## Ограничение halo" not in cards
    assert "## Ограничение halo" not in contrast
    assert context["schema"] == "context_evidence_v1"
    assert context["serve"] == "research"
    assert context["sets_anomaly_status"] is False
    assert context["operational"] is False
    assert context["clean"] == "median_x3"
    assert context["detector"] == "spatial_z_halo"
    assert context["threshold_z"] == 3.5
    assert all(card["expert_confirmed"] is False for card in context["cards"])
    assert "кольц" in halo and re.search(r"8\s*[–-]\s*40", halo)
    assert "serve=research" in halo
    assert "median_x3" in halo and "spatial_z_halo" in halo and "|Z| >= 3.5" in halo
    assert "не ставит anomaly_status" in halo
    assert "не операционная точность" in halo
    assert "expert_confirmed" in halo and "false" in halo
    assert "не подтверждает" in halo
    assert "docs/evidence/context_evidence.json" in halo
    assert "docs/evidence/spatial_vs_expert.json" in halo
    by_context = {card["id"]: card for card in context["cards"]}
    for card_id in ("next-0003", "next-0006", "next-0010"):
        line = next(row for row in halo.splitlines() if f"`{card_id}`" in row)
        spatial_card = by_spatial[card_id]
        context_card = by_context[card_id]
        width, width_places = _printed(line, r"ширина\s+(-?\d+(?:\.\d+)?)\s*м")
        rectangle, rectangle_places = _printed(line, r"медиана прямоугольника\s+(-?\d+(?:\.\d+)?)")
        far_cable, far_places = _printed(line, r"медиана кабеля дальше 40 м\s+(-?\d+(?:\.\d+)?)")
        fraction, fraction_places = _printed(line, r"доля кольца внутри\s+(-?\d+(?:\.\d+)?)")
        _same_at_precision(spatial_card["width_m"], width, width_places)
        _same_at_precision(context_card["rectangle_median_c"], rectangle, rectangle_places)
        _same_at_precision(context_card["far_cable_median_c"], far_cable, far_places)
        _same_at_precision(context_card["halo_inside_fraction"], fraction, fraction_places)
        _same_at_precision(spatial_card["rectangle"]["median_clean_c"], rectangle, rectangle_places)
        _same_at_precision(spatial_card["halo_inside_fraction"], fraction, fraction_places)
        assert spatial_card["expert_class"] in line
        if spatial_card["n_in_box"] == 0:
            assert "В ящике зон нет" in line
            assert context_card["in_box_zones"] == []
        for zone in spatial_card["in_box"]:
            assert _contrast_token(zone["contrast"]) in line
        for zone in spatial_card["left_flank_only"] + spatial_card["right_flank_only"]:
            assert _contrast_token(zone["contrast"]) in line
        if card_id == "next-0003":
            assert width > 40
            assert "шире 40" in line
            assert spatial_card["expert_class"] == "Local_Heating"
            assert "остаётся `Local_Heating`" in line
            residual, residual_places = _printed(line, r"остаток против фона кольца\s+(-?\d+(?:\.\d+)?)")
            left, left_places = _printed(line, r"дальний левый\s+(-?\d+(?:\.\d+)?)")
            right, right_places = _printed(line, r"дальний правый\s+(-?\d+(?:\.\d+)?)")
            _same_at_precision(spatial_card["rectangle"]["median_residual_c"], residual, residual_places)
            _same_at_precision(context_card["far_left"]["median_c"], left, left_places)
            _same_at_precision(spatial_card["far_left"]["median_clean_c"], left, left_places)
            _same_at_precision(context_card["far_right"]["median_c"], right, right_places)
            _same_at_precision(spatial_card["far_right"]["median_clean_c"], right, right_places)
        if card_id == "next-0006":
            assert spatial_card["presence"] == "no"
            assert spatial_card["expert_class"] == "Background"
            assert "presence=no" in line and "Background" in line
            assert "в норме" in line
            residual, residual_places = _printed(line, r"остаток против фона кольца\s+(-?\d+(?:\.\d+)?)")
            _same_at_precision(spatial_card["rectangle"]["median_residual_c"], residual, residual_places)
        if card_id == "next-0010":
            assert spatial_card["presence"] == "no"
            assert spatial_card["expert_class"] == "Background"
            assert "presence=no" in line and "Background" in line
            assert "не становится классом по температуре" in line


def test_windows_route_matches_evidence_files():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from fiber.viewer_app import app

    research = json.loads((ROOT / "docs" / "evidence" / "research_cnn_report.json").read_text(encoding="utf-8"))
    spatial = json.loads((ROOT / "docs" / "evidence" / "spatial_vs_expert.json").read_text(encoding="utf-8"))
    context_net = json.loads((ROOT / "docs" / "evidence" / "research_context_cnn_report.json").read_text(encoding="utf-8"))
    client = TestClient(app)
    first = client.get("/view/windows")
    second = client.get("/view/windows")
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()
    body = first.json()
    assert body["serve"] == "research"
    assert body["operational"] is False
    assert body["trained_classifier_operational"] is False
    windows = body["windows"]
    residual_ids = [card["id"] for card in research["cards"]]
    ask_cards = [card for card in context_net["cards"] if card["source"] == "ask_expert_seed"]
    assert [row["id"] for row in windows[: len(residual_ids)]] == residual_ids
    assert [row["id"] for row in windows[len(residual_ids) :]] == [card["id"] for card in ask_cards]
    assert len(windows) == len({row["id"] for row in windows})
    assert len(windows) >= 3
    spatial_by_id = {card["id"]: card for card in spatial["cards"]}
    research_by_id = {card["id"]: card for card in research["cards"]}
    for row in windows[: len(residual_ids)]:
        card = research_by_id[row["id"]]
        assert row["expert_class"] == card["expert_class"]
        assert row["network_class"] == card["network_class"]
        assert row["confirmed"] is (card["expert_class"] == card["network_class"])
        assert row["verdict"] == ("confirmed" if row["confirmed"] else "refuted")
        assert row["serve"] == "research"
        assert row["operational"] is False
        if row["id"] in spatial_by_id:
            expected = [float(zone["contrast"]) for zone in spatial_by_id[row["id"]]["in_box"]]
            assert row["baseline_source"] == "spatial_vs_expert"
        elif card["baseline_overlaps"]:
            expected = [float(zone["contrast_c"]) for zone in card["baseline_zones"]]
            assert row["baseline_source"] == "research_cnn_report"
        else:
            expected = []
            assert row["baseline_source"] == "research_cnn_report"
        assert row["baseline_contrasts_c"] == expected
        assert row["baseline_overlaps"] is bool(expected)
        if expected:
            for value in expected:
                assert _contrast_token(value) in row["baseline_text"]
        else:
            assert row["baseline_text"] == "baseline kept no zone"
    for row, card in zip(windows[len(residual_ids) :], ask_cards, strict=True):
        assert row["expert_class"] == card["expert_class"]
        assert row["network_class"] == card["network_class"]
        assert row["confirmed"] is (card["expert_class"] == card["network_class"])
        assert row["verdict"] == ("confirmed" if row["confirmed"] else "refuted")
        assert row["serve"] == "research"
        assert row["operational"] is False
        assert row["baseline_contrasts_c"] == []
        assert row["baseline_overlaps"] is False
        assert row["baseline_text"] == "baseline kept no zone"
        assert row["baseline_source"] == "research_context_cnn_report"
        assert row["rectangle_median_c"] == card["rectangle_median_c"]
        assert row["far_cable_median_c"] == card["far_cable_median_c"]
        assert row["halo_inside_fraction"] == card["halo_inside_fraction"]
        if card["far_cable_median_c"] is None:
            assert row["far_cable_median_c"] is None
            assert row["far_cable_median_c"] != row["rectangle_median_c"]
    assert any(row["confirmed"] for row in windows)
    assert any(not row["confirmed"] for row in windows)
    shown = next(row for row in windows if row["id"] == "next-0010")
    assert shown["confirmed"] is False
    assert shown["baseline_contrasts_c"] == [float(zone["contrast"]) for zone in spatial_by_id["next-0010"]["in_box"]]
    assert shown["baseline_contrasts_c"]


def _html_rows(page: str, card_id: str) -> list[str]:
    rows = []
    for row in page.split("<tr>"):
        head, _, _ = row.partition("</tr>")
        if f">{card_id}<" in head:
            rows.append(head)
    return rows


def test_report_and_viewer_context_and_wide_channel(monkeypatch):
    """Report prose and the research display follow the stored evidence files."""
    import inspect

    from fiber.viewer import expert_windows

    report = (ROOT / "REPORT.md").read_text(encoding="utf-8")
    residual = json.loads((ROOT / "docs" / "evidence" / "research_cnn_report.json").read_text(encoding="utf-8"))
    context_net = json.loads((ROOT / "docs" / "evidence" / "research_context_cnn_report.json").read_text(encoding="utf-8"))
    evidence = json.loads((ROOT / "docs" / "evidence" / "context_evidence.json").read_text(encoding="utf-8"))
    wide = json.loads((ROOT / "docs" / "evidence" / "wide_channel.json").read_text(encoding="utf-8"))
    assert residual["match_count"] == 9
    assert residual["previous_match_label"] == "10/15"
    assert len(residual["cards"]) == 15

    cards = _section(report, "## 15 экспертных карточек")
    assert f"{residual['match_count']}/{len(residual['cards'])}" in cards
    assert residual["previous_match_label"] in cards
    assert "serve=research" in cards
    assert "15/22" not in cards
    for card in residual["cards"]:
        line = next(row for row in cards.splitlines() if f"`{card['id']}`" in row)
        assert card["expert_class"] in line
        assert card["network_class"] in line

    statement = _section(report, "## Контекстная research-сеть")
    assert context_net["context_match_label"] in statement
    assert context_net["expert_seed_match_label"] in statement
    assert context_net["ask_seed_match_label"] in statement
    assert context_net["published_match_label"] in statement
    assert context_net["previous_match_label"] in statement
    assert context_net["historical_context_match_label"] == "15/22"
    assert context_net["context_match_label"] == "15/22"
    assert context_net["previous_match_label"] == "10/15"
    assert context_net["published_match_label"] == "9/15"
    ask_cards = [card for card in context_net["cards"] if card["source"] == "ask_expert_seed"]
    seed_cards = [card for card in context_net["cards"] if card["source"] == "expert_seed"]
    assert len(seed_cards) == 15
    assert ask_cards
    assert context_net["ask_seed_match_label"] == (
        f"{sum(bool(card['match']) for card in ask_cards)}/{len(ask_cards)}"
    )
    assert context_net["expert_seed_match_label"] == (
        f"{sum(bool(card['match']) for card in seed_cards)}/{len(seed_cards)}"
    )
    assert context_net["weights_sha256"] in statement
    assert context_net["weights_sha256"] != "144dea50ec6aaad4985afb1bbab0fdb4275b9dc89a094cd9403fa04ab31e4216"
    assert "serve=research" in statement
    assert "operational false" in statement
    assert "не были fit seeds" in statement
    assert context_net["gap_cards_in_fit"] is False
    assert context_net["unmapped_in_match_count"] is False
    bullets = [row for row in statement.splitlines() if row.startswith("- `")]
    assert [row.split("`")[1] for row in bullets] == [card["id"] for card in context_net["cards"]]
    assert len(context_net["cards"]) > 15
    for card, line in zip(context_net["cards"], bullets, strict=True):
        assert card["expert_class"] in line
        assert card["network_class"] in line
        if card["id"] == "next-0003":
            assert card["expert_class"] == "Local_Heating"
            assert "остаётся `Local_Heating`" in line
        if card["presence"] == "no":
            assert card["expert_class"] == "Background"
            assert "presence=no" in line and "остаётся `Background`" in line
    for unmapped in context_net["unmapped_gap_ids"]:
        assert unmapped not in [row.split("`")[1] for row in bullets]
        assert f"`{unmapped}`" in statement
    assert "не совпадение и не промах" in statement

    residual_by = {card["id"]: card for card in residual["cards"]}
    context_by = {card["id"]: card for card in context_net["cards"]}
    evidence_by = {card["id"]: card for card in evidence["cards"]}
    diff = _section(report, "## Расхождения и context evidence")
    assert "не заполняется медианой прямоугольника" in diff
    assert "не ставят anomaly_status" in diff
    assert "не являются операционной точностью" in diff
    assert "median_x3" in diff and "spatial_z_halo" in diff and "|Z| >= 3.5" in diff
    assert "`presence=no` остаётся `Background`" in diff
    covered = []
    for card in residual["cards"]:
        other = context_by[card["id"]]
        disagrees = card["network_class"] != card["expert_class"] or other["network_class"] != card["expert_class"]
        if not disagrees:
            assert f"`{card['id']}`" not in diff
            continue
        covered.append(card["id"])
        line = next(row for row in diff.splitlines() if f"`{card['id']}`" in row)
        assert f"эксперт `{card['expert_class']}`" in line
        assert f"Residual-сеть `{card['network_class']}`" in line
        assert f"Контекстная сеть `{other['network_class']}`" in line
        assert "промах" in line.lower()
        if card["network_class"] != other["network_class"]:
            assert "разошлись" in line
            assert f"residual сказал `{card['network_class']}`" in line
            assert f"контекстная сеть сказала `{other['network_class']}`" in line
        if card["presence"] == "no":
            assert "presence=no" in line
            assert "остаётся `Background`" in line
            assert card["expert_class"] == "Background"
        measured = evidence_by[card["id"]]
        rectangle, rectangle_places = _printed(line, r"медиана прямоугольника\s+(-?\d+(?:\.\d+)?)")
        fraction, fraction_places = _printed(line, r"доля кольца внутри\s+(-?\d+(?:\.\d+)?)")
        _same_at_precision(measured["rectangle_median_c"], rectangle, rectangle_places)
        _same_at_precision(measured["halo_inside_fraction"], fraction, fraction_places)
        if measured["far_cable_median_c"] is None:
            assert "отсутств" in line.lower()
            assert f"медиана кабеля дальше 40 м {rectangle}" not in line
        else:
            far_cable, far_places = _printed(line, r"медиана кабеля дальше 40 м\s+(-?\d+(?:\.\d+)?)")
            _same_at_precision(measured["far_cable_median_c"], far_cable, far_places)
            if far_cable == rectangle and measured["far_cable_median_c"] != measured["rectangle_median_c"]:
                raise AssertionError(card["id"])
    assert covered

    wide_section = _section(report, "## Широкий канал")
    assert "## Ограничение halo" not in wide_section
    assert "## 15 экспертных карточек" not in wide_section
    assert "serve=research" in wide_section and "operational false" in wide_section
    assert "не меньше 0.70" in wide_section and "не меньше 2.0" in wide_section
    assert "одни" in wide_section or "одно на все" in wide_section
    assert "не копируется из медианы прямоугольника" in wide_section
    assert "не седьмой класс" in wide_section
    assert "не ставит anomaly_status" in wide_section
    assert "не ставит expert_confirmed" in wide_section
    assert "median_x3" in wide_section and "spatial_z_halo" in wide_section and "|Z| >= 3.5" in wide_section
    cuts = re.findall(r"не меньше\s+(-?\d+(?:\.\d+)?)", wide_section)
    assert len(cuts) >= 2
    halo_token, gap_token = cuts[0], cuts[1]
    _same_at_precision(wide["wide_channel_halo_inside_min"], float(halo_token), len(halo_token.split(".")[1]))
    _same_at_precision(wide["wide_channel_far_gap_min_c"], float(gap_token), len(gap_token.split(".")[1]))
    wide_lines = [row for row in wide_section.splitlines() if row.startswith("- `")]
    assert [row.split("`")[1] for row in wide_lines] == [card["id"] for card in wide["cards"]]
    assert any(card["wide_channel"] for card in wide["cards"])
    assert any(not card["wide_channel"] for card in wide["cards"])
    assert any(card["expert_class"] is None for card in wide["cards"])
    for card, line in zip(wide["cards"], wide_lines, strict=True):
        assert ("флаг true" if card["wide_channel"] else "флаг false") in line
        if card["expert_class"] is None:
            assert "Эксперт null" in line
        else:
            assert f"эксперт `{card['expert_class']}`" in line
        assert "эксперт `Broad_Shift`" not in line
        if card["wide_channel_candidate"]:
            assert f"Кандидат `{card['wide_channel_candidate']}`" in line
            assert card["wide_channel"] is True
        else:
            assert card["wide_channel"] is False
            assert "Broad_Shift" not in line
            assert "Кандидата нет" in line
        if card["id"] == "next-0003":
            assert card["expert_class"] == "Local_Heating"
            assert "остаётся `Local_Heating`" in line
        rectangle, rectangle_places = _printed(line, r"медиана прямоугольника\s+(-?\d+(?:\.\d+)?)")
        fraction, fraction_places = _printed(line, r"доля кольца внутри\s+(-?\d+(?:\.\d+)?)")
        gap, gap_places = _printed(line, r"разрыв медиан\s+(-?\d+(?:\.\d+)?)")
        _same_at_precision(card["rectangle_median_c"], rectangle, rectangle_places)
        _same_at_precision(card["halo_inside_fraction"], fraction, fraction_places)
        _same_at_precision(card["far_gap_c"], gap, gap_places)
        if card["far_cable_median_c"] is None:
            assert "отсутств" in line.lower()
        else:
            far_cable, far_places = _printed(line, r"медиана кабеля дальше 40 м\s+(-?\d+(?:\.\d+)?)")
            _same_at_precision(card["far_cable_median_c"], far_cable, far_places)
        side_missing = card["far_left"]["present"] is False or card["far_right"]["present"] is False
        if side_missing:
            assert "не скопирована из прямоугольника" in line
    print(f"context_match {context_net['context_match_label']} seeds {context_net['expert_seed_match_label']}")
    print(f"published {context_net['published_match_label']} previous {context_net['previous_match_label']}")
    print(f"disagreements {len(covered)} wide_cards {len(wide['cards'])}")

    source = inspect.getsource(expert_windows)
    assert "analyze_field(" not in source
    assert ".pt" not in source
    assert "load_research" not in source
    real_open = open

    def guard(path, *args, **kwargs):
        name = str(path).replace("\\", "/")
        if name.endswith(".pt") or "/models/weights/" in name or "research_residual_cnn" in name:
            raise AssertionError(name)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", guard)
    payload = expert_windows()
    assert payload["serve"] == "research"
    assert payload["operational"] is False
    assert payload["trained_classifier_operational"] is False
    residual_ids = [card["id"] for card in residual["cards"]]
    window_ids = [row["id"] for row in payload["windows"]]
    assert window_ids[: len(residual_ids)] == residual_ids
    assert window_ids[len(residual_ids) :] == [card["id"] for card in ask_cards]
    for row in payload["windows"][: len(residual_ids)]:
        card = residual_by[row["id"]]
        measured = evidence_by[row["id"]]
        assert row["expert_class"] == card["expert_class"]
        assert row["network_class"] == card["network_class"]
        assert row["rectangle_median_c"] == measured["rectangle_median_c"]
        assert row["far_cable_median_c"] == measured["far_cable_median_c"]
        assert row["halo_inside_fraction"] == measured["halo_inside_fraction"]
        assert row["serve"] == "research"
        assert row["operational"] is False
        if measured["far_cable_median_c"] is None:
            assert row["far_cable_median_c"] is None
            assert row["far_cable_median_c"] != row["rectangle_median_c"]
    assert [row["id"] for row in payload["wide_channels"]] == [card["id"] for card in wide["cards"]]
    for row, card in zip(payload["wide_channels"], wide["cards"], strict=True):
        assert row["wide_channel"] is card["wide_channel"]
        assert row["wide_channel_candidate"] == card["wide_channel_candidate"]
        assert row["expert_class"] == card["expert_class"]
        assert row["rectangle_median_c"] == card["rectangle_median_c"]
        assert row["far_cable_median_c"] == card["far_cable_median_c"]
        assert row["halo_inside_fraction"] == card["halo_inside_fraction"]
        assert row["far_gap_c"] == card["far_gap_c"]
        assert row["serve"] == "research"
        assert row["operational"] is False

    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from fiber.viewer_app import app

    monkeypatch.undo()
    client = TestClient(app)
    example_first = client.get("/view/example").json()
    example_second = client.get("/view/example").json()
    assert example_first == example_second
    assert example_first["manifest"]["clean"] == "median_x3"
    assert example_first["manifest"]["detector"] == "spatial_z_halo"
    assert example_first["manifest"]["config"]["threshold_z"] == 3.5
    assert example_first["manifest"]["trained_classifier_operational"] is False
    display_first = client.get("/view/windows").json()
    display_second = client.get("/view/windows").json()
    assert display_first == display_second == payload
    page = client.get("/").text
    assert "serve=research" in page
    assert "operational false" in page
    assert "median_x3" in page and "spatial_z_halo" in page
    assert "__WINDOWS__" not in page and "__WIDE__" not in page
    for row in payload["windows"]:
        html_row = next(item for item in _html_rows(page, row["id"]) if "wide_channel" not in item)
        assert row["expert_class"] in html_row
        assert row["network_class"] in html_row
        assert "rectangle_median_c " + json.dumps(row["rectangle_median_c"]) in html_row
        far_text = "absent" if row["far_cable_median_c"] is None else json.dumps(row["far_cable_median_c"])
        assert "far_cable_median_c " + far_text in html_row
        assert "halo_inside_fraction " + json.dumps(row["halo_inside_fraction"]) in html_row
    for row in payload["wide_channels"]:
        html_row = next(item for item in _html_rows(page, row["id"]) if "wide_channel" in item)
        flag = "wide_channel true" if row["wide_channel"] else "wide_channel false"
        assert flag in html_row
        if row["wide_channel_candidate"] is None:
            assert "candidate absent" in html_row
            assert "Broad_Shift" not in html_row
        else:
            assert "candidate " + row["wide_channel_candidate"] in html_row
        if row["expert_class"] is None:
            assert "expert null" in html_row
        else:
            assert "expert " + row["expert_class"] in html_row
        assert "rectangle_median_c " + json.dumps(row["rectangle_median_c"]) in html_row
        far_text = "absent" if row["far_cable_median_c"] is None else json.dumps(row["far_cable_median_c"])
        assert "far_cable_median_c " + far_text in html_row
        assert "halo_inside_fraction " + json.dumps(row["halo_inside_fraction"]) in html_row


def test_ask_answer_section_and_windows(monkeypatch):
    """The new REPORT section and /view/windows follow the ask answers and the context report."""
    import inspect

    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from fiber.viewer import _stored_number, expert_windows
    from fiber.viewer_app import app, index, view_windows

    report = (ROOT / "REPORT.md").read_text(encoding="utf-8")
    for heading in (
        "## Задача",
        "## Методы",
        "## Найденные аномалии",
        "## 15 экспертных карточек",
        "## норма vs контраст",
        "## Ограничение halo на широком участке",
        "## Контекстная research-сеть",
        "## Расхождения и context evidence",
        "## Широкий канал",
        "## Ограничения",
        "## Viewer",
    ):
        assert heading in report
    assert report.count("## Около 150 ответов ask-shortlist") == 1
    section = _section(report, "## Около 150 ответов ask-shortlist")
    assert "## " not in section

    answers = [
        json.loads(line)
        for line in (ROOT / "docs" / "evidence" / "zakhar_ask_shortlist_answers.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = json.loads((ROOT / "docs" / "evidence" / "expert_seeds_ask_summary.json").read_text(encoding="utf-8"))
    context_net = json.loads((ROOT / "docs" / "evidence" / "research_context_cnn_report.json").read_text(encoding="utf-8"))
    residual = json.loads((ROOT / "docs" / "evidence" / "research_cnn_report.json").read_text(encoding="utf-8"))
    assert len(answers) == summary["n_answers"]
    assert str(len(answers)) in section
    unclear = [row for row in answers if row["presence"] == "unclear"]
    assert len(unclear) == summary["n_unclear"]
    assert str(len(unclear)) in section
    assert "presence=unclear" in section
    assert "не совпадение и не промах" in section
    kept = [card for card in summary["cards"] if card["status"] == "kept"]
    assert kept
    assert all(card["weight"] == 10 for card in kept)
    assert all(card["class_name"] is None and card["weight"] is None for card in summary["cards"] if card["status"] == "unclear")
    assert len(kept) == summary["n_kept"]
    assert str(len(kept)) in section
    assert "веса 10" in section
    ask_cards = [card for card in context_net["cards"] if card["source"] == "ask_expert_seed"]
    seed_cards = [card for card in context_net["cards"] if card["source"] == "expert_seed"]
    assert len(ask_cards) == len(kept)
    assert {card["id"] for card in ask_cards} == {card["id"] for card in kept}
    assert context_net["ask_seed_match_label"] == f"{sum(bool(card['match']) for card in ask_cards)}/{len(ask_cards)}"
    assert context_net["expert_seed_match_label"] == f"{sum(bool(card['match']) for card in seed_cards)}/{len(seed_cards)}"
    assert context_net["previous_match_label"] == "10/15"
    assert context_net["historical_context_match_label"] == "15/22"
    for key in (
        "previous_match_label",
        "historical_context_match_label",
        "expert_seed_match_label",
        "ask_seed_match_label",
    ):
        assert context_net[key] in section
    assert context_net["ask_recovery_is_in_sample"] is True
    assert "in-sample" in section
    assert "не является операционной точностью" in section
    assert "Промах остаётся промахом" in section
    assert context_net["serve"] == "research" and context_net["operational"] is False
    assert "serve=research" in section and "operational false" in section
    assert "не менялись" in section
    next_0003 = next(card for card in seed_cards if card["id"] == "next-0003")
    assert next_0003["expert_class"] == "Local_Heating"
    next_line = next(row for row in section.splitlines() if "next-0003" in row)
    assert f"остаётся экспертом `{next_0003['expert_class']}`" in next_line
    assert any(card["presence"] == "no" for card in ask_cards)
    assert all(card["expert_class"] == "Background" for card in ask_cards if card["presence"] == "no")
    assert "`presence=no` остаётся `Background`" in section
    misses = Counter(
        (card["expert_class"], card["network_class"])
        for card in ask_cards
        if not card["match"]
    )
    assert misses
    parsed = Counter()
    for expert, network, count in re.findall(r"`([A-Za-z_]+)` → `([A-Za-z_]+)` (\d+)", section):
        parsed[(expert, network)] += int(count)
    assert parsed == misses
    for card_id in (card["id"] for card in seed_cards if card["match"]):
        assert f"`{card_id}`" in section
    for row in unclear:
        assert row["id"] not in section

    weak_sha = hashlib.sha256((ROOT / "docs" / "evidence" / "weak_labels.jsonl").read_bytes()).hexdigest()
    residual_sha = hashlib.sha256((ROOT / "docs" / "evidence" / "research_cnn_report.json").read_bytes()).hexdigest()
    assert weak_sha == "48acffcc46679f465cbf6ba167043b02246666afc13c7491a1b3be7239846c57"
    assert residual_sha == "2c05c26d555fc52f4d372af5c818ad11697c88aa8b27f906836b5cf29f2a93fc"

    display_source = inspect.getsource(expert_windows)
    assert "analyze_field(" not in display_source
    assert ".pt" not in display_source
    assert "load_research" not in display_source
    assert "_stored_number(card[\"far_cable_median_c\"])" in display_source
    assert "analyze_field(" not in inspect.getsource(view_windows)
    assert ".pt" not in inspect.getsource(view_windows)
    assert "analyze_field(" not in inspect.getsource(index)
    assert _stored_number(None) is None
    real_open = open

    def guard(path, *args, **kwargs):
        name = str(path).replace("\\", "/")
        if name.endswith(".pt") or "/models/weights/" in name:
            raise AssertionError(name)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", guard)
    built = expert_windows()
    monkeypatch.undo()

    client = TestClient(app)
    example = client.get("/view/example").json()
    assert example["manifest"]["clean"] == "median_x3"
    assert example["manifest"]["detector"] == "spatial_z_halo"
    assert example["manifest"]["config"]["threshold_z"] == 3.5
    assert example["manifest"]["trained_classifier_operational"] is False
    first = client.get("/view/windows")
    second = client.get("/view/windows")
    assert first.status_code == 200
    assert first.json() == second.json() == built
    body = first.json()
    assert body["serve"] == "research"
    assert body["operational"] is not True
    assert body["trained_classifier_operational"] is False
    assert body["detector"] == "spatial_z_halo"
    windows = body["windows"]
    residual_ids = [card["id"] for card in residual["cards"]]
    assert [row["id"] for row in windows[: len(residual_ids)]] == residual_ids
    assert [row["id"] for row in windows[len(residual_ids) :]] == [card["id"] for card in ask_cards]
    assert len(windows) == len({row["id"] for row in windows})
    assert any(row["id"] == "next-0010" for row in windows[: len(residual_ids)])
    unclear_ids = {row["id"] for row in unclear}
    assert unclear_ids.isdisjoint(row["id"] for row in windows)
    page = client.get("/").text
    assert "next-0010" in page
    for card in ask_cards:
        matches = [row for row in windows if row["id"] == card["id"]]
        assert len(matches) == 1
        row = matches[0]
        assert row["expert_class"] == card["expert_class"]
        assert row["network_class"] == card["network_class"]
        assert row["rectangle_median_c"] == card["rectangle_median_c"]
        assert row["far_cable_median_c"] == card["far_cable_median_c"]
        assert row["halo_inside_fraction"] == card["halo_inside_fraction"]
        if card["far_cable_median_c"] is None:
            assert row["far_cable_median_c"] is None
            assert row["far_cable_median_c"] != row["rectangle_median_c"]
        html_rows = [item for item in _html_rows(page, card["id"]) if "wide_channel" not in item]
        assert len(html_rows) == 1
        html_row = html_rows[0]
        assert card["expert_class"] in html_row
        assert card["network_class"] in html_row
        assert "rectangle_median_c " + json.dumps(row["rectangle_median_c"]) in html_row
        far_text = "absent" if row["far_cable_median_c"] is None else json.dumps(row["far_cable_median_c"])
        assert "far_cable_median_c " + far_text in html_row
        if row["far_cable_median_c"] is None:
            assert "far_cable_median_c " + json.dumps(row["rectangle_median_c"]) not in html_row
        assert "halo_inside_fraction " + json.dumps(row["halo_inside_fraction"]) in html_row
