"""The research context channel reads analyze_field and does not change it."""
import math

import numpy as np

from fiber.classify.rules import LABELS
from fiber.pipeline import PipelineConfig, analyze_field
from fiber.research_context import (
    WIDE_CHANNEL_CANDIDATE,
    WIDE_CHANNEL_FAR_GAP_C,
    WIDE_CHANNEL_HALO_INSIDE_MIN,
    context_evidence,
    wide_channel_from_measurements,
)


def _snapshot(found):
    return {
        "anomaly_status": found["anomaly_status"],
        "n_zones": len(found["zones"]),
        "zones": [
            (zone["start_m"], zone["end_m"], zone["contrast"], zone["expert_confirmed"])
            for zone in found["zones"]
        ],
    }


def _unchanged(found, before):
    assert found["anomaly_status"] == before["anomaly_status"]
    assert len(found["zones"]) == before["n_zones"]
    for zone, saved in zip(found["zones"], before["zones"], strict=True):
        assert (zone["start_m"], zone["end_m"], zone["contrast"], zone["expert_confirmed"]) == saved
        assert zone["expert_confirmed"] is False
        assert "context" not in zone


def _assert_flag_matches_reader(row):
    again = wide_channel_from_measurements(
        row["halo_inside_fraction"],
        row["rectangle_median_c"],
        row["far_cable_median_c"],
    )
    assert row["wide_channel"] is again["wide_channel"]
    assert row["wide_channel_candidate"] == again["wide_channel_candidate"]
    assert row["far_gap_c"] == again["far_gap_c"]
    assert row["wide_channel_halo_inside_min"] == WIDE_CHANNEL_HALO_INSIDE_MIN
    assert row["wide_channel_far_gap_min_c"] == WIDE_CHANNEL_FAR_GAP_C
    assert row["wide_channel"] is (row["wide_channel_candidate"] == WIDE_CHANNEL_CANDIDATE)
    if row["wide_channel"]:
        assert row["wide_channel_candidate"] == "Broad_Shift"
        assert row["halo_inside_fraction"] >= row["wide_channel_halo_inside_min"]
        assert row["far_gap_c"] is not None
        assert abs(row["far_gap_c"]) >= row["wide_channel_far_gap_min_c"]
    else:
        assert row["wide_channel_candidate"] is None


def _assert_channel(row, x0, x1):
    assert math.isfinite(row["rectangle_median_c"])
    assert math.isfinite(row["halo_inside_fraction"])
    assert 0.0 <= row["halo_inside_fraction"] <= 1.0
    for side in (row["far_left"], row["far_right"]):
        assert "present" in side and "median_c" in side and "n_columns" in side
        if not side["present"]:
            assert side["median_c"] is None
            assert side["n_columns"] == 0
            assert side["median_c"] != row["rectangle_median_c"]
    if row["far_left"]["present"] or row["far_right"]["present"]:
        assert row["far_cable_median_c"] is not None
        assert math.isfinite(row["far_cable_median_c"])
    else:
        assert row["far_cable_median_c"] is None
    for zone in row["flank_zones"]:
        assert zone["end_m"] < x0 or zone["start_m"] > x1
        assert zone["expert_confirmed"] is False


def test_context_evidence_on_wide_span_and_hot_flank():
    assert PipelineConfig().threshold_z == 3.5
    assert PipelineConfig().inner_m == 8.0
    assert PipelineConfig().outer_m == 40.0
    assert len(LABELS) == 6
    assert WIDE_CHANNEL_CANDIDATE in LABELS

    wide_raw = np.full((12, 261), 20.0, dtype=np.float32)
    wide_raw[:, 30:151] += 8.0
    wide_x = np.arange(261, dtype=np.float64)
    wide_t = np.arange(12, dtype=np.float64) * 60.0
    wide = analyze_field(wide_raw, wide_x, wide_t)
    wide_before = _snapshot(wide)
    wide_row = context_evidence(
        wide, wide_x, wide_t, rows_half_open=(0, 12), x_span=(30.0, 150.0),
    )
    _unchanged(wide, wide_before)
    _assert_channel(wide_row, 30.0, 150.0)
    _assert_flag_matches_reader(wide_row)
    assert wide_row["halo_inside_fraction"] == 1.0
    assert wide_row["far_left"]["present"] is False
    assert wide_row["far_right"]["present"] is True
    assert wide_row["far_right"]["median_c"] < wide_row["rectangle_median_c"]
    assert wide_row["rectangle_median_c"] > 27.0
    assert wide_row["wide_channel"] is True
    assert wide_row["wide_channel_candidate"] == "Broad_Shift"
    assert wide["anomaly_status"] == wide_before["anomaly_status"]

    raw = np.full((12, 100), 22.0, dtype=np.float32)
    raw[:, 45:51] += 6.0
    x_m = np.arange(100, dtype=np.float64)
    time_s = np.arange(12, dtype=np.float64) * 60.0
    found = analyze_field(raw, x_m, time_s)
    before = _snapshot(found)
    assert found["anomaly_status"] == "Anomaly"
    assert len(found["zones"]) >= 1
    narrow = context_evidence(
        found, x_m, time_s, rows_half_open=(0, 12), x_span=(72.0, 78.0),
    )
    _unchanged(found, before)
    _assert_channel(narrow, 72.0, 78.0)
    _assert_flag_matches_reader(narrow)
    assert narrow["halo_inside_fraction"] == 0.0
    assert narrow["halo_inside_fraction"] < wide_row["halo_inside_fraction"]
    assert narrow["far_right"]["present"] is False
    assert narrow["in_box_zones"] == []
    assert narrow["flank_zones"]
    assert all(zone["polarity"] == "hot" and zone["contrast"] > 0 for zone in narrow["flank_zones"])
    assert narrow["wide_channel"] is False
    assert narrow["wide_channel_candidate"] is None
    assert narrow["wide_channel_halo_inside_min"] == wide_row["wide_channel_halo_inside_min"]
    assert narrow["wide_channel_far_gap_min_c"] == wide_row["wide_channel_far_gap_min_c"]

    # The hot patch itself is a few metres wide, so its 8–40 m ring sits outside
    # the rectangle even when the rectangle is warmer than the far cable.
    hot_narrow = context_evidence(
        found, x_m, time_s, rows_half_open=(0, 12), x_span=(45.0, 50.0),
    )
    _unchanged(found, before)
    _assert_channel(hot_narrow, 45.0, 50.0)
    _assert_flag_matches_reader(hot_narrow)
    assert hot_narrow["halo_inside_fraction"] == 0.0
    assert hot_narrow["far_cable_median_c"] is not None
    assert abs(hot_narrow["far_gap_c"]) >= hot_narrow["wide_channel_far_gap_min_c"]
    assert hot_narrow["wide_channel"] is False
    assert hot_narrow["wide_channel_candidate"] is None

    hotspot = next(zone for zone in found["zones"] if zone["polarity"] == "hot")
    zone_row = context_evidence(found, x_m, time_s, zone=hotspot)
    _unchanged(found, before)
    _assert_channel(zone_row, float(hotspot["start_m"]), float(hotspot["end_m"]))
    _assert_flag_matches_reader(zone_row)
    assert zone_row["in_box_zones"]
    assert all(
        item["end_m"] < hotspot["start_m"] or item["start_m"] > hotspot["end_m"]
        for item in zone_row["flank_zones"]
    )
    assert found["anomaly_status"] == before["anomaly_status"]
    assert [zone["expert_confirmed"] for zone in found["zones"]] == [False] * len(found["zones"])

    flat_raw = np.full((12, 261), 20.0, dtype=np.float32)
    flat = analyze_field(flat_raw, wide_x, wide_t)
    flat_before = _snapshot(flat)
    flat_row = context_evidence(
        flat, wide_x, wide_t, rows_half_open=(0, 12), x_span=(30.0, 150.0),
    )
    _unchanged(flat, flat_before)
    _assert_channel(flat_row, 30.0, 150.0)
    _assert_flag_matches_reader(flat_row)
    assert flat_row["halo_inside_fraction"] == 1.0
    assert abs(flat_row["rectangle_median_c"] - flat_row["far_cable_median_c"]) < 0.05
    assert flat_row["wide_channel"] is False
    assert flat_row["wide_channel_candidate"] is None
    assert flat_row["wide_channel_halo_inside_min"] == wide_row["wide_channel_halo_inside_min"]

    whole = context_evidence(
        wide, wide_x, wide_t, rows_half_open=(0, 12), x_span=(0.0, 260.0),
    )
    _unchanged(wide, wide_before)
    _assert_channel(whole, 0.0, 260.0)
    _assert_flag_matches_reader(whole)
    assert whole["halo_inside_fraction"] == 1.0
    assert whole["far_left"]["present"] is False
    assert whole["far_right"]["present"] is False
    assert whole["far_cable_median_c"] is None
    assert whole["far_left"]["median_c"] is None
    assert whole["far_right"]["median_c"] is None
    assert whole["far_gap_c"] is None
    assert whole["wide_channel"] is False
    assert whole["wide_channel_candidate"] is None
