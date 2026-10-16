"""Shipped loader, ensemble, and event functions on synthetic DTS fixtures."""

import csv
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mgts_anomaly import (
    BROAD_HEATING,
    BROAD_HIGH_AMPLITUDE_C,
    BROAD_MILD,
    EXTENDED_MILD,
    FIRE_HOLD_MINUTES,
    FIRE_MAX_DT_PER_MIN,
    FIRE_PEAK_TEMPERATURE_C,
    LOCAL_BG_SPACE,
    LOCAL_BROAD_FRACTION,
    LOCAL_COOLING,
    LOCAL_COOLING_MAX_AMPLITUDE_C,
    LOCAL_COOLING_MIN_DURATION_MIN,
    DETECTOR_PERCENTILE,
    DETECTOR_PERCENTILE_CAP,
    LOCAL_HEATING,
    ZONE_COOLING_MAX_AMPLITUDE_C,
    ZONE_COOLING_MIN_DURATION_MIN,
    ZONE_COOLING_MIN_EXTENT_M,
    MIN_DURATION_STEPS,
    MIN_LOCAL_AMPLITUDE_C,
    OTHER,
    POSSIBLE_FIRE,
    SEGMENT_ROWS,
    TEMPORAL_MEDIAN_STEPS,
    TRANSIENT,
    ZONE_COOLING,
    classify_event_type,
    compute_ensemble,
    denoise_matrix,
    detector_percentile_for_score,
    ensemble_score,
    extract_events,
    segment_anomaly_mask,
    format_event_report,
    load_dts_to_matrix,
    statistical_anomaly_score,
    subtract_slow_temporal_median,
    temporal_median_window,
)


def _write_points_csv(path: Path, rows: list[tuple[str, list]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "points"])
        for timestamp, points in rows:
            writer.writerow([timestamp, json.dumps(points)])


def test_loader_accepts_csv_without_points_size(tmp_path: Path) -> None:
    path = tmp_path / "no_points_size.csv"
    samples = [
        ("2026-05-26 00:04:00+00", [[0.0, 20.0], [0.5, 21.0], [1.0, 22.0]]),
        ("2026-05-26 00:00:00+00", [[0.0, 10.0], [1.0, 12.0]]),
        ("2026-05-26 00:02:00+00", [[0.0, 15.0], [0.5, 16.0], [1.0, 17.0]]),
    ]
    _write_points_csv(path, samples)
    matrix, dists, timestamps = load_dts_to_matrix(str(path))
    assert matrix.shape[0] == len(samples)
    assert matrix.shape[1] == dists.shape[0]
    assert np.isfinite(matrix).all()
    assert timestamps == [
        "2026-05-26 00:00:00+00",
        "2026-05-26 00:02:00+00",
        "2026-05-26 00:04:00+00",
    ]
    assert matrix[0, 0] < matrix[1, 0] < matrix[2, 0]


def _planted_rows() -> tuple[list[tuple[str, list]], dict]:
    # The cool zone is 91 samples (~22.5 m). The cable is long enough that
    # this stays under LOCAL_BROAD_FRACTION, but it is wider than 20 m, so
    # the zone rule takes it when the neighbor ring is cold and the hold is
    # longer than 25 min.
    # Both planted intervals are a minority of the absolute temporal window.
    n_time = 240
    # 1100 samples keeps the detected cool zone under 16% of the cable
    # after the score halo, while its meter width stays above 20 m.
    n_space = 1100
    max_dist = (n_space - 1) * 0.25
    dists = np.linspace(0.0, max_dist, n_space)
    spec = {
        "dists": dists,
        "hot_s": (20, 36),
        "cool_s": (120, 210),
        "hot_t": (10, 44),
        "cool_t": (8, 50),
        "background": 22.0,
        "hot_temp": 40.0,
        "cool_temp": 14.0,
    }
    start = pd.Timestamp("2026-05-26 00:00:00+00:00")
    rows = []
    for t_index in range(n_time):
        temps = np.full(n_space, spec["background"], dtype=np.float64)
        if spec["hot_t"][0] <= t_index <= spec["hot_t"][1]:
            temps[spec["hot_s"][0] : spec["hot_s"][1] + 1] = spec["hot_temp"]
        if spec["cool_t"][0] <= t_index <= spec["cool_t"][1]:
            temps[spec["cool_s"][0] : spec["cool_s"][1] + 1] = spec["cool_temp"]
        points = [[float(dists[i]), float(temps[i])] for i in range(n_space)]
        timestamp = (start + pd.Timedelta(minutes=2 * t_index)).strftime("%Y-%m-%d %H:%M:%S+00")
        rows.append((timestamp, points))
    rows[0], rows[5] = rows[5], rows[0]
    return rows, spec


def test_planted_hot_strip_and_cool_zone(tmp_path: Path) -> None:
    path = tmp_path / "planted.csv"
    rows, spec = _planted_rows()
    _write_points_csv(path, rows)
    matrix, dists, timestamps = load_dts_to_matrix(str(path))
    assert matrix.shape[0] == 240
    assert np.isfinite(matrix).all()
    assert timestamps[0] < timestamps[1]

    denoised = denoise_matrix(matrix)
    score = compute_ensemble(denoised, use_ae=True, seed=0, epochs=1, device="cpu")
    assert np.isfinite(score).all()
    hot_cell = float(score[25, 28])
    quiet_cell = float(score[2, 60])
    assert hot_cell > quiet_cell

    events = extract_events(
        score,
        matrix,
        dists=dists,
        percentile=95,
        timestamps=timestamps,
        use_cv2=False,
    )
    types = [event["type_hypothesis"] for event in events]
    assert LOCAL_HEATING in types
    assert ZONE_COOLING in types
    for event in events:
        assert event["n_pixels"] >= 15
        assert event["spatial_points"] >= 5
        s0, s1 = event["s_range"]
        assert event["spatial_extent_m"] == abs(float(dists[s1] - dists[s0]))
        assert event["duration_min"] == event["duration_steps"] * 2.0
        if event["type_hypothesis"] in (BROAD_HEATING, POSSIBLE_FIRE, BROAD_MILD):
            assert event["spatial_fraction"] >= LOCAL_BROAD_FRACTION
        else:
            assert event["duration_steps"] >= MIN_DURATION_STEPS
            assert event["spatial_fraction"] < LOCAL_BROAD_FRACTION

    heating = next(
        event
        for event in events
        if event["type_hypothesis"] == LOCAL_HEATING
        and event["s_range"][0] <= spec["hot_s"][0]
        and event["s_range"][1] >= spec["hot_s"][1]
    )
    cooling = next(
        event
        for event in events
        if event["type_hypothesis"] == ZONE_COOLING
        and event["s_range"][0] <= spec["cool_s"][0] + 5
        and event["s_range"][1] >= spec["cool_s"][1] - 5
    )
    assert heating["spatial_extent_m"] < 15
    assert heating["duration_min"] > 20
    assert heating["amplitude"] > 1.2
    assert heating["periodic"] is False
    assert cooling["spatial_extent_m"] > ZONE_COOLING_MIN_EXTENT_M
    assert cooling["amplitude"] < -0.8
    assert cooling["local_amplitude"] < ZONE_COOLING_MAX_AMPLITUDE_C
    assert cooling["duration_min"] > ZONE_COOLING_MIN_DURATION_MIN
    assert cooling["spatial_fraction"] < LOCAL_BROAD_FRACTION
    assert heating["s_range"][0] <= spec["hot_s"][0]
    assert heating["s_range"][1] >= spec["hot_s"][1]
    assert cooling["s_range"][0] <= spec["cool_s"][0] + 5
    assert cooling["s_range"][1] >= spec["cool_s"][1] - 5


def test_components_below_size_cutoffs_are_dropped() -> None:
    score = np.zeros((40, 100), dtype=np.float32)
    original = np.full((40, 100), 20.0, dtype=np.float32)
    dists = np.linspace(0, 25, 100)
    score[10:30, 20:50] = 5
    original[10:30, 20:50] = 30
    score[0:2, 90:92] = 5
    original[0:2, 90:92] = 40
    score[36:38, 20:45] = 5
    original[36:38, 20:45] = 35
    score[10:30, 2:4] = 5
    original[10:30, 2:4] = 35

    events = extract_events(
        score,
        original,
        dists=dists,
        percentile=90,
        time_step_min=2.0,
        use_cv2=False,
    )
    assert len(events) == 1
    kept = events[0]
    assert kept["n_pixels"] >= 15
    assert kept["duration_steps"] >= MIN_DURATION_STEPS
    assert kept["spatial_points"] >= 5
    # 30 samples of 100 is wider than the local cutoff, so the block is typed
    # rather than dropped. The 20→30 °C step is faster than 0.4 °C/min.
    assert kept["spatial_points"] / original.shape[1] >= LOCAL_BROAD_FRACTION
    assert kept["type_hypothesis"] == POSSIBLE_FIRE
    assert kept["max_dt_per_min"] > FIRE_MAX_DT_PER_MIN
    assert kept["duration_min"] >= FIRE_HOLD_MINUTES
    assert kept["t_range"] == (10, 29)
    assert kept["s_range"] == (20, 49)


def test_local_cooling_needs_a_cold_neighbor_amplitude_and_a_long_hold() -> None:
    cold = {
        "amplitude": -2.0,
        "local_amplitude": -1.5,
        "spatial_extent_m": 5.0,
        "duration_min": 30.0,
        "spatial_fraction": 0.05,
    }
    assert classify_event_type(cold) == LOCAL_COOLING
    assert classify_event_type({**cold, "local_amplitude": LOCAL_COOLING_MAX_AMPLITUDE_C}) == OTHER
    assert classify_event_type({**cold, "duration_min": LOCAL_COOLING_MIN_DURATION_MIN}) == OTHER
    assert classify_event_type({**cold, "spatial_extent_m": ZONE_COOLING_MIN_EXTENT_M}) == LOCAL_COOLING
    wide_cold = {**cold, "spatial_extent_m": ZONE_COOLING_MIN_EXTENT_M + 0.5, "spatial_fraction": 0.05}
    assert classify_event_type(wide_cold) == ZONE_COOLING
    broad = {
        **cold,
        "spatial_fraction": LOCAL_BROAD_FRACTION,
        "max_dt_per_min": 0.0,
        "peak_temperature_c": 20.0,
    }
    assert classify_event_type(broad) == ZONE_COOLING
    mild = {
        **broad,
        "amplitude": 0.3,
        "local_amplitude": -0.2,
    }
    assert classify_event_type(mild) == BROAD_MILD
    assert classify_event_type({**broad, "max_dt_per_min": 1.0}) == POSSIBLE_FIRE
    assert classify_event_type({**broad, "amplitude": 2.0, "max_dt_per_min": 0.1}) == BROAD_HEATING
    heat = {
        "amplitude": 2.0,
        "local_amplitude": 5.0,
        "spatial_extent_m": 6.0,
        "duration_min": 200.0,
        "spatial_fraction": 0.02,
    }
    assert classify_event_type(heat) == LOCAL_HEATING


def test_type_rules_cover_all_five_hypotheses() -> None:
    assert classify_event_type(
        {"amplitude": 2.0, "spatial_extent_m": 5.0, "duration_min": 30.0}
    ) == LOCAL_HEATING
    assert classify_event_type(
        {
            "amplitude": -2.0,
            "local_amplitude": -1.2,
            "spatial_extent_m": 25.0,
            "duration_min": 100.0,
        }
    ) == ZONE_COOLING
    assert classify_event_type(
        {"amplitude": 2.0, "spatial_extent_m": 5.0, "duration_min": 10.0}
    ) == TRANSIENT
    assert classify_event_type(
        {"amplitude": 0.5, "spatial_extent_m": 40.0, "duration_min": 100.0}
    ) == EXTENDED_MILD
    assert classify_event_type(
        {"amplitude": 0.2, "spatial_extent_m": 10.0, "duration_min": 100.0}
    ) == OTHER


def test_default_weights_and_percentile() -> None:
    stat_sig = inspect.signature(statistical_anomaly_score)
    assert stat_sig.parameters["residual_w"].default == 0.45
    assert stat_sig.parameters["spatial_w"].default == 0.35
    assert stat_sig.parameters["gradient_w"].default == 0.20
    ens_sig = inspect.signature(ensemble_score)
    assert ens_sig.parameters["stat_w"].default == 0.55
    assert ens_sig.parameters["ae_w"].default == 0.45
    event_sig = inspect.signature(extract_events)
    assert event_sig.parameters["percentile"].default == 99.2
    assert event_sig.parameters["min_pixels"].default == 15
    assert event_sig.parameters["min_duration"].default == 3
    assert event_sig.parameters["min_spatial"].default == 5
    assert event_sig.parameters["min_duration_steps"].default == MIN_DURATION_STEPS
    assert event_sig.parameters["min_local_amplitude_c"].default == MIN_LOCAL_AMPLITUDE_C
    assert event_sig.parameters["segment_rows"].default == SEGMENT_ROWS
    assert event_sig.parameters["local_broad_fraction"].default == LOCAL_BROAD_FRACTION
    assert inspect.signature(temporal_median_window).parameters["window_steps"].default == (
        TEMPORAL_MEDIAN_STEPS
    )
    assert inspect.signature(compute_ensemble).parameters["window_steps"].default == TEMPORAL_MEDIAN_STEPS
    assert MIN_DURATION_STEPS > 3
    assert 0.15 <= LOCAL_BROAD_FRACTION <= 0.18
    assert MIN_LOCAL_AMPLITUDE_C < 1.0
    assert FIRE_MAX_DT_PER_MIN == 0.4
    assert FIRE_PEAK_TEMPERATURE_C == 48.0
    assert FIRE_HOLD_MINUTES >= 5.0
    assert BROAD_HIGH_AMPLITUDE_C > MIN_LOCAL_AMPLITUDE_C
    assert temporal_median_window(708) == TEMPORAL_MEDIAN_STEPS
    assert temporal_median_window(5000) == TEMPORAL_MEDIAN_STEPS
    assert temporal_median_window(708) % 2 == 1


def test_noisy_score_raises_the_segment_cut() -> None:
    """Calm maps stay on percentile 99.2. A wide score bulk keeps a stricter tail."""
    calm = np.zeros((SEGMENT_ROWS, 200), dtype=np.float64)
    calm[10:30, 8:16] = 6.0
    calm_cut = detector_percentile_for_score(calm)
    assert calm_cut == DETECTOR_PERCENTILE
    assert np.array_equal(
        segment_anomaly_mask(calm, calm_cut),
        segment_anomaly_mask(calm, DETECTOR_PERCENTILE),
    )
    assert calm[segment_anomaly_mask(calm, calm_cut)].min() == 6.0

    score = np.random.default_rng(0).normal(0.0, 1.0, size=(SEGMENT_ROWS, 400))
    score = np.minimum(score, 1.5)
    # 120 peak pixels sit above the 99.8 tail. 400 moderate pixels fill the
    # 99.2 tail and fall out once the cut rises. The bulk stays below both.
    score[5:15, 10:22] = 40.0
    score[40:60, 80:100] = 5.0
    applied = detector_percentile_for_score(score)
    assert applied > DETECTOR_PERCENTILE
    assert detector_percentile_for_score(score, 97.0) == 97.0
    raised = segment_anomaly_mask(score, applied)
    fixed = segment_anomaly_mask(score, DETECTOR_PERCENTILE)
    assert int(raised.sum()) < int(fixed.sum())
    assert np.all(~raised | fixed)
    assert raised[8, 14]
    assert not raised[50, 90]
    assert fixed[50, 90]
    # A pixel placed on a value the raised mask already keeps stays selected.
    edge = float(score[raised].min())
    probe = score.copy()
    probe[0, 0] = edge
    assert segment_anomaly_mask(probe, applied)[0, 0]

    temperatures = np.full(score.shape, 20.0, dtype=np.float64)
    temperatures[5:15, 10:22] = 28.0
    temperatures[40:60, 80:100] = 24.0
    dists = np.arange(score.shape[1], dtype=np.float64) * 0.25
    kept = extract_events(score, temperatures, dists=dists, time_step_min=2.0)
    same = extract_events(
        score,
        temperatures,
        dists=dists,
        percentile=applied,
        time_step_min=2.0,
    )
    assert len(kept) == len(same)
    assert any(_overlaps(event, 5, 14, 10, 21) for event in kept)
    assert not any(_overlaps(event, 40, 59, 80, 99) for event in kept)
    assert applied <= DETECTOR_PERCENTILE_CAP


def _overlaps(event: dict, t0: int, t1: int, s0: int, s1: int) -> bool:
    event_t0, event_t1 = event["t_range"]
    event_s0, event_s1 = event["s_range"]
    return event_t0 <= t1 and event_t1 >= t0 and event_s0 <= s1 and event_s1 >= s0


def test_temporal_median_keeps_early_multihour_heat() -> None:
    """A 268-step heat in a 708-row record stays; a 600-step baseline is the median."""
    n_time = 708
    heat_steps = 268
    baseline_steps = 600
    matrix = np.full((n_time, 30), 20.0, dtype=np.float64)
    matrix[:baseline_steps, 2:5] = 40.0
    matrix[:heat_steps, 20:25] = 26.0
    cleaned = subtract_slow_temporal_median(matrix)
    assert cleaned[heat_steps // 2, 22] > 4.0
    assert abs(cleaned[baseline_steps // 2, 3]) < 1.0
    score = compute_ensemble(matrix, use_ae=False)
    assert np.isfinite(score).all()
    assert float(score[heat_steps // 2, 22]) > float(score[baseline_steps // 2, 3])


def test_quiet_segment_keeps_moderate_heat_beside_a_louder_spike() -> None:
    segment = SEGMENT_ROWS
    n_time = segment * 2
    n_space = 200
    matrix = np.full((n_time, n_space), 20.0, dtype=np.float64)
    spike_t0 = segment // 3
    heat_t0 = segment + segment // 3
    # Eight samples is inside the 15-step local median, so the strip scores
    # through its interior and the segment threshold can keep all 40 steps.
    matrix[spike_t0 : spike_t0 + 40, 10:18] = 55.0
    matrix[heat_t0 : heat_t0 + 40, 100:108] = 24.0
    assert heat_t0 >= segment
    score = compute_ensemble(matrix, use_ae=False)
    assert np.isfinite(score).all()
    global_threshold = float(np.percentile(score, 99.2))
    heat_score = score[heat_t0 : heat_t0 + 40, 100:108]
    assert float(heat_score.max()) < global_threshold
    dists = np.arange(n_space, dtype=np.float64) * 0.25
    events = extract_events(score, matrix, dists=dists, time_step_min=2.0, use_cv2=False)
    kept = [
        event
        for event in events
        if _overlaps(event, heat_t0, heat_t0 + 39, 100, 107)
    ]
    assert kept
    heat = kept[0]
    assert heat["duration_steps"] >= MIN_DURATION_STEPS
    assert heat["spatial_fraction"] < LOCAL_BROAD_FRACTION
    assert abs(heat["amplitude"]) >= MIN_LOCAL_AMPLITUDE_C


def test_appended_quiet_time_keeps_moderate_local_heat() -> None:
    """A heat the shipped score keeps does not vanish when quiet rows are added.

    The short series caps the median window. The longer series uses the full
    absolute window. Both still keep an overlapping local event.
    """
    segment = SEGMENT_ROWS
    n_space = 200
    short = np.full((segment * 2, n_space), 20.0, dtype=np.float64)
    spike_t0 = segment // 3
    heat_t0 = segment + segment // 3
    short[spike_t0 : spike_t0 + 40, 10:18] = 55.0
    short[heat_t0 : heat_t0 + 40, 100:108] = 24.0
    longer = np.vstack([short, np.full((segment * 4, n_space), 20.0, dtype=np.float64)])
    assert temporal_median_window(short.shape[0]) < TEMPORAL_MEDIAN_STEPS
    assert temporal_median_window(longer.shape[0]) == TEMPORAL_MEDIAN_STEPS
    dists = np.arange(n_space, dtype=np.float64) * 0.25

    def kept_heat(values: np.ndarray) -> list[dict]:
        score = compute_ensemble(values, use_ae=False)
        assert np.isfinite(score).all()
        events = extract_events(score, values, dists=dists, time_step_min=2.0, use_cv2=False)
        return [
            event
            for event in events
            if _overlaps(event, heat_t0, heat_t0 + 39, 100, 107)
            and event["spatial_fraction"] < LOCAL_BROAD_FRACTION
        ]

    assert kept_heat(short)
    assert kept_heat(longer)


def test_junk_filter_types_wide_and_drops_short_components() -> None:
    n_space = 200
    wide = int(LOCAL_BROAD_FRACTION * n_space) + 10
    assert wide / n_space >= LOCAL_BROAD_FRACTION
    short_steps = MIN_DURATION_STEPS - 1
    assert short_steps >= 3
    n_time = 80
    score = np.zeros((n_time, n_space), dtype=np.float32)
    original = np.full((n_time, n_space), 20.0, dtype=np.float32)
    # Fast wide jump, peak stays under 48 °C. This used to be discarded.
    score[5:35, 0:wide] = 10
    original[5:35, 0:wide] = 36.0
    score[0:short_steps, 160:168] = 10
    original[0:short_steps, 160:168] = 35
    score[10:41, 120:141] = 10
    original[10:41, 120:141] = 30
    dists = np.arange(n_space, dtype=np.float64) * 0.25
    events = extract_events(
        score,
        original,
        dists=dists,
        time_step_min=2.0,
        use_cv2=False,
    )
    assert events.tally["duration"] >= 1
    assert "spatial_fraction" not in events.tally
    wide_events = [
        event
        for event in events
        if event["spatial_fraction"] >= LOCAL_BROAD_FRACTION
        and _overlaps(event, 5, 34, 0, wide - 1)
    ]
    assert wide_events
    assert wide_events[0]["type_hypothesis"] == POSSIBLE_FIRE
    assert wide_events[0]["max_dt_per_min"] > FIRE_MAX_DT_PER_MIN
    assert wide_events[0]["peak_temperature_c"] < FIRE_PEAK_TEMPERATURE_C
    assert wide_events[0]["duration_min"] >= FIRE_HOLD_MINUTES
    assert "width_changes" in wide_events[0]
    assert any(_overlaps(event, 10, 40, 120, 140) for event in events)
    assert all(
        event["duration_steps"] >= MIN_DURATION_STEPS or event["type_hypothesis"] in {
            BROAD_HEATING,
            POSSIBLE_FIRE,
            BROAD_MILD,
        }
        for event in events
    )
    report = format_event_report(events)
    lines = report.splitlines()
    assert lines[0].startswith("Filtered ")
    assert "duration=" in lines[0]
    assert "amplitude=" in lines[0]
    assert "spatial_fraction=" not in lines[0]
    assert lines[1].startswith("Categories:")
    assert f"{POSSIBLE_FIRE}=" in lines[1]
    assert f"{BROAD_HEATING}=" in lines[1]
    assert f"{BROAD_MILD}=" in lines[1]
    assert f"{LOCAL_HEATING}=" in lines[1]
    assert lines[2].startswith("Detected ")
    assert "meters=" in report


def test_local_amplitude_keeps_heating_when_file_mean_would_flip_the_sign() -> None:
    n_time = 40
    n_space = 900
    original = np.full((n_time, n_space), 20.0, dtype=np.float64)
    original[:, 30:47] = 30.0
    original[:, 560:900] = 55.0
    global_amplitude = float(original[:, 30:47].mean() - original.mean())
    assert global_amplitude < 0.0
    score = np.zeros((n_time, n_space), dtype=np.float32)
    score[:, 30:47] = 10
    score[:, 560:900] = 10
    dists = np.arange(n_space, dtype=np.float64) * 0.25
    events = extract_events(score, original, dists=dists, time_step_min=2.0, use_cv2=False)
    heating = [
        event
        for event in events
        if event["type_hypothesis"] == LOCAL_HEATING and _overlaps(event, 0, n_time - 1, 30, 46)
    ]
    assert heating
    heat = heating[0]
    assert heat["amplitude"] > 1.2
    assert heat["local_amplitude"] > 5.0
    assert 0.0 <= heat["confidence"] <= 1.0
    assert heat["spatial_extent_m"] < 15
    assert heat["duration_min"] > 20
    assert heat["spatial_fraction"] < LOCAL_BROAD_FRACTION
    broad = [
        event
        for event in events
        if event["type_hypothesis"] == POSSIBLE_FIRE and _overlaps(event, 0, n_time - 1, 560, 899)
    ]
    assert broad
    assert broad[0]["peak_temperature_c"] > FIRE_PEAK_TEMPERATURE_C
    assert broad[0]["spatial_fraction"] >= LOCAL_BROAD_FRACTION
    assert broad[0]["duration_min"] >= FIRE_HOLD_MINUTES


def test_section_scale_heat_is_kept() -> None:
    n_time = SEGMENT_ROWS * 2
    n_space = 400
    matrix = np.full((n_time, n_space), 20.0, dtype=np.float64)
    heat_t0, heat_t1 = 150, 189
    heat_s0, heat_s1 = 100, 131
    matrix[heat_t0 : heat_t1 + 1, heat_s0 : heat_s1 + 1] = 26.0
    score = compute_ensemble(matrix, use_ae=False)
    assert np.isfinite(score).all()
    dists = np.arange(n_space, dtype=np.float64) * 0.25
    events = extract_events(score, matrix, dists=dists, time_step_min=2.0, use_cv2=False)
    kept = [event for event in events if _overlaps(event, heat_t0, heat_t1, heat_s0, heat_s1)]
    assert kept
    heat = kept[0]
    assert heat["duration_steps"] >= MIN_DURATION_STEPS
    assert heat["spatial_extent_m"] > 3.0
    assert heat["spatial_fraction"] < LOCAL_BROAD_FRACTION
    assert heat["type_hypothesis"] == LOCAL_HEATING
    assert abs(heat["amplitude"]) >= MIN_LOCAL_AMPLITUDE_C
    assert heat["duration_min"] > 20
    assert heat["spatial_extent_m"] < 15
    assert (heat_s1 - heat_s0) * 0.25 > 3.0


def _extract_wide(original: np.ndarray, s0: int, s1: int, t0: int, t1: int) -> dict:
    score = np.zeros(original.shape, dtype=np.float32)
    score[t0 : t1 + 1, s0 : s1 + 1] = 10
    dists = np.arange(original.shape[1], dtype=np.float64) * 0.25
    events = extract_events(score, original, dists=dists, time_step_min=2.0, use_cv2=False)
    matched = [event for event in events if _overlaps(event, t0, t1, s0, s1)]
    assert matched, format_event_report(events)
    return matched[0]


def test_wide_fast_rise_is_possible_fire() -> None:
    n_time, n_space = 80, 200
    s0, s1 = 10, 59
    t0, t1 = 12, 40
    original = np.full((n_time, n_space), 20.0, dtype=np.float64)
    original[t0 : t1 + 1, s0 : s1 + 1] = 36.0
    event = _extract_wide(original, s0, s1, t0, t1)
    assert event["spatial_fraction"] >= LOCAL_BROAD_FRACTION
    assert event["type_hypothesis"] == POSSIBLE_FIRE
    assert event["max_dt_per_min"] > FIRE_MAX_DT_PER_MIN
    assert event["peak_temperature_c"] < FIRE_PEAK_TEMPERATURE_C
    assert event["duration_min"] >= FIRE_HOLD_MINUTES
    assert isinstance(event["width_changes"], bool)


def test_wide_slow_high_amplitude_is_broad_heating() -> None:
    n_time, n_space = 80, 200
    s0, s1 = 10, 59
    original = np.full((n_time, n_space), 20.0, dtype=np.float64)
    # 0.25 °C per 2 min step is 0.125 °C/min, under the fire rate.
    # The extra columns rise with the block, so widening the mask is not a jump.
    for offset, row in enumerate(range(10, 30)):
        original[row, s0 : s1 + 6] = 20.0 + 0.25 * offset
    original[30:55, s0 : s1 + 6] = 24.75
    score = np.zeros(original.shape, dtype=np.float32)
    score[10:30, s0 : s1 + 1] = 10
    score[30:55, s0 : s1 + 6] = 10
    dists = np.arange(n_space, dtype=np.float64) * 0.25
    events = extract_events(score, original, dists=dists, time_step_min=2.0, use_cv2=False)
    matched = [event for event in events if _overlaps(event, 10, 54, s0, s1)]
    assert matched, format_event_report(events)
    event = matched[0]
    assert event["spatial_fraction"] >= LOCAL_BROAD_FRACTION
    assert event["type_hypothesis"] == BROAD_HEATING
    assert event["amplitude"] >= BROAD_HIGH_AMPLITUDE_C
    assert event["max_dt_per_min"] <= FIRE_MAX_DT_PER_MIN
    assert event["peak_temperature_c"] < FIRE_PEAK_TEMPERATURE_C
    assert event["width_changes"] is True


def test_wide_weak_event_is_broad_mild() -> None:
    n_time, n_space = 80, 200
    s0, s1 = 10, 59
    t0, t1 = 10, 49
    original = np.full((n_time, n_space), 20.0, dtype=np.float64)
    original[t0 : t1 + 1, s0 : s1 + 1] = 20.35
    event = _extract_wide(original, s0, s1, t0, t1)
    assert event["spatial_fraction"] >= LOCAL_BROAD_FRACTION
    assert event["type_hypothesis"] == BROAD_MILD
    assert abs(event["amplitude"]) < MIN_LOCAL_AMPLITUDE_C
    assert event["max_dt_per_min"] <= FIRE_MAX_DT_PER_MIN
    assert event["peak_temperature_c"] < FIRE_PEAK_TEMPERATURE_C
    assert isinstance(event["width_changes"], bool)


def test_slow_broad_heat_on_full_cable_stays_broad_heating() -> None:
    """A slow heat wider than the 241-step local median still has outside contrast.

    On a 1716-column cable the local median sits inside the block. Amplitude
    for a broad event is the cable outside the mask, so the heat stays
    broad_heating.
    """
    n_space = 1716
    n_time = 80
    assert n_space >= 1716
    assert LOCAL_BG_SPACE < int(LOCAL_BROAD_FRACTION * n_space)

    def planted(width: int, background: float, step_c: float, peak: float, s0: int) -> dict:
        assert width > LOCAL_BG_SPACE
        assert width / n_space >= LOCAL_BROAD_FRACTION
        original = np.full((n_time, n_space), background, dtype=np.float64)
        row = 10
        value = background
        while value < peak - 1e-9:
            original[row, s0 : s0 + width] = value
            value = min(peak, value + step_c)
            row += 1
        t1 = row + 20
        original[row:t1, s0 : s0 + width] = peak
        score = np.zeros(original.shape, dtype=np.float32)
        score[10:t1, s0 : s0 + width] = 10
        dists = np.arange(n_space, dtype=np.float64) * 0.25
        events = extract_events(score, original, dists=dists, time_step_min=2.0, use_cv2=False)
        matched = [event for event in events if _overlaps(event, 10, t1 - 1, s0, s0 + width - 1)]
        assert matched, format_event_report(events)
        return matched[0]

    # +0.2 °C per 2 min step is 0.1 °C/min. Peak stays under 48 °C.
    wide = planted(width=300, background=22.0, step_c=0.2, peak=28.0, s0=400)
    assert wide["type_hypothesis"] == BROAD_HEATING
    assert wide["amplitude"] >= BROAD_HIGH_AMPLITUDE_C
    assert wide["local_amplitude"] >= BROAD_HIGH_AMPLITUDE_C
    assert 0.0 <= wide["confidence"] <= 1.0
    assert wide["max_dt_per_min"] <= FIRE_MAX_DT_PER_MIN
    assert wide["peak_temperature_c"] < FIRE_PEAK_TEMPERATURE_C
    assert wide["spatial_fraction"] >= LOCAL_BROAD_FRACTION

    minimum = planted(width=275, background=20.0, step_c=0.2, peak=25.8, s0=500)
    assert minimum["type_hypothesis"] == BROAD_HEATING
    assert minimum["amplitude"] >= BROAD_HIGH_AMPLITUDE_C
    assert minimum["max_dt_per_min"] <= FIRE_MAX_DT_PER_MIN
    assert minimum["peak_temperature_c"] <= 25.8 + 1e-6


def test_close_same_sign_fragments_merge_and_opposites_stay() -> None:
    n_time, n_space = 80, 220
    score = np.zeros((n_time, n_space), dtype=np.float32)
    original = np.full((n_time, n_space), 20.0, dtype=np.float64)
    score[10:25, 40:52] = 10
    original[10:25, 40:52] = 32.0
    score[30:46, 42:55] = 10
    original[30:46, 42:55] = 31.0
    score[10:25, 58:70] = 10
    original[10:25, 58:70] = 12.0
    score[10:30, 150:162] = 10
    original[10:30, 150:162] = 33.0
    score[8:19, 100:112] = 10
    original[8:19, 100:112] = 30.0
    score[55:71, 100:112] = 10
    original[55:71, 100:112] = 30.0
    dists = np.arange(n_space, dtype=np.float64) * 0.25
    events = extract_events(score, original, dists=dists, time_step_min=2.0, use_cv2=False)
    merged = [
        event
        for event in events
        if _overlaps(event, 10, 24, 40, 51) and _overlaps(event, 30, 45, 42, 54)
    ]
    assert len(merged) == 1
    assert merged[0]["t_range"][0] <= 10
    assert merged[0]["t_range"][1] >= 45
    assert merged[0]["local_amplitude"] > 1.2
    assert 0.0 <= merged[0]["confidence"] <= 1.0
    assert merged[0]["type_hypothesis"] == LOCAL_HEATING
    cool = [event for event in events if _overlaps(event, 10, 24, 58, 69)]
    assert len(cool) == 1
    assert cool[0]["local_amplitude"] < 0
    assert not _overlaps(cool[0], 10, 24, 40, 51)
    far = [event for event in events if _overlaps(event, 10, 29, 150, 161)]
    assert len(far) == 1
    assert not _overlaps(far[0], 10, 24, 40, 51)
    early, late = (
        [event for event in events if _overlaps(event, 8, 18, 100, 111)],
        [event for event in events if _overlaps(event, 55, 70, 100, 111)],
    )
    assert len(early) == 1 and len(late) == 1
    assert early[0] is not late[0]
    assert early[0]["t_range"][1] < 55
    report = format_event_report(events)
    assert "local_amplitude=" in report
    assert "confidence=" in report


def test_min_confidence_drops_the_weaker_event() -> None:
    n_time, n_space = 40, 180
    score = np.zeros((n_time, n_space), dtype=np.float32)
    original = np.full((n_time, n_space), 20.0, dtype=np.float64)
    score[5:25, 20:32] = 10
    original[5:25, 20:32] = 20.7
    score[5:25, 120:132] = 10
    original[5:25, 120:132] = 32.0
    dists = np.arange(n_space, dtype=np.float64) * 0.25
    kept = extract_events(score, original, dists=dists, time_step_min=2.0, use_cv2=False)
    assert len(kept) == 2
    cut = extract_events(
        score,
        original,
        dists=dists,
        time_step_min=2.0,
        use_cv2=False,
        min_confidence=0.75,
    )
    assert len(cut) == 1
    assert _overlaps(cut[0], 5, 24, 120, 131)
    assert cut[0]["confidence"] >= 0.75
    assert cut.tally["confidence"] >= 1


def test_agreeing_score_branches_raise_confidence() -> None:
    n_time, n_space = 40, 80
    score = np.zeros((n_time, n_space), dtype=np.float32)
    original = np.full((n_time, n_space), 20.0, dtype=np.float64)
    score[8:28, 20:36] = 10
    original[8:28, 20:36] = 30.0
    ramp = np.linspace(0.2, 1.0, 20, dtype=np.float64)[:, None]
    stat = np.zeros((n_time, n_space), dtype=np.float64)
    stat[8:28, 20:36] = ramp
    agree = stat.copy()
    disagree = np.zeros_like(stat)
    disagree[8:28, 20:36] = ramp[::-1]
    dists = np.arange(n_space, dtype=np.float64) * 0.25
    common = dict(dists=dists, time_step_min=2.0, use_cv2=False)
    agreed = extract_events(score, original, stat_score=stat, ae_score=agree, **common)
    opposed = extract_events(score, original, stat_score=stat, ae_score=disagree, **common)
    assert len(agreed) == 1 and len(opposed) == 1
    assert agreed[0]["confidence"] > opposed[0]["confidence"]


def test_plot_labels_type_and_amplitude(tmp_path: Path) -> None:
    from mgts_anomaly import save_anomaly_plot

    n_time, n_space = 30, 40
    matrix = np.full((n_time, n_space), 20.0, dtype=np.float64)
    score = np.zeros((n_time, n_space), dtype=np.float32)
    matrix[5:20, 8:16] = 30.0
    score[5:20, 8:16] = 5
    events = extract_events(score, matrix, time_step_min=2.0, use_cv2=False)
    assert events
    assert "local_amplitude" in events[0]
    plot_path = tmp_path / "labeled.png"
    save_anomaly_plot(matrix, score, events, str(plot_path))
    payload = plot_path.read_bytes()
    assert payload[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(payload) > 32


def test_extract_keeps_narrow_heat_and_cold_and_types_a_wide_cold_zone() -> None:
    n_time, n_space = 40, 800
    original = np.full((n_time, n_space), 22.0, dtype=np.float64)
    score = np.zeros((n_time, n_space), dtype=np.float32)
    original[:, 40:140] = 12.0
    score[:, 40:140] = 10
    original[:, 400:408] = 12.0
    score[:, 400:408] = 10
    original[:, 600:612] = 36.0
    score[:, 600:612] = 10
    dists = np.arange(n_space, dtype=np.float64) * 0.25
    events = extract_events(score, original, dists=dists, time_step_min=2.0, use_cv2=False)
    zone = next(event for event in events if _overlaps(event, 0, n_time - 1, 40, 139))
    narrow = next(event for event in events if _overlaps(event, 0, n_time - 1, 400, 407))
    heat = next(event for event in events if _overlaps(event, 0, n_time - 1, 600, 611))
    assert zone["type_hypothesis"] == ZONE_COOLING
    assert zone["spatial_fraction"] < LOCAL_BROAD_FRACTION
    assert zone["spatial_extent_m"] > ZONE_COOLING_MIN_EXTENT_M
    assert zone["local_amplitude"] < ZONE_COOLING_MAX_AMPLITUDE_C
    assert zone["duration_min"] > ZONE_COOLING_MIN_DURATION_MIN
    assert narrow["type_hypothesis"] == LOCAL_COOLING
    assert narrow["spatial_extent_m"] <= ZONE_COOLING_MIN_EXTENT_M
    assert narrow["periodic"] is False
    assert heat["type_hypothesis"] == LOCAL_HEATING
    assert heat["spatial_extent_m"] < 15
    assert heat["periodic"] is False

    short_cable = np.full((n_time, 200), 22.0, dtype=np.float64)
    short_score = np.zeros(short_cable.shape, dtype=np.float32)
    short_cable[:, 10:90] = 12.0
    short_score[:, 10:90] = 10
    wide_share = extract_events(
        short_score,
        short_cable,
        dists=np.arange(200, dtype=np.float64) * 0.25,
        time_step_min=2.0,
        use_cv2=False,
    )
    shared = next(event for event in wide_share if _overlaps(event, 0, n_time - 1, 10, 89))
    assert shared["spatial_fraction"] >= LOCAL_BROAD_FRACTION
    assert shared["type_hypothesis"] == ZONE_COOLING


def test_zone_cooling_survives_when_legacy_amplitude_is_weak() -> None:
    """A wide cold block stays zone_cooling when the 3x241 amplitude is under 0.6 C.

    Both cases go through extract_events on a 1716-sample cable. Morphology
    keeps 38 steps of a 40-row block, which is 76 min at 2 min per step.
    """
    n_time, n_space = 40, 1716
    dists = np.arange(n_space, dtype=np.float64) * 0.25
    original = np.full((n_time, n_space), 22.0, dtype=np.float64)
    score = np.zeros((n_time, n_space), dtype=np.float32)
    original[:, 200:461] = 12.0
    score[:, 200:461] = 10
    events = extract_events(score, original, dists=dists, time_step_min=2.0, use_cv2=False)
    zone = next(event for event in events if _overlaps(event, 0, n_time - 1, 200, 460))
    assert zone["type_hypothesis"] == ZONE_COOLING
    assert zone["type_hypothesis"] == "zone_cooling (AC / occupancy)"
    assert abs(zone["spatial_fraction"] - 0.152) < 0.002
    assert zone["spatial_fraction"] < LOCAL_BROAD_FRACTION
    assert zone["spatial_extent_m"] == 65.0
    assert zone["duration_min"] == 76.0
    assert zone["local_amplitude"] == -10.0
    assert zone["amplitude"] == 0.0
    assert abs(zone["amplitude"]) < MIN_LOCAL_AMPLITUDE_C

    width = int(round(0.203 * n_space))
    weak = np.full((n_time, n_space), 20.40, dtype=np.float64)
    weak_score = np.zeros((n_time, n_space), dtype=np.float32)
    s0 = 400
    s1 = s0 + width
    weak[:, s0:s1] = 20.0
    weak[:, s0 - 16 : s0] = 22.0
    weak[:, s1 : s1 + 16] = 22.0
    weak_score[:, s0:s1] = 10
    kept = extract_events(weak_score, weak, dists=dists, time_step_min=2.0, use_cv2=False)
    broad = next(event for event in kept if _overlaps(event, 0, n_time - 1, s0, s1 - 1))
    assert broad["type_hypothesis"] == ZONE_COOLING
    assert broad["type_hypothesis"] == "zone_cooling (AC / occupancy)"
    assert abs(broad["spatial_fraction"] - 0.203) < 0.002
    assert broad["spatial_fraction"] >= LOCAL_BROAD_FRACTION
    assert broad["duration_min"] == 76.0
    assert broad["local_amplitude"] == -2.0
    assert abs(broad["amplitude"] - (-0.44)) < 0.01
    assert abs(broad["amplitude"]) < MIN_LOCAL_AMPLITUDE_C


def test_periodic_flag_marks_a_repeating_local_heat_only() -> None:
    n_time, n_space = 120, 220
    step = 2.0
    period_steps = 20
    clock = np.arange(n_time, dtype=np.float64)
    original = np.full((n_time, n_space), 22.0, dtype=np.float64)
    score = np.zeros((n_time, n_space), dtype=np.float32)
    wave = 30.0 + 8.0 * np.sin(2.0 * np.pi * clock / period_steps)
    original[:, 20:32] = wave[:, None]
    score[:, 20:32] = 10
    original[:, 80:92] = 34.0
    score[:, 80:92] = 10
    original[40:70, 150:162] = 40.0
    score[:, 150:162] = 10
    events = extract_events(
        score,
        original,
        dists=np.arange(n_space, dtype=np.float64) * 0.25,
        time_step_min=step,
        use_cv2=False,
    )
    oscillating = next(event for event in events if _overlaps(event, 0, n_time - 1, 20, 31))
    flat = next(event for event in events if _overlaps(event, 0, n_time - 1, 80, 91))
    pulse = next(event for event in events if _overlaps(event, 0, n_time - 1, 150, 161))
    assert oscillating["type_hypothesis"] == LOCAL_HEATING
    assert oscillating["periodic"] is True
    assert flat["type_hypothesis"] == LOCAL_HEATING
    assert flat["periodic"] is False
    assert pulse["type_hypothesis"] == LOCAL_HEATING
    assert pulse["periodic"] is False


def test_event_json_copies_periodic_flag(tmp_path: Path) -> None:
    """The shipped writer stores periodic, and false when the pipeline omitted it."""
    from main import _write_events_json

    marked = {
        "distance_start_m": 94.0,
        "distance_end_m": 100.0,
        "duration_min": 352.0,
        "amplitude": 4.21,
        "local_amplitude": 5.17,
        "confidence": 0.81,
        "type_hypothesis": LOCAL_HEATING,
        "periodic": True,
        "t_range": (10, 186),
        "s_range": (100, 120),
    }
    omitted = {
        "distance_start_m": 31.8,
        "distance_end_m": 43.7,
        "duration_min": 41.0,
        "amplitude": -5.21,
        "local_amplitude": -4.21,
        "confidence": 0.78,
        "type_hypothesis": ZONE_COOLING,
        "t_range": (0, 20),
        "s_range": (40, 90),
    }
    dest = tmp_path / "sample_events.json"
    _write_events_json([marked, omitted], str(dest))
    text = dest.read_text(encoding="utf-8")
    saved = json.loads(text)
    assert '"periodic": true' in text
    assert '"periodic": false' in text
    assert saved[0]["periodic"] is True
    assert saved[1]["periodic"] is False
    for source, row in ((marked, saved[0]), (omitted, saved[1])):
        assert row["type_hypothesis"] == source["type_hypothesis"]
        assert row["local_amplitude"] == source["local_amplitude"]
        assert row["confidence"] == source["confidence"]
        assert row["duration_min"] == source["duration_min"]
        assert row["amplitude"] == source["amplitude"]
        assert row["meters"] == [source["distance_start_m"], source["distance_end_m"]]


def test_extract_events_opencv_finds_a_blob_and_ignores_a_flat_map(monkeypatch) -> None:
    """The shipped OpenCV pass is a separate list and is empty on a flat map."""
    import mgts_anomaly as pipe

    assert pipe.HAS_CV2
    steps: list[tuple] = []
    real_components = pipe.cv2.connectedComponentsWithStats

    def components(image, *args, **kwargs):
        steps.append(("connectedComponentsWithStats", image.dtype, np.array(image, copy=True)))
        return real_components(image, *args, **kwargs)

    monkeypatch.setattr(pipe.cv2, "connectedComponentsWithStats", components)

    score = np.zeros((40, 80), dtype=np.float64)
    score[10:28, 20:40] = 9.0
    temps = np.zeros((40, 80), dtype=np.float64)
    temps[10:16, 20:26] = 1.2
    found = pipe.extract_events_opencv(score, temps)
    assert found
    assert all(item["source"] == "opencv" for item in found)
    assert any(item["s_range"][0] <= 22 <= item["s_range"][1] for item in found)
    assert steps
    kind, dtype, image = steps[0]
    assert kind == "connectedComponentsWithStats"
    assert dtype == np.dtype(np.uint8)
    assert set(np.unique(image).tolist()) <= {0, 255}
    assert image[12, 22] == 255
    assert image[20, 30] == 0
    assert not np.array_equal(image, temps)

    typed = extract_events(score, np.full(score.shape, 22.0), time_step_min=2.0, use_cv2=False)
    assert all(event.get("source") != "opencv" for event in typed)

    assert pipe.extract_events_opencv(np.zeros((40, 80)), np.zeros((40, 80))) == []
    assert pipe.extract_events_opencv(np.full((40, 80), 3.0), np.full((40, 80), 22.0)) == []


def test_opencv_finder_ands_score_percentile_and_contrast(tmp_path: Path, monkeypatch) -> None:
    """Shipped finder: both cuts, exact 1.2 °C, and a too-small box absent from the legend."""
    import copy
    import inspect

    import matplotlib.pyplot as plt

    import mgts_anomaly as pipe
    from main import parse_args
    from mgts_anomaly import (
        annotate_opencv_components,
        extract_events_opencv,
        filter_opencv_components,
        neighbor_local_amplitude,
        save_anomaly_plot,
    )

    assert list(inspect.signature(extract_events_opencv).parameters)[:2] == [
        "score_map",
        "temperatures",
    ]
    assert "events" not in inspect.signature(extract_events_opencv).parameters
    bare = parse_args(["4.csv"])
    assert bare.opencv_score_percentile == 95
    assert bare.opencv_contrast_min == 1.2
    assert bare.percentile == 99.2
    explicit = parse_args(
        [
            "4.csv",
            "--opencv-score-percentile",
            "90",
            "--opencv-contrast-min",
            "2.5",
            "--percentile",
            "99.2",
            "--use-opencv",
            "--no-opencv",
        ]
    )
    assert explicit.opencv_score_percentile == 90.0
    assert explicit.opencv_contrast_min == 2.5
    assert explicit.percentile == 99.2
    assert explicit.use_opencv is False

    def maps(both_delta: float):
        score = np.zeros((40, 80), dtype=np.float64)
        temps = np.zeros((40, 80), dtype=np.float64)
        score[2:22, 2:22] = 10.0
        temps[2:8, 50:56] = 5.0
        score[28:34, 50:56] = 10.0
        temps[28:34, 50:56] = both_delta
        return score, temps

    seen: list[np.ndarray] = []
    real_components = pipe.cv2.connectedComponentsWithStats

    def components(image, *args, **kwargs):
        seen.append(np.array(image, copy=True))
        return real_components(image, *args, **kwargs)

    monkeypatch.setattr(pipe.cv2, "connectedComponentsWithStats", components)
    score, temps = maps(1.2)
    pixel = np.zeros(temps.shape, dtype=bool)
    pixel[30, 52] = True
    assert neighbor_local_amplitude(temps, pixel, 30, 30, 52, 52) == 1.2
    detector = [
        {
            "type_hypothesis": LOCAL_HEATING,
            "distance_start_m": 94.0,
            "distance_end_m": 100.0,
            "duration_min": 352.0,
        }
    ]
    snapshot = copy.deepcopy(detector)
    found = extract_events_opencv(score, temps)
    assert detector == snapshot
    assert found
    assert all(item["t_range"][0] >= 20 and item["s_range"][0] >= 40 for item in found)
    assert any(item["s_range"][0] <= 52 <= item["s_range"][1] for item in found)
    image = seen[0]
    assert image.dtype == np.dtype(np.uint8)
    assert set(np.unique(image).tolist()) <= {0, 255}
    assert image[30, 52] == 255
    assert image[10, 10] == 0
    assert image[4, 52] == 0
    assert not np.array_equal(image, temps)

    shifted = temps + 22.0
    # 22 + 1.2 is 1.1999999999999993, under the cut. Use an exact 2 °C step.
    shifted[28:34, 50:56] = 24.0
    assert neighbor_local_amplitude(shifted, pixel, 30, 30, 52, 52) == 2.0
    assert extract_events_opencv(score, shifted)
    assert all(
        item["t_range"][0] >= 20 and item["s_range"][0] >= 40
        for item in extract_events_opencv(score, shifted)
    )
    score_low, temps_low = maps(1.1)
    assert neighbor_local_amplitude(temps_low, pixel, 30, 30, 52, 52) == 1.1
    assert extract_events_opencv(score_low, temps_low) == []
    honored = extract_events_opencv(score_low, temps_low, contrast_min_c=1.1)
    assert honored
    flat_score = np.zeros_like(score)
    varying = np.zeros_like(temps)
    varying[28:34, 50:56] = 5.0
    assert extract_events_opencv(flat_score, varying) == []
    assert extract_events_opencv(score, np.zeros_like(temps)) == []

    dists = np.arange(80, dtype=np.float64) * 0.25
    annotated = annotate_opencv_components(found, temps, dists, 2.0)
    dropped = annotated[0]
    kept = {
        "n_pixels": 200,
        "duration_min": 100.0,
        "meters": [94.0, 100.0],
        "t_range": (1, 6),
        "s_range": (10, 16),
        "type_hypothesis": "heating",
        "amplitude": 4.2,
        "source": "opencv",
    }
    filtered = filter_opencv_components([dropped, kept])
    assert dropped not in filtered
    assert filtered == [kept]
    kept_figures = []

    def _keep(figure=None, *args, **kwargs):
        if figure is not None:
            kept_figures.append(figure)

    monkeypatch.setattr(plt, "close", _keep)
    matrix = np.full((40, 80), 22.0, dtype=np.float64)
    save_anomaly_plot(
        matrix,
        np.zeros_like(matrix),
        [],
        str(tmp_path / "intersect.png"),
        mode="opencv",
        opencv_events=filtered,
    )
    legend = kept_figures[0].axes[0].get_legend()
    assert legend is not None
    text = " ".join(item.get_text() for item in legend.get_texts())
    assert "heating +4.2 C" in text
    assert "94.0-100.0 m" in text
    dropped_span = f"{dropped['meters'][0]:.1f}-{dropped['meters'][1]:.1f}"
    assert dropped_span not in text
    assert dropped_span != "94.0-100.0"


def test_opencv_defaults_on_and_no_opencv_turns_it_off() -> None:
    from main import parse_args

    bare = parse_args(["4.csv"])
    assert bare.use_opencv is True
    still_on = parse_args(["4.csv", "--use-opencv"])
    assert still_on.use_opencv is True
    off = parse_args(["4.csv", "--no-opencv"])
    assert off.use_opencv is False
    off_wins = parse_args(["4.csv", "--use-opencv", "--no-opencv"])
    assert off_wins.use_opencv is False


def test_opencv_writer_and_overlap_match(tmp_path: Path) -> None:
    """Overlap in meters and time is a match. A miss in either is only-OpenCV."""
    from main import _write_opencv_json, split_opencv_matches

    component = {
        "source": "opencv",
        "t_range": (4, 13),
        "s_range": (2, 5),
        "n_pixels": 37,
    }
    dists = np.array([0.0, 0.25, 10.0, 10.5, 11.0, 12.25], dtype=np.float64)
    step = 2.0
    dest = tmp_path / "sample_opencv.json"
    _write_opencv_json([component], dists, step, str(dest))
    saved = json.loads(dest.read_text(encoding="utf-8"))
    assert len(saved) == 1
    row = saved[0]
    s0, s1 = component["s_range"]
    t0, t1 = component["t_range"]
    assert row["meters"] == [round(float(dists[s0]), 4), round(float(dists[s1]), 4)]
    assert row["duration_min"] == round((t1 - t0 + 1) * step, 4)
    assert row["area"] == component["n_pixels"]
    assert row["t_range"] == [t0, t1]

    hit = {
        "meters": [11.0, 13.0],
        "t_range": [10, 20],
        "duration_min": row["duration_min"],
        "area": 40,
    }
    miss_meters = {
        "meters": [40.0, 42.0],
        "t_range": [4, 13],
        "duration_min": 20.0,
        "area": 80,
    }
    miss_time = {
        "meters": [10.0, 12.25],
        "t_range": [30, 40],
        "duration_min": 22.0,
        "area": 15,
    }
    detector = {"meters": [9.0, 12.0], "t_range": [8, 16]}
    matched, only = split_opencv_matches([row, hit, miss_meters, miss_time], [detector])
    assert row in matched and hit in matched
    assert miss_meters in only and miss_time in only
    assert len(matched) + len(only) == 4


def test_opencv_contrast_type_and_json_source(tmp_path: Path) -> None:
    """±0.5 °C is not heating or cooling. JSON records say who found the box."""
    from main import _write_events_json, _write_opencv_json
    from mgts_anomaly import opencv_type_from_temperatures

    background = 20.0
    heating = opencv_type_from_temperatures(np.full(6, background + 2.0), background)
    cooling = opencv_type_from_temperatures(np.full(6, background - 2.0), background)
    exactly = opencv_type_from_temperatures(np.array([background + 0.5]), background)
    inside_band = opencv_type_from_temperatures(np.array([background + 0.2]), background)
    cold_edge = opencv_type_from_temperatures(np.array([background - 0.5]), background)
    assert heating == "heating"
    assert cooling == "cooling"
    assert exactly == "unknown"
    assert exactly != "heating"
    assert inside_band == "unknown"
    assert cold_edge == "unknown"
    assert cold_edge != "cooling"

    event = {
        "distance_start_m": 94.0,
        "distance_end_m": 100.0,
        "duration_min": 352.0,
        "amplitude": 4.21,
        "local_amplitude": 5.17,
        "confidence": 0.81,
        "type_hypothesis": LOCAL_HEATING,
        "periodic": False,
        "t_range": (10, 186),
        "s_range": (100, 120),
    }
    event_path = tmp_path / "events.json"
    _write_events_json([event], str(event_path))
    saved_event = json.loads(event_path.read_text(encoding="utf-8"))[0]
    assert saved_event["source"] == "detector"

    component = {
        "source": "opencv",
        "t_range": (4, 13),
        "s_range": (2, 5),
        "n_pixels": 37,
        "type_hypothesis": heating,
    }
    dists = np.array([0.0, 0.25, 10.0, 10.5, 11.0, 12.25], dtype=np.float64)
    opencv_path = tmp_path / "opencv.json"
    _write_opencv_json([component], dists, 2.0, str(opencv_path))
    saved_opencv = json.loads(opencv_path.read_text(encoding="utf-8"))[0]
    assert saved_opencv["source"] == "opencv"
    assert saved_opencv["type_hypothesis"] == heating


def test_plot_writes_separate_model_opencv_and_comparison(tmp_path: Path, monkeypatch) -> None:
    """Shipped plot: model-only, OpenCV-only, and model above OpenCV."""
    import matplotlib.pyplot as plt
    from matplotlib.legend import Legend

    from mgts_anomaly import save_anomaly_plot

    kept = []

    def _keep(figure=None, *args, **kwargs):
        if figure is not None:
            kept.append(figure)

    monkeypatch.setattr(plt, "close", _keep)
    matrix = np.full((12, 24), 22.0, dtype=np.float64)
    score = np.zeros((12, 24), dtype=np.float32)
    event = {
        "t_range": (2, 8),
        "s_range": (3, 7),
        "type_hypothesis": LOCAL_HEATING,
        "local_amplitude": 2.4,
        "amplitude": 2.4,
        "confidence": 0.8,
        "distance_start_m": 10.0,
        "distance_end_m": 14.0,
    }
    component = {
        "t_range": (1, 6),
        "s_range": (12, 16),
        "meters": [94.0, 100.0],
        "amplitude": 4.2,
        "duration_min": 352.0,
        "source": "opencv",
        "type_hypothesis": "heating",
    }
    model_path = tmp_path / "model.png"
    opencv_path = tmp_path / "opencv.png"
    compare_path = tmp_path / "compare.png"
    save_anomaly_plot(matrix, score, [event], str(model_path))
    save_anomaly_plot(
        matrix,
        score,
        [event],
        str(opencv_path),
        mode="opencv",
        opencv_events=[component],
    )
    save_anomaly_plot(
        matrix,
        score,
        [event],
        str(compare_path),
        mode="compare",
        opencv_events=[component],
    )

    def legend_blocks(figure):
        found = []
        seen = set()
        for artist in figure.findobj(match=lambda item: isinstance(item, Legend)):
            if id(artist) in seen:
                continue
            seen.add(id(artist))
            title = artist.get_title().get_text()
            texts = [text.get_text() for text in artist.get_texts()]
            colors = []
            for handle in artist.legend_handles:
                color = handle.get_color()
                colors.append(color if isinstance(color, str) else tuple(color))
            found.append((title, texts, colors, artist.axes))
        return found

    def joined(blocks):
        return "\n".join(text for _title, texts, _colors, _axes in blocks for text in texts)

    model_blocks = legend_blocks(kept[0])
    model_titles = [title for title, _texts, _colors, _axes in model_blocks]
    model_text = joined(model_blocks)
    assert model_titles == ["top events"]
    assert "local_heating +2.4 C" in model_text
    assert "10.0-14.0 m" in model_text
    assert "94.0-100.0" not in model_text
    assert "OpenCV" not in model_titles

    opencv_blocks = legend_blocks(kept[1])
    opencv_titles = [title for title, _texts, _colors, _axes in opencv_blocks]
    opencv_text = joined(opencv_blocks)
    assert opencv_titles == ["OpenCV"]
    assert "heating +4.2 C" in opencv_text
    assert "94.0-100.0 m" in opencv_text
    assert "local_heating" not in opencv_text
    assert "10.0-14.0" not in opencv_text
    assert any(
        color == "magenta" or color == (1.0, 0.0, 1.0)
        for _title, _texts, colors, _axes in opencv_blocks
        for color in colors
    )

    compare = kept[2]
    upper = compare.axes[0].get_legend()
    lower = compare.axes[1].get_legend()
    assert upper is not None and lower is not None
    assert upper.get_title().get_text() == "top events"
    assert lower.get_title().get_text() == "OpenCV"
    upper_text = " ".join(text.get_text() for text in upper.get_texts())
    lower_text = " ".join(text.get_text() for text in lower.get_texts())
    assert "local_heating +2.4 C" in upper_text
    assert "10.0-14.0 m" in upper_text
    assert "94.0-100.0" not in upper_text
    assert "heating +4.2 C" in lower_text
    assert "94.0-100.0 m" in lower_text
    assert "local_heating" not in lower_text
    for path in (model_path, opencv_path, compare_path):
        payload = path.read_bytes()
        assert payload[:8] == b"\x89PNG\r\n\x1a\n"
        assert len(payload) > 32


def test_opencv_filter_keeps_area_40_and_duration_4_min(tmp_path: Path, monkeypatch) -> None:
    """Shipped filter: area under 40 or duration under 4 min is dropped. The plot uses that result."""
    import inspect

    import matplotlib.pyplot as plt
    from matplotlib.legend import Legend

    from mgts_anomaly import filter_opencv_components, save_anomaly_plot

    assert list(inspect.signature(filter_opencv_components).parameters) == ["components"]
    small = {
        "n_pixels": 39,
        "duration_min": 10.0,
        "meters": [1.0, 2.0],
        "t_range": (1, 4),
        "s_range": (1, 3),
        "type_hypothesis": "unknown",
        "amplitude": 0.2,
        "source": "opencv",
    }
    short = {
        "n_pixels": 100,
        "duration_min": 3.0,
        "meters": [12.0, 13.0],
        "t_range": (2, 3),
        "s_range": (4, 8),
        "type_hypothesis": "heating",
        "amplitude": 1.0,
        "source": "opencv",
    }
    edge = {
        "n_pixels": 40,
        "duration_min": 4.0,
        "meters": [20.0, 21.0],
        "t_range": (0, 5),
        "s_range": (2, 6),
        "type_hypothesis": "cooling",
        "amplitude": -1.5,
        "source": "opencv",
    }
    large = {
        "n_pixels": 80,
        "duration_min": 10.0,
        "meters": [94.0, 100.0],
        "t_range": (1, 6),
        "s_range": (10, 16),
        "type_hypothesis": "heating",
        "amplitude": 4.2,
        "source": "opencv",
    }
    detector = [
        {
            "type_hypothesis": LOCAL_HEATING,
            "distance_start_m": 94.0,
            "distance_end_m": 100.0,
            "duration_min": 352.0,
        }
    ]
    untouched = list(detector)
    filtered = filter_opencv_components([small, short, edge, large])
    assert detector == untouched
    assert filtered == [edge, large]

    kept = []

    def _keep(figure=None, *args, **kwargs):
        if figure is not None:
            kept.append(figure)

    monkeypatch.setattr(plt, "close", _keep)
    matrix = np.full((12, 24), 22.0, dtype=np.float64)
    score = np.zeros((12, 24), dtype=np.float32)
    save_anomaly_plot(
        matrix,
        score,
        [],
        str(tmp_path / "unfiltered.png"),
        mode="opencv",
        opencv_events=[small, large],
    )
    save_anomaly_plot(
        matrix,
        score,
        [],
        str(tmp_path / "filtered.png"),
        mode="opencv",
        opencv_events=filtered,
    )
    legend = kept[1].axes[0].get_legend()
    assert legend is not None
    assert legend.get_title().get_text() == "OpenCV"
    text = " ".join(item.get_text() for item in legend.get_texts())
    assert "heating +4.2 C" in text
    assert "94.0-100.0 m" in text
    assert "12.0-13.0" not in text


def test_opencv_filter_uses_file_medians(tmp_path: Path, monkeypatch) -> None:
    """Shipped adaptive filter: floors, raised cuts, even median, and the legend of what remains."""
    import copy
    import inspect

    import matplotlib.pyplot as plt

    from mgts_anomaly import filter_opencv_components, save_anomaly_plot

    assert list(inspect.signature(filter_opencv_components).parameters) == ["components"]

    def box(area, duration, start_m, end_m, kind="heating", amplitude=2.0, use_area_key=False):
        component = {
            "duration_min": float(duration),
            "meters": [float(start_m), float(end_m)],
            "t_range": (0, 4),
            "s_range": (1, 3),
            "type_hypothesis": kind,
            "amplitude": amplitude,
            "source": "opencv",
        }
        if use_area_key:
            component["area"] = int(area)
        else:
            component["n_pixels"] = int(area)
        return component

    floor_small = box(39, 10, 1.0, 2.0, kind="unknown", amplitude=0.2)
    floor_short = box(100, 3, 12.0, 13.0, kind="heating", amplitude=1.0)
    floor_edge = box(40, 4, 20.0, 21.0, kind="cooling", amplitude=-1.5)
    floor_kept = box(80, 10, 94.0, 100.0, kind="heating", amplitude=4.2)
    floor_kept["t_range"] = (1, 6)
    floor_kept["s_range"] = (10, 16)
    detector = [
        {
            "type_hypothesis": LOCAL_HEATING,
            "distance_start_m": 94.0,
            "distance_end_m": 100.0,
            "duration_min": 352.0,
        }
    ]
    snapshot = copy.deepcopy(detector)
    floor = filter_opencv_components([floor_small, floor_short, floor_edge, floor_kept])
    assert floor == [floor_edge, floor_kept]

    raised_area = [
        box(40, 100, 0.0, 1.0),
        box(49, 100, 2.0, 3.0),
        box(50, 100, 4.0, 5.0),
        box(200, 100, 6.0, 7.0),
        box(200, 100, 8.0, 9.0),
        box(200, 100, 10.0, 11.0),
        box(200, 100, 14.0, 15.0),
    ]
    assert filter_opencv_components(raised_area) == raised_area[2:]

    raised_duration = [
        box(100, 4, 0.0, 1.0),
        box(100, 11, 2.0, 3.0),
        box(100, 12, 4.0, 5.0),
        box(100, 40, 6.0, 7.0),
        box(100, 40, 8.0, 9.0),
        box(100, 40, 10.0, 11.0),
        box(100, 40, 14.0, 15.0),
    ]
    assert filter_opencv_components(raised_duration) == raised_duration[2:]

    # Two extra 400s keep the central pair at 200 and 400 after 74 and 75 join the list.
    even_median = [
        box(74, 20, 0.0, 1.0),
        box(75, 20, 30.0, 31.0, use_area_key=True),
        box(200, 20, 2.0, 3.0),
        box(200, 20, 4.0, 5.0),
        box(400, 20, 6.0, 7.0),
        box(400, 20, 8.0, 9.0),
        box(400, 20, 10.0, 11.0),
        box(400, 20, 14.0, 15.0),
    ]
    assert filter_opencv_components(even_median) == even_median[1:]
    assert detector == snapshot

    kept = []

    def _keep(figure=None, *args, **kwargs):
        if figure is not None:
            kept.append(figure)

    monkeypatch.setattr(plt, "close", _keep)
    matrix = np.full((12, 24), 22.0, dtype=np.float64)
    score = np.zeros((12, 24), dtype=np.float32)
    save_anomaly_plot(
        matrix,
        score,
        [],
        str(tmp_path / "adaptive.png"),
        mode="opencv",
        opencv_events=floor,
    )
    legend = kept[0].axes[0].get_legend()
    assert legend is not None
    text = " ".join(item.get_text() for item in legend.get_texts())
    assert "heating +4.2 C" in text
    assert "94.0-100.0 m" in text
    assert "12.0-13.0" not in text
    assert "1.0-2.0" not in text


def test_opencv_row_fill_drops_gaps_that_close_joins(tmp_path: Path, monkeypatch) -> None:
    """Calm scores use fill 0.6. Noisy scores use fill 0.45. An exact hit stays."""
    import copy
    import inspect

    import matplotlib.pyplot as plt

    from mgts_anomaly import (
        annotate_opencv_components,
        extract_events_opencv,
        filter_opencv_components,
        filter_opencv_row_fill,
        retype_opencv_components,
        save_anomaly_plot,
    )

    def score_mad(values: np.ndarray) -> float:
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        median = float(np.median(finite))
        return float(np.median(np.abs(finite - median)))

    assert list(inspect.signature(filter_opencv_row_fill).parameters) == [
        "components",
        "score",
    ]
    # The map stays large so the hot cells are under 5% and the score
    # percentile remains on the background. A smaller map changes the mask.
    n_time, n_space = 60, 120
    score = np.zeros((n_time, n_space), dtype=np.float64)
    temps = np.zeros((n_time, n_space), dtype=np.float64)
    hot = 4.0
    # Checkerboard minus one cell: the shipped 16-sample mask reports fill 0.6.
    for row in range(5):
        for col in range(20):
            if (row + col) % 2 == 0 and (row, col) != (0, 18):
                score[row, col] = 10.0
                temps[row, col] = hot
    # 45 cells in a 5 by 20 rectangle: the shipped mask reports fill 0.45.
    for col in range(20, 29):
        score[10, col] = 10.0
        temps[10, col] = hot
    for col in range(28, 37):
        score[11, col] = 10.0
        temps[11, col] = hot
    for col in range(32, 40):
        score[12, col] = 10.0
        temps[12, col] = hot
    for col in range(32, 40):
        score[13, col] = 10.0
        temps[13, col] = hot
    for col in range(20, 31):
        score[14, col] = 10.0
        temps[14, col] = hot
    # A 4-wide stroke on every row, stepped sideways. The bounding rectangle
    # is mostly empty, so the share stays under 0.45.
    for offset, row in enumerate(range(20, 40)):
        col = 50 + offset
        score[row, col : col + 4] = 10.0
        temps[row, col : col + 4] = hot
    dists = np.arange(n_space, dtype=np.float64)
    found = extract_events_opencv(score, temps)
    annotate_opencv_components(found, temps, dists, 2.0)
    solid = next(item for item in found if item["t_range"] == (0, 4))
    exact = next(item for item in found if item["t_range"] == (10, 14))
    streak = next(item for item in found if item["t_range"] == (20, 39))
    assert solid["row_fill"] == 0.6
    assert solid["duration_min"] == 10.0
    assert exact["row_fill"] == 0.45
    assert exact["duration_min"] == 10.0
    assert streak["n_pixels"] >= 40
    assert streak["duration_min"] >= 4.0
    assert streak["s_range"][1] - streak["s_range"][0] + 1 > 4
    assert streak["row_fill"] < 0.45
    area_kept = filter_opencv_components(found)
    assert len(area_kept) == 3
    assert score_mad(score) <= 0.04
    boundary = np.concatenate([np.zeros(80), np.full(80, 0.08)])
    noisy = np.concatenate([np.zeros(80), np.full(80, 0.1)])
    assert score_mad(boundary) == 0.04
    assert score_mad(noisy) > 0.04
    detector = [
        {
            "type_hypothesis": LOCAL_HEATING,
            "distance_start_m": 94.0,
            "distance_end_m": 100.0,
            "duration_min": 352.0,
        }
    ]
    snapshot = copy.deepcopy(detector)
    calm = filter_opencv_row_fill(area_kept, score)
    assert filter_opencv_row_fill(area_kept, boundary) == [solid]
    noisy_kept = filter_opencv_row_fill(area_kept, noisy)
    assert detector == snapshot
    assert calm == [solid]
    assert exact not in calm
    assert streak not in calm
    assert noisy_kept == [solid, exact]
    assert streak not in noisy_kept
    typed = retype_opencv_components(calm)
    assert len(typed) == 1
    assert typed[0]["duration_min"] == 10.0
    dropped = f"{streak['meters'][0]:.1f}-{streak['meters'][1]:.1f}"
    kept_span = f"{typed[0]['meters'][0]:.1f}-{typed[0]['meters'][1]:.1f}"
    exact_span = f"{exact['meters'][0]:.1f}-{exact['meters'][1]:.1f}"
    assert dropped != kept_span
    assert exact_span != kept_span

    figures = []

    def _keep(figure=None, *args, **kwargs):
        if figure is not None:
            figures.append(figure)

    monkeypatch.setattr(plt, "close", _keep)
    save_anomaly_plot(
        temps,
        score.astype(np.float32),
        [],
        str(tmp_path / "row_fill.png"),
        mode="opencv",
        opencv_events=typed,
    )
    legend = figures[0].axes[0].get_legend()
    assert legend is not None
    text = " ".join(item.get_text() for item in legend.get_texts())
    assert kept_span in text
    assert exact_span not in text
    assert dropped not in text


def test_opencv_calm_wide_ring_covers_plateau_noisy_ring_does_not() -> None:
    """Calm files use the wide ring. Noisy files stay on 16 samples.

    The score percentile and the 1.2 °C cut still both have to pass.
    """
    import mgts_anomaly as pipe
    from mgts_anomaly import OPENCV_CALM_HALF_POINTS, NEIGHBOR_HALF_POINTS, collect_opencv_components

    n_time, n_space = 100, 200
    temps = np.zeros((n_time, n_space), dtype=np.float64)
    temps[20:80, 40:90] = 3.0
    dists = np.arange(n_space, dtype=np.float64)
    calm_score = np.zeros((n_time, n_space), dtype=np.float64)
    calm_score[20:80, 40:90] = 1.0
    noisy_score = np.random.default_rng(0).normal(0.0, 0.2, size=(n_time, n_space))
    noisy_score[20:80, 40:90] = 5.0
    seen: list[int] = []
    real_delta = pipe._neighbor_ring_delta

    def _record(values, half_points=None):
        seen.append(NEIGHBOR_HALF_POINTS if half_points is None else int(half_points))
        return real_delta(values, half_points=half_points)

    pipe._neighbor_ring_delta = _record
    try:
        calm = collect_opencv_components(calm_score, temps, temps, dists, 2.0)
        calm_widths = list(seen)
        seen.clear()
        noisy = collect_opencv_components(noisy_score, temps, temps, dists, 2.0)
        noisy_widths = list(seen)
    finally:
        pipe._neighbor_ring_delta = real_delta

    def _covers_interior(components):
        return [
            item
            for item in components
            if item["s_range"][0] <= 65 <= item["s_range"][1]
            and item["t_range"][0] <= 50 <= item["t_range"][1]
        ]

    assert calm_widths[0] == OPENCV_CALM_HALF_POINTS
    assert OPENCV_CALM_HALF_POINTS > NEIGHBOR_HALF_POINTS
    assert NEIGHBOR_HALF_POINTS in calm_widths
    assert noisy_widths == [NEIGHBOR_HALF_POINTS]
    covered = _covers_interior(calm)
    assert len(covered) == 1
    assert covered[0]["row_fill"] >= 0.6
    assert covered[0]["duration_min"] >= 90.0
    assert _covers_interior(noisy) == []

    below = np.full((n_time, n_space), 2.0)
    below[20:80, 40:90] = 0.0
    assert _covers_interior(collect_opencv_components(below, temps, temps, dists, 2.0)) == []
    mild = np.zeros_like(temps)
    mild[20:80, 40:90] = 1.1
    assert _covers_interior(collect_opencv_components(calm_score, mild, mild, dists, 2.0)) == []


def test_opencv_calm_drop_removes_a_short_box_inside_a_long_one() -> None:
    """A calm file drops a short box nested in a long box. A noisy file keeps it."""
    from mgts_anomaly import _ensemble_score_mad, collect_opencv_components

    n_time, n_space = 160, 160
    temps = np.zeros((n_time, n_space), dtype=np.float64)
    blocks = (
        (0, 110, 10, 21),
        # More than 30 min after the long box, so the near merge leaves it
        # and the assertion still measures the calm fragment drop.
        (140, 160, 12, 19),
        (0, 40, 40, 51),
        (0, 40, 70, 81),
        (0, 40, 100, 111),
        (0, 40, 130, 141),
    )
    for t0, t1, s0, s1 in blocks:
        temps[t0:t1, s0:s1] = 4.0
    calm_score = np.zeros((n_time, n_space), dtype=np.float64)
    calm_score[temps > 0] = 10.0
    noisy_score = np.random.default_rng(1).normal(0.0, 0.25, size=(n_time, n_space))
    noisy_score[temps > 0] = 8.0
    dists = np.arange(n_space, dtype=np.float64)
    assert _ensemble_score_mad(calm_score) <= 0.04
    assert _ensemble_score_mad(noisy_score) > 0.04
    calm = collect_opencv_components(calm_score, temps, temps, dists, 2.0)
    noisy = collect_opencv_components(noisy_score, temps, temps, dists, 2.0)

    def _nested_short(components):
        return [
            item
            for item in components
            if float(item["duration_min"]) < 90.0
            and float(item["meters"][0]) >= 10.0
            and float(item["meters"][1]) <= 20.0
        ]

    assert any(float(item["duration_min"]) >= 200.0 for item in calm)
    assert _nested_short(calm) == []
    assert _nested_short(noisy)


def test_opencv_retype_drops_contrast_under_1_5_regardless_of_area(
    tmp_path: Path, monkeypatch
) -> None:
    """A large long box under 1.2 °C is omitted. Exactly ±1.2 °C stays typed."""
    import copy
    import inspect

    import matplotlib.pyplot as plt

    from mgts_anomaly import (
        annotate_opencv_components,
        filter_opencv_components,
        neighbor_local_amplitude,
        retype_opencv_components,
        save_anomaly_plot,
    )

    assert list(inspect.signature(retype_opencv_components).parameters) == ["components"]
    n_time, n_space = 50, 200
    temps = np.zeros((n_time, n_space), dtype=np.float64)
    # Identical rectangles, far enough apart that each ring stays on the background.
    # 31 by 19 keeps the neighbor mean at exactly ±1.2. 31 by 21 sums to
    # 1.1999999999999997, which the strict cut drops.
    spans = {
        "weak": (10, 40, 20, 38, 1.0),
        "hot": (10, 40, 80, 98, 1.2),
        "cold": (10, 40, 140, 158, -1.2),
    }
    for t0, t1, s0, s1, delta in spans.values():
        temps[t0 : t1 + 1, s0 : s1 + 1] = delta
    dists = np.zeros(n_space, dtype=np.float64)
    dists[20], dists[38] = 12.0, 20.0
    dists[80], dists[98] = 94.0, 100.0
    dists[140], dists[158] = 106.0, 113.0

    def ring(t0: int, t1: int, s0: int, s1: int) -> float:
        mask = np.zeros(temps.shape, dtype=bool)
        mask[t0 : t1 + 1, s0 : s1 + 1] = True
        return neighbor_local_amplitude(temps, mask, t0, t1, s0, s1)

    assert ring(10, 40, 20, 38) == 1.0
    assert ring(10, 40, 80, 98) == 1.2
    assert ring(10, 40, 140, 158) == -1.2
    raw = []
    for t0, t1, s0, s1, _delta in spans.values():
        raw.append(
            {
                "source": "opencv",
                "t_range": (t0, t1),
                "s_range": (s0, s1),
                "n_pixels": (t1 - t0 + 1) * (s1 - s0 + 1),
            }
        )
    annotated = annotate_opencv_components(raw, temps, dists, 2.0)
    assert [item["amplitude"] for item in annotated] == [1.0, 1.2, -1.2]
    assert all(item["n_pixels"] == 589 for item in annotated)
    assert all(item["duration_min"] == 62.0 for item in annotated)
    filtered = filter_opencv_components(annotated)
    assert len(filtered) == 3
    detector = [
        {
            "type_hypothesis": LOCAL_HEATING,
            "distance_start_m": 94.0,
            "distance_end_m": 100.0,
            "duration_min": 352.0,
        }
    ]
    snapshot = copy.deepcopy(detector)
    typed = retype_opencv_components(filtered)
    assert detector == snapshot
    assert len(typed) == 2
    assert typed[0]["type_hypothesis"] == "heating"
    assert typed[0]["amplitude"] == 1.2
    assert typed[0]["meters"] == [94.0, 100.0]
    assert typed[1]["type_hypothesis"] == "cooling"
    assert typed[1]["amplitude"] == -1.2
    assert typed[1]["meters"] == [106.0, 113.0]
    assert all(item["meters"] != [12.0, 20.0] for item in typed)

    kept = []

    def _keep(figure=None, *args, **kwargs):
        if figure is not None:
            kept.append(figure)

    monkeypatch.setattr(plt, "close", _keep)
    matrix = np.zeros((n_time, n_space), dtype=np.float64)
    score = np.zeros((n_time, n_space), dtype=np.float32)
    save_anomaly_plot(
        matrix,
        score,
        [],
        str(tmp_path / "hard_contrast.png"),
        mode="opencv",
        opencv_events=typed,
    )
    legend = kept[0].axes[0].get_legend()
    assert legend is not None
    text = " ".join(item.get_text() for item in legend.get_texts())
    assert "heating +1.2 C" in text
    assert "cooling -1.2 C" in text
    assert "94.0-100.0 m" in text
    assert "106.0-113.0 m" in text
    assert "12.0-20.0" not in text


def test_opencv_merge_joins_half_overlap_on_both_axes(tmp_path: Path, monkeypatch) -> None:
    """Shipped merge: half overlap on both axes joins; one axis, or the merged hull, does not."""
    import inspect
    import json

    import matplotlib.pyplot as plt

    from main import _write_opencv_json
    from mgts_anomaly import merge_opencv_components, save_anomaly_plot

    assert list(inspect.signature(merge_opencv_components).parameters) == ["components"]

    def box(t0, t1, s0, s1, start_m, end_m, kind, pixels, amplitude=2.0):
        steps = t1 - t0 + 1
        return {
            "t_range": (t0, t1),
            "s_range": (s0, s1),
            "meters": [float(start_m), float(end_m)],
            "distance_start_m": float(start_m),
            "distance_end_m": float(end_m),
            "duration_min": float(steps * 2.0),
            "type_hypothesis": kind,
            "n_pixels": pixels,
            "area": pixels,
            "amplitude": amplitude,
            "source": "opencv",
        }

    left = box(0, 9, 0, 10, 0.0, 10.0, "heating", 40)
    right = box(5, 14, 5, 15, 5.0, 15.0, "heating", 50)
    detector = [{"type_hypothesis": LOCAL_HEATING, "meters": [94.0, 100.0]}]
    untouched = list(detector)
    joined = merge_opencv_components([left, right])
    assert detector == untouched
    assert len(joined) == 1
    assert joined[0]["meters"] == [0.0, 15.0]
    assert joined[0]["duration_min"] == 30.0
    assert joined[0]["n_pixels"] == 90
    assert joined[0]["area"] == 90
    assert joined[0]["type_hypothesis"] == "heating"
    assert joined[0]["t_range"] == (0, 14)
    assert joined[0]["s_range"] == (0, 15)

    time_only_b = box(5, 14, 100, 110, 100.0, 110.0, "heating", 40)
    stayed = merge_opencv_components([left, time_only_b])
    assert stayed == [left, time_only_b]
    distance_only_b = box(100, 109, 5, 15, 5.0, 15.0, "heating", 40)
    stayed_distance = merge_opencv_components([left, distance_only_b])
    assert stayed_distance == [left, distance_only_b]

    chain_a = box(0, 9, 0, 10, 0.0, 10.0, "heating", 40)
    chain_b = box(5, 14, 5, 15, 5.0, 15.0, "heating", 40)
    chain_c = box(10, 19, 10, 20, 10.0, 20.0, "heating", 40)
    outsider = box(10, 19, 0, 5, 0.0, 5.0, "cooling", 40, amplitude=-1.0)
    clustered = merge_opencv_components([chain_a, chain_b, chain_c, outsider])
    assert len(clustered) == 2
    assert clustered[1] is outsider
    assert clustered[0]["meters"] == [0.0, 20.0]
    assert clustered[0]["duration_min"] == 40.0
    assert clustered[0]["duration_min"] != 60.0
    assert clustered[0]["area"] == 120
    assert clustered[0]["type_hypothesis"] == "heating"
    assert outsider["meters"] == [0.0, 5.0]
    assert outsider["type_hypothesis"] == "cooling"

    hot = box(0, 9, 0, 10, 0.0, 10.0, "heating", 40)
    cold = box(5, 14, 5, 15, 5.0, 15.0, "cooling", 40, amplitude=-1.2)
    mixed = merge_opencv_components([hot, cold])
    assert len(mixed) == 1
    assert mixed[0]["type_hypothesis"] == "unknown"

    alone = box(0, 9, 30, 34, 30.0, 34.0, "heating", 40)
    assert merge_opencv_components([alone]) == [alone]
    assert merge_opencv_components([alone])[0] is alone

    kept = []

    def _keep(figure=None, *args, **kwargs):
        if figure is not None:
            kept.append(figure)

    monkeypatch.setattr(plt, "close", _keep)
    matrix = np.full((20, 40), 22.0, dtype=np.float64)
    score = np.zeros((20, 40), dtype=np.float32)
    save_anomaly_plot(
        matrix,
        score,
        [],
        str(tmp_path / "merged.png"),
        mode="opencv",
        opencv_events=joined,
    )
    legend = kept[0].axes[0].get_legend()
    assert legend is not None
    legend_text = " ".join(item.get_text() for item in legend.get_texts())
    assert "heating +2.0 C" in legend_text
    assert "0.0-15.0 m" in legend_text
    assert "0.0-10.0" not in legend_text
    assert "5.0-15.0" not in legend_text

    dists = np.arange(21, dtype=np.float64)
    dest = tmp_path / "merged.json"
    _write_opencv_json(joined, dists, 2.0, str(dest))
    saved = json.loads(dest.read_text(encoding="utf-8"))
    assert len(saved) == 1
    assert saved[0]["source"] == "opencv"
    assert saved[0]["type_hypothesis"] == "heating"
    assert saved[0]["meters"] == [0.0, 15.0]
    assert saved[0]["area"] == 90
    assert saved[0]["duration_min"] == 30.0
    assert saved[0]["t_range"] == [0, 14]
    assert saved[0]["s_range"] == [0, 15]


def test_opencv_overlap_flag_uses_30_percent_on_both_axes(tmp_path: Path, monkeypatch) -> None:
    """Shipped 30% flag: both axes, JSON boolean, dashed overlap and solid unique outline."""
    import copy

    import matplotlib.pyplot as plt

    from main import _write_opencv_json
    from mgts_anomaly import mark_opencv_detector_overlaps, save_anomaly_plot

    def box(t0, t1, start_m, end_m, s0, s1):
        steps = t1 - t0 + 1
        return {
            "t_range": (t0, t1),
            "s_range": (s0, s1),
            "meters": [float(start_m), float(end_m)],
            "duration_min": float(steps * 2.0),
            "type_hypothesis": "heating",
            "n_pixels": 50,
            "area": 50,
            "amplitude": 2.0,
            "source": "opencv",
        }

    def detector(t0, t1, start_m, end_m):
        steps = t1 - t0 + 1
        return {
            "t_range": (t0, t1),
            "s_range": (0, 4),
            "meters": [float(start_m), float(end_m)],
            "distance_start_m": float(start_m),
            "distance_end_m": float(end_m),
            "duration_min": float(steps * 2.0),
            "type_hypothesis": LOCAL_HEATING,
            "local_amplitude": 5.0,
            "confidence": 0.8,
        }

    # Detector is the longer, wider box. 3 steps and 3 m are 30% of the smaller side.
    reference = detector(0, 19, 0.0, 20.0)
    both = box(17, 26, 17.0, 27.0, 0, 5)
    time_exact_meters_under = box(17, 26, 17.1, 27.1, 6, 10)
    meters_exact_time_under = box(18, 27, 17.0, 27.0, 11, 15)
    time_only = box(17, 26, 100.0, 110.0, 16, 20)
    meters_only = box(100, 109, 17.0, 27.0, 21, 25)
    missed = detector(200, 219, 200.0, 220.0)
    second_hit = detector(300, 319, 300.0, 320.0)
    hit_by_second = box(317, 326, 317.0, 327.0, 26, 30)
    zero_width = detector(17, 26, 50.0, 50.0)
    touches_zero_width = box(17, 26, 50.0, 60.0, 31, 35)

    events = [reference, missed, second_hit, zero_width]
    snapshot = copy.deepcopy(events)
    labeled = mark_opencv_detector_overlaps(
        [
            both,
            time_exact_meters_under,
            meters_exact_time_under,
            time_only,
            meters_only,
            hit_by_second,
            touches_zero_width,
        ],
        events,
    )
    assert events == snapshot
    assert [component["overlaps_detector"] for component in labeled] == [
        True,
        False,
        False,
        False,
        False,
        True,
        False,
    ]
    assert all(isinstance(component["overlaps_detector"], bool) for component in labeled)

    miss_only = copy.deepcopy(hit_by_second)
    miss_events = [reference, missed]
    miss_snapshot = copy.deepcopy(miss_events)
    missed_label = mark_opencv_detector_overlaps([miss_only], miss_events)
    assert miss_events == miss_snapshot
    assert missed_label[0]["overlaps_detector"] is False

    dists = np.arange(40, dtype=np.float64)
    dest = tmp_path / "overlap.json"
    _write_opencv_json(labeled, dists, 2.0, str(dest))
    saved = json.loads(dest.read_text(encoding="utf-8"))
    assert len(saved) == len(labeled)
    for record, component in zip(saved, labeled):
        assert record["source"] == "opencv"
        assert isinstance(record["overlaps_detector"], bool)
        assert record["overlaps_detector"] is component["overlaps_detector"]

    kept = []

    def _keep(figure=None, *args, **kwargs):
        if figure is not None:
            kept.append(figure)

    monkeypatch.setattr(plt, "close", _keep)
    matrix = np.full((340, 40), 22.0, dtype=np.float64)
    score = np.zeros((340, 40), dtype=np.float32)
    drawn = [both, meters_only]
    save_anomaly_plot(
        matrix,
        score,
        [],
        str(tmp_path / "overlap_opencv.png"),
        mode="opencv",
        opencv_events=drawn,
    )
    save_anomaly_plot(
        matrix,
        score,
        [reference],
        str(tmp_path / "overlap_compare.png"),
        mode="compare",
        opencv_events=drawn,
    )
    save_anomaly_plot(
        matrix,
        score,
        [reference],
        str(tmp_path / "overlap_model.png"),
        mode="model",
        opencv_events=drawn,
    )

    def _styles(axis):
        found = []
        for line in axis.get_lines():
            color = line.get_color()
            if color not in {"magenta", (1.0, 0.0, 1.0)}:
                continue
            xs = [float(value) for value in line.get_xdata()]
            found.append((xs[0], line.get_linestyle()))
        return found

    def _assert_outlines(axis):
        styles = dict(_styles(axis))
        assert styles[17.0] in {"--", "dashed"}
        assert styles[100.0] in {"-", "solid"}

    opencv_figure, compare_figure, model_figure = kept
    _assert_outlines(opencv_figure.axes[0])
    _assert_outlines(opencv_figure.axes[1])
    _assert_outlines(compare_figure.axes[1])
    assert _styles(compare_figure.axes[0]) == []
    assert _styles(model_figure.axes[0]) == []
    assert _styles(model_figure.axes[1]) == []
    legend = opencv_figure.axes[0].get_legend()
    assert legend is not None
    legend_text = " ".join(item.get_text() for item in legend.get_texts())
    assert "1. heating +2.0 C" in legend_text
    assert "17.0-27.0 m" in legend_text


def test_combined_list_marks_both_on_the_shipped_30_percent_rule(tmp_path: Path) -> None:
    """Exact 30% on both axes is one both row. One axis under 30% stays two rows."""
    import copy

    from main import _write_combined_json, _write_events_json, _write_opencv_json
    from mgts_anomaly import combine_detector_and_opencv, mark_opencv_detector_overlaps

    def box(t0, t1, start_m, end_m, s0, s1):
        steps = t1 - t0 + 1
        return {
            "t_range": (t0, t1),
            "s_range": (s0, s1),
            "meters": [float(start_m), float(end_m)],
            "distance_start_m": float(start_m),
            "distance_end_m": float(end_m),
            "duration_min": float(steps * 2.0),
            "type_hypothesis": "heating",
            "n_pixels": 50,
            "area": 50,
            "amplitude": 2.0,
            "source": "opencv",
        }

    def detector(t0, t1, start_m, end_m, s0, s1):
        steps = t1 - t0 + 1
        return {
            "t_range": (t0, t1),
            "s_range": (s0, s1),
            "meters": [float(start_m), float(end_m)],
            "distance_start_m": float(start_m),
            "distance_end_m": float(end_m),
            "duration_min": float(steps * 2.0),
            "amplitude": 4.0,
            "local_amplitude": 5.0,
            "confidence": 0.8,
            "periodic": False,
            "type_hypothesis": LOCAL_HEATING,
        }

    # 3 steps and 3 m are 30% of the smaller side. 2.9 m is under 30%.
    reference = detector(0, 19, 0.0, 20.0, 0, 4)
    untouched = detector(200, 219, 200.0, 220.0, 5, 9)
    exact = box(17, 26, 17.0, 27.0, 0, 5)
    one_axis = box(17, 26, 17.1, 27.1, 6, 10)
    events = [reference, untouched]
    components = [exact, one_axis]
    event_snapshot = copy.deepcopy(events)
    labeled = mark_opencv_detector_overlaps(components, events)
    assert events == event_snapshot
    assert [item["overlaps_detector"] for item in labeled] == [True, False]
    labeled_snapshot = copy.deepcopy(labeled)
    combined = combine_detector_and_opencv(events, labeled)
    assert events == event_snapshot
    assert labeled == labeled_snapshot
    assert [item["source"] for item in combined] == ["both", "detector", "opencv"]
    both_row, detector_row, opencv_row = combined
    assert both_row["meters"] == [0.0, 20.0]
    assert both_row["duration_min"] == 40.0
    assert both_row["type_hypothesis"] == LOCAL_HEATING
    assert "overlaps_detector" not in both_row
    assert "overlaps_detector" not in detector_row
    assert detector_row["meters"] == [200.0, 220.0]
    assert opencv_row["meters"] == [17.1, 27.1]
    assert opencv_row["t_range"] == one_axis["t_range"]
    unique = sum(1 for item in labeled if item["overlaps_detector"] is False)
    assert len(combined) == len(events) + unique

    dists = np.arange(40, dtype=np.float64)
    _write_events_json(events, str(tmp_path / "events.json"))
    _write_opencv_json(labeled, dists, 2.0, str(tmp_path / "opencv.json"))
    _write_combined_json(events, labeled, dists, 2.0, str(tmp_path / "combined.json"))
    saved_events = json.loads((tmp_path / "events.json").read_text(encoding="utf-8"))
    saved_opencv = json.loads((tmp_path / "opencv.json").read_text(encoding="utf-8"))
    saved_combined = json.loads((tmp_path / "combined.json").read_text(encoding="utf-8"))
    assert [item["source"] for item in saved_events] == ["detector", "detector"]
    assert all("overlaps_detector" not in item for item in saved_events)
    assert [item["source"] for item in saved_opencv] == ["opencv", "opencv"]
    assert [item["overlaps_detector"] for item in saved_opencv] == [True, False]
    assert [item["source"] for item in saved_combined] == ["both", "detector", "opencv"]
    assert saved_combined[0]["meters"] == [0.0, 20.0]
    assert saved_combined[0]["duration_min"] == 40.0
    assert saved_combined[0]["type_hypothesis"] == LOCAL_HEATING
    assert "overlaps_detector" not in saved_combined[0]
    assert len(saved_combined) == len(saved_events) + sum(
        1 for item in saved_opencv if item["overlaps_detector"] is False
    )


def test_opencv_nearby_merge_joins_same_sign_within_2m_and_30min() -> None:
    """Shipped near merge: exact 2 m and 30 min join; one axis, opposite signs, and the hull do not."""
    import inspect

    from mgts_anomaly import (
        collect_opencv_components,
        merge_opencv_nearby,
        run_pipeline,
    )

    assert list(inspect.signature(merge_opencv_nearby).parameters) == ["components"]
    collect_source = inspect.getsource(collect_opencv_components)
    assert collect_source.index("merge_opencv_components") < collect_source.index(
        "drop_opencv_fragments_covered_by_long_box"
    )
    assert collect_source.index("drop_opencv_fragments_covered_by_long_box") < collect_source.index(
        "merge_opencv_nearby"
    )
    pipeline_source = inspect.getsource(run_pipeline)
    assert pipeline_source.index("collect_opencv_components") < pipeline_source.index(
        "mark_opencv_detector_overlaps"
    )

    def box(t0, t1, start_m, end_m, kind, amplitude, s0=0, s1=4):
        steps = t1 - t0 + 1
        return {
            "t_range": (t0, t1),
            "s_range": (s0, s1),
            "meters": [float(start_m), float(end_m)],
            "distance_start_m": float(start_m),
            "distance_end_m": float(end_m),
            "duration_min": float(steps * 2.0),
            "type_hypothesis": kind,
            "n_pixels": 40,
            "area": 40,
            "amplitude": amplitude,
            "source": "opencv",
        }

    # 15 index holes at 2 min is 30 min. Meter gap 12 - 10 is 2 m.
    left = box(0, 9, 0.0, 10.0, "heating", 2.0)
    exact = box(25, 34, 12.0, 22.0, "heating", 1.5, s0=6, s1=10)
    joined = merge_opencv_nearby([left, exact])
    assert len(joined) == 1
    assert joined[0]["meters"] == [0.0, 22.0]
    assert joined[0]["duration_min"] == 70.0
    assert joined[0]["duration_min"] != 40.0
    assert joined[0]["type_hypothesis"] == "heating"
    assert joined[0]["t_range"] == (0, 34)
    assert joined[0]["n_pixels"] == 80
    assert left["meters"] == [0.0, 10.0]
    assert exact["duration_min"] == 20.0

    time_over = box(26, 35, 12.0, 22.0, "heating", 1.5)
    assert merge_opencv_nearby([left, time_over]) == [left, time_over]
    meters_over = box(25, 34, 12.1, 22.1, "heating", 1.5)
    assert merge_opencv_nearby([left, meters_over]) == [left, meters_over]

    hot = box(0, 9, 0.0, 10.0, "heating", 2.0)
    cold = box(10, 19, 10.0, 20.0, "cooling", -2.0)
    stayed = merge_opencv_nearby([hot, cold])
    assert stayed == [hot, cold]
    assert stayed[0] is hot
    assert stayed[1] is cold

    chain_a = box(0, 9, 0.0, 10.0, "heating", 2.0)
    chain_b = box(0, 9, 12.0, 20.0, "heating", 2.0)
    chain_c = box(0, 9, 22.0, 28.0, "heating", 2.0)
    chained = merge_opencv_nearby([chain_a, chain_b, chain_c])
    assert len(chained) == 1
    assert chained[0]["meters"] == [0.0, 28.0]
    assert chained[0]["duration_min"] == 20.0
    assert chained[0]["type_hypothesis"] == "heating"

    # The corner meets the union of this chain and none of the input boxes.
    corner = box(34, 35, 0.0, 1.0, "heating", 2.0)
    hull_inputs = [
        box(0, 1, 0.0, 1.0, "heating", 2.0),
        box(0, 1, 3.0, 4.0, "heating", 2.0),
        box(0, 1, 6.0, 7.0, "heating", 2.0),
        box(17, 18, 6.0, 7.0, "heating", 2.0),
        box(34, 35, 6.0, 7.0, "heating", 2.0),
        corner,
    ]
    hull = merge_opencv_nearby(hull_inputs)
    assert len(hull) == 2
    assert corner in hull
    assert hull[0]["meters"] == [0.0, 7.0]
    assert hull[0]["duration_min"] == 72.0
    assert hull[0]["type_hypothesis"] == "heating"


def test_combined_plot_colors_sources_and_leaves_the_three_modes(tmp_path: Path, monkeypatch) -> None:
    """Shipped combined drawer: white, magenta, and green, with the three older modes unchanged."""
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    from mgts_anomaly import _anomaly_plot_paths, run_pipeline, save_anomaly_plot

    paths = _anomaly_plot_paths("anomaly_4.png")
    assert paths["combined"].endswith("anomaly_4_combined.png")
    assert paths["model"].endswith("anomaly_4.png")
    assert paths["opencv"].endswith("anomaly_4_opencv.png")
    assert paths["compare"].endswith("anomaly_4_compare.png")
    pipeline_source = inspect.getsource(run_pipeline)
    assert pipeline_source.index('mode="compare"') < pipeline_source.index('mode="combined"')
    assert pipeline_source.index("combine_detector_and_opencv") < pipeline_source.index('mode="combined"')
    assert pipeline_source.index('mode="combined"') < pipeline_source.index(
        "save_anomaly_plot(matrix, score, events, plot_path)"
    )

    def record(t0, t1, s0, s1, source):
        return {
            "t_range": (t0, t1),
            "s_range": (s0, s1),
            "meters": [float(s0), float(s1)],
            "distance_start_m": float(s0),
            "distance_end_m": float(s1),
            "source": source,
            "type_hypothesis": "heating",
            "duration_min": 20.0,
            "amplitude": 2.0,
            "local_amplitude": 2.0,
            "confidence": 0.5,
            "n_pixels": 40,
        }

    detector = record(0, 10, 0, 4, "detector")
    opencv = record(40, 50, 10, 14, "opencv")
    both = record(80, 90, 20, 24, "both")
    matrix = np.full((100, 30), 22.0, dtype=np.float64)
    score = np.zeros((100, 30), dtype=np.float32)
    kept = []

    def _keep(figure=None, *args, **kwargs):
        if figure is not None:
            kept.append(figure)

    monkeypatch.setattr(plt, "close", _keep)
    save_anomaly_plot(
        matrix,
        score,
        [detector],
        str(tmp_path / "combined.png"),
        max_boxes=1,
        mode="combined",
        combined_events=[detector, opencv, both],
    )
    save_anomaly_plot(matrix, score, [detector], str(tmp_path / "model.png"), mode="model")
    save_anomaly_plot(
        matrix,
        score,
        [detector],
        str(tmp_path / "opencv.png"),
        mode="opencv",
        opencv_events=[opencv],
    )
    save_anomaly_plot(
        matrix,
        score,
        [detector],
        str(tmp_path / "compare.png"),
        mode="compare",
        opencv_events=[opencv],
    )

    def rgb(color):
        return tuple(round(channel, 5) for channel in mcolors.to_rgb(color))

    white, magenta, green = rgb("white"), rgb("magenta"), rgb("green")

    def line_colors(axis):
        found = []
        for line in axis.get_lines():
            xs = [float(value) for value in line.get_xdata()]
            found.append((xs[0], rgb(line.get_color())))
        return found

    combined_figure, model_figure, opencv_figure, compare_figure = kept
    combined_lines = dict(line_colors(combined_figure.axes[0]))
    assert combined_lines[0.0] == white
    assert combined_lines[40.0] == magenta
    assert combined_lines[80.0] == green
    assert line_colors(combined_figure.axes[1]) == []
    assert len(combined_figure.axes[0].get_lines()) == 3
    legend = combined_figure.axes[0].get_legend()
    assert legend is not None
    legend_text = " ".join(item.get_text() for item in legend.get_texts())
    assert "1. detector" in legend_text
    assert "2. opencv" in legend_text
    assert "3. both" in legend_text

    def colors_of(figure):
        return [color for axis in figure.axes for _x, color in line_colors(axis)]

    assert green not in colors_of(model_figure)
    assert white in colors_of(model_figure)
    assert green not in colors_of(opencv_figure)
    assert magenta in colors_of(opencv_figure)
    assert green not in colors_of(compare_figure)
    assert white in [color for _x, color in line_colors(compare_figure.axes[0])]
    assert magenta in [color for _x, color in line_colors(compare_figure.axes[1])]
