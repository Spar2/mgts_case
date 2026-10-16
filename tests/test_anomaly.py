import numpy as np

from fiber.anomaly import cusum_alarm, extract_zones, spatial_z, temporal_z


def _hotspot():
    x = np.arange(180, dtype=np.float64)
    base = 20.0 + 0.01 * x
    field = np.repeat(base[None, :], 16, axis=0).astype(np.float32)
    field[:, 90:96] += 6.0
    valid = np.ones(field.shape, dtype=bool)
    valid[:, 160:] = False
    field[:, 160:] = -300.0
    return field, valid, x


def test_spatial_z_finds_hotspot_and_ignores_dead_tail():
    field, valid, x = _hotspot()
    z, bg, scale = spatial_z(field, valid, x, inner_m=6.0, outer_m=30.0, bin_m=2.0)
    assert z.shape == field.shape
    assert np.abs(z[:, 160:]).max() == 0
    assert np.abs(z[:, 92]).min() > 4
    assert np.median(np.abs(z[:, 10:40])) < 1.5
    assert np.isfinite(bg[:, 90]).all()
    assert scale.min() >= 0.30


def test_slow_ramp_is_not_a_local_alarm():
    x = np.arange(120, dtype=np.float64)
    field = np.repeat((18.0 + 0.015 * x)[None, :], 10, axis=0).astype(np.float32)
    valid = np.ones(field.shape, dtype=bool)
    z, _, _ = spatial_z(field, valid, x, inner_m=6.0, outer_m=25.0, bin_m=2.0)
    assert np.percentile(np.abs(z[:, 20:100]), 95) < 3.5


def test_temporal_innovation_uses_only_earlier_profiles():
    """Shipped historical residual: no past, no future leak, past step matters."""
    from fiber.anomaly.detect import temporal_innovation

    field = np.full((12, 3), 21.0, dtype=np.float32)
    valid = np.ones(field.shape, dtype=bool)
    time_s = np.arange(12, dtype=np.float64) * 60.0
    # Rows 0..3 already jump, so a scorer that ignored min_past would be
    # nonzero there. With min_past=4 those rows stay 0. Row 4 has four
    # past profiles and its score is nonzero.
    short_past = field.copy()
    short_past[1::2, :] = 35.0
    assert np.ptp(short_past[:4, 0]) > 1.0
    early = temporal_innovation(short_past, valid, time_s, window=8, min_past=4)
    assert np.all(early[:4] == 0)
    assert abs(float(early[4, 0])) > 0.1

    stepped = field.copy()
    stepped[7:, 0] = 29.0
    before = temporal_innovation(stepped, valid, time_s, window=8, min_past=3)
    future = stepped.copy()
    future[9:, :] = 10.0
    after = temporal_innovation(future, valid, time_s, window=8, min_past=3)
    assert np.allclose(before[:9], after[:9])

    past_only = field.copy()
    past_only[2:6, 1] = 28.0
    changed = temporal_innovation(past_only, valid, time_s, window=8, min_past=3)
    flat = temporal_innovation(field, valid, time_s, window=8, min_past=3)
    assert changed[6, 1] < -3.5
    assert abs(float(flat[6, 1])) < 1e-6


def test_temporal_z_is_causal_and_zero_at_start():
    field = np.full((12, 8), 21.0, dtype=np.float32)
    field[8:, :] = 26.0
    valid = np.ones(field.shape, dtype=bool)
    time_s = np.arange(12, dtype=np.float64) * 60.0
    z = temporal_z(field, valid, time_s, window=4, min_past=3)
    assert np.all(z[0] == 0)
    assert np.median(z[8]) > 4
    assert np.median(z[3]) < 1


def test_cusum_trips_on_persistent_residual():
    resid = np.zeros((20, 4), dtype=np.float32)
    resid[:, 1] = 1.2
    valid = np.ones(resid.shape, dtype=bool)
    time_s = np.arange(20, dtype=np.float64) * 30.0
    alarm = cusum_alarm(resid, valid, time_s, drift_c=0.4, threshold_c=3.0)
    assert alarm[0, 1] == 0
    assert alarm[-1, 1] == 1
    assert alarm[-1, 0] == 0


def test_extract_zones_keeps_the_hot_blob():
    field, valid, x = _hotspot()
    z, bg, _ = spatial_z(field, valid, x, inner_m=6.0, outer_m=30.0, bin_m=2.0)
    contrast = field - bg
    times = [f"t{i:02d}" for i in range(field.shape[0])]
    time_s = np.arange(field.shape[0], dtype=np.float64) * 60.0
    zones = extract_zones(z, valid, x, times, time_s, contrast_c=contrast, min_cells=8, min_profiles=3)
    assert zones
    top = zones[0]
    assert top["polarity"] == "hot"
    assert 85 <= top["start_m"] <= 96
    assert top["end_m"] <= 100
    assert top["peak_contrast_c"] > 3
