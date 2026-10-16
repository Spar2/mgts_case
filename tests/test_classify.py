import numpy as np

from fiber.benchmark.synthetic import half_cable_shift, local_scene
from fiber.classify.rules import classify_zones, profile_shift


def _zone(**overrides):
    base = {
        "polarity": "hot",
        "start_m": 10.0,
        "end_m": 16.0,
        "width_m": 6.0,
        "t_start_s": 0.0,
        "median_contrast_c": 4.0,
        "peak_contrast_c": 6.0,
    }
    base.update(overrides)
    return base


def test_local_hot_and_cold():
    labels = classify_zones(
        [
            _zone(),
            _zone(polarity="cold", start_m=40, end_m=46, median_contrast_c=-3, peak_contrast_c=-5),
        ],
        cable_length_m=200,
    )
    assert labels == ["Local_Heating", "Local_Cooling"]


def test_narrow_jump_is_sensor_fault_and_wide_zone_is_broad():
    labels = classify_zones(
        [
            _zone(start_m=1, end_m=1.4, width_m=0.4, peak_contrast_c=18),
            _zone(start_m=0, end_m=90, width_m=90, peak_contrast_c=3),
        ],
        cable_length_m=120,
    )
    assert labels == ["Sensor_Fault", "Broad_Shift"]


def test_wide_strong_heat_stays_local():
    labels = classify_zones(
        [_zone(start_m=10, end_m=30, width_m=20, peak_contrast_c=32, median_contrast_c=20)],
        cable_length_m=1000,
    )
    assert labels == ["Local_Heating"]


def test_regular_repeats_are_periodic_and_irregular_are_not():
    from fiber.classify.rules import periodicity_cv

    regular = [
        _zone(t_start_s=step * 600.0, start_m=10, end_m=16, width_m=6)
        for step in range(4)
    ]
    irregular = [
        _zone(t_start_s=t, start_m=50, end_m=56, width_m=6)
        for t in (0.0, 100.0, 5000.0)
    ]
    labels = classify_zones(regular + irregular, cable_length_m=200)
    assert labels[:4] == ["Periodic_Thermal"] * 4
    assert labels[4:] == ["Local_Heating"] * 3
    assert periodicity_cv(regular) == [0.0, 0.0, 0.0, 0.0]
    assert periodicity_cv(irregular) == [None, None, None]
    assert periodicity_cv([_zone()]) == [None]


def test_half_cable_is_broad_and_local_blob_is_not():
    broad = half_cable_shift()
    found = profile_shift(broad["clean"], broad["valid"], broad["x_m"])
    assert found["label"] == "Broad_Shift"
    assert found["side"] == "left_warmer"
    local = local_scene()
    quiet = profile_shift(local["clean"], local["valid"], local["x_m"])
    assert quiet["label"] == "Background"
    x_m = np.arange(180, dtype=np.float64)
    slope = np.broadcast_to(20.0 + 0.01 * x_m, (12, 180)).copy()
    valid = np.ones(slope.shape, dtype=bool)
    assert profile_shift(slope, valid, x_m)["label"] == "Background"
