"""Label mapping for the synth trainer. No HDF5 and no hand-label fitting."""

import numpy as np

from train_synth import COOL_IDS, HEAT_IDS, expected_hand_name, label_from_mask


def test_event_mask_maps_to_heat_cool_or_background() -> None:
    mask = np.zeros((32, 128), dtype=np.uint8)
    assert label_from_mask(mask) == 0
    mask[0, 0] = HEAT_IDS[0]
    assert label_from_mask(mask) == 0
    mask[10:12, 20:30] = HEAT_IDS[1]
    assert label_from_mask(mask) == 1
    cool = np.zeros((32, 128), dtype=np.uint8)
    cool[5:15, 40:50] = COOL_IDS[1]
    assert label_from_mask(cool) == 2


def test_hand_names_stay_a_check_mapping() -> None:
    assert expected_hand_name({"suggested_class": "Local_Heating"}) == "heat"
    assert expected_hand_name({"suggested_class": "Local_Cooling"}) == "cool"
    assert expected_hand_name({"suggested_class": "Background"}) == "background"
    assert expected_hand_name({"suggested_class": "Broad_Shift", "delta_c": 3.0}) == "heat"
    assert expected_hand_name({"suggested_class": "Broad_Shift", "delta_c": -2.0}) == "cool"
