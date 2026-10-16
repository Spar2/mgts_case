"""Scene builders used by the threshold and refine reports."""

import hashlib

import numpy as np

from fiber.benchmark.synthetic import background_scene, half_cable_shift, local_scene, local_scene_var

# seed 0 fields of the functions that already existed at d288333
_LOCAL_SEED0 = "a6cb11fc9107f47c93b612fa1c9d58041df679bb736bcbdd3da6b3cd2908b63c"
_HALF_SEED0 = "f2d3eac43ea308e4664eb35c2a1fa8529eef7b030eff5bf9a4c5d2397ec02754"


def _sha(field: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(field).tobytes()).hexdigest()


def test_seed0_control_scenes_keep_their_fields():
    assert _sha(local_scene(0)["clean"]) == _LOCAL_SEED0
    assert _sha(half_cable_shift(0)["clean"]) == _HALF_SEED0


def test_background_scene_has_no_events():
    scene = background_scene(0)
    assert scene["clean"].shape == local_scene(0)["clean"].shape
    assert not scene["hot"].any()
    assert not scene["cold"].any()
    assert np.all(scene["truth_delta_c"] == 0)
    assert int((~scene["valid"]).sum()) == int((~local_scene(0)["valid"]).sum())


def test_local_scene_var_truth_matches_the_plant():
    scene = local_scene_var(0, amp_hot=4.0, amp_cold=-3.0, width=5, noise=0.0, ramp=0)
    delta = scene["truth_delta_c"]
    assert scene["hot"].sum() == 21 * 5
    assert scene["cold"].sum() == 17 * 5
    assert np.isclose(delta[scene["hot"]].max(), 4.0)
    assert np.isclose(delta[scene["cold"]].min(), -3.0)
    assert np.all(delta[~scene["hot"] & ~scene["cold"]] == 0)
    quiet = scene["valid"] & ~scene["hot"] & ~scene["cold"]
    assert np.allclose(scene["clean"][quiet], 21.0)

    ramped = local_scene_var(1, amp_hot=4.0, amp_cold=-4.0, width=5, noise=0.0, ramp=2)
    partial = ramped["truth_delta_c"]
    assert np.any((partial > 0) & (partial < 4.0))
    assert np.isclose(partial.max(), 4.0)
    assert ramped["hot"].sum() > scene["hot"].sum()
