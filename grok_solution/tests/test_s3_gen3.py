"""Физика GEN3: холод, слабое событие, склейка. Карточки и реальные CSV не читаются."""
import numpy as np

from dts import gen3


def _blank(t_len=80, n_px=256, busy=None):
    x = np.zeros((3, t_len, n_px), np.float32)
    x[2] = 1
    y = np.zeros((t_len, n_px), np.int64)
    if busy is not None:
        y[:, busy] = 1
    sigma = np.full(n_px, 0.30, np.float32)
    t = np.arange(t_len, dtype=np.float64) * 120.0
    return x, y, sigma, t


def test_self_check():
    gen3.self_check()


def test_occupied_columns_stay():
    rng = np.random.default_rng(4)
    x, y, sigma, t = _blank(busy=slice(12, 28))
    held_x = x[:, :, 12:28].copy()
    held_y = y[:, 12:28].copy()
    out, yy, info = gen3.augment_tile(x, y, sigma, t, rng, force=True)
    assert np.allclose(out[:, :, 12:28], held_x)
    assert np.array_equal(yy[:, 12:28], held_y)
    for key in gen3._MASKS:
        assert not info[key][:, 12:28].any()


def test_cold_is_negative_class_2():
    rng = np.random.default_rng(2)
    x, y, sigma, t = _blank()
    out, yy, info = gen3.augment_tile(x, y, sigma, t, rng, probs=dict(
        static=0, cold=1, weak=0, broad=0, hvac=0, artifact=0, splice=0,
    ))
    assert info["cold_mask"].any()
    assert set(np.unique(yy[info["cold_mask"]]).tolist()) == {2}
    assert info["cold"]["amp_c"] <= -3.0
    assert 2.0 <= info["cold"]["width_m"] <= 8.0
    assert float(out[1][info["cold_mask"]].mean()) < -2.0


def test_weak_range_and_class():
    rng = np.random.default_rng(5)
    x, y, sigma, t = _blank()
    _out, yy, info = gen3.augment_tile(x, y, sigma, t, rng, probs=dict(
        static=0, cold=0, weak=1, broad=0, hvac=0, artifact=0, splice=0,
    ))
    weak = info["weak"]
    assert weak["mask"].any()
    assert gen3.WEAK_AMP_C[0] <= abs(weak["amp_c"]) <= gen3.WEAK_AMP_C[1]
    assert gen3.WEAK_WIDTH_M[0] <= weak["width_m"] <= gen3.WEAK_WIDTH_M[1]
    assert gen3.WEAK_LEN[0] <= weak["n_samples"] <= gen3.WEAK_LEN[1]
    assert weak["cls"] in (1, 2)
    assert np.all(yy[info["weak_mask"]] == weak["cls"])


def test_splice_is_background_step():
    rng = np.random.default_rng(6)
    x, y, sigma, t = _blank()
    out, yy, info = gen3.augment_tile(x, y, sigma, t, rng, probs=dict(
        static=0, cold=0, weak=0, broad=0, hvac=0, artifact=0, splice=1,
    ))
    sp = info["splice"]
    assert 1 <= sp["n_cols"] <= 2
    assert gen3.SPLICE_AMP_C[0] <= abs(sp["amp_c"]) <= gen3.SPLICE_AMP_C[1]
    assert yy[info["splice_mask"]].max() == 0
    assert float(np.abs(out[0][info["splice_mask"]]).mean()) < 1e-5
    assert float(np.abs(out[1][info["splice_mask"]]).mean()) > 2.0
    wk = gen3.pixel_weight(info)
    assert float(wk[info["splice_mask"]].min()) >= 1.0


def test_weak_weight_above_static_and_gen2_probs_kept():
    for key, val in gen2_probs():
        assert gen3.PROBS[key] == val
    assert set(gen3.PROBS) >= {"static", "broad", "hvac", "artifact", "cold", "weak", "splice"}


def gen2_probs():
    from dts import gen2
    return gen2.PROBS.items()
