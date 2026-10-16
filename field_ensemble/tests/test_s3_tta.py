"""Геометрия TTA: отражения, симметрия знака, смеси, порог FAR. Без сетей и без real."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("s3_tta", ROOT / "scripts" / "s3_tta.py")
tta = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tta)

from dts import metrics as M


def test_flip_and_sign_roundtrip_probabilities():
    rng = np.random.default_rng(0)
    P = rng.random((3, 5, 7)).astype(np.float32)
    P /= P.sum(0, keepdims=True)
    assert np.allclose(tta.align_prob(np.flip(P, 2), flip_x=True), P)
    assert np.allclose(tta.align_prob(np.flip(P, 1), flip_t=True), P)
    swapped = np.stack([P[0], P[2], P[1]], 0)
    assert np.allclose(tta.align_prob(swapped, sign=True), P)
    both = np.flip(np.flip(swapped, 1), 2)
    back = tta.align_prob(both, flip_t=True, flip_x=True, sign=True)
    assert np.allclose(back, P)


def test_sign_negates_residual_channels_only_and_time_flip_keeps_class_order():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(3, 4, 6)).astype(np.float32)
    X[2] = (X[2] > 0).astype(np.float32)
    Y = tta.transform_input(X, sign=True)
    assert np.allclose(Y[0], -X[0])
    assert np.allclose(Y[1], -X[1])
    assert np.allclose(Y[2], X[2])
    Ft = tta.transform_input(X, flip_t=True)
    assert Ft.shape == X.shape
    assert np.allclose(Ft[:, 0, :], X[:, -1, :])
    assert np.allclose(Ft[2], np.flip(X[2], 0))
    # Отражение времени не меняет местами классы: в спецификации sign выключен.
    assert tta.VIEW_SPEC["flip_t"]["sign"] is False
    assert tta.VIEW_SPEC["flip_t"]["flip_t"] is True
    assert tta.VIEW_SPEC["flip_x"]["sign"] is False
    assert tta.VIEW_SPEC["sign"]["sign"] is True
    assert tta.views_required("mean_full") == {"flip_x", "flip_t", "sign"}
    assert tta.views_required("wsign_0.50") == {"sign"}
    assert tta.views_required("sign_member") == {"sign"}
    assert tta.LINEAR["sign_member"] == {"sign": 1.0}
    assert "flip_t" in tta.LINEAR[tta.PRIMARY]


def test_member_delta_replaces_one_slot_of_four():
    rng = np.random.default_rng(5)
    mats = [rng.random((4, 5)).astype(np.float32) for _ in range(4)]
    hot = [rng.normal(size=(4, 5)).astype(np.float32) for _ in range(4)]
    ens_s = (mats[0] + mats[1] + mats[2] + mats[3]) * np.float32(0.25)
    ens_h = (hot[0] + hot[1] + hot[2] + hot[3]) * np.float32(0.25)
    new_s = rng.random((4, 5)).astype(np.float32)
    new_h = rng.normal(size=(4, 5)).astype(np.float32)
    got_s, got_h = tta.member_delta(ens_s, ens_h, mats[0], hot[0], new_s, new_h, n=4)
    ref_s = (new_s + mats[1] + mats[2] + mats[3]) * np.float32(0.25)
    ref_h = (new_h + hot[1] + hot[2] + hot[3]) * np.float32(0.25)
    assert got_s.dtype == np.float32 and got_h.dtype == np.float32
    assert np.allclose(got_s, ref_s, atol=1e-5)
    assert np.allclose(got_h, ref_h, atol=1e-5)


def test_score_hotness_roundtrip_and_sign_negates_hotness():
    s = np.array([[0.2, 0.8]], np.float32)
    h = np.array([[0.1, -0.4]], np.float32)
    p = tta.prob_from_sh(s, h)
    s2, h2 = tta.sh_from_prob(p)
    assert s2.dtype == np.float32 and h2.dtype == np.float32
    assert np.allclose(s2, s) and np.allclose(h2, h)
    assert np.allclose(p.sum(0), 1)
    aligned = tta.align_prob(p, sign=True)
    sa, ha = tta.sh_from_prob(aligned)
    assert np.allclose(sa, s)
    assert np.allclose(ha, -h)


def test_linear_weights_and_shrink_of_two_is_min():
    a = (np.array([0.2, 0.8], np.float32), np.array([0.4, -0.2], np.float32))
    b = (np.array([0.6, 0.0], np.float32), np.array([-0.2, 0.2], np.float32))
    s, h = tta.linear_fuse({"id": a, "sign": b}, {"id": 0.75, "sign": 0.25})
    assert np.allclose(s, 0.75 * a[0] + 0.25 * b[0])
    assert np.allclose(h, 0.75 * a[1] + 0.25 * b[1])
    mv = {"id": a, "sign": b, "flip_x": a, "flip_t": b}
    sm, hm = tta.nonlinear_member(mv, "shrink_full")
    # Четыре вида (a, b, a, b): среднее (a+b)/2, std = |a-b|/2, mean-0.5*std = mean - |a-b|/4.
    mid = 0.5 * (a[0] + b[0])
    assert np.allclose(sm, mid - 0.25 * np.abs(a[0] - b[0]))
    two = {
        "id": a, "flip_x": a, "flip_t": b, "sign": b,
    }
    # geomean знака подавляет клетку, где один вид почти ноль
    g, _ = tta.nonlinear_member(two, "geomean_sign")
    assert g[1] < 1e-3


def test_member_mean_matches_ensemble_mean_for_linear_full():
    rng = np.random.default_rng(2)
    keys = ("id", "flip_x", "flip_t", "sign")
    members = []
    for _ in range(3):
        mv = {}
        for k in keys:
            mv[k] = (rng.random((2, 3)).astype(np.float32), rng.normal(size=(2, 3)).astype(np.float32))
        members.append(mv)
    lgbm = (rng.random((2, 3)).astype(np.float32), rng.normal(size=(2, 3)).astype(np.float32))
    ens = {k: tta.Acc() for k in keys}
    per = tta.Acc()
    for mv in members:
        for k in keys:
            ens[k].add(*mv[k])
        per.add(*tta.linear_fuse(mv, tta.LINEAR["mean_full"]))
    for k in keys:
        ens[k].add(*lgbm)
    per.add(*lgbm)
    fused = tta.linear_fuse({k: ens[k].mean() for k in keys}, tta.LINEAR["mean_full"])
    other = per.mean()
    assert np.allclose(fused[0], other[0], atol=1e-6)
    assert np.allclose(fused[1], other[1], atol=1e-6)


def test_maxpol_renormalizes_and_keeps_confident_polarity():
    # id уверен в нагреве, sign-вид (уже выровненный) уверен в охлаждении
    id_sh = (np.array([0.9], np.float32), np.array([0.8], np.float32))
    sign_sh = (np.array([0.9], np.float32), np.array([-0.8], np.float32))
    s, h = tta.maxpol(id_sh, sign_sh)
    p = tta.prob_from_sh(s, h)
    assert np.allclose(p.sum(0), 1, atol=1e-5)
    assert s[0] > 0.5
    # обе полярности сильные, hotness ближе к нулю, чем у одного вида
    assert abs(float(h[0])) < 0.8


def test_scorebag_matches_far_threshold_and_confusion():
    rng = np.random.default_rng(3)
    s = rng.random((40, 50)).astype(np.float32)
    h = np.where(rng.random((40, 50)) > 0.5, 1.0, -1.0).astype(np.float32)
    valid = np.ones(s.shape, bool)
    valid[0, :3] = False
    ev = np.zeros(s.shape, np.uint8)
    ev[5:8, 10:14] = 1
    ev[20:23, 30:33] = 6
    nu = np.zeros(s.shape, np.uint8)
    nu[1, 4:8] = 1
    bag = tta.ScoreBag(gate=0.0)
    bag.add("seq", s, h, valid, ev, nu)
    bg = s[valid & (ev == 0) & (nu == 0)]
    for far in (1e-2, 1e-3):
        thr_ref = M.far_threshold(bg, far)
        thr = bag.threshold(far)
        assert thr is not None
        assert abs(thr - thr_ref) < 1e-6
        pred = (s >= thr) & valid
        truth = ev > 0
        tp, fp, fn = M.confusion(pred, truth, valid)
        rep = bag.report(thr)
        assert rep["tp"] == tp and rep["fp"] == fp and rep["fn"] == fn
        assert abs(rep["iou"] - M.prf(tp, fp, fn)["iou"]) < 1e-12
        hot_t = valid & (ev >= 1) & (ev <= 4)
        ht = M.confusion(pred & (h > 0), hot_t, valid)
        assert rep["hot"]["iou"] == M.prf(*ht)["iou"]
        cold_t = valid & (ev >= 5)
        ct = M.confusion(pred & (h <= 0), cold_t, valid)
        assert abs(rep["cold"]["iou"] - M.prf(*ct)["iou"]) < 1e-12


def test_scorebag_gate_still_hits_upper_quantile():
    rng = np.random.default_rng(4)
    bg = rng.random(20000).astype(np.float32) * np.float32(0.5)
    # несколько высоких, чтобы квантиль 1e-3 был выше gate
    bg[:40] = np.linspace(0.8, 0.99, 40, dtype=np.float32)
    s = bg.reshape(100, 200)
    h = np.ones_like(s)
    valid = np.ones(s.shape, bool)
    ev = np.zeros(s.shape, np.uint8)
    nu = np.zeros(s.shape, np.uint8)
    ev[0, :10] = 2
    bag = tta.ScoreBag(gate=0.25)
    bag.add("a", s, h, valid, ev, nu)
    thr = bag.threshold(1e-3)
    ref = M.far_threshold(s[valid & (ev == 0) & (nu == 0)], 1e-3)
    assert thr is not None and thr >= bag.gate
    assert abs(thr - ref) < 1e-6
    # ниже gate квантиль не обещаем: FAR 0.5 должен вернуть None на этом распределении
    assert bag.threshold(0.5) is None or bag.threshold(0.5) >= bag.gate


def test_runner_is_inference_only_and_names_tta_files():
    src = (ROOT / "scripts" / "s3_tta_run.py").read_text(encoding="utf-8")
    assert "MGTS_data" not in src
    assert "backward" not in src
    assert "optimizer" not in src.lower()
    assert "ens_tta" in src
    assert "s3_tta_" in src
    assert "mean_full" in src
    lib = (ROOT / "scripts" / "s3_tta.py").read_text(encoding="utf-8")
    assert "flip_x" in lib and "flip_t" in lib and "sign" in lib
