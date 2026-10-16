"""SEL: веса ансамбля, полоса IoU и cooling recall. Синтетические кадры, без реальных csv."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import s3_sel_core as sel
from dts import metrics as M


def _frame(ev0, ev1, bg, cooling, n_bg=None):
    ev = np.stack([np.asarray(ev0, np.float32), np.asarray(ev1, np.float32)], 0)
    bg = np.asarray(bg, np.float32)
    bg_m = np.stack([bg, bg], 0)
    nu = np.zeros((2, 0), np.float32)
    return dict(
        ev_m=ev,
        nu_m=nu,
        bg_m=bg_m,
        n_bg=int(bg.shape[0] if n_bg is None else n_bg),
        cooling=[np.asarray(cooling, np.int32)],
    )


def test_far_threshold_matches_metrics():
    rng = np.random.default_rng(0)
    bg = rng.random(5000)
    for far in (1e-3, 1e-2, 0.2):
        assert abs(sel.far_threshold(bg, far) - M.far_threshold(bg, far)) < 1e-9


def test_hist_threshold_close_to_sort():
    rng = np.random.default_rng(1)
    bg = rng.random(20000)
    edges = np.linspace(0.0, 1.0, 5001)
    counts, _ = np.histogram(bg, bins=edges)
    thr_h = sel.threshold_from_hist(counts, edges, 1e-3)
    thr_s = sel.far_threshold(bg, 1e-3)
    assert abs(thr_h - thr_s) < 2.0 / (len(edges) - 1)


def test_weights_clip_and_sum_to_one():
    w = sel.weights_from_array(["a", "b", "c"], [-1.0, 0.0, 3.0])
    assert w["a"] == 0.0 and abs(w["c"] - 1.0) < 1e-9
    assert abs(sum(w.values()) - 1.0) < 1e-9
    z = sel.weights_from_array(["a", "b"], [0.0, 0.0])
    assert abs(z["a"] - 0.5) < 1e-9


def test_softmax_and_dirichlet_seed():
    w = sel.softmax_weights(["a", "b"], [0.0, 1.0], temperature=0.05)
    assert w["b"] > w["a"]
    a = sel.dirichlet_bank(4, 5, seed=11)
    b = sel.dirichlet_bank(4, 5, seed=11)
    c = sel.dirichlet_bank(4, 5, seed=12)
    assert a.shape == (5, 4)
    assert np.allclose(a, b) and not np.allclose(a, c)
    assert np.allclose(a.sum(1), 1.0)


def test_logit_and_geomean_fixed_point():
    stack = np.full((3, 6), 0.4, np.float32)
    w = np.array([0.2, 0.5, 0.3])
    for mode in ("mean", "logit", "geomean"):
        out = sel.mix_stack(stack, w, mode)
        assert out.shape == (6,)
        assert np.allclose(out, 0.4, atol=1e-5)


def test_mixture_beats_single_on_cooling():
    # Общий фон ниже 0.4, порог FAR 1e-3 почти у верхней точки фона.
    bg = np.linspace(0.0, 0.4, 1000, dtype=np.float32)
    hot = np.array([0.99, 0.99, 0.99, 0.99, 0.99, 0.99], np.float32)
    ev0 = np.concatenate([hot, np.full(4, 0.10, np.float32)])
    ev1 = np.concatenate([np.full(6, 0.20, np.float32), np.full(4, 0.99, np.float32)])
    fr = _frame(ev0, ev1, bg, cooling=[6, 7, 8, 9])
    names = ["hot_only", "cold_only"]
    m0 = sel.evaluate_frames([fr], [1.0, 0.0])
    m1 = sel.evaluate_frames([fr], [0.0, 1.0])
    mix = sel.evaluate_frames([fr], [0.5, 0.5])
    assert m0["cooling_recall"] == 0.0
    assert m1["cooling_recall"] == 1.0
    assert mix["cooling_recall"] == 1.0
    assert mix["iou"] > m0["iou"] and mix["iou"] > m1["iou"]
    # Полоса не пускает модель с высоким cooling и просевшим IoU.
    rows = [
        dict(name="hot", iou=m0["iou"], cooling_recall=m0["cooling_recall"], combined=m0["combined"]),
        dict(name="cold", iou=m1["iou"], cooling_recall=m1["cooling_recall"], combined=m1["combined"]),
        dict(name="mix", iou=mix["iou"], cooling_recall=mix["cooling_recall"], combined=mix["combined"]),
    ]
    chosen, best_iou, _ = sel.pick_band(rows, tol=0.005)
    assert chosen["name"] == "mix"
    assert best_iou == max(r["iou"] for r in rows)


def test_band_keeps_iou_leader_when_cooling_is_far_below():
    rows = [
        dict(name="leader", iou=0.40, cooling_recall=0.5, combined=0.90),
        dict(name="near", iou=0.397, cooling_recall=1.0, combined=1.397),
        dict(name="drop", iou=0.30, cooling_recall=1.0, combined=1.30),
    ]
    chosen, best, band = sel.pick_band(rows, tol=0.005)
    assert best == 0.40
    assert {r["name"] for r in band} == {"leader", "near"}
    assert chosen["name"] == "near"


def test_coordinate_ascent_recovers_cooling():
    bg = np.linspace(0.0, 0.4, 800, dtype=np.float32)
    ev0 = np.concatenate([np.full(6, 0.99, np.float32), np.full(4, 0.10, np.float32)])
    ev1 = np.concatenate([np.full(6, 0.20, np.float32), np.full(4, 0.99, np.float32)])
    fr = _frame(ev0, ev1, bg, cooling=[6, 7, 8, 9])
    names = ["hot_only", "cold_only"]

    def ev(wdict):
        return sel.evaluate_frames([fr], sel.as_vector(names, wdict))["combined"]

    start = sel.recipe_weights(names, ["hot_only"])
    w, sc = sel.coordinate_ascent(ev, names, start, steps=(0.0, 0.25, 0.5, 1.0), rounds=2)
    got = sel.evaluate_frames([fr], sel.as_vector(names, w))
    assert sc + 1e-9 >= ev(start)
    assert got["cooling_recall"] == 1.0
    assert got["combined"] > ev(start)


def test_recipe_ignores_unknown_and_is_deterministic():
    names = ["unet", "tcn", "pixel_lgbm"]
    w = sel.recipe_weights(names, ["unet", "tcn"])
    assert w["pixel_lgbm"] == 0.0
    assert abs(w["unet"] - 0.5) < 1e-9
    assert sel.recipe_weights(names, ["unet", "tcn"]) == w


def test_sequence_folds_partition():
    for k in (2, 5):
        folds = sel.sequence_folds(45, k)
        assert len(folds) == k
        flat = [i for fold in folds for i in fold]
        assert sorted(flat) == list(range(45))
        assert len(set(flat)) == 45
    two = sel.sequence_folds(45, 2)
    assert len(two[0]) == 23 and len(two[1]) == 22
    five = sel.sequence_folds(45, 5)
    assert all(len(f) == 9 for f in five)


def test_seeded_folds_and_frozen_threshold_and_delta():
    a = sel.seeded_folds(45, 5, seed=0)
    b = sel.seeded_folds(45, 5, seed=0)
    c = sel.seeded_folds(45, 5, seed=1)
    assert a == b and a != c
    flat = sorted(i for fold in a for i in fold)
    assert flat == list(range(45))
    assert all(len(f) == 9 for f in a)
    # Порог с обучающего фона, IoU на другом кадре.
    bg_lo = np.linspace(0.0, 0.3, 1000, dtype=np.float32)
    bg_hi = np.linspace(0.0, 0.9, 1000, dtype=np.float32)
    ev = np.array([0.5, 0.5, 0.95], np.float32)
    train = _frame(ev, ev, bg_lo, cooling=[0])
    hold = _frame(ev, ev, bg_hi, cooling=[2])
    thr = sel.background_threshold([train], [1.0, 0.0])
    own = sel.evaluate_frames([hold], [1.0, 0.0])
    moved = sel.evaluate_frames([hold], [1.0, 0.0], thr=thr)
    assert thr < own["threshold"]
    assert moved["threshold"] == thr
    assert moved["iou"] != own["iou"]
    summary = sel.delta_summary([0.01, -0.02, 0.0, 0.03])
    assert summary["n"] == 4
    assert summary["n_positive"] == 2
    assert abs(summary["frac_positive"] - 0.5) < 1e-12
    assert abs(summary["mean"] - 0.005) < 1e-12


def test_micro_iou_pools_pixels_and_bar_is_strict():
    parts = [
        dict(tp=50, fp=50, fn=0),
        dict(tp=0, fp=0, fn=50),
    ]
    # 50 / 150, не среднее 0.5 и 0.
    assert abs(sel.micro_iou(parts) - (50 / 150)) < 1e-12
    bar = sel.NESTED_BASE + sel.NESTED_MARGIN
    assert abs(bar - 0.3897) < 1e-12
    assert sel.nested_clears_bar(bar) is False
    assert sel.nested_clears_bar(bar + 1e-6) is True
    # Оба фолда обязаны быть выше планки. Один высокий in-sample не спасает.
    bad = sel.nested_verdict({"2": 0.3944, "5": 0.3870})
    assert bad["accept"] is False
    assert abs(bad["worst"] - 0.3870) < 1e-12
    good = sel.nested_verdict({"2": 0.3910, "5": 0.3905})
    assert good["accept"] is True


def test_scaled_background_matches_full_when_sample_is_all():
    rng = np.random.default_rng(2)
    bg = np.linspace(0.0, 0.9, 500, dtype=np.float32)
    ev = rng.random(40).astype(np.float32)
    fr = _frame(ev, ev, bg, cooling=[0, 1, 2])
    # n_bg больше выборки: оценка fp масштабируется, recall cooling не зависит от фона.
    fr_scaled = _frame(ev, ev, bg, cooling=[0, 1, 2], n_bg=5000)
    a = sel.evaluate_frames([fr], [0.5, 0.5])
    b = sel.evaluate_frames([fr_scaled], [0.5, 0.5])
    assert a["cooling_recall"] == b["cooling_recall"]
    assert b["fp"] > a["fp"]
