"""Пост-обработка score: CFAR и сглаживание без меток и без больших кэшей."""
import inspect
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dts import metrics, postproc as PP
from dts.infer import Analyzer


def test_temporal_max_never_lowers_and_keeps_invalid():
    s = np.zeros((9, 3), np.float32)
    s[4, 1] = 1
    v = np.ones_like(s, bool)
    v[0, 0] = False
    s[0, 0] = 5
    for name in ("box3", "box7", "gauss3", "gauss15"):
        out = PP.apply(s, v, name)
        assert out.shape == s.shape
        assert out[0, 0] == 5
        assert out[4, 1] >= 1 - 1e-5
        assert np.all(out[v] + 1e-6 >= s[v])
    assert PP.apply(s, v, "box3")[3, 1] > 0


def test_cfar_per_x_removes_spatial_offset():
    rng = np.random.default_rng(0)
    t, x = 80, 20
    s = rng.normal(0, 0.05, (t, x)).astype(np.float32)
    s[:, :10] += 1.0
    v = np.ones_like(s, bool)
    z = PP.apply(s, v, "cfar_x")
    assert np.max(np.abs(np.median(z, axis=0))) < 0.2
    s2 = s.copy()
    s2[10, 3] += 0.5
    s2[10, 15] += 0.5
    z2 = PP.apply(s2, v, "cfar_x")
    assert z2[10, 3] > 3 and z2[10, 15] > 3


def test_cfar_constant_and_spike():
    s = np.full((30, 16), 0.3, np.float32)
    v = np.ones_like(s, bool)
    for name in ("cfar_global", "cfar_x", "cfar_x_w21", "cfar_2d"):
        z = PP.apply(s, v, name)
        assert np.max(np.abs(z)) < 0.05
    s[5, 8] = 0.9
    z = PP.apply(s, v, "cfar_x_w21")
    assert z[5, 8] > 10


def test_far_after_per_x_matches_rate():
    rng = np.random.default_rng(1)
    t, x = 500, 24
    offset = np.linspace(0.0, 0.8, x).astype(np.float32)
    s = offset + rng.normal(0, 0.02, (t, x)).astype(np.float32)
    v = np.ones((t, x), bool)
    z = PP.apply(s, v, "cfar_x")
    thr = metrics.far_threshold(z.ravel(), 1e-3)
    rate = float((z.ravel() >= thr).mean())
    assert abs(rate - 1e-3) < 2e-3


def test_all_variants_finite_on_small_map():
    rng = np.random.default_rng(2)
    s = rng.random((36, 22)).astype(np.float32)
    v = np.ones_like(s, bool)
    v[:2, 0] = False
    base = PP.apply(s, v, "identity")
    assert np.allclose(base, s)
    for name in PP.VARIANTS:
        out = PP.apply(s, v, name)
        assert out.shape == s.shape
        assert np.isfinite(out).all()


def test_shift_preserves_decision():
    rng = np.random.default_rng(3)
    z = rng.normal(size=1000).astype(np.float32)
    src, dst = 2.5, 0.84
    out = PP.shift_to_threshold(z, src, dst)
    assert np.array_equal(z >= src, out >= dst)


def test_choose_keeps_iou_and_prefers_weak_recall():
    rows = {
        "identity": dict(iou=0.30, f1=0.40, weak_recall=0.02, event_recall=0.40, far=0.001),
        "cfar_x": dict(iou=0.295, f1=0.40, weak_recall=0.10, event_recall=0.40, far=0.001),
        "box15": dict(iou=0.20, f1=0.30, weak_recall=0.50, event_recall=0.90, far=0.001),
    }
    picked = PP.choose(rows)
    assert picked["variant"] == "cfar_x" and picked["enable"] is True
    assert "box15" in picked["rejected"]
    same_weak = dict(rows)
    same_weak["cfar_x"] = dict(iou=0.31, f1=0.45, weak_recall=0.02, event_recall=0.50, far=0.001)
    picked = PP.choose(same_weak)
    assert picked["variant"] == "cfar_x" and picked["enable"] is True
    worse = dict(rows)
    worse["cfar_x"] = dict(iou=0.30, f1=0.40, weak_recall=0.01, event_recall=0.30, far=0.001)
    picked = PP.choose(worse)
    assert picked["variant"] == "identity" and picked["enable"] is False


def test_flag_off_leaves_score_unchanged(monkeypatch):
    monkeypatch.setattr(PP, "DEPLOYED_VARIANT", "cfar_x")
    monkeypatch.setattr(PP, "THRESHOLD", 3.0)
    monkeypatch.setattr(PP, "ENABLED", True)
    s = np.linspace(0, 1, 24, dtype=np.float32).reshape(6, 4)
    s[:3] += 0.4
    v = np.ones_like(s, bool)
    off = PP.score_for_analyzer(s, v, 0.5, enabled=False)
    assert np.allclose(off, s)
    on = PP.score_for_analyzer(s, v, 0.5, enabled=True)
    assert not np.allclose(on, s)
    assert PP.resolve_enabled(None, "far_ref") is False
    assert PP.resolve_enabled(False, PP.OP) is False
    assert PP.resolve_enabled(None, PP.OP) is True
    monkeypatch.setattr(PP, "ENABLED", False)
    assert PP.resolve_enabled(None, PP.OP) is False


def test_analyzer_run_accepts_postproc_flag():
    sig = inspect.signature(Analyzer.run)
    params = list(sig.parameters)
    assert params[1:5] == ["T", "valid", "x", "t"]
    assert sig.parameters["postproc"].default is None
