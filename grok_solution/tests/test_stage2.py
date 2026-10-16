"""Признаки этапа 2, калибровка и сквозное сопоставление зон."""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dts import events


def test_crop_scalar_width_unchanged():
    rng = np.random.default_rng(2)
    T = 20 + rng.normal(0, 0.2, (80, 120)).astype(np.float32)
    T[30:50, 40:55] -= 3
    from dts import features
    ch, aux = features.make_channels(T, np.ones_like(T, bool))
    x = np.arange(120) * 0.25
    t = np.arange(80) * 120.0
    img, f = events.crop_and_features(ch, aux["sigma_x"], x, t, 10, 14, 30 * 120, 50 * 120)
    assert img.shape == (3,) + events.CROP
    assert len(f) == len(events.SCALAR_NAMES)


def test_image_bank_shape_and_lgbm_widths():
    img = np.zeros((4, 3, 32, 64), np.float32)
    img[:, 0, :, 30:34] = 1
    base = np.zeros((4, len(events.SCALAR_NAMES)), np.float32)
    ctx = np.zeros((4, len(events.CONTEXT_NAMES)), np.float32)
    assert events.image_bank(img).shape == (4, events.IMAGE_BANK_DIM)
    assert events.shape_features(img).shape == (4, len(events.SHAPE_NAMES))
    assert events.lgbm_matrix(base, img).shape[1] == len(events.SCALAR_NAMES) + events.IMAGE_BANK_DIM
    full = events.lgbm_matrix(base, img, ctx, "full")
    assert full.shape[1] == len(events.SCALAR_NAMES) + events.IMAGE_BANK_DIM + len(events.SHAPE_NAMES) + len(events.CONTEXT_NAMES)


def test_context_chain_and_singleton():
    xa = np.array([0.0, 50.0, 100.0])
    xb = xa + 0.4
    ta = np.array([0.0, 400.0, 800.0])
    tb = ta + 200.0
    c = events.context_features(xa, xb, ta, tb)
    assert c.shape == (3, len(events.CONTEXT_NAMES))
    assert c[:, events.CONTEXT_NAMES.index("n_chain")].min() >= 1
    alone = events.context_features([0.0], [1.0], [0.0], [10.0])
    assert alone[0, events.CONTEXT_NAMES.index("log_n_other")] == 0
    assert alone[0, events.CONTEXT_NAMES.index("log_nearest_dx_m")] > 5
    assert events.context_features([], [], [], []).shape == (0, len(events.CONTEXT_NAMES))


def test_dataset_context_follows_cache_order(tmp_path):
    import pandas as pd
    cache = tmp_path / "cache" / "train"
    cache.mkdir(parents=True)
    x = np.linspace(0, 200, 50)
    np.savez(cache / "seq0001.npz", x=x)
    pd.DataFrame([
        dict(seq_id="seq0001", t_start="2026-01-01T00:00:00+00:00", t_end="2026-01-01T01:00:00+00:00",
             dist_start_m=10.0, dist_end_m=12.0, label="hotspot"),
        dict(seq_id="seq0001", t_start="2026-01-01T00:10:00+00:00", t_end="2026-01-01T00:20:00+00:00",
             dist_start_m=60.0, dist_end_m=60.4, label="tracing"),
    ]).to_csv(tmp_path / "labels.csv", index=False)
    y = np.array([events.CLASSES.index("Local_Heating"), events.CLASSES.index("Contact_Trace")])
    ctx = events.dataset_context(tmp_path / "cache", tmp_path / "labels.csv", expect_y=y, expect_groups=["seq0001", "seq0001"])
    assert ctx.shape == (2, len(events.CONTEXT_NAMES))
    assert ctx[:, events.CONTEXT_NAMES.index("n_chain")].min() >= 1


def test_box_iou_and_greedy_match():
    assert events.box_iou((0, 2, 0, 2), (0, 2, 0, 2)) == pytest.approx(1)
    assert events.box_iou((0, 1, 0, 1), (2, 3, 2, 3)) == 0
    assert events.box_iou((0, 2, 0, 2), (1, 3, 0, 2)) == pytest.approx(1 / 3)
    matches, unmatched_p, unmatched_g = events.greedy_match(
        [(0, 2, 0, 2), (10, 12, 10, 12)], [(0.2, 1.8, 0.2, 1.8)], 0.1)
    assert len(matches) == 1 and matches[0][0] == 0
    assert unmatched_p == [1] and unmatched_g == []


def test_e2e_counts_match_mismatch_and_false_zone():
    pred = np.array([0, 1])
    gt = np.array([0, 2])
    matches = [(0, 0, 0.5)]
    tp, fp, fn, correct = events.e2e_counts(pred, gt, matches, 7)
    assert correct == 1
    assert tp[0] == 1 and fp[1] == 1 and fn[2] == 1
    macro, per = events.f1_from_counts(tp, fp, fn, events.CLASSES)
    assert per["Local_Heating"]["f1"] == pytest.approx(1)
    assert 0 <= macro <= 1


def test_temperature_preserves_argmax_and_lowers_overconfidence():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(80, 7)).astype(np.float32)
    logits[:, 0] += 5  # модель уверена в классе 0, а метки случайные
    y = rng.integers(0, 7, size=80)
    T = events.fit_temperature(logits, y)
    assert T > 1.0
    z = logits - logits.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    q = events.temper_proba(p, T)
    assert np.array_equal(p.argmax(1), q.argmax(1))
    assert q.max(1).mean() < p.max(1).mean()


def test_eventnet_legacy_graph_and_late_fusion_starts_equal():
    torch = pytest.importorskip("torch")
    from dts.models import EventNet, two_scale
    torch.manual_seed(0)
    base = EventNet(7, 19)
    assert "head_img.weight" not in base.state_dict()
    late = EventNet(7, 19, late_fusion=True)
    late.cnn.load_state_dict(base.cnn.state_dict())
    late.mlp.load_state_dict(base.mlp.state_dict())
    late.head.load_state_dict(base.head.state_dict())
    base.eval(); late.eval()
    img = torch.randn(2, 3, 32, 64)
    sc = torch.randn(2, 19)
    with torch.no_grad():
        assert torch.allclose(base(img, sc), late(img, sc), atol=1e-5)
    assert two_scale(img).shape == (2, 6, 32, 64)


def test_legacy_eventnet_checkpoint_still_loads():
    torch = pytest.importorskip("torch")
    from dts.models import EventNet
    ckpt = Path(__file__).resolve().parents[1] / "artifacts" / "stage2" / "eventnet.pt"
    if not ckpt.exists():
        pytest.skip("no eventnet checkpoint")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    net = EventNet(len(ck["classes"]), len(ck["mu"]))
    net.load_state_dict(ck["state"])
    net.eval()
    with torch.no_grad():
        out = net(torch.zeros(2, 3, 32, 64), torch.zeros(2, len(ck["mu"])))
    assert out.shape == (2, len(ck["classes"]))


def test_predict_zone_probs_empty_and_legacy_lgbm():
    torch = pytest.importorskip("torch")
    from dts.models import EventNet
    from dts.infer import predict_zone_probs
    torch.manual_seed(0)
    net = EventNet(7, len(events.SCALAR_NAMES))
    X = np.zeros((3, 40, 80), np.float32)
    X[2] = 1
    X[0, 10:25, 20:40] = 2
    sigma = np.ones(80, np.float32)
    x = np.arange(80) * 0.25
    t = np.arange(40) * 120.0
    assert predict_zone_probs(net, np.zeros(19), np.ones(19), X, sigma, x, t, []) == []

    class Stub:
        n_features_in_ = len(events.SCALAR_NAMES) + events.IMAGE_BANK_DIM

        def predict_proba(self, feat):
            assert feat.shape[1] == self.n_features_in_
            p = np.full((len(feat), 7), 0.01, np.float32)
            p[:, 0] = 0.94
            return p

    zones = [dict(start_m=5.0, end_m=10.0, t_start_s=1200.0, t_end_s=3000.0)]
    P = predict_zone_probs(net, np.zeros(19), np.ones(19), X, sigma, x, t, zones, lgbm=Stub(), lgbm_kind="legacy")
    assert P.shape == (1, 7)
    assert abs(P.sum() - 1) < 1e-4


def test_lgbm_seeds_are_averaged_when_cnn_weight_is_zero():
    torch = pytest.importorskip("torch")
    from dts.models import EventNet
    from dts.infer import predict_zone_probs
    net = EventNet(7, len(events.SCALAR_NAMES))
    X = np.zeros((3, 40, 80), np.float32); X[2] = 1; X[0, 10:20, 20:40] = 1
    sigma = np.ones(80, np.float32)
    x = np.arange(80) * 0.25
    t = np.arange(40) * 120.0
    zones = [dict(start_m=5.0, end_m=10.0, t_start_s=1200.0, t_end_s=2400.0)]

    class Stub:
        n_features_in_ = len(events.SCALAR_NAMES) + events.IMAGE_BANK_DIM

        def __init__(self, col):
            self.col = col

        def predict_proba(self, feat):
            p = np.full((len(feat), 7), 0.02, np.float32)
            p[:, self.col] = 0.88
            return p

    P = predict_zone_probs(net, np.zeros(19), np.ones(19), X, sigma, x, t, zones,
                           lgbm=[Stub(0), Stub(1)], lgbm_kind="legacy", cnn_weight=0.0)
    assert abs(float(P[0, 0]) - float(P[0, 1])) < 1e-5
    assert float(P[0, 0]) > float(P[0, 2])


def test_eval_end2end_cli_on_synthetic_maps(tmp_path):
    torch = pytest.importorskip("torch")
    from dts.models import EventNet
    root = Path(__file__).resolve().parents[1]
    cache = tmp_path / "cache" / "test"
    pred = tmp_path / "pred" / "test"
    cache.mkdir(parents=True)
    pred.mkdir(parents=True)
    x = np.arange(40) * 0.25
    t = np.arange(30) * 120.0
    X = np.zeros((3, 30, 40), np.float32)
    X[2] = 1
    valid = np.ones((30, 40), bool)
    np.savez(cache / "seq0001.npz", X=X, valid=valid, x=x, t=t, sigma=np.ones(40, np.float32))
    s = np.zeros((30, 40), np.float32)
    s[5:15, 8:20] = 0.9
    s[20:28, 30:38] = 0.9
    h = np.ones_like(s)
    np.savez(pred / "seq0001.npz", s=s, h=h)
    # метка покрывает только первый блок
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "seq_id,t_start,t_end,dist_start_m,dist_end_m,label\n"
        "seq0001,1970-01-01T00:10:00+00:00,1970-01-01T00:28:00+00:00,2.0,4.75,hotspot\n"
    )
    net = EventNet(7, len(events.SCALAR_NAMES))
    ckpt = tmp_path / "eventnet.pt"
    torch.save(dict(state=net.state_dict(), mu=np.zeros(19, np.float32), sd=np.ones(19, np.float32),
                    classes=events.CLASSES, temperature=1.0, cnn_weight=1.0), ckpt)
    out = tmp_path / "e2e.json"
    script = root / "scripts" / "eval_end2end.py"
    subprocess.check_call([
        sys.executable, str(script),
        "--pred", str(tmp_path / "pred"), "--cache", str(tmp_path / "cache"), "--split", "test",
        "--labels", str(labels), "--thr", "0.5", "--stage2", str(ckpt), "--no-lgbm",
        "--out", str(out), "--stage1-json", str(tmp_path / "missing.json"),
    ], cwd=root, env={**dict(**{k: v for k, v in __import__("os").environ.items()}), "OMP_NUM_THREADS": "1"})
    rep = json.loads(out.read_text())
    assert rep["n_zones"] == 2
    assert rep["n_false_zones"] == 1
    assert rep["false_zone_frac"] == pytest.approx(0.5)
    assert rep["n_matched"] == 1
    assert rep["n_event"] == 1 and rep["n_event_hit"] == 1
    assert rep["event_recall"] == pytest.approx(1.0)
    assert abs(sum(rep["false_zone_class_frac"].values()) - 1.0) < 1e-9
    assert rep["false_nuisance_frac"] == pytest.approx(
        rep["false_zone_class_frac"]["HVAC_Diurnal"] + rep["false_zone_class_frac"]["Sensor_Artifact"])
    assert 0.0 <= rep["alarm_precision"] <= 1.0
    assert 0.0 <= rep["macro_f1"] <= 1.0
    assert set(rep["per_class"]) == set(events.CLASSES)


def test_zone_alarm_stats_recall_and_precision():
    import importlib.util
    script = Path(__file__).resolve().parents[1] / "scripts" / "eval_end2end.py"
    spec = importlib.util.spec_from_file_location("eval_end2end_stats", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # зона 0 — тревога и попала в событие; зона 1 — HVAC и ложная; зона 2 — тревога и ложная
    # второе событие без пары
    pred = np.array([0, events.CLASSES.index("HVAC_Diurnal"), 0])
    is_event = np.array([True, True])
    false_cls, n_event, n_hit, n_alarm, n_alarm_hit = mod.zone_alarm_stats(
        pred, is_event, [(0, 0, 0.8)], len(events.CLASSES))
    assert (n_event, n_hit, n_alarm, n_alarm_hit) == (2, 1, 2, 1)
    assert false_cls.sum() == 2
    assert false_cls[events.CLASSES.index("HVAC_Diurnal")] == 1
    assert false_cls[0] == 1


def test_resolve_threshold_from_json_and_arg(tmp_path):
    import importlib.util
    script = Path(__file__).resolve().parents[1] / "scripts" / "eval_end2end.py"
    spec = importlib.util.spec_from_file_location("eval_end2end", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.resolve_threshold("nope.json", "matched_filter", "far_ref", 1.25) == (1.25, "arg")
    p = tmp_path / "stage1.json"
    p.write_text(json.dumps({"thresholds": {"matched_filter": {"far_ref": 2.5, "far_1e-3": 4.0}}}))
    thr, src = mod.resolve_threshold(p, "matched_filter", "far_ref", None)
    assert thr == pytest.approx(2.5) and "matched_filter.far_ref" in src
    with pytest.raises(SystemExit):
        mod.resolve_threshold(tmp_path / "absent.json", "matched_filter", "far_ref", None)
