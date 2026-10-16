"""Флаг static инференса: off оставляет скор победителя тем же объектом.

Ветка on — выбранное слияние STATIC2 (max, шаг 800). Реальные CSV здесь не читаются.
"""
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(2)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dts import events, infer
from dts.static_net import (
    INFER_TIME_OVERLAP, INFER_TIME_WIN, fuse, predict_map, predict_map_windows,
    residual_from_temperature,
)

ROOT = Path(__file__).resolve().parents[1]
WINNER_THR = 0.8608087301264272


def test_default_flag_is_off_and_keeps_the_same_score_object():
    assert inspect.signature(infer.Analyzer.__init__).parameters["static"].default == "off"
    assert infer._STATIC == ("off", "on")
    score = np.arange(12, dtype=np.float32).reshape(3, 4)
    before = score.tobytes()
    out = infer.score_with_static(score, None, None, 0.25, "off")
    assert out is score
    assert out.tobytes() == before
    thr = infer.operating_threshold(WINNER_THR, None, "off")
    assert thr is WINNER_THR


def test_unknown_static_rejected_before_models():
    try:
        infer.Analyzer(static="branch")
    except ValueError as exc:
        assert "off" in str(exc) and "on" in str(exc)
    else:
        raise AssertionError("ожидался ValueError")
    try:
        infer.operating_threshold(0.5, None, "max")
    except ValueError as exc:
        assert "off" in str(exc) and "on" in str(exc)
    else:
        raise AssertionError("ожидался ValueError")
    try:
        infer.score_with_static(np.zeros((2, 2), np.float32), None, None, 0.25, "blend")
    except ValueError as exc:
        assert "off" in str(exc) and "on" in str(exc)
    else:
        raise AssertionError("ожидался ValueError")


def test_spec_is_static2_max_step_800():
    spec = infer.static_spec()
    raw = json.loads((ROOT / "artifacts/static/s3_static2.json").read_text())
    assert spec["name"] == raw["selected"] == "max"
    assert spec["mode"] == "max"
    assert spec["scale"] == 1.0
    assert spec["gate"] == 0.0
    assert spec["step"] == 800
    assert spec["ckpt"].endswith("static_net2_s800.pt")
    assert abs(spec["threshold"] - 0.865895) < 1e-6
    assert abs(spec["threshold"] - raw["val"]["max"]["threshold"]) < 1e-12
    assert spec["threshold"] > WINNER_THR + 1e-4
    net = infer.load_static_net(infer.static_ckpt_path(spec))
    assert net.cin == 17 and net.max_rows == 256 and net.n_bins == 8
    got = infer.operating_threshold(WINNER_THR, spec, "on")
    assert abs(got - spec["threshold"]) < 1e-12


def test_selected_max_never_lowers_ensemble_and_frozen_thr_is_stricter():
    spec = infer.static_spec()
    assert spec["mode"] in ("max", "blend")
    rng = np.random.default_rng(1)
    score = rng.random((8, 32)).astype(np.float32)
    before = score.tobytes()
    p = rng.random(32).astype(np.float32)
    fused = fuse(score, p, spec["mode"], spec["scale"], spec["gate"])
    assert np.all(fused + 1e-6 >= score)
    assert score.tobytes() == before
    # между порогом выпуска и порогом слитой карты карта не режется
    mid = np.full((6, 16), 0.863, np.float32)
    held = fuse(mid, np.zeros(16, np.float32), spec["mode"], spec["scale"], spec["gate"])
    x = np.arange(16, dtype=np.float64) * 0.5
    t = np.arange(6, dtype=np.float64) * 60.0
    hot = np.zeros_like(held)
    valid = np.ones_like(held, bool)
    assert infer.pipeline_zones(held, hot, valid, x, t, spec["threshold"]) == []
    assert len(infer.pipeline_zones(held, hot, valid, x, t, WINNER_THR)) > 0
    cc = events.zones_from_prob(held, hot, valid, x, t, WINNER_THR)
    assert infer.pipeline_zones(held, hot, valid, x, t, WINNER_THR) == cc


def test_on_lifts_narrow_static_column_and_matches_fuse():
    spec = infer.static_spec()
    net = infer.load_static_net(infer.static_ckpt_path(spec))
    rng = np.random.default_rng(0)
    rows, cols = 64, 640
    T = rng.normal(18.0, 0.04, size=(rows, cols)).astype(np.float32)
    hot_x = slice(300, 316)  # 4 м при dx 0.25
    T[:, hot_x] += 10.0
    valid = np.ones_like(T, bool)
    score = np.full_like(T, 0.15)
    before = score.tobytes()
    fused = infer.apply_static_score(score, T, valid, 0.25, net, spec)
    assert fused.shape == score.shape
    assert fused.dtype == np.float32
    assert np.all(fused + 1e-5 >= score)
    assert score.tobytes() == before
    bump = float(fused[:, hot_x].max())
    far = float(fused[:, 40:80].max())
    # игрушечный ансамбль 0.15 не обязан перейти порог FAR; столбец должен подняться над фоном
    assert bump > 0.5
    assert bump > far + 0.2
    resid, rv = residual_from_temperature(T, valid, dx=0.25, max_rows=int(net.max_rows))
    column_p = predict_map(net, resid, rv)
    again = fuse(score, column_p, spec["mode"], spec["scale"], spec["gate"])
    assert np.allclose(fused, again)
    # то же правило: p выше замороженного порога открывает зону, сырой скор — нет
    p = np.zeros(cols, np.float32)
    p[hot_x] = 0.99
    lifted = fuse(score, p, spec["mode"], spec["scale"], spec["gate"])
    assert float(lifted[:, hot_x].min()) >= spec["threshold"]
    assert np.all(lifted + 1e-6 >= score)
    x = np.arange(cols, dtype=np.float64) * 0.25
    t = np.arange(rows, dtype=np.float64) * 120.0
    hot = np.ones_like(lifted)
    zones = infer.pipeline_zones(lifted, hot, np.ones_like(lifted, bool), x, t, spec["threshold"])
    assert any(z["x0"] <= 308 < z["x1"] for z in zones)
    assert infer.pipeline_zones(score, hot, np.ones_like(score, bool), x, t, spec["threshold"]) == []


def test_short_record_is_one_window_and_long_record_is_max_of_windows():
    spec = infer.static_spec()
    net = infer.load_static_net(infer.static_ckpt_path(spec))
    assert INFER_TIME_WIN == 640 and INFER_TIME_OVERLAP == 0.5
    rng = np.random.default_rng(2)
    rows, cols = 80, 200
    T = rng.normal(18.0, 0.05, size=(rows, cols)).astype(np.float32)
    T[:, 90:102] += 8.0
    valid = np.ones_like(T, bool)
    score = np.full_like(T, 0.2)
    fused = infer.apply_static_score(score, T, valid, 0.25, net, spec)
    resid, rv = residual_from_temperature(T, valid, dx=0.25, max_rows=int(net.max_rows))
    again = fuse(score, predict_map(net, resid, rv), spec["mode"], spec["scale"], spec["gate"])
    assert np.allclose(fused, again)
    R = resid
    assert np.array_equal(predict_map_windows(net, R, rv), predict_map(net, R, rv))

    long_rows = INFER_TIME_WIN + 200
    T = rng.normal(18.0, 0.05, size=(long_rows, cols)).astype(np.float32)
    T[:180, 90:102] += 12.0
    valid = np.ones_like(T, bool)
    score = np.full_like(T, 0.2)
    fused = infer.apply_static_score(score, T, valid, 0.25, net, spec)
    resid, rv = residual_from_temperature(T, valid, dx=0.25, max_rows=long_rows)
    manual = predict_map_windows(net, resid, rv)
    assert np.allclose(fused, fuse(score, manual, spec["mode"], spec["scale"], spec["gate"]))
    win = INFER_TIME_WIN
    stride = int(win * (1.0 - INFER_TIME_OVERLAP))
    starts = list(range(0, long_rows - win + 1, stride))
    if starts[-1] != long_rows - win:
        starts.append(long_rows - win)
    assert len(starts) > 1
    stacked = np.stack([predict_map(net, resid[s:s + win], rv[s:s + win]) for s in starts], 0)
    assert np.allclose(manual, stacked.max(0))
    whole = residual_from_temperature(T, valid, dx=0.25, max_rows=int(net.max_rows))
    single = predict_map(net, whole[0], whole[1])
    assert not np.allclose(manual, single)
