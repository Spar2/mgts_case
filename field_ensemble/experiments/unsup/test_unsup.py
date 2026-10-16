"""Слепой прогнозист: дырка не протекает, скор = |r|/(1.4826·MAD), горб внутри дырки не заучивается."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dts import export
from experiments.unsup.unsup import (
    BLIND_T, BLIND_X, BlindSpotForecaster, anomaly_from_residual, assemble_zones, blind_box,
    causal_time, delay_time, describe_zone, huber_masked, judge_zone, left_view, robust_abs_z,
    train_forecaster,
)


def _awaken(model):
    with torch.no_grad():
        for mod in model.modules():
            if isinstance(mod, nn.Conv2d):
                mod.weight.fill_(0.02)
                if mod.bias is not None:
                    mod.bias.zero_()
        model.head[-1].weight.fill_(0.02)
        model.head[-1].bias.zero_()


def test_left_view_last_tap_is_margin_away():
    conv = nn.Conv2d(1, 1, (1, 3), bias=False)
    with torch.no_grad():
        conv.weight.zero_()
        conv.weight[0, 0, 0, -1] = 1.0
    x = torch.zeros(1, 1, 1, 30)
    x[0, 0, 0, 10] = 5.0
    y = left_view(x, conv, margin=4)
    assert y.shape[-1] == 30
    assert y[0, 0, 0, 14].item() == pytest.approx(5.0)
    assert torch.count_nonzero(y[0, 0, 0, :14]) == 0


def test_delayed_causal_sees_only_past():
    conv = nn.Conv2d(1, 1, (3, 1), bias=False)
    with torch.no_grad():
        conv.weight.zero_()
        conv.weight[0, 0, -1, 0] = 1.0
    x = torch.zeros(1, 1, 20, 1)
    x[0, 0, 10, 0] = 7.0
    y = causal_time(delay_time(x, 5), conv)
    assert y[0, 0, 15, 0].item() == pytest.approx(7.0)
    assert y[0, 0, 14, 0].item() == pytest.approx(0.0)
    assert torch.count_nonzero(y[0, 0, :15, 0]) == 0


def test_blind_box_is_15_by_41():
    r0, r1, c0, c1 = blind_box(40, 80)
    assert (r1 - r0, c1 - c0) == (BLIND_T, BLIND_X)
    assert r0 <= 40 < r1 and c0 <= 80 < c1


def test_center_strip_and_future_do_not_leak():
    torch.manual_seed(0)
    model = BlindSpotForecaster()
    _awaken(model)
    model.eval()
    x = torch.zeros(1, 3, 80, 160)
    i, j = 40, 80
    y0 = model(x)
    assert y0.shape == (1, 1, 80, 160)
    r0, r1, c0, c1 = blind_box(i, j, model.blind_t, model.blind_x)
    # Полоса своего x видна соседям, поэтому сравниваем только прогноз в (i, j).
    leaked = x.clone()
    leaked[:, :2, :, c0:c1] = 50.0
    leaked[:, :2, i + 1:, :] = 50.0
    y1 = model(leaked)
    assert torch.equal(y0[:, :, i, j], y1[:, :, i, j])
    future = x.clone()
    future[:, :2, i + 1:, :] = 50.0
    # причинная сеть: строки ≤ i не зависят от будущего; более поздние строки — зависят
    assert torch.equal(y0[:, :, :i + 1], model(future)[:, :, :i + 1])
    # сосед сразу за дыркой в текущей строке обязан влиять
    near = x.clone()
    near[:, 0, i, c0 - 1] = 20.0
    y_near = model(near)
    assert not torch.equal(y0[:, :, i, j], y_near[:, :, i, j])
    # прошлое этого соседа на лаге blind_t тоже видно
    past = x.clone()
    past[:, 0, i - model.blind_t, c0 - 1] = 20.0
    y_past = model(past)
    assert not torch.equal(y0[:, :, i, j], y_past[:, :, i, j])


def test_huber_ignores_invalid():
    pred = torch.zeros(1, 1, 4, 4)
    target = torch.ones(1, 1, 4, 4)
    valid = torch.zeros(1, 1, 4, 4)
    assert float(huber_masked(pred, target, valid)) == 0.0
    valid[:] = 1
    assert float(huber_masked(pred, target, valid)) > 0


def test_robust_z_matches_mad_formula():
    r = np.array([[1., 2, 3, 4, 5, 6, 7, 8, 9]])
    valid = np.ones_like(r, bool)
    z = robust_abs_z(r, valid)
    med = 5.0
    mad = 2.0
    scale = 1.4826 * mad
    assert z[0, 0] == pytest.approx(abs(1.0) / scale, rel=1e-5)
    assert z[0, -1] == pytest.approx(9.0 / scale, rel=1e-5)
    # общий сдвиг строки не вычитается из числителя
    shift = r + 10
    zs = robust_abs_z(shift, valid)
    assert zs[0, 4] == pytest.approx(15.0 / scale, rel=1e-5)
    s, h = anomaly_from_residual(np.array([[-2., 3.]]), np.ones((1, 2), bool))
    assert s.shape == (1, 2) and h[0, 0] < 0 < h[0, 1]


def test_zones_export_roundtrip(tmp_path):
    T, W = 30, 40
    score = np.zeros((T, W), np.float32)
    score[10:16, 8:20] = 5
    hot = np.ones_like(score)
    valid = np.ones_like(score, bool)
    x = np.arange(W) * 0.25
    t = 1_700_000_000 + np.arange(T) * 120.0
    dT = np.zeros_like(score)
    dT[10:16, 8:20] = 1.5
    zones = assemble_zones(score, hot, valid, x, t, dT, thr=1.0)
    assert len(zones) == 1
    assert zones[0]["polarity"] == "hot"
    assert zones[0]["peak_dT_C"] == pytest.approx(1.5)
    assert zones[0]["alarm"] is True
    rows = list(export.zone_rows(zones, "1.csv", "abc", model_version="unsup-test"))
    assert list(rows[0].keys()) == export.FIELDS
    path = tmp_path / "zones_1.csv"
    assert export.write_csv(path, rows) == 1
    desc = describe_zone(zones[0], score, valid, x, t, 1.0)
    assert desc["width_m"] > 0 and desc["dur_h"] > 0 and desc["verdict"]


def test_judge_regime_and_thermal():
    regime = dict(row_flag_frac=0.4, width_frac=0.5, width_m=200, dur_h=0.3, abs_peak_C=1.2,
                  peak_dT_C=1.2, polarity="hot", touches_end=False)
    verdict, _ = judge_zone(regime)
    assert "режим" in verdict
    thermal = dict(row_flag_frac=0.01, width_frac=0.02, width_m=6, dur_h=1.5, abs_peak_C=1.1,
                   peak_dT_C=-1.1, polarity="cold", touches_end=False)
    verdict, reasons = judge_zone(thermal)
    assert "тепловое" in verdict
    assert any("охлаждение" in r for r in reasons)


def test_training_fits_background_and_keeps_hidden_bump():
    rng = np.random.default_rng(0)
    T, W = 48, 96
    X = np.zeros((3, T, W), np.float32)
    X[0] = 0.5
    X[1] = 0.5
    X[2] = 1
    X[0] += rng.normal(0, 0.02, (T, W)).astype(np.float32)
    model, hist = train_forecaster([X], minutes=0.25, max_steps=40, batch=4, tile_t=48, tile_x=96,
                                   probe_every=20, log_every=20, seed=0, tag="test ")
    probes = [h for h in hist if "bg" in h]
    assert probes[0]["bg"] > probes[-1]["bg"]
    assert probes[-1]["event"] > 4.0
    assert probes[-1]["bg"] < 0.25
