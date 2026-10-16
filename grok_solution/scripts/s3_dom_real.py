"""Смесь ens_dom для реального прогона. models_release только читается.

score = 0.5 * (ансамбль выпуска, U-Net из models_release_dom).
Порог — far_1e-3 из models_release_dom/stage1_thresholds.json, не из карточек.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from dts import classic, events as E, features, grid, infer, models, postproc as PP

ROOT = Path(__file__).resolve().parents[1]


def load_dom(op: str = "far_1e-3", zones: str = "cc"):
    rel = ROOT / "models_release_dom"
    spec = json.loads((rel / "stage1_thresholds.json").read_text())
    try:
        thr = float(spec["thresholds"]["cnn_unet"][op])
    except KeyError as e:
        raise SystemExit(f"models_release_dom не содержит порог {op}") from e
    width = int(spec.get("width", 16))
    unet = models.build("unet", width)
    unet.load_state_dict(torch.load(rel / spec["unet_file"], map_location="cpu", weights_only=True))
    unet.eval()
    release = ROOT / "models_release"
    # Члены ансамбля и EventNet читаются из models_release. Каталог не изменяется.
    analyzer = infer.Analyzer(
        stage1=release / "det1_unet.pt",
        stage2=release / "eventnet.pt",
        lgbm=release / "lgbm.joblib",
        thresholds=release / "stage1_thresholds.json",
        op=op,
        zones=zones,
    )
    analyzer.thr = thr
    analyzer.op = op
    return analyzer, unet, spec


def run_blend(analyzer, unet, T, valid, x, t, postproc=None, zones=None):
    """Тот же выход, что Analyzer.run, но score и hot — смесь 1/2."""
    mode = analyzer.zones if zones is None else zones
    if mode not in ("cc", "pdrs_hybrid"):
        raise ValueError("zones must be 'cc' or 'pdrs_hybrid'")
    p = grid.plan(x, t)
    Tc, v = grid.resample(T, valid, p, "mean")
    X, aux = features.make_channels(Tc, v)
    ss, hh = [], []
    for kind, mdl in analyzer.members:
        if kind == "net":
            prob = models.predict_full(mdl, X)
        else:
            feat = classic.pixel_features(X)
            prob = np.moveaxis(mdl.predict_proba(feat).reshape(X.shape[1], X.shape[2], 3), -1, 0)
        ss.append(1.0 - prob[0])
        hh.append(prob[1] - prob[2])
    prob_u = models.predict_full(unet, X)
    score = 0.5 * (np.mean(ss, 0) + (1.0 - prob_u[0]))
    hot = 0.5 * (np.mean(hh, 0) + (prob_u[1] - prob_u[2]))
    if PP.resolve_enabled(postproc, analyzer.op):
        score = PP.score_for_analyzer(score, v, analyzer.thr, True)
    found = infer.pipeline_zones(score, hot, v, p["x"], p["t"], analyzer.thr, mode)
    P = analyzer.classify(X, aux["sigma_x"], p["x"], p["t"], found)
    for z, pr in zip(found, P):
        z["class"] = E.CLASSES[int(pr.argmax())]
        z["class_prob"] = {c: round(float(q), 3) for c, q in zip(E.CLASSES, pr)}
        z["alarm"] = z["class"] not in ("HVAC_Diurnal", "Sensor_Artifact")
        sl = (slice(z["t0"], z["t1"]), slice(z["x0"], z["x1"]))
        z["peak_dT_C"] = float(np.max(np.abs(aux["dT_time"][sl])))
    return dict(grid=p, X=X, valid=v, score=score, hotness=hot, zones=found, sigma=aux["sigma_x"], dT=aux["dT_time"])
