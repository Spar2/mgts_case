"""Метрики детекции при фиксированной доле ложных тревог (FAR) и событийные метрики."""
from __future__ import annotations
import numpy as np
import pandas as pd


def far_threshold(bg_scores: np.ndarray, far: float) -> float:
    """Порог, при котором доля фоновых клеток со score ≥ thr равна far."""
    s = np.sort(bg_scores)
    k = int(np.ceil((1 - far) * len(s))) - 1
    return float(s[np.clip(k, 0, len(s) - 1)]) + 1e-12


def confusion(pred, truth, valid):
    p = pred & valid; t = truth & valid
    tp = int((p & t).sum()); fp = int((p & ~t).sum()); fn = int((~p & t).sum())
    return tp, fp, fn


def prf(tp, fp, fn):
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    return dict(precision=P, recall=R, f1=2 * P * R / max(P + R, 1e-12), iou=tp / max(tp + fp + fn, 1))


def label_boxes(labels: pd.DataFrame, seq_id: str) -> pd.DataFrame:
    d = labels[labels.seq_id == seq_id].copy()
    E0 = pd.Timestamp("1970-01-01", tz="UTC")
    d["ts"] = (pd.to_datetime(d.t_start, utc=True, format="mixed") - E0).dt.total_seconds()
    d["te"] = (pd.to_datetime(d.t_end, utc=True, format="mixed") - E0).dt.total_seconds()
    return d


def event_pixels(row, x, t, ev):
    ix = (x >= row.dist_start_m - 0.25) & (x <= row.dist_end_m + 0.25)
    it = (t >= row.ts - 120) & (t <= row.te + 120)
    m = np.zeros(ev.shape, bool)
    if ix.any() and it.any():
        sub = ev[np.ix_(it, ix)] == int(row.mask_value)
        m[np.ix_(it, ix)] = sub
    return m


def amp_bin(a):
    return "<=2C" if a <= 2 else ("2-5C" if a <= 5 else ">5C")


def bootstrap_ci(per_seq: dict, keys, n=1000, seed=0):
    """per_seq: seq -> (tp,fp,fn). CI для IoU по бутстрэпу последовательностей."""
    rng = np.random.default_rng(seed); arr = np.array([per_seq[k] for k in keys], float)
    vals = []
    for _ in range(n):
        s = arr[rng.integers(0, len(arr), len(arr))].sum(0); vals.append(s[0] / max(s.sum(), 1))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
