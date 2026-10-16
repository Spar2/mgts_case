"""Каноническая сетка: dx = 0.25 м, dt = 120 с. Все модели работают на ней,
поэтому синтетика разных шаблонов и реальные файлы 1–6.csv (шаг 0.1–1 м, 10–240 с)
приводятся к одному масштабу «пикселя»."""
from __future__ import annotations
import numpy as np

DX_M = 0.25
DT_S = 120.0
MAX_ROW_GAP_S = 300.0  # как max_gap_s в эталонном пайплайне


def _axis_index(src: np.ndarray, step: float):
    src = np.asarray(src, dtype=np.float64)
    n = int(np.floor((src[-1] - src[0]) / step)) + 1
    centers = src[0] + step * np.arange(n)
    b = np.clip(np.round((src - src[0]) / step).astype(np.int64), 0, n - 1)
    nearest = np.clip(np.searchsorted(src, centers), 1, len(src) - 1)
    left = src[nearest - 1]; right = src[nearest]
    nearest = np.where(np.abs(centers - left) <= np.abs(right - centers), nearest - 1, nearest)
    dist = np.abs(src[nearest] - centers)
    return centers, b, nearest, dist


def plan(x: np.ndarray, t: np.ndarray, dx: float = DX_M, dt: float = DT_S):
    xc, bx, nx_, dxd = _axis_index(x, dx)
    tc, bt, nt_, dtd = _axis_index(t, dt)
    return dict(x=xc, t=tc, bx=bx, nx=nx_, bt=bt, nt=nt_, row_ok=dtd <= max(MAX_ROW_GAP_S / 2, dt), dx=dx, dt=dt)


def _reduce_axis(a: np.ndarray, w: np.ndarray, b, nearest, n, axis, how):
    a = np.moveaxis(a, axis, 0); w = np.moveaxis(w, axis, 0)
    if how == "nearest":
        out = a[nearest]; ow = w[nearest]
    else:
        shape = (n,) + a.shape[1:]
        if how == "mean":
            s = np.zeros(shape, np.float64); c = np.zeros(shape, np.float64)
            np.add.at(s, b, np.where(w, a, 0.0)); np.add.at(c, b, w.astype(np.float64))
            out = np.where(c > 0, s / np.maximum(c, 1), np.nan); ow = c > 0
        elif how == "maxabs":
            out = np.full(shape, -np.inf); 
            np.maximum.at(out, b, np.where(w, np.abs(a), -np.inf)); ow = np.isfinite(out)
            out = np.where(ow, out, np.nan)
        elif how == "max":
            out = np.zeros(shape, a.dtype); np.maximum.at(out, b, np.where(w, a, 0)); ow = np.ones(shape, bool)
        else:
            raise ValueError(how)
        empty = ~ow
        if empty.any():  # пустые бины (апсемплинг) — ближайший исходный узел
            fill = a[nearest]; fw = w[nearest]
            out = np.where(empty, fill, out); ow = np.where(empty, fw, ow)
    return np.moveaxis(out, 0, axis), np.moveaxis(ow.astype(bool), 0, axis)


def resample(a: np.ndarray, valid: np.ndarray, p: dict, how: str = "mean"):
    """a[T, X] -> канонич. сетка. how: mean (температура), nearest (метки), maxabs (скоры), max."""
    a = np.asarray(a); valid = np.asarray(valid, bool) & (np.isfinite(a) if a.dtype.kind == "f" else True)
    o, w = _reduce_axis(a, valid, p["bx"], p["nx"], len(p["x"]), 1, how)
    o, w = _reduce_axis(o, w, p["bt"], p["nt"], len(p["t"]), 0, how)
    w &= p["row_ok"][:, None]
    return o, w
