"""Классические (не-CNN) детекторы для сравнения на общем протоколе."""
from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter

SCALES = ((1.5, 3), (4, 8), (10, 20))   # (σ_t в строках по 120 c, σ_x в столбцах по 0.25 м)


def matched_filter(X):
    """Согласованный фильтр: сглаживание нормированных каналов гауссовым ядром ~ размера типичного события.
    Без обучения. score = max(|G*dT_time|, |G*dT_space|)."""
    v = X[2] > 0
    a = gaussian_filter(X[0].astype(np.float32), (2, 6)); b = gaussian_filter(X[1].astype(np.float32), (2, 6))
    n = gaussian_filter(v.astype(np.float32), (2, 6)) + 1e-3
    a, b = a / n, b / n
    return np.maximum(np.abs(a), np.abs(b)) * v, (a + b)


def pixel_features(X):
    f = [X[0], X[1]]
    for s in SCALES:
        f += [gaussian_filter(X[0].astype(np.float32), s), gaussian_filter(X[1].astype(np.float32), s)]
    f.append(gaussian_filter(np.abs(X[0]).astype(np.float32), (4, 8)))
    return np.stack(f, -1).reshape(-1, len(f)).astype(np.float32)
