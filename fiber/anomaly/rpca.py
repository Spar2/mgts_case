"""Robust PCA (principal component pursuit) on a temperature field.

Low rank is the slow background. The sparse term is the local event.
The score handed to extract_zones is sign(S) * |S| / MAD.
"""

from __future__ import annotations

import numpy as np

from fiber.models.ssl_common import live_mask


def _svt(matrix: np.ndarray, tau: float) -> np.ndarray:
    left, values, right = np.linalg.svd(matrix, full_matrices=False)
    kept = np.maximum(values - tau, 0.0)
    return (left * kept) @ right


def _soft(matrix: np.ndarray, tau: float) -> np.ndarray:
    return np.sign(matrix) * np.maximum(np.abs(matrix) - tau, 0.0)


def decompose_rpca(matrix: np.ndarray, n_iter: int = 80) -> tuple[np.ndarray, np.ndarray]:
    """Inexact ALM for min ||L||_* + lambda ||S||_1 with L + S = matrix."""
    observed = np.asarray(matrix, dtype=np.float64)
    if observed.ndim != 2:
        raise ValueError("RPCA expects a 2D field")
    rows, cols = observed.shape
    lam = 1.0 / np.sqrt(max(rows, cols))
    spectral = np.linalg.norm(observed, ord=2)
    mu = 1.25 / spectral if spectral > 0 else 1.25
    mu_cap = mu * 1e7
    dual = np.zeros_like(observed)
    sparse = np.zeros_like(observed)
    low_rank = np.zeros_like(observed)
    scale = np.linalg.norm(observed, ord="fro")
    scale = scale if scale > 0 else 1.0
    for _ in range(n_iter):
        low_rank = _svt(observed - sparse + dual / mu, 1.0 / mu)
        sparse = _soft(observed - low_rank + dual / mu, lam / mu)
        gap = observed - low_rank - sparse
        dual = dual + mu * gap
        mu = min(mu * 1.5, mu_cap)
        if np.linalg.norm(gap, ord="fro") / scale < 1e-7:
            break
    return low_rank, sparse


def rpca_score(field: np.ndarray, valid: np.ndarray, n_iter: int = 80) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (signed score, low rank, sparse). Dead cells score 0."""
    mask = live_mask(field, valid)
    values = np.asarray(field, dtype=np.float64)
    fill = float(np.median(values[mask])) if np.any(mask) else 0.0
    observed = np.where(mask, values, fill)
    low_rank, sparse = decompose_rpca(observed, n_iter=n_iter)
    magnitude = np.abs(sparse)
    center = np.median(magnitude[mask]) if np.any(mask) else 0.0
    mad = 1.4826 * np.median(np.abs(magnitude[mask] - center)) if np.any(mask) else 0.0
    mad = float(mad) if np.isfinite(mad) and mad > 1e-6 else 1e-6
    score = np.sign(sparse) * (magnitude / mad)
    score = score.astype(np.float32)
    score[~mask] = 0.0
    return score, low_rank.astype(np.float32), sparse.astype(np.float32)
