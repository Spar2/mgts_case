"""Standard and modern DSP denoisers on masked DTS fields."""

from __future__ import annotations

import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

from fiber.denoise.base import apply_rows, nan_field, restore_invalid

try:
    import pywt
except ImportError:  # pragma: no cover
    pywt = None


def identity(raw_c: np.ndarray, valid: np.ndarray) -> np.ndarray:
    return restore_invalid(raw_c, valid)


def median_x(raw_c: np.ndarray, valid: np.ndarray, k: int = 3) -> np.ndarray:
    """Spatial nan-median along the fibre. Does not bridge invalid cells."""
    if k < 1 or k % 2 == 0:
        raise ValueError("median kernel must be a positive odd integer")
    if k > 3:
        def on_island(row: np.ndarray) -> np.ndarray:
            padded = np.pad(row, (k // 2, k // 2), constant_values=np.nan)
            return np.nanmedian(sliding_window_view(padded, k), axis=-1)
        return apply_rows(raw_c, valid, on_island, min_len=1)
    field = nan_field(raw_c, valid)
    radius = k // 2
    padded = np.pad(field, ((0, 0), (radius, radius)), constant_values=np.nan)
    windows = sliding_window_view(padded, k, axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = np.nanmedian(windows, axis=-1)
    return restore_invalid(out, valid)


def median_xt(raw_c: np.ndarray, valid: np.ndarray, kt: int = 3, kx: int = 3) -> np.ndarray:
    """2D nan-median on the waterfall (time × distance)."""
    if kt < 1 or kx < 1 or kt % 2 == 0 or kx % 2 == 0:
        raise ValueError("median_xt kernels must be positive odd integers")
    field = nan_field(raw_c, valid)
    pt, px = kt // 2, kx // 2
    padded = np.pad(field, ((pt, pt), (px, px)), constant_values=np.nan)
    windows = sliding_window_view(padded, (kt, kx))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = np.nanmedian(windows, axis=(-2, -1))
    return restore_invalid(out, valid)


def gaussian_x(raw_c: np.ndarray, valid: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    def _fn(row: np.ndarray) -> np.ndarray:
        return gaussian_filter1d(row, sigma=sigma, mode="nearest")

    return apply_rows(raw_c, valid, _fn)


def savgol_x(raw_c: np.ndarray, valid: np.ndarray, window: int = 11, poly: int = 2) -> np.ndarray:
    window = int(window)
    if window % 2 == 0:
        window += 1

    def _fn(row: np.ndarray) -> np.ndarray:
        w = min(window, row.size if row.size % 2 == 1 else row.size - 1)
        w = max(w, poly + 2 if (poly + 2) % 2 == 1 else poly + 3)
        if w > row.size:
            return row
        return savgol_filter(row, window_length=w, polyorder=min(poly, w - 1), mode="interp")

    return apply_rows(raw_c, valid, _fn, min_len=max(5, poly + 2))


def bilateral_x(
    raw_c: np.ndarray,
    valid: np.ndarray,
    radius: int = 5,
    spatial_sigma: float = 2.5,
    value_sigma: float = 1.5,
) -> np.ndarray:
    """1D bilateral filter on each valid run (edge-preserving)."""
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    spatial = np.exp(-0.5 * (offsets / spatial_sigma) ** 2)
    inv_val = 1.0 / (value_sigma ** 2)

    def _fn(row: np.ndarray) -> np.ndarray:
        padded = np.pad(row, radius, mode="edge")
        windows = sliding_window_view(padded, 2 * radius + 1)
        center = row[:, None]
        range_w = np.exp(-0.5 * (windows - center) ** 2 * inv_val)
        weights = spatial * range_w
        denom = weights.sum(axis=1)
        return (weights * windows).sum(axis=1) / np.maximum(denom, 1e-12)

    return apply_rows(raw_c, valid, _fn, min_len=2 * radius + 1)


def wavelet_x(raw_c: np.ndarray, valid: np.ndarray, wavelet: str = "db4") -> np.ndarray:
    """VisuShrink soft threshold on wavelet coefficients (Donoho–Johnstone)."""
    if pywt is None:
        raise RuntimeError("PyWavelets is required for wavelet_x")

    def _fn(row: np.ndarray) -> np.ndarray:
        level = min(pywt.dwt_max_level(row.size, pywt.Wavelet(wavelet).dec_len), 4)
        level = max(level, 1)
        coeffs = pywt.wavedec(row, wavelet, level=level)
        detail = coeffs[-1]
        sigma = np.median(np.abs(detail - np.median(detail))) / 0.6745
        if not np.isfinite(sigma) or sigma <= 0:
            return row
        thresh = sigma * np.sqrt(2.0 * np.log(row.size))
        rec = [coeffs[0]] + [pywt.threshold(c, thresh, mode="soft") for c in coeffs[1:]]
        out = pywt.waverec(rec, wavelet)
        return out[: row.size]

    return apply_rows(raw_c, valid, _fn, min_len=16)


def _bayes_threshold(detail: np.ndarray, sigma: float) -> float:
    """BayesShrink threshold for one wavelet level (Chang, Yu, Vetterli)."""
    var = float(np.mean(np.asarray(detail, dtype=np.float64) ** 2))
    var_signal = max(var - sigma * sigma, 0.0)
    if var_signal <= 1e-12:
        return float(np.max(np.abs(detail))) if detail.size else 0.0
    return (sigma * sigma) / np.sqrt(var_signal)


def wavelet_bayes(raw_c: np.ndarray, valid: np.ndarray, wavelet: str = "db4") -> np.ndarray:
    """Level-adaptive BayesShrink. Usually keeps edges better than VisuShrink."""
    if pywt is None:
        raise RuntimeError("PyWavelets is required for wavelet_bayes")

    def _fn(row: np.ndarray) -> np.ndarray:
        level = min(pywt.dwt_max_level(row.size, pywt.Wavelet(wavelet).dec_len), 4)
        level = max(level, 1)
        coeffs = pywt.wavedec(row, wavelet, level=level)
        detail = coeffs[-1]
        sigma = np.median(np.abs(detail - np.median(detail))) / 0.6745
        if not np.isfinite(sigma) or sigma <= 0:
            return row
        rec = [coeffs[0]]
        for band in coeffs[1:]:
            rec.append(pywt.threshold(band, _bayes_threshold(band, float(sigma)), mode="soft"))
        out = pywt.waverec(rec, wavelet)
        return out[: row.size]

    return apply_rows(raw_c, valid, _fn, min_len=16)


def nlm_x(
    raw_c: np.ndarray,
    valid: np.ndarray,
    search: int = 7,
    patch: int = 3,
    h_mult: float = 1.0,
) -> np.ndarray:
    """1D non-local means (Buades). Weights come from similar neighbouring patches.

    `h_mult` scales the decay. The short default search barely leaves the
    local neighbourhood on a 1 m grid, so smooth sections stay noisy.
    """
    from scipy.ndimage import uniform_filter1d

    if search < 3 or search % 2 == 0 or patch < 1 or patch % 2 == 0:
        raise ValueError("search and patch must be positive odd integers")
    if h_mult <= 0:
        raise ValueError("h_mult must be positive")

    def _fn(row: np.ndarray) -> np.ndarray:
        sigma = 1.4826 * np.median(np.abs(np.diff(row))) / np.sqrt(2.0)
        if not np.isfinite(sigma) or sigma <= 1e-6:
            sigma = 0.2
        h = max(float(sigma), 0.25) * float(h_mult)
        h2 = h * h
        radius = search // 2
        acc = row.astype(np.float64).copy()
        wsum = np.ones(row.shape, dtype=np.float64)
        padded = np.pad(row.astype(np.float64), radius, mode="reflect")
        for offset in range(-radius, radius + 1):
            if offset == 0:
                continue
            shifted = padded[radius + offset:radius + offset + row.size]
            dist = uniform_filter1d((row - shifted) ** 2, size=patch, mode="nearest")
            weight = np.exp(-dist / h2)
            acc += weight * shifted
            wsum += weight
        return acc / np.maximum(wsum, 1e-12)

    return apply_rows(raw_c, valid, _fn, min_len=search)


def nlm_wide(raw_c: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Non-local means with a 21-sample search and a softer decay."""
    return nlm_x(raw_c, valid, search=21, patch=5, h_mult=1.5)


def wiener_x(raw_c: np.ndarray, valid: np.ndarray, mysize: int = 5) -> np.ndarray:
    """Local Wiener filter along the fibre. Classical stationary-noise smoother."""
    from scipy.signal import wiener

    if mysize < 3 or mysize % 2 == 0:
        raise ValueError("wiener window must be an odd integer >= 3")

    def _fn(row: np.ndarray) -> np.ndarray:
        out = np.asarray(wiener(row, mysize=mysize), dtype=np.float64)
        return out

    return apply_rows(raw_c, valid, _fn, min_len=mysize)


def tv_x(raw_c: np.ndarray, valid: np.ndarray, weight: float = 0.35, n_iter: int = 80) -> np.ndarray:
    """1D Chambolle total-variation denoise on each valid run."""

    def _fn(row: np.ndarray) -> np.ndarray:
        u = row.copy()
        p = np.zeros_like(row)
        tau = 0.25
        for _ in range(n_iter):
            grad = np.diff(u, append=u[-1])
            p += tau * grad
            p /= np.maximum(1.0, np.abs(p))
            div = np.diff(p, prepend=0.0)
            u = row + weight * div
        return u

    return apply_rows(raw_c, valid, _fn, min_len=4)


def temporal_median(raw_c: np.ndarray, valid: np.ndarray, k: int = 3) -> np.ndarray:
    """Nan-median along time at each metre. Keeps spatial edges intact."""
    if k < 1 or k % 2 == 0:
        raise ValueError("temporal kernel must be a positive odd integer")
    field = nan_field(raw_c, valid)
    radius = k // 2
    padded = np.pad(field, ((radius, radius), (0, 0)), constant_values=np.nan)
    windows = sliding_window_view(padded, k, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = np.nanmedian(windows, axis=-1)
    return restore_invalid(out, valid)
