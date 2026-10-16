"""Общие операции классического CV на сетке dx=0.25 м, dt=120 с.

cv2 импортируется только внутри вызовов. Квадратный medianBlur по одной строке
даёт одномерную медиану вдоль кабеля: у строки высотой 1 ядро копирует её же
(BORDER_REPLICATE), поэтому медиана окна k×k совпадает с медианой k отсчётов.
Большое k на float32 OpenCV не принимает — квантование в uint8, шаг около
диапазона/255.
"""
from __future__ import annotations

import os

import numpy as np
from scipy.ndimage import gaussian_filter


def _cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "Нужен opencv-python-headless: pip install -r requirements-cv.txt"
        ) from exc
    cv2.setNumThreads(int(os.environ.get("CV_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "2"))))
    return cv2


def axes_steps(x: np.ndarray, t: np.ndarray) -> tuple[float, float]:
    """Типичные dx (м) и dt (с). На канонической сетке это 0.25 и 120."""
    x = np.asarray(x, np.float64)
    t = np.asarray(t, np.float64)
    dx = float(np.median(np.diff(x))) if x.size > 1 else 0.25
    dt = float(np.median(np.diff(t))) if t.size > 1 else 120.0
    if not np.isfinite(dx) or dx <= 0:
        dx = 0.25
    if not np.isfinite(dt) or dt <= 0:
        dt = 120.0
    return dx, dt


def fill_invalid(T: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Копия поля: невалидные клетки заменены медианой своего столбца.

    В кэше невалидные температуры уже обнулены (`nan_to_num`), ноль — не холод.
    """
    valid = np.asarray(valid, bool)
    work = np.where(valid, np.asarray(T, np.float32), np.nan)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            col = np.nanmedian(work, axis=0)
    if not np.isfinite(col).any():
        return np.zeros(work.shape, np.float32)
    fallback = float(np.nanmedian(col))
    col = np.where(np.isfinite(col), col, fallback)
    out = np.where(np.isfinite(work), work, col[None, :]).astype(np.float32)
    return np.nan_to_num(out, nan=fallback, posinf=fallback, neginf=fallback)


def _odd_limit(requested: int, limit: int) -> int:
    window = min(int(requested), int(limit))
    if window < 1:
        return 1
    if window % 2 == 0:
        window -= 1
    return max(window, 1)


def median_axis1(values: np.ndarray, k: int) -> np.ndarray:
    """Медианный фильтр только вдоль оси x (ось 1). k — полная ширина окна."""
    values = np.asarray(values, np.float32)
    width = int(values.shape[1]) if values.ndim == 2 else 0
    k = _odd_limit(k, width)
    if values.size == 0 or k < 3:
        return values.copy()
    cv2 = _cv2()
    finite = np.isfinite(values)
    if not finite.all():
        med = float(np.median(values[finite])) if finite.any() else 0.0
        work = np.where(finite, values, med).astype(np.float32)
    else:
        work = values
    lo = float(work.min())
    hi = float(work.max())
    if not np.isfinite(lo) or hi - lo < 1e-6:
        return np.full_like(work, lo if np.isfinite(lo) else 0.0)
    if hi - lo > 80.0:
        lo, hi = np.percentile(work, [0.5, 99.5]).astype(np.float64)
        if hi - lo < 1e-6:
            return np.full_like(work, float(lo))
    scale = np.float32(255.0 / (hi - lo))
    u8 = np.clip((work - np.float32(lo)) * scale, 0, 255).astype(np.uint8)
    out_u = np.empty_like(u8)
    for i in range(u8.shape[0]):
        out_u[i] = cv2.medianBlur(u8[i:i + 1], k)[0]
    return out_u.astype(np.float32) / scale + np.float32(lo)


def temporal_median_window(n_times: int, window_steps: int) -> int:
    if n_times <= 1:
        return 1
    window = min(max(int(window_steps), 1), int(n_times))
    if window % 2 == 0:
        window -= 1
    return max(window, 1)


def slow_temporal_background(matrix: np.ndarray, window_steps: int = 601, stride: int = 4) -> np.ndarray:
    """Скользящая медиана по времени в каждом столбце.

    Окно сдвигается внутрь записи и не повторяет крайний отсчёт. При длинном
    ряде медианы считаются с шагом `stride` и линейно интерполируются: окно
    в сотни шагов меняется медленно. Короткие ряды (мало окон) — без шага.
    """
    values = np.asarray(matrix, np.float32)
    n_times, n_space = values.shape
    window = temporal_median_window(n_times, window_steps)
    if window <= 1:
        return np.zeros((n_times, n_space), np.float32)
    if window >= n_times:
        mid = n_times // 2
        med = np.partition(values, mid, axis=0)[mid]
        return np.repeat(med[None, :], n_times, axis=0)
    n_windows = n_times - window + 1
    step = 1 if n_windows <= 24 else max(int(stride), 1)
    starts = list(range(0, n_windows, step))
    if starts[-1] != n_windows - 1:
        starts.append(n_windows - 1)
    mid = window // 2
    meds = np.empty((len(starts), n_space), np.float32)
    for i, start in enumerate(starts):
        meds[i] = np.partition(values[start:start + window], mid, axis=0)[mid]
    half = window // 2
    index = np.clip(np.arange(n_times) - half, 0, n_windows - 1).astype(np.float64)
    xp = np.asarray(starts, np.float64)
    right = np.clip(np.searchsorted(xp, index, side="left"), 1, len(xp) - 1)
    left = right - 1
    span = xp[right] - xp[left]
    weight = np.where(span > 0, (index - xp[left]) / span, 0.0).astype(np.float32)
    return (1.0 - weight)[:, None] * meds[left] + weight[:, None] * meds[right]


def remove_common_drift(matrix: np.ndarray) -> np.ndarray:
    """Вычитание медленного общекабельного дрейфа без пространственного сглаживания.

    Дрейф — низкочастотная составляющая медианного профиля по длине, как в
    denoise_matrix референса. Локальный нагрев и широкий холодный участок остаются.
    """
    values = np.asarray(matrix, np.float32)
    if values.shape[0] <= 1:
        return values.copy()
    profile = np.median(values, axis=1)
    sigma = max(5.0, values.shape[0] / 20.0)
    drift = gaussian_filter(profile.astype(np.float32), sigma=sigma)
    return (values - drift[:, None] + np.float32(np.median(values))).astype(np.float32)


def denoise_matrix(matrix: np.ndarray, sigma_time: float = 1.5, sigma_space: float = 2.5) -> np.ndarray:
    """Лёгкое гауссово сглаживание и вычитание медленного общекабельного дрейфа.

    Повторяет denoise_matrix из PiorlZ/test2_larpathon (mgts_anomaly.py).
    """
    denoised = gaussian_filter(np.asarray(matrix, np.float32), sigma=(sigma_time, sigma_space))
    return remove_common_drift(denoised)


def blur_masked(img: np.ndarray, valid: np.ndarray, sigma_t: float, sigma_x: float) -> np.ndarray:
    """Гауссово сглаживание, невалидные клетки не тянут среднее."""
    cv2 = _cv2()
    w = np.asarray(valid, np.float32)
    num = cv2.GaussianBlur(np.asarray(img, np.float32) * w, (0, 0), sigmaX=float(sigma_x), sigmaY=float(sigma_t), borderType=cv2.BORDER_REPLICATE)
    den = cv2.GaussianBlur(w, (0, 0), sigmaX=float(sigma_x), sigmaY=float(sigma_t), borderType=cv2.BORDER_REPLICATE)
    return num / np.maximum(den, 1e-3)


def morph(mask: np.ndarray, op: str, ksize_xt: tuple[int, int]) -> np.ndarray:
    """Морфология. ksize_xt = (ширина по x, высота по t) — порядок OpenCV."""
    cv2 = _cv2()
    kx, kt = int(ksize_xt[0]), int(ksize_xt[1])
    kx = max(kx, 1)
    kt = max(kt, 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, kt))
    image = (np.asarray(mask, bool).astype(np.uint8)) * 255
    if op == "open":
        out = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    elif op == "close":
        out = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    else:
        raise ValueError(op)
    return out > 0
