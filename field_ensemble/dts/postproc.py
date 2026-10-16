"""Пост-обработка карты score без переобучения.

Варианты не смотрят на метки: только score и маска valid. Порог FAR по-прежнему
подбирается снаружи, на фоне val. Analyzer включает сохранённый вариант только
если тот лучше identity на val; иначе флаг по умолчанию выключен.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter, uniform_filter1d

MAD_TO_SIGMA = 1.4826
EPS = 1e-3
# Окно по x: 21 отсчёт = 5.25 м при шаге 0.25 м. Половина окна — 10.
CFAR_X_HALF = 10
# 2D-окно: 31 строка (31×120 с = 62 мин) на 41 отсчёт (10.25 м).
# Медиана на полной карте слишком дорогая, поэтому сетка с шагом 4 и билинейное растяжение.
CFAR_2D_T = 31
CFAR_2D_X = 41
CFAR_2D_STEP = 4
OP = "far_1e-3"
IOU_GUARD = 0.01
FAR_SANITY = 0.0025

VARIANTS = (
    "identity",
    "cfar_global",
    "cfar_x",
    "cfar_x_w21",
    "cfar_2d",
    "box3",
    "box7",
    "box15",
    "gauss3",
    "gauss7",
    "gauss15",
    "cfar_x_box7",
    "cfar_x_gauss7",
)

# --- deployed:start ---
DEPLOYED_VARIANT = 'box15'
THRESHOLD = 0.8486673235903249  # порог val FAR 1e-3 в пространстве варианта; None => не сдвигать
ENABLED = False  # прирост val IoU +0.0006 (шум); порог привязан к одному методу -> выкл. по умолчанию
# --- deployed:end ---


def resolve_enabled(flag, op: str) -> bool:
    """None — автомат (ENABLED и рабочая точка OP). Явный флаг перекрывает автомат."""
    if flag is None:
        return bool(ENABLED) and op == OP and DEPLOYED_VARIANT != "identity"
    return bool(flag)


def shift_to_threshold(transformed, src_thr: float, dst_thr: float) -> np.ndarray:
    """Сдвиг: transformed >= src_thr совпадает с out >= dst_thr."""
    out = np.asarray(transformed, np.float32) + np.float32(float(dst_thr) - float(src_thr))
    return out.astype(np.float32, copy=False)


def score_for_analyzer(score, valid, analyzer_thr: float, enabled: bool) -> np.ndarray:
    """Карта, которую Analyzer сравнивает со своим порогом.

    Выключенный флаг возвращает исходный score. Включённый — вариант из
    DEPLOYED_VARIANT, сдвинутый так, чтобы старый порог остался рабочей точкой.
    """
    raw = np.asarray(score, np.float32)
    if not enabled:
        return raw
    transformed = apply(raw, valid, DEPLOYED_VARIANT)
    if THRESHOLD is None or DEPLOYED_VARIANT == "identity":
        return transformed
    return shift_to_threshold(transformed, THRESHOLD, analyzer_thr)


def choose(rows: dict, iou_guard: float = IOU_GUARD, far_sanity: float = FAR_SANITY):
    """Выбор по val. Сначала отсечь просадку IoU и сломанный FAR, затем
    weak recall, event recall, IoU, F1. Включать флаг только если ключ строго выше identity.
    """
    if "identity" not in rows:
        raise KeyError("identity")
    base = rows["identity"]

    def admissible(r):
        return float(r["iou"]) >= float(base["iou"]) - float(iou_guard) and float(r["far"]) <= float(far_sanity)

    def key(r):
        return (float(r["weak_recall"]), float(r["event_recall"]), float(r["iou"]), float(r["f1"]))

    pool = {name: r for name, r in rows.items() if admissible(r)}
    pool.setdefault("identity", base)
    best = max(pool, key=lambda name: key(pool[name]))
    enable = best != "identity" and key(pool[best]) > key(base)
    rejected = [name for name in rows if name not in pool]
    return {"variant": best, "enable": bool(enable), "rejected": rejected}


def apply(score, valid, variant: str = "identity") -> np.ndarray:
    """Преобразование карты (T, X). Метки не используются."""
    if variant not in VARIANTS:
        raise ValueError(variant)
    s = np.asarray(score, np.float32)
    v = np.asarray(valid, bool)
    if s.shape != v.shape:
        raise ValueError(f"shape {s.shape} vs valid {v.shape}")
    if variant == "identity":
        return s.copy()
    if variant == "cfar_global":
        med, mad = _median_mad_global(s, v)
        return _zscore(s, v, med, mad)
    if variant == "cfar_x":
        med, mad = _nanmedian_cols(s, v)
        return _zscore(s, v, med, mad)
    if variant == "cfar_x_w21":
        med, mad = _nanmedian_cols(s, v)
        k = 2 * CFAR_X_HALF + 1
        med = _smooth_stat(med, k)
        mad = _smooth_stat(mad, k)
        return _zscore(s, v, med, mad)
    if variant == "cfar_2d":
        return _cfar_2d(s, v)
    if variant.startswith("box") or variant.startswith("gauss"):
        kind = "box" if variant.startswith("box") else "gauss"
        k = int(variant[len(kind):])
        return temporal_max(s, v, kind, k)
    if variant == "cfar_x_box7":
        return apply(temporal_max(s, v, "box", 7), v, "cfar_x")
    if variant == "cfar_x_gauss7":
        return apply(temporal_max(s, v, "gauss", 7), v, "cfar_x")
    raise ValueError(variant)


def apply_deployed(score, valid) -> np.ndarray:
    """Лучший по val вариант ансамбля, без сдвига порога."""
    return apply(score, valid, DEPLOYED_VARIANT)


def temporal_max(score, valid, kind: str, k: int) -> np.ndarray:
    """Сглаживание вдоль времени и max(исходный, сглаженный). Дыры valid не разбавляют среднее."""
    if k < 1 or k % 2 == 0:
        raise ValueError(k)
    s = np.asarray(score, np.float32)
    v = np.asarray(valid, bool)
    w = v.astype(np.float32)
    x = np.where(v, s, np.float32(0))
    if kind == "box":
        num = uniform_filter1d(x, size=k, axis=0, mode="nearest")
        den = uniform_filter1d(w, size=k, axis=0, mode="nearest")
    elif kind == "gauss":
        # sigma растёт с окном: 3 → 0.5, 7 → 1.5, 15 → 3.5 строки.
        sigma = (k - 1) / 4.0
        num = gaussian_filter1d(x, sigma=sigma, axis=0, mode="nearest")
        den = gaussian_filter1d(w, sigma=sigma, axis=0, mode="nearest")
    else:
        raise ValueError(kind)
    sm = num / np.maximum(den, np.float32(1e-6))
    out = np.maximum(s, sm)
    return np.where(v, out, s).astype(np.float32)


def _zscore(s, valid, med, mad, eps: float = EPS) -> np.ndarray:
    scale = np.maximum(np.float32(MAD_TO_SIGMA) * np.asarray(mad, np.float32), np.float32(eps))
    z = (np.asarray(s, np.float32) - np.asarray(med, np.float32)) / scale
    z = np.where(valid, z, np.float32(0))
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _median_mad_global(s, valid):
    vals = np.asarray(s, np.float32)[np.asarray(valid, bool)]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.float32(0), np.float32(0)
    med = np.median(vals)
    mad = np.median(np.abs(vals - med))
    return np.float32(med), np.float32(mad)


def _nanmedian_cols(s, valid):
    a = np.where(valid, np.asarray(s, np.float32), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(a, axis=0)
        mad = np.nanmedian(np.abs(a - med), axis=0)
    return np.nan_to_num(med, nan=0.0).astype(np.float32), np.nan_to_num(mad, nan=0.0).astype(np.float32)


def _smooth_stat(stat, k: int) -> np.ndarray:
    stat = np.asarray(stat, np.float32)
    k = _odd_cap(k, int(stat.shape[0]))
    if k <= 1:
        return stat
    return median_filter(stat, size=k, mode="nearest").astype(np.float32)


def _fill_invalid(s, valid) -> np.ndarray:
    med, _ = _nanmedian_cols(s, valid)
    if not np.isfinite(med).all():
        med = np.nan_to_num(med, nan=0.0)
    return np.where(valid, s, med).astype(np.float32)


def _upsample(coarse, t: int, w: int) -> np.ndarray:
    coarse = np.asarray(coarse, np.float32)
    ct, cw = coarse.shape
    if ct == t and cw == w:
        return coarse
    ts = np.linspace(0, ct - 1, t)
    xs = np.linspace(0, cw - 1, w)
    t0 = np.floor(ts).astype(np.int32)
    x0 = np.floor(xs).astype(np.int32)
    t1 = np.minimum(t0 + 1, ct - 1)
    x1 = np.minimum(x0 + 1, cw - 1)
    wt = (ts - t0).astype(np.float32)[:, None]
    wx = (xs - x0).astype(np.float32)[None, :]
    c00 = coarse[t0][:, x0]
    c01 = coarse[t0][:, x1]
    c10 = coarse[t1][:, x0]
    c11 = coarse[t1][:, x1]
    return ((1.0 - wt) * ((1.0 - wx) * c00 + wx * c01) + wt * ((1.0 - wx) * c10 + wx * c11)).astype(np.float32)


def _odd_cap(k: int, n: int) -> int:
    if n <= 1:
        return 1
    k = min(k, n if n % 2 else n - 1)
    if k % 2 == 0:
        k -= 1
    return max(k, 1)


def _cfar_2d(s, valid, wt: int = CFAR_2D_T, wx: int = CFAR_2D_X, step: int = CFAR_2D_STEP) -> np.ndarray:
    s = np.asarray(s, np.float32)
    v = np.asarray(valid, bool)
    t, w = s.shape
    filled = _fill_invalid(s, v)
    st = step if t >= step * (wt + 2) else 1
    sx = step if w >= step * (wx + 2) else 1
    sub = filled[::st, ::sx]
    kt = _odd_cap(max(1, int(round(wt / st))), sub.shape[0])
    kx = _odd_cap(max(1, int(round(wx / sx))), sub.shape[1])
    med = _upsample(median_filter(sub, size=(kt, kx), mode="nearest"), t, w)
    dev = np.abs(filled - med)
    mad = _upsample(median_filter(dev[::st, ::sx], size=(kt, kx), mode="nearest"), t, w)
    return _zscore(s, v, med, mad)
