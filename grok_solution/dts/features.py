"""Физически мотивированная нормировка поля T(t, x) в «изображение» для CNN.

Каналы (все в единицах робастного шума σ(x), обрезаны в [-CLIP, CLIP]):
  0  dT_time  = T - синфазная составляющая(t) - статический профиль(x)   — «что изменилось во времени»
  1  dT_space = T - скользящая медиана по x (окно 40 м) - синфазная      — «чем точка отличается от соседей»
  2  valid    — маска валидных измерений (−300 / разрывы / за концом волокна = 0)
Синфазная составляющая (общий дрейф прибора/помещения) убирается построчной медианой —
так помехи типа common-mode не превращаются в «аномалию».
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

CLIP = 12.0
SPACE_WIN_M = 40.0


def _nanmedian_rows(a):
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.nanmedian(a, axis=1)


def robust_sigma_x(d: np.ndarray, valid: np.ndarray, smooth_px: int = 41) -> np.ndarray:
    """σ(x) по разностям во времени: 1.4826·MAD(ΔtT)/√2, сглажено по x."""
    import warnings
    diff = np.diff(np.where(valid, d, np.nan), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        med = np.nanmedian(diff, axis=0)
        s = 1.4826 * np.nanmedian(np.abs(diff - med), axis=0) / np.sqrt(2)
    good = np.isfinite(s)
    if not good.any():
        return np.ones(d.shape[1], np.float32)
    s = np.where(good, s, np.nanmedian(s))
    s = median_filter(s, size=smooth_px, mode="nearest")
    return np.maximum(s, 0.03).astype(np.float32)


def make_channels(T: np.ndarray, valid: np.ndarray, dx: float = 0.25) -> tuple[np.ndarray, dict]:
    import warnings
    T = np.where(valid, T, np.nan).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cm = np.nanmedian(T, axis=1)
        cm = np.where(np.isfinite(cm), cm, 0.0)
        D = T - cm[:, None]
        prof = np.nanmedian(D, axis=0)
        prof = np.where(np.isfinite(prof), prof, 0.0)
    dT_time = D - prof[None, :]
    sig = robust_sigma_x(D, valid)
    # пространственный фон: медиана по x (NaN заполняем профилем)
    Df = np.where(np.isfinite(D), D, prof[None, :])
    win = max(3, int(round(SPACE_WIN_M / dx)) | 1)
    # медиана по x дорогая — считаем на прореженной сетке и интерполируем
    step = 4
    sub = median_filter(Df[:, ::step], size=(1, max(3, (win // step) | 1)), mode="nearest")
    xs = np.arange(Df.shape[1]); bg = np.empty_like(Df)
    for i in range(Df.shape[0]):
        bg[i] = np.interp(xs, xs[::step], sub[i])
    dT_space = D - bg
    ch = np.stack([np.clip(np.nan_to_num(dT_time / sig[None, :]), -CLIP, CLIP),
                   np.clip(np.nan_to_num(dT_space / sig[None, :]), -CLIP, CLIP),
                   valid.astype(np.float32)]).astype(np.float32)
    ch[:2] *= valid[None]
    return ch, dict(sigma_x=sig, common_mode=cm.astype(np.float32), profile=prof.astype(np.float32),
                    dT_time=np.nan_to_num(dT_time).astype(np.float32))
