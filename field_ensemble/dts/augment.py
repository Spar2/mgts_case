"""Аугментации температуры до признаков.

Цветной шум — аддитивный AR(1) в градусах на остатке T(t, x).
Каналы CNN считаются после этого вызовом ``features.make_channels``:
синфазная мода и медиана по x видят шум, канал valid не меняется.
Нулевая амплитуда оставляет уже посчитанный тайл как есть.

Инверсия знака отражает остаток вокруг фона (на остатке это T → −T)
и меняет местами классы нагрев (1) и охлаждение (2).
"""
from __future__ import annotations

import re

import numpy as np

from dts import features

# Медиана σ real и лаг-1 АКФ фона из docs/EXPERIMENTS_LATE.md (без меток).
SIGMA_REAL_C = 0.283
PHI_T_REAL = 0.357
PHI_X_REAL = 0.736

_REAL_SIG = re.compile(
    r"\|\s*real,\s*все файлы\s*\|\s*[0-9.]+\s*\|\s*([0-9]*\.?[0-9]+)\s*\|"
)
_PHI = re.compile(
    r"φ_t\s*=\s*([+-]?[0-9]*\.?[0-9]+).*?φ_x\s*=\s*([+-]?[0-9]*\.?[0-9]+)",
    re.S,
)


class Spec:
    def __init__(self, sigma_c: float, phi_t: float, phi_x: float, jitter: float, sign_p: float):
        self.sigma_c = float(sigma_c)
        self.phi_t = float(phi_t)
        self.phi_x = float(phi_x)
        self.jitter = float(jitter)
        self.sign_p = float(sign_p)

    def as_dict(self) -> dict:
        return dict(sigma_c=self.sigma_c, phi_t=self.phi_t, phi_x=self.phi_x,
                    jitter=self.jitter, sign_p=self.sign_p)


def lit_temperature_recipe(text: str) -> dict | None:
    """σ real (строка «real, все файлы») и первый φ_t, φ_x рецепта noise-matched.

    Оценка спектра в документе — АКФ фона, без карточек и без меток.
    """
    i = text.find("Рецепт noise-matched")
    if i < 0:
        return None
    pm = _PHI.search(text[i:i + 1800])
    sm = _REAL_SIG.search(text)
    if not pm or not sm:
        return None
    return {"sigma_c": float(sm.group(1)), "phi_t": float(pm.group(1)), "phi_x": float(pm.group(2))}


def swap_hot_cold(y: np.ndarray) -> np.ndarray:
    y = np.array(y, copy=True)
    hot = y == 1
    cold = y == 2
    y[hot] = 2
    y[cold] = 1
    return y


def invert_around_background(T: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """T → фон − остаток. Фон = синфазная мода(t) + статический профиль(x)."""
    import warnings
    Tw = np.where(valid, np.asarray(T, np.float32), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cm = np.nanmedian(Tw, axis=1)
        cm = np.where(np.isfinite(cm), cm, 0.0)
        D = Tw - cm[:, None]
        prof = np.nanmedian(D, axis=0)
        prof = np.where(np.isfinite(prof), prof, 0.0)
    resid = D - prof
    inv = cm[:, None] + prof[None, :] - resid
    out = np.array(T, np.float32, copy=True)
    m = np.asarray(valid, bool)
    out[m] = inv[m]
    return out


def _ar1_along(white: np.ndarray, phi: float, axis: int) -> np.ndarray:
    phi = float(np.clip(phi, -0.99, 0.99))
    if abs(phi) < 1e-6:
        return white
    white = np.moveaxis(white, axis, -1)
    scale = np.sqrt(max(1.0 - phi * phi, 0.0))
    out = np.empty_like(white)
    out[..., 0] = white[..., 0]
    for i in range(1, white.shape[-1]):
        out[..., i] = phi * out[..., i - 1] + scale * white[..., i]
    return np.moveaxis(out, -1, axis)


def colored_noise(rng: np.random.Generator, shape, phi_t: float, phi_x: float,
                  std: float, valid: np.ndarray | None = None) -> np.ndarray:
    """AR(1) по времени, затем по x. std — СКО в тех же единицах, что T (°C)."""
    z = rng.standard_normal(shape).astype(np.float32)
    z = _ar1_along(z, phi_t, 0)
    z = _ar1_along(z, phi_x, 1)
    if valid is None:
        s = float(z.std())
    else:
        sel = z[np.asarray(valid, bool)]
        s = float(sel.std()) if sel.size else 0.0
    z *= np.float32(std / max(s, 1e-6))
    if valid is not None:
        z = np.where(valid, z, 0.0).astype(np.float32)
    return z


def draw_amplitude(rng: np.random.Generator, sigma_c: float, jitter: float) -> float:
    """Случайная амплитуда вокруг sigma_c. jitter=0 → ровно sigma_c."""
    if sigma_c <= 0:
        return 0.0
    jitter = min(abs(float(jitter)), 0.9)
    if jitter == 0:
        return float(sigma_c)
    return float(sigma_c) * float(rng.uniform(1.0 - jitter, 1.0 + jitter))


def add_colored_noise(T: np.ndarray, valid: np.ndarray, rng: np.random.Generator,
                      phi_t: float, phi_x: float, std: float) -> np.ndarray:
    """Аддитивный цветной шум только на валидных клетках. std<=0 — копия T."""
    out = np.array(T, np.float32, copy=True)
    if std is None or std <= 0:
        return out
    noise = colored_noise(rng, out.shape, phi_t, phi_x, std, valid)
    m = np.asarray(valid, bool)
    out[m] = out[m] + noise[m]
    return out


def featurize_noisy(T: np.ndarray, valid: np.ndarray, rng: np.random.Generator,
                    phi_t: float = PHI_T_REAL, phi_x: float = PHI_X_REAL,
                    sigma_c: float = SIGMA_REAL_C, jitter: float = 0.25,
                    dx: float = 0.25):
    """Шум в °C, затем make_channels. Возвращает (каналы, зашумлённое T, amp)."""
    amp = draw_amplitude(rng, sigma_c, jitter)
    Tn = add_colored_noise(T, valid, rng, phi_t, phi_x, amp)
    ch, _aux = features.make_channels(Tn, valid, dx)
    return ch, Tn, amp


def feature_delta(T: np.ndarray, noise: np.ndarray, valid: np.ndarray, dx: float = 0.25) -> np.ndarray:
    """Разница каналов make_channels(T+noise) − make_channels(T). Нулевой шум → 0."""
    m = np.asarray(valid, bool)
    base = np.where(m, T, 0.0).astype(np.float32)
    noisy = np.where(m, np.asarray(T, np.float32) + np.asarray(noise, np.float32), 0.0)
    c0, _ = features.make_channels(base, m, dx)
    c1, _ = features.make_channels(noisy, m, dx)
    return c1 - c0


def apply_tile(xt: np.ndarray, yt: np.ndarray, sigma, rng: np.random.Generator, spec: Spec,
               dx: float = 0.25):
    """Тайл каналов (3, t, x) и метки.

    Остаток °C = канал 0 × σ(x) последовательности. Знак и шум применяются
    к этому остатку до make_channels; в тайл добавляется только приращение
    каналов, поэтому при выключенном шуме тайл не меняется.
    """
    xt = np.array(xt, np.float32, copy=True)
    yt = np.array(yt, copy=True)
    valid = xt[2] > 0.5
    sig = np.maximum(np.asarray(sigma, np.float32), 0.03)
    if sig.shape[-1] != xt.shape[-1]:
        raise ValueError(f"sigma {sig.shape[-1]} != tile {xt.shape[-1]}")
    T = xt[0] * sig[None, :]
    if spec.sign_p > 0 and float(rng.random()) < spec.sign_p:
        T = -T
        xt[0] *= -1
        xt[1] *= -1
        yt = swap_hot_cold(yt)
    if spec.sigma_c > 0:
        amp = draw_amplitude(rng, spec.sigma_c, spec.jitter)
        noise = colored_noise(rng, T.shape, spec.phi_t, spec.phi_x, amp, valid)
        delta = feature_delta(T, noise, valid, dx)
        xt[0] = np.clip(xt[0] + delta[0], -features.CLIP, features.CLIP)
        xt[1] = np.clip(xt[1] + delta[1], -features.CLIP, features.CLIP)
        xt[:2] *= valid.astype(np.float32)
    return xt, yt
