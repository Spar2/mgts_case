"""AR(1) шум на уже нормированных каналах 0 и 1 (рецепт lit для ae).

Канал valid не меняется. std<=0 — тождественное преобразование.
φ по времени и по кабелю задают стационарный AR(1); поле потом
масштабируется к заданному std на каждом тайле.
"""
from __future__ import annotations
import re
import numpy as np

_STD = re.compile(r"std\s*=\s*([0-9]*\.?[0-9]+)")
_PHI = re.compile(
    r"φ_t\s*=\s*([+-]?[0-9]*\.?[0-9]+).*?φ_x\s*=\s*([+-]?[0-9]*\.?[0-9]+)",
    re.S,
)


def recipe_from_markdown(text: str) -> dict | None:
    """Числа из раздела lit. Берётся первый AR(1) и первый std после заголовка рецепта."""
    i = text.find("Рецепт noise-matched")
    if i < 0:
        return None
    chunk = text[i:i + 1800]
    sm = _STD.search(chunk)
    pm = _PHI.search(chunk)
    if not sm or not pm:
        return None
    return {"std": float(sm.group(1)), "phi_t": float(pm.group(1)), "phi_x": float(pm.group(2))}


def _ar1(white: np.ndarray, phi: float, axis: int) -> np.ndarray:
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


def _field(rng: np.random.Generator, shape, phi_t: float, phi_x: float, std: float) -> np.ndarray:
    z = rng.standard_normal(shape).astype(np.float32)
    z = _ar1(z, phi_t, axis=-2)
    z = _ar1(z, phi_x, axis=-1)
    flat = z.reshape(z.shape[0], -1)
    s = flat.std(axis=1, keepdims=True)
    s = np.maximum(s, 1e-6)
    z *= (std / s).astype(np.float32)[..., None]
    return z


def add_channel_ar1(x: np.ndarray, std: float, phi_t: float, phi_x: float, rng: np.random.Generator) -> np.ndarray:
    """x: (N, 3, T, W). Независимый AR(1) на каналах 0 и 1, затем clip [-12, 12]."""
    if std is None or std <= 0:
        return x
    out = np.array(x, dtype=np.float32, copy=True)
    n, _, t, w = out.shape
    for ch in (0, 1):
        out[:, ch] += _field(rng, (n, t, w), phi_t, phi_x, std)
    np.clip(out[:, :2], -12, 12, out=out[:, :2])
    return out
