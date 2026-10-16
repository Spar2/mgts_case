"""Синтетический шумовой приор для дообучения U-Net. Реальных спектров нет.

Коробка — вокруг фона синтетики v8, как в docs/S3_LIT_DG.md (конвейер P), не вокруг кабеля:
phi_t ~ U(0, 0.34), phi_x ~ U(0, 0.69), sigma = 0.297 * U(0.5, 2) °C.
Шум — AR(1) по времени и по x. Затем FreeSDG-шаг только на шуме: фаза БПФ
остаётся, амплитуды части частот берутся у второй реализации, постоянная
составляющая не масштабируется, СКО возвращается к уже вытянутому sigma.
Событие в чистом T не сдвигается. Сводный спектр кабеля и константы реального фона не читаются.
"""
from __future__ import annotations

import numpy as np

from dts.augment import colored_noise, feature_delta
from dts.features import CLIP

# Верх коробки = 2 * (0.170, 0.345) из фона синтетики, не из МГТС.
PHI_T_HI = 0.34
PHI_X_HI = 0.69
SIGMA_C = 0.297
SIGMA_LO = 0.5
SIGMA_HI = 2.0
AMP_LO = 0.5
AMP_HI = 2.0
P_APPLY = 0.7


def draw_prior(rng: np.random.Generator) -> dict:
    """Один тайл: равномерный DR, без подгонки к кабелю."""
    return {
        "phi_t": float(rng.uniform(0.0, PHI_T_HI)),
        "phi_x": float(rng.uniform(0.0, PHI_X_HI)),
        "sigma": float(SIGMA_C * rng.uniform(SIGMA_LO, SIGMA_HI)),
    }


def _scale_spectrum(noise_a: np.ndarray, noise_b: np.ndarray, rng: np.random.Generator):
    """Фаза первого шума. 2–4 прямоугольника амплитуды — от второго.

    Один общий множитель U(0.5, 2) на все бины, кроме постоянной составляющей.
    После обратного БПФ вызывающий возвращает СКО: общий множитель сам по себе
    почти снимается этой нормировкой, форму спектра меняют прямоугольники.
    """
    a = np.fft.rfft2(np.asarray(noise_a, np.float32))
    b = np.fft.rfft2(np.asarray(noise_b, np.float32))
    mag = np.abs(a)
    phase = np.ones_like(a)
    nz = mag > 1e-12
    phase[nz] = a[nz] / mag[nz]
    th, fw = mag.shape
    n_rect = int(rng.integers(2, 5))
    rects = []
    for _ in range(n_rect):
        h = int(rng.integers(1, max(th // 4, 2)))
        w = int(rng.integers(1, max(fw // 4, 2)))
        h = min(h, th)
        w = min(w, fw)
        t0 = int(rng.integers(0, th - h + 1))
        f0 = int(rng.integers(0, fw - w + 1))
        mag[t0:t0 + h, f0:f0 + w] = np.abs(b[t0:t0 + h, f0:f0 + w])
        rects.append((t0, f0, h, w))
    factor = float(rng.uniform(AMP_LO, AMP_HI))
    mag *= np.float32(factor)
    mag[0, 0] = np.abs(a[0, 0])
    field = np.fft.irfft2(mag * phase, s=noise_a.shape).astype(np.float32)
    return field, {"n_rect": n_rect, "amp_factor": factor, "rects": rects}


def _restore_std(field: np.ndarray, valid: np.ndarray, sigma: float) -> np.ndarray:
    m = np.asarray(valid, bool)
    sel = field[m]
    if sel.size == 0 or sigma <= 0:
        return np.zeros_like(field)
    scale = float(sel.std())
    if scale < 1e-8:
        return np.zeros_like(field)
    out = (field * np.float32(sigma / scale)).astype(np.float32)
    return np.where(m, out, np.float32(0))


def synthesize_noise(shape, valid, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """AR(1) в °C плюс частотная подмена амплитуды. Фаза первой реализации сохраняется."""
    first = draw_prior(rng)
    second = draw_prior(rng)
    m = np.asarray(valid, bool)
    a = colored_noise(rng, shape, first["phi_t"], first["phi_x"], first["sigma"], m)
    b = colored_noise(rng, shape, second["phi_t"], second["phi_x"], second["sigma"], m)
    mixed, spec = _scale_spectrum(a, b, rng)
    noise = _restore_std(mixed, m, first["sigma"])
    info = dict(first=first, second=second, **{k: spec[k] for k in ("n_rect", "amp_factor")})
    return noise, info


def apply_tile(xt: np.ndarray, yt: np.ndarray, sigma, rng: np.random.Generator, p_apply: float = P_APPLY):
    """Тайл (3, T, W). Шум в °C добавляется к остатку до make_channels.

    sigma — шкала канала 0 этой записи (°C), не априор. Априор шумит сам.
    С вероятностью 1 - p_apply тайл не меняется. Метки не трогаются.
    """
    xt = np.array(xt, np.float32, copy=True)
    yt = np.array(yt, copy=True)
    if p_apply <= 0 or float(rng.random()) > p_apply:
        return xt, yt
    valid = xt[2] > 0.5
    sig = np.maximum(np.asarray(sigma, np.float32), np.float32(0.03))
    if sig.shape[-1] != xt.shape[-1]:
        raise ValueError(f"sigma {sig.shape[-1]} != tile {xt.shape[-1]}")
    temp = xt[0] * sig[None, :]
    noise, _info = synthesize_noise(temp.shape, valid, rng)
    delta = feature_delta(temp, noise, valid)
    xt[0] = np.clip(xt[0] + delta[0], -CLIP, CLIP)
    xt[1] = np.clip(xt[1] + delta[1], -CLIP, CLIP)
    xt[:2] *= valid.astype(np.float32)
    return xt, yt


def self_check() -> None:
    rng = np.random.default_rng(0)
    for _ in range(64):
        p = draw_prior(rng)
        if not (0.0 <= p["phi_t"] <= PHI_T_HI and 0.0 <= p["phi_x"] <= PHI_X_HI):
            raise SystemExit("prior phi outside the synthetic box")
        lo, hi = SIGMA_C * SIGMA_LO, SIGMA_C * SIGMA_HI
        if not (lo - 1e-9 <= p["sigma"] <= hi + 1e-9):
            raise SystemExit("prior sigma outside the synthetic box")
    src = np.random.default_rng(1)
    noise = src.standard_normal((32, 40)).astype(np.float32)
    other = src.standard_normal((32, 40)).astype(np.float32)
    # Без прямоугольников множитель сохраняет фазу и не трогает DC.
    spec = np.fft.rfft2(noise)
    factor = 1.7
    mag = np.abs(spec) * np.float32(factor)
    mag[0, 0] = np.abs(spec[0, 0])
    phase = np.ones_like(spec)
    nz = np.abs(spec) > 1e-12
    phase[nz] = spec[nz] / np.abs(spec[nz])
    back = np.fft.rfft2(np.fft.irfft2(mag * phase, s=noise.shape))
    ratio = back / spec
    if abs(complex(ratio[0, 0]) - 1.0) > 1e-3:
        raise SystemExit("DC bin was scaled")
    if abs(abs(complex(ratio[3, 2])) - factor) > 1e-3:
        raise SystemExit("amplitude factor missed a bin")
    if abs(np.angle(ratio[3, 2])) > 1e-3:
        raise SystemExit("phase was not kept")
    valid = np.ones((48, 64), bool)
    valid[:, -3:] = False
    field, info = synthesize_noise((48, 64), valid, np.random.default_rng(2))
    got = float(field[valid].std())
    if abs(got - info["first"]["sigma"]) > 1e-3:
        raise SystemExit(f"std restore {got} != {info['first']['sigma']}")
    if int(field[:, -3:].sum()) != 0:
        raise SystemExit("noise leaked into invalid columns")
    xt = np.zeros((3, 24, 36), np.float32)
    xt[2] = 1
    yt = np.zeros((24, 36), np.int64)
    out, yt2 = apply_tile(xt, yt, np.full(36, 0.25, np.float32), np.random.default_rng(3), p_apply=1.0)
    if out.shape != xt.shape or not np.array_equal(yt2, yt):
        raise SystemExit("apply_tile changed shape or labels")
    if not np.isfinite(out).all():
        raise SystemExit("apply_tile produced non-finite channels")
    print("aug_dg self-check ok", flush=True)
