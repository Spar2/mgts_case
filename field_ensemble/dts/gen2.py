"""Аугментация GEN2 поверх уже посчитанных каналов кэша.

Явления — по физике описания, не по координатам и амплитудам карточек.
Каналы кэша (dts/features.py), единицы σ, обрезка ±12:

  0  dT_time  — из статического профиля вычтено, остаётся изменение во времени
  1  dT_space — контраст к скользящей медиане 40 м
  2  valid

Узкая статичная горячая точка (dts/static_synth.py, 2–8 м, +3…+24 °C)
почти целиком сидит в канале 1: профиль по времени её съедает.
Широкий медленный сдвиг и цикл HVAC 20–90 мин — климат, не аномалия:
разметка не меняется. Резкий провал до −20 °C на 1–3 отсчёта — класс
artifact: в цели это фон (y = 0), чтобы модель гасила ложную тревогу.
Новых больших файлов функция не пишет.
"""
from __future__ import annotations

import numpy as np

from dts.static_synth import DX, hot_spec, time_multiplier, unit_profile

CLIP = 12.0
SIGMA_FLOOR = 0.03
SPACE_WIN_M = 40.0
# Климатические диапазоны — порядок явления, не оценки с карточек.
BROAD_WIDTH_M = (30.0, 120.0)
BROAD_AMP_C = (0.8, 5.0)
BROAD_TAU_MIN = (40.0, 180.0)
HVAC_WIDTH_M = (8.0, 35.0)
HVAC_AMP_C = (0.8, 3.5)
HVAC_PERIOD_MIN = (20.0, 90.0)
ARTIFACT_C = -20.0
ARTIFACT_LEN = (1, 3)  # включительно
PROBS = dict(static=0.50, broad=0.35, hvac=0.35, artifact=0.50)


def _sigma(sigma, n: int) -> np.ndarray:
    s = np.asarray(sigma, np.float32).reshape(-1)
    if s.size == 1:
        s = np.full(n, float(s[0]), np.float32)
    if s.size != n:
        raise ValueError(f"sigma {s.size} != ширина {n}")
    return np.maximum(s, SIGMA_FLOOR)


def _free_cols(y: np.ndarray) -> np.ndarray:
    """Столбцы без события и хотя бы с одной валидной клеткой."""
    valid = (y >= 0).any(axis=0)
    busy = (y > 0).any(axis=0)
    return valid & ~busy


def _spatial_residual(profile: np.ndarray, dx: float = DX, win_m: float = SPACE_WIN_M) -> np.ndarray:
    """Профиль минус скользящее среднее 40 м: то, что остаётся в dT_space."""
    win = max(3, int(round(win_m / dx)) | 1)
    ker = np.ones(win, np.float64) / win
    bg = np.convolve(np.asarray(profile, np.float64), ker, mode="same")
    return (np.asarray(profile, np.float64) - bg).astype(np.float32)


def _apply(x, y, add0, add1, cols: np.ndarray) -> None:
    if not np.any(cols):
        return
    gate = ((y >= 0) & cols.reshape(1, -1)).astype(np.float32)
    x[0] += add0 * gate
    x[1] += add1 * gate


def _clip(x, y) -> None:
    valid = (y >= 0).astype(np.float32)
    x[0] = np.clip(x[0], -CLIP, CLIP) * valid
    x[1] = np.clip(x[1], -CLIP, CLIP) * valid


def inject_static(x, y, sigma, rng) -> dict:
    """Одна узкая статичная горячая точка. Уже включена на первом отсчёте тайла."""
    _, t_len, n_px = x.shape
    empty = np.zeros((t_len, n_px), bool)
    free = _free_cols(y)
    spec, _avoid = hot_spec(rng, n_px, DX, ~free)
    if spec is None:
        return dict(mask=empty, amp_c=None, width_m=None)
    prof = unit_profile(n_px, spec, dx=DX)
    cols = (prof >= 0.5) & free
    if not cols.any():
        return dict(mask=empty, amp_c=float(spec["amp"]), width_m=float(spec["width_m"]))
    mult = time_multiplier(t_len, spec).astype(np.float64)
    scale = (float(spec["amp"]) / _sigma(sigma, n_px)).astype(np.float64)
    field = mult[:, None] * prof[None, :] * scale[None, :]
    dev = (mult - float(mult.mean()))[:, None] * prof[None, :] * scale[None, :]
    _apply(x, y, dev.astype(np.float32), field.astype(np.float32), cols)
    mask = np.zeros((t_len, n_px), bool)
    mask[:, cols] = y[:, cols] == 0
    y[mask] = 1
    return dict(mask=mask, amp_c=float(spec["amp"]), width_m=float(spec["width_m"]))


def inject_broad(x, y, sigma, t, rng) -> dict:
    """Широкий сдвиг профиля. Медленный по времени, в разметку не входит."""
    _, t_len, n_px = x.shape
    free = _free_cols(y)
    width = float(rng.uniform(*BROAD_WIDTH_M))
    amp = float(rng.uniform(*BROAD_AMP_C))
    center = int(rng.integers(0, n_px))
    spec = dict(center=center, width_m=width, kind="broad", spot_m=1.5, dx=DX)
    prof = unit_profile(n_px, spec, dx=DX)
    resid = _spatial_residual(prof, DX)
    tt = np.asarray(t, np.float64).reshape(-1)
    if tt.size != t_len:
        tt = np.arange(t_len, dtype=np.float64) * 120.0
    tau = float(rng.uniform(*BROAD_TAU_MIN)) * 60.0
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    env = np.sin(2.0 * np.pi * (tt - tt[0]) / tau + phase)
    scale = amp / _sigma(sigma, n_px)
    zm = env - float(env.mean())
    add0 = (zm[:, None] * prof[None, :] * scale[None, :]).astype(np.float32)
    add1 = (env[:, None] * resid[None, :] * scale[None, :]).astype(np.float32)
    _apply(x, y, add0, add1, free)
    return dict(width_m=width, amp_c=amp, tau_min=tau / 60.0)


def inject_hvac(x, y, sigma, t, rng) -> dict:
    """Локальный цикл климатики. Период 20–90 мин. Не аномалия."""
    _, t_len, n_px = x.shape
    free = _free_cols(y)
    width = float(rng.uniform(*HVAC_WIDTH_M))
    amp = float(rng.uniform(*HVAC_AMP_C))
    period = float(rng.uniform(*HVAC_PERIOD_MIN))
    center = int(rng.integers(0, max(n_px, 1)))
    kind = "segment" if rng.random() < 0.5 else "gaussian"
    spec = dict(center=center, width_m=width, kind=kind, spot_m=1.5, edge_m=1.0, dx=DX)
    prof = unit_profile(n_px, spec, dx=DX)
    resid = _spatial_residual(prof, DX)
    tt = np.asarray(t, np.float64).reshape(-1)
    if tt.size != t_len:
        tt = np.arange(t_len, dtype=np.float64) * 120.0
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    wave = np.sin(2.0 * np.pi * (tt - tt[0]) / (period * 60.0) + phase)
    scale = amp / _sigma(sigma, n_px)
    add0 = (wave[:, None] * prof[None, :] * scale[None, :]).astype(np.float32)
    add1 = (wave[:, None] * resid[None, :] * scale[None, :]).astype(np.float32)
    _apply(x, y, add0, add1, free)
    return dict(period_min=period, width_m=width, amp_c=amp)


def inject_artifact(x, y, sigma, rng) -> dict:
    """Провал до −20 °C на 1–3 отсчёта. Класс artifact: цель — фон, не событие."""
    _, t_len, n_px = x.shape
    empty = np.zeros((t_len, n_px), bool)
    free = _free_cols(y)
    if t_len < 1 or not free.any():
        return dict(mask=empty, n_samples=0, mode=None)
    n_samp = int(rng.integers(ARTIFACT_LEN[0], ARTIFACT_LEN[1] + 1))
    t0 = int(rng.integers(0, t_len))
    rows = np.arange(t0, min(t_len, t0 + n_samp))
    mode = "section" if rng.random() < 0.5 else "connector"
    if mode == "connector":
        n_col = int(rng.integers(1, 4))
        c0 = int(rng.choice(np.flatnonzero(free)))
        cols = np.zeros(n_px, bool)
        cols[c0:c0 + n_col] = True
        cols &= free
    else:
        width = float(rng.uniform(3.0, 20.0))
        center = int(rng.choice(np.flatnonzero(free)))
        half = max(1, int(round(width / DX / 2.0)))
        cols = np.zeros(n_px, bool)
        cols[max(0, center - half):min(n_px, center + half + 1)] = True
        cols &= free
    if not cols.any() or rows.size == 0:
        return dict(mask=empty, n_samples=0, mode=mode)
    scale = (ARTIFACT_C / _sigma(sigma, n_px)).astype(np.float32)
    add = np.zeros((t_len, n_px), np.float32)
    add[np.ix_(rows, np.flatnonzero(cols))] = scale[cols]
    # Резкий провал не является ни статическим профилем, ни медианой 40 м.
    _apply(x, y, add, add, cols)
    mask = np.zeros((t_len, n_px), bool)
    mask[np.ix_(rows, np.flatnonzero(cols))] = y[np.ix_(rows, np.flatnonzero(cols))] == 0
    y[mask] = 0
    return dict(mask=mask, n_samples=int(rows.size), mode=mode, amp_c=ARTIFACT_C)


def augment_tile(x, y, sigma, t, rng, force: bool = False, probs: dict | None = None):
    """Копия тайла с явлениями. force — все четыре, для собственного gen2_val."""
    x = np.array(x, np.float32, copy=True)
    y = np.array(y, np.int64, copy=True)
    if x.ndim != 3 or x.shape[0] != 3:
        raise ValueError(f"ожидался X (3,T,W), получено {x.shape}")
    if y.shape != x.shape[1:]:
        raise ValueError(f"y {y.shape} != {x.shape[1:]}")
    p = dict(PROBS)
    if probs:
        p.update(probs)
    info = dict(
        static_mask=np.zeros(y.shape, bool),
        artifact_mask=np.zeros(y.shape, bool),
        static=None, broad=None, hvac=None, artifact=None,
    )
    want = {k: True for k in ("broad", "hvac", "static", "artifact")} if force else {
        k: rng.random() < float(p[k]) for k in ("broad", "hvac", "static", "artifact")
    }
    if want["broad"]:
        info["broad"] = inject_broad(x, y, sigma, t, rng)
    if want["hvac"]:
        info["hvac"] = inject_hvac(x, y, sigma, t, rng)
    if want["static"]:
        st = inject_static(x, y, sigma, rng)
        info["static"] = st
        info["static_mask"] = st["mask"]
    if want["artifact"]:
        art = inject_artifact(x, y, sigma, rng)
        info["artifact"] = art
        info["artifact_mask"] = art["mask"]
        # Провал на клетке события не ставим: inject и так берёт свободные столбцы.
        # Если маска всё же пересекла горячую точку — горячая точка важнее recall.
        clash = info["artifact_mask"] & info["static_mask"]
        if clash.any():
            info["artifact_mask"] = info["artifact_mask"] & ~clash
            y[clash] = 1
    _clip(x, y)
    return x, y, info


def pixel_weight(info, artifact_w: float = 1.0, static_w: float = 0.5) -> np.ndarray:
    """Вес для focal/CE: артефакт — отрицательный класс, статика — редкий положительный."""
    w = np.zeros(info["static_mask"].shape, np.float32)
    w[info["artifact_mask"]] = np.float32(artifact_w)
    hot = info["static_mask"] & (w == 0)
    w[hot] = np.float32(static_w)
    return w


def build_scene(seed: int, t_rows: int = 180, n_px: int = 512, sigma_c: float = 0.30) -> dict:
    """Свой кадр gen2_val в памяти. σ = 0.30 °C — порядок синтетики, не карточка."""
    rng = np.random.default_rng(int(seed) + 17)
    x = rng.normal(0.0, 0.85, size=(3, t_rows, n_px)).astype(np.float32)
    x[2] = 1.0
    y = np.zeros((t_rows, n_px), np.int64)
    sigma = np.full(n_px, sigma_c, np.float32)
    t = np.arange(t_rows, dtype=np.float64) * 120.0
    x, y, info = augment_tile(x, y, sigma, t, rng, force=True)
    return dict(X=x, y=y, sigma=sigma, t=t, info=info, seed=int(seed))


def self_check() -> None:
    rng = np.random.default_rng(0)
    t_len, n_px = 160, 400
    x = np.zeros((3, t_len, n_px), np.float32)
    x[2] = 1
    y = np.zeros((t_len, n_px), np.int64)
    y[:, 8:24] = 1
    held = x[:, :, 8:24].copy()
    sigma = np.full(n_px, 0.30, np.float32)
    t = np.arange(t_len, dtype=np.float64) * 120.0
    out, yy, info = augment_tile(x, y, sigma, t, rng, force=True)
    if not np.allclose(out[:, :, 8:24], held):
        raise SystemExit("аугментация изменила столбцы существующего события")
    if not np.array_equal(yy[:, 8:24], y[:, 8:24]):
        raise SystemExit("разметка события изменилась")
    sm = info["static_mask"]
    am = info["artifact_mask"]
    if not sm.any() or not am.any():
        raise SystemExit("не встали статика или артефакт")
    if yy[sm].min() != 1 or yy[am].max() != 0:
        raise SystemExit("классы static/artifact не совпали с целью")
    if float(out[1][sm].mean()) < 2.0:
        raise SystemExit("статика не видна в dT_space")
    if float(np.abs(out[0][sm]).mean()) > float(out[1][sm].mean()):
        raise SystemExit("статический горб протекает в dT_time сильнее, чем в dT_space")
    if float(out[0][am].max()) > -8.0:
        raise SystemExit("провал не дошёл до instrument clip")
    period = info["hvac"]["period_min"]
    if not (HVAC_PERIOD_MIN[0] <= period <= HVAC_PERIOD_MIN[1]):
        raise SystemExit(f"период HVAC {period}")
    if info["broad"]["width_m"] < BROAD_WIDTH_M[0]:
        raise SystemExit("широкий сдвиг узкий")
    if int(info["artifact"]["n_samples"]) < 1 or int(info["artifact"]["n_samples"]) > 3:
        raise SystemExit("длина провала не 1–3")
    scene = build_scene(3)
    if scene["X"].shape[0] != 3 or scene["info"]["static_mask"].sum() == 0:
        raise SystemExit("build_scene пустой")
