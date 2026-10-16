"""Аугментация GEN3: явления GEN2 плюс холод, слабые события и склейка.

Физика, не координаты карточек. Каналы те же, что у dts/gen2.py:
dT_time, dT_space, valid, единицы σ, обрезка ±12.

Новое:
  холодная статичная точка — зеркало hot_spec, амплитуда отрицательная, класс cold;
  слабое событие — 2–6 м, 0.6–2.0 °C, короткое по времени, класс hot или cold;
  склейка — постоянный сдвиг ±1–4 °C на 1–2 столбца, класс artifact (y = 0).

Столбцы, где в тайле уже есть событие, не перекрашиваются.
"""
from __future__ import annotations

import numpy as np

from dts import gen2
from dts.static_synth import DX, hot_spec, time_multiplier, unit_profile

WEAK_WIDTH_M = (2.0, 6.0)
WEAK_AMP_C = (0.6, 2.0)
WEAK_LEN = (2, 12)  # отсчёты при dt = 120 с: 4–24 мин
SPLICE_AMP_C = (1.0, 4.0)
SPLICE_COLS = (1, 2)
# Веса карты wk. weak_weight=3 в лоссе даёт множитель 1 + 3*wk.
WEIGHTS = dict(artifact=1.0, splice=1.0, static=0.5, cold=0.5, weak=1.5)
PROBS = dict(gen2.PROBS)
PROBS.update(cold=0.40, weak=0.55, splice=0.45)
_KEYS = ("broad", "hvac", "static", "cold", "weak", "artifact", "splice")
_MASKS = ("static_mask", "cold_mask", "weak_mask", "artifact_mask", "splice_mask")


def _blank(shape) -> np.ndarray:
    return np.zeros(shape, bool)


def _place_on_free(rng, free: np.ndarray, width_m: float):
    half = max(1, int(round(float(width_m) / DX / 2.0)))
    idx = np.flatnonzero(free)
    if idx.size == 0:
        return None
    n_px = int(free.size)
    for _ in range(32):
        c = int(rng.choice(idx))
        a, b = max(0, c - half), min(n_px, c + half + 1)
        if b - a > 2 and bool(free[a:b].all()):
            return c
    return int(rng.choice(idx))


def inject_cold(x, y, sigma, rng) -> dict:
    """Узкая статичная холодная точка. Класс cold (y = 2)."""
    _, t_len, n_px = x.shape
    empty = _blank((t_len, n_px))
    free = gen2._free_cols(y)
    spec, _avoid = hot_spec(rng, n_px, DX, ~free)
    if spec is None:
        return dict(mask=empty, amp_c=None, width_m=None)
    spec = dict(spec)
    spec["amp"] = -abs(float(spec["amp"]))
    prof = unit_profile(n_px, spec, dx=DX)
    cols = (prof >= 0.5) & free
    if not cols.any():
        return dict(mask=empty, amp_c=float(spec["amp"]), width_m=float(spec["width_m"]))
    mult = time_multiplier(t_len, spec).astype(np.float64)
    scale = (float(spec["amp"]) / gen2._sigma(sigma, n_px)).astype(np.float64)
    field = mult[:, None] * prof[None, :] * scale[None, :]
    dev = (mult - float(mult.mean()))[:, None] * prof[None, :] * scale[None, :]
    gen2._apply(x, y, dev.astype(np.float32), field.astype(np.float32), cols)
    mask = _blank((t_len, n_px))
    mask[:, cols] = y[:, cols] == 0
    y[mask] = 2
    return dict(mask=mask, amp_c=float(spec["amp"]), width_m=float(spec["width_m"]))


def inject_weak(x, y, sigma, rng) -> dict:
    """Короткое слабое горячее или холодное событие. Это положительный класс."""
    _, t_len, n_px = x.shape
    empty = _blank((t_len, n_px))
    free = gen2._free_cols(y)
    if t_len < 2 or not free.any():
        return dict(mask=empty, amp_c=None, width_m=None, n_samples=0, cls=None)
    width = float(rng.uniform(*WEAK_WIDTH_M))
    mag = float(rng.uniform(*WEAK_AMP_C))
    sign = 1.0 if float(rng.random()) < 0.5 else -1.0
    amp = sign * mag
    cls = 1 if sign > 0.0 else 2
    center = _place_on_free(rng, free, width)
    if center is None:
        return dict(mask=empty, amp_c=amp, width_m=width, n_samples=0, cls=cls)
    spec = dict(center=int(center), width_m=width, kind="segment", spot_m=1.0, edge_m=0.4, dx=DX)
    prof = unit_profile(n_px, spec, dx=DX)
    cols = (prof >= 0.5) & free
    if not cols.any():
        return dict(mask=empty, amp_c=amp, width_m=width, n_samples=0, cls=cls)
    n_samp = int(rng.integers(WEAK_LEN[0], WEAK_LEN[1] + 1))
    t0 = int(rng.integers(0, t_len))
    rows = np.arange(t0, min(t_len, t0 + n_samp))
    scale = (amp / gen2._sigma(sigma, n_px)).astype(np.float64)
    pulse = np.zeros(t_len, np.float64)
    pulse[rows] = 1.0
    add1 = (pulse[:, None] * prof[None, :] * scale[None, :]).astype(np.float32)
    dev = pulse - float(pulse.mean())
    add0 = (dev[:, None] * prof[None, :] * scale[None, :]).astype(np.float32)
    gen2._apply(x, y, add0, add1, cols)
    mask = _blank((t_len, n_px))
    mask[np.ix_(rows, np.flatnonzero(cols))] = y[np.ix_(rows, np.flatnonzero(cols))] == 0
    y[mask] = cls
    return dict(mask=mask, amp_c=float(amp), width_m=width, n_samples=int(rows.size), cls=int(cls))


def inject_splice(x, y, sigma, rng) -> dict:
    """Постоянный пространственный скачок на 1–2 столбца. Не событие: y остаётся 0."""
    _, t_len, n_px = x.shape
    empty = _blank((t_len, n_px))
    free = gen2._free_cols(y)
    if t_len < 1 or not free.any():
        return dict(mask=empty, amp_c=None, n_cols=0)
    n_col = int(rng.integers(SPLICE_COLS[0], SPLICE_COLS[1] + 1))
    c0 = int(rng.choice(np.flatnonzero(free)))
    cols = np.zeros(n_px, bool)
    cols[c0:c0 + n_col] = True
    cols &= free
    if not cols.any():
        return dict(mask=empty, amp_c=None, n_cols=0)
    mag = float(rng.uniform(*SPLICE_AMP_C))
    amp = mag if float(rng.random()) < 0.5 else -mag
    scale = (amp / gen2._sigma(sigma, n_px)).astype(np.float32)
    add1 = np.zeros((t_len, n_px), np.float32)
    add1[:, cols] = scale[cols]
    add0 = np.zeros_like(add1)
    gen2._apply(x, y, add0, add1, cols)
    mask = _blank((t_len, n_px))
    mask[:, cols] = y[:, cols] == 0
    y[mask] = 0
    return dict(mask=mask, amp_c=float(amp), n_cols=int(cols.sum()))


def _info(shape) -> dict:
    return dict(
        static_mask=_blank(shape),
        cold_mask=_blank(shape),
        weak_mask=_blank(shape),
        artifact_mask=_blank(shape),
        splice_mask=_blank(shape),
        static=None, cold=None, weak=None, broad=None, hvac=None, artifact=None, splice=None,
    )


def _unclash(y, info) -> None:
    """Положительный класс важнее артефакта, если маски всё же пересеклись."""
    pos = info["static_mask"] | info["cold_mask"] | info["weak_mask"]
    for key in ("artifact_mask", "splice_mask"):
        clash = info[key] & pos
        if clash.any():
            info[key] = info[key] & ~clash
            restore = np.zeros(y.shape, np.int64)
            restore[info["static_mask"]] = 1
            restore[info["cold_mask"]] = 2
            restore[info["weak_mask"]] = y[info["weak_mask"]]
            y[clash] = restore[clash]


def augment_tile(x, y, sigma, t, rng, force: bool = False, probs: dict | None = None):
    """Копия тайла. force включает все семь явлений."""
    x = np.array(x, np.float32, copy=True)
    y = np.array(y, np.int64, copy=True)
    if x.ndim != 3 or x.shape[0] != 3:
        raise ValueError(f"ожидался X (3,T,W), получено {x.shape}")
    if y.shape != x.shape[1:]:
        raise ValueError(f"y {y.shape} != {x.shape[1:]}")
    p = dict(PROBS)
    if probs:
        p.update(probs)
    info = _info(y.shape)
    want = {k: True for k in _KEYS} if force else {k: float(rng.random()) < float(p[k]) for k in _KEYS}
    if want["broad"]:
        info["broad"] = gen2.inject_broad(x, y, sigma, t, rng)
    if want["hvac"]:
        info["hvac"] = gen2.inject_hvac(x, y, sigma, t, rng)
    if want["static"]:
        st = gen2.inject_static(x, y, sigma, rng)
        info["static"] = st
        info["static_mask"] = st["mask"]
    if want["cold"]:
        cold = inject_cold(x, y, sigma, rng)
        info["cold"] = cold
        info["cold_mask"] = cold["mask"]
    if want["weak"]:
        weak = inject_weak(x, y, sigma, rng)
        info["weak"] = weak
        info["weak_mask"] = weak["mask"]
    if want["artifact"]:
        art = gen2.inject_artifact(x, y, sigma, rng)
        info["artifact"] = art
        info["artifact_mask"] = art["mask"]
    if want["splice"]:
        sp = inject_splice(x, y, sigma, rng)
        info["splice"] = sp
        info["splice_mask"] = sp["mask"]
    _unclash(y, info)
    gen2._clip(x, y)
    return x, y, info


def pixel_weight(info, artifact_w: float = 1.0, static_w: float = 0.5,
                 cold_w: float = 0.5, weak_w: float = 1.5, splice_w: float = 1.0) -> np.ndarray:
    """wk для focal/CE. Слабое событие тяжелее статики, склейка — как артефакт."""
    w = np.zeros(info["static_mask"].shape, np.float32)
    w[info["artifact_mask"]] = np.float32(artifact_w)
    w[info["splice_mask"]] = np.maximum(w[info["splice_mask"]], np.float32(splice_w))
    w[info["static_mask"]] = np.float32(static_w)
    w[info["cold_mask"]] = np.float32(cold_w)
    w[info["weak_mask"]] = np.float32(weak_w)
    return w


def flip_info(info) -> None:
    for key in _MASKS:
        info[key] = info[key][:, ::-1].copy()


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
    for key in _MASKS:
        if info[key][:, 8:24].any():
            raise SystemExit(f"{key} закрашивает занятый столбец")
    if yy[info["static_mask"]].min() != 1:
        raise SystemExit("статика не класс hot")
    if yy[info["cold_mask"]].min() != 2 or yy[info["cold_mask"]].max() != 2:
        raise SystemExit("холод не класс cold")
    if float(info["cold"]["amp_c"]) >= 0.0:
        raise SystemExit("холодная амплитуда не отрицательная")
    if float(out[1][info["cold_mask"]].mean()) > -2.0:
        raise SystemExit("холод не виден в dT_space")
    weak = info["weak"]
    if weak["mask"].sum() == 0:
        raise SystemExit("слабое событие не встало")
    if abs(float(weak["amp_c"])) < WEAK_AMP_C[0] or abs(float(weak["amp_c"])) > WEAK_AMP_C[1]:
        raise SystemExit("амплитуда слабого события вне 0.6–2.0 °C")
    if not (WEAK_WIDTH_M[0] <= float(weak["width_m"]) <= WEAK_WIDTH_M[1]):
        raise SystemExit("ширина слабого события вне 2–6 м")
    if int(weak["cls"]) not in (1, 2) or not np.all(yy[info["weak_mask"]] == int(weak["cls"])):
        raise SystemExit("класс слабого события не совпал с целью")
    if float(np.abs(out[1][info["weak_mask"]]).mean()) < 0.5:
        raise SystemExit("слабое событие не видно в dT_space")
    sp = info["splice"]
    if sp["mask"].sum() == 0 or not (1 <= int(sp["n_cols"]) <= 2):
        raise SystemExit("склейка не на 1–2 столбцах")
    if yy[info["splice_mask"]].max() != 0:
        raise SystemExit("склейка размечена как событие")
    if abs(float(sp["amp_c"])) < SPLICE_AMP_C[0] or abs(float(sp["amp_c"])) > SPLICE_AMP_C[1]:
        raise SystemExit("амплитуда склейки вне ±1–4 °C")
    if float(np.abs(out[1][info["splice_mask"]]).mean()) < 2.0:
        raise SystemExit("склейка не видна в dT_space")
    alone = np.zeros_like(x)
    alone[2] = 1
    y_alone = np.zeros_like(y)
    sp_alone = inject_splice(alone, y_alone, sigma, np.random.default_rng(1))
    if sp_alone["mask"].sum() == 0:
        raise SystemExit("отдельная склейка пустая")
    if float(np.abs(alone[0][sp_alone["mask"]]).mean()) > 1e-5:
        raise SystemExit("постоянный сдвиг протекает в dT_time")
    if yy[info["artifact_mask"]].max() != 0:
        raise SystemExit("провал −20 °C размечен как событие")
    if info["hvac"]["period_min"] < gen2.HVAC_PERIOD_MIN[0]:
        raise SystemExit("период HVAC")
    wk = pixel_weight(info)
    if float(wk[info["weak_mask"]].min()) <= float(wk[info["static_mask"]].max()):
        raise SystemExit("вес слабого события не выше статики")
    if float(wk[info["splice_mask"]].min()) < 1.0:
        raise SystemExit("вес склейки не повышен")
