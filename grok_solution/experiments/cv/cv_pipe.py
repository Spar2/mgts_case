"""Классический CV поверх воспроизведения OpenCV и постобработка маски U-Net.

Скор события — тот же, что у cv_ref: статистический остаток в °C и постоянная
добавка на компонентах. Деление на σ и раздача максимума компоненты всем её
пикселям на контроле раздували порог FAR (десятки и сотни градусов) и резали IoU.

Два шага сверх референса:
- открытие 2×2 снимает одиночный выброс вне компоненты и приглушает его ореол;
- линия Хафа с физическим наклоном добавляется к маске, только если таких
  пикселей не больше 0.05 % валидных. Более широкая добавка съедает бюджет FAR.

Многомасштабное сглаживание, связные компоненты с фильтром размера и поток
Farneback в модуле есть. На трёх контрольных записях val широкая маска блобов
и добавка потока поднимали хвост фона и снижали IoU, поэтому в s они не входят.
postprocess_unet только снижает скор: зона без опоры в каналах X или мельче
физического размера приглушается. Порог FAR ставится снаружи.
"""
from __future__ import annotations

import numpy as np

from dts.cv_common import (
    _cv2,
    axes_steps,
    blur_masked,
    fill_invalid,
    morph,
    remove_common_drift,
    slow_temporal_background,
    temporal_median_window,
)
from experiments.cv.cv_ref import assemble_score, reference_parts

# Многомасштабные σ в пикселях: (время, расстояние). 1 строка = 120 с, 1 столбец = 0.25 м.
SCALES = ((1.0, 2.0), (2.0, 6.0), (4.0, 14.0), (8.0, 28.0))
SEED_K_MAD = 3.5
MIN_AREA_PX = 12
MIN_DURATION_MIN = 6.0
MIN_WIDTH_M = 0.75
MIN_MEAN_Z = 0.9  # °C: средний модуль остатка внутри оставленного контура
MIN_FILL = 0.35
ATTEN = 0.15
# Скорость дрейфа контакта, м/мин. Вертикальная полоса (стоит на месте) не усиливается.
DRIFT_SPEED_MIN = 0.02
DRIFT_SPEED_MAX = 2.0
FLOW_WEIGHT = 0.10
FLOW_Z_MIN = 0.8

# Постобработка U-Net: только отсечение. Опора — модуль каналов времени/пространства.
UNET_HI = 0.80
UNET_SUPPORT_Z = 1.25
UNET_WEAK_Z = 0.80
UNET_FLOOR = 0.05


def thermal_residual(T, valid, sigma) -> tuple[np.ndarray, np.ndarray]:
    """Остаток после робастного фона, в градусах и в σ. Невалидное = 0.

    Сглаживание сюда не входит: семя компонент строится по резкому |z|,
    иначе одиночный выброс расползается и проходит фильтр размера.
    """
    valid = np.asarray(valid, bool)
    sig = np.maximum(np.asarray(sigma, np.float32), 0.03)
    filled = fill_invalid(T, valid)
    field = remove_common_drift(filled)
    window = temporal_median_window(field.shape[0], 601)
    background = slow_temporal_background(field, window_steps=window, stride=4)
    resid = (field - background).astype(np.float32)
    z = resid / sig[None, :]
    z = np.where(valid, z, 0.0).astype(np.float32)
    resid = np.where(valid, resid, 0.0).astype(np.float32)
    return z, resid


def spatial_contrast(z: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Локальный контраст: z минус гауссова подложка ~40 м вдоль кабеля."""
    low = blur_masked(z, valid, 2.0, 64.0)
    return np.where(valid, z - low, 0.0).astype(np.float32)


def multiscale(z: np.ndarray, z_space: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Максимум модулей сглаженных контрастов и знаковая карта того же масштаба."""
    acc = np.zeros(z.shape, np.float32)
    signed = np.zeros(z.shape, np.float32)
    for sigma_t, sigma_x in SCALES:
        zt = blur_masked(z, valid, sigma_t, sigma_x)
        zx = blur_masked(z_space, valid, sigma_t, sigma_x)
        mag = np.maximum(np.abs(zt), np.abs(zx))
        take = mag >= acc
        acc = np.maximum(acc, mag)
        choice = np.where(np.abs(zt) >= np.abs(zx), zt, zx)
        signed = np.where(take, choice, signed)
    acc = np.where(valid, acc, 0.0).astype(np.float32)
    signed = np.where(valid, signed, 0.0).astype(np.float32)
    return acc, signed


def _seed_threshold(raw_abs: np.ndarray, valid: np.ndarray) -> float:
    sel = raw_abs[valid]
    if sel.size == 0:
        return 1e9
    med = float(np.median(sel))
    mad = float(np.median(np.abs(sel - med))) * 1.4826
    return med + SEED_K_MAD * max(mad, 1e-3)


def _adaptive_close(density: float) -> tuple[int, int]:
    """(ширина x, высота t). На плотном поле ядро меньше, чтобы не склеить соседей."""
    if density > 0.02:
        return (3, 3)
    if density > 0.005:
        return (5, 3)
    return (7, 5)


def _drift_mask(score: np.ndarray, valid: np.ndarray, dx: float, dt_s: float) -> np.ndarray:
    """Линии Хафа с ненулевым наклоном: дрейф по кабелю, а не стоячая полоса."""
    cv2 = _cv2()
    image = np.zeros(score.shape, np.uint8)
    sel = score[valid]
    if sel.size == 0:
        return np.zeros(score.shape, bool)
    hi = float(np.percentile(sel, 99.5))
    lo = float(np.percentile(sel, 50))
    if hi - lo < 1e-3:
        return np.zeros(score.shape, bool)
    u8 = np.clip((score - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    u8 = np.where(valid, u8, 0).astype(np.uint8)
    # На очень широком кабеле Хаф считается на сетке ~0.5 м, линии возвращаются назад.
    step = 2 if u8.shape[1] > 2500 else 1
    small = u8[:, ::step]
    edges = cv2.Canny(small, 40, 120)
    min_len = 12 if step == 1 else 8
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=30, minLineLength=min_len, maxLineGap=6)
    canvas = np.zeros_like(small)
    if lines is not None:
        arr = np.asarray(lines)
        if arr.ndim == 3:
            arr = arr[:, 0]
        order = sorted(arr, key=lambda ln: abs(int(ln[2]) - int(ln[0])) + abs(int(ln[3]) - int(ln[1])), reverse=True)
        for x1, y1, x2, y2 in order[:400]:
            dt_px = abs(int(y2) - int(y1))
            dx_px = abs(int(x2) - int(x1)) * step
            if dt_px < 6 or dx_px < 4:
                continue
            minutes = dt_px * dt_s / 60.0
            meters = dx_px * dx
            if minutes <= 0:
                continue
            speed = meters / minutes
            if speed < DRIFT_SPEED_MIN or speed > DRIFT_SPEED_MAX:
                continue
            cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), 255, thickness=2)
    if step == 1:
        return canvas > 0
    full = np.repeat(canvas, step, axis=1)
    return full[:, : score.shape[1]] > 0


def _flow_boost(z_abs: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Согласованный сдвиг Farneback между половинами записи. Вес небольшой.

    Поток считается на уменьшенной карте и растягивается обратно. Это не
    маска события, а добавка к скору там, где поле заметно уехало по x.
    """
    cv2 = _cv2()
    t_len, x_len = z_abs.shape
    if t_len < 32 or x_len < 32:
        return np.zeros(z_abs.shape, np.float32)
    scale_t = max(1, t_len // 160)
    scale_x = max(1, x_len // 320)
    small = cv2.resize(z_abs.astype(np.float32), (max(8, x_len // scale_x), max(8, t_len // scale_t)), interpolation=cv2.INTER_AREA)
    h = small.shape[0] // 2
    if h < 8:
        return np.zeros(z_abs.shape, np.float32)
    def u8(img):
        lo = float(np.percentile(img, 5))
        hi = float(np.percentile(img, 99))
        if hi - lo < 1e-3:
            return np.zeros(img.shape, np.uint8)
        return np.clip((img - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    flow = cv2.calcOpticalFlowFarneback(u8(small[:h]), u8(small[-h:]), None, 0.5, 2, 15, 2, 5, 1.1, 0)
    mag = np.abs(flow[..., 0]).astype(np.float32)
    thr = float(np.percentile(mag, 75))
    if thr <= 1e-3:
        return np.zeros(z_abs.shape, np.float32)
    gate = (mag >= thr).astype(np.float32)
    # Обе половины записи видят один и тот же грубый сдвиг.
    stacked = np.concatenate([gate, gate], axis=0)
    if stacked.shape[0] < small.shape[0]:
        stacked = np.pad(stacked, ((0, small.shape[0] - stacked.shape[0]), (0, 0)))
    stacked = stacked[: small.shape[0]]
    full = cv2.resize(stacked, (x_len, t_len), interpolation=cv2.INTER_NEAREST)
    full = np.where(valid, full, 0.0)
    return full.astype(np.float32)


def _blob_keep(seed: np.ndarray, z_abs: np.ndarray, drift: np.ndarray, x, t, valid: np.ndarray) -> np.ndarray:
    """Связные компоненты семян, проверенные контуром (solidity) и физическим размером.

    Пиксельная маска берётся из connectedComponents: заливка контура теряет
    линию толщиной в один пиксель. findContours даёт solidity = площадь
    контура / площадь рамки. У дрейфующего следа solidity маленькая, поэтому
    след, лежащий на маске Хафа, этим порогом не снимается.
    """
    cv2 = _cv2()
    if not np.any(seed):
        return np.zeros(seed.shape, bool)
    dx, dt_s = axes_steps(x, t)
    cable_m = float(np.ptp(np.asarray(x, np.float64))) if len(x) > 1 else dx
    cable_m = max(cable_m, dx)
    image = seed.astype(np.uint8)
    n, labels, stats, _cent = cv2.connectedComponentsWithStats(image, connectivity=8)
    contours, _hier = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solidity = {}
    for contour in contours:
        rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(contour)
        if rect_w < 1 or rect_h < 1:
            continue
        area = float(cv2.contourArea(contour))
        point = contour.reshape(-1, 2)
        py = int(np.clip(point[0, 1], 0, labels.shape[0] - 1))
        px = int(np.clip(point[0, 0], 0, labels.shape[1] - 1))
        lab = int(labels[py, px])
        if lab <= 0:
            continue
        solidity[lab] = max(solidity.get(lab, 0.0), area / float(rect_w * rect_h))
    flat = labels.ravel()
    sums = np.bincount(flat, weights=z_abs.ravel().astype(np.float64), minlength=n)
    cnts = np.bincount(flat, minlength=n)
    drift_w = np.bincount(flat, weights=drift.ravel().astype(np.float64), minlength=n)
    keep_ids = np.zeros(n, bool)
    for label_id in range(1, int(n)):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        width = int(stats[label_id, cv2.CC_STAT_WIDTH])
        height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        if area < 1:
            continue
        mean_z = float(sums[label_id] / max(cnts[label_id], 1))
        on_drift = float(drift_w[label_id] / max(cnts[label_id], 1)) > 0.25
        width_m = width * dx
        duration_min = height * dt_s / 60.0
        width_frac = width_m / cable_m
        fill = float(area / max(width * height, 1))
        sol = float(solidity.get(label_id, fill))
        wide_glitch = width_frac > 0.7 and duration_min <= 4.0
        if on_drift:
            drop = wide_glitch or mean_z < 0.6 or duration_min < 4.0
        else:
            too_small = area < MIN_AREA_PX or fill < MIN_FILL or sol < 0.15
            too_brief = duration_min < MIN_DURATION_MIN and width_m < 2.0
            too_narrow = width_m < MIN_WIDTH_M and duration_min < 20.0
            weak = mean_z < MIN_MEAN_Z
            drop = wide_glitch or too_small or too_brief or too_narrow or weak
        if not drop:
            keep_ids[label_id] = True
    return keep_ids[labels] & valid


def _tight_keep(abs_r, valid, drift, x, t):
    """Семя по модулю остатка в °C. Порог поднимается, пока доля маски не станет умеренной."""
    sel = abs_r[valid]
    if sel.size == 0:
        return np.zeros(abs_r.shape, bool)
    med = float(np.median(sel))
    mad = float(np.median(np.abs(sel - med))) * 1.4826 + 1e-6
    chosen = np.zeros(abs_r.shape, bool)
    for k_mad, floor_c in ((12.0, 1.2), (8.0, 1.0), (6.0, 0.8)):
        thr = max(med + k_mad * mad, floor_c)
        raw_seed = valid & (abs_r >= thr)
        density = float(raw_seed.sum() / max(int(valid.sum()), 1))
        seed = raw_seed | (drift & valid & (abs_r >= FLOW_Z_MIN))
        seed = morph(seed, "open", (2, 2))
        seed = morph(seed, "close", _adaptive_close(density))
        seed &= valid
        keep = _blob_keep(seed, abs_r, drift, x, t, valid)
        frac = float(keep.sum() / max(int(valid.sum()), 1))
        chosen = keep
        if frac <= 0.03:
            break
    return chosen


def score_sequence(T, valid, x, t, sigma, X=None) -> tuple[np.ndarray, np.ndarray]:
    """s, h. Каналы X не читаются: обрезка ±12 портит хвост FAR."""
    del X
    valid = np.asarray(valid, bool)
    parts = reference_parts(T, valid, x, t, sigma)
    score, hot = assemble_score(parts)
    resid = np.abs(np.where(valid, parts["residual"], 0.0)).astype(np.float32)
    mask = parts["mask"] & valid
    if valid.any():
        med = float(np.median(resid[valid]))
        raw_hi = valid & (resid >= max(med + 1.0, 1.5))
    else:
        raw_hi = np.zeros(valid.shape, bool)
    opened = morph(raw_hi, "open", (2, 2)) if raw_hi.any() else raw_hi
    halo = morph(raw_hi & ~opened, "close", (3, 3)) & valid & ~mask
    score = np.where(halo, score * np.float32(ATTEN), score)
    # Дрейф добавляем только узкой полосой: широкая маска Хафа на seq0072
    # была десятки тысяч пикселей и подняла бы порог FAR.
    dx, dt_s = axes_steps(x, t)
    drift = _drift_mask(resid, valid, dx, dt_s)
    extra = drift & valid & ~mask & (resid >= 1.2)
    n_valid = max(int(valid.sum()), 1)
    if 0 < int(extra.sum()) <= int(0.0005 * n_valid):
        sel = parts["stat"][valid]
        peak = float(np.percentile(sel, 99.5)) + 0.05 if sel.size else 1.0
        score = score + np.float32(0.5 * peak) * extra.astype(np.float32)
    score = np.where(valid, score, 0.0).astype(np.float32)
    return score, hot


def postprocess_unet(s, h, T, valid, x, t, sigma, X=None) -> tuple[np.ndarray, np.ndarray]:
    """Снижает скор U-Net на спеклах и зонах без температурной опоры. Метки не читаются."""
    valid = np.asarray(valid, bool)
    score = np.where(valid, np.asarray(s, np.float32), 0.0).astype(np.float32)
    hot = np.asarray(h, np.float32)
    if hot.shape != score.shape:
        hot = np.zeros_like(score)
    hot = np.where(valid, hot, 0.0).astype(np.float32)
    if X is not None and np.shape(X)[0] >= 2 and np.shape(X)[1:] == score.shape:
        support = np.maximum(np.abs(np.asarray(X[0], np.float32)), np.abs(np.asarray(X[1], np.float32)))
    else:
        z, _resid = thermal_residual(T, valid, sigma)
        support = np.abs(z)
    support = np.where(valid, support, 0.0).astype(np.float32)
    hi = valid & (score >= UNET_HI)
    opened = morph(hi, "open", (2, 2))
    speckle = hi & ~opened & (support < UNET_SUPPORT_Z)
    dx, dt_s = axes_steps(x, t)
    cable_m = float(np.ptp(np.asarray(x, np.float64))) if len(x) > 1 else dx
    cable_m = max(cable_m, dx)
    cv2 = _cv2()
    work = (hi & ((support >= UNET_WEAK_Z) | opened)).astype(np.uint8)
    n, lab, stats, _cent = cv2.connectedComponentsWithStats(work, connectivity=8)
    drop = np.zeros(score.shape, bool)
    for label_id in range(1, int(n)):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        width = int(stats[label_id, cv2.CC_STAT_WIDTH])
        height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        region = lab == label_id
        mean_z = float(support[region].mean()) if area else 0.0
        duration_min = height * dt_s / 60.0
        width_m = width * dx
        width_frac = width_m / cable_m
        reject = False
        if area <= 4:
            reject = True
        elif duration_min < 4.0 and width_m < 1.0:
            reject = True
        elif mean_z < UNET_WEAK_Z and area < 50:
            reject = True
        elif mean_z < 0.45 and width_frac < 0.2:
            reject = True
        if reject:
            drop |= region
    out = score.copy()
    kill = speckle | drop
    out = np.where(kill, np.minimum(out, np.float32(UNET_FLOOR)), out)
    out = np.where(valid, out, 0.0).astype(np.float32)
    return out, hot
