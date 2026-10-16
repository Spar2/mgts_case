"""OpenCV-детектор по PiorlZ/test2_larpathon (mgts_anomaly.py), на нашей сетке.

Источник (только чтение, авторский репозиторий):
https://github.com/PiorlZ/test2_larpathon
Файл mgts_anomaly.py, функция collect_opencv_components и статистический
остаток statistical_anomaly_score / ensemble_score без автоэнкодера.

Их OpenCV-ветка — пересечение маски скора (процентиль 95 ensemble) и маски
контраста соседнего кольца (≥ 1.2 °C), закрытие 3×3, связные компоненты,
адаптивный отсев по площади и длительности, заполненность прямоугольника,
порог контраста 1.2 °C, слияние перекрытий ≥ 50 % и близких рамок одного знака
(зазор ≤ 2 м и ≤ 30 мин). На спокойном файле (MAD ensemble ≤ 0.04) кольцо
58 отсчётов, иначе 16. Автоэнкодер на каждый файл не обучается: без него их
ensemble_score — это 0.55 · нормированный статистический остаток, а процентиль
95 от такого скора совпадает с процентилем самого остатка.

Адаптация к кэшу artifacts/cache/<ver>/<split>/seq.npz:
валидная маска обязательна (в кэше дыры уже записаны нулём); медиана вдоль
кабеля — быстрый uint8-аналог их median_filter; медленная временная медиана
при длинном ряде считается с шагом 4 и интерполируется. Порог 1.2 °C, процентиль,
площади и минуты не подгонялись под метки.

Пиксельный скор s для протокола FAR — статистический остаток в °C (не в
единицах σ: на части записей σ сидит на полу 0.03 и раздувает хвост) плюс
0.7 °C на пикселях оставшихся компонент. Добавка одна на все файлы: свой
перцентиль 99.5 поднимал общий порог val до ~7.6 °C. 0.7 °C выбраны по val
IoU при far 1e-3. h — контраст кольца, °C.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, sobel

from dts.cv_common import (
    _cv2,
    axes_steps,
    denoise_matrix,
    fill_invalid,
    median_axis1,
    morph,
    slow_temporal_background,
    temporal_median_window,
)

# Константы их mgts_anomaly.py (имена и числа сохранены).
TEMPORAL_MEDIAN_STEPS = 601
LOCAL_MEDIAN_STEPS = 15
REGIONAL_MEDIAN_STEPS = 241
DETECTOR_CALM_MAD = 0.04
NEIGHBOR_HALF_POINTS = 16
NEIGHBOR_PRE_STEPS = 5
OPENCV_CALM_HALF_POINTS = 58
OPENCV_MIN_PIXELS = 15
OPENCV_CLOSE_SIZE = (3, 3)  # (x, t) в порядке OpenCV; у них MORPH_RECT (3, 3)
OPENCV_SCORE_PERCENTILE = 95.0
OPENCV_CONTRAST_MIN_C = 1.2
OPENCV_NEAR_METER_GAP_M = 2.0
OPENCV_NEAR_TIME_GAP_MIN = 30.0
OPENCV_CALM_FRAGMENT_MINUTES = 90.0
OPENCV_CALM_LONG_MINUTES = 200.0
STAT_RESIDUAL_W = 0.45
STAT_SPATIAL_W = 0.35
STAT_GRADIENT_W = 0.20
# Фиксированная добавка на маске, °C. Подобрана по val v8 при far 1e-3, не по test.
MASK_BONUS_C = 0.7


def _norm_branch(score: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = np.asarray(score, np.float64)
    sel = values[np.asarray(valid, bool) & np.isfinite(values)]
    if sel.size == 0:
        return np.zeros(values.shape, np.float32)
    scale = float(np.percentile(sel, 99) - np.percentile(sel, 1)) + 1e-6
    normalized = (values - float(np.median(sel))) / scale
    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _ensemble_mad(score: np.ndarray, valid: np.ndarray) -> float:
    sel = np.asarray(score, np.float64)[np.asarray(valid, bool)]
    sel = sel[np.isfinite(sel)]
    if sel.size == 0:
        return 0.0
    med = float(np.median(sel))
    mad = float(np.median(np.abs(sel - med)))
    return mad if np.isfinite(mad) else 0.0


def statistical_score(residual: np.ndarray) -> np.ndarray:
    """Их взвешенный остаток: край, отклонение от пространственных медиан, градиент по времени."""
    values = np.asarray(residual, np.float32)
    smooth = gaussian_filter(values, sigma=(2.0, 3.0))
    residual_edge = np.abs(values - smooth)
    local = np.abs(values - median_axis1(values, LOCAL_MEDIAN_STEPS))
    regional = np.abs(values - median_axis1(values, REGIONAL_MEDIAN_STEPS))
    spatial_dev = np.maximum(local, regional)
    temporal_gradient = np.abs(sobel(values, axis=0))
    score = (
        STAT_RESIDUAL_W * residual_edge
        + STAT_SPATIAL_W * spatial_dev
        + STAT_GRADIENT_W * temporal_gradient
    )
    return score.astype(np.float32)


def _percentile_mask(score: np.ndarray, valid: np.ndarray, percentile: float) -> np.ndarray:
    sel = score[valid]
    sel = sel[np.isfinite(sel)]
    if sel.size == 0 or float(sel.max() - sel.min()) <= 1e-12:
        return np.zeros(score.shape, bool)
    level = float(np.percentile(sel, percentile))
    return valid & np.isfinite(score) & (score >= level)


def _ring_delta(temps: np.ndarray, half_points: int) -> np.ndarray:
    """Температура минус медиана пространственного кольца ширины 2·half+1.

    Пять предыдущих строк их кольца — меньшинство выборок (5 против 32–116),
    в медиану окна они не входят. Центр окна у medianBlur есть; на окне в
    десятки отсчётов один пиксель медиану не сдвигает.
    """
    half = max(int(half_points), 1)
    ref = median_axis1(temps, 2 * half + 1)
    return (np.asarray(temps, np.float32) - ref).astype(np.float32)


def _components(mask: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """Связанные компоненты после закрытия 3×3. Мельче 15 пикселей отбрасываются."""
    if not np.any(mask):
        return np.zeros(mask.shape, np.int32), []
    cv2 = _cv2()
    closed = morph(mask, "close", OPENCV_CLOSE_SIZE)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(closed.astype(np.uint8), connectivity=8)
    comps = []
    for label_id in range(1, int(count)):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        width = int(stats[label_id, cv2.CC_STAT_WIDTH])
        height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        if area < OPENCV_MIN_PIXELS or width < 1 or height < 1:
            continue
        left = int(stats[label_id, cv2.CC_STAT_LEFT])
        top = int(stats[label_id, cv2.CC_STAT_TOP])
        t0, t1 = top, top + height - 1
        s0, s1 = left, left + width - 1
        owned = labels[t0:t1 + 1, s0:s1 + 1] == label_id
        pre = mask[t0:t1 + 1, s0:s1 + 1] & owned
        comps.append(dict(
            label_id=label_id,
            t_range=(t0, t1),
            s_range=(s0, s1),
            n_pixels=area,
            row_fill=float(pre.sum() / max(pre.size, 1)),
            members=[label_id],
        ))
    comps.sort(key=lambda item: (-int(item["n_pixels"]), item["t_range"][0], item["s_range"][0]))
    return labels.astype(np.int32), comps


def _neighbor_amplitude(temps: np.ndarray, t0: int, t1: int, s0: int, s1: int, half: int = NEIGHBOR_HALF_POINTS) -> float:
    """Среднее в прямоугольнике минус медиана кольца снаружи рамки (их neighbor_local_amplitude)."""
    values = np.asarray(temps, np.float32)
    n_time, n_space = values.shape
    t0 = int(np.clip(t0, 0, n_time - 1))
    t1 = int(np.clip(t1, 0, n_time - 1))
    s0 = int(np.clip(s0, 0, n_space - 1))
    s1 = int(np.clip(s1, 0, n_space - 1))
    if t1 < t0:
        t0, t1 = t1, t0
    if s1 < s0:
        s0, s1 = s1, s0
    inside = values[t0:t1 + 1, s0:s1 + 1].ravel()
    inside = inside[np.isfinite(inside)]
    if inside.size == 0:
        return 0.0
    samples = []
    left0 = max(0, s0 - int(half))
    if left0 < s0:
        samples.append(values[t0:t1 + 1, left0:s0].ravel())
    right1 = min(n_space, s1 + 1 + int(half))
    if s1 + 1 < right1:
        samples.append(values[t0:t1 + 1, s1 + 1:right1].ravel())
    pre0 = max(0, t0 - NEIGHBOR_PRE_STEPS)
    if pre0 < t0:
        samples.append(values[pre0:t0, s0:s1 + 1].ravel())
    if not samples:
        return 0.0
    ring = np.concatenate(samples)
    ring = ring[np.isfinite(ring)]
    if ring.size == 0:
        return 0.0
    return float(inside.mean() - np.median(ring))


def _annotate(comps: list[dict], temps: np.ndarray, x: np.ndarray, dt_min: float) -> list[dict]:
    for comp in comps:
        t0, t1 = comp["t_range"]
        s0, s1 = comp["s_range"]
        amp = _neighbor_amplitude(temps, t0, t1, s0, s1)
        comp["amplitude"] = amp
        comp["duration_min"] = float((t1 - t0 + 1) * dt_min)
        start_m = float(x[s0])
        end_m = float(x[s1])
        if start_m > end_m:
            start_m, end_m = end_m, start_m
        comp["meters"] = (start_m, end_m)
        if amp > 0.5:
            comp["type"] = "heating"
        elif amp < -0.5:
            comp["type"] = "cooling"
        else:
            comp["type"] = "unknown"
    return comps


def _median_value(values: list[float]) -> float:
    ordered = sorted(float(v) for v in values)
    count = len(ordered)
    mid = count // 2
    if count % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _filter_area_duration(comps: list[dict]) -> list[dict]:
    """max(40, 0.25·медиана площади) и max(4 мин, 0.3·медиана длительности)."""
    if not comps:
        return []
    areas = [float(c["n_pixels"]) for c in comps]
    durations = [float(c["duration_min"]) for c in comps]
    area_cut = max(40.0, 0.25 * _median_value(areas))
    duration_cut = max(4.0, 0.3 * _median_value(durations))
    return [c for c in comps if c["n_pixels"] >= area_cut and c["duration_min"] >= duration_cut]


def _filter_row_fill(comps: list[dict], calm: bool) -> list[dict]:
    """Спокойный файл: доля < 0.6 снимается. Шумный: доля < 0.45."""
    row_fill_min = 0.6 if calm else 0.45
    return [c for c in comps if float(c["row_fill"]) >= row_fill_min]


def _retype(comps: list[dict]) -> list[dict]:
    """|контраст| < 1.2 °C снимается. Знак ±0.8 °C задаёт heating/cooling."""
    kept = []
    for comp in comps:
        contrast = float(comp["amplitude"])
        duration = float(comp["duration_min"])
        if abs(contrast) < OPENCV_CONTRAST_MIN_C:
            continue
        if contrast >= 0.8:
            comp["type"] = "heating"
        elif contrast <= -0.8:
            comp["type"] = "cooling"
        elif abs(contrast) < 0.8 and duration < 10.0:
            continue
        else:
            comp["type"] = "unknown"
        kept.append(comp)
    return kept


def _meters(comp: dict) -> tuple[float, float]:
    a, b = comp["meters"]
    return (a, b) if a <= b else (b, a)


def _strong_overlap(left: dict, right: dict) -> bool:
    lt0, lt1 = left["t_range"]
    rt0, rt1 = right["t_range"]
    overlap_steps = min(lt1, rt1) - max(lt0, rt0) + 1
    if overlap_steps <= 0:
        return False
    shorter_steps = min(lt1 - lt0 + 1, rt1 - rt0 + 1)
    if overlap_steps + 1e-9 < 0.5 * shorter_steps:
        return False
    lm0, lm1 = _meters(left)
    rm0, rm1 = _meters(right)
    narrower = min(lm1 - lm0, rm1 - rm0)
    if narrower <= 0:
        return False
    overlap_m = min(lm1, rm1) - max(lm0, rm0)
    return overlap_m + 1e-9 >= 0.5 * narrower


def _gap_m(left: dict, right: dict) -> float:
    lm0, lm1 = _meters(left)
    rm0, rm1 = _meters(right)
    if lm1 < rm0:
        return rm0 - lm1
    if rm1 < lm0:
        return lm0 - rm1
    return 0.0


def _gap_min(left: dict, right: dict) -> float:
    lt0, lt1 = left["t_range"]
    rt0, rt1 = right["t_range"]
    if lt1 < rt0:
        gap_steps = rt0 - lt1 - 1
    elif rt1 < lt0:
        gap_steps = lt0 - rt1 - 1
    else:
        return 0.0
    left_steps = lt1 - lt0 + 1
    step = float(left["duration_min"]) / max(left_steps, 1)
    return gap_steps * step


def _same_sign(left: dict, right: dict) -> bool:
    a = float(left.get("amplitude", 0.0))
    b = float(right.get("amplitude", 0.0))
    if a == 0 or b == 0:
        return False
    return (a > 0) == (b > 0)


def _union_find_merge(comps: list[dict], predicate) -> list[dict]:
    items = list(comps)
    count = len(items)
    parent = list(range(count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for i in range(count):
        for j in range(i + 1, count):
            if not predicate(items[i], items[j]):
                continue
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri
    groups: dict[int, list[int]] = {}
    for index in range(count):
        groups.setdefault(find(index), []).append(index)
    merged = []
    for indexes in groups.values():
        if len(indexes) == 1:
            merged.append(items[indexes[0]])
            continue
        members = [items[k] for k in indexes]
        sample = dict(members[0])
        t0 = min(m["t_range"][0] for m in members)
        t1 = max(m["t_range"][1] for m in members)
        s0 = min(m["s_range"][0] for m in members)
        s1 = max(m["s_range"][1] for m in members)
        start_m = min(_meters(m)[0] for m in members)
        end_m = max(_meters(m)[1] for m in members)
        steps = members[0]["t_range"][1] - members[0]["t_range"][0] + 1
        step = float(members[0]["duration_min"]) / max(steps, 1)
        types = {m.get("type", "unknown") for m in members}
        sample["t_range"] = (t0, t1)
        sample["s_range"] = (s0, s1)
        sample["meters"] = (start_m, end_m)
        sample["duration_min"] = float((t1 - t0 + 1) * step)
        sample["n_pixels"] = int(sum(m["n_pixels"] for m in members))
        sample["members"] = [label for m in members for label in m["members"]]
        sample["type"] = next(iter(types)) if len(types) == 1 else "unknown"
        merged.append(sample)
    return merged


def _drop_calm_fragments(comps: list[dict], calm: bool) -> list[dict]:
    if not calm:
        return list(comps)
    long_boxes = [c for c in comps if float(c["duration_min"]) >= OPENCV_CALM_LONG_MINUTES]
    kept = []
    for comp in comps:
        if float(comp["duration_min"]) >= OPENCV_CALM_FRAGMENT_MINUTES:
            kept.append(comp)
            continue
        start_m, end_m = _meters(comp)
        covered = False
        for long_box in long_boxes:
            ls, le = _meters(long_box)
            if start_m + 1e-9 >= ls and end_m <= le + 1e-9:
                covered = True
                break
        if not covered:
            kept.append(comp)
    return kept


def _paint(labels: np.ndarray, comps: list[dict]) -> np.ndarray:
    mask = np.zeros(labels.shape, bool)
    if labels.size == 0 or not comps:
        return mask
    keep_ids = np.zeros(int(labels.max()) + 1, bool)
    for comp in comps:
        for label_id in comp["members"]:
            if 0 < label_id < keep_ids.size:
                keep_ids[label_id] = True
    return keep_ids[labels]


def _extract_layer(score, contrast_abs, valid, half: int, temps, x, dt_min):
    score_on = _percentile_mask(score, valid, OPENCV_SCORE_PERCENTILE)
    contrast_on = valid & np.isfinite(contrast_abs) & (contrast_abs >= OPENCV_CONTRAST_MIN_C)
    labels, comps = _components(score_on & contrast_on)
    _annotate(comps, temps, x, dt_min)
    return labels, comps


def reference_parts(T, valid, x, t, sigma=None) -> dict:
    """Промежуточные карты. Метки не читаются."""
    valid = np.asarray(valid, bool)
    x = np.asarray(x, np.float64)
    t = np.asarray(t, np.float64)
    _dx, dt_s = axes_steps(x, t)
    dt_min = dt_s / 60.0
    if sigma is None:
        sig = np.ones(valid.shape[1], np.float32)
    else:
        sig = np.maximum(np.asarray(sigma, np.float32), 0.03)
    filled = fill_invalid(T, valid)
    denoised = denoise_matrix(filled)
    window = temporal_median_window(denoised.shape[0], TEMPORAL_MEDIAN_STEPS)
    background = slow_temporal_background(denoised, window_steps=window, stride=4)
    residual = (denoised - background).astype(np.float32)
    stat = statistical_score(residual)
    stat = np.where(valid, stat, 0.0).astype(np.float32)
    norm = _norm_branch(stat, valid)
    # Без автоэнкодера их ensemble = 0.55 * norm(stat). MAD и процентиль
    # считаются по этой шкале, как в collect_opencv_components.
    ensemble = (0.55 * norm).astype(np.float32)
    calm = _ensemble_mad(ensemble, valid) <= DETECTOR_CALM_MAD
    half = OPENCV_CALM_HALF_POINTS if calm else NEIGHBOR_HALF_POINTS
    delta = _ring_delta(denoised, half)
    contrast_abs = np.abs(delta)
    labels, comps = _extract_layer(ensemble, contrast_abs, valid, half, filled, x, dt_min)
    if calm:
        delta_n = _ring_delta(denoised, NEIGHBOR_HALF_POINTS)
        labels_n, extra = _extract_layer(ensemble, np.abs(delta_n), valid, NEIGHBOR_HALF_POINTS, filled, x, dt_min)
        dense = [c for c in comps if float(c["row_fill"]) >= 0.6]
        for item in extra:
            if float(item["row_fill"]) < 0.6 or float(item["duration_min"]) < OPENCV_CALM_FRAGMENT_MINUTES:
                continue
            if any(_strong_overlap(item, other) for other in dense):
                continue
            # Пиксели узкого кольца лежат в другой карте меток. Переносим их
            # в общую карту отдельным идентификатором.
            region = (labels_n == item["label_id"]) & (labels == 0)
            if not region.any():
                continue
            new_id = int(labels.max()) + 1
            labels[region] = new_id
            item["label_id"] = new_id
            item["members"] = [new_id]
            comps.append(item)
            dense.append(item)
    comps = _filter_area_duration(comps)
    comps = _filter_row_fill(comps, calm)
    comps = _retype(comps)
    comps = _union_find_merge(comps, _strong_overlap)
    comps = _drop_calm_fragments(comps, calm)
    comps = _union_find_merge(
        comps,
        lambda a, b: _same_sign(a, b) and _gap_m(a, b) <= OPENCV_NEAR_METER_GAP_M + 1e-9 and _gap_min(a, b) <= OPENCV_NEAR_TIME_GAP_MIN + 1e-9,
    )
    mask = _paint(labels, comps) & valid
    return dict(
        stat=stat, ensemble=ensemble, delta=delta, contrast_abs=contrast_abs.astype(np.float32),
        mask=mask, valid=valid, sigma=sig, calm=calm, n_components=len(comps), residual=residual,
    )


def assemble_score(parts: dict) -> tuple[np.ndarray, np.ndarray]:
    """Непрерывный скор для порога FAR и знаковая горячность.

    База — статистический остаток в °C (не делим на σ: на части записей σ
    сидит на полу 0.03 и раздувает хвост). Пиксели компонент получают одну и
    ту же добавку MASK_BONUS_C на всех файлах, иначе шкала скора не сравнивается
    между записями и общий порог FAR садится по самым шумным.
    """
    valid = parts["valid"]
    stat = np.where(valid, parts["stat"], 0.0).astype(np.float32)
    score = stat + np.float32(MASK_BONUS_C) * parts["mask"].astype(np.float32)
    score = np.where(valid, score, 0.0).astype(np.float32)
    hot = np.where(valid, parts["delta"], 0.0).astype(np.float32)
    return score, hot


def score_sequence(T, valid, x, t, sigma=None) -> tuple[np.ndarray, np.ndarray]:
    """s float32 и h float32 по одной последовательности кэша. Метки не используются."""
    parts = reference_parts(T, valid, x, t, sigma)
    return assemble_score(parts)
