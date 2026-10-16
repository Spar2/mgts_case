"""OpenCV-аннотация уже найденных зон: форма, фронт, период.

cv2 импортируется только внутри вызовов. Порог по меткам и по карточкам
здесь не ставится: на вход приходит маска или скор и список зон.
"""
from __future__ import annotations

import numpy as np

from dts.cv_common import _cv2, axes_steps

# Скаляры, которые annotate пишет в словарь зоны и которые export кладёт в CSV.
CV_FIELDS = (
    "cv_area_px",
    "cv_solidity",
    "cv_eccentricity",
    "cv_orientation_deg",
    "cv_bbox_width_m",
    "cv_bbox_duration_h",
    "cv_drift_m_per_h",
    "cv_n_components",
    "cv_period_h",
)


def annotate(score_or_mask, dT, valid, x, t, zones):
    """Контур и скаляры формы для каждой зоны.

    `score_or_mask` — булева маска аномалии либо карта скора. Для скора внутри
    рамки зоны остаются верхние `n_px` валидных клеток (как у компоненты,
    из которой собрана рамка). `dT` — карта ΔT той же формы; по ней считается
    автокорреляция вдоль времени. Индексы зон — в эту сетку (`t0:t1`, `x0:x1`);
    если индексов нет, рамка берётся по метрам и секундам.

    На каждую зону дописываются скаляры `cv_*`. Контур и отрезок фронта
    остаются только в возвращаемом списке (в CSV они не входят).
    """
    valid = np.asarray(valid, bool)
    x = np.asarray(x, np.float64)
    t = np.asarray(t, np.float64)
    field = None if dT is None else np.asarray(dT, np.float32)
    score = np.asarray(score_or_mask)
    if score.shape != valid.shape:
        raise ValueError(f"score {score.shape} != valid {valid.shape}")
    if field is not None and field.shape != valid.shape:
        raise ValueError(f"dT {field.shape} != valid {valid.shape}")
    dx, dt_s = axes_steps(x, t)
    out = []
    for zone in zones:
        ann = _one_zone(score, field, valid, x, t, zone, dx, dt_s)
        for key in CV_FIELDS:
            zone[key] = ann[key]
        out.append(ann)
    return out


def overlay(fig, annotations, x, t, ax=None):
    """Контуры зон и линии фронта на карте, чья ось y — часы от t[0] сверху вниз.

    Так же задаёт extent `run_real` для ΔT: низ картинки — конец записи.
    `fig` — фигура matplotlib; если осей нет, создаётся одна.
    """
    if ax is None:
        ax = fig.axes[0] if getattr(fig, "axes", None) else fig.add_subplot(111)
    x = np.asarray(x, np.float64)
    t = np.asarray(t, np.float64)
    if x.size == 0 or t.size == 0:
        return ax
    t0 = float(t[0])

    def xy(col, row):
        col = int(np.clip(col, 0, x.size - 1))
        row = int(np.clip(row, 0, t.size - 1))
        return float(x[col]), (float(t[row]) - t0) / 3600.0

    for ann in annotations:
        contour = ann.get("contour")
        if contour is not None and len(contour) >= 2:
            pts = np.asarray(contour, np.int32)
            xs, ys = [], []
            for col, row in pts:
                px, py = xy(col, row)
                xs.append(px)
                ys.append(py)
            xs.append(xs[0])
            ys.append(ys[0])
            ax.plot(xs, ys, color="black", lw=0.9, solid_capstyle="round")
        line = ann.get("front_line")
        if line is None:
            continue
        x1, y1, x2, y2 = (int(v) for v in line)
        ax.plot([xy(x1, y1)[0], xy(x2, y2)[0]], [xy(x1, y1)[1], xy(x2, y2)[1]], color="lime", lw=1.3)
    return ax


def show_dT(ax, dT, valid, x, t):
    """ΔT в тех же координатах, в которых overlay кладёт контуры."""
    valid = np.asarray(valid, bool)
    show = np.where(valid, np.asarray(dT, np.float32), np.nan)
    finite = show[np.isfinite(show)]
    scale = float(np.percentile(np.abs(finite), 99.5)) if finite.size else 1.0
    scale = max(scale, 1e-3)
    x = np.asarray(x, np.float64)
    t = np.asarray(t, np.float64)
    extent = [float(x[0]), float(x[-1]), (float(t[-1]) - float(t[0])) / 3600.0, 0.0]
    image = ax.imshow(show, aspect="auto", cmap="RdBu_r", vmin=-scale, vmax=scale, extent=extent, interpolation="nearest")
    ax.set_xlabel("x, м")
    ax.set_ylabel("время от начала, ч")
    return image


def _bounds(zone, x, t, shape):
    n_t, n_x = shape
    if all(k in zone for k in ("t0", "t1", "x0", "x1")):
        t0, t1, x0, x1 = (int(zone["t0"]), int(zone["t1"]), int(zone["x0"]), int(zone["x1"]))
    else:
        x0 = int(np.searchsorted(x, float(zone["start_m"])))
        x1 = int(np.searchsorted(x, float(zone["end_m"]), side="right"))
        t0 = int(np.searchsorted(t, float(zone["t_start_s"])))
        t1 = int(np.searchsorted(t, float(zone["t_end_s"]), side="right"))
    t0 = int(np.clip(t0, 0, n_t))
    t1 = int(np.clip(t1, 0, n_t))
    x0 = int(np.clip(x0, 0, n_x))
    x1 = int(np.clip(x1, 0, n_x))
    if t1 < t0:
        t0, t1 = t1, t0
    if x1 < x0:
        x0, x1 = x1, x0
    return t0, t1, x0, x1


def _zone_mask(score, valid, t0, t1, x0, x1, n_px):
    crop = score[t0:t1, x0:x1]
    v = valid[t0:t1, x0:x1]
    if crop.size == 0:
        return np.zeros((max(t1 - t0, 0), max(x1 - x0, 0)), bool)
    if crop.dtype == bool:
        return crop & v
    values = np.asarray(crop, np.float32)
    if np.issubdtype(score.dtype, np.integer):
        return v & (values > 0)
    finite = v & np.isfinite(values)
    if not finite.any():
        return np.zeros(values.shape, bool)
    if n_px > 0:
        flat = values[finite]
        keep = min(int(n_px), int(flat.size))
        if keep <= 0:
            return np.zeros(values.shape, bool)
        order = np.argpartition(flat, flat.size - keep)
        level = float(flat[order[flat.size - keep]])
        return finite & (values >= level)
    level = 0.5 if float(np.nanmax(values)) <= 1.5 else 0.0
    return finite & (values >= level)


def _shape(mask_u8, contour):
    cv2 = _cv2()
    area_px = int(mask_u8.sum())
    solidity = 1.0 if area_px else 0.0
    ecc, orient = 0.0, 0.0
    if contour is None or len(contour) < 3:
        return area_px, solidity, ecc, orient
    area = abs(float(cv2.contourArea(contour)))
    hull = cv2.convexHull(contour)
    hull_area = abs(float(cv2.contourArea(hull)))
    if hull_area > 1e-6:
        solidity = float(np.clip(area / hull_area, 0.0, 1.0))
    ecc, orient = _eccentricity(contour)
    return area_px, solidity, ecc, orient


def _eccentricity(contour):
    cv2 = _cv2()
    pts = np.asarray(contour, np.float32)
    if pts.ndim == 3:
        pts = pts.reshape(-1, 2)
    if pts.shape[0] >= 5:
        try:
            (_center, (width, height), angle) = cv2.fitEllipse(pts)
            major = max(float(width), float(height)) / 2.0
            minor = min(float(width), float(height)) / 2.0
            if major > 1e-6:
                ecc = float(np.sqrt(max(0.0, 1.0 - (minor * minor) / (major * major))))
                return ecc, _wrap_deg(float(angle))
        except cv2.error:
            pass
    moments = cv2.moments(np.asarray(contour, np.int32))
    mu20, mu02, mu11 = moments["mu20"], moments["mu02"], moments["mu11"]
    common = float(np.hypot(mu20 - mu02, 2.0 * mu11))
    major = 0.5 * (mu20 + mu02 + common)
    minor = 0.5 * (mu20 + mu02 - common)
    ecc = float(np.sqrt(max(0.0, 1.0 - minor / major))) if major > 1e-9 else 0.0
    angle = 0.5 * float(np.degrees(np.arctan2(2.0 * mu11, mu20 - mu02)))
    return ecc, _wrap_deg(angle)


def _wrap_deg(angle):
    wrapped = (float(angle) + 90.0) % 180.0 - 90.0
    if wrapped == -90.0:
        return 90.0
    return wrapped


def _front(mask_u8, dx, dt_s):
    """Самый длинный не горизонтальный отрезок Хафа и его скорость, м/ч.

    Горизонтальный край рамки — это не дрейф по кабелю. Вертикальный след
    (пятно стоит на месте) даёт скорость около нуля.
    """
    cv2 = _cv2()
    if int(mask_u8.sum()) < 8:
        return None, None
    edges = cv2.Canny(mask_u8 * np.uint8(255), 40, 120)
    if int(edges.sum()) < 8:
        return None, None
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=8, minLineLength=8, maxLineGap=4)
    if lines is None:
        return None, None
    arr = np.asarray(lines)
    if arr.ndim == 3:
        arr = arr[:, 0]
    best = None
    best_len = -1.0
    for x1, y1, x2, y2 in arr:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        dy = y2 - y1
        if abs(dy) < 2:
            continue
        length = abs(x2 - x1) + abs(dy)
        if length > best_len:
            best_len = length
            speed = ((x2 - x1) * dx) / (dy * dt_s / 3600.0)
            best = (speed, (x1, y1, x2, y2))
    if best is None:
        return None, None
    return float(best[0]), best[1]


def _period_hours(profile, dt_s):
    values = np.asarray(profile, np.float64)
    values = values[np.isfinite(values)]
    if values.size < 8:
        return None
    centered = values - float(values.mean())
    if float(np.dot(centered, centered)) <= 1e-12:
        return None
    corr = np.correlate(centered, centered, mode="full")
    corr = corr[values.size - 1 :]
    corr = corr / (corr[0] + 1e-12)
    min_lag = 2
    if corr.size <= min_lag + 2:
        return None
    segment = corr[min_lag:]
    for i in range(1, segment.size - 1):
        if segment[i] >= segment[i - 1] and segment[i] >= segment[i + 1] and segment[i] >= 0.25:
            lag = i + min_lag
            return float(lag * dt_s / 3600.0)
    return None


def _profile(field, valid, t0, t1, x0, x1, mask):
    if field is not None:
        block = np.where(valid[t0:t1, x0:x1], field[t0:t1, x0:x1], np.nan)
    else:
        block = mask.astype(np.float32)
    with np.errstate(all="ignore"):
        row = np.nanmean(block, axis=1)
    return row


def _one_zone(score, field, valid, x, t, zone, dx, dt_s):
    t0, t1, x0, x1 = _bounds(zone, x, t, valid.shape)
    n_px = int(zone.get("n_px") or 0)
    mask = _zone_mask(score, valid, t0, t1, x0, x1, n_px)
    empty = dict(
        zone_id=zone.get("id"),
        contour=None,
        front_line=None,
        cv_area_px=0,
        cv_solidity=0.0,
        cv_eccentricity=0.0,
        cv_orientation_deg=0.0,
        cv_bbox_width_m=0.0,
        cv_bbox_duration_h=0.0,
        cv_drift_m_per_h=None,
        cv_n_components=0,
        cv_period_h=None,
    )
    width_m = float(max(x1 - x0 - 1, 0) * dx) if x1 > x0 else 0.0
    duration_h = float(max(t1 - t0 - 1, 0) * dt_s / 3600.0) if t1 > t0 else 0.0
    # Ширина и длительность по крайним пикселям маски, а не по пустому краю рамки.
    if mask.any():
        rows = np.flatnonzero(mask.any(axis=1))
        cols = np.flatnonzero(mask.any(axis=0))
        width_m = float((cols[-1] - cols[0]) * dx)
        duration_h = float((rows[-1] - rows[0]) * dt_s / 3600.0)
    empty["cv_bbox_width_m"] = width_m
    empty["cv_bbox_duration_h"] = duration_h
    if not mask.any():
        return empty
    cv2 = _cv2()
    image = mask.astype(np.uint8)
    n_labels, _lab = cv2.connectedComponents(image, connectivity=8)
    contours, _hier = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = None
    if contours:
        contour = max(contours, key=cv2.contourArea)
    area_px, solidity, ecc, orient = _shape(image, contour)
    full_contour = None
    if contour is not None and len(contour) >= 1:
        pts = contour.reshape(-1, 2).astype(np.int32)
        pts[:, 0] += x0
        pts[:, 1] += t0
        full_contour = pts
    speed, line = _front(image, dx, dt_s)
    full_line = None
    if line is not None:
        lx1, ly1, lx2, ly2 = line
        full_line = (lx1 + x0, ly1 + t0, lx2 + x0, ly2 + t0)
    period = _period_hours(_profile(field, valid, t0, t1, x0, x1, mask), dt_s)
    return dict(
        zone_id=zone.get("id"),
        contour=full_contour,
        front_line=full_line,
        cv_area_px=area_px,
        cv_solidity=solidity,
        cv_eccentricity=ecc,
        cv_orientation_deg=orient,
        cv_bbox_width_m=width_m,
        cv_bbox_duration_h=duration_h,
        cv_drift_m_per_h=speed,
        cv_n_components=int(max(n_labels - 1, 0)),
        cv_period_h=period,
    )
