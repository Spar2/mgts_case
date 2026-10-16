"""Признаки «климатика работает / не работает» по одному ряду температуры.

Пассивный участок описывается моделью первого порядка
    dT/dt = −k (T − T_inf)
(релаксация к равновесию помещения). Коэффициенты — робастная регрессия Хубера
на отсчётах с малой |d²T/dt²|. Активный отсчёт — там, где остаток производной
больше 3·MAD и такие отсчёты идут подряд не короче min_run.

Это не детектор по пространственному z-score. Признаки подаются дополнительным
входом в нейросеть этапа 2 (EventNet) только если use_runtime=True. При
use_runtime=False граф и размер скаляров совпадают с dts.models.EventNet.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d

# Имена скаляров, которые бенч клеит к SCALAR_NAMES. period и k пересчитаны
# в удобный для сети масштаб: log(минуты) и 1/час. Сырой словарь runtime_features
# хранит секунды и 1/с.
FEATURE_NAMES = (
    "active_frac",
    "log_period_min",
    "duty",
    "k_per_h",
    "T_inf",
    "resid_mad",
    "active_sign",
)
SERIES_LEN = 128


def _prepare(series, time_s):
    s = np.asarray(series, np.float64).reshape(-1)
    t = np.asarray(time_s, np.float64).reshape(-1)
    n = min(len(s), len(t))
    s, t = s[:n], t[:n]
    m = np.isfinite(s) & np.isfinite(t)
    s, t = s[m], t[m]
    if len(t) == 0:
        return s, t
    order = np.argsort(t, kind="mergesort")
    s, t = s[order], t[order]
    if len(t) > 1:
        keep = np.r_[True, np.diff(t) > 0]
        s, t = s[keep], t[keep]
    return _decimate(s, t)


def _decimate(s, t, max_n=4096):
    n = len(s)
    if n <= max_n or n < 4:
        return s, t
    m = int(np.ceil(n / max_n))
    n2 = n // m
    if n2 < 8:
        return s, t
    s = s[: n2 * m].reshape(n2, m).mean(1)
    t = t[: n2 * m].reshape(n2, m).mean(1)
    return s, t


def _huber_line(x, y, n_iter=20):
    """y ≈ a + b x, IRLS с весами Хубера. Возвращает (a, b)."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    A = np.column_stack([np.ones(len(x)), x])
    w = np.ones(len(x))
    coef = np.zeros(2)
    for _ in range(n_iter):
        sw = np.sqrt(w)[:, None]
        coef, *_ = np.linalg.lstsq(A * sw, y * sw[:, 0], rcond=None)
        resid = y - A @ coef
        med = np.median(resid)
        scale = 1.4826 * np.median(np.abs(resid - med)) + 1e-12
        u = np.abs(resid) / (1.345 * scale)
        w = np.where(u <= 1.0, 1.0, 1.0 / np.maximum(u, 1e-12))
    return float(coef[0]), float(coef[1])


def _dominant_period(y, t):
    """Период по локальному пику периодограммы. 0, если это тренд, а не цикл.

    Возвращает (period_s, confidence). confidence — доля энергии спектра в пике.
    """
    y = np.asarray(y, np.float64)
    t = np.asarray(t, np.float64)
    if len(y) < 16:
        return 0.0, 0.0
    dt = float(np.median(np.diff(t)))
    dur = float(t[-1] - t[0])
    if not np.isfinite(dt) or dt <= 0 or dur <= 4 * dt:
        return 0.0, 0.0
    n = int(round(dur / dt)) + 1
    n = int(np.clip(n, 16, 8192))
    tu = np.linspace(t[0], t[-1], n)
    yu = np.interp(tu, t, y)
    yu = yu - yu.mean()
    sd = float(yu.std())
    if sd < 1e-8:
        return 0.0, 0.0
    yu /= sd
    step = dur / (n - 1)
    spec = np.abs(np.fft.rfft(yu * np.hanning(n))) ** 2
    freqs = np.fft.rfftfreq(n, d=step)
    if len(freqs) < 5:
        return 0.0, 0.0
    df = float(freqs[1])
    # Меньше ~1.25 цикла на записи — не отличить от тренда. Короче 4 отсчётов — шум.
    f_min = 1.25 / dur
    f_max = 1.0 / (4.0 * step)
    band = np.where((freqs >= f_min) & (freqs <= f_max))[0]
    if len(band) < 3:
        return 0.0, 0.0
    cands = []
    for k in band:
        if k <= 0 or k >= len(spec) - 1:
            continue
        if spec[k] >= spec[k - 1] and spec[k] >= spec[k + 1]:
            cands.append(int(k))
    if not cands:
        return 0.0, 0.0
    peak = max(float(spec[k]) for k in cands)
    med = float(np.median(spec[band])) + 1e-12
    if peak / med < 4.0:
        return 0.0, 0.0
    # Фундаментальная, а не гармоника: самая низкая частота среди сильных пиков.
    strong = [k for k in cands if spec[k] >= 0.5 * peak]
    k = int(min(strong))
    a0, a1, a2 = float(spec[k - 1]), float(spec[k]), float(spec[k + 1])
    den = a0 - 2.0 * a1 + a2
    delta = 0.5 * (a0 - a2) / den if abs(den) > 1e-12 else 0.0
    delta = float(np.clip(delta, -0.5, 0.5))
    freq = float(freqs[k] + delta * df)
    if freq <= 0:
        return 0.0, 0.0
    period = 1.0 / freq
    if period < 4.0 * step or period > dur / 1.2:
        return 0.0, 0.0
    conf = float(spec[k] / (spec.sum() + 1e-12))
    return float(period), conf


def _runs(active, t, dT):
    """Список (t_start, t_end, median dT) по непрерывным активным участкам."""
    runs = []
    n = len(active)
    i = 0
    while i < n:
        if not active[i]:
            i += 1
            continue
        j = i + 1
        while j < n and active[j]:
            j += 1
        runs.append((float(t[i]), float(t[j - 1]), float(np.median(dT[i:j]))))
        i = j
    return runs


def _gap_period(runs):
    if len(runs) < 2:
        return 0.0
    gaps = np.diff([r[0] for r in runs])
    gaps = gaps[gaps > 0]
    if len(gaps) == 0:
        return 0.0
    return float(np.median(gaps))


def _choose_period(fft_period, gap_period):
    """Спектральный период, если он есть. Иначе медиана интервалов между включениями.

    На синусе активных включений нет, период виден только в спектре.
    """
    if fft_period > 0:
        return float(fft_period)
    return float(gap_period)


def runtime_features(series, time_s, *, min_run=5, mad_mult=3.0, curv_q=0.4):
    """Признаки режима по ряду средней температуры зоны.

    series, time_s — одинаковой длины (температура и секунды). Словарь:
      active_frac — доля активного времени;
      period_s — период циклов, секунды (0, если цикла нет);
      duty — скважность, медиана доли активного времени внутри цикла;
      k — коэффициент пассивной модели, 1/с;
      T_inf — равновесная температура пассивной модели (те же единицы, что series);
      resid_mad — MAD остатка dT/dt относительно пассивной модели;
      active_sign — +1 нагрев, −1 охлаждение, 0 если активного участка нет.
    """
    s, t = _prepare(series, time_s)
    empty = {
        "active_frac": 0.0,
        "period_s": 0.0,
        "duty": 0.0,
        "k": 0.0,
        "T_inf": float(np.median(s)) if len(s) else 0.0,
        "resid_mad": 0.0,
        "active_sign": 0.0,
        "period_conf": 0.0,
        "n_runs": 0,
    }
    n = len(s)
    if n < 8 or float(np.std(s)) < 1e-8:
        return empty
    w = 5 if n >= 21 else 3
    sm = uniform_filter1d(s, size=w, mode="nearest")
    dT = np.gradient(sm, t)
    d2 = np.gradient(dT, t)
    curv = np.abs(d2)
    border = max(w, 2)
    interior = np.zeros(n, dtype=bool)
    interior[border: n - border] = True
    thr = float(np.quantile(curv[interior], curv_q)) if interior.sum() >= 8 else float(np.quantile(curv, curv_q))
    mask = (curv <= thr) & interior
    if int(mask.sum()) < 8:
        mask = interior.copy() if int(interior.sum()) >= 8 else np.ones(n, dtype=bool)
    med = float(np.median(sm[mask]))
    a, b = _huber_line(sm[mask] - med, dT[mask])
    k = float(-b)
    if not np.isfinite(k) or abs(k) > 1.0:
        k = 0.0
        a, b = float(np.median(dT[mask])), 0.0
    if abs(k) < 1e-10:
        Tinf = float(np.median(sm))
    else:
        Tinf = float(med + a / k) if k != 0 else float(np.median(sm))
    if not np.isfinite(Tinf) or abs(Tinf) > 1e4:
        Tinf = float(np.median(sm))
    dhat = a + b * (sm - med)
    resid = dT - dhat
    rc = resid[mask]
    mad = float(1.4826 * np.median(np.abs(rc - np.median(rc))))
    if not np.isfinite(mad) or mad < 0:
        mad = 0.0
    scale = float(np.median(np.abs(dT))) + 1e-12
    # Идеальная пассива: остаток на порядке численного шума, активных точек нет.
    if mad < 1e-4 * scale and float(np.percentile(np.abs(resid), 90)) <= max(3.0 * mad, 1e-5 * scale):
        active = np.zeros(n, dtype=bool)
        mad = max(mad, 0.0)
    else:
        mad = max(mad, 1e-8 * scale)
        active = np.abs(resid) > mad_mult * mad
        active[:border] = False
        active[-border:] = False
        i = 0
        while i < n:
            if not active[i]:
                i += 1
                continue
            j = i + 1
            while j < n and active[j]:
                j += 1
            if j - i < int(min_run):
                active[i:j] = False
            i = j
    dt = np.empty(n, np.float64)
    if n > 1:
        dif = np.diff(t)
        dt[:-1] = dif
        dt[-1] = dif[-1]
    else:
        dt[0] = 0.0
    total = float(dt.sum())
    active_frac = float(dt[active].sum() / total) if total > 0 else 0.0
    if active.any():
        med_d = float(np.median(dT[active]))
        if abs(med_d) > 1e-10:
            sign = float(np.sign(med_d))
        else:
            signed = float(np.sum(dT[active] * dt[active]))
            if abs(signed) > 1e-10:
                sign = float(np.sign(signed))
            else:
                sign = float(np.sign(np.median(sm[active]) - Tinf))
    else:
        sign = 0.0
    runs = _runs(active, t, dT)
    fft_p, conf = _dominant_period(sm, t)
    gap_p = _gap_period(runs)
    period = _choose_period(fft_p, gap_p)
    dur = float(t[-1] - t[0])
    if period > 0 and dur >= 1.5 * period:
        n_cyc = int(np.floor(dur / period))
        duties = []
        for c in range(n_cyc):
            t0 = t[0] + c * period
            t1 = t0 + period
            sel = (t >= t0) & (t < t1)
            denom = float(dt[sel].sum())
            if denom <= 0 or int(sel.sum()) < 3:
                continue
            duties.append(float(dt[sel & active].sum() / denom))
        duty = float(np.median(duties)) if duties else active_frac
    else:
        duty = active_frac
    return {
        "active_frac": float(np.clip(active_frac, 0.0, 1.0)),
        "period_s": float(max(period, 0.0)),
        "duty": float(np.clip(duty, 0.0, 1.0)),
        "k": float(k),
        "T_inf": float(Tinf),
        "resid_mad": float(max(mad, 0.0)),
        "active_sign": float(0.0 if sign == 0 else np.sign(sign)),
        "period_conf": float(conf),
        "n_runs": int(len(runs)),
    }


def feature_vector(feat) -> np.ndarray:
    """Семь чисел в порядке FEATURE_NAMES."""
    period = max(float(feat["period_s"]), 0.0)
    return np.asarray(
        [
            float(feat["active_frac"]),
            np.log1p(period / 60.0),
            float(feat["duty"]),
            float(feat["k"]) * 3600.0,
            float(feat["T_inf"]),
            float(feat["resid_mad"]),
            float(feat["active_sign"]),
        ],
        np.float32,
    )


def stack_scalars(scal, runtime, enabled=False):
    """Дописать runtime-признаки к скалярам этапа 2.

    enabled=False возвращает тот же объект scal, без копии и без изменения ширины.
    """
    if not enabled:
        return scal
    base = np.asarray(scal, np.float32)
    extra = np.asarray(runtime, np.float32)
    if extra.ndim == 1:
        extra = np.broadcast_to(extra, (base.shape[0], extra.shape[0])).copy()
    if extra.shape[0] != base.shape[0]:
        raise ValueError(f"runtime rows {extra.shape[0]} != scalars {base.shape[0]}")
    return np.concatenate([base, extra], 1)


def resample_trace(series, time_s, length=SERIES_LEN) -> np.ndarray:
    """Равномерный ряд фиксированной длины, z-score по самому ряду (форма, не амплитуда)."""
    s, t = _prepare(series, time_s)
    out = np.zeros(int(length), np.float32)
    if len(s) < 2 or float(t[-1] - t[0]) <= 0:
        return out
    ti = np.linspace(t[0], t[-1], int(length))
    y = np.interp(ti, t, s)
    sd = float(y.std())
    if sd < 1e-8:
        return out
    return ((y - y.mean()) / sd).astype(np.float32)


def zone_mean_series(channel, valid, x, xa, xb):
    """Среднее по x в [xa, xb] вдоль всего времени. channel, valid: (T, W)."""
    x = np.asarray(x, np.float64)
    xa, xb = float(xa), float(xb)
    if xb < xa:
        xa, xb = xb, xa
    ix = (x >= xa) & (x <= xb)
    if not np.any(ix):
        ix = np.zeros(len(x), dtype=bool)
        ix[int(np.argmin(np.abs(x - 0.5 * (xa + xb))))] = True
    ch = np.asarray(channel[:, ix], np.float64)
    m = np.asarray(valid[:, ix], np.float64) > 0
    w = m.sum(1)
    acc = (ch * m).sum(1)
    series = np.zeros(ch.shape[0], np.float64)
    np.divide(acc, w, out=series, where=w > 0)
    if np.any(w == 0) and np.any(w > 0):
        tt = np.arange(len(series))
        series[w == 0] = np.interp(tt[w == 0], tt[w > 0], series[w > 0])
    return series


def make_runtime_net(n_classes, n_scalar, n_runtime=0, use_runtime=False, use_series=False,
                     series_len=SERIES_LEN, cin=3, w=24):
    """EventNet. use_runtime=False — тот же вход (n_scalar), без ветки ряда.

    use_runtime=True — скалярный вход шире на n_runtime (признаки клеит вызывающий
    код через stack_scalars). use_series=True добавляет свёртку по ряду температуры;
    её вклад в логиты на старте маленький (std=0.01), чтобы не сбить кроп.
    """
    import torch
    from torch import nn
    from . import models

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.use_runtime = bool(use_runtime)
            self.use_series = bool(use_series) and self.use_runtime
            self.series_len = int(series_len)
            n_in = int(n_scalar) + (int(n_runtime) if self.use_runtime else 0)
            self.n_in = n_in
            self.backbone = models.EventNet(n_classes, n_in, cin=cin, w=w)
            if self.use_series:
                self.seq = nn.Sequential(
                    nn.Conv1d(1, 16, 7, padding=3, bias=False),
                    nn.BatchNorm1d(16),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(16, 32, 5, padding=2, bias=False),
                    nn.BatchNorm1d(32),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool1d(1),
                )
                self.seq_head = nn.Linear(32, n_classes)
                nn.init.normal_(self.seq_head.weight, std=0.01)
                nn.init.zeros_(self.seq_head.bias)

        def forward(self, img, scal, series=None):
            logits = self.backbone(img, scal)
            if self.use_series:
                if series is None:
                    raise ValueError("use_series требует ряд температуры")
                h = self.seq(series.unsqueeze(1)).flatten(1)
                logits = logits + self.seq_head(h)
            return logits

    return _Net()
