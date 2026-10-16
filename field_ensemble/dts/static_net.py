"""Нейросетевой канал статичных узких горячих точек.

Победитель (dts.features.make_channels) кладёт в канал 0 величину
T − синфазная составляющая − статический профиль по времени. Точка, которая
есть с первого кадра и не включается внутри окна, уходит в этот профиль, и
детектор её не видит.

Эта ветвь смотрит на сырой пространственный остаток в градусах Цельсия:
T минус синфазная составляющая минус гладкая пространственная медиана (~40 м).
Деления на σ(x) нет — это не z-score. Дальше 1D-сеть с большим полем зрения и
обучаемой гладкой опорой (пул шире 6 м, узкий горб в опору не влезает).
Выход — вероятность столбца. На карту времени она растягивается и сливается
с ансамблем через max/blend, не опуская скор победителя.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .static_synth import DX

FEAT_SCALE_C = 10.0  # фиксированные градусы, не локальная σ
N_FEAT = 9
# Длинное окно STATIC2: до 256 кадров на всю запись и 8 медиан по равным долям времени.
LONG_ROWS = 256
LONG_BINS = 8
# Инференс длинной записи. Обучение кропает x до 1280 и по времени берёт либо всю
# запись (75 %), либо отрезок не короче 192 кадров. Опубликованный synth IoU —
# один predict_map на всю ширину. Кроп 1280 по x этот проход не повторяет:
# GroupNorm считается по ширине кропа, val IoU уезжает с 0.386027 на 0.386810.
# По x остаётся один проход. По времени — окно 640 (короче уже другие столбцы
# выше замороженного порога), перекрытие половина, склейка max. Запись короче
# окна равна одному predict_map.
INFER_TIME_WIN = 640
INFER_TIME_OVERLAP = 0.5
TREND_M = 40.0
DILATIONS = (1, 2, 4, 8, 16, 32, 16, 8, 4, 1)


def moving_average(y: np.ndarray, win: int) -> np.ndarray:
    win = max(1, int(win) | 1)
    if win <= 1 or len(y) < 2:
        return np.array(y, np.float32, copy=True)
    pad = win // 2
    yp = np.pad(np.asarray(y, np.float64), (pad, pad), mode="edge")
    c = np.empty(len(yp) + 1, np.float64)
    c[0] = 0.0
    np.cumsum(yp, out=c[1:])
    return ((c[win:] - c[:-win]) / win).astype(np.float32)


def _sorted_valid(R: np.ndarray, valid: np.ndarray):
    """Сортировка по времени. Невалидные уезжают в +inf, валидные занимают начало столбца."""
    filled = np.where(valid, R, np.inf)
    order = np.argsort(filled, axis=0)
    return np.take_along_axis(filled, order, axis=0), valid.sum(axis=0).astype(np.int32)


def _pick(sorted_v: np.ndarray, counts: np.ndarray, q: float) -> np.ndarray:
    rank = np.rint((q / 100.0) * np.maximum(counts - 1, 0)).astype(np.int64)
    rank = np.clip(rank, 0, sorted_v.shape[0] - 1)
    vals = sorted_v[rank, np.arange(sorted_v.shape[1])]
    return np.where(counts > 0, vals, np.nan).astype(np.float32)


def _std_valid(R: np.ndarray, valid: np.ndarray, counts: np.ndarray) -> np.ndarray:
    c = np.maximum(counts.astype(np.float32), 1.0)
    x = np.where(valid, R, 0.0)
    mean = x.sum(axis=0) / c
    var = np.maximum((x * x).sum(axis=0) / c - mean * mean, 0.0)
    std = np.sqrt(var)
    std = np.where(counts > 1, std, 0.0)
    return std.astype(np.float32)


def temporal_features(R: np.ndarray, valid: np.ndarray, dx: float = DX, max_rows: int = 64,
                      n_bins: int = 0):
    """Признаки столбца по окну времени. R — остаток в °C, не z-score.

    Каналы: медиана, p20, p80, минимум медиан четвертей (устойчивость),
    наклон первой/последней четверти, std, вычитание скользящего среднего
    16/40/80 м. Все делятся на FEAT_SCALE_C.

    max_rows — сколько кадров равномерно берётся из всего окна. n_bins > 0
    добавляет медиану каждой доли времени: горб, который стоит всю запись,
    высок во всех долях, короткий всплеск — только в своей.
    """
    R = np.asarray(R, np.float32)
    valid = np.asarray(valid, bool) & np.isfinite(R)
    if R.shape[0] > max_rows:
        idx = np.linspace(0, R.shape[0] - 1, max_rows).astype(np.int64)
        R = R[idx]
        valid = valid[idx]
    rows = R.shape[0]
    sorted_v, counts = _sorted_valid(R, valid)
    med = _pick(sorted_v, counts, 50)
    p20 = _pick(sorted_v, counts, 20)
    p80 = _pick(sorted_v, counts, 80)
    std = _std_valid(R, valid, counts)
    nch = 4
    edges = np.linspace(0, rows, nch + 1).astype(np.int64)
    chunks = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            chunks.append(med)
            continue
        sv, ct = _sorted_valid(R[a:b], valid[a:b])
        chunks.append(_pick(sv, ct, 50))
    cm = np.stack(chunks, 0)
    finite = np.isfinite(cm)
    if finite.any():
        persist = np.where(finite, cm, np.inf).min(axis=0)
        persist = np.where(finite.any(axis=0), persist, np.nan)
    else:
        persist = np.full(cm.shape[1], np.nan, np.float32)
    slope = np.nan_to_num(cm[-1] - cm[0])
    good = np.isfinite(med) & (valid.mean(0) >= 0.2)
    med = np.where(np.isfinite(med), med, 0.0)
    # дыры не должны давать ложный край у скользящего среднего
    if not good.all():
        idx = np.flatnonzero(good)
        if len(idx):
            xs = np.arange(len(med))
            med = np.interp(xs, idx, med[idx]).astype(np.float32)
        else:
            med = np.zeros_like(med)
    p20 = np.nan_to_num(p20)
    p80 = np.nan_to_num(p80)
    persist = np.nan_to_num(persist)
    std = np.nan_to_num(std)
    hp16 = med - moving_average(med, int(round(16.0 / dx)))
    hp40 = med - moving_average(med, int(round(40.0 / dx)))
    hp80 = med - moving_average(med, int(round(80.0 / dx)))
    feats = np.stack([med, p20, p80, persist, slope, std, hp16, hp40, hp80], 0).astype(np.float32)
    feats /= np.float32(FEAT_SCALE_C)
    feats = np.nan_to_num(feats)
    feats[:, ~good] = 0.0
    if n_bins and int(n_bins) > 0:
        edges_b = np.linspace(0, rows, int(n_bins) + 1).astype(np.int64)
        bins = []
        for a, b in zip(edges_b[:-1], edges_b[1:]):
            if b <= a:
                bins.append(np.zeros(R.shape[1], np.float32))
                continue
            sv, ct = _sorted_valid(R[a:b], valid[a:b])
            bins.append(np.nan_to_num(_pick(sv, ct, 50)))
        extra = np.stack(bins, 0).astype(np.float32)
        extra /= np.float32(FEAT_SCALE_C)
        extra[:, ~good] = 0.0
        feats = np.concatenate([feats, extra], 0)
    return feats, good


def _fill_invalid(D: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if valid.all():
        return D
    out = np.array(D, np.float32, copy=True)
    xs = np.arange(D.shape[1])
    for i in range(D.shape[0]):
        m = valid[i]
        if m.all() or m.sum() < 2:
            continue
        out[i] = np.interp(xs, xs[m], D[i, m]).astype(np.float32)
    return out


def residual_from_temperature(T: np.ndarray, valid: np.ndarray, dx: float = DX,
                              trend_m: float = TREND_M, max_rows: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """T минус синфазная медиана строки минус пространственная медиана ~trend_m.

    Результат в градусах. На σ не делится. Статический профиль по времени
    не вычитается — иначе узкая точка, стоящая всё окно, исчезнет.
    """
    from scipy.ndimage import median_filter
    T = np.asarray(T, np.float32)
    valid = np.asarray(valid, bool) & np.isfinite(T)
    if T.shape[0] > max_rows:
        idx = np.linspace(0, T.shape[0] - 1, max_rows).astype(np.int64)
        T = T[idx]
        valid = valid[idx]
    masked = np.where(valid, T, np.nan)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cm = np.nanmedian(masked, axis=1)
    cm = np.where(np.isfinite(cm), cm, 0.0).astype(np.float32)
    D = np.where(valid, T - cm[:, None], 0.0).astype(np.float32)
    filled = _fill_invalid(D, valid)
    win = max(3, int(round(trend_m / dx)) | 1)
    bg = median_filter(filled, size=(1, win), mode="nearest")
    resid = (filled - bg) * valid.astype(np.float32)
    return resid.astype(np.float32), valid


def residual_from_cache(z) -> tuple[np.ndarray, np.ndarray]:
    """Канал 1 кэша — это dT_space, уже без крупномасштабного пространственного фона.

    X[1] обрезан в единицах σ и умножен обратно на σ: получаются градусы,
    а не z-score. Канал 0 (dT_time) здесь намеренно не используется.
    """
    sig = np.asarray(z["sigma"], np.float32)
    R = np.asarray(z["X"][1], np.float32) * sig[None, :]
    valid = np.asarray(z["valid"], bool)
    return R, valid


class SmoothReference(nn.Module):
    """Обучаемая опора. AvgPool шире нагретого участка (2–8 м),
    поэтому узкий горб в опору не помещается и остаётся в остатке.
    Последний слой с нуля: в начале опора равна нулю.
    """

    def __init__(self, cin: int, width: int = 16, pool: int = 80):
        super().__init__()
        self.pool = int(pool)
        self.enc = nn.Sequential(
            nn.Conv1d(cin, width, 5, padding=2, bias=False),
            nn.GroupNorm(4, width),
            nn.ReLU(inplace=True),
            nn.Conv1d(width, cin, 1),
        )
        nn.init.zeros_(self.enc[-1].weight)
        nn.init.zeros_(self.enc[-1].bias)

    def forward(self, x):
        p = self.pool
        w = x.shape[-1]
        pad = (p - (w % p)) % p
        xp = F.pad(x, (0, pad), mode="replicate") if pad else x
        low = F.avg_pool1d(xp, kernel_size=p, stride=p)
        y = self.enc(low)
        y = F.interpolate(y, size=xp.shape[-1], mode="linear", align_corners=False)
        return y[..., :w]


class ResBlock1d(nn.Module):
    def __init__(self, c: int, dil: int):
        super().__init__()
        self.c1 = nn.Conv1d(c, c, 3, padding=dil, dilation=dil, bias=False)
        self.n1 = nn.GroupNorm(4, c)
        self.c2 = nn.Conv1d(c, c, 3, padding=dil, dilation=dil, bias=False)
        self.n2 = nn.GroupNorm(4, c)

    def forward(self, x):
        h = F.relu(self.n1(self.c1(x)), inplace=True)
        h = self.n2(self.c2(h))
        return F.relu(x + h, inplace=True)


class StaticNet(nn.Module):
    """1D-сеть по x: вход — признаки профиля, внутри — опора и дилатации.

    Поле зрения дилатаций ~90 м, опора сглажена пулом `pool` пикселей
    (80 px × 0.25 м = 20 м > 6 м).
    """

    def __init__(self, cin: int = N_FEAT, width: int = 16, pool: int = 80, dilations=DILATIONS,
                 max_rows: int = 64, n_bins: int = 0):
        super().__init__()
        if width % 4:
            raise ValueError("width должен делиться на 4 (GroupNorm)")
        self.cin = cin
        self.max_rows = int(max_rows)
        self.n_bins = int(n_bins)
        self.ref = SmoothReference(cin, width=width, pool=pool)
        self.in_proj = nn.Conv1d(cin * 2, width, 5, padding=2, bias=False)
        self.in_n = nn.GroupNorm(4, width)
        self.blocks = nn.Sequential(*[ResBlock1d(width, int(d)) for d in dilations])
        self.head = nn.Conv1d(width, 1, 1)
        nn.init.constant_(self.head.bias, -4.0)

    def forward(self, x):
        ref = self.ref(x)
        h = torch.cat([x, x - ref], dim=1)
        h = F.relu(self.in_n(self.in_proj(h)), inplace=True)
        h = self.blocks(h)
        return self.head(h).squeeze(1)


def static_loss(logit: torch.Tensor, y: torch.Tensor, w: torch.Tensor):
    """Сбалансированный focal BCE. Хвост фона прижимается вниз, пик события — вверх."""
    p = torch.sigmoid(logit)
    bce = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
    pt = torch.where(y > 0.5, p, 1.0 - p)
    focal = (1.0 - pt).pow(2) * bce
    pos = (y > 0.5) & (w > 0)
    neg = (y < 0.5) & (w > 0)
    loss_pos = (focal[pos] * w[pos]).sum() / w[pos].sum().clamp(min=1.0) if int(pos.sum()) else logit.new_zeros(())
    loss_neg = (focal[neg] * w[neg]).sum() / w[neg].sum().clamp(min=1.0) if int(neg.sum()) else logit.new_zeros(())
    tail = logit.new_zeros(())
    if int(neg.sum()) > 32:
        pn = p[neg]
        k = max(1, int(0.01 * pn.numel()))
        top = torch.topk(pn, k).values
        tail = F.relu(top - 0.20).pow(2).mean()
    lift = F.relu(0.90 - p[pos]).pow(2).mean() if int(pos.sum()) else logit.new_zeros(())
    total = loss_pos + 0.35 * loss_neg + 3.0 * tail + 0.5 * lift
    return total, dict(pos=float(loss_pos.detach()), neg=float(loss_neg.detach()),
                       tail=float(tail.detach()), lift=float(lift.detach()))


@torch.no_grad()
def predict_map(model: StaticNet, R: np.ndarray, valid: np.ndarray) -> np.ndarray:
    model.eval()
    max_rows = int(getattr(model, "max_rows", 64))
    n_bins = int(getattr(model, "n_bins", 0))
    feats, good = temporal_features(R, valid, max_rows=max_rows, n_bins=n_bins)
    if feats.shape[0] != int(getattr(model, "cin", feats.shape[0])):
        raise RuntimeError(f"признаков {feats.shape[0]}, сеть ждёт {model.cin}")
    x = torch.from_numpy(np.ascontiguousarray(feats[None]))
    logit = model(x)
    p = torch.sigmoid(logit)[0].detach().numpy().astype(np.float32)
    p = p.copy()
    p[~good] = 0.0
    return p


def _window_starts(n: int, win: int, stride: int) -> list[int]:
    if n <= win:
        return [0]
    starts = list(range(0, n - win + 1, max(1, int(stride))))
    last = n - win
    if starts[-1] != last:
        starts.append(last)
    return starts


@torch.no_grad()
def predict_map_windows(model: StaticNet, R: np.ndarray, valid: np.ndarray,
                        win: int = INFER_TIME_WIN, overlap: float = INFER_TIME_OVERLAP) -> np.ndarray:
    """Скользящие окна по времени. Склейка — max. Короткая запись — один predict_map.

    Окно короче INFER_TIME_WIN на synth v8 val меняет столбцы выше порога FAR 1e-3
    и static recall. Перекрытие 1/2. По x окно не режется: нормировка GroupNorm
    на кропе 1280 не совпадает с полным проходом, по которому заморожен порог.
    """
    rows = int(np.asarray(R).shape[0])
    win = int(win)
    if win < 1 or rows <= win:
        return predict_map(model, R, valid)
    stride = max(1, int(win * (1.0 - float(overlap))))
    parts = [predict_map(model, R[s:s + win], valid[s:s + win]) for s in _window_starts(rows, win, stride)]
    return np.stack(parts, 0).max(axis=0).astype(np.float32)


def threshold_from_hist(hist: np.ndarray, far: float) -> float:
    """Квантиль FAR по гистограмме скоров на [0, 1). Совпадает с metrics.far_threshold до ширины бина."""
    hist = np.asarray(hist)
    total = int(hist.sum())
    if total <= 0:
        return 1.0
    k = int(np.ceil((1.0 - far) * total)) - 1
    k = int(np.clip(k, 0, total - 1))
    c = np.cumsum(hist)
    bin_idx = int(np.searchsorted(c, k + 1, side="left"))
    bin_idx = min(bin_idx, len(hist) - 1)
    return float(bin_idx / len(hist)) + 1e-12


def add_score_hist(hist: np.ndarray, scores: np.ndarray):
    if scores.size == 0:
        return
    n = len(hist)
    s = np.clip(np.asarray(scores, np.float64), 0.0, 1.0 - 1e-15)
    idx = (s * n).astype(np.int64)
    np.add.at(hist, idx, 1)


def fuse(score: np.ndarray, p: np.ndarray, mode: str = "max", scale: float = 1.0, gate: float = 0.0) -> np.ndarray:
    """Слияние с картой ансамбля. Результат не ниже score (кроме identity).

    max: maximum(score, scale * p), опционально p обнуляется ниже gate.
    blend: maximum(score, (1-scale)*score + scale*p) — доля scale тянет к p.
    identity: вернуть score.
    branch: только карта ветви, растянутая по времени. Ансамбль не участвует,
    поэтому результат может быть ниже score.
    """
    score = np.asarray(score, np.float32)
    if mode == "identity":
        return score
    pp = np.asarray(p, np.float32)
    if pp.ndim == 1:
        pp = pp[None, :]
    if pp.shape[-1] != score.shape[-1]:
        raise ValueError(f"width {pp.shape[-1]} != {score.shape[-1]}")
    if pp.shape[0] == 1 and score.shape[0] != 1:
        pp = np.broadcast_to(pp, score.shape)
    sp = np.array(pp, np.float32, copy=True)
    if gate and gate > 0:
        sp = np.where(pp >= float(gate), sp, 0.0)
    if mode == "branch":
        return np.float32(scale) * sp
    if mode == "max":
        return np.maximum(score, np.float32(scale) * sp)
    if mode == "blend":
        a = np.float32(scale)
        mixed = (1.0 - a) * score + a * sp
        return np.maximum(score, mixed)
    raise ValueError(mode)


# STATIC6. StaticNet выше не меняется: старые чекпойнты и dts.infer грузят его как раньше.
# Вход — температура без пространственного детренда. По времени — средние
# 30 мин, 2 ч и всего тайла. Опора — dilated 1D с полем зрения не меньше 200 м.
WIN_30MIN_ROWS = 15  # 30 мин при dt = 120 с на сетке, которую видит модель
WIN_2H_ROWS = 60     # 2 ч при том же шаге
N_FEAT6 = 5
REF_DILATIONS = (1, 2, 4, 8, 16, 32, 64, 73)  # сумма 200; два ядра 3 → +800 px
TRUNK_DILATIONS = REF_DILATIONS


def receptive_field_px(layers) -> int:
    """Поле зрения стека свёрток в пикселях.

    layers — пары (ядро, дилатация, шаг). Каждая свёртка добавляет
    (ядро − 1) * дилатация * накопленный шаг. Это не константа сети:
    тест считает поле по слоям, которые реально стоят в модуле.
    """
    rf = 1
    jump = 1
    for k, d, s in layers:
        rf += (int(k) - 1) * int(d) * jump
        jump *= int(s)
    return int(rf)


def _window_mean_minmax(R: np.ndarray, valid: np.ndarray, win: int):
    """Мин и макс скользящего среднего длины win. Короче окна — среднее тайла дважды."""
    rows, cols = R.shape
    x = np.where(valid, R, 0.0).astype(np.float64)
    c = valid.astype(np.float64)
    cs = np.concatenate([np.zeros((1, cols), np.float64), np.cumsum(x, axis=0)], 0)
    cc = np.concatenate([np.zeros((1, cols), np.float64), np.cumsum(c, axis=0)], 0)
    win = int(win)
    if rows <= win:
        n = cc[-1]
        mean = np.divide(cs[-1], np.maximum(n, 1.0), out=np.full(cols, np.nan), where=n > 0)
        mean = mean.astype(np.float32)
        return mean, mean.copy()
    s = cs[win:] - cs[:-win]
    n = cc[win:] - cc[:-win]
    ok = n >= 1.0
    means = np.divide(s, np.maximum(n, 1.0), out=np.full(s.shape, np.nan), where=ok)
    mask = np.isfinite(means)
    if not mask.any():
        z = np.full(cols, np.nan, np.float32)
        return z, z.copy()
    mx = np.where(mask, means, -np.inf).max(axis=0)
    mn = np.where(mask, means, np.inf).min(axis=0)
    bad = ~mask.any(axis=0)
    mx = np.where(bad, np.nan, mx).astype(np.float32)
    mn = np.where(bad, np.nan, mn).astype(np.float32)
    return mx, mn


def temperature_features(T: np.ndarray, valid: np.ndarray, dx: float = DX, max_rows: int = LONG_ROWS,
                         win_30: int = WIN_30MIN_ROWS, win_2h: int = WIN_2H_ROWS):
    """Признаки столбца по сырой температуре. Пространственный тренд не вычитается.

    Один скаляр на окно — медиана всех валидных отсчётов — снимает только
    абсолютный ноль прибора. Медианный фильтр по x и скользящее среднее по x
    здесь не вызываются: крупный ход кабеля остаётся в признаках, его должна
    убрать обучаемая опора.

    Если кадров больше max_rows, сначала равномерный linspace. Окна 15 и 60
    строк считаются уже на этой сетке. При dt = 120 с и длине не больше
    max_rows это 30 мин и 2 ч. На более длинной записи тот же linspace
    растягивает шаг, и те же 15/60 строк покрывают большую долю файла.
    Среднее тайла — среднее по видимой сетке.

    Каналы, делённые на FEAT_SCALE_C: среднее тайла, максимум и минимум
    среднего окна 30 мин, максимум и минимум среднего окна 2 ч.
    Столбец с долей валидных кадров ниже 20 % обнуляется.
    """
    R = np.asarray(T, np.float32)
    valid = np.asarray(valid, bool) & np.isfinite(R)
    if R.shape[0] > max_rows:
        idx = np.linspace(0, R.shape[0] - 1, int(max_rows)).astype(np.int64)
        R = R[idx]
        valid = valid[idx]
    rows, cols = R.shape
    seen = R[valid]
    scalar = float(np.median(seen)) if seen.size else 0.0
    centered = np.where(valid, R - np.float32(scalar), 0.0).astype(np.float32)
    counts = valid.sum(axis=0).astype(np.float32)
    tile = np.divide(centered.sum(axis=0), np.maximum(counts, 1.0),
                     out=np.zeros(cols, np.float32), where=counts > 0)
    mx30, mn30 = _window_mean_minmax(centered, valid, int(win_30))
    mx2h, mn2h = _window_mean_minmax(centered, valid, int(win_2h))
    good = (counts >= max(1.0, 0.2 * rows)) & np.isfinite(tile)
    feats = np.stack([
        np.nan_to_num(tile),
        np.nan_to_num(mx30),
        np.nan_to_num(mn30),
        np.nan_to_num(mx2h),
        np.nan_to_num(mn2h),
    ], 0).astype(np.float32)
    feats /= np.float32(FEAT_SCALE_C)
    feats = np.nan_to_num(feats)
    feats[:, ~good] = 0.0
    return feats, good


class DilatedReference(nn.Module):
    """Обучаемая пространственная опора. Dilated 1D, поле зрения ≥ 200 м.

    Последний слой с нуля: на старте опора равна нулю, узкий горб в неё
    не заложен. Это не средний пул StaticNet.
    """

    def __init__(self, cin: int, width: int = 16, dilations=REF_DILATIONS):
        super().__init__()
        if width % 4:
            raise ValueError("width должен делиться на 4 (GroupNorm)")
        self.stem = nn.Sequential(
            nn.Conv1d(cin, width, 3, padding=1, bias=False),
            nn.GroupNorm(4, width),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResBlock1d(width, int(d)) for d in dilations])
        self.out = nn.Conv1d(width, cin, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        layers = [(3, 1, 1)]
        for d in dilations:
            layers.append((3, int(d), 1))
            layers.append((3, int(d), 1))
        self.layers = tuple(layers)

    def forward(self, x):
        return self.out(self.blocks(self.stem(x)))


class StaticNet6(nn.Module):
    """1D-сеть STATIC6: сырые многомасштабные средние и dilated-опора ≥ 200 м.

    StaticNet не заменяется. Выход по-прежнему один логит на столбец.
    """

    def __init__(self, cin: int = N_FEAT6, width: int = 16, max_rows: int = LONG_ROWS,
                 win_30: int = WIN_30MIN_ROWS, win_2h: int = WIN_2H_ROWS,
                 ref_dilations=REF_DILATIONS, trunk_dilations=TRUNK_DILATIONS):
        super().__init__()
        if width % 4:
            raise ValueError("width должен делиться на 4 (GroupNorm)")
        self.cin = int(cin)
        self.max_rows = int(max_rows)
        self.win_30 = int(win_30)
        self.win_2h = int(win_2h)
        self.n_bins = 0
        self.ref = DilatedReference(cin, width=width, dilations=ref_dilations)
        self.in_proj = nn.Conv1d(cin * 2, width, 5, padding=2, bias=False)
        self.in_n = nn.GroupNorm(4, width)
        self.blocks = nn.Sequential(*[ResBlock1d(width, int(d)) for d in trunk_dilations])
        self.head = nn.Conv1d(width, 1, 1)
        nn.init.constant_(self.head.bias, -4.0)
        trunk = [(5, 1, 1)]
        for d in trunk_dilations:
            trunk.append((3, int(d), 1))
            trunk.append((3, int(d), 1))
        self.trunk_layers = tuple(trunk)

    def forward(self, x):
        ref = self.ref(x)
        h = torch.cat([x, x - ref], dim=1)
        h = F.relu(self.in_n(self.in_proj(h)), inplace=True)
        h = self.blocks(h)
        return self.head(h).squeeze(1)


@torch.no_grad()
def predict_temperature(model: StaticNet6, T: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Вероятность столбца по сырой температуре. Пространственный медианный фильтр не вызывается."""
    model.eval()
    feats, good = temperature_features(
        T, valid, max_rows=int(model.max_rows), win_30=int(model.win_30), win_2h=int(model.win_2h),
    )
    if feats.shape[0] != int(model.cin):
        raise RuntimeError(f"признаков {feats.shape[0]}, сеть ждёт {model.cin}")
    x = torch.from_numpy(np.ascontiguousarray(feats[None]))
    logit = model(x)
    p = torch.sigmoid(logit)[0].detach().numpy().astype(np.float32)
    p = p.copy()
    p[~good] = 0.0
    return p


# Входы гейта: сводка s победителя, две карты ветви, локальная форма горба.
# Индексы temporal_features: 0 медиана, 3 persist, 4 наклон, 5 std, 6/7 hp 16/40 м.
GATE_LOC_IDX = (0, 3, 4, 5, 6, 7)
GATE_CIN = 7 + len(GATE_LOC_IDX) + 1  # + минимум длинных долей времени


def score_column_stats(s: np.ndarray, valid: np.ndarray, max_rows: int = 64):
    """Медиана, 90-й перцентиль и максимум карты s по времени. Не z-score."""
    s = np.asarray(s, np.float32)
    valid = np.asarray(valid, bool) & np.isfinite(s)
    if s.shape[0] > max_rows:
        idx = np.linspace(0, s.shape[0] - 1, max_rows).astype(np.int64)
        s = s[idx]
        valid = valid[idx]
    sorted_v, counts = _sorted_valid(s, valid)
    med = np.nan_to_num(_pick(sorted_v, counts, 50))
    p90 = np.nan_to_num(_pick(sorted_v, counts, 90))
    mx = np.where(valid, s, -np.inf).max(axis=0)
    mx = np.where(np.isfinite(mx), mx, 0.0).astype(np.float32)
    good = counts >= max(1, int(0.2 * s.shape[0]))
    med = np.where(good, med, 0.0).astype(np.float32)
    p90 = np.where(good, p90, 0.0).astype(np.float32)
    mx = np.where(good, mx, 0.0).astype(np.float32)
    return med, p90, mx


def gate_channels(s: np.ndarray, valid: np.ndarray, p1: np.ndarray, p2: np.ndarray,
                  feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Столбцы для гейта и средняя вероятность двух сетей.

    Каналы: s_med, s_p90, s_max, p1, p2, среднее, |p1-p2|,
    медиана остатка, persist, наклон, std, hp16, hp40, минимум длинного окна.
    Вероятности и доли — как есть. Пространственные каналы уже поделены на 10 °C.
    """
    p1 = np.asarray(p1, np.float32)
    p2 = np.asarray(p2, np.float32)
    med_s, p90, mx = score_column_stats(s, valid)
    mean = (0.5 * (p1 + p2)).astype(np.float32)
    ad = np.abs(p1 - p2).astype(np.float32)
    loc = np.stack([feats[i] for i in GATE_LOC_IDX], 0).astype(np.float32)
    if feats.shape[0] > N_FEAT:
        long_min = np.min(feats[N_FEAT:], axis=0).astype(np.float32)
    else:
        long_min = np.asarray(feats[3], np.float32)
    x = np.concatenate([
        np.stack([med_s, p90, mx, p1, p2, mean, ad], 0),
        loc,
        long_min[None, :],
    ], 0).astype(np.float32)
    x = np.nan_to_num(x)
    if x.shape[0] != GATE_CIN:
        raise RuntimeError(f"гейт ждёт {GATE_CIN} каналов, собрано {x.shape[0]}")
    return x, mean


class StaticGate(nn.Module):
    """Маленькая 1D-сеть по x. Выход — вероятность открыть статичную ветвь.

    Голова с нулевым весом и смещением −4: в начале гейт почти закрыт
    и слияние совпадает с картой победителя.
    """

    def __init__(self, cin: int = GATE_CIN, width: int = 16):
        super().__init__()
        if width % 4:
            raise ValueError("width должен делиться на 4 (GroupNorm)")
        self.cin = cin
        self.proj = nn.Conv1d(cin, width, 5, padding=2, bias=False)
        self.pn = nn.GroupNorm(4, width)
        self.blocks = nn.Sequential(
            ResBlock1d(width, 1),
            ResBlock1d(width, 2),
            ResBlock1d(width, 4),
            ResBlock1d(width, 8),
        )
        self.head = nn.Conv1d(width, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, -4.0)

    def forward(self, x):
        h = F.relu(self.pn(self.proj(x)), inplace=True)
        h = self.blocks(h)
        return torch.sigmoid(self.head(h).squeeze(1))


def fuse_gated(score: np.ndarray, p: np.ndarray, g: np.ndarray) -> np.ndarray:
    """max(s, g * p). Карту победителя не опускает. g и p — по столбцу или покадрово."""
    score = np.asarray(score, np.float32)
    pp = np.asarray(p, np.float32)
    gg = np.asarray(g, np.float32)
    if pp.ndim == 1:
        pp = pp[None, :]
    if gg.ndim == 1:
        gg = gg[None, :]
    if pp.shape[-1] != score.shape[-1] or gg.shape[-1] != score.shape[-1]:
        raise ValueError("width gate/p/score")
    if pp.shape[0] == 1 and score.shape[0] != 1:
        pp = np.broadcast_to(pp, score.shape)
    if gg.shape[0] == 1 and score.shape[0] != 1:
        gg = np.broadcast_to(gg, score.shape)
    return np.maximum(score, gg * pp)


def gate_loss(g: torch.Tensor, y: torch.Tensor, w: torch.Tensor, p: torch.Tensor):
    """Гейт открывается там, где ветвь уже уверена и столбец — горячая точка.

    На фоне с высокой p гейт держит g*p ниже 0.78, то есть под порогом FAR победителя.
    Столбцы с маленькой p на порог не влияют, их вес нулевой.
    """
    bce = F.binary_cross_entropy(g.clamp(1e-4, 1 - 1e-4), y, reduction="none")
    pt = torch.where(y > 0.5, g, 1.0 - g)
    focal = (1.0 - pt).pow(2) * bce
    active = w * (p > 0.40).float()
    pos = (y > 0.5) & (active > 0)
    neg = (y < 0.5) & (active > 0)
    zero = g.sum() * 0.0
    loss_pos = (focal[pos] * active[pos]).sum() / active[pos].sum().clamp(min=1.0) if int(pos.sum()) else zero
    loss_neg = (focal[neg] * active[neg]).sum() / active[neg].sum().clamp(min=1.0) if int(neg.sum()) else zero
    opened = g * p
    tail = F.relu(opened[neg] - 0.78).pow(2).mean() if int(neg.sum()) else zero
    lift = F.relu(p[pos].clamp(max=0.93) - opened[pos]).pow(2).mean() if int(pos.sum()) else zero
    total = loss_pos + 0.5 * loss_neg + 4.0 * tail + 0.5 * lift
    return total, dict(pos=float(loss_pos.detach()), neg=float(loss_neg.detach()),
                       tail=float(tail.detach()), lift=float(lift.detach()))


@torch.no_grad()
def predict_gate(gate: StaticGate, channels: np.ndarray) -> np.ndarray:
    gate.eval()
    x = torch.from_numpy(np.ascontiguousarray(channels[None]))
    return gate(x)[0].detach().numpy().astype(np.float32)
