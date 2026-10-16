"""Конвейер realnet: U-Net с синтетики и отдельный скор базовой линии.

Голова сегментации предобученного DTSUNet не меняется. Порог и веса скора
по реальным карточкам не подбираются.

Вход сети — те же три канала, на которых её учили (временной контраст,
пространственный контраст, valid). Вычитание собственной базовой линии
(профиль по x, гармоника суток и линейный тренд) в эти каналы не подставляется:
на v8 val такая замена роняет IoU. Остаток базовой линии идёт отдельным скором,
и val может дать ему нулевой вес.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter, median_filter

from dts import events as E
from dts import features
from dts import metrics
from dts import models

CLIP = 12.0
SPACE_WIN_M = 40.0
# Дырка маски: 8 строк (16 мин) × 16 точек (4 м). Ядро головы чуть шире дырки,
# чтобы с замороженного энкодера видеть контекст сразу за её краем.
HOLE_T, HOLE_X = 8, 16
PITCH_T, PITCH_X = 16, 32
MASK_KT, MASK_KX = 11, 21
# Сжатие робастного z в [0, 1]. Константы фиксированы, не по val и не по карточкам.
Z_FLOOR = 2.5
Z_SPAN = 6.0
WIDE_SIGMA = (4.0, 48.0)   # ~8 мин × 12 м: внутренность плато остаётся, спекл гаснет
MASK_SIGMA = (2.0, 6.0)
HUBER_DELTA = 1.5
MODEL_VERSION = "realnet-0.1"
NON_ALARM = ("HVAC_Diurnal", "Sensor_Artifact")


def n_harmonics_for(t) -> tuple[int, float]:
    """Сутки не идентифицируются на записи короче 16 ч; вторая гармоника — от 36 ч."""
    t = np.asarray(t, np.float64)
    if t.size < 2:
        return 0, 0.0
    span_h = float(t[-1] - t[0]) / 3600.0
    if span_h < 16.0:
        return 0, span_h
    if span_h < 36.0:
        return 1, span_h
    return 2, span_h


def design_matrix(t, n_harm: int) -> np.ndarray:
    """Столбцы: 1, линейный тренд, cos/sin суток и второй гармоники."""
    t = np.asarray(t, np.float64)
    span = max(float(t[-1] - t[0]), 1.0) if t.size else 1.0
    tau = (t - (float(t.mean()) if t.size else 0.0)) / span
    cols = [np.ones_like(tau), tau]
    t0 = float(t[0]) if t.size else 0.0
    for k in range(1, int(n_harm) + 1):
        ang = 2.0 * np.pi * k * (t - t0) / 86400.0
        cols.append(np.cos(ang))
        cols.append(np.sin(ang))
    return np.column_stack(cols)


def _irls(Phi, Y, valid, delta, n_iter=4, min_count=12):
    """Робастный МНК сразу по всем столбцам. delta — скаляр или вектор длины N.

    Веса Хьюбера: очаг короче записи не утаскивает гармонику и тренд.
    """
    Phi = np.asarray(Phi, np.float64)
    valid = np.asarray(valid, bool)
    T, K = Phi.shape
    N = int(Y.shape[1]) if np.ndim(Y) == 2 else 0
    if N == 0 or T == 0 or K == 0:
        return np.zeros((T, N), np.float32)
    Y0 = np.where(valid, np.asarray(Y, np.float64), 0.0)
    W = valid.astype(np.float64)
    active = W.sum(0) >= min_count
    if np.ndim(delta) == 0:
        delta = np.full(N, float(delta), np.float64)
    else:
        delta = np.asarray(delta, np.float64).reshape(N)
    delta = np.maximum(delta, 1e-3)
    beta = np.zeros((K, N), np.float64)
    eye = np.eye(K, dtype=np.float64)
    for _ in range(int(n_iter)):
        G = np.zeros((K, K, N), np.float64)
        rhs = np.zeros((K, N), np.float64)
        WY = W * Y0
        for a in range(K):
            pa = Phi[:, a]
            rhs[a] = pa @ WY
            for b in range(a, K):
                g = (pa * Phi[:, b]) @ W
                G[a, b] = g
                if b != a:
                    G[b, a] = g
        Gs = np.moveaxis(G, 2, 0)
        tr = np.trace(Gs, axis1=1, axis2=2) / max(K, 1)
        Gs = Gs + eye * (1e-3 * np.maximum(tr, 1e-6))[:, None, None]
        rh = np.moveaxis(rhs, 0, 1)[..., None]
        try:
            solved = np.linalg.solve(Gs, rh)[..., 0]
        except np.linalg.LinAlgError:
            solved = np.zeros((N, K), np.float64)
            flat = rh[..., 0]
            for i in range(N):
                solved[i] = np.linalg.lstsq(Gs[i], flat[i], rcond=None)[0]
        solved = np.where(np.isfinite(solved), solved, 0.0)
        solved[~active] = 0.0
        beta = solved.T
        pred = Phi @ beta
        resid = np.where(valid, Y0 - pred, 0.0)
        abs_r = np.abs(resid)
        d = delta[None, :]
        w = np.where(abs_r > d, d / np.maximum(abs_r, 1e-8), 1.0)
        W = np.where(valid, w, 0.0)
    pred = np.where(valid, Phi @ beta, 0.0)
    return pred.astype(np.float32)


def fit_baseline(T, valid, t, x, dx: float = 0.25, stride: int = 8):
    """Базовая линия B(t, x) = профиль(x) + общая гармоника(t) + локальный тренд(t, x).

    Профиль — медиана по времени. Общая часть — робастная регрессия построчной
    медианы (то, что синфазно на большей части кабеля). Локальная часть — та же
    регрессия по столбцам с шагом stride, коэффициенты сглажены по x.
    Короткое плато в остатке R = T − B, даже если оно шире 40 м.
    """
    T = np.asarray(T, np.float32)
    valid = np.asarray(valid, bool) & np.isfinite(T)
    n_t, n_x = T.shape
    t = np.asarray(t, np.float64)
    x = np.asarray(x, np.float64)
    if t.size != n_t:
        t = np.arange(n_t, dtype=np.float64) * 120.0
    n_harm, span_h = n_harmonics_for(t)
    Phi = design_matrix(t, n_harm)
    work = np.where(valid, T, np.nan)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prof = np.nanmedian(work, axis=0)
            D = work - prof[None, :]
            cm = np.nanmedian(D, axis=1)
    prof = np.where(np.isfinite(prof), prof, 0.0).astype(np.float64)
    cm = np.where(np.isfinite(cm), cm, 0.0).astype(np.float64)
    # масштаб Хьюбера по временным разностям, в градусах
    sig = features.robust_sigma_x(np.where(valid, T, np.nan).astype(np.float32), valid)
    delta_cm = float(np.median(sig) * 2.0)
    row_ok = valid.any(1)
    cm_hat = _irls(Phi, cm.reshape(n_t, 1), row_ok.reshape(n_t, 1), delta_cm, min_count=8)[:, 0]
    D2 = np.where(valid, np.nan_to_num(D) - cm_hat[:, None], 0.0)
    stride = max(int(stride), 1)
    idx = np.arange(0, n_x, stride)
    if idx.size == 0:
        idx = np.array([0], np.int64)
    delta_x = np.maximum(sig[idx] * 2.0, 0.05)
    local = _irls(Phi, D2[:, idx], valid[:, idx], delta_x)
    # интерполяция коэффициентов не нужна: раскладываем прогноз по столбцам idx и тянем по x
    pred_cols = local.astype(np.float64)
    full = np.zeros((n_t, n_x), np.float64)
    src = idx.astype(np.float64)
    grid = np.arange(n_x, dtype=np.float64)
    for i in range(n_t):
        full[i] = np.interp(grid, src, pred_cols[i])
    # сглаживание по x убирает ступеньку шага stride, не трогая ход по времени
    if n_x >= 3:
        k = max(3, (stride | 1))
        full = median_filter(full, size=(1, k), mode="nearest")
    B = prof[None, :] + cm_hat[:, None] + full
    R = np.where(valid, np.asarray(T, np.float64) - B, 0.0).astype(np.float32)
    info = dict(n_harm=int(n_harm), span_h=float(span_h), profile=prof.astype(np.float32),
                sigma_x=sig.astype(np.float32), dx=float(dx), x0=float(x[0]) if x.size else 0.0)
    return B.astype(np.float32), R, info


def _spatial_background(R, valid, dx: float) -> np.ndarray:
    """Медиана по x окном ~40 м, как канал 1 в make_channels. Внутри широкого плато ≈ само плато."""
    win = max(3, int(round(SPACE_WIN_M / max(dx, 1e-6))) | 1)
    step = 4
    filled = np.where(valid, R, 0.0).astype(np.float32, copy=False)
    k = max(3, (win // step) | 1)
    sub = median_filter(filled[:, ::step], size=(1, k), mode="nearest")
    n_x = R.shape[1]
    xs = np.arange(n_x, dtype=np.float64)
    src = xs[::step]
    bg = np.empty_like(filled)
    for i in range(R.shape[0]):
        bg[i] = np.interp(xs, src, sub[i])
    return bg


def make_channels(T, valid, t, x, dx: float = 0.25):
    """Остаток базовой линии в раскладке каналов U-Net. На вход сети не подаётся.

    Абляция v8 val: подстановка этих каналов вместо кэша роняет IoU.
    Канал 0 — остаток в единицах σ(x), широкая зона здесь видна.
    Канал 1 — тот же остаток минус медиана 40 м.
    Официальный скор берёт info['residual'] только как p_wide.
    """
    _B, R, info = fit_baseline(T, valid, t, x, dx=dx)
    valid = np.asarray(valid, bool) & np.isfinite(np.asarray(T, np.float32))
    sig = info["sigma_x"]
    sig_safe = np.maximum(sig, 0.03).astype(np.float32)
    rs = np.clip(R / sig_safe[None, :], -CLIP, CLIP)
    rs = np.where(valid, rs, 0.0).astype(np.float32)
    bg = _spatial_background(R, valid, dx)
    sp = np.clip((R - bg) / sig_safe[None, :], -CLIP, CLIP)
    sp = np.where(valid, sp, 0.0).astype(np.float32)
    X = np.stack([rs, sp, valid.astype(np.float32)]).astype(np.float32)
    info["residual"] = R
    return X, info


def smooth_valid(field, valid, sigma) -> np.ndarray:
    valid = np.asarray(valid, bool)
    w = gaussian_filter(valid.astype(np.float32), sigma, mode="nearest")
    num = gaussian_filter(np.where(valid, np.asarray(field, np.float32), 0.0), sigma, mode="nearest")
    out = num / np.maximum(w, 1e-3)
    return np.where(valid, out, 0.0).astype(np.float32)


def robust_z(field, valid) -> np.ndarray:
    """|поле| / (1.4826 · MAD) по валидным клеткам кадра. Один масштаб на кадр."""
    valid = np.asarray(valid, bool)
    r = np.where(valid, np.asarray(field, np.float64), np.nan)
    if not np.isfinite(r).any():
        return np.zeros(field.shape, np.float32)
    med = np.nanmedian(r)
    mad = np.nanmedian(np.abs(r - med))
    scale = max(float(1.4826 * mad), 1e-3)
    z = np.abs(np.nan_to_num(np.asarray(field, np.float64), nan=0.0)) / scale
    return np.where(valid, z, 0.0).astype(np.float32)


def squash_z(z) -> np.ndarray:
    return np.clip((np.asarray(z, np.float32) - Z_FLOOR) / Z_SPAN, 0.0, 1.0).astype(np.float32)


def component_score(signed, valid, sigma) -> np.ndarray:
    sm = smooth_valid(signed, valid, sigma)
    return np.where(valid, squash_z(robust_z(sm, valid)), 0.0).astype(np.float32)


def combine_scores(p_seg, p_mask, p_wide, weights) -> np.ndarray:
    ws, wm, ww = (float(v) for v in weights)
    score = ws * np.asarray(p_seg, np.float32) + wm * np.asarray(p_mask, np.float32) + ww * np.asarray(p_wide, np.float32)
    return score.astype(np.float32)


def valid_far_threshold(score, valid, far: float = 1e-3) -> float:
    """Порог доли far по валидным пикселям одного файла. Метки и карточки не читаются."""
    valid = np.asarray(valid, bool)
    values = np.asarray(score, np.float32)[valid]
    if values.size == 0:
        return 1.0
    return float(metrics.far_threshold(values, far))


def weight_grid():
    """Симплекс с шагом 1/4. Порядок фиксирован: при равном IoU берём больше веса головы."""
    vals = (0.0, 0.25, 0.5, 0.75, 1.0)
    out = []
    for ws in vals:
        for wm in vals:
            ww = round(1.0 - ws - wm, 2)
            if ww < -1e-9 or ww - 1e-9 > 1.0:
                continue
            if abs(ww * 4 - round(ww * 4)) > 1e-6:
                continue
            out.append((ws, wm, ww))
    out.sort(key=lambda w: (-w[0], -w[2], -w[1]))
    return out


def holdout_ids(names, held) -> list:
    """Имена файлов адаптации: все, кроме оцениваемого."""
    held = str(held)
    return [str(n) for n in names if str(n) != held]


class MaskHead(nn.Module):
    """Прогноз канала 0 по кольцу признаков сразу за дыркой 8×16.

    Дилатация (5, 9) и ядро 3×3 достают до первой клетки снаружи дырки.
    Центральный отвод заморожен нулём: свою клетку голова не читает.
    Полное ядро 11×21 на всём кадре слишком дорого для двух потоков.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.ring = nn.Conv2d(channels, channels, 3, dilation=(5, 9), padding=(5, 9), bias=False)
        self.act = nn.ReLU(inplace=True)
        self.out = nn.Conv2d(channels, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        with torch.no_grad():
            self.ring.weight[:, :, 1, 1] = 0
        self.ring.weight.register_hook(self._drop_center_grad)

    @staticmethod
    def _drop_center_grad(grad):
        grad = grad.clone()
        grad[:, :, 1, 1] = 0
        return grad

    def forward(self, d1):
        self.ring.weight.data[:, :, 1, 1] = 0
        return self.out(self.act(self.ring(d1)))


class RealNet(nn.Module):
    """Замороженный U-Net (голова сегментации на месте) и отдельная маскированная голова."""

    def __init__(self, unet: nn.Module):
        super().__init__()
        self.unet = unet
        channels = int(unet.head.weight.shape[1])
        self.mask_head = MaskHead(channels)
        self.freeze_unet()

    def freeze_unet(self):
        self.unet.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)

    def decoder_features(self, x: torch.Tensor) -> torch.Tensor:
        u = self.unet
        e1 = u.e1(x)
        e2 = u.e2(F.max_pool2d(e1, 2))
        e3 = u.e3(F.max_pool2d(e2, 2))
        e4 = u.gate(u.e4(F.max_pool2d(e3, 2)))
        d3 = u.d3(torch.cat([u.u3(e4), e3], 1))
        d2 = u.d2(torch.cat([u.u2(d3), e2], 1))
        d1 = u.d1(torch.cat([u.u1(d2), e1], 1))
        if d1.shape[-2:] != x.shape[-2:]:
            d1 = F.interpolate(d1, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return d1

    def mask_pred(self, x: torch.Tensor) -> torch.Tensor:
        return self.mask_head(self.decoder_features(x))


def segmentation_head_tensors(unet: nn.Module):
    """Веса головы, которые адаптация обязана вернуть байт-в-байт."""
    head = unet.head
    return head.weight.detach().clone(), head.bias.detach().clone()


def load_pretrained(path, width: int = 16) -> RealNet:
    unet = models.DTSUNet(w=width)
    state = torch.load(path, map_location="cpu", weights_only=True)
    unet.load_state_dict(state)
    net = RealNet(unet)
    net.eval()
    return net


def adapt_batchnorm(unet: nn.Module, tiles: list[np.ndarray], batch: int = 8) -> None:
    """AdaBN: заново накопить running mean/var. γ, β и свёртки не шагаются.

    momentum=None — накопление среднего по всем батчам адаптации, не экспонента
    с короткой памятью. После прохода сеть в eval, голова та же.
    """
    if not tiles:
        unet.eval()
        return
    unet.train()
    for mod in unet.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.reset_running_stats()
            mod.momentum = None
    with torch.no_grad():
        buf = []
        for tile in tiles:
            buf.append(torch.from_numpy(np.ascontiguousarray(tile, np.float32)))
            if len(buf) == batch:
                unet(torch.stack(buf))
                buf = []
        if buf:
            unet(torch.stack(buf))
    for mod in unet.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.momentum = 0.1
    unet.eval()


def _tile_starts(n: int, tile: int, overlap: int) -> list[int]:
    if n <= tile:
        return [0]
    step = max(tile - 2 * overlap, 1)
    starts = list(range(0, n - tile + 1, step))
    if starts[-1] != n - tile:
        starts.append(n - tile)
    return starts


def apply_hole_mask(x: torch.Tensor, t0: int, w0: int, hole_t: int = HOLE_T, hole_x: int = HOLE_X):
    """Занулить каналы 0–1 и valid в прямоугольнике. Исходный тензор не портится."""
    out = x.clone()
    t1 = min(t0 + hole_t, out.shape[-2])
    w1 = min(w0 + hole_x, out.shape[-1])
    out[..., :3, t0:t1, w0:w1] = 0
    # valid тоже 0: для предобученной сети это пропуск, а не настоящий ноль градусов
    return out


def _hole_starts(n: int, off: int, hole: int, pitch: int) -> list[int]:
    """Начала дырок одной фазы. Хвост ряда закрывает фаза, чья решётка ближе к хвосту."""
    if n <= 0:
        return []
    if n <= hole:
        return [0] if off == 0 else []
    starts = list(range(int(off), n - hole + 1, int(pitch)))
    covered = (starts[-1] + hole) if starts else 0
    if covered < n:
        tail = n - hole
        dist = (tail - int(off)) % int(pitch)
        if tail not in starts and dist <= pitch // 2:
            starts.append(int(tail))
    if not starts and off == 0:
        return [0]
    return starts


def phase_batch(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Четыре сдвига решётки дырок. Клетка кадра закрыта хотя бы одной фазой и в этой фазе не видна сети.

    x: (1, 3, H, W) или (3, H, W).
    Возвращает батч входов и маску дырок (P, 1, H, W).
    """
    if x.dim() == 3:
        x = x[None]
    _, _, H, W = x.shape
    masked, masks = [], []
    for t_off in (0, HOLE_T):
        for w_off in (0, HOLE_X):
            m = torch.zeros(1, 1, H, W)
            y = x.clone()
            for t0 in _hole_starts(H, t_off, HOLE_T, PITCH_T):
                for w0 in _hole_starts(W, w_off, HOLE_X, PITCH_X):
                    y[:, :3, t0:t0 + HOLE_T, w0:w0 + HOLE_X] = 0
                    m[:, :, t0:t0 + HOLE_T, w0:w0 + HOLE_X] = 1
            masked.append(y)
            masks.append(m)
    return torch.cat(masked, 0), torch.cat(masks, 0)


def random_holes(x: torch.Tensor, rng: np.random.Generator, n_holes: int = 3):
    """Случайные дырки для обучения. x: (B, 3, H, W)."""
    B, _, H, W = x.shape
    y = x.clone()
    m = torch.zeros(B, 1, H, W)
    if H < HOLE_T or W < HOLE_X:
        return y, m
    for b in range(B):
        for _ in range(n_holes):
            t0 = int(rng.integers(0, H - HOLE_T + 1))
            w0 = int(rng.integers(0, W - HOLE_X + 1))
            y[b, :3, t0:t0 + HOLE_T, w0:w0 + HOLE_X] = 0
            m[b, 0, t0:t0 + HOLE_T, w0:w0 + HOLE_X] = 1
    return y, m


def huber_weighted(pred, target, weight, delta: float = HUBER_DELTA):
    w = weight.to(dtype=pred.dtype)
    r = pred - target
    a = r.abs()
    hub = torch.where(a <= delta, 0.5 * r * r, delta * (a - 0.5 * delta))
    return (hub * w).sum() / w.sum().clamp(min=1.0)


def sample_tile(X: np.ndarray, rng: np.random.Generator, weight: np.ndarray | None = None,
                th: int = 128, tw: int = 256, tries: int = 6):
    """Кроп, по возможности из фона (большой средний weight) и с размером, кратным 8."""
    _c, T, W = X.shape
    th = min(th, T)
    tw = min(tw, W)
    th -= th % 8
    tw -= tw % 8
    if th < 16 or tw < 32:
        th, tw = T - T % 8, W - W % 8
    best, best_s = None, -1.0
    for _ in range(max(tries, 1)):
        t0 = 0 if T == th else int(rng.integers(0, T - th + 1))
        w0 = 0 if W == tw else int(rng.integers(0, W - tw + 1))
        if weight is None:
            score = 1.0
        else:
            score = float(weight[t0:t0 + th, w0:w0 + tw].mean())
        if score > best_s:
            best_s = score
            best = (t0, w0)
    t0, w0 = best
    tile = np.array(X[:, t0:t0 + th, w0:w0 + tw], dtype=np.float32, copy=True)
    bg = None if weight is None else np.array(weight[t0:t0 + th, w0:w0 + tw], dtype=np.float32, copy=True)
    if rng.random() < 0.5:
        tile = tile[:, :, ::-1].copy()
        if bg is not None:
            bg = bg[:, ::-1].copy()
    return tile, bg


def train_mask_head(model: RealNet, tiles: list[np.ndarray], weights: list[np.ndarray] | None = None,
                    minutes: float = 2.0, seed: int = 0, batch: int = 8, lr: float = 1e-3,
                    max_steps: int = 10**9) -> list:
    """Учится только MaskHead. U-Net в eval: BN-статистики AdaBN не плывут, голова сегментации заморожена."""
    if not tiles:
        model.eval()
        return []
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    head_w, head_b = segmentation_head_tensors(model.unet)
    model.freeze_unet()
    model.unet.eval()
    model.mask_head.train()
    opt = torch.optim.AdamW(model.mask_head.parameters(), lr=lr, weight_decay=1e-4)
    hist = []
    t0 = __import__("time").time()
    budget = float(minutes) * 60.0
    step = 0
    n = len(tiles)
    while step < max_steps and (__import__("time").time() - t0) < budget:
        pick = rng.integers(0, n, size=min(batch, n))
        xb = np.stack([tiles[int(i)] for i in pick])
        xt = torch.from_numpy(xb)
        masked, m = random_holes(xt, rng)
        w = m * (xt[:, 2:3] > 0.5).float()
        if weights is not None:
            wb = np.stack([weights[int(i)] for i in pick])
            w = w * torch.from_numpy(wb)[:, None]
        model.unet.eval()
        model.mask_head.train()
        opt.zero_grad(set_to_none=True)
        pred = model.mask_pred(masked)
        loss = huber_weighted(pred, xt[:, 0:1], w)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.mask_head.parameters(), 1.0)
        opt.step()
        step += 1
        if step == 1 or step % 20 == 0:
            hist.append(dict(step=step, loss=round(float(loss.detach()), 5),
                             sec=round(__import__("time").time() - t0, 1)))
            print(f"mask step {step} loss {float(loss.detach()):.4f}", flush=True)
    model.mask_head.eval()
    model.eval()
    w2, b2 = segmentation_head_tensors(model.unet)
    if not torch.equal(head_w, w2) or not torch.equal(head_b, b2):
        raise RuntimeError("голова сегментации изменилась во время обучения маски")
    hist.append(dict(step=step, sec=round(__import__("time").time() - t0, 1), final=True))
    return hist


def _lattice_mask(n_t: int, n_x: int, t_off: int, w_off: int) -> np.ndarray:
    """Декартово произведение дырок фазы. Хвост берётся из тех же начал, что и phase_batch."""
    rows = np.zeros(n_t, dtype=bool)
    cols = np.zeros(n_x, dtype=bool)
    for t0 in _hole_starts(n_t, t_off, HOLE_T, PITCH_T):
        rows[t0:t0 + HOLE_T] = True
    for w0 in _hole_starts(n_x, w_off, HOLE_X, PITCH_X):
        cols[w0:w0 + HOLE_X] = True
    if not rows.any() or not cols.any():
        return np.zeros((n_t, n_x), dtype=bool)
    return rows[:, None] & cols[None, :]


@torch.no_grad()
def predict_mask_residual(model: RealNet, X: np.ndarray, tile=(256, 512), overlap=32) -> np.ndarray:
    """Остаток канала 0 в дырках. Один проход на фазу по всему кадру: тайл 256×512 на каждом кадре слишком дорог.

    Кадр паддится до шага решётки (он же кратен 8, так что три пула U-Net сходятся).
    """
    model.eval()
    X = np.ascontiguousarray(X, dtype=np.float32)
    _c, T, W = X.shape
    pad_h = (PITCH_T - T % PITCH_T) % PITCH_T
    pad_w = (PITCH_X - W % PITCH_X) % PITCH_X
    # слишком широкий кадр режем по x, чтобы активации энкодера не раздували ОЗУ
    max_pix = 4_500_000
    if (T + pad_h) * (W + pad_w) <= max_pix:
        return _mask_residual_canvas(model, X, pad_h, pad_w)
    # полосы по x: _mask_residual_canvas уже возвращает остаток, не прогноз
    acc = np.zeros((T, W), np.float32)
    cnt = np.zeros((T, W), np.float32)
    tw = min(int(tile[1]), W)
    for w0 in _tile_starts(W, tw, overlap):
        sl = X[:, :, w0:w0 + tw]
        part = _mask_residual_canvas(model, sl, (PITCH_T - sl.shape[1] % PITCH_T) % PITCH_T,
                                     (PITCH_X - sl.shape[2] % PITCH_X) % PITCH_X)
        c0 = 0 if w0 == 0 else overlap
        c1 = part.shape[1] if w0 + tw >= W else max(part.shape[1] - overlap, c0)
        acc[:, w0 + c0:w0 + c1] += part[:, c0:c1]
        cnt[:, w0 + c0:w0 + c1] += 1.0
    return (acc / np.maximum(cnt, 1.0)).astype(np.float32)


def _mask_residual_canvas(model, X, pad_h: int, pad_w: int) -> np.ndarray:
    """Одна фаза за раз: батч из четырёх полных кадров раздувает активации U-Net."""
    _c, T, W = X.shape
    xp = torch.from_numpy(np.ascontiguousarray(X, np.float32))[None]
    if pad_h or pad_w:
        xp = F.pad(xp, (0, int(pad_w), 0, int(pad_h)))
    H, Ww = T + int(pad_h), W + int(pad_w)
    acc = np.zeros((T, W), np.float32)
    cnt = np.zeros((T, W), np.float32)
    for t_off in (0, HOLE_T):
        for w_off in (0, HOLE_X):
            m = _lattice_mask(H, Ww, t_off, w_off)
            masked = xp.masked_fill(torch.from_numpy(np.ascontiguousarray(m))[None, None], 0.0)
            pred = model.mask_pred(masked)[0, 0, :T, :W].numpy()
            mm = m[:T, :W]
            acc[mm] += pred[mm]
            cnt[mm] += 1.0
            del masked, pred
    pred_full = acc / np.maximum(cnt, 1.0)
    valid = X[2] > 0.5
    return np.where(valid, X[0] - pred_full, 0.0).astype(np.float32)


@torch.no_grad()
def predict_components(model: RealNet, X: np.ndarray, residual: np.ndarray, sigma: np.ndarray, valid: np.ndarray,
                       need_mask: bool = True, need_wide: bool = True):
    """p_seg, p_mask, p_wide и hotness. Сеть в eval, порог здесь не ставится.

    need_mask / need_wide можно выключить, если валидация уже дала этому члену вес 0.
    До выбора весов оба члена считаются.
    """
    model.eval()
    valid = np.asarray(valid, bool)
    prob = models.predict_full(model.unet, X)
    p_seg = np.where(valid, 1.0 - prob[0], 0.0).astype(np.float32)
    h_net = (prob[1] - prob[2]).astype(np.float32)
    if need_mask:
        p_mask = component_score(predict_mask_residual(model, X), valid, MASK_SIGMA)
    else:
        p_mask = np.zeros(valid.shape, np.float32)
    if need_wide:
        sig = np.maximum(np.asarray(sigma, np.float32), 0.03)
        rs = np.where(valid, np.asarray(residual, np.float32) / sig[None, :], 0.0)
        p_wide = component_score(rs, valid, WIDE_SIGMA)
    else:
        p_wide = np.zeros(valid.shape, np.float32)
    sign_r = np.sign(np.where(valid, residual, 0.0)).astype(np.float32)
    hot = np.where(np.abs(h_net) >= 0.05, h_net, sign_r)
    hot = np.where(valid, hot, 0.0).astype(np.float32)
    return p_seg, p_mask, p_wide, hot


def save_model(model: RealNet, path, **meta) -> None:
    path = __import__("pathlib").Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(unet=model.unet.state_dict(), mask=model.mask_head.state_dict(),
                    width=int(model.unet.head.weight.shape[1]), **meta), path)


def load_model(path, width: int | None = None) -> RealNet:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    w = int(width or ckpt.get("width") or 16)
    unet = models.DTSUNet(w=w)
    unet.load_state_dict(ckpt["unet"])
    model = RealNet(unet)
    model.mask_head.load_state_dict(ckpt["mask"])
    model.eval()
    return model


def assemble_zones(score, hotness, valid, x, t, residual, thr, min_px=12):
    """Зоны как в dts.events. Пик — остаток базовой линии в °C, не сырая температура."""
    zones = E.zones_from_prob(score, hotness, valid, x, t, float(thr), min_px=min_px)
    valid = np.asarray(valid, bool)
    residual = np.asarray(residual, np.float32)
    score = np.asarray(score, np.float32)
    dx = float(np.median(np.diff(x))) if len(x) > 1 else 0.25
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 120.0
    occupied = valid.any(0)
    cable_m = float(x[occupied][-1] - x[occupied][0]) + dx if occupied.any() else float(x[-1] - x[0]) + dx
    for z in zones:
        sl = (slice(z["t0"], z["t1"]), slice(z["x0"], z["x1"]))
        m = (score[sl] >= thr) & valid[sl]
        vals = residual[sl][m]
        if vals.size:
            z["peak_dT_C"] = float(vals.max() if z["polarity"] == "hot" else vals.min())
            z["median_dT_C"] = float(np.median(vals))
        else:
            z["peak_dT_C"] = 0.0
            z["median_dT_C"] = 0.0
        z["dur_h"] = (z["t1"] - z["t0"]) * dt / 3600.0
        z["width_m"] = (z["x1"] - z["x0"]) * dx
        z["width_frac"] = z["width_m"] / max(cable_m, 1.0)
        z["alarm"] = True
        z["class"] = ""
        z["class_prob"] = {"": 0.0}
    return zones


def judge_zone(z) -> tuple[str, list]:
    """Физическое чтение формы. Это не метка и не эталон."""
    peak = float(z.get("peak_dT_C", 0.0))
    amp = abs(peak)
    dur = float(z.get("dur_h", 0.0))
    width = float(z.get("width_m", 0.0))
    frac = float(z.get("width_frac", 0.0))
    reasons = []
    if z.get("polarity") == "hot":
        reasons.append(f"знак: нагрев, пик остатка {peak:+.2f} °C")
    else:
        reasons.append(f"знак: охлаждение, пик остатка {peak:+.2f} °C")
    reasons.append(f"длительность {dur:.2f} ч, протяжённость {width:.1f} м ({frac:.0%} кабеля)")
    cls = z.get("class") or ""
    if cls:
        reasons.append(f"тип выпущенного классификатора: {cls}")
    if frac >= 0.35 and dur >= 0.3:
        reasons.append("в те же сроки закрыта большая доля трассы — похоже на общий режим, а не на локальный очаг")
        verdict = "похоже на скачок общего режима"
    elif width >= 40.0 and dur >= 0.4 and amp >= 0.3:
        reasons.append("плато шире окна 40 м, которым обычно вычитают фон: это как раз зона, которую базовая линия должна оставить в остатке")
        verdict = "широкая плавная зона относительно своей базовой линии"
    elif z.get("polarity") == "cold" and amp >= 0.4 and width <= 30.0 and 0.15 <= dur <= 24.0:
        reasons.append("умеренное по ширине охлаждение; такие рамки ансамбль часто не дотягивал до порога")
        verdict = "правдоподобно как охлаждение участка"
    elif width <= 2.5 and dur >= 2.0:
        reasons.append("узкая полоса держится часами — так выглядят коннектор или стабильный точечный контакт")
        verdict = "возможно коннектор или узкий контакт"
    elif dur >= 8.0 and width >= 25.0:
        reasons.append("многочасовой широкий ход согласуется с суточным ходом секции, если гармоника его не забрала")
        verdict = "похоже на суточный или секционный ход"
    elif amp < 0.25:
        reasons.append("амплитуда остатка меньше 0.25 °C — уровень шума после вычитания базовой линии")
        verdict = "слабо, возможно шум"
    elif 0.4 <= width <= 80.0 and 0.05 <= dur <= 18.0 and amp >= 0.4:
        reasons.append("размер и амплитуда похожи на локальный прогрев или остывание участка кабеля")
        verdict = "правдоподобно как тепловое событие"
    else:
        reasons.append("форма не совпала с шаблонами очага, широкого плато, охлаждения и суточного хода")
        verdict = "неоднозначно"
    return verdict, reasons


def describe_zone(z) -> dict:
    verdict, reasons = judge_zone(z)
    return dict(id=int(z["id"]), start_m=round(float(z["start_m"]), 2), end_m=round(float(z["end_m"]), 2),
                t_start_s=float(z["t_start_s"]), t_end_s=float(z["t_end_s"]),
                polarity=z["polarity"], score=round(float(z["score"]), 4), n_px=int(z["n_px"]),
                peak_dT_C=round(float(z.get("peak_dT_C", 0.0)), 3),
                dur_h=round(float(z.get("dur_h", 0.0)), 3), width_m=round(float(z.get("width_m", 0.0)), 2),
                width_frac=round(float(z.get("width_frac", 0.0)), 4),
                class_name=z.get("class") or "", alarm=bool(z.get("alarm", True)),
                verdict=verdict, reasons=reasons)


def attach_zone_cv(score, dT, valid, x, t, zones):
    """OpenCV-геометрия из dts.zone_cv, если модуль уже есть. Порог не меняет."""
    try:
        from dts import zone_cv
    except ImportError:
        return []
    if not hasattr(zone_cv, "annotate"):
        return []
    return zone_cv.annotate(score, dT, valid, x, t, zones)
