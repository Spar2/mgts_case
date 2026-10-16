"""Слепой свёрточный автоэнкодер (Noise2Self) для карты аномалий Raman-DTS.

Поле уже приведено к σ: каналы 0 и 1 — ΔT/σ, канал 2 — маска valid.
Сеть структурно не видит блок 8×16 вокруг предсказываемого пикселя (четыре
причинных ветви: сверху, снизу, слева, справа) и на обучении дополнительно
получает вход с занулёнными случайными блоками 8×16 в каналах 0–1. Учим
только эти блоки, без меток.

Скор — сглаженный гауссом модуль ошибки восстановления (единицы σ).
Hotness — знак ошибки: плюс, если наблюдение горячее контекста.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

BLOCK_T, BLOCK_X = 8, 16
# шаг решётки масок: между блоками остаётся такой же зазор контекста
PITCH_T, PITCH_X = 16, 32
# сколько пикселей первая причинная свёртка не смотрит в сторону центра,
# чтобы дырка была ровно BLOCK_T × BLOCK_X (чётный размер → поля несимметричны)
MARGIN = {"above": 4, "below": 5, "left": 8, "right": 9}
WIDTH = 16
KERNEL = 5
SIDE_K = 5
DILATIONS = (1, 2, 4)
SIGMA = (2.0, 6.0)          # сглаживание скора, в пикселях (120 с × 0.25 м)
HUBER_DELTA = 2.0           # робастный Noise2Self: фон в ~1σ важнее хвостов


def blind_box(i: int, j: int) -> tuple[int, int, int, int]:
    """Полуинтервал [r0, r1) × [c0, c1) каналов 0–1, невидимый для пикселя (i, j)."""
    r0 = i - (BLOCK_T // 2 - 1)          # i-3
    r1 = i + (BLOCK_T // 2 + 1)          # i+5 → строки i-3..i+4
    c0 = j - (BLOCK_X // 2 - 1)          # j-7
    c1 = j + (BLOCK_X // 2 + 1)          # j+9 → столбцы j-7..j+8
    return r0, r1, c0, c1


class DirConv(nn.Module):
    """Разделённая свёртка, которая смотрит только по одну сторону от дырки.

    Сначала причинное ядро вдоль запрещённого направления (с дилатацией),
    затем симметричное ядро поперёк. Поперечное ядро не возвращает взгляд
    внутрь слепой зоны: столбцы/строки, куда ветвь имеет право смотреть,
    уже вне блока 8×16.
    """

    def __init__(self, cin, cout, k, direction, margin, dilation, side_k):
        super().__init__()
        self.k, self.direction, self.margin = k, direction, margin
        self.dilation, self.side_k = dilation, side_k
        vertical = direction in ("above", "below")
        if vertical:
            self.axis = nn.Conv2d(cin, cout, (k, 1), dilation=(dilation, 1), bias=False)
            self.side = nn.Conv2d(cout, cout, (1, side_k), bias=True)
        else:
            self.axis = nn.Conv2d(cin, cout, (1, k), dilation=(1, dilation), bias=False)
            self.side = nn.Conv2d(cout, cout, (side_k, 1), bias=True)
        self.act = nn.GELU()

    def forward(self, x):
        span = (self.k - 1) * self.dilation + self.margin
        m = self.margin
        if self.direction == "above":
            y = self.axis(F.pad(x, (0, 0, span, 0)))
            if m:
                y = y[:, :, :-m]
        elif self.direction == "below":
            y = self.axis(F.pad(x, (0, 0, 0, span)))
            if m:
                y = y[:, :, m:]
        elif self.direction == "left":
            y = self.axis(F.pad(x, (span, 0, 0, 0)))
            if m:
                y = y[:, :, :, :-m]
        else:
            y = self.axis(F.pad(x, (0, span, 0, 0)))
            if m:
                y = y[:, :, :, m:]
        p = self.side_k // 2
        if self.direction in ("above", "below"):
            y = self.side(F.pad(y, (p, p, 0, 0)))
        else:
            y = self.side(F.pad(y, (0, 0, p, p)))
        return self.act(y)


class CausalBranch(nn.Module):
    def __init__(self, direction, cin, width, k, dilations, margin, side_k):
        super().__init__()
        layers, c = [], cin
        for i, d in enumerate(dilations):
            layers.append(DirConv(c, width, k, direction, margin if i == 0 else 0, d, side_k))
            c = width
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class BlindSpotAE(nn.Module):
    """Четыре причинные ветви → узкое 1×1 бутылочное горлышко → 2 канала."""

    def __init__(self, width=WIDTH, k=KERNEL, dilations=DILATIONS, side_k=SIDE_K, cin=3):
        super().__init__()
        self.width, self.k, self.side_k = width, k, side_k
        self.dilations = tuple(dilations)
        self.branches = nn.ModuleDict({
            d: CausalBranch(d, cin, width, k, self.dilations, MARGIN[d], side_k)
            for d in ("above", "below", "left", "right")})
        self.head = nn.Sequential(
            nn.Conv2d(4 * width, width, 1), nn.GELU(),
            nn.Conv2d(width, 2, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.score_sigma = SIGMA

    def lookback(self):
        """Сколько соседей нужно с каждой стороны, чтобы контекст не упирался в край тайла."""
        reach = (self.k - 1) * int(sum(self.dilations))
        side = (self.side_k // 2) * len(self.dilations)
        return (
            max(MARGIN["above"] + reach, side),
            max(MARGIN["below"] + reach, side),
            max(MARGIN["left"] + reach, side),
            max(MARGIN["right"] + reach, side),
        )

    def forward(self, x):
        feats = [self.branches[d](x) for d in ("above", "below", "left", "right")]
        return self.head(torch.cat(feats, 1))


def sample_block_mask(n, h, w, rng, p=0.5) -> np.ndarray:
    """Случайная фаза решётки и бернуллиевские блоки 8×16. Форма (n, 1, h, w)."""
    mask = np.zeros((n, 1, h, w), np.float32)
    for i in range(n):
        t0 = int(rng.integers(0, PITCH_T))
        w0 = int(rng.integers(0, PITCH_X))
        ts = np.arange(t0, max(h - BLOCK_T, 0) + 1, PITCH_T)
        ws = np.arange(w0, max(w - BLOCK_X, 0) + 1, PITCH_X)
        if len(ts) and len(ws):
            pick = rng.random((len(ts), len(ws))) < p
            for ti, t in enumerate(ts):
                for wi, c in enumerate(ws):
                    if pick[ti, wi]:
                        mask[i, 0, int(t):int(t) + BLOCK_T, int(c):int(c) + BLOCK_X] = 1
        if mask[i].sum() == 0 and h >= BLOCK_T and w >= BLOCK_X:
            t = int(rng.integers(0, h - BLOCK_T + 1))
            c = int(rng.integers(0, w - BLOCK_X + 1))
            mask[i, 0, t:t + BLOCK_T, c:c + BLOCK_X] = 1
    return mask


def masked_loss(pred, target, mask, valid, delta=HUBER_DELTA):
    """Huber только на замаскированных валидных пикселях каналов 0–1."""
    w = mask * (valid > 0.5).to(dtype=pred.dtype)
    r = pred - target
    a = r.abs()
    hub = torch.where(a <= delta, 0.5 * r * r, delta * (a - 0.5 * delta))
    return (hub * w).sum() / (w.sum() * pred.shape[1]).clamp(min=1.0)


def coverage_spans(length: int, tile: int, before: int, after: int):
    """Окна (start, keep0, keep1): keep покрывают [0, length) без перекрытий и дыр.

    У внутреннего окна надёжны выходы [start+before, start+tile-after):
    у них полный контекст внутри тайла. Первое окно забирает край изображения.
    """
    if length <= 0:
        return []
    if length <= tile or before + after >= tile:
        return [(0, 0, length)]
    step = tile - before - after
    starts = list(range(0, length - tile + 1, step))
    last = length - tile
    if starts[-1] != last:
        starts.append(last)
    spans, covered = [], 0
    for i, s in enumerate(starts):
        k0 = 0 if i == 0 else covered
        k1 = length if i == len(starts) - 1 else min(length, s + tile - after)
        if k0 < s or k1 > s + tile or k1 < k0:
            raise RuntimeError(f"bad span start={s} keep=[{k0},{k1}) tile={tile} len={length}")
        spans.append((s, k0, k1))
        covered = k1
    if covered != length:
        raise RuntimeError(f"coverage {covered} != {length}")
    return spans


@torch.inference_mode()
def reconstruct(model: BlindSpotAE, X: np.ndarray, tile=(256, 768), batch=4, max_full_pixels=3_000_000) -> np.ndarray:
    """X (3, T, W) → восстановленные каналы 0–1, форма (2, T, W). Один проход: дырка структурная."""
    model.eval()
    X = np.asarray(X, np.float32)
    _, T, W = X.shape
    # целиком, пока активации ветви влезают в память (ширина 16 × float32 ≈ 64 байта/пиксель на карту)
    if T * W <= max_full_pixels:
        y = model(torch.from_numpy(X)[None])
        return y[0].numpy()
    up, down, left, right = model.lookback()
    th = min(max(tile[0], up + down + 32), T)
    tw = min(max(tile[1], left + right + 32), W)
    t_spans = coverage_spans(T, th, up if T > th else 0, down if T > th else 0)
    w_spans = coverage_spans(W, tw, left if W > tw else 0, right if W > tw else 0)
    # если изображение короче тайла, coverage схлопывается в один полный кадр — поправим размеры окна
    if len(t_spans) == 1 and t_spans[0][0] == 0 and t_spans[0][2] == T:
        th = T
    if len(w_spans) == 1 and w_spans[0][0] == 0 and w_spans[0][2] == W:
        tw = W
    pred = np.zeros((2, T, W), np.float32)
    jobs = [(ts, ws) for ts in t_spans for ws in w_spans]
    for i0 in range(0, len(jobs), batch):
        chunk = jobs[i0:i0 + batch]
        patches = np.zeros((len(chunk), 3, th, tw), np.float32)
        for i, ((t0, _, _), (w0, _, _)) in enumerate(chunk):
            src = X[:, t0:t0 + th, w0:w0 + tw]
            patches[i, :, :src.shape[1], :src.shape[2]] = src
        yb = model(torch.from_numpy(patches)).numpy()
        for y, ((t0, a, b), (w0, c, d)) in zip(yb, chunk):
            pred[:, a:b, c:d] = y[:, a - t0:b - t0, c - w0:d - w0]
    return pred


def anomaly_maps(X: np.ndarray, pred: np.ndarray, sigma=SIGMA) -> tuple[np.ndarray, np.ndarray]:
    """s = Gσ(|ошибка|) в единицах σ канала; h = знак ошибки (hot, если T выше контекста)."""
    valid = np.asarray(X[2] > 0.5, np.float32)
    e0 = np.asarray(X[0], np.float32) - np.asarray(pred[0], np.float32)
    e1 = np.asarray(X[1], np.float32) - np.asarray(pred[1], np.float32)
    err = np.where(np.abs(e1) > np.abs(e0), e1, e0).astype(np.float32)
    mag = gaussian_filter(np.abs(err) * valid, sigma, mode="nearest")
    den = gaussian_filter(valid, sigma, mode="nearest")
    s = (mag / np.maximum(den, 1e-3) * valid).astype(np.float32)
    h = (np.sign(err) * valid).astype(np.float16)
    return s, h


def score_field(model: BlindSpotAE, X: np.ndarray, sigma=None) -> tuple[np.ndarray, np.ndarray]:
    pred = reconstruct(model, X)
    return anomaly_maps(X, pred, sigma or getattr(model, "score_sigma", SIGMA))


def save_model(model: BlindSpotAE, path, **meta) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "config": {
            "width": model.width, "k": model.k, "side_k": model.side_k,
            "dilations": list(model.dilations),
            "block_t": BLOCK_T, "block_x": BLOCK_X,
            "sigma_t": float(model.score_sigma[0]), "sigma_x": float(model.score_sigma[1]),
        },
        **meta,
    }, path)


def load_model(path) -> BlindSpotAE:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    model = BlindSpotAE(
        width=cfg.get("width", WIDTH), k=cfg.get("k", KERNEL),
        dilations=tuple(cfg.get("dilations", DILATIONS)), side_k=cfg.get("side_k", SIDE_K))
    model.load_state_dict(ckpt["state_dict"])
    model.score_sigma = (float(cfg.get("sigma_t", SIGMA[0])), float(cfg.get("sigma_x", SIGMA[1])))
    model.eval()
    return model


class FieldTiles:
    """Тайлы только из mmap `split/*_X.npy`. Файлы меток *_y.npy не открываются."""

    def __init__(self, mm_root, split="train", seed=0, paths=None):
        self.paths = list(paths) if paths is not None else sorted(Path(mm_root).glob(f"{split}/*_X.npy"))
        if not self.paths:
            raise FileNotFoundError(f"no *_X.npy in {mm_root}/{split}")
        bad = [p.name for p in self.paths if not p.name.endswith("_X.npy")]
        if bad:
            raise ValueError(f"refusing non-X fields: {bad[:3]}")
        self.arrs = [np.load(p, mmap_mode="r") for p in self.paths]
        for p, a in zip(self.paths, self.arrs):
            if a.ndim != 3 or a.shape[0] != 3:
                raise ValueError(f"{p} shape {a.shape}, expected (3, T, X)")
        w = np.array([a.shape[1] * a.shape[2] for a in self.arrs], np.float64)
        self.prob = w / w.sum()
        self.rng = np.random.default_rng(seed)

    def sample(self, n, th, tw) -> np.ndarray:
        out = np.zeros((n, 3, th, tw), np.float32)
        for i in range(n):
            a = self.arrs[int(self.rng.choice(len(self.arrs), p=self.prob))]
            _, T, W = a.shape
            t0 = 0 if T <= th else int(self.rng.integers(0, T - th + 1))
            w0 = 0 if W <= tw else int(self.rng.integers(0, W - tw + 1))
            src = np.array(a[:, t0:t0 + th, w0:w0 + tw], dtype=np.float32, copy=True)
            if self.rng.random() < 0.5:
                src = src[:, :, ::-1].copy()
            out[i, :, :src.shape[1], :src.shape[2]] = src
        return out
