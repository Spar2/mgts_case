"""Одношаговый rectified flow фона (REFLECT / DiffusionAD, один шаг Эйлера).

Сеть учит скорость переноса шум → чистый фон только на тайлах без события и nuisance.
На инференсе один шаг к норме. Скор клетки — модуль остатка канала dT, знак — знак остатка.
Это не голова сегментации и не DTSUNet: на выходе одно поле скорости.
Реальные трассы и их спектр сюда не входят.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .models import TimeGate, block

T_STAR = 0.5
TILE = (256, 512)
OVERLAP = 32


class VelocityNet(nn.Module):
    """Поле скорости ширины 16. Вход: зашумлённый dT, маска valid и время потока."""

    def __init__(self, w: int = 16):
        super().__init__()
        c = [w, 2 * w, 4 * w, 8 * w]
        self.e1 = block(3, c[0], (7, 3))
        self.e2 = block(c[0], c[1], (7, 3))
        self.e3 = block(c[1], c[2])
        self.e4 = block(c[2], c[3])
        self.gate = TimeGate(c[3])
        self.u3 = nn.ConvTranspose2d(c[3], c[2], 2, 2)
        self.d3 = block(2 * c[2], c[2])
        self.u2 = nn.ConvTranspose2d(c[2], c[1], 2, 2)
        self.d2 = block(2 * c[1], c[1])
        self.u1 = nn.ConvTranspose2d(c[1], c[0], 2, 2)
        self.d1 = block(2 * c[0], c[0])
        self.head = nn.Conv2d(c[0], 1, 1)

    def forward(self, x_t: torch.Tensor, valid: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.view(-1, 1, 1, 1)
        tmap = t.expand(-1, 1, x_t.shape[2], x_t.shape[3])
        x = torch.cat([x_t, valid, tmap], 1)
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        e4 = self.gate(self.e4(F.max_pool2d(e3, 2)))
        d3 = self.d3(torch.cat([self.u3(e4), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.head(d1) * valid


def flow_loss(model: VelocityNet, x1: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Прямой путь z → x1. Вход сети всегда зашумлён, чистый фон только в цели."""
    z = torch.randn_like(x1)
    t = 0.15 + 0.70 * torch.rand(x1.shape[0], 1, 1, 1, device=x1.device)
    x_t = (t * x1 + (1.0 - t) * z) * valid
    target = (x1 - z) * valid
    pred = model(x_t, valid, t)
    denom = valid.sum().clamp(min=1.0)
    return ((pred - target) ** 2 * valid).sum() / denom


def _tile_starts(n: int, tile: int, step: int) -> list[int]:
    if n <= tile:
        return [0]
    starts = list(range(0, n - tile + 1, step))
    last = n - tile
    if starts[-1] != last:
        starts.append(last)
    return starts


@torch.no_grad()
def reconstruct(model: VelocityNet, field: np.ndarray, valid: np.ndarray,
                tile=TILE, overlap: int = OVERLAP, t_star: float = T_STAR, batch: int = 4,
                seed: int = 0) -> np.ndarray:
    """Один эйлеров шаг. Шум z зависит только от размера поля и seed, не от значений field.

    field, valid: (T, W). Возвращает оценку чистого фона той же формы.
    """
    was = model.training
    model.eval()
    field = np.asarray(field, np.float32)
    valid = np.asarray(valid, np.float32)
    th, tw = tile
    if th % 8 or tw % 8:
        raise ValueError("tile must be divisible by 8")
    t_len, w_len = field.shape
    z = np.random.default_rng(seed).standard_normal((t_len, w_len)).astype(np.float32)
    step_t, step_w = th - 2 * overlap, tw - 2 * overlap
    boxes = [(t0, w0) for t0 in _tile_starts(t_len, th, step_t) for w0 in _tile_starts(w_len, tw, step_w)]
    acc = np.zeros((t_len, w_len), np.float32)
    cnt = np.zeros((t_len, w_len), np.float32)
    t_col = torch.full((batch, 1, 1, 1), float(t_star))
    for i in range(0, len(boxes), batch):
        xs, vs, zs, meta = [], [], [], []
        for t0, w0 in boxes[i:i + batch]:
            sl_f = field[t0:t0 + th, w0:w0 + tw]
            sl_v = valid[t0:t0 + th, w0:w0 + tw]
            sl_z = z[t0:t0 + th, w0:w0 + tw]
            ht, wt = sl_f.shape
            if ht != th or wt != tw:
                pad = ((0, th - ht), (0, tw - wt))
                sl_f = np.pad(sl_f, pad)
                sl_v = np.pad(sl_v, pad)
                sl_z = np.pad(sl_z, pad)
            xs.append(sl_f)
            vs.append(sl_v)
            zs.append(sl_z)
            meta.append((t0, w0, ht, wt))
        b = len(xs)
        yt = torch.from_numpy(np.stack(xs))[:, None]
        vm = torch.from_numpy(np.stack(vs))[:, None]
        zt = torch.from_numpy(np.stack(zs))[:, None]
        tt = t_col[:b]
        x_t = (tt * yt + (1.0 - tt) * zt) * vm
        v = model(x_t, vm, tt)
        hat = (x_t + (1.0 - tt) * v) * vm
        hat_np = hat[:, 0].numpy()
        for p, (t0, w0, ht, wt) in zip(hat_np, meta):
            acc[t0:t0 + ht, w0:w0 + wt] += p[:ht, :wt]
            cnt[t0:t0 + ht, w0:w0 + wt] += 1.0
    model.train(was)
    return acc / np.maximum(cnt, 1.0)


def residual_map(model: VelocityNet, field: np.ndarray, valid: np.ndarray, **kw) -> np.ndarray:
    """Остаток = наблюдение − восстановленная норма. Знак сохраняется."""
    hat = reconstruct(model, field, valid, **kw)
    return (np.asarray(field, np.float32) - hat) * np.asarray(valid, np.float32)


def copy_probe(model: VelocityNet, field: np.ndarray, valid: np.ndarray, amp: float = 4.0) -> dict:
    """Проверка тождественного копирования: всплеск и шум на чистом фоне.

    restore → 1, если всплеск стёрт и поле вернулось к фону.
    denoise → 1, если добавленный шум снят. Оба около 0 — сеть копирует вход.
    """
    field = np.asarray(field, np.float32)
    valid = np.asarray(valid, np.float32)
    h, w = field.shape
    t0, t1 = max(0, h // 2 - 8), min(h, h // 2 + 8)
    w0, w1 = max(0, w // 2 - 20), min(w, w // 2 + 20)
    if h % 8 or w % 8:
        raise ValueError("copy_probe ждёт поле, кратное 8")
    spiked = field.copy()
    spiked[t0:t1, w0:w1] += np.float32(amp)
    hat = reconstruct(model, spiked, valid, tile=(h, w), overlap=0, batch=1)
    inj = amp
    left = float(np.mean(np.abs(hat[t0:t1, w0:w1] - field[t0:t1, w0:w1])))
    restore = float(1.0 - left / (abs(inj) + 1e-6))
    rng = np.random.default_rng(1)
    noisy = field + rng.standard_normal(field.shape).astype(np.float32)
    noisy *= valid
    hat_n = reconstruct(model, noisy, valid, tile=(h, w), overlap=0, batch=1)
    m = valid > 0.5
    base = float(np.mean((noisy[m] - field[m]) ** 2))
    left_n = float(np.mean((hat_n[m] - field[m]) ** 2))
    denoise = float(1.0 - left_n / (base + 1e-6))
    return {"restore": restore, "denoise": denoise, "spike_abs_left": left}


def assert_euler_identity() -> None:
    """Идеальная скорость x1−z за один шаг возвращает x1."""
    rng = np.random.default_rng(0)
    x1 = rng.standard_normal((2, 1, 8, 8)).astype(np.float32)
    z = rng.standard_normal(x1.shape).astype(np.float32)
    t = np.float32(0.5)
    x_t = t * x1 + (1.0 - t) * z
    v = x1 - z
    hat = x_t + (1.0 - t) * v
    err = float(np.max(np.abs(hat - x1)))
    if err > 1e-5:
        raise SystemExit(f"эйлеров шаг не сходится: {err}")
