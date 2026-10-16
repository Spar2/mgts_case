"""U-Net с осевым вниманием в бутылочном слое.

Костяк совпадает с ``dts.models.DTSUNet`` (ширина 16, TimeGate, те же имена весов),
поэтому чекпойнт ``artifacts/stage1/unet.pt`` загружается как есть. Поверх признака
e4 стоит осевой блок: многоголовое внимание по времени, поточечный FFN и внимание
по кабелю. Это факторное внимание Axial-DeepLab (Ho et al., 2020): полная
двумерная карта не строится, каждая ось смотрит только вдоль себя.

Выходные проекции внимания и FFN инициализированы нулём. До первого шага
оптимизатора блок добавляет нуль, и ``AxialUNet`` совпадает с ``DTSUNet`` на тех
же весах. Относительный сдвиг позиции (как в T5) учится вместе с запросами;
таблица ограничена ``max_len``, более дальние сдвиги прилипают к краю.

Канал 2 входа — маска valid. В бутылочном слое ключ закрыт, если в его клетке
8×8 нет ни одного валидного пикселя. Строка, где закрыты все ключи, даёт нуль,
а не NaN, поэтому нулевая проекция остаётся точным тождеством.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from dts import models


def _groups(channels: int) -> int:
    for g in (8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class AxisAttention(nn.Module):
    """Многоголовое внимание вдоль одной оси карты B,C,T,W.

    ``axis='time'`` смешивает строки (время) отдельно в каждом столбце кабеля.
    ``axis='x'`` смешивает отсчёты кабеля отдельно в каждом моменте времени.
    """

    def __init__(self, channels: int, heads: int = 4, axis: str = "time", max_len: int = 96):
        super().__init__()
        if axis not in ("time", "x"):
            raise ValueError(axis)
        if channels % heads != 0:
            raise ValueError(f"channels {channels} не делится на heads {heads}")
        if max_len < 2:
            raise ValueError(max_len)
        self.heads = heads
        self.axis = axis
        self.max_len = int(max_len)
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.rel = nn.Parameter(torch.zeros(heads, 2 * self.max_len - 1))
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.zeros_(self.proj.weight)

    def _rel_bias(self, length: int, device: torch.device) -> torch.Tensor:
        """logits[q, k] += bias[k - q]. Диагональ таблицы — индекс max_len-1."""
        pos = torch.arange(length, device=device)
        offset = (pos[None, :] - pos[:, None]).clamp(-(self.max_len - 1), self.max_len - 1)
        index = offset + (self.max_len - 1)
        return self.rel[:, index]

    def forward(self, x: torch.Tensor, key_ok: torch.Tensor | None = None) -> torch.Tensor:
        b, c, t, w = x.shape
        h, d = self.heads, c // self.heads
        qkv = self.qkv(x).reshape(b, 3, h, d, t, w)
        if self.axis == "time":
            qkv = qkv.permute(1, 0, 5, 2, 4, 3)  # 3, B, W, h, T, d
            q, k, v = (qkv[i].reshape(b * w, h, t, d) for i in range(3))
            mask = None if key_ok is None else key_ok.permute(0, 2, 1).reshape(b * w, t)
            y = self._attend(q, k, v, t, mask)
            y = y.reshape(b, w, h, t, d).permute(0, 2, 4, 3, 1).reshape(b, c, t, w)
        else:
            qkv = qkv.permute(1, 0, 4, 2, 5, 3)  # 3, B, T, h, W, d
            q, k, v = (qkv[i].reshape(b * t, h, w, d) for i in range(3))
            mask = None if key_ok is None else key_ok.reshape(b * t, w)
            y = self._attend(q, k, v, w, mask)
            y = y.reshape(b, t, h, w, d).permute(0, 2, 4, 1, 3).reshape(b, c, t, w)
        return self.proj(y)

    def _attend(self, q, k, v, length: int, key_ok: torch.Tensor | None) -> torch.Tensor:
        scale = q.shape[-1] ** -0.5
        logits = torch.matmul(q, k.transpose(-1, -2)) * scale
        logits = logits + self._rel_bias(length, q.device)
        if key_ok is not None:
            logits = logits.masked_fill(~key_ok[:, None, None, :], float("-inf"))
        weight = torch.softmax(logits, dim=-1)
        weight = torch.nan_to_num(weight, nan=0.0)
        return torch.matmul(weight, v)


class AxialBottleneck(nn.Module):
    """Временной трансформер (внимание + FFN) и следом внимание по кабелю.

    Каждый остаток умножается на свёртку 1×1 с нулевым весом, поэтому
    ``forward(x) == x`` до обучения этих свёрток.
    """

    def __init__(self, channels: int, heads: int = 4, mlp_ratio: int = 2, max_time: int = 96, max_x: int = 160):
        super().__init__()
        g = _groups(channels)
        hidden = channels * mlp_ratio
        self.n1 = nn.GroupNorm(g, channels)
        self.attn_t = AxisAttention(channels, heads=heads, axis="time", max_len=max_time)
        self.n2 = nn.GroupNorm(g, channels)
        self.ff = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.n3 = nn.GroupNorm(g, channels)
        self.attn_x = AxisAttention(channels, heads=heads, axis="x", max_len=max_x)
        nn.init.xavier_uniform_(self.ff[0].weight)
        nn.init.zeros_(self.ff[2].weight)

    def forward(self, x: torch.Tensor, key_ok: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn_t(self.n1(x), key_ok)
        x = x + self.ff(self.n2(x))
        x = x + self.attn_x(self.n3(x), key_ok)
        return x


class AxialUNet(nn.Module):
    """DTSUNet + ``AxialBottleneck`` после TimeGate.

    Имена e1..head совпадают с ``DTSUNet``, чтобы ``load_state_dict`` чекпойнта
    U-Net проходил с ``strict=False`` и не затирал нулевые проекции.
    """

    def __init__(self, cin: int = 3, n_classes: int = 3, w: int = 16, heads: int = 4):
        super().__init__()
        c = [w, 2 * w, 4 * w, 8 * w]
        self.e1 = models.block(cin, c[0], (7, 3))
        self.e2 = models.block(c[0], c[1], (7, 3))
        self.e3 = models.block(c[1], c[2])
        self.e4 = models.block(c[2], c[3])
        self.gate = models.TimeGate(c[3])
        self.axial = AxialBottleneck(c[3], heads=heads)
        self.u3 = nn.ConvTranspose2d(c[3], c[2], 2, 2)
        self.d3 = models.block(2 * c[2], c[2])
        self.u2 = nn.ConvTranspose2d(c[2], c[1], 2, 2)
        self.d2 = models.block(2 * c[1], c[1])
        self.u1 = nn.ConvTranspose2d(c[1], c[0], 2, 2)
        self.d1 = models.block(2 * c[0], c[0])
        self.head = nn.Conv2d(c[0], n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        e4 = self.gate(self.e4(F.max_pool2d(e3, 2)))
        valid = x[:, 2:3]
        for _ in range(3):
            valid = F.max_pool2d(valid, 2)
        e4 = self.axial(e4, valid[:, 0] > 0.5)
        d3 = self.d3(torch.cat([self.u3(e4), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.head(d1)


def load_unet_weights(model: AxialUNet, path: str | os.PathLike) -> list[str]:
    """Скопировать веса DTSUNet. Возвращает недостающие ключи (все из axial)."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    incompat = model.load_state_dict(state, strict=False)
    unexpected = list(incompat.unexpected_keys)
    if unexpected:
        raise RuntimeError(f"лишние ключи чекпойнта: {unexpected[:8]}")
    return list(incompat.missing_keys)


def param_counts(model: nn.Module) -> dict:
    axial = sum(p.numel() for n, p in model.named_parameters() if n.startswith("axial."))
    total = sum(p.numel() for p in model.parameters())
    return {"total": int(total), "axial": int(axial), "backbone": int(total - axial)}


def proj_l1(model: AxialUNet) -> float:
    """L1 выходных проекций: нуль означает тождество с U-Net."""
    parts = [model.axial.attn_t.proj.weight, model.axial.attn_x.proj.weight, model.axial.ff[2].weight]
    return float(sum(p.detach().abs().sum().item() for p in parts))


def scores_from_logits(prob: torch.Tensor | "object") -> tuple:
    """prob (3,T,W) после softmax -> s = 1-p(фон), h = p(нагрев)-p(охлаждение)."""
    import numpy as np
    pr = np.asarray(prob)
    return (1.0 - pr[0]).astype(np.float32), (pr[1] - pr[2]).astype(np.float32)


def far_threshold_hist(scores, far: float, nbin: int = 100_000) -> float:
    """Порядок statistics как у metrics.far_threshold, бин шириной 1/nbin на [0, 1]."""
    import numpy as np
    scores = np.clip(np.asarray(scores, np.float64).ravel(), 0.0, 1.0)
    if scores.size == 0:
        return 1.0
    idx = np.clip((scores * nbin).astype(np.int32), 0, nbin - 1)
    hist = np.bincount(idx, minlength=nbin)
    n = int(hist.sum())
    k = int(np.ceil((1.0 - far) * n)) - 1
    k = int(np.clip(k, 0, n - 1))
    i = int(np.searchsorted(np.cumsum(hist), k + 1, side="left"))
    i = min(max(i, 0), nbin - 1)
    return float((i + 0.5) / nbin) + 1e-12
