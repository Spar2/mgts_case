"""U-Net с ConvGRU в бутылочном слое.

Костяк совпадает с ``dts.models.DTSUNet`` (ширина 16, TimeGate, те же имена весов),
поэтому ``artifacts/stage1/unet_v810ft.pt`` загружается с ``strict=False``.
После TimeGate карта e4 читается ConvGRU вдоль времени: на каждом шаге свёртка
только по кабелю, скрытое состояние переходит к следующему моменту.

Уравнения — ConvGRU из Ballas et al., 2016 (свёрточный аналог GRU Cho et al.).
ConvLSTM Shi et al., 2015 держит отдельную ячейку памяти; здесь её нет, чтобы
на CPU не удваивать рекуррентные свёртки. Перенос состояния между тайлами
инференса — тот же приём длинного контекста: медленное событие длиннее одного
тайла 256 не обнуляет память на стыке.

Выходная свёртка 1×1 инициализирована нулём. До первого шага блок добавляет нуль,
и логиты совпадают с DTSUNet на тех же весах. Реальные трассы сюда не входят.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dts import models


class ConvGRUCell(nn.Module):
    """Один шаг Ballas 2016: z, r и кандидат — свёртки ядром (1, 3) по кабелю.

    h_t = (1 − z_t) ⊙ h_{t−1} + z_t ⊙ tanh(W_n ∗ x_t + U_n ∗ (r_t ⊙ h_{t−1})).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = int(channels)
        self.conv_x = nn.Conv2d(channels, 3 * channels, (1, 3), padding=(0, 1), bias=True)
        self.conv_h = nn.Conv2d(channels, 3 * channels, (1, 3), padding=(0, 1), bias=True)
        nn.init.xavier_uniform_(self.conv_x.weight)
        nn.init.xavier_uniform_(self.conv_h.weight)
        nn.init.zeros_(self.conv_x.bias)
        nn.init.zeros_(self.conv_h.bias)

    def step(self, x_t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """x_t, h: B,C,W → новое h."""
        gx = self.conv_x(x_t[:, :, None, :]).squeeze(2)
        gh = self.conv_h(h[:, :, None, :]).squeeze(2)
        c = self.channels
        zx, rx, nx = gx[:, :c], gx[:, c:2 * c], gx[:, 2 * c:]
        zh, rh, nh = gh[:, :c], gh[:, c:2 * c], gh[:, 2 * c:]
        z = torch.sigmoid(zx + zh)
        r = torch.sigmoid(rx + rh)
        n = torch.tanh(nx + r * nh)
        return (1.0 - z) * h + z * n


class ConvGRUBottleneck(nn.Module):
    """Проход ConvGRU по оси времени и нулевой остаток к карте e4.

    ``h0`` — состояние с предыдущего тайла, форма B,C,W. ``None`` — нули.
    Невалидная клетка бутылочного слоя (нет valid в окне 8×8) состояние не меняет.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.cell = ConvGRUCell(channels)
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None, valid: torch.Tensor | None = None):
        """x: B,C,T,W. valid: B,T,W или None. Возвращает (карта, h_seq B,C,T,W)."""
        b, c, t, w = x.shape
        h = x.new_zeros(b, c, w) if h0 is None else h0
        if h.shape != (b, c, w):
            raise ValueError(f"h0 {tuple(h.shape)} != {(b, c, w)}")
        steps = []
        for i in range(t):
            h_new = self.cell.step(x[:, :, i, :], h)
            if valid is not None:
                m = valid[:, i, :].unsqueeze(1)
                h = torch.where(m, h_new, h)
            else:
                h = h_new
            steps.append(h)
        h_seq = torch.stack(steps, dim=2)
        return x + self.proj(h_seq), h_seq


class ConvGRUNet(nn.Module):
    """DTSUNet + ConvGRU после TimeGate.

    Имена e1..head совпадают с DTSUNet. ``h0`` и ``return_state`` нужны инференсу,
    который несёт состояние между тайлами вдоль времени.
    """

    def __init__(self, cin: int = 3, n_classes: int = 3, w: int = 16):
        super().__init__()
        c = [w, 2 * w, 4 * w, 8 * w]
        self.e1 = models.block(cin, c[0], (7, 3))
        self.e2 = models.block(c[0], c[1], (7, 3))
        self.e3 = models.block(c[1], c[2])
        self.e4 = models.block(c[2], c[3])
        self.gate = models.TimeGate(c[3])
        self.gru = ConvGRUBottleneck(c[3])
        self.u3 = nn.ConvTranspose2d(c[3], c[2], 2, 2)
        self.d3 = models.block(2 * c[2], c[2])
        self.u2 = nn.ConvTranspose2d(c[2], c[1], 2, 2)
        self.d2 = models.block(2 * c[1], c[1])
        self.u1 = nn.ConvTranspose2d(c[1], c[0], 2, 2)
        self.d1 = models.block(2 * c[0], c[0])
        self.head = nn.Conv2d(c[0], n_classes, 1)
        self.bottleneck_channels = c[3]

    def encode(self, x: torch.Tensor):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        e4 = self.gate(self.e4(F.max_pool2d(e3, 2)))
        valid = x[:, 2:3]
        for _ in range(3):
            valid = F.max_pool2d(valid, 2)
        return e1, e2, e3, e4, valid[:, 0] > 0.5

    def decode(self, e4, e3, e2, e1):
        d3 = self.d3(torch.cat([self.u3(e4), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.head(d1)

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None, return_state: bool = False):
        e1, e2, e3, e4, valid = self.encode(x)
        e4, h_seq = self.gru(e4, h0, valid)
        logits = self.decode(e4, e3, e2, e1)
        if return_state:
            return logits, h_seq
        return logits


def load_unet_weights(model: ConvGRUNet, path: str | os.PathLike) -> list[str]:
    """Скопировать веса DTSUNet. Возвращает недостающие ключи (блок gru)."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    incompat = model.load_state_dict(state, strict=False)
    unexpected = list(incompat.unexpected_keys)
    if unexpected:
        raise RuntimeError(f"лишние ключи чекпойнта: {unexpected[:8]}")
    return list(incompat.missing_keys)


def param_counts(model: nn.Module) -> dict:
    gru = sum(p.numel() for n, p in model.named_parameters() if n.startswith("gru."))
    total = sum(p.numel() for p in model.parameters())
    return {"total": int(total), "gru": int(gru), "backbone": int(total - gru)}


def proj_l1(model: ConvGRUNet) -> float:
    return float(model.gru.proj.weight.detach().abs().sum().item())


def tile_starts(n: int, tile: int, step: int) -> list[int]:
    ts = list(range(0, max(n - tile, 0) + 1, step)) or [0]
    if ts[-1] + tile < n:
        ts.append(max(n - tile, 0))
    return ts


@torch.inference_mode()
def predict_full(model: ConvGRUNet, X, tile=(256, 512), overlap=32) -> np.ndarray:
    """Тайлы как у ``models.predict_full``, но внутри каждой колонки кабеля
    скрытое состояние ConvGRU переходит к следующему тайлу по времени.

    Индекс состояния — последняя строка бутылочного слоя строго до начала
    следующего тайла (шаг входных пикселей, делённый на 8). Если шаг не кратен 8,
    берётся целая часть; так закрыт последний неровный тайл кадра.
    """
    model.eval()
    X = np.asarray(X, np.float32)
    _, t_len, w_len = X.shape
    th, tw = tile
    step_t, step_w = th - 2 * overlap, tw - 2 * overlap
    ts = tile_starts(t_len, th, step_t)
    ws = tile_starts(w_len, tw, step_w)
    out = np.zeros((3, t_len, w_len), np.float32)
    cnt = np.zeros((t_len, w_len), np.float32)
    h = None
    prev_t = None
    for t0 in ts:
        if h is not None and prev_t is not None:
            k = (t0 - prev_t) // 8 - 1
            if h.shape[2] > k >= 0:
                h0 = h[:, :, k, :]
            else:
                h0 = None
        else:
            h0 = None
        xs, meta = [], []
        for w0 in ws:
            sl = X[:, t0:t0 + th, w0:w0 + tw]
            ht, wt = sl.shape[1], sl.shape[2]
            if ht != th or wt != tw:
                sl = np.pad(sl, ((0, 0), (0, th - ht), (0, tw - wt)))
            xs.append(sl)
            meta.append((w0, ht, wt))
        logits, h_seq = model(torch.from_numpy(np.stack(xs)), h0, return_state=True)
        prob = torch.softmax(logits, 1).numpy()
        for p, (w0, ht, wt) in zip(prob, meta):
            out[:, t0:t0 + ht, w0:w0 + wt] += p[:, :ht, :wt]
            cnt[t0:t0 + ht, w0:w0 + wt] += 1
        h = h_seq
        prev_t = t0
        del logits, h_seq, prob
    return out / np.maximum(cnt, 1.0)


def scores_from_prob(prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (1.0 - prob[0]).astype(np.float32), (prob[1] - prob[2]).astype(np.float32)
