"""Свёртка слиятия победителя и пространственного контраста.

Вход (4, T, W):
  0  s_winner          — ens_u2ft_tcn_lgbm
  1  CDF(|dT_space|)   — узлы CDF с обучения MAE, не с v8 val
  2  |dT_time| / 12    — фиксированный масштаб клипа канала
  3  valid

Все свёртки 3×3. Параметров не больше 5000. Выход — логит пикселя события.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from dts.mae_anom import apply_cdf

CLIP = 12.0
N_PARAMS_MAX = 5000
CHANNELS = ("s_winner", "cdf_abs_dT_space", "abs_dT_time_over_12", "valid")


class FuseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 3, padding=1),
        )
        last = self.body[-1]
        nn.init.normal_(last.weight, std=0.01)
        nn.init.constant_(last.bias, -2.0)
        n = n_params(self)
        if n > N_PARAMS_MAX:
            raise RuntimeError(f"fusenet {n} параметров, лимит {N_PARAMS_MAX}")
        for mod in self.modules():
            if isinstance(mod, nn.Conv2d) and mod.kernel_size != (3, 3):
                raise RuntimeError(f"ядро {mod.kernel_size}, нужно 3×3")

    def forward(self, x):
        return self.body(x)


def n_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def fuse_input(winner_s, X, knots) -> np.ndarray:
    """Каналы слияния. Невалидные клетки обнулены, в том числе маска valid."""
    ch0 = np.nan_to_num(np.asarray(X[0], np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    ch1 = np.nan_to_num(np.asarray(X[1], np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    valid = (np.asarray(X[2], np.float32) > 0.5).astype(np.float32)
    score = np.nan_to_num(np.asarray(winner_s, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    cdf = apply_cdf(np.abs(ch1), knots)
    dt = np.clip(np.abs(ch0), 0.0, CLIP) / CLIP
    feat = np.stack([score, cdf, dt, valid], 0)
    feat *= valid[None]
    return feat


def blend4(winner_s, X, knots) -> np.ndarray:
    """Фиксированная смесь (4×победитель + CDF(|dT_space|)) / 5."""
    ch1 = np.nan_to_num(np.asarray(X[1], np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    valid = np.asarray(X[2], np.float32) > 0.5
    score = np.nan_to_num(np.asarray(winner_s, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    cdf = apply_cdf(np.abs(ch1), knots)
    out = (4.0 * score + cdf) / 5.0
    out = out.astype(np.float32)
    out[~valid] = 0.0
    return out


@torch.no_grad()
def predict_prob(model: nn.Module, feat: np.ndarray) -> np.ndarray:
    """Полная карта. Рецептивное поле 3×3×4 мало, шов тайлов не нужен."""
    was = model.training
    model.eval()
    try:
        x = torch.from_numpy(np.ascontiguousarray(feat, dtype=np.float32))[None]
        logit = model(x)[0, 0]
        prob = torch.sigmoid(logit).numpy().astype(np.float32)
    finally:
        model.train(was)
    valid = feat[3] > 0.5
    prob = prob.copy()
    prob[~valid] = 0.0
    return prob
