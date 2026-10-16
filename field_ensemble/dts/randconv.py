"""RandConv (Xu et al., ICLR 2021) по входным каналам.

Фильтр k×k, k ∈ {1, 3}, веса Θ ~ N(0, 1/(C k²)), свёртка с сохранением размера.
Смесь α x + (1−α) (x ∗ Θ), α ~ U(0, 1), отдельно на каждый образец.
Реальные трассы и их статистики не читаются.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

KERNELS = (1, 3)


def _one(x: torch.Tensor, generator: torch.Generator, kernel: int | None, alpha: float | None) -> torch.Tensor:
    """x: (1, C, H, W)."""
    c = int(x.shape[1])
    if kernel is None:
        kernel = KERNELS[int(torch.randint(0, len(KERNELS), (1,), generator=generator).item())]
    if kernel not in KERNELS:
        raise ValueError(f"ядро {kernel} не из {KERNELS}")
    std = 1.0 / math.sqrt(c * kernel * kernel)
    weight = torch.randn(c, c, kernel, kernel, generator=generator, dtype=x.dtype) * std
    rc = F.conv2d(x, weight, padding=kernel // 2)
    if alpha is None:
        alpha = float(torch.rand(1, generator=generator).item())
    a = float(alpha)
    if not 0.0 <= a <= 1.0:
        raise ValueError("alpha должен быть в [0, 1]")
    return a * x + (1.0 - a) * rc


def apply_batch(x: torch.Tensor, generator: torch.Generator, kernel: int | None = None,
                alpha: float | None = None) -> torch.Tensor:
    """x: (N, C, H, W) → тот же размер. Кадр получает свой фильтр и свой α."""
    if x.ndim != 4:
        raise ValueError("ожидался батч NCHW")
    return torch.cat([_one(x[i:i + 1], generator, kernel, alpha) for i in range(x.shape[0])], 0)


def consistency_kl(logits: torch.Tensor) -> torch.Tensor:
    """KL каждого вида к среднему softmax. logits: (V, N, C, H, W), как в статье (три вида)."""
    if logits.ndim != 5 or logits.shape[0] < 2:
        raise ValueError("нужны хотя бы два вида (V, N, C, H, W)")
    logp = F.log_softmax(logits, dim=2)
    p = logp.exp()
    bar = p.mean(dim=0, keepdim=True).clamp_min(1e-6)
    return (p * (logp - bar.log())).sum(dim=2).mean()


def self_check() -> None:
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 3, 8, 8)
    y = apply_batch(x, g)
    if y.shape != x.shape or not torch.isfinite(y).all():
        raise RuntimeError("форма или NaN")
    g2 = torch.Generator().manual_seed(1)
    same = apply_batch(x, g2, kernel=3, alpha=1.0)
    if not torch.allclose(same, x):
        raise RuntimeError("alpha=1 должен оставлять вход")
    changed = apply_batch(x, torch.Generator().manual_seed(2), kernel=1, alpha=0.0)
    if torch.allclose(changed, x):
        raise RuntimeError("случайная 1×1 не должна быть тождеством")
    logits = torch.zeros(3, 2, 3, 4, 4)
    if float(consistency_kl(logits)) != 0.0:
        raise RuntimeError("одинаковые логиты дают KL 0")
