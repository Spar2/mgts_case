"""MixStyle (Zhou et al., Domain Generalization with MixStyle, ICLR 2021).

Смешиваются среднее и СКО карты признаков по пакету. Параметров нет.
На eval модуль — тождество, поэтому U-Net с теми же весами совпадает с DTSUNet.
Реальные трассы и их статистики сюда не входят.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from . import models


class MixStyle(nn.Module):
    """Смешение стиля: нормализация по пространству, затем выпуклая смесь статистик.

    Один бросок монеты на прямой проход (вероятность ``p``). ``alpha`` — параметр Beta.
    Статистики отсоединены от графа, как в авторской реализации.
    """

    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6):
        super().__init__()
        if not 0.0 <= float(p) <= 1.0:
            raise ValueError("p must be in [0, 1]")
        if float(alpha) <= 0:
            raise ValueError("alpha must be positive")
        self.p = float(p)
        self.alpha = float(alpha)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.p <= 0.0:
            return x
        if float(torch.rand((), device=x.device)) > self.p:
            return x
        b = int(x.shape[0])
        if b < 2:
            return x
        mu = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_norm = (x - mu) / sig
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((b, 1, 1, 1))
        lam = lam.to(device=x.device, dtype=x.dtype)
        perm = torch.randperm(b, device=x.device)
        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = lam * mu + (1.0 - lam) * mu2
        sig_mix = lam * sig + (1.0 - lam) * sig2
        return x_norm * sig_mix + mu_mix


class MixStyleUNet(nn.Module):
    """DTSUNet ширины 16. MixStyle стоит после блоков e1 и e2, нигде больше.

    Веса ``unet_v810ft`` грузятся в ``backbone`` один в один. На eval MixStyle молчит.
    """

    def __init__(self, w: int = 16, p: float = 0.5, alpha: float = 0.1):
        super().__init__()
        self.backbone = models.DTSUNet(w=int(w))
        self.mix1 = MixStyle(p=p, alpha=alpha)
        self.mix2 = MixStyle(p=p, alpha=alpha)

    def load_init(self, path) -> None:
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.backbone.load_state_dict(state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        net = self.backbone
        e1 = self.mix1(net.e1(x))
        e2 = self.mix2(net.e2(F.max_pool2d(e1, 2)))
        e3 = net.e3(F.max_pool2d(e2, 2))
        e4 = net.gate(net.e4(F.max_pool2d(e3, 2)))
        d3 = net.d3(torch.cat([net.u3(e4), e3], 1))
        d2 = net.d2(torch.cat([net.u2(d3), e2], 1))
        d1 = net.d1(torch.cat([net.u1(d2), e1], 1))
        return net.head(d1)
