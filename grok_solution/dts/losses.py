"""Потери сегментации stage 1. По умолчанию train_stage1 их не вызывает.

ce — взвешенный cross-entropy (веса 1, 3, 3), ignore_index=-1.
focal — тот же вес и γ=2 (Lin et al., 2017).
tversky_ce — тот же CE плюс Tversky по переднему плану (α=0.7, β=0.3):
FN штрафуется сильнее FP. Передний план — как в прежнем dice: y>0, p=1-p(фон).
Сумма пишется через '+': focal+tversky_ce.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

TERMS = ("ce", "focal", "tversky_ce")


def parse_spec(spec: str) -> list[str]:
    parts = [p.strip() for p in str(spec).replace("+", ",").split(",") if p.strip()]
    bad = [p for p in parts if p not in TERMS]
    if not parts or bad:
        raise ValueError(f"loss: ожидается комбинация {TERMS}, получено {spec!r}")
    return parts


def weighted_ce(logit, y, cw, wk=None, weak_weight=0.0):
    if wk is not None and weak_weight > 0:
        pix = F.cross_entropy(logit, y, weight=cw, ignore_index=-1, reduction="none")
        w = (1 + weak_weight * wk) * (y >= 0)
        return (pix * w).sum() / w.sum().clamp(min=1)
    return F.cross_entropy(logit, y, weight=cw, ignore_index=-1)


def focal(logit, y, cw, gamma=2.0, wk=None, weak_weight=0.0):
    """Среднее по валидным пикселям: -w (1-p_t)^γ log p_t. Класс с ignore не входит."""
    logp = F.log_softmax(logit, 1)
    valid = y >= 0
    y_safe = y.clamp(min=0)
    logpt = logp.gather(1, y_safe.unsqueeze(1)).squeeze(1)
    pt = logpt.exp()
    w = cw.to(device=logit.device, dtype=logit.dtype)[y_safe]
    if wk is not None and weak_weight > 0:
        w = w * (1 + weak_weight * wk)
    loss = -(w * (1 - pt).pow(gamma) * logpt)
    loss = torch.where(valid, loss, torch.zeros_like(loss))
    return loss.sum() / valid.sum().clamp(min=1).to(loss.dtype)


def tversky(logit, y, alpha=0.7, beta=0.3):
    """1 - TI по переднему плану. Невалидные пиксели (y<0) в суммы не входят."""
    p = torch.softmax(logit, 1)
    m = (y >= 0).float()
    fg = 1 - p[:, 0]
    t = (y > 0).float()
    tp = (fg * t * m).sum()
    fp = (fg * (1 - t) * m).sum()
    fn = ((1 - fg) * t * m).sum()
    return 1 - (tp + 1) / (tp + alpha * fn + beta * fp + 1)


def compute(spec, logit, y, cw, wk=None, weak_weight=0.0):
    total = logit.new_zeros(())
    for term in parse_spec(spec):
        if term == "ce":
            total = total + weighted_ce(logit, y, cw, wk, weak_weight)
        elif term == "focal":
            total = total + focal(logit, y, cw, wk=wk, weak_weight=weak_weight)
        else:
            total = total + weighted_ce(logit, y, cw, wk, weak_weight) + tversky(logit, y)
    return total
