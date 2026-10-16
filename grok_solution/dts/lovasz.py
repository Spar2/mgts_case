"""Lovász-hinge и Lovász-softmax (Berman, Triki, Blaschko, CVPR 2018).

Прямое продолжение Жаккара на вещественные ошибки: пиксели сортируются по
убыванию ошибки, градиент расширения — разности Жаккара по Алг. 1 статьи.
Невалидные пиксели (y < 0) выбрасываются до сортировки. Среднее — по
изображениям батча; у softmax ещё по классам, которые есть на этом изображении.

Бинарный hinge смотрит на передний план детектора: метка y > 0, логит —
log-odds оценки s = 1 − p(фон). Softmax — три класса {фон, нагрев, охлаждение}.
"""
from __future__ import annotations

import torch


def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Градиент расширения Ловаса по уже отсортированным меткам 0/1. Константа для backward."""
    gt = gt_sorted.detach().to(dtype=torch.float32)
    n = int(gt.numel())
    if n == 0:
        return gt
    gts = gt.sum()
    intersection = gts - gt.cumsum(0)
    union = gts + (1.0 - gt).cumsum(0)
    jaccard = 1.0 - intersection / union
    if n > 1:
        # Не вычитать срезы на месте: у пересекающихся view PyTorch затирает уже записанное.
        jaccard = torch.cat([jaccard[:1], jaccard[1:] - jaccard[:-1]])
    return jaccard


def lovasz_hinge_flat(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """logits — логиты переднего плана, labels — 0/1 той же длины."""
    if logits.numel() == 0:
        return logits.sum() * 0.0
    signs = labels.to(dtype=logits.dtype) * 2.0 - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    grad = lovasz_grad(labels.to(dtype=torch.float32)[perm])
    return torch.dot(torch.relu(errors_sorted), grad.to(dtype=errors_sorted.dtype))


def lovasz_softmax_flat(probas: torch.Tensor, labels: torch.Tensor, classes: str = "present") -> torch.Tensor:
    """probas (P, C) в [0, 1], labels (P) — индексы классов."""
    if probas.numel() == 0:
        return probas.sum() * 0.0
    losses = []
    for c in range(probas.shape[1]):
        fg = (labels == c).to(dtype=probas.dtype)
        if classes == "present" and not bool(fg.detach().any()):
            continue
        errors = (fg - probas[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        grad = lovasz_grad(fg[perm])
        losses.append(torch.dot(errors_sorted, grad.to(dtype=errors_sorted.dtype)))
    if not losses:
        return probas.sum() * 0.0
    return torch.stack(losses).mean()


def _per_image(logit_or_prob, y, flat_fn, prob: bool):
    losses = []
    for i in range(y.shape[0]):
        lab = y[i].reshape(-1)
        valid = lab >= 0
        labels = lab[valid]
        if prob:
            flat = logit_or_prob[i].reshape(logit_or_prob.shape[1], -1).T
            losses.append(flat_fn(flat[valid], labels))
        else:
            losses.append(flat_fn(logit_or_prob[i].reshape(-1)[valid], (labels > 0).to(dtype=logit_or_prob.dtype)))
    return torch.stack(losses).mean()


def lovasz_hinge(logit_fg: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """logit_fg (B, H, W), y (B, H, W). Передний план — y > 0, ignore — y < 0."""
    return _per_image(logit_fg, y, lovasz_hinge_flat, prob=False)


def lovasz_softmax(logit: torch.Tensor, y: torch.Tensor, classes: str = "present") -> torch.Tensor:
    """logit (B, C, H, W). Классы 'present' — только те, что есть на изображении."""
    prob = torch.softmax(logit, dim=1)
    return _per_image(prob, y, lambda p, lab: lovasz_softmax_flat(p, lab, classes=classes), prob=True)


def foreground_logit(logit: torch.Tensor) -> torch.Tensor:
    """log(s / (1 − s)), s = 1 − p(фон). Совпадает с бинарным логитом детектора."""
    p0 = torch.softmax(logit, dim=1)[:, 0].clamp(1e-6, 1.0 - 1e-6)
    return torch.log(1.0 - p0) - torch.log(p0)


def lovasz_iou_loss(logit: torch.Tensor, y: torch.Tensor, hinge_weight: float = 0.5, softmax_weight: float = 0.5):
    """Смесь hinge по переднему плану и softmax по трём классам. Возвращает loss, hinge, softmax."""
    hinge = lovasz_hinge(foreground_logit(logit), y)
    soft = lovasz_softmax(logit, y, classes="present")
    return hinge_weight * hinge + softmax_weight * soft, hinge, soft


def self_check() -> None:
    """Сверка Алг. 1 и нулевой потери на идеальном прогнозе. Падает, если формула уехала."""
    cases = {
        (1.0,): [1.0],
        (0.0,): [1.0],
        (1.0, 0.0): [1.0, 0.0],
        (1.0, 1.0, 0.0): [0.5, 0.5, 0.0],
        (0.0, 0.0, 1.0): [0.5, 1.0 / 6.0, 1.0 / 3.0],
    }
    for gt, expect in cases.items():
        got = lovasz_grad(torch.tensor(gt, dtype=torch.float32))
        ref = torch.tensor(expect, dtype=torch.float32)
        if not torch.allclose(got, ref, atol=1e-6):
            raise RuntimeError(f"lovasz_grad {gt} -> {got.tolist()} != {expect}")

    logit = torch.zeros(2, 3, 4, 4)
    y = torch.zeros(2, 4, 4, dtype=torch.long)
    y[0, :2, :] = 1
    y[1, :, :2] = 2
    y[0, 0, 0] = -1
    # Идеальные вероятности: ошибка softmax нулевая на валидных пикселях.
    prob = torch.zeros_like(logit)
    for b in range(2):
        for c in range(3):
            prob[b, c][y[b] == c] = 1.0
    prob[0, :, 0, 0] = 1.0 / 3.0
    soft = _per_image(prob, y, lambda p, lab: lovasz_softmax_flat(p, lab), prob=True)
    if float(soft) > 1e-6:
        raise RuntimeError(f"идеальный softmax не нулевой: {float(soft)}")

    # Большой зазор hinge на верных знаках — потеря около нуля.
    signs = torch.where(y > 0, torch.ones_like(logit[:, 0]), -torch.ones_like(logit[:, 0]))
    signs = torch.where(y >= 0, signs, torch.zeros_like(signs))
    wide = signs * 20.0
    hinge = lovasz_hinge(wide, y)
    if float(hinge) > 1e-5:
        raise RuntimeError(f"идеальный hinge не нулевой: {float(hinge)}")

    bad = torch.zeros(1, 3, 6, 6, requires_grad=True)
    yb = torch.randint(0, 3, (1, 6, 6))
    loss, h, s = lovasz_iou_loss(bad, yb)
    loss.backward()
    if bad.grad is None or not torch.isfinite(bad.grad).all() or not torch.isfinite(loss):
        raise RuntimeError("градиент Lovász не конечный")
    if float(h.detach()) < 0 or float(s.detach()) < 0:
        raise RuntimeError("отрицательная потеря")
