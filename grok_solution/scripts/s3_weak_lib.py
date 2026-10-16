"""Дообучение U-Net на слабые события (≤2 °C).

Потеря: focal + Tversky, вес пикселя растёт при малой амплитуде события.
Curriculum сужает потолок амплитуды и усиливает выборку малых событий.
Порог FAR считается по хвосту фона тем же правилом, что dts.metrics.far_threshold.
Реальные трассы сюда не входят.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def curriculum(epoch: int, n_epochs: int) -> dict:
    """От более широких событий к амплитуде ≤2 °C. f — smoothstep 0→1."""
    n = max(int(n_epochs), 1)
    e = min(max(int(epoch), 0), n - 1)
    f = 0.0 if n == 1 else e / (n - 1)
    f = f * f * (3.0 - 2.0 * f)
    return {
        "amp_hi": float(8.0 * (1.0 - f) + 2.0 * f),
        "p_small": float(0.20 + 0.60 * f),
        "gamma": float(0.40 + 1.20 * f),
        "rank_w": float(0.25 + 0.75 * f),
    }


def amp_gain(amp, gamma: float, amp_ref: float = 2.0, lo: float = 1.0, hi: float = 4.0):
    """Множитель потери. Амплитуда ≥ amp_ref остаётся 1, меньшая растёт как (ref/a)^γ.

    Сильные события не занижаются: иначе IoU на крупных зонах проседает.
    """
    a = np.maximum(np.asarray(amp, np.float64), 0.2)
    rel = float(amp_ref) / np.minimum(a, float(amp_ref))
    return np.clip(rel ** float(gamma), lo, hi)


def boxes_from_axes(x, t, records) -> list[dict]:
    """Боксы в индексах сетки. Поля как у metrics.event_pixels: ±0.25 м и ±120 с."""
    x = np.asarray(x)
    t = np.asarray(t)
    out = []
    for r in records:
        if r.get("kind", "event") != "event":
            continue
        ix = np.flatnonzero((x >= r["dist_start_m"] - 0.25) & (x <= r["dist_end_m"] + 0.25))
        it = np.flatnonzero((t >= r["ts"] - 120.0) & (t <= r["te"] + 120.0))
        if ix.size == 0 or it.size == 0:
            continue
        out.append(dict(
            t0=int(it[0]), t1=int(it[-1]) + 1,
            x0=int(ix[0]), x1=int(ix[-1]) + 1,
            amp=float(r["amp"]), mask=int(r["mask"]),
        ))
    return out


def paint_weights(ev, boxes, gamma: float) -> np.ndarray:
    """Карта весов: 1 на фоне, amp_gain на пикселях события с тем же mask_value."""
    w = np.ones(ev.shape, np.float32)
    for b in boxes:
        t0, t1 = max(int(b["t0"]), 0), min(int(b["t1"]), ev.shape[0])
        x0, x1 = max(int(b["x0"]), 0), min(int(b["x1"]), ev.shape[1])
        if t1 <= t0 or x1 <= x0:
            continue
        m = ev[t0:t1, x0:x1] == int(b["mask"])
        if not np.any(m):
            continue
        g = float(np.reshape(amp_gain(b["amp"], gamma), ()))
        sl = w[t0:t1, x0:x1]
        sl[m] = np.maximum(sl[m], np.float32(g))
    return w


def _rank_term(fg: torch.Tensor, y: torch.Tensor, amp_w: torch.Tensor, target: float) -> torch.Tensor:
    """Хинж: верхние 30% пикселей слабого события дотягиваются до target.

    Локальный хвост фона слишком низкий, а softplus продолжает толкать логиты,
    когда пик уже выше тайла, и вместе с ними растёт глобальный порог FAR.
    """
    losses = []
    for b in range(fg.shape[0]):
        pos = (y[b] > 0) & (amp_w[b] > 1.05)
        if int(pos.sum()) < 4:
            continue
        pv = fg[b][pos]
        kpos = max(1, int(np.ceil(0.30 * pv.numel())))
        ps = pv.topk(min(kpos, pv.numel())).values.mean()
        losses.append(F.relu(target - ps))
    if not losses:
        return fg.new_zeros(())
    return torch.stack(losses).mean()


def bg_anchor(fg: torch.Tensor, ref_fg: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Фон не должен стать увереннее, чем у замороженного учителя. События не якорятся."""
    bg = y == 0
    excess = F.relu(fg - ref_fg)
    m = bg.to(fg.dtype)
    base = (excess.pow(2) * m).sum() / m.sum().clamp(min=1.0)
    tails = []
    for b in range(fg.shape[0]):
        if int(bg[b].sum()) < 64:
            continue
        e = excess[b][bg[b]]
        k = max(4, int(round(0.005 * int(e.numel()))))
        tails.append(e.topk(min(k, int(e.numel()))).values.mean())
    tail = torch.stack(tails).mean() if tails else fg.new_zeros(())
    return base + tail


def focal_tversky(logit, y, amp_w, cw, *, gamma: float = 2.0, alpha: float = 0.7, beta: float = 0.3,
                  rank_w: float = 0.0, margin: float = 0.96):
    """focal (Lin 2017) + Tversky (α на FN) с весом амплитуды + ранжирование хвоста.

    Невалидные пиксели (y<0) в суммы не входят. FP Tversky не усиливается весом,
    FN — усиливается, чтобы слабое событие не терялось в сильных.
    """
    logp = F.log_softmax(logit, dim=1)
    valid = y >= 0
    y_safe = y.clamp(min=0)
    logpt = logp.gather(1, y_safe.unsqueeze(1)).squeeze(1)
    pt = logpt.exp()
    w = cw.to(dtype=logit.dtype, device=logit.device)[y_safe] * amp_w.to(dtype=logit.dtype)
    focal = -(w * (1.0 - pt).pow(gamma) * logpt)
    focal = torch.where(valid, focal, torch.zeros_like(focal))
    denom = torch.where(valid, w, torch.zeros_like(w)).sum().clamp(min=1.0)
    focal_m = focal.sum() / denom

    p = torch.softmax(logit, dim=1)
    m = valid.to(logit.dtype)
    fg = 1.0 - p[:, 0]
    t = (y > 0).to(logit.dtype)
    aw = amp_w.to(dtype=logit.dtype) * m
    tp = (fg * t * aw).sum()
    fp = (fg * (1.0 - t) * m).sum()
    fn = ((1.0 - fg) * t * aw).sum()
    tv = 1.0 - (tp + 1.0) / (tp + alpha * fn + beta * fp + 1.0)
    loss = focal_m + tv
    rank = fg.new_zeros(())
    if rank_w > 0:
        rank = _rank_term(fg, y, amp_w, margin)
        loss = loss + float(rank_w) * rank
    return loss, {"focal": float(focal_m.detach()), "tversky": float(tv.detach()), "rank": float(rank.detach())}


class ScoreTail:
    """Квантиль фона для FAR. Точное значение, если квантиль выше cutoff, иначе гистограмма."""

    def __init__(self, cutoff: float = 0.5, bins: int = 100_000):
        self.cutoff = float(cutoff)
        self.bins = int(bins)
        self.hist = np.zeros(self.bins, np.int64)
        self.total = 0
        self.chunks: list[np.ndarray] = []
        self.last_source = ""

    def add(self, scores) -> None:
        s = np.asarray(scores, np.float32).ravel()
        if s.size == 0:
            return
        s = s[np.isfinite(s)]
        self.total += int(s.size)
        if s.size == 0:
            return
        clipped = np.clip(s, 0.0, 1.0)
        idx = np.minimum((clipped * self.bins).astype(np.int64), self.bins - 1)
        self.hist += np.bincount(idx, minlength=self.bins)
        hi = s[s > self.cutoff]
        if hi.size:
            self.chunks.append(np.ascontiguousarray(hi))

    def threshold(self, far: float) -> float:
        n = self.total
        if n <= 0:
            self.last_source = "empty"
            return 1.0
        k = int(np.ceil((1.0 - far) * n)) - 1
        k = int(np.clip(k, 0, n - 1))
        kept = np.concatenate(self.chunks) if self.chunks else np.empty(0, np.float32)
        n_below = n - int(kept.size)
        if kept.size and k >= n_below:
            kept.sort()
            self.last_source = "exact"
            return float(kept[k - n_below]) + 1e-12
        c = np.cumsum(self.hist)
        idx = int(np.searchsorted(c, k + 1, side="left"))
        idx = min(max(idx, 0), self.bins - 1)
        self.last_source = "hist"
        return float(idx / self.bins) + 1e-12


def new_acc() -> dict:
    return dict(tot=np.zeros(3, np.int64), hot=np.zeros(3, np.int64), cold=np.zeros(3, np.int64),
                far_n=0, far_d=0, per_seq={}, events=[])


def update_acc(acc, s, h, ev, valid, nu, x, t, seq_id, labels, thr: float) -> None:
    from dts import metrics as M
    pred = (s >= thr) & valid
    truth = ev > 0
    c = M.confusion(pred, truth, valid)
    acc["tot"] += np.asarray(c, np.int64)
    acc["per_seq"][seq_id] = c
    acc["hot"] += np.asarray(M.confusion(pred & (h > 0), (ev >= 1) & (ev <= 4), valid), np.int64)
    acc["cold"] += np.asarray(M.confusion(pred & (h <= 0), ev >= 5, valid), np.int64)
    b = valid & (ev == 0) & (nu == 0)
    acc["far_n"] += int((pred & b).sum())
    acc["far_d"] += int(b.sum())
    for r in M.label_boxes(labels, seq_id).itertuples():
        if r.kind != "event":
            continue
        px = M.event_pixels(r, x, t, ev)
        if int(px.sum()) == 0:
            continue
        frac = float(pred[px].mean())
        acc["events"].append(dict(
            seq_id=seq_id, label=str(r.label), amp=float(r.amp_obs_C),
            amp_bin=M.amp_bin(float(r.amp_obs_C)), frac=frac, hit=bool(frac >= 0.3),
            tp=int((pred & px).sum()), fn=int((~pred & px).sum()),
        ))


def _agg(g) -> dict:
    tp = int(g.tp.sum())
    fn = int(g.fn.sum())
    return dict(n=int(len(g)), recall=float(g.hit.mean()), pixel_recall=float(tp / max(tp + fn, 1)))


def finalize(acc) -> dict:
    import pandas as pd
    from dts import metrics as M
    res = dict(far=float(acc["far_n"] / max(acc["far_d"], 1)), **M.prf(*map(int, acc["tot"])),
               hot=M.prf(*map(int, acc["hot"])), cold=M.prf(*map(int, acc["cold"])))
    if acc["per_seq"]:
        res["iou_ci95"] = M.bootstrap_ci(acc["per_seq"], sorted(acc["per_seq"]))
    E = pd.DataFrame(acc["events"])
    if len(E):
        res["events_all"] = _agg(E)
        res["by_amp"] = {str(k): _agg(g) for k, g in E.groupby("amp_bin")}
        res["by_label"] = {str(k): _agg(g) for k, g in E.groupby("label")}
        weak = E[E.amp <= 2]
        res["weak_events"] = _agg(weak) if len(weak) else dict(n=0, recall=None, pixel_recall=None)
    else:
        res["events_all"] = dict(n=0, recall=None, pixel_recall=None)
        res["by_amp"] = {}
        res["by_label"] = {}
        res["weak_events"] = dict(n=0, recall=None, pixel_recall=None)
    return res


def bg_mask(ev, valid, nu) -> np.ndarray:
    return valid & (ev == 0) & (nu == 0)


def crossing_level(xs, ys, level: float = 0.5):
    """Первое восходящее пересечение level. По x — логарифм, если оба конца > 0.

    xs и ys — соседние бины (центр бина и recall). None, если пересечения нет.
    """
    xs = np.asarray(xs, np.float64)
    ys = np.asarray(ys, np.float64)
    if xs.size < 2:
        return None
    for i in range(int(xs.size) - 1):
        y0, y1 = float(ys[i]), float(ys[i + 1])
        if not (y0 < level <= y1):
            continue
        span = y1 - y0
        if span <= 0:
            continue
        t = (level - y0) / span
        x0, x1 = float(xs[i]), float(xs[i + 1])
        if x0 > 0 and x1 > 0:
            return float(np.exp(np.log(x0) + t * (np.log(x1) - np.log(x0))))
        return float(x0 + t * (x1 - x0))
    return None


def predict_sh(model, X) -> tuple[np.ndarray, np.ndarray]:
    from dts import models
    pr = models.predict_full(model, X)
    return (1.0 - pr[0]).astype(np.float32), (pr[1] - pr[2]).astype(np.float32)
