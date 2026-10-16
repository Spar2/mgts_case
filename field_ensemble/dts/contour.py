"""Контур события: 16 запросов, 8 вершин, один шаг уточнения.

Энкодер — ранние блоки U-Net победителя (unet_v810ft, e1 и e2), веса заморожены.
Голова не является второй сегментационной головой DTSUNet: выход — вершины и
уверенность запроса. Пиксельный скор — уверенность внутри растеризованного
многоугольника и 0 снаружи. Знак нагрев/охлаждение — отдельная голова запроса.
Реальные трассы сюда не входят. dts/models.py не меняется.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .models import block

N_QUERIES = 16
N_VERTS = 8
TILE = (128, 256)
OVERLAP = 16
WINNER_ENCODER = "artifacts/stage1/unet_v810ft.pt"

# Обход прямоугольника: углы и середины сторон. Нулевые смещения дают рамку.
_CANON = (
    (-0.5, -0.5),
    (-0.5, 0.0),
    (-0.5, 0.5),
    (0.0, 0.5),
    (0.5, 0.5),
    (0.5, 0.0),
    (0.5, -0.5),
    (0.0, -0.5),
)


class FrozenEncoder(nn.Module):
    """e1+e2 победителя, шаг 2. Очаг в 4 клетки ещё виден; e4 его съедает."""

    def __init__(self, w: int = 16):
        super().__init__()
        c0, c1 = w, 2 * w
        self.e1 = block(3, c0, (7, 3))
        self.e2 = block(c0, c1, (7, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        return self.e2(F.max_pool2d(e1, 2))

    def load_winner(self, path: str = WINNER_ENCODER) -> None:
        sd = torch.load(path, map_location="cpu", weights_only=True)
        own = self.state_dict()
        missing = [k for k in own if k not in sd]
        if missing:
            raise RuntimeError(f"в чекпойнте нет {missing[:4]}")
        self.load_state_dict({k: sd[k] for k in own}, strict=True)
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()


class ContourHead(nn.Module):
    """Запросы смотрят на карту e2. Один refine — выборка вдоль рамки и остаток вершин."""

    def __init__(self, cin: int = 32, d: int = 64, nq: int = N_QUERIES, nv: int = N_VERTS):
        super().__init__()
        self.nq = nq
        self.nv = nv
        self.d = d
        self.proj = nn.Conv2d(cin, d, 1)
        self.queries = nn.Parameter(torch.randn(nq, d) * 0.02)
        self.norm = nn.LayerNorm(d)
        self.to_q = nn.Linear(d, d, bias=False)
        self.to_k = nn.Linear(d, d, bias=False)
        self.to_v = nn.Linear(d, d, bias=False)
        self.ff = nn.Sequential(nn.Linear(d, d), nn.ReLU(inplace=True), nn.Linear(d, d))
        self.box = nn.Linear(d, 4)
        self.samp = nn.Linear(d, d)
        self.off = nn.Linear(d, nv * 2)
        self.obj = nn.Linear(d, 1)
        self.hot = nn.Linear(d, 1)
        nn.init.zeros_(self.off.weight)
        nn.init.zeros_(self.off.bias)
        nn.init.zeros_(self.box.weight)
        with torch.no_grad():
            # центр 0.5, размер около 0.2 тайла: вершины стартуют у рамки
            self.box.bias.copy_(torch.tensor([0.0, 0.0, -1.4, -1.4]))
        nn.init.constant_(self.obj.bias, -2.0)
        canon = torch.tensor(_CANON, dtype=torch.float32)
        self.register_buffer("canon", canon, persistent=False)

    def forward(self, feat: torch.Tensor):
        x = self.proj(feat)
        b, d, _, _ = x.shape
        tok = x.flatten(2).transpose(1, 2)
        q = self.norm(self.queries.unsqueeze(0).expand(b, -1, -1))
        attn = torch.softmax(self.to_q(q) @ self.to_k(tok).transpose(1, 2) / (d ** 0.5), dim=-1)
        q = q + attn @ self.to_v(tok)
        q = q + self.ff(q)
        raw = self.box(q)
        cy = torch.sigmoid(raw[..., 0])
        cx = torch.sigmoid(raw[..., 1])
        sh = torch.sigmoid(raw[..., 2]) * 0.90 + 0.02
        sw = torch.sigmoid(raw[..., 3]) * 0.90 + 0.02
        center = torch.stack([cy, cx], -1)
        size = torch.stack([sh, sw], -1)
        base = center.unsqueeze(2) + size.unsqueeze(2) * self.canon
        grid = torch.stack([base[..., 1] * 2 - 1, base[..., 0] * 2 - 1], -1)
        sampled = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)
        sampled = self.samp(sampled.permute(0, 2, 3, 1)).mean(2)
        q2 = q + sampled
        delta = torch.tanh(self.off(q2)).view(b, self.nq, self.nv, 2) * 0.45
        verts = (center.unsqueeze(2) + size.unsqueeze(2) * (self.canon + delta)).clamp(0, 1)
        return self.obj(q2), self.hot(q2), verts


class ContourNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FrozenEncoder()
        self.head = ContourHead()

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, x: torch.Tensor):
        self.encoder.eval()
        with torch.no_grad():
            feat = self.encoder(x)
        return self.head(feat.detach())


def octagon_pixels(ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """8 вершин компонента: ближайшая точка маски к каждому узлу рамки."""
    y0, y1 = float(ys.min()), float(ys.max())
    x0, x1 = float(xs.min()), float(xs.max())
    cy = 0.5 * (y0 + y1)
    cx = 0.5 * (x0 + x1)
    sh = max(y1 - y0, 1.0)
    sw = max(x1 - x0, 1.0)
    if ys.size > 4000:
        sel = np.linspace(0, ys.size - 1, 4000).astype(np.int64)
        ys = ys[sel]
        xs = xs[sel]
    out = np.empty((N_VERTS, 2), np.float32)
    for i, (dy, dx) in enumerate(_CANON):
        d2 = (ys - (cy + dy * sh)) ** 2 + (xs - (cx + dx * sw)) ** 2
        k = int(np.argmin(d2))
        out[i, 0] = ys[k]
        out[i, 1] = xs[k]
    return out


def instances_from_lab(lab: np.ndarray, max_inst: int = N_QUERIES, min_px: int = 8):
    """lab: -1 пусто, 0 фон, 1 нагрев, 2 охлаждение. Вершины нормированы на тайл."""
    import cv2

    H, W = lab.shape
    found = []
    for code, is_hot in ((1, 1.0), (2, 0.0)):
        mask = (lab == code).astype(np.uint8)
        if int(mask.sum()) < min_px:
            continue
        n, comp = cv2.connectedComponents(mask, connectivity=8)
        for k in range(1, n):
            ys, xs = np.where(comp == k)
            if ys.size < min_px:
                continue
            pix = octagon_pixels(ys, xs)
            vn = np.empty_like(pix)
            vn[:, 0] = (pix[:, 0] + 0.5) / H
            vn[:, 1] = (pix[:, 1] + 0.5) / W
            found.append((int(ys.size), is_hot, vn))
    found.sort(key=lambda z: -z[0])
    found = found[:max_inst]
    if not found:
        return np.zeros((0, N_VERTS, 2), np.float32), np.zeros((0,), np.float32)
    verts = np.stack([z[2] for z in found]).astype(np.float32)
    hot = np.asarray([z[1] for z in found], np.float32)
    return verts, hot


def contour_loss(obj, hot, verts, targets, vert_w: float = 4.0) -> torch.Tensor:
    """Венгерское сопоставление запросов с экземплярами тайла. Пустые запросы — фон."""
    from scipy.optimize import linear_sum_assignment

    total = obj.new_zeros(())
    b = obj.shape[0]
    pw = obj.new_tensor([4.0])
    for i in range(b):
        tv, th = targets[i]
        logits = obj[i, :, 0]
        if tv.shape[0] == 0:
            total = total + F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))
            continue
        with torch.no_grad():
            pv = verts[i].detach().cpu().numpy()
            ph = torch.sigmoid(hot[i, :, 0]).detach().cpu().numpy()
            cost = np.abs(pv[:, None] - tv[None]).mean(axis=(2, 3))
            cost = cost + 0.15 * np.abs(ph[:, None] - th[None])
            ri, ci = linear_sum_assignment(cost)
        ri_t = torch.as_tensor(ri, dtype=torch.long)
        tgt_v = torch.from_numpy(tv[ci])
        tgt_h = torch.from_numpy(th[ci])
        vloss = (verts[i, ri_t] - tgt_v).abs().mean()
        hloss = F.binary_cross_entropy_with_logits(hot[i, ri_t, 0], tgt_h)
        obj_t = torch.zeros_like(logits)
        obj_t[ri_t] = 1.0
        oloss = F.binary_cross_entropy_with_logits(logits, obj_t, pos_weight=pw)
        total = total + oloss + hloss + vert_w * vloss
    return total / b


def raster_tile(verts: np.ndarray, conf: np.ndarray, hot: np.ndarray, height: int, width: int):
    """Уверенность внутри многоугольника, 0 снаружи. Перекрытие — максимум уверенности."""
    import cv2

    score = np.zeros((height, width), np.float32)
    sign = np.zeros((height, width), np.float32)
    for q in range(verts.shape[0]):
        c = float(conf[q])
        if c < 1e-4:
            continue
        yy = np.rint(np.clip(verts[q, :, 0] * height - 0.5, 0, height - 1)).astype(np.int32)
        xx = np.rint(np.clip(verts[q, :, 1] * width - 0.5, 0, width - 1)).astype(np.int32)
        if len(np.unique(np.stack([yy, xx], 1), axis=0)) < 3:
            continue
        canvas = np.zeros((height, width), np.uint8)
        cv2.fillPoly(canvas, [np.stack([xx, yy], 1)], 1)
        upd = (canvas > 0) & (c > score)
        if not np.any(upd):
            continue
        score[upd] = c
        sign[upd] = 1.0 if float(hot[q]) > 0.5 else -1.0
    return score, sign


def tile_starts(n: int, tile: int, step: int) -> list[int]:
    if n <= tile:
        return [0]
    starts = list(range(0, n - tile + 1, step))
    last = n - tile
    if starts[-1] != last:
        starts.append(last)
    return starts


@torch.no_grad()
def score_field(model: ContourNet, x: np.ndarray, tile=TILE, overlap: int = OVERLAP, batch: int = 4):
    """x: (3, T, W) float. Тот же тайл, что на обучении: нормировка вершин от размера тайла."""
    model.eval()
    x = np.asarray(x, np.float32)
    _, t_len, w_len = x.shape
    th, tw = tile
    step_t = max(th - 2 * overlap, 1)
    step_w = max(tw - 2 * overlap, 1)
    boxes = [(t0, w0) for t0 in tile_starts(t_len, th, step_t) for w0 in tile_starts(w_len, tw, step_w)]
    score = np.zeros((t_len, w_len), np.float32)
    sign = np.zeros((t_len, w_len), np.float32)
    for i in range(0, len(boxes), batch):
        chunk = boxes[i:i + batch]
        xs, meta = [], []
        for t0, w0 in chunk:
            sl = x[:, t0:t0 + th, w0:w0 + tw]
            ht, wt = sl.shape[1], sl.shape[2]
            if ht != th or wt != tw:
                pad = np.zeros((3, th, tw), np.float32)
                pad[:, :ht, :wt] = sl
                sl = pad
            xs.append(sl)
            meta.append((t0, w0, ht, wt))
        xt = torch.from_numpy(np.stack(xs))
        obj, hot, verts = model(xt)
        conf = torch.sigmoid(obj[:, :, 0]).cpu().numpy()
        hs = torch.sigmoid(hot[:, :, 0]).cpu().numpy()
        vp = verts.cpu().numpy()
        for b, (t0, w0, ht, wt) in enumerate(meta):
            sb, hb = raster_tile(vp[b], conf[b], hs[b], th, tw)
            sb, hb = sb[:ht, :wt], hb[:ht, :wt]
            dest = score[t0:t0 + ht, w0:w0 + wt]
            better = sb > dest
            dest[better] = sb[better]
            sign[t0:t0 + ht, w0:w0 + wt][better] = hb[better]
    return score, sign


def geometry_check() -> None:
    """Прямоугольник и узкая полоса растеризуются обратно в маску. Сеть не нужна."""
    lab = np.zeros((128, 256), np.int8)
    lab[20:61, 40:121] = 1
    lab[70:111, 30:35] = 2
    verts, hot = instances_from_lab(lab)
    if verts.shape != (2, 8, 2) or not np.allclose(hot, [1.0, 0.0]):
        raise RuntimeError(f"экземпляры {verts.shape} {hot}")
    conf = np.ones(2, np.float32)
    score, sign = raster_tile(verts, conf, hot, 128, 256)
    inter = np.logical_and(score > 0, lab == 1).sum()
    union = np.logical_or(score > 0, lab == 1).sum()
    # холодная полоса тоже закрашена; IoU только горячего прямоугольника считаем по его рамке
    box = np.zeros_like(lab, bool)
    box[20:61, 40:121] = True
    # временно один горячий полигон
    s1, _ = raster_tile(verts[:1], conf[:1], hot[:1], 128, 256)
    iou = np.logical_and(s1 > 0, box).sum() / max(np.logical_or(s1 > 0, box).sum(), 1)
    thin = np.zeros_like(lab, bool)
    thin[70:111, 30:35] = True
    s2, h2 = raster_tile(verts[1:], conf[1:], hot[1:], 128, 256)
    iou_t = np.logical_and(s2 > 0, thin).sum() / max(np.logical_or(s2 > 0, thin).sum(), 1)
    if iou < 0.9 or iou_t < 0.7:
        raise RuntimeError(f"растеризация прямоуг {iou:.3f} полоса {iou_t:.3f}")
    if float((h2[thin] < 0).mean()) < 0.9:
        raise RuntimeError("знак охлаждения не попал в полосу")
    _ = inter, union
