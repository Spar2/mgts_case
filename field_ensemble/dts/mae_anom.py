"""Маскированный реконструктор фона для synth-only детекции.

Это не поле скорости и не rectified flow. Сеть видит каналы dT только там,
где маска открыта, и учится восстановить скрытые клетки по соседнему контексту.
На инференсе каждая колонка оценивается лишь в тех фазах, где она скрыта,
поэтому значение этой колонки в ответ не копируется.

Канал 0 (dT_time) на фоне слабо коррелирован по x: оптимум восстановления близок к нулю,
а хвост |dT| упирается в клип. Канал 1 (dT_space) пространственно коррелирован,
его и восстанавливаем как величину скора. Знак нагрев/холод берётся из остатка
канала 0 — знака температуры относительно реконструкции.

Обучение только на синтетике. Реальные трассы этот модуль не читает.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

WIDTH_MAX = 16
BAND = 8
RADIUS = 1
TILE = (128, 256)
OVERLAP = 32
SCORE_CHANNEL = 1
SIGN_CHANNEL = 0
LOSS_W = (0.35, 1.0)
WINNER_VAL = 0.38671235821914546
WINNER_TEST = 0.4052762065230473
WINNER_V10 = 0.4038977883892501
WINNER_V11 = 0.40390028295846503
NESTED_GAIN = 0.003
NESTED_BAR = 0.3867 + 0.003
CANDIDATES = ("ens_mean4", "ens_mean", "ens_max", "mae")
PROTOCOL_TEXT = (
    "Принять только если val IoU > 0.38671235821914546, "
    "вложенный прирост k=5 не меньше 0.003, min(k=2, k=5) > 0.3897, "
    "и v10/v11 test не хуже победителя. "
    "Порог FAR 1e-3 считается по фону v8 val. Реальные трассы не участвуют в обучении и пороге. "
    "Обучение — синтетика v8+v10, преимущественно фон."
)


def _conv(cin, cout, k=3, stride=1, padding=None, dilation=1):
    if padding is None:
        rad = k // 2 if isinstance(k, int) else None
        if rad is None:
            padding = tuple(dilation * (kk // 2) for kk in k)
        else:
            padding = dilation * rad
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, stride=stride, padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


def channel_width(model: nn.Module) -> int:
    """Максимальная ширина свёрток. Входной conv с 4 каналами не считается шириной сети."""
    width = 0
    for mod in model.modules():
        if isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d)):
            width = max(width, int(mod.out_channels))
            if int(mod.in_channels) > 4:
                width = max(width, int(mod.in_channels))
    return int(width)


class MaskedReconstructor(nn.Module):
    """Лёгкий энкодер-декодер. Все свёртки ширины <= 16, сквозные связи складываются, не склеиваются."""

    def __init__(self, width: int = 16):
        super().__init__()
        if int(width) > WIDTH_MAX or int(width) < 4:
            raise ValueError(f"ширина {width} вне 4..{WIDTH_MAX}")
        w = int(width)
        self.width = w
        self.e1 = nn.Sequential(_conv(4, w, k=(5, 3), padding=(2, 1)), _conv(w, w, 3))
        self.e2 = nn.Sequential(_conv(w, w, 3, stride=2, padding=1), _conv(w, w, 3))
        self.e3 = nn.Sequential(_conv(w, w, 3, stride=2, padding=1), _conv(w, w, 3))
        self.up2 = nn.ConvTranspose2d(w, w, 2, stride=2)
        self.d2 = _conv(w, w, 3)
        self.up1 = nn.ConvTranspose2d(w, w, 2, stride=2)
        self.d1 = _conv(w, w, 3)
        self.ctx = nn.Sequential(
            _conv(w, w, 3, padding=2, dilation=2),
            _conv(w, w, 3, padding=4, dilation=4),
        )
        self.head = nn.Conv2d(w, 2, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        if channel_width(self) > WIDTH_MAX:
            raise RuntimeError(f"ширина сети {channel_width(self)}")

    @staticmethod
    def _add(up, skip):
        if up.shape[-2:] != skip.shape[-2:]:
            up = F.interpolate(up, size=skip.shape[-2:], mode="nearest")
        return up + skip

    def forward(self, ch, valid, mask):
        """ch (B,2,H,W), valid/mask (B,1,H,W). Скрытые клетки обнуляются до первой свёртки."""
        if valid.dim() == 3:
            valid = valid.unsqueeze(1)
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        vis = (1.0 - mask) * valid
        x = torch.cat([ch * vis, vis, mask], 1)
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        d2 = self.d2(self._add(self.up2(e3), e2))
        d1 = self.d1(self._add(self.up1(d2), e1))
        return self.head(self.ctx(d1)) * valid


def spatial_band_mask(h: int, w: int, band: int, offset: int, parity: int) -> np.ndarray:
    """1 = скрыто. Полоса по времени целиком, шаг по x. Событие уже полосы не видно соседям."""
    xs = np.arange(w, dtype=np.int32)
    col = (((xs + int(offset)) // int(band)) % 2) == int(parity)
    mask = np.zeros((h, w), np.float32)
    mask[:, col] = 1.0
    return mask


def infer_masks(h: int, w: int, band: int = BAND) -> list[np.ndarray]:
    """Четыре фазы: две решётки (сдвиг на полполосы) и две чётности. Каждая колонка скрыта дважды."""
    out = []
    for offset in (0, int(band) // 2):
        for parity in (0, 1):
            out.append(spatial_band_mask(h, w, band, offset, parity))
    return out


def interior_mask(mask: np.ndarray, radius: int = RADIUS) -> np.ndarray:
    """Скрытые клетки, у которых в радиусе по x нет открытой. Край дыры в скор не идёт."""
    vis = mask < 0.5
    bad = vis.copy()
    for d in range(1, int(radius) + 1):
        bad[:, d:] |= vis[:, :-d]
        bad[:, :-d] |= vis[:, d:]
    return ((mask > 0.5) & ~bad).astype(np.float32)


def sample_train_mask(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """В основном те же полосы, что на инференсе. Иногда шире дыра или прямоугольники MAE."""
    if float(rng.random()) < 0.8:
        band = int(rng.choice(np.array([4, 8, 16])))
        band = min(band, max(2, w // 2))
        offset = int(rng.integers(0, band))
        if float(rng.random()) < 0.75:
            return spatial_band_mask(h, w, band, offset, int(rng.integers(0, 2)))
        xs = np.arange(w, dtype=np.int32)
        group = ((xs + offset) // band) % 4
        keep = int(rng.integers(0, 4))
        mask = np.ones((h, w), np.float32)
        mask[:, group == keep] = 0.0
        if mask.min() == 1.0:
            mask[:, 0] = 0.0
        return mask
    ph, pw = 8, 16
    gh = (h + ph - 1) // ph
    gw = (w + pw - 1) // pw
    drop = rng.random((gh, gw)) < 0.6
    if not bool(drop.any()):
        drop[0, 0] = True
    if bool(drop.all()):
        drop[0, 0] = False
    up = np.repeat(np.repeat(drop.astype(np.float32), ph, 0), pw, 1)
    return up[:h, :w].astype(np.float32)


def mae_loss(hat, target, mask, valid, loss_w=LOSS_W):
    if valid.dim() == 3:
        valid = valid.unsqueeze(1)
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    weight = hat.new_tensor(loss_w).view(1, 2, 1, 1)
    sel = mask * valid
    err = (hat - target).abs() * weight * sel
    denom = (sel.sum() * float(sum(loss_w))).clamp(min=1.0)
    return err.sum() / denom


def training_loss(model, ch, valid, mask, sigma: float = 0.15):
    """Шум только на открытом контексте. Цель — чистые каналы в скрытых клетках."""
    if sigma > 0:
        vis = (1.0 - mask) * valid
        ch_in = ch + sigma * torch.randn_like(ch) * vis
    else:
        ch_in = ch
    hat = model(ch_in, valid, mask)
    return mae_loss(hat, ch, mask, valid)


def _starts(n: int, tile: int, step: int) -> list[int]:
    if n <= tile:
        return [0]
    starts = list(range(0, n - tile + 1, step))
    last = n - tile
    if starts[-1] != last:
        starts.append(last)
    return starts


def reconstruct(model: nn.Module, ch: np.ndarray, valid: np.ndarray,
                tile=TILE, overlap: int = OVERLAP, band: int = BAND, radius: int = RADIUS,
                tile_batch: int = 2) -> np.ndarray:
    """hat (2,H,W). В каждой клетке среднее только по фазам, где клетка скрыта."""
    was = model.training
    model.eval()
    ch = np.nan_to_num(np.asarray(ch, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    ch = np.clip(ch, -12.0, 12.0)
    valid = (np.asarray(valid, np.float32) > 0.5).astype(np.float32)
    if ch.ndim != 3 or ch.shape[0] != 2:
        raise ValueError(f"ch {ch.shape}")
    _, height, width = ch.shape
    if valid.shape != (height, width):
        raise ValueError("valid shape")
    th, tw = int(tile[0]), int(tile[1])
    if th % 4 or tw % 4:
        raise ValueError("tile должен делиться на 4")
    phases = [(m, interior_mask(m, radius)) for m in infer_masks(height, width, band)]
    nphase = len(phases)
    if overlap <= 0:
        step_t, step_w = th, tw
    else:
        step_t, step_w = th - 2 * overlap, tw - 2 * overlap
        if step_t <= 0 or step_w <= 0:
            raise ValueError("overlap слишком большой")
    boxes = [(t0, w0) for t0 in _starts(height, th, step_t) for w0 in _starts(width, tw, step_w)]
    acc = np.zeros((2, height, width), np.float32)
    cnt = np.zeros((height, width), np.float32)
    acc_f = np.zeros((2, height, width), np.float32)
    cnt_f = np.zeros((height, width), np.float32)
    tb = max(1, int(tile_batch))
    try:
        with torch.no_grad():
            for i0 in range(0, len(boxes), tb):
                chunk = boxes[i0:i0 + tb]
                xs, vs, ms = [], [], []
                meta = []
                for t0, w0 in chunk:
                    ht = min(th, height - t0)
                    wt = min(tw, width - w0)
                    ch_s = np.zeros((2, th, tw), np.float32)
                    va_s = np.zeros((th, tw), np.float32)
                    ch_s[:, :ht, :wt] = ch[:, t0:t0 + ht, w0:w0 + wt]
                    va_s[:ht, :wt] = valid[t0:t0 + ht, w0:w0 + wt]
                    for mask, inn in phases:
                        m = np.ones((th, tw), np.float32)
                        inner = np.zeros((th, tw), np.float32)
                        m[:ht, :wt] = mask[t0:t0 + ht, w0:w0 + wt]
                        inner[:ht, :wt] = inn[t0:t0 + ht, w0:w0 + wt]
                        xs.append(ch_s)
                        vs.append(va_s)
                        ms.append(m)
                        meta.append((t0, w0, ht, wt, inner, m))
                xb = torch.from_numpy(np.stack(xs))
                vb = torch.from_numpy(np.stack(vs)[:, None])
                mb = torch.from_numpy(np.stack(ms)[:, None])
                hat = model(xb, vb, mb).numpy()
                for p, (t0, w0, ht, wt, inner, m) in enumerate(meta):
                    pred = hat[p, :, :ht, :wt]
                    inn = inner[:ht, :wt]
                    mk = m[:ht, :wt]
                    acc[:, t0:t0 + ht, w0:w0 + wt] += pred * inn
                    cnt[t0:t0 + ht, w0:w0 + wt] += inn
                    use = (mk > 0.5).astype(np.float32)
                    acc_f[:, t0:t0 + ht, w0:w0 + wt] += pred * use
                    cnt_f[t0:t0 + ht, w0:w0 + wt] += use
                del hat, xb, vb, mb
    finally:
        model.train(was)
    if nphase < 1:
        raise RuntimeError("нет фаз маски")
    out = np.zeros((2, height, width), np.float32)
    good = cnt > 0
    np.divide(acc[0], cnt, out=out[0], where=good)
    np.divide(acc[1], cnt, out=out[1], where=good)
    miss = ~good
    fall = miss & (cnt_f > 0)
    np.divide(acc_f[0], cnt_f, out=out[0], where=fall)
    np.divide(acc_f[1], cnt_f, out=out[1], where=fall)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def residuals(model, ch, valid, **kw) -> tuple[np.ndarray, np.ndarray]:
    """(r_time, r_space), оба со знаком, уже умноженные на valid."""
    hat = reconstruct(model, ch, valid, **kw)
    v = (np.asarray(valid, np.float32) > 0.5).astype(np.float32)
    ch = np.nan_to_num(np.asarray(ch, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    r0 = (ch[0] - hat[0]) * v
    r1 = (ch[1] - hat[1]) * v
    return r0.astype(np.float32), r1.astype(np.float32)


def head_is_zero(model) -> bool:
    """Нулевая голова даёт hat=0 на любом входе. Это ровно остаток = канал."""
    head = getattr(model, "head", None)
    if head is None or not hasattr(head, "weight"):
        return False
    w = head.weight.detach()
    b = head.bias.detach()
    return bool(w.numel()) and float(w.abs().max()) == 0.0 and float(b.abs().max()) == 0.0


def member_maps(model, ch, valid, knots, **kw) -> tuple[np.ndarray, np.ndarray]:
    """Скор CDF(|остаток dT_space|) и знак остатка dT_time в {-1,0,+1}.

    Нулевая голова не гоняет свёртки: ответ совпадает с residuals при hat=0.
    """
    if head_is_zero(model):
        # hat=0. Клип ±12 живёт только внутри reconstruct и на нулевой голове не влияет.
        ch = np.nan_to_num(np.asarray(ch, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        v = (np.asarray(valid, np.float32) > 0.5).astype(np.float32)
        r0 = ch[0] * v
        r1 = ch[1] * v
    else:
        r0, r1 = residuals(model, ch, valid, **kw)
    s = apply_cdf(np.abs(r1), knots)
    h = np.sign(r0).astype(np.float32)
    h *= (np.asarray(valid, np.float32) > 0.5).astype(np.float32)
    return s.astype(np.float32), h.astype(np.float32)


def apply_cdf(values: np.ndarray, knots: np.ndarray) -> np.ndarray:
    knots = np.asarray(knots, np.float64).ravel()
    if knots.size < 2:
        raise ValueError("мало узлов CDF")
    for i in range(1, knots.size):
        if knots[i] <= knots[i - 1]:
            knots[i] = knots[i - 1] + 1e-6
    grid = np.linspace(0.0, 1.0, knots.size)
    out = np.interp(np.asarray(values, np.float64), knots, grid, left=0.0, right=1.0)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def blend_scores(mae_s: np.ndarray, winner_s: np.ndarray, kind: str) -> np.ndarray:
    mae_s = np.asarray(mae_s, np.float32)
    winner_s = np.asarray(winner_s, np.float32)
    if kind == "mae":
        return mae_s
    if kind == "ens_mean":
        return (0.5 * (mae_s + winner_s)).astype(np.float32)
    if kind == "ens_max":
        return np.maximum(mae_s, winner_s).astype(np.float32)
    if kind == "ens_mean4":
        return ((4.0 * winner_s + mae_s) / 5.0).astype(np.float32)
    raise ValueError(kind)


def choose_candidate(ious: dict) -> str:
    """Максимум val IoU. Ничья — более консервативная смесь (сначала ens_mean4)."""
    best = max(float(ious[k]) for k in CANDIDATES)
    for name in CANDIDATES:
        if float(ious[name]) >= best - 1e-12:
            return name
    raise RuntimeError("нет кандидата")


def accept_decision(val_iou, winner_val, nested_gain, nested_min,
                    v10_sel, v10_win, v11_sel, v11_win, protocol_ok: bool = True) -> dict:
    """Правила задачи. BEATS требует ещё зазор 1e-4 к опубликованному val и живой протокол."""
    val_iou = float(val_iou)
    winner_val = float(winner_val)
    nested_gain = float(nested_gain)
    nested_min = float(nested_min)
    val_ok = bool(val_iou > WINNER_VAL and val_iou > winner_val)
    nested_ok = bool(nested_gain + 1e-15 >= NESTED_GAIN and nested_min > NESTED_BAR)
    v10_ok = bool(float(v10_sel) + 1e-6 >= float(v10_win))
    v11_ok = bool(float(v11_sel) + 1e-6 >= float(v11_win))
    accept = bool(protocol_ok and val_ok and nested_ok and v10_ok and v11_ok)
    beats = bool(accept and val_iou > WINNER_VAL + 1e-4)
    return dict(
        val_ok=val_ok, nested_ok=nested_ok, v10_ok=v10_ok, v11_ok=v11_ok,
        accept=accept, beats=beats, protocol_ok=bool(protocol_ok),
    )


def spike_probe(model, ch, valid, amp: float = 4.0, tile=TILE, overlap: int = 0) -> dict:
    """Всплеск в обоих каналах. restore→1, если реконструкция пространства осталась фоном."""
    ch = np.asarray(ch, np.float32).copy()
    valid = np.asarray(valid, np.float32)
    height, width = valid.shape
    r0, r1 = max(0, height // 2 - 8), min(height, height // 2 + 8)
    c0, c1 = max(0, width // 2 - 4), min(width, width // 2 + 4)
    spiked = ch.copy()
    spiked[0, r0:r1, c0:c1] += np.float32(amp)
    spiked[1, r0:r1, c0:c1] += np.float32(amp)
    hat = reconstruct(model, spiked, valid, tile=tile, overlap=overlap)
    zone = np.zeros((height, width), bool)
    zone[r0:r1, c0:c1] = True
    zone &= valid > 0.5
    if not bool(zone.any()):
        return {"restore": 0.0, "left": None}
    left = float(np.mean(np.abs(hat[SCORE_CHANNEL][zone] - ch[SCORE_CHANNEL][zone])))
    restore = float(1.0 - left / (abs(amp) + 1e-6))
    return {"restore": restore, "left": left}
