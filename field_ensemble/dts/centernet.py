"""CenterNet-голова (Zhou, Wang, Krähenbühl, Objects as Points, CVPR 2019).

Замороженный энкодер DTSUNet победителя (unet_v810ft, ширина 16) отдаёт признаки
на шаге 2: e2 и ближайший подъём e3. Декодер и TimeGate не используются: гейт
усредняет по всей оси x и на тайле не совпал бы с полным кабелем. Голова учит
тепловую карту центров (нагрев и охлаждение отдельно), log(Δt), log(Δx) и
дробный сдвиг центра внутри клетки шага 2. Растеризация рамки — уверенность
пика внутри прямоугольника, снаружи 0.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage

from dts import models

STRIDE = 2
N_CLASS = 2
PEAK_MIN = 0.05
TOPK = 100
WH_WEIGHT = 0.1
OFF_WEIGHT = 1.0
FOCAL_A = 2.0
FOCAL_B = 4.0
MIN_OVERLAP = 0.7
STRUCT8 = ndimage.generate_binary_structure(2, 2)


def load_encoder(path: str, width: int = 16) -> models.DTSUNet:
    net = models.DTSUNet(w=width)
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    net.load_state_dict(state)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def encoder_width(net: nn.Module) -> int:
    return int(net.e1[0].out_channels)


def encoder_features(net: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Признаки шага 2: e2 и e3, поднятый до сетки e2. x: (B, 3, H, W), H и W делятся на 4."""
    e1 = net.e1(x)
    e2 = net.e2(F.max_pool2d(e1, 2))
    e3 = net.e3(F.max_pool2d(e2, 2))
    e3u = F.interpolate(e3, size=e2.shape[-2:], mode="nearest")
    return torch.cat([e2, e3u], dim=1)


def feature_channels(width: int) -> int:
    return 6 * int(width)


class CenterNetHead(nn.Module):
    """1×1 снижает каналы, дальше dilated 3×3 (1,2,4,8,16).

    Поле на сетке шага 2 — 63 клетки, около 31 м по кабелю. Без этого голова видит
    только соседей и не отличает центр длинного очага от его середины: фокальная
    потеря тогда уводит всю карту вниз.
    """

    def __init__(self, cin: int = 96, mid: int = 32, n_class: int = N_CLASS):
        super().__init__()
        self.cin = int(cin)
        self.mid = int(mid)
        self.n_class = int(n_class)
        groups = 8 if self.mid % 8 == 0 else 1
        layers = [
            nn.Conv2d(self.cin, self.mid, 1, bias=False),
            nn.GroupNorm(groups, self.mid),
            nn.ReLU(inplace=True),
        ]
        for d in (1, 2, 4, 8, 16):
            layers += [
                nn.Conv2d(self.mid, self.mid, 3, padding=d, dilation=d, bias=False),
                nn.GroupNorm(groups, self.mid),
                nn.ReLU(inplace=True),
            ]
        self.proj = nn.Sequential(*layers)
        self.hm = nn.Conv2d(self.mid, self.n_class, 1)
        self.wh = nn.Conv2d(self.mid, 2, 1)
        self.off = nn.Conv2d(self.mid, 2, 1)
        nn.init.constant_(self.hm.bias, -2.19)
        nn.init.normal_(self.hm.weight, std=0.001)
        nn.init.zeros_(self.wh.bias)
        nn.init.normal_(self.wh.weight, std=0.001)
        nn.init.zeros_(self.off.bias)
        nn.init.normal_(self.off.weight, std=0.001)

    def forward(self, feat: torch.Tensor) -> dict:
        z = self.proj(feat)
        return {"hm": self.hm(z), "wh": self.wh(z), "off": self.off(z)}


def gaussian_radius(height: float, width: float, min_overlap: float = MIN_OVERLAP) -> int:
    """Радиус гауссианы CornerNet/CenterNet при минимальном IoU min_overlap."""
    h = max(float(height), 1e-3)
    w = max(float(width), 1e-3)
    a1, b1 = 1.0, h + w
    c1 = w * h * (1.0 - min_overlap) / (1.0 + min_overlap)
    a2, b2 = 4.0, 2.0 * (h + w)
    c2 = (1.0 - min_overlap) * w * h
    a3 = 4.0 * min_overlap
    b3 = -2.0 * min_overlap * (h + w)
    c3 = (min_overlap - 1.0) * w * h
    radii = []
    for a, b, c in ((a1, b1, c1), (a2, b2, c2), (a3, b3, c3)):
        disc = b * b - 4.0 * a * c
        if disc < 0 or a == 0:
            continue
        radii.append((b + np.sqrt(disc)) / (2.0 * a))
    if not radii:
        return 0
    r = float(min(radii))
    if not np.isfinite(r) or r < 1:
        return 0
    return int(r)


def axis_radius(length: float) -> int:
    """Радиус вдоль одной стороны рамки.

    Формула CornerNet для квадрата этой стороны часто даёт < 1 на узком кабеле
    и тогда вся рамка, кроме центра, становится жёстким отрицательным примером.
    Пол — половина стороны (от 2 клеток), чтобы гауссиана росла вместе с рамкой.
    """
    length = max(float(length), 1e-3)
    r = gaussian_radius(length, length)
    floor = int(length / 2.0) if length >= 2.0 else 0
    return int(max(r, floor))


def size_radii(dt: float, dx: float, stride: int = STRIDE) -> tuple[int, int]:
    return axis_radius(float(dt) / stride), axis_radius(float(dx) / stride)


def draw_gaussian(hm: np.ndarray, cy: int, cx: int, radius: int, radius_x: int | None = None) -> None:
    """Эллиптическая гауссиана. radius — полуось по времени, radius_x — по кабелю.

    Центр клетки равен 1. При наложении остаётся максимум, центр каждого объекта снова 1.
    """
    h, w = hm.shape
    if not (0 <= cy < h and 0 <= cx < w):
        return
    ry = int(radius)
    rx = ry if radius_x is None else int(radius_x)
    if ry < 1 and rx < 1:
        hm[cy, cx] = 1.0
        return
    ry = max(0, min(ry, h))
    rx = max(0, min(rx, w))
    y0, y1 = cy - ry, cy + ry + 1
    x0, x1 = cx - rx, cx + rx + 1
    gy0, gy1 = 0, 2 * ry + 1
    gx0, gx1 = 0, 2 * rx + 1
    if y0 < 0:
        gy0 -= y0
        y0 = 0
    if x0 < 0:
        gx0 -= x0
        x0 = 0
    if y1 > h:
        gy1 -= y1 - h
        y1 = h
    if x1 > w:
        gx1 -= x1 - w
        x1 = w
    if y0 >= y1 or x0 >= x1:
        return
    yy = np.arange(-ry, ry + 1, dtype=np.float32)[:, None]
    xx = np.arange(-rx, rx + 1, dtype=np.float32)[None, :]
    sig_y = (2 * ry + 1) / 6.0 if ry >= 1 else 1.0
    sig_x = (2 * rx + 1) / 6.0 if rx >= 1 else 1.0
    g = np.exp(-(yy * yy) / (2.0 * sig_y * sig_y) - (xx * xx) / (2.0 * sig_x * sig_x)).astype(np.float32)
    sl = hm[y0:y1, x0:x1]
    np.maximum(sl, g[gy0:gy1, gx0:gx1], out=sl)
    hm[cy, cx] = 1.0


def _components(mask: np.ndarray, cls: int) -> list[dict]:
    lab, n = ndimage.label(mask, structure=STRUCT8)
    if n <= 0:
        return []
    out = []
    for sl in ndimage.find_objects(lab):
        if sl is None:
            continue
        t0, t1 = int(sl[0].start), int(sl[0].stop) - 1
        x0, x1 = int(sl[1].start), int(sl[1].stop) - 1
        dt = t1 - t0 + 1
        dx = x1 - x0 + 1
        if dt < 1 or dx < 1:
            continue
        out.append(dict(cls=cls, t0=t0, t1=t1, x0=x0, x1=x1, dt=dt, dx=dx))
    return out


def objects_from_label(y: np.ndarray) -> list[dict]:
    """y: -1 вне кабеля, 0 фон, 1 нагрев, 2 охлаждение. Рамка — AABB компоненты."""
    hot = _components(y == 1, 0)
    cold = _components(y == 2, 1)
    objs = hot + cold
    objs.sort(key=lambda o: -(o["dt"] * o["dx"]))
    return objs


def build_targets(ys: np.ndarray, stride: int = STRIDE):
    """ys (B, H, W) int. Возвращает hm, valid, n_obj, reg (N, 6): b, iy, ix, log_dt, log_dx, off_t, off_x."""
    bsz, height, width = ys.shape
    if height % stride or width % stride:
        raise ValueError("тайл должен делиться на stride")
    ho, wo = height // stride, width // stride
    hm = np.zeros((bsz, N_CLASS, ho, wo), np.float32)
    # Клетка в потере, если в окне шага есть кабель (фон 0 или событие). Чистое −1 — ignore.
    cells = ys.reshape(bsz, ho, stride, wo, stride)
    valid = (cells >= 0).any(axis=(2, 4))
    rows = []
    n_obj = 0
    for b in range(bsz):
        for o in objects_from_label(ys[b]):
            ct = 0.5 * (o["t0"] + o["t1"])
            cx = 0.5 * (o["x0"] + o["x1"])
            oy, ox = ct / stride, cx / stride
            iy, ix = int(np.floor(oy)), int(np.floor(ox))
            if not (0 <= iy < ho and 0 <= ix < wo):
                continue
            rad_t, rad_x = size_radii(o["dt"], o["dx"], stride)
            draw_gaussian(hm[b, o["cls"]], iy, ix, rad_t, rad_x)
            rows.append((b, iy, ix, np.log(float(o["dt"])), np.log(float(o["dx"])), oy - iy, ox - ix))
            n_obj += 1
    reg = np.zeros((len(rows), 7), np.float32)
    for i, (b, iy, ix, ldt, ldx, ot, ox) in enumerate(rows):
        reg[i] = (b, iy, ix, ldt, ldx, ot, ox)
    return hm, valid.astype(np.bool_), n_obj, reg


def centernet_loss(pred: dict, hm: torch.Tensor, valid: torch.Tensor, n_obj: int, reg: torch.Tensor) -> dict:
    """Фокальная потеря карты (α=2, β=4) и L1 размера и сдвига только в центрах."""
    logits = pred["hm"]
    prob = torch.sigmoid(logits)
    m = valid[:, None]
    pos = (hm >= 0.999) & m
    neg = (hm < 0.999) & m
    loss_hm = logits.new_zeros(())
    if pos.any():
        p = prob[pos].clamp(1e-6, 1 - 1e-6)
        loss_hm = loss_hm + (-(1 - p).pow(FOCAL_A) * p.log()).sum()
    if neg.any():
        p = prob[neg].clamp(1e-6, 1 - 1e-6)
        t = hm[neg]
        loss_hm = loss_hm + (-p.pow(FOCAL_A) * (1 - t).pow(FOCAL_B) * (1 - p).log()).sum()
    denom = max(int(n_obj), 1)
    loss_hm = loss_hm / denom
    loss_wh = logits.new_zeros(())
    loss_off = logits.new_zeros(())
    if n_obj > 0 and reg.numel():
        b = reg[:, 0].long()
        iy = reg[:, 1].long()
        ix = reg[:, 2].long()
        got_wh = pred["wh"][b, :, iy, ix]
        got_off = pred["off"][b, :, iy, ix]
        loss_wh = (got_wh - reg[:, 3:5]).abs().sum() / denom
        loss_off = (got_off - reg[:, 5:7]).abs().sum() / denom
    total = loss_hm + WH_WEIGHT * loss_wh + OFF_WEIGHT * loss_off
    return {"loss": total, "hm": loss_hm.detach(), "wh": loss_wh.detach(), "off": loss_off.detach()}


def box_bounds(ct: float, cx: float, dt: float, dx: float, height: int, width: int):
    dt = float(np.clip(dt, 1.0, 8192.0))
    dx = float(np.clip(dx, 1.0, 8192.0))
    t_lo = int(np.round(ct - (dt - 1.0) / 2.0))
    x_lo = int(np.round(cx - (dx - 1.0) / 2.0))
    t_hi = t_lo + int(np.round(dt))
    x_hi = x_lo + int(np.round(dx))
    t_lo, t_hi = max(0, t_lo), min(int(height), t_hi)
    x_lo, x_hi = max(0, x_lo), min(int(width), x_hi)
    return t_lo, t_hi, x_lo, x_hi


def rasterize(dets: list[dict], height: int, width: int, n_class: int = N_CLASS):
    """Внутри рамки — уверенность пика (max при перекрытии). Снаружи 0. h = hot − cold."""
    acc = np.zeros((n_class, height, width), np.float32)
    for d in dets:
        t_lo, t_hi, x_lo, x_hi = box_bounds(d["ct"], d["cx"], d["dt"], d["dx"], height, width)
        if t_hi <= t_lo or x_hi <= x_lo:
            continue
        sl = acc[d["cls"], t_lo:t_hi, x_lo:x_hi]
        np.maximum(sl, float(d["score"]), out=sl)
    score = acc.max(axis=0)
    hotness = acc[0] - acc[1] if n_class >= 2 else score
    return score, hotness


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))).astype(np.float32)


def extract_dets(hm_logits: np.ndarray, wh: np.ndarray, off: np.ndarray,
                 stride: int = STRIDE, peak_min: float = PEAK_MIN, topk: int = TOPK) -> list[dict]:
    """Пики 3×3 max-pool, как в статье. Не NMS рамок. До topk пиков на класс."""
    prob = _sigmoid(hm_logits)
    xt = torch.from_numpy(np.ascontiguousarray(prob[None]))
    hmax = F.max_pool2d(xt, 3, stride=1, padding=1)
    keep = (xt == hmax) & (xt >= peak_min)
    dets = []
    n_class, _, _ = prob.shape
    for c in range(n_class):
        ys, xs = torch.where(keep[0, c])
        if ys.numel() == 0:
            continue
        scores = xt[0, c, ys, xs]
        if scores.numel() > topk:
            sel = torch.topk(scores, topk).indices
            ys, xs, scores = ys[sel], xs[sel], scores[sel]
        for y, x, s in zip(ys.tolist(), xs.tolist(), scores.tolist()):
            ot = float(np.clip(off[0, y, x], -1.0, 2.0))
            ox = float(np.clip(off[1, y, x], -1.0, 2.0))
            dt = float(np.clip(np.exp(wh[0, y, x]), 1.0, 8192.0))
            dx = float(np.clip(np.exp(wh[1, y, x]), 1.0, 8192.0))
            dets.append(dict(
                cls=c, score=float(s),
                ct=(y + ot) * stride, cx=(x + ox) * stride,
                dt=dt, dx=dx,
            ))
    return dets


def _origins(n: int, tile: int, step: int) -> list[int]:
    if n < tile:
        raise ValueError("канву нужно дополнить до тайла")
    if n == tile:
        return [0]
    xs = list(range(0, n - tile + 1, step))
    last = n - tile
    if xs[-1] != last:
        xs.append(last)
    return xs


@torch.no_grad()
def predict_score(net, head, image: np.ndarray, tile=(256, 512), overlap=64, batch=4,
                  stride: int = STRIDE, peak_min: float = PEAK_MIN, topk: int = TOPK):
    """image (3, H, W) float32 → score (H, W), hotness (H, W), сведения о пиках.

    Начала тайлов кратны 2, канва кратна 8, чтобы сетка пулинга на тайлах совпала.
    Логиты в перекрытии усредняются, пики ищутся уже на сшитой карте.
    """
    net.eval()
    head.eval()
    image = np.asarray(image, np.float32)
    _, height, width = image.shape
    th, tw = int(tile[0]), int(tile[1])
    if th % 8 or tw % 8 or int(overlap) % 2:
        raise ValueError("тайл кратен 8, перекрытие кратно 2")
    hp, wp = max(height, th), max(width, tw)
    if hp % 8:
        hp += 8 - hp % 8
    if wp % 8:
        wp += 8 - wp % 8
    canvas = np.zeros((3, hp, wp), np.float32)
    canvas[:, :height, :width] = image
    ho, wo = hp // stride, wp // stride
    acc_hm = np.zeros((N_CLASS, ho, wo), np.float32)
    acc_wh = np.zeros((2, ho, wo), np.float32)
    acc_off = np.zeros((2, ho, wo), np.float32)
    acc_n = np.zeros((ho, wo), np.float32)
    step_t, step_w = th - int(overlap), tw - int(overlap)
    ts, xs = _origins(hp, th, step_t), _origins(wp, tw, step_w)
    buf_x, buf_xy = [], []

    def flush():
        if not buf_x:
            return
        xt = torch.from_numpy(np.stack(buf_x))
        feat = encoder_features(net, xt)
        out = head(feat)
        hm = out["hm"].numpy()
        wh = out["wh"].numpy()
        off = out["off"].numpy()
        for i, (t0, x0) in enumerate(buf_xy):
            ot, ox = t0 // stride, x0 // stride
            hs, ws = hm.shape[-2], hm.shape[-1]
            acc_hm[:, ot:ot + hs, ox:ox + ws] += hm[i]
            acc_wh[:, ot:ot + hs, ox:ox + ws] += wh[i]
            acc_off[:, ot:ot + hs, ox:ox + ws] += off[i]
            acc_n[ot:ot + hs, ox:ox + ws] += 1.0
        buf_x.clear()
        buf_xy.clear()

    for t0 in ts:
        for x0 in xs:
            buf_x.append(canvas[:, t0:t0 + th, x0:x0 + tw])
            buf_xy.append((t0, x0))
            if len(buf_x) >= batch:
                flush()
    flush()
    if (acc_n <= 0).any():
        raise RuntimeError("тайлы не покрыли канву")
    inv = 1.0 / acc_n
    dets = extract_dets(acc_hm * inv[None], acc_wh * inv[None], acc_off * inv[None],
                        stride=stride, peak_min=peak_min, topk=topk)
    score, hotness = rasterize(dets, hp, wp)
    score, hotness = score[:height, :width], hotness[:height, :width]
    info = dict(
        n_det=len(dets),
        pos_frac=float((score > 0).mean()) if score.size else 0.0,
        max_score=float(score.max()) if score.size else 0.0,
    )
    return score, hotness, info


def self_check(encoder_path: str | None = None) -> None:
    """Геометрия рамки, гауссиана в центре и то, что энкодер не учится."""
    y = np.zeros((32, 48), np.int8)
    y[10:15, 20:24] = 1
    hm, valid, n_obj, reg = build_targets(y[None], STRIDE)
    if n_obj != 1:
        raise SystemExit(f"ожидался 1 объект, есть {n_obj}")
    # центр (12, 21.5) → клетка (6, 10)
    if hm[0, 0, 6, 10] < 0.999:
        raise SystemExit(f"пик карты не в клетке центра: {hm[0, 0, 6, 10]}")
    if abs(float(reg[0, 3]) - np.log(5)) > 1e-5 or abs(float(reg[0, 4]) - np.log(4)) > 1e-5:
        raise SystemExit(f"размер цели {reg[0, 3:5]}")
    if abs(float(reg[0, 6]) - 0.75) > 1e-5:
        raise SystemExit(f"сдвиг x {reg[0, 6]}")
    det = dict(cls=0, score=0.8, ct=12.0, cx=21.5, dt=5.0, dx=4.0)
    score, hot = rasterize([det], 32, 48)
    if score[10:15, 20:24].min() < 0.8 - 1e-6 or score[9, 22] != 0 or score[12, 19] != 0:
        raise SystemExit("растеризация не совпала с рамкой 10:15, 20:24")
    if hot[12, 21] <= 0:
        raise SystemExit("знак нагрева не положительный")
    cold = dict(cls=1, score=0.4, ct=4.0, cx=6.0, dt=1.0, dx=1.0)
    score2, hot2 = rasterize([det, cold], 32, 48)
    if score2[4, 6] < 0.4 - 1e-6 or hot2[4, 6] >= 0:
        raise SystemExit("холодная точка не легла отдельно")
    if not valid.any():
        raise SystemExit("маска валидности пустая")
    if gaussian_radius(80, 8) <= gaussian_radius(2, 2):
        raise SystemExit(
            f"радиус не растёт с рамкой: {gaussian_radius(80, 8)} против {gaussian_radius(2, 2)}"
        )
    wide = np.zeros((1, 64, 64), np.int8)
    wide[0, 8:48, 20:28] = 1
    tiny = np.zeros((1, 64, 64), np.int8)
    tiny[0, 30:32, 40:42] = 1
    hm_w, _, _, reg_w = build_targets(wide, STRIDE)
    hm_t, _, _, reg_t = build_targets(tiny, STRIDE)
    rw = size_radii(float(np.exp(reg_w[0, 3])), float(np.exp(reg_w[0, 4])))
    rt = size_radii(float(np.exp(reg_t[0, 3])), float(np.exp(reg_t[0, 4])))
    if max(rw) <= max(rt):
        raise SystemExit(f"гауссиана крупной рамки не шире мелкой: {rw} <= {rt}")
    if hm_w[0, 0].sum() <= hm_t[0, 0].sum():
        raise SystemExit("крупная рамка не рисует более широкую гауссиану")
    head = CenterNetHead(cin=96, mid=32)
    feat = torch.zeros(2, 96, 16, 24)
    out = head(feat)
    if out["hm"].shape != (2, 2, 16, 24) or out["wh"].shape[-3] != 2:
        raise SystemExit(f"форма головы {out['hm'].shape}")
    tgt = torch.zeros_like(out["hm"])
    v = torch.ones(2, 16, 24, dtype=torch.bool)
    loss = centernet_loss(out, tgt, v, 0, torch.zeros(0, 7))
    loss["loss"].backward()
    if head.hm.weight.grad is None:
        raise SystemExit("голова без градиента")
    if encoder_path:
        net = load_encoder(encoder_path, 16)
        if encoder_width(net) != 16 or feature_channels(16) != 96:
            raise SystemExit("ширина энкодера не 16")
        x = torch.zeros(1, 3, 128, 256)
        with torch.no_grad():
            f = encoder_features(net, x)
        if tuple(f.shape) != (1, 96, 64, 128):
            raise SystemExit(f"признаки {tuple(f.shape)}")
        if any(p.requires_grad for p in net.parameters()):
            raise SystemExit("энкодер не заморожен")
        head2 = CenterNetHead(cin=96)
        opt = torch.optim.SGD(head2.parameters(), lr=1e-2)
        xb = torch.randn(2, 3, 128, 256)
        yb = np.zeros((2, 128, 256), np.int8)
        yb[0, 40:50, 80:88] = 1
        hm_b, v_b, n_b, reg_b = build_targets(yb, STRIDE)
        with torch.no_grad():
            fb = encoder_features(net, xb)
        pred = head2(fb.detach())
        loss2 = centernet_loss(
            pred, torch.from_numpy(hm_b), torch.from_numpy(v_b), n_b, torch.from_numpy(reg_b))
        opt.zero_grad()
        loss2["loss"].backward()
        opt.step()
        if not any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in head2.parameters()):
            raise SystemExit("шаг головы не сдвинул градиент")
    print("centernet self_check ok", flush=True)
