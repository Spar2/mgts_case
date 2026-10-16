"""CNN-модели.

Stage 1 — DTSUNet: компактный U-Net (Ronneberger et al., 2015) для семантической сегментации
«изображения» время×расстояние на 3 класса {фон, нагрев, охлаждение}. Полностью свёрточный —
работает на трассе любой длины. Вероятность аномалии = 1 − p(фон).
Добавлены: (а) анизотропные ядра в энкодере (7×3: время×x) — события в DTS вытянуты во времени;
(б) «колонковое» внимание по времени (squeeze по x) — ловит синхронность включений.

Stage 2 — EventNet: 2D-CNN по вырезке события (32×64, 3 канала) + MLP по скалярным признакам
(длительность, ширина, амплитуда, доля кабеля...) -> тип события.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def block(cin, cout, k=(3, 3)):
    pad = (k[0] // 2, k[1] // 2)
    return nn.Sequential(nn.Conv2d(cin, cout, k, padding=pad, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class TimeGate(nn.Module):
    """Squeeze-excitation по оси времени: признак строки (среднее по x) модулирует всю строку."""
    def __init__(self, c):
        super().__init__(); self.fc = nn.Conv1d(c, c, 5, padding=2)
    def forward(self, x):
        s = x.mean(dim=3)                      # B,C,T
        g = torch.sigmoid(self.fc(s))[..., None]
        return x * g


class DTSUNet(nn.Module):
    def __init__(self, cin=3, n_classes=3, w=16):
        super().__init__()
        c = [w, 2 * w, 4 * w, 8 * w]
        self.e1 = block(cin, c[0], (7, 3)); self.e2 = block(c[0], c[1], (7, 3)); self.e3 = block(c[1], c[2]); self.e4 = block(c[2], c[3])
        self.gate = TimeGate(c[3])
        self.u3 = nn.ConvTranspose2d(c[3], c[2], 2, 2); self.d3 = block(2 * c[2], c[2])
        self.u2 = nn.ConvTranspose2d(c[2], c[1], 2, 2); self.d2 = block(2 * c[1], c[1])
        self.u1 = nn.ConvTranspose2d(c[1], c[0], 2, 2); self.d1 = block(2 * c[0], c[0])
        self.head = nn.Conv2d(c[0], n_classes, 1)

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(F.max_pool2d(e1, 2)); e3 = self.e3(F.max_pool2d(e2, 2))
        e4 = self.gate(self.e4(F.max_pool2d(e3, 2)))
        d3 = self.d3(torch.cat([self.u3(e4), e3], 1)); d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.head(d1)


@torch.no_grad()
def predict_full(model, X, tile=(256, 512), overlap=32):
    """X: np/tensor (3,T,X) -> prob (3,T,X). Тайлы с перекрытием, края дополняются нулями."""
    import numpy as np
    model.eval()
    X = torch.as_tensor(np.asarray(X, np.float32))
    C, T, W = X.shape
    th, tw = tile
    out = torch.zeros(3, T, W); cnt = torch.zeros(1, T, W)
    step_t, step_w = th - 2 * overlap, tw - 2 * overlap
    ts = list(range(0, max(T - th, 0) + 1, step_t)) or [0]
    if ts[-1] + th < T: ts.append(T - th)
    ws = list(range(0, max(W - tw, 0) + 1, step_w)) or [0]
    if ws[-1] + tw < W: ws.append(W - tw)
    for t0 in ts:
        for w0 in ws:
            x = X[:, max(t0, 0):t0 + th, max(w0, 0):w0 + tw]
            ph, pw = th - x.shape[1], tw - x.shape[2]
            xp = F.pad(x, (0, pw, 0, ph))[None]
            p = torch.softmax(model(xp), 1)[0, :, :x.shape[1], :x.shape[2]]
            out[:, t0:t0 + x.shape[1], w0:w0 + x.shape[2]] += p
            cnt[:, t0:t0 + x.shape[1], w0:w0 + x.shape[2]] += 1
    return (out / cnt.clamp(min=1)).numpy()


class EventNet(nn.Module):
    def __init__(self, n_classes, n_scalar, cin=3, w=24, late_fusion=False):
        super().__init__()
        # late_fusion=False — тот же граф, что у уже сохранённого eventnet.pt.
        # late_fusion=True добавляет нулевые линейные головы по CNN и по скалярам:
        # на старте это тот же EventNet, дальше головы учат поправку.
        self.late_fusion = late_fusion
        self.cnn = nn.Sequential(
            nn.Conv2d(cin, w, 3, padding=1), nn.BatchNorm2d(w), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(w, 2 * w, 3, padding=1), nn.BatchNorm2d(2 * w), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(2 * w, 4 * w, 3, padding=1), nn.BatchNorm2d(4 * w), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten())
        self.mlp = nn.Sequential(nn.Linear(n_scalar, 32), nn.ReLU())
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(16 * w + 32, 64), nn.ReLU(), nn.Linear(64, n_classes))
        if late_fusion:
            self.head_img = nn.Linear(16 * w, n_classes)
            self.head_sc = nn.Linear(32, n_classes)
            nn.init.zeros_(self.head_img.weight); nn.init.zeros_(self.head_img.bias)
            nn.init.zeros_(self.head_sc.weight); nn.init.zeros_(self.head_sc.bias)

    def forward(self, img, scal):
        fi, fs = self.cnn(img), self.mlp(scal)
        logits = self.head(torch.cat([fi, fs], 1))
        if self.late_fusion:
            logits = logits + self.head_img(fi) + self.head_sc(fs)
        return logits


def two_scale(img):
    """Широкая вырезка и билинейно растянутый центр (две ветки одним тензором, cin удваивается)."""
    h, w = img.shape[-2:]
    tight = img[..., h // 4:h - h // 4, w // 4:w - w // 4]
    if tight.shape[-2] < 1 or tight.shape[-1] < 1:
        tight = img
    up = F.interpolate(tight, size=(h, w), mode="bilinear", align_corners=False)
    return torch.cat([img, up], 1)


class TCN(nn.Module):
    """Временная свёрточная сеть (Bai et al., 2018, TCN): дилатированные свёртки только по оси времени
    (каждый столбец x — свой ряд), плюс узкое ядро 3 по x. Без пулинга, полностью свёрточная."""
    def __init__(self, cin=3, n_classes=3, w=32, dil=(1, 2, 4, 8, 16, 32)):
        super().__init__()
        layers, c = [], cin
        for d in dil:
            layers += [nn.Conv2d(c, w, (5, 3), padding=(2 * d, 1), dilation=(d, 1), bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True)]
            c = w
        self.body = nn.Sequential(*layers); self.head = nn.Conv2d(w, n_classes, 1)
    def forward(self, x):
        return self.head(self.body(x))


class DilatedFCN(nn.Module):
    """2D контекстная сеть с дилатациями (Yu & Koltun, 2016): большое поле зрения без потери разрешения."""
    def __init__(self, cin=3, n_classes=3, w=32, dil=(1, 2, 4, 8, 16, 1)):
        super().__init__()
        layers, c = [], cin
        for d in dil:
            layers += [nn.Conv2d(c, w, 3, padding=d, dilation=d, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True)]
            c = w
        self.body = nn.Sequential(*layers); self.head = nn.Conv2d(w, n_classes, 1)
    def forward(self, x):
        return self.head(self.body(x))


def build(arch: str, width: int | None = None):
    if arch == "unet":
        return DTSUNet(w=width or 16)
    if arch == "unet_ms":
        return DTSUNetMS(w=width or 16)
    if arch == "tcn":
        return TCN(w=width or 32)
    if arch == "dilated":
        return DilatedFCN(w=width or 32)
    raise ValueError(arch)


class Integrate(nn.Module):
    """Физически мотивированный «интегрирующий» вход: слабое долгое событие (0.3–2 °C при σ≈0.3 °C)
    не видно в одном пикселе, но его SNR растёт как √N при усреднении по N клеткам (согласованный фильтр).
    Фиксированные (не обучаемые) усредняющие окна по (t, x), выход нормирован на √N, чтобы остаться в единицах σ."""
    def __init__(self, scales=((7, 15), (21, 61))):
        super().__init__(); self.scales = scales

    def forward(self, x):
        v = x[:, 2:3]; outs = []
        for kt, kx in self.scales:
            pad = (kx // 2, kx // 2, kt // 2, kt // 2)
            num = F.avg_pool2d(F.pad(x[:, :2] * v, pad, mode="replicate"), (kt, kx), stride=1)
            den = F.avg_pool2d(F.pad(v, pad, mode="replicate"), (kt, kx), stride=1).clamp(min=1e-3)
            n = den * kt * kx
            outs.append(torch.clamp(num / den * torch.sqrt(n) / 3.0, -12, 12))   # /3: ~ поправка на корреляцию шума
        return torch.cat([x] + outs, 1)


class DTSUNetMS(nn.Module):
    """U-Net + интегрирующие каналы (3 + 2·2 = 7 входов)."""
    def __init__(self, w=16, scales=((7, 15), (21, 61))):
        super().__init__(); self.integ = Integrate(scales); self.net = DTSUNet(cin=3 + 2 * len(scales), w=w)

    def forward(self, x):
        return self.net(self.integ(x))
