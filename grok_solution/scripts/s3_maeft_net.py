"""Сегментационный U-Net ширины MAE и перенос энкодера.

MaskedReconstructor (dts/mae_anom.py) держит все свёртки ширины 16 и видит 4 канала:
dT_time, dT_space, valid, маска. DTSUNet победителя — другой граф: вход 3 канала,
ядро 7×3 и ширины 16-32-64-128. Общих тензоров энкодера, которые можно скопировать
без смены формы, нет (ствол (16,4,5,3) против (16,3,7,3)).

Поэтому предобучение MAE переносится в U-Net той же ширины: e1/e2/e3 — те же модули,
что у реконструктора, декодер складывает пропуски и заканчивается 3 классами.
Сравнение «от unet_v810ft» — это тот же график обучения на DTSUNet, строго загруженный
из artifacts/stage1/unet_v810ft.pt. Случайная инициализация — тот же SegUNet без копии.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from dts.mae_anom import MaskedReconstructor
from dts.models import DTSUNet

WIDTH = 16
ARMS = ("mae", "random", "v810")
BLENDS = ("ens_mean4", "ens_mean", "ens_max")
# Порядок ничьей фиксирован до val: сначала ens_mean4, сначала инициализация MAE.
CANDIDATES = tuple(f"{blend}_{arm}" for arm in ARMS for blend in BLENDS)
MAE_CKPT = "artifacts/eval/s3_mae_recon.pt"
V810_CKPT = "artifacts/stage1/unet_v810ft.pt"
WINNER_VAL = 0.38671235821914546
WINNER_TEST = 0.4052762065230473
WINNER_V10 = 0.4038977883892501
WINNER_V11 = 0.40390028295846503
NESTED_GAIN = 0.003
ENC_LR = 1e-5
DEC_LR = 1e-4


class SegUNet(nn.Module):
    """Энкодер совпадает с MaskedReconstructor. Голова — фон / нагрев / охлаждение."""

    def __init__(self, width: int = WIDTH):
        super().__init__()
        donor = MaskedReconstructor(width=int(width))
        self.width = int(donor.width)
        self.e1 = donor.e1
        self.e2 = donor.e2
        self.e3 = donor.e3
        self.up2 = donor.up2
        self.d2 = donor.d2
        self.up1 = donor.up1
        self.d1 = donor.d1
        self.ctx = donor.ctx
        self.head = nn.Conv2d(self.width, 3, 1)
        nn.init.normal_(self.head.weight, 0.0, 0.01)
        self.head.bias.data.copy_(torch.tensor([1.0, -1.0, -1.0]))

    @staticmethod
    def _add(up, skip):
        if up.shape[-2:] != skip.shape[-2:]:
            up = F.interpolate(up, size=skip.shape[-2:], mode="nearest")
        return up + skip

    def forward(self, x):
        """x (B,3,H,W): dT_time, dT_space, valid. Маска MAE на сегментации нулевая."""
        ch = x[:, :2].clamp(-12.0, 12.0)
        valid = (x[:, 2:3] > 0.5).to(dtype=ch.dtype)
        mask = torch.zeros_like(valid)
        z = torch.cat([ch * valid, valid, mask], 1)
        e1 = self.e1(z)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        d2 = self.d2(self._add(self.up2(e3), e2))
        d1 = self.d1(self._add(self.up1(d2), e1))
        return self.head(self.ctx(d1))


def _load_sd(path: str) -> dict:
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(raw, dict) and "state_dict" in raw and not any(str(k).startswith("e1.") for k in raw):
        raw = raw["state_dict"]
    return raw


def copy_mae_encoder(model: SegUNet, path: str = MAE_CKPT) -> dict:
    """Копирует e1/e2/e3, включая BN. Декодер и голова не берутся из реконструктора."""
    src = _load_sd(path)
    own = model.state_dict()
    copied, skipped = [], []
    for key, value in src.items():
        if not str(key).startswith(("e1.", "e2.", "e3.")):
            continue
        if key not in own or tuple(own[key].shape) != tuple(value.shape):
            skipped.append(str(key))
            continue
        own[key] = value.detach().clone()
        copied.append(str(key))
    enc_keys = [k for k in own if k.startswith(("e1.", "e2.", "e3."))]
    if skipped or len(copied) != len(enc_keys):
        raise RuntimeError(f"энкодер MAE скопирован не целиком: copied={len(copied)} enc={len(enc_keys)} skipped={skipped[:8]}")
    model.load_state_dict(own)
    with torch.no_grad():
        delta = 0.0
        for key in enc_keys:
            if key.endswith("num_batches_tracked"):
                continue
            delta = max(delta, float((model.state_dict()[key] - src[key]).abs().max()))
    return dict(copied=len(copied), skipped=len(skipped), max_abs=delta, keys=copied)


def encoder_delta(model: nn.Module, path: str, prefixes: tuple[str, ...]) -> dict:
    src = _load_sd(path)
    own = model.state_dict()
    delta = 0.0
    worst = ""
    n = 0
    for key, value in src.items():
        if not str(key).startswith(prefixes):
            continue
        if key not in own or tuple(own[key].shape) != tuple(value.shape):
            continue
        if str(key).endswith("num_batches_tracked"):
            continue
        gap = float((own[key].detach() - value.detach()).abs().max())
        if gap >= delta:
            delta = gap
            worst = str(key)
        n += 1
    if n == 0:
        raise RuntimeError("нет тензоров для сравнения энкодера")
    return dict(max_abs=float(delta), key=worst, n=n)


def build_arm(arm: str, seed: int) -> nn.Module:
    if arm not in ARMS:
        raise ValueError(arm)
    torch.manual_seed(int(seed))
    if arm == "v810":
        model = DTSUNet(cin=3, n_classes=3, w=WIDTH)
        model.load_state_dict(_load_sd(V810_CKPT))
        return model
    model = SegUNet(WIDTH)
    if arm == "mae":
        report = copy_mae_encoder(model, MAE_CKPT)
        if report["max_abs"] != 0.0:
            raise RuntimeError(f"копия энкодера MAE не точная: {report['max_abs']}")
    return model


def param_groups(model: nn.Module, arm: str) -> list[dict]:
    if arm == "v810":
        prefixes = ("e1.", "e2.", "e3.", "e4.", "gate.")
    else:
        prefixes = ("e1.", "e2.", "e3.")
    enc, dec = [], []
    for name, param in model.named_parameters():
        (enc if name.startswith(prefixes) else dec).append(param)
    if not enc or not dec:
        raise RuntimeError(f"{arm}: пустая группа параметров")
    return [
        dict(params=enc, lr=ENC_LR),
        dict(params=dec, lr=DEC_LR),
    ]


def blend_scores(score: np.ndarray, winner: np.ndarray, kind: str) -> np.ndarray:
    score = np.asarray(score, np.float32)
    winner = np.asarray(winner, np.float32)
    if kind == "ens_mean":
        return (0.5 * (score + winner)).astype(np.float32)
    if kind == "ens_max":
        return np.maximum(score, winner).astype(np.float32)
    if kind == "ens_mean4":
        return ((4.0 * winner + score) / 5.0).astype(np.float32)
    raise ValueError(kind)


def split_candidate(name: str) -> tuple[str, str]:
    arm = name.rsplit("_", 1)[-1]
    if arm not in ARMS:
        raise ValueError(name)
    kind = name[: -(len(arm) + 1)]
    if kind not in BLENDS:
        raise ValueError(name)
    return kind, arm


def choose_candidate(ious: dict) -> str:
    """Максимум val IoU среди заранее перечисленных смесей. Ничья — порядок CANDIDATES."""
    best = max(float(ious[name]) for name in CANDIDATES)
    for name in CANDIDATES:
        if float(ious[name]) >= best - 1e-12:
            return name
    raise RuntimeError("нет кандидата")


def _starts(n: int, tile: int, step: int) -> list[int]:
    if n <= tile:
        return [0]
    starts = list(range(0, n - tile + 1, step))
    last = n - tile
    if starts[-1] != last:
        starts.append(last)
    return starts


@torch.no_grad()
def predict_map(model: nn.Module, image: np.ndarray, tile=(256, 512), overlap: int = 32, batch: int = 4):
    """image (3,T,W) -> s=1-p(фон), h=p(нагрев)-p(охлаждение). Тайлы с перекрытием."""
    was = model.training
    model.eval()
    image = np.nan_to_num(np.asarray(image, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"ожидался (3,T,W), пришло {image.shape}")
    _, height, width = image.shape
    th, tw = int(tile[0]), int(tile[1])
    if overlap <= 0:
        step_t, step_w = th, tw
    else:
        step_t, step_w = th - 2 * overlap, tw - 2 * overlap
        if step_t <= 0 or step_w <= 0:
            raise ValueError("overlap слишком большой")
    boxes = [(t0, w0) for t0 in _starts(height, th, step_t) for w0 in _starts(width, tw, step_w)]
    acc = np.zeros((3, height, width), np.float32)
    cnt = np.zeros((height, width), np.float32)
    tb = max(1, int(batch))
    try:
        for i0 in range(0, len(boxes), tb):
            chunk = boxes[i0:i0 + tb]
            xs = np.zeros((len(chunk), 3, th, tw), np.float32)
            meta = []
            for i, (t0, w0) in enumerate(chunk):
                ht = min(th, height - t0)
                wt = min(tw, width - w0)
                xs[i, :, :ht, :wt] = image[:, t0:t0 + ht, w0:w0 + wt]
                meta.append((t0, w0, ht, wt))
            prob = torch.softmax(model(torch.from_numpy(xs)), 1).numpy()
            for (t0, w0, ht, wt), pi in zip(meta, prob):
                acc[:, t0:t0 + ht, w0:w0 + wt] += pi[:, :ht, :wt]
                cnt[t0:t0 + ht, w0:w0 + wt] += 1.0
    finally:
        model.train(was)
    acc /= np.maximum(cnt, 1.0)
    score = (1.0 - acc[0]).astype(np.float32)
    sign = (acc[1] - acc[2]).astype(np.float32)
    return score, sign
