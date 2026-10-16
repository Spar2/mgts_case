"""Банк нормальных патчей и обучаемая голова аномалии.

Энкодер — e1 и e2 замороженного unet_v810ft. В банк попадают только клетки фона
synth train (ev==0, nu не нужен: сюда кладётся метка 0). Размер сырого банка
ограничен FIFO, в поиске соседей — корежет не больше CORESET векторов.

Опубликованный скор — сигмоида головы. Голова видит запрос, соседа и статистику
расстояний (MLP и билинеар). Голое расстояние скором не является.

Ядро контекста 9×3 (время × кабель) зафиксировано. Реальные CSV сюда не входят.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .models import block

WINNER_ENCODER = "artifacts/stage1/unet_v810ft.pt"
TILE = (128, 256)
INFER_TILE = (256, 512)
OVERLAP = 32
EMBED = 16
ENC_CH = 32
BANK_CAP = 8192
CORESET = 1024
K_NN = 3
AGG_KERNEL = (9, 3)
# Смесь с ансамблем зафиксирована до val: (4 * winner + s) / 5.
BLEND_WINNER_PARTS = 4.0


def l2_normalize(x: torch.Tensor, dim: int) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp(min=1e-6)


def fifo_append(old: torch.Tensor, new: torch.Tensor, cap: int) -> torch.Tensor:
    """Дописать векторы и выбросить самые старые, если длина больше cap."""
    if new.numel() == 0:
        return old
    if old.numel() == 0:
        cat = new
    else:
        cat = torch.cat([old, new], 0)
    cap = int(cap)
    if cat.shape[0] > cap:
        cat = cat[-cap:].contiguous()
    return cat


def greedy_coreset(x: torch.Tensor, m: int, seed: int) -> torch.Tensor:
    """Жадный furthest-point. При избытке кандидатов — случайная подвыборка seed."""
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("пустой набор для корезета")
    m = int(m)
    if m < 1:
        raise ValueError("m")
    x = x.detach().float().cpu()
    if x.shape[0] <= m:
        return x.clone()
    g = torch.Generator()
    g.manual_seed(int(seed))
    cap = m * 8
    if x.shape[0] > cap:
        perm = torch.randperm(x.shape[0], generator=g)[:cap]
        x = x[perm]
    n = x.shape[0]
    start = int(torch.randint(0, n, (1,), generator=g))
    chosen = [start]
    dmin = torch.cdist(x, x[start:start + 1]).squeeze(1)
    dmin[start] = -1.0
    while len(chosen) < m:
        nxt = int(torch.argmax(dmin))
        chosen.append(nxt)
        dmin = torch.minimum(dmin, torch.cdist(x, x[nxt:nxt + 1]).squeeze(1))
        dmin[nxt] = -1.0
    return x[chosen].clone()


def pure_background(lab: torch.Tensor, stride: int = 2) -> torch.Tensor:
    """True, где блок stride×stride целиком имеет метку 0. Край, не кратный stride, отброшен."""
    if lab.ndim == 2:
        lab = lab.unsqueeze(0)
    b, height, width = lab.shape
    h = height // stride
    w = width // stride
    if h == 0 or w == 0:
        return torch.zeros(b, h, w, dtype=torch.bool, device=lab.device)
    sl = lab[:, : h * stride, : w * stride]
    sl = sl.reshape(b, h, stride, w, stride)
    return (sl == 0).all(dim=(2, 4))


def _knn_index(q: torch.Tensor, bank: torch.Tensor, k: int) -> torch.Tensor:
    """Индексы k ближайших по косинусу. q: (B,D,H,W), bank: (N,D). Оба уже нормированы.

    Большая карта режется по банке, чтобы не держать матрицу N×HW целиком.
    """
    b, d, height, width = q.shape
    hw = height * width
    qf = q.reshape(b, d, hw)
    n = int(bank.shape[0])
    k = min(int(k), n)
    # ~128 МБ на кусок сходства
    chunk = max(k, int((128 * 2**20) // max(b * hw * 4, 1)))
    best_s = None
    best_i = None
    ar = torch.arange(n)
    for s in range(0, n, chunk):
        bk = bank[s:s + chunk]
        sim = torch.einsum("bdh,nd->bnh", qf, bk)
        if best_s is None:
            kk = min(k, sim.shape[1])
            best_s, local = torch.topk(sim, kk, dim=1, largest=True)
            best_i = local + s
            if kk < k:
                pad_s = torch.full((b, k - kk, hw), -2.0, dtype=sim.dtype)
                pad_i = torch.zeros((b, k - kk, hw), dtype=torch.long)
                best_s = torch.cat([best_s, pad_s], 1)
                best_i = torch.cat([best_i, pad_i], 1)
        else:
            take = bk.shape[0]
            scat = torch.cat([best_s, sim], 1)
            icat = torch.cat([best_i, ar[s:s + take][None, :, None].expand(b, -1, hw)], 1)
            best_s, local = torch.topk(scat, k, dim=1, largest=True)
            best_i = torch.gather(icat, 1, local)
    return best_i.view(b, k, height, width)


def knn_pack(q: torch.Tensor, bank: torch.Tensor, k: int = K_NN):
    """nearest (B,D,H,W) — среднее k соседей; stats (B,3,H,W) — min, mean, max расстояние.

    Индекс соседа отсоединён от графа. Расстояние считается заново и зависит от q.
    """
    if bank.ndim != 2 or bank.shape[0] < 1:
        raise RuntimeError("банк пуст")
    qn = l2_normalize(q, 1)
    bn = l2_normalize(bank.detach(), 1)
    k_eff = min(int(k), int(bn.shape[0]))
    with torch.no_grad():
        idx = _knn_index(qn.detach(), bn, k_eff)
    nb = bn[idx]  # B,k,H,W,D
    diff = qn.permute(0, 2, 3, 1).unsqueeze(1) - nb
    dist = torch.sqrt((diff * diff).sum(-1).clamp(min=1e-12))
    if k_eff < int(k):
        pad = dist[:, -1:].expand(-1, int(k) - k_eff, -1, -1)
        dist = torch.cat([dist, pad], 1)
        nb = torch.cat([nb, nb[:, -1:].expand(-1, int(k) - k_eff, -1, -1, -1)], 1)
    nearest = nb.mean(1).permute(0, 3, 1, 2).contiguous()
    stats = torch.stack([dist[:, 0], dist.mean(1), dist[:, -1]], 1)
    return nearest, stats


class FrozenEncoder(nn.Module):
    """e1+e2, шаг 2. e4 для короткого следа слишком груб."""

    def __init__(self, w: int = 16):
        super().__init__()
        self.e1 = block(3, w, (7, 3))
        self.e2 = block(w, 2 * w, (7, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.e2(F.max_pool2d(self.e1(x), 2))

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


class ScoreHead(nn.Module):
    """Две свёртки 1×1 и билинеар. Вход — запрос, разность с соседом, три дистанции."""

    def __init__(self, d: int = EMBED):
        super().__init__()
        self.d = d
        self.mlp = nn.Sequential(
            nn.Conv2d(d + d + 3, 32, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1, bias=True),
        )
        self.bilinear = nn.Bilinear(d, d, 1, bias=False)
        self.hot = nn.Conv2d(d, 1, 1)
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.normal_(self.mlp[2].weight, std=0.01)
        nn.init.constant_(self.mlp[2].bias, -2.0)
        nn.init.normal_(self.bilinear.weight, std=0.01)
        nn.init.zeros_(self.hot.weight)
        nn.init.zeros_(self.hot.bias)

    def forward(self, q: torch.Tensor, nearest: torch.Tensor, stats: torch.Tensor):
        mix = torch.cat([q, q - nearest, stats], 1)
        base = self.mlp(mix)
        b, d, height, width = q.shape
        qf = q.permute(0, 2, 3, 1).reshape(-1, d)
        nf = nearest.permute(0, 2, 3, 1).reshape(-1, d)
        bil = self.bilinear(qf, nf).view(b, height, width, 1).permute(0, 3, 1, 2)
        return base + bil, self.hot(q)


class MemBankNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FrozenEncoder()
        self.adapter = nn.Conv2d(ENC_CH, EMBED, 1, bias=False)
        self.agg = nn.Conv2d(EMBED, EMBED, AGG_KERNEL, padding=(AGG_KERNEL[0] // 2, AGG_KERNEL[1] // 2), bias=False)
        self.head = ScoreHead(EMBED)
        self.register_buffer("bank", torch.zeros(CORESET, EMBED))
        self.register_buffer("bank_count", torch.zeros((), dtype=torch.long))
        self._init_agg_identity()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()

    def _init_agg_identity(self) -> None:
        nn.init.zeros_(self.agg.weight)
        cy, cx = AGG_KERNEL[0] // 2, AGG_KERNEL[1] // 2
        with torch.no_grad():
            for c in range(EMBED):
                self.agg.weight[c, c, cy, cx] = 1.0

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def set_bank(self, proto: torch.Tensor) -> None:
        proto = proto.detach().float().cpu()
        if proto.ndim != 2 or proto.shape[1] != EMBED:
            raise RuntimeError(f"банк {tuple(proto.shape)} не {EMBED}")
        keep = proto.norm(dim=1) > 1e-4
        proto = proto[keep]
        if proto.shape[0] == 0:
            raise RuntimeError("пустой банк")
        if proto.shape[0] < K_NN:
            reps = (K_NN + proto.shape[0] - 1) // proto.shape[0]
            proto = proto.repeat(reps, 1)[:K_NN]
        proto = l2_normalize(proto, 1)
        n = min(int(proto.shape[0]), CORESET)
        self.bank.zero_()
        self.bank[:n].copy_(proto[:n])
        self.bank_count.fill_(n)

    def active_bank(self) -> torch.Tensor:
        n = int(self.bank_count.item())
        if n < 1:
            raise RuntimeError("банк пуст")
        return self.bank[:n]

    def _perturb(self, q: torch.Tensor):
        """Узкая по кабелю полоса шума. Метка этой полосы добавляется к событию, не заменяет его."""
        b, _d, h, w = q.shape
        apply = torch.rand(b, device=q.device) < 0.5
        x0 = torch.randint(0, w, (b,), device=q.device)
        width = torch.randint(1, 3, (b,), device=q.device)
        t0 = torch.randint(0, max(h // 2, 1), (b,), device=q.device)
        tt = torch.arange(h, device=q.device)
        xx = torch.arange(w, device=q.device)
        time_ok = tt[None, :] >= t0[:, None]
        space_ok = (xx[None, :] >= x0[:, None]) & (xx[None, :] < (x0 + width)[:, None])
        mask = (time_ok[:, :, None] & space_ok[:, None, :] & apply[:, None, None]).float()[:, None]
        return q + torch.randn_like(q) * mask * 0.8, mask

    def features(self, x: torch.Tensor) -> torch.Tensor:
        self.encoder.eval()
        with torch.no_grad():
            feat = self.encoder(x)
        return self.agg(self.adapter(feat))

    def forward(self, x: torch.Tensor, perturb: bool = False):
        q = self.features(x)
        extra = None
        if perturb and self.training:
            q, mask_f = self._perturb(q)
            extra = F.interpolate(mask_f, size=x.shape[-2:], mode="nearest")[:, 0]
        nearest, stats = knn_pack(q, self.active_bank(), K_NN)
        qn = l2_normalize(q, 1)
        logit, hot = self.head(qn, nearest, stats)
        size = x.shape[-2:]
        logit = F.interpolate(logit, size=size, mode="bilinear", align_corners=False)
        hot = F.interpolate(hot, size=size, mode="bilinear", align_corners=False)
        return logit, hot, extra

    @torch.no_grad()
    def embed_background(self, x: torch.Tensor, lab: torch.Tensor) -> torch.Tensor:
        """Векторы клеток, чей блок 2×2 целиком фон. Уже после адаптера и ядра 9×3."""
        q = self.features(x)
        if not torch.is_tensor(lab):
            lab = torch.as_tensor(np.ascontiguousarray(lab))
        bg = pure_background(lab.long(), 2)
        q = q[:, :, ::2, ::2]
        bg = bg[:, ::2, ::2]
        flat = q.permute(0, 2, 3, 1).reshape(-1, EMBED)
        return flat[bg.reshape(-1)].detach().cpu()


def membank_loss(logit, hot, lab, extra=None, pos_weight: float = 4.0, hot_weight: float = 0.25):
    """BCE аномалии на valid. Шум латента только добавляет положительные клетки фона."""
    if lab.ndim == 2:
        lab = lab.unsqueeze(0)
    valid = lab >= 0
    target = lab > 0
    if extra is not None:
        if extra.ndim == 4:
            extra = extra[:, 0]
        target = target | (extra > 0.5)
    target = target & valid
    if not bool(valid.any()):
        return logit.sum() * 0
    inp = logit[:, 0][valid]
    tgt = target[valid].float()
    per = F.binary_cross_entropy_with_logits(inp, tgt, reduction="none")
    weight = torch.where(tgt > 0.5, per.new_tensor(float(pos_weight)), per.new_ones(()))
    main = (per * weight).mean()
    ev = (lab == 1) | (lab == 2)
    if bool(ev.any()):
        hot_t = (lab == 1).float()
        hloss = F.binary_cross_entropy_with_logits(hot[:, 0][ev], hot_t[ev])
    else:
        hloss = logit.new_zeros(())
    return main + float(hot_weight) * hloss


def blend_with_winner(score: np.ndarray, winner: np.ndarray, winner_parts: float = BLEND_WINNER_PARTS) -> np.ndarray:
    return ((float(winner_parts) * winner + score) / (float(winner_parts) + 1.0)).astype(np.float32)


def tile_starts(n: int, tile: int, step: int) -> list[int]:
    if n <= tile:
        return [0]
    starts = list(range(0, n - tile + 1, step))
    last = n - tile
    if starts[-1] != last:
        starts.append(last)
    return starts


def _interior(t0, w0, ht, wt, t_len, w_len, margin: int):
    t1, w1, t2, w2 = 0, 0, ht, wt
    if t0 > 0:
        t1 = min(margin, ht // 4)
    if w0 > 0:
        w1 = min(margin, wt // 4)
    if t0 + ht < t_len:
        t2 = ht - min(margin, ht // 4)
    if w0 + wt < w_len:
        w2 = wt - min(margin, wt // 4)
    if t2 <= t1 or w2 <= w1:
        return 0, ht, 0, wt
    return t1, t2, w1, w2


@torch.inference_mode()
def score_field(model: MemBankNet, x: np.ndarray, tile=INFER_TILE, overlap: int = OVERLAP, batch: int = 2):
    """x: (3,T,W) → s в [0,1], h в [-1,1]. Перекрытие усредняется, край тайла с неполным ядром отбрасывается."""
    model.eval()
    x = np.asarray(x, np.float32)
    _, t_len, w_len = x.shape
    th, tw = tile
    margin = max(overlap // 2, 8)
    step_t = max(th - 2 * margin, 1)
    step_w = max(tw - 2 * margin, 1)
    boxes = [(t0, w0) for t0 in tile_starts(t_len, th, step_t) for w0 in tile_starts(w_len, tw, step_w)]
    acc = np.zeros((t_len, w_len), np.float32)
    sgn = np.zeros((t_len, w_len), np.float32)
    wgt = np.zeros((t_len, w_len), np.float32)
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
        logit, hot, _extra = model(torch.from_numpy(np.stack(xs)), perturb=False)
        prob = torch.sigmoid(logit[:, 0]).cpu().numpy()
        sign = torch.tanh(hot[:, 0]).cpu().numpy()
        for b, (t0, w0, ht, wt) in enumerate(meta):
            a, c, d, e = _interior(t0, w0, ht, wt, t_len, w_len, margin)
            acc[t0 + a:t0 + c, w0 + d:w0 + e] += prob[b, a:c, d:e]
            sgn[t0 + a:t0 + c, w0 + d:w0 + e] += sign[b, a:c, d:e]
            wgt[t0 + a:t0 + c, w0 + d:w0 + e] += 1.0
    out = acc / np.maximum(wgt, 1.0)
    h = sgn / np.maximum(wgt, 1.0)
    if not np.isfinite(out).all():
        raise RuntimeError("нечисловой скор")
    return out, h


def trainable_parameters(model: MemBankNet):
    for name, p in model.named_parameters():
        if name.startswith("encoder."):
            continue
        if p.requires_grad:
            yield p
