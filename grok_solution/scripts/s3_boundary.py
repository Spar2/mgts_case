"""Boundary loss (Kervadec et al., MIDL 2019) и deep supervision на U-Net.

Только синтетика v8 и v10 train. Старт — artifacts/stage1/unet_v810ft.pt.
Реальные CSV и статистики шума не читаются.
Выбор — v8 val IoU@FAR 1e-3 смеси с ens_u2ft_tcn_lgbm.
Вложенные 5 фолдов — процедура scripts/s3_sel2_cv.py: i % 5, порог на holdout, micro IoU.
Прирост к победителю под этой процедурой должен быть не меньше 0.003.

  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \\
  /workspace/repo/run/.venv/bin/python -u scripts/s3_boundary.py
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_k] = "2"

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)

from dts import data, losses, metrics as M, models

WINNER = "ens_u2ft_tcn_lgbm"
WINNER_VAL = 0.38671235821914546
WINNER_TEST = 0.4052762065230473
NESTED_GAIN = 0.003
INIT = Path("artifacts/stage1/unet_v810ft.pt")
CKPT = Path("artifacts/eval/s3_boundary_unet.pt")
OUT_TRAIN = Path("artifacts/eval/s3_boundary_train.json")
OUT_EVAL = Path("artifacts/eval/s3_boundary_v8.json")
ENS = Path("artifacts/pred/v8") / WINNER
TRAIN_BUDGET_S = 40 * 60
STEPS = 36
MAX_EPOCHS = 24
BATCH = 6
TILE = (128, 256)
LR = 8e-5
LR_ENC = 1e-5


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S', time.gmtime())} {msg}", flush=True)


def plain(obj):
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return plain(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    return obj


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(plain(obj), indent=1, ensure_ascii=False))
    os.replace(tmp, path)


class BoundaryUNet(torch.nn.Module):
    """DTSUNet ширины 16 плюс боковые 1×1 на d3 и d2. На eval боковые головы не вызываются."""

    def __init__(self, w: int = 16):
        super().__init__()
        self.net = models.DTSUNet(w=w)
        c1, c2 = 2 * w, 4 * w
        self.side2 = torch.nn.Conv2d(c1, 3, 1)
        self.side3 = torch.nn.Conv2d(c2, 3, 1)
        torch.nn.init.zeros_(self.side2.weight)
        torch.nn.init.zeros_(self.side2.bias)
        torch.nn.init.zeros_(self.side3.weight)
        torch.nn.init.zeros_(self.side3.bias)

    def load_init(self, path) -> None:
        self.net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))

    def forward(self, x, deep: bool = False):
        n = self.net
        e1 = n.e1(x)
        e2 = n.e2(F.max_pool2d(e1, 2))
        e3 = n.e3(F.max_pool2d(e2, 2))
        e4 = n.gate(n.e4(F.max_pool2d(e3, 2)))
        d3 = n.d3(torch.cat([n.u3(e4), e3], 1))
        d2 = n.d2(torch.cat([n.u2(d3), e2], 1))
        d1 = n.d1(torch.cat([n.u1(d2), e1], 1))
        logit = n.head(d1)
        if not deep:
            return logit
        size = logit.shape[-2:]
        s2 = F.interpolate(self.side2(d2), size=size, mode="bilinear", align_corners=False)
        s3 = F.interpolate(self.side3(d3), size=size, mode="bilinear", align_corners=False)
        return logit, s2, s3


def levelset_one(y: np.ndarray) -> np.ndarray:
    """φ Карведека: расстояние до границы, минус внутри объекта, плюс снаружи.

    Делится на среднее |φ| по валидным клеткам, чтобы член был порядка 1.
    y<0 — ignore, в сумму не входит и в расстояние считается как фон.
    """
    fg = y > 0
    if not np.any(fg):
        phi = np.ones(y.shape, np.float32)
    elif np.all(fg):
        phi = -np.ones(y.shape, np.float32)
    else:
        inside = ndimage.distance_transform_edt(fg)
        outside = ndimage.distance_transform_edt(~fg)
        phi = (outside - inside).astype(np.float32)
    valid = y >= 0
    phi = phi.copy()
    phi[~valid] = 0.0
    if np.any(valid):
        scale = float(np.mean(np.abs(phi[valid])))
        if scale > 1e-6:
            phi /= scale
    return phi


def levelset_batch(y: np.ndarray) -> np.ndarray:
    return np.stack([levelset_one(y[i]) for i in range(y.shape[0])]).astype(np.float32)


def boundary_term(logit, phi, y) -> torch.Tensor:
    """∫ φ s d q, s = 1 − p(фон). Минимизация кладёт массу туда, где φ < 0."""
    valid = (y >= 0).to(dtype=logit.dtype)
    p_fg = 1.0 - torch.softmax(logit, 1)[:, 0]
    return (p_fg * phi * valid).sum() / valid.sum().clamp(min=1)


def regional(logit, y, cw) -> torch.Tensor:
    return losses.compute("focal+tversky_ce", logit, y, cw)


def total_loss(logits, y, phi, cw, w_b: float) -> tuple[torch.Tensor, float]:
    main, side2, side3 = logits
    reg = regional(main, y, cw)
    bnd = boundary_term(main, phi, y)
    aux = (
        0.5 * (regional(side2, y, cw) + w_b * boundary_term(side2, phi, y))
        + 0.25 * (regional(side3, y, cw) + w_b * boundary_term(side3, phi, y))
    )
    return reg + w_b * bnd + aux, float(bnd.detach())


def freeze_bn(model) -> None:
    for mod in model.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.eval()
            for p in mod.parameters():
                p.requires_grad_(False)


def bn_train_frozen(model) -> None:
    model.train()
    freeze_bn(model)


def tile_starts(n: int, tile: int, step: int) -> list[int]:
    ts = list(range(0, max(n - tile, 0) + 1, step)) or [0]
    if ts[-1] + tile < n:
        ts.append(n - tile)
    return ts


@torch.inference_mode()
def predict_batched(model, X, tile=(256, 512), overlap=32, batch=4):
    X = np.asarray(X, np.float32)
    _, t_len, w_len = X.shape
    th, tw = tile
    step_t, step_w = th - 2 * overlap, tw - 2 * overlap
    boxes = [(t0, w0) for t0 in tile_starts(t_len, th, step_t) for w0 in tile_starts(w_len, tw, step_w)]
    out = np.zeros((3, t_len, w_len), np.float32)
    cnt = np.zeros((t_len, w_len), np.float32)
    model.eval()
    for i in range(0, len(boxes), batch):
        xs, meta = [], []
        for t0, w0 in boxes[i:i + batch]:
            sl = X[:, t0:t0 + th, w0:w0 + tw]
            ht, wt = sl.shape[1], sl.shape[2]
            if ht != th or wt != tw:
                sl = np.pad(sl, ((0, 0), (0, th - ht), (0, tw - wt)))
            xs.append(sl)
            meta.append((t0, w0, ht, wt))
        prob = torch.softmax(model(torch.from_numpy(np.stack(xs))), 1).numpy()
        for p, (t0, w0, ht, wt) in zip(prob, meta):
            out[:, t0:t0 + ht, w0:w0 + wt] += p[:, :ht, :wt]
            cnt[t0:t0 + ht, w0:w0 + wt] += 1
    out /= np.maximum(cnt, 1.0)
    return out


class NpzSampler:
    """Тайлы из npz без нового mmap. Зеркало и множитель шума — как у TileSampler."""

    def __init__(self, paths, tile=TILE, seed=0, p_event=0.6):
        self.paths = list(paths)
        self.tile = tile
        self.p_event = p_event
        self.rng = np.random.default_rng(seed)
        self.ev_idx = []
        self.shapes = []
        for i, p in enumerate(self.paths):
            with np.load(p) as z:
                ev = np.asarray(z["ev"])
                valid = np.asarray(z["valid"], bool)
            y = data.target(ev, valid)
            idx = np.argwhere(y > 0)
            if len(idx) > 4000:
                idx = idx[self.rng.choice(len(idx), 4000, replace=False)]
            self.ev_idx.append(idx)
            self.shapes.append(y.shape)
            if (i + 1) % 40 == 0 or i + 1 == len(self.paths):
                log(f"индекс v10 {i + 1}/{len(self.paths)}")
        self.weights = np.array([a * b for a, b in self.shapes], np.float64)
        self.weights /= self.weights.sum()
        self._cache: dict = {}

    def _get(self, k):
        p = self.paths[k]
        if p in self._cache:
            self._cache[p] = self._cache.pop(p)
            return self._cache[p]
        with np.load(p) as z:
            X = np.asarray(z["X"])
            y = data.target(np.asarray(z["ev"]), np.asarray(z["valid"], bool))
        self._cache[p] = (X, y)
        while len(self._cache) > 3:
            self._cache.pop(next(iter(self._cache)))
        return X, y

    def sample(self, n):
        th, tw = self.tile
        xs, ys = [], []
        for _ in range(n):
            k = int(self.rng.choice(len(self.paths), p=self.weights))
            X, y = self._get(k)
            T, W = y.shape
            if self.rng.random() < self.p_event and len(self.ev_idx[k]):
                c = self.ev_idx[k][self.rng.integers(len(self.ev_idx[k]))]
                t0 = int(np.clip(int(c[0]) - self.rng.integers(th), 0, max(T - th, 0)))
                w0 = int(np.clip(int(c[1]) - self.rng.integers(tw), 0, max(W - tw, 0)))
            else:
                t0 = int(self.rng.integers(max(T - th, 0) + 1))
                w0 = int(self.rng.integers(max(W - tw, 0) + 1))
            xt = np.zeros((3, th, tw), np.float32)
            yt = np.full((th, tw), -1, np.int64)
            a = np.asarray(X[:, t0:t0 + th, w0:w0 + tw], np.float32)
            b = y[t0:t0 + th, w0:w0 + tw]
            xt[:, :a.shape[1], :a.shape[2]] = a
            yt[:b.shape[0], :b.shape[1]] = b
            if self.rng.random() < 0.5:
                xt = xt[:, :, ::-1]
                yt = yt[:, ::-1]
            g = float(self.rng.uniform(0.75, 1.3))
            xt[:2] = np.clip(xt[:2] * g, -12, 12)
            xs.append(np.ascontiguousarray(xt))
            ys.append(np.ascontiguousarray(yt))
        return np.stack(xs), np.stack(ys)


def assert_mmap_v8() -> None:
    mm = Path("artifacts/mm/v8/train")
    missing = [p.name for p in sorted(Path("artifacts/cache/v8/train").glob("seq*.npz"))
               if not (mm / f"{p.stem}_y.npy").exists()]
    if missing:
        raise SystemExit(f"нет mmap v8/train: {len(missing)}, новый кэш не пишу")


def bg_mask(ev, nu, valid):
    return np.asarray(valid, bool) & (np.asarray(ev) == 0) & (np.asarray(nu) == 0)


def proxy_blend(model, files) -> tuple[float, float]:
    model.eval()
    parts, maps = [], []
    for p in files:
        with np.load(p) as z:
            X = np.asarray(z["X"], np.float32)
            valid = np.asarray(z["valid"], bool)
            ev = np.asarray(z["ev"])
            nu = np.asarray(z["nu"])
        prob = predict_batched(model, X)
        s = (1.0 - prob[0]).astype(np.float32)
        with np.load(ENS / "val" / p.name) as pred:
            ens = np.asarray(pred["s"], np.float32)
        blend = (0.5 * (s + ens)).astype(np.float32)
        parts.append(blend[bg_mask(ev, nu, valid)])
        maps.append((blend, valid, ev > 0))
    thr = float(M.far_threshold(np.concatenate(parts), 1e-3))
    tot = np.zeros(3, np.int64)
    for s, valid, truth in maps:
        tot += np.asarray(M.confusion(s >= thr, truth, valid), np.int64)
    return float(M.prf(*tot)["iou"]), thr


def _self_check() -> None:
    y = np.zeros((32, 40), np.int64)
    y[8:24, 10:30] = 1
    y[0, :] = -1
    phi = levelset_one(y)
    if not (phi[16, 20] < 0 and phi[16, 2] > 0 and phi[0, 5] == 0):
        raise SystemExit(f"levelset знак неверен {phi[16, 20]} {phi[16, 2]} {phi[0, 5]}")
    net = BoundaryUNet(16)
    net.eval()
    x = torch.zeros(1, 3, 64, 80)
    with torch.no_grad():
        main, s2, s3 = net(x, deep=True)
        bare = net(x, deep=False)
    if main.shape != bare.shape or s2.shape != main.shape or s3.shape != main.shape:
        raise SystemExit(f"формы {main.shape} {s2.shape} {s3.shape}")
    # нулевые боковые головы: не участвуют в eval-графе
    if not torch.equal(main, bare):
        raise SystemExit("deep и eval разошлись на главной голове")
    ref = models.DTSUNet(w=16)
    ref.load_state_dict(net.net.state_dict())
    ref.eval()
    with torch.no_grad():
        err = float((bare - ref(x)).abs().max())
    if err > 1e-6:
        raise SystemExit(f"главная голова не совпала с DTSUNet {err}")
    del net, x, ref


def train() -> dict:
    _self_check()
    assert_mmap_v8()
    log("self-check ok, сбор v8 mmap")
    tr = data.mmap_items("artifacts/cache/v8", "train", Path("artifacts/mm/v8"))
    sampler8 = data.TileSampler(tr, tile=TILE, seed=0, p_weak=0.0)
    v10 = sorted(Path("artifacts/cache/v10/train").glob("seq*.npz"))
    if len(v10) < 10:
        raise SystemExit(f"мало v10 train: {len(v10)}")
    log(f"v8 {len(tr)} v10 {len(v10)}")
    sampler10 = NpzSampler(v10, tile=TILE, seed=1)
    model = BoundaryUNet(16)
    model.load_init(INIT)
    freeze_bn(model)
    enc_ids = set()
    enc = []
    for name in ("e1", "e2", "e3", "e4", "gate"):
        for p in getattr(model.net, name).parameters():
            if p.requires_grad:
                enc.append(p)
                enc_ids.add(id(p))
    dec = [p for p in model.parameters() if p.requires_grad and id(p) not in enc_ids]
    opt = torch.optim.AdamW(
        [{"params": enc, "lr": LR_ENC}, {"params": dec, "lr": LR}],
        weight_decay=0.0,
    )
    cw = torch.tensor([1.0, 3.0, 3.0])
    proxy_files = data.list_cache("artifacts/cache/v8", "val")[:4]
    best = -1.0
    best_epoch = None
    hist = []
    t0 = time.time()
    deadline = t0 + TRAIN_BUDGET_S
    stopped = "epochs"
    payload = dict(history=[], best_epoch=None, stopped=stopped, seconds=0.0, ckpt=None,
                   best_proxy_blend_iou=None)
    for ep in range(MAX_EPOCHS):
        if time.time() >= deadline:
            stopped = "time"
            break
        w_b = 0.05 + 0.20 * min(ep, 8) / 8.0
        bn_train_frozen(model)
        tl, bl, nstep = [], [], 0
        t_ep = time.time()
        for step in range(STEPS):
            if time.time() >= deadline and nstep > 0:
                stopped = "time"
                break
            x, y = sampler8.sample(BATCH) if step % 2 == 0 else sampler10.sample(BATCH)
            phi = torch.from_numpy(levelset_batch(y))
            yt = torch.from_numpy(y)
            loss, bval = total_loss(model(torch.from_numpy(x), deep=True), yt, phi, cw, w_b)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_v = float(loss.detach())
            if not np.isfinite(loss_v) or not np.isfinite(bval):
                stopped = "nan"
                log(f"loss не число epoch {ep} step {step}")
                break
            tl.append(loss_v)
            bl.append(bval)
            nstep += 1
        if stopped == "nan":
            break
        if nstep == 0:
            stopped = "time"
            break
        pi, pt = proxy_blend(model, proxy_files)
        if not np.isfinite(pi):
            stopped = "nan"
            log(f"proxy не число epoch {ep}")
            break
        rec = dict(epoch=ep, steps=nstep, train=float(np.mean(tl)), boundary=float(np.mean(bl)),
                   w_b=w_b, proxy_blend_iou=pi, proxy_thr=pt,
                   sec=round(time.time() - t0, 1), epoch_sec=round(time.time() - t_ep, 1))
        hist.append(rec)
        log(str(rec))
        if pi > best + 1e-4:
            best = pi
            best_epoch = ep
            tmp = CKPT.with_suffix(".pt.tmp")
            torch.save(model.state_dict(), tmp)
            os.replace(tmp, CKPT)
        payload = dict(
            name="BOUNDARY",
            init=str(INIT),
            arch="DTSUNet w=16 + 1x1 side heads on d2 and d3",
            loss="focal+tversky_ce + Kervadec boundary on foreground + deep supervision",
            boundary="signed distance, negative inside the event, normalized by mean |phi|",
            paper="Kervadec et al., MIDL 2019, Boundary loss for highly unbalanced segmentation",
            deep_supervision="0.5 on d2 and 0.25 on d3, regional and boundary, bilinear to full resolution",
            w_b_schedule="0.05 → 0.25 over 8 epochs",
            lr=LR, lr_enc=LR_ENC, batch=BATCH, tile=list(TILE),
            steps_per_epoch=STEPS, max_epochs=MAX_EPOCHS,
            train_caches=["artifacts/cache/v8/train", "artifacts/cache/v10/train"],
            v10_mmap=False,
            bn="frozen",
            real_traces_used=False,
            real_noise_used=False,
            history=hist,
            best_epoch=best_epoch,
            best_proxy_blend_iou=None if best < 0 else float(best),
            stopped=stopped,
            seconds=round(time.time() - t0, 1),
            ckpt=str(CKPT) if best_epoch is not None else None,
        )
        dump(OUT_TRAIN, payload)
    payload["stopped"] = stopped
    payload["seconds"] = round(time.time() - t0, 1)
    dump(OUT_TRAIN, payload)
    log(f"обучение {payload['seconds']} с epoch {best_epoch} proxy {best:.4f} stop {stopped}")
    return payload


def _score_of(name, s, ws):
    if name == "unet_boundary":
        return s
    if name == "ens_blend":
        return (0.5 * (s + ws)).astype(np.float32)
    return ws


def fit_thresholds(tmp, files, split: str) -> tuple[dict, int]:
    """По одному методу: три полных фона сразу не держим."""
    names = ("unet_boundary", "ens_blend", WINNER)
    n_bg = 0
    masks = []
    for p in files:
        with np.load(p) as z:
            mask = bg_mask(z["ev"], z["nu"], z["valid"])
        masks.append(mask)
        n_bg += int(mask.sum())
    thr = {}
    for n in names:
        parts = []
        for p, mask in zip(files, masks):
            s = np.load(tmp / f"{p.stem}_s.npy", mmap_mode="r")
            with np.load(ENS / split / p.name) as pred:
                ws = np.asarray(pred["s"], np.float32)
            parts.append(np.asarray(_score_of(n, s, ws)[mask], np.float32))
            del s, ws
        bg = np.concatenate(parts)
        del parts
        thr[n] = float(M.far_threshold(bg, 1e-3))
        log(f"порог {n} {thr[n]:.6f} n_bg {n_bg}")
        del bg
    return thr, n_bg


def compact_pair(tmp, files):
    """Кадры s3_sel2_cv из уже посчитанных карт val. Второй проход сети не нужен."""
    import s3_sel2_cv as s2
    import s3_sel_eval as ev

    labels = pd.read_csv("data/synth_meta/v8/labels.csv")
    seqs = pd.read_csv("data/synth_meta/v8/sequences.csv").set_index("seq_id")
    winner, blend = [], []
    for p in files:
        with np.load(p) as z:
            ev_map = np.asarray(z["ev"])
            valid = np.asarray(z["valid"])
            nu = np.asarray(z["nu"])
            x = np.asarray(z["x"], np.float64)
            t = np.asarray(z["t"], np.float64)
        ev_idx, nu_idx, bg_idx, n_bg = ev._indices(ev_map, valid, nu, p.stem)
        s = np.asarray(np.load(tmp / f"{p.stem}_s.npy"), np.float32)
        with np.load(ENS / "val" / p.name) as pred:
            ws = np.asarray(pred["s"], np.float32)
        cooling, _rows = ev._events_for(p.stem, ev_map, x, t, labels, seqs, ev_idx)

        def frame(arr, stem=p.stem, cooling=cooling, n_bg=n_bg, ev_idx=ev_idx, nu_idx=nu_idx, bg_idx=bg_idx):
            e, n, b = s2._cols(arr, ev_idx, nu_idx, bg_idx)
            return dict(ev_m=e[None], nu_m=n[None], bg_m=b[None], n_bg=n_bg, cooling=cooling, stem=stem)

        winner.append(frame(ws))
        blend.append(frame((0.5 * (s + ws)).astype(np.float32)))
        del s, ws
    order = np.argsort([fr["stem"] for fr in winner])
    return [winner[i] for i in order], [blend[i] for i in order]


def score_split(model, split: str, thr: dict | None, collect_nested: bool = False):
    tmp = Path(f"/tmp/s3_boundary_{split}_{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    files = data.list_cache("artifacts/cache/v8", split)
    names = ("unet_boundary", "ens_blend", WINNER)
    t0 = time.time()
    pair = None
    try:
        for i, p in enumerate(files, 1):
            with np.load(p) as z:
                X = np.asarray(z["X"], np.float32)
            prob = predict_batched(model, X)
            np.save(tmp / f"{p.stem}_s.npy", (1.0 - prob[0]).astype(np.float32))
            np.save(tmp / f"{p.stem}_h.npy", (prob[1] - prob[2]).astype(np.float16))
            del X, prob
            if i == 1 or i % 15 == 0 or i == len(files):
                log(f"pred {split} {i}/{len(files)} {time.time() - t0:.0f}s")
        if thr is None:
            thr, n_bg_fit = fit_thresholds(tmp, files, split)
        else:
            n_bg_fit = None
        if collect_nested:
            pair = compact_pair(tmp, files)
            log(f"compact {len(pair[0])} кадров из карт val")
        tot = {n: np.zeros(3, np.int64) for n in names}
        hot = {n: np.zeros(3, np.int64) for n in names}
        cold = {n: np.zeros(3, np.int64) for n in names}
        far_n = {n: 0 for n in names}
        far_d = {n: 0 for n in names}
        for p in files:
            with np.load(p) as z:
                ev = np.asarray(z["ev"])
                valid = np.asarray(z["valid"], bool)
                nu = np.asarray(z["nu"])
            s = np.load(tmp / f"{p.stem}_s.npy")
            h = np.load(tmp / f"{p.stem}_h.npy").astype(np.float32)
            with np.load(ENS / split / p.name) as pred:
                ws = np.asarray(pred["s"], np.float32)
                wh = np.asarray(pred["h"], np.float32)
            packed = {
                "unet_boundary": (s, h),
                "ens_blend": ((0.5 * (s + ws)).astype(np.float32), (0.5 * (h + wh)).astype(np.float32)),
                WINNER: (ws, wh),
            }
            truth = ev > 0
            hot_t = (ev >= 1) & (ev <= 4)
            cold_t = ev >= 5
            mask = bg_mask(ev, nu, valid)
            for n, (ps, ph) in packed.items():
                pred = (ps >= thr[n]) & valid
                tot[n] += np.asarray(M.confusion(pred, truth, valid), np.int64)
                hot[n] += np.asarray(M.confusion(pred & (ph > 0), hot_t, valid), np.int64)
                cold[n] += np.asarray(M.confusion(pred & (ph <= 0), cold_t, valid), np.int64)
                far_n[n] += int((pred & mask).sum())
                far_d[n] += int(mask.sum())
            del s, h, ws, wh
        report = {}
        for n in names:
            row = dict(threshold=float(thr[n]), far=float(far_n[n] / max(far_d[n], 1)),
                       **M.prf(*map(int, tot[n])), hot=M.prf(*map(int, hot[n])), cold=M.prf(*map(int, cold[n])))
            report[n] = row
            log(f"{split} {n} IoU {row['iou']:.4f} F1 {row['f1']:.4f} FAR {row['far']:.5f}")
        report["_n_bg"] = int(far_d[WINNER])
        return report, pair
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def nested5(winner, blend) -> dict:
    """5 фолдов как s3_sel2_cv.nested_fixed: отсортированные val, i % 5, порог на holdout."""
    import s3_sel2_cv as s2
    import s3_sel_core as sel

    folds = sel.sequence_folds(len(winner), 5)
    parts_w, parts_b, details = [], [], []
    for r, hold_idx in enumerate(folds):
        mw = s2.eval_one(s2.take(winner, hold_idx))
        mb = s2.eval_one(s2.take(blend, hold_idx))
        parts_w.append(mw)
        parts_b.append(mb)
        details.append(dict(
            fold=r, n_seq=len(hold_idx),
            stems=[winner[i]["stem"] for i in hold_idx],
            winner=dict(iou=float(mw["iou"]), tp=float(mw["tp"]), fp=float(mw["fp"]), fn=float(mw["fn"])),
            blend=dict(iou=float(mb["iou"]), tp=float(mb["tp"]), fp=float(mb["fp"]), fn=float(mb["fn"])),
            delta=float(mb["iou"] - mw["iou"]),
        ))
        log(f"nested k=5 fold {r} winner {mw['iou']:.4f} blend {mb['iou']:.4f} delta {mb['iou'] - mw['iou']:+.4f}")
    iw = float(sel.micro_iou(parts_w))
    ib = float(sel.micro_iou(parts_b))
    return dict(k=5, winner_iou=iw, blend_iou=ib, delta=float(ib - iw), folds=details,
                procedure="scripts/s3_sel2_cv.py sequence_folds i%5, holdout FAR 1e-3, micro IoU")


def evaluate(train_meta: dict) -> dict:
    if not CKPT.exists():
        raise SystemExit("нет чекпойнта")
    model = BoundaryUNet(16)
    model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True))
    model.eval()
    freeze_bn(model)
    # главная голова на eval совпадает с DTSUNet тех же весов
    ref = models.DTSUNet(w=16)
    ref.load_state_dict(model.net.state_dict())
    ref.eval()
    probe = torch.zeros(1, 3, 64, 80)
    with torch.no_grad():
        err = float((model(probe) - ref(probe)).abs().max())
    log(f"совпадение с DTSUNet max|Δ| {err:.3e}")
    if err > 1e-5:
        raise SystemExit("eval-граф не совпал с DTSUNet")
    del ref, probe
    val, pair = score_split(model, "val", None, collect_nested=True)
    n_bg = val.pop("_n_bg")
    thr = {n: val[n]["threshold"] for n in ("unet_boundary", "ens_blend", WINNER)}
    if abs(val[WINNER]["iou"] - WINNER_VAL) > 0.005:
        raise SystemExit(f"пересчёт победителя {val[WINNER]['iou']} далёк от {WINNER_VAL}")
    if pair is None:
        raise SystemExit("нет компактных кадров для 5-fold")
    nest = nested5(*pair)
    test, _pair = score_split(model, "test", thr)
    test.pop("_n_bg", None)
    if abs(test[WINNER]["iou"] - WINNER_TEST) > 0.005:
        raise SystemExit(f"test победителя {test[WINNER]['iou']} далёк от {WINNER_TEST}")
    sel_val = float(val["ens_blend"]["iou"])
    win_val = float(val[WINNER]["iou"])
    beats_val = bool(sel_val > win_val)
    nested_ok = bool(nest["delta"] >= NESTED_GAIN)
    accept = bool(beats_val and nested_ok)
    if accept:
        decision = (
            f"Принято. Ансамбль val IoU {sel_val:.4f} выше победителя {win_val:.4f}, "
            f"вложенный прирост {nest['delta']:+.4f} не меньше 0.003."
        )
    elif beats_val:
        decision = (
            f"Отклонено вложенной проверкой. Полный val ансамбля {sel_val:.4f} выше победителя {win_val:.4f}, "
            f"но прирост 5-fold {nest['delta']:+.4f} меньше 0.003."
        )
    else:
        decision = (
            f"Отклонено. Полный val ансамбля {sel_val:.4f} не выше победителя {win_val:.4f}. "
            f"Вложенный прирост {nest['delta']:+.4f}."
        )
    report = dict(
        name="BOUNDARY",
        protocol=(
            "Official: v8 val IoU at FAR 1e-3, threshold from the full v8 val background. "
            "Candidate is the mean of the boundary U-Net and ens_u2ft_tcn_lgbm. "
            "Nested: scripts/s3_sel2_cv.py procedure, k=5, threshold and IoU on the holdout, "
            "micro IoU, gain vs the winner under the same procedure must be >= 0.003. "
            "Test is not used for selection. Synthetic v8 and v10 train only."
        ),
        decision=decision,
        accept=accept,
        beats_val=beats_val,
        nested_ok=nested_ok,
        n_bg=n_bg,
        published_winner=dict(name=WINNER, val_iou=WINNER_VAL, test_iou=WINNER_TEST),
        val=val,
        test=test,
        nested5=nest,
        selected="ens_blend",
        selected_val_iou=sel_val,
        selected_test_iou=float(test["ens_blend"]["iou"]),
        train=train_meta,
        real_used_for_training=False,
        real_used_for_threshold=False,
        real_noise_used=False,
    )
    dump(OUT_EVAL, report)
    log(decision)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    t0 = time.time()
    if args.eval_only:
        train_meta = json.loads(OUT_TRAIN.read_text()) if OUT_TRAIN.exists() else {}
    else:
        train_meta = train()
    report = evaluate(train_meta)
    report["seconds"] = round(time.time() - t0, 1)
    dump(OUT_EVAL, report)
    log(f"записан {OUT_EVAL}")


if __name__ == "__main__":
    main()
