"""Дообучение U-Net участника с аугментацией dts/gen2.py.

Только синтетика v8 и v10 train, на лету, без нового mmap.
Старт — artifacts/stage1/unet_v810ft.pt. Потолок 58 минут, 2 потока.
Снимок artifacts/ckpt/s3_gen2/last.pt не реже чем раз в 15 минут.
В конце — файл DONE.

  scripts/run_resumable.sh s3_gen2 /workspace/repo/run/.venv/bin/python -u scripts/s3_gen2_train.py
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_k] = "2"

import argparse
import json
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from dts import data, gen2, losses, metrics as M, models

RUN = "s3_gen2"
CKPT_DIR = Path("artifacts/ckpt") / RUN
LAST = CKPT_DIR / "last.pt"
BEST = Path("artifacts/eval/s3_gen2_unet.pt")
TRAIN_JSON = Path("artifacts/eval/s3_gen2_train.json")
INIT = Path("artifacts/stage1/unet_v810ft.pt")
ENS = Path("artifacts/pred/v8/ens_u2ft_tcn_lgbm/val")
SEED = 20261015
BUDGET_S = 54 * 60
SAVE_EVERY = 15 * 60
TILE = (128, 256)
BATCH = 6
LR = 8e-5
LR_ENC = 1e-5
WEAK_W = 3.0


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S", time.gmtime()) + " " + msg, flush=True)


def plain(obj):
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return plain(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(plain(obj), indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def atomic_torch(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def freeze_bn(model) -> None:
    for mod in model.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.eval()
            for p in mod.parameters():
                p.requires_grad_(False)


def bn_train_frozen(model) -> None:
    model.train()
    freeze_bn(model)


def rng_blob(rng):
    return dict(
        torch=torch.get_rng_state(),
        numpy=np.random.get_state(),
        python=random.getstate(),
        generator=rng.bit_generator.state,
    )


def restore_rng(rng, blob) -> None:
    torch.set_rng_state(blob["torch"])
    np.random.set_state(blob["numpy"])
    random.setstate(blob["python"])
    rng.bit_generator.state = blob["generator"]


def tile_starts(n, tile, step):
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


class Source:
    """Тайлы v8 (mmap) или v10 (npz LRU). Метаданные σ, t, x — маленькие."""

    def __init__(self, split_paths, mmap_root: Path | None, rng_index):
        self.paths = list(split_paths)
        self.mmap = []
        self.sigma = []
        self.t = []
        self.ev_idx = []
        self.shapes = []
        self.lru = OrderedDict()
        self.use_mmap = mmap_root is not None
        for i, p in enumerate(self.paths):
            with np.load(p) as z:
                ev = np.asarray(z["ev"])
                valid = np.asarray(z["valid"], bool)
                self.sigma.append(np.asarray(z["sigma"], np.float32))
                self.t.append(np.asarray(z["t"], np.float64))
            y = data.target(ev, valid)
            idx = np.argwhere(y > 0)
            if len(idx) > 2500:
                idx = idx[rng_index.choice(len(idx), 2500, replace=False)]
            self.ev_idx.append(idx)
            self.shapes.append(y.shape)
            if self.use_mmap:
                fx = mmap_root / f"{p.stem}_X.npy"
                fy = mmap_root / f"{p.stem}_y.npy"
                if not fx.exists() or not fy.exists():
                    raise SystemExit(f"нет mmap {p.stem}, новый кэш не создаю")
                self.mmap.append((
                    np.load(fx, mmap_mode="r"),
                    np.load(fy, mmap_mode="r"),
                ))
            if (i + 1) % 40 == 0 or i + 1 == len(self.paths):
                log(f"индекс {p.parent.parent.name}/{p.parent.name} {i + 1}/{len(self.paths)}")

    def _arrays(self, i):
        if self.use_mmap:
            X, y = self.mmap[i]
            return X, y
        path = str(self.paths[i])
        if path in self.lru:
            self.lru.move_to_end(path)
        else:
            with np.load(path) as z:
                X = np.asarray(z["X"])
                y = data.target(np.asarray(z["ev"]), np.asarray(z["valid"], bool))
            self.lru[path] = (X, y)
            while len(self.lru) > 3:
                self.lru.popitem(last=False)
        return self.lru[path]


def crop_pair(X, y, sigma, t, t0, w0, th, tw):
    T, W = y.shape
    xs = np.zeros((3, th, tw), np.float32)
    ys = np.full((th, tw), -1, np.int64)
    ht, wt = min(th, T - t0), min(tw, W - w0)
    xs[:, :ht, :wt] = np.array(X[:, t0:t0 + ht, w0:w0 + wt], np.float32, copy=True)
    ys[:ht, :wt] = np.array(y[t0:t0 + ht, w0:w0 + wt], np.int64, copy=True)
    sig = np.full(tw, 0.30, np.float32)
    sl = np.asarray(sigma[w0:w0 + wt], np.float32)
    sig[:sl.size] = sl
    tt = np.arange(th, dtype=np.float64) * 120.0
    if t0 < t.size:
        raw = np.asarray(t[t0:t0 + ht], np.float64)
        tt[:raw.size] = raw
        if raw.size < th and raw.size:
            dt = 120.0 if raw.size == 1 else float(raw[-1] - raw[-2] if raw.size > 1 else 120.0)
            if raw.size > 1:
                dt = float(raw[-1] - raw[-2])
            for k in range(raw.size, th):
                tt[k] = raw[-1] + dt * (k - raw.size + 1)
    return xs, ys, sig, tt


class Mixer:
    def __init__(self, v8: Source, v10: Source, rng):
        self.v8 = v8
        self.v10 = v10
        self.rng = rng
        self.th, self.tw = TILE

    def one(self, which: str):
        src = self.v8 if which == "v8" else self.v10
        i = int(self.rng.integers(0, len(src.paths)))
        T, W = src.shapes[i]
        th, tw = self.th, self.tw
        idx = src.ev_idx[i]
        if idx.size and self.rng.random() < 0.6:
            c = idx[int(self.rng.integers(0, len(idx)))]
            t0 = int(c[0]) - int(self.rng.integers(0, th))
            w0 = int(c[1]) - int(self.rng.integers(0, tw))
        else:
            t0 = int(self.rng.integers(0, max(T - th, 0) + 1))
            w0 = int(self.rng.integers(0, max(W - tw, 0) + 1))
        t0 = int(np.clip(t0, 0, max(T - th, 0)))
        w0 = int(np.clip(w0, 0, max(W - tw, 0)))
        X, y = src._arrays(i)
        xs, ys, sig, tt = crop_pair(X, y, src.sigma[i], src.t[i], t0, w0, th, tw)
        xs, ys, info = gen2.augment_tile(xs, ys, sig, tt, self.rng, force=False)
        if self.rng.random() < 0.5:
            xs = xs[:, :, ::-1].copy()
            ys = ys[:, ::-1].copy()
            info["static_mask"] = info["static_mask"][:, ::-1].copy()
            info["artifact_mask"] = info["artifact_mask"][:, ::-1].copy()
        g = float(self.rng.uniform(0.75, 1.3))
        xs[0] = np.clip(xs[0] * g, -12, 12)
        xs[1] = np.clip(xs[1] * g, -12, 12)
        valid = ys >= 0
        xs[0] *= valid
        xs[1] *= valid
        return xs, ys, gen2.pixel_weight(info)


def proxy_blend(model, files) -> float:
    """Один проход по четырём файлам v8 val. Это не официальный полный val."""
    model.eval()
    parts = []
    maps = []
    for p in files:
        with np.load(p) as z:
            X = np.asarray(z["X"], np.float32)
            ev = np.asarray(z["ev"])
            valid = np.asarray(z["valid"], bool)
            nu = np.asarray(z["nu"])
        prob = predict_batched(model, X)
        s = (1.0 - prob[0]).astype(np.float32)
        with np.load(ENS / p.name) as pred:
            ws = np.asarray(pred["s"], np.float32)
        blend = (0.5 * (s + ws)).astype(np.float32)
        bg = valid & (ev == 0) & (nu == 0)
        parts.append(np.asarray(blend[bg], np.float32))
        maps.append((blend, ev > 0, valid))
        del X, prob, s, ws
    thr = float(M.far_threshold(np.concatenate(parts), 1e-3))
    del parts
    tot = np.zeros(3, np.int64)
    for blend, truth, valid in maps:
        tot += np.asarray(M.confusion((blend >= thr) & valid, truth, valid), np.int64)
    iou = float(M.prf(*map(int, tot))["iou"])
    log(f"proxy ens_blend IoU {iou:.4f} thr {thr:.4f}")
    return iou


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", default="")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    gen2.self_check()
    log("self-check ok")
    rng = np.random.default_rng(args.seed)
    index_rng = np.random.default_rng(0)
    v8_paths = data.list_cache("artifacts/cache/v8", "train")
    v10_paths = data.list_cache("artifacts/cache/v10", "train")
    if len(v8_paths) < 4 or len(v10_paths) < 4:
        raise SystemExit(f"мало train: v8 {len(v8_paths)} v10 {len(v10_paths)}")
    v8 = Source(v8_paths, Path("artifacts/mm/v8/train"), index_rng)
    v10 = Source(v10_paths, None, index_rng)
    mixer = Mixer(v8, v10, rng)
    proxy_files = data.list_cache("artifacts/cache/v8", "val")[:4]
    model = models.build("unet", 16)
    model.load_state_dict(torch.load(INIT, map_location="cpu", weights_only=True))
    freeze_bn(model)
    enc, dec = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (enc if name.startswith(("e1", "e2", "e3", "e4", "gate")) else dec).append(p)
    opt = torch.optim.AdamW(
        [{"params": enc, "lr": LR_ENC}, {"params": dec, "lr": LR}],
        weight_decay=0.0,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=4000, eta_min=LR * 0.2)
    cw = torch.tensor([1.0, 3.0, 3.0])
    step = 0
    best = -1.0
    best_step = None
    history = []
    elapsed_prev = 0.0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    resume_path = None
    if args.resume == "auto" and LAST.is_file():
        resume_path = LAST
    elif args.resume and args.resume != "auto":
        resume_path = Path(args.resume)
    if resume_path is not None:
        blob = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model"])
        freeze_bn(model)
        opt.load_state_dict(blob["optim"])
        sched.load_state_dict(blob["sched"])
        restore_rng(rng, blob["rng"])
        step = int(blob["step"])
        best = float(blob["best_iou"])
        best_step = blob.get("best_step")
        history = list(blob.get("history") or [])
        elapsed_prev = float(blob.get("seconds") or 0.0)
        if blob.get("best_state"):
            best_state = blob["best_state"]
        log(f"resume step {step} best {best:.4f} spent {elapsed_prev:.0f}s")

    def snapshot(seconds: float) -> None:
        payload = dict(
            model=model.state_dict(),
            optim=opt.state_dict(),
            sched=sched.state_dict(),
            rng=rng_blob(rng),
            step=step,
            best_iou=best,
            best_step=best_step,
            best_state=best_state,
            history=history,
            seconds=seconds,
            seed=args.seed,
            init=str(INIT),
        )
        atomic_torch(LAST, payload)
        meta = dict(
            name="GEN2",
            init=str(INIT),
            step=step,
            best_step=best_step,
            best_proxy_iou=None if best < 0 else best,
            seconds=seconds,
            history=history,
            real_traces_used=False,
            real_noise_used=False,
            generator="dts/gen2.py",
            phenomena=["static_hot", "broad_shift", "hvac_20_90min", "artifact_-20C"],
        )
        dump_json(TRAIN_JSON, meta)

    t0 = time.time()
    deadline = t0 + max(0.0, BUDGET_S - elapsed_prev)
    if best < 0 and step == 0:
        bi = proxy_blend(model, proxy_files)
        best = bi
        best_step = 0
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(dict(step=0, proxy_blend_iou=bi, sec=0.0))
        atomic_torch(BEST, best_state)
        log(f"proxy step 0 {bi:.4f} (инициализация)")

    last_save = time.time()
    stopped = "budget"
    running = []
    while time.time() < deadline:
        bn_train_frozen(model)
        xs, ys, ws = [], [], []
        which = "v8" if step % 2 == 0 else "v10"
        for _ in range(BATCH):
            a, b, c = mixer.one(which)
            xs.append(a)
            ys.append(b)
            ws.append(c)
        xb = torch.from_numpy(np.stack(xs))
        yb = torch.from_numpy(np.stack(ys))
        wb = torch.from_numpy(np.stack(ws))
        opt.zero_grad(set_to_none=True)
        loss = losses.compute("focal+tversky_ce", model(xb), yb, cw, wk=wb, weak_weight=WEAK_W)
        if not torch.isfinite(loss):
            stopped = "nan"
            log(f"loss не число на шаге {step}")
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        running.append(float(loss.detach()))
        if step % 25 == 0:
            spent = elapsed_prev + (time.time() - t0)
            log(f"step {step} loss {float(np.mean(running[-25:])):.4f} {which} {spent:.0f}s")
        now = time.time()
        if now - last_save >= SAVE_EVERY and now < deadline - 30:
            bi = proxy_blend(model, proxy_files)
            spent = elapsed_prev + (now - t0)
            history.append(dict(step=step, proxy_blend_iou=bi, loss=float(np.mean(running[-25:])), sec=round(spent, 1)))
            if bi > best + 1e-4:
                best = bi
                best_step = step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                atomic_torch(BEST, best_state)
                log(f"новый лучший прокси {bi:.4f} шаг {step}")
            snapshot(spent)
            last_save = time.time()
            bn_train_frozen(model)
    else:
        stopped = "budget"

    if stopped != "nan" and time.time() < deadline + 180:
        bi = proxy_blend(model, proxy_files)
        spent = elapsed_prev + (time.time() - t0)
        history.append(dict(step=step, proxy_blend_iou=bi, sec=round(spent, 1), final=True))
        if bi > best + 1e-4:
            best = bi
            best_step = step
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            atomic_torch(BEST, best_state)
    spent = elapsed_prev + (time.time() - t0)
    snapshot(spent)
    atomic_torch(BEST, best_state)
    (CKPT_DIR / "DONE").touch()
    log(f"TRAIN_DONE step {step} best {best:.4f} at {best_step} stop {stopped} {spent:.0f}s")


if __name__ == "__main__":
    main()
