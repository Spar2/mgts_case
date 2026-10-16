"""Обучение одношагового flow-прайора фона. Только синтетика v8+v10 train.

Тайл целиком без события и без nuisance. Реальные CSV и спектр шума не читаются.
Чекпойнт artifacts/ckpt/s3_flow/last.pt (веса, оптимизатор, scheduler, шаг, RNG, лучшая метрика)
пишется не реже чем раз в 15 минут. Запуск: scripts/run_resumable.sh s3_flow <python> -u scripts/s3_flow_train.py

  /workspace/repo/run/.venv/bin/python -u scripts/s3_flow_train.py --resume auto
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "NUMEXPR_MAX_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_k] = "2"

import argparse
import json
import random
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
import sys
sys.path.insert(0, str(ROOT))

from dts import data
from dts.flowprior import T_STAR, VelocityNet, assert_euler_identity, copy_probe, flow_loss, residual_map

RUN = "s3_flow"
CKPT_DIR = Path("artifacts/ckpt") / RUN
LAST = CKPT_DIR / "last.pt"
BEST = Path("artifacts/eval/s3_flow_prior.pt")
TRAIN_JSON = Path("artifacts/eval/s3_flow_train.json")
TILE = (128, 256)
SAVE_EVERY = 15 * 60


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S', time.gmtime())} {msg}", flush=True)


def plain(obj):
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(plain(obj), indent=1, ensure_ascii=False))
    os.replace(tmp, path)


class TileSource:
    def __init__(self):
        self.mm = {}
        self.lru = OrderedDict()

    def get(self, win):
        path, mmap_path, t0, w0 = win
        th, tw = TILE
        if mmap_path:
            key = str(mmap_path)
            if key not in self.mm:
                self.mm[key] = np.load(mmap_path, mmap_mode="r")
            X = self.mm[key]
            a = np.array(X[0, t0:t0 + th, w0:w0 + tw], dtype=np.float32, copy=True)
            v = np.array(X[2, t0:t0 + th, w0:w0 + tw], dtype=np.float32, copy=True)
            return a, v
        key = str(path)
        if key not in self.lru:
            with np.load(path) as z:
                self.lru[key] = np.asarray(z["X"])
            while len(self.lru) > 4:
                self.lru.popitem(last=False)
        else:
            self.lru.move_to_end(key)
        X = self.lru[key]
        a = np.array(X[0, t0:t0 + th, w0:w0 + tw], dtype=np.float32, copy=True)
        v = np.array(X[2, t0:t0 + th, w0:w0 + tw], dtype=np.float32, copy=True)
        return a, v


def index_windows() -> tuple[list, list]:
    th, tw = TILE
    train, hold = [], []
    for ver, use_mm in (("v8", True), ("v10", False)):
        files = data.list_cache(f"artifacts/cache/{ver}", "train")
        if len(files) < 100:
            raise SystemExit(f"мало {ver} train: {len(files)}")
        t0c = time.time()
        for i, p in enumerate(files, 1):
            with np.load(p) as z:
                ev = np.asarray(z["ev"])
                nu = np.asarray(z["nu"])
                valid = np.asarray(z["valid"])
            mmap = Path(f"artifacts/mm/v8/train/{p.stem}_X.npy") if use_mm else None
            if use_mm and not mmap.exists():
                raise SystemExit(f"нет mmap {mmap}")
            T, W = ev.shape
            n_here = 0
            for t0 in range(0, max(T - th, 0) + 1, th // 2):
                for w0 in range(0, max(W - tw, 0) + 1, tw // 2):
                    if int(ev[t0:t0 + th, w0:w0 + tw].max()) > 0:
                        continue
                    if int(nu[t0:t0 + th, w0:w0 + tw].max()) > 0:
                        continue
                    if float(valid[t0:t0 + th, w0:w0 + tw].mean()) < 0.8:
                        continue
                    win = (p, mmap, t0, w0)
                    # Стабильный holdout, не зависит от RNG обучения.
                    tag = zlib_key(f"{p}:{t0}:{w0}")
                    if tag % 100 < 15:
                        hold.append(win)
                    else:
                        train.append(win)
                    n_here += 1
            if i == 1 or i % 40 == 0 or i == len(files):
                log(f"индекс {ver} {i}/{len(files)} окон+ {n_here} {time.time() - t0c:.0f}s")
    if len(train) < 200 or len(hold) < 16:
        raise SystemExit(f"мало чистых тайлов train {len(train)} hold {len(hold)}")
    return train, hold


def zlib_key(text: str) -> int:
    import zlib
    return zlib.crc32(text.encode()) & 0xFFFFFFFF


def sample_batch(rng: np.random.Generator, wins, src: TileSource, n: int):
    xs, vs = [], []
    for _ in range(n):
        w = wins[int(rng.integers(0, len(wins)))]
        a, v = src.get(w)
        if rng.random() < 0.5:
            a = a[:, ::-1].copy()
            v = v[:, ::-1].copy()
        xs.append(a)
        vs.append(v)
    x = np.stack(xs)[:, None]
    m = np.stack(vs)[:, None]
    return torch.from_numpy(x), torch.from_numpy(m)


def rng_blob(rng: np.random.Generator) -> dict:
    return {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
        "sample": rng.bit_generator.state,
    }


def rng_restore(blob, rng: np.random.Generator) -> None:
    torch.set_rng_state(blob["torch"])
    np.random.set_state(blob["numpy"])
    random.setstate(blob["python"])
    rng.bit_generator.state = blob["sample"]


def save_last(model, opt, sched, step, epoch, rng, best, history, elapsed) -> None:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": None if sched is None else sched.state_dict(),
        "step": int(step),
        "epoch": int(epoch),
        "rng": rng_blob(rng),
        "best": best,
        "history": history,
        "elapsed": float(elapsed),
        "t_star": T_STAR,
    }
    tmp = LAST.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, LAST)


def save_best_weights(model) -> None:
    BEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = BEST.with_suffix(".pt.tmp")
    torch.save(model.state_dict(), tmp)
    os.replace(tmp, BEST)


def proxy_metrics(model, hold, src, n_bg: int = 4) -> dict:
    model.eval()
    abs_r = []
    for w in hold[:n_bg]:
        a, v = src.get(w)
        r = residual_map(model, a, v, tile=TILE, overlap=0, batch=1)
        m = v > 0.5
        abs_r.append(float(np.mean(np.abs(r[m]))))
    bg = float(np.mean(abs_r))
    probes = [copy_probe(model, *src.get(w)) for w in hold[:2]]
    restore = float(np.mean([p["restore"] for p in probes]))
    denoise = float(np.mean([p["denoise"] for p in probes]))
    model.train()
    return {"bg_abs": bg, "restore": restore, "denoise": denoise, "score": restore - bg}


def fit_cdf(model, hold, src, n: int = 12, cap: int = 200_000) -> np.ndarray:
    model.eval()
    parts = []
    for w in hold[:n]:
        a, v = src.get(w)
        r = residual_map(model, a, v, tile=TILE, overlap=0, batch=1)
        m = v > 0.5
        parts.append(np.abs(r[m]).ravel())
    cat = np.concatenate(parts)
    if cat.size > cap:
        cat = np.random.default_rng(0).choice(cat, cap, replace=False)
    knots = np.quantile(cat, np.linspace(0.0, 1.0, 2049)).astype(np.float64)
    model.train()
    return knots


def payload(stopped, elapsed, best, history, knots, n_train, n_hold) -> dict:
    return dict(
        name="FLOW",
        arch="velocity_unet",
        width=16,
        t_star=T_STAR,
        loss="rectified_flow_mse",
        lr=2e-4,
        weight_decay=0.0,
        tile=[128, 256],
        batch=int(os.environ.get("FLOW_BATCH", "8")),
        train_windows=n_train,
        hold_windows=n_hold,
        data="v8+v10 train, tiles with ev=0 and nu=0",
        real_traces_used=False,
        real_noise_used=False,
        blend="(4*winner + cdf(|residual|))/5",
        best=best,
        history=history,
        cdf_knots=knots,
        stopped=stopped,
        seconds=round(float(elapsed), 1),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", default="auto", choices=("auto", "never", "must"))
    args = ap.parse_args()
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    epochs = int(os.environ.get("FLOW_EPOCHS", "40"))
    steps = int(os.environ.get("FLOW_STEPS", "60"))
    batch = int(os.environ.get("FLOW_BATCH", "8"))
    lr = float(os.environ.get("FLOW_LR", "2e-4"))
    max_minutes = float(os.environ.get("FLOW_MAX_MINUTES", "58"))
    patience = int(os.environ.get("FLOW_PATIENCE", "6"))
    seed = int(os.environ.get("FLOW_SEED", "0"))

    assert_euler_identity()
    train_w, hold_w = index_windows()
    log(f"чистые тайлы train {len(train_w)} hold {len(hold_w)}")
    src = TileSource()
    rng = np.random.default_rng(seed)
    model = VelocityNet(w=16)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda s: 1.0)
    step = 0
    start_ep = 0
    history = []
    best = {"score": -1e9, "epoch": -1, "bg_abs": None, "restore": None, "denoise": None}
    elapsed0 = 0.0
    bad = 0
    if args.resume == "never":
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    elif LAST.exists():
        ckpt = torch.load(LAST, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler"):
            sched.load_state_dict(ckpt["scheduler"])
        step = int(ckpt["step"])
        start_ep = int(ckpt["epoch"]) + 1
        history = list(ckpt.get("history") or [])
        best = dict(ckpt["best"])
        elapsed0 = float(ckpt.get("elapsed") or 0.0)
        rng_restore(ckpt["rng"], rng)
        bad = sum(1 for h in history if int(h["epoch"]) > int(best["epoch"]))
        log(f"resume epoch {ckpt['epoch']} step {step} best {best['score']:.4f} elapsed {elapsed0:.0f}s")
    elif args.resume == "must":
        raise SystemExit("resume must: нет last.pt")
    else:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        log("старт с нуля")

    _probe = VelocityNet(w=16)
    _probe.eval()
    with torch.no_grad():
        _y = _probe(torch.zeros(1, 1, 128, 256), torch.ones(1, 1, 128, 256), torch.zeros(1, 1, 1, 1))
    if not torch.isfinite(_y).all() or tuple(_y.shape) != (1, 1, 128, 256):
        raise SystemExit(f"проба сети {_y.shape}")
    del _probe, _y
    model.train()

    t0 = time.time()

    def spent() -> float:
        return elapsed0 + (time.time() - t0)

    last_save = time.time()
    stopped = "epochs"
    model.train()
    for ep in range(start_ep, start_ep + epochs):
        if spent() > (max_minutes - 3) * 60:
            stopped = "time"
            break
        losses = []
        for _ in range(steps):
            if spent() > max_minutes * 60:
                stopped = "time"
                break
            xb, mb = sample_batch(rng, train_w, src, batch)
            opt.zero_grad()
            loss = flow_loss(model, xb, mb)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            losses.append(float(loss.item()))
            if time.time() - last_save >= SAVE_EVERY:
                save_last(model, opt, sched, step, ep, rng, best, history, spent())
                last_save = time.time()
                log(f"last.pt step {step}")
        if not losses:
            stopped = "time"
            break
        pr = proxy_metrics(model, hold_w, src)
        rec = dict(epoch=ep, train=float(np.mean(losses)), steps_done=len(losses), step=step,
                   sec=round(spent(), 1), **pr)
        history.append(rec)
        log(str({k: rec[k] for k in ("epoch", "train", "bg_abs", "restore", "denoise", "score", "sec")}))
        if pr["score"] > float(best["score"]) + 1e-4:
            best = dict(score=pr["score"], epoch=ep, bg_abs=pr["bg_abs"], restore=pr["restore"], denoise=pr["denoise"])
            bad = 0
            save_best_weights(model)
            log(f"сохранён epoch {ep} score {pr['score']:.4f}")
        else:
            bad += 1
            if bad >= patience and ep >= 2:
                stopped = "patience"
        save_last(model, opt, sched, step, ep, rng, best, history, spent())
        last_save = time.time()
        if stopped in ("time", "patience"):
            break
    save_last(model, opt, sched, step, history[-1]["epoch"] if history else -1, rng, best, history, spent())
    scored = VelocityNet(w=16)
    if best["epoch"] >= 0 and BEST.exists():
        scored.load_state_dict(torch.load(BEST, map_location="cpu", weights_only=True))
    else:
        scored.load_state_dict(model.state_dict())
    knots = fit_cdf(scored, hold_w, src)
    final_probe = proxy_metrics(scored, hold_w, src)
    body = payload(stopped, spent(), best, history, knots, len(train_w), len(hold_w))
    body["final_probe"] = final_probe
    dump_json(TRAIN_JSON, body)
    (CKPT_DIR / "DONE").touch()
    log(f"готово best_epoch {best['epoch']} score {best['score']:.4f} stop {stopped} sec {spent():.0f}")


if __name__ == "__main__":
    main()
