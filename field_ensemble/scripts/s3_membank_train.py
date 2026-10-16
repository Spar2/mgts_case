"""Голова банка нормы на замороженном энкодере unet_v810ft. Только synth v8+v10 train.

Банк — FIFO фона train и корежет. Val и реальные CSV не читаются.
Чекпойнт artifacts/ckpt/s3_membank/last.pt: веса, банк, оптимизатор, шаг, RNG.
Не реже раза в 15 минут. Запуск:

  scripts/run_resumable.sh s3_membank /workspace/repo/run/.venv/bin/python -u scripts/s3_membank_train.py
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
import zlib
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
import sys
sys.path.insert(0, str(ROOT))

from dts import data
from dts import metrics as M
from dts.membank import (
    BANK_CAP,
    CORESET,
    EMBED,
    TILE,
    WINNER_ENCODER,
    MemBankNet,
    fifo_append,
    greedy_coreset,
    membank_loss,
    trainable_parameters,
)

RUN = "s3_membank"
CKPT_DIR = Path("artifacts/ckpt") / RUN
LAST = CKPT_DIR / "last.pt"
BEST = Path("artifacts/eval/s3_membank_prior.pt")
TRAIN_JSON = Path("artifacts/eval/s3_membank_train.json")
LOG = Path("/workspace/run_logs/_events.log")
SAVE_EVERY = 15 * 60


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S', time.gmtime())} {msg}"
    print(line, flush=True)


def event(msg: str) -> None:
    log(msg)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%H:%M', time.gmtime())} {msg}\n")


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


def forbid_real(path: str) -> None:
    s = str(path)
    for bad in ("MGTS", "real_noise", "models_release"):
        if bad in s:
            raise SystemExit(f"реальный след в обучении: {s}")


class TileSource:
    def __init__(self):
        self.mm = {}
        self.lru = OrderedDict()

    def get(self, win):
        path, mx, my, t0, w0 = win
        th, tw = TILE
        forbid_real(path)
        if mx:
            if mx not in self.mm:
                self.mm[mx] = np.load(mx, mmap_mode="r")
                self.mm[my] = np.load(my, mmap_mode="r")
            x = np.array(self.mm[mx][:, t0:t0 + th, w0:w0 + tw], dtype=np.float32, copy=True)
            lab = np.array(self.mm[my][t0:t0 + th, w0:w0 + tw], dtype=np.int16, copy=True)
            return x, lab
        key = str(path)
        if key not in self.lru:
            with np.load(path) as z:
                xfull = np.asarray(z["X"])
                labfull = data.target(np.asarray(z["ev"]), np.asarray(z["valid"]))
            self.lru[key] = (xfull, labfull)
            while len(self.lru) > 3:
                self.lru.popitem(last=False)
        else:
            self.lru.move_to_end(key)
        xfull, labfull = self.lru[key]
        x = np.array(xfull[:, t0:t0 + th, w0:w0 + tw], dtype=np.float32, copy=True)
        lab = np.array(labfull[t0:t0 + th, w0:w0 + tw], dtype=np.int16, copy=True)
        return x, lab


def _hold(key: str) -> bool:
    return (zlib.crc32(key.encode()) & 0xFFFFFFFF) % 100 < 12


def index_windows():
    th, tw = TILE
    groups = {"train_pos": [], "train_neg": [], "hold_pos": [], "hold_neg": []}
    seen = set()
    for ver in ("v8", "v10"):
        files = data.list_cache(f"artifacts/cache/{ver}", "train")
        if len(files) < 50:
            raise SystemExit(f"мало {ver} train: {len(files)}")
        t0c = time.time()
        n_pos = n_neg = 0
        for i, p in enumerate(files, 1):
            forbid_real(str(p))
            mx = Path(f"artifacts/mm/{ver}/train/{p.stem}_X.npy")
            my = Path(f"artifacts/mm/{ver}/train/{p.stem}_y.npy")
            if mx.exists() and my.exists():
                lab = np.load(my, mmap_mode="r")
                mx_s, my_s = str(mx), str(my)
            else:
                mx_s = my_s = ""
                with np.load(p) as z:
                    lab = data.target(np.asarray(z["ev"]), np.asarray(z["valid"]))
            t_len, w_len = lab.shape
            if t_len < th or w_len < tw:
                continue
            rng = np.random.default_rng(zlib.crc32(p.stem.encode()) & 0xFFFFFFFF)
            ys, xs = np.where(np.asarray(lab) > 0)
            if ys.size:
                take = min(3, int(ys.size))
                pick = rng.choice(int(ys.size), size=take, replace=False)
                for j in np.atleast_1d(pick):
                    t0 = int(np.clip(int(ys[j]) - th // 2, 0, t_len - th))
                    w0 = int(np.clip(int(xs[j]) - tw // 2, 0, w_len - tw))
                    key = (str(p), t0, w0)
                    if key in seen:
                        continue
                    seen.add(key)
                    bucket = "hold_pos" if _hold(f"{p}:{t0}:{w0}") else "train_pos"
                    groups[bucket].append((str(p), mx_s, my_s, t0, w0))
                    n_pos += 1
            got = 0
            for _ in range(12):
                t0 = int(rng.integers(0, t_len - th + 1))
                w0 = int(rng.integers(0, w_len - tw + 1))
                sl = np.asarray(lab[t0:t0 + th, w0:w0 + tw])
                if int(sl.max()) > 0 or float((sl != -1).mean()) < 0.85:
                    continue
                key = (str(p), t0, w0)
                if key in seen:
                    continue
                seen.add(key)
                bucket = "hold_neg" if _hold(f"{p}:{t0}:{w0}") else "train_neg"
                groups[bucket].append((str(p), mx_s, my_s, t0, w0))
                got += 1
                n_neg += 1
                if got >= 2:
                    break
            if i == 1 or i % 40 == 0 or i == len(files):
                log(f"индекс {ver} {i}/{len(files)} pos {n_pos} neg {n_neg} {time.time() - t0c:.0f}s")
    if len(groups["train_pos"]) < 40 or len(groups["train_neg"]) < 20:
        raise SystemExit(
            f"мало окон pos {len(groups['train_pos'])} neg {len(groups['train_neg'])}"
        )
    return groups


def sample_batch(rng, pos, neg, src: TileSource, n: int):
    xs, labs = [], []
    for _ in range(n):
        use_neg = bool(neg) and float(rng.random()) < 0.3
        pool = neg if use_neg else pos
        if not pool:
            pool = pos or neg
        win = pool[int(rng.integers(0, len(pool)))]
        x, lab = src.get(win)
        if rng.random() < 0.5:
            x = x[:, :, ::-1].copy()
            lab = lab[:, ::-1].copy()
        xs.append(x)
        labs.append(lab)
    xb = torch.from_numpy(np.stack(xs))
    yb = torch.from_numpy(np.stack(labs))
    return xb, yb


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


def save_last(model, opt, sched, step, epoch, rng, best, history, elapsed, raw) -> None:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    blob = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "step": int(step),
        "epoch": int(epoch),
        "rng": rng_blob(rng),
        "best": best,
        "history": history,
        "elapsed": float(elapsed),
        "bank_raw": raw.detach().cpu(),
        "tile": list(TILE),
        "encoder": WINNER_ENCODER,
        "coreset": CORESET,
        "bank_cap": BANK_CAP,
    }
    tmp = LAST.with_suffix(".pt.tmp")
    torch.save(blob, tmp)
    os.replace(tmp, LAST)


def save_best_weights(model) -> None:
    BEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = BEST.with_suffix(".pt.tmp")
    torch.save({"model": model.state_dict(), "encoder": WINNER_ENCODER}, tmp)
    os.replace(tmp, BEST)


@torch.no_grad()
def collect_vectors(model, windows, src, rng, n_tiles: int, per_tile: int = 192) -> torch.Tensor:
    model.eval()
    if not windows:
        return torch.zeros(0, EMBED)
    n_tiles = min(int(n_tiles), len(windows))
    pick = rng.choice(len(windows), size=n_tiles, replace=False)
    vecs = []
    for i in np.atleast_1d(pick):
        x, lab = src.get(windows[int(i)])
        v = model.embed_background(torch.from_numpy(x[None]), lab)
        if v.shape[0] == 0:
            continue
        if v.shape[0] > per_tile:
            sel = torch.randperm(v.shape[0])[:per_tile]
            v = v[sel]
        vecs.append(v)
    if not vecs:
        return torch.zeros(0, EMBED)
    return torch.cat(vecs, 0)


def refresh_bank(model, raw, windows, src, rng, n_tiles: int, seed: int):
    was = model.training
    vecs = collect_vectors(model, windows, src, rng, n_tiles)
    raw = fifo_append(raw, vecs, BANK_CAP)
    if raw.shape[0] < 8:
        raise RuntimeError(f"мало векторов фона {raw.shape[0]}")
    proto = greedy_coreset(raw, min(CORESET, int(raw.shape[0])), seed)
    model.set_bank(proto)
    if was:
        model.train()
    return raw


@torch.no_grad()
def proxy_metrics(model, hold_pos, hold_neg, src) -> dict:
    model.eval()
    scores, truth, bg = [], [], []
    for win in list(hold_pos)[:12] + list(hold_neg)[:6]:
        x, lab = src.get(win)
        logit, _hot, _extra = model(torch.from_numpy(x[None]), perturb=False)
        s = torch.sigmoid(logit[0, 0]).cpu().numpy()
        scores.append(s)
        truth.append(lab > 0)
        bg.append(lab == 0)
    s = np.stack(scores)
    truth_m = np.stack(truth)
    bg_m = np.stack(bg)
    if int(bg_m.sum()) < 100 or not truth_m.any():
        return {"iou": 0.0, "far": 1.0, "thr": 1.0, "sep": 0.0, "score": 0.0}
    thr = float(M.far_threshold(s[bg_m], 1e-3))
    pred = (s >= thr) & (truth_m | bg_m)
    tp = int((pred & truth_m).sum())
    fp = int((pred & bg_m).sum())
    fn = int((~pred & truth_m).sum())
    iou = tp / max(tp + fp + fn, 1)
    far = fp / max(int(bg_m.sum()), 1)
    sep = float(s[truth_m].mean() - s[bg_m].mean())
    model.train()
    return {"iou": float(iou), "far": float(far), "thr": thr, "sep": sep, "score": float(iou + sep)}


def encoder_delta(model) -> float:
    ref = torch.load(WINNER_ENCODER, map_location="cpu", weights_only=True)
    delta = 0.0
    for k, v in model.encoder.state_dict().items():
        if v.dtype.is_floating_point and k in ref:
            delta = max(delta, float((v - ref[k]).abs().max()))
    return delta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", default="auto", choices=("auto", "never", "must"))
    args = ap.parse_args()
    if (CKPT_DIR / "DONE").exists() and args.resume != "never":
        log("DONE уже есть, обучение не повторяю")
        return
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    epochs = int(os.environ.get("MEMBANK_EPOCHS", "30"))
    steps = int(os.environ.get("MEMBANK_STEPS", "60"))
    batch = int(os.environ.get("MEMBANK_BATCH", "2"))
    lr = float(os.environ.get("MEMBANK_LR", "3e-4"))
    max_minutes = float(os.environ.get("MEMBANK_MAX_MINUTES", "35"))
    patience = int(os.environ.get("MEMBANK_PATIENCE", "8"))
    seed = int(os.environ.get("MEMBANK_SEED", "0"))

    log("индекс окон")
    groups = index_windows()
    log(
        f"окна pos {len(groups['train_pos'])}/{len(groups['hold_pos'])} "
        f"neg {len(groups['train_neg'])}/{len(groups['hold_neg'])}"
    )
    src = TileSource()
    rng = np.random.default_rng(seed)
    model = MemBankNet()
    model.encoder.load_winner(WINNER_ENCODER)
    model.train()
    params = list(trainable_parameters(model))
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda s: 1.0)
    step = 0
    start_ep = 0
    history = []
    best = {"score": -1e9, "epoch": -1, "iou": None, "far": None}
    elapsed0 = 0.0
    bad = 0
    raw = torch.zeros(0, EMBED)
    if args.resume == "never":
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        log("старт с нуля (resume never)")
    elif LAST.exists():
        ckpt = torch.load(LAST, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        for p in model.encoder.parameters():
            p.requires_grad_(False)
        model.encoder.eval()
        opt.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler"):
            sched.load_state_dict(ckpt["scheduler"])
        step = int(ckpt["step"])
        start_ep = int(ckpt["epoch"]) + 1
        history = list(ckpt.get("history") or [])
        best = dict(ckpt["best"])
        elapsed0 = float(ckpt.get("elapsed") or 0.0)
        raw = ckpt.get("bank_raw")
        if raw is None or raw.numel() == 0:
            raw = torch.zeros(0, EMBED)
        rng_restore(ckpt["rng"], rng)
        bad = sum(1 for h in history if int(h["epoch"]) > int(best["epoch"]))
        log(f"resume epoch {ckpt['epoch']} step {step} best {best['score']:.4f} bank {int(model.bank_count)} elapsed {elapsed0:.0f}s")
    elif args.resume == "must":
        raise SystemExit("resume must: нет last.pt")
    else:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        log("старт с нуля")

    t0 = time.time()

    def spent() -> float:
        return elapsed0 + (time.time() - t0)

    if int(model.bank_count) < 3:
        log("сбор начального банка фона")
        raw = refresh_bank(model, raw if raw.numel() else torch.zeros(0, EMBED), groups["train_neg"], src, rng, 48, seed=1)
        log(f"банк raw {raw.shape[0]} proto {int(model.bank_count)}")
    save_last(model, opt, sched, step, start_ep - 1, rng, best, history, spent(), raw)

    model.train()
    probe_x = torch.zeros(1, 3, TILE[0], TILE[1])
    probe_y = torch.zeros(1, TILE[0], TILE[1], dtype=torch.int16)
    probe_y[:, 40:70, 30:50] = 1
    before = {k: v.detach().clone() for k, v in model.encoder.state_dict().items()}
    opt.zero_grad()
    logit, hot, _extra = model(probe_x, perturb=False)
    loss = membank_loss(logit, hot, probe_y)
    if not torch.isfinite(loss):
        raise SystemExit("проба: нечисло")
    loss.backward()
    opt.zero_grad()
    for k, v in model.encoder.state_dict().items():
        if v.dtype.is_floating_point and not torch.equal(v, before[k]):
            raise SystemExit(f"энкодер сдвинулся на пробе {k}")
    log(f"проба loss {float(loss.detach()):.4f} энкодер заморожен")

    last_save = time.time()
    stopped = "epochs"
    for ep in range(start_ep, start_ep + epochs):
        if spent() > (max_minutes - 2) * 60:
            stopped = "time"
            break
        losses = []
        model.train()
        for _ in range(steps):
            if spent() > max_minutes * 60:
                stopped = "time"
                break
            xb, yb = sample_batch(rng, groups["train_pos"], groups["train_neg"], src, batch)
            opt.zero_grad()
            logit, hot, extra = model(xb, perturb=True)
            loss = membank_loss(logit, hot, yb, extra=extra)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            step += 1
            losses.append(float(loss.detach()))
            if time.time() - last_save >= SAVE_EVERY:
                save_last(model, opt, sched, step, ep, rng, best, history, spent(), raw)
                last_save = time.time()
                log(f"last.pt step {step}")
        if not losses:
            stopped = "time"
            break
        pr = proxy_metrics(model, groups["hold_pos"], groups["hold_neg"], src)
        rec = dict(epoch=ep, train=float(np.mean(losses)), step=step, sec=round(spent(), 1), **pr)
        history.append(rec)
        msg = f"MEMBANK_TRAIN epoch={ep} loss={rec['train']:.4f} proxy_iou={pr['iou']:.4f} far={pr['far']:.4f} sec={rec['sec']:.0f}"
        event(msg)
        if pr["score"] > float(best["score"]) + 1e-4:
            best = dict(score=pr["score"], epoch=ep, iou=pr["iou"], far=pr["far"])
            bad = 0
            save_best_weights(model)
            log(f"сохранён epoch {ep} score {pr['score']:.4f}")
        else:
            bad += 1
            if bad >= patience and ep >= 2:
                stopped = "patience"
        raw = refresh_bank(model, raw, groups["train_neg"], src, rng, 16, seed=1000 + ep)
        save_last(model, opt, sched, step, ep, rng, best, history, spent(), raw)
        last_save = time.time()
        log(f"банк обновлён raw {raw.shape[0]} proto {int(model.bank_count)}")
        if stopped in ("time", "patience"):
            break

    save_last(model, opt, sched, step, history[-1]["epoch"] if history else -1, rng, best, history, spent(), raw)
    scored = MemBankNet()
    scored.encoder.load_winner(WINNER_ENCODER)
    if best["epoch"] >= 0 and BEST.exists():
        blob = torch.load(BEST, map_location="cpu", weights_only=False)
        scored.load_state_dict(blob["model"])
    else:
        scored.load_state_dict(model.state_dict())
    for p in scored.encoder.parameters():
        p.requires_grad_(False)
    scored.encoder.eval()
    final_probe = proxy_metrics(scored, groups["hold_pos"], groups["hold_neg"], src)
    dw = encoder_delta(scored)
    n_head = sum(p.numel() for p in trainable_parameters(model))
    body = dict(
        name="MEMBANK",
        encoder=WINNER_ENCODER,
        encoder_frozen=True,
        encoder_max_dw=dw,
        encoder_stages="e1+e2",
        embed=EMBED,
        bank_cap=BANK_CAP,
        coreset=CORESET,
        knn=3,
        agg_kernel=list((9, 3)),
        blend="(4*ens_u2ft_tcn_lgbm+s_membank)/5",
        score="sigmoid MLP+bilinear(query, nearest, dist stats)",
        tile=list(TILE),
        lr=lr,
        weight_decay=1e-4,
        batch=batch,
        steps_per_epoch=steps,
        head_params=int(n_head),
        stopped=stopped,
        seconds=round(spent(), 1),
        best=best,
        final_probe=final_probe,
        bank_raw=int(raw.shape[0]),
        bank_proto=int(scored.bank_count),
        history=history,
        train_pos=len(groups["train_pos"]),
        train_neg=len(groups["train_neg"]),
        hold_pos=len(groups["hold_pos"]),
        hold_neg=len(groups["hold_neg"]),
        real_traces_used=False,
        real_noise_used=False,
        imagenet_weights=False,
    )
    dump_json(TRAIN_JSON, body)
    (CKPT_DIR / "DONE").touch()
    event(
        f"MEMBANK_TRAIN_END best_epoch={best['epoch']} proxy={best['score']:.4f} "
        f"dw={dw:.3e} stop={stopped} sec={spent():.0f} bank={int(scored.bank_count)}"
    )


if __name__ == "__main__":
    main()
