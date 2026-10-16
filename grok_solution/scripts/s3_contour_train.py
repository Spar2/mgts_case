"""Голова контура на замороженном энкодере unet_v810ft. Только synth v8+v10 train.

Чекпойнт artifacts/ckpt/s3_contour/last.pt: голова+энкодер, оптимизатор, scheduler,
шаг, RNG, лучшая метрика. Не реже раза в 15 минут. Запуск:

  scripts/run_resumable.sh s3_contour /workspace/repo/run/.venv/bin/python -u scripts/s3_contour_train.py
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
from dts.contour import (
    N_QUERIES,
    N_VERTS,
    TILE,
    WINNER_ENCODER,
    ContourNet,
    contour_loss,
    geometry_check,
    instances_from_lab,
    raster_tile,
)

RUN = "s3_contour"
CKPT_DIR = Path("artifacts/ckpt") / RUN
LAST = CKPT_DIR / "last.pt"
BEST = Path("artifacts/eval/s3_contour_prior.pt")
TRAIN_JSON = Path("artifacts/eval/s3_contour_train.json")
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


def cover_starts(a0: int, a1: int, n: int, tile: int) -> list[int]:
    if n <= tile:
        return [0]
    starts = []
    s = max(0, min(int(a0), n - tile))
    guard = 0
    while guard < 64:
        starts.append(s)
        if s + tile - 1 >= a1 or s >= n - tile:
            break
        nxt = min(s + tile // 2, n - tile)
        if nxt == s:
            break
        s = nxt
        guard += 1
    return starts


class TileSource:
    """v8: mmap X и y. v10: LRU из 4 npz, без записи npy."""

    def __init__(self):
        self.mm = {}
        self.lru = OrderedDict()

    def get(self, win):
        path, mx, my, t0, w0 = win
        th, tw = TILE
        if mx:
            kx, ky = str(mx), str(my)
            if kx not in self.mm:
                self.mm[kx] = np.load(mx, mmap_mode="r")
                self.mm[ky] = np.load(my, mmap_mode="r")
            x = np.array(self.mm[kx][:, t0:t0 + th, w0:w0 + tw], dtype=np.float32, copy=True)
            lab = np.array(self.mm[ky][t0:t0 + th, w0:w0 + tw], dtype=np.int8, copy=True)
            return x, lab
        key = str(path)
        if key not in self.lru:
            with np.load(path) as z:
                ev = np.asarray(z["ev"])
                valid = np.asarray(z["valid"])
                xfull = np.asarray(z["X"])
            self.lru[key] = (xfull, data.target(ev, valid))
            while len(self.lru) > 4:
                self.lru.popitem(last=False)
        else:
            self.lru.move_to_end(key)
        xfull, labfull = self.lru[key]
        x = np.array(xfull[:, t0:t0 + th, w0:w0 + tw], dtype=np.float32, copy=True)
        lab = np.array(labfull[t0:t0 + th, w0:w0 + tw], dtype=np.int8, copy=True)
        return x, lab


def index_windows():
    th, tw = TILE
    groups = {"train_pos": [], "train_neg": [], "hold_pos": [], "hold_neg": []}
    seen = set()
    for ver, use_mm in (("v8", True), ("v10", False)):
        files = data.list_cache(f"artifacts/cache/{ver}", "train")
        if len(files) < 100:
            raise SystemExit(f"мало {ver} train: {len(files)}")
        t0c = time.time()
        n_pos = n_neg = 0
        for i, p in enumerate(files, 1):
            if use_mm:
                my = Path(f"artifacts/mm/v8/train/{p.stem}_y.npy")
                mx = Path(f"artifacts/mm/v8/train/{p.stem}_X.npy")
                if not my.exists() or not mx.exists():
                    raise SystemExit(f"нет mmap {p.stem}")
                lab = np.load(my, mmap_mode="r")
            else:
                mx = my = None
                with np.load(p) as z:
                    lab = data.target(np.asarray(z["ev"]), np.asarray(z["valid"]))
            T, W = lab.shape
            if T < th or W < tw:
                continue
            hot = (np.asarray(lab) == 1).astype(np.uint8)
            cold = (np.asarray(lab) == 2).astype(np.uint8)
            import cv2
            for mask in (hot, cold):
                if int(mask.sum()) < 8:
                    continue
                n, comp = cv2.connectedComponents(mask, connectivity=8)
                added = 0
                for k in range(1, n):
                    ys, xs = np.where(comp == k)
                    if ys.size < 8:
                        continue
                    for t0 in cover_starts(int(ys.min()), int(ys.max()), T, th):
                        for w0 in cover_starts(int(xs.min()), int(xs.max()), W, tw):
                            key = (str(p), t0, w0)
                            if key in seen:
                                continue
                            seen.add(key)
                            bucket = "hold_pos" if (zlib.crc32(f"{p}:{t0}:{w0}".encode()) & 0xFFFFFFFF) % 100 < 15 else "train_pos"
                            groups[bucket].append((p, mx, my, t0, w0))
                            added += 1
                            n_pos += 1
                            if added >= 24:
                                break
                        if added >= 24:
                            break
                    if added >= 24:
                        break
            rng = np.random.default_rng(zlib.crc32(p.stem.encode()) & 0xFFFFFFFF)
            got = 0
            for _ in range(16):
                t0 = int(rng.integers(0, T - th + 1))
                w0 = int(rng.integers(0, W - tw + 1))
                sl = lab[t0:t0 + th, w0:w0 + tw]
                if int(sl.max()) > 0 or float((sl != -1).mean()) < 0.8:
                    continue
                key = (str(p), t0, w0)
                if key in seen:
                    continue
                seen.add(key)
                bucket = "hold_neg" if (zlib.crc32(f"{p}:{t0}:{w0}".encode()) & 0xFFFFFFFF) % 100 < 15 else "train_neg"
                groups[bucket].append((p, mx, my, t0, w0))
                got += 1
                n_neg += 1
                if got >= 2:
                    break
            if i == 1 or i % 40 == 0 or i == len(files):
                log(f"индекс {ver} {i}/{len(files)} pos {n_pos} neg {n_neg} {time.time() - t0c:.0f}s")
    if len(groups["train_pos"]) < 80 or len(groups["hold_pos"]) < 8:
        raise SystemExit(
            f"мало окон pos {len(groups['train_pos'])}/{len(groups['hold_pos'])} "
            f"neg {len(groups['train_neg'])}/{len(groups['hold_neg'])}"
        )
    return groups


def sample_batch(rng, pos, neg, src: TileSource, n: int):
    xs, labs = [], []
    for _ in range(n):
        use_neg = bool(neg) and float(rng.random()) < 0.25
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
    targets = [instances_from_lab(l) for l in labs]
    return torch.from_numpy(np.stack(xs)), targets


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
        "n_queries": N_QUERIES,
        "n_verts": N_VERTS,
        "tile": list(TILE),
        "encoder": WINNER_ENCODER,
    }
    tmp = LAST.with_suffix(".pt.tmp")
    torch.save(blob, tmp)
    os.replace(tmp, LAST)


def save_best_weights(model) -> None:
    BEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = BEST.with_suffix(".pt.tmp")
    torch.save(model.state_dict(), tmp)
    os.replace(tmp, BEST)


def tile_iou(score, lab) -> float:
    pred = score >= 0.5
    truth = (lab == 1) | (lab == 2)
    inter = np.logical_and(pred, truth).sum()
    union = np.logical_or(pred, truth).sum()
    if union == 0:
        return 1.0 if not pred.any() else 0.0
    return float(inter / union)


@torch.no_grad()
def proxy_metrics(model, hold_pos, hold_neg, src: TileSource) -> dict:
    model.eval()
    ious = []
    for win in hold_pos[:12]:
        x, lab = src.get(win)
        obj, hot, verts = model(torch.from_numpy(x[None]))
        conf = torch.sigmoid(obj[0, :, 0]).cpu().numpy()
        hs = torch.sigmoid(hot[0, :, 0]).cpu().numpy()
        score, _ = raster_tile(verts[0].cpu().numpy(), conf, hs, TILE[0], TILE[1])
        ious.append(tile_iou(score, lab))
    empty = []
    for win in hold_neg[:8]:
        x, lab = src.get(win)
        obj, hot, verts = model(torch.from_numpy(x[None]))
        conf = torch.sigmoid(obj[0, :, 0]).cpu().numpy()
        hs = torch.sigmoid(hot[0, :, 0]).cpu().numpy()
        score, _ = raster_tile(verts[0].cpu().numpy(), conf, hs, TILE[0], TILE[1])
        empty.append(float((score >= 0.5).mean()))
    iou = float(np.mean(ious)) if ious else 0.0
    emp = float(np.mean(empty)) if empty else 0.0
    return {"iou": iou, "empty": emp, "score": iou - emp}


def encoder_delta(model) -> float:
    ref = torch.load(WINNER_ENCODER, map_location="cpu", weights_only=True)
    delta = 0.0
    for k, v in model.encoder.state_dict().items():
        if v.dtype.is_floating_point:
            delta = max(delta, float((v - ref[k]).abs().max()))
    return delta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", default="auto", choices=("auto", "never", "must"))
    args = ap.parse_args()
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    import cv2
    cv2.setNumThreads(2)

    epochs = int(os.environ.get("CONTOUR_EPOCHS", "40"))
    steps = int(os.environ.get("CONTOUR_STEPS", "80"))
    batch = int(os.environ.get("CONTOUR_BATCH", "4"))
    lr = float(os.environ.get("CONTOUR_LR", "3e-4"))
    max_minutes = float(os.environ.get("CONTOUR_MAX_MINUTES", "58"))
    patience = int(os.environ.get("CONTOUR_PATIENCE", "8"))
    seed = int(os.environ.get("CONTOUR_SEED", "0"))

    geometry_check()
    log("геометрия ок")
    groups = index_windows()
    log(
        f"окна pos {len(groups['train_pos'])}/{len(groups['hold_pos'])} "
        f"neg {len(groups['train_neg'])}/{len(groups['hold_neg'])}"
    )
    src = TileSource()
    rng = np.random.default_rng(seed)
    model = ContourNet()
    model.encoder.load_winner(WINNER_ENCODER)
    model.train()
    opt = torch.optim.AdamW(model.head.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda s: 1.0)
    step = 0
    start_ep = 0
    history = []
    best = {"score": -1e9, "epoch": -1, "iou": None, "empty": None}
    elapsed0 = 0.0
    bad = 0
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

    # Проба графа на отдельном входе: шаг оптимизатора не делаем, RNG обучения не трогаем.
    model.train()
    probe_x = torch.zeros(1, 3, TILE[0], TILE[1])
    probe_lab = np.zeros(TILE, np.int8)
    probe_lab[30:50, 40:80] = 1
    before = {k: v.detach().clone() for k, v in model.encoder.state_dict().items()}
    opt.zero_grad()
    obj, hot, verts = model(probe_x)
    loss = contour_loss(obj, hot, verts, [instances_from_lab(probe_lab)])
    if not torch.isfinite(loss):
        raise SystemExit("проба: нечисло")
    loss.backward()
    opt.zero_grad()
    for k, v in model.encoder.state_dict().items():
        if v.dtype.is_floating_point and float((v - before[k]).abs().max()) != 0.0:
            raise SystemExit(f"энкодер сдвинулся на пробе {k}")
    log(f"проба loss {float(loss.detach()):.4f} энкодер заморожен")

    t0 = time.time()

    def spent() -> float:
        return elapsed0 + (time.time() - t0)

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
            xb, targets = sample_batch(rng, groups["train_pos"], groups["train_neg"], src, batch)
            opt.zero_grad()
            obj, hot, verts = model(xb)
            loss = contour_loss(obj, hot, verts, targets)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.head.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            losses.append(float(loss.detach()))
            if time.time() - last_save >= SAVE_EVERY:
                save_last(model, opt, sched, step, ep, rng, best, history, spent())
                last_save = time.time()
                log(f"last.pt step {step}")
        if not losses:
            stopped = "time"
            break
        pr = proxy_metrics(model, groups["hold_pos"], groups["hold_neg"], src)
        rec = dict(epoch=ep, train=float(np.mean(losses)), step=step, sec=round(spent(), 1), **pr)
        history.append(rec)
        log(str({k: rec[k] for k in ("epoch", "train", "iou", "empty", "score", "sec")}))
        if pr["score"] > float(best["score"]) + 1e-4:
            best = dict(score=pr["score"], epoch=ep, iou=pr["iou"], empty=pr["empty"])
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
    scored = ContourNet()
    scored.encoder.load_winner(WINNER_ENCODER)
    if best["epoch"] >= 0 and BEST.exists():
        scored.load_state_dict(torch.load(BEST, map_location="cpu", weights_only=True))
        for p in scored.encoder.parameters():
            p.requires_grad_(False)
        scored.encoder.eval()
    else:
        scored.load_state_dict(model.state_dict())
    final_probe = proxy_metrics(scored, groups["hold_pos"], groups["hold_neg"], src)
    dw = encoder_delta(scored)
    n_head = sum(p.numel() for p in model.head.parameters())
    body = dict(
        name="CONTOUR",
        encoder=WINNER_ENCODER,
        encoder_frozen=True,
        encoder_max_dw=dw,
        encoder_stages="e1+e2",
        n_queries=N_QUERIES,
        n_verts=N_VERTS,
        tile=list(TILE),
        overlap=16,
        refine_steps=1,
        blend="(4*winner+conf_inside_polygon)/5",
        lr=lr,
        weight_decay=1e-4,
        batch=batch,
        steps_per_epoch=steps,
        head_params=int(n_head),
        stopped=stopped,
        seconds=round(spent(), 1),
        best=best,
        final_probe=final_probe,
        history=history,
        train_pos=len(groups["train_pos"]),
        train_neg=len(groups["train_neg"]),
        hold_pos=len(groups["hold_pos"]),
        hold_neg=len(groups["hold_neg"]),
        real_traces_used=False,
        real_noise_used=False,
        coco_weights=False,
        imagenet_weights=False,
    )
    dump_json(TRAIN_JSON, body)
    (CKPT_DIR / "DONE").touch()
    log(f"готово best_epoch {best['epoch']} score {best['score']:.4f} dw {dw:.3e} stop {stopped} sec {spent():.0f}")


if __name__ == "__main__":
    main()
