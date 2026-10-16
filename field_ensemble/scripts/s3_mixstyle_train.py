"""Дообучение U-Net с MixStyle на синтетике v8+v10.

Старт — artifacts/stage1/unet_v810ft.pt. Реальные CSV, спектры и σ реального
фона не читаются. Чекпойнт — прокси IoU@FAR 1e-3 на первых восьми v8 val.
Официальный отбор — scripts/s3_mixstyle_eval.py.

  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \\
  /workspace/repo/run/.venv/bin/python -u scripts/s3_mixstyle_train.py
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_k] = "2"

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from dts import data, losses, metrics as M, models
from dts.mixstyle import MixStyleUNet


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
    return obj


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plain(obj), indent=1, ensure_ascii=False))


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
    was = model.training
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
    model.train(was)
    return out


def assert_identity(model: MixStyleUNet) -> float:
    """На eval MixStyle молчит: логиты совпадают с обычным DTSUNet тех же весов.

    Пробный train-проход не должен менять BN: после него состояние восстанавливается.
    """
    snap = {k: v.detach().clone() for k, v in model.state_dict().items()}
    ref = models.DTSUNet(w=16)
    ref.load_state_dict(model.backbone.state_dict())
    ref.eval()
    model.eval()
    x = torch.randn(2, 3, 64, 96)
    with torch.no_grad():
        err = float((model(x) - ref(x)).abs().max())
    if err > 1e-5:
        raise SystemExit(f"MixStyle на eval не тождество: max|Δ| {err}")
    model.train()
    with torch.no_grad():
        y = model(x)
    if not torch.isfinite(y).all():
        raise SystemExit("MixStyle дал нечисло на train")
    model.load_state_dict(snap)
    model.train()
    return err


def assert_mmap(ver: str, split: str) -> None:
    cache = Path("artifacts/cache") / ver / split
    mm = Path("artifacts/mm") / ver / split
    missing = [p.name for p in sorted(cache.glob("seq*.npz")) if not (mm / f"{p.stem}_y.npy").exists()]
    if missing:
        raise SystemExit(f"нет mmap {ver}/{split}: {len(missing)} — новый npy не пишу")


class LazyX:
    """Срез X из npz по запросу. Весь v10 train в RAM не держим и npy не пишем."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def __getitem__(self, item):
        with np.load(self.path) as z:
            return np.array(z["X"][item], copy=True)


def load_v10_ram() -> list:
    """v10 train: y int8 в памяти, X читается из npz. На диск npy не пишутся."""
    items = []
    files = data.list_cache("artifacts/cache/v10", "train")
    if len(files) < 100:
        raise SystemExit(f"мало v10 train: {len(files)}")
    t0 = time.time()
    for i, p in enumerate(files, 1):
        with np.load(p) as z:
            y = data.target(np.asarray(z["ev"]), np.asarray(z["valid"]))
        items.append(dict(X=LazyX(p), y=y, name=p.stem, src=p))
        if i == 1 or i % 40 == 0 or i == len(files):
            log(f"v10 метки {i}/{len(files)} {time.time() - t0:.0f}s")
    return items


def proxy_iou(model, files) -> tuple[float, float]:
    model.eval()
    parts, maps = [], []
    for p in files:
        with np.load(p) as z:
            prob = predict_batched(model, np.asarray(z["X"], np.float32))
            valid = np.asarray(z["valid"], bool)
            ev = np.asarray(z["ev"])
            nu = np.asarray(z["nu"])
        s = 1.0 - prob[0]
        parts.append(s[valid & (ev == 0) & (nu == 0)])
        maps.append((s, valid, ev > 0))
    thr = M.far_threshold(np.concatenate(parts), 1e-3)
    tot = np.zeros(3, np.int64)
    for s, valid, truth in maps:
        tot += np.asarray(M.confusion((s >= thr) & valid, truth, valid), np.int64)
    model.train()
    return float(M.prf(*tot)["iou"]), float(thr)


def main() -> None:
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    epochs = int(os.environ.get("MIX_EPOCHS", "8"))
    steps = int(os.environ.get("MIX_STEPS", "80"))
    batch = int(os.environ.get("MIX_BATCH", "8"))
    lr = float(os.environ.get("MIX_LR", "1e-4"))
    max_minutes = float(os.environ.get("MIX_MAX_MINUTES", "44"))
    seed = int(os.environ.get("MIX_SEED", "0"))
    patience = int(os.environ.get("MIX_PATIENCE", "3"))
    tile = (128, 256)
    init = Path(os.environ.get("MIX_INIT", "artifacts/stage1/unet_v810ft.pt"))
    resume = os.environ.get("MIX_RESUME", "0") == "1"
    out_json = Path("artifacts/eval/s3_mixstyle_train.json")
    out_pt = Path("artifacts/eval/s3_mixstyle_unet.pt")

    torch.manual_seed(seed)
    np.random.seed(seed)
    assert_mmap("v8", "train")
    assert_mmap("v8", "val")
    tr = data.mmap_items("artifacts/cache/v8", "train", Path("artifacts/mm/v8"))
    va = data.mmap_items("artifacts/cache/v8", "val", Path("artifacts/mm/v8"))
    tr.extend(load_v10_ram())
    log(f"train items {len(tr)} (v8 mmap + v10 ram) val {len(va)}")

    sampler = data.TileSampler(tr, tile=tile, seed=seed, p_weak=0.0, aug=None)
    sval = data.TileSampler(va, tile=tile, seed=1234, p_event=0.6)
    vx, vy = sval.sample(64)
    del va, sval

    model = MixStyleUNet(w=16, p=0.5, alpha=0.1)
    model.load_init(init)
    err = assert_identity(model)
    log(f"identity eval max|Δ| {err:.3e}")
    opt = torch.optim.AdamW(model.backbone.parameters(), lr=lr, weight_decay=0.0)
    cw = torch.tensor([1.0, 3.0, 3.0])

    @torch.no_grad()
    def val_loss() -> float:
        model.eval()
        losses_ep = []
        for i in range(0, len(vx), 8):
            logit = model(torch.from_numpy(vx[i:i + 8]))
            losses_ep.append(losses.compute("focal+tversky_ce", logit, torch.from_numpy(vy[i:i + 8]), cw).item())
        model.train()
        return float(np.mean(losses_ep))

    def save_ckpt():
        tmp = out_pt.with_suffix(".pt.tmp")
        torch.save(model.backbone.state_dict(), tmp)
        os.replace(tmp, out_pt)

    proxy_files = data.list_cache("artifacts/cache/v8", "val")[:8]
    prior_seconds = 0.0
    if resume:
        if not out_json.exists() or not out_pt.exists():
            raise SystemExit("resume: нет чекпойнта или json")
        prior = json.loads(out_json.read_text())
        hist = list(prior["history"])
        best_proxy = float(prior["best_proxy_iou"])
        best_epoch = int(prior["best_epoch"])
        best_loss = float(prior["best_val"])
        prior_seconds = float(prior.get("seconds") or 0.0)
        start_ep = max(int(h["epoch"]) for h in hist) + 1
        bad = sum(1 for h in hist if int(h["epoch"]) > best_epoch)
        log(f"resume с epoch {best_epoch} proxy {best_proxy:.4f} дальше с {start_ep} bad {bad}")
    else:
        best_loss = val_loss()
        best_proxy, best_thr = proxy_iou(model, proxy_files)
        best_epoch = -1
        save_ckpt()
        hist = [dict(epoch=-1, train=None, val=best_loss, proxy_iou=best_proxy, proxy_thr=best_thr, sec=0.0)]
        start_ep = 0
        bad = 0
        log(f"init val_loss {best_loss:.4f} proxy_iou {best_proxy:.4f} thr {best_thr:.4f}")

    def payload(stopped, seconds):
        return dict(
            name="MIXSTYLE",
            init=str(init),
            arch="unet",
            width=16,
            mixstyle=dict(after=["e1", "e2"], p=0.5, alpha=0.1, eval="identity"),
            loss="focal+tversky_ce",
            lr=lr,
            epochs=epochs,
            steps=steps,
            batch=batch,
            tile=[128, 256],
            max_minutes=max_minutes,
            seed=seed,
            weight_decay=0.0,
            threads=2,
            train_cache=["artifacts/cache/v8/train", "artifacts/cache/v10/train"],
            v10_x="lazy npz, y in ram",
            real_traces_used=False,
            real_noise_used=False,
            prior_seconds=prior_seconds,
            ckpt_rule="max proxy IoU@FAR1e-3 on the first 8 v8 val sequences",
            history=hist,
            best_val=float(best_loss),
            best_proxy_iou=float(best_proxy),
            best_epoch=int(best_epoch),
            stopped=stopped,
            seconds=round(float(seconds) + prior_seconds, 1),
            ckpt=str(out_pt),
        )

    t0 = time.time()
    stopped = "epochs"
    for ep in range(start_ep, start_ep + epochs):
        if (time.time() - t0) / 60 > max_minutes - 4:
            stopped = "time"
            break
        model.train()
        tl = []
        for _ in range(steps):
            if (time.time() - t0) / 60 > max_minutes:
                stopped = "time"
                break
            x, y = sampler.sample(batch)
            logit = model(torch.from_numpy(x))
            loss = losses.compute("focal+tversky_ce", logit, torch.from_numpy(y), cw)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss")
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.backbone.parameters(), 1.0)
            opt.step()
            tl.append(float(loss.item()))
        if not tl:
            stopped = "time"
            break
        vl = val_loss()
        pi, pt = proxy_iou(model, proxy_files)
        rec = dict(epoch=ep, train=float(np.mean(tl)), val=vl, proxy_iou=pi, proxy_thr=pt,
                   steps_done=len(tl), sec=round(time.time() - t0 + prior_seconds, 1))
        hist.append(rec)
        log(str(rec))
        if pi > best_proxy + 1e-4:
            best_proxy = pi
            best_loss = vl
            best_epoch = ep
            bad = 0
            save_ckpt()
            log(f"сохранён epoch {ep} proxy {pi:.4f}")
        else:
            bad += 1
            if bad >= patience and ep >= 2:
                stopped = "patience"
        dump(out_json, payload(stopped if stopped in ("time", "patience") else "running", time.time() - t0))
        if stopped in ("time", "patience"):
            break
    dump(out_json, payload(stopped, time.time() - t0))
    log(f"готово best_proxy {best_proxy:.4f} epoch {best_epoch} stop {stopped} sec {time.time() - t0:.0f}")


if __name__ == "__main__":
    main()
