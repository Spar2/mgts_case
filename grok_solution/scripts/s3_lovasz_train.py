"""Дообучение U-Net потерей Lovász. Только синтетика v8 и v10 train.

Старт — artifacts/stage1/unet_v810ft.pt. mmap v10 не создаётся.
Чекпойнт — максимум прокси IoU@FAR 1e-3 на четырёх последовательностях v8 val.
Официальный отбор — в scripts/s3_lovasz_eval.py. Реальные CSV не читаются.

  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \\
  /workspace/repo/run/.venv/bin/python -u scripts/s3_lovasz_train.py
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_k] = "2"

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from dts import data, metrics as M, models
from dts.lovasz import lovasz_iou_loss, self_check

TILE = (128, 256)


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S', time.gmtime())} {msg}", flush=True)


def _predict():
    spec = importlib.util.spec_from_file_location("s3_dom_eval", ROOT / "scripts" / "s3_dom_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.predict_batched


def assert_mmap(ver: str, split: str) -> None:
    cache = Path("artifacts/cache") / ver / split
    mm = Path("artifacts/mm") / ver / split
    missing = [
        p.name for p in sorted(cache.glob("seq*.npz"))
        if not (mm / f"{p.stem}_y.npy").exists() or not (mm / f"{p.stem}_X.npy").exists()
    ]
    if missing:
        raise SystemExit(f"нет mmap {ver}/{split}: {len(missing)} файлов, новый кэш не пишу")


class StreamTiles:
    """Один кадр в памяти. v8 уже mmap, v10 читается из npz и на диск не пишется."""

    def __init__(self, kind: str, items, seed: int):
        self.kind = kind
        self.items = items
        self.rng = np.random.default_rng(seed)
        self.cache = None

    def _load(self, it):
        if self.kind == "v8":
            X, y = it["X"], it["y"]
            key = it["name"]
        else:
            with np.load(it) as z:
                X = np.array(z["X"], np.float16, copy=True)
                y = data.target(np.asarray(z["ev"]), np.asarray(z["valid"]))
            key = Path(it).name
        ev = np.argwhere(np.asarray(y) > 0)
        if len(ev) > 4000:
            ev = ev[self.rng.choice(len(ev), 4000, replace=False)]
        return key, X, y, ev

    def tiles(self, n: int):
        it = self.items[int(self.rng.integers(0, len(self.items)))]
        key = it["name"] if self.kind == "v8" else Path(it).name
        if self.cache is None or self.cache[0] != key:
            self.cache = self._load(it)
        _, X, y, ev = self.cache
        th, tw = TILE
        T, W = int(y.shape[0]), int(y.shape[1])
        xs, ys = [], []
        for _ in range(n):
            if len(ev) and self.rng.random() < 0.6:
                c = ev[int(self.rng.integers(0, len(ev)))]
                t0 = int(c[0]) - int(self.rng.integers(0, th))
                w0 = int(c[1]) - int(self.rng.integers(0, tw))
            else:
                t0 = int(self.rng.integers(0, max(T - th, 0) + 1))
                w0 = int(self.rng.integers(0, max(W - tw, 0) + 1))
            t0 = int(np.clip(t0, 0, max(T - th, 0)))
            w0 = int(np.clip(w0, 0, max(W - tw, 0)))
            xt = np.zeros((3, th, tw), np.float32)
            yt = np.full((th, tw), -1, np.int64)
            a = np.asarray(X[:, t0:t0 + th, w0:w0 + tw], np.float32)
            b = np.asarray(y[t0:t0 + th, w0:w0 + tw])
            xt[:, :a.shape[1], :a.shape[2]] = a
            yt[:b.shape[0], :b.shape[1]] = b
            if self.rng.random() < 0.5:
                xt = xt[:, :, ::-1].copy()
                yt = yt[:, ::-1].copy()
            g = float(self.rng.uniform(0.75, 1.3))
            xt[:2] = np.clip(xt[:2] * g, -12, 12)
            xs.append(np.ascontiguousarray(xt))
            ys.append(np.ascontiguousarray(yt))
        return xs, ys


def proxy_iou(model, files, predict_batched) -> tuple[float, float]:
    """IoU@FAR 1e-3 на нескольких val-последовательностях. Порог только по их фону."""
    model.eval()
    parts, maps = [], []
    for p in files:
        with np.load(p) as z:
            prob = predict_batched(model, np.asarray(z["X"], np.float32), batch=4)
            s = 1.0 - prob[0]
            valid = np.asarray(z["valid"], bool)
            ev, nu = z["ev"], z["nu"]
        parts.append(np.asarray(s[valid & (ev == 0) & (nu == 0)], np.float32))
        maps.append((s, valid, ev > 0))
    bg = np.concatenate(parts)
    thr = float(M.far_threshold(bg, 1e-3))
    tot = np.zeros(3, np.int64)
    for s, valid, truth in maps:
        tot += np.asarray(M.confusion((s >= thr) & valid, truth, valid), np.int64)
    return float(M.prf(*[int(v) for v in tot])["iou"]), thr


def main() -> None:
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    self_check()
    log("self_check Lovász ok")
    epochs = int(os.environ.get("LOVASZ_EPOCHS", "8"))
    steps = int(os.environ.get("LOVASZ_STEPS", "64"))
    batch = int(os.environ.get("LOVASZ_BATCH", "8"))
    lr = float(os.environ.get("LOVASZ_LR", "1e-5"))
    max_minutes = float(os.environ.get("LOVASZ_MAX_MINUTES", "30"))
    patience = int(os.environ.get("LOVASZ_PATIENCE", "3"))
    seed = int(os.environ.get("LOVASZ_SEED", "0"))
    hinge_w = float(os.environ.get("LOVASZ_HINGE", "0.5"))
    soft_w = float(os.environ.get("LOVASZ_SOFTMAX", "0.5"))
    init = Path("artifacts/stage1/unet_v810ft.pt")
    out_json = Path("artifacts/eval/s3_lovasz_train.json")
    out_pt = Path("artifacts/eval/s3_lovasz_unet.pt")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    mm10 = Path("artifacts/mm/v10")
    mm10_before = mm10.exists()

    torch.manual_seed(seed)
    np.random.seed(seed)
    assert_mmap("v8", "train")
    v8 = data.mmap_items("artifacts/cache/v8", "train", Path("artifacts/mm/v8"))
    v10_paths = data.list_cache("artifacts/cache/v10", "train")
    if len(v8) < 10 or len(v10_paths) < 10:
        raise SystemExit(f"мало train: v8 {len(v8)} v10 {len(v10_paths)}")
    src8 = StreamTiles("v8", v8, seed)
    src10 = StreamTiles("v10", v10_paths, seed + 1)
    log(f"train v8 {len(v8)} v10 {len(v10_paths)} без mmap v10")

    model = models.build("unet", 16)
    model.load_state_dict(torch.load(init, map_location="cpu", weights_only=True))
    for mod in model.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.eval()
            for p in mod.parameters():
                p.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)

    def bn_eval():
        model.train()
        for mod in model.modules():
            if isinstance(mod, torch.nn.BatchNorm2d):
                mod.eval()

    def save_ckpt():
        tmp = out_pt.with_suffix(".pt.tmp")
        torch.save(model.state_dict(), tmp)
        os.replace(tmp, out_pt)

    predict_batched = _predict()
    files = data.list_cache("artifacts/cache/v8", "val")
    proxy_files = [files[i] for i in (0, 11, 22, 33) if i < len(files)]
    best_proxy, best_thr = proxy_iou(model, proxy_files, predict_batched)
    best_epoch = -1
    save_ckpt()
    hist = [dict(epoch=-1, train=None, hinge=None, softmax=None, proxy_iou=best_proxy, proxy_thr=best_thr, sec=0.0)]
    log(f"init proxy_iou {best_proxy:.4f} thr {best_thr:.4f}")

    def payload(stopped, seconds):
        return dict(
            name="LOVASZ",
            paper="Berman, Triki, Blaschko, CVPR 2018",
            init=str(init),
            arch="unet",
            width=16,
            loss="0.5*lovasz_hinge(fg log-odds of s=1-p_bg) + 0.5*lovasz_softmax(classes=present)",
            hinge_weight=hinge_w,
            softmax_weight=soft_w,
            per_image=True,
            ignore="y<0",
            lr=lr,
            epochs=epochs,
            steps=steps,
            batch=batch,
            tile=[128, 256],
            max_minutes=max_minutes,
            seed=seed,
            bn="frozen_running_stats_and_affine",
            weight_decay=0.0,
            grad_clip=1.0,
            train_cache=["artifacts/cache/v8/train", "artifacts/cache/v10/train"],
            v10_mmap=False,
            aug="x-flip 0.5, gain 0.75-1.3; no real spectrum, no real csv",
            ckpt_rule="max proxy IoU@FAR1e-3 on 4 spread v8 val sequences; full 45-seq ensemble decides",
            proxy_files=[p.name for p in proxy_files],
            history=hist,
            best_proxy_iou=float(best_proxy),
            best_proxy_thr=float(best_thr),
            best_epoch=int(best_epoch),
            stopped=stopped,
            seconds=round(seconds, 1),
            ckpt=str(out_pt),
            real_traces_used=False,
            real_csv_used=False,
            lr_note="Прогон с lr 1e-4 поднял hinge и уронил прокси 0.2036 → 0.1926 → 0.1865 за две эпохи, чекпойнт остался инициализацией. Этот прогон — lr 1e-5.",
        )

    out_json.write_text(json.dumps(payload("init", 0.0), indent=1, ensure_ascii=False))
    bad = 0
    t0 = time.time()
    stopped = "epochs"
    n8 = batch // 2
    n10 = batch - n8
    for ep in range(epochs):
        if (time.time() - t0) / 60.0 > max_minutes:
            stopped = "time"
            break
        bn_eval()
        tl, ths, tss = [], [], []
        for step in range(steps):
            if step % 8 == 0 and (time.time() - t0) / 60.0 > max_minutes:
                stopped = "time"
                break
            xs, ys = src8.tiles(n8)
            x10, y10 = src10.tiles(n10)
            x = torch.from_numpy(np.stack(xs + x10))
            y = torch.from_numpy(np.stack(ys + y10))
            st = time.time()
            loss, hinge, soft = lovasz_iou_loss(model(x), y, hinge_w, soft_w)
            if not torch.isfinite(loss):
                log(f"non-finite loss epoch {ep} step {step}, шаг пропущен")
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl.append(float(loss.detach()))
            ths.append(float(hinge.detach()))
            tss.append(float(soft.detach()))
            if ep == 0 and step == 0:
                log(f"first step {time.time() - st:.2f}s loss {tl[-1]:.4f} hinge {ths[-1]:.4f} soft {tss[-1]:.4f}")
        if stopped == "time" and not tl:
            break
        pi, pt = proxy_iou(model, proxy_files, predict_batched)
        rec = dict(
            epoch=ep,
            train=float(np.mean(tl)) if tl else None,
            hinge=float(np.mean(ths)) if ths else None,
            softmax=float(np.mean(tss)) if tss else None,
            proxy_iou=pi,
            proxy_thr=pt,
            sec=round(time.time() - t0, 1),
        )
        hist.append(rec)
        log(str(rec))
        if pi > best_proxy + 1e-4:
            best_proxy = pi
            best_thr = pt
            best_epoch = ep
            bad = 0
            save_ckpt()
        else:
            bad += 1
            if bad >= patience and ep >= 1:
                stopped = "patience"
                break
        out_json.write_text(json.dumps(payload(stopped if stopped == "time" else "running", time.time() - t0), indent=1, ensure_ascii=False))
        if stopped == "time":
            break

    if mm10.exists() and not mm10_before:
        raise SystemExit("появился artifacts/mm/v10 — обучение его создавать не должно")
    out_json.write_text(json.dumps(payload(stopped, time.time() - t0), indent=1, ensure_ascii=False))
    log(f"готово best_proxy {best_proxy:.4f} epoch {best_epoch} stop {stopped} sec {time.time() - t0:.0f}")


if __name__ == "__main__":
    main()
