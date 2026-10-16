"""Второй сид GEN2. Тот же рецепт, другой seed, прокси — весь v10 val.

Старт artifacts/stage1/unet_v810ft.pt, не веса первого сида.
Чекпойнт — IoU mean(эта сеть, победитель) при FAR 1e-3 на всех
artifacts/cache/v10/val. v8 val не читается. Победитель на v10 val —
среднее четырёх членов; на v8 val такое среднее совпало с сохранённым
ансамблем (max abs 0).

Свои пути: artifacts/ckpt/s3_gen2_seed2/ и artifacts/eval/s3_gen2_unet_s2.pt.
Первый сид не перезаписывается.

  scripts/run_resumable.sh s3_gen2_seed2 /workspace/repo/run/.venv/bin/python -u scripts/s3_gen2_seed2_train.py
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_k] = "2"

import argparse
import gc
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)

from dts import data, gen2, losses, metrics as M, models
from s3_gen2_train import (
    BATCH, INIT, LR, LR_ENC, WEAK_W,
    Mixer, Source, atomic_torch, bn_train_frozen, dump_json, freeze_bn,
    log, predict_batched, restore_rng, rng_blob,
)

RUN = "s3_gen2_seed2"
CKPT_DIR = Path("artifacts/ckpt") / RUN
LAST = CKPT_DIR / "last.pt"
BEST = Path("artifacts/eval/s3_gen2_unet_s2.pt")
TRAIN_JSON = Path("artifacts/eval/s3_gen2_seed2_train.json")
SEED = 20261018
BUDGET_S = 54 * 60
SAVE_EVERY = 15 * 60
MEMBERS = ("unet_w24", "unet_v810ft", "tcn", "pixel_lgbm")
SEED1_BEST = Path("artifacts/eval/s3_gen2_unet.pt")


def winner_v10_val(stem: str, shape) -> np.ndarray:
    ss = []
    for name in MEMBERS:
        path = Path("artifacts/pred/v10") / name / "val" / f"{stem}.npz"
        with np.load(path) as z:
            s = np.asarray(z["s"], np.float32)
        if s.shape != shape:
            raise SystemExit(f"{name} {stem} {s.shape} != {shape}")
        ss.append(s)
    return np.mean(ss, axis=0).astype(np.float32)


def proxy_v10(model, files) -> dict:
    """FAR 1e-3 и IoU mean(сеть, победитель) на полном v10 val. Карты во /tmp."""
    model.eval()
    tmp = Path(f"/tmp/s3_gen2_seed2_proxy_{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    bg_parts = []
    try:
        for i, p in enumerate(files, 1):
            with np.load(p) as z:
                X = np.asarray(z["X"], np.float32)
                ev = np.asarray(z["ev"])
                valid = np.asarray(z["valid"], bool)
                nu = np.asarray(z["nu"])
            prob = predict_batched(model, X)
            s = (1.0 - prob[0]).astype(np.float32)
            del prob, X
            ws = winner_v10_val(p.stem, ev.shape)
            blend = (0.5 * (s + ws)).astype(np.float32)
            del s, ws
            bg = valid & (ev == 0) & (nu == 0)
            bg_parts.append(np.asarray(blend[bg], np.float32))
            np.save(tmp / f"{p.stem}_b.npy", blend)
            np.save(tmp / f"{p.stem}_y.npy", np.stack([
                (ev > 0).astype(np.uint8), valid.astype(np.uint8),
            ]))
            del blend, ev, valid, nu, bg
            if i == 1 or i % 15 == 0 or i == len(files):
                log(f"proxy v10 val {i}/{len(files)}")
        bg = np.concatenate(bg_parts)
        del bg_parts
        thr = float(M.far_threshold(bg, 1e-3))
        n_bg = int(bg.size)
        del bg
        tot = np.zeros(3, np.int64)
        for p in files:
            y = np.load(tmp / f"{p.stem}_y.npy")
            truth, valid = y[0].astype(bool), y[1].astype(bool)
            blend = np.load(tmp / f"{p.stem}_b.npy")
            tot += np.asarray(M.confusion((blend >= thr) & valid, truth, valid), np.int64)
            del y, blend, truth, valid
        iou = float(M.prf(*map(int, tot))["iou"])
        log(f"proxy mean(seed2,winner) IoU {iou:.4f} thr {thr:.4f} n_bg {n_bg}")
        return dict(iou=iou, threshold=thr, n_bg=n_bg)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        gc.collect()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", default="")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    if BEST.resolve() == SEED1_BEST.resolve():
        raise SystemExit("второй сид не пишет чекпойнт первого")
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
    proxy_files = data.list_cache("artifacts/cache/v10", "val")
    if len(v8_paths) < 4 or len(v10_paths) < 4 or len(proxy_files) != 45:
        raise SystemExit(
            f"кэш v8 {len(v8_paths)} v10 train {len(v10_paths)} v10 val {len(proxy_files)}"
        )
    if any(p.parent.name != "val" or "v10" not in p.parts for p in proxy_files):
        raise SystemExit("прокси не весь v10 val")
    log("прокси v10 val " + " ".join(p.stem for p in proxy_files[:4]) + f" ... n={len(proxy_files)}")
    v8 = Source(v8_paths, Path("artifacts/mm/v8/train"), index_rng)
    v10 = Source(v10_paths, None, index_rng)
    mixer = Mixer(v8, v10, rng)
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
        if int(blob.get("seed", args.seed)) != int(args.seed):
            raise SystemExit(f"resume seed {blob.get('seed')} != {args.seed}")
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

    def meta(seconds: float) -> dict:
        return dict(
            name="GEN2_SEED2",
            init=str(INIT),
            seed=int(args.seed),
            seed1=20261015,
            step=step,
            best_step=best_step,
            best_proxy_iou=None if best < 0 else best,
            seconds=seconds,
            history=history,
            proxy="mean(seed2, winner) IoU at FAR 1e-3 on all artifacts/cache/v10/val",
            proxy_n_files=len(proxy_files),
            proxy_files=[p.stem for p in proxy_files],
            proxy_not_v8_val=True,
            winner_on_v10_val="mean of unet_w24, unet_v810ft, tcn, pixel_lgbm",
            real_traces_used=False,
            real_noise_used=False,
            generator="dts/gen2.py",
            phenomena=["static_hot", "broad_shift", "hvac_20_90min", "artifact_-20C"],
            budget_s=BUDGET_S,
        )

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
            seed=int(args.seed),
            init=str(INIT),
        )
        atomic_torch(LAST, payload)
        dump_json(TRAIN_JSON, meta(seconds))

    if best < 0 and step == 0:
        row = proxy_v10(model, proxy_files)
        best = float(row["iou"])
        best_step = 0
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(dict(step=0, proxy_blend_iou=best, threshold=row["threshold"], sec=0.0))
        atomic_torch(BEST, best_state)
        snapshot(elapsed_prev)
        log(f"proxy step 0 {best:.4f} (инициализация)")

    # 54 минуты — цикл шагов. Стартовый прокси в этот бюджет не входит.
    t0 = time.time()
    deadline = t0 + max(0.0, BUDGET_S - elapsed_prev)

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
            spent = elapsed_prev + (now - t0)
            snapshot(spent)
            last_save = time.time()
            row = proxy_v10(model, proxy_files)
            bi = float(row["iou"])
            spent = elapsed_prev + (time.time() - t0)
            history.append(dict(
                step=step, proxy_blend_iou=bi, threshold=row["threshold"],
                loss=float(np.mean(running[-25:])), sec=round(spent, 1),
            ))
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

    if stopped != "nan":
        row = proxy_v10(model, proxy_files)
        spent = elapsed_prev + (time.time() - t0)
        bi = float(row["iou"])
        history.append(dict(step=step, proxy_blend_iou=bi, threshold=row["threshold"], sec=round(spent, 1), final=True))
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
