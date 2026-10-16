"""Дообучение unet_w24 аугментацией dts/gen2.py. Тот же рецепт, что scripts/s3_gen2_train.py.

Старт — artifacts/stage1/unet_w24.pt, ширина 24. Только синтетика v8 и v10 train.
Потолок 50 минут вместе с индексацией и прокси, 2 потока.
Снимок artifacts/ckpt/s3_gen2w24/last.pt не реже чем раз в 15 минут. В конце — DONE.

Прокси чекпойнта — IoU mean(gen2, gen2w24) на всём кэше v10 val при FAR 1e-3.
Это смесь ens_gens. v8 val в прокси не входит: у GEN2 прокси сидел на четырёх
файлах v8 val. Карт победителя на v10 val нет, поэтому в прокси его нет.
Официальный выбор смеси — в scripts/s3_gen2w24_eval.py по полному v8 val.

  scripts/run_resumable.sh s3_gen2w24 /workspace/repo/run/.venv/bin/python -u scripts/s3_gen2w24_train.py
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
    Mixer,
    Source,
    atomic_torch,
    bn_train_frozen,
    dump_json,
    freeze_bn,
    predict_batched,
    restore_rng,
    rng_blob,
)

RUN = "s3_gen2w24"
CKPT_DIR = Path("artifacts/ckpt") / RUN
LAST = CKPT_DIR / "last.pt"
BEST = Path("artifacts/eval/s3_gen2w24_unet.pt")
TRAIN_JSON = Path("artifacts/eval/s3_gen2w24_train.json")
INIT = Path("artifacts/stage1/unet_w24.pt")
GEN2_W = Path("artifacts/eval/s3_gen2_unet.pt")
SEED = 20261016
BUDGET_S = 50 * 60
SAVE_EVERY = 15 * 60
PROXY_GUARD_S = 8 * 60
BATCH = 6
LR = 8e-5
LR_ENC = 1e-5
WEAK_W = 3.0
WIDTH = 24


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S", time.gmtime()) + " " + msg, flush=True)


def load_unet(path: Path, width: int):
    model = models.build("unet", width)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state and not any(k.startswith("e1") for k in state):
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    freeze_bn(model)
    return model


@torch.inference_mode()
def proxy_v10(model, files, gen2_net, cache: dict) -> dict:
    """IoU на полном v10 val. Порог только по фону этих файлов. Карты на диске, не в RAM."""
    model.eval()
    tmp = Path(f"/tmp/s3_gen2w24_proxy_{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    bg_b, bg_s = [], []
    try:
        for i, p in enumerate(files, 1):
            with np.load(p) as z:
                X = np.asarray(z["X"], np.float32)
                ev = np.asarray(z["ev"])
                valid = np.asarray(z["valid"], bool)
                nu = np.asarray(z["nu"])
            prob = predict_batched(model, X)
            s = (1.0 - prob[0]).astype(np.float32)
            del prob
            if p.stem not in cache:
                gp = predict_batched(gen2_net, X)
                cache[p.stem] = (1.0 - gp[0]).astype(np.float16)
                del gp
            g = np.asarray(cache[p.stem], np.float32)
            blend = (0.5 * (s + g)).astype(np.float32)
            bg = valid & (ev == 0) & (nu == 0)
            bg_b.append(np.asarray(blend[bg], np.float16))
            bg_s.append(np.asarray(s[bg], np.float16))
            np.save(tmp / f"{p.stem}_b.npy", blend.astype(np.float16))
            np.save(tmp / f"{p.stem}_s.npy", s.astype(np.float16))
            np.save(tmp / f"{p.stem}_y.npy", np.stack([
                (ev > 0).astype(np.uint8), valid.astype(np.uint8),
            ]))
            del X, s, g, blend, ev, valid, nu
            if i == 1 or i % 15 == 0 or i == len(files):
                log(f"proxy pred {i}/{len(files)}")
        thr_b = float(M.far_threshold(np.concatenate(bg_b).astype(np.float32), 1e-3))
        del bg_b
        thr_s = float(M.far_threshold(np.concatenate(bg_s).astype(np.float32), 1e-3))
        del bg_s
        tot_b = np.zeros(3, np.int64)
        tot_s = np.zeros(3, np.int64)
        for p in files:
            y = np.load(tmp / f"{p.stem}_y.npy")
            truth, valid = y[0].astype(bool), y[1].astype(bool)
            blend = np.load(tmp / f"{p.stem}_b.npy").astype(np.float32)
            solo = np.load(tmp / f"{p.stem}_s.npy").astype(np.float32)
            tot_b += np.asarray(M.confusion((blend >= thr_b) & valid, truth, valid), np.int64)
            tot_s += np.asarray(M.confusion((solo >= thr_s) & valid, truth, valid), np.int64)
            del y, blend, solo
        pair = float(M.prf(*map(int, tot_b))["iou"])
        solo = float(M.prf(*map(int, tot_s))["iou"])
        log(f"proxy v10 val ens_gens IoU {pair:.4f} thr {thr_b:.4f} solo {solo:.4f} thr {thr_s:.4f}")
        return dict(pair_iou=pair, pair_thr=thr_b, solo_iou=solo, solo_thr=thr_s)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        gc.collect()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", default="")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    wall0 = time.time()
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    gen2.self_check()
    log("gen2 self-check ok")
    if not INIT.is_file() or not GEN2_W.is_file():
        raise SystemExit(f"нет веса {INIT} или {GEN2_W}")
    rng = np.random.default_rng(args.seed)
    index_rng = np.random.default_rng(0)
    v8_paths = data.list_cache("artifacts/cache/v8", "train")
    v10_paths = data.list_cache("artifacts/cache/v10", "train")
    proxy_files = data.list_cache("artifacts/cache/v10", "val")
    if len(v8_paths) < 4 or len(v10_paths) < 4 or len(proxy_files) < 4:
        raise SystemExit(f"мало кэша: v8 {len(v8_paths)} v10 train {len(v10_paths)} v10 val {len(proxy_files)}")
    if any("cache/v8" in str(p) for p in proxy_files):
        raise SystemExit("прокси указывает на v8")
    log(f"индексация, прокси файлов v10 val {len(proxy_files)}")
    v8 = Source(v8_paths, Path("artifacts/mm/v8/train"), index_rng)
    v10 = Source(v10_paths, None, index_rng)
    mixer = Mixer(v8, v10, rng)
    model = load_unet(INIT, WIDTH)
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

    gen2_net = load_unet(GEN2_W, 16)
    gen2_cache: dict = {}
    deadline = wall0 + max(0.0, BUDGET_S - elapsed_prev)

    def spent_now() -> float:
        return elapsed_prev + (time.time() - wall0)

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
            width=WIDTH,
        )
        atomic_torch(LAST, payload)
        dump_json(TRAIN_JSON, dict(
            name="GEN2W24",
            init=str(INIT),
            width=WIDTH,
            generator="dts/gen2.py",
            gen2_weights=str(GEN2_W),
            step=step,
            best_step=best_step,
            best_proxy_iou=None if best < 0 else best,
            seconds=seconds,
            history=history,
            proxy="mean(gen2, gen2w24) IoU at FAR 1e-3 on all artifacts/cache/v10/val",
            proxy_n_files=len(proxy_files),
            proxy_not_v8_val=True,
            phenomena=["static_hot", "broad_shift", "hvac_20_90min", "artifact_-20C"],
            real_traces_used=False,
            real_noise_used=False,
            budget_s=BUDGET_S,
        ))

    def consider(row: dict) -> None:
        nonlocal best, best_step, best_state
        history.append(row)
        bi = float(row["proxy_blend_iou"])
        if bi > best + 1e-4:
            best = bi
            best_step = int(row["step"])
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            atomic_torch(BEST, best_state)
            log(f"новый лучший прокси {bi:.4f} шаг {best_step}")

    if best < 0 and step == 0 and time.time() + PROXY_GUARD_S < deadline:
        row = proxy_v10(model, proxy_files, gen2_net, gen2_cache)
        consider(dict(step=0, proxy_blend_iou=row["pair_iou"], solo_iou=row["solo_iou"], sec=round(spent_now(), 1)))
        log(f"proxy step 0 pair {row['pair_iou']:.4f} solo {row['solo_iou']:.4f}")
        bn_train_frozen(model)

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
        del xb, yb, wb, loss
        if step % 25 == 0:
            log(f"step {step} loss {float(np.mean(running[-25:])):.4f} {which} {spent_now():.0f}s")
        now = time.time()
        if now - last_save >= SAVE_EVERY and now + 30 < deadline:
            if now + PROXY_GUARD_S < deadline:
                row = proxy_v10(model, proxy_files, gen2_net, gen2_cache)
                consider(dict(
                    step=step,
                    proxy_blend_iou=row["pair_iou"],
                    solo_iou=row["solo_iou"],
                    loss=float(np.mean(running[-25:])) if running else None,
                    sec=round(spent_now(), 1),
                ))
            snapshot(spent_now())
            last_save = time.time()
            bn_train_frozen(model)
            gc.collect()
    else:
        stopped = "budget"

    if stopped != "nan" and time.time() + 30 < deadline:
        row = proxy_v10(model, proxy_files, gen2_net, gen2_cache)
        consider(dict(step=step, proxy_blend_iou=row["pair_iou"], solo_iou=row["solo_iou"],
                      sec=round(spent_now(), 1), final=True))
    spent = spent_now()
    snapshot(spent)
    atomic_torch(BEST, best_state)
    if stopped == "nan":
        log(f"TRAIN_FAIL step {step} best {best:.4f} at {best_step} {spent:.0f}s")
        raise SystemExit(1)
    (CKPT_DIR / "DONE").touch()
    log(f"TRAIN_DONE step {step} best {best:.4f} at {best_step} stop {stopped} {spent:.0f}s")


if __name__ == "__main__":
    main()
