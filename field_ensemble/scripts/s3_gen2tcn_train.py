"""Дообучение TCN участника аугментацией dts/gen2.py. Тот же рецепт, что scripts/s3_gen2_train.py.

Старт — artifacts/stage1/tcn.pt, ширина 32. Только синтетика v8 и v10 train.
Потолок 50 минут вместе с индексацией и прокси, 2 потока.
Снимок artifacts/ckpt/s3_gen2tcn/last.pt не реже чем раз в 15 минут. В конце — DONE.

Прокси чекпойнта — IoU mean(gen2, gen2tcn) на всём кэше v10 val при FAR 1e-3.
v8 val в прокси не входит. Карт победителя на v10 val нет, поэтому в прокси его нет.
Официальный выбор смеси — в scripts/s3_gen2tcn_eval.py по полному v8 val.

  scripts/run_resumable.sh s3_gen2tcn /workspace/repo/run/.venv/bin/python -u scripts/s3_gen2tcn_train.py
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

RUN = "s3_gen2tcn"
CKPT_DIR = Path("artifacts/ckpt") / RUN
LAST = CKPT_DIR / "last.pt"
BEST = Path("artifacts/eval/s3_gen2tcn_tcn.pt")
TRAIN_JSON = Path("artifacts/eval/s3_gen2tcn_train.json")
INIT = Path("artifacts/stage1/tcn.pt")
GEN2_W = Path("artifacts/eval/s3_gen2_unet.pt")
LOG = Path("/workspace/run_logs/_events.log")
SEED = 20261017
BUDGET_S = 50 * 60
SAVE_EVERY = 15 * 60
PROXY_GUARD_S = 8 * 60
BATCH = 6
LR = 8e-5
LR_BODY = 1e-5
WEAK_W = 3.0
WIDTH = 32


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S", time.gmtime()) + " " + msg, flush=True)


def elog(msg: str) -> None:
    log(msg)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(time.strftime("%H:%M", time.gmtime()) + " " + msg + "\n")


def load_tcn(path: Path):
    model = models.build("tcn", WIDTH)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state and "body.0.weight" not in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    freeze_bn(model)
    return model


def load_gen2():
    model = models.build("unet", 16)
    state = torch.load(GEN2_W, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state and not any(str(k).startswith("e1") for k in state):
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    freeze_bn(model)
    return model


@torch.inference_mode()
def proxy_v10(model, files, gen2_net, cache: dict) -> dict:
    """IoU на полном v10 val. Порог только по фону этих файлов. Карты на диске, не в RAM."""
    model.eval()
    tmp = Path(f"/tmp/s3_gen2tcn_proxy_{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    bg_b, bg_s = [], []
    try:
        for i, p in enumerate(files, 1):
            if "v10" not in p.parts or p.parent.name != "val":
                raise RuntimeError(f"прокси не v10 val: {p}")
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
            if i == 1 or i % 5 == 0 or i == len(files):
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
    ap.add_argument("--resume", default="auto", choices=("auto", "never", ""))
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    wall0 = time.time()
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if (CKPT_DIR / "DONE").is_file() and args.resume != "never":
        log("DONE already")
        return
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
    if any(p.parent.name != "val" or "v10" not in p.parts for p in proxy_files):
        raise SystemExit("прокси указывает не на v10 val")
    log(f"индексация, прокси файлов v10 val {len(proxy_files)}")
    v8 = Source(v8_paths, Path("artifacts/mm/v8/train"), index_rng)
    v10 = Source(v10_paths, None, index_rng)
    mixer = Mixer(v8, v10, rng)
    model = load_tcn(INIT)
    freeze_bn(model)
    body, head = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head if name.startswith("head") else body).append(p)
    if not body or not head:
        raise SystemExit(f"пустые группы TCN body {len(body)} head {len(head)}")
    n_params = int(sum(p.numel() for p in model.parameters()))
    log(f"tcn width {WIDTH} params {n_params} body {len(body)} head {len(head)}")
    opt = torch.optim.AdamW(
        [{"params": body, "lr": LR_BODY}, {"params": head, "lr": LR}],
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
    if args.resume == "never" and LAST.is_file():
        LAST.unlink()
    elif args.resume == "auto" and LAST.is_file():
        resume_path = LAST
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

    gen2_net = load_gen2()
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
            name="GEN2TCN",
            init=str(INIT),
            width=WIDTH,
            params=n_params,
            arch="tcn",
            generator="dts/gen2.py",
            recipe="scripts/s3_gen2_train.py",
            gen2_weights=str(GEN2_W),
            step=step,
            best_step=best_step,
            best_proxy_iou=None if best < 0 else best,
            seconds=seconds,
            history=history,
            proxy="mean(gen2, gen2tcn) IoU at FAR 1e-3 on all artifacts/cache/v10/val",
            proxy_n_files=len(proxy_files),
            proxy_not_v8_val=True,
            val_used_for_train=False,
            phenomena=["static_hot", "broad_shift", "hvac_20_90min", "artifact_-20C"],
            lr_body=LR_BODY,
            lr_head=LR,
            batch=BATCH,
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
    elog(
        f"GEN2TCN_TRAIN_DONE step={step} best_step={best_step} proxy_iou={best:.6f} "
        f"stop={stopped} sec={spent:.0f} params={n_params} real_train=0"
    )


if __name__ == "__main__":
    main()
