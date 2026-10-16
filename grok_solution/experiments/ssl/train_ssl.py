"""Обучение слепого автоэнкодера на train/*_X.npy. Меток нет. Бюджет — max-minutes."""
import os
import sys


def _arg(name, default):
    if name in sys.argv:
        return sys.argv[sys.argv.index(name) + 1]
    return default


_THREADS = _arg("--threads", "2")
os.environ["OMP_NUM_THREADS"] = _THREADS
os.environ["MKL_NUM_THREADS"] = _THREADS
os.environ["OPENBLAS_NUM_THREADS"] = _THREADS
os.environ["NUMEXPR_NUM_THREADS"] = _THREADS

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.ssl.ssl import BlindSpotAE, FieldTiles, masked_loss, sample_block_mask, save_model


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mm", default="artifacts/mm/v8", help="корень mmap-кэша с train/*_X.npy")
    ap.add_argument("--out", default="artifacts/stage1/ssl_ae.pt")
    ap.add_argument("--scratch", default="artifacts/ae_scratch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--max-minutes", type=float, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--tile-t", type=int, default=128)
    ap.add_argument("--tile-x", type=int, default=192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--mask-p", type=float, default=0.5)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--steps", type=int, default=10**9, help="верхняя граница шагов; обычно режет max-minutes")
    a = ap.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as exc:  # pragma: no cover
        print("deterministic algorithms not fully available:", exc, flush=True)

    scratch = Path(a.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    paths = sorted(Path(a.mm).glob("train/*_X.npy"))
    if not paths:
        raise SystemExit(f"no train/*_X.npy under {a.mm}")
    # часть train-файлов — только монитор реконструкции, не val и не метки
    order = np.random.default_rng(a.seed + 1).permutation(len(paths))
    n_mon = max(1, len(paths) // 10)
    mon_paths = [paths[i] for i in order[:n_mon]]
    fit_paths = [paths[i] for i in order[n_mon:]]
    fit = FieldTiles(a.mm, seed=a.seed, paths=fit_paths)
    mon = FieldTiles(a.mm, seed=a.seed + 2, paths=mon_paths)
    print(f"train_files {len(fit_paths)} monitor_files {len(mon_paths)} threads {a.threads}", flush=True)

    torch.manual_seed(a.seed)
    model = BlindSpotAE(width=a.width)
    up, down, left, right = model.lookback()
    if a.tile_t <= up + down or a.tile_x <= left + right:
        raise SystemExit(f"tile {a.tile_t}x{a.tile_x} меньше lookback {(up, down, left, right)}")
    print(f"lookback up,down,left,right {up, down, left, right} tile {a.tile_t}x{a.tile_x}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    mask_rng = np.random.default_rng(a.seed + 3)
    interior = torch.zeros(1, 1, a.tile_t, a.tile_x)
    interior[:, :, up:a.tile_t - down, left:a.tile_x - right] = 1

    mon_x = torch.from_numpy(mon.sample(a.batch, a.tile_t, a.tile_x))
    mon_mask = torch.from_numpy(sample_block_mask(a.batch, a.tile_t, a.tile_x, np.random.default_rng(a.seed + 4), a.mask_p))
    mon_mask = mon_mask * interior
    mon_inp = mon_x.clone()
    mon_inp[:, :2] *= 1.0 - mon_mask

    def monitor_loss():
        model.eval()
        with torch.no_grad():
            return float(masked_loss(model(mon_inp), mon_x[:, :2], mon_mask, mon_x[:, 2:3]))

    hist, best, best_state = [], float("inf"), None
    t0 = time.time()
    budget = a.max_minutes * 60.0
    step = 0
    while step < a.steps and (time.time() - t0) < budget:
        model.train()
        xb = torch.from_numpy(fit.sample(a.batch, a.tile_t, a.tile_x))
        mask = torch.from_numpy(sample_block_mask(a.batch, a.tile_t, a.tile_x, mask_rng, a.mask_p)) * interior
        inp = xb.clone()
        inp[:, :2] *= 1.0 - mask
        opt.zero_grad(set_to_none=True)
        loss = masked_loss(model(inp), xb[:, :2], mask, xb[:, 2:3])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        frac = min(1.0, (time.time() - t0) / budget)
        lr = a.lr * (1.0 - 0.9 * frac)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.step()
        step += 1
        if step % 20 == 0 or step == 1:
            rec = dict(step=step, train=round(float(loss.detach()), 5), lr=round(lr, 6), sec=round(time.time() - t0, 1))
            if step % 100 == 0 or step == 1:
                rec["monitor"] = round(monitor_loss(), 5)
                if rec["monitor"] < best:
                    best = rec["monitor"]
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    save_model(model, a.out, seed=a.seed, steps=step, seconds=rec["sec"], best_monitor=best)
            print(rec, flush=True)
            hist.append(rec)
    last_mon = monitor_loss()
    if last_mon < best or best_state is None:
        best = last_mon
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    seconds = round(time.time() - t0, 1)
    save_model(model, a.out, seed=a.seed, steps=step, seconds=seconds, best_monitor=float(best), args=vars(a))
    log = dict(args=vars(a), steps=step, seconds=seconds, best_monitor=float(best),
               n_fit=len(fit_paths), n_monitor=len(mon_paths), history=hist,
               lookback=dict(up=up, down=down, left=left, right=right))
    with open(scratch / "train_ssl.json", "w") as f:
        json.dump(log, f, indent=1)
    print(f"saved {a.out} steps {step} sec {seconds} best_monitor {best:.5f}", flush=True)


if __name__ == "__main__":
    main()
