"""Дообучение U-Net аугментацией RandConv. Только синтетика v8 train.

Старт artifacts/stage1/unet_v810ft.pt. Реальные CSV и real_noise_psd не читаются.
Порог официального ансамбля здесь не считается: чекпойнт выбирается по чистому
прокси-IoU первых 8 последовательностей v8 val.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_k] = "2"

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from dts import data, losses, metrics as M, models, randconv


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S', time.gmtime())} {msg}", flush=True)


def _predict():
    spec = importlib.util.spec_from_file_location("s3_dom_eval", ROOT / "scripts" / "s3_dom_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.predict_batched


def assert_mmap() -> None:
    cache = Path("artifacts/cache/v8/train")
    mm = Path("artifacts/mm/v8/train")
    missing = [p.name for p in sorted(cache.glob("seq*.npz")) if not (mm / f"{p.stem}_y.npy").exists()]
    if missing:
        raise SystemExit(f"нет mmap y для {len(missing)} train-файлов; новый кэш не пишем")


@torch.no_grad()
def proxy_iou(model, files, predict) -> tuple[float, float]:
    model.eval()
    parts, maps = [], []
    for p in files:
        with np.load(p) as z:
            prob = predict(model, np.asarray(z["X"], np.float32))
            valid = np.asarray(z["valid"], bool)
            ev = np.asarray(z["ev"])
            nu = np.asarray(z["nu"])
        s = 1.0 - prob[0]
        parts.append(s[valid & (ev == 0) & (nu == 0)])
        maps.append((s, valid, ev > 0))
    thr = float(M.far_threshold(np.concatenate(parts), 1e-3))
    tot = np.zeros(3, np.int64)
    for s, valid, truth in maps:
        tot += np.asarray(M.confusion((s >= thr) & valid, truth, valid), np.int64)
    return float(M.prf(*tot)["iou"]), thr


def main() -> None:
    randconv.self_check()
    torch.set_num_threads(2)
    assert_mmap()
    seed = int(os.environ.get("RC_SEED", "0"))
    epochs = int(os.environ.get("RC_EPOCHS", "8"))
    steps = int(os.environ.get("RC_STEPS", "30"))
    batch = int(os.environ.get("RC_BATCH", "8"))
    lr = float(os.environ.get("RC_LR", "1e-4"))
    lam = float(os.environ.get("RC_LAMBDA", "1.0"))
    views = int(os.environ.get("RC_VIEWS", "3"))
    max_minutes = float(os.environ.get("RC_MAX_MINUTES", "40"))
    patience = int(os.environ.get("RC_PATIENCE", "3"))
    tile = (128, 256)
    out_pt = Path(os.environ.get("RC_OUT_PT", "artifacts/eval/s3_randconv_unet.pt"))
    out_json = Path(os.environ.get("RC_OUT_JSON", "artifacts/eval/s3_randconv_train.json"))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    gen = torch.Generator().manual_seed(seed)

    init = Path("artifacts/stage1/unet_v810ft.pt")
    tr = data.mmap_items("artifacts/cache/v8", "train", Path("artifacts/mm/v8"))
    va = data.mmap_items("artifacts/cache/v8", "val", Path("artifacts/mm/v8"))
    log(f"train {len(tr)} val {len(va)} seed {seed} views {views} lambda {lam}")
    sampler = data.TileSampler(tr, tile=tile, seed=seed, p_weak=0.0, aug=None)
    sval = data.TileSampler(va, tile=tile, seed=1234, p_event=0.6)
    vx, vy = sval.sample(64)

    model = models.build("unet", 16)
    model.load_state_dict(torch.load(init, map_location="cpu", weights_only=True))
    for mod in model.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.eval()
            for p in mod.parameters():
                p.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    cw = torch.tensor([1.0, 3.0, 3.0])

    def bn_eval():
        model.train()
        for mod in model.modules():
            if isinstance(mod, torch.nn.BatchNorm2d):
                mod.eval()

    @torch.no_grad()
    def val_loss() -> float:
        model.eval()
        losses_ep = []
        for i in range(0, len(vx), 8):
            logit = model(torch.from_numpy(vx[i:i + 8]))
            losses_ep.append(losses.compute("focal+tversky_ce", logit, torch.from_numpy(vy[i:i + 8]), cw).item())
        return float(np.mean(losses_ep))

    def save_ckpt():
        tmp = out_pt.with_suffix(".pt.tmp")
        torch.save(model.state_dict(), tmp)
        os.replace(tmp, out_pt)

    predict = _predict()
    proxy_files = data.list_cache("artifacts/cache/v8", "val")[:8]
    t_wall = time.time()
    best_loss = val_loss()
    best_proxy, best_thr = proxy_iou(model, proxy_files, predict)
    best_epoch = -1
    save_ckpt()
    hist = [dict(epoch=-1, train=None, task=None, cons=None, val=best_loss, proxy_iou=best_proxy, proxy_thr=best_thr, sec=0)]
    log(f"init val_loss {best_loss:.4f} proxy_iou {best_proxy:.4f}")

    def payload(stopped: str) -> dict:
        return dict(
            name="RANDCONV",
            paper="Xu et al., ICLR 2021, RandConv",
            init=str(init),
            arch="unet",
            width=16,
            loss="focal+tversky_ce on view 0 + lambda * KL(views || mean)",
            lambda_cons=lam,
            n_views=views,
            kernels=list(randconv.KERNELS),
            weight_law="N(0, 1/(C k^2))",
            mix="alpha * x + (1-alpha) * conv, alpha~U(0,1) per sample",
            lr=lr,
            epochs=epochs,
            steps=steps,
            batch=batch,
            tile=list(tile),
            max_minutes=max_minutes,
            seed=seed,
            bn="frozen_running_stats_and_affine",
            weight_decay=0.0,
            ckpt_rule="max proxy IoU@FAR1e-3 on the first 8 v8 val sequences, clean input",
            train_cache="artifacts/cache/v8/train",
            real_stats_used=False,
            real_csv_used=False,
            history=hist,
            best_val=float(best_loss),
            best_proxy_iou=float(best_proxy),
            best_epoch=int(best_epoch),
            stopped=stopped,
            seconds=round(time.time() - t_wall, 1),
            ckpt=str(out_pt),
        )

    bad = 0
    stopped = "epochs"
    for ep in range(epochs):
        if (time.time() - t_wall) / 60.0 > max_minutes:
            stopped = "time"
            break
        bn_eval()
        task_l, cons_l = [], []
        for _ in range(steps):
            x, y = sampler.sample(batch)
            xt = torch.from_numpy(np.ascontiguousarray(x))
            aug = torch.cat([randconv.apply_batch(xt, gen) for _v in range(views)], 0)
            logit = model(aug).view(views, batch, 3, tile[0], tile[1])
            task = losses.compute("focal+tversky_ce", logit[0], torch.from_numpy(y), cw)
            cons = randconv.consistency_kl(logit)
            loss = task + lam * cons
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            task_l.append(float(task.detach()))
            cons_l.append(float(cons.detach()))
        vl = val_loss()
        pi, pt = proxy_iou(model, proxy_files, predict)
        rec = dict(epoch=ep, train=float(np.mean(task_l) + lam * np.mean(cons_l)),
                   task=float(np.mean(task_l)), cons=float(np.mean(cons_l)),
                   val=vl, proxy_iou=pi, proxy_thr=pt, sec=round(time.time() - t_wall, 1))
        hist.append(rec)
        log(str(rec))
        if pi > best_proxy + 1e-4:
            best_proxy = pi
            best_loss = vl
            best_epoch = ep
            bad = 0
            save_ckpt()
        else:
            bad += 1
            if bad >= patience and ep >= 2:
                stopped = "patience"
                break
        out_json.write_text(json.dumps(payload(stopped), indent=1, ensure_ascii=False))
    out_json.write_text(json.dumps(payload(stopped), indent=1, ensure_ascii=False))
    log(f"готово best_proxy {best_proxy:.4f} epoch {best_epoch} stop {stopped} sec {time.time() - t_wall:.0f}")


if __name__ == "__main__":
    main()
