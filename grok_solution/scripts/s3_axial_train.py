"""Дообучение AxialUNet от artifacts/stage1/unet.pt на train синтетики v8.

Порог FAR и официальный IoU здесь не считаются: их ставит scripts/s3_axial_eval.py
по полному v8 val. Внутри цикла чекпойнт выбирается по IoU@FAR 1e-3 на нескольких
val-трассах (прокси), чтобы не сохранять минимум тайлового лосса в стороне от
рабочей точки. Реальные CSV не читаются.
"""
import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dts import data, models
from dts.axial import AxialUNet, load_unet_weights, param_counts, proj_l1

NBIN = 20000


def ce_dice_hardneg(logit, y, cw, neg_frac=0.02, neg_boost=2.0):
    """CE+dice как у stage-1, плюс усиление самых ярких фоновых пикселей.

    Рабочая точка — хвост FAR 1e-3. Доля neg_frac шире этого хвоста, вес
    стоит только на y==0, пиксели событий не штрафуются дополнительно.
    """
    ce_map = F.cross_entropy(logit, y, weight=cw, ignore_index=-1, reduction="none")
    with torch.no_grad():
        prob_fg = 1.0 - torch.softmax(logit, 1)[:, 0]
        bg = y == 0
        boost = torch.ones_like(ce_map)
        n_bg = int(bg.sum())
        if n_bg:
            k = max(1, int(neg_frac * n_bg))
            k = min(k, n_bg)
            thr = torch.topk(prob_fg[bg], k).values.min()
            boost = torch.where(bg & (prob_fg >= thr), boost * neg_boost, boost)
        valid = (y >= 0).to(ce_map.dtype)
        w = boost * valid
    ce = (ce_map * w).sum() / w.sum().clamp(min=1)
    p = torch.softmax(logit, 1)
    m = (y >= 0).float()
    fg = 1 - p[:, 0]
    t = (y > 0).float()
    dice = 1 - (2 * (fg * t * m).sum() + 1) / ((fg * m).sum() + (t * m).sum() + 1)
    return ce + dice


def thr_from_hist(hist, far):
    n = int(hist.sum())
    if n <= 0:
        return 1.0
    k = int(np.ceil((1.0 - far) * n)) - 1
    k = int(np.clip(k, 0, n - 1))
    cdf = np.cumsum(hist)
    i = int(np.searchsorted(cdf, k + 1, side="left"))
    i = min(max(i, 0), len(hist) - 1)
    return float((i + 0.5) / len(hist))


@torch.inference_mode()
def proxy_iou(model, paths, far=1e-3):
    """Пул фона нескольких трасс, порог FAR, пиксельный IoU. Модель уже в eval."""
    hist = np.zeros(NBIN, np.int64)
    packs = []
    for p in paths:
        z = np.load(p)
        pr = models.predict_full(model, np.asarray(z["X"], np.float32))
        s = 1.0 - pr[0]
        valid = np.asarray(z["valid"])
        ev = np.asarray(z["ev"])
        nu = np.asarray(z["nu"]) if "nu" in z.files else np.zeros_like(ev)
        bg = valid & (ev == 0) & (nu == 0)
        if bg.any():
            idx = np.clip((np.clip(s[bg], 0, 1) * (NBIN - 1)).astype(np.int32), 0, NBIN - 1)
            hist += np.bincount(idx, minlength=NBIN)
        packs.append((s, valid, ev > 0))
    thr = thr_from_hist(hist, far)
    tot = np.zeros(3, np.int64)
    for s, valid, truth in packs:
        pred = (s >= thr) & valid
        t = truth & valid
        tp = int((pred & t).sum())
        fp = int((pred & ~t).sum())
        fn = int((~pred & t).sum())
        tot += (tp, fp, fn)
    iou = float(tot[0] / max(tot.sum(), 1))
    return iou, thr, int(tot.sum())


def set_phase(model, frozen: bool):
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("axial.") or not frozen
    if frozen:
        model.eval()
        model.axial.train()
    else:
        model.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="artifacts/cache/v8")
    ap.add_argument("--init", default="artifacts/stage1/unet.pt")
    ap.add_argument("--out", default="artifacts/eval/s3_axial_model.pt")
    ap.add_argument("--log", default="artifacts/eval/s3_axial_train.json")
    ap.add_argument("--epochs-frozen", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=6, help="эпохи со размороженным костяком, после frozen")
    ap.add_argument("--steps-frozen", type=int, default=30)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--tile", default="128,256")
    ap.add_argument("--lr", type=float, default=1e-4, help="костяк")
    ap.add_argument("--lr-axial", type=float, default=1e-3)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--max-minutes", type=float, default=26)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--proxy-n", type=int, default=4)
    ap.add_argument("--proxy-every", type=int, default=2)
    a = ap.parse_args()

    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tile = tuple(int(v) for v in a.tile.split(","))
    mm = Path("artifacts/mm") / Path(a.cache).name
    tr = data.mmap_items(a.cache, "train", mm)
    va = data.mmap_items(a.cache, "val", mm)
    if not tr or not va:
        raise SystemExit(f"нет mmap {a.cache}: train {len(tr)} val {len(va)}")
    print("train", len(tr), "val", len(va), "tile", tile, flush=True)

    model = AxialUNet()
    missing = load_unet_weights(model, a.init)
    if any(not k.startswith("axial.") for k in missing):
        raise SystemExit(f"не загрузились веса U-Net: {missing[:8]}")
    counts = param_counts(model)
    print("params", counts, "proj_l1", proj_l1(model), "missing", len(missing), flush=True)

    ref = models.DTSUNet()
    ref.load_state_dict(torch.load(a.init, map_location="cpu", weights_only=False))
    ref.eval()
    model.eval()
    probe = torch.randn(1, 3, 64, 80)
    probe[:, 2] = 1
    with torch.no_grad():
        err = float((model(probe) - ref(probe)).abs().max())
    print("identity_maxabs", err, flush=True)
    if err > 1e-4:
        raise SystemExit("нулевая проекция не совпала с U-Net")
    del ref

    sampler = data.TileSampler(tr, tile=tile, seed=a.seed)
    sval = data.TileSampler(va, tile=tile, seed=1234)
    vx, vy = sval.sample(48)
    cw = torch.tensor([1.0, 3.0, 3.0])
    val_paths = data.list_cache(a.cache, "val")
    if a.proxy_n > 0:
        pick = np.linspace(0, len(val_paths) - 1, a.proxy_n).astype(int)
        proxy_paths = [val_paths[i] for i in pick]
    else:
        proxy_paths = []
    print("proxy", [p.stem for p in proxy_paths], flush=True)

    best_proxy, best_state, best_epoch = -1.0, None, -1
    best_loss, best_loss_state = 1e9, None
    hist = []
    t0 = time.time()
    if proxy_paths:
        model.eval()
        piou, pthr, _ = proxy_iou(model, proxy_paths)
        best_proxy, best_epoch = piou, -1
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        hist.append(dict(epoch=-1, phase="init", train=None, val=None, proxy_iou=piou, proxy_thr=pthr,
                         proj_l1=proj_l1(model), sec=round(time.time() - t0, 1)))
        print("proxy_init", hist[-1], flush=True)
    phases = [("frozen", a.epochs_frozen, a.steps_frozen)] + [("joint", a.epochs, a.steps)]
    global_ep = 0
    stop = False
    for phase, n_ep, steps in phases:
        frozen = phase == "frozen"
        set_phase(model, frozen)
        axial_params = [p for n, p in model.named_parameters() if n.startswith("axial.") and p.requires_grad]
        backbone_params = [p for n, p in model.named_parameters() if not n.startswith("axial.") and p.requires_grad]
        # После разморозки шаг осевого блока ниже, чем в замороженной фазе: проекция уже открыта.
        ax_lr = a.lr_axial if frozen else a.lr_axial * 0.3
        groups = [{"params": axial_params, "lr": ax_lr, "name": "axial"}]
        if backbone_params:
            groups.append({"params": backbone_params, "lr": a.lr, "name": "backbone"})
        opt = torch.optim.AdamW(groups, weight_decay=1e-4)
        base_lr = {"axial": ax_lr, "backbone": a.lr}
        print("phase", phase, "epochs", n_ep, "steps", steps,
              "axial", sum(p.numel() for p in axial_params),
              "backbone", sum(p.numel() for p in backbone_params), flush=True)
        phase_step = 0
        warm = min(10, steps)
        for _ in range(n_ep):
            if (time.time() - t0) / 60 > a.max_minutes:
                print("time budget", flush=True)
                stop = True
                break
            set_phase(model, frozen)
            losses = []
            for _step in range(steps):
                scale = (phase_step + 1) / warm if phase_step < warm else 1.0
                for group in opt.param_groups:
                    group["lr"] = base_lr[group["name"]] * scale
                x, y = sampler.sample(a.batch)
                logit = model(torch.from_numpy(np.ascontiguousarray(x)))
                loss = ce_dice_hardneg(logit, torch.from_numpy(np.ascontiguousarray(y)), cw)
                if not torch.isfinite(loss):
                    raise SystemExit("loss не конечен")
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                losses.append(float(loss.detach()))
                phase_step += 1
            model.eval()
            with torch.no_grad():
                vloss = float(np.mean([
                    ce_dice_hardneg(
                        model(torch.from_numpy(np.ascontiguousarray(vx[i:i + 8]))),
                        torch.from_numpy(np.ascontiguousarray(vy[i:i + 8])),
                        cw,
                    ).item()
                    for i in range(0, len(vx), 8)
                ]))
            do_proxy = bool(proxy_paths) and (global_ep % a.proxy_every == 0 or _ == n_ep - 1)
            piou, pthr = None, None
            if do_proxy:
                piou, pthr, _ = proxy_iou(model, proxy_paths)
                if piou > best_proxy:
                    best_proxy = piou
                    best_epoch = global_ep
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if vloss < best_loss:
                best_loss = vloss
                best_loss_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            row = dict(epoch=global_ep, phase=phase, train=float(np.mean(losses)), val=vloss,
                       proxy_iou=piou, proxy_thr=pthr, proj_l1=proj_l1(model),
                       sec=round(time.time() - t0, 1))
            hist.append(row)
            print(row, flush=True)
            global_ep += 1
        if stop:
            break

    if best_state is None:
        best_state = best_loss_state if best_loss_state is not None else model.state_dict()
        best_epoch = -1
    torch.save(best_state, out)
    # контрольный прогон проекции на сохранённых весах
    model.load_state_dict(best_state)
    report = dict(args=vars(a), params=counts, identity_maxabs=err, history=hist,
                  best_proxy_iou=None if best_proxy < 0 else best_proxy, best_epoch=best_epoch,
                  best_tile_val=best_loss, proj_l1=proj_l1(model), seconds=round(time.time() - t0, 1))
    Path(a.log).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(a.log, "w"), indent=1, ensure_ascii=False)
    print("saved", out, "best_epoch", best_epoch, "best_proxy", best_proxy, "proj_l1", report["proj_l1"], flush=True)


if __name__ == "__main__":
    main()
