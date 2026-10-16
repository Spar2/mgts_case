"""Дообучение U-Net шумом со спектром реального фона.

Порог и тест здесь не считаются. Реальные CSV не читаются.
Старт — artifacts/stage1/unet_v810ft.pt, отбор чекпойнта по чистому val-loss.
Запуск из /workspace/dts_solution:

  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \\
  /workspace/repo/run/.venv/bin/python -u scripts/s3_dom_train.py
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

from dts import aug_real, data, losses, metrics as M, models
import importlib.util


def _eval_mod():
    spec = importlib.util.spec_from_file_location("s3_dom_eval", ROOT / "scripts" / "s3_dom_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S', time.gmtime())} {msg}", flush=True)


def assert_mmap(ver: str, split: str) -> None:
    cache = Path("artifacts/cache") / ver / split
    mm = Path("artifacts/mm") / ver / split
    missing = [p.name for p in sorted(cache.glob("seq*.npz")) if not (mm / f"{p.stem}_y.npy").exists()]
    if missing:
        raise SystemExit(f"нет mmap {ver}/{split}: {len(missing)} файлов, новый кэш не пишу")


def main() -> None:
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    epochs = int(os.environ.get("DOM_EPOCHS", "8"))
    steps = int(os.environ.get("DOM_STEPS", "80"))
    batch = int(os.environ.get("DOM_BATCH", "8"))
    lr = float(os.environ.get("DOM_LR", "1e-4"))
    max_minutes = float(os.environ.get("DOM_MAX_MINUTES", "22"))
    seed = int(os.environ.get("DOM_SEED", "0"))
    patience = int(os.environ.get("DOM_PATIENCE", "3"))
    tile = (128, 256)
    init = Path("artifacts/stage1/unet_v810ft.pt")
    out_json = Path(os.environ.get("DOM_OUT_JSON", "artifacts/eval/s3_dom_train.json"))
    out_pt = Path(os.environ.get("DOM_OUT_PT", "artifacts/eval/s3_dom_unet.pt"))
    out_json.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    log("загрузка спектра фона")
    bank = aug_real.load_bank()
    shaper, fit_info = aug_real.fit_shaper(bank, steps=12, seed=seed, tile=(160, 256), max_lag=16)
    log(f"шейпер {fit_info}")
    spec = aug_real.AugSpec(std_lo=0.25, std_hi=0.70, p_apply=0.7, sign_p=0.0)

    assert_mmap("v8", "train")
    assert_mmap("v8", "val")
    tr = data.mmap_items("artifacts/cache/v8", "train", Path("artifacts/mm/v8"))
    va = data.mmap_items("artifacts/cache/v8", "val", Path("artifacts/mm/v8"))
    n_sig = 0
    for it in tr:
        with np.load(it["src"]) as z:
            it["sigma"] = np.asarray(z["sigma"], np.float32)
        n_sig += 1
    log(f"train {len(tr)} val {len(va)} sigma {n_sig}")

    aug_rng = np.random.default_rng(seed + 23)

    def _aug(xt, yt, it, t0, w0, shaper=shaper, spec=spec, rng=aug_rng):
        sig = it["sigma"]
        tw = xt.shape[-1]
        col = np.full(tw, 0.03, np.float32)
        n = min(tw, max(int(sig.shape[0]) - int(w0), 0))
        if n > 0:
            col[:n] = np.asarray(sig[w0:w0 + n], np.float32)
        return aug_real.apply_tile(xt, yt, col, rng, shaper, spec)

    sampler = data.TileSampler(tr, tile=tile, seed=seed, p_weak=0.0, aug=_aug)
    sval = data.TileSampler(va, tile=tile, seed=1234, p_event=0.6)
    vx, vy = sval.sample(64)
    # одна пачка — оценка цены аугментации
    t_aug = time.time()
    sampler.sample(batch)
    log(f"aug batch {batch} за {time.time() - t_aug:.2f} с")

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

    predict_batched = _eval_mod().predict_batched
    proxy_files = data.list_cache("artifacts/cache/v8", "val")[:8]
    best_loss = val_loss()
    best_proxy, best_thr = proxy_iou(model, proxy_files, predict_batched)
    best_epoch = -1
    save_ckpt()
    hist = [dict(epoch=-1, train=None, val=best_loss, proxy_iou=best_proxy, proxy_thr=best_thr, sec=0)]
    log(f"init val_loss {best_loss:.4f} proxy_iou {best_proxy:.4f}")
    bad = 0
    t0 = time.time()
    stopped = "epochs"
    for ep in range(epochs):
        if (time.time() - t0) / 60 > max_minutes:
            stopped = "time"
            break
        bn_eval()
        tl = []
        for _ in range(steps):
            x, y = sampler.sample(batch)
            logit = model(torch.from_numpy(x))
            loss = losses.compute("focal+tversky_ce", logit, torch.from_numpy(y), cw)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl.append(float(loss.item()))
        vl = val_loss()
        pi, pt = proxy_iou(model, proxy_files, predict_batched)
        rec = dict(epoch=ep, train=float(np.mean(tl)), val=vl, proxy_iou=pi, proxy_thr=pt, sec=round(time.time() - t0, 1))
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
        payload = _payload(bank, spec, fit_info, hist, best_loss, best_proxy, best_epoch, init, lr, epochs, steps, batch, max_minutes, seed, stopped, time.time() - t0, out_pt)
        out_json.write_text(json.dumps(payload, indent=1, ensure_ascii=False))

    gap = _gap(bank, shaper, va)
    payload = _payload(bank, spec, fit_info, hist, best_loss, best_proxy, best_epoch, init, lr, epochs, steps, batch, max_minutes, seed, stopped, time.time() - t0, out_pt)
    payload["spectrum_gap"] = gap
    out_json.write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    log(f"готово best_proxy {best_proxy:.4f} epoch {best_epoch} stop {stopped} gap {gap}")


def proxy_iou(model, files, predict_batched) -> tuple[float, float]:
    """IoU@FAR 1e-3 на нескольких val-последовательностях. Порог только по их фону.

    Это внутренний отбор эпохи. Официальная цифра считается в s3_dom_eval по всем 45.
    """
    model.eval()
    maps = []
    parts = []
    for p in files:
        z = np.load(p)
        prob = predict_batched(model, np.asarray(z["X"], np.float32))
        s = 1.0 - prob[0]
        valid = np.asarray(z["valid"], bool)
        ev = z["ev"]
        nu = z["nu"]
        parts.append(s[valid & (ev == 0) & (nu == 0)])
        maps.append((s, valid, ev > 0))
    bg = np.concatenate(parts)
    thr = M.far_threshold(bg, 1e-3)
    tot = np.zeros(3, np.int64)
    for s, valid, truth in maps:
        tot += np.asarray(M.confusion((s >= thr) & valid, truth, valid), np.int64)
    return float(M.prf(*tot)["iou"]), float(thr)


def _payload(bank, spec, fit_info, hist, best_loss, best_proxy, best_epoch, init, lr, epochs, steps, batch, max_minutes, seed, stopped, seconds, ckpt):
    return dict(
        name="DOM",
        init=str(init),
        arch="unet",
        width=16,
        loss="focal+tversky_ce",
        lr=lr,
        epochs=epochs,
        steps=steps,
        batch=batch,
        tile=[128, 256],
        max_minutes=max_minutes,
        seed=seed,
        bn="frozen_running_stats_and_affine",
        weight_decay=0.0,
        ckpt_rule="max proxy IoU@FAR1e-3 on the first 8 v8 val sequences; full 45-seq val decides the candidate",
        train_cache="artifacts/cache/v8/train",
        val_cache_loss="artifacts/cache/v8/val tiles, no aug",
        aug=spec.as_dict(),
        psd=bank.as_meta(),
        shaper=fit_info,
        history=hist,
        best_val=float(best_loss),
        best_proxy_iou=float(best_proxy),
        best_epoch=int(best_epoch),
        stopped=stopped,
        seconds=round(seconds, 1),
        ckpt=str(ckpt),
        real_traces_used=False,
    )


def _gap(bank, shaper, val_items) -> dict:
    """АКФ фона synth до и после добавки шума. Метки не читаются, только маска ev==0."""
    rng = np.random.default_rng(0)
    src = bank.n_src - 1
    before, after = [], []
    noise_only = shaper.sample(rng, (512, 384), src, std=1.0)
    noise_match = aug_real.acf_match(noise_only, bank.acf_t[src], bank.acf_x[src], max_lag=24)
    for it in val_items[:4]:
        with np.load(it["src"]) as z:
            ch = np.asarray(z["X"][0], np.float32)
            valid = np.asarray(z["valid"], bool) & (np.asarray(z["ev"]) == 0) & (np.asarray(z["nu"]) == 0)
        # узкая полоса по времени, чтобы не держать всё поле
        t0 = max(int(ch.shape[0]) // 3, 0)
        sl = (slice(t0, t0 + 180), slice(0, min(800, ch.shape[1])))
        field = ch[sl]
        m = valid[sl]
        if int(m.sum()) < 1000:
            continue
        before.append(aug_real.spectrum_gap(field, m, bank, src=src, max_lag=16))
        noise = shaper.sample(rng, field.shape, src, std=0.45, valid=m)
        after.append(aug_real.spectrum_gap(np.where(m, field + noise, 0.0), m, bank, src=src, max_lag=16))
    def pack(rows):
        if not rows:
            return {}
        keys = ("corr_t", "corr_x", "phi_t", "phi_x")
        return {k: float(np.nanmean([r[k] for r in rows])) for k in keys}
    return {
        "noise_vs_pooled": noise_match,
        "synth_bg_before": pack(before),
        "synth_bg_plus_noise_after": pack(after),
        "target_phi_t": float(bank.acf_t[src, 1]),
        "target_phi_x": float(bank.acf_x[src, 1]),
        "n_seq": len(before),
    }


if __name__ == "__main__":
    main()
