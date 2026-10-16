"""Многозадачная U-Net: сегментация и гетероскедастичная регрессия ΔT.

Только синтетика v8 train. Реальные CSV и спектр кабеля не читаются.
Старт — artifacts/stage1/unet_v810ft.pt. Потоки 2, стена не больше 45 минут.
Чекпойнт — максимум proxy IoU@FAR 1e-3 на первых 8 val. Официальный выбор
делает scripts/s3_mtl_eval.py.

  OMP_NUM_THREADS=2 /workspace/repo/run/.venv/bin/python -u scripts/s3_mtl_train.py
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_k] = "2"

import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from dts import data, losses, metrics as M, models

AMP_DIR = Path("/tmp/s3_mtl_amp")


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S', time.gmtime())} {msg}", flush=True)


class MTLUNet(nn.Module):
    """Сегментационная U-Net и 1×1 голова (μ, log σ²) с последнего декодера.

    На выводе без aux возвращаются только логиты сегментации: карта детектора
    считается так же, как у обычной U-Net. dts/models.py не меняется.
    """

    def __init__(self, unet: nn.Module):
        super().__init__()
        self.unet = unet
        self.amp = nn.Conv2d(16, 2, 1)
        nn.init.zeros_(self.amp.weight)
        nn.init.zeros_(self.amp.bias)
        # σ ≈ e^1 °C на старте: регрессия не орёт громче сегментации.
        self.amp.bias.data[1] = 2.0
        self.s_seg = nn.Parameter(torch.zeros(()))
        self.s_reg = nn.Parameter(torch.tensor(2.0))
        self._d1 = None
        self.unet.d1.register_forward_hook(self._keep_d1)

    def _keep_d1(self, _module, _inp, out):
        self._d1 = out

    def forward(self, x, aux: bool = False):
        logits = self.unet(x)
        if not aux:
            return logits
        raw = self.amp(self._d1)
        mu = raw[:, 0]
        s = raw[:, 1].clamp(-8.0, 8.0)
        return logits, mu, s


def task_loss(logits, y, mu, s_pix, amp, cw, s_seg, s_reg):
    """Сегментация focal+tversky и Gaussian NLL ΔT, веса Кендалла и Гала.

    s_pix, s_seg, s_reg — log σ².
    NLL_i = ½ e^{-s_i} (y_i−μ_i)² + ½ s_i только на пикселях события.
    L = e^{-s_seg} L_seg + ½ s_seg + e^{-s_reg} L_reg + ½ s_reg.
    Это 1/σ²·L + log σ для каждой задачи. У регрессии множитель ½ уже внутри NLL.
    """
    seg = losses.compute("focal+tversky_ce", logits, y, cw)
    ss = s_seg.clamp(-8.0, 8.0)
    sr = s_reg.clamp(-8.0, 8.0)
    total = torch.exp(-ss) * seg + 0.5 * ss
    event = y > 0
    n_ev = int(event.sum())
    reg_value = None
    if n_ev > 0:
        err2 = (mu - amp) ** 2
        nll = 0.5 * torch.exp(-s_pix) * err2 + 0.5 * s_pix
        reg = nll[event].mean()
        total = total + torch.exp(-sr) * reg + 0.5 * sr
        reg_value = float(reg.detach().item())
    return total, float(seg.detach().item()), reg_value, float(ss.detach().item()), float(sr.detach().item())


def _check_loss() -> None:
    torch.manual_seed(0)
    logits = torch.zeros(2, 3, 4, 4)
    logits[:, 1] = 2.0
    y = torch.zeros(2, 4, 4, dtype=torch.long)
    y[:, 0, 0] = 1
    mu = torch.zeros(2, 4, 4)
    mu[:, 0, 0] = 3.0
    s_pix = torch.zeros(2, 4, 4)
    amp = torch.zeros(2, 4, 4)
    amp[:, 0, 0] = 3.0
    cw = torch.tensor([1.0, 3.0, 3.0])
    s_seg = torch.zeros(())
    s_reg = torch.zeros(())
    total, seg, reg, _, _ = task_loss(logits, y, mu, s_pix, amp, cw, s_seg, s_reg)
    # ошибка 0 и s_pix=0 → NLL = 0 на этих пикселях
    if reg is None or abs(reg) > 1e-5:
        raise SystemExit(f"NLL при нулевой ошибке {reg}")
    if not torch.isfinite(total):
        raise SystemExit("потеря не конечна на проверке")
    # вес e^{0}=1 не меняет сегментацию относительно голой потери, когда регрессия 0 и s=0
    bare = losses.compute("focal+tversky_ce", logits, y, cw)
    if abs(float(total) - float(bare)) > 1e-4:
        raise SystemExit(f"при s=0 и нулевой NLL сумма {float(total)} != сегментация {float(bare)}")
    log("проверка потери ок")


class AmpSampler:
    """Как TileSampler, плюс карта подписанного ΔT (°C) того же тайла."""

    def __init__(self, items, tile=(128, 256), p_event=0.6, seed=0):
        self.items = items
        self.tile = tile
        self.p_event = p_event
        self.rng = np.random.default_rng(seed)
        self.ev_idx = []
        for it in items:
            idx = np.argwhere(np.asarray(it["y"]) > 0)
            if len(idx) > 5000:
                idx = idx[self.rng.choice(len(idx), 5000, replace=False)]
            self.ev_idx.append(idx)
        self.weights = np.array([it["y"].size for it in items], float)
        self.weights /= self.weights.sum()

    def sample(self, n):
        th, tw = self.tile
        xs, ys, amps = [], [], []
        for _ in range(n):
            k = int(self.rng.choice(len(self.items), p=self.weights))
            it = self.items[k]
            X, y, amp = it["X"], it["y"], it["amp"]
            T, W = y.shape
            if self.rng.random() < self.p_event and len(self.ev_idx[k]):
                c = self.ev_idx[k][self.rng.integers(len(self.ev_idx[k]))]
                t0 = int(c[0]) - int(self.rng.integers(th))
                w0 = int(c[1]) - int(self.rng.integers(tw))
            else:
                t0 = int(self.rng.integers(max(T - th, 0) + 1))
                w0 = int(self.rng.integers(max(W - tw, 0) + 1))
            t0 = int(np.clip(t0, 0, max(T - th, 0)))
            w0 = int(np.clip(w0, 0, max(W - tw, 0)))
            xt = np.zeros((3, th, tw), np.float32)
            yt = np.full((th, tw), -1, np.int64)
            at = np.zeros((th, tw), np.float32)
            a = X[:, t0:t0 + th, w0:w0 + tw].astype(np.float32)
            b = y[t0:t0 + th, w0:w0 + tw]
            c = np.asarray(amp[t0:t0 + th, w0:w0 + tw], np.float32)
            xt[:, :a.shape[1], :a.shape[2]] = a
            yt[:b.shape[0], :b.shape[1]] = b
            at[:c.shape[0], :c.shape[1]] = c
            if self.rng.random() < 0.5:
                xt = xt[:, :, ::-1]
                yt = yt[:, ::-1]
                at = at[:, ::-1]
            g = float(self.rng.uniform(0.75, 1.3))
            xt[:2] = np.clip(xt[:2] * g, -12, 12)
            xs.append(np.ascontiguousarray(xt))
            ys.append(np.ascontiguousarray(yt))
            amps.append(np.ascontiguousarray(at))
        return np.stack(xs), np.stack(ys), np.stack(amps)


def _eval_mod():
    spec = importlib.util.spec_from_file_location("s3_mtl_eval", ROOT / "scripts" / "s3_mtl_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_amp_maps(files, labels) -> None:
    if AMP_DIR.exists():
        shutil.rmtree(AMP_DIR)
    AMP_DIR.mkdir(parents=True)
    ids = {p.stem for p in files}
    lab = labels[labels.seq_id.isin(ids)]
    t0 = time.time()
    n_ev = 0
    for i, p in enumerate(files, 1):
        with np.load(p) as z:
            ev = np.asarray(z["ev"])
            x = np.asarray(z["x"])
            t = np.asarray(z["t"])
        amp = np.zeros(ev.shape, np.float16)
        for r in M.label_boxes(lab, p.stem).itertuples():
            if r.kind != "event":
                continue
            px = M.event_pixels(r, x, t, ev)
            if not px.any():
                continue
            amp[px] = np.float16(r.severity)
            n_ev += 1
        np.save(AMP_DIR / f"{p.stem}.npy", amp)
        del amp, ev
        if i == 1 or i % 40 == 0 or i == len(files):
            log(f"ΔT карты {i}/{len(files)} событий {n_ev} {time.time() - t0:.0f}с")
    log(f"подписанный ΔT = severity, |severity|≈amp_obs_C, событий {n_ev}")


def assert_mmap(ver: str, split: str) -> None:
    cache = Path("artifacts/cache") / ver / split
    mm = Path("artifacts/mm") / ver / split
    missing = [p.name for p in sorted(cache.glob("seq*.npz")) if not (mm / f"{p.stem}_y.npy").exists()]
    if missing:
        raise SystemExit(f"нет mmap {ver}/{split}: {len(missing)} файлов, новый кэш не пишу")


def proxy_iou(model, files, predict_batched) -> tuple[float, float]:
    model.eval()
    maps = []
    parts = []
    for p in files:
        with np.load(p) as z:
            prob = predict_batched(model, np.asarray(z["X"], np.float32))
            valid = np.asarray(z["valid"], bool)
            ev = np.asarray(z["ev"])
            nu = np.asarray(z["nu"])
        s = 1.0 - prob[0]
        parts.append(s[valid & (ev == 0) & (nu == 0)])
        maps.append((s, valid, ev > 0))
        del prob
    bg = np.concatenate(parts)
    thr = M.far_threshold(bg, 1e-3)
    tot = np.zeros(3, np.int64)
    for s, valid, truth in maps:
        tot += np.asarray(M.confusion((s >= thr) & valid, truth, valid), np.int64)
    return float(M.prf(*tot)["iou"]), float(thr)


def freeze_bn(model) -> None:
    for mod in model.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.eval()
            for p in mod.parameters():
                p.requires_grad_(False)


def bn_eval(model) -> None:
    model.train()
    freeze_bn(model)


def main() -> None:
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    _check_loss()
    epochs = int(os.environ.get("MTL_EPOCHS", "8"))
    steps = int(os.environ.get("MTL_STEPS", "80"))
    batch = int(os.environ.get("MTL_BATCH", "8"))
    lr = float(os.environ.get("MTL_LR", "1e-4"))
    max_minutes = min(float(os.environ.get("MTL_MAX_MINUTES", "40")), 45.0)
    seed = int(os.environ.get("MTL_SEED", "0"))
    patience = int(os.environ.get("MTL_PATIENCE", "3"))
    tile = (128, 256)
    init = Path("artifacts/stage1/unet_v810ft.pt")
    out_json = Path("artifacts/eval/s3_mtl_train.json")
    out_pt = Path("artifacts/eval/s3_mtl_unet.pt")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    import pandas as pd
    labels = pd.read_csv("data/synth_meta/v8/labels.csv")
    assert_mmap("v8", "train")
    train_files = data.list_cache("artifacts/cache/v8", "train")
    try:
        build_amp_maps(train_files, labels)
        tr = data.mmap_items("artifacts/cache/v8", "train", Path("artifacts/mm/v8"))
        for it in tr:
            it["amp"] = np.load(AMP_DIR / f"{it['name']}.npy", mmap_mode="r")
        log(f"train {len(tr)}")
        sampler = AmpSampler(tr, tile=tile, seed=seed)
        t_aug = time.time()
        sampler.sample(batch)
        log(f"batch {batch} за {time.time() - t_aug:.2f} с")

        unet = models.build("unet", 16)
        unet.load_state_dict(torch.load(init, map_location="cpu", weights_only=True))
        model = MTLUNet(unet)
        freeze_bn(model)
        with torch.no_grad():
            probe = torch.zeros(1, 3, 32, 40)
            a = model.unet(probe)
            b = model(probe)
            if float((a - b).abs().max()) > 0:
                raise SystemExit("обёртка меняет логиты сегментации")
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
        cw = torch.tensor([1.0, 3.0, 3.0])
        predict_batched = _eval_mod().predict_batched
        err = _eval_mod().check_predict(model)
        log(f"predict max|Δ| {err:.3e}")
        proxy_files = data.list_cache("artifacts/cache/v8", "val")[:8]

        def save_ckpt():
            tmp = out_pt.with_suffix(".pt.tmp")
            torch.save(model.state_dict(), tmp)
            os.replace(tmp, out_pt)

        def payload(hist, best_proxy, best_epoch, stopped, seconds):
            return dict(
                task="MTL",
                init=str(init),
                arch="unet+amp_head",
                width=16,
                seg_loss="focal+tversky_ce",
                reg="heteroscedastic Gaussian NLL of signed severity °C on event pixels",
                weighting="Kendall&Gal homoscedastic task weights, s=log sigma^2, L = exp(-s)*L + 0.5*s",
                s_reg_init=2.0,
                s_pix_bias_init=2.0,
                lr=lr,
                weight_decay=0.0,
                epochs=epochs,
                steps=steps,
                batch=batch,
                tile=[128, 256],
                max_minutes=max_minutes,
                seed=seed,
                bn="frozen_running_stats_and_affine",
                target="label severity (°C), signed; magnitude matches amp_obs_C; event pixels only",
                ckpt_rule="max proxy IoU@FAR1e-3 on the first 8 sorted v8 val sequences",
                train_cache="artifacts/cache/v8/train",
                real_traces_used=False,
                replaced_release=False,
                history=hist,
                best_proxy_iou=None if best_proxy is None else float(best_proxy),
                best_epoch=int(best_epoch),
                stopped=stopped,
                seconds=round(seconds, 1),
                ckpt=str(out_pt),
                predict_max_abs_err=err,
            )

        best_proxy, best_thr = proxy_iou(model, proxy_files, predict_batched)
        best_epoch = -1
        save_ckpt()
        t0 = time.time()
        hist = [dict(epoch=-1, train=None, proxy_iou=best_proxy, proxy_thr=best_thr, sec=0.0)]
        log(f"init proxy_iou {best_proxy:.4f} thr {best_thr:.6f}")
        out_json.write_text(json.dumps(payload(hist, best_proxy, best_epoch, "init", 0.0), indent=1, ensure_ascii=False))
        bad = 0
        stopped = "epochs"
        for ep in range(epochs):
            if (time.time() - t0) / 60.0 > max_minutes:
                stopped = "time"
                break
            bn_eval(model)
            tl, seg_l, reg_l = [], [], []
            last_ss, last_sr = 0.0, 0.0
            cut = False
            for step in range(steps):
                if step % 10 == 0 and (time.time() - t0) / 60.0 > max_minutes:
                    cut = True
                    stopped = "time"
                    break
                x, y, amp = sampler.sample(batch)
                logits, mu, s_pix = model(torch.from_numpy(x), aux=True)
                loss, seg_v, reg_v, last_ss, last_sr = task_loss(
                    logits, torch.from_numpy(y), mu, s_pix, torch.from_numpy(amp), cw, model.s_seg, model.s_reg,
                )
                if not torch.isfinite(loss):
                    raise SystemExit(f"loss не конечен на эпохе {ep} шаге {step}")
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                model._d1 = None
                tl.append(float(loss.item()))
                seg_l.append(seg_v)
                if reg_v is not None:
                    reg_l.append(reg_v)
            if not tl:
                break
            pi, pt = proxy_iou(model, proxy_files, predict_batched)
            rec = dict(
                epoch=ep,
                train=float(np.mean(tl)),
                seg=float(np.mean(seg_l)),
                reg=None if not reg_l else float(np.mean(reg_l)),
                s_seg=last_ss,
                s_reg=last_sr,
                steps_done=len(tl),
                proxy_iou=pi,
                proxy_thr=pt,
                sec=round(time.time() - t0, 1),
            )
            hist.append(rec)
            log(str(rec))
            if pi > best_proxy + 1e-4:
                best_proxy = pi
                best_epoch = ep
                bad = 0
                save_ckpt()
            else:
                bad += 1
                if bad >= patience and ep >= 2:
                    stopped = "patience"
                    break
            if cut:
                stopped = "time"
                break
            out_json.write_text(json.dumps(payload(hist, best_proxy, best_epoch, stopped, time.time() - t0), indent=1, ensure_ascii=False))
        model.load_state_dict(torch.load(out_pt, map_location="cpu", weights_only=True))
        final = payload(hist, best_proxy, best_epoch, stopped, time.time() - t0)
        out_json.write_text(json.dumps(final, indent=1, ensure_ascii=False))
        log(f"готово best_proxy {best_proxy:.4f} epoch {best_epoch} stop {stopped} sec {final['seconds']}")
    finally:
        if AMP_DIR.exists():
            shutil.rmtree(AMP_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
