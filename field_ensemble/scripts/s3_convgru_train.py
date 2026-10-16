"""Дообучение ConvGRU-U-Net от unet_v810ft. Только synth v8 и v10 train.

Реальные CSV не читаются, mmap v10 не создаётся. Бюджет 45 минут, 2 потока.
Чекпойнт — максимум прокси IoU@FAR 1e-3 на четырёх последовательностях v8 val.
Официальный отбор — в scripts/s3_convgru_eval.py.

  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \\
  /workspace/repo/run/.venv/bin/python -u scripts/s3_convgru_train.py
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
from dts.convgru import ConvGRUNet, load_unet_weights, param_counts, predict_full, proj_l1

TILE = (128, 256)
STEP_T = 96  # 128 − 32, кратно 8: состояние берётся после строки бутылочного слоя 11
STATE_K = STEP_T // 8 - 1


def _bg_tail(logit, y):
    """Квадрат оценки аномалии на фоне: хвост FAR 1e-3 не должен раздуваться."""
    s = 1.0 - torch.softmax(logit, 1)[:, 0]
    bg = y == 0
    if int(bg.sum()) < 8:
        return logit.new_zeros(())
    return 0.05 * (s.square() * bg).sum() / bg.sum()


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S', time.gmtime())} {msg}", flush=True)


def crop(X, y, t0, w0, th, tw):
    xt = np.zeros((3, th, tw), np.float32)
    yt = np.full((th, tw), -1, np.int64)
    a = np.asarray(X[:, t0:t0 + th, w0:w0 + tw], np.float32)
    b = np.asarray(y[t0:t0 + th, w0:w0 + tw])
    xt[:, :a.shape[1], :a.shape[2]] = a
    yt[:b.shape[0], :b.shape[1]] = b
    return xt, yt


class PairSource:
    """Один кадр в памяти. v8 уже mmap, v10 читается из npz и не сохраняется."""

    def __init__(self, kind: str, items, seed: int):
        self.kind = kind
        self.items = items
        self.rng = np.random.default_rng(seed)
        self.cache = None

    def _materialize(self, it):
        if self.kind == "v8":
            X, y = it["X"], it["y"]
            ev = np.argwhere(np.asarray(y) > 0)
        else:
            with np.load(it) as z:
                X = np.array(z["X"], np.float16, copy=True)
                y = data.target(np.asarray(z["ev"]), np.asarray(z["valid"]))
            ev = np.argwhere(y > 0)
        if len(ev) > 4000:
            ev = ev[self.rng.choice(len(ev), 4000, replace=False)]
        return X, y, ev

    def batch(self, n: int):
        it = self.items[int(self.rng.integers(len(self.items)))]
        key = it["name"] if self.kind == "v8" else it.name
        if self.cache is None or self.cache[0] != key:
            X, y, ev = self._materialize(it)
            self.cache = (key, X, y, ev)
        _, X, y, ev = self.cache
        th, tw = TILE
        T, W = y.shape
        can_pair = T >= th + STEP_T
        xs0, ys0, xs1, ys1 = [], [], [], []
        for _ in range(n):
            if len(ev) and self.rng.random() < 0.6:
                c = ev[int(self.rng.integers(len(ev)))]
                ta, wa = int(c[0]), int(c[1])
            else:
                ta, wa = int(self.rng.integers(max(T, 1))), int(self.rng.integers(max(W, 1)))
            w0 = int(np.clip(wa - int(self.rng.integers(tw)), 0, max(W - tw, 0)))
            if can_pair:
                t0 = int(np.clip(ta - int(self.rng.integers(th)), 0, T - (th + STEP_T)))
            else:
                t0 = int(np.clip(ta - int(self.rng.integers(max(th, 1))), 0, max(T - th, 0)))
            x0, y0 = crop(X, y, t0, w0, th, tw)
            if can_pair:
                x1, y1 = crop(X, y, t0 + STEP_T, w0, th, tw)
            else:
                x1, y1 = x0, y0
            if self.rng.random() < 0.5:
                x0 = x0[:, :, ::-1].copy()
                y0 = y0[:, ::-1].copy()
                x1 = x1[:, :, ::-1].copy()
                y1 = y1[:, ::-1].copy()
            g = float(self.rng.uniform(0.75, 1.3))
            x0[:2] = np.clip(x0[:2] * g, -12, 12)
            x1[:2] = np.clip(x1[:2] * g, -12, 12)
            xs0.append(x0)
            ys0.append(y0)
            xs1.append(x1)
            ys1.append(y1)
        return np.stack(xs0), np.stack(ys0), np.stack(xs1), np.stack(ys1), can_pair


def proxy_iou(model, files) -> tuple[float, float]:
    model.eval()
    parts, maps = [], []
    for p in files:
        z = np.load(p)
        prob = predict_full(model, np.asarray(z["X"], np.float32))
        s = 1.0 - prob[0]
        valid = np.asarray(z["valid"], bool)
        ev, nu = z["ev"], z["nu"]
        parts.append(s[valid & (ev == 0) & (nu == 0)])
        maps.append((s, valid, ev > 0))
        z.close()
    bg = np.concatenate(parts)
    thr = float(M.far_threshold(bg, 1e-3))
    tot = np.zeros(3, np.int64)
    for s, valid, truth in maps:
        tot += np.asarray(M.confusion((s >= thr) & valid, truth, valid), np.int64)
    return float(M.prf(*[int(v) for v in tot])["iou"]), thr


def assert_identity() -> float:
    torch.manual_seed(0)
    net = ConvGRUNet(w=16)
    load_unet_weights(net, "artifacts/stage1/unet_v810ft.pt")
    base = models.build("unet", 16)
    base.load_state_dict(torch.load("artifacts/stage1/unet_v810ft.pt", map_location="cpu", weights_only=True))
    net.eval()
    base.eval()
    rng = np.random.default_rng(0)
    X = rng.standard_normal((3, 280, 640)).astype(np.float32)
    X[2] = (rng.random((280, 640)) > 0.1).astype(np.float32)
    err = float(np.max(np.abs(predict_full(net, X) - models.predict_full(base, X))))
    if err > 1e-4:
        raise SystemExit(f"ConvGRU при нулевой проекции не совпал с U-Net: {err}")
    return err


def main() -> None:
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    epochs = int(os.environ.get("CONVGRU_EPOCHS", "24"))
    steps = int(os.environ.get("CONVGRU_STEPS", "80"))
    batch = int(os.environ.get("CONVGRU_BATCH", "4"))
    max_minutes = float(os.environ.get("CONVGRU_MAX_MINUTES", "45"))
    patience = int(os.environ.get("CONVGRU_PATIENCE", "6"))
    seed = int(os.environ.get("CONVGRU_SEED", "0"))
    lr_back = float(os.environ.get("CONVGRU_LR", "0"))
    lr_gru = float(os.environ.get("CONVGRU_LR_GRU", "1e-5"))
    init = Path("artifacts/stage1/unet_v810ft.pt")
    out_json = Path("artifacts/eval/s3_convgru_train.json")
    out_pt = Path("artifacts/eval/s3_convgru_unet.pt")
    out_json.parent.mkdir(parents=True, exist_ok=True)

    err = assert_identity()
    log(f"тождество с unet_v810ft max|Δ| {err:.3e}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    v8 = data.mmap_items("artifacts/cache/v8", "train", Path("artifacts/mm/v8"))
    v10_paths = data.list_cache("artifacts/cache/v10", "train")
    if len(v8) < 10 or len(v10_paths) < 10:
        raise SystemExit(f"мало train: v8 {len(v8)} v10 {len(v10_paths)}")
    src8 = PairSource("v8", v8, seed)
    src10 = PairSource("v10", v10_paths, seed + 1)
    log(f"train v8 {len(v8)} v10 {len(v10_paths)} без нового mmap")

    model = ConvGRUNet(w=16)
    missing = load_unet_weights(model, init)
    counts = param_counts(model)
    log(f"параметры {counts} недостающие ключи {len(missing)} proj {proj_l1(model):.3e}")
    for name, p in model.named_parameters():
        # Остов уже откалиброван под FAR 1e-3. Крупный шаг его ломает за эпоху.
        # Учится только ConvGRU; нулевая проекция сохраняет старт.
        p.requires_grad_(name.startswith("gru."))
    for mod in model.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.eval()
    gru = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(gru, lr=lr_gru, weight_decay=0.0)
    cw = torch.tensor([1.0, 3.0, 3.0])

    def bn_eval():
        model.train()
        for mod in model.modules():
            if isinstance(mod, torch.nn.BatchNorm2d):
                mod.eval()

    def save_ckpt():
        tmp = out_pt.with_suffix(".pt.tmp")
        torch.save(model.state_dict(), tmp)
        os.replace(tmp, out_pt)

    files = data.list_cache("artifacts/cache/v8", "val")
    proxy_files = [files[i] for i in (0, 11, 22, 33) if i < len(files)]
    best_proxy, best_thr = proxy_iou(model, proxy_files)
    best_epoch = -1
    save_ckpt()
    hist = [dict(epoch=-1, train=None, proxy_iou=best_proxy, proxy_thr=best_thr, sec=0.0, proj_l1=proj_l1(model))]
    log(f"init proxy_iou {best_proxy:.4f} thr {best_thr:.4f}")

    def payload(stopped, seconds):
        return dict(
            name="CONVGRU",
            init=str(init),
            arch="convgru_unet",
            width=16,
            params=counts,
            loss="focal+tversky_ce",
            lr_backbone=lr_back,
            lr_gru=lr_gru,
            epochs=epochs,
            steps=steps,
            batch=batch,
            tile=[128, 256],
            pair_step=STEP_T,
            state_index=STATE_K,
            max_minutes=max_minutes,
            seed=seed,
            bn="frozen_running_stats_and_affine",
            train_cache=["artifacts/cache/v8/train", "artifacts/cache/v10/train"],
            v10_mmap=False,
            ckpt_rule="max proxy IoU@FAR1e-3 on 4 spread v8 val sequences; full 45-seq val decides the ensemble",
            proxy_files=[p.name for p in proxy_files],
            history=hist,
            best_proxy_iou=float(best_proxy),
            best_epoch=int(best_epoch),
            stopped=stopped,
            seconds=round(seconds, 1),
            ckpt="artifacts/eval/s3_convgru_unet.pt",
            identity_max_abs=err,
            real_traces_used=False,
            reference="ConvGRU Ballas et al. 2016; tile state carry. ConvLSTM Shi et al. 2015 not used (extra cell).",
        )

    out_json.write_text(json.dumps(payload("init", 0.0), indent=1, ensure_ascii=False))
    bad = 0
    t0 = time.time()
    deadline = t0 + max_minutes * 60.0
    stopped = "epochs"
    for ep in range(epochs):
        if time.time() >= deadline:
            stopped = "time"
            break
        bn_eval()
        tl = []
        cut = False
        for i in range(steps):
            if time.time() >= deadline:
                cut = True
                stopped = "time"
                break
            src = src8 if i % 2 == 0 else src10
            x0, y0, x1, y1, paired = src.batch(batch)
            x0_t = torch.from_numpy(x0)
            y0_t = torch.from_numpy(y0)
            logits, h_seq = model(x0_t, return_state=True)
            loss = losses.compute("focal+tversky_ce", logits, y0_t, cw)
            loss = loss + _bg_tail(logits, y0_t)
            if paired:
                h0 = h_seq[:, :, STATE_K].detach()
                x1_t = torch.from_numpy(x1)
                y1_t = torch.from_numpy(y1)
                logits1 = model(x1_t, h0)
                loss = 0.5 * (loss + losses.compute("focal+tversky_ce", logits1, y1_t, cw) + _bg_tail(logits1, y1_t))
            # Нулевая проекция легко раздувает хвост фона. Лёгкий L2 держит поправку маленькой.
            loss = loss + 20.0 * model.gru.proj.weight.square().sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl.append(float(loss.detach()))
        if cut or not tl:
            log(f"стоп на эпохе {ep}, шагов {len(tl)}")
            break
        if time.time() >= deadline:
            stopped = "time"
            hist.append(dict(epoch=ep, train=float(np.mean(tl)), proxy_iou=None, sec=round(time.time() - t0, 1)))
            log(f"эпоха {ep} без прокси, бюджет вышел")
            break
        pi, pt = proxy_iou(model, proxy_files)
        rec = dict(
            epoch=ep, train=float(np.mean(tl)), proxy_iou=pi, proxy_thr=pt,
            sec=round(time.time() - t0, 1), proj_l1=proj_l1(model),
        )
        hist.append(rec)
        log(str(rec))
        if pi > best_proxy + 1e-4:
            best_proxy = pi
            best_epoch = ep
            bad = 0
            save_ckpt()
            log(f"сохранён чекпойнт эпоха {ep} proxy {pi:.4f}")
        else:
            bad += 1
            if bad >= patience and ep >= 8:
                stopped = "patience"
                out_json.write_text(json.dumps(payload(stopped, time.time() - t0), indent=1, ensure_ascii=False))
                break
        out_json.write_text(json.dumps(payload("running", time.time() - t0), indent=1, ensure_ascii=False))

    out_json.write_text(json.dumps(payload(stopped, time.time() - t0), indent=1, ensure_ascii=False))
    log(f"готово best_proxy {best_proxy:.4f} epoch {best_epoch} stop {stopped} sec {time.time() - t0:.0f}")


if __name__ == "__main__":
    main()
