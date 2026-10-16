"""Сетка силы реалистичного шума и доли аугментированных батчей.

Старт и потеря — как у DOM (unet_v810ft, focal+tversky_ce, BN заморожен).
Сила ×{0.5, 1.5} умножает интервал DOM std Uniform(0.25, 0.70)·σ.
Доля батчей — вероятность, что весь батч получает шум (внутри такого батча
p_apply=1). У DOM шум решается по тайлу с p=0.7; здесь это отдельный фактор,
уровни 0.5 и 1.0 окружают 0.7.

Официальный выбор — val IoU@FAR 1e-3 смеси ens_blend = среднее
ens_u2ft_tcn_lgbm и нового U-Net. Порог только по фону v8 val. Test не выбирает.
Реальные CSV не читаются. Бюджет обучения 35 мин, 2 потока.

  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \\
  /workspace/repo/run/.venv/bin/python -u scripts/s3_dom_sweep.py
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_k] = "2"

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from dts import aug_real, data, losses, metrics as M, models

WINNER = "ens_u2ft_tcn_lgbm"
WINNER_VAL = 0.38671235821914546
WINNER_TEST = 0.4052762065230473
DOM_BLEND_VAL = 0.3898
DOM_STD = (0.25, 0.70)
# (имя, множитель силы, доля аугментированных батчей)
GRID = (
    ("s05_f05", 0.5, 0.5),
    ("s05_f10", 0.5, 1.0),
    ("s15_f05", 1.5, 0.5),
    ("s15_f10", 1.5, 1.0),
)
TRAIN_BUDGET_S = 35 * 60
STEPS = 40
MAX_EPOCHS = 3
BATCH = 8
TILE = (128, 256)
LR = 1e-4
ENS = Path("artifacts/pred/v8") / WINNER
INIT = Path("artifacts/stage1/unet_v810ft.pt")
OUT_TRAIN = Path("artifacts/eval/s3_domsweep_train.json")
OUT_EVAL = Path("artifacts/eval/s3_domsweep_v8.json")


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S', time.gmtime())} {msg}", flush=True)


def plain(obj):
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return plain(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    return obj


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(plain(obj), indent=1, ensure_ascii=False))
    os.replace(tmp, path)


def scaled_spec(scale: float, batch_frac: float) -> aug_real.AugSpec:
    """Интервал DOM, умноженный на scale. p_apply=1: решение принято на батче."""
    if scale <= 0 or not 0.0 <= batch_frac <= 1.0:
        raise ValueError(f"scale {scale} frac {batch_frac}")
    return aug_real.AugSpec(DOM_STD[0] * scale, DOM_STD[1] * scale, p_apply=1.0, sign_p=0.0)


def tile_starts(n: int, tile: int, step: int) -> list[int]:
    ts = list(range(0, max(n - tile, 0) + 1, step)) or [0]
    if ts[-1] + tile < n:
        ts.append(n - tile)
    return ts


@torch.inference_mode()
def predict_batched(model, X, tile=(256, 512), overlap=32, batch=4):
    """Тот же тайлинг, что models.predict_full (сверка в DOM: max|Δ| ~ 1e-7)."""
    X = np.asarray(X, np.float32)
    _, t_len, w_len = X.shape
    th, tw = tile
    step_t, step_w = th - 2 * overlap, tw - 2 * overlap
    boxes = [(t0, w0) for t0 in tile_starts(t_len, th, step_t) for w0 in tile_starts(w_len, tw, step_w)]
    out = np.zeros((3, t_len, w_len), np.float32)
    cnt = np.zeros((t_len, w_len), np.float32)
    for i in range(0, len(boxes), batch):
        chunk = boxes[i:i + batch]
        xs, meta = [], []
        for t0, w0 in chunk:
            sl = X[:, t0:t0 + th, w0:w0 + tw]
            ht, wt = sl.shape[1], sl.shape[2]
            if ht != th or wt != tw:
                sl = np.pad(sl, ((0, 0), (0, th - ht), (0, tw - wt)))
            xs.append(sl)
            meta.append((t0, w0, ht, wt))
        prob = torch.softmax(model(torch.from_numpy(np.stack(xs))), 1).numpy()
        for p, (t0, w0, ht, wt) in zip(prob, meta):
            out[:, t0:t0 + ht, w0:w0 + wt] += p[:, :ht, :wt]
            cnt[t0:t0 + ht, w0:w0 + wt] += 1
    out /= np.maximum(cnt, 1.0)
    return out


def bg_mask(ev, nu, valid):
    return valid & (ev == 0) & (nu == 0)


def freeze_bn(model) -> None:
    for mod in model.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.eval()
            for p in mod.parameters():
                p.requires_grad_(False)


def bn_train_frozen(model) -> None:
    model.train()
    freeze_bn(model)


def load_init():
    model = models.build("unet", 16)
    sd = torch.load(INIT, map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    freeze_bn(model)
    return model, sd


def assert_mmap(split: str) -> None:
    cache = Path("artifacts/cache/v8") / split
    mm = Path("artifacts/mm/v8") / split
    missing = [p.name for p in sorted(cache.glob("seq*.npz")) if not (mm / f"{p.stem}_y.npy").exists()]
    if missing:
        raise SystemExit(f"нет mmap v8/{split}: {len(missing)} файлов, новый кэш не пишу")


def attach_sigma(items) -> None:
    for it in items:
        with np.load(it["src"]) as z:
            it["sigma"] = np.asarray(z["sigma"], np.float32)


def _self_check() -> None:
    a = scaled_spec(0.5, 0.5)
    b = scaled_spec(1.5, 1.0)
    assert abs(a.std_lo - 0.125) < 1e-12 and abs(a.std_hi - 0.35) < 1e-12
    assert abs(b.std_lo - 0.375) < 1e-12 and abs(b.std_hi - 1.05) < 1e-12
    assert a.p_apply == 1.0 and a.sign_p == 0.0
    rng = np.random.default_rng(0)
    draws = [rng.random() < 0.5 for _ in range(400)]
    # доля около 0.5, не константа
    rate = float(np.mean(draws))
    if not 0.4 < rate < 0.6:
        raise SystemExit(f"монета батча вне допуска: {rate}")


def cached_split(split: str, thr: float | None = None) -> dict:
    """IoU ансамбля по уже записанным картам. Порог — только если split==val и thr is None."""
    files = data.list_cache("artifacts/cache/v8", split)
    if thr is None:
        parts = []
        for p in files:
            with np.load(p) as z:
                mask = bg_mask(np.asarray(z["ev"]), np.asarray(z["nu"]), np.asarray(z["valid"], bool))
            with np.load(ENS / split / p.name) as pred:
                s = np.asarray(pred["s"], np.float32)
            parts.append(s[mask])
        bg = np.concatenate(parts)
        thr = float(M.far_threshold(bg, 1e-3))
        n_bg = int(bg.size)
        del parts, bg
    else:
        n_bg = None
    tot = np.zeros(3, np.int64)
    far_n = far_d = 0
    for p in files:
        with np.load(p) as z:
            ev = np.asarray(z["ev"])
            valid = np.asarray(z["valid"], bool)
            nu = np.asarray(z["nu"])
        with np.load(ENS / split / p.name) as pred:
            s = np.asarray(pred["s"], np.float32)
        pred_m = s >= thr
        tot += np.asarray(M.confusion(pred_m, ev > 0, valid), np.int64)
        mask = bg_mask(ev, nu, valid)
        far_n += int((pred_m & mask).sum())
        far_d += int(mask.sum())
    row = dict(threshold=float(thr), n_background=n_bg if n_bg is not None else int(far_d),
               far=float(far_n / max(far_d, 1)), **M.prf(*tot))
    return row


def sanity() -> dict:
    t0 = time.time()
    val = cached_split("val")
    test = cached_split("test", thr=val["threshold"])
    val_err = abs(val["iou"] - WINNER_VAL)
    test_err = abs(test["iou"] - WINNER_TEST)
    ok = val_err < 0.005 and test_err < 0.005
    log(f"sanity winner val {val['iou']:.6f} err {val_err:.3e} test {test['iou']:.6f} err {test_err:.3e} ok {ok}")
    if not ok:
        raise SystemExit("протокол не сошёлся с опубликованным ансамблем, обучение не запускаю")
    return dict(val=val, test=test, val_abs_err=val_err, test_abs_err=test_err,
                protocol_ok=True, seconds=round(time.time() - t0, 1))


def make_sampler(items, holder):
    def _aug(xt, yt, it, t0, w0, holder=holder):
        if not holder["on"]:
            return xt, yt
        sig = it["sigma"]
        tw = xt.shape[-1]
        col = np.full(tw, 0.03, np.float32)
        n = min(tw, max(int(sig.shape[0]) - int(w0), 0))
        if n > 0:
            col[:n] = np.asarray(sig[w0:w0 + n], np.float32)
        return aug_real.apply_tile(xt, yt, col, holder["rng"], holder["shaper"], holder["spec"])

    return data.TileSampler(items, tile=TILE, seed=0, p_weak=0.0, aug=_aug)


def proxy_blend(model, files) -> tuple[float, float]:
    """ens_blend IoU@FAR 1e-3 на нескольких val. Порог только по их фону. Не официальная цифра."""
    model.eval()
    parts, maps = [], []
    for p in files:
        with np.load(p) as z:
            X = np.asarray(z["X"], np.float32)
            valid = np.asarray(z["valid"], bool)
            ev = np.asarray(z["ev"])
            nu = np.asarray(z["nu"])
        prob = predict_batched(model, X)
        s = (1.0 - prob[0]).astype(np.float32)
        with np.load(ENS / "val" / p.name) as pred:
            ens = np.asarray(pred["s"], np.float32)
        if ens.shape != s.shape:
            raise SystemExit(f"{p.name}: ens {ens.shape} != unet {s.shape}")
        blend = (0.5 * (s + ens)).astype(np.float32)
        mask = bg_mask(ev, nu, valid)
        parts.append(blend[mask])
        maps.append((blend, valid, ev > 0))
        del X, prob, s, ens
    thr = float(M.far_threshold(np.concatenate(parts), 1e-3))
    tot = np.zeros(3, np.int64)
    for s, valid, truth in maps:
        tot += np.asarray(M.confusion(s >= thr, truth, valid), np.int64)
    return float(M.prf(*tot)["iou"]), thr


def train_all(proxy_files) -> list[dict]:
    assert_mmap("train")
    assert_mmap("val")
    log("mmap и спектр")
    bank = aug_real.load_bank()
    shaper, fit = aug_real.fit_shaper(bank, steps=0, seed=0, tile=(160, 256), max_lag=16)
    log(f"шейпер steps=0 corr {fit.get('base_acf_corr')}")
    tr = data.mmap_items("artifacts/cache/v8", "train", Path("artifacts/mm/v8"))
    attach_sigma(tr)
    holder = {"on": False, "rng": np.random.default_rng(0), "shaper": shaper, "spec": scaled_spec(1.0, 1.0)}
    sampler = make_sampler(tr, holder)
    _, init_sd = load_init()
    cw = torch.tensor([1.0, 3.0, 3.0])
    t0 = time.time()
    deadline = t0 + TRAIN_BUDGET_S
    rows = []
    for i, (name, scale, frac) in enumerate(GRID):
        left_cfg = len(GRID) - i
        now = time.time()
        if now >= deadline - 45:
            log(f"бюджет кончился, {name} пропущен")
            break
        slice_end = now + (deadline - now) / left_cfg
        log(f"старт {name} scale {scale} batch_frac {frac} срез {slice_end - now:.0f} с")
        model = models.build("unet", 16)
        model.load_state_dict(init_sd)
        freeze_bn(model)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
        holder["spec"] = scaled_spec(scale, frac)
        holder["rng"] = np.random.default_rng(4000 + i)
        coin = np.random.default_rng(3000 + i)
        sampler.rng = np.random.default_rng(0)
        best = -1.0
        best_epoch = None
        hist = []
        n_on = n_bat = 0
        bad = 0
        ckpt = Path(f"artifacts/eval/s3_domsweep_{name}.pt")
        stopped = "epochs"
        for ep in range(MAX_EPOCHS):
            if time.time() >= slice_end:
                stopped = "slice"
                break
            bn_train_frozen(model)
            tl = []
            ep_steps = 0
            t_ep = time.time()
            for _ in range(STEPS):
                if time.time() >= slice_end and ep_steps > 0:
                    stopped = "slice"
                    break
                holder["on"] = bool(coin.random() < frac)
                n_bat += 1
                n_on += int(holder["on"])
                x, y = sampler.sample(BATCH)
                logit = model(torch.from_numpy(x))
                loss = losses.compute("focal+tversky_ce", logit, torch.from_numpy(y), cw)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tl.append(float(loss.item()))
                ep_steps += 1
            if ep_steps == 0:
                stopped = "slice"
                break
            pi, pt = proxy_blend(model, proxy_files)
            rec = dict(epoch=ep, steps=ep_steps, train=float(np.mean(tl)),
                       proxy_blend_iou=pi, proxy_thr=pt,
                       sec=round(time.time() - t0, 1), epoch_sec=round(time.time() - t_ep, 1))
            hist.append(rec)
            log(f"{name} {rec}")
            if pi > best + 1e-4:
                best = pi
                best_epoch = ep
                bad = 0
                tmp = ckpt.with_suffix(".pt.tmp")
                torch.save(model.state_dict(), tmp)
                os.replace(tmp, ckpt)
            else:
                bad += 1
                if bad >= 2:
                    stopped = "patience"
                    break
        row = dict(
            name=name,
            strength_scale=scale,
            batch_frac=frac,
            aug=holder["spec"].as_dict(),
            batch_frac_realized=float(n_on / max(n_bat, 1)),
            n_batches=int(n_bat),
            history=hist,
            best_proxy_blend_iou=None if best < 0 else float(best),
            best_epoch=best_epoch,
            stopped=stopped,
            ckpt=str(ckpt) if best_epoch is not None else None,
            init=str(INIT),
        )
        rows.append(row)
        dump(OUT_TRAIN, dict(
            name="DOMSWEEP",
            budget_s=TRAIN_BUDGET_S,
            steps_per_epoch=STEPS,
            max_epochs=MAX_EPOCHS,
            batch=BATCH,
            tile=list(TILE),
            lr=LR,
            loss="focal+tversky_ce",
            bn="frozen_running_stats_and_affine",
            init=str(INIT),
            dom_reference=dict(std_lo=DOM_STD[0], std_hi=DOM_STD[1], tile_p_apply=0.7,
                               ens_blend_val_iou=DOM_BLEND_VAL),
            shaper=fit,
            psd=bank.as_meta(),
            proxy_files=[p.name for p in proxy_files],
            proxy_rule="max ens_blend IoU@FAR1e-3 on the first 4 v8 val files; official score is all 45",
            configs=rows,
            train_seconds=round(time.time() - t0, 1),
            real_traces_used=False,
        ))
        del model, opt
    log(f"обучение {time.time() - t0:.0f} с, конфигов {len(rows)}")
    return rows


def _agg(df):
    if df is None or len(df) == 0:
        return dict(n=0, recall=None, pixel_recall=None)
    tp = int(df.tp.sum())
    fn = int(df.fn.sum())
    return dict(n=int(len(df)), recall=float(df.hit.mean()),
                pixel_recall=float(tp / max(tp + fn, 1)))


def score_maps(model, split: str, thr: dict | None, labels, seqs) -> dict:
    """Предсказание одного U-Net и метрики unet / ens_blend. Карты во временном каталоге."""
    tmp = Path(f"/tmp/s3_domsweep_{split}_{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    files = data.list_cache("artifacts/cache/v8", split)
    t0 = time.time()
    try:
        for i, p in enumerate(files, 1):
            with np.load(p) as z:
                X = np.asarray(z["X"], np.float32)
            prob = predict_batched(model, X)
            s = (1.0 - prob[0]).astype(np.float32)
            h = (prob[1] - prob[2]).astype(np.float16)
            np.save(tmp / f"{p.stem}_s.npy", s)
            np.save(tmp / f"{p.stem}_h.npy", h)
            del X, prob, s, h
            if i == 1 or i % 15 == 0 or i == len(files):
                log(f"pred {split} {i}/{len(files)} {time.time() - t0:.0f}s")
        names = ("unet", "ens_blend")
        if thr is None:
            bags = {n: [] for n in names}
            for p in files:
                s = np.load(tmp / f"{p.stem}_s.npy")
                with np.load(p) as z:
                    mask = bg_mask(np.asarray(z["ev"]), np.asarray(z["nu"]), np.asarray(z["valid"], bool))
                with np.load(ENS / split / p.name) as pred:
                    ens = np.asarray(pred["s"], np.float32)
                if ens.shape != s.shape:
                    raise SystemExit(f"{p.name}: ens {ens.shape} != unet {s.shape}")
                blend = 0.5 * (s + ens)
                bags["unet"].append(np.asarray(s[mask], np.float32))
                bags["ens_blend"].append(np.asarray(blend[mask], np.float32))
                del s, ens, blend
            thr = {}
            for n in names:
                bg = np.concatenate(bags[n])
                thr[n] = float(M.far_threshold(bg, 1e-3))
                log(f"порог {split} {n} {thr[n]:.6f} n_bg {bg.size}")
                del bg
            del bags
        tot = {n: np.zeros(3, np.int64) for n in names}
        hot = {n: np.zeros(3, np.int64) for n in names}
        cold = {n: np.zeros(3, np.int64) for n in names}
        far_n = {n: 0 for n in names}
        far_d = {n: 0 for n in names}
        per = {n: {} for n in names}
        ev_rows = {n: [] for n in names}
        for p in files:
            with np.load(p) as z:
                ev = np.asarray(z["ev"])
                valid = np.asarray(z["valid"], bool)
                nu = np.asarray(z["nu"])
                x = np.asarray(z["x"])
                t = np.asarray(z["t"])
            s = np.load(tmp / f"{p.stem}_s.npy")
            h_u = np.load(tmp / f"{p.stem}_h.npy").astype(np.float32)
            with np.load(ENS / split / p.name) as pred:
                ens_s = np.asarray(pred["s"], np.float32)
                ens_h = np.asarray(pred["h"], np.float32)
            packed = {
                "unet": (s, h_u),
                "ens_blend": ((0.5 * (s + ens_s)).astype(np.float32), (0.5 * (h_u + ens_h)).astype(np.float32)),
            }
            truth = ev > 0
            hot_t = (ev >= 1) & (ev <= 4)
            cold_t = ev >= 5
            mask = bg_mask(ev, nu, valid)
            boxes = []
            for r in M.label_boxes(labels, p.stem).itertuples():
                if r.kind != "event":
                    continue
                px = M.event_pixels(r, x, t, ev)
                if px.any():
                    boxes.append((r, px))
            for n in names:
                sc, h = packed[n]
                pred = (sc >= thr[n]) & valid
                tot[n] += np.asarray(M.confusion(pred, truth, valid), np.int64)
                per[n][p.stem] = M.confusion(pred, truth, valid)
                hot[n] += np.asarray(M.confusion(pred & (h > 0), hot_t, valid), np.int64)
                cold[n] += np.asarray(M.confusion(pred & (h <= 0), cold_t, valid), np.int64)
                far_n[n] += int((pred & mask).sum())
                far_d[n] += int(mask.sum())
                for r, px in boxes:
                    frac = float(pred[px].mean())
                    ev_rows[n].append(dict(
                        label=str(r.label), amp=float(r.amp_obs_C), amp_bin=M.amp_bin(float(r.amp_obs_C)),
                        frac=frac, hit=bool(frac >= 0.3),
                        tp=int((pred & px).sum()), fn=int((~pred & px).sum()),
                    ))
            del s, h_u, ens_s, ens_h, packed
        report = {}
        for n in names:
            E = pd.DataFrame(ev_rows[n])
            row = dict(
                threshold=float(thr[n]),
                far=float(far_n[n] / max(far_d[n], 1)),
                **M.prf(*map(int, tot[n])),
                hot=M.prf(*map(int, hot[n])),
                cold=M.prf(*map(int, cold[n])),
                iou_ci95=M.bootstrap_ci(per[n], sorted(per[n]), n=200, seed=0),
            )
            if len(E):
                row["events_all"] = _agg(E)
                row["weak_events"] = _agg(E[E.amp <= 2]) if (E.amp <= 2).any() else dict(n=0, recall=None, pixel_recall=None)
                row["by_amp"] = {str(k): _agg(g) for k, g in E.groupby("amp_bin")}
            report[n] = row
            log(f"{split} {n} IoU {row['iou']:.4f} F1 {row['f1']:.4f} FAR {row['far']:.5f}")
        return report
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def evaluate(rows: list[dict], sanity_rep: dict) -> dict:
    labels = pd.read_csv("data/synth_meta/v8/labels.csv")
    seqs = pd.read_csv("data/synth_meta/v8/sequences.csv").set_index("seq_id")
    usable = [r for r in rows if r.get("ckpt") and Path(r["ckpt"]).exists()]
    if not usable:
        raise SystemExit("нет чекпойнта для оценки")
    # официально сравниваем лучший чекпойнт каждого конфига
    val_rows = {}
    models_live = {}
    for r in usable:
        model = models.build("unet", 16)
        model.load_state_dict(torch.load(r["ckpt"], map_location="cpu", weights_only=True))
        model.eval()
        freeze_bn(model)
        log(f"val {r['name']}")
        val_rows[r["name"]] = score_maps(model, "val", None, labels, seqs)
        models_live[r["name"]] = model
    selected = max(usable, key=lambda r: val_rows[r["name"]]["ens_blend"]["iou"])
    name = selected["name"]
    thr = {k: val_rows[name][k]["threshold"] for k in ("unet", "ens_blend")}
    log(f"test {name}")
    test_row = score_maps(models_live[name], "test", thr, labels, seqs)
    for m in models_live.values():
        del m
    sel_val = val_rows[name]["ens_blend"]["iou"]
    sel_test = test_row["ens_blend"]["iou"]
    beats = bool(sanity_rep["protocol_ok"] and sel_val > DOM_BLEND_VAL)
    report = dict(
        name="DOMSWEEP",
        protocol="v8 val IoU at FAR 1e-3 of ens_blend=mean(ens_u2ft_tcn_lgbm, swept U-Net); threshold from v8 val background; test not used for selection",
        reference_dom_ens_blend_val=DOM_BLEND_VAL,
        reference_note="DOM ens_blend val is the 4 d.p. figure 0.3898 from docs/S3_DOM.md. Beat requires a strictly larger full-val IoU.",
        published_winner=dict(name=WINNER, val_iou=WINNER_VAL, test_iou=WINNER_TEST),
        sanity_winner=sanity_rep,
        grid=[dict(name=n, strength_scale=s, batch_frac=f,
                   std_lo=DOM_STD[0] * s, std_hi=DOM_STD[1] * s) for n, s, f in GRID],
        aug_definition="strength multiplies DOM Uniform(0.25, 0.70)*sigma; batch_frac is the probability a whole batch is noised with p_apply=1. DOM itself used per-tile p_apply=0.7 and is not retrained.",
        val={k: val_rows[k] for k in val_rows},
        selected=name,
        selected_strength_scale=selected["strength_scale"],
        selected_batch_frac=selected["batch_frac"],
        selected_val_iou=sel_val,
        selected_val_unet_iou=val_rows[name]["unet"]["iou"],
        selected_test=test_row,
        selected_test_iou=sel_test,
        threshold_ens_blend=thr["ens_blend"],
        protocol_ok=bool(sanity_rep["protocol_ok"]),
        beats_dom_blend=beats,
        real_used_for_training=False,
        real_used_for_threshold=False,
        cards="not used",
    )
    return report


def continue_all(minutes: float) -> list[dict]:
    """Добор бюджета: те же четыре клетки, старт с уже выбранного чекпойнта."""
    blob = json.loads(OUT_TRAIN.read_text())
    spent = float(blob.get("train_seconds") or 0)
    room = TRAIN_BUDGET_S - spent
    budget = min(float(minutes) * 60, max(room, 0))
    if budget < 60:
        log(f"добор не нужен, остаток бюджета {room:.0f} с")
        return blob["configs"]
    rows = blob["configs"]
    bank = aug_real.load_bank()
    shaper, fit = aug_real.fit_shaper(bank, steps=0, seed=1, tile=(160, 256), max_lag=16)
    tr = data.mmap_items("artifacts/cache/v8", "train", Path("artifacts/mm/v8"))
    attach_sigma(tr)
    holder = {"on": False, "rng": np.random.default_rng(1), "shaper": shaper, "spec": scaled_spec(1.0, 1.0)}
    sampler = make_sampler(tr, holder)
    proxy_files = [Path("artifacts/cache/v8/val") / n for n in blob["proxy_files"]]
    cw = torch.tensor([1.0, 3.0, 3.0])
    t0 = time.time()
    deadline = t0 + budget
    pending = [r for r in rows if r.get("ckpt") and Path(r["ckpt"]).exists()]
    log(f"добор {budget:.0f} с, конфигов {len(pending)}")
    for i, row in enumerate(pending):
        left = len(pending) - i
        now = time.time()
        if now >= deadline - 40:
            log(f"добор: время вышло, {row['name']} без новых эпох")
            break
        slice_end = now + (deadline - now) / left
        name = row["name"]
        scale, frac = float(row["strength_scale"]), float(row["batch_frac"])
        model = models.build("unet", 16)
        model.load_state_dict(torch.load(row["ckpt"], map_location="cpu", weights_only=True))
        freeze_bn(model)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
        holder["spec"] = scaled_spec(scale, frac)
        holder["rng"] = np.random.default_rng(8000 + i)
        coin = np.random.default_rng(7000 + i)
        sampler.rng = np.random.default_rng(100 + i)
        best = float(row["best_proxy_blend_iou"] or -1)
        best_epoch = row["best_epoch"]
        hist = list(row["history"])
        n_on = n_bat = 0
        ep0 = (hist[-1]["epoch"] + 1) if hist else 0
        ckpt = Path(row["ckpt"])
        log(f"добор {name} с эпохи {ep0}, срез {slice_end - now:.0f} с, proxy {best:.4f}")
        for ep in range(ep0, ep0 + 8):
            if time.time() >= slice_end:
                break
            bn_train_frozen(model)
            tl, ep_steps = [], 0
            t_ep = time.time()
            for _ in range(STEPS):
                if time.time() >= slice_end and ep_steps > 0:
                    break
                holder["on"] = bool(coin.random() < frac)
                n_bat += 1
                n_on += int(holder["on"])
                x, y = sampler.sample(BATCH)
                logit = model(torch.from_numpy(x))
                loss = losses.compute("focal+tversky_ce", logit, torch.from_numpy(y), cw)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tl.append(float(loss.item()))
                ep_steps += 1
            if ep_steps == 0:
                break
            pi, pt = proxy_blend(model, proxy_files)
            rec = dict(epoch=ep, steps=ep_steps, train=float(np.mean(tl)),
                       proxy_blend_iou=pi, proxy_thr=pt,
                       sec=round(spent + time.time() - t0, 1), epoch_sec=round(time.time() - t_ep, 1),
                       stage="continue")
            hist.append(rec)
            log(f"{name} {rec}")
            if pi > best + 1e-4:
                best = pi
                best_epoch = ep
                tmp = ckpt.with_suffix(".pt.tmp")
                torch.save(model.state_dict(), tmp)
                os.replace(tmp, ckpt)
        row["history"] = hist
        row["best_proxy_blend_iou"] = float(best)
        row["best_epoch"] = best_epoch
        row["continue_batches"] = int(n_bat)
        row["continue_batch_frac_realized"] = float(n_on / max(n_bat, 1))
        del model, opt
        blob["configs"] = rows
        blob["train_seconds"] = round(spent + time.time() - t0, 1)
        blob["continue_seconds"] = round(time.time() - t0, 1)
        blob["shaper_continue"] = fit
        dump(OUT_TRAIN, blob)
    log(f"добор готов {time.time() - t0:.0f} с")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity-only", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--continue-min", type=float, default=0.0)
    args = ap.parse_args()
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    _self_check()
    t0 = time.time()
    sanity_rep = sanity()
    if args.sanity_only:
        dump(Path("artifacts/eval/s3_domsweep_sanity.json"), sanity_rep)
        log("sanity only")
        return
    if args.continue_min > 0:
        rows = continue_all(args.continue_min)
    elif args.eval_only:
        blob = json.loads(OUT_TRAIN.read_text())
        rows = blob["configs"]
    else:
        proxy_files = data.list_cache("artifacts/cache/v8", "val")[:4]
        rows = train_all(proxy_files)
    report = evaluate(rows, sanity_rep)
    report["seconds"] = round(time.time() - t0, 1)
    dump(OUT_EVAL, report)
    log(f"selected {report['selected']} val {report['selected_val_iou']:.4f} "
        f"test {report['selected_test_iou']:.4f} beats {report['beats_dom_blend']}")
    log(f"записан {OUT_EVAL}")


if __name__ == "__main__":
    main()
