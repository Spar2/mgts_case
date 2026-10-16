"""Обучение маскированного реконструктора фона. Только синтетика v8+v10 train.

Чекпойнт artifacts/ckpt/<run>/last.pt (веса, оптимизатор, шаг, RNG, лучший прокси)
пишется не реже чем раз в 15 минут. В конце — touch DONE.
Запуск только так:
  scripts/run_resumable.sh s3_mae /workspace/repo/run/.venv/bin/python -u scripts/s3_mae_train.py
Дополнительные сиды — тот же рецепт, другие --seed/--run/--out, чтобы не затереть s3_mae.
torch.manual_seed вызывается до MaskedReconstructor: иначе все сиды делят одну init.
Рядом с --out пишутся init-веса (*_init.pt) и лучшая эпоха epoch>=0 (сам --out),
даже если прокси так и не обогнал init. Опубликованные s3_mae_recon.pt и
s3_mae_train.json этот скрипт не перезаписывает.

Реальные CSV, карточки и спектр шума не читаются.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "NUMEXPR_MAX_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_k] = "2"

import argparse
import json
import random
import time
import zlib
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
import sys
sys.path.insert(0, str(ROOT))

from dts import data
from dts.mae_anom import (
    BAND, RADIUS, TILE, MaskedReconstructor, channel_width, residuals, sample_train_mask,
    spike_probe, training_loss,
)

RUN = "s3_mae"
CKPT_DIR = Path("artifacts/ckpt") / RUN
LAST = CKPT_DIR / "last.pt"
BEST = Path("artifacts/eval/s3_mae_recon.pt")
TRAIN_JSON = Path("artifacts/eval/s3_mae_train.json")
LOG = Path("/workspace/grok_logs/_events.log")
TAG = "MAE_TRAIN"
SEED = 20261015
SAVE_EVERY = 15 * 60
TILE_H, TILE_W = TILE


def log(msg: str) -> None:
    line = time.strftime("%H:%M", time.gmtime()) + " " + msg
    print(line, flush=True)


def elog(msg: str) -> None:
    log(msg)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(time.strftime("%H:%M", time.gmtime()) + " " + msg + "\n")


def plain(obj):
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(plain(obj), indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def crc(text: str) -> int:
    return zlib.crc32(text.encode()) & 0xFFFFFFFF


class TileSource:
    def __init__(self):
        self.mm = {}
        self.lru = OrderedDict()

    def _npz(self, path: str):
        if path not in self.lru:
            with np.load(path) as z:
                self.lru[path] = (
                    np.asarray(z["X"][:2]),
                    np.asarray(z["valid"]),
                    np.asarray(z["ev"]),
                )
            while len(self.lru) > 3:
                self.lru.popitem(last=False)
        else:
            self.lru.move_to_end(path)
        return self.lru[path]

    def get(self, win, with_ev: bool = False):
        path, mmap, t0, w0 = win
        th, tw = TILE_H, TILE_W
        if mmap:
            if mmap not in self.mm:
                self.mm[mmap] = np.load(mmap, mmap_mode="r")
            X = self.mm[mmap]
            ch = np.array(X[0:2, t0:t0 + th, w0:w0 + tw], dtype=np.float32, copy=True)
            valid = np.array(X[2, t0:t0 + th, w0:w0 + tw], dtype=np.float32, copy=True)
            ev = np.zeros(valid.shape, np.uint8) if with_ev else None
        else:
            X, valid_full, ev_full = self._npz(path)
            ch = np.array(X[:, t0:t0 + th, w0:w0 + tw], dtype=np.float32, copy=True)
            valid = np.array(valid_full[t0:t0 + th, w0:w0 + tw], dtype=np.float32, copy=True)
            ev = np.array(ev_full[t0:t0 + th, w0:w0 + tw], copy=True) if with_ev else None
        ch = np.nan_to_num(ch, nan=0.0, posinf=0.0, neginf=0.0)
        ch = np.clip(ch, -12.0, 12.0)
        valid = (valid > 0.5).astype(np.float32)
        if with_ev:
            return ch, valid, ev
        return ch, valid


def _collect(nu_limit: float) -> tuple[list, list, list]:
    train, hold, events = [], [], []
    th, tw = TILE_H, TILE_W
    step_t, step_w = th // 2, tw // 2
    for ver, use_mm in (("v8", True), ("v10", False)):
        files = data.list_cache(f"artifacts/cache/{ver}", "train")
        if len(files) < 100:
            raise SystemExit(f"мало {ver} train: {len(files)}")
        t0c = time.time()
        n_keep = 0
        for i, p in enumerate(files, 1):
            with np.load(p) as z:
                ev = np.asarray(z["ev"])
                nu = np.asarray(z["nu"])
                valid = np.asarray(z["valid"])
            mmap = ""
            if use_mm:
                cand = Path(f"artifacts/mm/v8/train/{p.stem}_X.npy")
                if not cand.exists():
                    raise SystemExit(f"нет mmap {cand}")
                mmap = str(cand)
            T, W = ev.shape
            if T < th or W < tw:
                continue
            for t0 in range(0, T - th + 1, step_t):
                for w0 in range(0, W - tw + 1, step_w):
                    evs = ev[t0:t0 + th, w0:w0 + tw]
                    nus = nu[t0:t0 + th, w0:w0 + tw]
                    vas = valid[t0:t0 + th, w0:w0 + tw]
                    if float(vas.mean()) < 0.8:
                        continue
                    win = (str(p), mmap if int(evs.max()) == 0 else "", t0, w0)
                    ev_max = int(evs.max())
                    if ev_max == 0 and float(nus.mean()) <= nu_limit:
                        tag = crc(f"{p}:{t0}:{w0}")
                        if tag % 100 < 12:
                            hold.append((str(p), mmap, t0, w0))
                        else:
                            train.append((str(p), mmap, t0, w0))
                        n_keep += 1
                    elif (
                        len(events) < 48
                        and ev_max > 0
                        and 0.002 <= float(evs.mean()) <= 0.08
                        and float(nus.mean()) <= max(nu_limit, 0.02)
                    ):
                        events.append((str(p), "", t0, w0))
            if i == 1 or i % 40 == 0 or i == len(files):
                log(f"индекс {ver} nu<={nu_limit} {i}/{len(files)} +{n_keep} {time.time() - t0c:.0f}s")
    return train, hold, events


def index_windows() -> tuple[list, list, list, float]:
    train, hold, events = _collect(0.0)
    used = 0.0
    if len(train) < 200 or len(hold) < 16:
        log(f"чистых тайлов мало train {len(train)} hold {len(hold)}, допуск nu 1%")
        train, hold, events = _collect(0.01)
        used = 0.01
    if len(train) < 200 or len(hold) < 8:
        raise SystemExit(f"мало тайлов фона train {len(train)} hold {len(hold)}")
    log(f"тайлы train {len(train)} hold {len(hold)} event_proxy {len(events)} nu_limit {used}")
    return train, hold, events, used


def sample_batch(rng: np.random.Generator, wins, src: TileSource, n: int):
    chs, vs, ms = [], [], []
    th, tw = TILE_H, TILE_W
    for _ in range(n):
        ch, valid = src.get(wins[int(rng.integers(0, len(wins)))])
        if rng.random() < 0.5:
            ch = ch[:, :, ::-1].copy()
            valid = valid[:, ::-1].copy()
        if rng.random() < 0.5:
            ch = ch[:, ::-1].copy()
            valid = valid[::-1].copy()
        chs.append(ch)
        vs.append(valid)
        ms.append(sample_train_mask(th, tw, rng))
    return (
        torch.from_numpy(np.stack(chs)),
        torch.from_numpy(np.stack(vs)[:, None]),
        torch.from_numpy(np.stack(ms)[:, None]),
    )


def rng_blob(rng: np.random.Generator) -> dict:
    return {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
        "sample": rng.bit_generator.state,
    }


def rng_restore(blob, rng: np.random.Generator) -> None:
    torch.set_rng_state(blob["torch"])
    np.random.set_state(blob["numpy"])
    random.setstate(blob["python"])
    rng.bit_generator.state = blob["sample"]


def clone_sd(model) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def conv0_sum(model) -> float:
    conv = model.e1[0][0]
    return float(conv.weight.detach().sum())


def head_abs_max(model) -> float:
    return float(max(
        model.head.weight.detach().abs().max().item(),
        model.head.bias.detach().abs().max().item(),
    ))


def apply_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    random.seed(int(seed))


def save_weights(path: Path, state) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sd = state if isinstance(state, dict) else state.state_dict()
    tmp = path.with_suffix(".pt.tmp")
    torch.save(sd, tmp)
    os.replace(tmp, path)


def save_last(model, opt, step, epoch, rng, best, history, elapsed, steps_per_epoch, nu_limit, extra) -> None:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": None,
        "step": int(step),
        "epoch": int(epoch),
        "rng": rng_blob(rng),
        "best": best,
        "history": history,
        "elapsed": float(elapsed),
        "steps_per_epoch": int(steps_per_epoch),
        "nu_limit": float(nu_limit),
        "seed": int(SEED),
        "band": BAND,
        "radius": RADIUS,
        "width": int(model.width),
        "seeded_before_model": True,
    }
    payload.update(extra)
    tmp = LAST.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, LAST)


def proxy_metrics(model, hold, events, src: TileSource) -> dict:
    """Доля пикселей события выше квантиля фона 1-1e-3. Это ближе к IoU@FAR, чем медиана."""
    model.eval()
    bg_chunks, ev_chunks = [], []
    ref_abs, restores = [], []
    kw = dict(tile=TILE, overlap=0, band=BAND, radius=RADIUS, tile_batch=1)
    for w in hold[:6]:
        ch, valid = src.get(w)
        _r0, r1 = residuals(model, ch, valid, **kw)
        m = valid > 0.5
        bg_chunks.append(np.abs(r1[m]).ravel())
        ref_abs.append(float(np.mean(np.abs(ch[1][m]))))
    for w in hold[:2]:
        ch, valid = src.get(w)
        restores.append(float(spike_probe(model, ch, valid, tile=TILE, overlap=0)["restore"]))
    seps = []
    for w in events[:6]:
        ch, valid, ev = src.get(w, with_ev=True)
        _r0, r1 = residuals(model, ch, valid, **kw)
        bg = (ev == 0) & (valid > 0.5)
        evp = (ev > 0) & (valid > 0.5)
        if int(bg.sum()) >= 20:
            bg_chunks.append(np.abs(r1[bg]).ravel())
        if int(evp.sum()) < 20:
            continue
        ev_abs = np.abs(r1[evp])
        ev_chunks.append(ev_abs.ravel())
        if int(bg.sum()) >= 20:
            seps.append(float(np.median(ev_abs) / max(float(np.median(np.abs(r1[bg]))), 1e-6)))
    model.train()
    bg_cat = np.concatenate(bg_chunks) if bg_chunks else np.ones(8, np.float32)
    ev_cat = np.concatenate(ev_chunks) if ev_chunks else np.zeros(8, np.float32)
    if bg_cat.size > 250_000:
        bg_cat = bg_cat[:: max(1, bg_cat.size // 250_000)]
    thr = float(np.quantile(bg_cat, 1.0 - 1e-3))
    recall = float(np.mean(ev_cat >= thr)) if ev_cat.size else 0.0
    bg = float(np.mean(bg_cat))
    sep = float(np.mean(seps)) if seps else 1.0
    restore = float(np.mean(restores)) if restores else 0.0
    ref = float(np.mean(ref_abs)) if ref_abs else 0.0
    # Хвост фона важнее медианы: сырой |dT| уже хорошо разделён по медиане и плох на FAR 1e-3.
    score = float(recall - 0.002 * max(0.0, bg - ref))
    return {
        "bg_abs": bg, "ref_abs": ref, "sep": sep, "restore": restore, "score": score,
        "recall": recall, "proxy_thr": thr, "n_sep": len(seps),
    }


def fit_cdf(model, hold, src: TileSource, n: int = 12, cap: int = 200_000) -> np.ndarray:
    model.eval()
    parts = []
    kw = dict(tile=TILE, overlap=0, band=BAND, radius=RADIUS, tile_batch=1)
    for w in hold[:n]:
        ch, valid = src.get(w)
        _r0, r1 = residuals(model, ch, valid, **kw)
        m = valid > 0.5
        parts.append(np.abs(r1[m]).ravel())
    cat = np.concatenate(parts) if parts else np.zeros(8, np.float32)
    if cat.size > cap:
        cat = np.random.default_rng(0).choice(cat, cap, replace=False)
    knots = np.quantile(cat, np.linspace(0.0, 1.0, 257)).astype(np.float64)
    for i in range(1, knots.size):
        if knots[i] <= knots[i - 1]:
            knots[i] = knots[i - 1] + 1e-6
    model.train()
    return knots


def _resolved(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def refuse_published(path: Path) -> None:
    published = {
        _resolved(Path("artifacts/eval/s3_mae_recon.pt")),
        _resolved(Path("artifacts/eval/s3_mae_train.json")),
    }
    if _resolved(path) in published:
        raise SystemExit(f"отказ: не перезаписываю опубликованный {path}")


def main() -> None:
    global RUN, CKPT_DIR, LAST, BEST, TRAIN_JSON, TAG, SEED
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", default="auto", choices=("auto", "never", "must"))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--run", default="s3_mae")
    ap.add_argument("--out", default="")
    ap.add_argument("--log-json", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--init-from", default="", help="загрузить эти веса как init, не перезаписывая источник")
    args = ap.parse_args()
    RUN = str(args.run)
    CKPT_DIR = Path("artifacts/ckpt") / RUN
    LAST = CKPT_DIR / "last.pt"
    BEST = Path(args.out) if args.out else Path("artifacts/eval/s3_mae_recon.pt")
    TRAIN_JSON = Path(args.log_json) if args.log_json else Path("artifacts/eval/s3_mae_train.json")
    INIT_PT = BEST.with_name(BEST.stem + "_init.pt")
    TAG = str(args.tag) if args.tag else "MAE_TRAIN"
    SEED = int(args.seed) if args.seed is not None else int(os.environ.get("MAE_SEED", "20261015"))
    for path in (BEST, INIT_PT, TRAIN_JSON):
        refuse_published(path)
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    max_minutes = float(os.environ.get("MAE_MAX_MINUTES", "40"))
    patience = int(os.environ.get("MAE_PATIENCE", "6"))
    min_epochs = int(os.environ.get("MAE_MIN_EPOCHS", "8"))
    steps_default = int(os.environ.get("MAE_STEPS", "80"))
    lr = float(os.environ.get("MAE_LR", "1e-3"))
    batch = int(os.environ.get("MAE_BATCH", "8"))
    seed = int(SEED)

    train_w, hold_w, event_w, nu_limit = index_windows()
    # Сид до конструктора. Иначе MaskedReconstructor берёт один и тот же поток и все сиды совпадают.
    apply_seed(seed)
    model = MaskedReconstructor(16)
    if channel_width(model) > 16:
        raise SystemExit("ширина > 16")
    n_param = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    rng = np.random.default_rng(seed)
    step = 0
    start_ep = 0
    history = []
    best = {"score": -1e9, "epoch": -1, "bg_abs": None, "sep": None, "restore": None}
    best_trained = {"score": -1e9, "epoch": -1, "bg_abs": None, "sep": None, "restore": None}
    elapsed0 = 0.0
    bad = 0
    steps_per = steps_default
    init_sd = None
    trained_sd = None
    init_probe = None
    init_from = str(args.init_from or "")
    if args.resume != "never" and LAST.exists():
        ckpt = torch.load(LAST, map_location="cpu", weights_only=False)
        if not ckpt.get("seeded_before_model"):
            raise SystemExit("resume: чекпойнт собран до сида модели, одинаковую init не продолжаю")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        step = int(ckpt["step"])
        start_ep = int(ckpt["epoch"]) + 1
        history = list(ckpt.get("history") or [])
        best = dict(ckpt["best"])
        best_trained = dict(ckpt.get("best_trained") or best_trained)
        elapsed0 = float(ckpt.get("elapsed") or 0.0)
        steps_per = int(ckpt.get("steps_per_epoch") or steps_default)
        nu_limit = float(ckpt.get("nu_limit") or nu_limit)
        if "seed" in ckpt and int(ckpt["seed"]) != seed:
            raise SystemExit(f"resume: seed {ckpt['seed']} != {seed}")
        init_sd = ckpt.get("init_model")
        trained_sd = ckpt.get("trained_model")
        init_probe = ckpt.get("init_probe")
        init_from = str(ckpt.get("init_from") or init_from)
        if not isinstance(init_sd, dict):
            raise SystemExit("resume: нет init_model")
        rng_restore(ckpt["rng"], rng)
        bad = sum(1 for h in history if int(h["epoch"]) > int(best["epoch"]))
        elog(
            f"{TAG}_RESUME epoch={ckpt['epoch']} step={step} seed={seed} "
            f"best={best.get('score')} trained_epoch={best_trained.get('epoch')} "
            f"elapsed={elapsed0:.0f}s run={RUN}"
        )
    elif args.resume == "must":
        raise SystemExit("resume must: нет last.pt")
    else:
        if init_from:
            src_pt = Path(init_from)
            if _resolved(src_pt) == _resolved(BEST) or _resolved(src_pt) == _resolved(INIT_PT):
                raise SystemExit("init-from совпадает с выходом")
            model.load_state_dict(torch.load(src_pt, map_location="cpu", weights_only=True))
            # Опубликованный прогон сеял torch уже после конструктора. Повторяем это после загрузки,
            # чтобы шум обучения шёл с того же места, а загруженные веса не заменялись новой init.
            apply_seed(seed)
            rng = np.random.default_rng(seed)
        init_sd = clone_sd(model)
        save_weights(INIT_PT, init_sd)
        elog(
            f"{TAG}_START params={n_param} width=16 band={BAND} threads=2 "
            f"batch={batch} nu_limit={nu_limit} seed={seed} run={RUN} "
            f"conv0={conv0_sum(model):.8f} head_max={head_abs_max(model):.6g}"
        )

    held = {
        "init_model": init_sd,
        "trained_model": trained_sd,
        "best_trained": best_trained,
        "init_probe": init_probe,
        "init_from": init_from,
    }

    def persist(ep_saved: int) -> None:
        held["trained_model"] = trained_sd
        held["best_trained"] = best_trained
        held["init_probe"] = init_probe
        save_last(
            model, opt, step, ep_saved, rng, best, history, spent(), steps_per, nu_limit, held,
        )

    # Прокси init. Голова архитектуры нулевая, поэтому скор init не видит свёртки.
    # В --out init не пишем: туда попадает только лучшая эпоха epoch>=0.
    if start_ep == 0 and int(best.get("epoch", -999)) < 0 and not history:
        pr0 = proxy_metrics(model, hold_w, event_w, TileSource())
        init_probe = dict(pr0)
        best = dict(score=pr0["score"], epoch=-1, bg_abs=pr0["bg_abs"], sep=pr0["sep"], restore=pr0["restore"])
        held["init_probe"] = init_probe
        elog(
            f"{TAG}_INIT score={pr0['score']:.4f} sep={pr0['sep']:.3f} "
            f"bg={pr0['bg_abs']:.3f} ref={pr0['ref_abs']:.3f} seed={seed} "
            f"conv0={conv0_sum(model):.8f} head_max={head_abs_max(model):.6g}"
        )

    src = TileSource()
    t0 = time.time()

    def spent() -> float:
        return elapsed0 + (time.time() - t0)

    last_save = time.time()
    stopped = "epochs"
    model.train()
    epoch = start_ep - 1
    for epoch in range(start_ep, start_ep + 80):
        if spent() > max_minutes * 60:
            stopped = "time"
            break
        losses = []
        for _ in range(steps_per):
            if spent() > max_minutes * 60:
                stopped = "time"
                break
            xb, vb, mb = sample_batch(rng, train_w, src, batch)
            opt.zero_grad(set_to_none=True)
            loss = training_loss(model, xb, vb, mb)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            losses.append(float(loss.detach().item()))
            if time.time() - last_save >= SAVE_EVERY:
                persist(epoch - 1 if not losses else epoch)
                last_save = time.time()
                log(f"last.pt step {step}")
        if not losses:
            stopped = "time"
            break
        pr = proxy_metrics(model, hold_w, event_w, src)
        rec = dict(epoch=epoch, train=float(np.mean(losses)), step=step, sec=round(spent(), 1), **pr)
        history.append(rec)
        elog(
            f"{TAG}_TRAIN epoch={epoch} loss={rec['train']:.4f} score={pr['score']:.4f} "
            f"sep={pr['sep']:.3f} bg={pr['bg_abs']:.3f} restore={pr['restore']:.3f} sec={rec['sec']:.0f}"
        )
        if pr["score"] > float(best["score"]) + 1e-3:
            best = dict(score=pr["score"], epoch=epoch, bg_abs=pr["bg_abs"], sep=pr["sep"], restore=pr["restore"])
            bad = 0
            log(f"прокси лучше init, epoch {epoch} score {pr['score']:.4f}")
        else:
            bad += 1
        if int(best_trained["epoch"]) < 0 or pr["score"] > float(best_trained["score"]):
            best_trained = dict(
                score=pr["score"], epoch=int(epoch), bg_abs=pr["bg_abs"],
                sep=pr["sep"], restore=pr["restore"], head_max=head_abs_max(model),
            )
            trained_sd = clone_sd(model)
            save_weights(BEST, trained_sd)
            log(f"сохранена обученная эпоха {epoch} score {pr['score']:.4f}")
        persist(epoch)
        last_save = time.time()
        if bad >= patience and epoch >= min_epochs:
            stopped = "patience"
            break
        if stopped == "time":
            break
    if trained_sd is None or int(best_trained["epoch"]) < 0:
        raise SystemExit("нет сохранённой эпохи epoch>=0")
    persist(epoch)
    init_net = MaskedReconstructor(16)
    init_net.load_state_dict(init_sd)
    trained_net = MaskedReconstructor(16)
    trained_net.load_state_dict(trained_sd)
    save_weights(INIT_PT, init_sd)
    save_weights(BEST, trained_sd)
    knots_init = fit_cdf(init_net, hold_w, src)
    knots_trained = fit_cdf(trained_net, hold_w, src)
    final_probe = proxy_metrics(trained_net, hold_w, event_w, src)
    body = dict(
        name="MAE",
        arch="masked_band_reconstructor",
        width=16,
        channel_width=channel_width(trained_net),
        params=n_param,
        band=BAND,
        radius=RADIUS,
        tile=[TILE_H, TILE_W],
        score_channel="dT_space",
        sign_channel="dT_time",
        loss="masked_l1_space_heavy",
        lr=lr,
        batch=batch,
        seed=seed,
        seed_before_model=True,
        steps_per_epoch=steps_per,
        train_windows=len(train_w),
        hold_windows=len(hold_w),
        event_proxy_windows=len(event_w),
        nu_limit=nu_limit,
        data="v8+v10 train, tiles без события, nuisance в допуске nu_limit",
        real_traces_used=False,
        real_noise_used=False,
        init_from=init_from,
        init_weights=str(INIT_PT),
        trained_weights=str(BEST),
        init_conv0=conv0_sum(init_net),
        trained_conv0=conv0_sum(trained_net),
        init_head_max=head_abs_max(init_net),
        trained_head_max=head_abs_max(trained_net),
        best=best,
        best_trained=best_trained,
        init_probe=init_probe,
        final_probe=final_probe,
        history=history,
        cdf_knots=knots_trained,
        cdf_knots_init=knots_init,
        cdf_knots_trained=knots_trained,
        stopped=stopped,
        seconds=round(float(spent()), 1),
    )
    dump_json(TRAIN_JSON, body)
    (CKPT_DIR / "DONE").touch()
    elog(
        f"{TAG}_TRAIN_DONE best_epoch={best['epoch']} score={best['score']:.4f} "
        f"trained_epoch={best_trained['epoch']} trained_score={best_trained['score']:.4f} "
        f"trained_head_max={best_trained.get('head_max')} "
        f"sep={best.get('sep')} bg={best.get('bg_abs')} restore={best.get('restore')} "
        f"stop={stopped} sec={spent():.0f} params={n_param} seed={seed} run={RUN}"
    )


if __name__ == "__main__":
    main()
