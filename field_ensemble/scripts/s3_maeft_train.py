"""Дообучение сегментации после MAE-энкодера. Только синтетика v8+v10 train.

Три руки одного бюджета шагов:
  mae    — SegUNet, энкодер скопирован из artifacts/eval/s3_mae_recon.pt
  random — тот же SegUNet от того же seed, без копии
  v810   — DTSUNet из artifacts/stage1/unet_v810ft.pt

Чекпойнт не смотрит на официальный v8 val. Прокси — IoU@FAR 1e-3 на тайлах
train-holdout (crc32 % 10 == 0). Запуск:
  scripts/run_resumable.sh s3_maeft /workspace/repo/run/.venv/bin/python -u scripts/s3_maeft_train.py
Лаунчер сам дописывает --resume auto. Реальные CSV, карточки и спектр шума не читаются.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "NUMEXPR_MAX_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_k] = "2"

import argparse
import json
import random
import shutil
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
sys.path.insert(0, str(ROOT / "scripts"))

from dts import data, losses, metrics as M
from s3_maeft_net import (
    ARMS, DEC_LR, ENC_LR, MAE_CKPT, V810_CKPT, WIDTH,
    build_arm, copy_mae_encoder, encoder_delta, param_groups,
)

RUN = "s3_maeft"
CKPT_DIR = Path("artifacts/ckpt") / RUN
LAST = CKPT_DIR / "last.pt"
TRAIN_JSON = Path("artifacts/eval/s3_maeft_train.json")
LOG = Path("/workspace/run_logs/_events.log")
TAG = "MAEFT_TRAIN"
SEED = 20261015
SAVE_EVERY = 14 * 60
TILE = (128, 256)
TMP_V10 = Path("/tmp/s3_maeft_v10")
FORBIDDEN = ("MGTS", "real_noise", "models_release", "real_zones")


def log(msg: str) -> None:
    print(time.strftime("%H:%M", time.gmtime()) + " " + msg, flush=True)


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


def forbid_real() -> None:
    for path in (MAE_CKPT, V810_CKPT, "artifacts/cache/v8/train", "artifacts/cache/v10/train"):
        text = str(path)
        for bad in FORBIDDEN:
            if bad in text:
                raise SystemExit(f"запрещённый путь {path}")


def rng_blob(rng: np.random.Generator) -> dict:
    return dict(
        torch=torch.get_rng_state(),
        numpy=np.random.get_state(),
        py=random.getstate(),
        bit=rng.bit_generator.state,
    )


def rng_restore(blob: dict, rng: np.random.Generator) -> None:
    torch.set_rng_state(blob["torch"])
    np.random.set_state(blob["numpy"])
    random.setstate(blob["py"])
    rng.bit_generator.state = blob["bit"]


class Cache:
    """v8 читается из готового mmap. v10 один раз кладётся в /tmp и удаляется в конце."""

    def __init__(self, smoke: bool):
        self.smoke = smoke
        self.mm = OrderedDict()
        self.items = []
        self._open()

    def _mm(self, path: Path):
        key = str(path)
        if key in self.mm:
            self.mm.move_to_end(key)
            return self.mm[key]
        arr = np.load(path, mmap_mode="r")
        self.mm[key] = arr
        while len(self.mm) > 8:
            self.mm.popitem(last=False)
        return arr

    def _index_file(self, stem: str, x_path: Path, y_path: Path, hold: bool) -> None:
        y = self._mm(y_path)
        height, width = y.shape
        th, tw = TILE
        if height < th or width < tw:
            return
        pos = np.argwhere(np.asarray(y) > 0)
        if len(pos) > 48:
            pick = np.random.default_rng(crc(stem)).choice(len(pos), 48, replace=False)
            pos = pos[pick]
        self.items.append(dict(
            stem=stem, x=x_path, y=y_path, hold=hold,
            T=int(height), W=int(width), ev=np.asarray(pos, np.int32),
        ))

    def _ensure_v10(self, path: Path) -> tuple[Path, Path]:
        TMP_V10.mkdir(parents=True, exist_ok=True)
        fx, fy = TMP_V10 / f"{path.stem}_X.npy", TMP_V10 / f"{path.stem}_y.npy"
        if fx.exists() and fy.exists() and fx.stat().st_size > 0 and fy.stat().st_size > 0:
            return fx, fy
        with np.load(path) as z:
            image = np.asarray(z["X"])
            target = data.target(np.asarray(z["ev"]), np.asarray(z["valid"]))
        tmp_x = TMP_V10 / f"{path.stem}_X.tmp.npy"
        tmp_y = TMP_V10 / f"{path.stem}_y.tmp.npy"
        np.save(tmp_x, image)
        np.save(tmp_y, target)
        os.replace(tmp_x, fx)
        os.replace(tmp_y, fy)
        return fx, fy

    def _open(self) -> None:
        v8 = sorted(Path("artifacts/cache/v8/train").glob("seq*.npz"))
        v10 = sorted(Path("artifacts/cache/v10/train").glob("seq*.npz"))
        if self.smoke:
            v8, v10 = v8[:4], []
        if not v8:
            raise SystemExit("нет v8 train")
        mm = Path("artifacts/mm/v8/train")
        for i, path in enumerate(v8):
            fx, fy = mm / f"{path.stem}_X.npy", mm / f"{path.stem}_y.npy"
            if not fx.exists() or not fy.exists():
                raise SystemExit(f"нет mmap {path.stem}")
            hold = (i == len(v8) - 1) if self.smoke else ((crc(path.stem) % 10) == 0)
            self._index_file(path.stem, fx, fy, hold)
        for i, path in enumerate(v10, 1):
            fx, fy = self._ensure_v10(path)
            self._index_file(path.stem, fx, fy, (crc(path.stem) % 10) == 0)
            if i == 1 or i % 40 == 0 or i == len(v10):
                log(f"v10 cache {i}/{len(v10)}")
        n_hold = sum(1 for it in self.items if it["hold"])
        need = 1 if self.smoke else 2
        if n_hold < need or n_hold >= len(self.items):
            raise SystemExit(f"плохой holdout {n_hold}/{len(self.items)}")
        log(f"индекс train {len(self.items) - n_hold} hold {n_hold}")

    def crop(self, item: dict, t0: int, w0: int, flip: bool, gain: float) -> tuple[np.ndarray, np.ndarray]:
        th, tw = TILE
        image = np.array(self._mm(item["x"])[:, t0:t0 + th, w0:w0 + tw], dtype=np.float32, copy=True)
        target = np.array(self._mm(item["y"])[t0:t0 + th, w0:w0 + tw], dtype=np.int64, copy=True)
        if image.shape[-2:] != (th, tw):
            pad_x = np.zeros((3, th, tw), np.float32)
            pad_y = np.full((th, tw), -1, np.int64)
            pad_x[:, :image.shape[1], :image.shape[2]] = image
            pad_y[:target.shape[0], :target.shape[1]] = target
            image, target = pad_x, pad_y
        if flip:
            image = image[:, :, ::-1].copy()
            target = target[:, ::-1].copy()
        image[:2] = np.clip(image[:2] * gain, -12.0, 12.0)
        return image, target

    def sample(self, n: int, rng: np.random.Generator, hold: bool) -> tuple[np.ndarray, np.ndarray]:
        pool = [it for it in self.items if it["hold"] is hold]
        weights = np.array([it["T"] * it["W"] for it in pool], np.float64)
        weights /= weights.sum()
        th, tw = TILE
        xs, ys = [], []
        for _ in range(n):
            item = pool[int(rng.choice(len(pool), p=weights))]
            if item["ev"].shape[0] and float(rng.random()) < 0.6:
                cy, cx = item["ev"][int(rng.integers(item["ev"].shape[0]))]
                t0 = int(cy) - int(rng.integers(th))
                w0 = int(cx) - int(rng.integers(tw))
            else:
                t0 = int(rng.integers(item["T"] - th + 1))
                w0 = int(rng.integers(item["W"] - tw + 1))
            t0 = int(np.clip(t0, 0, item["T"] - th))
            w0 = int(np.clip(w0, 0, item["W"] - tw))
            image, target = self.crop(item, t0, w0, bool(rng.random() < 0.5), float(rng.uniform(0.75, 1.3)))
            xs.append(image)
            ys.append(target)
        return np.stack(xs), np.stack(ys)


def proxy_iou(model: nn.Module, image: np.ndarray, target: np.ndarray) -> float:
    was = model.training
    model.eval()
    scores, truths, valids = [], [], []
    with torch.no_grad():
        for i in range(0, len(image), 4):
            logit = model(torch.from_numpy(np.ascontiguousarray(image[i:i + 4])))
            prob = torch.softmax(logit, 1)
            scores.append((1.0 - prob[:, 0]).numpy())
            y = target[i:i + 4]
            truths.append(y > 0)
            valids.append(y >= 0)
    model.train(was)
    score = np.concatenate([a.ravel() for a in scores])
    truth = np.concatenate([a.ravel() for a in truths])
    valid = np.concatenate([a.ravel() for a in valids])
    bg = valid & ~truth
    if int(bg.sum()) < 200 or int((valid & truth).sum()) < 20:
        return 0.0
    thr = M.far_threshold(score[bg], 1e-3)
    return float(M.prf(*M.confusion((score >= thr) & valid, truth, valid))["iou"])


def save_last(payload: dict) -> None:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = LAST.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, LAST)


def save_best(arm: str, state) -> None:
    path = Path(f"artifacts/eval/s3_maeft_{arm}.pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pt.tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", default="never", choices=("auto", "never"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    forbid_real()
    epochs = 2 if args.smoke else int(os.environ.get("MAEFT_EPOCHS", "30"))
    steps = 2 if args.smoke else int(os.environ.get("MAEFT_STEPS", "40"))
    batch = 2 if args.smoke else int(os.environ.get("MAEFT_BATCH", "8"))
    max_minutes = 1.0 if args.smoke else float(os.environ.get("MAEFT_MAX_MINUTES", "58"))
    hold_n = 4 if args.smoke else 24
    if args.smoke:
        global CKPT_DIR, LAST, TRAIN_JSON
        CKPT_DIR = Path("/tmp/s3_maeft_smoke_ckpt")
        LAST = CKPT_DIR / "last.pt"
        TRAIN_JSON = Path("/tmp/s3_maeft_smoke_train.json")
        if CKPT_DIR.exists():
            shutil.rmtree(CKPT_DIR)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    done = CKPT_DIR / "DONE"
    if done.exists() and args.resume != "never" and not args.smoke:
        log("уже DONE")
        return

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    cache = Cache(smoke=args.smoke)
    hold_x, hold_y = cache.sample(hold_n, np.random.default_rng(SEED + 17), hold=True)

    models = {arm: build_arm(arm, SEED) for arm in ARMS}
    # Случайная рука и MAE делят один seed конструктора. Энкодер MAE после копии другой.
    same = torch.equal(models["mae"].head.weight, models["random"].head.weight)
    if not same:
        raise SystemExit("головы mae и random разошлись: seed конструктора не общий")
    report = copy_mae_encoder(models["mae"], MAE_CKPT)
    if report["max_abs"] != 0.0 or report["copied"] < 30:
        raise SystemExit(f"плохая копия MAE {report}")
    rand_delta = float((models["random"].e1[0][0].weight.detach() - models["mae"].e1[0][0].weight.detach()).abs().max())
    if rand_delta == 0.0:
        raise SystemExit("случайный энкодер совпал с MAE")
    v810_delta = encoder_delta(models["v810"], V810_CKPT, ("e1.", "e2.", "e3.", "e4.", "head."))
    if v810_delta["max_abs"] != 0.0:
        raise SystemExit(f"v810 не совпал с чекпойнтом {v810_delta}")
    log(f"копия MAE тензоров {report['copied']} max|Δ| {report['max_abs']} random|Δствол| {rand_delta:.4f} v810|Δ| {v810_delta['max_abs']}")

    opts = {
        arm: torch.optim.AdamW(param_groups(models[arm], arm), weight_decay=0.0)
        for arm in ARMS
    }
    cw = torch.tensor([1.0, 3.0, 3.0])
    best = {arm: dict(proxy=-1.0, epoch=-1, state=None) for arm in ARMS}
    history = []
    step = 0
    epoch = 0
    t0 = time.time()
    last_save = t0

    if args.resume == "auto" and LAST.exists() and not args.smoke:
        blob = torch.load(LAST, map_location="cpu", weights_only=False)
        for arm in ARMS:
            models[arm].load_state_dict(blob["models"][arm])
            opts[arm].load_state_dict(blob["opts"][arm])
        rng_restore(blob["rng"], rng)
        best = blob["best"]
        history = list(blob["history"])
        step = int(blob["step"])
        epoch = int(blob["epoch"])
        t0 = time.time() - float(blob["seconds"])
        last_save = time.time()
        log(f"resume step {step} epoch {epoch} sec {blob['seconds']:.0f}")

    def snapshot(reason: str) -> dict:
        return dict(
            models={arm: models[arm].state_dict() for arm in ARMS},
            opts={arm: opts[arm].state_dict() for arm in ARMS},
            rng=rng_blob(rng),
            best=best,
            history=history,
            step=step,
            epoch=epoch,
            seconds=time.time() - t0,
            reason=reason,
            seed=SEED,
        )

    def persist(reason: str) -> None:
        nonlocal last_save
        save_last(snapshot(reason))
        last_save = time.time()

    def consider(arm: str, proxy: float, ep: int) -> None:
        if proxy > float(best[arm]["proxy"]) + 1e-4:
            best[arm] = dict(
                proxy=float(proxy), epoch=int(ep),
                state={k: v.detach().cpu().clone() for k, v in models[arm].state_dict().items()},
            )
            if not args.smoke:
                save_best(arm, best[arm]["state"])
            log(f"лучший {arm} epoch {ep} proxy {proxy:.4f}")

    if epoch == 0 and step == 0:
        for arm in ARMS:
            proxy = proxy_iou(models[arm], hold_x, hold_y)
            consider(arm, proxy, -1)
            history.append(dict(epoch=-1, arm=arm, proxy=proxy, loss=None, step=0, sec=0.0))
            log(f"init {arm} proxy {proxy:.4f}")
        persist("init")

    stopped = "epochs"
    while epoch < epochs:
        if (time.time() - t0) / 60.0 >= max_minutes:
            stopped = "time"
            break
        losses_ep = {arm: [] for arm in ARMS}
        for _ in range(steps):
            if (time.time() - t0) / 60.0 >= max_minutes:
                stopped = "time"
                break
            image, target = cache.sample(batch, rng, hold=False)
            xt = torch.from_numpy(np.ascontiguousarray(image))
            yt = torch.from_numpy(np.ascontiguousarray(target))
            for arm in ARMS:
                models[arm].train()
                logit = models[arm](xt)
                loss = losses.compute("focal+tversky_ce", logit, yt, cw)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss {arm} step {step}")
                opts[arm].zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(models[arm].parameters(), 1.0)
                opts[arm].step()
                losses_ep[arm].append(float(loss.detach()))
            step += 1
            if time.time() - last_save >= SAVE_EVERY:
                persist("step")
                elog(f"{TAG} step={step} epoch={epoch} loss={losses_ep['mae'][-1]:.4f} sec={time.time() - t0:.0f}")
        if stopped == "time" and not any(losses_ep[arm] for arm in ARMS):
            break
        row = dict(epoch=epoch, step=step, sec=round(time.time() - t0, 1))
        for arm in ARMS:
            proxy = proxy_iou(models[arm], hold_x, hold_y)
            mean_loss = float(np.mean(losses_ep[arm])) if losses_ep[arm] else None
            row[arm] = dict(loss=mean_loss, proxy=proxy)
            consider(arm, proxy, epoch)
            log(f"epoch {epoch} {arm} loss {mean_loss} proxy {proxy:.4f}")
        history.append(row)
        if not args.smoke:
            elog(
                f"{TAG} epoch={epoch} step={step} mae={row['mae']['proxy']:.4f} "
                f"rand={row['random']['proxy']:.4f} v810={row['v810']['proxy']:.4f} sec={row['sec']:.0f}"
            )
        epoch += 1
        persist("epoch")
        if stopped == "time":
            break

    for arm in ARMS:
        if best[arm]["state"] is None:
            best[arm]["state"] = {k: v.detach().cpu().clone() for k, v in models[arm].state_dict().items()}
        models[arm].load_state_dict(best[arm]["state"])
        if not args.smoke:
            save_best(arm, best[arm]["state"])

    def payload(stopped: str) -> dict:
        deltas = {}
        for arm, path, prefixes in (
            ("mae", MAE_CKPT, ("e1.", "e2.", "e3.")),
            ("v810", V810_CKPT, ("e1.", "e2.", "e3.", "e4.", "head.")),
        ):
            deltas[arm] = encoder_delta(models[arm], path, prefixes)
        return dict(
            name="MAEFT",
            arms=list(ARMS),
            arch=dict(
                mae="SegUNet width 16, encoder = MaskedReconstructor e1/e2/e3",
                random="SegUNet width 16, same constructor seed, encoder not copied",
                v810="DTSUNet width 16 strict load unet_v810ft",
            ),
            note=(
                "Ствол DTSUNet (16,3,7,3) не совпадает со стволом MAE (16,4,5,3). "
                "Совместимые веса энкодера переносятся в U-Net ширины MAE. "
                "Рука v810 — дообучение опубликованного U-Net тем же числом шагов."
            ),
            mae_copied=report["copied"],
            mae_copy_max_abs=report["max_abs"],
            random_stem_abs=rand_delta,
            encoder_max_dw=deltas,
            loss="focal+tversky_ce",
            class_weight=[1, 3, 3],
            lr=dict(encoder=ENC_LR, decoder=DEC_LR),
            weight_decay=0.0,
            tile=[128, 256],
            batch=batch,
            steps_per_epoch=steps,
            epochs_cap=epochs,
            max_minutes=max_minutes,
            seed=SEED,
            threads=2,
            train_cache=["artifacts/cache/v8/train", "artifacts/cache/v10/train"],
            proxy="IoU@FAR1e-3 on crc32%10==0 train tiles, not v8 val",
            real_traces_used=False,
            real_noise_used=False,
            best={arm: dict(epoch=best[arm]["epoch"], proxy=best[arm]["proxy"]) for arm in ARMS},
            history=history,
            step=step,
            stopped=stopped,
            seconds=round(time.time() - t0, 1),
            ckpt={arm: f"artifacts/eval/s3_maeft_{arm}.pt" for arm in ARMS},
        )

    if args.smoke:
        body = payload(stopped)
        if body["mae_copy_max_abs"] != 0.0:
            raise SystemExit("smoke: копия до обучения не была нулевой")
        log(f"smoke ok step {step} best {body['best']} dw {body['encoder_max_dw']}")
        shutil.rmtree(CKPT_DIR, ignore_errors=True)
        return

    body = payload(stopped)
    dump_json(TRAIN_JSON, body)
    persist("final")
    done.write_text("ok\n", encoding="utf-8")
    shutil.rmtree(TMP_V10, ignore_errors=True)
    elog(
        f"{TAG}_END step={step} stopped={stopped} sec={body['seconds']:.0f} "
        f"mae_ep={best['mae']['epoch']} mae_proxy={best['mae']['proxy']:.4f} "
        f"rand_ep={best['random']['epoch']} rand_proxy={best['random']['proxy']:.4f} "
        f"v810_ep={best['v810']['epoch']} v810_proxy={best['v810']['proxy']:.4f} "
        f"copied={report['copied']} dw_mae={body['encoder_max_dw']['mae']['max_abs']:.6f} "
        f"dw_key={body['encoder_max_dw']['mae']['key']} real_train=0"
    )
    log("train DONE")


if __name__ == "__main__":
    main()
