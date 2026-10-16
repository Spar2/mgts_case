"""Обучение головы CenterNet на замороженном энкодере unet_v810ft.

Только synth v8+v10 train. Val, test и реальные CSV не читаются.
Снимок artifacts/ckpt/<run>/last.pt: голова, оптимизатор, scheduler, шаг, RNG, лучшая прокси.
CENTERNET2: до 20000 шагов или 45 минут. Батч принимается только если в нём есть
положительный центр и не меньше половины окон содержат событие.

  scripts/run_resumable.sh centernet2 \\
    /workspace/repo/run/.venv/bin/python -u scripts/s3_centernet_train.py \\
    --steps 20000 --seed 20261015 --max-minutes 45 --run centernet2 --tag CENTERNET2
"""
from __future__ import annotations

import os
import zlib

_nthr = os.environ.get("OMP_NUM_THREADS", "3")
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, _nthr)

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
import sys
sys.path.insert(0, str(ROOT))

from dts import data
from dts.centernet import (
    STRIDE, CenterNetHead, build_targets, centernet_loss, encoder_features,
    encoder_width, feature_channels, gaussian_radius, load_encoder, size_radii,
)

LOG = Path("/workspace/grok_logs/_events.log")
ENC = "artifacts/stage1/unet_v810ft.pt"


def elog(msg: str) -> None:
    line = f"{time.strftime('%H:%M', time.gmtime())} {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def has_tag(tag: str) -> bool:
    if not LOG.exists():
        return False
    text = LOG.read_text(encoding="utf-8", errors="ignore")
    return tag in text


def load_item(path: Path) -> dict:
    with np.load(path) as z:
        x = np.asarray(z["X"])
        y = data.target(np.asarray(z["ev"]), np.asarray(z["valid"]))
    idx = np.argwhere(y > 0)
    if len(idx) > 4000:
        sub = np.random.default_rng(zlib.crc32(path.stem.encode()) & 0xFFFFFFFF)
        idx = idx[sub.choice(len(idx), 4000, replace=False)]
    return dict(X=x, y=y, ev_idx=np.asarray(idx, np.int32), name=path.stem)


def even_origin(n: int, tile: int, rng: np.random.Generator, center: int | None) -> int:
    """Чётное начало окна. Если задан пиксель события, он остаётся внутри кропа."""
    if n <= tile:
        return 0
    if center is None:
        t0 = int(rng.integers(0, n - tile + 1))
        t0 -= t0 % STRIDE
        return int(max(0, t0))
    c = int(center)
    t0 = c - tile // 2
    t0 -= t0 % STRIDE
    if t0 < 0:
        t0 = 0
    if t0 <= c < min(n, t0 + tile):
        return int(t0)
    t0 = c - (c % STRIDE)
    t0 = max(0, t0 - (tile - 1))
    t0 -= t0 % STRIDE
    if t0 > c:
        t0 = c - (c % STRIDE)
    return int(max(0, t0))


def crop_sample(it: dict, rng: np.random.Generator, th: int, tw: int, p_event: float):
    x, y = it["X"], it["y"]
    _, t, w = x.shape
    want = len(it["ev_idx"]) > 0 and rng.random() < p_event
    if want:
        k = int(rng.integers(0, len(it["ev_idx"])))
        tc, xc = int(it["ev_idx"][k, 0]), int(it["ev_idx"][k, 1])
    else:
        tc = xc = None
    t0 = even_origin(t, th, rng, tc)
    x0 = even_origin(w, tw, rng, xc)
    t1, x1 = min(t0 + th, t), min(x0 + tw, w)
    tile_x = np.zeros((x.shape[0], th, tw), np.float32)
    tile_y = np.full((th, tw), -1, np.int8)
    tile_x[:, :t1 - t0, :x1 - x0] = np.asarray(x[:, t0:t1, x0:x1], np.float32)
    tile_y[:t1 - t0, :x1 - x0] = y[t0:t1, x0:x1]
    if rng.random() < 0.5:
        tile_x = tile_x[:, :, ::-1].copy()
        tile_y = tile_y[:, ::-1].copy()
    has_event = bool((tile_y > 0).any())
    return tile_x, tile_y, has_event


class Pool:
    def __init__(self, paths: list[Path], rng: np.random.Generator, n: int = 8):
        self.paths = paths
        self.rng = rng
        self.n = n
        self.items: list[dict] = []
        self.names: list[str] = []

    def fill(self) -> None:
        pick = self.rng.choice(len(self.paths), size=min(self.n, len(self.paths)), replace=False)
        self.items = [load_item(self.paths[int(i)]) for i in pick]
        need = max(1, (len(self.items) + 1) // 2)
        have = sum(len(it["ev_idx"]) > 0 for it in self.items)
        tries = 0
        while have < need and tries < 40:
            tries += 1
            it = load_item(self.paths[int(self.rng.integers(0, len(self.paths)))])
            if len(it["ev_idx"]) == 0:
                continue
            for k, old in enumerate(self.items):
                if len(old["ev_idx"]) == 0:
                    self.items[k] = it
                    have += 1
                    break
            else:
                break
        self.names = [it["name"] for it in self.items]

    def load_names(self, names: list[str]) -> None:
        by = {p.stem: p for p in self.paths}
        self.items = [load_item(by[n]) for n in names]
        self.names = list(names)

    def _pick(self, force_event: bool) -> dict:
        for _ in range(8):
            it = self.items[int(self.rng.integers(0, len(self.items)))]
            if force_event and len(it["ev_idx"]) == 0:
                continue
            return it
        return self.items[int(self.rng.integers(0, len(self.items)))]

    def _crop(self, th: int, tw: int, force_event: bool):
        last = None
        tries = 8 if force_event else 1
        for _ in range(tries):
            it = self._pick(force_event)
            tile_x, tile_y, has = crop_sample(
                it, self.rng, th, tw, 1.0 if force_event else 0.0)
            last = (tile_x, tile_y, has)
            if has or not force_event:
                return last
        return last

    def sample(self, n: int, th: int, tw: int):
        """Не меньше половины окон — с пикселем события. Остальные — случайный фон."""
        n_force = max(1, (n + 1) // 2)
        xs, ys = [], []
        n_hit = 0
        for i in range(n):
            tile_x, tile_y, has = self._crop(th, tw, i < n_force)
            xs.append(tile_x)
            ys.append(tile_y)
            n_hit += int(has)
        return np.stack(xs), np.stack(ys), n_hit / float(n)


def rng_blob(gen: np.random.Generator) -> dict:
    return dict(
        torch=torch.get_rng_state(),
        python=random.getstate(),
        numpy=np.random.get_state(),
        generator=gen.bit_generator.state,
    )


def rng_restore(blob: dict, gen: np.random.Generator) -> None:
    torch.set_rng_state(blob["torch"])
    random.setstate(blob["python"])
    np.random.set_state(blob["numpy"])
    gen.bit_generator.state = blob["generator"]


def save_last(path: Path, blob: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pt.tmp")
    torch.save(blob, tmp)
    tmp.replace(path)


def export_head(path: Path, head: CenterNetHead, step: int, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(
        arch="centernet",
        head=head.state_dict(),
        cin=head.cin, mid=head.mid, n_class=head.n_class,
        stride=STRIDE, encoder=ENC, width=16, step=int(step), seed=int(seed),
        peak_min=0.05, topk=100,
    ), path)


def proxy_hit(net, head, it: dict) -> dict:
    """Прокси на одном train-тайле: средняя вероятность в клетках цели. Не отбор."""
    head.eval()
    rng = np.random.default_rng(0)
    tile_x, tile_y, _has = crop_sample(it, rng, 128, 256, 1.0)
    hm, valid, n_obj, reg = build_targets(tile_y[None], STRIDE)
    with torch.no_grad():
        feat = encoder_features(net, torch.from_numpy(tile_x[None]))
        prob = torch.sigmoid(head(feat)["hm"]).numpy()
    if n_obj == 0 or not valid.any():
        hit, bg = 0.0, float(prob.mean())
    else:
        vals = []
        for r in reg:
            iy, ix = int(r[1]), int(r[2])
            cls = int(np.argmax(hm[0, :, iy, ix]))
            vals.append(float(prob[0, cls, iy, ix]))
        hit = float(np.mean(vals)) if vals else 0.0
        bg = float(prob[0][:, valid[0]].mean())
    head.train()
    return dict(center_hit=hit, bg_mean=bg, n_obj=int(n_obj), score=float(hit - bg))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=20261015)
    ap.add_argument("--max-minutes", type=float, default=45.0)
    ap.add_argument("--run", default="centernet2")
    ap.add_argument("--tag", default="CENTERNET2")
    ap.add_argument("--resume", default="")
    ap.add_argument("--out", default="artifacts/eval/s3_centernet2_head.pt")
    ap.add_argument("--log-json", default="artifacts/eval/s3_centernet2_train.json")
    args = ap.parse_args()
    tag = str(args.tag)
    if gaussian_radius(80, 8) <= gaussian_radius(2, 2):
        raise SystemExit("радиус гауссианы не растёт с размером рамки")

    nthr = int(os.environ.get("OMP_NUM_THREADS", "2"))
    torch.set_num_threads(nthr)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    ckpt_dir = Path("artifacts/ckpt") / args.run
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_path = ckpt_dir / "last.pt"
    done_path = ckpt_dir / "DONE"
    if done_path.exists() and args.resume in ("", "auto"):
        elog(f"{tag}_TRAIN_DONE already_complete run={args.run}")
        return

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    gen = np.random.default_rng(args.seed)

    files = data.list_cache("artifacts/cache/v8", "train") + data.list_cache("artifacts/cache/v10", "train")
    if len(files) < 8:
        raise SystemExit(f"мало train-файлов: {len(files)}")
    net = load_encoder(ENC, 16)
    if encoder_width(net) != 16:
        raise SystemExit("энкодер не ширины 16")
    cin = feature_channels(16)
    head = CenterNetHead(cin=cin, mid=32, n_class=2)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=0.15 * args.lr)

    resume_path = None
    if args.resume == "auto" and last_path.exists():
        resume_path = last_path
    elif args.resume not in ("", "auto"):
        resume_path = Path(args.resume)

    start = 1
    elapsed_prev = 0.0
    history = []
    best = dict(step=0, center_hit=0.0, bg_mean=1.0, score=-1.0, n_obj=0)
    pool = Pool(files, gen, n=8)
    if resume_path is not None:
        blob = torch.load(resume_path, map_location="cpu", weights_only=False)
        if int(blob.get("seed", -1)) != args.seed or int(blob.get("steps", -1)) != args.steps:
            raise SystemExit("resume: seed или steps не совпали")
        if blob.get("arch") != "centernet":
            raise SystemExit("resume: не centernet")
        head.load_state_dict(blob["head"])
        opt.load_state_dict(blob["optim"])
        sched.load_state_dict(blob["sched"])
        rng_restore(blob["rng"], gen)
        best = blob.get("best") or best
        history = list(blob.get("history") or [])
        elapsed_prev = float(blob.get("seconds") or 0.0)
        start = int(blob["step"]) + 1
        pool.load_names(blob["cursor"]["names"])
        elog(f"{tag}_RESUME step={blob['step']} run={args.run}")
        if int(blob["step"]) >= args.steps:
            export_head(Path(args.out), head, int(blob["step"]), args.seed)
            done_path.touch()
            elog(f"{tag}_TRAIN_DONE already_complete step={blob['step']}")
            return
    else:
        pool.fill()
        msg = (
            f"steps={args.steps} batch={args.batch} max_min={args.max_minutes} "
            f"train_files={len(files)} threads={nthr} seed={args.seed} "
            f"encoder=unet_v810ft arch=centernet_e2e3_dilated stride={STRIDE} "
            f"event_windows>=0.5 n_pos>0 gaussian=size"
        )
        if not has_tag(f"{tag}_START"):
            elog(f"{tag}_START {msg}")

    head.train()
    budget = float(args.max_minutes) * 60.0
    t0 = time.time()
    last_save = t0
    stopped = "steps"
    step = start - 1
    loss_v = 0.0
    pos_p = 0.0
    rad_mean = 0.0
    rad_x_mean = 0.0

    def elapsed() -> float:
        return elapsed_prev + (time.time() - t0)

    def dump(step_done: int) -> None:
        blob = dict(
            arch="centernet",
            head=head.state_dict(),
            optim=opt.state_dict(),
            sched=sched.state_dict(),
            step=int(step_done),
            rng=rng_blob(gen),
            best=best,
            history=history[-80:],
            cursor=dict(names=pool.names),
            seconds=round(elapsed(), 1),
            cin=cin, mid=32, n_class=2, stride=STRIDE,
            seed=args.seed, steps=args.steps, encoder=ENC, width=16,
        )
        save_last(last_path, blob)

    while step < args.steps:
        if elapsed() >= budget and step >= start:
            stopped = "budget"
            break
        nxt = step + 1
        if nxt != start and nxt % 40 == 0:
            pool.fill()
        accepted = False
        ev_frac = 0.0
        n_obj = 0
        hm = valid = reg = None
        xb = yb = None
        for attempt in range(12):
            if attempt in (4, 8):
                pool.fill()
            xb, yb, ev_frac = pool.sample(args.batch, 128, 256)
            hm, valid, n_obj, reg = build_targets(yb, STRIDE)
            if n_obj > 0 and ev_frac + 1e-9 >= 0.5:
                accepted = True
                break
        if not accepted:
            raise SystemExit(
                f"батч без положительных центров next={nxt} n_obj={n_obj} ev_frac={ev_frac:.2f}"
            )
        step = nxt
        with torch.no_grad():
            feat = encoder_features(net, torch.from_numpy(xb))
        pred = head(feat.detach())
        losses = centernet_loss(
            pred, torch.from_numpy(hm), torch.from_numpy(valid), n_obj, torch.from_numpy(reg))
        loss = losses["loss"]
        if not torch.isfinite(loss):
            raise SystemExit(f"неконечная потеря на шаге {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
        opt.step()
        sched.step()
        loss_v = float(loss.detach())
        pairs = [size_radii(float(np.exp(r[3])), float(np.exp(r[4]))) for r in reg]
        rad_mean = float(np.mean([p[0] for p in pairs])) if pairs else 0.0
        rad_x_mean = float(np.mean([p[1] for p in pairs])) if pairs else 0.0
        with torch.no_grad():
            prb = torch.sigmoid(pred["hm"].detach())
            pos = torch.from_numpy(hm) >= 0.999
            pos_p = float(prb[pos].mean()) if int(pos.sum()) else 0.0
        if step % 20 == 0 or step == 1 or step == args.steps:
            print(
                f"step {step} loss {loss_v:.4f} hm {float(losses['hm']):.4f} "
                f"wh {float(losses['wh']):.4f} off {float(losses['off']):.4f} "
                f"pos_p {pos_p:.3f} n {n_obj} ev {ev_frac:.2f} "
                f"rad {rad_mean:.1f}/{rad_x_mean:.1f} sec {elapsed():.0f}",
                flush=True,
            )
        if step % 200 == 0 or step == args.steps:
            pr = proxy_hit(net, head, pool.items[0])
            rec = dict(step=step, loss=loss_v, sec=round(elapsed(), 1), pos_p=pos_p,
                       ev_frac=ev_frac, rad_t=rad_mean, rad_x=rad_x_mean, **pr)
            history.append(rec)
            if pr["score"] > float(best.get("score", -1e9)):
                best = dict(step=step, **pr)
            elog(
                f"{tag}_TRAIN step={step} loss={loss_v:.4f} pos_p={pos_p:.3f} "
                f"center_hit={pr['center_hit']:.3f} "
                f"bg={pr['bg_mean']:.3f} n={n_obj} ev={ev_frac:.2f} "
                f"rad={rad_mean:.1f}/{rad_x_mean:.1f} "
                f"sec={elapsed():.0f}"
            )
        wall = time.time() - last_save
        if step % 100 == 0 or wall >= 15 * 60 or step == args.steps:
            dump(step)
            last_save = time.time()
        if elapsed() >= budget:
            stopped = "budget"
            dump(step)
            break

    export_head(Path(args.out), head, step, args.seed)
    report = dict(
        arch="centernet",
        encoder=ENC,
        width=16,
        stride=STRIDE,
        features="e2+nearest(e3), no decoder, no TimeGate",
        classes="hot,cold",
        size="log Δt, log Δx",
        offset=True,
        seed=args.seed,
        steps_target=args.steps,
        step=int(step),
        stopped=stopped,
        seconds=round(elapsed(), 1),
        threads=nthr,
        batch=args.batch,
        lr=args.lr,
        train_files=len(files),
        final_loss=loss_v,
        final_pos_p=pos_p,
        final_radius_t=rad_mean,
        final_radius_x=rad_x_mean,
        event_window_min=0.5,
        gaussian="size min_overlap 0.7",
        best_proxy=best,
        real_traces_used=False,
        val_used_for_training=False,
        history=history[-40:],
    )
    out_json = Path(args.log_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    done_path.touch()
    elog(
        f"{tag}_TRAIN_DONE step={step} sec={elapsed():.1f} stopped={stopped} "
        f"loss={loss_v:.4f} pos_p={pos_p:.3f} rad={rad_mean:.1f}/{rad_x_mean:.1f} "
        f"proxy_best_step={best.get('step')} center_hit={best.get('center_hit')}"
    )


if __name__ == "__main__":
    main()
