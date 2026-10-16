"""Обучение StaticNet6 на генераторе STATIC2.

Вход ветви — температура без пространственного детренда. В train-кэше сырого T
нет: поле — остаток канала 1 плюс гладкий тренд STATIC2. Val/test/real в это
обучение не входят.

Возобновление: --resume auto читает artifacts/ckpt/<run>/last.pt
(модель, оптимизатор, scheduler, шаг, RNG, лучшая прокси-метрика).
Снимок не реже чем раз в 15 минут. В конце — файл DONE.
Запуск: scripts/run_resumable.sh static6 <python> scripts/s3_static6_train.py ...
"""
from __future__ import annotations

import argparse, json, os, random, sys, time
from pathlib import Path

_NTHR = os.environ.get("OMP_NUM_THREADS", "3")
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, _NTHR)

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dts.static_net import (
    LONG_ROWS, REF_DILATIONS, TRUNK_DILATIONS, WIN_2H_ROWS, WIN_30MIN_ROWS,
    StaticNet6, predict_temperature, residual_from_cache, static_loss,
)
from dts.static_synth import (
    DX, S2_AMP, S2_WIDTH, add_smooth_trend, apply_profiles, seq_seed, static2_fixed_hots,
    static2_train_specs, static2_view, unit_profile,
)

CROP = 1280
CKPT_EVERY_S = 15 * 60
LOG = Path("/workspace/run_logs/_events.log")


def log(msg: str):
    line = time.strftime("%H:%M") + " " + msg
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def load_item(path: Path):
    z = np.load(path)
    R, valid = residual_from_cache(z)
    ev = np.asarray(z["ev"])
    return dict(R=R.astype(np.float16), valid=valid, ev=ev, name=path.stem, path=path)


def event_columns(ev: np.ndarray) -> np.ndarray:
    return (ev > 0).mean(axis=0) > 0.02


def make_sample(item, rng, max_rows: int):
    R = np.asarray(item["R"], np.float32)
    valid = item["valid"]
    rows, width = R.shape
    if width < 256 or rows < 16:
        return None
    t1 = rows
    t0 = 0
    if rows > max_rows and rng.random() < 0.25:
        span = int(rng.integers(max(192, max_rows // 2), rows + 1))
        t0 = int(rng.integers(0, rows - span + 1))
        t1 = t0 + span
    avoid = event_columns(item["ev"][t0:t1])
    specs = static2_train_specs(rng, width, DX, avoid)
    field = R[t0:t1]
    v = np.array(valid[t0:t1], copy=True)
    field, pos, hard = apply_profiles(field, [static2_view(s) for s in specs])
    if rng.random() < 0.7:
        field = add_smooth_trend(field, rng, DX)
    field = field * np.float32(rng.uniform(0.85, 1.15))
    if rng.random() < 0.5:
        field = field[:, ::-1]
        v = v[:, ::-1]
        pos = pos[::-1].copy()
        hard = hard[::-1].copy()
        avoid = avoid[::-1].copy()
    if width > CROP:
        if pos.any():
            c = int(np.flatnonzero(pos).mean())
            start = c - int(rng.integers(CROP // 5, 4 * CROP // 5))
        else:
            start = int(rng.integers(0, width - CROP + 1))
        start = int(np.clip(start, 0, width - CROP))
        sl = slice(start, start + CROP)
        field, v, pos, hard, avoid = field[:, sl], v[:, sl], pos[sl], hard[sl], avoid[sl]
    feats, good = temperature_call(field, v, max_rows)
    y = pos.astype(np.float32)
    w = np.ones(y.shape, np.float32)
    w[~good] = 0
    w[avoid & ~pos] = 0
    w[hard] = 4.0
    peak = good & ~pos & (feats[0] > 0.25)
    w[peak] = np.maximum(w[peak], 3.0)
    w[pos] = 1.0
    return feats, y, w


def temperature_call(field, valid, max_rows):
    from dts.static_net import temperature_features
    return temperature_features(field, valid, max_rows=max_rows, win_30=WIN_30MIN_ROWS, win_2h=WIN_2H_ROWS)


def proxy(model, files, seed_base: int):
    hits, n_ev, fired, cols = [], 0, 0, 0
    for p in files:
        z = np.load(p)
        R, valid = residual_from_cache(z)
        ev = np.asarray(z["ev"])
        p0 = predict_temperature(model, R, valid)
        bg = (ev == 0).all(0) & (valid.mean(0) >= 0.5)
        fired += int((p0[bg] >= 0.90).sum())
        cols += int(bg.sum())
        avoid = event_columns(ev)
        specs = static2_fixed_hots(R.shape[1], seq_seed(seed_base, p.stem), n_events=2, avoid=avoid)
        strong = [sp for sp in specs if sp["amp"] >= 8.0]
        if not strong:
            continue
        Rp, _, _ = apply_profiles(R, [static2_view(sp) for sp in strong])
        pp = predict_temperature(model, Rp, valid)
        for sp in strong:
            cols_ev = unit_profile(R.shape[1], static2_view(sp)) >= 0.5
            if not cols_ev.any():
                continue
            n_ev += 1
            frac = float((pp[cols_ev] >= 0.80).mean())
            hits.append(frac >= 0.3)
    rec = float(np.mean(hits)) if hits else 0.0
    rate = fired / max(cols, 1)
    return dict(strong_recall=rec, n_strong=n_ev, bg_col_rate=rate)


def rng_blob(rng):
    return dict(
        torch=torch.get_rng_state(),
        numpy=np.random.get_state(),
        python=random.getstate(),
        generator=rng.bit_generator.state,
    )


def restore_rng(rng, blob):
    torch.set_rng_state(blob["torch"])
    np.random.set_state(blob["numpy"])
    random.setstate(blob["python"])
    rng.bit_generator.state = blob["generator"]


def atomic_save(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=20261015)
    ap.add_argument("--max-rows", type=int, default=LONG_ROWS)
    ap.add_argument("--early-stop", type=int, default=0)
    ap.add_argument("--tag", default="STATIC6")
    ap.add_argument("--out", default="artifacts/static/static_net6.pt")
    ap.add_argument("--log-json", default="artifacts/static/train_log6.json")
    ap.add_argument("--run", default="static6")
    ap.add_argument("--resume", default="", help="auto = artifacts/ckpt/<run>/last.pt, если файл есть")
    a = ap.parse_args()
    nthr = int(os.environ.get("OMP_NUM_THREADS", "3"))
    torch.set_num_threads(nthr)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    rng = np.random.default_rng(a.seed)
    paths = sorted((ROOT / "artifacts/cache/v8/train").glob("seq*.npz"))
    paths += sorted((ROOT / "artifacts/cache/v10/train").glob("seq*.npz"))
    if len(paths) < 4:
        raise SystemExit(f"мало train-кэшей: {len(paths)}")
    val_files = [ROOT / "artifacts/cache/v8/val" / n for n in
                 ("seq0272.npz", "seq0126.npz", "seq0083.npz", "seq0242.npz")]
    val_files = [p for p in val_files if p.exists()]
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    run_dir = ROOT / "artifacts/ckpt" / a.run
    run_dir.mkdir(parents=True, exist_ok=True)
    last_path = run_dir / "last.pt"
    model = StaticNet6(max_rows=a.max_rows, win_30=WIN_30MIN_ROWS, win_2h=WIN_2H_ROWS)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps, eta_min=a.lr * 0.15)

    resume_path = None
    if a.resume == "auto":
        resume_path = last_path if last_path.is_file() else None
    elif a.resume:
        resume_path = Path(a.resume)
        if not resume_path.is_file():
            raise SystemExit(f"нет чекпойнта {resume_path}")

    start = 1
    best = dict(score=-1e9, step=0, strong_recall=None, bg_col_rate=None)
    history = []
    elapsed_prev = 0.0
    order = rng.permutation(len(paths))
    pool, pool_i = [], 0

    def refill():
        nonlocal pool, pool_i, order
        pool = []
        for _ in range(6):
            if pool_i >= len(order):
                order = rng.permutation(len(paths))
                pool_i = 0
            pool.append(load_item(paths[int(order[pool_i])]))
            pool_i += 1

    if resume_path is not None:
        blob = torch.load(resume_path, map_location="cpu", weights_only=False)
        if int(blob.get("seed", a.seed)) != a.seed or int(blob.get("steps", a.steps)) != a.steps:
            raise SystemExit("resume: seed или steps не совпадают с запуском")
        model.load_state_dict(blob["state"])
        opt.load_state_dict(blob["optim"])
        sched.load_state_dict(blob["sched"])
        restore_rng(rng, blob["rng"])
        start = int(blob["step"]) + 1
        best = blob.get("best") or best
        history = list(blob.get("history") or [])
        elapsed_prev = float(blob.get("seconds") or 0.0)
        order = np.asarray(blob["cursor"]["order"])
        pool_i = int(blob["cursor"]["pool_i"])
        pool = [load_item(ROOT / p) for p in blob["cursor"]["pool"]]
        log(f"{a.tag}_RESUME step={blob['step']} from={resume_path}")
    else:
        refill()
        log(f"{a.tag}_START steps={a.steps} batch={a.batch} train_files={len(paths)} "
            f"threads={nthr} max_rows={a.max_rows} generator=static2 arch=rawT_ms_dilated")

    def infer_payload(step: int, proxy_info: dict):
        return dict(
            arch="static6", state=model.state_dict(), cin=model.cin, width=16,
            max_rows=model.max_rows, win_30=model.win_30, win_2h=model.win_2h,
            feat_scale=10.0, dx=DX, step=step,
            ref_dilations=list(REF_DILATIONS), trunk_dilations=list(TRUNK_DILATIONS),
            generator="static2", proxy=proxy_info,
        )

    def save_last(step: int, seconds: float):
        cursor = dict(
            order=np.asarray(order).tolist(),
            pool_i=int(pool_i),
            pool=[str(it["path"].relative_to(ROOT)) for it in pool],
        )
        blob = infer_payload(step, history[-1] if history else {})
        blob.update(dict(
            optim=opt.state_dict(), sched=sched.state_dict(), rng=rng_blob(rng),
            best=best, history=history, seconds=seconds, seed=a.seed, steps=a.steps,
            cursor=cursor,
        ))
        atomic_save(last_path, blob)

    if start > a.steps:
        atomic_save(out, infer_payload(a.steps, history[-1] if history else {}))
        (run_dir / "DONE").touch()
        log(f"{a.tag}_TRAIN_DONE step={a.steps} sec={elapsed_prev} already_complete")
        return

    t0 = time.time()
    last_save_wall = t0
    model.train()
    run_loss = []
    step = start - 1
    for step in range(start, a.steps + 1):
        feats, ys, ws = [], [], []
        guard = 0
        while len(feats) < a.batch and guard < 40:
            guard += 1
            item = pool[int(rng.integers(0, len(pool)))]
            sample = make_sample(item, rng, a.max_rows)
            if sample is None:
                continue
            feats.append(sample[0])
            ys.append(sample[1])
            ws.append(sample[2])
        if len(feats) < 2:
            sched.step()
            continue
        wmax = max(f.shape[1] for f in feats)

        def _pad(arr, w):
            if arr.shape[-1] == w:
                return arr
            pad = np.zeros(arr.shape[:-1] + (w,), arr.dtype)
            pad[..., :arr.shape[-1]] = arr
            return pad

        xb = torch.from_numpy(np.stack([_pad(f, wmax) for f in feats]))
        yb = torch.from_numpy(np.stack([_pad(y, wmax) for y in ys]))
        wb = torch.from_numpy(np.stack([_pad(w, wmax) for w in ws]))
        opt.zero_grad(set_to_none=True)
        logit = model(xb)
        loss, parts = static_loss(logit, yb, wb)
        if not torch.isfinite(loss):
            log(f"{a.tag}_TRAIN nonfinite step={step}")
            sched.step()
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        sched.step()
        run_loss.append(float(loss.detach()))
        seconds = round(elapsed_prev + time.time() - t0, 1)
        if step % 50 == 0:
            print(f"step {step} loss {np.mean(run_loss[-50:]):.4f} pos {parts['pos']:.3f} "
                  f"neg {parts['neg']:.3f} tail {parts['tail']:.3f} lr {sched.get_last_lr()[0]:.5f}",
                  flush=True)
        if step % 200 == 0:
            model.eval()
            pr = proxy(model, val_files, a.seed)
            model.train()
            rec = dict(step=step, seconds=seconds, loss=float(np.mean(run_loss[-50:])), **pr)
            history.append(rec)
            score = pr["strong_recall"] - 40.0 * max(0.0, pr["bg_col_rate"] - 5e-4)
            log(f"{a.tag}_TRAIN step={step} loss={rec['loss']:.4f} strong_recall={pr['strong_recall']:.3f} "
                f"n={pr['n_strong']} bg_col={pr['bg_col_rate']:.5f} sec={seconds}")
            if score > best.get("score", -1e9):
                best = dict(score=score, step=step, **pr)
            if step % 400 == 0:
                snap = out.with_name(f"{out.stem}_s{step}{out.suffix}")
                atomic_save(snap, infer_payload(step, pr))
            if a.early_stop and step >= a.early_stop and pr["strong_recall"] >= 0.9 and pr["bg_col_rate"] <= 5e-4:
                log(f"{a.tag}_TRAIN early_stop step={step}")
                save_last(step, seconds)
                break
        if step % 250 == 0:
            refill()
        now = time.time()
        if step % 200 == 0 or now - last_save_wall >= CKPT_EVERY_S or step == a.steps:
            save_last(step, seconds)
            last_save_wall = now
        if elapsed_prev + (time.time() - t0) > 60 * 60:
            log(f"{a.tag}_TRAIN budget_stop step={step}")
            save_last(step, seconds)
            break
    seconds = round(elapsed_prev + time.time() - t0, 1)
    payload = infer_payload(step, history[-1] if history else {})
    payload["seconds"] = seconds
    payload["proxy_best"] = {k: v for k, v in best.items() if k != "score"}
    atomic_save(out, payload)
    save_last(step, seconds)
    meta = dict(
        steps_ran=step, seconds=seconds, final_step=step,
        proxy_best=payload["proxy_best"], history=history,
        n_train_files=len(paths), crop=CROP, batch=a.batch,
        max_rows=a.max_rows, seed=a.seed, threads=nthr,
        arch="static6",
        generator=dict(
            name="static2", hot_width=list(S2_WIDTH), hot_amp=list(S2_AMP),
            amp_law="log-uniform", kind="70% gaussian / 30% raised_cosine",
            drift=0.15, spot_m=0,
            distractors="broad 18-90 m amp 3-18 p=0.6; spike 0.35-1.3 m amp 4-18 p=0.45",
            trend="add_smooth_trend p=0.7, 50-250 m, 0.4-5 C",
        ),
        train_field="cache residual X[1]*sigma + STATIC2 trend. Train npz has no raw T. "
                    "Features subtract one scalar and do not run a spatial median.",
        resume=str(resume_path) if resume_path else "",
    )
    Path(ROOT / a.log_json).write_text(json.dumps(meta, indent=1))
    (run_dir / "DONE").touch()
    log(f"{a.tag}_TRAIN_DONE step={step} sec={seconds} "
        f"proxy_best_step={payload['proxy_best'].get('step')} "
        f"strong_recall={payload['proxy_best'].get('strong_recall')} "
        f"bg_col={payload['proxy_best'].get('bg_col_rate')}")


if __name__ == "__main__":
    main()
