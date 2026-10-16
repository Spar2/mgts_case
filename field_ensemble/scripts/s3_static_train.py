"""Обучение StaticNet только на синтетике v8/v10 train.

Вклейка статичных горбов — на лету, кэши не изменяются. Реальные CSV не читаются.
Поток CPU: torch.set_num_threads(3).
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("MKL_NUM_THREADS", "3")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "3")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "3")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dts.static_net import (
    LONG_BINS, LONG_ROWS, N_FEAT, StaticNet, predict_map, residual_from_cache, static_loss,
    temporal_features,
)
from dts.static_synth import (
    DX, EDGE_RANGE, ENV_LO, ENV_HI, HOT_AMP, HOT_WIDTH, SPOT_RANGE,
    apply_baseline, apply_profiles, apply_row_gaps, fixed_hot_specs, sample_baseline,
    sample_gap, sample_train_specs, seq_seed,
)

CROP = 1280
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


def make_sample(item, rng, max_rows: int, n_bins: int):
    R = np.asarray(item["R"], np.float32)
    valid = item["valid"]
    rows, width = R.shape
    if width < 256 or rows < 16:
        return None
    t1 = rows
    t0 = 0
    # Длинное окно: чаще вся запись. Короткий кроп оставляем редко, не короче 192 кадров.
    if rows > max_rows and rng.random() < 0.25:
        span = int(rng.integers(max(192, max_rows // 2), rows + 1))
        t0 = int(rng.integers(0, rows - span + 1))
        t1 = t0 + span
    avoid = event_columns(item["ev"][t0:t1])
    specs = sample_train_specs(rng, width, DX, avoid)
    field = R[t0:t1]
    # Копия: пропуски не должны портить кэш и не обнуляют температуру.
    v = np.array(valid[t0:t1], copy=True)
    baseline = sample_baseline(rng)
    gap = sample_gap(rng)
    field, pos, hard = apply_profiles(field, specs)
    field = apply_baseline(field, baseline, DX)
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
    v = apply_row_gaps(v, gap)
    feats, good = temporal_features(field, v, max_rows=max_rows, n_bins=n_bins)
    y = pos.astype(np.float32)
    w = np.ones(y.shape, np.float32)
    w[~good] = 0
    w[avoid & ~pos] = 0
    # естественные пики фона и широкие/узкие отвлечения — трудные отрицательные
    w[hard] = 4.0
    peak = good & ~pos & (feats[0] > 0.25)
    w[peak] = np.maximum(w[peak], 3.0)
    w[pos] = 1.0
    return feats, y, w


def proxy(model, files, seed_base: int):
    """Грубая проверка: фон почти молчит, горбы ≥ 8 °C находятся при пороге 0.8."""
    hits, n_ev, fired, cols = [], 0, 0, 0
    for p in files:
        z = np.load(p)
        R, valid = residual_from_cache(z)
        ev = np.asarray(z["ev"])
        p0 = predict_map(model, R, valid)
        bg = (ev == 0).all(0) & (valid.mean(0) >= 0.5)
        fired += int((p0[bg] >= 0.90).sum())
        cols += int(bg.sum())
        avoid = event_columns(ev)
        specs = fixed_hot_specs(R.shape[1], seq_seed(seed_base, p.stem), n_events=2, avoid=avoid)
        strong = [sp for sp in specs if sp["amp"] >= 8.0]
        if not strong:
            continue
        Rp, _, _ = apply_profiles(R, strong)
        pp = predict_map(model, Rp, valid)
        for sp in strong:
            from dts.static_synth import unit_profile
            cols_ev = unit_profile(R.shape[1], sp) >= 0.5
            if not cols_ev.any():
                continue
            n_ev += 1
            frac = float((pp[cols_ev] >= 0.80).mean())
            hits.append(frac >= 0.3)
    rec = float(np.mean(hits)) if hits else 0.0
    rate = fired / max(cols, 1)
    return dict(strong_recall=rec, n_strong=n_ev, bg_col_rate=rate)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=20261015)
    ap.add_argument("--max-rows", type=int, default=LONG_ROWS)
    ap.add_argument("--n-bins", type=int, default=LONG_BINS)
    ap.add_argument("--early-stop", type=int, default=0,
                    help="Шаг, после которого можно остановиться. 0 — дойти до --steps.")
    ap.add_argument("--tag", default="STATIC2")
    ap.add_argument("--out", default="artifacts/static/static_net2.pt")
    ap.add_argument("--log-json", default="artifacts/static/train_log2.json")
    a = ap.parse_args()
    torch.set_num_threads(3)
    torch.manual_seed(a.seed)
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
    cin = N_FEAT + (a.n_bins if a.n_bins > 0 else 0)
    model = StaticNet(cin=cin, max_rows=a.max_rows, n_bins=a.n_bins)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)

    def save_ckpt(path: Path, step: int, proxy_info: dict):
        torch.save(dict(
            state=model.state_dict(), cin=model.cin, width=16, pool=int(model.ref.pool),
            feat_scale=10.0, dx=DX, step=step, max_rows=model.max_rows, n_bins=model.n_bins,
            proxy=proxy_info,
        ), path)

    log(f"{a.tag}_START steps={a.steps} batch={a.batch} train_files={len(paths)} "
        f"threads=3 max_rows={a.max_rows} n_bins={a.n_bins}")
    t0 = time.time()
    pool, pool_i = [], 0
    order = rng.permutation(len(paths))
    best = None
    history = []

    def refill():
        nonlocal pool, pool_i, order
        pool = []
        for _ in range(6):
            if pool_i >= len(order):
                order = rng.permutation(len(paths))
                pool_i = 0
            pool.append(load_item(paths[int(order[pool_i])]))
            pool_i += 1

    refill()
    model.train()
    run_loss = []
    for step in range(1, a.steps + 1):
        frac = step / a.steps
        lr = a.lr * (0.15 + 0.85 * 0.5 * (1 + np.cos(np.pi * frac)))
        for g in opt.param_groups:
            g["lr"] = float(lr)
        feats, ys, ws = [], [], []
        guard = 0
        while len(feats) < a.batch and guard < 40:
            guard += 1
            item = pool[int(rng.integers(0, len(pool)))]
            sample = make_sample(item, rng, a.max_rows, a.n_bins)
            if sample is None:
                continue
            feats.append(sample[0])
            ys.append(sample[1])
            ws.append(sample[2])
        if len(feats) < 2:
            continue
        wmax = max(f.shape[1] for f in feats)

        def _pad(a, w):
            if a.shape[-1] == w:
                return a
            out = np.zeros(a.shape[:-1] + (w,), a.dtype)
            out[..., :a.shape[-1]] = a
            return out

        xb = torch.from_numpy(np.stack([_pad(f, wmax) for f in feats]))
        yb = torch.from_numpy(np.stack([_pad(y, wmax) for y in ys]))
        wb = torch.from_numpy(np.stack([_pad(w, wmax) for w in ws]))
        opt.zero_grad(set_to_none=True)
        logit = model(xb)
        loss, parts = static_loss(logit, yb, wb)
        if not torch.isfinite(loss):
            log(f"{a.tag}_TRAIN nonfinite step={step}")
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        run_loss.append(float(loss.detach()))
        if step % 50 == 0:
            print(f"step {step} loss {np.mean(run_loss[-50:]):.4f} pos {parts['pos']:.3f} "
                  f"neg {parts['neg']:.3f} tail {parts['tail']:.3f} lr {lr:.5f}", flush=True)
        if step % 200 == 0:
            model.eval()
            pr = proxy(model, val_files, a.seed)
            model.train()
            rec = dict(step=step, seconds=round(time.time() - t0, 1),
                       loss=float(np.mean(run_loss[-50:])), **pr)
            history.append(rec)
            score = pr["strong_recall"] - 40.0 * max(0.0, pr["bg_col_rate"] - 5e-4)
            log(f"{a.tag}_TRAIN step={step} loss={rec['loss']:.4f} strong_recall={pr['strong_recall']:.3f} "
                f"n={pr['n_strong']} bg_col={pr['bg_col_rate']:.5f} sec={rec['seconds']}")
            if best is None or score > best["score"]:
                best = dict(score=score, step=step, **pr)
            if step % 400 == 0:
                snap = out.with_name(f"{out.stem}_s{step}{out.suffix}")
                save_ckpt(snap, step, pr)
            if a.early_stop and step >= a.early_stop and pr["strong_recall"] >= 0.9 and pr["bg_col_rate"] <= 5e-4:
                log(f"{a.tag}_TRAIN early_stop step={step}")
                break
        if step % 250 == 0:
            refill()
    if best is None:
        best = dict(step=step, score=0, strong_recall=None, bg_col_rate=None)
    # На диск кладётся последний шаг. Выбор чекпойнта — по static_val на оценке, не по прокси.
    save_ckpt(out, step, history[-1] if history else {})
    ckpt = torch.load(out, map_location="cpu", weights_only=False)
    ckpt["seconds"] = round(time.time() - t0, 1)
    ckpt["proxy_best"] = {k: v for k, v in best.items() if k != "score"}
    torch.save(ckpt, out)
    meta = dict(steps_ran=step, seconds=ckpt["seconds"], final_step=step,
                proxy_best=ckpt["proxy_best"], history=history,
                n_train_files=len(paths), crop=CROP, batch=a.batch,
                max_rows=a.max_rows, n_bins=a.n_bins, seed=a.seed,
                generator=dict(hot_width=list(HOT_WIDTH), hot_amp=list(HOT_AMP),
                               spot_m=list(SPOT_RANGE), edge_m=list(EDGE_RANGE),
                               env=(ENV_LO, ENV_HI)))
    Path(ROOT / a.log_json).write_text(json.dumps(meta, indent=1))
    log(f"{a.tag}_TRAIN_DONE step={step} sec={ckpt['seconds']} "
        f"proxy_best_step={ckpt['proxy_best'].get('step')} "
        f"strong_recall={ckpt['proxy_best'].get('strong_recall')} "
        f"bg_col={ckpt['proxy_best'].get('bg_col_rate')}")


if __name__ == "__main__":
    main()
