"""Обучение FuseNet на подмножестве synth v10 train.

Карты победителя пишутся в /tmp/s3_fusenet_maps и удаляются после успешного обучения.
v8 val в лосс и в раннюю остановку не входит. Прокси — IoU@FAR 1e-3 на 12
последовательностях того же подмножества v10 train.

Запуск:
  scripts/run_resumable.sh s3_fusenet /workspace/repo/run/.venv/bin/python -u scripts/s3_fusenet_train.py
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
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)

import joblib

from dts import classic, data, models
from dts import metrics as M
from s3_fusenet_net import CLIP, FuseNet, fuse_input, n_params, predict_prob

RUN = "s3_fusenet"
CKPT = Path("artifacts/ckpt") / RUN
LAST = CKPT / "last.pt"
STEMS_JSON = CKPT / "stems.json"
MAPDIR = Path("/tmp/s3_fusenet_maps")
PT = Path("artifacts/eval/s3_fusenet.pt")
TRAIN_JSON = Path("artifacts/eval/s3_fusenet_train.json")
KNOTS_JSON = Path("artifacts/eval/s3_mae_train.json")
WINNER_THR_JSON = Path("models_release/stage1_thresholds.json")
LOG = Path("/workspace/grok_logs/_events.log")
N_SEQ = 60
N_HOLD = 12
TILE = (128, 256)
BATCH = 8
STEPS = 40
MAX_EPOCHS = 10
PATIENCE = 3
LR = 1e-3
WD = 1e-4
BUDGET_S = 20 * 60
POS_CAP = 30.0
SAVE_EVERY_S = 15 * 60


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def elog(msg: str) -> None:
    line = time.strftime("%H:%M") + " " + msg
    log(msg)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_knots() -> np.ndarray:
    body = json.loads(KNOTS_JSON.read_text())
    knots = np.asarray(body["cdf_knots"], np.float64).copy()
    if knots.size < 2:
        raise RuntimeError("пустые узлы CDF")
    return knots


def choose_stems() -> dict:
    files = data.list_cache("artifacts/cache/v10", "train")
    if len(files) < N_SEQ:
        raise RuntimeError(f"v10 train {len(files)} < {N_SEQ}")
    rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(len(files), N_SEQ, replace=False))
    chosen = [files[int(i)] for i in pick]
    order = np.random.default_rng(1).permutation(N_SEQ)
    hold_i = set(int(i) for i in order[:N_HOLD])
    train, hold = [], []
    for i, p in enumerate(chosen):
        (hold if i in hold_i else train).append(p.stem)
    body = dict(
        seed_subset=0, seed_split=1, n_seq=N_SEQ, n_hold=N_HOLD,
        train=train, hold=hold,
    )
    STEMS_JSON.parent.mkdir(parents=True, exist_ok=True)
    STEMS_JSON.write_text(json.dumps(body, indent=1), encoding="utf-8")
    return body


def cache_path(stem: str) -> Path:
    return Path("artifacts/cache/v10/train") / f"{stem}.npz"


def load_winner_members():
    th = json.loads(WINNER_THR_JSON.read_text())
    members = []
    for m in th["members"]:
        f = Path("models_release") / m["file"]
        if m["kind"] == "pixel_lgbm":
            mdl = joblib.load(f)
            if hasattr(mdl, "set_params"):
                try:
                    mdl.set_params(n_jobs=2)
                except Exception:
                    pass
            members.append(("lgbm", mdl))
        else:
            net = models.build(m["kind"], m.get("width"))
            net.load_state_dict(torch.load(f, map_location="cpu", weights_only=True))
            net.eval()
            members.append(("net", net))
    if len(members) != 4:
        raise RuntimeError(f"победитель ждёт 4 члена, есть {len(members)}")
    return members


@torch.no_grad()
def winner_score(members, X) -> np.ndarray:
    X = np.asarray(X, np.float32)
    ss = []
    for kind, mdl in members:
        if kind == "net":
            prob = models.predict_full(mdl, X)
        else:
            feat = classic.pixel_features(X)
            prob = np.moveaxis(
                mdl.predict_proba(feat).reshape(X.shape[1], X.shape[2], 3), -1, 0,
            )
        ss.append(1.0 - prob[0])
    return np.mean(ss, 0).astype(np.float32)


def sanity_match(members) -> float:
    """Один файл v8 val: генератор должен совпасть с уже лежащей картой. В обучение не идёт."""
    p = sorted(Path("artifacts/cache/v8/val").glob("seq*.npz"))[0]
    with np.load(p) as z:
        X = np.asarray(z["X"], np.float32)
    got = winner_score(members, X)
    with np.load(Path("artifacts/pred/v8/ens_u2ft_tcn_lgbm/val") / f"{p.stem}.npz") as z:
        ref = np.asarray(z["s"], np.float32)
    err = float(np.max(np.abs(got - ref)))
    log(f"sanity {p.stem} maxabs {err:.3e}")
    if err > 1e-4:
        raise RuntimeError(f"карта победителя не совпала с artifacts/pred: {err}")
    return err


def save_score(path: Path, score: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        np.save(f, score)


def ensure_maps(stems: list[str], members) -> None:
    MAPDIR.mkdir(parents=True, exist_ok=True)
    pending = [s for s in stems if not (MAPDIR / f"{s}.npy").exists()]
    log(f"maps {len(stems) - len(pending)}/{len(stems)} ready, todo {len(pending)}")
    for i, stem in enumerate(pending, 1):
        with np.load(cache_path(stem)) as z:
            X = np.asarray(z["X"], np.float32)
        score = winner_score(members, X)
        save_score(MAPDIR / f"{stem}.npy", score)
        log(f"map {i}/{len(pending)} {stem} {tuple(score.shape)}")


def event_points(ev, valid, cap: int, rng: np.random.Generator) -> np.ndarray:
    idx = np.flatnonzero(valid & (ev > 0))
    if idx.size == 0:
        return np.zeros((0, 2), np.int32)
    if idx.size > cap:
        idx = rng.choice(idx, cap, replace=False)
    t, w = np.unravel_index(idx, ev.shape)
    return np.stack([t, w], 1).astype(np.int32)


def build_index(stems: list[str]) -> tuple[list[dict], float, float]:
    rng = np.random.default_rng(2)
    items = []
    pos = 0
    neg = 0
    for stem in stems:
        path = cache_path(stem)
        with np.load(path) as z:
            ev = np.asarray(z["ev"])
            valid = np.asarray(z["valid"]) > 0.5
        pos += int((valid & (ev > 0)).sum())
        neg += int((valid & (ev == 0)).sum())
        items.append(dict(
            stem=stem, cache=path, score=MAPDIR / f"{stem}.npy",
            shape=tuple(int(v) for v in ev.shape),
            ev_pts=event_points(ev, valid, 4000, rng),
        ))
    raw = float(neg / max(pos, 1))
    pw = float(min(max(raw, 1.0), POS_CAP))
    log(f"train pixels pos {pos} neg {neg} pos_weight {pw:.2f} raw {raw:.1f}")
    return items, pw, raw


def crop_example(feat, ev, valid, t0, w0):
    th, tw = TILE
    T, W = ev.shape
    patch = np.zeros((4, th, tw), np.float32)
    y = np.zeros((th, tw), np.float32)
    m = np.zeros((th, tw), np.float32)
    ht = min(th, T - t0)
    wt = min(tw, W - w0)
    patch[:, :ht, :wt] = feat[:, t0:t0 + ht, w0:w0 + wt]
    ev_c = ev[t0:t0 + ht, w0:w0 + wt]
    va = valid[t0:t0 + ht, w0:w0 + wt]
    y[:ht, :wt] = ((ev_c > 0) & va).astype(np.float32)
    m[:ht, :wt] = va.astype(np.float32)
    return patch, y, m


def sample_batch(items, knots, rng: np.random.Generator, pw: float):
    xs, ys, ms = [], [], []
    th, tw = TILE
    for _ in range(BATCH):
        it = items[int(rng.integers(0, len(items)))]
        with np.load(it["cache"]) as z:
            X = np.asarray(z["X"])
            ev = np.asarray(z["ev"])
            valid = np.asarray(z["valid"]) > 0.5
        score = np.load(it["score"])
        feat = fuse_input(score, X, knots)
        T, W = ev.shape
        use_ev = it["ev_pts"].shape[0] > 0 and float(rng.random()) < 0.5
        if use_ev:
            t, w = it["ev_pts"][int(rng.integers(0, it["ev_pts"].shape[0]))]
            t, w = int(t), int(w)
        else:
            t = int(rng.integers(0, T))
            w = int(rng.integers(0, W))
        if T <= th:
            t0 = 0
        else:
            t0 = int(np.clip(t - int(rng.integers(0, th)), 0, T - th))
        if W <= tw:
            w0 = 0
        else:
            w0 = int(np.clip(w - int(rng.integers(0, tw)), 0, W - tw))
        patch, y, msk = crop_example(feat, ev, valid, t0, w0)
        xs.append(patch)
        ys.append(y)
        ms.append(msk)
    xb = torch.from_numpy(np.stack(xs))
    yb = torch.from_numpy(np.stack(ys))
    mb = torch.from_numpy(np.stack(ms))
    return xb, yb, mb, pw


def counts_at(prob, ev, valid, nu, thr: float) -> dict:
    pred = prob >= thr
    truth = valid & (ev > 0)
    bg = valid & (ev == 0) & (nu == 0)
    nup = valid & (ev == 0) & (nu > 0)
    tp = int((pred & truth).sum())
    fn = int((~pred & truth).sum())
    fp_bg = int((pred & bg).sum())
    fp_nu = int((pred & nup).sum())
    return dict(tp=tp, fn=fn, fp_bg=fp_bg, fp_nu=fp_nu, n_bg=int(bg.sum()), bg=prob[bg])


def proxy_iou(model, stems: list[str], knots) -> dict:
    bags = []
    rows = []
    for stem in stems:
        with np.load(cache_path(stem)) as z:
            X = z["X"]
            ev = np.asarray(z["ev"])
            valid = np.asarray(z["valid"]) > 0.5
            nu = np.asarray(z["nu"])
        score = np.load(MAPDIR / f"{stem}.npy")
        prob = predict_prob(model, fuse_input(score, X, knots))
        row = counts_at(prob, ev, valid, nu, thr=1.0)
        bags.append(row.pop("bg"))
        rows.append(row)
    bg = np.concatenate(bags) if bags else np.zeros(1, np.float32)
    thr = M.far_threshold(bg, 1e-3)
    tp = fn = fp_bg = fp_nu = n_bg = 0
    for stem, row in zip(stems, rows):
        with np.load(cache_path(stem)) as z:
            X = z["X"]
            ev = np.asarray(z["ev"])
            valid = np.asarray(z["valid"]) > 0.5
            nu = np.asarray(z["nu"])
        score = np.load(MAPDIR / f"{stem}.npy")
        prob = predict_prob(model, fuse_input(score, X, knots))
        c = counts_at(prob, ev, valid, nu, thr)
        tp += c["tp"]; fn += c["fn"]; fp_bg += c["fp_bg"]; fp_nu += c["fp_nu"]; n_bg += c["n_bg"]
    fp = fp_bg + fp_nu
    iou = float(tp / max(tp + fp + fn, 1))
    return dict(
        iou=iou, threshold=float(thr), tp=tp, fp=fp, fn=fn,
        fp_bg=fp_bg, fp_nu=fp_nu, n_bg=n_bg,
        far=float(fp_bg / max(n_bg, 1)),
    )


def rng_blob(rng: np.random.Generator) -> dict:
    return dict(
        torch=torch.get_rng_state(),
        numpy=np.random.get_state(),
        py=random.getstate(),
        gen=rng.bit_generator.state,
    )


def restore_rng(blob, rng: np.random.Generator) -> None:
    torch.set_rng_state(blob["torch"])
    np.random.set_state(blob["numpy"])
    random.setstate(blob["py"])
    rng.bit_generator.state = blob["gen"]


def save_last(model, opt, epoch, step, rng, best, history, spent, pw) -> None:
    CKPT.mkdir(parents=True, exist_ok=True)
    payload = dict(
        model=model.state_dict(), optimizer=opt.state_dict(),
        epoch=int(epoch), step=int(step), rng=rng_blob(rng),
        best=best, history=history, elapsed=float(spent), pos_weight=float(pw),
    )
    tmp = LAST.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    tmp.replace(LAST)


def save_best_pt(model) -> None:
    PT.parent.mkdir(parents=True, exist_ok=True)
    tmp = PT.with_suffix(".pt.tmp")
    torch.save(model.state_dict(), tmp)
    tmp.replace(PT)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", default="auto", choices=("auto", "never", "must"))
    args = ap.parse_args()
    torch.set_num_threads(2)
    if CKPT.joinpath("DONE").exists() and args.resume != "never":
        log("DONE already")
        return
    if args.resume == "never":
        if LAST.exists():
            LAST.unlink()
        if STEMS_JSON.exists():
            STEMS_JSON.unlink()
    knots = load_knots()
    if STEMS_JSON.exists() and args.resume != "never":
        stems = json.loads(STEMS_JSON.read_text())
    else:
        stems = choose_stems()
    all_stems = list(stems["train"]) + list(stems["hold"])
    for stem in all_stems:
        path = cache_path(stem)
        if "train" not in path.parts:
            raise RuntimeError(f"не train: {path}")
    members = load_winner_members()
    err = sanity_match(members)
    ensure_maps(all_stems, members)
    del members
    t_wall = time.time()
    items, pw, raw_pw = build_index(stems["train"])
    model = FuseNet()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    rng = np.random.default_rng(0)
    start_ep = 0
    step = 0
    history = []
    best = dict(epoch=-1, iou=-1.0)
    elapsed0 = 0.0
    if args.resume == "must" and not LAST.exists():
        raise SystemExit("resume must: нет last.pt")
    if args.resume != "never" and LAST.exists():
        ck = torch.load(LAST, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        start_ep = int(ck["epoch"]) + 1
        step = int(ck["step"])
        history = list(ck.get("history") or [])
        best = dict(ck.get("best") or best)
        elapsed0 = float(ck.get("elapsed") or 0.0)
        pw = float(ck.get("pos_weight") or pw)
        restore_rng(ck["rng"], rng)
        log(f"resume epoch {ck['epoch']} best {best}")
    elif start_ep == 0 and int(best["epoch"]) < 0 and not history:
        pr0 = proxy_iou(model, stems["hold"], knots)
        best = dict(epoch=-1, iou=float(pr0["iou"]), threshold=pr0["threshold"], far=pr0["far"])
        save_best_pt(model)
        elog(f"FUSENET_INIT proxy_iou={pr0['iou']:.4f} far={pr0['far']:.5f} params={n_params(model)}")
    bad = sum(1 for h in history if int(h["epoch"]) > int(best["epoch"]))
    spent = lambda: elapsed0 + (time.time() - t_wall)
    last_save = time.time()
    stopped = "epochs"
    pw_t = torch.tensor(pw, dtype=torch.float32)
    for epoch in range(start_ep, MAX_EPOCHS):
        if spent() > BUDGET_S:
            stopped = "time"
            break
        model.train()
        losses = []
        for _ in range(STEPS):
            xb, yb, mb, _ = sample_batch(items, knots, rng, pw)
            opt.zero_grad(set_to_none=True)
            logit = model(xb)[:, 0]
            loss_map = F.binary_cross_entropy_with_logits(logit, yb, pos_weight=pw_t, reduction="none")
            denom = mb.sum().clamp(min=1.0)
            loss = (loss_map * mb).sum() / denom
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            losses.append(float(loss))
            if time.time() - last_save > SAVE_EVERY_S:
                save_last(model, opt, epoch, step, rng, best, history, spent(), pw)
                last_save = time.time()
                log(f"last.pt step {step}")
        pr = proxy_iou(model, stems["hold"], knots)
        rec = dict(epoch=epoch, loss=float(np.mean(losses)), sec=round(spent(), 1), **pr)
        history.append(rec)
        log(f"epoch {epoch} loss {rec['loss']:.4f} proxy {pr['iou']:.4f} far {pr['far']:.5f}")
        if pr["iou"] > float(best["iou"]) + 1e-4:
            best = dict(epoch=epoch, iou=float(pr["iou"]), threshold=pr["threshold"], far=pr["far"])
            bad = 0
            save_best_pt(model)
            log(f"saved epoch {epoch}")
        else:
            bad += 1
        save_last(model, opt, epoch, step, rng, best, history, spent(), pw)
        last_save = time.time()
        if bad >= PATIENCE and epoch >= 1:
            stopped = "patience"
            break
    # Веса оценки — лучший прокси, не последняя эпоха.
    if PT.exists():
        model.load_state_dict(torch.load(PT, map_location="cpu", weights_only=True))
    body = dict(
        name="FUSENET",
        arch="conv3x3",
        params=n_params(model),
        channels=list(("s_winner", "cdf_abs_dT_space", "abs_dT_time_over_12", "valid")),
        clip=CLIP,
        kernels=3,
        n_seq=N_SEQ,
        n_train=len(stems["train"]),
        n_hold=len(stems["hold"]),
        train_stems=stems["train"],
        hold_stems=stems["hold"],
        data="v10 train subset only",
        val_used_for_train=False,
        real_traces_used=False,
        knots=str(KNOTS_JSON),
        n_knots=int(knots.size),
        winner_sanity_maxabs=err,
        pos_weight=pw,
        pos_weight_raw=raw_pw,
        lr=LR,
        weight_decay=WD,
        batch=BATCH,
        steps_per_epoch=STEPS,
        tile=list(TILE),
        best=best,
        history=history,
        stopped=stopped,
        seconds=round(float(spent()), 1),
        maps_dir=str(MAPDIR),
    )
    TRAIN_JSON.parent.mkdir(parents=True, exist_ok=True)
    tmp = TRAIN_JSON.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(body, indent=1), encoding="utf-8")
    tmp.replace(TRAIN_JSON)
    shutil.rmtree(MAPDIR, ignore_errors=True)
    body["maps_deleted"] = not MAPDIR.exists()
    TRAIN_JSON.write_text(json.dumps(body, indent=1), encoding="utf-8")
    (CKPT / "DONE").touch()
    elog(
        f"FUSENET_TRAIN_DONE best_epoch={best.get('epoch')} proxy_iou={best.get('iou')} "
        f"stop={stopped} sec={spent():.0f} params={n_params(model)} "
        f"n_train={len(stems['train'])} n_hold={len(stems['hold'])} real_train=0 maps_deleted=1"
    )


if __name__ == "__main__":
    main()
