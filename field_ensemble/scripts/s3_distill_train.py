"""Дистилляция ens_u2ft_tcn_lgbm → узкий DTSUNet (ширина 16). Только synth v8+v10 train.

Карт учителя на train нет: s, h считаются здесь из замороженных
artifacts/stage1/{unet_v810ft,unet_w24,tcn}.pt и pixel_lgbm.joblib
и пишутся в artifacts/ckpt/s3_distill/teacher/. Релиз и stage1 только читаются.

  scripts/run_resumable.sh s3_distill /workspace/repo/run/.venv/bin/python -u scripts/s3_distill_train.py --max-minutes 42
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "NUMEXPR_MAX_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_k] = "2"

import argparse
import copy
import gc
import hashlib
import json
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from dts import classic, data, distill, models

torch.set_num_threads(2)

TILE = (128, 256)
INIT_PT = Path("artifacts/stage1/unet_v810ft.pt")
TEACHER_FILES = (
    Path("artifacts/stage1/unet_v810ft.pt"),
    Path("artifacts/stage1/unet_w24.pt"),
    Path("artifacts/stage1/tcn.pt"),
    Path("artifacts/stage1/pixel_lgbm.joblib"),
)
EVENTS = Path("/workspace/run_logs/_events.log")


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S', time.gmtime())} {msg}", flush=True)


def event(msg: str) -> None:
    line = f"{time.strftime('%H:%M', time.gmtime())} {msg}"
    print(line, flush=True)
    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    with EVENTS.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def plain(obj):
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def dump_json(path: Path, obj) -> None:
    path = distill.assert_safe_output(path, ROOT)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(plain(obj), indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def teacher_hashes() -> dict:
    return {str(p): sha256_file(p) for p in TEACHER_FILES}


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


def freeze_bn(model) -> None:
    model.train()
    for mod in model.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.eval()
            for p in mod.parameters():
                p.requires_grad = False


def param_groups(model, lr_enc: float, lr_dec: float):
    enc, dec = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        top = name.split(".")[0]
        if top in {"e1", "e2", "e3", "e4", "gate"}:
            enc.append(p)
        else:
            dec.append(p)
    if not enc or not dec:
        raise RuntimeError("пустые группы параметров")
    return [{"params": enc, "lr": lr_enc}, {"params": dec, "lr": lr_dec}]


def candidate_rows():
    rows = []
    for ver in ("v8", "v10"):
        df = pd.read_csv(ROOT / f"data/synth_meta/{ver}/sequences.csv")
        have = {p.stem: p for p in data.list_cache(f"artifacts/cache/{ver}", "train")}
        sub = df[df.split == "train"]
        for r in sub.itertuples(index=False):
            if r.seq_id not in have:
                continue
            rows.append(dict(
                ver=ver,
                stem=str(r.seq_id),
                path=have[r.seq_id],
                template=str(r.template),
                n_pix=int(r.n_traces) * int(r.n_x),
                n_events=int(r.n_events),
                proxy=distill.is_proxy(ver, str(r.seq_id)),
            ))
    return rows


def teacher_dst(ckpt: Path, row) -> Path:
    return ckpt / "teacher" / row["ver"] / f"{row['stem']}.npz"


def load_teachers():
    import joblib

    nets = []
    spec = (
        ("unet", 16, TEACHER_FILES[0]),
        ("unet", 24, TEACHER_FILES[1]),
        ("tcn", 32, TEACHER_FILES[2]),
    )
    for arch, width, path in spec:
        net = models.build(arch, width)
        net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        net.eval()
        for p in net.parameters():
            p.requires_grad = False
        nets.append(net)
    gbm = joblib.load(TEACHER_FILES[3])
    try:
        gbm.set_params(n_jobs=2)
    except Exception:
        pass
    return nets, gbm


@torch.inference_mode()
def member_sh(net, X):
    pr = models.predict_full(net, X)
    return (1.0 - pr[0]).astype(np.float32), (pr[1] - pr[2]).astype(np.float32)


def lgbm_sh(gbm, X):
    t_len, w_len = X.shape[1], X.shape[2]
    feat = classic.pixel_features(X)
    pr = gbm.predict_proba(feat).reshape(t_len, w_len, 3)
    return (1.0 - pr[..., 0]).astype(np.float32), (pr[..., 1] - pr[..., 2]).astype(np.float32)


def ensemble_sh(nets, gbm, X):
    parts = [member_sh(net, X) for net in nets]
    parts.append(lgbm_sh(gbm, X))
    return distill.mean_scores(parts)


def verify_teacher(nets, gbm, ckpt: Path) -> dict:
    flag = ckpt / "teacher_check.json"
    if flag.exists():
        rec = json.loads(flag.read_text(encoding="utf-8"))
        if rec.get("ok"):
            log(f"teacher check cached max|ds|={rec['max_abs_s']}")
            return rec
    stem = "seq0115"
    cache = next(p for p in data.list_cache("artifacts/cache/v8", "val") if p.stem == stem)
    X = np.asarray(np.load(cache)["X"], np.float32)
    names = ("unet_v810ft", "unet_w24", "tcn", "pixel_lgbm")
    parts = [member_sh(net, X) for net in nets]
    parts.append(lgbm_sh(gbm, X))
    per = {}
    for name, (s, h) in zip(names, parts):
        ref = np.load(ROOT / "artifacts/pred/v8" / name / "val" / f"{stem}.npz")
        per[name] = dict(
            max_abs_s=float(np.max(np.abs(s - ref["s"].astype(np.float32)))),
            max_abs_h=float(np.max(np.abs(h - ref["h"].astype(np.float32)))),
        )
        log(f"teacher {name} max|ds|={per[name]['max_abs_s']:.3e} max|dh|={per[name]['max_abs_h']:.3e}")
    s, h = distill.mean_scores(parts)
    ens = np.load(ROOT / "artifacts/pred/v8/ens_u2ft_tcn_lgbm/val" / f"{stem}.npz")
    err_s = float(np.max(np.abs(s - ens["s"].astype(np.float32))))
    err_h = float(np.max(np.abs(h - ens["h"].astype(np.float32))))
    ok = err_s <= 1e-4 and err_h <= 5e-4 and all(v["max_abs_s"] <= 1e-4 for v in per.values())
    rec = dict(stem=stem, ok=ok, max_abs_s=err_s, max_abs_h=err_h, members=per)
    dump_json(flag, rec)
    if not ok:
        raise SystemExit(f"учитель не совпал с сохранённым ансамблем: ds={err_s} dh={err_h}")
    log(f"teacher check ok max|ds|={err_s:.3e} max|dh|={err_h:.3e}")
    return rec


def build_one(nets, gbm, row, dst: Path) -> None:
    X = np.asarray(np.load(row["path"])["X"], np.float32)
    s, h = ensemble_sh(nets, gbm, X)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".tmp.npz")
    np.savez_compressed(tmp, s=s.astype(np.float16), h=h.astype(np.float16))
    os.replace(tmp, dst)
    del X, s, h


def load_xy(row):
    ver, stem, path = row["ver"], row["stem"], row["path"]
    xp = ROOT / "artifacts/mm" / ver / "train" / f"{stem}_X.npy"
    yp = ROOT / "artifacts/mm" / ver / "train" / f"{stem}_y.npy"
    if xp.exists() and yp.exists():
        X = np.array(np.load(xp, mmap_mode="r"), copy=True)
        y = np.array(np.load(yp, mmap_mode="r"), copy=True)
    else:
        with np.load(path) as z:
            X = np.asarray(z["X"])
            y = data.target(np.asarray(z["ev"]), np.asarray(z["valid"]))
    with np.load(teacher_dst(row["_ckpt"], row)) as z:
        s = np.asarray(z["s"], np.float16)
        h = np.asarray(z["h"], np.float16)
    if s.shape != y.shape or h.shape != y.shape or X.shape[1:] != y.shape:
        raise RuntimeError(f"форма {stem}: X {X.shape} y {y.shape} s {s.shape}")
    rng = np.random.default_rng(zlib_key(f"{ver}:{stem}"))
    idx = np.argwhere(np.asarray(y) > 0)
    if len(idx) > 4000:
        idx = idx[rng.choice(len(idx), 4000, replace=False)]
    return dict(X=X, y=y, s=s, h=h, ev_idx=idx, stem=stem, ver=ver)


def zlib_key(text: str) -> int:
    import zlib
    return zlib.crc32(text.encode()) & 0xFFFFFFFF


def sample_batch(rng, items, batch: int, tile):
    th, tw = tile
    xs, ys, ss, hs = [], [], [], []
    for _ in range(batch):
        it = items[int(rng.integers(0, len(items)))]
        X, y, s, h = it["X"], it["y"], it["s"], it["h"]
        t_len, w_len = y.shape
        if rng.random() < 0.55 and len(it["ev_idx"]):
            c = it["ev_idx"][int(rng.integers(0, len(it["ev_idx"])))]
            t0 = int(c[0]) - int(rng.integers(0, th))
            w0 = int(c[1]) - int(rng.integers(0, tw))
        else:
            t0 = int(rng.integers(0, max(t_len - th, 0) + 1))
            w0 = int(rng.integers(0, max(w_len - tw, 0) + 1))
        t0 = int(np.clip(t0, 0, max(t_len - th, 0)))
        w0 = int(np.clip(w0, 0, max(w_len - tw, 0)))
        xt = np.zeros((3, th, tw), np.float32)
        yt = np.full((th, tw), -1, np.int64)
        st = np.zeros((th, tw), np.float32)
        ht = np.zeros((th, tw), np.float32)
        a = np.asarray(X[:, t0:t0 + th, w0:w0 + tw], np.float32)
        b = np.asarray(y[t0:t0 + th, w0:w0 + tw])
        ss_ = np.asarray(s[t0:t0 + th, w0:w0 + tw], np.float32)
        hh = np.asarray(h[t0:t0 + th, w0:w0 + tw], np.float32)
        xt[:, :a.shape[1], :a.shape[2]] = a
        yt[:b.shape[0], :b.shape[1]] = b
        st[:ss_.shape[0], :ss_.shape[1]] = ss_
        ht[:hh.shape[0], :hh.shape[1]] = hh
        if rng.random() < 0.5:
            xt = xt[:, :, ::-1].copy()
            yt = yt[:, ::-1].copy()
            st = st[:, ::-1].copy()
            ht = ht[:, ::-1].copy()
        xs.append(np.ascontiguousarray(xt))
        ys.append(np.ascontiguousarray(yt))
        ss.append(np.ascontiguousarray(st))
        hs.append(np.ascontiguousarray(ht))
    return (
        torch.from_numpy(np.stack(xs)),
        torch.from_numpy(np.stack(ys)),
        torch.from_numpy(np.stack(ss)),
        torch.from_numpy(np.stack(hs)),
    )


def make_student(width: int):
    if width > distill.WIDTH_MAX:
        raise SystemExit(f"width {width} > {distill.WIDTH_MAX}")
    model = models.build("unet", width)
    if width == 16:
        model.load_state_dict(torch.load(INIT_PT, map_location="cpu", weights_only=True))
    elif width == 24:
        model.load_state_dict(torch.load(TEACHER_FILES[1], map_location="cpu", weights_only=True))
    freeze_bn(model)
    return model


def save_last(path, model, opt, sched, step, rng, best, history, elapsed, best_state, width: int) -> None:
    path = distill.assert_safe_output(path, ROOT)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": None if sched is None else sched.state_dict(),
        "step": int(step),
        "epoch": 0,
        "rng": rng_blob(rng),
        "best": best,
        "history": history[-80:],
        "elapsed": float(elapsed),
        "best_model": best_state,
        "width": int(width),
    }
    tmp = path.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def export_best(path: Path, state) -> None:
    path = distill.assert_safe_output(path, ROOT)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pt.tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", default="auto", choices=("auto", "off"))
    ap.add_argument("--max-minutes", type=float, default=42.0)
    ap.add_argument("--width", type=int, default=distill.STUDENT_WIDTH)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-teacher", type=int, default=32)
    ap.add_argument("--min-teacher", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--lr-enc", type=float, default=1e-5)
    ap.add_argument("--lr-dec", type=float, default=8e-5)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--w-kl", type=float, default=1.0)
    ap.add_argument("--w-bce", type=float, default=1.0)
    ap.add_argument("--w-hard", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=20261015)
    ap.add_argument("--ckpt-dir", default="artifacts/ckpt/s3_distill")
    ap.add_argument("--out-pt", default="artifacts/eval/s3_distill_unet.pt")
    ap.add_argument("--train-json", default="artifacts/eval/s3_distill_train.json")
    args = ap.parse_args()
    if args.width > distill.WIDTH_MAX or args.width < 4:
        raise SystemExit(f"width должна быть 4..{distill.WIDTH_MAX}")

    ckpt = Path(args.ckpt_dir)
    ckpt.mkdir(parents=True, exist_ok=True)
    last_pt = ckpt / "last.pt"
    clock_path = ckpt / "clock.json"
    done = ckpt / "DONE"
    out_pt = Path(args.out_pt)
    train_json = Path(args.train_json)
    for p in (last_pt, out_pt, train_json, ckpt / "teacher_check.json"):
        distill.assert_safe_output(p, ROOT)

    t0 = time.time()
    elapsed_prev = 0.0
    if args.resume == "auto" and clock_path.exists():
        elapsed_prev = float(json.loads(clock_path.read_text())["elapsed"])
    budget = float(args.max_minutes) * 60.0
    remain = budget - elapsed_prev
    if remain < 30 and done.exists():
        log("бюджет исчерпан и DONE есть")
        return

    def current_elapsed():
        return elapsed_prev + (time.time() - t0)

    def save_clock():
        dump_json(clock_path, {"elapsed": current_elapsed()})

    deadline = time.time() + max(remain, 0.0)
    # Учителю — не больше 14 мин и не меньше 16 мин ученику, если бюджет это позволяет.
    teacher_deadline = min(time.time() + 14 * 60, deadline - 16 * 60)
    hashes_before = teacher_hashes()
    event(
        f"DISTILL_START width={args.width} max_min={args.max_minutes} "
        f"elapsed_prev={elapsed_prev:.0f} remain={remain:.0f} threads=2 real_train=0"
    )

    rows = candidate_rows()
    selected = distill.select_teacher_rows(rows, args.max_teacher)
    for row in selected:
        row["_ckpt"] = ckpt
    train_rows, proxy_rows = distill.split_train_proxy(selected)
    log(f"teacher pool {len(selected)} train {len(train_rows)} proxy {len(proxy_rows)} "
        f"из {len(rows)} кандидатов")
    if len(train_rows) < 1:
        raise SystemExit("нет train-файлов для дистилляции")

    nets, gbm = load_teachers()
    check = verify_teacher(nets, gbm, ckpt)
    done_n = 0
    for i, row in enumerate(selected, 1):
        dst = teacher_dst(ckpt, row)
        if dst.exists() and dst.stat().st_size > 0:
            done_n += 1
            continue
        if done_n >= args.min_teacher and time.time() >= teacher_deadline:
            log(f"teacher stop по времени, готово {done_n}")
            break
        if time.time() >= deadline - 8 * 60 and done_n >= 1:
            log("teacher stop: мало времени до дедлайна")
            break
        t_file = time.time()
        build_one(nets, gbm, row, dst)
        done_n += 1
        save_clock()
        sec = time.time() - t_file
        log(f"teacher {done_n}/{len(selected)} {row['ver']} {row['stem']} {row['template']} {sec:.1f}s")
        if done_n % 4 == 0:
            event(f"DISTILL_TEACHER n={done_n} last={row['ver']}/{row['stem']} sec={sec:.1f}")
    del nets, gbm
    gc.collect()
    ready = [r for r in selected if teacher_dst(ckpt, r).exists()]
    train_rows = [r for r in train_rows if teacher_dst(ckpt, r).exists()]
    proxy_rows = [r for r in proxy_rows if teacher_dst(ckpt, r).exists()]
    if len(train_rows) < 1:
        raise SystemExit("учитель не записал ни одной карты")
    log(f"teacher ready {len(ready)} train {len(train_rows)} proxy {len(proxy_rows)}")

    items, proxy_items = [], []
    for row in train_rows:
        row["_ckpt"] = ckpt
        items.append(load_xy(row))
    for row in proxy_rows:
        row["_ckpt"] = ckpt
        proxy_items.append(load_xy(row))
    log(f"loaded tilesources train={len(items)} proxy={len(proxy_items)}")

    model = make_student(args.width)
    n_param = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(param_groups(model, args.lr_enc, args.lr_dec), weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=4000, eta_min=1e-6)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    step = 0
    best = {"proxy": 1e9, "step": 0}
    history = []
    best_state = copy.deepcopy(model.state_dict())
    if args.resume == "auto" and last_pt.exists():
        blob = torch.load(last_pt, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model"])
        freeze_bn(model)
        opt = torch.optim.AdamW(param_groups(model, args.lr_enc, args.lr_dec), weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=4000, eta_min=1e-6)
        opt.load_state_dict(blob["optimizer"])
        if blob.get("scheduler"):
            sched.load_state_dict(blob["scheduler"])
        rng_restore(blob["rng"], rng)
        step = int(blob["step"])
        best = blob["best"]
        history = list(blob.get("history") or [])
        if blob.get("best_model"):
            best_state = blob["best_model"]
        log(f"resume step={step} best_proxy={best.get('proxy')}")

    def proxy_loss():
        if not proxy_items:
            return None
        freeze_bn(model)
        model.eval()
        losses = []
        with torch.no_grad():
            for _ in range(2):
                x, y, s, h = sample_batch(rng, proxy_items, args.batch, TILE)
                teacher_p = distill.probs_from_sh(s, h)
                loss, _parts = distill.distill_loss(
                    model(x), teacher_p, y,
                    temperature=args.temperature, w_kl=args.w_kl, w_bce=args.w_bce, w_hard=args.w_hard,
                )
                losses.append(float(loss))
        freeze_bn(model)
        return float(np.mean(losses))

    next_save = time.time() + 5 * 60
    stopped = "time"
    while time.time() < deadline and (args.max_steps <= 0 or step < args.max_steps):
        freeze_bn(model)
        opt.zero_grad(set_to_none=True)
        x, y, s, h = sample_batch(rng, items, args.batch, TILE)
        teacher_p = distill.probs_from_sh(s, h)
        loss, parts = distill.distill_loss(
            model(x), teacher_p, y,
            temperature=args.temperature, w_kl=args.w_kl, w_bce=args.w_bce, w_hard=args.w_hard,
        )
        if not torch.isfinite(loss):
            stopped = "nan"
            log(f"nan loss at step {step}")
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        rec = dict(step=step, loss=float(loss.detach()), **parts, sec=round(current_elapsed(), 1))
        if step % 100 == 0 or step == 1:
            pv = proxy_loss()
            rec["proxy"] = pv
            score = rec["loss"] if pv is None else pv
            if score < float(best["proxy"]):
                best = {"proxy": score, "step": step, "held_out": pv is not None}
                best_state = copy.deepcopy(model.state_dict())
                export_best(ckpt / "best.pt", best_state)
            history.append(rec)
            log(f"step={step} loss={rec['loss']:.4f} kl={parts['kl']:.4f} bce={parts['bce']:.4f} "
                f"hard={parts['hard']:.4f} proxy={pv} sec={rec['sec']:.0f}")
            if step % 200 == 0:
                event(f"DISTILL_TRAIN step={step} loss={rec['loss']:.4f} proxy={pv} sec={rec['sec']:.0f}")
        if time.time() >= next_save:
            save_last(last_pt, model, opt, sched, step, rng, best, history, current_elapsed(), best_state, args.width)
            save_clock()
            next_save = time.time() + 5 * 60
            log(f"ckpt step={step}")
    else:
        if args.max_steps > 0 and step >= args.max_steps:
            stopped = "steps"

    save_last(last_pt, model, opt, sched, step, rng, best, history, current_elapsed(), best_state, args.width)
    save_clock()
    export_best(out_pt, best_state)
    hashes_after = teacher_hashes()
    if hashes_after != hashes_before:
        event("DISTILL_TRAIN_FAIL teacher ckpt hash changed")
        raise SystemExit("хеш весов учителя изменился — запись прервана")
    report = dict(
        width=args.width,
        n_param=n_param,
        n_trainable=n_train,
        init=str(INIT_PT if args.width == 16 else "random_or_w24"),
        temperature=args.temperature,
        w_kl=args.w_kl,
        w_bce=args.w_bce,
        w_hard=args.w_hard,
        lr_enc=args.lr_enc,
        lr_dec=args.lr_dec,
        batch=args.batch,
        tile=list(TILE),
        step=step,
        stopped=stopped,
        best=best,
        elapsed_sec=current_elapsed(),
        n_teacher=len(ready),
        n_train_files=len(items),
        n_proxy_files=len(proxy_items),
        teacher_stems=[f"{r['ver']}/{r['stem']}" for r in ready],
        teacher_check=check,
        teacher_sha256=hashes_after,
        real_train=False,
        history=history,
        bn_frozen=True,
        loss="T^2 KL(soften teacher || student) + BCE(s) + 0.25 CE(hard ev)",
    )
    dump_json(train_json, report)
    if stopped == "nan":
        event(f"DISTILL_TRAIN_FAIL nan step={step}")
        raise SystemExit("nan loss")
    done.write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    event(
        f"DISTILL_TRAIN_DONE step={step} stopped={stopped} best_step={best.get('step')} "
        f"best_proxy={best.get('proxy')} sec={current_elapsed():.0f} n_teacher={len(ready)} "
        f"width={args.width} real_train=0"
    )


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        event("DISTILL_TRAIN_FAIL " + traceback.format_exc().splitlines()[-1][:240])
        raise
