"""Дообучение U-Net: focal + Tversky, вес по амплитуде, curriculum малых событий.

Данные — synth v10 train (v11 train и кэш v12, если появятся). v8 val только
как прокси ранней остановки, порог официальной оценки здесь не подбирается.
Реальные трассы не читаются.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import argparse, gc, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_weak_lib as W
from dts import data, models

ROOT = Path(__file__).resolve().parents[1]


def records_for(labels, seq_id):
    from dts import metrics as M
    rows = []
    for r in M.label_boxes(labels, seq_id).itertuples():
        if r.kind != "event":
            continue
        rows.append(dict(
            kind="event", dist_start_m=float(r.dist_start_m), dist_end_m=float(r.dist_end_m),
            ts=float(r.ts), te=float(r.te), amp=float(r.amp_obs_C), mask=int(r.mask_value),
        ))
    return rows


def index_split(cache, split, labels):
    metas = []
    for p in data.list_cache(cache, split):
        z = np.load(p)
        boxes = W.boxes_from_axes(z["x"], z["t"], records_for(labels, p.stem))
        amps = [b["amp"] for b in boxes]
        metas.append(dict(
            path=p, boxes=boxes,
            min_amp=float(min(amps)) if amps else 99.0,
            n_weak=int(sum(a <= 2.0 for a in amps)),
            name=p.stem,
        ))
    return metas


def choose_pool(metas, k, rng, amp_hi):
    weak = [m for m in metas if m["min_amp"] <= amp_hi]
    if len(weak) < 2:
        weak = list(metas)
    n_w = min(len(weak), max(1, int(round(k * 0.85))))
    pick = list(rng.choice(len(weak), size=n_w, replace=False))
    chosen = [weak[i] for i in pick]
    rest = [m for m in metas if all(m["name"] != c["name"] for c in chosen)]
    need = k - len(chosen)
    if need > 0 and rest:
        extra = list(rng.choice(len(rest), size=min(need, len(rest)), replace=False))
        chosen += [rest[i] for i in extra]
    return chosen


def load_item(meta, gamma, amp_hi, rng):
    z = np.load(meta["path"])
    ev = np.array(z["ev"])
    valid = np.array(z["valid"])
    X = np.array(z["X"], dtype=np.float16, copy=True)
    y = data.target(ev, valid)
    w = W.paint_weights(ev, meta["boxes"], gamma)
    small, allc = [], []
    T, Wd = y.shape
    for b in meta["boxes"]:
        t0, t1 = max(b["t0"], 0), min(b["t1"], T)
        x0, x1 = max(b["x0"], 0), min(b["x1"], Wd)
        if t1 <= t0 or x1 <= x0:
            continue
        m = ev[t0:t1, x0:x1] == b["mask"]
        ys, xs = np.nonzero(m)
        if ys.size == 0:
            continue
        if ys.size > 16:
            sel = rng.choice(ys.size, 16, replace=False)
            ys, xs = ys[sel], xs[sel]
        pts = [(int(t0 + yy), int(x0 + xx)) for yy, xx in zip(ys, xs)]
        allc.extend(pts)
        if b["amp"] <= amp_hi:
            small.extend(pts)
    return dict(X=X, y=y, w=w, small=small, all=allc, name=meta["name"])


def sample_batch(pool, batch, tile, rng, p_small, p_event=0.55):
    th, tw = tile
    xs, ys, ws = [], [], []
    for _ in range(batch):
        it = pool[int(rng.integers(0, len(pool)))]
        T, Wd = it["y"].shape
        if rng.random() < p_small and it["small"]:
            c = it["small"][int(rng.integers(0, len(it["small"])))]
            t0 = int(c[0]) - int(rng.integers(0, th))
            w0 = int(c[1]) - int(rng.integers(0, tw))
        elif rng.random() < p_event and it["all"]:
            c = it["all"][int(rng.integers(0, len(it["all"])))]
            t0 = int(c[0]) - int(rng.integers(0, th))
            w0 = int(c[1]) - int(rng.integers(0, tw))
        else:
            t0 = int(rng.integers(0, max(T - th, 0) + 1))
            w0 = int(rng.integers(0, max(Wd - tw, 0) + 1))
        t0 = int(np.clip(t0, 0, max(T - th, 0)))
        w0 = int(np.clip(w0, 0, max(Wd - tw, 0)))
        xt = np.zeros((3, th, tw), np.float32)
        yt = np.full((th, tw), -1, np.int64)
        wt = np.ones((th, tw), np.float32)
        a = np.asarray(it["X"][:, t0:t0 + th, w0:w0 + tw], np.float32)
        b = np.asarray(it["y"][t0:t0 + th, w0:w0 + tw])
        c = np.asarray(it["w"][t0:t0 + th, w0:w0 + tw], np.float32)
        xt[:, :a.shape[1], :a.shape[2]] = a
        yt[:b.shape[0], :b.shape[1]] = b
        wt[:c.shape[0], :c.shape[1]] = c
        wt[yt < 0] = 0
        if rng.random() < 0.5:
            xt = xt[:, :, ::-1].copy()
            yt = yt[:, ::-1].copy()
            wt = wt[:, ::-1].copy()
        g = float(rng.uniform(0.75, 1.3))
        xt[:2] = np.clip(xt[:2] * g, -12, 12)
        xs.append(np.ascontiguousarray(xt))
        ys.append(np.ascontiguousarray(yt))
        ws.append(np.ascontiguousarray(wt))
    return np.stack(xs), np.stack(ys), np.stack(ws)


def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)


# v8 val: файлы с большим числом событий ≤2 °C и файлы с тяжёлым хвостом фона
# ансамбля (они задают глобальный квантиль FAR 1e-3). Тихий seq0111 один
# даёт порог ~0.3 и не похож на полный val.
PROXY_VAL = ["seq0242", "seq0111", "seq0177", "seq0086", "seq0115", "seq0224"]


def proxy_ids(val_metas, n_weak=4, n_plain=2):
    have = {m["name"] for m in val_metas}
    picked = [n for n in PROXY_VAL if n in have]
    if len(picked) >= 4:
        return picked
    ranked = sorted(val_metas, key=lambda m: (-m["n_weak"], m["name"]))
    weak = [m["name"] for m in ranked if m["n_weak"] > 0][:n_weak]
    plain = [m["name"] for m in ranked if m["n_weak"] == 0][:n_plain]
    return weak + plain


def run_proxy(model, cache, labels, names, fixed_thr=None):
    tail = W.ScoreTail(cutoff=0.25)
    # первый проход: порог по фону этих файлов
    packed = []
    for name in names:
        p = Path(cache) / "val" / f"{name}.npz"
        z = np.load(p)
        X = np.asarray(z["X"], np.float32)
        with torch.inference_mode():
            s, h = W.predict_sh(model, X)
        ev, valid, nu = z["ev"], z["valid"], z["nu"]
        tail.add(s[W.bg_mask(ev, valid, nu)])
        packed.append((name, s, h, np.array(ev), np.array(valid), np.array(nu), np.array(z["x"]), np.array(z["t"])))
        del X, z
    thr = tail.threshold(1e-3)
    acc = W.new_acc()
    for name, s, h, ev, valid, nu, x, t in packed:
        W.update_acc(acc, s, h, ev, valid, nu, x, t, name, labels, thr)
    res = W.finalize(acc)
    weak = res["weak_events"]
    out = dict(iou=float(res["iou"]), f1=float(res["f1"]), far=float(res["far"]), thr=float(thr),
               weak_recall=float(weak["recall"] or 0.0), weak_n=int(weak["n"]),
               weak_pixel=float(weak["pixel_recall"] or 0.0), source=tail.last_source,
               by_amp=res.get("by_amp"))
    if fixed_thr is not None:
        acc_f = W.new_acc()
        for name, s, h, ev, valid, nu, x, t in packed:
            W.update_acc(acc_f, s, h, ev, valid, nu, x, t, name, labels, float(fixed_thr))
        fr = W.finalize(acc_f)
        fw = fr["weak_events"]
        out["iou_fixed"] = float(fr["iou"])
        out["far_fixed"] = float(fr["far"])
        out["weak_fixed"] = float(fw["recall"] or 0.0)
        out["fixed_thr"] = float(fixed_thr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="artifacts/stage1/unet_v810ft.pt")
    ap.add_argument("--cache", default="artifacts/cache/v10")
    ap.add_argument("--val-cache", default="artifacts/cache/v8")
    ap.add_argument("--labels", default="data/synth_meta/v10/labels.csv")
    ap.add_argument("--val-labels", default="data/synth_meta/v8/labels.csv")
    ap.add_argument("--out", default="artifacts/s3_weak")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--pool", type=int, default=8)
    ap.add_argument("--tile", default="128,256")
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--max-minutes", type=float, default=28.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-enc", type=float, default=2e-5)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.epochs, a.steps, a.batch, a.pool = 1, 2, 2, 2
        a.max_minutes = 6.0
    os.chdir(ROOT)
    torch.set_num_threads(a.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    import pandas as pd
    labels = pd.read_csv(a.labels)
    val_labels = pd.read_csv(a.val_labels)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    caches = [a.cache]
    v11 = Path("artifacts/cache/v11/train")
    v12 = Path("artifacts/cache/v12/train")
    extra = []
    if any(v11.glob("seq*.npz")):
        caches.append("artifacts/cache/v11")
        extra.append("v11")
    if any(v12.glob("seq*.npz")):
        caches.append("artifacts/cache/v12")
        extra.append("v12")
    print("train caches", caches, "extra", extra or "none", flush=True)
    t0 = time.time()
    metas = []
    for c in caches:
        lab = labels if c == a.cache else pd.read_csv(Path("data/synth_meta") / Path(c).name / "labels.csv")
        part = index_split(c, "train", lab)
        print("indexed", c, len(part), "weak seq", sum(m["n_weak"] > 0 for m in part), flush=True)
        metas.extend(part)
    val_metas = index_split(a.val_cache, "val", val_labels)
    names = proxy_ids(val_metas, 2 if a.smoke else 4, 1 if a.smoke else 2)
    if a.smoke:
        names = names[:1] or [val_metas[0]["name"]]
    print("proxy", names, "index_sec", round(time.time() - t0), flush=True)

    model = models.build("unet", a.width)
    model.load_state_dict(torch.load(a.init, map_location="cpu"))
    init_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    teacher = models.build("unet", a.width)
    teacher.load_state_dict(init_state)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    enc, dec = [], []
    enc_names = {"e1", "e2", "e3", "e4", "gate"}
    for name, mod in model.named_children():
        (enc if name in enc_names else dec).extend(list(mod.parameters()))
    opt = torch.optim.AdamW(
        [{"params": enc, "lr": a.lr_enc, "base": a.lr_enc},
         {"params": dec, "lr": a.lr, "base": a.lr}],
        weight_decay=1e-4,
    )
    cw = torch.tensor([1.0, 3.0, 3.0])
    tile = tuple(int(v) for v in a.tile.split(","))
    rng = np.random.default_rng(a.seed)

    model.eval()
    freeze_bn(model)
    init_proxy = run_proxy(model, a.val_cache, val_labels, names)
    print("init proxy", {k: v for k, v in init_proxy.items() if k != "by_amp"}, "by_amp", init_proxy.get("by_amp"), flush=True)
    fixed_thr = float(init_proxy["thr"])
    best = dict(score=-1e9, state=None, epoch=None, proxy=None, kind=None)
    best_iou = dict(score=-1e9, state=None, epoch=None, proxy=None)
    hist = []
    rolled = False

    def consider(ep, pr, kind_ok):
        score = pr["iou"] + 0.35 * pr["weak_recall"]
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if pr["iou"] > best_iou["score"]:
            best_iou.update(score=pr["iou"], state=state, epoch=ep, proxy=pr)
        if kind_ok and score > best["score"]:
            best.update(score=score, state=state, epoch=ep, proxy=pr, kind="balanced")
            torch.save(state, out / "unet_s3_weak.pt")
            print("save best", ep, "score", round(score, 4), flush=True)

    for ep in range(a.epochs):
        if (time.time() - t0) / 60.0 > a.max_minutes:
            print("time budget", flush=True)
            break
        cur = W.curriculum(ep, a.epochs)
        # косинус к концу, базы lr уже в группах
        scale = 0.5 * (1.0 + np.cos(np.pi * ep / max(a.epochs, 1)))
        scale = float(max(scale, 0.2))
        for g in opt.param_groups:
            g["lr"] = g["base"] * scale
        pool_meta = choose_pool(metas, a.pool, rng, cur["amp_hi"])
        pool = [load_item(m, cur["gamma"], cur["amp_hi"], rng) for m in pool_meta]
        model.train()
        freeze_bn(model)
        parts = []
        tl = []
        for _ in range(a.steps):
            x, y, w = sample_batch(pool, a.batch, tile, rng, cur["p_small"])
            xt = torch.from_numpy(x)
            yt = torch.from_numpy(y)
            logit = model(xt)
            loss, part = W.focal_tversky(logit, yt, torch.from_numpy(w), cw, rank_w=cur["rank_w"])
            with torch.no_grad():
                ref_fg = 1.0 - torch.softmax(teacher(xt), 1)[:, 0]
            fg = 1.0 - torch.softmax(logit, 1)[:, 0]
            anc = W.bg_anchor(fg, ref_fg, yt)
            loss = loss + anc
            part = dict(part, anchor=float(anc.detach()))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl.append(float(loss.detach()))
            parts.append(part)
        del pool
        gc.collect()
        model.eval()
        freeze_bn(model)
        pr = run_proxy(model, a.val_cache, val_labels, names, fixed_thr=fixed_thr)
        far_ok = pr.get("far_fixed", 0.0) <= max(init_proxy["far"] * 1.5, 0.002)
        ok = pr["iou"] >= init_proxy["iou"] - 0.025 and far_ok
        row = dict(epoch=ep, train=float(np.mean(tl)), cur=cur, lr=opt.param_groups[-1]["lr"],
                   proxy=pr, ok=bool(ok), sec=round(time.time() - t0),
                   loss_parts={k: float(np.mean([p[k] for p in parts])) for k in parts[0]})
        hist.append(row)
        print(json.dumps(row), flush=True)
        if ep == 0 and (not rolled) and pr["iou"] < init_proxy["iou"] - 0.05:
            print("rollback epoch 0, lr x0.3", flush=True)
            model.load_state_dict(init_state)
            for g in opt.param_groups:
                g["base"] *= 0.3
            rolled = True
            continue
        consider(ep, pr, ok)
        torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, out / "unet_s3_weak_last.pt")

    if best["state"] is None:
        # ни одна эпоха не уложилась в допуск IoU — берём лучший finetune по proxy IoU
        src = best_iou if best_iou["state"] is not None else dict(state=init_state, epoch=None, proxy=init_proxy, kind="init")
        best.update(state=src["state"], epoch=src.get("epoch"), proxy=src.get("proxy"), kind=src.get("kind", "best_iou"))
        torch.save(best["state"], out / "unet_s3_weak.pt")
        print("fallback save", best["kind"], best["epoch"], flush=True)
    summary = dict(
        args=vars(a), caches=caches, v11_train=bool(extra and "v11" in extra) or any(v11.glob("seq*.npz")),
        v12_train=any(v12.glob("seq*.npz")), real_used=False,
        proxy_ids=names, init_proxy=init_proxy,
        best_epoch=best["epoch"], best_kind=best["kind"], best_proxy=best["proxy"],
        best_iou_epoch=best_iou["epoch"], history=hist,
        init=a.init, seconds=round(time.time() - t0),
    )
    # args содержит Path? нет
    json.dump(summary, open(out / "train_log.json", "w"), indent=1, default=float)
    print("done", best["kind"], best["epoch"], "sec", summary["seconds"], flush=True)


if __name__ == "__main__":
    main()
