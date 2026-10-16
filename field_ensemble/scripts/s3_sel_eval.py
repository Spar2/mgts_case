"""SEL: переоценка ens_cnsym и взвешенный ансамбль чекпойнтов.

Порог FAR 1e-3 только по фону v8 val. Test и реальные csv не двигают веса и порог.
Реальные файлы — описание карточек, без обучения и без калибровки.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import s3_sel_core as sel
from dts import data, events as EV, metrics as M, models

torch.set_num_threads(2)

ROOT = Path(__file__).resolve().parents[1]
NAMES = [
    "unet", "unet_s2", "unet_w24", "unet_v810ft", "unet_nm",
    "unet_cn", "unet_cnsym", "unet_cnsymlong", "tcn", "pixel_lgbm",
]
INFER = ("unet_s2", "unet_nm")
NEURAL = [n for n in NAMES if n != "pixel_lgbm"]
WINNER = "ens_u2ft_tcn_lgbm"
CNSYM = "ens_cnsym"
PUB_WIN_VAL = 0.38671235821914546
PUB_WIN_TEST = 0.4052762065230473
BG_CAP = 200_000
HIT = sel.HIT_FRAC


def elog(msg: str) -> None:
    line = time.strftime("%H:%M") + " " + msg
    print(line, flush=True)
    path = Path("/workspace/run_logs/_events.log")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def conv(o):
    if isinstance(o, dict):
        return {str(k): conv(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [conv(v) for v in o]
    if isinstance(o, np.ndarray):
        return conv(o.tolist())
    if isinstance(o, np.generic):
        return o.item()
    return o


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(conv(obj), indent=1, ensure_ascii=False), encoding="utf-8")


def load_nets():
    nets = {}
    for name in INFER:
        net = models.build("unet", 16)
        net.load_state_dict(torch.load(ROOT / "artifacts/stage1" / f"{name}.pt", map_location="cpu"))
        net.eval()
        nets[name] = net
    return nets


def _indices(ev, valid, nu, stem, cap=BG_CAP):
    valid = np.asarray(valid, bool)
    ev = np.asarray(ev)
    nu = np.asarray(nu)
    ev_idx = np.flatnonzero(valid & (ev > 0))
    nu_idx = np.flatnonzero(valid & (ev == 0) & (nu > 0))
    bg_all = np.flatnonzero(valid & (ev == 0) & (nu == 0))
    rng = np.random.default_rng(zlib.crc32(stem.encode()) & 0xFFFFFFFF)
    if bg_all.size > cap:
        bg_idx = rng.choice(bg_all, cap, replace=False)
    else:
        bg_idx = bg_all
    return ev_idx, nu_idx, bg_idx, int(bg_all.size)


def _take(arr, ev_idx, nu_idx, bg_idx):
    flat = np.asarray(arr, np.float32).ravel()
    return flat[ev_idx], flat[nu_idx], flat[bg_idx]


def _events_for(stem, ev, x, t, labels, seqs, ev_idx):
    """Индексы событий в компактной оси ev_idx и плоские индексы кадра."""
    cooling = []
    rows = []
    if ev_idx.size == 0:
        return cooling, rows
    template = str(seqs.loc[stem, "template"]) if stem in seqs.index else ""
    for r in M.label_boxes(labels, stem).itertuples():
        if r.kind != "event":
            continue
        px = M.event_pixels(r, x, t, ev)
        if not px.any():
            continue
        flat = np.flatnonzero(px)
        loc = np.searchsorted(ev_idx, flat)
        ok = loc < ev_idx.size
        loc = loc[ok]
        flat_ok = flat[ok]
        ok2 = ev_idx[loc] == flat_ok
        loc = loc[ok2].astype(np.int32)
        flat_ok = flat_ok[ok2].astype(np.int32)
        if loc.size == 0:
            continue
        rec = dict(label=str(r.label), amp=float(r.amp_obs_C), template=template, loc=loc, flat=flat_ok)
        rows.append(rec)
        if r.label == "cooling":
            cooling.append(loc)
    return cooling, rows


def load_compact(cache, pred, labels, seqs, nets, split="val"):
    frames = []
    infer_full = {n: {} for n in INFER}
    files = data.list_cache(cache, split)
    t0 = time.time()
    for i, path in enumerate(files):
        stem = path.stem
        with np.load(path) as z:
            ev = np.asarray(z["ev"])
            valid = np.asarray(z["valid"])
            nu = np.asarray(z["nu"])
            x = np.asarray(z["x"], np.float64)
            t = np.asarray(z["t"], np.float64)
            ev_idx, nu_idx, bg_idx, n_bg = _indices(ev, valid, nu, stem)
            cols_e, cols_n, cols_b = [], [], []
            X = np.asarray(z["X"], np.float32)
            for name in NAMES:
                if name in INFER:
                    pr = models.predict_full(nets[name], X)
                    s = (1.0 - pr[0]).astype(np.float32)
                    h = (pr[1] - pr[2]).astype(np.float32)
                    infer_full[name][stem] = (s.astype(np.float16), h.astype(np.float16))
                    del pr
                else:
                    q = np.load(pred / name / split / path.name)
                    s = np.asarray(q["s"], np.float32)
                    h = None
                if s.shape != ev.shape:
                    raise SystemExit(f"форма {name} {s.shape} != ev {ev.shape} {stem}")
                e, n, b = _take(s, ev_idx, nu_idx, bg_idx)
                cols_e.append(e.astype(np.float16))
                cols_n.append(n.astype(np.float16))
                cols_b.append(b.astype(np.float16))
                del s
            del X
            cooling, events = _events_for(stem, ev, x, t, labels, seqs, ev_idx)
        frames.append(dict(
            stem=stem,
            ev_m=np.stack(cols_e, 0),
            nu_m=np.stack(cols_n, 0),
            bg_m=np.stack(cols_b, 0),
            n_bg=n_bg,
            cooling=cooling,
            events=events,
        ))
        if i % 5 == 0 or i + 1 == len(files):
            print(f"compact {split} {i+1}/{len(files)} {stem} {round(time.time()-t0)}s", flush=True)
    return frames, infer_full


def subset(frames, stems):
    want = set(stems)
    return [fr for fr in frames if fr["stem"] in want]


def row_of(name, metrics, extra=None):
    d = dict(
        name=name,
        iou=float(metrics["iou"]),
        cooling_recall=float(metrics["cooling_recall"]),
        cooling_hits=int(metrics["cooling_hits"]),
        cooling_n=int(metrics["cooling_n"]),
        combined=float(metrics["combined"]),
        f1=float(metrics["f1"]),
        precision=float(metrics["precision"]),
        recall=float(metrics["recall"]),
        threshold=float(metrics["threshold"]),
        far=float(metrics["far"]),
    )
    if extra:
        d.update(extra)
    return d


def fit_select_stems(frames):
    stems = [fr["stem"] for fr in frames]
    fit, select = [], []
    for i, s in enumerate(stems):
        (select if i % 3 == 0 else fit).append(s)
    return fit, select


def search(frames):
    fit_stems, sel_stems = fit_select_stems(frames)
    fit_fr = subset(frames, fit_stems)
    sel_fr = subset(frames, sel_stems)
    print(f"split fit {len(fit_fr)} select {len(sel_fr)}", flush=True)

    member_rows = []
    for i, name in enumerate(NAMES):
        w = np.zeros(len(NAMES))
        w[i] = 1.0
        full = sel.evaluate_frames(frames, w)
        on_fit = sel.evaluate_frames(fit_fr, w)
        member_rows.append(dict(
            name=name, neural=name != "pixel_lgbm",
            val=row_of(name, full), fit=row_of(name, on_fit),
        ))
        print(f"member {name} val IoU {full['iou']:.4f} cool {full['cooling_hits']}/{full['cooling_n']} "
              f"fit IoU {on_fit['iou']:.4f} cool {on_fit['cooling_hits']}/{on_fit['cooling_n']}", flush=True)

    def pack(name, wdict, mode="mean"):
        return dict(name=name, weights=sel.weights_from_array(NAMES, [float(wdict.get(n, 0.0)) for n in NAMES]), mode=mode)

    fixed = [
        pack("equal_all", {n: 1.0 for n in NAMES}),
        pack("equal_neural", {n: 1.0 for n in NEURAL}),
        pack("winner4", sel.recipe_weights(NAMES, ["unet_w24", "unet_v810ft", "tcn", "pixel_lgbm"])),
        pack("cnsym4", sel.recipe_weights(NAMES, ["unet_w24", "unet_cnsym", "tcn", "pixel_lgbm"])),
        pack("cn4", sel.recipe_weights(NAMES, ["unet_w24", "unet_cn", "tcn", "pixel_lgbm"])),
        pack("seeds_unet", sel.recipe_weights(NAMES, ["unet", "unet_s2", "unet_w24", "unet_v810ft", "unet_nm", "unet_cn"])),
        pack("sym_wide", sel.recipe_weights(NAMES, ["unet_w24", "unet_cn", "unet_cnsym", "unet_cnsymlong", "unet_v810ft", "unet_nm", "unet_s2", "tcn", "pixel_lgbm"])),
        pack("winner4_logit", sel.recipe_weights(NAMES, ["unet_w24", "unet_v810ft", "tcn", "pixel_lgbm"]), "logit"),
        pack("neural_logit", {n: 1.0 for n in NEURAL}, "logit"),
        pack("neural_geomean", {n: 1.0 for n in NEURAL}, "geomean"),
    ]

    fit_iou = np.array([r["fit"]["iou"] for r in member_rows])
    fit_cool = np.array([r["fit"]["cooling_recall"] for r in member_rows])
    adaptive = []
    for temp in (0.03, 0.1):
        adaptive.append(pack(f"soft_iou_{temp}", sel.softmax_weights(NAMES, fit_iou, temp)))
        adaptive.append(pack(f"soft_sum_{temp}", sel.softmax_weights(NAMES, fit_iou + fit_cool, temp)))

    cache = {}

    def ev_on(frs, mode):
        def fn(wdict):
            vec = sel.as_vector(NAMES, wdict)
            key = (mode, tuple(np.round(vec, 5)))
            if key not in cache:
                cache[key] = sel.evaluate_frames(frs, vec, mode=mode)
            return cache[key]
        return fn

    fit_mean = ev_on(fit_fr, "mean")
    diri = sel.dirichlet_bank(len(NAMES), 40, seed=7, concentration=0.7)
    diri_rows = []
    for i, vec in enumerate(diri):
        wdict = sel.weights_from_array(NAMES, vec)
        m = fit_mean(wdict)
        diri_rows.append(row_of(f"diri_{i}", m, extra=dict(weights=wdict, mode="mean")))
    diri_pick, _, _ = sel.pick_band(diri_rows)
    best_iou_diri = max(diri_rows, key=lambda r: (r["iou"], r["cooling_recall"]))
    adaptive.append(pack("diri_band", diri_pick["weights"]))
    if best_iou_diri["name"] != diri_pick["name"]:
        adaptive.append(pack("diri_iou", best_iou_diri["weights"]))

    inits = {
        "winner4": fixed[2]["weights"],
        "cnsym4": fixed[3]["weights"],
        "equal_neural": fixed[1]["weights"],
        "best_single": sel.recipe_weights(NAMES, [max(member_rows, key=lambda r: r["fit"]["combined"])["name"]]),
    }
    asc_specs = [
        ("asc_sum_winner4", "winner4", "combined"),
        ("asc_sum_cnsym4", "cnsym4", "combined"),
        ("asc_iou_winner4", "winner4", "iou"),
        ("asc_iou_neural", "equal_neural", "iou"),
    ]
    for name, init_key, obj in asc_specs:
        w, _ = sel.coordinate_ascent(lambda wd, obj=obj: fit_mean(wd)[obj], NAMES, inits[init_key])
        adaptive.append(pack(name, w))
        print("ascent", name, "fit", {k: round(v, 3) for k, v in fit_mean(w).items() if k in ("iou", "cooling_recall", "combined")}, flush=True)

    mlp_model, mlp_alpha = train_mlp(fit_fr)
    mlp_frames = apply_mlp_frames(frames, mlp_model)
    mlp_sel = subset(mlp_frames, sel_stems)
    mlp_m = sel.evaluate_frames(mlp_sel, np.array([1.0]))
    adaptive_rows = []
    for sp in adaptive:
        m = sel.evaluate_frames(sel_fr, sel.as_vector(NAMES, sp["weights"]), mode=sp["mode"])
        adaptive_rows.append(row_of(sp["name"], m, extra=dict(weights=sp["weights"], mode=sp["mode"])))
        print(f"select {sp['name']} IoU {m['iou']:.4f} cool {m['cooling_hits']}/{m['cooling_n']}", flush=True)
    adaptive_rows.append(row_of("mlp_stack", mlp_m, extra=dict(weights={n: float(a) for n, a in zip(NAMES, mlp_alpha)}, mode="mlp")))
    print(f"select mlp_stack IoU {mlp_m['iou']:.4f} cool {mlp_m['cooling_hits']}/{mlp_m['cooling_n']}", flush=True)
    adaptive_winner, _, adaptive_band = sel.pick_band(adaptive_rows)
    print("adaptive", adaptive_winner["name"], "select IoU", round(adaptive_winner["iou"], 4),
          "cool", adaptive_winner["cooling_hits"], "/", adaptive_winner["cooling_n"], flush=True)

    # Компактный val всех фиксированных и одного адаптивного победителя — только скрининг.
    screen = []
    specs = list(fixed)
    if adaptive_winner["name"] == "mlp_stack":
        m = sel.evaluate_frames(mlp_frames, np.array([1.0]))
        screen.append(row_of("mlp_stack", m, extra=dict(weights=adaptive_winner["weights"], mode="mlp", family="adaptive")))
    else:
        win_sp = next(sp for sp in adaptive if sp["name"] == adaptive_winner["name"])
        specs.append(win_sp)
    for sp in specs:
        m = sel.evaluate_frames(frames, sel.as_vector(NAMES, sp["weights"]), mode=sp["mode"])
        family = "adaptive" if sp["name"] == adaptive_winner["name"] else "fixed"
        screen.append(row_of(sp["name"], m, extra=dict(weights=sp["weights"], mode=sp["mode"], family=family)))
        print(f"screen {sp['name']} IoU {m['iou']:.4f} cool {m['cooling_hits']}/{m['cooling_n']}", flush=True)
    screen_pick, screen_best, _ = sel.pick_band(screen)
    return dict(
        fit_stems=fit_stems, select_stems=sel_stems, members=member_rows,
        adaptive_select=adaptive_rows, adaptive_winner=adaptive_winner["name"],
        adaptive_band=[r["name"] for r in adaptive_band],
        screen=screen, screen_pick=screen_pick["name"], screen_best_iou=screen_best,
        mlp_alpha={n: float(a) for n, a in zip(NAMES, mlp_alpha)},
        mlp_model=mlp_model,
    )


def train_mlp(frames, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    xs, ys, ws = [], [], []
    mdim = frames[0]["ev_m"].shape[0]
    for fr in frames:
        ev = fr["ev_m"]
        cool = np.unique(np.concatenate([np.asarray(c, np.int64) for c in fr["cooling"]])) if fr["cooling"] else np.zeros(0, np.int64)
        other = np.setdiff1d(np.arange(ev.shape[1]), cool)
        if other.size > 4000:
            other = rng.choice(other, 4000, replace=False)
        take = np.concatenate([cool, other]).astype(np.int64) if (cool.size or other.size) else np.zeros(0, np.int64)
        if take.size:
            xs.append(np.asarray(ev[:, take].T, np.float32))
            ys.append(np.ones(take.size, np.float32))
            ww = np.ones(take.size, np.float32)
            if cool.size:
                ww[np.isin(take, cool)] = 4.0
            ws.append(ww)
        for key, cap in (("bg_m", 8000), ("nu_m", 2000)):
            block = fr[key]
            n = block.shape[1]
            if n == 0:
                continue
            cols = rng.choice(n, min(n, cap), replace=False)
            xs.append(np.asarray(block[:, cols].T, np.float32))
            ys.append(np.zeros(len(cols), np.float32))
            ws.append(np.ones(len(cols), np.float32))
    X = np.concatenate(xs)
    y = np.concatenate(ys)
    w = np.concatenate(ws)
    pos = y > 0.5
    w[pos] *= float(w[~pos].sum() / max(w[pos].sum(), 1e-6))
    model = torch.nn.Sequential(torch.nn.Linear(mdim, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
    xt = torch.from_numpy(X)
    yt = torch.from_numpy(y)
    wt = torch.from_numpy(w)
    for ep in range(20):
        perm = torch.randperm(len(xt))
        loss_sum = 0.0
        for i in range(0, len(xt), 8192):
            idx = perm[i:i + 8192]
            opt.zero_grad()
            logit = model(xt[idx]).squeeze(1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, yt[idx], weight=wt[idx])
            loss.backward()
            opt.step()
            loss_sum += float(loss.detach()) * len(idx)
        if ep == 0 or ep == 19:
            print(f"mlp epoch {ep} loss {loss_sum / len(xt):.4f}", flush=True)
    model.eval()
    lin = model[0].weight.detach().numpy()
    imp = np.abs(lin).sum(0)
    alpha = imp / max(imp.sum(), 1e-12)
    return model, alpha


def _sigmoid_rows(model, feats):
    """feats (M, N) → (N,)"""
    if feats.shape[1] == 0:
        return np.zeros(0, np.float32)
    out = np.empty(feats.shape[1], np.float32)
    with torch.no_grad():
        step = 250_000
        for i in range(0, feats.shape[1], step):
            batch = torch.from_numpy(np.ascontiguousarray(feats[:, i:i + step].T, np.float32))
            out[i:i + step] = torch.sigmoid(model(batch)).reshape(-1).numpy()
    return out


def apply_mlp_frames(frames, model):
    out = []
    for fr in frames:
        out.append(dict(
            fr,
            ev_m=_sigmoid_rows(model, fr["ev_m"].astype(np.float32))[None, :],
            nu_m=_sigmoid_rows(model, fr["nu_m"].astype(np.float32))[None, :],
            bg_m=_sigmoid_rows(model, fr["bg_m"].astype(np.float32))[None, :],
        ))
    return out


def new_bucket():
    return dict(tot=np.zeros(3, np.float64), hot=np.zeros(3, np.float64), cold=np.zeros(3, np.float64),
                far_n=0, far_d=0, events=[])


def add_maps(bucket, s, h, ev, valid, nu, x, t, stem, thr, labels, seqs):
    valid = np.asarray(valid, bool)
    ev = np.asarray(ev)
    nu = np.asarray(nu)
    pred = (s >= thr) & valid
    truth = ev > 0
    bucket["tot"] += np.asarray(M.confusion(pred, truth, valid), np.float64)
    bucket["hot"] += np.asarray(M.confusion(pred & (h > 0), (ev >= 1) & (ev <= 4), valid), np.float64)
    bucket["cold"] += np.asarray(M.confusion(pred & (h <= 0), ev >= 5, valid), np.float64)
    b = valid & (ev == 0) & (nu == 0)
    bucket["far_n"] += int((pred & b).sum())
    bucket["far_d"] += int(b.sum())
    template = str(seqs.loc[stem, "template"]) if stem in seqs.index else ""
    for r in M.label_boxes(labels, stem).itertuples():
        if r.kind != "event":
            continue
        px = M.event_pixels(r, x, t, ev)
        if not px.any():
            continue
        frac = float(pred[px].mean())
        bucket["events"].append(dict(
            label=str(r.label), amp=float(r.amp_obs_C), template=template,
            frac=frac, hit=bool(frac >= HIT),
            tp=int((pred & px).sum()), fn=int((~pred & px).sum()),
        ))


def finalize_bucket(bucket, thr, zones=None):
    stats = M.prf(*bucket["tot"])
    out = dict(
        threshold=float(thr),
        far=float(bucket["far_n"] / max(bucket["far_d"], 1)),
        **stats,
        hot=M.prf(*bucket["hot"]),
        cold=M.prf(*bucket["cold"]),
    )
    E = pd.DataFrame(bucket["events"])
    def agg(g):
        tp = float(g.tp.sum())
        fn = float(g.fn.sum())
        return dict(n=int(len(g)), recall=float(g.hit.mean()) if len(g) else 0.0,
                    pixel_recall=float(tp / max(tp + fn, 1.0)))
    if len(E):
        out["events_all"] = agg(E)
        out["by_label"] = {str(k): agg(g) for k, g in E.groupby("label")}
        out["by_amp"] = {str(k): agg(g) for k, g in E.groupby(E.amp.map(M.amp_bin))}
        out["weak_events"] = agg(E[E.amp <= 2])
        cool = out["by_label"].get("cooling", dict(n=0, recall=0.0, pixel_recall=0.0))
    else:
        cool = dict(n=0, recall=0.0, pixel_recall=0.0)
        out["events_all"] = dict(n=0, recall=0.0, pixel_recall=0.0)
        out["by_label"] = {}
        out["weak_events"] = dict(n=0, recall=0.0, pixel_recall=0.0)
    out["cooling_recall"] = float(cool["recall"])
    out["cooling_n"] = int(cool["n"])
    out["cooling_hits"] = int(round(float(cool["recall"]) * int(cool["n"])))
    out["cooling_pixel_recall"] = float(cool.get("pixel_recall", 0.0))
    out["combined"] = sel.combined_score(out["iou"], out["cooling_recall"])
    if zones is not None:
        out["zones"] = zones
    return out


def iter_meta(cache, split):
    for path in data.list_cache(cache, split):
        with np.load(path) as z:
            yield path.stem, np.asarray(z["ev"]), np.asarray(z["valid"]), np.asarray(z["nu"]), np.asarray(z["x"], np.float64), np.asarray(z["t"], np.float64)


def mix_spatial(s_maps, h_maps, names, weights, mode, shape):
    w = sel.as_vector(names, weights)
    s = np.zeros(shape, np.float32)
    h = np.zeros(shape, np.float32)
    if mode == "mlp":
        raise RuntimeError("mlp смешивается отдельно")
    if mode == "mean":
        for n, wi in zip(names, w):
            if wi <= 0:
                continue
            s += wi * s_maps[n]
            h += wi * h_maps[n]
        return s, h
    acc = np.zeros(shape, np.float32)
    for n, wi in zip(names, w):
        if wi <= 0:
            continue
        p = np.clip(s_maps[n], 1e-4, 1.0 if mode == "geomean" else 1.0 - 1e-4)
        acc += wi * (np.log(p) if mode == "geomean" else (np.log(p) - np.log1p(-p)))
        h += wi * h_maps[n]
    if mode == "geomean":
        return np.exp(acc), h
    if mode == "logit":
        return 1.0 / (1.0 + np.exp(-acc)), h
    raise ValueError(mode)


def load_member_maps(pred, split, stem, names, infer_bank):
    s_maps, h_maps = {}, {}
    for name in names:
        if infer_bank is not None and name in infer_bank and stem in infer_bank[name]:
            s16, h16 = infer_bank[name][stem]
            s_maps[name] = np.asarray(s16, np.float32)
            h_maps[name] = np.asarray(h16, np.float32)
        else:
            q = np.load(pred / name / split / f"{stem}.npz")
            s_maps[name] = np.asarray(q["s"], np.float32)
            h_maps[name] = np.asarray(q["h"], np.float32)
    return s_maps, h_maps


def needed_names(specs):
    names = []
    for sp in specs:
        if sp.get("saved"):
            continue
        if sp.get("mode") == "mlp":
            for n in NAMES:
                if n not in names:
                    names.append(n)
            continue
        for n, w in sp["weights"].items():
            if w > 1e-8 and n not in names and n in NAMES:
                names.append(n)
    return names


def spec_maps(sp, s_maps, h_maps, shape, mlp_model=None):
    if sp.get("saved"):
        return s_maps[sp["saved"]], h_maps[sp["saved"]]
    if sp["mode"] == "mlp":
        arr = np.stack([s_maps[n].ravel() for n in NAMES], 0)
        s = _sigmoid_rows(mlp_model, arr).reshape(shape)
        alpha = sel.as_vector(NAMES, sp["weights"])
        h = sum(float(a) * h_maps[n] for n, a in zip(NAMES, alpha))
        return s, h
    return mix_spatial(s_maps, h_maps, NAMES, sp["weights"], sp["mode"], shape)


def exact_specs(cache, pred, split, specs, labels, seqs, infer_bank=None, mlp_model=None, zones_for=None):
    """Два прохода: гистограмма порога, затем метрики. Для официальных строк порог ещё раз по полной сортировке."""
    edges = np.linspace(0.0, 1.0, 20001)
    hists = {sp["name"]: np.zeros(len(edges) - 1, np.int64) for sp in specs}
    files = list(iter_meta(cache, split))
    names = needed_names(specs)
    saved = [sp["saved"] for sp in specs if sp.get("saved")]
    t0 = time.time()
    for i, (stem, ev, valid, nu, x, t) in enumerate(files):
        s_maps, h_maps = ({}, {})
        if names:
            s_maps, h_maps = load_member_maps(pred, split, stem, names, infer_bank)
        for method in saved:
            q = np.load(pred / method / split / f"{stem}.npz")
            s_maps[method] = np.asarray(q["s"], np.float32)
            h_maps[method] = np.asarray(q["h"], np.float32)
        bg = np.asarray(valid, bool) & (ev == 0) & (nu == 0)
        for sp in specs:
            s, _h = spec_maps(sp, s_maps, h_maps, ev.shape, mlp_model)
            hists[sp["name"]] += np.histogram(np.clip(s[bg], 0.0, 1.0), bins=edges)[0]
        if i % 15 == 0:
            print(f"hist {split} {i+1}/{len(files)} {round(time.time()-t0)}s", flush=True)
    thrs = {name: sel.threshold_from_hist(hists[name], edges, sel.FAR) for name in hists}
    buckets = {sp["name"]: new_bucket() for sp in specs}
    zone_acc = {sp["name"]: dict(n_zones=0, n_false=0, kmh=0.0) for sp in specs} if zones_for else {}
    for stem, ev, valid, nu, x, t in files:
        s_maps, h_maps = ({}, {})
        if names:
            s_maps, h_maps = load_member_maps(pred, split, stem, names, infer_bank)
        for method in saved:
            q = np.load(pred / method / split / f"{stem}.npz")
            s_maps[method] = np.asarray(q["s"], np.float32)
            h_maps[method] = np.asarray(q["h"], np.float32)
        valid_b = np.asarray(valid, bool)
        for sp in specs:
            s, h = spec_maps(sp, s_maps, h_maps, ev.shape, mlp_model)
            add_maps(buckets[sp["name"]], s, h, ev, valid_b, nu, x, t, stem, thrs[sp["name"]], labels, seqs)
            if zones_for and sp["name"] in zones_for:
                zs = EV.zones_from_prob(s, h, valid_b, x, t, thrs[sp["name"]])
                acc = zone_acc[sp["name"]]
                for z in zs:
                    sl = (slice(z["t0"], z["t1"]), slice(z["x0"], z["x1"]))
                    comp = (s[sl] >= thrs[sp["name"]]) & valid_b[sl]
                    acc["n_zones"] += 1
                    acc["n_false"] += int((ev[sl][comp] > 0).mean() < 0.1) if comp.any() else 1
                acc["kmh"] += (float(x[-1] - x[0]) / 1000.0) * (float(t[-1] - t[0]) / 3600.0)
    out = {}
    for sp in specs:
        zones = None
        if sp["name"] in zone_acc:
            acc = zone_acc[sp["name"]]
            zones = dict(
                n_zones=acc["n_zones"], n_false=acc["n_false"],
                false_zones_per_km_h=acc["n_false"] / max(acc["kmh"], 1e-9),
                false_frac=acc["n_false"] / max(acc["n_zones"], 1),
            )
        metrics = finalize_bucket(buckets[sp["name"]], thrs[sp["name"]], zones)
        metrics["threshold_kind"] = "hist_20000"
        out[sp["name"]] = metrics
    return out


def confirm_sort_threshold(cache, pred, split, sp, labels, seqs, infer_bank=None, mlp_model=None, with_zones=False, thr=None):
    """Порог — порядковый статистик фона val. На test передаётся уже замороженный thr."""
    names = [] if sp.get("saved") else needed_names([sp])
    saved = [sp["saved"]] if sp.get("saved") else []
    if thr is None:
        bg_parts = []
        for stem, ev, valid, nu, x, t in iter_meta(cache, split):
            s_maps, h_maps = ({}, {})
            if names:
                s_maps, h_maps = load_member_maps(pred, split, stem, names, infer_bank)
            for method in saved:
                q = np.load(pred / method / split / f"{stem}.npz")
                s_maps[method] = np.asarray(q["s"], np.float32)
                h_maps[method] = np.asarray(q["h"], np.float32)
            s, _h = spec_maps(sp, s_maps, h_maps, ev.shape, mlp_model)
            bg = np.asarray(valid, bool) & (ev == 0) & (nu == 0)
            bg_parts.append(np.asarray(s[bg], np.float32))
        thr = sel.far_threshold(np.concatenate(bg_parts), sel.FAR)
        del bg_parts
    thr = float(thr)
    bucket = new_bucket()
    acc = dict(n_zones=0, n_false=0, kmh=0.0)
    for stem, ev, valid, nu, x, t in iter_meta(cache, split):
        s_maps, h_maps = ({}, {})
        if names:
            s_maps, h_maps = load_member_maps(pred, split, stem, names, infer_bank)
        for method in saved:
            q = np.load(pred / method / split / f"{stem}.npz")
            s_maps[method] = np.asarray(q["s"], np.float32)
            h_maps[method] = np.asarray(q["h"], np.float32)
        s, h = spec_maps(sp, s_maps, h_maps, ev.shape, mlp_model)
        valid_b = np.asarray(valid, bool)
        add_maps(bucket, s, h, ev, valid_b, nu, x, t, stem, thr, labels, seqs)
        if with_zones:
            zs = EV.zones_from_prob(s, h, valid_b, x, t, thr)
            for z in zs:
                sl = (slice(z["t0"], z["t1"]), slice(z["x0"], z["x1"]))
                comp = (s[sl] >= thr) & valid_b[sl]
                acc["n_zones"] += 1
                acc["n_false"] += int((ev[sl][comp] > 0).mean() < 0.1) if comp.any() else 1
            acc["kmh"] += (float(x[-1] - x[0]) / 1000.0) * (float(t[-1] - t[0]) / 3600.0)
    zones = None
    if with_zones:
        zones = dict(
            n_zones=acc["n_zones"], n_false=acc["n_false"],
            false_zones_per_km_h=acc["n_false"] / max(acc["kmh"], 1e-9),
            false_frac=acc["n_false"] / max(acc["n_zones"], 1),
        )
    metrics = finalize_bucket(bucket, thr, zones)
    metrics["threshold_kind"] = "sort"
    return metrics


def slim_metrics(m):
    cool = (m.get("by_label") or {}).get("cooling", {})
    coldspot = (m.get("by_label") or {}).get("coldspot", {})
    return dict(
        threshold=m["threshold"], threshold_kind=m.get("threshold_kind"),
        far=m["far"], iou=m["iou"], f1=m["f1"], precision=m["precision"], recall=m["recall"],
        hot_iou=m.get("hot", {}).get("iou"), cold_iou=m.get("cold", {}).get("iou"),
        cold_recall=m.get("cold", {}).get("recall"),
        cooling_recall=m["cooling_recall"], cooling_hits=m["cooling_hits"], cooling_n=m["cooling_n"],
        cooling_pixel_recall=m.get("cooling_pixel_recall"),
        coldspot_recall=coldspot.get("recall"), coldspot_n=coldspot.get("n"),
        weak=m.get("weak_events"), combined=m["combined"],
        by_label=m.get("by_label"), zones=m.get("zones"),
    )


def run_real(weights, mode, thr, mlp_model, out_path):
    """Оценка на MGTS. Порог заморожен. Карточки не участвуют в весах."""
    from dts import features, grid, infer, real
    cards = []
    for p in (ROOT / "data/operator_cards/cards_batch1.jsonl", ROOT / "data/operator_cards/cards_batch2.jsonl"):
        cards += [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    positive = [(n, float(w)) for n, w in weights.items() if float(w) > 1e-8 and n in NAMES]
    nets = {}
    gbm = None
    for name, _w in positive:
        if name == "pixel_lgbm":
            import joblib
            gbm = joblib.load(ROOT / "artifacts/stage1/pixel_lgbm.joblib")
            continue
        if name == "tcn":
            arch, width = "tcn", 32
        elif name == "unet_w24":
            arch, width = "unet", 24
        else:
            arch, width = "unet", 16
        net = models.build(arch, width)
        net.load_state_dict(torch.load(ROOT / "artifacts/stage1" / f"{name}.pt", map_location="cpu"))
        net.eval()
        nets[name] = (arch, net)
    analyzer = infer.Analyzer(op="far_1e-3")
    root = Path("/workspace/mgts/MGTS_data")
    files_out = {}
    card_rows = []
    vec = sel.as_vector(NAMES, weights) if mode != "mlp" else None
    alpha = sel.as_vector(NAMES, weights) if mode == "mlp" else None
    for csv_path in sorted(root.glob("*.csv")):
        rec = real.load_csv(csv_path)
        name = rec["name"]
        t1 = time.time()
        plan = grid.plan(rec["x"], rec["t"])
        Tc, v = grid.resample(rec["T"], rec["valid"], plan, "mean")
        X, aux = features.make_channels(Tc, v)
        s_maps, h_maps = {}, {}
        for mname, _w in positive:
            if mname == "pixel_lgbm":
                from dts import classic
                pr = gbm.predict_proba(classic.pixel_features(X)).reshape(X.shape[1], X.shape[2], 3)
                s_maps[mname] = (1.0 - pr[..., 0]).astype(np.float32)
                h_maps[mname] = (pr[..., 1] - pr[..., 2]).astype(np.float32)
            else:
                pr = models.predict_full(nets[mname][1], X)
                s_maps[mname] = (1.0 - pr[0]).astype(np.float32)
                h_maps[mname] = (pr[1] - pr[2]).astype(np.float32)
        if mode == "mlp":
            # недостающие члены с нулевым вкладом не грузились; для MLP нужен полный вектор.
            # На real MLP не гоняем, если режим mlp: горячность и оценка по альфе только загруженных.
            # Скрипт вызывает run_real только для mean/logit/geomean.
            raise SystemExit("real для mlp не собран: на csv нет сохранённых карт всех членов")
        s, h = mix_spatial(s_maps, h_maps, [n for n, _ in positive], {n: w for n, w in positive}, mode, Tc.shape)
        zones = EV.zones_from_prob(s, h, v, plan["x"], plan["t"], thr)
        probs = analyzer.classify(X, aux["sigma_x"], plan["x"], plan["t"], zones)
        for z, prb in zip(zones, probs):
            z["class"] = EV.CLASSES[int(np.argmax(prb))]
            z["class_prob"] = {c: round(float(q), 3) for c, q in zip(EV.CLASSES, prb)}
            z["alarm"] = z["class"] not in ("HVAC_Diurnal", "Sensor_Artifact")
        cls = {}
        for z in zones:
            cls[z["class"]] = cls.get(z["class"], 0) + 1
        n_alarm = sum(1 for z in zones if z["alarm"])
        files_out[name] = dict(
            sha256=rec["sha256"], threshold=float(thr), shape=list(rec["T"].shape),
            grid_shape=list(v.shape), n_zones=len(zones), n_alarm=n_alarm, classes=cls,
        )
        for c in cards:
            if str(c.get("record")) != name:
                continue
            card_rows.append(card_overlap(c, zones, s, v, plan, thr))
        print(f"real {name} zones {len(zones)} alarm {n_alarm} {round(time.time()-t1)}s", flush=True)
        del X, Tc, s, h, s_maps, h_maps
    yes = [r for r in card_rows if r.get("presence") == "yes" and r.get("status") == "ok"]
    no = [r for r in card_rows if r.get("presence") == "no" and r.get("status") == "ok"]
    summary = dict(
        yes_alarm=int(sum(r["hit_alarm"] for r in yes)), yes_n=len(yes),
        no_alarm=int(sum(r["hit_alarm"] for r in no)), no_n=len(no),
        note="Карточки только описывают пересечение зон с рамкой. Порог и веса по ним не подбирались.",
    )
    payload = dict(
        op="far_1e-3", threshold=float(thr), mode=mode, weights=weights,
        calib="frozen_v8_val", trained_on_real=False, files=files_out, cards=card_rows, presence=summary,
    )
    dump_json(out_path, payload)
    return payload


def card_overlap(card, zones, score, valid, plan, thr):
    import pandas as pd
    E0 = pd.Timestamp("1970-01-01", tz="UTC")
    ts = []
    for s in card["time"]:
        stamp = pd.Timestamp(s)
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize("UTC")
        else:
            stamp = stamp.tz_convert("UTC")
        ts.append(stamp)
    ta, tb = [(s - E0).total_seconds() for s in ts]
    xa, xb = card["meters"]
    x = plan["x"]
    t = plan["t"]
    ix = (x >= xa) & (x <= xb)
    it = (t >= ta) & (t <= tb)
    if not ix.any() or not it.any():
        return dict(id=card["id"], record=str(card["record"]), presence=card["presence"], status="box_outside_record")
    box = np.ix_(it, ix)
    vb = valid[box]
    alarm = [z for z in zones if z.get("alarm") and z["end_m"] >= xa and z["start_m"] <= xb and z["t_end_s"] >= ta and z["t_start_s"] <= tb]
    any_z = [z for z in zones if z["end_m"] >= xa and z["start_m"] <= xb and z["t_end_s"] >= ta and z["t_start_s"] <= tb]
    return dict(
        id=card["id"], record=str(card["record"]), presence=card["presence"], shape=card.get("shape", ""),
        status="ok", hit_alarm=bool(alarm), hit_zone=bool(any_z), n_alarm=len(alarm), n_zones=len(any_z),
        max_score=float(score[box][vb].max()) if vb.any() else None,
        frac_flagged=float((score[box] >= thr)[vb].mean()) if vb.any() else None,
    )


def write_md(path: Path, report: dict) -> None:
    ch = report["choice"]
    lines = []
    lines.append("# SEL. Ансамбль весов и cooling на val")
    lines.append("")
    lines.append("Рабочая точка этапа 1 по-прежнему считается по val IoU при FAR = 1e-3, порог — квантиль фона v8 val. "
                 "Test и реальные файлы в отбор не входят. Внутри полосы 0.005 по IoU выше ставится recall событий с меткой `cooling` "
                 "(событие найдено, если помечено не меньше 30 % его пикселей). Сумма IoU и этого recall записывается, но одна не выбирает строку: "
                 "на val таких событий шесть, шаг 1/6 больше типичного разрыва IoU.")
    lines.append("")
    lines.append("Члены — карты уже обученных сетей и пиксельного LightGBM из `artifacts/stage1` и `artifacts/pred/v8`. "
                 "`unet_s2` (seed 2) и `unet_nm` посчитаны на лету с чекпойнтов, без новой подгонки и без записи больших npz. "
                 "Пространственный детектор |Z| в смесь не входил. Веса неотрицательные и в сумме 1, кроме режима logit/geomean, где это те же коэффициенты смеси. "
                 "Случайные и покоординатные веса учились на 30 последовательностях val (каждая третья, по порядку имени, отдана в отбор). "
                 "Маленький MLP (линейный слой 10→16, ReLU, логит) учился только на этой обучающей трети, с повышенным весом пикселей cooling. "
                 "В полный val он попадал лишь если выигрывал отборочную треть по правилу полосы.")
    lines.append("")
    lines.append("## Члены на полном val, компактная оценка")
    lines.append("")
    lines.append("Фон здесь — подвыборка до 200 000 клеток на файл, ложные на чистом фоне масштабируются. Это скрининг, не официальная строка.")
    lines.append("")
    lines.append("| член | val IoU | cooling | fit IoU | fit cooling |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for r in report["members"]:
        lines.append(f"| `{r['name']}` | {r['val']['iou']:.4f} | {r['val']['cooling_hits']}/{r['val']['cooling_n']} | {r['fit']['iou']:.4f} | {r['fit']['cooling_hits']}/{r['fit']['cooling_n']} |")
    lines.append("")
    lines.append("## Официальные строки, порог по полной сортировке фона")
    lines.append("")
    lines.append("| метод | val IoU | val cooling | val cold IoU | test IoU | test cooling | test F1 | test FAR |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, block in report["official"].items():
        v, t = block["val"], block.get("test")
        if not t:
            lines.append(
                f"| `{name}` | {v['iou']:.4f} | {v['cooling_hits']}/{v['cooling_n']} | {v['cold_iou']:.4f} | — | — | — | — |"
            )
            continue
        lines.append(
            f"| `{name}` | {v['iou']:.4f} | {v['cooling_hits']}/{v['cooling_n']} | {v['cold_iou']:.4f} | "
            f"{t['iou']:.4f} | {t['cooling_hits']}/{t['cooling_n']} | {t['f1']:.4f} | {t['far']:.4f} |"
        )
    lines.append("")
    lines.append(f"Опубликованный победитель `ens_u2ft_tcn_lgbm`: val IoU {PUB_WIN_VAL:.4f}, test IoU {PUB_WIN_TEST:.4f}. "
                 f"Пересчёт той же сохранённой карты в этом скрипте: val {report['sanity']['winner_val_iou']:.4f}, "
                 f"test {report['sanity']['winner_test_iou']:.4f}, cooling test {report['sanity']['winner_test_cooling_hits']}/"
                 f"{report['sanity']['winner_test_cooling_n']}. "
                 f"`ens_cnsym`: val {report['sanity']['cnsym_val_iou']:.4f}, test {report['sanity']['cnsym_test_iou']:.4f}, "
                 f"cooling test {report['sanity']['cnsym_test_cooling_hits']}/{report['sanity']['cnsym_test_cooling_n']}.")
    lines.append("")
    lines.append(f"По правилу полосы выбран `{ch['name']}`. "
                 f"Val IoU {ch['val_iou']:.4f}, cooling {ch['val_cooling_hits']}/{ch['val_cooling_n']}, "
                 f"test IoU {ch['test_iou']:.4f}, cooling {ch['test_cooling_hits']}/{ch['test_cooling_n']}. "
                 f"Сумма на val {ch['val_combined']:.4f}. Режим `{ch['mode']}`.")
    lines.append("")
    if ch.get("weights"):
        lines.append("Веса выбранной строки:")
        lines.append("")
        lines.append("| член | вес |")
        lines.append("| --- | ---: |")
        for n, w in ch["weights"].items():
            if w > 1e-4:
                lines.append(f"| `{n}` | {w:.4f} |")
        lines.append("")
    if report["beats_winner"]:
        lines.append(f"Val IoU этой строки выше пересчитанного победителя на {report['beats_delta']:.4f}. "
                     "В журнал записан BEATS_WINNER. Test в это решение не входил.")
    else:
        lines.append("Val IoU не выше пересчитанного `ens_u2ft_tcn_lgbm`. Победитель этапа 1 не меняется. "
                     "Более высокий test или более высокий cooling при более низком IoU победителя не заменяют.")
    lines.append("")
    lines.append("Сравнение `ens_cnsym` с победителем на одном коде. "
                 f"Разрыв val IoU {report['cnsym_vs_winner']['val_iou_gap']:.4f} "
                 f"(полоса 0.005: {'внутри' if report['cnsym_vs_winner']['inside_band'] else 'снаружи'}). "
                 f"Cooling val: cnsym {report['cnsym_vs_winner']['cnsym_cool']} , победитель {report['cnsym_vs_winner']['winner_cool']}. "
                 f"Cooling test: cnsym {report['cnsym_vs_winner']['cnsym_cool_test']} , победитель {report['cnsym_vs_winner']['winner_cool_test']}. "
                 "Если обе строки в полосе, правило отдаёт предпочтение большему cooling на val; test при этом только отчёт.")
    lines.append("")
    real = report.get("real")
    lines.append("## Реальные файлы")
    lines.append("")
    if not real:
        lines.append("Прогон по MGTS не записан. Карточки из прошлых json не использовались для весов.")
    else:
        pres = real["presence"]
        lines.append(f"Порог {real['threshold']:.5f} заморожен с v8 val, калибровка по файлам не делалась, метки не выдумывались. "
                     f"Карточки presence=yes {pres['yes_alarm']}/{pres['yes_n']}, presence=no {pres['no_alarm']}/{pres['no_n']}. "
                     "Попадание — хотя бы одна тревожная зона (класс вне HVAC_Diurnal и Sensor_Artifact) пересекает рамку. "
                     "Это описание, не критерий отбора.")
        lines.append("")
        lines.append("| файл | зоны | тревоги |")
        lines.append("| --- | ---: | ---: |")
        for name, fr in real["files"].items():
            lines.append(f"| {name} | {fr['n_zones']} | {fr['n_alarm']} |")
        lines.append("")
        lines.append("Пропуски yes и ложные no:")
        lines.append("")
        for c in real["cards"]:
            if c.get("status") != "ok":
                lines.append(f"- {c.get('id')}: {c.get('status')}")
                continue
            if c["presence"] == "yes" and not c["hit_alarm"]:
                lines.append(f"- пропуск yes {c['id']} файл {c['record']} max score {c.get('max_score')}")
            if c["presence"] == "no" and c["hit_alarm"]:
                lines.append(f"- ложная no {c['id']} файл {c['record']} зон {c['n_alarm']}")
    lines.append("")
    lines.append("JSON: `artifacts/eval/s3_sel_v8.json`, при наличии real — `artifacts/eval/s3_sel_real.json`.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(ROOT / "artifacts/cache/v8"))
    ap.add_argument("--pred", default=str(ROOT / "artifacts/pred/v8"))
    ap.add_argument("--skip-real", action="store_true")
    ap.add_argument("--skip-search", action="store_true")
    ap.add_argument("--search-only", action="store_true")
    a = ap.parse_args()
    cache = Path(a.cache)
    pred = Path(a.pred)
    labels = pd.read_csv(ROOT / "data/synth_meta/v8/labels.csv")
    seqs = pd.read_csv(ROOT / "data/synth_meta/v8/sequences.csv").set_index("seq_id")
    t_all = time.time()

    if a.search_only:
        nets = load_nets()
        frames, _infer_full = load_compact(cache, pred, labels, seqs, nets, "val")
        found = search(frames)
        found.pop("mlp_model", None)
        dump_json(ROOT / "artifacts/eval/s3_sel_weights.json", dict(
            members=found["members"], screen=found["screen"],
            adaptive_winner=found["adaptive_winner"],
            adaptive_select=found["adaptive_select"],
            fit_stems=found["fit_stems"], select_stems=found["select_stems"],
        ))
        print("search-only wrote weights", flush=True)
        return

    print("sanity saved maps", flush=True)
    ref_specs = [
        dict(name=WINNER, saved=WINNER, weights={}, mode="saved"),
        dict(name=CNSYM, saved=CNSYM, weights={}, mode="saved"),
    ]
    official = {}
    for sp in ref_specs:
        print("confirm", sp["name"], "val", flush=True)
        val_m = confirm_sort_threshold(cache, pred, "val", sp, labels, seqs)
        print("confirm", sp["name"], "test", "thr", round(val_m["threshold"], 5), flush=True)
        test_m = confirm_sort_threshold(cache, pred, "test", sp, labels, seqs, with_zones=False, thr=val_m["threshold"])
        official[sp["name"]] = dict(val=slim_metrics(val_m), test=slim_metrics(test_m), weights={}, mode="saved", family="reference")
        print(sp["name"], "val", round(val_m["iou"], 4), "cool", val_m["cooling_hits"], "/", val_m["cooling_n"],
              "test", round(test_m["iou"], 4), "cool", test_m["cooling_hits"], "/", test_m["cooling_n"], flush=True)
    win_val = official[WINNER]["val"]["iou"]
    win_test = official[WINNER]["test"]["iou"]
    if abs(win_val - PUB_WIN_VAL) > 0.005 or abs(win_test - PUB_WIN_TEST) > 0.005:
        raise SystemExit(f"пересчёт победителя val {win_val} test {win_test} далеко от {PUB_WIN_VAL}/{PUB_WIN_TEST}")

    nets = load_nets()
    frames, infer_full = load_compact(cache, pred, labels, seqs, nets, "val")
    found = search(frames)
    mlp_model = found.pop("mlp_model")
    del frames

    screen = found["screen"]
    best_screen = max(r["iou"] for r in screen)
    short = []
    seen = set()
    for r in screen:
        keep = r["iou"] >= best_screen - 0.01 or r["name"] == found["adaptive_winner"] or r["name"] in ("winner4", "cnsym4", "equal_neural")
        if keep and r["name"] not in seen:
            seen.add(r["name"])
            short.append(dict(name=r["name"], weights=r["weights"], mode=r["mode"]))
    print("exact shortlist", [s["name"] for s in short], flush=True)
    hist_val = exact_specs(cache, pred, "val", short, labels, seqs, infer_full, mlp_model)
    hist_rows = []
    for sp in short:
        m = hist_val[sp["name"]]
        hist_rows.append(row_of(sp["name"], m, extra=dict(weights=sp["weights"], mode=sp["mode"])))
        print(f"hist-val {sp['name']} IoU {m['iou']:.4f} cool {m['cooling_hits']}/{m['cooling_n']}", flush=True)
    hist_pick, _, _ = sel.pick_band(hist_rows)
    confirm_names = []
    for r in hist_rows:
        near_pick = r["iou"] >= hist_pick["iou"] - sel.IOU_BAND
        near_winner = r["iou"] >= float(official[WINNER]["val"]["iou"]) - 0.003
        forced = r["name"] in ("winner4", "cnsym4")
        if (near_pick or near_winner or forced) and r["name"] not in confirm_names:
            confirm_names.append(r["name"])
    print("sort-confirm", confirm_names, flush=True)
    sorted_val = {}
    for name in confirm_names:
        sp = next(s for s in short if s["name"] == name)
        print("sort val", name, flush=True)
        val_m = confirm_sort_threshold(cache, pred, "val", sp, labels, seqs, infer_full, mlp_model)
        sorted_val[name] = dict(val=slim_metrics(val_m), weights=sp["weights"], mode=sp["mode"], family="mixture")
        print(name, "sort val", round(val_m["iou"], 4), "cool", val_m["cooling_hits"], "/", val_m["cooling_n"], flush=True)
    del infer_full

    choice_rows = []
    for name, block in list(official.items()) + list(sorted_val.items()):
        v = block["val"]
        choice_rows.append(dict(
            name=name, iou=v["iou"], cooling_recall=v["cooling_recall"],
            cooling_hits=v["cooling_hits"], cooling_n=v["cooling_n"], combined=v["combined"],
            mode=block["mode"], weights=block["weights"], family=block["family"],
        ))
    choice, choice_best, choice_band = sel.pick_band(choice_rows)
    print("CHOICE", choice["name"], "iou", round(choice["iou"], 4), "cool", choice["cooling_hits"], "/", choice["cooling_n"], flush=True)
    our_rows = [r for r in choice_rows if r["family"] == "mixture"]
    our_best = max(our_rows, key=lambda r: (r["iou"], r["cooling_recall"])) if our_rows else None
    test_names = []
    for r in our_rows:
        in_band = r["iou"] >= choice_best - sel.IOU_BAND
        if in_band or r["name"] in ("winner4", "cnsym4") or (our_best and r["name"] == our_best["name"]):
            if r["name"] not in test_names:
                test_names.append(r["name"])
    test_bank = build_infer_bank(cache, "test", nets) if test_names else None
    for name in test_names:
        sp = dict(name=name, weights=sorted_val[name]["weights"], mode=sorted_val[name]["mode"])
        print("sort test", name, flush=True)
        test_m = confirm_sort_threshold(
            cache, pred, "test", sp, labels, seqs, test_bank, mlp_model,
            with_zones=(name == choice["name"] or name == (our_best or {}).get("name")),
            thr=sorted_val[name]["val"]["threshold"],
        )
        sorted_val[name]["test"] = slim_metrics(test_m)
        print(name, "test", round(test_m["iou"], 4), "cool", test_m["cooling_hits"], "/", test_m["cooling_n"], flush=True)
    if test_bank is not None:
        del test_bank
    for name, block in sorted_val.items():
        if "test" not in block:
            block["test"] = None
        official[name] = block

    choice_block = official[choice["name"]]
    win_block = official[WINNER]
    cn_block = official[CNSYM]
    beats = float(choice_block["val"]["iou"]) > float(win_block["val"]["iou"]) + 1e-4 and choice["family"] == "mixture"
    # Также отметим, если любая наша смесь выше по val IoU, даже если полоса выбрала другую за cooling.
    best_mixture_iou = max(official[r["name"]]["val"]["iou"] for r in our_rows) if our_rows else 0.0
    beats_any = best_mixture_iou > float(win_block["val"]["iou"]) + 1e-4

    real_payload = None
    real_target = choice if choice["family"] == "mixture" else (our_best or None)
    if real_target and real_target["mode"] == "mlp":
        print("real пропущен: выбран mlp, на csv нет готовых карт", flush=True)
        real_target_name = None
    else:
        real_target_name = None if (a.skip_real or real_target is None) else real_target["name"]
    if real_target_name:
        sp = official[real_target_name]
        try:
            real_payload = run_real(sp["weights"], sp["mode"], sp["val"]["threshold"], mlp_model, ROOT / "artifacts/eval/s3_sel_real.json")
        except Exception as exc:
            real_payload = dict(error=str(exc), presence=None)
            print("real failed", exc, flush=True)

    def cool_pair(block, split):
        m = block[split]
        return f"{m['cooling_hits']}/{m['cooling_n']}"

    report = dict(
        task="SEL",
        version="v8",
        far=sel.FAR,
        hit_frac=HIT,
        iou_band=sel.IOU_BAND,
        published_winner=dict(name=WINNER, val_iou=PUB_WIN_VAL, test_iou=PUB_WIN_TEST),
        rule="val IoU @ FAR 1e-3; внутри полосы 0.005 максимум cooling event recall на val; test не выбирает",
        trained_on_real=False,
        sanity=dict(
            winner_val_iou=win_block["val"]["iou"],
            winner_test_iou=win_block["test"]["iou"],
            winner_test_cooling_hits=win_block["test"]["cooling_hits"],
            winner_test_cooling_n=win_block["test"]["cooling_n"],
            cnsym_val_iou=cn_block["val"]["iou"],
            cnsym_test_iou=cn_block["test"]["iou"],
            cnsym_test_cooling_hits=cn_block["test"]["cooling_hits"],
            cnsym_test_cooling_n=cn_block["test"]["cooling_n"],
            winner_val_gap_vs_published=win_block["val"]["iou"] - PUB_WIN_VAL,
        ),
        members=[{k: v for k, v in r.items() if k != "events"} for r in found["members"]],
        fit_stems=found["fit_stems"],
        select_stems=found["select_stems"],
        adaptive_winner=found["adaptive_winner"],
        adaptive_select=[{k: v for k, v in r.items() if k != "weights"} | {"weights": r.get("weights")} for r in found["adaptive_select"]],
        screen=found["screen"],
        hist_val=hist_rows,
        official={k: {kk: vv for kk, vv in block.items() if kk != "unused"} for k, block in official.items()},
        choice=dict(
            name=choice["name"], family=choice["family"], mode=choice["mode"], weights=choice["weights"],
            val_iou=choice_block["val"]["iou"], val_cooling_hits=choice_block["val"]["cooling_hits"],
            val_cooling_n=choice_block["val"]["cooling_n"], val_combined=choice_block["val"]["combined"],
            test_iou=choice_block["test"]["iou"], test_cooling_hits=choice_block["test"]["cooling_hits"],
            test_cooling_n=choice_block["test"]["cooling_n"],
            band=[r["name"] for r in choice_band],
        ),
        cnsym_vs_winner=dict(
            val_iou_gap=cn_block["val"]["iou"] - win_block["val"]["iou"],
            inside_band=abs(cn_block["val"]["iou"] - win_block["val"]["iou"]) <= sel.IOU_BAND or cn_block["val"]["iou"] >= choice_best - sel.IOU_BAND,
            cnsym_cool=cool_pair(cn_block, "val"),
            winner_cool=cool_pair(win_block, "val"),
            cnsym_cool_test=cool_pair(cn_block, "test"),
            winner_cool_test=cool_pair(win_block, "test"),
        ),
        beats_winner=bool(beats or beats_any),
        beats_delta=float(max(best_mixture_iou, choice_block["val"]["iou"]) - win_block["val"]["iou"]),
        our_best_val_iou=float(best_mixture_iou),
        real=None if not real_payload or real_payload.get("error") else dict(
            threshold=real_payload["threshold"], files=real_payload["files"],
            cards=real_payload["cards"], presence=real_payload["presence"], mode=real_payload["mode"],
        ),
        seconds=round(time.time() - t_all, 1),
    )
    # inside_band для cnsym относительно победителя, не относительно choice_best
    report["cnsym_vs_winner"]["inside_band"] = (win_block["val"]["iou"] - cn_block["val"]["iou"]) <= sel.IOU_BAND
    dump_json(ROOT / "artifacts/eval/s3_sel_v8.json", report)
    write_md(ROOT / "docs/S3_SEL.md", report)
    ch = report["choice"]
    msg = (
        f"SEL_DONE method {ch['name']} val IoU {ch['val_iou']:.4f} cooling {ch['val_cooling_hits']}/{ch['val_cooling_n']} "
        f"test IoU {ch['test_iou']:.4f} cooling {ch['test_cooling_hits']}/{ch['test_cooling_n']} "
        f"winner_reval {win_block['val']['iou']:.4f}/{win_block['test']['iou']:.4f} "
        f"cnsym_reval {cn_block['val']['iou']:.4f}/{cn_block['test']['iou']:.4f} "
        f"cnsym_cool_val {cn_block['val']['cooling_hits']}/{cn_block['val']['cooling_n']} "
        f"winner_cool_val {win_block['val']['cooling_hits']}/{win_block['val']['cooling_n']} "
        f"beats {int(report['beats_winner'])}"
    )
    elog(msg)
    if report["beats_winner"]:
        elog(
            f"BEATS_WINNER {ch['name']} val {ch['val_iou']:.4f} test {ch['test_iou']:.4f} "
            f"winner val {win_block['val']['iou']:.4f} test {win_block['test']['iou']:.4f}"
        )
    print("wrote", ROOT / "artifacts/eval/s3_sel_v8.json", "sec", report["seconds"], flush=True)


def build_infer_bank(cache, split, nets):
    bank = {n: {} for n in INFER}
    t0 = time.time()
    files = data.list_cache(cache, split)
    for i, path in enumerate(files):
        X = np.load(path)["X"].astype(np.float32)
        for name, net in nets.items():
            pr = models.predict_full(net, X)
            bank[name][path.stem] = ((1.0 - pr[0]).astype(np.float16), (pr[1] - pr[2]).astype(np.float16))
        del X
        if i % 5 == 0 or i + 1 == len(files):
            print(f"infer {split} {i+1}/{len(files)} {round(time.time()-t0)}s", flush=True)
    return bank


if __name__ == "__main__":
    main()
