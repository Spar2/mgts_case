"""Realnet: U-Net с синтетики и отдельный скор базовой линии.

Абляция v8 val оставила исходный unet_v810ft на каналах кэша. AdaBN и маска
выключены. Остаток базовой линии — дополнительный скор; его вес и порог
FAR 1e-3 выбирает pooled val v8+v10. Test только отчёт.

Реальные 1–6.csv: одна общая модель, без leave-one-file-out, потому что
обе адаптации проиграли на val. Карточки только в конце, порог не двигается.
Длинный прогон: из корня dts_solution, 2 потока.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path


def _arg(name, default):
    if name in sys.argv:
        return sys.argv[sys.argv.index(name) + 1]
    return default


_THREADS = _arg("--threads", "2")
os.environ["OMP_NUM_THREADS"] = _THREADS
os.environ["MKL_NUM_THREADS"] = _THREADS
os.environ["OPENBLAS_NUM_THREADS"] = _THREADS
os.environ["NUMEXPR_NUM_THREADS"] = _THREADS

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dts import data, domain, events as EV, export, features, grid, infer, metrics, models, real
from experiments.realnet.realnet import (
    MODEL_VERSION, NON_ALARM, adapt_batchnorm, assemble_zones, attach_zone_cv,
    combine_scores, describe_zone, holdout_ids, load_pretrained, make_channels,
    n_harmonics_for, predict_components, sample_tile, save_model, train_mask_head, weight_grid,
)

import torch

# Абляция v8 val (artifacts/eval/realnet_ablation.json):
# замена входа U-Net остатком базовой линии роняет IoU 0.375 → 0.091,
# AdaBN по фону train — 0.375 → 0.304. Маска и широкий скор на симплексе
# получают вес 0: они не лучше чистой головы. Поэтому сеть видит каналы,
# на которых её учили, адаптации нет, а остаток базовой линии остаётся
# только дополнительным скором (вес всё равно выбирает pooled val).
USE_ADABN = False
USE_MASK = False
USE_WIDE = True


def event(msg: str) -> None:
    line = time.strftime("%H:%M", time.gmtime()) + " realnet: " + msg
    try:
        with open("/workspace/run_logs/_events.log", "a") as f:
            f.write(line + "\n")
    except OSError:
        pass
    print(line, flush=True)


def dump_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def conv(o):
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    with open(path, "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=conv)


def new_acc():
    return dict(tot=np.zeros(3), hot=np.zeros(3), cold=np.zeros(3), far_n=0, far_d=0,
                per_seq={}, ev_rows=[], zones_n=0, zones_false=0, kmh=0.0)


def add_frame(acc, score, hot, valid, ev, nu, x, t, thr, labels, sid):
    valid = np.asarray(valid, bool)
    ev = np.asarray(ev)
    nu = np.asarray(nu)
    pred = (score >= thr) & valid
    truth = ev > 0
    c = metrics.confusion(pred, truth, valid)
    acc["tot"] += c
    acc["per_seq"][sid] = tuple(int(v) for v in c)
    bg = valid & (ev == 0) & (nu == 0)
    acc["far_n"] += int((pred & bg).sum())
    acc["far_d"] += int(bg.sum())
    acc["hot"] += metrics.confusion(pred & (hot > 0), (ev >= 1) & (ev <= 4), valid)
    acc["cold"] += metrics.confusion(pred & (hot <= 0), ev >= 5, valid)
    if labels is not None:
        for r in metrics.label_boxes(labels, sid).itertuples():
            if r.kind != "event":
                continue
            px = metrics.event_pixels(r, x, t, ev)
            if px.sum() == 0:
                continue
            frac = float(pred[px].mean())
            acc["ev_rows"].append(dict(
                seq_id=sid, label=r.label, amp=float(r.amp_obs_C), amp_bin=metrics.amp_bin(float(r.amp_obs_C)),
                template=getattr(r, "template", ""), difficulty=r.difficulty, frac=frac, hit=frac >= 0.3,
                tp=int((pred & px).sum()), fn=int((~pred & px).sum())))
    zs = EV.zones_from_prob(score, hot, valid, x, t, float(thr))
    for z in zs:
        sl = (slice(z["t0"], z["t1"]), slice(z["x0"], z["x1"]))
        comp = (score[sl] >= thr) & valid[sl]
        acc["zones_n"] += 1
        if comp.any():
            acc["zones_false"] += int(float((ev[sl][comp] > 0).mean()) < 0.1)
        else:
            acc["zones_false"] += 1
    if len(x) > 1 and len(t) > 1:
        acc["kmh"] += float(x[-1] - x[0]) / 1000.0 * float(t[-1] - t[0]) / 3600.0


def finish_acc(acc, seqs):
    import pandas as pd
    tot = acc["tot"]
    res = dict(far=float(acc["far_n"] / max(acc["far_d"], 1)), **metrics.prf(*tot),
               hot=metrics.prf(*acc["hot"]), cold=metrics.prf(*acc["cold"]),
               iou_ci95=metrics.bootstrap_ci(acc["per_seq"], sorted(acc["per_seq"])) if acc["per_seq"] else [None, None])
    E = pd.DataFrame(acc["ev_rows"])
    if len(E):
        if seqs is not None:
            E["template"] = [seqs.loc[s, "template"] if s in seqs.index else "" for s in E.seq_id]

        def agg(g):
            return dict(n=int(len(g)), recall=float(g.hit.mean()),
                        pixel_recall=float(g.tp.sum() / max(g.tp.sum() + g.fn.sum(), 1)))

        res["events_all"] = agg(E)
        res["by_amp"] = {str(k): agg(g) for k, g in E.groupby("amp_bin")}
        res["by_label"] = {str(k): agg(g) for k, g in E.groupby("label")}
        res["by_template"] = {str(k): agg(g) for k, g in E.groupby("template")}
        res["weak_events"] = agg(E[E.amp <= 2]) if (E.amp <= 2).any() else dict(n=0, recall=None, pixel_recall=None)
    tm = {}
    for sid, c in acc["per_seq"].items():
        key = str(seqs.loc[sid, "template"]) if seqs is not None and sid in seqs.index else "?"
        tm.setdefault(key, np.zeros(3))
        tm[key] += c
    res["iou_by_template"] = {k: metrics.prf(*v)["iou"] for k, v in tm.items()}
    res["zones"] = dict(n_zones=int(acc["zones_n"]), n_false=int(acc["zones_false"]),
                        false_frac=float(acc["zones_false"] / max(acc["zones_n"], 1)),
                        false_zones_per_km_h=float(acc["zones_false"] / max(acc["kmh"], 1e-9)))
    return res


def slim(res):
    return {k: res[k] for k in ("far", "precision", "recall", "f1", "iou", "iou_ci95") if k in res}


def build_synth(path, need_wide=True):
    """Вход сети — каналы кэша, на которых учили U-Net. Остаток базовой линии только для доп. скора."""
    path = Path(path)
    z = np.load(path)
    X = np.asarray(z["X"], np.float32)
    valid = np.asarray(z["valid"], bool)
    x = np.asarray(z["x"], np.float64)
    t = np.asarray(z["t"], np.float64)
    ev = np.asarray(z["ev"])
    nu = np.asarray(z["nu"])
    sigma = np.asarray(z["sigma"], np.float32)
    span_h = float(t[-1] - t[0]) / 3600.0 if t.size > 1 else 0.0
    n_harm = -1
    residual = X[0] * sigma[None, :]
    if need_wide and "T" in z.files:
        T = np.asarray(z["T"], np.float32)
        valid_t = valid & np.isfinite(T)
        dx = float(np.median(np.diff(x))) if x.size > 1 else 0.25
        _Xb, info = make_channels(T, valid_t, t, x, dx=dx)
        residual = info["residual"]
        sigma = info["sigma_x"]
        n_harm = int(info["n_harm"])
        span_h = float(info["span_h"])
    return dict(name=path.stem, X=X, residual=residual, sigma=sigma, valid=valid,
                ev=ev, nu=nu, x=x, t=t, span_h=span_h, n_harm=n_harm)


def save_components(path, comps):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    p_seg, p_mask, p_wide, hot = comps
    np.savez_compressed(path, p_seg=p_seg.astype(np.float16), p_mask=p_mask.astype(np.float16),
                        p_wide=p_wide.astype(np.float16), hot=hot.astype(np.float16))


def load_components(path):
    z = np.load(path)
    return (z["p_seg"].astype(np.float32), z["p_mask"].astype(np.float32),
            z["p_wide"].astype(np.float32), z["hot"].astype(np.float32))


def adapt_record(path):
    """Кадр для AdaBN и маски. Val/test хранят T. Train его лишён (slim_train_cache), сырых h5 v8/v10 нет.

    Тогда берём каналы кэша, на которых U-Net уже учился. Фон по-прежнему ev==0 и nu==0.
    Оценка val/test идёт через build_synth и свою базовую линию, не через этот путь.
    """
    path = Path(path)
    z = np.load(path)
    if "T" in z.files:
        rec = build_synth(path)
        rec["channel_source"] = "harmonic_baseline"
        return rec
    t = np.asarray(z["t"], np.float64)
    span = float(t[-1] - t[0]) / 3600.0 if t.size > 1 else 0.0
    return dict(name=path.stem, X=np.asarray(z["X"], np.float32), valid=np.asarray(z["valid"], bool),
                ev=np.asarray(z["ev"]), nu=np.asarray(z["nu"]), x=np.asarray(z["x"], np.float64), t=t,
                span_h=span, n_harm=-1, channel_source="cache_X")


def collect_tiles(paths, n_files, n_tiles, seed):
    rng = np.random.default_rng(seed)
    order = list(paths)
    rng.shuffle(order)
    tiles, weights = [], []
    used = []
    for p in order[:n_files]:
        rec = adapt_record(p)
        bg = (rec["valid"] & (rec["ev"] == 0) & (rec["nu"] == 0)).astype(np.float32)
        for _ in range(n_tiles):
            tile, w = sample_tile(rec["X"], rng, bg)
            tiles.append(tile)
            weights.append(w if w is not None else np.ones(tile.shape[-2:], np.float32))
        used.append(f"{Path(p).stem}:{rec['channel_source']}")
        print(f"adapt tile {Path(p).name} src {rec['channel_source']} harm {rec['n_harm']} span {rec['span_h']:.1f}h",
              flush=True)
        del rec
    return tiles, weights, used


def score_file(model, rec):
    return predict_components(model, rec["X"], rec["residual"], rec["sigma"], rec["valid"],
                              need_mask=USE_MASK, need_wide=USE_WIDE)


def _frame_arrays(sp, cp):
    comp = load_components(sp)
    z = np.load(cp)
    valid = np.asarray(z["valid"], bool)
    ev = np.asarray(z["ev"])
    nu = np.asarray(z["nu"])
    bg = valid & (ev == 0) & (nu == 0)
    return comp, valid, ev, nu, bg


def select_weights(items, far, allow_mask=False, allow_wide=True):
    """items: (component_npz, cache_npz, version). Порог и IoU только по этим кадрам.

    Порог — точный квантиль всего фона val, как far_threshold в eval_stage1.
    Веса идут по одному: все карты сразу в памяти не лежат.
    Маска и широкий скор выключаются, если абляция их не оставила.
    """
    grid = [w for w in weight_grid() if (allow_mask or w[1] == 0.0) and (allow_wide or w[2] == 0.0)]
    thresholds = []
    for w in grid:
        parts = []
        n_bg = 0
        for sp, cp, _ver in items:
            comp, _valid, _ev, _nu, bg = _frame_arrays(sp, cp)
            score = combine_scores(comp[0], comp[1], comp[2], w)
            part = np.asarray(score[bg], np.float32).ravel()
            parts.append(part)
            n_bg += int(part.size)
        bg_scores = np.concatenate(parts) if parts else np.zeros(1, np.float32)
        del parts
        thr = float(metrics.far_threshold(bg_scores, far))
        thresholds.append(thr)
        print(f"thr {w} n_bg {n_bg} {thr:.6f}", flush=True)
        del bg_scores
    rows = []
    best = None
    for i, w in enumerate(grid):
        tot = {v: np.zeros(3) for v in ("v8", "v10")}
        pooled = np.zeros(3)
        far_n = far_d = 0
        for sp, cp, ver in items:
            comp, valid, ev, _nu, bg = _frame_arrays(sp, cp)
            score = combine_scores(comp[0], comp[1], comp[2], w)
            pred = (score >= thresholds[i]) & valid
            c = np.array(metrics.confusion(pred, ev > 0, valid), float)
            pooled += c
            if ver in tot:
                tot[ver] += c
            far_n += int((pred & bg).sum())
            far_d += int(bg.sum())
        rec = dict(weights=w, threshold=thresholds[i], pooled=metrics.prf(*pooled),
                   far=float(far_n / max(far_d, 1)),
                   v8=metrics.prf(*tot["v8"]), v10=metrics.prf(*tot["v10"]))
        rows.append(dict(weights=list(w), threshold=rec["threshold"], iou=rec["pooled"]["iou"],
                         f1=rec["pooled"]["f1"], far=rec["far"],
                         iou_v8=rec["v8"]["iou"], iou_v10=rec["v10"]["iou"]))
        # сетка уже отсортирована в пользу головы сегментации: ничья остаётся за более ранним
        if best is None or rec["pooled"]["iou"] > best["pooled"]["iou"] + 1e-6:
            best = rec
    return best, rows


def metrics_from_saved(items, weights, thr, labels_of, seqs_of):
    by = {}
    for sp, cp, ver in items:
        comp = load_components(sp)
        z = np.load(cp)
        acc = by.setdefault(ver, new_acc())
        score = combine_scores(comp[0], comp[1], comp[2], weights)
        add_frame(acc, score, comp[3], z["valid"], z["ev"], z["nu"], z["x"], z["t"], thr,
                  labels_of[ver], Path(cp).stem)
        # шаблон в строках событий дописывается в finish из sequences
        del z
    out = {}
    pooled = new_acc()
    for ver, acc in by.items():
        out[ver] = finish_acc(acc, seqs_of[ver])
        pooled["tot"] += acc["tot"]
        pooled["hot"] += acc["hot"]
        pooled["cold"] += acc["cold"]
        pooled["far_n"] += acc["far_n"]
        pooled["far_d"] += acc["far_d"]
        pooled["per_seq"].update({f"{ver}:{k}": v for k, v in acc["per_seq"].items()})
        pooled["zones_n"] += acc["zones_n"]
        pooled["zones_false"] += acc["zones_false"]
        pooled["kmh"] += acc["kmh"]
        pooled["ev_rows"] += acc["ev_rows"]
    out["pooled"] = finish_acc(pooled, None)
    return out


def stream_test(model, paths, weights, thr, labels, seqs, ver):
    acc = new_acc()
    t0 = time.time()
    need_mask = weights[1] > 0
    need_wide = weights[2] > 0
    for i, p in enumerate(paths):
        rec = build_synth(p, need_wide=need_wide)
        comps = predict_components(model, rec["X"], rec["residual"], rec["sigma"], rec["valid"],
                                   need_mask=need_mask, need_wide=need_wide)
        score = combine_scores(*comps[:3], weights)
        add_frame(acc, score, comps[3], rec["valid"], rec["ev"], rec["nu"], rec["x"], rec["t"], thr, labels, rec["name"])
        if (i + 1) % 5 == 0 or i + 1 == len(paths):
            print(f"test {ver} {i+1}/{len(paths)} {time.time()-t0:.0f}s", flush=True)
        del rec
    return finish_acc(acc, seqs)


def load_meta(ver):
    import pandas as pd
    root = Path("data/synth_meta") / ver
    labels = pd.read_csv(root / "labels.csv")
    seqs = pd.read_csv(root / "sequences.csv").set_index("seq_id")
    return labels, seqs


def reference_numbers():
    out = {}
    r7 = Path("artifacts/eval/r7_ens_u2ft_tcn_lgbm.json")
    if r7.exists():
        blob = json.load(open(r7))
        for split in ("val", "test"):
            block = blob.get(split, {}).get("ens_u2ft_tcn_lgbm@far_1e-3")
            if block:
                out[f"winner_v8_{split}"] = {k: block.get(k) for k in ("iou", "f1", "precision", "recall", "far")}
    v10 = Path("artifacts/eval/v10_ens_u2ft_tcn_lgbm.json")
    if v10.exists():
        blob = json.load(open(v10))
        block = blob.get("test", {}).get("ens_u2ft_tcn_lgbm@far_1e-3")
        if block:
            out["winner_v10_test"] = {k: block.get(k) for k in ("iou", "f1", "precision", "recall", "far")}
    return out


def load_typer():
    root = Path("models_release")
    ck = torch.load(root / "eventnet.pt", map_location="cpu", weights_only=False)
    enet = models.EventNet(len(EV.CLASSES), len(ck["mu"]), cin=int(ck.get("cin", 3)),
                           late_fusion=bool(ck.get("late_fusion", False)))
    enet.load_state_dict(ck["state"])
    enet.eval()
    lgbm, kind = None, "legacy"
    try:
        import joblib
        obj = joblib.load(root / "lgbm.joblib")
        if isinstance(obj, dict) and "model" in obj:
            lgbm = obj["seeds"] if obj.get("seeds") else obj["model"]
            kind = obj.get("kind", "full")
        else:
            lgbm = obj
    except Exception as exc:
        print("lgbm не загрузился", exc, flush=True)
    return dict(enet=enet, ck=ck, lgbm=lgbm, kind=kind)


def type_zones(typer, zones, X, sigma, x, t):
    if not zones or typer is None:
        return
    ck = typer["ck"]
    P = infer.predict_zone_probs(
        typer["enet"], ck["mu"], ck["sd"], X, sigma, x, t, zones, lgbm=typer["lgbm"], lgbm_kind=typer["kind"],
        temperature=float(ck.get("temperature", 1.0)), lgbm_temperature=float(ck.get("lgbm_temperature", 1.0)),
        cnn_weight=float(ck.get("cnn_weight", 0.5)), scalar_names=ck.get("scalar_names"),
        two_scale=bool(ck.get("two_scale", False)),
        stack_coef=None if ck.get("stack_coef") is None else np.asarray(ck["stack_coef"], np.float32),
        stack_intercept=None if ck.get("stack_intercept") is None else np.asarray(ck["stack_intercept"], np.float32))
    for z, pr in zip(zones, P):
        k = int(np.argmax(pr))
        z["class"] = EV.CLASSES[k]
        z["class_prob"] = {c: round(float(q), 3) for c, q in zip(EV.CLASSES, pr)}
        z["alarm"] = z["class"] not in NON_ALARM


def load_real_grid(path, cache_dir: Path):
    path = Path(path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = real.sha256(path)
    cache = cache_dir / f"{path.stem}.npz"
    if cache.exists():
        z = np.load(cache)
        if str(z["sha256"].item()) == digest:
            print(f"grid cache {path.name} {tuple(z['T'].shape)}", flush=True)
            return dict(name=str(z["name"].item()), sha256=digest, tz_missing=bool(z["tz_missing"].item()),
                        x=z["x"], t=z["t"], T=z["T"].astype(np.float32), valid=z["valid"].astype(bool))
    print(f"parse {path.name}", flush=True)
    raw = real.load_csv(path)
    plan = grid.plan(raw["x"], raw["t"])
    Tg, vg = grid.resample(raw["T"], raw["valid"], plan, "mean")
    Tg = np.asarray(Tg, np.float32)
    vg = np.asarray(vg, bool) & np.isfinite(Tg)
    out = dict(name=raw["name"], sha256=raw["sha256"], tz_missing=bool(raw["tz_missing"]),
               x=np.asarray(plan["x"], np.float64), t=np.asarray(plan["t"], np.float64), T=Tg, valid=vg)
    np.savez(cache, T=out["T"], valid=out["valid"].astype(np.uint8), x=out["x"], t=out["t"],
             sha256=np.array(out["sha256"]), tz_missing=np.array(out["tz_missing"]), name=np.array(out["name"]))
    print(f"gridded {out['name']} {out['T'].shape}", flush=True)
    return out


def channelize(rec, need_wide=False):
    """Вход сети — обычные три канала features.make_channels. Базовая линия — отдельный скор."""
    dx = float(np.median(np.diff(rec["x"]))) if len(rec["x"]) > 1 else 0.25
    ch, aux = features.make_channels(rec["T"], rec["valid"], dx=dx)
    n_harm, span_h = n_harmonics_for(rec["t"])
    info = dict(sigma_x=aux["sigma_x"], residual=np.asarray(aux["dT_time"], np.float32),
                n_harm=int(n_harm), span_h=float(span_h))
    if need_wide:
        _Xb, base = make_channels(rec["T"], rec["valid"], rec["t"], rec["x"], dx=dx)
        info["residual"] = base["residual"]
        info["sigma_x"] = base["sigma_x"]
        info["n_harm"] = int(base["n_harm"])
        info["span_h"] = float(base["span_h"])
    return ch.astype(np.float32), info


def save_figure(path, rec, residual, score, zones, thr, annotations, variant=""):
    x, t, valid = rec["x"], rec["t"], rec["valid"]
    fig, ax = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    hours = (t - t[0]) / 3600.0
    ext = [float(x[0]), float(x[-1]), float(hours[-1]) if len(hours) else 0.0, 0.0]
    show = np.where(valid, residual, np.nan)
    finite = show[np.isfinite(show)]
    v = float(np.percentile(np.abs(finite), 99.5)) if finite.size else 1.0
    v = max(v, 0.2)
    ax[0].imshow(show, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v, extent=ext, interpolation="nearest")
    ax[0].set_title(f"{rec['name']}.csv: ΔT, °C")
    zshow = np.where(valid, score, np.nan)
    ax[1].imshow(zshow, aspect="auto", cmap="magma", vmin=0, vmax=1, extent=ext, interpolation="nearest")
    tag = f"вариант {variant}, " if variant else ""
    ax[1].set_title(f"{tag}скор realnet, порог {thr:.3f}")
    drawn = 0
    for z in zones:
        if not z.get("alarm", True):
            continue
        if drawn >= 200:
            break
        drawn += 1
        dur_h = max((z["t_end_s"] - z["t_start_s"]) / 3600.0, 0.02)
        width = max(z["end_m"] - z["start_m"], 0.5)
        y0 = (z["t_start_s"] - t[0]) / 3600.0
        color = "#c0392b" if z["polarity"] == "hot" else "#2471a3"
        for a in ax:
            a.add_patch(Rectangle((z["start_m"], y0), width, dur_h, fill=False, ec=color, lw=0.7))
    if annotations:
        from dts import zone_cv
        zone_cv.overlay(fig, annotations, x, t, ax=ax[0])
        zone_cv.overlay(fig, annotations, x, t, ax=ax[1])
    ax[0].set_ylabel("время от начала, ч")
    ax[0].set_xlabel("x, м")
    ax[1].set_xlabel("x, м")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def saved_card_tally(path):
    if not Path(path).exists():
        return None
    blob = json.load(open(path))
    rows = blob.get("cards") or []
    yes = [c for c in rows if c.get("presence") == "yes" and c.get("status", "ok") == "ok"]
    no = [c for c in rows if c.get("presence") == "no" and c.get("status", "ok") == "ok"]
    return dict(yes_alarm=sum(1 for c in yes if c.get("cnn_n_alarm_zones")), yes_n=len(yes),
                no_alarm=sum(1 for c in no if c.get("cnn_n_alarm_zones")), no_n=len(no),
                yes_base=sum(1 for c in yes if c.get("base_n_zones")),
                no_base=sum(1 for c in no if c.get("base_n_zones")),
                threshold=blob.get("threshold"), op=blob.get("op"))


def run_synth(a):
    out_path = Path("artifacts/eval/realnet_synth.json")
    if a.resume and out_path.exists() and not a.redo_synth:
        blob = json.load(open(out_path))
        if blob.get("done") and "weights" in blob and "threshold" in blob:
            event(f"synth json уже готов, веса {blob['weights']} порог {blob['threshold']:.4f}")
            return blob
    ckpt = Path(a.ckpt)
    model = load_pretrained(ckpt)
    head_w, head_b = model.unet.head.weight.detach().clone(), model.unet.head.bias.detach().clone()
    used, hist = [], []
    if USE_ADABN or USE_MASK:
        v8_train = data.list_cache(a.cache_v8, "train")
        v10_train = data.list_cache(a.cache_v10, "train") if Path(a.cache_v10).exists() else []
        n8 = min(8, a.n_adapt, len(v8_train))
        n10 = min(4, max(a.n_adapt - n8, 0), len(v10_train))
        event(f"адаптация synth train-фон: {n8} файлов v8 + {n10} v10, AdaBN={USE_ADABN} маска={USE_MASK}")
        tiles, tw = [], []
        if n8:
            t, w, u = collect_tiles(v8_train, n8, a.tiles_per_file, seed=0)
            tiles += t
            tw += w
            used += [f"v8:{s}" for s in u]
        if n10:
            t, w, u = collect_tiles(v10_train, n10, a.tiles_per_file, seed=1)
            tiles += t
            tw += w
            used += [f"v10:{s}" for s in u]
        if USE_ADABN:
            adapt_batchnorm(model.unet, tiles, batch=4)
        if USE_MASK:
            hist = train_mask_head(model, tiles, tw, minutes=a.adapt_minutes, seed=0, batch=4)
        if not torch.equal(head_w, model.unet.head.weight.detach()) or not torch.equal(head_b, model.unet.head.bias.detach()):
            raise RuntimeError("голова сегментации сдвинулась на synth")
        del tiles, tw
    else:
        event("адаптации нет: абляция v8 val оставила исходный U-Net")
    save_model(model, "artifacts/realnet/model_synth.pt", adapted_on=used, mask_hist=hist, source=str(ckpt),
               use_adabn=USE_ADABN, use_mask=USE_MASK, use_wide=USE_WIDE)

    tmp = Path("artifacts/realnet/tmp_scores")
    val_items = []
    for ver, cache in (("v8", a.cache_v8), ("v10", a.cache_v10)):
        paths = data.list_cache(cache, "val")
        if not paths:
            continue
        t0 = time.time()
        for i, p in enumerate(paths):
            dst = tmp / ver / "val" / p.name
            if not dst.exists():
                rec = build_synth(p)
                save_components(dst, score_file(model, rec))
                del rec
            val_items.append((dst, p, ver))
            if (i + 1) % 5 == 0 or i + 1 == len(paths):
                print(f"val {ver} {i+1}/{len(paths)} {time.time()-t0:.0f}s", flush=True)
    event(f"val карты {len(val_items)} файлов, выбор весов")
    best, table = select_weights(val_items, far=1e-3, allow_mask=USE_MASK, allow_wide=USE_WIDE)
    weights = tuple(best["weights"])
    thr = float(best["threshold"])
    event(f"выбрано по val IoU {best['pooled']['iou']:.4f}: веса seg/mask/wide {weights} порог {thr:.4f}")
    labels_of, seqs_of = {}, {}
    for ver in ("v8", "v10"):
        try:
            labels_of[ver], seqs_of[ver] = load_meta(ver)
        except FileNotFoundError:
            labels_of[ver], seqs_of[ver] = None, None
    val_metrics = metrics_from_saved(val_items, weights, thr, labels_of, seqs_of)
    test_metrics = {}
    for ver, cache in (("v8", a.cache_v8), ("v10", a.cache_v10)):
        paths = data.list_cache(cache, "test")
        if not paths:
            continue
        test_metrics[ver] = stream_test(model, paths, weights, thr, labels_of.get(ver), seqs_of.get(ver), ver)
        event(f"test {ver} IoU {test_metrics[ver]['iou']:.4f} F1 {test_metrics[ver]['f1']:.4f} "
              f"FAR {test_metrics[ver]['far']:.5f}")
    blob = dict(method="realnet", done=True, far_target=1e-3, weights=dict(seg=weights[0], mask=weights[1], wide=weights[2]),
                threshold=thr, adapted_on=used, mask_steps=hist[-1].get("step") if hist else 0,
                adapt_input="pretrained channels (cache X / features.make_channels); harmonic baseline is an extra score only",
                use_adabn=USE_ADABN, use_mask=USE_MASK, use_wide_score=USE_WIDE,
                ablation="artifacts/eval/realnet_ablation.json",
                selection="pooled v8+v10 val pixel IoU at far 1e-3; ties keep the larger segmentation weight; mask/wide only if ablation left them on",
                weight_table=table, val=val_metrics, test=test_metrics, reference=reference_numbers(),
                backbone=str(ckpt), model="artifacts/realnet/model_synth.pt")
    dump_json(blob, out_path)
    # промежуточные карты val больше не нужны: метрики уже в json
    removed = 0
    if tmp.exists():
        for p in tmp.rglob("*.npz"):
            p.unlink()
            removed += 1
    event(f"synth json записан, временные карты удалены ({removed})")
    return blob


def run_real(a, synth):
    out_path = Path("artifacts/eval/realnet_real.json")
    if a.resume and out_path.exists() and not a.redo_real:
        blob = json.load(open(out_path))
        if blob.get("done"):
            event("real json уже готов")
            return blob
    weights = (float(synth["weights"]["seg"]), float(synth["weights"]["mask"]), float(synth["weights"]["wide"]))
    thr = float(synth["threshold"])
    root = Path(a.real_root)
    files = sorted(root.glob("*.csv"))
    if not files:
        raise SystemExit(f"нет csv в {root}")
    grids = [load_real_grid(p, Path("artifacts/realnet/grid")) for p in files]
    names = [g["name"] for g in grids]
    by_name = {g["name"]: g for g in grids}
    adapt = USE_ADABN or USE_MASK
    if adapt:
        event(f"real LOFO на {names}, порог synth val {thr:.4f}, веса {weights}")
    else:
        event(f"real без дообучения (абляция отклонила AdaBN и маску) на {names}, порог {thr:.4f}, веса {weights}")
    try:
        typer = load_typer()
    except Exception as exc:
        typer = None
        print("типизатор не загрузился", exc, flush=True)
    art = Path("artifacts/realnet")
    fig_dir = Path("docs/figures")
    all_rows = []
    file_summaries = {}
    expert = {}
    zone_store = {}
    shared = None if adapt else load_pretrained(a.ckpt)
    if shared is not None:
        save_model(shared, art / "model_real.pt", source=str(a.ckpt), use_adabn=False, use_mask=False)
    for held in names:
        t0 = time.time()
        hist = []
        if adapt:
            model = load_pretrained(a.ckpt)
            rng = np.random.default_rng(1000 + (int(held) if str(held).isdigit() else 0))
            tiles, wtiles = [], []
            for name in holdout_ids(names, held):
                X, info = channelize(by_name[name], need_wide=False)
                weight = X[2]
                n_tiles = max(a.tiles_per_file, 1)
                for _ in range(n_tiles):
                    tile, w = sample_tile(X, rng, weight)
                    tiles.append(tile)
                    wtiles.append(np.ones(tile.shape[-2:], np.float32) * (w > 0.5))
                del X
            if USE_ADABN:
                adapt_batchnorm(model.unet, tiles, batch=4)
            if USE_MASK:
                hist = train_mask_head(model, tiles, wtiles, minutes=a.real_minutes, seed=0, batch=4)
            save_model(model, art / f"model_holdout_{held}.pt", held_out=held,
                       adapted_on=holdout_ids(names, held), mask_steps=hist[-1].get("step") if hist else 0)
            del tiles, wtiles
        else:
            model = shared
        rec = by_name[held]
        need_wide = weights[2] > 0
        X, info = channelize(rec, need_wide=need_wide)
        comps = predict_components(model, X, info["residual"], info["sigma_x"], rec["valid"],
                                   need_mask=weights[1] > 0, need_wide=need_wide)
        score = combine_scores(*comps[:3], weights)
        zones = assemble_zones(score, comps[3], rec["valid"], rec["x"], rec["t"], info["residual"], thr)
        type_zones(typer, zones, X, info["sigma_x"], rec["x"], rec["t"])
        annotations = []
        try:
            annotations = attach_zone_cv(score, info["residual"], rec["valid"], rec["x"], rec["t"], zones)
        except Exception as exc:
            print(f"zone_cv {held}: {exc}", flush=True)
        rows = list(export.zone_rows(zones, f"{held}.csv", rec["sha256"], model_version=MODEL_VERSION))
        export.write_csv(art / f"zones_{held}.csv", rows)
        all_rows += rows
        save_figure(fig_dir / f"realnet_{held}.png", rec, info["residual"], score, zones, thr, annotations)
        alarm = [z for z in zones if z.get("alarm", True)]
        cls = {}
        for z in zones:
            cls[z.get("class") or ""] = cls.get(z.get("class") or "", 0) + 1
        flagged = (score >= thr) & rec["valid"]
        file_summaries[held] = dict(
            sha256=rec["sha256"], tz_missing=rec["tz_missing"], shape=list(rec["T"].shape),
            hours=round(float(rec["t"][-1] - rec["t"][0]) / 3600.0, 2),
            km=round(float(np.ptp(rec["x"][rec["valid"].any(0)])) / 1000.0, 3) if rec["valid"].any() else 0.0,
            n_harm=int(info["n_harm"]), span_h=round(float(info["span_h"]), 2),
            n_zones=len(zones), n_alarm=len(alarm),
            alarm_fraction=float(flagged[rec["valid"]].mean()) if rec["valid"].any() else 0.0,
            classes=cls, threshold=thr, weights=dict(seg=weights[0], mask=weights[1], wide=weights[2]),
            adapt_sec=round(time.time() - t0, 1), mask_steps=hist[-1].get("step") if hist else 0)
        described = [describe_zone(z) for z in zones]
        described.sort(key=lambda d: (-d["score"], -d["width_m"]))
        keep = []
        for d in described:
            if len(keep) < 15 or d["width_m"] >= 40 or (d["polarity"] == "cold" and abs(d["peak_dT_C"]) >= 0.4):
                keep.append(d)
        expert[held] = keep
        zone_store[held] = zones
        event(f"файл {held}: зон {len(zones)} тревог {len(alarm)} доля {file_summaries[held]['alarm_fraction']:.4f} "
              f"за {file_summaries[held]['adapt_sec']:.0f} с")
        del X, score, comps
        if adapt:
            del model
    export.write_csv(art / "zones_all.csv", all_rows)
    # карточки — после зон и порога, только подсчёт
    cards = []
    for cpath in a.cards:
        with open(cpath) as f:
            cards += [json.loads(line) for line in f if line.strip()]
    card_rows = []
    for c in cards:
        rec = by_name.get(str(c.get("record")))
        if rec is None:
            card_rows.append(dict(id=c.get("id"), record=str(c.get("record")), presence=c.get("presence"),
                                  status="missing_file"))
            continue
        # score уже не лежит в памяти: пересечение считаем по зонам, как hit_alarm в run_real
        zones = zone_store[str(c.get("record"))]
        xa, xb, ta, tb = domain.card_bounds(c)
        ix = (rec["x"] >= xa) & (rec["x"] <= xb)
        it = (rec["t"] >= ta) & (rec["t"] <= tb)
        if not ix.any() or not it.any():
            card_rows.append(dict(id=c.get("id"), record=str(c.get("record")), presence=c.get("presence"),
                                  status="box_outside_record", hit_alarm=False, hit_zone=False))
            continue
        alarm = domain.overlapping_zones(zones, xa, xb, ta, tb, True)
        any_z = domain.overlapping_zones(zones, xa, xb, ta, tb, False)
        dom = max(alarm or any_z, key=lambda z: z.get("n_px", 0)) if (alarm or any_z) else None
        card_rows.append(dict(
            id=c.get("id"), record=str(c.get("record")), presence=c.get("presence"), status="ok",
            shape=c.get("shape", ""), notes=c.get("notes", ""),
            hit_alarm=bool(alarm), hit_zone=bool(any_z),
            n_alarm=len(alarm), n_zones=len(any_z),
            dominant_class=None if dom is None else dom.get("class"),
            dominant_polarity=None if dom is None else dom.get("polarity")))
    tally_alarm = domain.presence_tally(card_rows, "hit_alarm")
    tally_zone = domain.presence_tally(card_rows, "hit_zone")
    blob = dict(done=True, method="realnet", threshold=thr, weights=dict(seg=weights[0], mask=weights[1], wide=weights[2]),
                threshold_source="synth pooled val far 1e-3, not cards, not real files",
                files=file_summaries, cards=card_rows, presence_alarm=tally_alarm, presence_zone=tally_zone,
                reference_real_v8=saved_card_tally("artifacts/eval/real_v8.json"),
                expert=expert, model_version=MODEL_VERSION)
    dump_json(blob, out_path)
    event(f"real cards yes {tally_alarm.get('yes_hit_alarm')}/{tally_alarm.get('yes_n')} "
          f"no {tally_alarm.get('no_hit_alarm')}/{tally_alarm.get('no_n')}")
    return blob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--ckpt", default="artifacts/stage1/unet_v810ft.pt")
    ap.add_argument("--cache-v8", default="artifacts/cache/v8")
    ap.add_argument("--cache-v10", default="artifacts/cache/v10")
    ap.add_argument("--real-root", default="/workspace/repo/src/MGTS_data")
    ap.add_argument("--cards", nargs="*", default=["data/operator_cards/cards_batch1.jsonl",
                                                   "data/operator_cards/cards_batch2.jsonl"])
    ap.add_argument("--adapt-minutes", type=float, default=2.0)
    ap.add_argument("--real-minutes", type=float, default=1.5)
    ap.add_argument("--n-adapt", type=int, default=12)
    ap.add_argument("--tiles-per-file", type=int, default=4)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--redo-synth", action="store_true")
    ap.add_argument("--redo-real", action="store_true")
    ap.add_argument("--skip-real", action="store_true")
    ap.add_argument("--skip-synth", action="store_true")
    a = ap.parse_args()
    if a.no_resume:
        a.resume = False
    torch.set_num_threads(a.threads)
    event("старт")
    if a.skip_synth:
        synth = json.load(open("artifacts/eval/realnet_synth.json"))
    else:
        synth = run_synth(a)
    if not a.skip_real:
        run_real(a, synth)
    event("REALNET_DONE")


if __name__ == "__main__":
    main()
