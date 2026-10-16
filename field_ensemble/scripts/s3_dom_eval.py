"""Оценка DOM: val IoU@FAR 1e-3 на synth v8, порог только по фону v8 val.

Кандидаты заранее фиксированы. Победитель — максимум val IoU.
Test в выбор не входит. Реальные CSV не калибруют порог.
Карты dom пишутся во временный memmap и удаляются в конце.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_k] = "2"

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from dts import data, events as EV, metrics as M, models

WINNER = "ens_u2ft_tcn_lgbm"
WINNER_VAL = 0.38671235821914546
WINNER_TEST = 0.4052762065230473
MEMBERS = ("unet_w24", "unet_v810ft", "tcn", "pixel_lgbm", WINNER)
CANDIDATES = ("unet_dom", "ens_replace", "ens_plus", "ens_blend")
TMP = Path("/tmp/s3_dom_maps")


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S', time.gmtime())} {msg}", flush=True)


def plain(obj):
    """numpy 2 bool/float не подкласс Python и ломает json.dumps."""
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return plain(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def tile_starts(n: int, tile: int, step: int) -> list[int]:
    ts = list(range(0, max(n - tile, 0) + 1, step)) or [0]
    if ts[-1] + tile < n:
        ts.append(n - tile)
    return ts


@torch.inference_mode()
def predict_batched(model, X, tile=(256, 512), overlap=32, batch=4):
    X = np.asarray(X, np.float32)
    _, t_len, w_len = X.shape
    th, tw = tile
    step_t, step_w = th - 2 * overlap, tw - 2 * overlap
    ts = tile_starts(t_len, th, step_t)
    ws = tile_starts(w_len, tw, step_w)
    boxes = [(t0, w0) for t0 in ts for w0 in ws]
    out = np.zeros((3, t_len, w_len), np.float32)
    cnt = np.zeros((t_len, w_len), np.float32)
    for i in range(0, len(boxes), batch):
        chunk = boxes[i:i + batch]
        xs, meta = [], []
        for t0, w0 in chunk:
            sl = X[:, t0:t0 + th, w0:w0 + tw]
            ht, wt = sl.shape[1], sl.shape[2]
            if ht != th or wt != tw:
                sl = np.pad(sl, ((0, 0), (0, th - ht), (0, tw - wt)))
            xs.append(sl)
            meta.append((t0, w0, ht, wt))
        prob = torch.softmax(model(torch.from_numpy(np.stack(xs))), 1).numpy()
        for p, (t0, w0, ht, wt) in zip(prob, meta):
            out[:, t0:t0 + ht, w0:w0 + wt] += p[:, :ht, :wt]
            cnt[t0:t0 + ht, w0:w0 + wt] += 1
    out /= np.maximum(cnt, 1.0)
    return out


def check_predict(model) -> float:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((3, 300, 700)).astype(np.float32)
    a = predict_batched(model, X, batch=4)
    b = models.predict_full(model, X)
    err = float(np.max(np.abs(a - b)))
    log(f"сверка predict max|Δ| {err:.3e}")
    if err > 1e-4:
        raise SystemExit(f"predict_batched расходится с models.predict_full: {err}")
    return err


def load_members(split: str, name: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out = {}
    for m in MEMBERS:
        with np.load(Path("artifacts/pred/v8") / m / split / name) as z:
            out[m] = (np.asarray(z["s"], np.float32), np.asarray(z["h"], np.float32))
    return out


def combine(mem, dom_s, dom_h) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    def avg(pairs):
        s = sum(p[0] for p in pairs) / len(pairs)
        h = sum(p[1] for p in pairs) / len(pairs)
        return s.astype(np.float32), h.astype(np.float32)
    w24, v8, tcn, lgb, ens = (mem[k] for k in MEMBERS)
    dom = (dom_s, dom_h)
    return {
        "unet_dom": dom,
        "ens_replace": avg([w24, dom, tcn, lgb]),
        "ens_plus": avg([w24, v8, dom, tcn, lgb]),
        "ens_blend": avg([ens, dom]),
        WINNER: ens,
    }


def bg_mask(ev, nu, valid):
    return valid & (ev == 0) & (nu == 0)


def open_map(split: str, stem: str, key: str, shape, dtype):
    path = TMP / split / f"{stem}_{key}.mmap"
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.memmap(path, dtype=dtype, mode="w+", shape=shape)


def read_map(split: str, stem: str, key: str, shape, dtype):
    path = TMP / split / f"{stem}_{key}.mmap"
    return np.memmap(path, dtype=dtype, mode="r", shape=shape)


def shapes_of(split: str) -> dict[str, tuple[int, int]]:
    out = {}
    for p in data.list_cache("artifacts/cache/v8", split):
        z = np.load(Path("artifacts/pred/v8") / WINNER / split / p.name)
        out[p.stem] = tuple(z["s"].shape)
    return out


def predict_split(model, split: str, shapes: dict[str, tuple[int, int]]) -> None:
    t0 = time.time()
    files = data.list_cache("artifacts/cache/v8", split)
    for i, p in enumerate(files, 1):
        shape = shapes[p.stem]
        with np.load(p) as z:
            X = np.asarray(z["X"], np.float32)
        if X.shape[1:] != shape:
            raise SystemExit(f"{p.name}: X {X.shape} != pred {shape}")
        prob = predict_batched(model, X)
        s = (1.0 - prob[0]).astype(np.float32)
        h = (prob[1] - prob[2]).astype(np.float16)
        sm = open_map(split, p.stem, "s", shape, np.float32)
        hm = open_map(split, p.stem, "h", shape, np.float16)
        sm[:] = s
        hm[:] = h
        sm.flush()
        hm.flush()
        del sm, hm, X, prob, s, h
        if i == 1 or i % 10 == 0 or i == len(files):
            log(f"pred {split} {i}/{len(files)} {time.time() - t0:.0f}s")


def collect_thresholds(split_shapes, names) -> dict[str, float]:
    """По одному кандидату: фон val — десятки миллионов клеток, все сразу не держатся."""
    thr = {}
    for n in names:
        parts = []
        for stem, shape in split_shapes.items():
            with np.load(Path("artifacts/cache/v8/val") / f"{stem}.npz") as meta:
                mask = np.asarray(bg_mask(np.asarray(meta["ev"]), np.asarray(meta["nu"]), np.asarray(meta["valid"])))
            dom_s = np.asarray(read_map("val", stem, "s", shape, np.float32))
            dom_h = np.asarray(read_map("val", stem, "h", shape, np.float16), np.float32)
            score = combine(load_members("val", f"{stem}.npz"), dom_s, dom_h)[n][0]
            parts.append(np.asarray(score[mask], np.float32))
            del dom_s, dom_h, score
        bg = np.concatenate(parts)
        del parts
        thr[n] = M.far_threshold(bg, 1e-3)
        log(f"порог {n} {thr[n]:.6f} n_bg {bg.size}")
        del bg
    return thr


def prf_add(acc, pred, truth, valid):
    acc += np.asarray(M.confusion(pred, truth, valid), np.int64)


def evaluate(split: str, shapes, thr: dict[str, float], labels, seqs) -> dict:
    names = list(thr)
    tot = {n: np.zeros(3, np.int64) for n in names}
    hot = {n: np.zeros(3, np.int64) for n in names}
    cold = {n: np.zeros(3, np.int64) for n in names}
    far_n = {n: 0 for n in names}
    far_d = {n: 0 for n in names}
    per = {n: {} for n in names}
    ev_rows = {n: [] for n in names}
    cache = Path("artifacts/cache/v8") / split
    for stem, shape in shapes.items():
        with np.load(cache / f"{stem}.npz") as meta:
            ev = np.asarray(meta["ev"])
            valid = np.asarray(meta["valid"], bool)
            nu = np.asarray(meta["nu"])
            x = np.asarray(meta["x"])
            t = np.asarray(meta["t"])
        mask_bg = bg_mask(ev, nu, valid)
        truth = ev > 0
        hot_t = (ev >= 1) & (ev <= 4)
        cold_t = ev >= 5
        dom_s = np.asarray(read_map(split, stem, "s", shape, np.float32))
        dom_h = np.asarray(read_map(split, stem, "h", shape, np.float16), np.float32)
        packed = combine(load_members(split, f"{stem}.npz"), dom_s, dom_h)
        boxes = []
        for r in M.label_boxes(labels, stem).itertuples():
            if r.kind != "event":
                continue
            px = M.event_pixels(r, x, t, ev)
            if not px.any():
                continue
            boxes.append((r, px))
        for n in names:
            s, h = packed[n]
            pred = (s >= thr[n]) & valid
            prf_add(tot[n], pred, truth, valid)
            per[n][stem] = M.confusion(pred, truth, valid)
            prf_add(hot[n], pred & (h > 0), hot_t, valid)
            prf_add(cold[n], pred & (h <= 0), cold_t, valid)
            far_n[n] += int((pred & mask_bg).sum())
            far_d[n] += int(mask_bg.sum())
            for r, px in boxes:
                frac = float(pred[px].mean())
                ev_rows[n].append(dict(
                    label=r.label, amp=float(r.amp_obs_C), amp_bin=M.amp_bin(float(r.amp_obs_C)),
                    template=seqs.loc[stem, "template"] if stem in seqs.index else "",
                    frac=frac, hit=frac >= 0.3,
                    tp=int((pred & px).sum()), fn=int((~pred & px).sum()),
                ))
        del packed, dom_s, dom_h
    report = {}
    for n in names:
        E = pd.DataFrame(ev_rows[n])
        row = dict(
            threshold=float(thr[n]),
            far=float(far_n[n] / max(far_d[n], 1)),
            **M.prf(*tot[n]),
            hot=M.prf(*hot[n]),
            cold=M.prf(*cold[n]),
            iou_ci95=M.bootstrap_ci(per[n], sorted(per[n]), n=200, seed=0),
        )
        if len(E):
            def agg(g):
                return dict(n=int(len(g)), recall=float(g.hit.mean()),
                            pixel_recall=float(g.tp.sum() / max(int(g.tp.sum() + g.fn.sum()), 1)))
            row["events_all"] = agg(E)
            row["weak_events"] = agg(E[E.amp <= 2]) if (E.amp <= 2).any() else dict(n=0, recall=None, pixel_recall=None)
            row["by_amp"] = {k: agg(g) for k, g in E.groupby("amp_bin")}
            row["by_label"] = {k: agg(g) for k, g in E.groupby("label")}
        report[n] = row
        log(f"{split} {n} IoU {row['iou']:.4f} F1 {row['f1']:.4f} FAR {row['far']:.5f} "
            f"hot {row['hot']['iou']:.4f} cold {row['cold']['iou']:.4f}")
    return report


def _take_patches(X, valid, ban, sigma, n, seed, rng, shaper, augment: bool):
    """Патчи каналов. augment — шум сводного спектра фона, std=0.45 σ, до make_channels."""
    from dts import fbd
    from dts.augment import feature_delta

    raw = fbd.sample_patches(X, valid, n, (64, 128), seed, ban=ban, min_valid=0.85)
    if not augment or len(raw) == 0:
        return raw
    sig = np.full(raw.shape[-1], float(np.median(sigma)), np.float32)
    spec_src = shaper.n_src - 1
    out = np.empty_like(raw)
    for i in range(len(raw)):
        xt = raw[i]
        valid_t = xt[2] > 0.5
        T = xt[0] * sig[None, :]
        noise = shaper.sample(rng, T.shape, spec_src, 0.45, valid_t) * sig[None, :]
        delta = feature_delta(T, noise, valid_t)
        xt = np.array(xt, np.float32, copy=True)
        xt[0] = np.clip(xt[0] + delta[0], -12, 12)
        xt[1] = np.clip(xt[1] + delta[1], -12, 12)
        xt[:2] *= valid_t.astype(np.float32)
        out[i] = xt
    return out


def domain_fbd() -> dict:
    """FBD Фреше: фон synth v8 val против фона real, до и после спектральной добавки.

    Энкодер заморожен (models_release/det0_unet.pt). Реальные каналы — уже посчитанный
    domain_cache, только чтение. Порог и веса U-Net по этим патчам не подбираются.
    """
    from dts import aug_real, fbd

    enc_path = Path("models_release/det0_unet.pt")
    if not enc_path.exists():
        return {"available": False, "error": f"нет {enc_path}"}
    enc = fbd.BottleneckEncoder.load(enc_path)
    bank = aug_real.load_bank()
    shaper, fit = aug_real.fit_shaper(bank, steps=0, seed=0, tile=(64, 128), max_lag=12)
    rng = np.random.default_rng(0)
    synth_raw, synth_aug = [], []
    for i, p in enumerate(data.list_cache("artifacts/cache/v8", "val")[:6]):
        with np.load(p) as z:
            X = np.asarray(z["X"], np.float32)
            valid = np.asarray(z["valid"], bool)
            ban = np.asarray(z["ev"]) > 0
            sigma = np.asarray(z["sigma"], np.float32)
        synth_raw.append(_take_patches(X, valid, ban, sigma, 8, 1000 + i, rng, shaper, False))
        synth_aug.append(_take_patches(X, valid, ban, sigma, 8, 1000 + i, rng, shaper, True))
        del X
    real = []
    cache = Path("artifacts/eval/domain_cache")
    for fid in (1, 2, 3, 4, 5, 6):
        with np.load(cache / f"{fid}_ch.npz") as z:
            X = np.asarray(z["X"], np.float32)
            valid = np.asarray(z["valid"], bool)
            sigma = np.asarray(z["sigma"], np.float32)
            t = np.asarray(z["t"])
            x = np.asarray(z["x"])
        zones = json.loads((cache / f"{fid}_base.json").read_text())
        ban = fbd.boxes_to_mask(t, x, zones)
        real.append(_take_patches(X, valid & ~ban, ban, sigma, 8, 2000 + fid, rng, shaper, False))
        del X
    def cat(rows):
        rows = [r for r in rows if len(r)]
        return np.concatenate(rows, 0) if rows else np.zeros((0, 3, 64, 128), np.float32)
    a, b, c = cat(synth_raw), cat(synth_aug), cat(real)
    log(f"FBD патчи synth {len(a)} aug {len(b)} real {len(c)} dim {enc.dim}")
    if min(len(a), len(b), len(c)) < 8:
        return {"available": False, "error": "мало патчей", "n": [int(len(a)), int(len(b)), int(len(c))]}
    fa, fb, fc = enc.embed(a, batch=4), enc.embed(b, batch=4), enc.embed(c, batch=4)
    before = fbd.frechet_distance(fa, fc, shrink=True)
    after = fbd.frechet_distance(fb, fc, shrink=True)
    return {
        "available": True,
        "encoder": str(enc_path),
        "feature_dim": enc.dim,
        "patch": [64, 128],
        "n_synth": int(len(a)),
        "n_real": int(len(c)),
        "aug": "pooled ACF noise, std 0.45 sigma, p=1, median sigma of the file",
        "shaper_steps": 0,
        "shaper_base_acf_corr": fit.get("base_acf_corr"),
        "real_source": "artifacts/eval/domain_cache/{1..6}_ch.npz, ban = baseline zones",
        "before_synth_vs_real": before,
        "after_aug_synth_vs_real": after,
        "fbd_dropped": bool(after["fbd"] < before["fbd"]),
        "note": "Энкодер не дообучался. Меньший FBD — облако признаков ближе к реальному фону. В выбор детектора не входит.",
    }


def score_kl(split_shapes) -> dict:
    """KL зона/тишина по фиксированным меткам синтетики. Карта сети сюда зоны не выбирает."""
    from dts import fbd

    out = {}
    files = list(split_shapes.items())[:4]
    acc = {n: {"hot": [], "cold": [], "n_skipped": 0} for n in list(CANDIDATES) + [WINNER]}
    for stem, shape in files:
        with np.load(Path("artifacts/cache/v8/val") / f"{stem}.npz") as meta:
            ev = np.asarray(meta["ev"])
            valid = np.asarray(meta["valid"], bool)
            nu = np.asarray(meta["nu"])
            t = np.asarray(meta["t"])
        specs = fbd.zone_specs(ev, valid, t, nu)
        dom_s = np.asarray(read_map("val", stem, "s", shape, np.float32))
        dom_h = np.asarray(read_map("val", stem, "h", shape, np.float16), np.float32)
        packed = combine(load_members("val", f"{stem}.npz"), dom_s, dom_h)
        for n, (s, _) in packed.items():
            part = fbd.kl_zone_quiet(s, specs)
            acc[n]["hot"].extend(part["hot"])
            acc[n]["cold"].extend(part["cold"])
            acc[n]["n_skipped"] += part["n_skipped"]
        del packed, dom_s, dom_h
    for n, bag in acc.items():
        out[n] = {
            "hot": fbd.kl_summary(bag["hot"]),
            "cold": fbd.kl_summary(bag["cold"]),
            "n_skipped": bag["n_skipped"],
            "n_seq": len(files),
        }
    return out


def fbd_hook(kl: dict | None) -> dict:
    """Фреше до/после и, если есть, json агента task_A. Ошибка FBD не роняет IoU."""
    out = {"kl_zone_quiet_val4": kl}
    p = Path("artifacts/eval/fbd.json")
    if p.exists() and p.stat().st_size < 2_000_000:
        try:
            raw = json.loads(p.read_text())
            text = json.dumps(raw)
            out["fbd_json"] = raw if len(text) < 8000 else {"keys": list(raw)[:40] if isinstance(raw, dict) else "list"}
        except Exception as exc:
            out["fbd_json_error"] = str(exc)
    elif p.exists():
        out["fbd_json_bytes"] = p.stat().st_size
    try:
        out["domain"] = domain_fbd()
    except Exception as exc:
        out["domain"] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
        log(f"FBD ошибка {out['domain']['error']}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="artifacts/eval/s3_dom_unet.pt")
    ap.add_argument("--skip-predict", action="store_true")
    ap.add_argument("--no-fbd", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    t0 = time.time()
    labels = pd.read_csv("data/synth_meta/v8/labels.csv")
    seqs = pd.read_csv("data/synth_meta/v8/sequences.csv").set_index("seq_id")
    model = models.build("unet", 16)
    model.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=True))
    model.eval()
    err = check_predict(model)
    val_shapes = shapes_of("val")
    test_shapes = shapes_of("test")
    log(f"val {len(val_shapes)} test {len(test_shapes)}")
    kl = None
    try:
        if not a.skip_predict:
            if TMP.exists():
                shutil.rmtree(TMP)
            predict_split(model, "val", val_shapes)
            predict_split(model, "test", test_shapes)
        names = list(CANDIDATES) + [WINNER]
        thr = collect_thresholds(val_shapes, names)
        log("метрики val")
        val_rep = evaluate("val", val_shapes, thr, labels, seqs)
        log("метрики test")
        test_rep = evaluate("test", test_shapes, thr, labels, seqs)
        if not a.no_fbd:
            log("KL зон")
            kl = score_kl(val_shapes)
    finally:
        if TMP.exists() and not a.skip_predict:
            shutil.rmtree(TMP, ignore_errors=True)

    sanity = val_rep[WINNER]["iou"]
    log(f"sanity winner val {sanity:.4f} published {WINNER_VAL:.4f}")
    # выбор только по val IoU среди своих кандидатов; при равенстве — более раннее имя
    selected = max(CANDIDATES, key=lambda n: (val_rep[n]["iou"], -CANDIDATES.index(n)))
    protocol_ok = abs(sanity - WINNER_VAL) < 0.005 and abs(test_rep[WINNER]["iou"] - WINNER_TEST) < 0.005
    beats = bool(protocol_ok and val_rep[selected]["iou"] > max(WINNER_VAL, sanity) + 1e-4)
    train_meta = {}
    tp = Path("artifacts/eval/s3_dom_train.json")
    if tp.exists():
        train_meta = json.loads(tp.read_text())
    report = dict(
        name="DOM",
        protocol="v8 val IoU at FAR 1e-3; threshold = quantile of background (valid & ev==0 & nu==0) on v8 val; test not used for selection",
        published_winner=dict(name=WINNER, val_iou=WINNER_VAL, test_iou=WINNER_TEST),
        sanity_winner=dict(val=val_rep[WINNER], test=test_rep[WINNER],
                           val_abs_err=abs(val_rep[WINNER]["iou"] - WINNER_VAL),
                           test_abs_err=abs(test_rep[WINNER]["iou"] - WINNER_TEST)),
        predict_max_abs_err=err,
        candidates=CANDIDATES,
        candidate_meaning=dict(
            unet_dom="U-Net w=16, init unet_v810ft, fine-tuned with PSD/ACF background noise",
            ens_replace="mean(unet_w24, unet_dom, tcn, pixel_lgbm)",
            ens_plus="mean(unet_w24, unet_v810ft, unet_dom, tcn, pixel_lgbm)",
            ens_blend="mean(ens_u2ft_tcn_lgbm, unet_dom)",
        ),
        thresholds=thr,
        val={n: val_rep[n] for n in CANDIDATES},
        test={n: test_rep[n] for n in CANDIDATES},
        selected=selected,
        selected_val_iou=val_rep[selected]["iou"],
        selected_test_iou=test_rep[selected]["iou"],
        protocol_ok=protocol_ok,
        beats_winner=beats,
        spectrum_gap=train_meta.get("spectrum_gap"),
        train_best_val_loss=train_meta.get("best_val"),
        train_best_epoch=train_meta.get("best_epoch"),
        train_seconds=train_meta.get("seconds"),
        aug=train_meta.get("aug"),
        shaper=train_meta.get("shaper"),
        fbd={"pending": True},
        real_used_for_training=False,
        real_used_for_threshold=False,
        cards="not used for selection",
        seconds=round(time.time() - t0, 1),
    )
    out = Path("artifacts/eval/s3_dom_v8.json")
    out.write_text(json.dumps(plain(report), indent=1, ensure_ascii=False))
    log(f"selected {selected} val {report['selected_val_iou']:.4f} test {report['selected_test_iou']:.4f} beats {beats}")
    log(f"записан {out}")
    if not a.no_fbd:
        report["fbd"] = fbd_hook(kl)
        out.write_text(json.dumps(plain(report), indent=1, ensure_ascii=False))
        dom = report["fbd"].get("domain") if isinstance(report["fbd"], dict) else None
        if isinstance(dom, dict) and dom.get("available"):
            b = dom["before_synth_vs_real"]["fbd"]
            c = dom["after_aug_synth_vs_real"]["fbd"]
            log(f"FBD до {b:.4f} после {c:.4f} dropped {dom.get('fbd_dropped')}")
    else:
        report["fbd"] = {"skipped": True}
        out.write_text(json.dumps(plain(report), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
