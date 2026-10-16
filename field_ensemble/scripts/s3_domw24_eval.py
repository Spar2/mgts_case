"""Оценка второго DOM U-Net и ансамбля с обоими DOM-участниками.

Кандидаты заданы до test. Выбор — максимум v8 val IoU при FAR 1e-3.
Порог — квантиль фона v8 val (valid & ev==0 & nu==0). Test в выбор не входит.
Реальные CSV не читаются. Карты пишутся во временный memmap и удаляются.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_k] = "2"

import importlib.util
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

from dts import data, metrics as M, models

WINNER = "ens_u2ft_tcn_lgbm"
WINNER_VAL = 0.38671235821914546
WINNER_TEST = 0.4052762065230473
# Выбор только среди новых карт. ens_dom — пересчёт уже опубликованного blend, в выбор не входит.
CANDIDATES = ("unet_domw24", "ens_w24", "ens_blend")
REFS = ("ens_dom", WINNER)
TMP = Path("/tmp/s3_domw24_maps")
DOM_W16 = "dom"
DOM_W24 = "w24"


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S', time.gmtime())} {msg}", flush=True)


def plain(obj):
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return plain(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _predict_mod():
    spec = importlib.util.spec_from_file_location("s3_dom_eval", ROOT / "scripts" / "s3_dom_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def shapes_of(split: str) -> dict[str, tuple[int, int]]:
    out = {}
    for p in data.list_cache("artifacts/cache/v8", split):
        z = np.load(Path("artifacts/pred/v8") / WINNER / split / p.name)
        out[p.stem] = tuple(int(v) for v in z["s"].shape)
        z.close()
    return out


def open_map(tag: str, split: str, stem: str, key: str, shape, dtype):
    path = TMP / tag / split / f"{stem}_{key}.mmap"
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.memmap(path, dtype=dtype, mode="w+", shape=shape)


def read_map(tag: str, split: str, stem: str, key: str, shape, dtype):
    path = TMP / tag / split / f"{stem}_{key}.mmap"
    return np.memmap(path, dtype=dtype, mode="r", shape=shape)


def predict_split(model, predict_batched, tag: str, split: str, shapes: dict[str, tuple[int, int]]) -> None:
    t0 = time.time()
    files = data.list_cache("artifacts/cache/v8", split)
    for i, p in enumerate(files, 1):
        shape = shapes[p.stem]
        with np.load(p) as z:
            X = np.asarray(z["X"], np.float32)
        if X.shape[1:] != shape:
            raise SystemExit(f"{p.name}: X {X.shape[1:]} != pred {shape}")
        prob = predict_batched(model, X)
        s = (1.0 - prob[0]).astype(np.float32)
        h = (prob[1] - prob[2]).astype(np.float16)
        sm = open_map(tag, split, p.stem, "s", shape, np.float32)
        hm = open_map(tag, split, p.stem, "h", shape, np.float16)
        sm[:] = s
        hm[:] = h
        sm.flush()
        hm.flush()
        del sm, hm, X, prob, s, h
        if i == 1 or i % 10 == 0 or i == len(files):
            log(f"pred {tag} {split} {i}/{len(files)} {time.time() - t0:.0f}s")


def load_pair(tag: str, split: str, stem: str, shape):
    s = np.asarray(read_map(tag, split, stem, "s", shape, np.float32))
    h = np.asarray(read_map(tag, split, stem, "h", shape, np.float16), np.float32)
    return s, h


def load_winner(split: str, stem: str):
    with np.load(Path("artifacts/pred/v8") / WINNER / split / f"{stem}.npz") as z:
        return np.asarray(z["s"], np.float32), np.asarray(z["h"], np.float32)


def avg(pairs):
    s = sum(p[0] for p in pairs) / len(pairs)
    h = sum(p[1] for p in pairs) / len(pairs)
    return s.astype(np.float32, copy=False), h.astype(np.float32, copy=False)


def combine(ens, dom, w24) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """ens_blend — равное среднее опубликованного ансамбля и обоих DOM U-Net."""
    return {
        "unet_domw24": w24,
        "ens_w24": avg([ens, w24]),
        "ens_blend": avg([ens, dom, w24]),
        "ens_dom": avg([ens, dom]),
        WINNER: ens,
    }


def bg_mask(ev, nu, valid):
    return valid & (ev == 0) & (nu == 0)


def collect_thresholds(shapes, names) -> dict[str, float]:
    thr = {}
    cache = Path("artifacts/cache/v8/val")
    for n in names:
        parts = []
        for stem, shape in shapes.items():
            with np.load(cache / f"{stem}.npz") as meta:
                mask = np.asarray(bg_mask(np.asarray(meta["ev"]), np.asarray(meta["nu"]), np.asarray(meta["valid"])))
            packed = combine(load_winner("val", stem), load_pair(DOM_W16, "val", stem, shape), load_pair(DOM_W24, "val", stem, shape))
            parts.append(np.asarray(packed[n][0][mask], np.float32))
            del packed
        bg = np.concatenate(parts)
        del parts
        thr[n] = float(M.far_threshold(bg, 1e-3))
        log(f"порог {n} {thr[n]:.6f} n_bg {bg.size}")
        del bg
    return thr


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
    t0 = time.time()
    items = list(shapes.items())
    for i, (stem, shape) in enumerate(items, 1):
        with np.load(cache / f"{stem}.npz") as meta:
            ev = np.asarray(meta["ev"])
            valid = np.asarray(meta["valid"], bool)
            nu = np.asarray(meta["nu"])
            x = np.asarray(meta["x"])
            t = np.asarray(meta["t"])
        mask = bg_mask(ev, nu, valid)
        truth = ev > 0
        hot_t = (ev >= 1) & (ev <= 4)
        cold_t = ev >= 5
        packed = combine(load_winner(split, stem), load_pair(DOM_W16, split, stem, shape), load_pair(DOM_W24, split, stem, shape))
        boxes = []
        for r in M.label_boxes(labels, stem).itertuples():
            if r.kind != "event":
                continue
            px = M.event_pixels(r, x, t, ev)
            if not px.any():
                continue
            boxes.append((r, px))
        template = ""
        if stem in seqs.index:
            template = seqs.loc[stem, "template"]
            if not np.ndim(template) == 0:
                template = template.iloc[0]
            template = str(template)
        for n in names:
            s, h = packed[n]
            pred = (s >= thr[n]) & valid
            tot[n] += np.asarray(M.confusion(pred, truth, valid), np.int64)
            per[n][stem] = M.confusion(pred, truth, valid)
            hot[n] += np.asarray(M.confusion(pred & (h > 0), hot_t, valid), np.int64)
            cold[n] += np.asarray(M.confusion(pred & (h <= 0), cold_t, valid), np.int64)
            far_n[n] += int((pred & mask).sum())
            far_d[n] += int(mask.sum())
            for r, px in boxes:
                frac = float(pred[px].mean())
                ev_rows[n].append(dict(
                    label=str(r.label), amp=float(r.amp_obs_C), amp_bin=M.amp_bin(float(r.amp_obs_C)),
                    template=template, frac=frac, hit=bool(frac >= 0.3),
                    tp=int((pred & px).sum()), fn=int((~pred & px).sum()),
                ))
        del packed
        if i == 1 or i % 15 == 0 or i == len(items):
            log(f"eval {split} {i}/{len(items)} {time.time() - t0:.0f}s")
    report = {}
    for n in names:
        E = pd.DataFrame(ev_rows[n])
        row = dict(
            threshold=float(thr[n]),
            far=float(far_n[n] / max(far_d[n], 1)),
            **M.prf(*[int(v) for v in tot[n]]),
            hot=M.prf(*[int(v) for v in hot[n]]),
            cold=M.prf(*[int(v) for v in cold[n]]),
            n_bg=int(far_d[n]),
            n_seq=len(per[n]),
            iou_ci95=M.bootstrap_ci(per[n], sorted(per[n]), n=200, seed=0),
        )
        if len(E):
            def agg(g):
                tp = int(g.tp.sum())
                fn = int(g.fn.sum())
                return dict(n=int(len(g)), recall=float(g.hit.mean()), pixel_recall=float(tp / max(tp + fn, 1)))
            row["events_all"] = agg(E)
            row["weak_events"] = agg(E[E.amp <= 2]) if (E.amp <= 2).any() else dict(n=0, recall=None, pixel_recall=None)
            row["by_amp"] = {str(k): agg(g) for k, g in E.groupby("amp_bin")}
            row["by_label"] = {str(k): agg(g) for k, g in E.groupby("label")}
        report[n] = row
        log(f"{split} {n} IoU {row['iou']:.4f} F1 {row['f1']:.4f} FAR {row['far']:.5f} "
            f"hot {row['hot']['iou']:.4f} cold {row['cold']['iou']:.4f}")
    return report


def load_unet(path: Path, width: int):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    got = int(sd["e1.0.weight"].shape[0])
    if got != width:
        raise SystemExit(f"{path}: ширина {got}, ожидалась {width}")
    model = models.build("unet", width)
    model.load_state_dict(sd)
    model.eval()
    return model


def main() -> None:
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    t0 = time.time()
    ckpt_w24 = Path("artifacts/eval/s3_domw24_unet.pt")
    ckpt_dom = Path("artifacts/eval/s3_dom_unet.pt")
    if not ckpt_w24.exists():
        raise SystemExit(f"нет {ckpt_w24}")
    if not ckpt_dom.exists():
        raise SystemExit(f"нет {ckpt_dom}")
    mod = _predict_mod()
    labels = pd.read_csv("data/synth_meta/v8/labels.csv")
    seqs = pd.read_csv("data/synth_meta/v8/sequences.csv").set_index("seq_id")
    val_shapes = shapes_of("val")
    test_shapes = shapes_of("test")
    log(f"val {len(val_shapes)} test {len(test_shapes)}")
    if TMP.exists():
        shutil.rmtree(TMP)
    try:
        model = load_unet(ckpt_w24, 24)
        err = mod.check_predict(model)
        predict_split(model, mod.predict_batched, DOM_W24, "val", val_shapes)
        predict_split(model, mod.predict_batched, DOM_W24, "test", test_shapes)
        del model
        model = load_unet(ckpt_dom, 16)
        predict_split(model, mod.predict_batched, DOM_W16, "val", val_shapes)
        predict_split(model, mod.predict_batched, DOM_W16, "test", test_shapes)
        del model
        names = list(CANDIDATES) + list(REFS)
        thr = collect_thresholds(val_shapes, names)
        log("метрики val")
        val_rep = evaluate("val", val_shapes, thr, labels, seqs)
        log("метрики test")
        test_rep = evaluate("test", test_shapes, thr, labels, seqs)
    finally:
        if TMP.exists():
            shutil.rmtree(TMP, ignore_errors=True)

    sanity = float(val_rep[WINNER]["iou"])
    sanity_test = float(test_rep[WINNER]["iou"])
    protocol_ok = abs(sanity - WINNER_VAL) < 0.005 and abs(sanity_test - WINNER_TEST) < 0.005
    if not protocol_ok:
        log(f"WARNING winner val {sanity:.4f} test {sanity_test:.4f} vs {WINNER_VAL:.4f}/{WINNER_TEST:.4f}")
    selected = max(CANDIDATES, key=lambda n: (float(val_rep[n]["iou"]), -CANDIDATES.index(n)))
    sel_val = float(val_rep[selected]["iou"])
    sel_test = float(test_rep[selected]["iou"])
    beats = bool(protocol_ok and sel_val > max(WINNER_VAL, sanity) + 1e-4)
    train_meta = {}
    tp = Path("artifacts/eval/s3_domw24_train.json")
    if tp.exists():
        train_meta = json.loads(tp.read_text())
    print(
        f"exact selected {selected} val {sel_val:.4f} test {sel_test:.4f} "
        f"ens_blend {float(val_rep['ens_blend']['iou']):.4f}/{float(test_rep['ens_blend']['iou']):.4f} "
        f"w24 {float(val_rep['unet_domw24']['iou']):.4f} "
        f"ens_dom {float(val_rep['ens_dom']['iou']):.4f} winner {sanity:.4f}",
        flush=True,
    )
    report = dict(
        name="DOMW24",
        protocol="v8 val IoU at FAR 1e-3; threshold from v8 val background only; test not used for selection",
        published_winner=dict(name=WINNER, val_iou=WINNER_VAL, test_iou=WINNER_TEST),
        sanity_winner=dict(
            val_iou=sanity, test_iou=sanity_test,
            val_abs_err=abs(sanity - WINNER_VAL), test_abs_err=abs(sanity_test - WINNER_TEST),
        ),
        predict_max_abs_err=err,
        candidates=list(CANDIDATES),
        candidate_meaning=dict(
            unet_domw24="U-Net width 24, init artifacts/stage1/unet_w24.pt, fine-tuned with aug_real",
            ens_w24="mean(ens_u2ft_tcn_lgbm, unet_domw24)",
            ens_blend="mean(ens_u2ft_tcn_lgbm, unet_dom, unet_domw24)",
            ens_dom="reference mean(ens_u2ft_tcn_lgbm, unet_dom); not a candidate",
        ),
        members=dict(
            ens_u2ft_tcn_lgbm="artifacts/pred/v8/ens_u2ft_tcn_lgbm",
            unet_dom=str(ckpt_dom),
            unet_domw24=str(ckpt_w24),
        ),
        thresholds=thr,
        val={n: val_rep[n] for n in list(CANDIDATES) + list(REFS)},
        test={n: test_rep[n] for n in list(CANDIDATES) + list(REFS)},
        selected=selected,
        selected_val_iou=sel_val,
        selected_test_iou=sel_test,
        protocol_ok=protocol_ok,
        beats_winner=beats,
        winner_margin=1e-4,
        train_best_epoch=train_meta.get("best_epoch"),
        train_best_proxy_iou=train_meta.get("best_proxy_iou"),
        train_seconds=train_meta.get("seconds"),
        train_stopped=train_meta.get("stopped"),
        aug=train_meta.get("aug"),
        shaper=train_meta.get("shaper"),
        spectrum_gap=train_meta.get("spectrum_gap"),
        real_used_for_training=False,
        real_used_for_threshold=False,
        cards="not used for selection",
        seconds=round(time.time() - t0, 1),
    )
    out = Path("artifacts/eval/s3_domw24_v8.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plain(report), indent=1, ensure_ascii=False))
    line = (
        f"DOMW24_DONE selected={selected} val_iou={sel_val:.4f} test_iou={sel_test:.4f} "
        f"thr={float(val_rep[selected]['threshold']):.6f} "
        f"val_far={float(val_rep[selected]['far']):.6f} test_far={float(test_rep[selected]['far']):.6f} "
        f"ens_blend_val={float(val_rep['ens_blend']['iou']):.4f} ens_blend_test={float(test_rep['ens_blend']['iou']):.4f} "
        f"ens_w24_val={float(val_rep['ens_w24']['iou']):.4f} unet_domw24_val={float(val_rep['unet_domw24']['iou']):.4f} "
        f"ens_dom_val={float(val_rep['ens_dom']['iou']):.4f} "
        f"winner_val={sanity:.4f} winner_test={sanity_test:.4f} "
        f"beats_winner={str(beats).lower()} epoch={train_meta.get('best_epoch')}"
    )
    elog = Path("/workspace/run_logs/_events.log")
    with elog.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
        if beats:
            f.write(f"BEATS_WINNER DOMW24 val {sel_val:.4f} test {sel_test:.4f}\n")
    print(line, flush=True)
    log(f"записан {out}")


if __name__ == "__main__":
    main()
