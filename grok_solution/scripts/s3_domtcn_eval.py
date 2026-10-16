"""Оценка DOM-TCN: val IoU@FAR 1e-3 на synth v8, порог только по фону v8 val.

Кандидаты фиксированы до test. Выбор — максимум val IoU.
Test в выбор не входит. Реальные CSV не калибруют порог.
Ансамбль с DOM U-Net — среднее вероятностей, не дообучение порога по test.

  /workspace/repo/run/.venv/bin/python -u scripts/s3_domtcn_eval.py
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_k] = "2"

import argparse
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
DOM_BLEND_VAL = 0.3898
MEMBERS = ("unet_w24", "unet_v810ft", "tcn", "pixel_lgbm", WINNER)
CANDIDATES = ("tcn_dom", "ens_pair", "ens_swap", "ens_swap_dom")
REFERENCES = (WINNER, "ens_blend_ref")
TMP = Path("/tmp/s3_domtcn_maps")
LOG = Path("/workspace/grok_logs/_events.log")


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


def _predict():
    spec = importlib.util.spec_from_file_location("s3_domtcn_train", ROOT / "scripts" / "s3_domtcn_train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.predict_batched


def load_members(split: str, name: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out = {}
    for m in MEMBERS:
        with np.load(Path("artifacts/pred/v8") / m / split / name) as z:
            out[m] = (np.asarray(z["s"], np.float32), np.asarray(z["h"], np.float32))
    return out


def combine(mem, tcn_s, tcn_h, dom_s, dom_h):
    def avg(pairs):
        s = sum(p[0] for p in pairs) / len(pairs)
        h = sum(p[1] for p in pairs) / len(pairs)
        return s.astype(np.float32), h.astype(np.float32)

    w24, v8ft, tcn, lgb, ens = (mem[k] for k in MEMBERS)
    td = (np.asarray(tcn_s, np.float32), np.asarray(tcn_h, np.float32))
    ud = (np.asarray(dom_s, np.float32), np.asarray(dom_h, np.float32))
    return {
        "tcn_dom": td,
        "ens_pair": avg([td, ud]),
        "ens_swap": avg([w24, v8ft, td, lgb]),
        "ens_swap_dom": avg([w24, v8ft, td, lgb, ud]),
        WINNER: ens,
        "ens_blend_ref": avg([ens, ud]),
    }


def bg_mask(ev, nu, valid):
    return valid & (ev == 0) & (nu == 0)


def open_map(model_name: str, split: str, stem: str, key: str, shape, dtype):
    path = TMP / model_name / split / f"{stem}_{key}.mmap"
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.memmap(path, dtype=dtype, mode="w+", shape=shape)


def read_map(model_name: str, split: str, stem: str, key: str, shape, dtype):
    path = TMP / model_name / split / f"{stem}_{key}.mmap"
    return np.memmap(path, dtype=dtype, mode="r", shape=shape)


def shapes_of(split: str) -> dict[str, tuple[int, int]]:
    out = {}
    for p in data.list_cache("artifacts/cache/v8", split):
        with np.load(Path("artifacts/pred/v8") / WINNER / split / p.name) as z:
            out[p.stem] = tuple(z["s"].shape)
    return out


def predict_split(named_models, predict_batched, split: str, shapes) -> None:
    t0 = time.time()
    files = data.list_cache("artifacts/cache/v8", split)
    for i, p in enumerate(files, 1):
        shape = shapes[p.stem]
        with np.load(p) as z:
            X = np.asarray(z["X"], np.float32)
        if X.shape[1:] != shape:
            raise SystemExit(f"{p.name}: X {X.shape[1:]} != pred {shape}")
        for name, model in named_models:
            prob = predict_batched(model, X)
            if not np.isfinite(prob).all():
                raise SystemExit(f"non-finite probs {name} {p.name}")
            s = (1.0 - prob[0]).astype(np.float32)
            h = (prob[1] - prob[2]).astype(np.float16)
            sm = open_map(name, split, p.stem, "s", shape, np.float32)
            hm = open_map(name, split, p.stem, "h", shape, np.float16)
            sm[:] = s
            hm[:] = h
            sm.flush()
            hm.flush()
            del sm, hm, prob, s, h
        del X
        if i == 1 or i % 10 == 0 or i == len(files):
            log(f"pred {split} {i}/{len(files)} {time.time() - t0:.0f}s")


def maps_of(split: str, stem: str, shape):
    tcn_s = np.asarray(read_map("tcn_dom", split, stem, "s", shape, np.float32))
    tcn_h = np.asarray(read_map("tcn_dom", split, stem, "h", shape, np.float16), np.float32)
    dom_s = np.asarray(read_map("unet_dom", split, stem, "s", shape, np.float32))
    dom_h = np.asarray(read_map("unet_dom", split, stem, "h", shape, np.float16), np.float32)
    return tcn_s, tcn_h, dom_s, dom_h


def collect_thresholds(split_shapes, names) -> dict[str, float]:
    thr = {}
    n_bg = None
    for n in names:
        parts = []
        for stem, shape in split_shapes.items():
            with np.load(Path("artifacts/cache/v8/val") / f"{stem}.npz") as meta:
                mask = np.asarray(bg_mask(np.asarray(meta["ev"]), np.asarray(meta["nu"]), np.asarray(meta["valid"])))
            tcn_s, tcn_h, dom_s, dom_h = maps_of("val", stem, shape)
            score = combine(load_members("val", f"{stem}.npz"), tcn_s, tcn_h, dom_s, dom_h)[n][0]
            parts.append(np.asarray(score[mask], np.float32))
            del tcn_s, tcn_h, dom_s, dom_h, score
        bg = np.concatenate(parts)
        del parts
        thr[n] = float(M.far_threshold(bg, 1e-3))
        n_bg = int(bg.size)
        log(f"порог {n} {thr[n]:.6f} n_bg {n_bg}")
        del bg
    return thr, n_bg


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
        tcn_s, tcn_h, dom_s, dom_h = maps_of(split, stem, shape)
        packed = combine(load_members(split, f"{stem}.npz"), tcn_s, tcn_h, dom_s, dom_h)
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
        del packed, tcn_s, tcn_h, dom_s, dom_h
    report = {}
    for n in names:
        frame = pd.DataFrame(ev_rows[n])
        row = dict(
            threshold=float(thr[n]),
            far=float(far_n[n] / max(far_d[n], 1)),
            **M.prf(*tot[n]),
            hot=M.prf(*hot[n]),
            cold=M.prf(*cold[n]),
            iou_ci95=M.bootstrap_ci(per[n], sorted(per[n]), n=200, seed=0),
        )
        if len(frame):
            def agg(g):
                return dict(n=int(len(g)), recall=float(g.hit.mean()),
                            pixel_recall=float(g.tp.sum() / max(int(g.tp.sum() + g.fn.sum()), 1)))
            row["events_all"] = agg(frame)
            row["weak_events"] = agg(frame[frame.amp <= 2]) if (frame.amp <= 2).any() else dict(n=0, recall=None, pixel_recall=None)
            row["by_amp"] = {k: agg(g) for k, g in frame.groupby("amp_bin")}
        report[n] = row
        log(f"{split} {n} IoU {row['iou']:.4f} F1 {row['f1']:.4f} FAR {row['far']:.5f} "
            f"hot {row['hot']['iou']:.4f} cold {row['cold']['iou']:.4f}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tcn", default="artifacts/eval/s3_domtcn_tcn.pt")
    ap.add_argument("--unet", default="artifacts/eval/s3_dom_unet.pt")
    ap.add_argument("--skip-predict", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    t0 = time.time()
    predict_batched = _predict()
    labels = pd.read_csv("data/synth_meta/v8/labels.csv")
    seqs = pd.read_csv("data/synth_meta/v8/sequences.csv").set_index("seq_id")
    tcn = models.build("tcn", 32)
    tcn.load_state_dict(torch.load(a.tcn, map_location="cpu", weights_only=True))
    unet = models.build("unet", 16)
    unet.load_state_dict(torch.load(a.unet, map_location="cpu", weights_only=True))
    tcn.eval()
    unet.eval()
    rng = np.random.default_rng(0)
    probe = rng.standard_normal((3, 280, 600)).astype(np.float32)
    for name, model in (("tcn", tcn), ("unet", unet)):
        got = predict_batched(model, probe, batch=4)
        ref = models.predict_full(model, probe)
        err = float(np.max(np.abs(got - ref)))
        log(f"сверка {name} max|Δ| {err:.3e}")
        if err > 1e-4:
            raise SystemExit(f"predict расходится у {name}: {err}")
    del probe
    val_shapes = shapes_of("val")
    test_shapes = shapes_of("test")
    log(f"val {len(val_shapes)} test {len(test_shapes)}")
    named = (("tcn_dom", tcn), ("unet_dom", unet))
    if not a.skip_predict:
        if TMP.exists():
            shutil.rmtree(TMP)
        predict_split(named, predict_batched, "val", val_shapes)
        predict_split(named, predict_batched, "test", test_shapes)
    names = list(CANDIDATES) + list(REFERENCES)
    thr, n_bg = collect_thresholds(val_shapes, names)
    log("метрики val")
    val_rep = evaluate("val", val_shapes, thr, labels, seqs)
    log("метрики test")
    test_rep = evaluate("test", test_shapes, thr, labels, seqs)
    if TMP.exists() and not a.skip_predict:
        shutil.rmtree(TMP, ignore_errors=True)

    sanity = float(val_rep[WINNER]["iou"])
    sanity_test = float(test_rep[WINNER]["iou"])
    protocol_ok = abs(sanity - WINNER_VAL) < 0.005 and abs(sanity_test - WINNER_TEST) < 0.005
    selected = max(CANDIDATES, key=lambda n: (float(val_rep[n]["iou"]), -CANDIDATES.index(n)))
    blend_val = float(val_rep["ens_blend_ref"]["iou"])
    sel_val = float(val_rep[selected]["iou"])
    sel_test = float(test_rep[selected]["iou"])
    blend_close = abs(blend_val - DOM_BLEND_VAL) < 0.002
    beats_winner = bool(protocol_ok and sel_val > max(WINNER_VAL, sanity) + 1e-4)
    beats_dom = bool(protocol_ok and blend_close and sel_val > blend_val + 1e-4)
    # Новый победитель только если выше и опубликованного ансамбля, и DOM ens_blend.
    beats = bool(beats_winner and beats_dom)
    train_meta = {}
    tp = Path("artifacts/eval/s3_domtcn_train.json")
    if tp.exists():
        train_meta = json.loads(tp.read_text())
    report = dict(
        name="DOMTCN",
        protocol="v8 val IoU at FAR 1e-3; threshold from v8 val background (valid & ev==0 & nu==0); test not used for selection",
        published_winner=dict(name=WINNER, val_iou=WINNER_VAL, test_iou=WINNER_TEST),
        published_dom_blend=dict(name="ens_blend", val_iou=DOM_BLEND_VAL, source="artifacts/eval/s3_dom_v8.json"),
        sanity_winner=dict(val_iou=sanity, test_iou=sanity_test, protocol_ok=protocol_ok,
                           val_abs_err=abs(sanity - WINNER_VAL), test_abs_err=abs(sanity_test - WINNER_TEST)),
        n_bg=n_bg,
        candidates=list(CANDIDATES),
        candidate_meaning={
            "tcn_dom": "TCN w=32, init stage1/tcn.pt, fine-tuned with the DOM PSD/ACF background noise",
            "ens_pair": "mean(tcn_dom, unet_dom)",
            "ens_swap": "mean(unet_w24, unet_v810ft, tcn_dom, pixel_lgbm)",
            "ens_swap_dom": "mean(unet_w24, unet_v810ft, tcn_dom, pixel_lgbm, unet_dom)",
            "ens_blend_ref": "mean(ens_u2ft_tcn_lgbm, unet_dom), reference only, not a candidate",
        },
        thresholds=thr,
        val=val_rep,
        test=test_rep,
        selected=selected,
        selected_val_iou=sel_val,
        selected_test_iou=sel_test,
        beats_published_winner=beats_winner,
        beats_dom_blend=beats_dom,
        beats=beats,
        train=dict(
            best_epoch=train_meta.get("best_epoch"),
            best_proxy_iou=train_meta.get("best_proxy_iou"),
            seconds=train_meta.get("seconds"),
            stopped=train_meta.get("stopped"),
            real_traces_used=False,
        ),
        seconds=round(time.time() - t0, 1),
        real_traces_used=False,
    )
    out = Path("artifacts/eval/s3_domtcn_v8.json")
    out.write_text(json.dumps(plain(report), indent=1, ensure_ascii=False))
    weak = val_rep[selected].get("weak_events") or {}
    line = (
        f"DOMTCN_DONE selected={selected} val={sel_val:.4f} test={sel_test:.4f} "
        f"tcn_dom={float(val_rep['tcn_dom']['iou']):.4f} ens_pair={float(val_rep['ens_pair']['iou']):.4f} "
        f"ens_swap={float(val_rep['ens_swap']['iou']):.4f} ens_swap_dom={float(val_rep['ens_swap_dom']['iou']):.4f} "
        f"winner_val={sanity:.4f} dom_blend_val={blend_val:.4f} "
        f"beats={int(beats)} beats_winner={int(beats_winner)} beats_dom={int(beats_dom)} "
        f"best_epoch={train_meta.get('best_epoch')} train_sec={train_meta.get('seconds')} "
        f"eval_sec={report['seconds']} weak_val_recall={weak.get('recall')} real=0\n"
    )
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line)
    if beats:
        with LOG.open("a", encoding="utf-8") as fh:
            fh.write(f"BEATS_WINNER DOMTCN {selected} val {sel_val:.4f} test {sel_test:.4f}\n")
    log(line.strip())
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
