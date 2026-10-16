"""CFAR и временное сглаживание score без переобучения.

Порог FAR=1e-3 считается на фоне val после преобразования. Test только в отчёте.
Выбор варианта — dts.postproc.choose — записывается до чтения test.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dts import data, metrics as M, postproc as PP

FAR = 1e-3


def load_split(cache, pred_root, split, limit):
    out = {}
    paths = data.list_cache(cache, split)
    if limit:
        paths = paths[:limit]
    for p in paths:
        z = np.load(p)
        q = np.load(Path(pred_root) / split / p.name)
        out[p.stem] = dict(
            s=q["s"].astype(np.float32),
            ev=z["ev"],
            nu=z["nu"],
            valid=z["valid"],
            x=z["x"],
            t=z["t"],
        )
    return out


def event_table(labels, sids):
    tables = {}
    for sid in sids:
        d = M.label_boxes(labels, sid)
        d = d[d.kind == "event"]
        tables[sid] = list(d.itertuples())
    return tables


def concat_bg(split, variant):
    parts = []
    n = 0
    for d in split.values():
        z = PP.apply(d["s"], d["valid"], variant)
        b = d["valid"] & (d["ev"] == 0) & (d["nu"] == 0)
        part = np.ascontiguousarray(z[b])
        parts.append(part)
        n += part.size
    out = np.empty(n, np.float32)
    i = 0
    for part in parts:
        out[i:i + part.size] = part
        i += part.size
    return out


def evaluate(split, boxes, variant, thr):
    tot = np.zeros(3, np.int64)
    far_n = far_d = 0
    rows = []
    for sid, d in split.items():
        z = PP.apply(d["s"], d["valid"], variant)
        pred = (z >= thr) & d["valid"]
        truth = d["ev"] > 0
        tot += M.confusion(pred, truth, d["valid"])
        b = d["valid"] & (d["ev"] == 0) & (d["nu"] == 0)
        far_n += int((pred & b).sum())
        far_d += int(b.sum())
        for r in boxes[sid]:
            px = M.event_pixels(r, d["x"], d["t"], d["ev"])
            if px.sum() == 0:
                continue
            frac = float(pred[px].mean())
            rows.append((float(r.amp_obs_C), frac >= 0.3))
    pr = M.prf(*tot)
    ev = np.array([h for _, h in rows], bool) if rows else np.zeros(0, bool)
    amp = np.array([a for a, _ in rows], np.float32) if rows else np.zeros(0, np.float32)
    weak = amp <= 2 if amp.size else amp
    return dict(
        threshold=float(thr),
        far=float(far_n / max(far_d, 1)),
        precision=float(pr["precision"]),
        recall=float(pr["recall"]),
        f1=float(pr["f1"]),
        iou=float(pr["iou"]),
        event_recall=float(ev.mean()) if ev.size else 0.0,
        event_n=int(ev.size),
        weak_recall=float(ev[weak].mean()) if weak.size and weak.any() else 0.0,
        weak_n=int(weak.sum()) if weak.size else 0,
    )


def write_deployed(variant, enable, threshold):
    path = ROOT / "dts" / "postproc.py"
    text = path.read_text()
    start = text.find("# --- deployed:start ---")
    end = text.find("# --- deployed:end ---")
    if start < 0 or end < 0:
        raise RuntimeError("deployed markers missing")
    thr = "None" if (not enable and variant == "identity") or threshold is None else repr(float(threshold))
    if variant == "identity":
        thr = "None"
        enable = False
    block = (
        "# --- deployed:start ---\n"
        f"DEPLOYED_VARIANT = {variant!r}\n"
        f"THRESHOLD = {thr}  # порог val FAR 1e-3 в пространстве варианта; None => не сдвигать\n"
        f"ENABLED = {bool(enable)}\n"
    )
    path.write_text(text[:start] + block + text[end:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(ROOT / "artifacts/cache/v8"))
    ap.add_argument("--pred", default=str(ROOT / "artifacts/pred/v8"))
    ap.add_argument("--labels", default=str(ROOT / "data/synth_meta/v8/labels.csv"))
    ap.add_argument("--ref", default=str(ROOT / "artifacts/eval/stage1_v8_ens.json"))
    ap.add_argument("--out", default=str(ROOT / "artifacts/eval/stage2b_postproc_v8.json"))
    ap.add_argument("--methods", default="unet,ens_unet_tcn_lgbm")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-deploy", action="store_true")
    a = ap.parse_args()
    methods = [m for m in a.methods.split(",") if m]
    labels = pd.read_csv(a.labels)
    ref = json.load(open(a.ref))
    report = dict(
        version="v8",
        far=FAR,
        protocol=(
            "Преобразование без меток. Порог — квантиль фона val (valid & ev==0 & nu==0) "
            "после преобразования, FAR=1e-3, dts.metrics.far_threshold. "
            "Пиксельные IoU/F1 как в scripts/eval_stage1.py. "
            "Событие зачтено, если порог покрыл ≥30% пикселей маски (±0.25 м, ±120 с). "
            "Слабое событие: amp_obs_C ≤ 2. Выбор: dts.postproc.choose, до чтения test."
        ),
        variants=list(PP.VARIANTS),
        variant_params=dict(
            cfar_eps=PP.EPS,
            mad_to_sigma=PP.MAD_TO_SIGMA,
            cfar_x_window=2 * PP.CFAR_X_HALF + 1,
            cfar_2d_t=PP.CFAR_2D_T,
            cfar_2d_x=PP.CFAR_2D_X,
            cfar_2d_step=PP.CFAR_2D_STEP,
            gauss_sigma="(k-1)/4",
            temporal="max(score, smooth along t)",
            iou_guard=PP.IOU_GUARD,
        ),
        published_identity=dict(
            unet_val_iou=ref["val"]["unet@far_1e-3"]["iou"],
            ens_val_iou=ref["val"]["ens_unet_tcn_lgbm@far_1e-3"]["iou"],
            unet_test_iou=ref["test"]["unet@far_1e-3"]["iou"],
            ens_test_iou=ref["test"]["ens_unet_tcn_lgbm@far_1e-3"]["iou"],
        ),
        selection={},
        val={},
        test={},
    )
    t_all = time.time()
    frozen = {}
    for method in methods:
        t0 = time.time()
        val = load_split(a.cache, Path(a.pred) / method, "val", a.limit)
        boxes = event_table(labels, val)
        print(method, "val loaded", len(val), "sec", round(time.time() - t0, 1), flush=True)
        rows = {}
        for variant in PP.VARIANTS:
            tv = time.time()
            bg = concat_bg(val, variant)
            thr = M.far_threshold(bg, FAR)
            del bg
            rows[variant] = evaluate(val, boxes, variant, thr)
            r = rows[variant]
            print(
                f"{method} val {variant} thr={r['threshold']:.6g} far={r['far']:.6f} "
                f"iou={r['iou']:.4f} f1={r['f1']:.4f} ev={r['event_recall']:.4f} "
                f"weak={r['weak_recall']:.4f} n={r['weak_n']} sec={time.time() - tv:.1f}",
                flush=True,
            )
        if not a.limit:
            published = ref["val"][f"{method}@far_1e-3"]["iou"]
            got = rows["identity"]["iou"]
            if abs(got - published) > 5e-4:
                print(f"SANITY MISMATCH {method} val iou {got} vs published {published}", flush=True)
                sys.exit(2)
            print(f"SANITY OK {method} val iou {got:.6f}", flush=True)
        picked = PP.choose(rows)
        picked["threshold"] = rows[picked["variant"]]["threshold"]
        picked["val"] = rows[picked["variant"]]
        picked["identity"] = rows["identity"]
        report["selection"][method] = picked
        report["val"][method] = rows
        frozen[method] = {k: rows[k]["threshold"] for k in rows}
        print(f"SELECTED {method} {picked['variant']} enable={picked['enable']}", flush=True)
        del val, boxes
        gc.collect()
    ens = "ens_unet_tcn_lgbm"
    if ens in report["selection"] and not a.limit and not a.no_deploy:
        sel = report["selection"][ens]
        write_deployed(sel["variant"], sel["enable"], sel["threshold"])
        print("DEPLOYED", sel["variant"], "enable", sel["enable"], "thr", sel["threshold"], flush=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out, "w"), indent=1, ensure_ascii=False)
    print("val frozen", out, flush=True)
    for method in methods:
        t0 = time.time()
        test = load_split(a.cache, Path(a.pred) / method, "test", a.limit)
        boxes = event_table(labels, test)
        print(method, "test loaded", len(test), "sec", round(time.time() - t0, 1), flush=True)
        rows = {}
        for variant in PP.VARIANTS:
            tv = time.time()
            rows[variant] = evaluate(test, boxes, variant, frozen[method][variant])
            r = rows[variant]
            print(
                f"{method} test {variant} far={r['far']:.6f} iou={r['iou']:.4f} "
                f"f1={r['f1']:.4f} ev={r['event_recall']:.4f} weak={r['weak_recall']:.4f} "
                f"sec={time.time() - tv:.1f}",
                flush=True,
            )
        if not a.limit:
            published = ref["test"][f"{method}@far_1e-3"]["iou"]
            got = rows["identity"]["iou"]
            if abs(got - published) > 5e-4:
                print(f"SANITY MISMATCH {method} test iou {got} vs published {published}", flush=True)
                sys.exit(2)
            print(f"SANITY OK {method} test iou {got:.6f}", flush=True)
        report["test"][method] = rows
        del test, boxes
        gc.collect()
        json.dump(report, open(out, "w"), indent=1, ensure_ascii=False)
    report["seconds"] = round(time.time() - t_all, 1)
    json.dump(report, open(out, "w"), indent=1, ensure_ascii=False)
    print("EVAL_DONE", report["seconds"], flush=True)


if __name__ == "__main__":
    main()
