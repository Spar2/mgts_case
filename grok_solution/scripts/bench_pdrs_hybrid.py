#!/usr/bin/env python3
"""Hybrid PDRS vs CC. Parameter choice on v8 val only, then frozen test v8/v10/v11.

Candidates:
  seed   — PDRS with seed = far 1e-3 (or a higher seed grown down to that floor)
  refine — CC support at far 1e-3, PDRS resegments inside it; fallback keeps a CC
           component that PDRS dropped

Selection (val only): event_recall >= CC on the same split. Among those,
maximise the mean of three gains from CC toward the reference PDRS point
(zone precision, boundary time, boundary distance). Tie-break: iou_all,
then event_recall. Reference PDRS is the frozen val point seed=0.95,
floor=0.7, r_saddle=0.3, region_median=0.5 — not refit here.
"""
import importlib.util
import itertools
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dts.pdrs import extract_zones_cc, extract_zones_hybrid, extract_zones_pdrs

_spec = importlib.util.spec_from_file_location("bench_pdrs", ROOT / "scripts" / "bench_pdrs.py")
_bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bench)
score_seq = _bench.score_seq

THR = 0.8608087301264272
PDRS_REF = dict(seed=0.95, floor=0.7, r_saddle=0.3, region_median=0.5)
MODEL = "ens_u2ft_tcn_lgbm"
RULE = ("eligible if event_recall >= cc val; then max mean gap-fraction "
        "from CC toward reference PDRS on zone_precision, bnd_err_t_s, bnd_err_x_m; "
        "tie-break iou_all, event_recall")


def grid():
    rows = []
    for floor, rs, rm in itertools.product([0.5, 0.7, THR], [0.3, 0.6, 0.9], [0.5, 0.7]):
        rows.append(dict(mode="seed", thr=THR, seed=THR, floor=floor, r_saddle=rs,
                         region_median=rm, fallback=False))
    for rs in (0.3, 0.6):
        rows.append(dict(mode="seed", thr=THR, seed=0.95, floor=THR, r_saddle=rs,
                         region_median=0.5, fallback=False))
    for core, rs, fb in itertools.product([0.9, 0.95], [0.3, 0.9], [True, False]):
        rows.append(dict(mode="refine", thr=THR, seed=core, floor=core, r_saddle=rs,
                         region_median=0.5, fallback=fb))
    return rows


def _call(method, params, score, valid, sign, time_s, dist_m):
    if method == "cc":
        return extract_zones_cc(score, valid, sign, **params)
    if method == "pdrs":
        return extract_zones_pdrs(score, valid, sign, time_s, dist_m, **params)
    return extract_zones_hybrid(score, valid, sign, time_s, dist_m, **params)


def run(method, params, files, pred_dir):
    acc = {k: np.zeros(3) for k in ("hot", "cold", "all")}
    ng = det = nz = zt = 0
    be, days, elapsed = [], 0.0, 0.0
    for path in files:
        data = np.load(path)
        pred = np.load(os.path.join(pred_dir, os.path.basename(path)))
        t0 = time.time()
        lab, zones = _call(method, params, pred["s"], data["valid"], pred["h"], data["t"], data["x"])
        elapsed += time.time() - t0
        counts, n_gt, n_det, n_zones, n_hit, bounds, span = score_seq(lab, zones, data)
        for key in acc:
            acc[key] += counts[key]
        ng += n_gt
        det += n_det
        nz += n_zones
        zt += n_hit
        be += bounds
        days += span
    out = {"method": method, "params": params}
    for key, (tp, fp, fn) in acc.items():
        out[f"iou_{key}"] = round(float(tp) / max(tp + fp + fn, 1), 4)
    out.update(
        event_recall=round(det / max(ng, 1), 4),
        zone_precision=round(zt / max(nz, 1), 4),
        zones_per_day=round(nz / max(days, 1e-9), 2),
        bnd_err_t_s=round(float(np.median([b[0] for b in be])) if be else -1, 1),
        bnd_err_x_m=round(float(np.median([b[1] for b in be])) if be else -1, 2),
        sec_per_seq=round(elapsed / max(len(files), 1), 3),
        n_files=len(files),
    )
    return out


def files_for(ver, split):
    cache = sorted((ROOT / "artifacts" / "cache" / ver / split).glob("*.npz"))
    pred = ROOT / "artifacts" / "pred" / ver / MODEL / split
    return [str(p) for p in cache if (pred / p.name).is_file()], str(pred)


def _gap_frac(value, worse, better, higher_is_better):
    span = (better - worse) if higher_is_better else (worse - better)
    if abs(span) < 1e-9:
        return 0.0
    return ((value - worse) if higher_is_better else (worse - value)) / span


def joint_score(row, cc, ref):
    parts = (
        _gap_frac(row["zone_precision"], cc["zone_precision"], ref["zone_precision"], True),
        _gap_frac(row["bnd_err_t_s"], cc["bnd_err_t_s"], ref["bnd_err_t_s"], False),
        _gap_frac(row["bnd_err_x_m"], cc["bnd_err_x_m"], ref["bnd_err_x_m"], False),
    )
    return sum(parts) / 3.0


def choose(rows, cc, ref):
    cc_recall = cc["event_recall"]
    eligible = [r for r in rows if r["event_recall"] + 1e-12 >= cc_recall]
    pool = eligible or rows

    def rank(row):
        return (joint_score(row, cc, ref), row["iou_all"], row["event_recall"], row["zone_precision"])

    best = max(pool, key=rank)
    picked = dict(best)
    picked["eligible_recall"] = bool(eligible)
    picked["cc_event_recall"] = cc_recall
    picked["joint_score"] = round(joint_score(best, cc, ref), 4)
    return picked


def main():
    ver = sys.argv[1] if len(sys.argv) > 1 else "all"
    versions = ["v8", "v10", "v11"] if ver == "all" else [ver]
    best_path = ROOT / "artifacts" / "eval" / "pdrs_hybrid_best_params.json"
    if "v8" in versions:
        val_files, val_pred = files_for("v8", "val")
        print(f"val files {len(val_files)}", flush=True)
        cc = run("cc", dict(thr=THR), val_files, val_pred)
        ref = run("pdrs", PDRS_REF, val_files, val_pred)
        print("val cc", cc["event_recall"], cc["zone_precision"], flush=True)
        print("val pdrs", ref["event_recall"], ref["zone_precision"], flush=True)
        rows = []
        for i, params in enumerate(grid(), 1):
            row = run("hybrid", params, val_files, val_pred)
            rows.append(row)
            print(f"val {i} {params['mode']} rec {row['event_recall']} prec {row['zone_precision']} "
                  f"iou {row['iou_all']} bnd {row['bnd_err_t_s']}/{row['bnd_err_x_m']}", flush=True)
        picked = choose(rows, cc, ref)
        best_params = picked["params"]
        best_path.parent.mkdir(parents=True, exist_ok=True)
        best_path.write_text(json.dumps(best_params))
        print("BEST", best_params, "eligible", picked["eligible_recall"], flush=True)
    else:
        best_params = json.loads(best_path.read_text())
        cc = ref = picked = rows = None
    for name in versions:
        test_files, test_pred = files_for(name, "test")
        payload = {"selection_rule": RULE, "threshold": THR}
        if name == "v8" and rows is not None:
            payload["val_cc"] = cc
            payload["val_pdrs"] = ref
            payload["val_grid"] = rows
            payload["best"] = {k: picked[k] for k in picked if k != "params"}
            payload["best"]["params"] = best_params
            payload["best_params"] = best_params
        else:
            payload["best_params"] = best_params
        payload["test"] = [
            run("cc", dict(thr=THR), test_files, test_pred),
            run("pdrs", PDRS_REF, test_files, test_pred),
            run("hybrid", best_params, test_files, test_pred),
        ]
        out = ROOT / "artifacts" / "eval" / f"pdrs_hybrid_{name}.json"
        out.write_text(json.dumps(payload, indent=1))
        for row in payload["test"]:
            print(name, row["method"], row["event_recall"], row["zone_precision"],
                  row["iou_all"], row["bnd_err_t_s"], row["bnd_err_x_m"], flush=True)


if __name__ == "__main__":
    main()
