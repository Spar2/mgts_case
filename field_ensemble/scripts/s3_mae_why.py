"""Почему прирост ens_mean4 на test больше, чем на val.

Считает гистограммы скора и точные счётчики. Карты не пишет.
Порог FAR берётся из уже посчитанного artifacts/eval/s3_mae_v8.json.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "2")

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from dts import data
from dts import metrics as M
from dts.mae_anom import BAND, RADIUS, TILE, MaskedReconstructor, blend_scores, member_maps

WINNER = "ens_u2ft_tcn_lgbm"
OUT = Path("artifacts/eval/s3_mae_why.json")
EDGES = np.linspace(0.0, 1.0, 2001)
FAR_GRID = np.geomspace(1e-4, 1e-2, 17)
SPLITS = (("v8", "val"), ("v8", "test"), ("v10", "test"), ("v11", "test"))


def log(msg: str) -> None:
    print(time.strftime("%H:%M", time.gmtime()), msg, flush=True)


def add_hist(hist, values) -> None:
    if values.size == 0:
        return
    hist += np.histogram(np.clip(values, 0.0, 1.0), bins=EDGES)[0]


def survival(hist) -> np.ndarray:
    """Число значений >= EDGES[i] для i < последнего узла. Бин i — [EDGES[i], EDGES[i+1])."""
    return np.cumsum(hist[::-1])[::-1]


def at_thr(surv, thr: float) -> int:
    i = int(np.searchsorted(EDGES[:-1], min(thr, float(EDGES[-2])), side="right") - 1)
    i = min(max(i, 0), surv.size - 1)
    return int(surv[i])


def rates(tp, fp_bg, fp_nu, fn, n_bg, n_nu) -> dict:
    fp = fp_bg + fp_nu
    return dict(
        tp=int(tp), fp=int(fp), fn=int(fn),
        fp_bg=int(fp_bg), fp_nu=int(fp_nu),
        precision=float(tp / max(tp + fp, 1)),
        recall=float(tp / max(tp + fn, 1)),
        iou=float(tp / max(tp + fp + fn, 1)),
        far=float(fp_bg / max(n_bg, 1)),
        nu_fpr=float(fp_nu / max(n_nu, 1)),
        n_bg=int(n_bg), n_nu=int(n_nu),
    )


def from_surv(sb, sn, se, thr, n_bg, n_nu, n_ev) -> dict:
    fp_bg = at_thr(sb, thr)
    fp_nu = at_thr(sn, thr)
    tp = at_thr(se, thr)
    fn = int(n_ev - tp)
    row = rates(tp, fp_bg, fp_nu, fn, n_bg, n_nu)
    row["threshold"] = float(thr)
    return row


def thr_for_far(surv_bg, n_bg, far: float) -> float:
    """Порог, при котором доля фона >= thr ближе всего к far (гистограмма)."""
    target = far * n_bg
    ge = surv_bg.astype(np.float64)
    i = int(np.argmin(np.abs(ge - target)))
    return float(EDGES[i])


def thr_for_recall(surv_ev, n_ev, recall: float) -> float:
    target = recall * n_ev
    ge = surv_ev.astype(np.float64)
    i = int(np.argmin(np.abs(ge - target)))
    return float(EDGES[min(i, EDGES.size - 2)])


def empty_bucket():
    return dict(n=0, hit=0, tp=0, fn=0, px=0)


def bucket_row(b) -> dict:
    return dict(
        n=int(b["n"]),
        recall=float(b["hit"] / b["n"]) if b["n"] else None,
        pixel_recall=float(b["tp"] / max(b["tp"] + b["fn"], 1)) if b["px"] else None,
        tp=int(b["tp"]), fn=int(b["fn"]), px=int(b["px"]),
    )


def main() -> None:
    torch.set_num_threads(2)
    published = json.loads(Path("artifacts/eval/s3_mae_v8.json").read_text())
    train = json.loads(Path("artifacts/eval/s3_mae_train.json").read_text())
    knots = np.asarray(train["cdf_knots"], np.float64)
    model = MaskedReconstructor(16)
    model.load_state_dict(torch.load("artifacts/eval/s3_mae_recon.pt", map_location="cpu", weights_only=True))
    model.eval()
    thr = {
        "mae": float(published["val"]["mae"]["threshold"]),
        "ens_mean4": float(published["val"]["ens_mean4"]["threshold"]),
        WINNER: float(published["val"][WINNER]["threshold"]),
    }
    names = ("mae", "ens_mean4", WINNER)
    report = dict(
        question="почему delta test IoU ens_mean4 больше delta val, при mae alone val IoU около 0.05",
        frozen_thresholds=thr,
        histogram_bins=int(EDGES.size - 1),
        hit_frac=0.3,
        note="FP в IoU = фон + помеха (ev==0). FAR = fp_bg / n_bg, только valid & ev==0 & nu==0. Карты не сохранялись.",
        splits={},
    )
    kw = dict(tile=TILE, overlap=32, band=BAND, radius=RADIUS, tile_batch=2)
    for ver, split in SPLITS:
        t0 = time.time()
        labels = pd.read_csv(f"data/synth_meta/{ver}/labels.csv")
        files = data.list_cache(f"artifacts/cache/{ver}", split)
        hist = {n: {k: np.zeros(EDGES.size - 1, np.int64) for k in ("bg", "nu", "ev")} for n in names}
        exact = {n: dict(tp=0, fp_bg=0, fp_nu=0, fn=0) for n in names}
        shift = dict(n_bg=0, n_nu=0, n_ev=0, sum_w=0.0, sum_b=0.0, sum_m=0.0,
                     n_m_lt_w_bg=0, n_m_lt_w_ev=0, sum_w_ev=0.0, sum_m_ev=0.0, sum_b_ev=0.0)
        boxes = {n: {} for n in ("label", "amp")}
        for key in ("label", "amp"):
            boxes[key] = {n: {} for n in ("ens_mean4", WINNER)}
        comp = {"label_events": {}, "label_px": {}, "amp_events": {}, "amp_px": {}}
        for i, p in enumerate(files, 1):
            with np.load(p) as z:
                X = np.asarray(z["X"], np.float32)
                ev = np.asarray(z["ev"])
                valid = np.asarray(z["valid"]) > 0.5
                nu = np.asarray(z["nu"])
                x = np.asarray(z["x"], np.float64)
                t = np.asarray(z["t"], np.float64)
            s_mae, _h = member_maps(model, X[:2], X[2], knots, **kw)
            with np.load(Path("artifacts/pred") / ver / WINNER / split / f"{p.stem}.npz") as z:
                ws = np.asarray(z["s"], np.float32)
            blend = blend_scores(s_mae, ws, "ens_mean4")
            maps = {"mae": s_mae, "ens_mean4": blend, WINNER: ws}
            truth = valid & (ev > 0)
            bg = valid & (ev == 0) & (nu == 0)
            nup = valid & (ev == 0) & (nu > 0)
            shift["n_bg"] += int(bg.sum())
            shift["n_nu"] += int(nup.sum())
            shift["n_ev"] += int(truth.sum())
            if bg.any():
                wb, mb, bb = ws[bg], s_mae[bg], blend[bg]
                shift["sum_w"] += float(wb.sum())
                shift["sum_m"] += float(mb.sum())
                shift["sum_b"] += float(bb.sum())
                shift["n_m_lt_w_bg"] += int((mb < wb).sum())
            if truth.any():
                we, me, be = ws[truth], s_mae[truth], blend[truth]
                shift["sum_w_ev"] += float(we.sum())
                shift["sum_m_ev"] += float(me.sum())
                shift["sum_b_ev"] += float(be.sum())
                shift["n_m_lt_w_ev"] += int((me < we).sum())
            for n, s in maps.items():
                add_hist(hist[n]["bg"], s[bg])
                add_hist(hist[n]["nu"], s[nup])
                add_hist(hist[n]["ev"], s[truth])
                pred = s >= thr[n]
                exact[n]["tp"] += int((pred & truth).sum())
                exact[n]["fn"] += int((~pred & truth).sum())
                exact[n]["fp_bg"] += int((pred & bg).sum())
                exact[n]["fp_nu"] += int((pred & nup).sum())
            ev_boxes = []
            for r in M.label_boxes(labels, p.stem).itertuples():
                if r.kind != "event":
                    continue
                px = M.event_pixels(r, x, t, ev)
                if not px.any():
                    continue
                ev_boxes.append((str(r.label), M.amp_bin(float(r.amp_obs_C)), px))
            for lab, amp, px in ev_boxes:
                comp["label_events"][lab] = comp["label_events"].get(lab, 0) + 1
                comp["amp_events"][amp] = comp["amp_events"].get(amp, 0) + 1
                npx = int(px.sum())
                comp["label_px"][lab] = comp["label_px"].get(lab, 0) + npx
                comp["amp_px"][amp] = comp["amp_px"].get(amp, 0) + npx
                for n in ("ens_mean4", WINNER):
                    pred = maps[n] >= thr[n]
                    tp = int((pred & px).sum())
                    fn = int((~pred & px).sum())
                    frac = tp / max(tp + fn, 1)
                    for kind, key in (("label", lab), ("amp", amp)):
                        b = boxes[kind][n].setdefault(key, empty_bucket())
                        b["n"] += 1
                        b["hit"] += int(frac >= 0.3)
                        b["tp"] += tp
                        b["fn"] += fn
                        b["px"] += npx
            del X, s_mae, ws, blend, maps
            if i == 1 or i % 15 == 0 or i == len(files):
                log(f"{ver} {split} {i}/{len(files)} {time.time() - t0:.0f}s")
        n_bg, n_nu, n_ev = shift["n_bg"], shift["n_nu"], shift["n_ev"]
        surv = {n: {k: survival(hist[n][k]) for k in ("bg", "nu", "ev")} for n in names}
        frozen = {}
        for n in names:
            row = rates(exact[n]["tp"], exact[n]["fp_bg"], exact[n]["fp_nu"], exact[n]["fn"], n_bg, n_nu)
            row["threshold"] = thr[n]
            frozen[n] = row
        w, b = frozen[WINNER], frozen["ens_mean4"]
        dtp = b["tp"] - w["tp"]
        dfp = b["fp"] - w["fp"]
        iou_tp = (w["tp"] + dtp) / max(w["tp"] + dtp + w["fp"] + w["fn"] - dtp, 1)
        iou_fp = w["tp"] / max(w["tp"] + w["fp"] + dfp + w["fn"], 1)
        matched_far = []
        for far in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2):
            item = {"target_far": far}
            for n in ("ens_mean4", WINNER):
                t = thr_for_far(surv[n]["bg"], n_bg, far)
                item[n] = from_surv(surv[n]["bg"], surv[n]["nu"], surv[n]["ev"], t, n_bg, n_nu, n_ev)
            matched_far.append(item)
        # один recall: recall победителя и recall смеси на замороженном пороге
        matched_recall = {}
        for tag, src, dst in (
            ("at_winner_recall", WINNER, "ens_mean4"),
            ("at_blend_recall", "ens_mean4", WINNER),
        ):
            target = frozen[src]["recall"]
            t = thr_for_recall(surv[dst]["ev"], n_ev, target)
            matched_recall[tag] = dict(
                target_recall=target,
                source=src,
                other=dst,
                other_at_target=from_surv(surv[dst]["bg"], surv[dst]["nu"], surv[dst]["ev"], t, n_bg, n_nu, n_ev),
                source_frozen=frozen[src],
            )
        curve = []
        for far in FAR_GRID:
            point = {"target_far": float(far)}
            for n, key in (("ens_mean4", "ens_mean4"), (WINNER, "winner")):
                t = thr_for_far(surv[n]["bg"], n_bg, far)
                point[key] = from_surv(surv[n]["bg"], surv[n]["nu"], surv[n]["ev"], t, n_bg, n_nu, n_ev)
            curve.append(point)
        def pack_boxes(kind):
            keys = sorted(set(boxes[kind]["ens_mean4"]) | set(boxes[kind][WINNER]))
            out = {}
            for k in keys:
                out[k] = {
                    "ens_mean4": bucket_row(boxes[kind]["ens_mean4"].get(k, empty_bucket())),
                    WINNER: bucket_row(boxes[kind][WINNER].get(k, empty_bucket())),
                }
            return out
        report["splits"][f"{ver}/{split}"] = dict(
            seconds=round(time.time() - t0, 1),
            n_files=len(files),
            n_bg=n_bg, n_nu=n_nu, n_ev=n_ev,
            score_shift=dict(
                bg_mean_winner=shift["sum_w"] / max(n_bg, 1),
                bg_mean_mae=shift["sum_m"] / max(n_bg, 1),
                bg_mean_blend=shift["sum_b"] / max(n_bg, 1),
                ev_mean_winner=shift["sum_w_ev"] / max(n_ev, 1),
                ev_mean_mae=shift["sum_m_ev"] / max(n_ev, 1),
                ev_mean_blend=shift["sum_b_ev"] / max(n_ev, 1),
                frac_mae_lt_winner_bg=shift["n_m_lt_w_bg"] / max(n_bg, 1),
                frac_mae_lt_winner_ev=shift["n_m_lt_w_ev"] / max(n_ev, 1),
            ),
            frozen=frozen,
            attribution=dict(
                delta_tp=int(dtp),
                delta_fp=int(dfp),
                delta_fp_bg=int(b["fp_bg"] - w["fp_bg"]),
                delta_fp_nu=int(b["fp_nu"] - w["fp_nu"]),
                delta_fn=int(b["fn"] - w["fn"]),
                delta_iou=float(b["iou"] - w["iou"]),
                iou_if_only_tp_changes=float(iou_tp),
                iou_if_only_fp_changes=float(iou_fp),
            ),
            matched_far=matched_far,
            matched_recall=matched_recall,
            curve=curve,
            by_label=pack_boxes("label"),
            by_amp=pack_boxes("amp"),
            composition=comp,
        )
        log(f"{ver} {split} done blend {frozen['ens_mean4']['iou']:.4f} winner {frozen[WINNER]['iou']:.4f}")
        del hist, surv
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"wrote {OUT} bytes {OUT.stat().st_size}")


if __name__ == "__main__":
    main()
