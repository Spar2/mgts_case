"""Ансамбль RandConv U-Net с победителем и вложенная проверка как в s3_sel2_cv.

Полный val IoU — по всем пикселям, порог по фону v8 val.
5-fold и 2-fold — sequence_folds, evaluate_frames, micro_iou, nested_verdict
из scripts/s3_sel_core.py; компактный фон — _indices из scripts/s3_sel_eval.py.
Alpha, если считается, выбирается только на train-фолде.
Test порог не двигает. Реальные CSV не читаются.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ.setdefault(_k, "2")

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)

import s3_sel2_cv as cv
import s3_sel_core as sel
import s3_sel_eval as sev
from dts import data, metrics as M, models

WINNER_VAL = 0.38671235821914546
WINNER_THR = 0.8608087301264272
TMP = Path("/tmp/s3_randconv_maps")


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
    spec = importlib.util.spec_from_file_location("s3_dom_eval", ROOT / "scripts" / "s3_dom_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_model():
    model = models.build("unet", 16)
    model.load_state_dict(torch.load("artifacts/eval/s3_randconv_unet.pt", map_location="cpu", weights_only=True))
    model.eval()
    return model


def _cols(s, ev_idx, nu_idx, bg_idx):
    e, n, b = sev._take(s, ev_idx, nu_idx, bg_idx)
    return e.astype(np.float16), n.astype(np.float16), b.astype(np.float16)


def as_frames(rows, a_ens: float):
    """Один канал: a * победитель + (1-a) * RandConv. Вес evaluate_frames = 1."""
    a = float(a_ens)
    out = []
    for fr in rows:
        ev = a * np.asarray(fr["ens_ev"], np.float32) + (1.0 - a) * np.asarray(fr["rc_ev"], np.float32)
        nu = a * np.asarray(fr["ens_nu"], np.float32) + (1.0 - a) * np.asarray(fr["rc_nu"], np.float32)
        bg = a * np.asarray(fr["ens_bg"], np.float32) + (1.0 - a) * np.asarray(fr["rc_bg"], np.float32)
        out.append(dict(
            ev_m=ev[None], nu_m=nu[None], bg_m=bg[None],
            n_bg=fr["n_bg"], cooling=fr["cooling"],
        ))
    return out


def nested(rows, a_ens, ks=(2, 5)):
    made = as_frames(rows, a_ens)
    out = {}
    for k in ks:
        folds = sel.sequence_folds(len(made), k)
        parts, details = [], []
        for r, hold_idx in enumerate(folds):
            hold = sel.evaluate_frames(cv.take(made, hold_idx), np.array([1.0]))
            parts.append(hold)
            details.append(dict(
                fold=r, n_seq=len(hold_idx),
                stems=[rows[i]["stem"] for i in hold_idx],
                hold=cv.pack_metrics(hold),
            ))
            log(f"fixed a={a_ens:.2f} k={k} fold {r} IoU {hold['iou']:.4f}")
        micro = sel.micro_iou(parts)
        out[str(k)] = dict(micro_iou=micro, folds=details)
        log(f"fixed a={a_ens:.2f} k={k} micro {micro:.4f}")
    return dict(by_k=out, verdict=sel.nested_verdict({k: v["micro_iou"] for k, v in out.items()}))


def nested_alpha(rows, ks=(2, 5)):
    out = {}
    for k in ks:
        folds = sel.sequence_folds(len(rows), k)
        parts, details = [], []
        for r, hold_idx in enumerate(folds):
            train_idx = [i for i in range(len(rows)) if i not in set(hold_idx)]
            best_a, best_iou = None, -1.0
            train = cv.take(rows, train_idx)
            for a in cv.ALPHAS:
                m = sel.evaluate_frames(as_frames(train, a), np.array([1.0]))
                if m["iou"] > best_iou:
                    best_iou = float(m["iou"])
                    best_a = float(a)
            hold = sel.evaluate_frames(as_frames(cv.take(rows, hold_idx), best_a), np.array([1.0]))
            parts.append(hold)
            details.append(dict(
                fold=r, n_seq=len(hold_idx), alpha_ens=best_a, train_iou=best_iou,
                stems=[rows[i]["stem"] for i in hold_idx], hold=cv.pack_metrics(hold),
            ))
            log(f"alpha k={k} fold {r} a={best_a:.2f} train {best_iou:.4f} hold {hold['iou']:.4f}")
        micro = sel.micro_iou(parts)
        out[str(k)] = dict(micro_iou=micro, folds=details)
        log(f"alpha k={k} micro {micro:.4f}")
    return dict(by_k=out, verdict=sel.nested_verdict({k: v["micro_iou"] for k, v in out.items()}))


def prf_pass(pairs, thr):
    tot = np.zeros(3, np.int64)
    hot = np.zeros(3, np.int64)
    cold = np.zeros(3, np.int64)
    far_n = far_d = 0
    for s, h, ev, valid, nu in pairs:
        truth = ev > 0
        pred = (s >= thr) & valid
        tot += np.asarray(M.confusion(pred, truth, valid), np.int64)
        hot += np.asarray(M.confusion(pred & (h > 0), (ev >= 1) & (ev <= 4), valid), np.int64)
        cold += np.asarray(M.confusion(pred & (h <= 0), ev >= 5, valid), np.int64)
        bg = valid & (ev == 0) & (nu == 0)
        far_n += int((pred & bg).sum())
        far_d += int(bg.sum())
    row = dict(threshold=float(thr), far=float(far_n / max(far_d, 1)), **M.prf(*tot),
               hot=M.prf(*hot), cold=M.prf(*cold), n_bg=int(far_d))
    return row


def main() -> None:
    torch.set_num_threads(2)
    ev = _predict()
    model = load_model()
    err = ev.check_predict(model)
    log(f"сверка predict {err:.3e}")
    labels = pd.read_csv("data/synth_meta/v8/labels.csv")
    seqs = pd.read_csv("data/synth_meta/v8/sequences.csv").set_index("seq_id")
    files = data.list_cache("artifacts/cache/v8", "val")
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True)
    frames = []
    bg = {k: [] for k in ("rc", "ens", "blend")}
    t0 = time.time()
    try:
        for i, p in enumerate(files, 1):
            with np.load(p) as z:
                X = np.asarray(z["X"], np.float32)
                ev_map = np.asarray(z["ev"])
                valid = np.asarray(z["valid"], bool)
                nu = np.asarray(z["nu"])
                x = np.asarray(z["x"], np.float64)
                t = np.asarray(z["t"], np.float64)
            prob = ev.predict_batched(model, X)
            s = (1.0 - prob[0]).astype(np.float32)
            h = (prob[1] - prob[2]).astype(np.float32)
            with np.load(Path("artifacts/pred/v8/ens_u2ft_tcn_lgbm/val") / p.name) as w:
                sw = np.asarray(w["s"], np.float32)
                hw = np.asarray(w["h"], np.float32)
            if sw.shape != s.shape:
                raise SystemExit(f"{p.stem} {s.shape} != winner {sw.shape}")
            blend = (0.5 * (sw + s)).astype(np.float32)
            bh = (0.5 * (hw + h)).astype(np.float32)
            mask = valid & (ev_map == 0) & (nu == 0)
            bg["rc"].append(s[mask])
            bg["ens"].append(sw[mask])
            bg["blend"].append(blend[mask])
            for key, arr, dt in (("s", s, np.float32), ("h", h, np.float16),
                                 ("bs", blend, np.float32), ("bh", bh, np.float16),
                                 ("ws", sw, np.float32), ("wh", hw, np.float16)):
                mm = np.memmap(TMP / f"{p.stem}_{key}.mmap", dtype=dt, mode="w+", shape=arr.shape)
                mm[:] = arr
                mm.flush()
                del mm
            np.save(TMP / f"{p.stem}_ev.npy", ev_map)
            np.save(TMP / f"{p.stem}_valid.npy", valid)
            np.save(TMP / f"{p.stem}_nu.npy", nu)
            ev_idx, nu_idx, bg_idx, n_bg = sev._indices(ev_map, valid, nu, p.stem)
            re, rn, rb = _cols(s, ev_idx, nu_idx, bg_idx)
            ee, en, eb = _cols(sw, ev_idx, nu_idx, bg_idx)
            cooling, _rows = sev._events_for(p.stem, ev_map, x, t, labels, seqs, ev_idx)
            frames.append(dict(
                stem=p.stem, n_bg=n_bg, cooling=cooling,
                rc_ev=re, rc_nu=rn, rc_bg=rb, ens_ev=ee, ens_nu=en, ens_bg=eb,
            ))
            del X, prob, s, h, sw, hw, blend, bh
            if i == 1 or i % 10 == 0 or i == len(files):
                log(f"val {i}/{len(files)} {time.time() - t0:.0f}s")
        frames.sort(key=lambda fr: fr["stem"])
        thr = {}
        for k, parts in bg.items():
            cat = np.concatenate(parts)
            thr[k] = float(M.far_threshold(cat, 1e-3))
            log(f"порог {k} {thr[k]:.6f} n {cat.size}")
            del cat
        del bg

        def load_pairs(kind):
            pairs = []
            for p in files:
                shape = None
                ev_map = np.load(TMP / f"{p.stem}_ev.npy")
                shape = ev_map.shape
                valid = np.load(TMP / f"{p.stem}_valid.npy")
                nu = np.load(TMP / f"{p.stem}_nu.npy")
                sk, hk = {"rc": ("s", "h"), "ens": ("ws", "wh"), "blend": ("bs", "bh")}[kind]
                dt_s = np.float32
                s = np.asarray(np.memmap(TMP / f"{p.stem}_{sk}.mmap", dtype=dt_s, mode="r", shape=shape))
                h = np.asarray(np.memmap(TMP / f"{p.stem}_{hk}.mmap", dtype=np.float16, mode="r", shape=shape), np.float32)
                pairs.append((s, h, ev_map, valid, nu))
            return pairs

        val = {
            "unet_randconv": prf_pass(load_pairs("rc"), thr["rc"]),
            "ens_u2ft_tcn_lgbm": prf_pass(load_pairs("ens"), thr["ens"]),
            "ens_randconv": prf_pass(load_pairs("blend"), thr["blend"]),
        }
        for name, row in val.items():
            log(f"val {name} IoU {row['iou']:.4f} thr {row['threshold']:.6f}")
        shutil.rmtree(TMP, ignore_errors=True)

        log("nested winner / blend 0.5 / alpha")
        nest_w = nested(frames, 1.0)
        nest_b = nested(frames, 0.5)
        nest_a = nested_alpha(frames)
        gain = float(nest_b["verdict"]["worst"] - nest_w["verdict"]["worst"])
        gain_alpha = float(nest_a["verdict"]["worst"] - nest_w["verdict"]["worst"])
        log(f"nested gain fixed {gain:.4f} alpha {gain_alpha:.4f}")

        # test, порог val уже заморожен
        acc_n = {k: [0, 0] for k in ("rc", "ens", "blend")}
        acc = {k: np.zeros(3, np.int64) for k in ("rc", "ens", "blend", "rch", "ensh", "blendh", "rcc", "ensc", "blendc")}
        test_files = data.list_cache("artifacts/cache/v8", "test")
        t1 = time.time()
        for i, p in enumerate(test_files, 1):
            with np.load(p) as z:
                X = np.asarray(z["X"], np.float32)
                ev_map = np.asarray(z["ev"])
                valid = np.asarray(z["valid"], bool)
                nu = np.asarray(z["nu"])
            prob = ev.predict_batched(model, X)
            s = (1.0 - prob[0]).astype(np.float32)
            h = (prob[1] - prob[2]).astype(np.float32)
            with np.load(Path("artifacts/pred/v8/ens_u2ft_tcn_lgbm/test") / p.name) as w:
                sw = np.asarray(w["s"], np.float32)
                hw = np.asarray(w["h"], np.float32)
            fields = {
                "rc": (s, h, thr["rc"]),
                "ens": (sw, hw, thr["ens"]),
                "blend": ((0.5 * (sw + s)).astype(np.float32), (0.5 * (hw + h)).astype(np.float32), thr["blend"]),
            }
            truth = ev_map > 0
            bgm = valid & (ev_map == 0) & (nu == 0)
            for key, (ss, hh, t) in fields.items():
                pred = (ss >= t) & valid
                acc[key] += np.asarray(M.confusion(pred, truth, valid), np.int64)
                acc[key + "h"] += np.asarray(M.confusion(pred & (hh > 0), (ev_map >= 1) & (ev_map <= 4), valid), np.int64)
                acc[key + "c"] += np.asarray(M.confusion(pred & (hh <= 0), ev_map >= 5, valid), np.int64)
                acc_n[key][0] += int((pred & bgm).sum())
                acc_n[key][1] += int(bgm.sum())
            del X, prob
            if i == 1 or i % 15 == 0 or i == len(test_files):
                log(f"test {i}/{len(test_files)} {time.time() - t1:.0f}s")
    finally:
        shutil.rmtree(TMP, ignore_errors=True)

    def pack(key, t):
        return dict(threshold=float(t), far=float(acc_n[key][0] / max(acc_n[key][1], 1)),
                    **M.prf(*acc[key]), hot=M.prf(*acc[key + "h"]), cold=M.prf(*acc[key + "c"]),
                    n_bg=int(acc_n[key][1]))

    test = {
        "unet_randconv": pack("rc", thr["rc"]),
        "ens_u2ft_tcn_lgbm": pack("ens", thr["ens"]),
        "ens_randconv": pack("blend", thr["blend"]),
    }
    sanity = abs(val["ens_u2ft_tcn_lgbm"]["iou"] - WINNER_VAL) < 0.005
    full_gain = float(val["ens_randconv"]["iou"] - val["ens_u2ft_tcn_lgbm"]["iou"])
    accepted = bool(sanity and gain >= 0.003 and full_gain > 1e-4)
    report = dict(
        name="RANDCONV",
        paper="Xu et al., ICLR 2021",
        ensemble="0.5 * (ens_u2ft_tcn_lgbm + unet_randconv)",
        selection="v8 val pixel IoU at FAR 1e-3 of the fixed 0.5 ensemble; test not used",
        nested_protocol=(
            "Та же процедура, что scripts/s3_sel2_cv.py: sequence_folds по отсортированным "
            "последовательностям, 2 и 5 фолдов, порог FAR 1e-3 только по фону holdout, "
            "micro IoU = сумма tp/fp/fn, компактный фон до 200000. "
            "Прирост = min(k=2,k=5) смеси 0.5 минус тот же минимум у победителя. "
            "Принять, если прирост ≥ 0.003 и полный val тоже выше."
        ),
        nested_bar=sel.NESTED_BAR,
        nested_margin=sel.NESTED_MARGIN,
        predict_max_abs_err=err,
        published_winner_val_iou=WINNER_VAL,
        published_winner_threshold=WINNER_THR,
        sanity_winner_val_abs_err=abs(val["ens_u2ft_tcn_lgbm"]["iou"] - WINNER_VAL),
        sanity_ok=bool(sanity),
        val=val,
        test=test,
        full_val_gain=full_gain,
        nested_winner=nest_w,
        nested_blend_fixed=nest_b,
        nested_blend_alpha=nest_a,
        nested_gain_fixed=gain,
        nested_gain_alpha=gain_alpha,
        accepted=accepted,
        real_stats_used=False,
        real_csv_used=False,
    )
    out = Path("artifacts/eval/s3_randconv_v8.json")
    out.write_text(json.dumps(plain(report), indent=1, ensure_ascii=False))
    log(f"val blend {val['ens_randconv']['iou']:.4f} winner {val['ens_u2ft_tcn_lgbm']['iou']:.4f} "
        f"gain_full {full_gain:.4f} gain_nested {gain:.4f} accept {accepted}")
    log(f"записан {out}")


if __name__ == "__main__":
    main()
