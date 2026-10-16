"""Второй сид DOM и ансамбль двух U-Net. Выбор только по v8 val IoU@FAR 1e-3.

Кандидаты заранее: unet_s0, unet_s1, ens_two = среднее их score.
Порог каждого — квантиль его фона на v8 val. Test не выбирает.
ens_dom (половина победителя и s0) считается для сверки и в argmax не входит.
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
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from dts import data, metrics as M, models

WINNER_THR = 0.8608087301264272
DOM2_THR = 0.8960151672373281
DOM2_VAL = 0.38976295844350645
WINNER_VAL = 0.38671235821914546
CANDIDATES = ("unet_s0", "unet_s1", "ens_two")
TMP = Path("/tmp/s3_dom3_maps")


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


def _eval():
    spec = importlib.util.spec_from_file_location("s3_dom_eval", ROOT / "scripts" / "s3_dom_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_unet(path: str):
    model = models.build("unet", 16)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def _maps(prob):
    return (1.0 - prob[0]).astype(np.float32), (prob[1] - prob[2]).astype(np.float32)


def forward_pair(m0, m1, predict, X):
    s0, h0 = _maps(predict(m0, X))
    s1, h1 = _maps(predict(m1, X))
    s = (0.5 * (s0 + s1)).astype(np.float32)
    h = (0.5 * (h0 + h1)).astype(np.float32)
    return s0, h0, s1, h1, s, h


def _add(acc, pred, truth, valid):
    acc += np.asarray(M.confusion(pred, truth, valid), np.int64)


def empty_acc():
    keys = ("s0", "s1", "two", "dom2", "s0h", "s1h", "twoh", "dom2h", "s0c", "s1c", "twoc", "dom2c")
    return {k: np.zeros(3, np.int64) for k in keys}


def score_one(s, h, thr, ev, valid, nu, acc, tag):
    bg = valid & (ev == 0) & (nu == 0)
    truth = ev > 0
    hot_t = (ev >= 1) & (ev <= 4)
    cold_t = ev >= 5
    pred = (s >= thr) & valid
    _add(acc[tag], pred, truth, valid)
    _add(acc[tag + "h"], pred & (h > 0), hot_t, valid)
    _add(acc[tag + "c"], pred & (h <= 0), cold_t, valid)
    return int((pred & bg).sum()), int(bg.sum())


def summarize(acc, tag, far_n, far_d, thr) -> dict:
    return dict(threshold=float(thr), far=float(far_n / max(far_d, 1)), **M.prf(*acc[tag]),
                hot=M.prf(*acc[tag + "h"]), cold=M.prf(*acc[tag + "c"]), n_bg=int(far_d))


def store_val(m0, m1, predict, files):
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True)
    shapes = {}
    bg = {k: [] for k in ("s0", "s1", "two", "dom2")}
    t0 = time.time()
    for i, p in enumerate(files, 1):
        with np.load(p) as z:
            X = np.asarray(z["X"], np.float32)
            ev = np.asarray(z["ev"])
            valid = np.asarray(z["valid"], bool)
            nu = np.asarray(z["nu"])
        s0, h0, s1, h1, s, h = forward_pair(m0, m1, predict, X)
        win = Path("artifacts/pred/v8/ens_u2ft_tcn_lgbm/val") / f"{p.stem}.npz"
        with np.load(win) as w:
            sw = np.asarray(w["s"], np.float32)
        if sw.shape != s0.shape:
            raise RuntimeError(f"{p.stem} shape {s0.shape} vs winner {sw.shape}")
        dom2 = (0.5 * (sw + s0)).astype(np.float32)
        mask = valid & (ev == 0) & (nu == 0)
        bg["s0"].append(np.asarray(s0[mask]))
        bg["s1"].append(np.asarray(s1[mask]))
        bg["two"].append(np.asarray(s[mask]))
        bg["dom2"].append(np.asarray(dom2[mask]))
        shapes[p.stem] = list(s0.shape)
        for key, arr, dt in (("s0", s0, np.float32), ("h0", h0, np.float16),
                             ("s1", s1, np.float32), ("h1", h1, np.float16)):
            mm = np.memmap(TMP / f"{p.stem}_{key}.mmap", dtype=dt, mode="w+", shape=arr.shape)
            mm[:] = arr
            mm.flush()
            del mm
        np.save(TMP / f"{p.stem}_ev.npy", ev)
        np.save(TMP / f"{p.stem}_valid.npy", valid)
        np.save(TMP / f"{p.stem}_nu.npy", nu)
        del X, s0, h0, s1, h1, s, h, sw, dom2
        if i == 1 or i % 10 == 0 or i == len(files):
            log(f"v8 val {i}/{len(files)} {time.time() - t0:.0f}s")
    thr = {}
    for k, parts in bg.items():
        cat = np.concatenate(parts)
        thr[k] = float(M.far_threshold(cat, 1e-3))
        log(f"порог {k} {thr[k]:.6f} n_bg {cat.size}")
        del cat
    return shapes, thr


def metrics_stored(files, shapes, thr) -> dict:
    acc = empty_acc()
    far = {k: [0, 0] for k in ("s0", "s1", "two", "dom2")}
    tags = (("s0", "s0", "h0"), ("s1", "s1", "h1"), ("two", None, None))
    for p in files:
        stem = p.stem
        shape = tuple(shapes[stem])
        s0 = np.asarray(np.memmap(TMP / f"{stem}_s0.mmap", dtype=np.float32, mode="r", shape=shape))
        s1 = np.asarray(np.memmap(TMP / f"{stem}_s1.mmap", dtype=np.float32, mode="r", shape=shape))
        h0 = np.asarray(np.memmap(TMP / f"{stem}_h0.mmap", dtype=np.float16, mode="r", shape=shape), np.float32)
        h1 = np.asarray(np.memmap(TMP / f"{stem}_h1.mmap", dtype=np.float16, mode="r", shape=shape), np.float32)
        ev = np.load(TMP / f"{stem}_ev.npy")
        valid = np.load(TMP / f"{stem}_valid.npy")
        nu = np.load(TMP / f"{stem}_nu.npy")
        fields = {
            "s0": (s0, h0, thr["s0"]),
            "s1": (s1, h1, thr["s1"]),
            "two": ((0.5 * (s0 + s1)).astype(np.float32), (0.5 * (h0 + h1)).astype(np.float32), thr["two"]),
        }
        with np.load(Path("artifacts/pred/v8/ens_u2ft_tcn_lgbm/val") / f"{stem}.npz") as w:
            sw = np.asarray(w["s"], np.float32)
            hw = np.asarray(w["h"], np.float32)
        fields["dom2"] = ((0.5 * (sw + s0)).astype(np.float32), (0.5 * (hw + h0)).astype(np.float32), thr["dom2"])
        for tag, (s, h, t) in fields.items():
            n, d = score_one(s, h, t, ev, valid, nu, acc, tag)
            far[tag][0] += n
            far[tag][1] += d
    return acc, far


def metrics_test(m0, m1, predict, thr) -> dict:
    files = data.list_cache("artifacts/cache/v8", "test")
    acc = empty_acc()
    far = {k: [0, 0] for k in ("s0", "s1", "two")}
    t0 = time.time()
    for i, p in enumerate(files, 1):
        with np.load(p) as z:
            X = np.asarray(z["X"], np.float32)
            ev = np.asarray(z["ev"])
            valid = np.asarray(z["valid"], bool)
            nu = np.asarray(z["nu"])
        s0, h0, s1, h1, s, h = forward_pair(m0, m1, predict, X)
        fields = {"s0": (s0, h0, thr["s0"]), "s1": (s1, h1, thr["s1"]), "two": (s, h, thr["two"])}
        for tag, (ss, hh, t) in fields.items():
            n, d = score_one(ss, hh, t, ev, valid, nu, acc, tag)
            far[tag][0] += n
            far[tag][1] += d
        del X, s0, h0, s1, h1, s, h
        if i == 1 or i % 10 == 0 or i == len(files):
            log(f"v8 test {i}/{len(files)} {time.time() - t0:.0f}s")
    return acc, far


def row(acc, far, tag, thr):
    return summarize(acc, tag, far[tag][0], far[tag][1], thr)


def main() -> None:
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    ev = _eval()
    m0 = load_unet("artifacts/eval/s3_dom_unet.pt")
    m1 = load_unet("artifacts/eval/s3_dom_unet_s1.pt")
    err0 = ev.check_predict(m0)
    err1 = ev.check_predict(m1)
    log(f"сверка s0 {err0:.3e} s1 {err1:.3e}")
    files = data.list_cache("artifacts/cache/v8", "val")
    try:
        shapes, thr = store_val(m0, m1, ev.predict_batched, files)
        acc_v, far_v = metrics_stored(files, shapes, thr)
        shutil.rmtree(TMP, ignore_errors=True)
        acc_t, far_t = metrics_test(m0, m1, ev.predict_batched, thr)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)

    val = {
        "unet_s0": row(acc_v, far_v, "s0", thr["s0"]),
        "unet_s1": row(acc_v, far_v, "s1", thr["s1"]),
        "ens_two": row(acc_v, far_v, "two", thr["two"]),
        "ens_dom_ref": row(acc_v, far_v, "dom2", thr["dom2"]),
    }
    test = {
        "unet_s0": row(acc_t, far_t, "s0", thr["s0"]),
        "unet_s1": row(acc_t, far_t, "s1", thr["s1"]),
        "ens_two": row(acc_t, far_t, "two", thr["two"]),
    }
    for name in CANDIDATES:
        log(f"val {name} IoU {val[name]['iou']:.4f} thr {val[name]['threshold']:.6f}")
    log(f"val ens_dom_ref IoU {val['ens_dom_ref']['iou']:.4f} published {DOM2_VAL:.4f}")
    selected = max(CANDIDATES, key=lambda n: (val[n]["iou"], -CANDIDATES.index(n)))
    sanity_ok = abs(val["ens_dom_ref"]["iou"] - DOM2_VAL) < 0.002
    beats_dom2 = bool(sanity_ok and val[selected]["iou"] > DOM2_VAL + 1e-4)
    beats_winner = bool(sanity_ok and val[selected]["iou"] > WINNER_VAL + 1e-4)
    report = {
        "name": "DOM3",
        "recipe": "same as seed 0: init unet_v810ft, focal+tversky_ce, AdamW 1e-4, wd 0, BN frozen, tile 128x256, batch 8, 80 steps, budget 22 min",
        "seeds": {"unet_s0": 0, "unet_s1": 1},
        "candidates": CANDIDATES,
        "candidate_meaning": {
            "unet_s0": "DOM U-Net seed 0, artifacts/eval/s3_dom_unet.pt",
            "unet_s1": "DOM U-Net seed 1, same recipe, artifacts/eval/s3_dom_unet_s1.pt",
            "ens_two": "mean score of unet_s0 and unet_s1",
        },
        "selection": "max v8 val IoU at FAR 1e-3; tie keeps the earlier name; test not used",
        "selected": selected,
        "selected_val_iou": val[selected]["iou"],
        "selected_test_iou": test[selected]["iou"],
        "beats_dom2": beats_dom2,
        "beats_winner": beats_winner,
        "dom2_reference_val_iou": DOM2_VAL,
        "dom2_reference_threshold": DOM2_THR,
        "winner_val_iou": WINNER_VAL,
        "winner_threshold": WINNER_THR,
        "sanity_ens_dom_abs_err": abs(val["ens_dom_ref"]["iou"] - DOM2_VAL),
        "sanity_ok": sanity_ok,
        "predict_max_abs_err": {"s0": err0, "s1": err1},
        "val": val,
        "test": test,
        "real_used_for_training": False,
        "real_used_for_threshold": False,
    }
    out = Path("artifacts/eval/s3_dom3.json")
    out.write_text(json.dumps(plain(report), indent=1, ensure_ascii=False))
    log(f"selected {selected} val {val[selected]['iou']:.4f} test {test[selected]['iou']:.4f} "
        f"beats_dom2 {beats_dom2} beats_winner {beats_winner}")
    log(f"записан {out}")


if __name__ == "__main__":
    main()
