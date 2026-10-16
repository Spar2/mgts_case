"""ens_dom = 0.5 * (ens_u2ft_tcn_lgbm, unet_dom).

Порог только по фону v8 val. Бутстреп — пары последовательностей v8 val:
на каждой выборке с возвращением суммируются tp/fp/fn и берётся разность IoU.
v10 и v11 test тем же порогом, в порог не входят.
Карты не хранятся дольше прогона.
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
WINNER_VAL = 0.38671235821914546
WINNER_V10 = 0.4038977883892501
WINNER_V11 = 0.40390028295846503
TMP = Path("/tmp/s3_dom2_maps")
REL = Path("models_release_dom")


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


def paired_delta_ci(dom: np.ndarray, win: np.ndarray, n: int = 1000, seed: int = 0) -> dict:
    """dom/win: (N, 3) = tp, fp, fn по последовательностям. ДИ разности pooled IoU."""
    dom = np.asarray(dom, np.float64)
    win = np.asarray(win, np.float64)
    if dom.shape != win.shape or dom.ndim != 2 or dom.shape[1] != 3:
        raise ValueError("need paired (N, 3) counts")

    def iou(block):
        s = block.sum(0)
        return float(s[0] / max(s.sum(), 1.0))

    point = iou(dom) - iou(win)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n, np.float64)
    for i in range(n):
        idx = rng.integers(0, len(dom), len(dom))
        diffs[i] = iou(dom[idx]) - iou(win[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "n_seq": int(len(dom)),
        "n_boot": int(n),
        "seed": int(seed),
        "point_iou_dom": iou(dom),
        "point_iou_winner": iou(win),
        "point_delta": float(point),
        "boot_mean_delta": float(diffs.mean()),
        "ci95": [float(lo), float(hi)],
        "frac_delta_positive": float(np.mean(diffs > 0)),
        "includes_zero": bool(lo <= 0.0 <= hi),
    }


def _self_check() -> None:
    dom = np.array([[10, 0, 0], [10, 0, 0]], np.float64)
    win = np.array([[5, 5, 0], [5, 5, 0]], np.float64)
    out = paired_delta_ci(dom, win, n=20, seed=0)
    if abs(out["point_delta"] - 0.5) > 1e-9:
        raise RuntimeError(f"bootstrap self-check {out}")


def load_model():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    model = models.build("unet", 16)
    model.load_state_dict(torch.load("artifacts/eval/s3_dom_unet.pt", map_location="cpu", weights_only=True))
    model.eval()
    return model


def blend_file(model, predict, ver: str, split: str, path: Path):
    stem = path.stem
    win_path = Path("artifacts/pred") / ver / "ens_u2ft_tcn_lgbm" / split / f"{stem}.npz"
    with np.load(path) as z:
        X = np.asarray(z["X"], np.float32)
        ev = np.asarray(z["ev"])
        valid = np.asarray(z["valid"], bool)
        nu = np.asarray(z["nu"])
    if not win_path.is_file():
        raise FileNotFoundError(win_path)
    prob = predict(model, X)
    s_u = (1.0 - prob[0]).astype(np.float32)
    h_u = (prob[1] - prob[2]).astype(np.float32)
    with np.load(win_path) as w:
        s_w = np.asarray(w["s"], np.float32)
        h_w = np.asarray(w["h"], np.float32)
    if s_w.shape != s_u.shape:
        raise RuntimeError(f"{stem} shape unet {s_u.shape} winner {s_w.shape}")
    s = (0.5 * (s_w + s_u)).astype(np.float32)
    h = (0.5 * (h_w + h_u)).astype(np.float32)
    return stem, ev, valid, nu, s, h, s_w, h_w


def _add(acc, pred, truth, valid):
    acc += np.asarray(M.confusion(pred, truth, valid), np.int64)


def summarize(tp_fp_fn, far_n, far_d, hot, cold, thr) -> dict:
    row = dict(threshold=float(thr), far=float(far_n / max(far_d, 1)), **M.prf(*tp_fp_fn),
               hot=M.prf(*hot), cold=M.prf(*cold), n_bg=int(far_d))
    return row


def eval_stored(stems, shapes, thr_dom, thr_win) -> dict:
    per_d, per_w = [], []
    names = []
    acc = {k: np.zeros(3, np.int64) for k in ("d", "w", "dh", "wh", "dc", "wc")}
    far = {k: 0 for k in ("dn", "dd", "wn", "wd")}
    for stem in stems:
        shape = tuple(shapes[stem])
        s = np.memmap(TMP / f"{stem}_s.mmap", dtype=np.float32, mode="r", shape=shape)
        h = np.memmap(TMP / f"{stem}_h.mmap", dtype=np.float16, mode="r", shape=shape)
        sw = np.memmap(TMP / f"{stem}_ws.mmap", dtype=np.float32, mode="r", shape=shape)
        hw = np.memmap(TMP / f"{stem}_wh.mmap", dtype=np.float16, mode="r", shape=shape)
        ev = np.load(TMP / f"{stem}_ev.npy")
        valid = np.load(TMP / f"{stem}_valid.npy")
        nu = np.load(TMP / f"{stem}_nu.npy")
        bg = valid & (ev == 0) & (nu == 0)
        truth = ev > 0
        hot_t = (ev >= 1) & (ev <= 4)
        cold_t = ev >= 5
        pred_d = (np.asarray(s) >= thr_dom) & valid
        pred_w = (np.asarray(sw) >= thr_win) & valid
        hv = np.asarray(h, np.float32)
        wv = np.asarray(hw, np.float32)
        cd = np.asarray(M.confusion(pred_d, truth, valid), np.int64)
        cw = np.asarray(M.confusion(pred_w, truth, valid), np.int64)
        per_d.append(cd)
        per_w.append(cw)
        names.append(stem)
        _add(acc["d"], pred_d, truth, valid)
        _add(acc["w"], pred_w, truth, valid)
        _add(acc["dh"], pred_d & (hv > 0), hot_t, valid)
        _add(acc["wh"], pred_w & (wv > 0), hot_t, valid)
        _add(acc["dc"], pred_d & (hv <= 0), cold_t, valid)
        _add(acc["wc"], pred_w & (wv <= 0), cold_t, valid)
        far["dn"] += int((pred_d & bg).sum())
        far["wn"] += int((pred_w & bg).sum())
        far["dd"] += int(bg.sum())
        far["wd"] += int(bg.sum())
        del s, h, sw, hw
    return dict(names=names, per_d=np.stack(per_d), per_w=np.stack(per_w), acc=acc, far=far)


def store_val(model, predict, files) -> dict:
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True)
    shapes = {}
    bg_parts = []
    t0 = time.time()
    for i, p in enumerate(files, 1):
        stem, ev, valid, nu, s, h, sw, hw = blend_file(model, predict, "v8", "val", p)
        shapes[stem] = list(s.shape)
        bg = valid & (ev == 0) & (nu == 0)
        bg_parts.append(np.asarray(s[bg], np.float32))
        sm = np.memmap(TMP / f"{stem}_s.mmap", dtype=np.float32, mode="w+", shape=s.shape)
        hm = np.memmap(TMP / f"{stem}_h.mmap", dtype=np.float16, mode="w+", shape=h.shape)
        wm = np.memmap(TMP / f"{stem}_ws.mmap", dtype=np.float32, mode="w+", shape=sw.shape)
        wh = np.memmap(TMP / f"{stem}_wh.mmap", dtype=np.float16, mode="w+", shape=hw.shape)
        sm[:] = s
        hm[:] = h
        wm[:] = sw
        wh[:] = hw
        sm.flush(); hm.flush(); wm.flush(); wh.flush()
        np.save(TMP / f"{stem}_ev.npy", np.asarray(ev))
        np.save(TMP / f"{stem}_valid.npy", np.asarray(valid))
        np.save(TMP / f"{stem}_nu.npy", np.asarray(nu))
        del sm, hm, wm, wh, s, h, sw, hw
        if i == 1 or i % 10 == 0 or i == len(files):
            log(f"v8 val {i}/{len(files)} {time.time() - t0:.0f}s")
    bg = np.concatenate(bg_parts)
    thr = float(M.far_threshold(bg, 1e-3))
    log(f"порог ens_dom {thr:.6f} n_bg {bg.size}")
    del bg, bg_parts
    return shapes, thr


def eval_stream(model, predict, ver, split, thr_dom, thr_win) -> dict:
    files = data.list_cache(f"artifacts/cache/{ver}", split)
    acc = {k: np.zeros(3, np.int64) for k in ("d", "w", "dh", "wh", "dc", "wc")}
    far = {k: 0 for k in ("dn", "dd", "wn", "wd")}
    t0 = time.time()
    for i, p in enumerate(files, 1):
        stem, ev, valid, nu, s, h, sw, hw = blend_file(model, predict, ver, split, p)
        bg = valid & (ev == 0) & (nu == 0)
        truth = ev > 0
        hot_t = (ev >= 1) & (ev <= 4)
        cold_t = ev >= 5
        pred_d = (s >= thr_dom) & valid
        pred_w = (sw >= thr_win) & valid
        _add(acc["d"], pred_d, truth, valid)
        _add(acc["w"], pred_w, truth, valid)
        _add(acc["dh"], pred_d & (h > 0), hot_t, valid)
        _add(acc["wh"], pred_w & (hw > 0), hot_t, valid)
        _add(acc["dc"], pred_d & (h <= 0), cold_t, valid)
        _add(acc["wc"], pred_w & (hw <= 0), cold_t, valid)
        far["dn"] += int((pred_d & bg).sum())
        far["wn"] += int((pred_w & bg).sum())
        nbg = int(bg.sum())
        far["dd"] += nbg
        far["wd"] += nbg
        del s, h, sw, hw
        if i == 1 or i % 10 == 0 or i == len(files):
            log(f"{ver} {split} {i}/{len(files)} {time.time() - t0:.0f}s")
    return dict(acc=acc, far=far, n_files=len(files))


def write_release(thr: float) -> None:
    REL.mkdir(parents=True, exist_ok=True)
    src = Path("artifacts/eval/s3_dom_unet.pt")
    dst = REL / "unet_dom.pt"
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        shutil.copyfile(src, dst)
    spec = {
        "method": "ens_dom",
        "definition": "score = 0.5 * (ens_u2ft_tcn_lgbm from models_release + unet_dom)",
        "unet_file": "unet_dom.pt",
        "width": 16,
        "arch": "unet",
        "winner_release": "models_release",
        "winner_method": "ens_u2ft_tcn_lgbm",
        "winner_threshold_far_1e-3": WINNER_THR,
        "thresholds": {"cnn_unet": {"far_1e-3": thr}},
        "threshold_source": "v8 val background valid & ev==0 & nu==0; test and real files do not move it",
        "init": "artifacts/stage1/unet_v810ft.pt",
        "train": "artifacts/eval/s3_dom_train.json",
        "real_policy": "MGTS csv evaluation only; no fit and no card calibration",
    }
    (REL / "stage1_thresholds.json").write_text(json.dumps(plain(spec), indent=1, ensure_ascii=False))
    log(f"релиз {REL}")


def pack(block, key_d, key_w, thr_dom, thr_win) -> dict:
    acc, far = block["acc"], block["far"]
    return {
        "ens_dom": summarize(acc[key_d], far["dn"], far["dd"], acc["dh"], acc["dc"], thr_dom),
        "ens_u2ft_tcn_lgbm": summarize(acc[key_w], far["wn"], far["wd"], acc["wh"], acc["wc"], thr_win),
    }


def main() -> None:
    _self_check()
    ev = _eval()
    model = load_model()
    err = ev.check_predict(model)
    log(f"сверка predict max|Δ| {err:.3e}")
    files = data.list_cache("artifacts/cache/v8", "val")
    try:
        shapes, thr = store_val(model, ev.predict_batched, files)
        stems = [p.stem for p in files]
        val = eval_stored(stems, shapes, thr, WINNER_THR)
        ci = paired_delta_ci(val["per_d"], val["per_w"], n=1000, seed=0)
        log(f"val ens_dom {ci['point_iou_dom']:.4f} winner {ci['point_iou_winner']:.4f} "
            f"delta {ci['point_delta']:.4f} CI [{ci['ci95'][0]:.4f}, {ci['ci95'][1]:.4f}]")
        shutil.rmtree(TMP, ignore_errors=True)
        write_release(thr)
        v10 = eval_stream(model, ev.predict_batched, "v10", "test", thr, WINNER_THR)
        v11 = eval_stream(model, ev.predict_batched, "v11", "test", thr, WINNER_THR)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)

    report = {
        "name": "DOM2",
        "ens_dom": "0.5 * (ens_u2ft_tcn_lgbm, unet_dom)",
        "threshold_rule": "FAR 1e-3 quantile on v8 val background only; same number on v10 test and v11 test",
        "predict_max_abs_err": err,
        "winner_threshold_far_1e-3": WINNER_THR,
        "published_winner_val_iou": WINNER_VAL,
        "published_winner_v10_test_iou": WINNER_V10,
        "published_winner_v11_test_iou": WINNER_V11,
        "v8_val": pack(val, "d", "w", thr, WINNER_THR),
        "bootstrap_val_iou_delta": ci,
        "per_seq_counts": {
            "order": val["names"],
            "ens_dom_tp_fp_fn": val["per_d"].astype(int).tolist(),
            "winner_tp_fp_fn": val["per_w"].astype(int).tolist(),
        },
        "v10_test": pack(v10, "d", "w", thr, WINNER_THR),
        "v11_test": pack(v11, "d", "w", thr, WINNER_THR),
        "release": "models_release_dom",
        "real_used_for_threshold": False,
    }
    out = Path("artifacts/eval/s3_dom2.json")
    out.write_text(json.dumps(plain(report), indent=1, ensure_ascii=False))
    log(f"v10 ens_dom {report['v10_test']['ens_dom']['iou']:.4f} winner {report['v10_test']['ens_u2ft_tcn_lgbm']['iou']:.4f}")
    log(f"v11 ens_dom {report['v11_test']['ens_dom']['iou']:.4f} winner {report['v11_test']['ens_u2ft_tcn_lgbm']['iou']:.4f}")
    log(f"записан {out}")


if __name__ == "__main__":
    main()
