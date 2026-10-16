"""Проверка RandConv: v10/v11 тем же порогом v8, бутстреп val, утечка отбора.

Порог смеси и победителя заново считается только по фону v8 val и на v10/v11 не двигается.
Alpha 0.5 зафиксирована заранее. Сетка alpha из вложенной проверки здесь не используется.
Промоушн: v10 test и v11 test смеси не хуже победителя при этих порогах.
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

WINNER_VAL = 0.38671235821914546
WINNER_TEST = 0.4052762065230473
WINNER_V10 = 0.4038977883892501
WINNER_V11 = 0.40390028295846503
TMP = Path("/tmp/s3_randconv2_maps")


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


def paired_delta_ci(ens: np.ndarray, win: np.ndarray, n: int = 1000, seed: int = 0) -> dict:
    ens = np.asarray(ens, np.float64)
    win = np.asarray(win, np.float64)

    def iou(block):
        s = block.sum(0)
        return float(s[0] / max(s.sum(), 1.0))

    point = iou(ens) - iou(win)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n, np.float64)
    for i in range(n):
        idx = rng.integers(0, len(ens), len(ens))
        diffs[i] = iou(ens[idx]) - iou(win[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "n_seq": int(len(ens)),
        "n_boot": int(n),
        "seed": int(seed),
        "point_iou_ens": iou(ens),
        "point_iou_winner": iou(win),
        "point_delta": float(point),
        "boot_mean_delta": float(diffs.mean()),
        "ci95": [float(lo), float(hi)],
        "frac_delta_positive": float(np.mean(diffs > 0)),
        "includes_zero": bool(lo <= 0.0 <= hi),
    }


def _predict():
    spec = importlib.util.spec_from_file_location("s3_dom_eval", ROOT / "scripts" / "s3_dom_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_model(path: str):
    model = models.build("unet", 16)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def counts_of(s, thr, ev, valid):
    return np.asarray(M.confusion((s >= thr) & valid, ev > 0, valid), np.int64)


def iou_block(block):
    s = np.asarray(block, np.float64).sum(0)
    return float(s[0] / max(s.sum(), 1.0))


def row_from(block, far_n, far_d, thr):
    return dict(threshold=float(thr), far=float(far_n / max(far_d, 1)), n_bg=int(far_d), **M.prf(*block.sum(0)))


def predict_split(model, predict, ver, split, files=None):
    files = files or data.list_cache(f"artifacts/cache/{ver}", split)
    t0 = time.time()
    out = []
    for i, p in enumerate(files, 1):
        with np.load(p) as z:
            X = np.asarray(z["X"], np.float32)
            ev = np.asarray(z["ev"])
            valid = np.asarray(z["valid"], bool)
            nu = np.asarray(z["nu"])
        prob = predict(model, X)
        s = (1.0 - prob[0]).astype(np.float32)
        win = Path(f"artifacts/pred/{ver}/ens_u2ft_tcn_lgbm/{split}") / p.name
        with np.load(win) as w:
            sw = np.asarray(w["s"], np.float32)
        if sw.shape != s.shape:
            raise SystemExit(f"{ver}/{split}/{p.stem} {s.shape} != {sw.shape}")
        blend = (0.5 * (sw + s)).astype(np.float32)
        out.append((p.stem, ev, valid, nu, s, sw, blend))
        del X, prob
        if i == 1 or i % 15 == 0 or i == len(files):
            log(f"{ver} {split} {i}/{len(files)} {time.time() - t0:.0f}s")
    return out


def bg_concat(rows):
    parts = {k: [] for k in ("rc", "ens", "blend")}
    for _stem, ev, valid, nu, s, sw, blend in rows:
        mask = valid & (ev == 0) & (nu == 0)
        parts["rc"].append(s[mask])
        parts["ens"].append(sw[mask])
        parts["blend"].append(blend[mask])
    return {k: np.concatenate(v) for k, v in parts.items()}


def score_rows(rows, thr):
    per = {k: [] for k in ("rc", "ens", "blend")}
    names = []
    far = {k: [0, 0] for k in per}
    for stem, ev, valid, nu, s, sw, blend in rows:
        names.append(stem)
        fields = {"rc": s, "ens": sw, "blend": blend}
        bg = valid & (ev == 0) & (nu == 0)
        for k, arr in fields.items():
            per[k].append(counts_of(arr, thr[k], ev, valid))
            pred = (arr >= thr[k]) & valid
            far[k][0] += int((pred & bg).sum())
            far[k][1] += int(bg.sum())
    return names, {k: np.stack(v) for k, v in per.items()}, far


def main() -> None:
    torch.set_num_threads(2)
    ckpt = os.environ.get("RC2_CKPT", "artifacts/eval/s3_randconv_unet.pt")
    tag = os.environ.get("RC2_TAG", "s0")
    out = Path(os.environ.get("RC2_OUT", f"artifacts/eval/s3_randconv2_{tag}.json"))
    ev = _predict()
    model = load_model(ckpt)
    err = ev.check_predict(model)
    log(f"{tag} сверка {err:.3e} ckpt {ckpt}")
    val_files = data.list_cache("artifacts/cache/v8", "val")
    proxy_stems = [p.stem for p in val_files[:8]]
    try:
        val_rows = predict_split(model, ev.predict_batched, "v8", "val", val_files)
        bg = bg_concat(val_rows)
        thr = {k: float(M.far_threshold(v, 1e-3)) for k, v in bg.items()}
        for k, v in bg.items():
            log(f"порог {k} {thr[k]:.6f} n {v.size}")
        del bg
        names, per, far = score_rows(val_rows, thr)
        del val_rows
        ci = paired_delta_ci(per["blend"], per["ens"], n=1000, seed=0)
        log(f"val blend {ci['point_iou_ens']:.4f} winner {ci['point_iou_winner']:.4f} "
            f"delta {ci['point_delta']:.4f} CI [{ci['ci95'][0]:.4f}, {ci['ci95'][1]:.4f}]")
        proxy_mask = np.array([n in set(proxy_stems) for n in names])
        if int(proxy_mask.sum()) != 8:
            raise SystemExit("прокси-8 не совпали со списком val")
        hold = {
            "proxy8": {
                "ens_randconv": iou_block(per["blend"][proxy_mask]),
                "winner": iou_block(per["ens"][proxy_mask]),
            },
            "rest37": {
                "ens_randconv": iou_block(per["blend"][~proxy_mask]),
                "winner": iou_block(per["ens"][~proxy_mask]),
            },
        }
        for part, d in hold.items():
            d["delta"] = float(d["ens_randconv"] - d["winner"])
            log(f"{part} delta {d['delta']:.4f}")
        splits = {}
        for ver in ("v10", "v11"):
            rows = predict_split(model, ev.predict_batched, ver, "test")
            _names, per_s, far_s = score_rows(rows, thr)
            del rows
            splits[ver] = {
                "ens_randconv": row_from(per_s["blend"], far_s["blend"][0], far_s["blend"][1], thr["blend"]),
                "ens_u2ft_tcn_lgbm": row_from(per_s["ens"], far_s["ens"][0], far_s["ens"][1], thr["ens"]),
                "unet_randconv": row_from(per_s["rc"], far_s["rc"][0], far_s["rc"][1], thr["rc"]),
            }
            log(f"{ver} blend {splits[ver]['ens_randconv']['iou']:.4f} "
                f"winner {splits[ver]['ens_u2ft_tcn_lgbm']['iou']:.4f}")
    finally:
        shutil.rmtree(TMP, ignore_errors=True)

    val_report = {
        "ens_randconv": row_from(per["blend"], far["blend"][0], far["blend"][1], thr["blend"]),
        "ens_u2ft_tcn_lgbm": row_from(per["ens"], far["ens"][0], far["ens"][1], thr["ens"]),
        "unet_randconv": row_from(per["rc"], far["rc"][0], far["rc"][1], thr["rc"]),
    }
    v10_ok = splits["v10"]["ens_randconv"]["iou"] + 1e-6 >= splits["v10"]["ens_u2ft_tcn_lgbm"]["iou"]
    v11_ok = splits["v11"]["ens_randconv"]["iou"] + 1e-6 >= splits["v11"]["ens_u2ft_tcn_lgbm"]["iou"]
    report = {
        "name": "RANDCONV2",
        "tag": tag,
        "ckpt": ckpt,
        "ensemble": "0.5 * (ens_u2ft_tcn_lgbm + unet_randconv), weight fixed before this run",
        "threshold_source": "v8 val background valid & ev==0 & nu==0; frozen for v10 test and v11 test",
        "alpha_used": False,
        "leak_audit": {
            "reported_blend_weight": 0.5,
            "alpha_grid_not_used": True,
            "alpha_note": "Сетка 0..1 в s3_randconv_eval выбиралась только на train-фолде вложенной проверки и в эту цифру не входит.",
            "threshold_fit_on_events": False,
            "threshold_refit_on_v10_v11": False,
            "threshold_refit_on_v8_test": False,
            "epoch_selected_on_proxy8": proxy_stems,
            "proxy8_vs_rest37": hold,
        },
        "predict_max_abs_err": err,
        "published": {"val": WINNER_VAL, "v8_test": WINNER_TEST, "v10_test": WINNER_V10, "v11_test": WINNER_V11},
        "val": val_report,
        "bootstrap_val_iou_delta": ci,
        "per_seq": {"order": names, "blend_tp_fp_fn": per["blend"].astype(int).tolist(),
                    "winner_tp_fp_fn": per["ens"].astype(int).tolist()},
        "v10_test": splits["v10"],
        "v11_test": splits["v11"],
        "v10_not_worse": bool(v10_ok),
        "v11_not_worse": bool(v11_ok),
        "promote": bool(v10_ok and v11_ok),
        "real_stats_used": False,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plain(report), indent=1, ensure_ascii=False))
    log(f"promote {report['promote']} записан {out}")


if __name__ == "__main__":
    main()
