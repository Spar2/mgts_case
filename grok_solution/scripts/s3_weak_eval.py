"""Оценка s3_weak на synth v8: порог по фону val при FAR 1e-3, by_amp ≤2 °C.

Порог не подбирается по test и не по реальным файлам. Реальные трассы не читаются:
меток событий на них нет, by_amp по ним не считается.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_weak_lib as W
from dts import data, models

ROOT = Path(__file__).resolve().parents[1]
WINNER_VAL = 0.38671235821914546
WINNER_TEST = 0.4052762065230473


def blend_maps(s_u, h_u, s_e, h_e):
    use_u = s_u >= s_e
    h = np.where(use_u, h_u, h_e).astype(np.float32)
    s_or = (1.0 - (1.0 - s_u) * (1.0 - s_e)).astype(np.float32)
    s_max = np.maximum(s_u, s_e).astype(np.float32)
    return {
        "s3_weak": (s_u, h_u),
        "s3_weak_or": (s_or, h),
        "s3_weak_max": (s_max, h),
        "ens_u2ft_tcn_lgbm": (s_e, h_e),
    }


def pass_split(split, cache, labels, model, ens_dir, thr, collect_tail, limit=None):
    names = ["s3_weak", "s3_weak_or", "s3_weak_max", "ens_u2ft_tcn_lgbm"] if model is not None else ["ens_u2ft_tcn_lgbm"]
    tails = {k: W.ScoreTail() for k in names} if collect_tail else None
    acc = {k: W.new_acc() for k in names} if thr is not None else None
    files = data.list_cache(cache, split)
    if limit:
        files = files[:limit]
    t0 = time.time()
    for i, p in enumerate(files):
        z = np.load(p)
        ev, valid, nu = z["ev"], z["valid"], z["nu"]
        x, t = z["x"], z["t"]
        ens_p = Path(ens_dir) / split / p.name
        ez = np.load(ens_p)
        s_e = np.asarray(ez["s"], np.float32)
        h_e = np.asarray(ez["h"], np.float32)
        if model is not None:
            X = np.asarray(z["X"], np.float32)
            with torch.inference_mode():
                s_u, h_u = W.predict_sh(model, X)
            del X
            maps = blend_maps(s_u, h_u, s_e, h_e)
        else:
            maps = {"ens_u2ft_tcn_lgbm": (s_e, h_e)}
        bg = W.bg_mask(ev, valid, nu)
        for k, (s, h) in maps.items():
            if tails is not None:
                tails[k].add(s[bg])
            if acc is not None:
                W.update_acc(acc[k], s, h, ev, valid, nu, x, t, p.stem, labels, thr[k])
        if (i + 1) % 5 == 0 or i + 1 == len(files):
            print(split, i + 1, "/", len(files), "sec", round(time.time() - t0), flush=True)
        del z, maps
    return tails, acc


def dumpable(o):
    if isinstance(o, dict):
        return {str(k): dumpable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [dumpable(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.floating, float)):
        return float(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="artifacts/s3_weak/unet_s3_weak.pt")
    ap.add_argument("--cache", default="artifacts/cache/v8")
    ap.add_argument("--labels", default="data/synth_meta/v8/labels.csv")
    ap.add_argument("--ens", default="artifacts/pred/v8/ens_u2ft_tcn_lgbm")
    ap.add_argument("--out", default="artifacts/eval/s3_weak_v8.json")
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--far", type=float, default=1e-3)
    ap.add_argument("--ens-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--splits", default="val,test")
    a = ap.parse_args()
    os.chdir(ROOT)
    torch.set_num_threads(a.threads)
    import pandas as pd
    labels = pd.read_csv(a.labels)
    model = None
    if not a.ens_only:
        model = models.build("unet", a.width)
        model.load_state_dict(torch.load(a.ckpt, map_location="cpu"))
        model.eval()
    splits = [s.strip() for s in a.splits.split(",") if s.strip()]
    limit = a.limit or None
    t0 = time.time()
    tails, _ = pass_split("val", a.cache, labels, model, a.ens, thr=None, collect_tail=True, limit=limit)
    thr = {k: float(tails[k].threshold(a.far)) for k in tails}
    src = {k: tails[k].last_source for k in tails}
    print("thresholds", thr, src, flush=True)
    report = dict(
        name="s3_weak",
        selection_rule="val IoU @ FAR 1e-3 on synth v8, threshold from v8 val background",
        far=a.far,
        ckpt=None if a.ens_only else a.ckpt,
        real_used_for_train_or_threshold=False,
        winner_ref=dict(name="ens_u2ft_tcn_lgbm", val_iou=WINNER_VAL, test_iou=WINNER_TEST),
        thresholds=thr, threshold_source=src,
        val={}, test={},
    )
    _, acc_val = pass_split("val", a.cache, labels, model, a.ens, thr=thr, collect_tail=False, limit=limit)
    for k, acc in acc_val.items():
        block = W.finalize(acc)
        block["threshold"] = thr[k]
        report["val"][k] = block
        weak = block["weak_events"]
        print("val", k, "iou", round(block["iou"], 4), "f1", round(block["f1"], 4),
              "far", round(block["far"], 6), "weak", weak, flush=True)
    if "test" in splits:
        _, acc_te = pass_split("test", a.cache, labels, model, a.ens, thr=thr, collect_tail=False, limit=limit)
        for k, acc in acc_te.items():
            block = W.finalize(acc)
            block["threshold"] = thr[k]
            report["test"][k] = block
            weak = block["weak_events"]
            print("test", k, "iou", round(block["iou"], 4), "weak", weak, flush=True)
    # выбор по val IoU; ничья — выше recall ≤2 °C
    cands = [k for k in report["val"] if k != "ens_u2ft_tcn_lgbm"]
    if cands:
        def key(k):
            b = report["val"][k]
            wr = b["weak_events"]["recall"] or 0.0
            return (b["iou"], wr)
        selected = max(cands, key=key)
    else:
        selected = "ens_u2ft_tcn_lgbm"
    report["selected"] = selected
    sel_val = report["val"][selected]["iou"]
    sel_test = report["test"].get(selected, {}).get("iou")
    report["beats_winner"] = bool(sel_val > WINNER_VAL)
    report["seconds"] = round(time.time() - t0)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dumpable(report), open(out, "w"), indent=1)
    print("wrote", out, "selected", selected, "beats", report["beats_winner"], "sec", report["seconds"], flush=True)


if __name__ == "__main__":
    main()
