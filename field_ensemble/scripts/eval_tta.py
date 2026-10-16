"""TTA U-Net: среднее score исходной карты и отражений по t и по x.

Запускать только если оценка времени < 20 мин. Порог FAR=1e-3 по val.
Ансамбль не трогает: у него нет того же входа в эту проверку.
"""
from __future__ import annotations

import importlib.util
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
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dts import data, models

BUDGET_S = 20 * 60
CACHE = ROOT / "artifacts/eval/stage2b_tta_pred"
OUT = ROOT / "artifacts/eval/stage2b_tta_v8.json"


def load_eval():
    spec = importlib.util.spec_from_file_location("eval_postproc", ROOT / "scripts" / "eval_postproc.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def flipped_score(net, x, flip_t, flip_x):
    xin = x
    if flip_t or flip_x:
        axes = ([1] if flip_t else []) + ([2] if flip_x else [])
        xin = np.flip(x, axis=tuple(axes)).copy()
    pr = models.predict_full(net, xin)
    s = (1.0 - pr[0]).astype(np.float32)
    if flip_t:
        s = np.flip(s, 0)
    if flip_x:
        s = np.flip(s, 1)
    return np.ascontiguousarray(s, dtype=np.float32)


def main():
    torch.set_num_threads(2)
    ev = load_eval()
    net = models.build("unet", 16)
    net.load_state_dict(torch.load(ROOT / "artifacts/stage1/unet.pt", map_location="cpu"))
    net.eval()
    splits = {}
    for split in ("val", "test"):
        splits[split] = data.list_cache(ROOT / "artifacts/cache/v8", split)
    # одна карта задаёт скорость и проверяет, что сохранённый score — тот же predict_full
    sample = splits["val"][0]
    x = np.load(sample)["X"].astype(np.float32)
    saved = np.load(ROOT / "artifacts/pred/v8/unet/val" / sample.name)["s"].astype(np.float32)
    t0 = time.time()
    with torch.no_grad():
        s_net = flipped_score(net, x, False, False)
    dt = time.time() - t0
    diff = float(np.max(np.abs(s_net - saved)))
    recompute_id = diff > 1e-3
    per = 3 if recompute_id else 2
    est_val = len(splits["val"]) * per * dt
    est_test = len(splits["test"]) * per * dt
    est = est_val + est_test
    run_splits = ("val", "test") if est <= BUDGET_S else (("val",) if est_val <= BUDGET_S else ())
    print(
        f"tta probe sec={dt:.2f} identity_max_abs={diff:.3g} est_sec={est:.0f} "
        f"est_val={est_val:.0f} recompute_id={recompute_id} splits={run_splits or 'none'}",
        flush=True,
    )
    if not run_splits:
        json.dump({"skipped": True, "reason": "over 20 min", "est_sec": est, "est_val_sec": est_val, "probe_sec": dt, "identity_max_abs": diff}, open(OUT, "w"), indent=1)
        print("TTA_SKIP", flush=True)
        return
    for split in run_splits:
        paths = splits[split]
        for p in paths:
            dst = CACHE / split / p.name
            if dst.exists():
                continue
            x = np.load(p)["X"].astype(np.float32)
            saved = np.load(ROOT / "artifacts/pred/v8/unet" / split / p.name)["s"].astype(np.float32)
            with torch.no_grad():
                parts = []
                if recompute_id:
                    parts.append(flipped_score(net, x, False, False))
                else:
                    parts.append(saved)
                parts.append(flipped_score(net, x, True, False))
                parts.append(flipped_score(net, x, False, True))
            s = np.mean(parts, axis=0).astype(np.float32)
            dst.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dst, s=s)
            print("tta", split, p.stem, flush=True)
    labels = __import__("pandas").read_csv(ROOT / "data/synth_meta/v8/labels.csv")
    report = {
        "method": "unet",
        "augment": "mean(id, flip_t, flip_x)",
        "probe_sec": dt,
        "identity_max_abs": diff,
        "recompute_id": recompute_id,
        "est_sec": est,
        "splits": list(run_splits),
    }
    if "test" not in run_splits:
        report["test_skipped"] = "val+test estimate exceeds 20 min; test flips were not run"
    for split in run_splits:
        maps = ev.load_split(ROOT / "artifacts/cache/v8", CACHE, split, 0)
        boxes = ev.event_table(labels, maps)
        if split == "val":
            bg = ev.concat_bg(maps, "identity")
            from dts import metrics as M
            thr = M.far_threshold(bg, 1e-3)
            del bg
            report["threshold"] = float(thr)
        rows = ev.evaluate(maps, boxes, "identity", report["threshold"])
        report[split] = rows
        print(f"tta {split} far={rows['far']:.6f} iou={rows['iou']:.4f} f1={rows['f1']:.4f} ev={rows['event_recall']:.4f} weak={rows['weak_recall']:.4f}", flush=True)
        del maps
    json.dump(report, open(OUT, "w"), indent=1)
    print("TTA_DONE", flush=True)


if __name__ == "__main__":
    main()
