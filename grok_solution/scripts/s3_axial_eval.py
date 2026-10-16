"""IoU@FAR AxialUNet на синтетике v8. Пороги — только по фону v8 val.

Реальные CSV не читаются и не калибруют порог. Карточки оператора не используются.
Счета пишутся в JSON, карты score на диск не складываются.
"""
import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dts import data, events as EV
from dts import metrics as M
from dts import models
from dts.axial import AxialUNet, param_counts, proj_l1

NBIN = 100_000
OPS = ("far_1e-4", "far_1e-3", "far_1e-2")
FAR_OF = {"far_1e-4": 1e-4, "far_1e-3": 1e-3, "far_1e-2": 1e-2}


def thr_from_hist(hist: np.ndarray, far: float) -> float:
    """Порядок как в metrics.far_threshold, бин шириной 1e-5."""
    n = int(hist.sum())
    if n <= 0:
        return 1.0
    k = int(np.ceil((1.0 - far) * n)) - 1
    k = int(np.clip(k, 0, n - 1))
    cdf = np.cumsum(hist)
    i = int(np.searchsorted(cdf, k + 1, side="left"))
    i = min(max(i, 0), len(hist) - 1)
    return float((i + 0.5) / len(hist)) + 1e-12


def add_hist(hist: np.ndarray, scores: np.ndarray) -> None:
    if scores.size == 0:
        return
    idx = np.clip((np.clip(scores, 0.0, 1.0) * NBIN).astype(np.int32), 0, NBIN - 1)
    hist += np.bincount(idx, minlength=NBIN)


@torch.inference_mode()
def predict_sh(model, X):
    pr = models.predict_full(model, np.asarray(X, np.float32))
    return (1.0 - pr[0]).astype(np.float32), (pr[1] - pr[2]).astype(np.float32)


def empty_acc():
    return dict(tot=np.zeros(3, np.int64), hot=np.zeros(3, np.int64), cold=np.zeros(3, np.int64),
                tot_ni=np.zeros(3, np.int64), per_seq={}, far_n=0, far_d=0, nu_n=0, nu_d=0, ev_rows=[],
                zones_n=0, zones_false=0, kmh=0.0, tpl= {})


def update(acc, sid, s, h, d, thr, labels_seq, template, with_events):
    valid = d["valid"]
    ev = d["ev"]
    nu = d["nu"]
    pred = (s >= thr) & valid
    truth = ev > 0
    c = M.confusion(pred, truth, valid)
    acc["tot"] += c
    acc["per_seq"][sid] = c
    acc["tot_ni"] += M.confusion(pred, truth, valid & ~((nu > 0) & ~truth))
    acc["hot"] += M.confusion(pred & (h > 0), (ev >= 1) & (ev <= 4), valid)
    acc["cold"] += M.confusion(pred & (h <= 0), ev >= 5, valid)
    b = valid & (ev == 0) & (nu == 0)
    acc["far_n"] += int((pred & b).sum())
    acc["far_d"] += int(b.sum())
    nm = valid & (nu > 0) & ~truth
    acc["nu_n"] += int((pred & nm).sum())
    acc["nu_d"] += int(nm.sum())
    acc["tpl"].setdefault(template, np.zeros(3, np.int64))
    acc["tpl"][template] += c
    if with_events and labels_seq is not None:
        for r in M.label_boxes(labels_seq, sid).itertuples():
            if r.kind != "event":
                continue
            px = M.event_pixels(r, d["x"], d["t"], ev)
            if px.sum() == 0:
                continue
            frac = float(pred[px].mean())
            acc["ev_rows"].append(dict(
                label=r.label, amp=float(r.amp_obs_C), amp_bin=M.amp_bin(float(r.amp_obs_C)),
                template=template, frac=frac, hit=frac >= 0.3,
                tp=int((pred & px).sum()), fn=int((~pred & px).sum()),
            ))
        zs = EV.zones_from_prob(s, h, valid, d["x"], d["t"], thr)
        for z in zs:
            sl = (slice(z["t0"], z["t1"]), slice(z["x0"], z["x1"]))
            comp = (s[sl] >= thr) & valid[sl]
            acc["zones_n"] += 1
            acc["zones_false"] += int((ev[sl][comp] > 0).mean() < 0.1) if comp.any() else 1
        acc["kmh"] += (d["x"][-1] - d["x"][0]) / 1000.0 * (d["t"][-1] - d["t"][0]) / 3600.0


def pack(acc, thr, far_ref):
    def agg(rows):
        if not rows:
            return dict(n=0, recall=None, pixel_recall=None)
        tp = sum(r["tp"] for r in rows)
        fn = sum(r["fn"] for r in rows)
        hit = sum(r["hit"] for r in rows)
        return dict(n=len(rows), recall=hit / len(rows), pixel_recall=tp / max(tp + fn, 1))

    E = acc["ev_rows"]
    res = dict(
        threshold=float(thr),
        far=float(acc["far_n"] / max(acc["far_d"], 1)),
        far_ratio=float(acc["far_n"] / max(acc["far_d"], 1) / max(far_ref, 1e-12)),
        nuisance_alarm_frac=float(acc["nu_n"] / max(acc["nu_d"], 1)),
        **M.prf(*acc["tot"]),
        hot=M.prf(*acc["hot"]),
        cold=M.prf(*acc["cold"]),
        nuisance_ignored=M.prf(*acc["tot_ni"]),
        iou_ci95=M.bootstrap_ci(acc["per_seq"], sorted(acc["per_seq"])) if acc["per_seq"] else None,
        iou_by_template={k: M.prf(*v)["iou"] for k, v in acc["tpl"].items()},
    )
    if E:
        res["events_all"] = agg(E)
        res["weak_events"] = agg([r for r in E if r["amp"] <= 2])
        res["by_amp"] = {k: agg([r for r in E if r["amp_bin"] == k]) for k in sorted({r["amp_bin"] for r in E})}
        res["by_label"] = {k: agg([r for r in E if r["label"] == k]) for k in sorted({r["label"] for r in E})}
        res["by_template"] = {k: agg([r for r in E if r["template"] == k]) for k in sorted({r["template"] for r in E})}
    res["zones"] = dict(
        n_zones=acc["zones_n"], n_false=acc["zones_false"],
        false_zones_per_km_h=acc["zones_false"] / max(acc["kmh"], 1e-9),
        false_frac=acc["zones_false"] / max(acc["zones_n"], 1),
    )
    return res


def load_np(path, with_x=True):
    z = np.load(path)
    d = dict(ev=z["ev"], nu=z["nu"], valid=z["valid"], x=z["x"], t=z["t"])
    if with_x:
        d["X"] = z["X"]
    return d


def _dump(path, arr):
    mm = np.memmap(path, dtype=arr.dtype, mode="w+", shape=arr.shape)
    mm[:] = arr
    mm.flush()
    del mm


def _load(path, shape, dtype):
    return np.array(np.memmap(path, dtype=dtype, mode="r", shape=shape))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="artifacts/cache/v8")
    ap.add_argument("--ckpt", default="artifacts/eval/s3_axial_model.pt")
    ap.add_argument("--labels", default="data/synth_meta/v8/labels.csv")
    ap.add_argument("--sequences", default="data/synth_meta/v8/sequences.csv")
    ap.add_argument("--out", default="artifacts/eval/s3_axial_v8.json")
    ap.add_argument("--winner", default="artifacts/eval/r7_ens_u2ft_tcn_lgbm.json")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--event-log", default="/workspace/grok_logs/_events.log")
    a = ap.parse_args()

    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.set_num_threads(a.threads)
    model = AxialUNet()
    state = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.eval()
    labels = pd.read_csv(a.labels)
    by_seq = {sid: g for sid, g in labels.groupby("seq_id")}
    seqs = pd.read_csv(a.sequences).set_index("seq_id")
    t0 = time.time()

    # Один проход val: гистограмма фона и карты s/h во временный memmap (удаляется в finally).
    # Второй проход по картам считает IoU уже известным порогом, без повторной сети.
    hist = np.zeros(NBIN, np.int64)
    far_ref_n = far_ref_d = 0
    val_paths = data.list_cache(a.cache, "val")
    cache_dir = Path("/tmp/s3_axial_valcache")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True)
    shapes = {}
    print("val infer", len(val_paths), flush=True)
    try:
        for i, p in enumerate(val_paths):
            z = np.load(p)
            s, h = predict_sh(model, z["X"])
            valid = z["valid"]
            ev = z["ev"]
            nu = z["nu"]
            bg = valid & (ev == 0) & (nu == 0)
            add_hist(hist, s[bg])
            if "base_z" in z.files:
                far_ref_n += int((np.abs(z["base_z"][bg]) >= 3.5).sum())
                far_ref_d += int(bg.sum())
            _dump(cache_dir / f"{p.stem}_s.f32", np.ascontiguousarray(s, np.float32))
            _dump(cache_dir / f"{p.stem}_h.f32", np.ascontiguousarray(h, np.float32))
            shapes[p.stem] = list(s.shape)
            del s, h
            if i % 5 == 0:
                print(" val", i, p.stem, round(time.time() - t0, 1), flush=True)
        far_ref = float(far_ref_n / max(far_ref_d, 1))
        thresholds = {op: thr_from_hist(hist, FAR_OF[op]) for op in OPS}
        thresholds["far_ref"] = thr_from_hist(hist, far_ref) if far_ref > 0 else 1.0
        print("thresholds", thresholds, "far_ref", far_ref, "sec", round(time.time() - t0, 1), flush=True)

        def score_split(paths, cached):
            accs = {op: empty_acc() for op in OPS}
            print("metrics", "cached" if cached else "live", len(paths), flush=True)
            for i, p in enumerate(paths):
                d = load_np(p, with_x=not cached)
                if cached:
                    shape = tuple(shapes[p.stem])
                    s = _load(cache_dir / f"{p.stem}_s.f32", shape, np.float32)
                    h = _load(cache_dir / f"{p.stem}_h.f32", shape, np.float32)
                else:
                    s, h = predict_sh(model, d["X"])
                    del d["X"]
                template = seqs.loc[p.stem, "template"] if p.stem in seqs.index else "?"
                if not np.ndim(template) == 0:
                    template = str(template.iloc[0] if hasattr(template, "iloc") else template)
                lab = by_seq.get(p.stem)
                for op, thr in thresholds.items():
                    if op not in accs:
                        continue
                    update(accs[op], p.stem, s, h, d, thr, lab, str(template), with_events=(op == "far_1e-3"))
                if i % 5 == 0:
                    print(" ", p.stem, round(time.time() - t0, 1), flush=True)
            return {op: pack(accs[op], thresholds[op], far_ref) for op in accs}

        val_report = score_split(val_paths, True)
        test_report = score_split(data.list_cache(a.cache, "test"), False)
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)

    winner = {}
    wpath = Path(a.winner)
    if wpath.exists():
        ref = json.load(open(wpath))
        winner = dict(
            name="ens_u2ft_tcn_lgbm",
            val_iou=ref["val"]["ens_u2ft_tcn_lgbm@far_1e-3"]["iou"],
            test_iou=ref["test"]["ens_u2ft_tcn_lgbm@far_1e-3"]["iou"],
            source=str(wpath),
        )
    our_val = float(val_report["far_1e-3"]["iou"])
    our_test = float(test_report["far_1e-3"]["iou"])
    beats = bool(winner) and our_val > float(winner["val_iou"])
    print(f"VAL {our_val:.4f} TEST {our_test:.4f} beats {beats}", flush=True)

    report = dict(
        name="axial",
        version=Path(a.cache).name,
        ckpt=a.ckpt,
        params=param_counts(model),
        proj_l1=proj_l1(model),
        bin_width=1.0 / NBIN,
        far_ref=far_ref,
        thresholds={"axial": thresholds},
        val={f"axial@{op}": val_report[op] for op in val_report},
        test={f"axial@{op}": test_report[op] for op in test_report},
        winner=winner,
        beats_winner_val=beats,
        seconds=round(time.time() - t0, 1),
        note="Порог по фону v8 val. Реальные файлы и карточки не использовались.",
    )
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

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(plain(report), open(out, "w"), indent=1, ensure_ascii=False)
    print("VAL", our_val, "TEST", our_test, "beats", beats, "wrote", out, flush=True)

    if a.event_log:
        line = (
            f"AXIAL_DONE val_iou={our_val:.4f} test_iou={our_test:.4f} "
            f"thr={thresholds['far_1e-3']:.4f} val_far={val_report['far_1e-3']['far']:.6f} "
            f"test_far={test_report['far_1e-3']['far']:.6f} "
            f"hot_val={val_report['far_1e-3']['hot']['iou']:.4f} "
            f"cold_val={val_report['far_1e-3']['cold']['iou']:.4f} "
            f"hot_test={test_report['far_1e-3']['hot']['iou']:.4f} "
            f"cold_test={test_report['far_1e-3']['cold']['iou']:.4f} "
            f"beats_winner={str(beats).lower()} "
            f"winner_val={winner.get('val_iou', float('nan')):.4f} "
            f"winner_test={winner.get('test_iou', float('nan')):.4f}"
        )
        elog = Path(a.event_log)
        elog.parent.mkdir(parents=True, exist_ok=True)
        with elog.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            if beats:
                f.write(f"BEATS_WINNER axial val {our_val:.4f} test {our_test:.4f}\n")
        print(line, flush=True)


if __name__ == "__main__":
    main()
