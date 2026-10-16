"""Оценка второго сида GEN2. v8 val в обучение и в прокси не входил.

41 последовательность — v8 val без seq0006, seq0021, seq0047, seq0048
(прокси первого сида). Порог FAR 1e-3 только по их фону и заморожен
для v8 test, v10 test и v11 test.

mean2 = 0.5 * (seed2 + победитель)
mean3 = (seed1 + seed2 + победитель) / 3

Бутстреп — 1000 перестановок последовательностей. Вложенные k=2 и k=5 —
sequence_folds i%k, порог фолда по своему фону, micro, как scripts/s3_sel2_cv.py.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_k] = "2"

import gc
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)

from dts import data, metrics as M, models
from s3_gen2_train import predict_batched

WINNER = "ens_u2ft_tcn_lgbm"
WINNER_VAL = 0.38671235821914546
WINNER_THR = 0.8608087301264272
HOLD_DROP = ("seq0006", "seq0021", "seq0047", "seq0048")
BOOT_N = 1000
BOOT_SEED = 20261018
SEED1 = Path("artifacts/eval/s3_gen2_unet.pt")
SEED2 = Path("artifacts/eval/s3_gen2_unet_s2.pt")
OUT = Path("artifacts/eval/s3_gen2_seed2.json")
NAMES = ("mean2", "mean3", "seed2", WINNER)


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S", time.gmtime()) + " " + msg, flush=True)


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


def dump(obj) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(plain(obj), indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, OUT)


def iou_of(tp, fp, fn):
    return float(tp) / max(float(tp) + float(fp) + float(fn), 1.0)


def pack(rows):
    tp = int(sum(r[0] for r in rows))
    fp = int(sum(r[1] for r in rows))
    fn = int(sum(r[2] for r in rows))
    pr = M.prf(tp, fp, fn)
    return dict(
        tp=tp, fp=fp, fn=fn, iou=float(pr["iou"]),
        precision=float(pr["precision"]), recall=float(pr["recall"]),
        n_seq=len(rows),
    )


def blend_of(name, s1, s2, ws):
    if name == "mean2":
        return (0.5 * (np.asarray(s2, np.float32) + ws)).astype(np.float32)
    if name == "mean3":
        return ((np.asarray(s1, np.float32) + np.asarray(s2, np.float32) + ws) / 3.0).astype(np.float32)
    if name == "seed2":
        return np.asarray(s2, np.float32)
    return np.asarray(ws, np.float32)


def bootstrap(rows_a, rows_b, n=BOOT_N, seed=BOOT_SEED):
    a = np.asarray(rows_a, np.float64)
    b = np.asarray(rows_b, np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    sa = a[idx].sum(axis=1)
    sb = b[idx].sum(axis=1)
    da = sa[:, 0] / np.maximum(sa.sum(axis=1), 1.0)
    db = sb[:, 0] / np.maximum(sb.sum(axis=1), 1.0)
    delta = da - db
    lo, mid, hi = np.percentile(delta, [2.5, 50, 97.5])
    point = pack(rows_a)["iou"] - pack(rows_b)["iou"]
    return dict(
        n=int(n), seed=int(seed), unit="sequence",
        point=float(point), mean=float(delta.mean()),
        p2_5=float(lo), p50=float(mid), p97_5=float(hi),
        frac_positive=float(np.mean(delta > 0)),
        covers_zero=bool(lo <= 0.0 <= hi),
    )


def load_net(path):
    model = models.build("unet", 16)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def predict_split(m1, m2, ver, split, tmp: Path):
    files = data.list_cache(f"artifacts/cache/{ver}", split)
    ens = Path("artifacts/pred") / ver / WINNER / split
    t0 = time.time()
    for i, p in enumerate(files, 1):
        with np.load(p) as z:
            X = np.asarray(z["X"], np.float32)
            shape = z["ev"].shape
        s1 = (1.0 - predict_batched(m1, X)[0]).astype(np.float32)
        s2 = (1.0 - predict_batched(m2, X)[0]).astype(np.float32)
        del X
        with np.load(ens / p.name) as pred:
            if pred["s"].shape != shape:
                raise SystemExit(f"winner {ver} {split} {p.stem} {pred['s'].shape} != {shape}")
        np.save(tmp / f"{ver}_{split}_{p.stem}_s1.npy", s1)
        np.save(tmp / f"{ver}_{split}_{p.stem}_s2.npy", s2)
        del s1, s2
        if i == 1 or i % 15 == 0 or i == len(files):
            log(f"pred {ver} {split} {i}/{len(files)} {time.time() - t0:.0f}s")
    return files


def load_trio(tmp, ver, split, path):
    s1 = np.load(tmp / f"{ver}_{split}_{path.stem}_s1.npy")
    s2 = np.load(tmp / f"{ver}_{split}_{path.stem}_s2.npy")
    with np.load(path) as z:
        ev = np.asarray(z["ev"])
        valid = np.asarray(z["valid"], bool)
        nu = np.asarray(z["nu"])
    with np.load(Path("artifacts/pred") / ver / WINNER / split / path.name) as pred:
        ws = np.asarray(pred["s"], np.float32)
    return s1, s2, ws, ev, valid, nu


def clear_split(tmp, ver, split, files):
    for p in files:
        for tag in ("s1", "s2"):
            f = tmp / f"{ver}_{split}_{p.stem}_{tag}.npy"
            if f.exists():
                f.unlink()
    gc.collect()


def winner_full_val():
    files = data.list_cache("artifacts/cache/v8", "val")
    rows = []
    for p in files:
        with np.load(p) as z:
            ev = np.asarray(z["ev"])
            valid = np.asarray(z["valid"], bool)
        with np.load(Path("artifacts/pred/v8") / WINNER / "val" / p.name) as pred:
            ws = np.asarray(pred["s"], np.float32)
        rows.append(M.confusion((ws >= WINNER_THR) & valid, ev > 0, valid))
    iou = pack(rows)["iou"]
    if abs(iou - WINNER_VAL) > 0.005:
        raise SystemExit(f"победитель на 45 файлах {iou} != {WINNER_VAL}")
    log(f"winner full val {iou:.6f} match")
    return iou, files


def fit_thr(files, tmp, ver, split):
    thr = {}
    for name in NAMES:
        parts = []
        for p in files:
            s1, s2, ws, ev, valid, nu = load_trio(tmp, ver, split, p)
            score = blend_of(name, s1, s2, ws)
            bg = valid & (ev == 0) & (nu == 0)
            parts.append(np.asarray(score[bg], np.float32))
            del s1, s2, ws, score
        bg = np.concatenate(parts)
        thr[name] = float(M.far_threshold(bg, 1e-3))
        log(f"порог {name} {thr[name]:.6f} n_bg {bg.size}")
        del parts, bg
    return thr


def score_files(files, tmp, ver, split, thr):
    rows = {n: [] for n in NAMES}
    for p in files:
        s1, s2, ws, ev, valid, _nu = load_trio(tmp, ver, split, p)
        truth = ev > 0
        for name in NAMES:
            score = blend_of(name, s1, s2, ws)
            rows[name].append(M.confusion((score >= thr[name]) & valid, truth, valid))
            del score
        del s1, s2, ws
    summary = {n: pack(rows[n]) for n in NAMES}
    for n in NAMES:
        summary[n]["threshold"] = float(thr[n])
    return summary, rows


def nested_files(files, tmp, ver, split):
    import s3_sel2_cv as sel2
    import s3_sel_core as sel
    import s3_sel_eval as ev
    import pandas as pd

    labels = pd.read_csv(f"data/synth_meta/{ver}/labels.csv")
    seqs = pd.read_csv(f"data/synth_meta/{ver}/sequences.csv").set_index("seq_id")
    buckets = {n: [] for n in ("mean2", "mean3", WINNER)}
    for p in files:
        s1, s2, ws, ev_map, valid, nu = load_trio(tmp, ver, split, p)
        with np.load(p) as z:
            x = np.asarray(z["x"], np.float64)
            t = np.asarray(z["t"], np.float64)
        ev_idx, nu_idx, bg_idx, n_bg = ev._indices(ev_map, valid, nu, p.stem)
        cooling, _rows = ev._events_for(p.stem, ev_map, x, t, labels, seqs, ev_idx)

        def frame(arr, stem=p.stem, cooling=cooling, n_bg=n_bg, ev_idx=ev_idx, nu_idx=nu_idx, bg_idx=bg_idx):
            e, n, b = sel2._cols(arr, ev_idx, nu_idx, bg_idx)
            return dict(ev_m=e[None], nu_m=n[None], bg_m=b[None], n_bg=n_bg, cooling=cooling, stem=stem)

        buckets["mean2"].append(frame(blend_of("mean2", s1, s2, ws)))
        buckets["mean3"].append(frame(blend_of("mean3", s1, s2, ws)))
        buckets[WINNER].append(frame(ws))
        del s1, s2, ws
    order = np.argsort([fr["stem"] for fr in buckets[WINNER]])
    frames = {n: [buckets[n][i] for i in order] for n in buckets}
    winner = frames[WINNER]
    out = {
        "stems": [fr["stem"] for fr in winner],
        "procedure": "s3_sel2_cv sequence_folds i%k, holdout FAR 1e-3, micro, bg cap 200k",
    }
    for name in ("mean2", "mean3"):
        out[name] = {}
        for k in (2, 5):
            folds = sel.sequence_folds(len(winner), k)
            parts_w, parts_b, fold_rows = [], [], []
            for r, idx in enumerate(folds):
                hw = sel2.eval_one(sel2.take(winner, idx))
                hb = sel2.eval_one(sel2.take(frames[name], idx))
                parts_w.append(hw)
                parts_b.append(hb)
                fold_rows.append(dict(
                    fold=r, n_seq=len(idx),
                    winner=float(hw["iou"]), selected=float(hb["iou"]),
                    delta=float(hb["iou"] - hw["iou"]),
                ))
            iw = float(sel.micro_iou(parts_w))
            ib = float(sel.micro_iou(parts_b))
            out[name][str(k)] = dict(
                winner_iou=iw, selected_iou=ib, delta=float(ib - iw), folds=fold_rows,
            )
            log(f"nested {ver} {split} {name} k={k} {ib:.4f} winner {iw:.4f} delta {ib - iw:+.4f}")
    return out


def eval_split(m1, m2, ver, split, thr, tmp):
    files = predict_split(m1, m2, ver, split, tmp)
    summary, rows = score_files(files, tmp, ver, split, thr)
    nest = nested_files(files, tmp, ver, split)
    boot = {}
    for name in ("mean2", "mean3"):
        boot[name] = bootstrap(rows[name], rows[WINNER])
        log(
            f"{ver} {split} {name} {summary[name]['iou']:.4f} winner {summary[WINNER]['iou']:.4f} "
            f"delta {boot[name]['point']:+.4f} CI [{boot[name]['p2_5']:+.4f}, {boot[name]['p97_5']:+.4f}] "
            f"covers0 {boot[name]['covers_zero']}"
        )
    clear_split(tmp, ver, split, files)
    return summary, boot, nest


def main():
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if not SEED2.is_file():
        raise SystemExit(f"нет {SEED2}")
    t0 = time.time()
    win_full, val_files = winner_full_val()
    if len(val_files) != 45:
        raise SystemExit(f"v8 val {len(val_files)}")
    dropped = tuple(p.stem for p in val_files[:4])
    if dropped != HOLD_DROP:
        raise SystemExit(f"первые 4 v8 val {dropped}")
    hold = val_files[4:]
    if len(hold) != 41:
        raise SystemExit(f"holdout {len(hold)}")
    tmp = Path(f"/tmp/s3_gen2_seed2_eval_{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        m1 = load_net(SEED1)
        m2 = load_net(SEED2)
        log("сети загружены")
        predict_split(m1, m2, "v8", "val", tmp)
        # Порог только по 41. Четыре файла прокси первого сида в него не входят.
        thr = fit_thr(hold, tmp, "v8", "val")
        hold_sum, hold_rows = score_files(hold, tmp, "v8", "val", thr)
        frozen_rows = []
        for p in hold:
            _s1, _s2, ws, ev, valid, _nu = load_trio(tmp, "v8", "val", p)
            frozen_rows.append(M.confusion((ws >= WINNER_THR) & valid, ev > 0, valid))
            del ws
        win_frozen = pack(frozen_rows)["iou"]
        log(f"winner 41 frozen {win_frozen:.6f}")
        if abs(win_frozen - 0.3784001812423094) > 1e-4:
            raise SystemExit(f"победитель на 41 при опубликованном пороге {win_frozen}")
        nest41 = nested_files(hold, tmp, "v8", "val")
        boot = {
            "val41": {
                "mean2": bootstrap(hold_rows["mean2"], hold_rows[WINNER]),
                "mean3": bootstrap(hold_rows["mean3"], hold_rows[WINNER]),
            }
        }
        for name in ("mean2", "mean3"):
            b = boot["val41"][name]
            log(
                f"val41 {name} {hold_sum[name]['iou']:.4f} winner {hold_sum[WINNER]['iou']:.4f} "
                f"delta {b['point']:+.4f} CI [{b['p2_5']:+.4f}, {b['p97_5']:+.4f}] covers0 {b['covers_zero']}"
            )
        clear_split(tmp, "v8", "val", val_files)
        splits = {}
        nested = {"val41": nest41}
        for ver, split, key in (("v8", "test", "test"), ("v10", "test", "v10"), ("v11", "test", "v11")):
            summary, b, nest = eval_split(m1, m2, ver, split, thr, tmp)
            splits[key] = summary
            boot[key] = b
            nested[key] = nest
        report = dict(
            name="GEN2_SEED2",
            seed=20261018,
            seed1=20261015,
            definitions={
                "mean2": "0.5 * (seed2 + ens_u2ft_tcn_lgbm)",
                "mean3": "(seed1 + seed2 + ens_u2ft_tcn_lgbm) / 3",
            },
            proxy="all artifacts/cache/v10/val, not v8 val",
            holdout_dropped=list(HOLD_DROP),
            holdout_n=41,
            threshold_source="FAR 1e-3 on the 41 v8 val backgrounds only; frozen for test, v10, v11",
            winner_full_val_iou=win_full,
            winner_match=abs(win_full - WINNER_VAL) < 1e-9,
            winner_41_at_published_threshold=win_frozen,
            thresholds=thr,
            val41=hold_sum,
            splits=splits,
            bootstrap=boot,
            nested=nested,
            bootstrap_definition="1000 resamples of sequences with replacement, micro IoU delta versus winner, percentile 2.5 and 97.5, seed 20261018",
            real_used=False,
            seconds=round(time.time() - t0, 1),
        )
        dump(report)
        log(f"записан {OUT}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
