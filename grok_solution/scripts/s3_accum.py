"""Накопление свидетельств поверх вероятностей ансамбля, без обучения.

Преобразования смотрят только на карту s и маску valid. Порог FAR 1e-3
подбирается на фоне v8 val (valid и ev==0 и nu==0) и замораживается для
test v8/v10/v11. Метки нужны только для отчёта IoU и recall.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

# Потоки до импорта numpy: общая машина, не забирать все ядра.
_thr = "2"
if "--threads" in sys.argv:
    _thr = sys.argv[sys.argv.index("--threads") + 1]
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = _thr

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter, uniform_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dts import data, metrics as M

CLIP = np.float32(1e-4)
FAR = 1e-3
METHOD = "ens_u2ft_tcn_lgbm"
PUBLISHED = {
    "thr": 0.8608087301264272,
    "val_iou": 0.38671235821914546,
    "val_far": 0.0010000046700476173,
    "test_iou": 0.4052762065230473,
    "test_weak_recall": 0.017543859649122806,
    "test_weak_n": 57,
    "v10_test_iou": 0.4038977883892501,
    "v11_test_iou": 0.40390028295846503,
}
# Принять вариант, если val IoU >= 0.387, либо weak recall вырос и val IoU >= 0.38.
IOU_ACCEPT = 0.387
IOU_FLOOR = 0.38
EVENT_LOG = Path(os.environ.get("ACCUM_EVENT_LOG", "/workspace/grok_logs/_events.log"))

VARIANTS = (
    {"name": "identity", "kind": "identity"},
    {"name": "ma_s_w3", "kind": "ma", "w": 3},
    {"name": "ma_s_w7", "kind": "ma", "w": 7},
    {"name": "ma_s_w15", "kind": "ma", "w": 15},
    {"name": "logit_sum_w3", "kind": "logit_sum", "w": 3},
    {"name": "logit_sum_w7", "kind": "logit_sum", "w": 7},
    {"name": "logit_sum_w15", "kind": "logit_sum", "w": 15},
    {"name": "cusum_fwd", "kind": "cusum", "mode": "forward", "drift": 0.0},
    {"name": "cusum_fwd_d05", "kind": "cusum", "mode": "forward", "drift": 0.5},
    {"name": "cusum_ctr", "kind": "cusum", "mode": "center", "drift": 0.0},
    {"name": "cusum_ctr_d05", "kind": "cusum", "mode": "center", "drift": 0.5},
    {"name": "box_t3_x3", "kind": "box", "kt": 3, "kx": 3},
    {"name": "box_t7_x3", "kind": "box", "kt": 7, "kx": 3},
    {"name": "box_t7_x9", "kind": "box", "kt": 7, "kx": 9},
    {"name": "box_t15_x3", "kind": "box", "kt": 15, "kx": 3},
    {"name": "box_t15_x9", "kind": "box", "kt": 15, "kx": 9},
    {"name": "box_t5_x21", "kind": "box", "kt": 5, "kx": 21},
)


def logit(s: np.ndarray) -> np.ndarray:
    p = np.clip(s, CLIP, np.float32(1.0) - CLIP)
    return (np.log(p) - np.log1p(-p)).astype(np.float32, copy=False)


def _finite(a: np.ndarray) -> np.ndarray:
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def masked_reduce(arr, valid, size, how: str) -> np.ndarray:
    """Маскированное среднее или сумма. Паддинг 0 не входит в знаменатель.

    При постоянном числе valid в окне сумма и среднее монотонно эквивалентны,
    порог FAR их не различает. Сумма оставлена как запрошенное накопление
    лог-шансов; на краях valid они расходятся.
    """
    x = np.where(valid, arr, np.float32(0))
    wgt = valid.astype(np.float32)
    if isinstance(size, tuple):
        num = uniform_filter(x, size=size, mode="constant", cval=0.0)
        den = uniform_filter(wgt, size=size, mode="constant", cval=0.0)
        scale = np.float32(size[0] * size[1])
    else:
        num = uniform_filter1d(x, size=int(size), axis=0, mode="constant", cval=0.0)
        den = uniform_filter1d(wgt, size=int(size), axis=0, mode="constant", cval=0.0)
        scale = np.float32(size)
    if how == "mean":
        out = num / np.maximum(den, np.float32(1e-6))
    elif how == "sum":
        out = num * scale
    else:
        raise ValueError(how)
    return _finite(out)


def col_median(z: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Медиана logit по столбцу x. Невалидные не входят. Метки не используются."""
    a = np.where(valid, np.asarray(z, np.float32), np.float32(np.inf))
    a.sort(axis=0)
    cnt = valid.sum(axis=0).astype(np.int32)
    idx = np.maximum((cnt - 1) // 2, 0)
    cols = np.arange(a.shape[1])
    lo = a[idx, cols]
    hi = a[np.minimum(idx + 1, a.shape[0] - 1), cols]
    even = (cnt % 2) == 0
    med = np.where(even & (cnt > 0), np.float32(0.5) * (lo + hi), lo)
    med = np.where(cnt > 0, med, np.float32(0))
    return med.astype(np.float32, copy=False)


def page_cusum(z: np.ndarray, valid: np.ndarray, level: np.ndarray, backward: bool) -> np.ndarray:
    """Page CUSUM: C = max(0, C + logit - (медиана столбца + drift))."""
    t_len, x_len = z.shape
    inc = np.where(valid, z - level[None, :], np.float32(0)).astype(np.float32, copy=False)
    c = np.zeros(x_len, np.float32)
    out = np.empty((t_len, x_len), np.float32)
    steps = range(t_len - 1, -1, -1) if backward else range(t_len)
    for t in steps:
        np.add(c, inc[t], out=c)
        np.maximum(c, np.float32(0), out=c)
        out[t] = c
    return out


def transform(s: np.ndarray, valid: np.ndarray, spec: dict) -> np.ndarray:
    kind = spec["kind"]
    if kind == "identity":
        return s
    if kind == "ma":
        return masked_reduce(s, valid, spec["w"], "mean")
    if kind == "logit_sum":
        return masked_reduce(logit(s), valid, spec["w"], "sum")
    if kind == "box":
        return masked_reduce(logit(s), valid, (spec["kt"], spec["kx"]), "mean")
    if kind == "cusum":
        z = logit(s)
        level = col_median(z, valid) + np.float32(spec["drift"])
        fwd = page_cusum(z, valid, level, backward=False)
        if spec["mode"] == "forward":
            return fwd
        if spec["mode"] == "center":
            # min(вперёд, назад): масса внутри серии, а не на её хвосте.
            bwd = page_cusum(z, valid, level, backward=True)
            return _finite(np.minimum(fwd, bwd))
        raise ValueError(spec["mode"])
    raise ValueError(kind)


def far_threshold(bg: np.ndarray, far: float = FAR) -> float:
    """Та же формула, что dts.metrics.far_threshold, сортировка на месте."""
    if bg.size == 0:
        return 1.0
    bg.sort()
    k = int(np.ceil((1.0 - far) * bg.size)) - 1
    k = int(np.clip(k, 0, bg.size - 1))
    return float(bg[k]) + 1e-12


def load_split(ver: str, split: str):
    rows = []
    pred_root = Path("artifacts/pred") / ver / METHOD / split
    for p in data.list_cache(f"artifacts/cache/{ver}", split):
        z = np.load(p)
        q = np.load(pred_root / p.name)
        valid = np.asarray(z["valid"], bool)
        ev = np.asarray(z["ev"])
        nu = np.asarray(z["nu"])
        s = np.asarray(q["s"], np.float32)
        rows.append({
            "sid": p.stem,
            "s": s,
            "valid": valid,
            "ev": ev,
            "x": np.asarray(z["x"]),
            "t": np.asarray(z["t"]),
            "bg": valid & (ev == 0) & (nu == 0),
        })
    return rows


def precompute_events(rows, labels: pd.DataFrame):
    """Плоские индексы пикселей события. Та же геометрия, что metrics.event_pixels."""
    by_sid = {d["sid"]: [] for d in rows}
    for d in rows:
        boxes = M.label_boxes(labels, d["sid"])
        ev, x, t = d["ev"], d["x"], d["t"]
        width = ev.shape[1]
        for r in boxes.itertuples():
            if r.kind != "event":
                continue
            ix = np.flatnonzero((x >= r.dist_start_m - 0.25) & (x <= r.dist_end_m + 0.25))
            it = np.flatnonzero((t >= r.ts - 120) & (t <= r.te + 120))
            if ix.size == 0 or it.size == 0:
                continue
            sub = ev[np.ix_(it, ix)] == int(r.mask_value)
            if not np.any(sub):
                continue
            tt, xx = np.nonzero(sub)
            flat = it[tt].astype(np.int64) * width + ix[xx].astype(np.int64)
            by_sid[d["sid"]].append({
                "amp": float(r.amp_obs_C),
                "flat": flat,
                "n": int(flat.size),
                "bin": M.amp_bin(float(r.amp_obs_C)),
            })
    return by_sid


def _agg(hits):
    if not hits:
        return {"n": 0, "recall": None, "pixel_recall": None}
    tp = sum(h[1] for h in hits)
    fn = sum(h[2] for h in hits)
    return {
        "n": int(len(hits)),
        "recall": float(sum(h[0] for h in hits) / len(hits)),
        "pixel_recall": float(tp / max(tp + fn, 1)),
    }


def evaluate(rows, scores, thr: float, events) -> dict:
    tp = fp = fn = 0
    far_n = far_d = 0
    bins = {"<=2C": [], "2-5C": [], ">5C": []}
    for d, sc in zip(rows, scores):
        pred = (sc >= thr) & d["valid"]
        truth = d["ev"] > 0
        ctp, cfp, cfn = M.confusion(pred, truth, d["valid"])
        tp += ctp
        fp += cfp
        fn += cfn
        b = d["bg"]
        far_n += int((pred & b).sum())
        far_d += int(b.sum())
        pr = pred.ravel()
        for e in events[d["sid"]]:
            etp = int(pr[e["flat"]].sum())
            efn = e["n"] - etp
            bins[e["bin"]].append((etp / e["n"] >= 0.3, etp, efn))
    prf = M.prf(tp, fp, fn)
    weak = _agg(bins["<=2C"])
    return {
        "threshold": float(thr),
        "far": float(far_n / max(far_d, 1)),
        "n_bg": int(far_d),
        "precision": prf["precision"],
        "recall": prf["recall"],
        "f1": prf["f1"],
        "iou": prf["iou"],
        "weak_n": weak["n"],
        "weak_recall": weak["recall"],
        "weak_pixel_recall": weak["pixel_recall"],
        "by_amp": {k: _agg(v) for k, v in bins.items()},
    }


def apply_all(rows, spec: dict):
    scores = []
    chunks = []
    for d in rows:
        sc = transform(d["s"], d["valid"], spec)
        scores.append(sc)
        chunks.append(np.ascontiguousarray(sc[d["bg"]]))
    return scores, chunks


def fit_and_eval(rows, events, spec: dict) -> dict:
    t0 = time.time()
    scores, chunks = apply_all(rows, spec)
    bg = np.concatenate(chunks)
    del chunks
    thr = far_threshold(bg, FAR)
    del bg
    metrics = evaluate(rows, scores, thr, events)
    del scores
    metrics["sec"] = round(time.time() - t0, 2)
    return metrics


def eval_frozen(rows, events, spec: dict, thr: float) -> dict:
    t0 = time.time()
    scores = [transform(d["s"], d["valid"], spec) for d in rows]
    metrics = evaluate(rows, scores, thr, events)
    del scores
    metrics["sec"] = round(time.time() - t0, 2)
    metrics["threshold"] = float(thr)
    return metrics


def params_of(spec: dict) -> dict:
    skip = {"name", "kind"}
    return {k: v for k, v in spec.items() if k not in skip}


def qualifies(iou: float, weak: float, base_weak: float) -> bool:
    if iou >= IOU_ACCEPT:
        return True
    if iou >= IOU_FLOOR and weak is not None and base_weak is not None and weak > base_weak:
        return True
    return False


def rank_key(m: dict):
    weak = -1.0 if m["weak_recall"] is None else float(m["weak_recall"])
    return (weak, float(m["iou"]), float(m["f1"]))


def _self_check() -> None:
    rng = np.random.default_rng(0)
    z = rng.normal(size=(30, 6)).astype(np.float32)
    valid = rng.random((30, 6)) > 0.2
    valid[0, 0] = True
    got = col_median(z, valid)
    masked = np.where(valid, z, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ref = np.nanmedian(masked, axis=0)
    if not np.allclose(got, ref, atol=1e-5, equal_nan=True):
        raise SystemExit("self-check: col_median != nanmedian")
    s = np.full((21, 9), np.float32(0.2))
    v = np.ones((21, 9), bool)
    v[:2, :] = False
    ma = transform(s, v, {"kind": "ma", "w": 5})
    if not np.allclose(ma[10], np.float32(0.2), atol=1e-5):
        raise SystemExit("self-check: moving average")
    sm = transform(s, v, {"kind": "logit_sum", "w": 5})
    if not np.allclose(sm[10], 5.0 * logit(s)[10], atol=1e-4):
        raise SystemExit("self-check: logit sum")
    bx = transform(s, v, {"kind": "box", "kt": 3, "kx": 3})
    if not np.allclose(bx[10, 4], logit(s)[10, 4], atol=1e-4):
        raise SystemExit("self-check: logit box")
    s2 = s.copy()
    s2[8:14, 4] = np.float32(0.85)
    ctr = transform(s2, v, {"kind": "cusum", "mode": "center", "drift": 0.0})
    fwd = transform(s2, v, {"kind": "cusum", "mode": "forward", "drift": 0.0})
    if not (ctr[11, 4] > ctr[11, 1] and ctr[4, 4] < ctr[11, 4]):
        raise SystemExit("self-check: centered cusum")
    # Вперёд копит к концу серии, центр сидит внутри.
    if not (fwd[13, 4] >= fwd[8, 4] and ctr[10, 4] >= ctr[13, 4]):
        raise SystemExit("self-check: cusum shape")
    bg = np.linspace(0, 1, 1000, dtype=np.float32)
    thr = far_threshold(bg.copy(), 1e-3)
    if abs(thr - (float(np.sort(bg)[int(np.ceil(0.999 * 1000)) - 1]) + 1e-12)) > 1e-9:
        raise SystemExit("self-check: far threshold")
    print("self-check ok", flush=True)


def _brief(m: dict) -> str:
    wr = m["weak_recall"]
    wp = m["weak_pixel_recall"]
    wr_s = "None" if wr is None else f"{wr:.4f}"
    wp_s = "None" if wp is None else f"{wp:.4f}"
    return (
        f"thr={m['threshold']:.6g} iou={m['iou']:.6f} f1={m['f1']:.4f} "
        f"far={m['far']:.6f} weak={wr_s} n={m['weak_n']} px={wp_s} {m.get('sec', 0)}s"
    )


def dump(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n")
    print("wrote", path, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out-dir", default="artifacts/eval")
    ap.add_argument("--self-check-only", action="store_true")
    ap.add_argument("--skip-transfer", action="store_true")
    a = ap.parse_args()
    _self_check()
    if a.self_check_only:
        return

    out_dir = Path(a.out_dir)
    labels_v8 = pd.read_csv("data/synth_meta/v8/labels.csv")
    print("load v8 val", flush=True)
    t0 = time.time()
    val_rows = load_split("v8", "val")
    val_ev = precompute_events(val_rows, labels_v8)
    n_weak = sum(1 for xs in val_ev.values() for e in xs if e["amp"] <= 2)
    print(f"val seq {len(val_rows)} weak_events {n_weak} load {time.time() - t0:.1f}s", flush=True)

    grid = []
    by_name = {}
    for spec in VARIANTS:
        metrics = fit_and_eval(val_rows, val_ev, spec)
        row = {
            "name": spec["name"],
            "kind": spec["kind"],
            "params": params_of(spec),
            "val": metrics,
        }
        grid.append(row)
        by_name[spec["name"]] = row
        print(f"[val] {spec['name']} {_brief(metrics)}", flush=True)
        if spec["name"] == "identity":
            di = abs(metrics["iou"] - PUBLISHED["val_iou"])
            dt = abs(metrics["threshold"] - PUBLISHED["thr"])
            print(f"identity check iou_delta={di:.3e} thr_delta={dt:.3e}", flush=True)
            if di > 1e-4 or dt > 1e-5:
                dump(out_dir / "s3_accum_v8.json", {"error": "identity mismatch", "grid": grid})
                raise SystemExit(f"identity val mismatch iou_delta={di} thr_delta={dt}")

    base = by_name["identity"]["val"]
    base_weak = float(base["weak_recall"] or 0.0)
    for row in grid:
        m = row["val"]
        row["qualifies"] = qualifies(float(m["iou"]), None if m["weak_recall"] is None else float(m["weak_recall"]), base_weak)
    pool = [r for r in grid if r["qualifies"]]
    best = max(grid, key=lambda r: rank_key(r["val"]))
    if pool:
        chosen_row = max(pool, key=lambda r: rank_key(r["val"]))
        accepted = True
        reason = (
            "val IoU >= 0.387"
            if chosen_row["val"]["iou"] >= IOU_ACCEPT
            else "weak recall вырос при val IoU >= 0.38"
        )
    else:
        chosen_row = None
        accepted = False
        reason = "ни один вариант не прошёл правило приёмки"
    chosen_name = None if chosen_row is None else chosen_row["name"]
    print(f"decision accepted={accepted} chosen={chosen_name} best={best['name']} {reason}", flush=True)

    del val_rows, val_ev
    print("load v8 test", flush=True)
    test_rows = load_split("v8", "test")
    test_ev = precompute_events(test_rows, labels_v8)
    for row in grid:
        tm = eval_frozen(test_rows, test_ev, _spec(row["name"]), row["val"]["threshold"])
        row["test"] = tm
        print(f"[test v8] {row['name']} {_brief(tm)}", flush=True)
    ident_test = by_name["identity"]["test"]
    d_iou = abs(ident_test["iou"] - PUBLISHED["test_iou"])
    d_w = abs(float(ident_test["weak_recall"] or 0) - PUBLISHED["test_weak_recall"])
    print(f"identity test check iou_delta={d_iou:.3e} weak_delta={d_w:.3e}", flush=True)
    if d_iou > 1e-4 or d_w > 1e-6:
        print("WARNING identity test does not match published", flush=True)

    report = {
        "task": "s3_accum",
        "model": METHOD,
        "version_fit": "v8",
        "far": FAR,
        "trained": False,
        "transform_sees_labels": False,
        "published_identity": PUBLISHED,
        "identity_check": {
            "val_iou_delta": abs(base["iou"] - PUBLISHED["val_iou"]),
            "thr_delta": abs(base["threshold"] - PUBLISHED["thr"]),
            "test_iou_delta": d_iou,
            "test_weak_delta": d_w,
        },
        "accept_rule": "val IoU >= 0.387 OR (weak_recall > identity AND val IoU >= 0.38)",
        "accepted": accepted,
        "chosen": chosen_name,
        "best_effort": best["name"],
        "reason": reason,
        "grid": grid,
    }
    dump(out_dir / "s3_accum_v8.json", report)
    del test_rows, test_ev

    transfer_names = ["identity"]
    if chosen_name and chosen_name not in transfer_names:
        transfer_names.append(chosen_name)
    if best["name"] not in transfer_names:
        transfer_names.append(best["name"])
    frozen = {n: by_name[n]["val"]["threshold"] for n in transfer_names}

    if not a.skip_transfer:
        for ver in ("v10", "v11"):
            lab = pd.read_csv(f"data/synth_meta/{ver}/labels.csv")
            print(f"load {ver} test", flush=True)
            rows = load_split(ver, "test")
            ev = precompute_events(rows, lab)
            block = {
                "task": "s3_accum",
                "model": METHOD,
                "version": ver,
                "split": "test",
                "threshold_source": "v8 val background FAR 1e-3, frozen",
                "accepted": accepted,
                "chosen": chosen_name,
                "best_effort": best["name"],
                "variants": {},
            }
            for name in transfer_names:
                tm = eval_frozen(rows, ev, _spec(name), frozen[name])
                block["variants"][name] = tm
                print(f"[test {ver}] {name} {_brief(tm)}", flush=True)
            dump(out_dir / f"s3_accum_{ver}.json", block)
            del rows, ev

    chosen_val = None if chosen_row is None else chosen_row["val"]
    chosen_test = None if chosen_row is None else chosen_row["test"]
    best_val, best_test = best["val"], best["test"]
    # Для строки лога: принятый вариант, иначе лучший по val (не принят).
    shown = chosen_row if chosen_row is not None else best
    v10 = json.loads((out_dir / "s3_accum_v10.json").read_text()) if (out_dir / "s3_accum_v10.json").exists() else {}
    v11 = json.loads((out_dir / "s3_accum_v11.json").read_text()) if (out_dir / "s3_accum_v11.json").exists() else {}

    def _pull(blob, name, key, default="nan"):
        m = (blob.get("variants") or {}).get(name) or {}
        val = m.get(key, default)
        return default if val is None else val

    show_name = shown["name"]
    line = (
        "ACCUM_DONE "
        f"accepted={'yes' if accepted else 'no'} variant={show_name} "
        f"val_iou={shown['val']['iou']:.6f} val_iou_id={base['iou']:.6f} "
        f"val_weak={shown['val']['weak_recall']} val_weak_id={base['weak_recall']} "
        f"val_weak_px={shown['val']['weak_pixel_recall']} val_weak_px_id={base['weak_pixel_recall']} "
        f"test_iou={shown['test']['iou']:.6f} test_iou_id={ident_test['iou']:.6f} "
        f"test_weak={shown['test']['weak_recall']} test_weak_id={ident_test['weak_recall']} "
        f"v10_iou={_pull(v10, show_name, 'iou')} v10_weak={_pull(v10, show_name, 'weak_recall')} "
        f"v10_iou_id={_pull(v10, 'identity', 'iou')} v10_weak_id={_pull(v10, 'identity', 'weak_recall')} "
        f"v11_iou={_pull(v11, show_name, 'iou')} v11_weak={_pull(v11, show_name, 'weak_recall')} "
        f"v11_iou_id={_pull(v11, 'identity', 'iou')} v11_weak_id={_pull(v11, 'identity', 'weak_recall')} "
        f"best={best['name']} best_val_iou={best_val['iou']:.6f} best_val_weak={best_val['weak_recall']}"
    )
    print(line, flush=True)
    with EVENT_LOG.open("a") as f:
        f.write(line + "\n")
    # silence unused in case of future edits
    _ = (chosen_val, chosen_test, best_test)


def _spec(name: str) -> dict:
    for spec in VARIANTS:
        if spec["name"] == name:
            return spec
    raise KeyError(name)


if __name__ == "__main__":
    main()
