"""Оценка многозадачной U-Net в смеси с победителем.

Кандидат заморожен: ens_blend = 0.5 * ens_u2ft_tcn_lgbm + 0.5 * unet_mtl.
Порог FAR 1e-3 — фон v8 val. Вложенные фолды — sequence_folds / micro_iou /
nested_verdict из scripts/s3_sel_core.py (то же, что s3_sel2_cv.nested_fixed):
k=2 и k=5, порог только по фону holdout. Принять вложенную проверку можно
только если min(micro) строго выше 0.3867+0.003. v10 и v11 test — тем же
порогом v8, без подгонки. Реальные CSV не читаются.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_k] = "2"

import argparse
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from dts import data, metrics as M, models

WINNER = "ens_u2ft_tcn_lgbm"
WINNER_VAL = 0.38671235821914546
WINNER_TEST = 0.4052762065230473
WINNER_THR = 0.8608087301264272
WINNER_V10 = 0.4038977883892501
WINNER_V11 = 0.40390028295846503
NESTED_BAR = 0.3867 + 0.003
NAMES = ("winner", "unet_mtl", "ens_blend")
TMP = Path("/tmp/s3_mtl_maps")
LOG_PATH = Path("/workspace/grok_logs/_events.log")


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S', time.gmtime())} {msg}", flush=True)


def _sel():
    spec = importlib.util.spec_from_file_location("s3_sel_core", ROOT / "scripts" / "s3_sel_core.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _train_mod():
    spec = importlib.util.spec_from_file_location("s3_mtl_train", ROOT / "scripts" / "s3_mtl_train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


def tile_starts(n: int, tile: int, step: int) -> list[int]:
    ts = list(range(0, max(n - tile, 0) + 1, step)) or [0]
    if ts[-1] + tile < n:
        ts.append(n - tile)
    return ts


@torch.inference_mode()
def predict_batched(model, X, tile=(256, 512), overlap=32, batch=4):
    X = np.asarray(X, np.float32)
    _, t_len, w_len = X.shape
    th, tw = tile
    step_t, step_w = th - 2 * overlap, tw - 2 * overlap
    ts = tile_starts(t_len, th, step_t)
    ws = tile_starts(w_len, tw, step_w)
    boxes = [(t0, w0) for t0 in ts for w0 in ws]
    out = np.zeros((3, t_len, w_len), np.float32)
    cnt = np.zeros((t_len, w_len), np.float32)
    for i in range(0, len(boxes), batch):
        chunk = boxes[i:i + batch]
        xs, meta = [], []
        for t0, w0 in chunk:
            sl = X[:, t0:t0 + th, w0:w0 + tw]
            ht, wt = sl.shape[1], sl.shape[2]
            if ht != th or wt != tw:
                sl = np.pad(sl, ((0, 0), (0, th - ht), (0, tw - wt)))
            xs.append(sl)
            meta.append((t0, w0, ht, wt))
        prob = torch.softmax(model(torch.from_numpy(np.stack(xs))), 1).numpy()
        for p, (t0, w0, ht, wt) in zip(prob, meta):
            out[:, t0:t0 + ht, w0:w0 + wt] += p[:, :ht, :wt]
            cnt[t0:t0 + ht, w0:w0 + wt] += 1
    out /= np.maximum(cnt, 1.0)
    return out


def check_predict(model) -> float:
    model.eval()
    rng = np.random.default_rng(0)
    X = rng.standard_normal((3, 180, 400)).astype(np.float32)
    a = predict_batched(model, X, batch=4)
    b = models.predict_full(model, X)
    err = float(np.max(np.abs(a - b)))
    if err > 1e-4:
        raise SystemExit(f"predict_batched расходится с models.predict_full: {err}")
    return err


def bg_mask(ev, nu, valid):
    return valid & (ev == 0) & (nu == 0)


def open_map(split, stem, key, shape, dtype):
    path = TMP / split / f"{stem}_{key}.mmap"
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.memmap(path, dtype=dtype, mode="w+", shape=shape)


def read_map(split, stem, key, shape, dtype):
    path = TMP / split / f"{stem}_{key}.mmap"
    return np.memmap(path, dtype=dtype, mode="r", shape=shape)


def shapes_of(ver: str, split: str) -> dict[str, tuple[int, int]]:
    out = {}
    for p in data.list_cache(f"artifacts/cache/{ver}", split):
        with np.load(Path("artifacts/pred") / ver / WINNER / split / p.name) as z:
            out[p.stem] = tuple(z["s"].shape)
    return out


def predict_split(model, ver: str, split: str, shapes) -> None:
    t0 = time.time()
    files = data.list_cache(f"artifacts/cache/{ver}", split)
    for i, p in enumerate(files, 1):
        shape = shapes[p.stem]
        with np.load(p) as z:
            X = np.asarray(z["X"], np.float32)
        if X.shape[1:] != shape:
            raise SystemExit(f"{ver}/{p.name}: X {X.shape[1:]} != pred {shape}")
        prob = predict_batched(model, X)
        s = (1.0 - prob[0]).astype(np.float32)
        h = (prob[1] - prob[2]).astype(np.float16)
        sm = open_map(f"{ver}_{split}", p.stem, "s", shape, np.float32)
        hm = open_map(f"{ver}_{split}", p.stem, "h", shape, np.float16)
        sm[:] = s
        hm[:] = h
        sm.flush()
        hm.flush()
        del sm, hm, X, prob
        if i == 1 or i % 10 == 0 or i == len(files):
            log(f"pred {ver}/{split} {i}/{len(files)} {time.time() - t0:.0f}s")


def load_winner(ver, split, stem, shape):
    with np.load(Path("artifacts/pred") / ver / WINNER / split / f"{stem}.npz") as z:
        s = np.asarray(z["s"], np.float32)
        h = np.asarray(z["h"], np.float32)
    if s.shape != shape:
        raise SystemExit(f"winner {ver}/{stem} {s.shape} != {shape}")
    return s, h


def load_cached(tag, stem, shape):
    s = np.asarray(read_map(tag, stem, "s", shape, np.float32))
    h = np.asarray(read_map(tag, stem, "h", shape, np.float16), np.float32)
    return s, h


def score_maps(ver, split, stem, shape, name):
    tag = f"{ver}_{split}"
    if name == "winner":
        return load_winner(ver, split, stem, shape)
    dg_s, dg_h = load_cached(tag, stem, shape)
    if name == "unet_mtl":
        return dg_s, dg_h
    w_s, w_h = load_winner(ver, split, stem, shape)
    return (0.5 * (w_s + dg_s)).astype(np.float32), (0.5 * (w_h + dg_h)).astype(np.float32)


def meta_of(ver, split, stem):
    with np.load(Path("artifacts/cache") / ver / split / f"{stem}.npz") as z:
        return (
            np.asarray(z["ev"]),
            np.asarray(z["valid"], bool),
            np.asarray(z["nu"]),
            np.asarray(z["x"]),
            np.asarray(z["t"]),
        )


def threshold_of(ver, split, files, shapes, name) -> tuple[float, int]:
    parts = []
    for p in files:
        ev, valid, nu, _x, _t = meta_of(ver, split, p.stem)
        s, _h = score_maps(ver, split, p.stem, shapes[p.stem], name)
        parts.append(np.asarray(s[bg_mask(ev, nu, valid)], np.float32))
        del s
    bg = np.concatenate(parts)
    thr = float(M.far_threshold(bg, 1e-3))
    n = int(bg.size)
    del bg, parts
    return thr, n


def _agg(g):
    tp = int(g.tp.sum())
    fn = int(g.fn.sum())
    return dict(n=int(len(g)), recall=float(g.hit.mean()) if len(g) else None,
                pixel_recall=float(tp / max(tp + fn, 1)))


def evaluate(ver, split, files, shapes, thr, labels) -> dict:
    tot = {n: np.zeros(3, np.int64) for n in thr}
    far_n = {n: 0 for n in thr}
    far_d = {n: 0 for n in thr}
    ev_rows = {n: [] for n in thr}
    for p in files:
        ev, valid, nu, x, t = meta_of(ver, split, p.stem)
        mask = bg_mask(ev, nu, valid)
        truth = ev > 0
        boxes = []
        for r in M.label_boxes(labels, p.stem).itertuples():
            if r.kind != "event":
                continue
            px = M.event_pixels(r, x, t, ev)
            if px.any():
                boxes.append((r, px))
        for n in thr:
            s, _h = score_maps(ver, split, p.stem, shapes[p.stem], n)
            pred = (s >= thr[n]) & valid
            cm = np.asarray(M.confusion(pred, truth, valid), np.int64)
            tot[n] += cm
            far_n[n] += int((pred & mask).sum())
            far_d[n] += int(mask.sum())
            for r, px in boxes:
                frac = float(pred[px].mean())
                ev_rows[n].append(dict(
                    amp=float(r.amp_obs_C), amp_bin=M.amp_bin(float(r.amp_obs_C)),
                    hit=frac >= 0.3, tp=int((pred & px).sum()), fn=int((~pred & px).sum()),
                ))
            del s, pred
    report = {}
    for n in thr:
        E = pd.DataFrame(ev_rows[n])
        row = dict(
            threshold=float(thr[n]),
            far=float(far_n[n] / max(far_d[n], 1)),
            n_bg=int(far_d[n]),
            **M.prf(*tot[n]),
            tp=int(tot[n][0]), fp=int(tot[n][1]), fn=int(tot[n][2]),
        )
        if len(E):
            row["events_all"] = _agg(E)
            row["weak_events"] = _agg(E[E.amp <= 2]) if (E.amp <= 2).any() else dict(n=0, recall=None, pixel_recall=None)
            row["by_amp"] = {k: _agg(g) for k, g in E.groupby("amp_bin")}
        report[n] = row
        weak = (row.get("by_amp") or {}).get("<=2C", {})
        log(f"{ver}/{split} {n} IoU {row['iou']:.6f} FAR {row['far']:.6f} weak {weak.get('recall')}")
    return report


def counts_on(ver, split, files, shapes, name, thr) -> dict:
    tot = np.zeros(3, np.int64)
    for p in files:
        ev, valid, _nu, _x, _t = meta_of(ver, split, p.stem)
        s, _h = score_maps(ver, split, p.stem, shapes[p.stem], name)
        tot += np.asarray(M.confusion((s >= thr) & valid, ev > 0, valid), np.int64)
        del s
    tp, fp, fn = (int(v) for v in tot)
    return dict(tp=tp, fp=fp, fn=fn, **M.prf(tp, fp, fn), threshold=float(thr), n_seq=len(files))


def nested(files, shapes, sel) -> dict:
    out = {}
    for k in (2, 5):
        folds = sel.sequence_folds(len(files), k)
        parts = {n: [] for n in NAMES}
        details = []
        for r, hold_idx in enumerate(folds):
            hold = [files[i] for i in hold_idx]
            fold_row = dict(fold=r, n_seq=len(hold), stems=[p.stem for p in hold])
            for n in NAMES:
                thr, n_bg = threshold_of("v8", "val", hold, shapes, n)
                got = counts_on("v8", "val", hold, shapes, n, thr)
                got["n_bg"] = n_bg
                parts[n].append(got)
                fold_row[n] = dict(iou=got["iou"], tp=got["tp"], fp=got["fp"], fn=got["fn"], threshold=got["threshold"])
            log(f"nested k={k} fold {r} blend {fold_row['ens_blend']['iou']:.4f} winner {fold_row['winner']['iou']:.4f}")
            details.append(fold_row)
        micros = {n: float(sel.micro_iou(parts[n])) for n in NAMES}
        out[str(k)] = dict(micro_iou=micros, folds=details)
        log(f"nested k={k} micro blend {micros['ens_blend']:.6f} winner {micros['winner']:.6f}")
    blend_by_k = {k: v["micro_iou"]["ens_blend"] for k, v in out.items()}
    bar = sel.nested_verdict(blend_by_k)
    return dict(by_k=out, nested_bar=bar)


def weak_recall(row):
    block = (row.get("by_amp") or {}).get("<=2C")
    if not block or block.get("recall") is None:
        return None
    return float(block["recall"])


def append_done(line: str) -> None:
    text = LOG_PATH.read_text(encoding="utf-8") if LOG_PATH.exists() else ""
    if "\nMTL_DONE " in ("\n" + text):
        raise SystemExit("MTL_DONE уже есть в _events.log, вторую строку не пишу")
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="artifacts/eval/s3_mtl_unet.pt")
    ap.add_argument("--skip-predict", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    sel = _sel()
    train = _train_mod()
    t0 = time.time()
    labels = {ver: pd.read_csv(f"data/synth_meta/{ver}/labels.csv") for ver in ("v8", "v10", "v11")}
    unet = models.build("unet", 16)
    model = train.MTLUNet(unet)
    model.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=True))
    model.eval()
    err = check_predict(model)
    log(f"сверка predict max|Δ| {err:.3e}")
    val_shapes = shapes_of("v8", "val")
    test_shapes = shapes_of("v8", "test")
    v10_shapes = shapes_of("v10", "test")
    v11_shapes = shapes_of("v11", "test")
    val_files = data.list_cache("artifacts/cache/v8", "val")
    test_files = data.list_cache("artifacts/cache/v8", "test")
    v10_files = data.list_cache("artifacts/cache/v10", "test")
    v11_files = data.list_cache("artifacts/cache/v11", "test")
    try:
        if not a.skip_predict:
            if TMP.exists():
                shutil.rmtree(TMP)
            predict_split(model, "v8", "val", val_shapes)
            predict_split(model, "v8", "test", test_shapes)
            predict_split(model, "v10", "test", v10_shapes)
            predict_split(model, "v11", "test", v11_shapes)
        thr = {}
        for n in NAMES:
            thr[n], n_bg = threshold_of("v8", "val", val_files, val_shapes, n)
            log(f"порог v8 {n} {thr[n]:.6f} n_bg {n_bg}")
        val_rep = evaluate("v8", "val", val_files, val_shapes, thr, labels["v8"])
        nest = nested(val_files, val_shapes, sel)
        test_rep = evaluate("v8", "test", test_files, test_shapes, thr, labels["v8"])
        v10_rep = evaluate("v10", "test", v10_files, v10_shapes, thr, labels["v10"])
        v11_rep = evaluate("v11", "test", v11_files, v11_shapes, thr, labels["v11"])
    finally:
        if TMP.exists():
            shutil.rmtree(TMP, ignore_errors=True)

    win_val = float(val_rep["winner"]["iou"])
    blend_val = float(val_rep["ens_blend"]["iou"])
    unet_val = float(val_rep["unet_mtl"]["iou"])
    win_test = float(test_rep["winner"]["iou"])
    blend_test = float(test_rep["ens_blend"]["iou"])
    v10_b = float(v10_rep["ens_blend"]["iou"])
    v10_w = float(v10_rep["winner"]["iou"])
    v11_b = float(v11_rep["ens_blend"]["iou"])
    v11_w = float(v11_rep["winner"]["iou"])
    protocol_ok = (
        abs(win_val - WINNER_VAL) <= 1e-4
        and abs(win_test - WINNER_TEST) <= 1e-4
        and abs(v10_w - WINNER_V10) <= 1e-4
        and abs(v11_w - WINNER_V11) <= 1e-4
    )
    if not protocol_ok:
        log(f"протокол не сошёлся val {win_val} test {win_test} v10 {v10_w} v11 {v11_w}")
    n5_b = float(nest["by_k"]["5"]["micro_iou"]["ens_blend"])
    n5_w = float(nest["by_k"]["5"]["micro_iou"]["winner"])
    n2_b = float(nest["by_k"]["2"]["micro_iou"]["ens_blend"])
    n2_w = float(nest["by_k"]["2"]["micro_iou"]["winner"])
    bar_accept = bool(nest["nested_bar"]["accept"])
    gain5 = n5_b - n5_w
    # Планка скрипта: min(k=2, k=5) > 0.3867+0.003. Прирост k=5 против победителя пишется рядом.
    nested_accept = bool(bar_accept and gain5 >= 0.003)
    v10_ok = bool(v10_b + 1e-6 >= v10_w)
    v11_ok = bool(v11_b + 1e-6 >= v11_w)
    beats = bool(protocol_ok and blend_val > WINNER_VAL + 1e-4 and nested_accept and v10_ok and v11_ok)
    train_meta = {}
    tp = Path("artifacts/eval/s3_mtl_train.json")
    if tp.exists():
        train_meta = json.loads(tp.read_text(encoding="utf-8"))
    sec = round(float(train_meta.get("seconds") or 0) + (time.time() - t0), 1)
    report = dict(
        task="MTL",
        protocol=(
            "Кандидат ens_blend = 0.5 * победитель + 0.5 * unet_mtl. Alpha не подбиралась. "
            "Официальная цифра — pixel IoU при FAR 1e-3 на всём v8 val. "
            "beats, только если val строго выше опубликованного победителя больше чем на 1e-4, "
            "min(micro k=2, k=5) строго выше 0.3897, прирост micro k=5 против победителя не меньше 0.003, "
            "и test IoU на v10 и v11 не хуже победителя при пороге, замороженном на v8 val. "
            "Строка BEATS_WINNER не пишется."
        ),
        published_winner=dict(
            val_iou=WINNER_VAL, test_iou=WINNER_TEST, thr=WINNER_THR,
            v10_test_iou=WINNER_V10, v11_test_iou=WINNER_V11,
        ),
        protocol_ok=protocol_ok,
        winner_val_abs_err=abs(win_val - WINNER_VAL),
        winner_test_abs_err=abs(win_test - WINNER_TEST),
        winner_v10_abs_err=abs(v10_w - WINNER_V10),
        winner_v11_abs_err=abs(v11_w - WINNER_V11),
        predict_max_abs_err=err,
        alpha=0.5,
        chosen="ens_blend",
        beats=beats,
        nested_accept=nested_accept,
        nested_bar_value=NESTED_BAR,
        gain5=gain5,
        v10_not_worse=v10_ok,
        v11_not_worse=v11_ok,
        thresholds=thr,
        val=val_rep,
        test=test_rep,
        v10_test=v10_rep,
        v11_test=v11_rep,
        nested=nest,
        train=dict(
            best_epoch=train_meta.get("best_epoch"),
            best_proxy_iou=train_meta.get("best_proxy_iou"),
            stopped=train_meta.get("stopped"),
            seconds=train_meta.get("seconds"),
            real_traces_used=False,
        ),
        real_traces_used=False,
        replaced_release=False,
        seconds=sec,
    )
    out = Path("artifacts/eval/s3_mtl_v8.json")
    out.write_text(json.dumps(plain(report), indent=1, ensure_ascii=False), encoding="utf-8")
    line = (
        f"MTL_DONE chosen=ens_blend beats={'yes' if beats else 'no'} "
        f"val_iou={blend_val:.10f} val_iou_winner={win_val:.10f} "
        f"val_weak={weak_recall(val_rep['ens_blend'])} "
        f"test_iou={blend_test:.10f} test_iou_winner={win_test:.10f} "
        f"unet_mtl_val={unet_val:.10f} "
        f"nested5_blend={n5_b:.10f} nested5_winner={n5_w:.10f} nested5_gain={gain5:.10f} "
        f"nested2_blend={n2_b:.10f} nested2_winner={n2_w:.10f} "
        f"nested_accept={'yes' if nested_accept else 'no'} "
        f"nested_bar={'yes' if bar_accept else 'no'} "
        f"v10_iou={v10_b:.10f} v10_winner={v10_w:.10f} "
        f"v11_iou={v11_b:.10f} v11_winner={v11_w:.10f} "
        f"thr={thr['ens_blend']:.10f} sec={sec}"
    )
    append_done(line)
    log(line)
    log(f"записан {out}")


if __name__ == "__main__":
    main()
