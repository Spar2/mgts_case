"""Оценка U-Net с синтетическим априором и смеси ens_blend.

Рецепт заморожен: ens_blend = 0.5 * карта победителя + 0.5 * новая карта.
Порог FAR 1e-3 — квантиль фона (valid & ev==0 & nu==0). Полный v8 val выбирает.
Вложенные фолды — sequence_folds из scripts/s3_sel_core.py (тот же разрез, что
s3_sel2_cv.nested_fixed): порог только по фону holdout, micro IoU по сумме tp/fp/fn.
Карты полные, без подвыборки 200k. Test порог не двигает. Реальные CSV не читаются.
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
NAMES = ("winner", "unet_dg", "ens_blend")
PRED = Path("artifacts/pred/v8")
TMP = Path("/tmp/s3_dg_maps")
LOG_PATH = Path("/workspace/grok_logs/_events.log")


def _sel():
    spec = importlib.util.spec_from_file_location("s3_sel_core", ROOT / "scripts" / "s3_sel_core.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


def tile_starts(n: int, tile: int, step: int) -> list[int]:
    ts = list(range(0, max(n - tile, 0) + 1, step)) or [0]
    if ts[-1] + tile < n:
        ts.append(n - tile)
    return ts


@torch.inference_mode()
def predict_batched(model, X, tile=(256, 512), overlap=32, batch=4):
    """Совпадает с models.predict_full: тайлы, softmax, среднее в перекрытии."""
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


def open_map(split: str, stem: str, key: str, shape, dtype):
    path = TMP / split / f"{stem}_{key}.mmap"
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.memmap(path, dtype=dtype, mode="w+", shape=shape)


def read_map(split: str, stem: str, key: str, shape, dtype):
    path = TMP / split / f"{stem}_{key}.mmap"
    return np.memmap(path, dtype=dtype, mode="r", shape=shape)


def shapes_of(split: str) -> dict[str, tuple[int, int]]:
    out = {}
    for p in data.list_cache("artifacts/cache/v8", split):
        with np.load(PRED / WINNER / split / p.name) as z:
            out[p.stem] = tuple(z["s"].shape)
    return out


def predict_split(model, split: str, shapes: dict[str, tuple[int, int]]) -> None:
    t0 = time.time()
    files = data.list_cache("artifacts/cache/v8", split)
    for i, p in enumerate(files, 1):
        shape = shapes[p.stem]
        with np.load(p) as z:
            X = np.asarray(z["X"], np.float32)
        if X.shape[1:] != shape:
            raise SystemExit(f"{p.name}: X {X.shape[1:]} != pred {shape}")
        prob = predict_batched(model, X)
        s = (1.0 - prob[0]).astype(np.float32)
        h = (prob[1] - prob[2]).astype(np.float16)
        sm = open_map(split, p.stem, "s", shape, np.float32)
        hm = open_map(split, p.stem, "h", shape, np.float16)
        sm[:] = s
        hm[:] = h
        sm.flush()
        hm.flush()
        del sm, hm, X, prob, s, h
        if i == 1 or i % 10 == 0 or i == len(files):
            log(f"pred {split} {i}/{len(files)} {time.time() - t0:.0f}s")


def load_cached(split: str, stem: str, shape):
    s = np.asarray(read_map(split, stem, "s", shape, np.float32))
    h = np.asarray(read_map(split, stem, "h", shape, np.float16), np.float32)
    return s, h


def load_winner(split: str, stem: str, shape):
    with np.load(PRED / WINNER / split / f"{stem}.npz") as z:
        s = np.asarray(z["s"], np.float32)
        h = np.asarray(z["h"], np.float32)
    if s.shape != shape:
        raise SystemExit(f"winner {stem} {s.shape} != {shape}")
    return s, h


def score_maps(split: str, stem: str, shape, name: str):
    if name == "winner":
        return load_winner(split, stem, shape)
    dg_s, dg_h = load_cached(split, stem, shape)
    if name == "unet_dg":
        return dg_s, dg_h
    w_s, w_h = load_winner(split, stem, shape)
    return (0.5 * (w_s + dg_s)).astype(np.float32), (0.5 * (w_h + dg_h)).astype(np.float32)


def meta_of(split: str, stem: str):
    with np.load(Path("artifacts/cache/v8") / split / f"{stem}.npz") as z:
        return (
            np.asarray(z["ev"]),
            np.asarray(z["valid"], bool),
            np.asarray(z["nu"]),
            np.asarray(z["x"]),
            np.asarray(z["t"]),
        )


def threshold_of(split: str, files, shapes, name: str) -> tuple[float, int]:
    parts = []
    for p in files:
        ev, valid, nu, _x, _t = meta_of(split, p.stem)
        s, _h = score_maps(split, p.stem, shapes[p.stem], name)
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
    return dict(
        n=int(len(g)),
        recall=float(g.hit.mean()) if len(g) else None,
        pixel_recall=float(tp / max(tp + fn, 1)),
    )


def evaluate(split: str, files, shapes, thr: dict[str, float], labels) -> dict:
    tot = {n: np.zeros(3, np.int64) for n in thr}
    hot = {n: np.zeros(3, np.int64) for n in thr}
    cold = {n: np.zeros(3, np.int64) for n in thr}
    far_n = {n: 0 for n in thr}
    far_d = {n: 0 for n in thr}
    per = {n: {} for n in thr}
    ev_rows = {n: [] for n in thr}
    for p in files:
        ev, valid, nu, x, t = meta_of(split, p.stem)
        mask = bg_mask(ev, nu, valid)
        truth = ev > 0
        hot_t = (ev >= 1) & (ev <= 4)
        cold_t = ev >= 5
        boxes = []
        for r in M.label_boxes(labels, p.stem).itertuples():
            if r.kind != "event":
                continue
            px = M.event_pixels(r, x, t, ev)
            if px.any():
                boxes.append((r, px))
        for n in thr:
            s, h = score_maps(split, p.stem, shapes[p.stem], n)
            pred = (s >= thr[n]) & valid
            cm = np.asarray(M.confusion(pred, truth, valid), np.int64)
            tot[n] += cm
            per[n][p.stem] = tuple(int(v) for v in cm)
            hot[n] += np.asarray(M.confusion(pred & (h > 0), hot_t, valid), np.int64)
            cold[n] += np.asarray(M.confusion(pred & (h <= 0), cold_t, valid), np.int64)
            far_n[n] += int((pred & mask).sum())
            far_d[n] += int(mask.sum())
            for r, px in boxes:
                frac = float(pred[px].mean())
                ev_rows[n].append(dict(
                    amp=float(r.amp_obs_C),
                    amp_bin=M.amp_bin(float(r.amp_obs_C)),
                    label=str(r.label),
                    frac=frac,
                    hit=frac >= 0.3,
                    tp=int((pred & px).sum()),
                    fn=int((~pred & px).sum()),
                ))
            del s, h, pred
        del ev, valid, nu
    report = {}
    for n in thr:
        E = pd.DataFrame(ev_rows[n])
        row = dict(
            threshold=float(thr[n]),
            far=float(far_n[n] / max(far_d[n], 1)),
            n_bg=int(far_d[n]),
            **M.prf(*tot[n]),
            tp=int(tot[n][0]), fp=int(tot[n][1]), fn=int(tot[n][2]),
            hot=M.prf(*hot[n]),
            cold=M.prf(*cold[n]),
            iou_ci95=M.bootstrap_ci(per[n], sorted(per[n]), n=200, seed=0),
        )
        if len(E):
            row["events_all"] = _agg(E)
            row["weak_events"] = _agg(E[E.amp <= 2]) if (E.amp <= 2).any() else dict(n=0, recall=None, pixel_recall=None)
            row["by_amp"] = {k: _agg(g) for k, g in E.groupby("amp_bin")}
            row["by_label"] = {k: _agg(g) for k, g in E.groupby("label")}
        report[n] = row
        weak = (row.get("by_amp") or {}).get("<=2C", {})
        log(f"{split} {n} IoU {row['iou']:.6f} FAR {row['far']:.6f} thr {row['threshold']:.6f} weak {weak.get('recall')}")
    return report


def counts_on(split, files, shapes, name, thr) -> dict:
    tot = np.zeros(3, np.int64)
    for p in files:
        ev, valid, _nu, _x, _t = meta_of(split, p.stem)
        s, _h = score_maps(split, p.stem, shapes[p.stem], name)
        pred = (s >= thr) & valid
        tot += np.asarray(M.confusion(pred, ev > 0, valid), np.int64)
        del s, pred
    tp, fp, fn = (int(v) for v in tot)
    return dict(tp=tp, fp=fp, fn=fn, **M.prf(tp, fp, fn), threshold=float(thr), n_seq=len(files))


def nested(split, files, shapes, sel) -> dict:
    """k=2 и k=5. Порог каждого фолда — только его фон. Сравнение с победителем на тех же фолдах."""
    files = list(files)
    out = {}
    for k in (2, 5):
        folds = sel.sequence_folds(len(files), k)
        parts = {n: [] for n in NAMES}
        details = []
        for r, hold_idx in enumerate(folds):
            hold = [files[i] for i in hold_idx]
            fold_row = dict(fold=r, n_seq=len(hold), stems=[p.stem for p in hold])
            for n in NAMES:
                thr, n_bg = threshold_of(split, hold, shapes, n)
                got = counts_on(split, hold, shapes, n, thr)
                got["n_bg"] = n_bg
                parts[n].append(got)
                fold_row[n] = dict(iou=got["iou"], tp=got["tp"], fp=got["fp"], fn=got["fn"], threshold=got["threshold"])
            log(
                f"nested k={k} fold {r} blend {fold_row['ens_blend']['iou']:.4f} "
                f"winner {fold_row['winner']['iou']:.4f} unet {fold_row['unet_dg']['iou']:.4f}"
            )
            details.append(fold_row)
        micros = {n: float(sel.micro_iou(parts[n])) for n in NAMES}
        out[str(k)] = dict(micro_iou=micros, folds=details)
        log(f"nested k={k} micro blend {micros['ens_blend']:.6f} winner {micros['winner']:.6f}")
    blend_by_k = {k: v["micro_iou"]["ens_blend"] for k, v in out.items()}
    bar = sel.nested_verdict(blend_by_k)
    n5_b = out["5"]["micro_iou"]["ens_blend"]
    n5_w = out["5"]["micro_iou"]["winner"]
    n2_b = out["2"]["micro_iou"]["ens_blend"]
    n2_w = out["2"]["micro_iou"]["winner"]
    vs = bool(n5_b > n5_w and n2_b > n2_w)
    return dict(
        by_k=out,
        nested_bar=bar,
        nested_vs_winner=vs,
        rule=(
            "Фолды i%k по отсортированным последовательностям v8 val, как sequence_folds в s3_sel_core "
            "(его вызывает s3_sel2_cv.nested_fixed). Порог FAR 1e-3 только по фону holdout. "
            "micro IoU — сумма tp/fp/fn. Карты полные. "
            "nested_vs_winner: micro смеси строго выше micro победителя и при k=5, и при k=2. "
            "nested_bar — правило скрипта: min(k=2, k=5) строго выше 0.3867+0.003. "
            "beats смотрит полный val и отдельно micro k=5 против победителя."
        ),
    )


def weak_recall(row) -> float | None:
    block = (row.get("by_amp") or {}).get("<=2C")
    if not block or block.get("recall") is None:
        return None
    return float(block["recall"])


def append_done(line: str) -> None:
    text = LOG_PATH.read_text(encoding="utf-8") if LOG_PATH.exists() else ""
    if "\nDG_DONE " in ("\n" + text):
        raise SystemExit("DG_DONE уже есть в _events.log, вторую строку не пишу")
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="artifacts/eval/s3_dg_unet.pt")
    ap.add_argument("--skip-predict", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    sel = _sel()
    t0 = time.time()
    labels = pd.read_csv("data/synth_meta/v8/labels.csv")
    model = models.build("unet", 16)
    model.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=True))
    model.eval()
    err = check_predict(model)
    log(f"сверка predict max|Δ| {err:.3e}")
    val_shapes = shapes_of("val")
    test_shapes = shapes_of("test")
    val_files = data.list_cache("artifacts/cache/v8", "val")
    test_files = data.list_cache("artifacts/cache/v8", "test")
    log(f"val {len(val_files)} test {len(test_files)}")
    try:
        if not a.skip_predict:
            if TMP.exists():
                shutil.rmtree(TMP)
            predict_split(model, "val", val_shapes)
            predict_split(model, "test", test_shapes)
        thr = {}
        for n in NAMES:
            thr[n], n_bg = threshold_of("val", val_files, val_shapes, n)
            log(f"порог val {n} {thr[n]:.6f} n_bg {n_bg}")
        if abs(thr["winner"] - WINNER_THR) > 1e-6:
            log(f"порог победителя {thr['winner']} не совпал с опубликованным {WINNER_THR}")
        val_rep = evaluate("val", val_files, val_shapes, thr, labels)
        nest = nested("val", val_files, val_shapes, sel)
        log("метрики test, порог заморожен по полному val")
        test_rep = evaluate("test", test_files, test_shapes, thr, labels)
    finally:
        if TMP.exists():
            shutil.rmtree(TMP, ignore_errors=True)

    win_val = float(val_rep["winner"]["iou"])
    blend_val = float(val_rep["ens_blend"]["iou"])
    unet_val = float(val_rep["unet_dg"]["iou"])
    win_test = float(test_rep["winner"]["iou"])
    blend_test = float(test_rep["ens_blend"]["iou"])
    protocol_ok = abs(win_val - WINNER_VAL) <= 1e-4 and abs(win_test - WINNER_TEST) <= 1e-4
    if not protocol_ok:
        log(f"протокол не сошёлся: winner val {win_val} test {win_test}")
    n5_b = float(nest["by_k"]["5"]["micro_iou"]["ens_blend"])
    n5_w = float(nest["by_k"]["5"]["micro_iou"]["winner"])
    n2_b = float(nest["by_k"]["2"]["micro_iou"]["ens_blend"])
    n2_w = float(nest["by_k"]["2"]["micro_iou"]["winner"])
    nested_accept = bool(n5_b > n5_w and n2_b > n2_w)
    beats = bool(protocol_ok and blend_val > WINNER_VAL and n5_b > n5_w)
    train_meta = {}
    tp = Path("artifacts/eval/s3_dg_train.json")
    if tp.exists():
        train_meta = json.loads(tp.read_text(encoding="utf-8"))
    sec = round(float(train_meta.get("seconds") or 0) + (time.time() - t0), 1)
    report = dict(
        task="DG",
        protocol=(
            "Кандидат один: ens_blend = 0.5 * ens_u2ft_tcn_lgbm + 0.5 * unet_dg. "
            "Alpha не подбиралась на val. Официальная цифра — pixel IoU при FAR 1e-3 на всём v8 val, "
            "порог по фону этого val. Test тем же порогом, в выбор не входит. "
            "beats, только если val IoU строго выше 0.38671235821914546 и micro IoU 5-fold смеси "
            "строго выше micro победителя на тех же фолдах. Вложенная проверка ещё требует то же для k=2 "
            "(nested_accept). Планка скрипта 0.3897 пишется в nested.nested_bar и сама beats не включает. "
            "Строка BEATS_WINNER не пишется."
        ),
        published_winner=dict(name=WINNER, val_iou=WINNER_VAL, test_iou=WINNER_TEST, thr=WINNER_THR),
        protocol_ok=protocol_ok,
        winner_val_abs_err=abs(win_val - WINNER_VAL),
        winner_test_abs_err=abs(win_test - WINNER_TEST),
        predict_max_abs_err=err,
        alpha=0.5,
        chosen="ens_blend",
        beats=beats,
        nested_accept=nested_accept,
        thresholds=thr,
        val=val_rep,
        test=test_rep,
        nested=nest,
        train=dict(
            best_epoch=train_meta.get("best_epoch"),
            best_proxy_iou=train_meta.get("best_proxy_iou"),
            stopped=train_meta.get("stopped"),
            seconds=train_meta.get("seconds"),
            init=train_meta.get("init"),
            prior=train_meta.get("prior"),
            real_traces_used=False,
        ),
        real_traces_used=False,
        replaced_release=False,
        cards="not used for selection",
        seconds=sec,
    )
    out = Path("artifacts/eval/s3_dg_v8.json")
    out.write_text(json.dumps(plain(report), indent=1, ensure_ascii=False), encoding="utf-8")
    vw = weak_recall(val_rep["ens_blend"])
    tw = weak_recall(test_rep["ens_blend"])
    line = (
        f"DG_DONE chosen=ens_blend beats={'yes' if beats else 'no'} "
        f"val_iou={blend_val:.10f} val_iou_winner={win_val:.10f} "
        f"val_weak={vw} "
        f"test_iou={blend_test:.10f} test_iou_winner={win_test:.10f} test_weak={tw} "
        f"unet_dg_val={unet_val:.10f} "
        f"nested5_blend={n5_b:.10f} nested5_winner={n5_w:.10f} "
        f"nested2_blend={n2_b:.10f} nested2_winner={n2_w:.10f} "
        f"nested_accept={'yes' if nested_accept else 'no'} "
        f"nested_bar={'yes' if nest['nested_bar']['accept'] else 'no'} "
        f"thr={thr['ens_blend']:.10f} sec={sec}"
    )
    append_done(line)
    log(line)
    log(f"записан {out}")


if __name__ == "__main__":
    main()
