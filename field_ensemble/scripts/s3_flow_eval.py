"""Оценка flow-прайора: ансамбль с победителем, 5 фолдов, v10/v11.

Скор потока — эмпирическая CDF |остатка|, узлы только с holdout-фона train.
Ансамбль заранее фиксирован: (4 * ens_u2ft_tcn_lgbm + s_flow) / 5.
Порог — квантиль фона всего v8 val. Test и v10/v11 порог не двигают.
Реальные CSV — только диагностика засветки кабеля, не выбор.

  /workspace/repo/run/.venv/bin/python -u scripts/s3_flow_eval.py
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "NUMEXPR_MAX_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_k] = "2"

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
import sys
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import s3_sel2_cv as s2
import s3_sel_core as sel
import s3_sel_eval as ev
from dts import data, metrics as M
from dts.features import make_channels
from dts.flowprior import VelocityNet, residual_map
from dts.real import load_csv

WINNER = "ens_u2ft_tcn_lgbm"
WINNER_VAL = 0.38671235821914546
WINNER_TEST = 0.4052762065230473
WINNER_V10 = 0.4038977883892501
WINNER_V11 = 0.40390028295846503
NESTED_GAIN = 0.003
TMP = Path("/tmp/s3_flow_maps")
LOG = Path("/workspace/run_logs/_events.log")
FLOW = "flow"
BLEND = "ens_blend"


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S', time.gmtime())} {msg}", flush=True)


def plain(obj):
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def open_map(ver, split, stem, kind, shape, dtype):
    d = TMP / ver / split
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{stem}_{kind}.mmap"
    return np.memmap(path, dtype=dtype, mode="w+", shape=shape)


def read_map(ver, split, stem, kind, shape, dtype):
    path = TMP / ver / split / f"{stem}_{kind}.mmap"
    return np.memmap(path, dtype=dtype, mode="r", shape=shape)


def shapes_of(ver, split) -> dict:
    out = {}
    pred = Path("artifacts/pred") / ver / WINNER / split
    for p in data.list_cache(f"artifacts/cache/{ver}", split):
        with np.load(pred / p.name) as z:
            out[p.stem] = tuple(z["s"].shape)
    return out


def predict_split(model, knots, ver, split, shapes) -> None:
    t0 = time.time()
    files = data.list_cache(f"artifacts/cache/{ver}", split)
    xs = np.linspace(0.0, 1.0, knots.size)
    for i, p in enumerate(files, 1):
        shape = shapes[p.stem]
        with np.load(p) as z:
            X = np.asarray(z["X"], np.float32)
        if X.shape[1:] != shape:
            raise SystemExit(f"{ver} {p.name}: X {X.shape[1:]} != {shape}")
        resid = residual_map(model, X[0], X[2])
        if not np.isfinite(resid).all():
            raise SystemExit(f"non-finite {p.name}")
        s = np.interp(np.abs(resid), knots, xs).astype(np.float32)
        sm = open_map(ver, split, p.stem, "s", shape, np.float32)
        hm = open_map(ver, split, p.stem, "h", shape, np.float16)
        sm[:] = s
        hm[:] = resid.astype(np.float16)
        sm.flush()
        hm.flush()
        del sm, hm, X, resid
        if i == 1 or i % 10 == 0 or i == len(files):
            log(f"pred {ver} {split} {i}/{len(files)} {time.time() - t0:.0f}s")


def load_winner(ver, split, name):
    with np.load(Path("artifacts/pred") / ver / WINNER / split / name) as z:
        return np.asarray(z["s"], np.float32), np.asarray(z["h"], np.float32)


def blend_of(s, ws):
    return ((4.0 * ws + s) / 5.0).astype(np.float32)


def thresholds_of(shapes) -> tuple[dict, int, float]:
    names = (FLOW, BLEND, WINNER)
    parts = {n: [] for n in names}
    raw_parts = []
    n_bg = 0
    for stem, shape in shapes.items():
        with np.load(Path("artifacts/cache/v8/val") / f"{stem}.npz") as meta:
            evm = np.asarray(meta["ev"])
            valid = np.asarray(meta["valid"], bool)
            nu = np.asarray(meta["nu"])
        bg = valid & (evm == 0) & (nu == 0)
        s = np.asarray(read_map("v8", "val", stem, "s", shape, np.float32))
        h = np.asarray(read_map("v8", "val", stem, "h", shape, np.float16), np.float32)
        ws, _wh = load_winner("v8", "val", f"{stem}.npz")
        parts[FLOW].append(s[bg])
        parts[BLEND].append(blend_of(s, ws)[bg])
        parts[WINNER].append(ws[bg])
        raw_parts.append(np.abs(h[bg]))
        n_bg += int(bg.sum())
        del s, h, ws
    thr = {}
    for n in names:
        bg = np.concatenate(parts[n])
        thr[n] = float(M.far_threshold(bg, 1e-3))
        log(f"порог {n} {thr[n]:.6f} n_bg {n_bg}")
        del bg
    raw = np.concatenate(raw_parts)
    raw_thr = float(M.far_threshold(raw, 1e-3))
    log(f"порог |остаток| {raw_thr:.6f}")
    return thr, n_bg, raw_thr


def score_split(ver, split, shapes, thr) -> dict:
    names = (FLOW, BLEND, WINNER)
    tot = {n: np.zeros(3, np.int64) for n in names}
    hot = {n: np.zeros(3, np.int64) for n in names}
    cold = {n: np.zeros(3, np.int64) for n in names}
    far_n = {n: 0 for n in names}
    far_d = {n: 0 for n in names}
    cache = Path("artifacts/cache") / ver / split
    for stem, shape in shapes.items():
        with np.load(cache / f"{stem}.npz") as meta:
            evm = np.asarray(meta["ev"])
            valid = np.asarray(meta["valid"], bool)
            nu = np.asarray(meta["nu"])
        bg = valid & (evm == 0) & (nu == 0)
        truth = evm > 0
        hot_t = (evm >= 1) & (evm <= 4)
        cold_t = evm >= 5
        s = np.asarray(read_map(ver, split, stem, "s", shape, np.float32))
        h = np.asarray(read_map(ver, split, stem, "h", shape, np.float16), np.float32)
        ws, wh = load_winner(ver, split, f"{stem}.npz")
        packed = {
            FLOW: (s, h),
            BLEND: (blend_of(s, ws), h),
            WINNER: (ws, wh),
        }
        for n, (ps, ph) in packed.items():
            pred = (ps >= thr[n]) & valid
            tot[n] += np.asarray(M.confusion(pred, truth, valid), np.int64)
            hot[n] += np.asarray(M.confusion(pred & (ph > 0), hot_t, valid), np.int64)
            cold[n] += np.asarray(M.confusion(pred & (ph <= 0), cold_t, valid), np.int64)
            far_n[n] += int((pred & bg).sum())
            far_d[n] += int(bg.sum())
        del s, h, ws, wh
    report = {}
    for n in names:
        row = dict(threshold=float(thr[n]), far=float(far_n[n] / max(far_d[n], 1)), **M.prf(*tot[n]),
                   hot=M.prf(*hot[n]), cold=M.prf(*cold[n]))
        report[n] = row
        log(f"{ver} {split} {n} IoU {row['iou']:.4f} F1 {row['f1']:.4f} FAR {row['far']:.5f}")
    return report


def separation(shapes) -> dict:
    bg_s, ev_s = [], []
    col_hot = 0
    col_n = 0
    for stem, shape in shapes.items():
        with np.load(Path("artifacts/cache/v8/val") / f"{stem}.npz") as meta:
            evm = np.asarray(meta["ev"])
            valid = np.asarray(meta["valid"], bool)
            nu = np.asarray(meta["nu"])
        h = np.abs(np.asarray(read_map("v8", "val", stem, "h", shape, np.float16), np.float32))
        bg = valid & (evm == 0) & (nu == 0)
        evp = valid & (evm > 0)
        if bg.any():
            bg_s.append(h[bg][:: max(1, int(bg.sum()) // 20000)])
            col = []
            for x in range(h.shape[1]):
                m = bg[:, x]
                if int(m.sum()) < 8:
                    continue
                col.append(float(np.median(h[:, x][m])))
            col = np.asarray(col, np.float32)
            if evp.any() and col.size:
                ev_med = float(np.median(h[evp]))
                col_hot += int((col > 0.5 * ev_med).sum())
                col_n += int(col.size)
        if evp.any():
            ev_s.append(h[evp])
    bg_cat = np.concatenate(bg_s) if bg_s else np.zeros(1)
    ev_cat = np.concatenate(ev_s) if ev_s else np.zeros(1)
    bg_med = float(np.median(bg_cat))
    ev_med = float(np.median(ev_cat))
    return {
        "bg_median_abs": bg_med,
        "event_median_abs": ev_med,
        "sep_event_over_bg": float(ev_med / max(bg_med, 1e-6)),
        "columns_half_event": float(col_hot / max(col_n, 1)),
    }


def frame_of(stem, score, ev_map, valid, nu, x, t, labels, seqs):
    ev_idx, nu_idx, bg_idx, n_bg = ev._indices(ev_map, valid, nu, stem)
    e, n, b = s2._cols(score, ev_idx, nu_idx, bg_idx)
    cooling, _rows = ev._events_for(stem, ev_map, x, t, labels, seqs, ev_idx)
    return dict(stem=stem, ev_m=e[None], nu_m=n[None], bg_m=b[None], n_bg=n_bg, cooling=cooling)


def nested5(shapes, labels, seqs) -> dict:
    files = data.list_cache("artifacts/cache/v8", "val")
    winner_f, blend_f = [], []
    for p in files:
        stem = p.stem
        shape = shapes[stem]
        with np.load(p) as z:
            ev_map = np.asarray(z["ev"])
            valid = np.asarray(z["valid"])
            nu = np.asarray(z["nu"])
            x = np.asarray(z["x"], np.float64)
            t = np.asarray(z["t"], np.float64)
        s = np.asarray(read_map("v8", "val", stem, "s", shape, np.float32))
        ws, _wh = load_winner("v8", "val", f"{stem}.npz")
        blend = blend_of(s, ws)
        winner_f.append(frame_of(stem, ws, ev_map, valid, nu, x, t, labels, seqs))
        blend_f.append(frame_of(stem, blend, ev_map, valid, nu, x, t, labels, seqs))
        del s, ws
    order = np.argsort([f["stem"] for f in winner_f])
    winner_f = [winner_f[i] for i in order]
    blend_f = [blend_f[i] for i in order]
    folds = sel.sequence_folds(len(winner_f), 5)
    parts_w, parts_b, rows = [], [], []
    for r, idx in enumerate(folds):
        hw = s2.eval_one(s2.take(winner_f, idx))
        hb = s2.eval_one(s2.take(blend_f, idx))
        parts_w.append(hw)
        parts_b.append(hb)
        rows.append(dict(fold=r, winner=float(hw["iou"]), blend=float(hb["iou"]),
                         delta=float(hb["iou"] - hw["iou"])))
        log(f"nested k=5 fold {r} winner {hw['iou']:.4f} blend {hb['iou']:.4f} delta {hb['iou'] - hw['iou']:+.4f}")
    iw = float(sel.micro_iou(parts_w))
    ib = float(sel.micro_iou(parts_b))
    return dict(k=5, winner_iou=iw, blend_iou=ib, delta=float(ib - iw), folds=rows,
                procedure="s3_sel2_cv.eval_one holdout threshold, sequence_folds i%5, micro_iou")


def real_cable(model, raw_thr: float) -> dict:
    """Один реальный кабель, несколько окон по времени. Порог не подгоняется."""
    path = Path("/workspace/mgts/MGTS_data/4.csv")
    try:
        rec = load_csv(path)
    except Exception as exc:
        return {"error": str(exc), "used_for_selection": False}
    T = rec["T"]
    valid = rec["valid"]
    n, w = T.shape
    height = 256
    if n <= height:
        starts = [0]
    else:
        starts = [int(v) for v in np.linspace(0, n - height, num=4)]
    fracs = []
    for t0 in starts:
        ch, _meta = make_channels(T[t0:t0 + height], valid[t0:t0 + height])
        resid = residual_map(model, ch[0], ch[2])
        m = ch[2] > 0.5
        if int(m.sum()) == 0:
            continue
        fracs.append(float(np.mean(np.abs(resid[m]) >= raw_thr)))
    return {
        "file": "4.csv",
        "sha256": rec["sha256"],
        "shape": [int(n), int(w)],
        "n_chunks": len(fracs),
        "frac_above_synth_far": float(np.mean(fracs)) if fracs else None,
        "chunks": fracs,
        "used_for_selection": False,
        "real_noise_psd": False,
        "real_zones": False,
    }


def render_md(rep: dict) -> str:
    def row(name, block):
        b = block[name]
        return (f"| {name} | {b['threshold']:.6f} | {b['iou']:.4f} | {b['f1']:.4f} | {b['far']:.5f} |")

    lines = [
        "# S3 Flow",
        "",
        "Одношаговый rectified flow фона (REFLECT: один эйлеров шаг, без VAE и без 1000 шагов DDPM).",
        "Учится только на тайлах синтетики v8 и v10 train, где нет события и nuisance.",
        f"Инференс при t={rep['t_star']}: поле скорости переводит зашумлённый dT к норме.",
        "Скор — CDF модуля остатка (узлы с holdout-фона train). Знак нагрев/охлаждение — знак остатка dT.",
        "Ансамбль фиксирован до val: (4 × ens_u2ft_tcn_lgbm + s_flow) / 5.",
        "Порог FAR 1e-3 — квантиль фона всего v8 val. v10 и v11 этот порог не двигают.",
        "Реальные CSV в обучение и в выбор порога не входят.",
        "",
        f"Решение: **{'принят' if rep['accept'] else 'отклонён'}**.",
        "",
        rep["decision"],
        "",
        "## Обучение",
        "",
        f"Лучшая эпоха {rep['train'].get('best', {}).get('epoch')}, "
        f"прокси {rep['train'].get('best', {}).get('score')}, "
        f"остановка {rep['train'].get('stopped')}, {rep['train'].get('seconds')} с.",
        f"Проба копирования: restore {rep['copy'].get('restore')}, denoise {rep['copy'].get('denoise')}, "
        f"фон |остаток| {rep['copy'].get('bg_abs')}.",
        "restore около 0 и denoise около 0 — сеть копирует вход. "
        "Оба около 1 — всплеск и шум снимаются.",
        "",
        "## Официальный v8",
        "",
        f"Фон {rep['n_bg']} клеток. Пересчёт победителя: val {rep['val'][WINNER]['iou']:.4f}, "
        f"test {rep['test'][WINNER]['iou']:.4f}.",
        "",
        "| кандидат | порог | val IoU | val F1 | val FAR |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in (FLOW, BLEND, WINNER):
        lines.append(row(name, rep["val"]))
    lines += [
        "",
        "| кандидат | test IoU | test F1 | test FAR |",
        "|---|---:|---:|---:|",
    ]
    for name in (FLOW, BLEND, WINNER):
        b = rep["test"][name]
        lines.append(f"| {name} | {b['iou']:.4f} | {b['f1']:.4f} | {b['far']:.5f} |")
    lines += [
        "",
        "## v10 и v11 test, порог v8",
        "",
        "| версия | flow IoU | ансамбль | победитель |",
        "|---|---:|---:|---:|",
        f"| v10 | {rep['v10'][FLOW]['iou']:.4f} | {rep['v10'][BLEND]['iou']:.4f} | {rep['v10'][WINNER]['iou']:.4f} |",
        f"| v11 | {rep['v11'][FLOW]['iou']:.4f} | {rep['v11'][BLEND]['iou']:.4f} | {rep['v11'][WINNER]['iou']:.4f} |",
        "",
        "## Вложенные 5 фолдов",
        "",
        "Та же процедура, что в `scripts/s3_sel2_cv.py`: фолды `i % 5`, порог и IoU на holdout, micro по сумме tp/fp/fn.",
        "",
        "| фолд | победитель | ансамбль | дельта |",
        "|---:|---:|---:|---:|",
    ]
    for r in rep["nested5"]["folds"]:
        lines.append(f"| {r['fold']} | {r['winner']:.4f} | {r['blend']:.4f} | {r['delta']:+.4f} |")
    lines += [
        "",
        f"Micro: ансамбль {rep['nested5']['blend_iou']:.4f}, победитель {rep['nested5']['winner_iou']:.4f}, "
        f"дельта {rep['nested5']['delta']:+.4f}. Порог принятия дельты {NESTED_GAIN}.",
        "",
        "## Риски",
        "",
        f"Разделение на synth val: медиана |остатка| фон {rep['separation']['bg_median_abs']:.4f}, "
        f"событие {rep['separation']['event_median_abs']:.4f}, "
        f"отношение {rep['separation']['sep_event_over_bg']:.3f}.",
        f"Доля столбцов кабеля, где медиана фона > половины медианы события: "
        f"{rep['separation']['columns_half_event']:.4f}.",
        f"Реальный кабель 4.csv, доля клеток выше synth-порога FAR: {rep['real_cable'].get('frac_above_synth_far')}. "
        "Это описание, не выбор модели.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    t0 = time.time()
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    train_meta = json.loads(Path("artifacts/eval/s3_flow_train.json").read_text())
    knots = np.asarray(train_meta["cdf_knots"], np.float64)
    if knots.ndim != 1 or knots.size < 8:
        raise SystemExit("нет узлов CDF")
    model = VelocityNet(w=16)
    model.load_state_dict(torch.load("artifacts/eval/s3_flow_prior.pt", map_location="cpu", weights_only=True))
    model.eval()
    labels = pd.read_csv("data/synth_meta/v8/labels.csv")
    seqs = pd.read_csv("data/synth_meta/v8/sequences.csv").set_index("seq_id")
    if TMP.exists():
        shutil.rmtree(TMP)
    try:
        shapes = shapes_of("v8", "val")
        predict_split(model, knots, "v8", "val", shapes)
        thr, n_bg, raw_thr = thresholds_of(shapes)
        if n_bg != 65095696:
            log(f"внимание n_bg {n_bg}")
        val_rep = score_split("v8", "val", shapes, thr)
        if abs(val_rep[WINNER]["iou"] - WINNER_VAL) > 0.005:
            raise SystemExit(f"победитель val {val_rep[WINNER]['iou']} далёк от {WINNER_VAL}")
        sep = separation(shapes)
        nest = nested5(shapes, labels, seqs)
        log(f"nested5 blend {nest['blend_iou']:.4f} winner {nest['winner_iou']:.4f} delta {nest['delta']:+.4f}")
        shapes_te = shapes_of("v8", "test")
        predict_split(model, knots, "v8", "test", shapes_te)
        test_rep = score_split("v8", "test", shapes_te, thr)
        if abs(test_rep[WINNER]["iou"] - WINNER_TEST) > 0.005:
            raise SystemExit(f"победитель test {test_rep[WINNER]['iou']} далёк от {WINNER_TEST}")
        extra = {}
        for ver, ref in (("v10", WINNER_V10), ("v11", WINNER_V11)):
            sh = shapes_of(ver, "test")
            predict_split(model, knots, ver, "test", sh)
            extra[ver] = score_split(ver, "test", sh, thr)
            if abs(extra[ver][WINNER]["iou"] - ref) > 0.01:
                raise SystemExit(f"победитель {ver} {extra[ver][WINNER]['iou']} далёк от {ref}")
        win_val = float(val_rep[WINNER]["iou"])
        sel_val = float(val_rep[BLEND]["iou"])
        beats_val = sel_val > win_val
        nested_ok = float(nest["delta"]) >= NESTED_GAIN
        v10_ok = float(extra["v10"][BLEND]["iou"]) >= float(extra["v10"][WINNER]["iou"]) - 1e-6
        v11_ok = float(extra["v11"][BLEND]["iou"]) >= float(extra["v11"][WINNER]["iou"]) - 1e-6
        accept = bool(beats_val and nested_ok and v10_ok and v11_ok)
        cable = real_cable(model, raw_thr)
        log(f"real cable frac {cable.get('frac_above_synth_far')}")
        if accept:
            decision = (
                f"Принят. Val ансамбля {sel_val:.4f} выше победителя {win_val:.4f}, "
                f"вложенный прирост {nest['delta']:+.4f}, v10/v11 не хуже."
            )
        else:
            decision = (
                f"Отклонено. Val ансамбля {sel_val:.4f} против победителя {win_val:.4f}. "
                f"Вложенный прирост {nest['delta']:+.4f} (нужно ≥ {NESTED_GAIN}). "
                f"v10 ансамбль {extra['v10'][BLEND]['iou']:.4f} / победитель {extra['v10'][WINNER]['iou']:.4f}, "
                f"v11 {extra['v11'][BLEND]['iou']:.4f} / {extra['v11'][WINNER]['iou']:.4f}. "
                "Победитель не заменяется."
            )
        copy = train_meta.get("final_probe") or train_meta.get("best") or {}
        report = dict(
            name="FLOW",
            t_star=0.5,
            blend="(4*winner+cdf(|residual|))/5",
            accept=accept,
            beats_val=beats_val,
            nested_ok=nested_ok,
            v10_ok=v10_ok,
            v11_ok=v11_ok,
            decision=decision,
            n_bg=n_bg,
            raw_far_threshold=raw_thr,
            val=val_rep,
            test=test_rep,
            v10=extra["v10"],
            v11=extra["v11"],
            nested5=nest,
            separation=sep,
            copy=copy,
            real_cable=cable,
            train=train_meta,
            seconds=round(time.time() - t0, 1),
            real_traces_used_for_train=False,
            real_noise_used=False,
        )
        # train json is large because of knots; don't nest the knots twice
        report["train"] = {k: train_meta[k] for k in train_meta if k != "cdf_knots"}
        out = Path("artifacts/eval/s3_flow_v8.json")
        out.write_text(json.dumps(plain(report), indent=1, ensure_ascii=False))
        Path("docs/S3_FLOW.md").write_text(render_md(report))
        line = (
            f"FLOW_DONE selected=ens_blend val={sel_val:.4f} test={test_rep[BLEND]['iou']:.4f} "
            f"flow_val={val_rep[FLOW]['iou']:.4f} winner_val={win_val:.4f} "
            f"nested5_blend={nest['blend_iou']:.4f} nested5_winner={nest['winner_iou']:.4f} "
            f"delta5={nest['delta']:+.4f} "
            f"v10_blend={extra['v10'][BLEND]['iou']:.4f} v10_winner={extra['v10'][WINNER]['iou']:.4f} "
            f"v11_blend={extra['v11'][BLEND]['iou']:.4f} v11_winner={extra['v11'][WINNER]['iou']:.4f} "
            f"accept={int(accept)} copy_restore={copy.get('restore')} "
            f"cable_frac={cable.get('frac_above_synth_far')} "
            f"best_epoch={train_meta.get('best', {}).get('epoch')} train_sec={train_meta.get('seconds')} "
            f"eval_sec={report['seconds']} real_train=0\n"
        )
        with LOG.open("a", encoding="utf-8") as fh:
            fh.write(line)
        if accept and sel_val > WINNER_VAL + 1e-4:
            with LOG.open("a", encoding="utf-8") as fh:
                fh.write(f"BEATS_WINNER FLOW ens_blend val {sel_val:.4f} test {test_rep[BLEND]['iou']:.4f}\n")
        log(line.strip())
    finally:
        if TMP.exists():
            shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
