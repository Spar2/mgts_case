"""Оценка контура: ансамбль с победителем, 5 фолдов, v10/v11.

Скор — уверенность запроса внутри растеризованного многоугольника, снаружи 0.
Ансамбль заранее фиксирован: (4 * ens_u2ft_tcn_lgbm + s_contour) / 5.
Порог — квантиль фона всего v8 val. Test и v10/v11 порог не двигают.
Реальные CSV не читаются.

  /workspace/repo/run/.venv/bin/python -u scripts/s3_contour_eval.py
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
from dts.contour import TILE, ContourNet, score_field

WINNER = "ens_u2ft_tcn_lgbm"
WINNER_VAL = 0.38671235821914546
WINNER_TEST = 0.4052762065230473
WINNER_V10 = 0.4038977883892501
WINNER_V11 = 0.40390028295846503
NESTED_GAIN = 0.003
TMP = Path("/tmp/s3_contour_maps")
LOG = Path("/workspace/grok_logs/_events.log")
FLOW = "contour"
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


def predict_split(model, ver, split, shapes) -> None:
    t0 = time.time()
    files = data.list_cache(f"artifacts/cache/{ver}", split)
    for i, p in enumerate(files, 1):
        shape = shapes[p.stem]
        with np.load(p) as z:
            X = np.asarray(z["X"], np.float32)
        if X.shape[1:] != shape:
            raise SystemExit(f"{ver} {p.name}: X {X.shape[1:]} != {shape}")
        s, h = score_field(model, X, tile=TILE, batch=4)
        if not np.isfinite(s).all():
            raise SystemExit(f"non-finite {p.name}")
        sm = open_map(ver, split, p.stem, "s", shape, np.float32)
        hm = open_map(ver, split, p.stem, "h", shape, np.float16)
        sm[:] = s
        hm[:] = h.astype(np.float16)
        sm.flush()
        hm.flush()
        del sm, hm, X, s, h
        if i == 1 or i % 10 == 0 or i == len(files):
            log(f"pred {ver} {split} {i}/{len(files)} {time.time() - t0:.0f}s")


def load_winner(ver, split, name):
    with np.load(Path("artifacts/pred") / ver / WINNER / split / name) as z:
        return np.asarray(z["s"], np.float32), np.asarray(z["h"], np.float32)


def blend_of(s, ws):
    return ((4.0 * ws + s) / 5.0).astype(np.float32)


def thresholds_of(shapes) -> tuple[dict, int, dict]:
    names = (FLOW, BLEND, WINNER)
    parts = {n: [] for n in names}
    n_bg = 0
    pos = 0
    pos_bg = 0
    valid_n = 0
    for stem, shape in shapes.items():
        with np.load(Path("artifacts/cache/v8/val") / f"{stem}.npz") as meta:
            evm = np.asarray(meta["ev"])
            valid = np.asarray(meta["valid"], bool)
            nu = np.asarray(meta["nu"])
        bg = valid & (evm == 0) & (nu == 0)
        s = np.asarray(read_map("v8", "val", stem, "s", shape, np.float32))
        ws, _wh = load_winner("v8", "val", f"{stem}.npz")
        parts[FLOW].append(s[bg])
        parts[BLEND].append(blend_of(s, ws)[bg])
        parts[WINNER].append(ws[bg])
        pos += int((s[valid] > 0).sum())
        pos_bg += int((s[bg] > 0).sum())
        valid_n += int(valid.sum())
        n_bg += int(bg.sum())
        del s, ws
    thr = {}
    for n in names:
        bg = np.concatenate(parts[n])
        thr[n] = float(M.far_threshold(bg, 1e-3))
        log(f"порог {n} {thr[n]:.6f} n_bg {n_bg}")
        del bg
    sparse = {
        "frac_positive_valid": float(pos / max(valid_n, 1)),
        "frac_positive_bg": float(pos_bg / max(n_bg, 1)),
        "n_valid": int(valid_n),
        "n_bg": int(n_bg),
    }
    log(f"доля клеток contour>0 valid {sparse['frac_positive_valid']:.6f} фон {sparse['frac_positive_bg']:.6f}")
    return thr, n_bg, sparse


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


def event_recall(shapes, thr_contour, thr_blend) -> dict:
    """Доля событий v8 val, у которых ≥10% пикселей выше порога. Не двигает выбор."""
    labels = pd.read_csv("data/synth_meta/v8/labels.csv")
    counts = {k: {"n": 0, "hit": 0, "n_weak": 0, "hit_weak": 0} for k in (FLOW, BLEND)}
    thr = {FLOW: thr_contour, BLEND: thr_blend}
    for stem, shape in shapes.items():
        with np.load(Path("artifacts/cache/v8/val") / f"{stem}.npz") as z:
            ev_map = np.asarray(z["ev"])
            x = np.asarray(z["x"], np.float64)
            t = np.asarray(z["t"], np.float64)
        s = np.asarray(read_map("v8", "val", stem, "s", shape, np.float32))
        ws, _wh = load_winner("v8", "val", f"{stem}.npz")
        maps = {FLOW: s, BLEND: blend_of(s, ws)}
        rows = M.label_boxes(labels, stem)
        for r in rows.itertuples():
            if r.kind != "event":
                continue
            px = M.event_pixels(r, x, t, ev_map)
            if not px.any():
                continue
            weak = float(r.amp_obs_C) <= 2.0
            for name, mp in maps.items():
                hit = float((mp[px] >= thr[name]).mean()) >= 0.1
                counts[name]["n"] += 1
                counts[name]["hit"] += int(hit)
                counts[name]["n_weak"] += int(weak)
                counts[name]["hit_weak"] += int(hit and weak)
        del s, ws
    out = {}
    for name, c in counts.items():
        out[name] = dict(
            n=c["n"],
            recall=float(c["hit"] / max(c["n"], 1)),
            n_weak=c["n_weak"],
            recall_le2c=float(c["hit_weak"] / max(c["n_weak"], 1)),
            hit_frac=0.1,
            used_for_selection=False,
        )
        log(f"события {name} recall {out[name]['recall']:.3f} ≤2C {out[name]['recall_le2c']:.3f} n {c['n']}")
    return out


def render_md(rep: dict) -> str:
    def row(name, block):
        b = block[name]
        return f"| {name} | {b['threshold']:.6f} | {b['iou']:.4f} | {b['f1']:.4f} | {b['far']:.5f} |"

    lines = [
        "# S3 Contour",
        "",
        "Контур экземпляра: не больше 16 запросов и не больше 8 вершин, один шаг уточнения.",
        "Энкодер — e1 и e2 `artifacts/stage1/unet_v810ft.pt` (U-Net из ансамбля победителя), веса заморожены.",
        "Голова учится только на синтетике v8 и v10 train. Веса COCO и ImageNet не загружаются.",
        "Нулевые смещения вершин — восьмиугольник рамки. Уточнение сдвигает середины сторон.",
        "Скор клетки — уверенность запроса внутри многоугольника и 0 снаружи.",
        "Знак нагрев/охлаждение — отдельная голова запроса, не форма контура.",
        "Ансамбль фиксирован до val: (4 × ens_u2ft_tcn_lgbm + s_contour) / 5.",
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
        f"Энкодер max|Δw| {rep['train'].get('encoder_max_dw')}.",
        f"Прокси на holdout train: IoU {rep['copy'].get('iou')}, пустые тайлы {rep['copy'].get('empty')}.",
        "",
        "## Официальный v8",
        "",
        f"Фон {rep['n_bg']} клеток. Пересчёт победителя: val {rep['val'][WINNER]['iou']:.4f}, "
        f"test {rep['test'][WINNER]['iou']:.4f}.",
        f"Доля клеток контура строго выше нуля: valid {rep['sparse']['frac_positive_valid']:.4f}, "
        f"фон {rep['sparse']['frac_positive_bg']:.4f}.",
        "Если ложных закрасок меньше 10⁻³ фона, квантиль садится низко и в зачёт идёт почти вся заливка.",
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
        "| версия | contour IoU | ансамбль | победитель |",
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
    evr = rep["events"]
    lines += [
        "",
        f"Micro: ансамбль {rep['nested5']['blend_iou']:.4f}, победитель {rep['nested5']['winner_iou']:.4f}, "
        f"дельта {rep['nested5']['delta']:+.4f}. Порог принятия дельты {NESTED_GAIN}.",
        "",
        "## События v8 val",
        "",
        "Попадание: не меньше 10% пикселей события выше порога. Это описание, не выбор модели.",
        "",
        f"Контур: recall {evr[FLOW]['recall']:.3f}, ≤2°C {evr[FLOW]['recall_le2c']:.3f} "
        f"(n {evr[FLOW]['n']}, слабых {evr[FLOW]['n_weak']}).",
        f"Ансамбль: recall {evr[BLEND]['recall']:.3f}, ≤2°C {evr[BLEND]['recall_le2c']:.3f}.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    if LOG.exists() and "CONTOUR_DONE" in LOG.read_text(encoding="utf-8"):
        raise SystemExit("CONTOUR_DONE уже записан")
    t0 = time.time()
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    import cv2
    cv2.setNumThreads(2)
    train_meta = json.loads(Path("artifacts/eval/s3_contour_train.json").read_text())
    model = ContourNet()
    model.encoder.load_winner()
    model.load_state_dict(torch.load("artifacts/eval/s3_contour_prior.pt", map_location="cpu", weights_only=True))
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    model.eval()
    labels = pd.read_csv("data/synth_meta/v8/labels.csv")
    seqs = pd.read_csv("data/synth_meta/v8/sequences.csv").set_index("seq_id")
    if TMP.exists():
        shutil.rmtree(TMP)
    try:
        shapes = shapes_of("v8", "val")
        predict_split(model, "v8", "val", shapes)
        thr, n_bg, sparse = thresholds_of(shapes)
        if n_bg != 65095696:
            log(f"внимание n_bg {n_bg}")
        val_rep = score_split("v8", "val", shapes, thr)
        if abs(val_rep[WINNER]["iou"] - WINNER_VAL) > 0.005:
            raise SystemExit(f"победитель val {val_rep[WINNER]['iou']} далёк от {WINNER_VAL}")
        nest = nested5(shapes, labels, seqs)
        log(f"nested5 blend {nest['blend_iou']:.4f} winner {nest['winner_iou']:.4f} delta {nest['delta']:+.4f}")
        events = event_recall(shapes, thr[FLOW], thr[BLEND])
        shapes_te = shapes_of("v8", "test")
        predict_split(model, "v8", "test", shapes_te)
        test_rep = score_split("v8", "test", shapes_te, thr)
        if abs(test_rep[WINNER]["iou"] - WINNER_TEST) > 0.005:
            raise SystemExit(f"победитель test {test_rep[WINNER]['iou']} далёк от {WINNER_TEST}")
        extra = {}
        for ver, ref in (("v10", WINNER_V10), ("v11", WINNER_V11)):
            sh = shapes_of(ver, "test")
            predict_split(model, ver, "test", sh)
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
            name="CONTOUR",
            tile=list(TILE),
            n_queries=16,
            n_verts=8,
            blend="(4*winner+conf_inside_polygon)/5",
            accept=accept,
            beats_val=beats_val,
            nested_ok=nested_ok,
            v10_ok=v10_ok,
            v11_ok=v11_ok,
            decision=decision,
            n_bg=n_bg,
            sparse=sparse,
            val=val_rep,
            test=test_rep,
            v10=extra["v10"],
            v11=extra["v11"],
            nested5=nest,
            events=events,
            copy=copy,
            train={k: train_meta[k] for k in train_meta if k != "history"},
            seconds=round(time.time() - t0, 1),
            real_traces_used_for_train=False,
            real_noise_used=False,
        )
        out = Path("artifacts/eval/s3_contour_v8.json")
        out.write_text(json.dumps(plain(report), indent=1, ensure_ascii=False))
        Path("docs/S3_CONTOUR.md").write_text(render_md(report))
        line = (
            f"CONTOUR_DONE selected=ens_blend val={sel_val:.4f} test={test_rep[BLEND]['iou']:.4f} "
            f"contour_val={val_rep[FLOW]['iou']:.4f} winner_val={win_val:.4f} "
            f"nested5_blend={nest['blend_iou']:.4f} nested5_winner={nest['winner_iou']:.4f} "
            f"delta5={nest['delta']:+.4f} "
            f"v10_blend={extra['v10'][BLEND]['iou']:.4f} v10_winner={extra['v10'][WINNER]['iou']:.4f} "
            f"v11_blend={extra['v11'][BLEND]['iou']:.4f} v11_winner={extra['v11'][WINNER]['iou']:.4f} "
            f"accept={int(accept)} frac_pos={sparse['frac_positive_bg']:.6f} "
            f"event_recall={events[FLOW]['recall']:.4f} weak_recall={events[FLOW]['recall_le2c']:.4f} "
            f"best_epoch={train_meta.get('best', {}).get('epoch')} train_sec={train_meta.get('seconds')} "
            f"eval_sec={report['seconds']} real_train=0\n"
        )
        with LOG.open("a", encoding="utf-8") as fh:
            fh.write(line)
        if accept and sel_val > WINNER_VAL + 1e-4:
            with LOG.open("a", encoding="utf-8") as fh:
                fh.write(f"BEATS_WINNER CONTOUR ens_blend val {sel_val:.4f} test {test_rep[BLEND]['iou']:.4f}\n")
        log(line.strip())
    finally:
        if TMP.exists():
            shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
