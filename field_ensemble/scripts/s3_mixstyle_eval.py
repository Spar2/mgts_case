"""Оценка MixStyle U-Net: val IoU@FAR 1e-3 и вложенные 5 фолдов.

Кандидат — среднее с сохранённой картой ens_u2ft_tcn_lgbm.
Порог официальной точки — квантиль фона всего v8 val.
Вложенная проверка идёт через scripts/s3_sel2_cv.py: те же фолды i%5,
порог и IoU на holdout, micro по сумме tp/fp/fn.
Прирост к победителю под этой процедурой должен быть не меньше 0.003.
Test в выбор не входит. Реальные CSV не читаются.

  /workspace/repo/run/.venv/bin/python -u scripts/s3_mixstyle_eval.py
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_k] = "2"

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
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)

import s3_sel2_cv as s2
import s3_sel_core as sel
import s3_sel_eval as ev
from dts import data, metrics as M, models
from dts.mixstyle import MixStyleUNet

WINNER = "ens_u2ft_tcn_lgbm"
WINNER_VAL = 0.38671235821914546
WINNER_TEST = 0.4052762065230473
NESTED_GAIN = 0.003
TMP = Path("/tmp/s3_mixstyle_maps")
LOG = Path("/workspace/run_logs/_events.log")


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


def _predict():
    spec = importlib.util.spec_from_file_location("s3_mixstyle_train", ROOT / "scripts" / "s3_mixstyle_train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.predict_batched


def open_map(split, stem, key, shape, dtype):
    path = TMP / split / f"{stem}_{key}.mmap"
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.memmap(path, dtype=dtype, mode="w+", shape=shape)


def read_map(split, stem, key, shape, dtype):
    path = TMP / split / f"{stem}_{key}.mmap"
    return np.memmap(path, dtype=dtype, mode="r", shape=shape)


def shapes_of(split):
    out = {}
    for p in data.list_cache("artifacts/cache/v8", split):
        with np.load(Path("artifacts/pred/v8") / WINNER / split / p.name) as z:
            out[p.stem] = tuple(z["s"].shape)
    return out


def predict_split(model, predict_batched, split, shapes):
    t0 = time.time()
    files = data.list_cache("artifacts/cache/v8", split)
    for i, p in enumerate(files, 1):
        shape = shapes[p.stem]
        with np.load(p) as z:
            X = np.asarray(z["X"], np.float32)
        if X.shape[1:] != shape:
            raise SystemExit(f"{p.name}: X {X.shape[1:]} != pred {shape}")
        prob = predict_batched(model, X)
        if not np.isfinite(prob).all():
            raise SystemExit(f"non-finite {p.name}")
        s = (1.0 - prob[0]).astype(np.float32)
        h = (prob[1] - prob[2]).astype(np.float16)
        sm = open_map(split, p.stem, "s", shape, np.float32)
        hm = open_map(split, p.stem, "h", shape, np.float16)
        sm[:] = s
        hm[:] = h
        sm.flush()
        hm.flush()
        del sm, hm, X, prob
        if i == 1 or i % 10 == 0 or i == len(files):
            log(f"pred {split} {i}/{len(files)} {time.time() - t0:.0f}s")


def load_winner(split, name):
    with np.load(Path("artifacts/pred/v8") / WINNER / split / name) as z:
        return np.asarray(z["s"], np.float32), np.asarray(z["h"], np.float32)


def score_split(split, shapes, thr):
    """Полный фон, не компактный. thr задан снаружи (с val)."""
    names = ("unet_mix", "ens_blend", WINNER)
    tot = {n: np.zeros(3, np.int64) for n in names}
    hot = {n: np.zeros(3, np.int64) for n in names}
    cold = {n: np.zeros(3, np.int64) for n in names}
    far_n = {n: 0 for n in names}
    far_d = {n: 0 for n in names}
    cache = Path("artifacts/cache/v8") / split
    for stem, shape in shapes.items():
        with np.load(cache / f"{stem}.npz") as meta:
            ev = np.asarray(meta["ev"])
            valid = np.asarray(meta["valid"], bool)
            nu = np.asarray(meta["nu"])
        bg = valid & (ev == 0) & (nu == 0)
        truth = ev > 0
        hot_t = (ev >= 1) & (ev <= 4)
        cold_t = ev >= 5
        s = np.asarray(read_map(split, stem, "s", shape, np.float32))
        h = np.asarray(read_map(split, stem, "h", shape, np.float16), np.float32)
        ws, wh = load_winner(split, f"{stem}.npz")
        packed = {
            "unet_mix": (s, h),
            "ens_blend": ((0.5 * (s + ws)).astype(np.float32), (0.5 * (h + wh)).astype(np.float32)),
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
        log(f"{split} {n} IoU {row['iou']:.4f} F1 {row['f1']:.4f} FAR {row['far']:.5f}")
    return report


def thresholds_of(shapes):
    names = ("unet_mix", "ens_blend", WINNER)
    parts = {n: [] for n in names}
    n_bg = 0
    for stem, shape in shapes.items():
        with np.load(Path("artifacts/cache/v8/val") / f"{stem}.npz") as meta:
            ev = np.asarray(meta["ev"])
            valid = np.asarray(meta["valid"], bool)
            nu = np.asarray(meta["nu"])
        bg = valid & (ev == 0) & (nu == 0)
        s = np.asarray(read_map("val", stem, "s", shape, np.float32))
        ws, _wh = load_winner("val", f"{stem}.npz")
        parts["unet_mix"].append(s[bg])
        parts["ens_blend"].append((0.5 * (s + ws))[bg])
        parts[WINNER].append(ws[bg])
        n_bg += int(bg.sum())
        del s, ws
    thr = {}
    for n in names:
        bg = np.concatenate(parts[n])
        thr[n] = float(M.far_threshold(bg, 1e-3))
        log(f"порог {n} {thr[n]:.6f} n_bg {n_bg}")
        del bg
    return thr, n_bg


def compact_val(shapes, labels, seqs):
    """Компактные кадры в формате s3_sel2_cv: один канал, фон до 200k."""
    winner, blend, alone = [], [], []
    files = data.list_cache("artifacts/cache/v8", "val")
    for p in files:
        stem = p.stem
        shape = shapes[stem]
        with np.load(p) as z:
            ev_map = np.asarray(z["ev"])
            valid = np.asarray(z["valid"])
            nu = np.asarray(z["nu"])
            x = np.asarray(z["x"], np.float64)
            t = np.asarray(z["t"], np.float64)
        ev_idx, nu_idx, bg_idx, n_bg = ev._indices(ev_map, valid, nu, stem)
        s = np.asarray(read_map("val", stem, "s", shape, np.float32))
        ws, _wh = load_winner("val", f"{stem}.npz")
        blend_s = (0.5 * (s + ws)).astype(np.float32)
        cooling, _rows = ev._events_for(stem, ev_map, x, t, labels, seqs, ev_idx)

        def frame(arr):
            e, n, b = s2._cols(arr, ev_idx, nu_idx, bg_idx)
            return dict(ev_m=e[None], nu_m=n[None], bg_m=b[None], n_bg=n_bg, cooling=cooling, stem=stem)

        winner.append(frame(ws))
        blend.append(frame(blend_s))
        alone.append(frame(s))
        del s, ws, blend_s
    order = np.argsort([fr["stem"] for fr in winner])
    def reorder(rows):
        return [rows[i] for i in order]
    return reorder(winner), reorder(blend), reorder(alone)


def nested_k(winner, blend, k=5):
    """Та же процедура, что s3_sel2_cv.nested_fixed: фолды i%k, порог на holdout, micro IoU."""
    folds = sel.sequence_folds(len(winner), k)
    parts_w, parts_b, details = [], [], []
    for r, hold_idx in enumerate(folds):
        mw = s2.eval_one(s2.take(winner, hold_idx))
        mb = s2.eval_one(s2.take(blend, hold_idx))
        parts_w.append(mw)
        parts_b.append(mb)
        details.append(dict(
            fold=r, n_seq=len(hold_idx),
            stems=[winner[i]["stem"] for i in hold_idx],
            winner=dict(iou=float(mw["iou"]), tp=float(mw["tp"]), fp=float(mw["fp"]), fn=float(mw["fn"])),
            blend=dict(iou=float(mb["iou"]), tp=float(mb["tp"]), fp=float(mb["fp"]), fn=float(mb["fn"])),
            delta=float(mb["iou"] - mw["iou"]),
        ))
        log(f"nested k={k} fold {r} winner {mw['iou']:.4f} blend {mb['iou']:.4f} delta {mb['iou'] - mw['iou']:+.4f}")
    iw = float(sel.micro_iou(parts_w))
    ib = float(sel.micro_iou(parts_b))
    return dict(k=k, winner_iou=iw, blend_iou=ib, delta=float(ib - iw), folds=details)


def render_md(report) -> str:
    v, t = report["val"], report["test"]
    nest = report["nested5"]
    acc = "принят" if report["accept"] else "отклонён"
    lines = [
        "# S3 MixStyle",
        "",
        "Нейросетевой детектор. MixStyle (Zhou et al., ICLR 2021) стоит в энкодере U-Net после блоков e1 и e2.",
        "Вероятность применения 0.5, Beta(0.1, 0.1). На eval модуль — тождество.",
        "Старт — `artifacts/stage1/unet_v810ft.pt`. Обучение только на синтетике v8 и v10 train.",
        "Реальные CSV, спектр фона и σ реального шума не читались.",
        "",
        "Порог официальной точки — квантиль фона всего v8 val при FAR 1e-3. Test в выбор не входит.",
        "Кандидат — среднее карты MixStyle и сохранённого `ens_u2ft_tcn_lgbm`.",
        "Вложенная проверка — 5 фолдов через `scripts/s3_sel2_cv.py`: последовательности v8 val по имени,",
        "фолд `i % 5`, порог FAR 1e-3 и IoU на holdout, micro по сумме tp/fp/fn.",
        "Принятие: официальный val IoU ансамбля строго выше победителя и прирост micro на 5 фолдах не меньше 0.003",
        "против победителя под той же процедурой.",
        "",
        f"Решение: **{acc}**.",
        "",
        report["decision"],
        "",
        "## Обучение",
        "",
        f"Эпоха чекпойнта {report['train'].get('best_epoch')}, прокси IoU {report['train'].get('best_proxy_iou')},",
        f"остановка {report['train'].get('stopped')}, {report['train'].get('seconds')} с, 2 потока.",
        "Потеря `focal+tversky_ce`, AdamW lr 1e-4, weight decay 0. Тайл 128×256, batch 8.",
        "Аугментация — штатный семплер тайлов (зеркало по x и множитель 0.75–1.3). Это не оценка по реальным файлам.",
        "",
        "## Официальный v8",
        "",
        f"Фон {report['n_bg']} клеток. Пересчёт победителя: val {v[WINNER]['iou']:.4f}, test {t[WINNER]['iou']:.4f}.",
        "",
        "| кандидат | порог | val IoU | val F1 | val FAR | test IoU | test F1 | test FAR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("unet_mix", "ens_blend", WINNER):
        lines.append(
            f"| {name} | {v[name]['threshold']:.6f} | {v[name]['iou']:.4f} | {v[name]['f1']:.4f} | "
            f"{v[name]['far']:.5f} | {t[name]['iou']:.4f} | {t[name]['f1']:.4f} | {t[name]['far']:.5f} |"
        )
    lines += [
        "",
        "### F1-разложение val, hot / cold IoU",
        "",
        "| кандидат | hot IoU | cold IoU |",
        "|---|---:|---:|",
    ]
    for name in ("unet_mix", "ens_blend", WINNER):
        lines.append(f"| {name} | {v[name]['hot']['iou']:.4f} | {v[name]['cold']['iou']:.4f} |")
    lines += [
        "",
        "## Вложенные 5 фолдов",
        "",
        f"micro победитель {nest['winner_iou']:.4f}, ансамбль {nest['blend_iou']:.4f}, "
        f"прирост {nest['delta']:+.4f}. Планка прироста {NESTED_GAIN:.3f}.",
        "",
        "| фолд | n | winner | blend | Δ |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in nest["folds"]:
        lines.append(
            f"| {row['fold']} | {row['n_seq']} | {row['winner']['iou']:.4f} | "
            f"{row['blend']['iou']:.4f} | {row['delta']:+.4f} |"
        )
    lines += [
        "",
        "Файлы: `dts/mixstyle.py`, `scripts/s3_mixstyle_train.py`, `scripts/s3_mixstyle_eval.py`,",
        "`artifacts/eval/s3_mixstyle_train.json`, `artifacts/eval/s3_mixstyle_v8.json`,",
        "`artifacts/eval/s3_mixstyle_unet.pt`.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    t0 = time.time()
    predict_batched = _predict()
    labels = pd.read_csv("data/synth_meta/v8/labels.csv")
    seqs = pd.read_csv("data/synth_meta/v8/sequences.csv").set_index("seq_id")
    model = MixStyleUNet(w=16, p=0.5, alpha=0.1)
    model.load_init("artifacts/eval/s3_mixstyle_unet.pt")
    model.eval()
    ref = models.DTSUNet(w=16)
    ref.load_state_dict(model.backbone.state_dict())
    ref.eval()
    probe = torch.randn(1, 3, 128, 256)
    with torch.no_grad():
        err = float((model(probe) - ref(probe)).abs().max())
    log(f"identity max|Δ| {err:.3e}")
    if err > 1e-5:
        raise SystemExit("чекпойнт на eval не совпал с DTSUNet")
    rng = np.random.default_rng(0)
    img = rng.standard_normal((3, 256, 512)).astype(np.float32)
    got = predict_batched(model, img, batch=2)
    base = models.predict_full(model, img)
    pred_err = float(np.max(np.abs(got - base)))
    log(f"сверка predict max|Δ| {pred_err:.3e}")
    if pred_err > 1e-4:
        raise SystemExit(f"predict расходится: {pred_err}")
    del probe, img
    val_shapes = shapes_of("val")
    test_shapes = shapes_of("test")
    if TMP.exists():
        shutil.rmtree(TMP)
    predict_split(model, predict_batched, "val", val_shapes)
    thr, n_bg = thresholds_of(val_shapes)
    log("метрики val")
    val_rep = score_split("val", val_shapes, thr)
    if abs(val_rep[WINNER]["iou"] - WINNER_VAL) > 0.005:
        raise SystemExit("пересчёт победителя на val далёк от 0.3867")
    winner_c, blend_c, alone_c = compact_val(val_shapes, labels, seqs)
    if len(winner_c) != 45:
        raise SystemExit(f"ожидалось 45 val, есть {len(winner_c)}")
    full_w = s2.eval_one(winner_c)
    full_b = s2.eval_one(blend_c)
    log(f"compact full winner {full_w['iou']:.4f} blend {full_b['iou']:.4f}")
    nest = nested_k(winner_c, blend_c, k=5)
    # Выбор только по полному val. Вложенный прирост — отдельный фильтр приёмки.
    sel_val = float(val_rep["ens_blend"]["iou"])
    win_val = float(val_rep[WINNER]["iou"])
    beats_val = bool(sel_val > win_val)
    nested_ok = bool(nest["delta"] >= NESTED_GAIN)
    accept = bool(beats_val and nested_ok)
    predict_split(model, predict_batched, "test", test_shapes)
    log("метрики test")
    test_rep = score_split("test", test_shapes, thr)
    if abs(test_rep[WINNER]["iou"] - WINNER_TEST) > 0.005:
        raise SystemExit("пересчёт победителя на test далёк от 0.4053")
    if TMP.exists():
        shutil.rmtree(TMP, ignore_errors=True)
    train_meta = {}
    tp = Path("artifacts/eval/s3_mixstyle_train.json")
    if tp.exists():
        train_meta = json.loads(tp.read_text())
    if accept:
        decision = (
            f"Принято. Ансамбль val IoU {sel_val:.4f} выше победителя {win_val:.4f}, "
            f"вложенный прирост {nest['delta']:+.4f} не меньше 0.003."
        )
    elif beats_val:
        decision = (
            f"Отклонено вложенной проверкой. Полный val ансамбля {sel_val:.4f} выше победителя {win_val:.4f}, "
            f"но прирост 5-fold {nest['delta']:+.4f} меньше 0.003. Победитель не заменяется."
        )
    else:
        decision = (
            f"Отклонено. Полный val ансамбля {sel_val:.4f} не выше победителя {win_val:.4f}. "
            f"Вложенный прирост {nest['delta']:+.4f}. Победитель не заменяется."
        )
    report = dict(
        name="MIXSTYLE",
        protocol=(
            "Official: v8 val IoU at FAR 1e-3, threshold from the full v8 val background. "
            "Candidate is the mean of MixStyle U-Net and ens_u2ft_tcn_lgbm. "
            "Nested: scripts/s3_sel2_cv.py procedure, k=5, threshold and IoU on the holdout, "
            "micro IoU, gain vs the winner under the same procedure must be >= 0.003. "
            "Test is not used for selection."
        ),
        published_winner=dict(name=WINNER, val_iou=WINNER_VAL, test_iou=WINNER_TEST),
        n_bg=n_bg,
        predict_max_abs_err=pred_err,
        thresholds=thr,
        val=val_rep,
        test=test_rep,
        compact_full=dict(winner_iou=float(full_w["iou"]), blend_iou=float(full_b["iou"]),
                          unet_mix_iou=float(s2.eval_one(alone_c)["iou"])),
        nested5=nest,
        nested_gain_bar=NESTED_GAIN,
        selected="ens_blend",
        selected_val_iou=sel_val,
        selected_test_iou=float(test_rep["ens_blend"]["iou"]),
        beats_val=beats_val,
        nested_ok=nested_ok,
        accept=accept,
        decision=decision,
        train=dict(
            best_epoch=train_meta.get("best_epoch"),
            best_proxy_iou=train_meta.get("best_proxy_iou"),
            seconds=train_meta.get("seconds"),
            stopped=train_meta.get("stopped"),
            real_traces_used=False,
            real_noise_used=False,
        ),
        seconds=round(time.time() - t0, 1),
        real_traces_used=False,
        real_noise_used=False,
    )
    out = Path("artifacts/eval/s3_mixstyle_v8.json")
    out.write_text(json.dumps(plain(report), indent=1, ensure_ascii=False))
    Path("docs/S3_MIXSTYLE.md").write_text(render_md(report))
    line = (
        f"MIXSTYLE_DONE selected=ens_blend val={sel_val:.4f} test={test_rep['ens_blend']['iou']:.4f} "
        f"unet_val={val_rep['unet_mix']['iou']:.4f} winner_val={win_val:.4f} "
        f"nested5_blend={nest['blend_iou']:.4f} nested5_winner={nest['winner_iou']:.4f} "
        f"delta5={nest['delta']:+.4f} accept={int(accept)} "
        f"best_epoch={train_meta.get('best_epoch')} train_sec={train_meta.get('seconds')} "
        f"eval_sec={report['seconds']} real=0\n"
    )
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line)
    if accept and sel_val > WINNER_VAL:
        with LOG.open("a", encoding="utf-8") as fh:
            fh.write(f"BEATS_WINNER MIXSTYLE ens_blend val {sel_val:.4f} test {test_rep['ens_blend']['iou']:.4f}\n")
    log(line.strip())
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
