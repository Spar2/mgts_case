"""Официальный IoU@FAR 1e-3 и вложенная проверка Lovász U-Net.

Кандидат заранее фиксирован: равное среднее ens_u2ft_tcn_lgbm и U-Net Lovász.
Порог полного поля — фон v8 val (valid & ev==0 & nu==0). Test порог не двигает.
Вложенная проверка — s3_sel2_cv.nested_fixed, k=2 и k=5, планка 0.3867+0.003.
Реальные CSV не читаются.

  OMP_NUM_THREADS=2 /workspace/repo/run/.venv/bin/python -u scripts/s3_lovasz_eval.py
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

import s3_sel2_cv as cv
import s3_sel_core as sel
import s3_sel_eval as sev
from dts import data, metrics as M, models

WINNER = "ens_u2ft_tcn_lgbm"
WINNER_VAL = 0.38671235821914546
WINNER_TEST = 0.4052762065230473
BEATS_MARGIN = 1e-4
TMP = Path("/tmp/s3_lovasz_maps")
NAMES = ("unet_lovasz", "ens_blend", WINNER)


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


def _dom():
    spec = importlib.util.spec_from_file_location("s3_dom_eval", ROOT / "scripts" / "s3_dom_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bg_mask(ev, nu, valid):
    return np.asarray(valid, bool) & (np.asarray(ev) == 0) & (np.asarray(nu) == 0)


def load_model():
    model = models.build("unet", 16)
    state = torch.load(ROOT / "artifacts/eval/s3_lovasz_unet.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def save_maps(stem, s, h):
    d = TMP / "val"
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"{stem}_s.npy", np.asarray(s, np.float16))
    np.save(d / f"{stem}_h.npy", np.asarray(h, np.float16))


def lovasz_maps(stem):
    d = TMP / "val"
    s = np.load(d / f"{stem}_s.npy").astype(np.float32)
    h = np.load(d / f"{stem}_h.npy").astype(np.float32)
    return s, h


def winner_maps(split, name):
    with np.load(Path("artifacts/pred/v8") / WINNER / split / name) as z:
        return np.asarray(z["s"], np.float32), np.asarray(z["h"], np.float32)


def fields_of(stem, split, ens_s=None, ens_h=None, lov_s=None, lov_h=None):
    if lov_s is None:
        lov_s, lov_h = lovasz_maps(stem)
    if ens_s is None:
        ens_s, ens_h = winner_maps(split, f"{stem}.npz")
    blend_s = (0.5 * (lov_s + ens_s)).astype(np.float32)
    blend_h = (0.5 * (lov_h + ens_h)).astype(np.float32)
    return {
        "unet_lovasz": (lov_s, lov_h),
        "ens_blend": (blend_s, blend_h),
        WINNER: (ens_s, ens_h),
    }


def collect_thresholds(files) -> tuple[dict, int]:
    """По одному кандидату: фон val — десятки миллионов клеток."""
    thr = {}
    n_bg = 0
    for n in NAMES:
        parts = []
        counted = 0
        for p in files:
            with np.load(p) as z:
                mask = bg_mask(z["ev"], z["nu"], z["valid"])
            score = fields_of(p.stem, "val")[n][0]
            parts.append(np.asarray(score[mask], np.float32))
            counted += int(mask.sum())
            del score
        bg = np.concatenate(parts)
        del parts
        thr[n] = float(M.far_threshold(bg, 1e-3))
        n_bg = counted
        log(f"порог {n} {thr[n]:.6f} n_bg {bg.size}")
        del bg
    return thr, n_bg


def evaluate_split(files, split, thr, labels, seqs, online=None) -> dict:
    """online(p) -> dict name -> (s, h), если карты не лежат в TMP."""
    tot = {n: np.zeros(3, np.int64) for n in NAMES}
    hot = {n: np.zeros(3, np.int64) for n in NAMES}
    cold = {n: np.zeros(3, np.int64) for n in NAMES}
    far_n = {n: 0 for n in NAMES}
    far_d = {n: 0 for n in NAMES}
    per = {n: {} for n in NAMES}
    rows = {n: [] for n in NAMES}
    cache = Path("artifacts/cache/v8") / split
    t0 = time.time()
    for i, p in enumerate(files, 1):
        with np.load(cache / p.name) as z:
            ev = np.asarray(z["ev"])
            valid = np.asarray(z["valid"], bool)
            nu = np.asarray(z["nu"])
            x = np.asarray(z["x"], np.float64)
            t = np.asarray(z["t"], np.float64)
        if online is None:
            packed = fields_of(p.stem, split)
        else:
            packed = online(p)
        mask_bg = bg_mask(ev, nu, valid)
        truth = ev > 0
        hot_t = (ev >= 1) & (ev <= 4)
        cold_t = ev >= 5
        boxes = []
        for r in M.label_boxes(labels, p.stem).itertuples():
            if r.kind != "event":
                continue
            px = M.event_pixels(r, x, t, ev)
            if not px.any():
                continue
            boxes.append((r, px))
        for n in NAMES:
            s, h = packed[n]
            pred = (s >= thr[n]) & valid
            c = np.asarray(M.confusion(pred, truth, valid), np.int64)
            tot[n] += c
            per[n][p.stem] = tuple(int(v) for v in c)
            hot[n] += np.asarray(M.confusion(pred & (h > 0), hot_t, valid), np.int64)
            cold[n] += np.asarray(M.confusion(pred & (h <= 0), cold_t, valid), np.int64)
            far_n[n] += int((pred & mask_bg).sum())
            far_d[n] += int(mask_bg.sum())
            for r, px in boxes:
                frac = float(pred[px].mean()) if px.any() else 0.0
                rows[n].append(dict(amp=float(r.amp_obs_C), label=str(r.label), frac=frac, hit=frac >= 0.3,
                                     tp=int((pred & px).sum()), fn=int((~pred & px).sum())))
        del packed
        if online is not None and (i == 1 or i % 5 == 0 or i == len(files)):
            log(f"{split} pred {i}/{len(files)} {time.time() - t0:.0f}s")
    report = {}
    for n in NAMES:
        E = pd.DataFrame(rows[n])
        row = dict(
            threshold=float(thr[n]),
            far=float(far_n[n] / max(far_d[n], 1)),
            n_bg=int(far_d[n]),
            **M.prf(*[int(v) for v in tot[n]]),
            hot=M.prf(*[int(v) for v in hot[n]]),
            cold=M.prf(*[int(v) for v in cold[n]]),
            iou_ci95=M.bootstrap_ci(per[n], sorted(per[n]), n=200, seed=0),
        )
        if len(E):
            def agg(g):
                tp = int(g.tp.sum())
                fn = int(g.fn.sum())
                return dict(n=int(len(g)), recall=float(g.hit.mean()),
                            pixel_recall=float(tp / max(tp + fn, 1)))
            row["events_all"] = agg(E)
            weak = E[E.amp <= 2]
            row["weak_events"] = agg(weak) if len(weak) else dict(n=0, recall=None, pixel_recall=None)
        report[n] = row
        log(f"{split} {n} IoU {row['iou']:.4f} F1 {row['f1']:.4f} FAR {row['far']:.5f}")
    return report


def build_frames(files, labels, seqs):
    frames = []
    for p in files:
        with np.load(p) as z:
            ev_map = np.asarray(z["ev"])
            valid = np.asarray(z["valid"])
            nu = np.asarray(z["nu"])
            x = np.asarray(z["x"], np.float64)
            t = np.asarray(z["t"], np.float64)
        lov_s, _h = lovasz_maps(p.stem)
        ens_s, _eh = winner_maps("val", p.name)
        if lov_s.shape != ev_map.shape or ens_s.shape != ev_map.shape:
            raise SystemExit(f"форма {p.stem} lov {lov_s.shape} ens {ens_s.shape} ev {ev_map.shape}")
        ev_idx, nu_idx, bg_idx, n_bg = sev._indices(ev_map, valid, nu, p.stem)
        le, ln, lb = sev._take(lov_s, ev_idx, nu_idx, bg_idx)
        ee, en, eb = sev._take(ens_s, ev_idx, nu_idx, bg_idx)
        cooling, _rows = sev._events_for(p.stem, ev_map, x, t, labels, seqs, ev_idx)
        frames.append(dict(
            stem=p.stem,
            ev_m=le.astype(np.float16)[None],
            nu_m=ln.astype(np.float16)[None],
            bg_m=lb.astype(np.float16)[None],
            n_bg=n_bg,
            cooling=cooling,
            ens_ev=ee.astype(np.float16),
            ens_nu=en.astype(np.float16),
            ens_bg=eb.astype(np.float16),
        ))
        del lov_s, ens_s
    frames.sort(key=lambda fr: fr["stem"])
    return frames


def nested_pair(frames):
    """Равное среднее через s3_sel2_cv.nested_fixed. Колонка lovasz — единственная в стеке."""
    saved = list(cv.ASC_DOM_NAMES)
    cv.ASC_DOM_NAMES = ["lovasz"]
    try:
        if frames[0]["ev_m"].shape[0] != 1:
            raise SystemExit("ожидался один канал Lovász")
        blend = cv.nested_fixed(frames, [("ens", 1.0), ("lovasz", 1.0)], ks=(2, 5))
        winner = cv.nested_fixed(frames, [("ens", 1.0)], ks=(2, 5))
        compact = cv.eval_one(cv.channel_frames(frames, [("ens", 1.0)]))
    finally:
        cv.ASC_DOM_NAMES = saved
    return blend, winner, compact


def render_md(report) -> str:
    v, t = report["val"], report["test"]
    tr = report["train"]
    nest = report["nested_blend_fixed"]["verdict"]
    nw = report["nested_winner"]["verdict"]
    decision = (
        f"Вложенный min(k=2, k=5) ансамбля {nest['worst']:.4f} "
        f"при планке {nest['bar']:.4f}: {'выше, принято' if nest['accept'] else 'не выше, не принято'}. "
        f"Тот же минимум у победителя {nw['worst']:.4f}, прирост {report['nested_gain_vs_winner']:+.4f}. "
        f"Полный val IoU ансамбля {v['ens_blend']['iou']:.6f} "
        f"против опубликованных {WINNER_VAL:.6f}: "
        f"{'выше более чем на 1e-4' if report['beats_winner'] else 'замены победителя нет'}."
    )
    lines = [
        "# S3 Lovász — прямая оптимизация IoU",
        "",
        "Нейросетевой детектор. Потеря — Lovász-hinge и Lovász-softmax (Berman, Triki, Blaschko, CVPR 2018):",
        "продолжение Жаккара на отсортированные ошибки пикселей. Hinge берёт передний план",
        "`y > 0` и логит `log(s / (1 − s))`, где `s = 1 − p(фон)`. Softmax усредняет классы,",
        "которые есть на тайле (`present`): фон, нагрев, охлаждение. Невалидные пиксели `y < 0`",
        "в сортировку не входят. Смесь 0.5 / 0.5, среднее по изображениям батча.",
        "",
        "Старт — `artifacts/stage1/unet_v810ft.pt`, ширина 16, BatchNorm заморожен.",
        "AdamW lr 10⁻⁵ (прогон с 10⁻⁴ поднял потерю и уронил прокси, в чекпойнт не вошёл),",
        "weight decay 0, клип градиента 1, тайл 128×256, batch 8, 2 потока.",
        "Обучение только на синтетике `artifacts/cache/v8/train` (уже готовый mmap) и",
        "`artifacts/cache/v10/train` (npz по одному файлу, `artifacts/mm/v10` не создаётся).",
        "Реальные CSV, спектр фона и σ реального шума не читались. Аугментация тайла — зеркало по x",
        "и множитель 0.75–1.3, без добавки реального шума.",
        "",
        "Чекпойнт внутри обучения — прокси IoU@FAR 1e-3 на четырёх разнесённых последовательностях v8 val.",
        "Это не официальная цифра. Официальный кандидат заранее фиксирован и на val не перебирается:",
        "равное среднее сохранённого `ens_u2ft_tcn_lgbm` и U-Net после шагов Lovász.",
        "Порог — квантиль чистого фона всех 45 последовательностей v8 val при FAR 1e-3.",
        "Test тем же порогом, в выбор не входит.",
        "",
        "Вложенная проверка идёт через `scripts/s3_sel2_cv.py` (`nested_fixed`):",
        "последовательности v8 val по имени, фолды `i % k` для k = 2 и k = 5,",
        "веса смеси фиксированы (0.5 / 0.5), порог FAR 1e-3 и IoU только на holdout по его фону,",
        "micro IoU — сумма tp/fp/fn. Компактный фон до 200000 клеток на файл, как в `s3_sel_eval.py`.",
        "Принятие — `nested_verdict`: min(k=2, k=5) строго выше 0.3867+0.003 = 0.3897.",
        "Равенство планке — отказ. Прирост против вложенного IoU самого победителя пишется рядом",
        "и планку не подменяет.",
        "",
        decision,
        "",
        "## Обучение",
        "",
        f"Эпоха чекпойнта {tr.get('best_epoch')}, прокси IoU {tr.get('best_proxy_iou')},",
        f"остановка {tr.get('stopped')}, {tr.get('seconds')} с.",
        "Прокси на четырёх последовательностях не выбирает ансамбль.",
        (
            "Ни одна эпоха не подняла прокси выше старта, поэтому в официальную оценку вошли стартовые веса `unet_v810ft.pt`."
            if tr.get("best_epoch") == -1
            else "В оценку вошли веса эпохи с лучшим прокси."
        ),
        "История потери — в `artifacts/eval/s3_lovasz_train.json`.",
        "",
        "## Официальный v8",
        "",
        f"Фон val {report['n_bg']} клеток. Пересчёт победителя: val {v[WINNER]['iou']:.6f}, test {t[WINNER]['iou']:.6f}.",
        f"Расхождение с публикацией val {report['sanity_winner_val_abs_err']:.6f}, test {report['sanity_winner_test_abs_err']:.6f}.",
        f"Сверка тайлинга с `models.predict_full`: max|Δ| = {report['predict_max_abs_err']:.3e}.",
        "",
        "| кандидат | порог | val IoU | val F1 | val FAR | test IoU | test F1 | test FAR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in NAMES:
        lines.append(
            f"| {name} | {v[name]['threshold']:.6f} | {v[name]['iou']:.4f} | {v[name]['f1']:.4f} | "
            f"{v[name]['far']:.5f} | {t[name]['iou']:.4f} | {t[name]['f1']:.4f} | {t[name]['far']:.5f} |"
        )
    lines += [
        "",
        "| кандидат | val hot IoU | val cold IoU | val события | val слабые ≤2 °C |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in NAMES:
        evs = v[name].get("events_all") or {}
        weak = v[name].get("weak_events") or {}
        lines.append(
            f"| {name} | {v[name]['hot']['iou']:.4f} | {v[name]['cold']['iou']:.4f} | "
            f"{evs.get('recall')} ({evs.get('n')}) | {weak.get('recall')} ({weak.get('n')}) |"
        )
    b2 = nest["iou_by_k"]["2"]
    b5 = nest["iou_by_k"]["5"]
    w2 = nw["iou_by_k"]["2"]
    w5 = nw["iou_by_k"]["5"]
    lines += [
        "",
        "## Вложенные фолды",
        "",
        f"Компактный полный val победителя (проверка загрузки, не критерий): {report['sanity_winner_compact_iou']:.4f}.",
        "",
        "| | k=2 | k=5 | min |",
        "|---|---:|---:|---:|",
        f"| ens_blend | {b2:.4f} | {b5:.4f} | {nest['worst']:.4f} |",
        f"| победитель | {w2:.4f} | {w5:.4f} | {nw['worst']:.4f} |",
        "",
        f"Планка {sel.NESTED_BAR:.4f}. Принятие ансамбля: {'да' if report['nested_accept'] else 'нет'}.",
        f"Прирост min против победителя: {report['nested_gain_vs_winner']:+.4f}.",
        f"BEATS_WINNER по полному val: {'да' if report['beats_winner'] else 'нет'}.",
        "",
        "Слабые события и test в выбор не входят. Карточки оператора не считались.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    t_all = time.time()
    train = json.loads(Path("artifacts/eval/s3_lovasz_train.json").read_text(encoding="utf-8"))
    dom = _dom()
    model = load_model()
    err = float(dom.check_predict(model))
    labels = pd.read_csv(ROOT / "data/synth_meta/v8/labels.csv")
    seqs = pd.read_csv(ROOT / "data/synth_meta/v8/sequences.csv").set_index("seq_id")
    val_files = data.list_cache("artifacts/cache/v8", "val")
    if len(val_files) != 45:
        raise SystemExit(f"ожидалось 45 val, есть {len(val_files)}")
    TMP.mkdir(parents=True, exist_ok=True)
    try:
        t0 = time.time()
        for i, p in enumerate(val_files, 1):
            with np.load(p) as z:
                X = np.asarray(z["X"], np.float32)
            prob = dom.predict_batched(model, X, batch=4)
            s = (1.0 - prob[0]).astype(np.float32)
            h = (prob[1] - prob[2]).astype(np.float32)
            save_maps(p.stem, s, h)
            del X, prob, s, h
            if i == 1 or i % 5 == 0 or i == len(val_files):
                log(f"val pred {i}/{len(val_files)} {time.time() - t0:.0f}s")
        thr, n_bg = collect_thresholds(val_files)
        log("метрики val")
        val = evaluate_split(val_files, "val", thr, labels, seqs)
        if abs(val[WINNER]["iou"] - WINNER_VAL) > 0.005:
            raise SystemExit(
                f"пересчёт победителя {val[WINNER]['iou']:.6f} далеко от {WINNER_VAL:.6f}, тест не считаю"
            )
        frames = build_frames(val_files, labels, seqs)
        log("nested")
        blend_n, winner_n, compact = nested_pair(frames)
        if abs(float(compact["iou"]) - 0.3867) > 0.02:
            raise SystemExit(f"компактный победитель {compact['iou']:.4f} далеко от 0.3867")
        log("test")
        test_files = data.list_cache("artifacts/cache/v8", "test")
        t1 = time.time()

        def online(p, _model=model, _dom=dom):
            with np.load(p) as z:
                X = np.asarray(z["X"], np.float32)
            prob = _dom.predict_batched(_model, X, batch=4)
            lov_s = (1.0 - prob[0]).astype(np.float32)
            lov_h = (prob[1] - prob[2]).astype(np.float32)
            ens_s, ens_h = winner_maps("test", p.name)
            del X, prob
            return fields_of(p.stem, "test", ens_s=ens_s, ens_h=ens_h, lov_s=lov_s, lov_h=lov_h)

        # evaluate_split с online не должен читать TMP. Передаём файлы test.
        test = evaluate_split(test_files, "test", thr, labels, seqs, online=online)
        log(f"test готов {time.time() - t1:.0f}s")
    finally:
        shutil.rmtree(TMP, ignore_errors=True)

    sanity_val = abs(val[WINNER]["iou"] - WINNER_VAL)
    sanity_test = abs(test[WINNER]["iou"] - WINNER_TEST)
    sanity_ok = bool(sanity_val < 0.005 and sanity_test < 0.005)
    beats = bool(sanity_ok and val["ens_blend"]["iou"] > WINNER_VAL + BEATS_MARGIN)
    gain = float(blend_n["verdict"]["worst"] - winner_n["verdict"]["worst"])
    report = dict(
        name="LOVASZ",
        paper="Berman, Triki, Blaschko, CVPR 2018",
        ensemble="0.5 * (ens_u2ft_tcn_lgbm + unet_lovasz)",
        selection="v8 val pixel IoU at FAR 1e-3 of the fixed equal ensemble; test not used",
        nested_protocol=(
            "scripts/s3_sel2_cv.nested_fixed: 2-fold и 5-fold по отсортированным последовательностям v8 val. "
            "Веса 0.5/0.5 фиксированы, на train-фолде не подбираются. "
            "Порог FAR 1e-3 и IoU — на holdout, фон holdout. "
            "Принятие: min(micro IoU k=2, k=5) > 0.3867+0.003. "
            "Компактный фон до 200000 клеток на файл. Реальные csv не читаются."
        ),
        nested_bar=sel.NESTED_BAR,
        predict_max_abs_err=err,
        published_winner_val_iou=WINNER_VAL,
        published_winner_test_iou=WINNER_TEST,
        sanity_winner_val_abs_err=float(sanity_val),
        sanity_winner_test_abs_err=float(sanity_test),
        sanity_ok=sanity_ok,
        sanity_winner_compact_iou=float(compact["iou"]),
        n_bg=int(n_bg),
        val=val,
        test=test,
        nested_blend_fixed=blend_n,
        nested_winner=winner_n,
        nested_accept=bool(blend_n["verdict"]["accept"]),
        nested_gain_vs_winner=gain,
        nested_gain_vs_winner_ge_0_003=bool(gain >= 0.003),
        beats_winner=beats,
        train=dict(
            best_epoch=train.get("best_epoch"),
            best_proxy_iou=train.get("best_proxy_iou"),
            stopped=train.get("stopped"),
            seconds=train.get("seconds"),
            loss=train.get("loss"),
            ckpt=train.get("ckpt"),
        ),
        real_traces_used=False,
        real_csv_used=False,
        seconds=round(time.time() - t_all, 1),
    )
    out = Path("artifacts/eval/s3_lovasz_v8.json")
    out.write_text(json.dumps(plain(report), indent=1, ensure_ascii=False), encoding="utf-8")
    Path("docs/S3_LOVASZ.md").write_text(render_md(report), encoding="utf-8")
    log(
        f"LOVASZ_RESULT val_blend {val['ens_blend']['iou']:.6f} "
        f"val_alone {val['unet_lovasz']['iou']:.6f} "
        f"winner {val[WINNER]['iou']:.6f} "
        f"nested {blend_n['verdict']['worst']:.4f} accept {int(blend_n['verdict']['accept'])} "
        f"gain_vs_winner {gain:+.4f} beats {int(beats)}"
    )
    log(f"записан {out}")


if __name__ == "__main__":
    main()
