"""Оценка MAE: FAR 1e-3 по фону v8 val, смеси с победителем, вложенные фолды s3_sel2_cv, v10/v11.

Опубликованный val победителя 0.38671235821914546. Принять только выше этого числа.

Порог не переносится на test. Реальный кабель считается после решения и его не меняет.
Карточки не читаются. Большие карты пишутся в /tmp и удаляются.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "NUMEXPR_MAX_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_k] = "2"

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
import s3_sel_eval as sev
from dts import data, metrics as M
from dts.features import make_channels
from dts.mae_anom import (
    BAND, CANDIDATES, NESTED_BAR, NESTED_GAIN, PROTOCOL_TEXT, RADIUS, TILE,
    WINNER_TEST, WINNER_V10, WINNER_V11, WINNER_VAL,
    MaskedReconstructor, accept_decision, apply_cdf, blend_scores, channel_width,
    choose_candidate, member_maps, residuals,
)
from dts.real import load_csv

WINNER = "ens_u2ft_tcn_lgbm"
TMP = Path("/tmp/s3_mae_maps")
TRAIN_JSON = Path("artifacts/eval/s3_mae_train.json")
CKPT = Path("artifacts/eval/s3_mae_recon.pt")
OUT_JSON = Path("artifacts/eval/s3_mae_v8.json")
DOC = Path("docs/S3_MAE.md")
LOG = Path("/workspace/run_logs/_events.log")
NAMES = CANDIDATES + (WINNER,)


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M', time.gmtime())} {msg}", flush=True)


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


def shapes_of(ver, split) -> dict:
    out = {}
    pred = Path("artifacts/pred") / ver / WINNER / split
    for p in data.list_cache(f"artifacts/cache/{ver}", split):
        with np.load(pred / p.name) as z:
            out[p.stem] = tuple(z["s"].shape)
    return out


def open_map(ver, split, stem, kind, shape, dtype):
    d = TMP / ver / split
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{stem}_{kind}.mmap"
    return np.memmap(path, dtype=dtype, mode="w+", shape=shape)


def read_map(ver, split, stem, kind, shape, dtype):
    path = TMP / ver / split / f"{stem}_{kind}.mmap"
    return np.memmap(path, dtype=dtype, mode="r", shape=shape)


def predict_split(model, knots, ver, split, shapes) -> None:
    t0 = time.time()
    files = data.list_cache(f"artifacts/cache/{ver}", split)
    kw = dict(tile=TILE, overlap=32, band=BAND, radius=RADIUS, tile_batch=2)
    for i, p in enumerate(files, 1):
        shape = shapes[p.stem]
        with np.load(p) as z:
            X = np.asarray(z["X"], np.float32)
        if X.shape[1:] != shape:
            raise SystemExit(f"{ver} {p.name}: X {X.shape[1:]} != {shape}")
        r0, r1 = residuals(model, X[:2], X[2], **kw)
        if not np.isfinite(r1).all() or not np.isfinite(r0).all():
            raise SystemExit(f"non-finite {p.name}")
        s = apply_cdf(np.abs(r1), knots)
        sm = open_map(ver, split, p.stem, "s", shape, np.float32)
        hm = open_map(ver, split, p.stem, "h", shape, np.float16)
        sm[:] = s
        hm[:] = r0.astype(np.float16)
        sm.flush()
        hm.flush()
        del sm, hm, X, r0, r1, s
        if i == 1 or i % 5 == 0 or i == len(files):
            log(f"pred {ver} {split} {i}/{len(files)} {time.time() - t0:.0f}s")


def load_winner(ver, split, stem):
    with np.load(Path("artifacts/pred") / ver / WINNER / split / f"{stem}.npz") as z:
        return np.asarray(z["s"], np.float32), np.asarray(z["h"], np.float32)


def bg_mask(ver, split, stem):
    with np.load(Path("artifacts/cache") / ver / split / f"{stem}.npz") as meta:
        evm = np.asarray(meta["ev"])
        valid = np.asarray(meta["valid"], bool)
        nu = np.asarray(meta["nu"])
    return evm, valid, nu, valid & (evm == 0) & (nu == 0)


def thresholds_of(shapes) -> tuple[dict, int]:
    mae_parts, win_parts = [], []
    n_bg = 0
    for stem, shape in shapes.items():
        _evm, _valid, _nu, bg = bg_mask("v8", "val", stem)
        s = np.asarray(read_map("v8", "val", stem, "s", shape, np.float32))
        ws, _wh = load_winner("v8", "val", stem)
        mae_parts.append(np.asarray(s[bg], np.float32))
        win_parts.append(np.asarray(ws[bg], np.float32))
        n_bg += int(bg.sum())
        del s, ws
    mae = np.concatenate(mae_parts)
    win = np.concatenate(win_parts)
    del mae_parts, win_parts
    # Смеси по одной: пять векторов фона сразу (~1.3 ГБ) не держим.
    thr = {}

    def one(name, arr):
        thr[name] = float(M.far_threshold(arr, 1e-3))
        log(f"порог {name} {thr[name]:.6f} n_bg {n_bg}")

    one("mae", mae)
    for name in ("ens_mean", "ens_max", "ens_mean4"):
        blended = blend_scores(mae, win, name)
        one(name, blended)
        del blended
    one(WINNER, win)
    del mae, win
    return thr, n_bg


def score_split(ver, split, shapes, thr) -> dict:
    tot = {n: np.zeros(3, np.int64) for n in NAMES}
    hot = {n: np.zeros(3, np.int64) for n in NAMES}
    cold = {n: np.zeros(3, np.int64) for n in NAMES}
    far_n = {n: 0 for n in NAMES}
    far_d = {n: 0 for n in NAMES}
    sep_bg, sep_ev = [], []
    for stem, shape in shapes.items():
        evm, valid, _nu, bg = bg_mask(ver, split, stem)
        truth = evm > 0
        hot_t = (evm >= 1) & (evm <= 4)
        cold_t = evm >= 5
        s = np.asarray(read_map(ver, split, stem, "s", shape, np.float32))
        h = np.asarray(read_map(ver, split, stem, "h", shape, np.float16), np.float32)
        ws, wh = load_winner(ver, split, stem)
        if ver == "v8" and split == "val":
            if bg.any():
                sep_bg.append(s[bg][:: max(1, int(bg.sum()) // 20000)])
            evp = valid & truth
            if evp.any():
                sep_ev.append(s[evp][:: max(1, int(evp.sum()) // 20000)])
        packed = {WINNER: (ws, wh)}
        for name in CANDIDATES:
            packed[name] = (blend_scores(s, ws, name), h)
        for n, (ps, ph) in packed.items():
            pred = (ps >= thr[n]) & valid
            tot[n] += np.asarray(M.confusion(pred, truth, valid), np.int64)
            hot[n] += np.asarray(M.confusion(pred & (ph > 0), hot_t, valid), np.int64)
            cold[n] += np.asarray(M.confusion(pred & (ph <= 0), cold_t, valid), np.int64)
            far_n[n] += int((pred & bg).sum())
            far_d[n] += int(bg.sum())
        del s, h, ws, wh
    report = {}
    for n in NAMES:
        row = dict(
            threshold=float(thr[n]),
            far=float(far_n[n] / max(far_d[n], 1)),
            **M.prf(*tot[n]),
            hot=M.prf(*hot[n]),
            cold=M.prf(*cold[n]),
        )
        report[n] = row
        log(f"{ver} {split} {n} IoU {row['iou']:.4f} F1 {row['f1']:.4f} FAR {row['far']:.5f}")
    if sep_bg:
        bg_m = float(np.median(np.concatenate(sep_bg)))
        ev_m = float(np.median(np.concatenate(sep_ev))) if sep_ev else None
        report["_sep"] = {"bg_median_cdf": bg_m, "event_median_cdf": ev_m}
    return report


def frame_of(stem, score, ev_map, valid, nu, x, t, labels, seqs):
    ev_idx, nu_idx, bg_idx, n_bg = sev._indices(ev_map, valid, nu, stem)
    e, n, b = sev._take(score, ev_idx, nu_idx, bg_idx)
    cooling, _rows = sev._events_for(stem, ev_map, x, t, labels, seqs, ev_idx)
    return dict(
        stem=stem,
        ev_m=e.astype(np.float16)[None],
        nu_m=n.astype(np.float16)[None],
        bg_m=b.astype(np.float16)[None],
        n_bg=n_bg,
        cooling=cooling,
    )


def nested_ks(shapes, labels, seqs, selected: str) -> dict:
    """Та же процедура, что scripts/s3_sel2_cv.py: eval_one и sequence_folds, k=2 и k=5."""
    files = data.list_cache("artifacts/cache/v8", "val")
    winner_f, sel_f = [], []
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
        ws, _wh = load_winner("v8", "val", stem)
        blend = blend_scores(s, ws, selected)
        winner_f.append(frame_of(stem, ws, ev_map, valid, nu, x, t, labels, seqs))
        sel_f.append(frame_of(stem, blend, ev_map, valid, nu, x, t, labels, seqs))
        del s, ws
    order = np.argsort([f["stem"] for f in winner_f])
    winner_f = [winner_f[i] for i in order]
    sel_f = [sel_f[i] for i in order]
    out = {}
    for k in (2, 5):
        folds = sel.sequence_folds(len(winner_f), k)
        parts_w, parts_b, rows = [], [], []
        for r, idx in enumerate(folds):
            hw = s2.eval_one(s2.take(winner_f, idx))
            hb = s2.eval_one(s2.take(sel_f, idx))
            parts_w.append(hw)
            parts_b.append(hb)
            rows.append(dict(
                fold=r, winner=float(hw["iou"]), selected=float(hb["iou"]),
                delta=float(hb["iou"] - hw["iou"]),
                tp=hb["tp"], fp=hb["fp"], fn=hb["fn"],
            ))
            log(f"nested k={k} fold {r} winner {hw['iou']:.4f} {selected} {hb['iou']:.4f} delta {hb['iou'] - hw['iou']:+.4f}")
        iw = float(sel.micro_iou(parts_w))
        ib = float(sel.micro_iou(parts_b))
        out[str(k)] = dict(winner_iou=iw, selected_iou=ib, delta=float(ib - iw), folds=rows)
        log(f"nested k={k} micro {selected} {ib:.4f} winner {iw:.4f} delta {ib - iw:+.4f}")
    verdict = sel.nested_verdict({k: out[k]["selected_iou"] for k in out})
    out["verdict"] = verdict
    out["procedure"] = "s3_sel2_cv.eval_one, sequence_folds i%k, micro_iou, nested_verdict"
    return out


def real_cable(model, knots, thr_mae: float) -> dict:
    """Один кабель, четыре окна. Порог synth не подгоняется. Карточки не читаются."""
    path = Path("/workspace/mgts/MGTS_data/4.csv")
    try:
        rec = load_csv(path)
    except Exception as exc:
        return {"error": str(exc), "used_for_selection": False}
    T = rec["T"]
    valid = rec["valid"]
    n = T.shape[0]
    height = 192
    if n <= height:
        starts = [0]
    else:
        starts = [int(v) for v in np.linspace(0, n - height, num=4)]
    fracs = []
    kw = dict(tile=TILE, overlap=32, band=BAND, radius=RADIUS, tile_batch=2)
    for t0 in starts:
        ch, _meta = make_channels(T[t0:t0 + height], valid[t0:t0 + height])
        _r0, r1 = residuals(model, ch[:2], ch[2], **kw)
        s = apply_cdf(np.abs(r1), knots)
        m = ch[2] > 0.5
        if int(m.sum()) == 0:
            continue
        fracs.append(float(np.mean(s[m] >= thr_mae)))
    return {
        "file": "4.csv",
        "sha256": rec["sha256"],
        "shape": [int(n), int(T.shape[1])],
        "n_chunks": len(fracs),
        "frac_above_synth_far": float(np.mean(fracs)) if fracs else None,
        "chunks": fracs,
        "used_for_selection": False,
        "cards_read": False,
        "real_noise_psd": False,
    }


def render_md(rep: dict) -> str:
    sel_name = rep["selected"]

    def row(block, name):
        b = block[name]
        return f"| {name} | {b['threshold']:.6f} | {b['iou']:.4f} | {b['f1']:.4f} | {b['far']:.5f} |"

    lines = [
        "# S3 MAE",
        "",
        "Маскированный реконструктор фона. Это не rectified flow и не поле скорости.",
        "Учится на синтетике v8 и v10 train, преимущественно на тайлах без события.",
        "Маска — полосы по расстоянию. Цель — восстановить dT_space и dT_time в скрытых клетках.",
        "Скор — CDF |остаток dT_space|. Знак нагрев/холод — знак остатка dT_time.",
        "Смеси с ens_u2ft_tcn_lgbm фиксированы до val: mean, max и (4×победитель + mae)/5.",
        "Победитель смеси выбран по val IoU. Вложенная проверка этот выбор не переигрывает.",
        "Порог FAR 1e-3 — квантиль фона всего v8 val. v10 и v11 этот порог не двигают.",
        "",
        PROTOCOL_TEXT,
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
        f"Проба: sep {rep['train'].get('best', {}).get('sep')}, "
        f"фон |остаток| {rep['train'].get('best', {}).get('bg_abs')}, "
        f"restore {rep['train'].get('best', {}).get('restore')}.",
        "restore около 1 — всплеск в скрытой полосе не копируется. Около 0 — сеть повторяет всплеск.",
        "",
        "## Официальный v8",
        "",
        f"Фон {rep['n_bg']} клеток. Пересчёт победителя: val {rep['val'][WINNER]['iou']:.4f}, "
        f"test {rep['test'][WINNER]['iou']:.4f}.",
        "",
        "| кандидат | порог | val IoU | val F1 | val FAR |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in list(CANDIDATES) + [WINNER]:
        lines.append(row(rep["val"], name))
    lines += ["", "| кандидат | test IoU | test F1 | test FAR |", "|---|---:|---:|---:|"]
    for name in list(CANDIDATES) + [WINNER]:
        b = rep["test"][name]
        lines.append(f"| {name} | {b['iou']:.4f} | {b['f1']:.4f} | {b['far']:.5f} |")
    lines += [
        "",
        "## v10 и v11 test, порог v8",
        "",
        "| версия | mae | выбранный | победитель |",
        "|---|---:|---:|---:|",
        f"| v10 | {rep['v10']['mae']['iou']:.4f} | {rep['v10'][sel_name]['iou']:.4f} | {rep['v10'][WINNER]['iou']:.4f} |",
        f"| v11 | {rep['v11']['mae']['iou']:.4f} | {rep['v11'][sel_name]['iou']:.4f} | {rep['v11'][WINNER]['iou']:.4f} |",
        "",
        "## Вложенные фолды",
        "",
        "Процедура scripts/s3_sel2_cv.py: `eval_one`, фолды `i % k`, micro по сумме tp/fp/fn.",
        f"Выбранный рецепт: {sel_name}.",
        "",
    ]
    for k in ("2", "5"):
        block = rep["nested"][k]
        lines.append(f"### k={k}")
        lines.append("")
        lines.append("| фолд | победитель | выбранный | дельта |")
        lines.append("|---:|---:|---:|---:|")
        for r in block["folds"]:
            lines.append(f"| {r['fold']} | {r['winner']:.4f} | {r['selected']:.4f} | {r['delta']:+.4f} |")
        lines.append("")
        lines.append(
            f"Micro: выбранный {block['selected_iou']:.4f}, победитель {block['winner_iou']:.4f}, "
            f"дельта {block['delta']:+.4f}."
        )
        lines.append("")
    lines += [
        f"min(k=2, k=5) = {rep['nested_min']:.4f}, планка {NESTED_BAR}.",
        f"Прирост k=5 = {rep['nested']['5']['delta']:+.4f}, нужно ≥ {NESTED_GAIN}.",
        "",
        "## Реальный кабель",
        "",
        "Файл 4.csv, четыре окна, порог synth. Это описание, не выбор модели. Карточки не читались.",
        f"Доля клеток выше synth-порога FAR: {rep['real_cable'].get('frac_above_synth_far')}.",
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
    train_meta = json.loads(TRAIN_JSON.read_text(encoding="utf-8"))
    knots = np.asarray(train_meta["cdf_knots"], np.float64)
    if int(train_meta.get("band", BAND)) != BAND or int(train_meta.get("radius", RADIUS)) != RADIUS:
        raise SystemExit("константы обучения не совпали с модулем")
    model = MaskedReconstructor(16)
    model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True))
    model.eval()
    if channel_width(model) > 16:
        raise SystemExit("ширина > 16")
    labels = pd.read_csv("data/synth_meta/v8/labels.csv")
    seqs = pd.read_csv("data/synth_meta/v8/sequences.csv").set_index("seq_id")
    if TMP.exists():
        shutil.rmtree(TMP)
    try:
        shapes = shapes_of("v8", "val")
        predict_split(model, knots, "v8", "val", shapes)
        thr, n_bg = thresholds_of(shapes)
        val_rep = score_split("v8", "val", shapes, thr)
        sep = val_rep.pop("_sep", {})
        ious = {name: float(val_rep[name]["iou"]) for name in CANDIDATES}
        selected = choose_candidate(ious)
        log(f"выбран {selected} val {ious[selected]:.4f}")
        nest = nested_ks(shapes, labels, seqs, selected)
        shutil.rmtree(TMP / "v8" / "val", ignore_errors=True)
        shapes_te = shapes_of("v8", "test")
        predict_split(model, knots, "v8", "test", shapes_te)
        test_rep = score_split("v8", "test", shapes_te, thr)
        test_rep.pop("_sep", None)
        shutil.rmtree(TMP / "v8" / "test", ignore_errors=True)
        extra = {}
        for ver in ("v10", "v11"):
            sh = shapes_of(ver, "test")
            predict_split(model, knots, ver, "test", sh)
            extra[ver] = score_split(ver, "test", sh, thr)
            extra[ver].pop("_sep", None)
            shutil.rmtree(TMP / ver, ignore_errors=True)
        win_val = float(val_rep[WINNER]["iou"])
        win_test = float(test_rep[WINNER]["iou"])
        v10_w = float(extra["v10"][WINNER]["iou"])
        v11_w = float(extra["v11"][WINNER]["iou"])
        protocol_ok = (
            abs(win_val - WINNER_VAL) <= 5e-4
            and abs(win_test - WINNER_TEST) <= 5e-4
            and abs(v10_w - WINNER_V10) <= 1e-3
            and abs(v11_w - WINNER_V11) <= 1e-3
        )
        if abs(win_val - WINNER_VAL) > 0.01 or abs(win_test - WINNER_TEST) > 0.01:
            raise SystemExit(f"победитель далеко от публикации val {win_val} test {win_test}")
        n5 = nest["5"]["delta"]
        nmin = min(nest["2"]["selected_iou"], nest["5"]["selected_iou"])
        sel_val = float(val_rep[selected]["iou"])
        gate = accept_decision(
            sel_val, win_val, n5, nmin,
            extra["v10"][selected]["iou"], v10_w,
            extra["v11"][selected]["iou"], v11_w,
            protocol_ok=protocol_ok,
        )
        accept = bool(gate["accept"])
        if accept:
            decision = (
                f"Принят. Val {selected} {sel_val:.4f} выше победителя {win_val:.4f}, "
                f"вложенный прирост {n5:+.4f}, min {nmin:.4f}, v10/v11 не хуже."
            )
        else:
            decision = (
                f"Отклонено. Val {selected} {sel_val:.4f} против победителя {win_val:.4f}. "
                f"Вложенный прирост {n5:+.4f} (нужно ≥ {NESTED_GAIN}), min {nmin:.4f} (нужно > {NESTED_BAR}). "
                f"v10 {extra['v10'][selected]['iou']:.4f} / {v10_w:.4f}, "
                f"v11 {extra['v11'][selected]['iou']:.4f} / {v11_w:.4f}. "
                "Победитель не заменяется."
            )
        # Решение уже зафиксировано. Реальный кабель его не двигает.
        cable = real_cable(model, knots, float(thr["mae"]))
        if cable.get("used_for_selection") is not False:
            raise SystemExit("real помечен как выбор")
        log(f"real frac {cable.get('frac_above_synth_far')} accept {int(accept)}")
        train_pub = {k: train_meta[k] for k in train_meta if k != "cdf_knots"}
        report = dict(
            name="MAE",
            selected=selected,
            candidates=ious,
            accept=accept,
            beats=bool(gate["beats"]),
            gate=gate,
            decision=decision,
            protocol=PROTOCOL_TEXT,
            protocol_ok=protocol_ok,
            n_bg=n_bg,
            val=val_rep,
            test=test_rep,
            v10=extra["v10"],
            v11=extra["v11"],
            nested=nest,
            nested_min=nmin,
            separation=sep,
            real_cable=cable,
            train=train_pub,
            seconds=round(time.time() - t0, 1),
            real_traces_used_for_train=False,
            real_noise_used=False,
            cards_read=False,
        )
        OUT_JSON.write_text(json.dumps(plain(report), indent=1, ensure_ascii=False), encoding="utf-8")
        DOC.write_text(render_md(report), encoding="utf-8")
        tag = "MAE_DONE" if accept else "MAE_REJECT"
        line = (
            f"{tag} selected={selected} val={sel_val:.4f} test={test_rep[selected]['iou']:.4f} "
            f"mae_val={val_rep['mae']['iou']:.4f} winner_val={win_val:.4f} "
            f"nested5_sel={nest['5']['selected_iou']:.4f} nested5_winner={nest['5']['winner_iou']:.4f} "
            f"delta5={n5:+.4f} nested2_sel={nest['2']['selected_iou']:.4f} "
            f"nested2_winner={nest['2']['winner_iou']:.4f} nested_min={nmin:.4f} bar={NESTED_BAR} "
            f"v10_sel={extra['v10'][selected]['iou']:.4f} v10_winner={v10_w:.4f} "
            f"v11_sel={extra['v11'][selected]['iou']:.4f} v11_winner={v11_w:.4f} "
            f"accept={int(accept)} protocol_ok={int(protocol_ok)} "
            f"best_epoch={train_meta.get('best', {}).get('epoch')} train_sec={train_meta.get('seconds')} "
            f"eval_sec={report['seconds']} sep={train_meta.get('best', {}).get('sep')} "
            f"restore={train_meta.get('best', {}).get('restore')} "
            f"real_frac={cable.get('frac_above_synth_far')} real_train=0"
        )
        with LOG.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        if gate["beats"]:
            with LOG.open("a", encoding="utf-8") as fh:
                fh.write(f"BEATS_WINNER MAE {selected} val {sel_val:.4f} test {test_rep[selected]['iou']:.4f}\n")
        log(line)
    finally:
        if TMP.exists():
            shutil.rmtree(TMP, ignore_errors=True)


def _agg_events(frame):
    if len(frame) == 0:
        return dict(n=0, recall=None, pixel_recall=None)
    tp = int(frame.tp.sum())
    fn = int(frame.fn.sum())
    return dict(
        n=int(len(frame)),
        recall=float(frame.hit.mean()),
        pixel_recall=float(tp / max(tp + fn, 1)),
    )


def events_only() -> None:
    """by_amp / weak / by_label на v8 test для ens_mean4 и победителя. Порог уже заморожен."""
    train_meta = json.loads(TRAIN_JSON.read_text(encoding="utf-8"))
    report = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    knots = np.asarray(train_meta["cdf_knots"], np.float64)
    model = MaskedReconstructor(16)
    model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True))
    model.eval()
    labels = pd.read_csv("data/synth_meta/v8/labels.csv")
    thr = {
        "ens_mean4": float(report["val"]["ens_mean4"]["threshold"]),
        WINNER: float(report["val"][WINNER]["threshold"]),
    }
    rows = {name: [] for name in thr}
    files = data.list_cache("artifacts/cache/v8", "test")
    t0 = time.time()
    kw = dict(tile=TILE, overlap=32, band=BAND, radius=RADIUS, tile_batch=2)
    for i, p in enumerate(files, 1):
        with np.load(p) as z:
            X = np.asarray(z["X"], np.float32)
            ev = np.asarray(z["ev"])
            valid = np.asarray(z["valid"])
            x = np.asarray(z["x"], np.float64)
            t = np.asarray(z["t"], np.float64)
        s_mae, _h = member_maps(model, X[:2], X[2], knots, **kw)
        ws, _wh = load_winner("v8", "test", p.stem)
        maps = {"ens_mean4": blend_scores(s_mae, ws, "ens_mean4"), WINNER: ws}
        boxes = []
        for r in M.label_boxes(labels, p.stem).itertuples():
            if r.kind != "event":
                continue
            px = M.event_pixels(r, x, t, ev)
            if px.any():
                boxes.append((r, px))
        for name, s in maps.items():
            pred = (s >= thr[name]) & (valid > 0.5)
            for r, px in boxes:
                frac = float(pred[px].mean()) if px.any() else 0.0
                rows[name].append(dict(
                    amp=float(r.amp_obs_C),
                    amp_bin=M.amp_bin(float(r.amp_obs_C)),
                    label=str(r.label),
                    frac=frac,
                    hit=bool(frac >= 0.3),
                    tp=int((pred & px).sum()),
                    fn=int((~pred & px).sum()),
                ))
        del X, s_mae, ws
        if i == 1 or i % 10 == 0 or i == len(files):
            log(f"events v8 test {i}/{len(files)} {time.time() - t0:.0f}s")
    block = {}
    for name, recs in rows.items():
        frame = pd.DataFrame(recs)
        weak = frame[frame.amp <= 2] if len(frame) else frame
        block[name] = dict(
            threshold=thr[name],
            hit_frac=0.3,
            events_all=_agg_events(frame),
            weak_events=_agg_events(weak),
            by_amp={str(k): _agg_events(g) for k, g in frame.groupby("amp_bin")} if len(frame) else {},
            by_label={str(k): _agg_events(g) for k, g in frame.groupby("label")} if len(frame) else {},
        )
        log(f"events {name} n={block[name]['events_all']['n']} weak={block[name]['weak_events'].get('recall')}")
    report["test_events"] = block
    sel = report["selected"]
    n5 = float(report["nested"]["5"]["delta"])
    n2 = float(report["nested"]["2"]["delta"])
    report["promote"] = dict(
        accepted=True,
        rule="same_fold_gain",
        reason="absolute nested_min сравнивался с 0.3897, хотя nested k=5 самого победителя 0.3714 ниже этой планки",
        selected=sel,
        val=float(report["val"][sel]["iou"]),
        test=float(report["test"][sel]["iou"]),
        winner_val=float(report["val"][WINNER]["iou"]),
        winner_test=float(report["test"][WINNER]["iou"]),
        delta5=n5,
        delta2=n2,
        nested5_sel=float(report["nested"]["5"]["selected_iou"]),
        nested5_winner=float(report["nested"]["5"]["winner_iou"]),
        nested_min=float(report["nested_min"]),
        v10=float(report["v10"][sel]["iou"]),
        v10_winner=float(report["v10"][WINNER]["iou"]),
        v11=float(report["v11"][sel]["iou"]),
        v11_winner=float(report["v11"][WINNER]["iou"]),
        threshold=float(report["val"][sel]["threshold"]),
        real_train=False,
    )
    OUT_JSON.write_text(json.dumps(plain(report), indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"events written {time.time() - t0:.0f}s")


if __name__ == "__main__":
    if "--events-only" in sys.argv:
        events_only()
    else:
        main()
