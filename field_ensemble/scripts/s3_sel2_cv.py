"""Вложенная проверка SEL и DOM ens_blend по последовательностям v8 val.

Веса (или alpha, или выбор кандидата) считаются только на train-фолде.
IoU и порог FAR 1e-3 — только на оставшихся последовательностях, по их фону.
Компактная подвыборка фона та же, что в scripts/s3_sel_eval.py (до 200k на файл).
Реальные csv не читаются. Большие карты не пишутся.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_k] = "2"

import json
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

import s3_dom_eval as dom
import s3_sel_core as sel
import s3_sel_eval as ev
from dts import data, models

torch.set_num_threads(2)

DISK = [
    "unet", "unet_w24", "unet_v810ft", "unet_cn", "unet_cnsym",
    "unet_cnsymlong", "tcn", "pixel_lgbm",
]
DOM_NAME = "unet_dom"
ENS_NAME = "ens_u2ft_tcn_lgbm"
ASC_NAMES = DISK
ASC_DOM_NAMES = DISK + [DOM_NAME]
WINNER4 = ["unet_w24", "unet_v810ft", "tcn", "pixel_lgbm"]
ASC_DOM_INIT = ["unet_v810ft", "unet_cn", "pixel_lgbm", DOM_NAME]
ALPHAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


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


def _cols(s, ev_idx, nu_idx, bg_idx):
    e, n, b = ev._take(s, ev_idx, nu_idx, bg_idx)
    return e.astype(np.float16), n.astype(np.float16), b.astype(np.float16)


def load_dom():
    model = models.build("unet", 16)
    state = torch.load(ROOT / "artifacts/eval/s3_dom_unet.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def load_frames():
    labels = pd.read_csv(ROOT / "data/synth_meta/v8/labels.csv")
    seqs = pd.read_csv(ROOT / "data/synth_meta/v8/sequences.csv").set_index("seq_id")
    cache = ROOT / "artifacts/cache/v8"
    pred = ROOT / "artifacts/pred/v8"
    model = load_dom()
    files = data.list_cache(cache, "val")
    frames = []
    t0 = time.time()
    names = DISK + [ENS_NAME]
    for i, path in enumerate(files):
        stem = path.stem
        with np.load(path) as z:
            ev_map = np.asarray(z["ev"])
            valid = np.asarray(z["valid"])
            nu = np.asarray(z["nu"])
            x = np.asarray(z["x"], np.float64)
            t = np.asarray(z["t"], np.float64)
            X = np.asarray(z["X"], np.float32)
        ev_idx, nu_idx, bg_idx, n_bg = ev._indices(ev_map, valid, nu, stem)
        cols_e, cols_n, cols_b = [], [], []
        ens_e = ens_n = ens_b = None
        for name in names:
            with np.load(pred / name / "val" / path.name) as q:
                s = np.asarray(q["s"], np.float32)
            if s.shape != ev_map.shape:
                raise SystemExit(f"{name} {s.shape} != {ev_map.shape} {stem}")
            e, n, b = _cols(s, ev_idx, nu_idx, bg_idx)
            if name == ENS_NAME:
                ens_e, ens_n, ens_b = e, n, b
            else:
                cols_e.append(e)
                cols_n.append(n)
                cols_b.append(b)
            del s
        prob = dom.predict_batched(model, X, batch=4)
        s = (1.0 - prob[0]).astype(np.float32)
        if s.shape != ev_map.shape:
            raise SystemExit(f"dom {s.shape} != {ev_map.shape} {stem}")
        e, n, b = _cols(s, ev_idx, nu_idx, bg_idx)
        cols_e.append(e)
        cols_n.append(n)
        cols_b.append(b)
        cooling, _rows = ev._events_for(stem, ev_map, x, t, labels, seqs, ev_idx)
        frames.append(dict(
            stem=stem,
            ev_m=np.stack(cols_e, 0),
            nu_m=np.stack(cols_n, 0),
            bg_m=np.stack(cols_b, 0),
            n_bg=n_bg,
            cooling=cooling,
            ens_ev=ens_e,
            ens_nu=ens_n,
            ens_bg=ens_b,
        ))
        del X, prob, s
        if i == 0 or (i + 1) % 5 == 0 or i + 1 == len(files):
            log(f"compact {i+1}/{len(files)} {stem} {time.time()-t0:.0f}s")
    frames.sort(key=lambda fr: fr["stem"])
    return frames


def take(frames, idxs):
    return [frames[i] for i in idxs]


def project(frames, names):
    """Оставить каналы names. ens не входит в стек."""
    index = {n: i for i, n in enumerate(ASC_DOM_NAMES)}
    cols = [index[n] for n in names]
    out = []
    for fr in frames:
        out.append(dict(
            stem=fr["stem"],
            ev_m=fr["ev_m"][cols],
            nu_m=fr["nu_m"][cols],
            bg_m=fr["bg_m"][cols],
            n_bg=fr["n_bg"],
            cooling=fr["cooling"],
        ))
    return out


def fit_ascent(train, names, inits):
    """Несколько заранее заданных стартов. Выбор старта — по IoU train-фолда."""
    cache = {}

    def score(wdict):
        vec = sel.as_vector(names, wdict)
        key = tuple(np.round(vec, 5))
        if key not in cache:
            cache[key] = sel.evaluate_frames(train, vec)["iou"]
        return cache[key]

    best_w = None
    best = -1.0
    best_init = None
    for init_name, active in inits:
        start = sel.recipe_weights(names, active)
        w, sc = sel.coordinate_ascent(score, names, start, steps=(0.0, 0.05, 0.15, 0.4), rounds=2)
        got = score(w)
        log(f"  ascent {init_name} train IoU {got:.4f}")
        if got > best:
            best = got
            best_w = w
            best_init = init_name
    return best_w, best_init, best


def pack_metrics(m):
    return dict(
        iou=float(m["iou"]), tp=float(m["tp"]), fp=float(m["fp"]), fn=float(m["fn"]),
        threshold=float(m["threshold"]), cooling_hits=int(m["cooling_hits"]),
        cooling_n=int(m["cooling_n"]),
    )


def nested_ascent(frames, names, inits, ks=(2, 5)):
    out = {}
    for k in ks:
        folds = sel.sequence_folds(len(frames), k)
        parts = []
        details = []
        for r, hold_idx in enumerate(folds):
            train_idx = [i for i in range(len(frames)) if i not in set(hold_idx)]
            w, init_name, train_iou = fit_ascent(take(frames, train_idx), names, inits)
            hold = sel.evaluate_frames(take(frames, hold_idx), sel.as_vector(names, w))
            parts.append(hold)
            nz = {n: round(float(v), 4) for n, v in w.items() if v > 1e-4}
            details.append(dict(
                fold=r, n_seq=len(hold_idx), stems=[frames[i]["stem"] for i in hold_idx],
                init=init_name, train_iou=float(train_iou), hold=pack_metrics(hold), weights=nz,
            ))
            log(f"k={k} fold {r} hold IoU {hold['iou']:.4f} cool {hold['cooling_hits']}/{hold['cooling_n']} init {init_name}")
        micro = sel.micro_iou(parts)
        out[str(k)] = dict(micro_iou=micro, folds=details)
        log(f"k={k} micro IoU {micro:.4f}")
    verdict = sel.nested_verdict({k: v["micro_iou"] for k, v in out.items()})
    return dict(by_k=out, verdict=verdict)


def channel_frames(frames, spec):
    """spec: list (name, weight). name='ens' берёт сохранённую карту ансамбля."""
    index = {n: i for i, n in enumerate(ASC_DOM_NAMES)}
    wsum = sum(w for _, w in spec)
    out = []
    for fr in frames:
        evs, nus, bgs = [], [], []
        for name, w in spec:
            coef = float(w) / wsum
            if name == "ens":
                e, n, b = fr["ens_ev"], fr["ens_nu"], fr["ens_bg"]
            else:
                i = index[name]
                e, n, b = fr["ev_m"][i], fr["nu_m"][i], fr["bg_m"][i]
            evs.append(coef * np.asarray(e, np.float32))
            nus.append(coef * np.asarray(n, np.float32))
            bgs.append(coef * np.asarray(b, np.float32))
        out.append(dict(
            ev_m=np.stack([np.sum(evs, 0)], 0),
            nu_m=np.stack([np.sum(nus, 0)], 0),
            bg_m=np.stack([np.sum(bgs, 0)], 0),
            n_bg=fr["n_bg"],
            cooling=fr["cooling"],
        ))
    return out


def eval_one(frames):
    return sel.evaluate_frames(frames, np.array([1.0]))


DOM_CANDIDATES = {
    "ens": [("ens", 1.0)],
    "unet_dom": [(DOM_NAME, 1.0)],
    "ens_replace": [("unet_w24", 1.0), (DOM_NAME, 1.0), ("tcn", 1.0), ("pixel_lgbm", 1.0)],
    "ens_plus": [("unet_w24", 1.0), ("unet_v810ft", 1.0), (DOM_NAME, 1.0), ("tcn", 1.0), ("pixel_lgbm", 1.0)],
    "ens_blend": [("ens", 1.0), (DOM_NAME, 1.0)],
}


def nested_fixed(frames, spec, ks=(2, 5)):
    # Кадры собираются один раз: у фиксированного рецепта нет весов с train.
    made = channel_frames(frames, spec)
    out = {}
    for k in ks:
        folds = sel.sequence_folds(len(made), k)
        parts = []
        details = []
        for r, hold_idx in enumerate(folds):
            hold = eval_one(take(made, hold_idx))
            parts.append(hold)
            details.append(dict(
                fold=r, n_seq=len(hold_idx),
                stems=[frames[i]["stem"] for i in hold_idx],
                hold=pack_metrics(hold),
            ))
            log(f"fixed k={k} fold {r} IoU {hold['iou']:.4f}")
        micro = sel.micro_iou(parts)
        out[str(k)] = dict(micro_iou=micro, folds=details)
        log(f"fixed k={k} micro IoU {micro:.4f}")
    return dict(by_k=out, verdict=sel.nested_verdict({k: v["micro_iou"] for k, v in out.items()}))


def nested_alpha(frames, ks=(2, 5)):
    out = {}
    for k in ks:
        folds = sel.sequence_folds(len(frames), k)
        parts = []
        details = []
        for r, hold_idx in enumerate(folds):
            train_idx = [i for i in range(len(frames)) if i not in set(hold_idx)]
            train = take(frames, train_idx)
            best_a, best_iou = None, -1.0
            for a in ALPHAS:
                m = eval_one(channel_frames(train, [("ens", a), (DOM_NAME, 1.0 - a)]))
                if m["iou"] > best_iou:
                    best_iou = float(m["iou"])
                    best_a = float(a)
            hold = eval_one(channel_frames(take(frames, hold_idx), [("ens", best_a), (DOM_NAME, 1.0 - best_a)]))
            parts.append(hold)
            details.append(dict(
                fold=r, n_seq=len(hold_idx), alpha_ens=best_a, train_iou=best_iou,
                stems=[frames[i]["stem"] for i in hold_idx], hold=pack_metrics(hold),
            ))
            log(f"alpha k={k} fold {r} a={best_a:.2f} train {best_iou:.4f} hold {hold['iou']:.4f}")
        micro = sel.micro_iou(parts)
        out[str(k)] = dict(micro_iou=micro, folds=details)
        log(f"alpha k={k} micro IoU {micro:.4f}")
    return dict(by_k=out, verdict=sel.nested_verdict({k: v["micro_iou"] for k, v in out.items()}))


def nested_dom_select(frames, ks=(2, 5)):
    out = {}
    for k in ks:
        folds = sel.sequence_folds(len(frames), k)
        parts = []
        details = []
        for r, hold_idx in enumerate(folds):
            train_idx = [i for i in range(len(frames)) if i not in set(hold_idx)]
            train = take(frames, train_idx)
            hold_fr = take(frames, hold_idx)
            best_name, best_iou = None, -1.0
            for name, spec in DOM_CANDIDATES.items():
                m = eval_one(channel_frames(train, spec))
                if m["iou"] > best_iou:
                    best_iou = float(m["iou"])
                    best_name = name
            hold = eval_one(channel_frames(hold_fr, DOM_CANDIDATES[best_name]))
            parts.append(hold)
            details.append(dict(
                fold=r, n_seq=len(hold_idx), chosen=best_name, train_iou=best_iou,
                stems=[frames[i]["stem"] for i in hold_idx], hold=pack_metrics(hold),
            ))
            log(f"select k={k} fold {r} {best_name} train {best_iou:.4f} hold {hold['iou']:.4f}")
        micro = sel.micro_iou(parts)
        out[str(k)] = dict(micro_iou=micro, folds=details)
        log(f"select k={k} micro IoU {micro:.4f}")
    return dict(by_k=out, verdict=sel.nested_verdict({k: v["micro_iou"] for k, v in out.items()}))


def main():
    t_all = time.time()
    frames = load_frames()
    if len(frames) != 45:
        raise SystemExit(f"ожидалось 45 последовательностей val, есть {len(frames)}")
    log(f"каналы {ASC_DOM_NAMES} n={frames[0]['ev_m'].shape[0]}")

    winner_frames = project(frames, DISK)
    w4 = sel.as_vector(DISK, sel.recipe_weights(DISK, WINNER4))
    winner_full = sel.evaluate_frames(winner_frames, w4)
    log(f"sanity winner4 compact full-val IoU {winner_full['iou']:.4f}")
    if abs(winner_full["iou"] - 0.3867) > 0.02:
        raise SystemExit("компактный winner4 далеко от 0.3867, загрузка карт сомнительна")

    blend_full = eval_one(channel_frames(frames, DOM_CANDIDATES["ens_blend"]))
    log(f"in-sample ens_blend compact IoU {blend_full['iou']:.4f} (опубликовано 0.3898, точная сортировка)")

    asc_inits = [("winner4", WINNER4)]
    asc_dom_inits = [
        ("winner4", WINNER4),
        ("asc_dom_equal", ASC_DOM_INIT),
    ]
    log("nested asc")
    asc = nested_ascent(winner_frames, DISK, asc_inits)
    log("nested asc+dom")
    asc_dom = nested_ascent(project(frames, ASC_DOM_NAMES), ASC_DOM_NAMES, asc_dom_inits)
    log("nested ens_blend frozen 0.5")
    blend_fixed = nested_fixed(frames, DOM_CANDIDATES["ens_blend"])
    log("nested ens/dom alpha")
    blend_alpha = nested_alpha(frames)
    log("nested DOM candidate select")
    dom_select = nested_dom_select(frames)

    # Оптимистичный подъём на всех 45 — только чтобы показать смещение, не для приёмки.
    log("in-sample ascents")
    w_asc, init_asc, iou_asc = fit_ascent(winner_frames, DISK, asc_inits)
    w_both, init_both, iou_both = fit_ascent(project(frames, ASC_DOM_NAMES), ASC_DOM_NAMES, asc_dom_inits)

    report = dict(
        task="SEL2",
        protocol=(
            "2-fold и 5-fold по отсортированным последовательностям v8 val. "
            "Веса, alpha и выбор кандидата — только на train-фолде. "
            "Порог FAR 1e-3 и IoU — на holdout, фон holdout. "
            "Принятие: min(micro IoU k=2, k=5) > 0.3867+0.003. "
            "Компактный фон до 200000 клеток на файл. "
            "unet_s2 и unet_nm не входят: на полном поиске их вес был 0."
        ),
        bar=sel.NESTED_BAR,
        trained_on_real=False,
        sanity_winner4_compact_iou=float(winner_full["iou"]),
        in_sample=dict(
            note="IoU подъёма на всех 45 последовательностях. Это та же смещённая схема, что val 0.3944. Для приёмки не используется.",
            ens_blend_compact_iou=float(blend_full["iou"]),
            ens_blend_published_exact_iou=0.3898,
            asc=dict(init=init_asc, iou=float(iou_asc), weights={k: v for k, v in w_asc.items() if v > 1e-4}),
            asc_dom=dict(init=init_both, iou=float(iou_both), weights={k: v for k, v in w_both.items() if v > 1e-4}),
        ),
        asc=asc,
        asc_dom=asc_dom,
        ens_blend_fixed=blend_fixed,
        ens_blend_alpha=blend_alpha,
        dom_select=dom_select,
        accept=dict(
            asc=bool(asc["verdict"]["accept"]),
            asc_dom=bool(asc_dom["verdict"]["accept"]),
            ens_blend_fixed=bool(blend_fixed["verdict"]["accept"]),
            ens_blend_alpha=bool(blend_alpha["verdict"]["accept"]),
            dom_select=bool(dom_select["verdict"]["accept"]),
        ),
        seconds=round(time.time() - t_all, 1),
    )
    path = ROOT / "artifacts/eval/s3_sel2_cv.json"
    path.write_text(json.dumps(plain(report), indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"wrote {path}")
    acc = report["accept"]
    log(
        "SEL2_RESULT "
        f"asc {asc['verdict']['worst']:.4f} accept {int(acc['asc'])} "
        f"dom_fixed {blend_fixed['verdict']['worst']:.4f} accept {int(acc['ens_blend_fixed'])} "
        f"dom_alpha {blend_alpha['verdict']['worst']:.4f} accept {int(acc['ens_blend_alpha'])} "
        f"dom_select {dom_select['verdict']['worst']:.4f} accept {int(acc['dom_select'])} "
        f"asc_dom {asc_dom['verdict']['worst']:.4f} accept {int(acc['asc_dom'])}"
    )


if __name__ == "__main__":
    main()
