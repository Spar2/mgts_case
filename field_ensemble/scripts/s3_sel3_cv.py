"""SEL3: победитель и ens_blend под одним вложенным протоколом.

Порог FAR 1e-3 считается по фону обучающих последовательностей val.
IoU считается на отложенных последовательностях с этим замороженным порогом.
Пять перестановок (seed 0..4), k = 2 и k = 5.
Веса не подбираются: winner — сохранённая карта ens_u2ft_tcn_lgbm,
ens_blend_fixed — среднее этой карты и unet_dom.
Реальные csv не читаются.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)

import s3_sel2_cv as s2
import s3_sel_core as sel

SEEDS = (0, 1, 2, 3, 4)
KS = (2, 5)
W = np.array([1.0])


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


def one_repeat(winner, blend, k: int, seed: int) -> dict:
    n = len(winner)
    folds = sel.seeded_folds(n, k, seed)
    parts_w = []
    parts_b = []
    fold_rows = []
    for r, hold_idx in enumerate(folds):
        hold_set = set(hold_idx)
        train_idx = [i for i in range(n) if i not in hold_set]
        train_w = s2.take(winner, train_idx)
        train_b = s2.take(blend, train_idx)
        hold_w = s2.take(winner, hold_idx)
        hold_b = s2.take(blend, hold_idx)
        thr_w = sel.background_threshold(train_w, W)
        thr_b = sel.background_threshold(train_b, W)
        mw = sel.evaluate_frames(hold_w, W, thr=thr_w)
        mb = sel.evaluate_frames(hold_b, W, thr=thr_b)
        parts_w.append(mw)
        parts_b.append(mb)
        fold_rows.append(dict(
            fold=r,
            n_seq=len(hold_idx),
            winner_threshold=thr_w,
            blend_threshold=thr_b,
            winner=dict(iou=float(mw["iou"]), tp=float(mw["tp"]), fp=float(mw["fp"]), fn=float(mw["fn"]), far=float(mw["far"])),
            blend=dict(iou=float(mb["iou"]), tp=float(mb["tp"]), fp=float(mb["fp"]), fn=float(mb["fn"]), far=float(mb["far"])),
        ))
    iw = sel.micro_iou(parts_w)
    ib = sel.micro_iou(parts_b)
    return dict(seed=int(seed), winner_iou=iw, blend_iou=ib, delta=float(ib - iw), folds=fold_rows)


def main():
    t0 = time.time()
    frames = s2.load_frames()
    if len(frames) != 45:
        raise SystemExit(f"ожидалось 45 последовательностей val, есть {len(frames)}")
    winner = s2.channel_frames(frames, [("ens", 1.0)])
    blend = s2.channel_frames(frames, [("ens", 1.0), (s2.DOM_NAME, 1.0)])
    # Сверка с полной выборкой: порог и IoU на всех 45, компактный фон.
    full_w = sel.evaluate_frames(winner, W)
    full_b = sel.evaluate_frames(blend, W)
    log(f"full winner {full_w['iou']:.4f} blend {full_b['iou']:.4f}")
    if abs(full_w["iou"] - 0.3867) > 0.02:
        raise SystemExit("компактный winner далеко от 0.3867")
    if abs(full_b["iou"] - 0.3898) > 0.02:
        raise SystemExit("компактный ens_blend далеко от 0.3898")

    by_k = {}
    for k in KS:
        repeats = []
        for seed in SEEDS:
            row = one_repeat(winner, blend, k, seed)
            repeats.append(row)
            log(f"k={k} seed {seed} winner {row['winner_iou']:.4f} blend {row['blend_iou']:.4f} delta {row['delta']:+.4f}")
        summary = sel.delta_summary([r["delta"] for r in repeats])
        by_k[str(k)] = dict(
            repeats=[{k_: v for k_, v in r.items() if k_ != "folds"} | {"folds": r["folds"]} for r in repeats],
            winner_mean=float(np.mean([r["winner_iou"] for r in repeats])),
            blend_mean=float(np.mean([r["blend_iou"] for r in repeats])),
            delta=summary,
        )
        log(
            f"k={k} mean delta {summary['mean']:+.4f} "
            f"positive {summary['n_positive']}/{summary['n']}"
        )

    report = dict(
        task="SEL3",
        protocol=(
            "Одинаковая процедура для ens_u2ft_tcn_lgbm и ens_blend_fixed. "
            "Пять seed 0..4, перестановка последовательностей v8 val, фолды pos % k. "
            "Порог FAR 1e-3 по фону обучающих последовательностей, IoU на отложенных с этим порогом. "
            "Число повтора — micro IoU по фолдам. "
            "Дельта = IoU ens_blend_fixed − IoU winner. "
            "Компактный фон до 200000 клеток на файл. Веса не учились."
        ),
        seeds=list(SEEDS),
        trained_on_real=False,
        full_val_compact=dict(
            winner_iou=float(full_w["iou"]),
            blend_iou=float(full_b["iou"]),
            note="Порог здесь по всему val. Это сверка загрузки, не вложенная оценка.",
        ),
        by_k=by_k,
        seconds=round(time.time() - t0, 1),
    )
    path = ROOT / "artifacts/eval/s3_sel3_cv.json"
    path.write_text(json.dumps(plain(report), indent=1, ensure_ascii=False), encoding="utf-8")
    k2 = by_k["2"]["delta"]
    k5 = by_k["5"]["delta"]
    log(
        "SEL3_RESULT "
        f"k2 mean_delta {k2['mean']:+.4f} pos {k2['n_positive']}/{k2['n']} "
        f"winner {by_k['2']['winner_mean']:.4f} blend {by_k['2']['blend_mean']:.4f} "
        f"k5 mean_delta {k5['mean']:+.4f} pos {k5['n_positive']}/{k5['n']} "
        f"winner {by_k['5']['winner_mean']:.4f} blend {by_k['5']['blend_mean']:.4f}"
    )
    log(f"wrote {path}")


if __name__ == "__main__":
    main()
