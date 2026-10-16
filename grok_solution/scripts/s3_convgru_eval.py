"""Официальный отбор ConvGRU: v8 val IoU@FAR 1e-3 и вложенные 5 фолдов.

Кандидаты заданы до test. Порог — квантиль фона v8 val.
Вложенная проверка вызывает scripts/s3_sel2_cv.py: те же фолды i%5,
компактный фон до 200000 клеток, порог и IoU только на holdout,
планка sel.NESTED_BAR = 0.3867+0.003. Test в выбор не входит.
Реальные CSV не читаются.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
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

import s3_sel2_cv as cv
import s3_sel_core as sel
import s3_sel_eval as compact
from dts import data, metrics as M
from dts.convgru import ConvGRUNet, predict_full

WINNER = "ens_u2ft_tcn_lgbm"
WINNER_VAL = 0.38671235821914546
WINNER_TEST = 0.4052762065230473
CANDIDATES = ("convgru", "ens_blend", "ens_replace")
REPLACE = ("unet_w24", "convgru", "tcn", "pixel_lgbm")
TMP = Path("/tmp/s3_convgru_maps")
# Каналы компактного кадра для s3_sel2_cv.eval_one.
CH = ("convgru", "ens", "unet_w24", "tcn", "pixel_lgbm")


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


def shapes_of(split: str) -> dict[str, tuple[int, int]]:
    out = {}
    for p in data.list_cache("artifacts/cache/v8", split):
        z = np.load(Path("artifacts/pred/v8") / WINNER / split / p.name)
        out[p.stem] = tuple(int(v) for v in z["s"].shape)
        z.close()
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
        prob = predict_full(model, X)
        s = (1.0 - prob[0]).astype(np.float32)
        h = (prob[1] - prob[2]).astype(np.float16)
        d = TMP / split
        d.mkdir(parents=True, exist_ok=True)
        sm = np.memmap(d / f"{p.stem}_s.mmap", dtype=np.float32, mode="w+", shape=shape)
        hm = np.memmap(d / f"{p.stem}_h.mmap", dtype=np.float16, mode="w+", shape=shape)
        sm[:] = s
        hm[:] = h
        sm.flush()
        hm.flush()
        del sm, hm, prob, s, h, X
        if i == 1 or i % 10 == 0 or i == len(files):
            log(f"pred {split} {i}/{len(files)} {time.time() - t0:.0f}s")


def read_map(split: str, stem: str, key: str, shape, dtype):
    path = TMP / split / f"{stem}_{key}.mmap"
    return np.memmap(path, dtype=dtype, mode="r", shape=shape)


def load_pred(name: str, split: str, stem: str):
    with np.load(Path("artifacts/pred/v8") / name / split / f"{stem}.npz") as z:
        return np.asarray(z["s"], np.float32), np.asarray(z["h"], np.float32)


def conv_maps(split: str, stem: str, shape):
    s = np.asarray(read_map(split, stem, "s", shape, np.float32))
    h = np.asarray(read_map(split, stem, "h", shape, np.float16), np.float32)
    return s, h


def pack_scores(split: str, stem: str, shape) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    ens_s, ens_h = load_pred(WINNER, split, stem)
    cg_s, cg_h = conv_maps(split, stem, shape)
    members = {WINNER: (ens_s, ens_h), "convgru": (cg_s, cg_h)}
    for name in ("unet_w24", "tcn", "pixel_lgbm"):
        members[name] = load_pred(name, split, stem)

    def avg(names):
        s = sum(members[n][0] for n in names) / len(names)
        h = sum(members[n][1] for n in names) / len(names)
        return s.astype(np.float32, copy=False), h.astype(np.float32, copy=False)

    return {
        "convgru": members["convgru"],
        "ens_blend": avg((WINNER, "convgru")),
        "ens_replace": avg(REPLACE),
        WINNER: members[WINNER],
        "_ch": {
            "convgru": members["convgru"][0],
            "ens": members[WINNER][0],
            "unet_w24": members["unet_w24"][0],
            "tcn": members["tcn"][0],
            "pixel_lgbm": members["pixel_lgbm"][0],
        },
    }


def collect_thresholds(shapes) -> dict[str, float]:
    names = list(CANDIDATES) + [WINNER]
    thr = {}
    cache = Path("artifacts/cache/v8/val")
    for n in names:
        parts = []
        for stem, shape in shapes.items():
            with np.load(cache / f"{stem}.npz") as meta:
                ev = np.asarray(meta["ev"])
                nu = np.asarray(meta["nu"])
                valid = np.asarray(meta["valid"], bool)
            mask = valid & (ev == 0) & (nu == 0)
            s = pack_scores("val", stem, shape)[n][0]
            parts.append(np.asarray(s[mask], np.float32))
        bg = np.concatenate(parts)
        thr[n] = float(M.far_threshold(bg, 1e-3))
        log(f"порог {n} {thr[n]:.6f} n_bg {bg.size}")
        del bg, parts
    return thr


def evaluate(split: str, shapes, thr: dict[str, float], labels, seqs, collect: bool):
    names = list(thr)
    tot = {n: np.zeros(3, np.int64) for n in names}
    hot = {n: np.zeros(3, np.int64) for n in names}
    cold = {n: np.zeros(3, np.int64) for n in names}
    far_n = {n: 0 for n in names}
    far_d = {n: 0 for n in names}
    per = {n: {} for n in names}
    ev_rows = {n: [] for n in names}
    frames = []
    cache = Path("artifacts/cache/v8") / split
    t0 = time.time()
    items = list(shapes.items())
    for i, (stem, shape) in enumerate(items, 1):
        with np.load(cache / f"{stem}.npz") as meta:
            ev = np.asarray(meta["ev"])
            valid = np.asarray(meta["valid"], bool)
            nu = np.asarray(meta["nu"])
            x = np.asarray(meta["x"])
            t = np.asarray(meta["t"])
        packed = pack_scores(split, stem, shape)
        truth = ev > 0
        hot_t = (ev >= 1) & (ev <= 4)
        cold_t = ev >= 5
        mask = valid & (ev == 0) & (nu == 0)
        boxes = []
        for r in M.label_boxes(labels, stem).itertuples():
            if r.kind != "event":
                continue
            px = M.event_pixels(r, x, t, ev)
            if px.any():
                boxes.append((r, px))
        for n in names:
            s, h = packed[n]
            pred = (s >= thr[n]) & valid
            tot[n] += np.asarray(M.confusion(pred, truth, valid), np.int64)
            per[n][stem] = M.confusion(pred, truth, valid)
            hot[n] += np.asarray(M.confusion(pred & (h > 0), hot_t, valid), np.int64)
            cold[n] += np.asarray(M.confusion(pred & (h <= 0), cold_t, valid), np.int64)
            far_n[n] += int((pred & mask).sum())
            far_d[n] += int(mask.sum())
            for r, px in boxes:
                frac = float(pred[px].mean())
                ev_rows[n].append(dict(
                    label=str(r.label), amp=float(r.amp_obs_C), amp_bin=M.amp_bin(float(r.amp_obs_C)),
                    frac=frac, hit=bool(frac >= 0.3),
                    tp=int((pred & px).sum()), fn=int((~pred & px).sum()),
                ))
        if collect:
            ev_idx, nu_idx, bg_idx, n_bg = compact._indices(ev, valid, nu, stem)
            cols_e, cols_n, cols_b = [], [], []
            for name in CH:
                e, nu_c, b = compact._take(packed["_ch"][name], ev_idx, nu_idx, bg_idx)
                cols_e.append(e.astype(np.float16))
                cols_n.append(nu_c.astype(np.float16))
                cols_b.append(b.astype(np.float16))
            cooling, _rows = compact._events_for(stem, ev, x, t, labels, seqs, ev_idx)
            frames.append(dict(
                stem=stem,
                ev_m=np.stack(cols_e, 0),
                nu_m=np.stack(cols_n, 0),
                bg_m=np.stack(cols_b, 0),
                n_bg=n_bg,
                cooling=cooling,
            ))
        del packed
        if i == 1 or i % 15 == 0 or i == len(items):
            log(f"eval {split} {i}/{len(items)} {time.time() - t0:.0f}s")
    report = {}
    for n in names:
        E = pd.DataFrame(ev_rows[n])
        row = dict(
            threshold=float(thr[n]),
            far=float(far_n[n] / max(far_d[n], 1)),
            **M.prf(*[int(v) for v in tot[n]]),
            hot=M.prf(*[int(v) for v in hot[n]]),
            cold=M.prf(*[int(v) for v in cold[n]]),
            n_bg=int(far_d[n]),
            n_seq=len(per[n]),
            iou_ci95=M.bootstrap_ci(per[n], sorted(per[n]), n=200, seed=0),
        )
        if len(E):
            def agg(g):
                tp = int(g.tp.sum())
                fn = int(g.fn.sum())
                return dict(n=int(len(g)), recall=float(g.hit.mean()), pixel_recall=float(tp / max(tp + fn, 1)))
            row["events_all"] = agg(E)
            weak = E[E.amp <= 2]
            row["weak_events"] = agg(weak) if len(weak) else dict(n=0, recall=None, pixel_recall=None)
            row["by_amp"] = {str(k): agg(g) for k, g in E.groupby("amp_bin")}
        report[n] = row
        log(f"{split} {n} IoU {row['iou']:.4f} F1 {row['f1']:.4f} FAR {row['far']:.5f} "
            f"hot {row['hot']['iou']:.4f} cold {row['cold']['iou']:.4f}")
    return report, frames


def channel_frames(frames, indexes: tuple[int, ...]):
    """Равная смесь каналов → один канал, формат s3_sel2_cv.eval_one."""
    out = []
    for fr in frames:
        def mix(stack):
            acc = np.zeros(stack.shape[1], np.float32)
            for i in indexes:
                acc += np.asarray(stack[i], np.float32)
            acc /= float(len(indexes))
            return acc
        out.append(dict(
            stem=fr["stem"],
            ev_m=np.stack([mix(fr["ev_m"])], 0),
            nu_m=np.stack([mix(fr["nu_m"])], 0),
            bg_m=np.stack([mix(fr["bg_m"])], 0),
            n_bg=fr["n_bg"],
            cooling=fr["cooling"],
        ))
    return out


def alpha_frames(frames, alpha_ens: float):
    """alpha на карте победителя, остаток на convgru."""
    out = []
    a = float(alpha_ens)
    for fr in frames:
        def mix(stack):
            return a * np.asarray(stack[1], np.float32) + (1.0 - a) * np.asarray(stack[0], np.float32)
        out.append(dict(
            stem=fr["stem"],
            ev_m=np.stack([mix(fr["ev_m"])], 0),
            nu_m=np.stack([mix(fr["nu_m"])], 0),
            bg_m=np.stack([mix(fr["bg_m"])], 0),
            n_bg=fr["n_bg"],
            cooling=fr["cooling"],
        ))
    return out


RECIPES = {
    "convgru": (0,),
    "ens": (1,),
    "ens_blend": (0, 1),
    "ens_replace": (0, 2, 3, 4),
}


def nested_fixed(frames, indexes, k: int = 5) -> dict:
    made = channel_frames(frames, indexes)
    folds = sel.sequence_folds(len(made), k)
    parts, win_parts, details = [], [], []
    winner = channel_frames(frames, (1,))
    for r, hold_idx in enumerate(folds):
        hold = cv.eval_one(cv.take(made, hold_idx))
        base = cv.eval_one(cv.take(winner, hold_idx))
        parts.append(hold)
        win_parts.append(base)
        details.append(dict(
            fold=r, n_seq=len(hold_idx),
            stems=[frames[i]["stem"] for i in hold_idx],
            hold=cv.pack_metrics(hold),
            winner_hold=cv.pack_metrics(base),
            delta=float(hold["iou"] - base["iou"]),
        ))
        log(f"fixed k={k} fold {r} IoU {hold['iou']:.4f} winner {base['iou']:.4f} Δ {hold['iou'] - base['iou']:+.4f}")
    micro = sel.micro_iou(parts)
    win_micro = sel.micro_iou(win_parts)
    verdict = sel.nested_verdict({str(k): micro})
    log(f"fixed k={k} micro {micro:.4f} winner {win_micro:.4f} accept {verdict['accept']}")
    return dict(by_k={str(k): dict(micro_iou=micro, winner_micro_iou=win_micro, delta=float(micro - win_micro), folds=details)}, verdict=verdict)


def nested_select(frames, k: int = 5) -> dict:
    folds = sel.sequence_folds(len(frames), k)
    parts, details = [], []
    names = tuple(RECIPES)
    for r, hold_idx in enumerate(folds):
        train_idx = [i for i in range(len(frames)) if i not in set(hold_idx)]
        best_name, best_iou = None, -1.0
        for name in names:
            m = cv.eval_one(channel_frames(cv.take(frames, train_idx), RECIPES[name]))
            if m["iou"] > best_iou:
                best_iou = float(m["iou"])
                best_name = name
        hold = cv.eval_one(channel_frames(cv.take(frames, hold_idx), RECIPES[best_name]))
        parts.append(hold)
        details.append(dict(
            fold=r, n_seq=len(hold_idx), chosen=best_name, train_iou=best_iou,
            stems=[frames[i]["stem"] for i in hold_idx], hold=cv.pack_metrics(hold),
        ))
        log(f"select k={k} fold {r} {best_name} train {best_iou:.4f} hold {hold['iou']:.4f}")
    micro = sel.micro_iou(parts)
    verdict = sel.nested_verdict({str(k): micro})
    log(f"select k={k} micro {micro:.4f} accept {verdict['accept']}")
    return dict(by_k={str(k): dict(micro_iou=micro, folds=details)}, verdict=verdict)


def nested_alpha(frames, k: int = 5) -> dict:
    folds = sel.sequence_folds(len(frames), k)
    parts, details = [], []
    for r, hold_idx in enumerate(folds):
        train_idx = [i for i in range(len(frames)) if i not in set(hold_idx)]
        train = cv.take(frames, train_idx)
        best_a, best_iou = None, -1.0
        for a in cv.ALPHAS:
            m = cv.eval_one(alpha_frames(train, a))
            if m["iou"] > best_iou:
                best_iou = float(m["iou"])
                best_a = float(a)
        hold = cv.eval_one(alpha_frames(cv.take(frames, hold_idx), best_a))
        parts.append(hold)
        details.append(dict(
            fold=r, n_seq=len(hold_idx), alpha_ens=best_a, train_iou=best_iou,
            stems=[frames[i]["stem"] for i in hold_idx], hold=cv.pack_metrics(hold),
        ))
        log(f"alpha k={k} fold {r} a={best_a:.2f} train {best_iou:.4f} hold {hold['iou']:.4f}")
    micro = sel.micro_iou(parts)
    verdict = sel.nested_verdict({str(k): micro})
    log(f"alpha k={k} micro {micro:.4f} accept {verdict['accept']}")
    return dict(by_k={str(k): dict(micro_iou=micro, folds=details)}, verdict=verdict)


def main() -> None:
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    t0 = time.time()
    ckpt = Path("artifacts/eval/s3_convgru_unet.pt")
    if not ckpt.exists():
        raise SystemExit(f"нет {ckpt}")
    labels = pd.read_csv("data/synth_meta/v8/labels.csv")
    seqs = pd.read_csv("data/synth_meta/v8/sequences.csv").set_index("seq_id")
    model = ConvGRUNet(w=16)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval()
    val_shapes = shapes_of("val")
    test_shapes = shapes_of("test")
    log(f"val {len(val_shapes)} test {len(test_shapes)}")
    if TMP.exists():
        shutil.rmtree(TMP)
    try:
        predict_split(model, "val", val_shapes)
        predict_split(model, "test", test_shapes)
        del model
        names = list(CANDIDATES) + [WINNER]
        thr = collect_thresholds(val_shapes)
        log("метрики val")
        val_rep, frames = evaluate("val", val_shapes, {n: thr[n] for n in names}, labels, seqs, True)
        log("метрики test")
        test_rep, _ = evaluate("test", test_shapes, {n: thr[n] for n in names}, labels, seqs, False)
        frames.sort(key=lambda fr: fr["stem"])
        if len(frames) != 45:
            raise SystemExit(f"компактных кадров {len(frames)}, ожидалось 45")
        log("вложенные 5 фолдов через s3_sel2_cv")
        selected = max(CANDIDATES, key=lambda n: (float(val_rep[n]["iou"]), -CANDIDATES.index(n)))
        fixed = nested_fixed(frames, RECIPES[selected], 5)
        chosen = nested_select(frames, 5)
        alpha = nested_alpha(frames, 5)
    finally:
        if TMP.exists():
            shutil.rmtree(TMP, ignore_errors=True)

    sanity = float(val_rep[WINNER]["iou"])
    sanity_test = float(test_rep[WINNER]["iou"])
    protocol_ok = abs(sanity - WINNER_VAL) < 0.005 and abs(sanity_test - WINNER_TEST) < 0.005
    if not protocol_ok:
        log(f"WARNING winner {sanity:.4f}/{sanity_test:.4f}")
    sel_val = float(val_rep[selected]["iou"])
    sel_test = float(test_rep[selected]["iou"])
    nested_micro = float(fixed["by_k"]["5"]["micro_iou"])
    nested_delta = float(fixed["by_k"]["5"]["delta"])
    # Приёмка — планка s3_sel_core на вложенном 5-fold уже выбранного рецепта.
    # Полный val сам по себе победителя не заменяет: вес рецепта смотрел те же 45 кадров.
    accept = bool(protocol_ok and fixed["verdict"]["accept"])
    beats = bool(accept and sel_val > WINNER_VAL + sel.NESTED_MARGIN)
    train_meta = {}
    tp = Path("artifacts/eval/s3_convgru_train.json")
    if tp.exists():
        train_meta = json.loads(tp.read_text())
    print(
        f"exact selected {selected} val {sel_val:.4f} test {sel_test:.4f} "
        f"nested5 {nested_micro:.4f} delta {nested_delta:+.4f} accept {accept} "
        f"select {chosen['by_k']['5']['micro_iou']:.4f} alpha {alpha['by_k']['5']['micro_iou']:.4f}",
        flush=True,
    )
    report = dict(
        name="CONVGRU",
        protocol=(
            "Выбор кандидата по полному v8 val IoU при FAR 1e-3. "
            "Вложенные 5 фолдов — scripts/s3_sel2_cv.py: sequence_folds, eval_one, "
            "micro_iou, nested_verdict, компактный фон 200000. "
            "Порог holdout только по фону holdout. Планка 0.3867+0.003. Test не выбирает."
        ),
        published_winner=dict(name=WINNER, val_iou=WINNER_VAL, test_iou=WINNER_TEST),
        sanity_winner=dict(val_iou=sanity, test_iou=sanity_test,
                           val_abs_err=abs(sanity - WINNER_VAL), test_abs_err=abs(sanity_test - WINNER_TEST)),
        candidates=list(CANDIDATES),
        candidate_meaning=dict(
            convgru="ConvGRU-U-Net width 16, init unet_v810ft, synth v8+v10 train",
            ens_blend="mean(ens_u2ft_tcn_lgbm, convgru)",
            ens_replace="mean(unet_w24, convgru, tcn, pixel_lgbm)",
        ),
        thresholds=thr,
        val={n: val_rep[n] for n in names},
        test={n: test_rep[n] for n in names},
        selected=selected,
        selected_val_iou=sel_val,
        selected_test_iou=sel_test,
        protocol_ok=protocol_ok,
        nested_5fold_fixed=fixed,
        nested_5fold_select=chosen,
        nested_5fold_alpha=alpha,
        nested_bar=sel.NESTED_BAR,
        nested_margin=sel.NESTED_MARGIN,
        accept_nested=accept,
        beats_winner=beats,
        train_best_epoch=train_meta.get("best_epoch"),
        train_best_proxy_iou=train_meta.get("best_proxy_iou"),
        train_seconds=train_meta.get("seconds"),
        train_stopped=train_meta.get("stopped"),
        real_used_for_training=False,
        real_used_for_threshold=False,
        cards="not used for selection",
        seconds=round(time.time() - t0, 1),
    )
    out = Path("artifacts/eval/s3_convgru_v8.json")
    out.write_text(json.dumps(plain(report), indent=1, ensure_ascii=False))
    line = (
        f"CONVGRU_DONE selected={selected} val_iou={sel_val:.4f} test_iou={sel_test:.4f} "
        f"thr={float(val_rep[selected]['threshold']):.6f} "
        f"val_far={float(val_rep[selected]['far']):.6f} test_far={float(test_rep[selected]['far']):.6f} "
        f"convgru_val={float(val_rep['convgru']['iou']):.4f} "
        f"ens_blend_val={float(val_rep['ens_blend']['iou']):.4f} "
        f"ens_replace_val={float(val_rep['ens_replace']['iou']):.4f} "
        f"nested5={nested_micro:.4f} nested_delta={nested_delta:+.4f} "
        f"nested_select={float(chosen['by_k']['5']['micro_iou']):.4f} "
        f"nested_alpha={float(alpha['by_k']['5']['micro_iou']):.4f} "
        f"bar={sel.NESTED_BAR:.4f} accept={str(accept).lower()} "
        f"winner_val={sanity:.4f} beats_winner={str(beats).lower()} epoch={train_meta.get('best_epoch')}"
    )
    elog = Path("/workspace/grok_logs/_events.log")
    with elog.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
        if beats:
            f.write(f"BEATS_WINNER CONVGRU val {sel_val:.4f} test {sel_test:.4f}\n")
    print(line, flush=True)
    log(f"записан {out}")


if __name__ == "__main__":
    main()
