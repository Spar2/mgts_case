"""Оценка банка нормы: FAR 1e-3 на фоне v8 val, смесь с ансамблем, вложенные фолды, v10/v11.

Скор — сигмоида MLP+bilinear по (запрос, сосед, статистика расстояний).
Смесь зафиксирована до val: (4 * ens_u2ft_tcn_lgbm + s) / 5.
Порог — квантиль фона всего v8 val. Test и v10/v11 порог не двигают.
Вложенная проверка — scripts/s3_sel2_cv.py: k=2 и k=5, порог на holdout, micro IoU.
Прирост = min(дельта k=2, дельта k=5). Реальные CSV не читаются.

  /workspace/repo/run/.venv/bin/python -u scripts/s3_membank_eval.py
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
from dts.membank import INFER_TILE, MemBankNet, blend_with_winner

WINNER = "ens_u2ft_tcn_lgbm"
WINNER_VAL = 0.38671235821914546
WINNER_TEST = 0.4052762065230473
WINNER_V10 = 0.4038977883892501
WINNER_V11 = 0.40390028295846503
NESTED_GAIN = 0.003
FLOW = "membank"
BLEND = "ens_blend"
TMP = Path("/tmp/s3_membank_maps")
LOG = Path("/workspace/grok_logs/_events.log")
PRIOR = Path("artifacts/eval/s3_membank_prior.pt")
TRAIN_JSON = Path("artifacts/eval/s3_membank_train.json")


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
        s, h = score_one(model, X)
        if not np.isfinite(s).all():
            raise SystemExit(f"non-finite {p.name}")
        sm = open_map(ver, split, p.stem, "s", shape, np.float32)
        hm = open_map(ver, split, p.stem, "h", shape, np.float16)
        sm[:] = s
        hm[:] = h.astype(np.float16)
        sm.flush()
        hm.flush()
        del sm, hm, X, s, h
        if i == 1 or i % 5 == 0 or i == len(files):
            log(f"pred {ver} {split} {i}/{len(files)} {time.time() - t0:.0f}s")


def score_one(model, x):
    from dts.membank import score_field
    return score_field(model, x, tile=INFER_TILE, overlap=32, batch=1)


def load_winner(ver, split, name):
    with np.load(Path("artifacts/pred") / ver / WINNER / split / name) as z:
        return np.asarray(z["s"], np.float32), np.asarray(z["h"], np.float32)


def pack_scores(ver, split, stem, shape):
    s = np.asarray(read_map(ver, split, stem, "s", shape, np.float32))
    h = np.asarray(read_map(ver, split, stem, "h", shape, np.float16), np.float32)
    ws, wh = load_winner(ver, split, f"{stem}.npz")
    return {
        FLOW: (s, h),
        BLEND: (blend_with_winner(s, ws), h),
        WINNER: (ws, wh),
    }


def thresholds_of(shapes) -> tuple[dict, int]:
    """По одному кандидату: иначе три копии фона (~65e6) сидят в памяти сразу."""
    names = (FLOW, BLEND, WINNER)
    thr = {}
    n_bg = 0
    for n in names:
        parts = []
        n_bg = 0
        for stem, shape in shapes.items():
            with np.load(Path("artifacts/cache/v8/val") / f"{stem}.npz") as meta:
                evm = np.asarray(meta["ev"])
                valid = np.asarray(meta["valid"], bool)
                nu = np.asarray(meta["nu"])
            bg = valid & (evm == 0) & (nu == 0)
            packed = pack_scores("v8", "val", stem, shape)
            parts.append(np.asarray(packed[n][0][bg], np.float32))
            n_bg += int(bg.sum())
            del packed
        scores = np.concatenate(parts)
        del parts
        thr[n] = float(M.far_threshold(scores, 1e-3))
        log(f"порог {n} {thr[n]:.6f} n_bg {n_bg}")
        del scores
    return thr, n_bg


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
        packed = pack_scores(ver, split, stem, shape)
        for n, (ps, ph) in packed.items():
            pred = (ps >= thr[n]) & valid
            tot[n] += np.asarray(M.confusion(pred, truth, valid), np.int64)
            hot[n] += np.asarray(M.confusion(pred & (ph > 0), hot_t, valid), np.int64)
            cold[n] += np.asarray(M.confusion(pred & (ph <= 0), cold_t, valid), np.int64)
            far_n[n] += int((pred & bg).sum())
            far_d[n] += int(bg.sum())
        del packed
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


def nested_ks(shapes, labels, seqs) -> dict:
    files = data.list_cache("artifacts/cache/v8", "val")
    frames = {FLOW: [], BLEND: [], WINNER: []}
    for p in files:
        stem = p.stem
        shape = shapes[stem]
        with np.load(p) as z:
            ev_map = np.asarray(z["ev"])
            valid = np.asarray(z["valid"])
            nu = np.asarray(z["nu"])
            x = np.asarray(z["x"], np.float64)
            t = np.asarray(z["t"], np.float64)
        packed = pack_scores("v8", "val", stem, shape)
        for name in frames:
            frames[name].append(frame_of(stem, packed[name][0], ev_map, valid, nu, x, t, labels, seqs))
        del packed
    order = np.argsort([f["stem"] for f in frames[WINNER]])
    for name in frames:
        frames[name] = [frames[name][i] for i in order]
    out = {}
    for k in (2, 5):
        folds = sel.sequence_folds(len(frames[WINNER]), k)
        parts = {name: [] for name in frames}
        rows = []
        for r, idx in enumerate(folds):
            got = {}
            for name in frames:
                hold = s2.eval_one(s2.take(frames[name], idx))
                parts[name].append(hold)
                got[name] = float(hold["iou"])
            rows.append(dict(
                fold=r,
                winner=got[WINNER],
                membank=got[FLOW],
                blend=got[BLEND],
                delta_membank=got[FLOW] - got[WINNER],
                delta_blend=got[BLEND] - got[WINNER],
            ))
            log(
                f"nested k={k} fold {r} winner {got[WINNER]:.4f} "
                f"membank {got[FLOW]:.4f} blend {got[BLEND]:.4f}"
            )
        micros = {name: float(sel.micro_iou(parts[name])) for name in frames}
        out[str(k)] = dict(
            winner_iou=micros[WINNER],
            membank_iou=micros[FLOW],
            blend_iou=micros[BLEND],
            delta_membank=float(micros[FLOW] - micros[WINNER]),
            delta_blend=float(micros[BLEND] - micros[WINNER]),
            folds=rows,
        )
        log(
            f"nested k={k} micro winner {micros[WINNER]:.4f} "
            f"membank {micros[FLOW] - micros[WINNER]:+.4f} blend {micros[BLEND] - micros[WINNER]:+.4f}"
        )
    def gain(key):
        return float(min(out["2"][key], out["5"][key]))
    out["gain_membank"] = gain("delta_membank")
    out["gain_blend"] = gain("delta_blend")
    out["procedure"] = "s3_sel2_cv.eval_one holdout threshold, sequence_folds i%k, micro_iou, gain=min(k=2,k=5)"
    return out


def event_recall(shapes, thr) -> dict:
    labels = pd.read_csv("data/synth_meta/v8/labels.csv")
    counts = {k: {"n": 0, "hit": 0, "n_weak": 0, "hit_weak": 0} for k in (FLOW, BLEND)}
    for stem, shape in shapes.items():
        with np.load(Path("artifacts/cache/v8/val") / f"{stem}.npz") as z:
            ev_map = np.asarray(z["ev"])
            x = np.asarray(z["x"], np.float64)
            t = np.asarray(z["t"], np.float64)
        packed = pack_scores("v8", "val", stem, shape)
        rows = M.label_boxes(labels, stem)
        for r in rows.itertuples():
            if r.kind != "event":
                continue
            px = M.event_pixels(r, x, t, ev_map)
            if not px.any():
                continue
            weak = float(r.amp_obs_C) <= 2.0
            for name in (FLOW, BLEND):
                hit = float((packed[name][0][px] >= thr[name]).mean()) >= 0.1
                counts[name]["n"] += 1
                counts[name]["hit"] += int(hit)
                counts[name]["n_weak"] += int(weak)
                counts[name]["hit_weak"] += int(hit and weak)
        del packed
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


def gates(val_iou, gain, v10_iou, v10_w, v11_iou, v11_w) -> dict:
    beats_val = float(val_iou) > WINNER_VAL
    nested_ok = float(gain) >= NESTED_GAIN
    v10_ok = float(v10_iou) + 1e-12 >= float(v10_w)
    v11_ok = float(v11_iou) + 1e-12 >= float(v11_w)
    return dict(
        beats_val=beats_val,
        nested_ok=nested_ok,
        v10_ok=v10_ok,
        v11_ok=v11_ok,
        accept=bool(beats_val and nested_ok and v10_ok and v11_ok),
    )


def render_md(rep: dict) -> str:
    def row(name, block):
        b = block[name]
        return f"| {name} | {b['threshold']:.6f} | {b['iou']:.4f} | {b['f1']:.4f} | {b['far']:.5f} |"

    lines = [
        "# S3 MemBank",
        "",
        "Банк нормальных патчей на картах e2 замороженного `artifacts/stage1/unet_v810ft.pt`.",
        "Сырой банк — FIFO не больше 8192 векторов фона synth train. В поиске соседей — корежет 1024.",
        "Опубликованный скор — сигмоида головы (две свёртки 1×1 и билинеар) по запросу, среднему соседу и min/mean/max расстоянию.",
        "Голое расстояние скором не является. Ядро контекста 9×3 зафиксировано.",
        "Смесь с ансамблем зафиксирована до val: (4 × ens_u2ft_tcn_lgbm + s) / 5.",
        "Порог FAR 1e-3 — квантиль фона всего v8 val. v10 и v11 этот порог не двигают.",
        "Вложенный прирост — минимум дельт k=2 и k=5 той же процедуры, что `scripts/s3_sel2_cv.py`.",
        "Реальные CSV в обучение и в выбор порога не входят.",
        "",
        f"Решение: **{'принят' if rep['accept'] else 'отклонён'}** ({rep['selected']}).",
        "",
        rep["decision"],
        "",
        "## Обучение",
        "",
        f"Лучшая эпоха {rep['train'].get('best', {}).get('epoch')}, "
        f"прокси {rep['train'].get('best', {}).get('score')}, "
        f"остановка {rep['train'].get('stopped')}, {rep['train'].get('seconds')} с.",
        f"Энкодер max|Δw| {rep['train'].get('encoder_max_dw')}.",
        f"Банк raw {rep['train'].get('bank_raw')}, прототипов {rep['train'].get('bank_proto')}.",
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
    lines += ["", "| кандидат | test IoU | test F1 | test FAR |", "|---|---:|---:|---:|"]
    for name in (FLOW, BLEND, WINNER):
        b = rep["test"][name]
        lines.append(f"| {name} | {b['iou']:.4f} | {b['f1']:.4f} | {b['far']:.5f} |")
    lines += [
        "",
        "## v10 и v11 test, порог v8",
        "",
        "| версия | membank IoU | ансамбль | победитель |",
        "|---|---:|---:|---:|",
        f"| v10 | {rep['v10'][FLOW]['iou']:.4f} | {rep['v10'][BLEND]['iou']:.4f} | {rep['v10'][WINNER]['iou']:.4f} |",
        f"| v11 | {rep['v11'][FLOW]['iou']:.4f} | {rep['v11'][BLEND]['iou']:.4f} | {rep['v11'][WINNER]['iou']:.4f} |",
        "",
        "## Вложенные фолды",
        "",
        "Порог и IoU на holdout. Micro — сумма tp/fp/fn. Прирост кандидата — min(k=2, k=5).",
        "",
        "| k | победитель | membank | дельта | ансамбль | дельта |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for k in ("2", "5"):
        block = rep["nested"][k]
        lines.append(
            f"| {k} | {block['winner_iou']:.4f} | {block['membank_iou']:.4f} | "
            f"{block['delta_membank']:+.4f} | {block['blend_iou']:.4f} | {block['delta_blend']:+.4f} |"
        )
    evr = rep["events"]
    lines += [
        "",
        f"Прирост membank {rep['nested']['gain_membank']:+.4f}, ансамбля {rep['nested']['gain_blend']:+.4f}. "
        f"Порог принятия {NESTED_GAIN}.",
        "",
        "## События v8 val",
        "",
        "Попадание: не меньше 10% пикселей события выше порога. Это описание, не выбор модели.",
        "",
        f"Банк: recall {evr[FLOW]['recall']:.3f}, ≤2°C {evr[FLOW]['recall_le2c']:.3f} "
        f"(n {evr[FLOW]['n']}, слабых {evr[FLOW]['n_weak']}).",
        f"Ансамбль: recall {evr[BLEND]['recall']:.3f}, ≤2°C {evr[BLEND]['recall_le2c']:.3f}.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    if LOG.exists():
        text = LOG.read_text(encoding="utf-8")
        if "MEMBANK_DONE " in text:
            raise SystemExit("MEMBANK_DONE уже записан")
    if not PRIOR.exists() or not TRAIN_JSON.exists():
        raise SystemExit("нет prior или train json — сначала обучение")
    t0 = time.time()
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    train_meta = json.loads(TRAIN_JSON.read_text())
    model = MemBankNet()
    model.encoder.load_winner()
    blob = torch.load(PRIOR, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["model"])
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    model.eval()
    if int(model.bank_count) < 3:
        raise SystemExit("в prior пустой банк")
    labels = pd.read_csv("data/synth_meta/v8/labels.csv")
    seqs = pd.read_csv("data/synth_meta/v8/sequences.csv").set_index("seq_id")
    if TMP.exists():
        shutil.rmtree(TMP)
    try:
        shapes = shapes_of("v8", "val")
        predict_split(model, "v8", "val", shapes)
        thr, n_bg = thresholds_of(shapes)
        if n_bg != 65095696:
            log(f"внимание n_bg {n_bg}")
        val_rep = score_split("v8", "val", shapes, thr)
        if abs(val_rep[WINNER]["iou"] - WINNER_VAL) > 0.005:
            raise SystemExit(f"победитель val {val_rep[WINNER]['iou']} далёк от {WINNER_VAL}")
        nest = nested_ks(shapes, labels, seqs)
        events = event_recall(shapes, thr)
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
        g_mb = gates(
            val_rep[FLOW]["iou"], nest["gain_membank"],
            extra["v10"][FLOW]["iou"], extra["v10"][WINNER]["iou"],
            extra["v11"][FLOW]["iou"], extra["v11"][WINNER]["iou"],
        )
        g_bl = gates(
            val_rep[BLEND]["iou"], nest["gain_blend"],
            extra["v10"][BLEND]["iou"], extra["v10"][WINNER]["iou"],
            extra["v11"][BLEND]["iou"], extra["v11"][WINNER]["iou"],
        )
        if g_mb["accept"] and g_bl["accept"]:
            selected = FLOW if val_rep[FLOW]["iou"] >= val_rep[BLEND]["iou"] else BLEND
        elif g_mb["accept"]:
            selected = FLOW
        elif g_bl["accept"]:
            selected = BLEND
        else:
            selected = BLEND
        chosen_g = g_mb if selected == FLOW else g_bl
        accept = bool(chosen_g["accept"])
        sel_val = float(val_rep[selected]["iou"])
        win_val = float(val_rep[WINNER]["iou"])
        gain = nest["gain_membank"] if selected == FLOW else nest["gain_blend"]
        if accept:
            decision = (
                f"Принят {selected}. Val {sel_val:.4f} выше {WINNER_VAL:.4f}, "
                f"вложенный прирост {gain:+.4f}, v10/v11 не хуже победителя."
            )
        else:
            decision = (
                f"Отклонено. Val membank {val_rep[FLOW]['iou']:.4f}, смесь {val_rep[BLEND]['iou']:.4f}, "
                f"победитель {win_val:.4f} (порог {WINNER_VAL:.4f}). "
                f"Вложенный прирост membank {nest['gain_membank']:+.4f}, смесь {nest['gain_blend']:+.4f} "
                f"(нужно ≥ {NESTED_GAIN}). "
                f"v10 смесь {extra['v10'][BLEND]['iou']:.4f} / победитель {extra['v10'][WINNER]['iou']:.4f}, "
                f"v11 {extra['v11'][BLEND]['iou']:.4f} / {extra['v11'][WINNER]['iou']:.4f}. "
                "Победитель не заменяется."
            )
        report = dict(
            name="MEMBANK",
            selected=selected,
            accept=accept,
            gates_membank=g_mb,
            gates_blend=g_bl,
            decision=decision,
            n_bg=n_bg,
            val=val_rep,
            test=test_rep,
            v10=extra["v10"],
            v11=extra["v11"],
            nested=nest,
            events=events,
            train={k: train_meta[k] for k in train_meta if k != "history"},
            seconds=round(time.time() - t0, 1),
            real_traces_used_for_train=False,
            real_noise_used=False,
            blend="(4*winner+s)/5",
            published_winner_val=WINNER_VAL,
        )
        out = Path("artifacts/eval/s3_membank_v8.json")
        out.write_text(json.dumps(plain(report), indent=1, ensure_ascii=False))
        Path("docs/S3_MEMBANK.md").write_text(render_md(report))
        decision_word = "accept" if accept else "reject"
        line = (
            f"MEMBANK_DONE decision={decision_word} selected={selected} val={sel_val:.4f} test={test_rep[selected]['iou']:.4f} "
            f"membank_val={val_rep[FLOW]['iou']:.4f} blend_val={val_rep[BLEND]['iou']:.4f} "
            f"winner_val={win_val:.4f} "
            f"nested_gain_membank={nest['gain_membank']:+.4f} nested_gain_blend={nest['gain_blend']:+.4f} "
            f"nested5_blend={nest['5']['blend_iou']:.4f} nested5_winner={nest['5']['winner_iou']:.4f} "
            f"v10_blend={extra['v10'][BLEND]['iou']:.4f} v10_winner={extra['v10'][WINNER]['iou']:.4f} "
            f"v11_blend={extra['v11'][BLEND]['iou']:.4f} v11_winner={extra['v11'][WINNER]['iou']:.4f} "
            f"v10_membank={extra['v10'][FLOW]['iou']:.4f} v11_membank={extra['v11'][FLOW]['iou']:.4f} "
            f"accept={int(accept)} "
            f"event_recall={events[FLOW]['recall']:.4f} weak_recall={events[FLOW]['recall_le2c']:.4f} "
            f"best_epoch={train_meta.get('best', {}).get('epoch')} train_sec={train_meta.get('seconds')} "
            f"eval_sec={report['seconds']} real_train=0\n"
        )
        with LOG.open("a", encoding="utf-8") as fh:
            fh.write(line)
        if accept and sel_val > WINNER_VAL + 1e-4:
            with LOG.open("a", encoding="utf-8") as fh:
                fh.write(f"BEATS_WINNER MEMBANK {selected} val {sel_val:.4f} test {test_rep[selected]['iou']:.4f}\n")
        log(line.strip())
    finally:
        if TMP.exists():
            shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
