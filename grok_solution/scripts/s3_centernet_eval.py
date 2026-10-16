"""Оценка CenterNet: v8 val IoU@FAR 1e-3 в ансамбле, вложенные 5 фолдов, v10/v11.

Кандидат — среднее карты рамок и сохранённого ens_u2ft_tcn_lgbm. Alpha 0.5 заранее,
на val не подбирается. Порог каждой карты — квантиль фона всего v8 val.
На v10 и v11 порог заморожен. Вложенная проверка: фолды i%5, порог и IoU на holdout,
micro по сумме tp/fp/fn, прирост к победителю не меньше 0.003.
Test в выбор не входит. Реальные CSV не читаются.

  /workspace/repo/run/.venv/bin/python -u scripts/s3_centernet_eval.py
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_k] = "2"

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)

import s3_sel_core as sel
import s3_sel_eval as ev
from dts import data, metrics as M
from dts.centernet import STRUCT8, CenterNetHead, load_encoder, predict_score

WINNER = "ens_u2ft_tcn_lgbm"
WINNER_VAL = 0.38671235821914546
WINNER_TEST = 0.4052762065230473
WINNER_V10 = 0.4038977883892501
WINNER_V11 = 0.40390028295846503
NESTED_GAIN = 0.003
PUB_MARGIN = 1e-4
HIT_FRAC = 0.3
MIN_ZONE = 12
NAMES = ("centernet", "ens_blend", WINNER)
TMP = Path("/tmp/s3_centernet_maps")
LOG = Path("/workspace/grok_logs/_events.log")
HEAD = Path("artifacts/eval/s3_centernet2_head.pt")


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S', time.gmtime())} {msg}", flush=True)


def elog(msg: str) -> None:
    line = f"{time.strftime('%H:%M', time.gmtime())} {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def has_tag(tag: str) -> bool:
    if not LOG.exists():
        return False
    return tag in LOG.read_text(encoding="utf-8", errors="ignore")


def plain(obj):
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return plain(obj.tolist())
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def load_head(path: Path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if blob.get("arch") != "centernet":
        raise SystemExit(f"не голова CenterNet: {path}")
    head = CenterNetHead(cin=int(blob["cin"]), mid=int(blob["mid"]), n_class=int(blob["n_class"]))
    head.load_state_dict(blob["head"])
    head.eval()
    net = load_encoder(blob["encoder"], int(blob["width"]))
    return net, head, blob


def shapes_of(ver: str, split: str) -> dict:
    out = {}
    for p in data.list_cache(f"artifacts/cache/{ver}", split):
        wp = Path("artifacts/pred") / ver / WINNER / split / p.name
        with np.load(wp) as z:
            out[p.stem] = tuple(z["s"].shape)
    return out


def map_path(ver: str, split: str, stem: str) -> Path:
    return TMP / f"{ver}_{split}" / f"{stem}_s.mmap"


def write_map(ver, split, stem, score):
    path = map_path(ver, split, stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    mm = np.memmap(path, dtype=np.float32, mode="w+", shape=score.shape)
    mm[:] = score
    mm.flush()
    del mm


def read_map(ver, split, stem, shape):
    return np.memmap(map_path(ver, split, stem), dtype=np.float32, mode="r", shape=shape)


def predict_split(net, head, ver, split, shapes) -> list[dict]:
    t0 = time.time()
    files = data.list_cache(f"artifacts/cache/{ver}", split)
    infos = []
    for i, p in enumerate(files, 1):
        shape = shapes[p.stem]
        with np.load(p) as z:
            x = np.asarray(z["X"], np.float32)
        if x.shape[1:] != shape:
            raise SystemExit(f"{ver}/{split}/{p.name}: X {x.shape[1:]} != pred {shape}")
        score, _hot, info = predict_score(net, head, x, tile=(256, 512), overlap=64, batch=4)
        if score.shape != shape or not np.isfinite(score).all():
            raise SystemExit(f"плохая карта {p.name}")
        write_map(ver, split, p.stem, score)
        info = dict(stem=p.stem, **info)
        infos.append(info)
        del x, score
        if i == 1 or i % 5 == 0 or i == len(files):
            log(f"pred {ver}/{split} {i}/{len(files)} {time.time() - t0:.0f}s det {info['n_det']}")
    return infos


def load_winner(ver, split, stem):
    with np.load(Path("artifacts/pred") / ver / WINNER / split / f"{stem}.npz") as z:
        return np.asarray(z["s"], np.float32)


def score_of(name, s, ws):
    if name == "centernet":
        return s
    if name == "ens_blend":
        return (0.5 * (s + ws)).astype(np.float32)
    if name == WINNER:
        return ws
    raise KeyError(name)


def meta_of(ver, split, stem):
    with np.load(Path("artifacts/cache") / ver / split / f"{stem}.npz") as z:
        return (
            np.asarray(z["ev"]),
            np.asarray(z["valid"], bool),
            np.asarray(z["nu"]),
            np.asarray(z["x"], np.float64),
            np.asarray(z["t"], np.float64),
        )


def thresholds_of(ver, split, shapes) -> tuple[dict, int]:
    files = data.list_cache(f"artifacts/cache/{ver}", split)
    parts = {n: [] for n in NAMES}
    n_bg = 0
    for p in files:
        ev_map, valid, nu, _x, _t = meta_of(ver, split, p.stem)
        bg = valid & (ev_map == 0) & (nu == 0)
        s = np.asarray(read_map(ver, split, p.stem, shapes[p.stem]))
        ws = load_winner(ver, split, p.stem)
        for n in NAMES:
            parts[n].append(np.asarray(score_of(n, s, ws)[bg], np.float32))
        n_bg += int(bg.sum())
        del s, ws
    thr = {}
    for n in NAMES:
        bg = np.concatenate(parts[n])
        thr[n] = float(M.far_threshold(bg, 1e-3))
        log(f"порог {n} {thr[n]:.6f} n_bg {n_bg}")
        del bg
    return thr, n_bg


def _agg(g: pd.DataFrame) -> dict:
    if len(g) == 0:
        return dict(n=0, recall=None, pixel_recall=None)
    tp, fn = int(g.tp.sum()), int(g.fn.sum())
    return dict(n=int(len(g)), recall=float(g.hit.mean()), pixel_recall=float(tp / max(tp + fn, 1)))


def event_zones(pred: np.ndarray, truth: np.ndarray):
    """Связные компоненты предсказания от 12 клеток. TP, если ≥30 % площади на событии."""
    lab, n = ndimage.label(pred, structure=STRUCT8)
    if n <= 0:
        return 0, 0
    tp = fp = 0
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is None:
            continue
        comp = lab[sl] == i
        area = int(comp.sum())
        if area < MIN_ZONE:
            continue
        inter = int((comp & truth[sl]).sum())
        if inter / area >= HIT_FRAC:
            tp += 1
        else:
            fp += 1
    return tp, fp


def gt_boxes(labels, stem, ev_map, x, t):
    boxes = []
    for r in M.label_boxes(labels, stem).itertuples():
        if r.kind != "event":
            continue
        px = M.event_pixels(r, x, t, ev_map)
        if px.any():
            boxes.append((float(r.amp_obs_C), px))
    return boxes


def score_split(ver, split, shapes, thr, labels) -> dict:
    files = data.list_cache(f"artifacts/cache/{ver}", split)
    tot = {n: np.zeros(3, np.int64) for n in NAMES}
    far_n = {n: 0 for n in NAMES}
    far_d = {n: 0 for n in NAMES}
    ev_rows = {n: [] for n in NAMES}
    zone_tp = {n: 0 for n in NAMES}
    zone_fp = {n: 0 for n in NAMES}
    pos_frac = []
    for p in files:
        ev_map, valid, nu, x, t = meta_of(ver, split, p.stem)
        bg = valid & (ev_map == 0) & (nu == 0)
        truth = ev_map > 0
        s = np.asarray(read_map(ver, split, p.stem, shapes[p.stem]))
        ws = load_winner(ver, split, p.stem)
        pos_frac.append(float((s > 0).mean()))
        boxes = gt_boxes(labels, p.stem, ev_map, x, t) if labels is not None else []
        for n in NAMES:
            ps = score_of(n, s, ws)
            pred = (ps >= thr[n]) & valid
            tot[n] += np.asarray(M.confusion(pred, truth, valid), np.int64)
            far_n[n] += int((pred & bg).sum())
            far_d[n] += int(bg.sum())
            ztp, zfp = event_zones(pred, truth)
            zone_tp[n] += ztp
            zone_fp[n] += zfp
            for amp, px in boxes:
                frac = float(pred[px].mean())
                ev_rows[n].append(dict(
                    amp=amp, amp_bin=M.amp_bin(amp), hit=frac >= HIT_FRAC,
                    tp=int((pred & px).sum()), fn=int((~pred & px).sum()),
                ))
            del ps, pred
        del s, ws
    report = {}
    for n in NAMES:
        E = pd.DataFrame(ev_rows[n])
        zden = zone_tp[n] + zone_fp[n]
        row = dict(
            threshold=float(thr[n]),
            far=float(far_n[n] / max(far_d[n], 1)),
            n_bg=int(far_d[n]),
            **M.prf(*tot[n]),
            tp=int(tot[n][0]), fp=int(tot[n][1]), fn=int(tot[n][2]),
            event_zone_tp=int(zone_tp[n]),
            event_zone_fp=int(zone_fp[n]),
            event_precision=(float(zone_tp[n] / zden) if zden else None),
        )
        if len(E):
            row["events_all"] = _agg(E)
            row["event_recall"] = row["events_all"]["recall"]
            row["weak_events"] = _agg(E[E.amp <= 2]) if (E.amp <= 2).any() else dict(n=0, recall=None)
            row["by_amp"] = {k: _agg(g) for k, g in E.groupby("amp_bin")}
        else:
            row["events_all"] = dict(n=0, recall=None)
            row["event_recall"] = None
            row["weak_events"] = dict(n=0, recall=None)
            row["by_amp"] = {}
        report[n] = row
        log(
            f"{ver}/{split} {n} IoU {row['iou']:.4f} FAR {row['far']:.5f} "
            f"evR {row['event_recall']} evP {row['event_precision']}"
        )
    report["_pos_frac_centernet"] = float(np.mean(pos_frac)) if pos_frac else 0.0
    return report


def compact_frames(shapes):
    files = data.list_cache("artifacts/cache/v8", "val")
    rows = {n: [] for n in ("centernet", "ens_blend", WINNER)}
    for p in files:
        ev_map, valid, nu, _x, _t = meta_of("v8", "val", p.stem)
        ev_idx, nu_idx, bg_idx, n_bg = ev._indices(ev_map, valid, nu, p.stem)
        s = np.asarray(read_map("v8", "val", p.stem, shapes[p.stem]))
        ws = load_winner("v8", "val", p.stem)

        def frame(arr):
            flat = np.asarray(arr, np.float32).ravel()
            return dict(
                ev_m=flat[ev_idx][None], nu_m=flat[nu_idx][None], bg_m=flat[bg_idx][None],
                n_bg=n_bg, cooling=[], stem=p.stem,
            )

        rows["centernet"].append(frame(s))
        rows["ens_blend"].append(frame((0.5 * (s + ws)).astype(np.float32)))
        rows[WINNER].append(frame(ws))
        del s, ws
    order = np.argsort([fr["stem"] for fr in rows[WINNER]])

    def reorder(items):
        return [items[i] for i in order]

    return {n: reorder(v) for n, v in rows.items()}


def nested5(frames) -> dict:
    folds = sel.sequence_folds(len(frames[WINNER]), 5)
    parts = {n: [] for n in ("ens_blend", WINNER)}
    details = []
    for r, hold_idx in enumerate(folds):
        fold = dict(fold=r, n_seq=len(hold_idx), stems=[frames[WINNER][i]["stem"] for i in hold_idx])
        for n in ("ens_blend", WINNER):
            got = sel.evaluate_frames([frames[n][i] for i in hold_idx], np.array([1.0]))
            parts[n].append(got)
            fold[n] = dict(iou=float(got["iou"]), tp=float(got["tp"]), fp=float(got["fp"]), fn=float(got["fn"]))
        fold["delta"] = float(fold["ens_blend"]["iou"] - fold[WINNER]["iou"])
        details.append(fold)
        log(f"nested k=5 fold {r} blend {fold['ens_blend']['iou']:.4f} winner {fold[WINNER]['iou']:.4f} d {fold['delta']:+.4f}")
    ib = float(sel.micro_iou(parts["ens_blend"]))
    iw = float(sel.micro_iou(parts[WINNER]))
    return dict(k=5, blend_iou=ib, winner_iou=iw, delta=float(ib - iw), folds=details)


def load_labels(ver: str):
    path = Path(f"data/synth_meta/{ver}/labels.csv")
    if not path.exists():
        return None
    return pd.read_csv(path)


def render_md(report) -> str:
    v, t = report["val"], report["test"]
    nest = report["nested5"]
    acc = "принят" if report["accept"] else "отклонён"
    tr = report["train"]
    lines = [
        "# S3 CenterNet",
        "",
        "Облегчённый LIT6 D. Голова — Objects as Points (Zhou, Wang, Krähenbühl, CVPR 2019):",
        "тепловая карта центров, log-размер (Δt, Δx) и дробный сдвиг внутри клетки.",
        "Энкодер — замороженный `artifacts/stage1/unet_v810ft.pt` (ширина 16, член победителя).",
        "Признаки на шаге 2: блок e2 и ближайший подъём e3. Декодер не учится и не считается.",
        "TimeGate (e4) не берётся: он усредняет по оси x, и на тайле это не тот же признак, что на всём кабеле.",
        "Два канала карты — нагрев и охлаждение. Пики — 3×3 max-pool, до 100 на класс, порог пика 0.05.",
        "Это отбор пиков из статьи, не NMS рамок. Растеризация: внутри рамки уверенность пика, снаружи 0.",
        "",
        "Обучение только на synth v8 и v10 train. Val, test и реальные CSV в потерю не входят.",
        "CENTERNET2: до 20000 шагов или 45 минут, фокальная потеря с гауссианой по размеру рамки,",
        "батч только если есть положительный центр и не меньше половины окон содержат событие.",
        "Запуск через `scripts/run_resumable.sh centernet2`, `--resume auto`, снимок `artifacts/ckpt/centernet2/last.pt`.",
        "",
        "Кандидат ансамбля — среднее этой карты и сохранённого `ens_u2ft_tcn_lgbm`. Вес 0.5 заранее.",
        "Порог официальной точки — квантиль фона всего v8 val при FAR 1e-3, у каждой карты свой.",
        "На v10 и v11 порог не пересчитывается. Test в выбор не входит.",
        "Вложенная проверка — 5 фолдов по имени последовательности v8 val, фолд `i % 5`.",
        "Порог FAR 1e-3 и IoU считаются на holdout, micro — по сумме tp/fp/fn.",
        "Принятие: val IoU ансамбля строго выше пересчитанного победителя, прирост micro не меньше 0.003,",
        "и test IoU на v10 и v11 не хуже победителя при замороженном пороге.",
        "",
        f"Решение: **{acc}**.",
        "",
        report["decision"],
        "",
        "## Обучение",
        "",
        f"Шаг {tr.get('step')}, остановка {tr.get('stopped')}, {tr.get('seconds')} с, потоки {tr.get('threads')}.",
        f"Потеря в конце {tr.get('final_loss')}. Прокси центра на train-тайле (не отбор) — шаг {((tr.get('best_proxy') or {}).get('step'))}.",
        "В оценку идёт последний шаг, не прокси. Реальные трассы не читались.",
        "",
        "## Официальный v8",
        "",
        f"Фон {report['n_bg']} клеток. Доля клеток CenterNet со скором > 0 на val: {report['pos_frac_val']:.4f}.",
        f"Среднее число пиков на val-последовательность: {report['mean_dets_val']:.1f}.",
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
        "### События v8 val",
        "",
        "Recall — доля событий разметки, у которых не меньше 30 % пикселей выше порога.",
        "Precision — доля связных компонент предсказания от 12 пикселей, у которых не меньше 30 % площади лежит на событии.",
        "",
        "| кандидат | event recall | event precision | зоны TP/FP | слабые ≤2 °C |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in NAMES:
        er = v[name].get("event_recall")
        ep = v[name].get("event_precision")
        weak = (v[name].get("weak_events") or {}).get("recall")
        er_s = "—" if er is None else f"{er:.3f} ({v[name]['events_all']['n']})"
        ep_s = "—" if ep is None else f"{ep:.3f}"
        weak_s = "—" if weak is None else f"{weak:.3f}"
        lines.append(
            f"| {name} | {er_s} | {ep_s} | {v[name]['event_zone_tp']}/{v[name]['event_zone_fp']} | {weak_s} |"
        )
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
            f"| {row['fold']} | {row['n_seq']} | {row[WINNER]['iou']:.4f} | "
            f"{row['ens_blend']['iou']:.4f} | {row['delta']:+.4f} |"
        )
    lines += [
        "",
        "## v10 и v11, порог v8 val",
        "",
        "| набор | победитель IoU | ансамбль IoU | CenterNet IoU | не хуже |",
        "|---|---:|---:|---:|---|",
        f"| v10 test | {report['v10'][WINNER]['iou']:.4f} | {report['v10']['ens_blend']['iou']:.4f} | "
        f"{report['v10']['centernet']['iou']:.4f} | {'да' if report['v10_ok'] else 'нет'} |",
        f"| v11 test | {report['v11'][WINNER]['iou']:.4f} | {report['v11']['ens_blend']['iou']:.4f} | "
        f"{report['v11']['centernet']['iou']:.4f} | {'да' if report['v11_ok'] else 'нет'} |",
        "",
        "Выпуск не менялся: `ens_u2ft_tcn_lgbm`. Голова не подмешана в `dts/infer.py`.",
        "",
        "Файлы: `dts/centernet.py`, `scripts/s3_centernet_train.py`, `scripts/s3_centernet_eval.py`,",
        "`artifacts/eval/s3_centernet2_train.json`, `artifacts/eval/s3_centernet_v8.json`,",
        "`artifacts/eval/s3_centernet2_head.pt`, `artifacts/ckpt/centernet2/last.pt`.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default=str(HEAD))
    ap.add_argument("--out", default="artifacts/eval/s3_centernet_v8.json")
    ap.add_argument("--resume", default="")
    args = ap.parse_args()
    if has_tag("CENTERNET2_DONE"):
        raise SystemExit("CENTERNET2_DONE уже есть, вторую строку не пишу")
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    t0 = time.time()
    net, head, blob = load_head(Path(args.head))
    elog(f"CENTERNET2_EVAL head={args.head} step={blob.get('step')} descriptive_protocol=v8+nested5+v10+v11")
    if TMP.exists():
        shutil.rmtree(TMP)
    try:
        val_shapes = shapes_of("v8", "val")
        test_shapes = shapes_of("v8", "test")
        v10_shapes = shapes_of("v10", "test")
        v11_shapes = shapes_of("v11", "test")
        infos = predict_split(net, head, "v8", "val", val_shapes)
        thr, n_bg = thresholds_of("v8", "val", val_shapes)
        labels8 = load_labels("v8")
        val_rep = score_split("v8", "val", val_shapes, thr, labels8)
        if abs(val_rep[WINNER]["iou"] - WINNER_VAL) > 0.005:
            raise SystemExit(f"пересчёт победителя на val далёк от публикации: {val_rep[WINNER]['iou']}")
        frames = compact_frames(val_shapes)
        if len(frames[WINNER]) != 45:
            raise SystemExit(f"ожидалось 45 val, есть {len(frames[WINNER])}")
        nest = nested5(frames)
        predict_split(net, head, "v8", "test", test_shapes)
        test_rep = score_split("v8", "test", test_shapes, thr, labels8)
        if abs(test_rep[WINNER]["iou"] - WINNER_TEST) > 0.005:
            raise SystemExit(f"пересчёт победителя на test далёк от публикации: {test_rep[WINNER]['iou']}")
        predict_split(net, head, "v10", "test", v10_shapes)
        v10_rep = score_split("v10", "test", v10_shapes, thr, load_labels("v10"))
        if abs(v10_rep[WINNER]["iou"] - WINNER_V10) > 0.005:
            raise SystemExit(f"v10 победитель далёк от публикации: {v10_rep[WINNER]['iou']}")
        predict_split(net, head, "v11", "test", v11_shapes)
        v11_rep = score_split("v11", "test", v11_shapes, thr, load_labels("v11"))
        if abs(v11_rep[WINNER]["iou"] - WINNER_V11) > 0.005:
            raise SystemExit(f"v11 победитель далёк от публикации: {v11_rep[WINNER]['iou']}")
    finally:
        if TMP.exists():
            shutil.rmtree(TMP, ignore_errors=True)

    sel_val = float(val_rep["ens_blend"]["iou"])
    win_val = float(val_rep[WINNER]["iou"])
    beats_val = bool(sel_val > win_val)
    nested_ok = bool(nest["delta"] >= NESTED_GAIN)
    v10_ok = bool(v10_rep["ens_blend"]["iou"] + 1e-6 >= v10_rep[WINNER]["iou"])
    v11_ok = bool(v11_rep["ens_blend"]["iou"] + 1e-6 >= v11_rep[WINNER]["iou"])
    accept = bool(beats_val and nested_ok and v10_ok and v11_ok)
    pub_beat = bool(sel_val > WINNER_VAL + PUB_MARGIN)
    if accept:
        decision = (
            f"Принято. Ансамбль val IoU {sel_val:.6f} выше победителя {win_val:.6f}, "
            f"вложенный прирост {nest['delta']:+.4f}, v10 и v11 не хуже."
        )
    elif beats_val and not nested_ok:
        decision = (
            f"Отклонено вложенной проверкой. Полный val ансамбля {sel_val:.6f} выше победителя {win_val:.6f}, "
            f"но прирост 5-fold {nest['delta']:+.4f} меньше 0.003. Выпуск не заменяется."
        )
    elif beats_val and not (v10_ok and v11_ok):
        decision = (
            f"Отклонено переносом. Val ансамбля {sel_val:.6f} выше победителя {win_val:.6f}, "
            f"вложенный прирост {nest['delta']:+.4f}, но v10/v11 хуже. Выпуск не заменяется."
        )
    else:
        decision = (
            f"Отклонено. Val ансамбля {sel_val:.6f} не выше победителя {win_val:.6f}. "
            f"Вложенный прирост {nest['delta']:+.4f}. v10 {'не хуже' if v10_ok else 'хуже'}, "
            f"v11 {'не хуже' if v11_ok else 'хуже'}. Выпуск не заменяется."
        )
    train_meta = {}
    tp = Path("artifacts/eval/s3_centernet2_train.json")
    if tp.exists():
        train_meta = json.loads(tp.read_text(encoding="utf-8"))
    pos = val_rep.pop("_pos_frac_centernet", 0.0)
    test_rep.pop("_pos_frac_centernet", None)
    v10_pos = v10_rep.pop("_pos_frac_centernet", None)
    v11_pos = v11_rep.pop("_pos_frac_centernet", None)
    report = dict(
        name="CENTERNET2",
        protocol=(
            "Official score is v8 val pixel IoU at FAR 1e-3. "
            "The candidate is the mean of the rasterized CenterNet map and ens_u2ft_tcn_lgbm, alpha 0.5 fixed. "
            "Each threshold is the full v8-val background quantile and is frozen for v8 test, v10 and v11. "
            "Nested 5-fold uses holdout background only, micro IoU, and requires a gain of at least 0.003. "
            "v10 and v11 test IoU of the blend must be no worse than the winner. "
            "Test is not used to choose the threshold. Real CSVs are not read."
        ),
        published_winner=dict(name=WINNER, val_iou=WINNER_VAL, test_iou=WINNER_TEST,
                              v10_iou=WINNER_V10, v11_iou=WINNER_V11),
        head=dict(step=blob.get("step"), encoder=blob.get("encoder"), stride=blob.get("stride"),
                  peak_min=blob.get("peak_min"), topk=blob.get("topk")),
        n_bg=n_bg,
        pos_frac_val=pos,
        pos_frac_v10=v10_pos,
        pos_frac_v11=v11_pos,
        mean_dets_val=float(np.mean([r["n_det"] for r in infos])) if infos else 0.0,
        thresholds=thr,
        val=val_rep,
        test=test_rep,
        v10=v10_rep,
        v11=v11_rep,
        nested5=nest,
        nested_gain_bar=NESTED_GAIN,
        selected="ens_blend",
        selected_val_iou=sel_val,
        selected_test_iou=float(test_rep["ens_blend"]["iou"]),
        beats_val=beats_val,
        nested_ok=nested_ok,
        v10_ok=v10_ok,
        v11_ok=v11_ok,
        accept=accept,
        pub_margin_cleared=pub_beat,
        decision=decision,
        train=dict(
            step=train_meta.get("step"),
            seconds=train_meta.get("seconds"),
            stopped=train_meta.get("stopped"),
            final_loss=train_meta.get("final_loss"),
            best_proxy=train_meta.get("best_proxy"),
            threads=train_meta.get("threads"),
            real_traces_used=False,
        ),
        seconds=round(time.time() - t0, 1),
        real_traces_used=False,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plain(report), indent=1, ensure_ascii=False), encoding="utf-8")
    Path("docs/S3_CENTERNET.md").write_text(render_md(report), encoding="utf-8")
    er = val_rep["ens_blend"].get("event_recall")
    ep = val_rep["ens_blend"].get("event_precision")
    er_s = "na" if er is None else f"{er:.3f}"
    ep_s = "na" if ep is None else f"{ep:.3f}"
    line = (
        f"CENTERNET2_DONE selected=ens_blend step={blob.get('step')} "
        f"val_iou={sel_val:.6f} dval={sel_val - win_val:+.6f} "
        f"test_iou={test_rep['ens_blend']['iou']:.6f} "
        f"winner_val={win_val:.6f} nested5_delta={nest['delta']:+.4f} "
        f"v10={v10_rep['ens_blend']['iou']:.6f} v11={v11_rep['ens_blend']['iou']:.6f} "
        f"event_recall={er_s} event_precision={ep_s} "
        f"accept={int(accept)} beats={int(accept and pub_beat)} real=0"
    )
    elog(line)
    if accept and pub_beat:
        elog(
            f"BEATS_WINNER CENTERNET ens_blend val {sel_val:.4f} "
            f"test {test_rep['ens_blend']['iou']:.4f}"
        )
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
