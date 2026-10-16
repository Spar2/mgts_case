"""Сиды MAE: init и лучшая обученная эпоха, смеси 3:1, 4:1, 5:1.

Опубликованный s3_mae_recon.pt — init (best_epoch=-1), не обученная сеть.
Для каждого сида оцениваются оба состояния:
  init — веса сразу после сидированного конструктора (у s0 это опубликованный файл);
  trained — лучшая эпоха с epoch>=0, даже если прокси хуже init.
Смесь (w * победитель + mae) / (w + 1), w = 3, 4, 5.
Плюс среднее трёх init и среднее трёх trained, каждое в смеси 4:1.
Порог FAR 1e-3 — фон всего v8 val, заморожен для остальных сплитов.
Бутстреп — 1000 выборок последовательностей, микро-IoU, перцентили 2.5 и 97.5.
Вложенные k=5 и k=2 — те же фолды i % k, порог только на holdout.
Временные карты: /tmp/s3_mae_seeds_maps, удаляются в конце.
Кэш v14 не строится. Реальные CSV не читаются. infer.py не меняется.
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
import zlib
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
from dts.mae_anom import (
    BAND, RADIUS, TILE, WINNER_TEST, WINNER_V10, WINNER_V11, WINNER_VAL,
    MaskedReconstructor, apply_cdf, channel_width, residuals,
)

WINNER = "ens_u2ft_tcn_lgbm"
TMP = Path("/tmp/s3_mae_seeds_maps")
LOG = Path("/workspace/grok_logs/_events.log")
OUT_JSON = Path("artifacts/eval/s3_mae_seeds.json")
DOC = Path("docs/S3_MAE.md")
RATIOS = (3, 4, 5)
N_BOOT = 1000
REF_S0_W4 = 0.3994942836
SPECS = (
    dict(slot="s0_init", seed=20261015, kind="init",
         ckpt="artifacts/eval/s3_mae_recon.pt",
         meta="artifacts/eval/s3_mae_train.json", knots="cdf_knots"),
    dict(slot="s0_trained", seed=20261015, kind="trained",
         ckpt="artifacts/eval/s3_mae_recon_s0_trained.pt",
         meta="artifacts/eval/s3_mae_train_s0_trained.json", knots="cdf_knots"),
    dict(slot="s1_init", seed=20261016, kind="init",
         ckpt="artifacts/eval/s3_mae_recon_s1_init.pt",
         meta="artifacts/eval/s3_mae_train_s1.json", knots="cdf_knots_init"),
    dict(slot="s1_trained", seed=20261016, kind="trained",
         ckpt="artifacts/eval/s3_mae_recon_s1.pt",
         meta="artifacts/eval/s3_mae_train_s1.json", knots="cdf_knots"),
    dict(slot="s2_init", seed=20261017, kind="init",
         ckpt="artifacts/eval/s3_mae_recon_s2_init.pt",
         meta="artifacts/eval/s3_mae_train_s2.json", knots="cdf_knots_init"),
    dict(slot="s2_trained", seed=20261017, kind="trained",
         ckpt="artifacts/eval/s3_mae_recon_s2.pt",
         meta="artifacts/eval/s3_mae_train_s2.json", knots="cdf_knots"),
)
INIT_SLOTS = ("s0_init", "s1_init", "s2_init")
TRAINED_SLOTS = ("s0_trained", "s1_trained", "s2_trained")
ALIASES: dict[str, str] = {}


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M', time.gmtime())} {msg}", flush=True)


def elog(msg: str) -> None:
    log(msg)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(time.strftime("%H:%M", time.gmtime()) + " " + msg + "\n")


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
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def mix(mae: np.ndarray, win: np.ndarray, w: int) -> np.ndarray:
    return ((float(w) * win + mae) / float(w + 1)).astype(np.float32)


def candidate_keys() -> list[str]:
    keys = [f"{slot}_w{w}" for slot in INIT_SLOTS + TRAINED_SLOTS for w in RATIOS]
    keys.append("mean3_init_w4")
    keys.append("mean3_trained_w4")
    return keys


def kind_of(key: str) -> str:
    if key.startswith("mean3_init"):
        return "init"
    if key.startswith("mean3_trained"):
        return "trained"
    if "_trained_" in key:
        return "trained"
    return "init"


def real_slot(slot: str) -> str:
    return ALIASES.get(slot, slot)


def shapes_of(ver: str, split: str) -> dict:
    out = {}
    pred = Path("artifacts/pred") / ver / WINNER / split
    for p in data.list_cache(f"artifacts/cache/{ver}", split):
        with np.load(pred / p.name) as z:
            out[p.stem] = tuple(z["s"].shape)
    return out


def map_path(slot: str, ver: str, split: str, stem: str) -> Path:
    return TMP / real_slot(slot) / f"{ver}_{split}" / f"{stem}.mmap"


def write_map(slot, ver, split, stem, score: np.ndarray) -> None:
    path = map_path(slot, ver, split, stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    mm = np.memmap(path, dtype=np.float32, mode="w+", shape=score.shape)
    mm[:] = score
    mm.flush()
    del mm


def read_map(slot, ver, split, stem, shape) -> np.ndarray:
    return np.memmap(map_path(slot, ver, split, stem), dtype=np.float32, mode="r", shape=shape)


def drop_split(ver: str, split: str) -> None:
    seen = set()
    for spec in SPECS:
        slot = real_slot(spec["slot"])
        if slot in seen:
            continue
        seen.add(slot)
        d = TMP / slot / f"{ver}_{split}"
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)


def conv0_of(path: Path) -> float:
    sd = torch.load(path, map_location="cpu", weights_only=True)
    for key, value in sd.items():
        if key.endswith("weight") and hasattr(value, "ndim") and int(value.ndim) == 4:
            return float(value.detach().float().sum())
    raise SystemExit(f"нет свёртки в {path}")


def head_max_of(path: Path) -> float:
    sd = torch.load(path, map_location="cpu", weights_only=True)
    w = sd["head.weight"].detach().abs().max().item()
    b = sd["head.bias"].detach().abs().max().item()
    return float(max(w, b))


def load_model(spec: dict):
    meta = json.loads(Path(spec["meta"]).read_text(encoding="utf-8"))
    if spec["knots"] not in meta:
        raise SystemExit(f"{spec['slot']}: нет {spec['knots']} в {spec['meta']}")
    knots = np.asarray(meta[spec["knots"]], np.float64)
    if int(meta.get("band", BAND)) != BAND or int(meta.get("radius", RADIUS)) != RADIUS:
        raise SystemExit(f"{spec['slot']}: band/radius не совпали")
    model = MaskedReconstructor(16)
    model.load_state_dict(torch.load(spec["ckpt"], map_location="cpu", weights_only=True))
    model.eval()
    if channel_width(model) > 16:
        raise SystemExit(f"{spec['slot']}: ширина > 16")
    return model, knots, meta


def score_array(model, knots, npz_path: Path) -> np.ndarray:
    with np.load(npz_path) as z:
        X = np.asarray(z["X"], np.float32)
    kw = dict(tile=TILE, overlap=32, band=BAND, radius=RADIUS, tile_batch=2)
    _r0, r1 = residuals(model, X[:2], X[2], **kw)
    if not np.isfinite(r1).all():
        raise SystemExit(f"non-finite {npz_path.name}")
    return apply_cdf(np.abs(r1), knots)


def predict_split(model, knots, slot: str, ver: str, split: str, shapes: dict) -> None:
    if slot in ALIASES:
        log(f"pred {slot} {ver}/{split} алиас {ALIASES[slot]}")
        return
    t0 = time.time()
    files = data.list_cache(f"artifacts/cache/{ver}", split)
    for i, p in enumerate(files, 1):
        if p.stem not in shapes:
            raise SystemExit(f"нет карты победителя {ver}/{split}/{p.name}")
        s = score_array(model, knots, p)
        if s.shape != shapes[p.stem]:
            raise SystemExit(f"{slot} {ver}/{split}/{p.name}: {s.shape} != {shapes[p.stem]}")
        write_map(slot, ver, split, p.stem, s)
        del s
        if i == 1 or i % 10 == 0 or i == len(files):
            log(f"pred {slot} {ver}/{split} {i}/{len(files)} {time.time() - t0:.0f}s")


def load_winner(ver, split, stem):
    with np.load(Path("artifacts/pred") / ver / WINNER / split / f"{stem}.npz") as z:
        return np.asarray(z["s"], np.float32)


def meta_of(ver, split, stem):
    with np.load(Path("artifacts/cache") / ver / split / f"{stem}.npz") as z:
        ev = np.asarray(z["ev"])
        valid = np.asarray(z["valid"], bool)
        nu = np.asarray(z["nu"])
    bg = valid & (ev == 0) & (nu == 0)
    return ev, valid, nu, bg


def iou_of(tp, fp, fn) -> float:
    return float(tp / max(tp + fp + fn, 1.0))


def boot_delta(sel_rows, win_rows, draws: np.ndarray) -> dict:
    sel = np.asarray(sel_rows, np.float64)
    win = np.asarray(win_rows, np.float64)
    deltas = np.empty(draws.shape[0], np.float64)
    for i, idx in enumerate(draws):
        s = sel[idx].sum(axis=0)
        w = win[idx].sum(axis=0)
        deltas[i] = iou_of(s[0], s[1], s[2]) - iou_of(w[0], w[1], w[2])
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    return dict(
        n=int(draws.shape[0]),
        mean=float(deltas.mean()),
        lo=float(lo),
        hi=float(hi),
        excludes_0=bool(lo > 0.0 or hi < 0.0),
        positive=bool(lo > 0.0),
    )


def draw_index(n: int, split_key: str) -> np.ndarray:
    seed = (20261015 ^ zlib.crc32(split_key.encode())) & 0xFFFFFFFF
    return np.random.default_rng(seed).integers(0, n, size=(N_BOOT, n))


def frame_from_score(stem, score, ev_map, valid, nu, x, t, labels, seqs):
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


def nested_pair(sel_frames, win_frames) -> dict:
    order = np.argsort([f["stem"] for f in win_frames])
    win_frames = [win_frames[i] for i in order]
    sel_frames = [sel_frames[i] for i in order]
    out = {}
    for k in (5, 2):
        folds = sel.sequence_folds(len(win_frames), k)
        parts_w, parts_s, rows = [], [], []
        for r, idx in enumerate(folds):
            hw = s2.eval_one(s2.take(win_frames, idx))
            hs = s2.eval_one(s2.take(sel_frames, idx))
            parts_w.append(hw)
            parts_s.append(hs)
            rows.append(dict(
                fold=r,
                winner=float(hw["iou"]),
                selected=float(hs["iou"]),
                delta=float(hs["iou"] - hw["iou"]),
            ))
        iw = float(sel.micro_iou(parts_w))
        ib = float(sel.micro_iou(parts_s))
        out[str(k)] = dict(
            winner_iou=iw, selected_iou=ib, delta=float(ib - iw), folds=rows,
        )
        log(f"nested k={k} micro {ib:.4f} winner {iw:.4f} delta {ib - iw:+.4f}")
    return out


def slot_maps(ver, split, stem, shape) -> dict:
    loaded = {}
    out = {}
    for slot in INIT_SLOTS + TRAINED_SLOTS:
        src = real_slot(slot)
        if src not in loaded:
            loaded[src] = np.asarray(read_map(src, ver, split, stem, shape))
        out[slot] = loaded[src]
    return out


def blend_of(key: str, maps: dict, win: np.ndarray) -> np.ndarray:
    if key == "mean3_init_w4":
        mae = (maps["s0_init"] + maps["s1_init"] + maps["s2_init"]) / 3.0
        return mix(mae.astype(np.float32), win, 4)
    if key == "mean3_trained_w4":
        mae = (maps["s0_trained"] + maps["s1_trained"] + maps["s2_trained"]) / 3.0
        return mix(mae.astype(np.float32), win, 4)
    slot, w = key.rsplit("_w", 1)
    return mix(maps[slot], win, int(w))


def score_split(keys, shapes, ver, split, thr_by_key, win_thr):
    rows = {k: [] for k in keys}
    win_rows = []
    for stem, shape in shapes.items():
        ev, valid, _nu, _bg = meta_of(ver, split, stem)
        truth = ev > 0
        win = load_winner(ver, split, stem)
        maps = slot_maps(ver, split, stem, shape)
        pred_w = (win >= win_thr) & valid
        win_rows.append(M.confusion(pred_w, truth, valid))
        for key in keys:
            blended = blend_of(key, maps, win)
            pred = (blended >= thr_by_key[key]) & valid
            rows[key].append(M.confusion(pred, truth, valid))
            del blended
        del maps, win
    return rows, win_rows


def bg_vector(slot: str, shapes: dict) -> np.ndarray:
    parts = []
    for stem, shape in shapes.items():
        _ev, _valid, _nu, bg = meta_of("v8", "val", stem)
        mae = np.asarray(read_map(slot, "v8", "val", stem, shape))
        parts.append(np.asarray(mae[bg], np.float32))
        del mae
    return np.concatenate(parts)


def thresholds_of(shapes) -> tuple[dict, float, int]:
    win_parts = []
    n_bg = 0
    for stem, _shape in shapes.items():
        _ev, _valid, _nu, bg = meta_of("v8", "val", stem)
        win = load_winner("v8", "val", stem)
        win_parts.append(np.asarray(win[bg], np.float32))
        n_bg += int(bg.sum())
        del win
    win_bg = np.concatenate(win_parts)
    del win_parts
    win_thr = float(M.far_threshold(win_bg, 1e-3))
    log(f"порог winner {win_thr:.6f} n_bg {n_bg}")
    thr = {}
    acc_init = np.zeros(win_bg.shape[0], np.float32)
    acc_trained = np.zeros(win_bg.shape[0], np.float32)
    held = {}
    for slot in INIT_SLOTS + TRAINED_SLOTS:
        src = real_slot(slot)
        if slot in ALIASES:
            if src not in held:
                raise SystemExit(f"алиас {slot} раньше источника {src}")
            for w in RATIOS:
                thr[f"{slot}_w{w}"] = thr[f"{src}_w{w}"]
                log(f"порог {slot}_w{w} {thr[f'{slot}_w{w}']:.6f}")
            acc_init += held[src]
            continue
        mae_bg = bg_vector(slot, shapes)
        log(f"фон {slot} {mae_bg.size}")
        for w in RATIOS:
            arr = mix(mae_bg, win_bg, w)
            thr[f"{slot}_w{w}"] = float(M.far_threshold(arr, 1e-3))
            log(f"порог {slot}_w{w} {thr[f'{slot}_w{w}']:.6f}")
            del arr
        if slot in INIT_SLOTS:
            acc_init += mae_bg
            if slot in set(ALIASES.values()):
                held[slot] = mae_bg
            else:
                del mae_bg
        else:
            acc_trained += mae_bg
            del mae_bg
    del held
    for name, acc in (("mean3_init_w4", acc_init), ("mean3_trained_w4", acc_trained)):
        arr = mix((acc / 3.0).astype(np.float32), win_bg, 4)
        thr[name] = float(M.far_threshold(arr, 1e-3))
        log(f"порог {name} {thr[name]:.6f}")
        del arr
    del acc_init, acc_trained, win_bg
    return thr, win_thr, n_bg


def summarize(key, rows, win_rows, draws, nest) -> dict:
    sel = np.asarray(rows, np.int64).sum(axis=0)
    win = np.asarray(win_rows, np.int64).sum(axis=0)
    point = iou_of(int(sel[0]), int(sel[1]), int(sel[2]))
    base = iou_of(int(win[0]), int(win[1]), int(win[2]))
    ci = boot_delta(rows, win_rows, draws)
    block = dict(iou=point, winner_iou=base, delta=float(point - base), ci=ci)
    if nest is not None:
        block["nested5"] = nest["5"]
        block["nested2"] = nest["2"]
    log(f"{key} iou {point:.4f} d {point - base:+.4f} ci {ci['lo']:+.4f},{ci['hi']:+.4f}")
    return block


def winner_frames(shapes, labels, seqs):
    frames = []
    for stem in shapes:
        ev, valid, nu, _bg = meta_of("v8", "val", stem)
        win = load_winner("v8", "val", stem)
        with np.load(Path("artifacts/cache/v8/val") / f"{stem}.npz") as z:
            x = np.asarray(z["x"], np.float64)
            t = np.asarray(z["t"], np.float64)
        frames.append(frame_from_score(stem, win, ev, valid, nu, x, t, labels, seqs))
        del win
    return frames


def blend_frames(key, shapes, labels, seqs):
    frames = []
    for stem, shape in shapes.items():
        ev, valid, nu, _bg = meta_of("v8", "val", stem)
        win = load_winner("v8", "val", stem)
        maps = slot_maps("v8", "val", stem, shape)
        blended = blend_of(key, maps, win)
        with np.load(Path("artifacts/cache/v8/val") / f"{stem}.npz") as z:
            x = np.asarray(z["x"], np.float64)
            t = np.asarray(z["t"], np.float64)
        frames.append(frame_from_score(stem, blended, ev, valid, nu, x, t, labels, seqs))
        del win, maps, blended
    return frames


def maybe_alias_inits(models) -> dict:
    """Нулевая голова даёт hat=0. Если карты init совпали на одном файле — не гонять их трижды."""
    files = data.list_cache("artifacts/cache/v8", "val")
    ref = score_array(models["s0_init"][0], models["s0_init"][1], files[0])
    diffs = {}
    for slot in ("s1_init", "s2_init"):
        got = score_array(models[slot][0], models[slot][1], files[0])
        diff = float(np.max(np.abs(got.astype(np.float32) - ref)))
        diffs[slot] = diff
        log(f"init map {slot} vs s0_init max|d| {diff:.3e} file {files[0].name}")
        if diff <= 1e-6:
            ALIASES[slot] = "s0_init"
        del got
    del ref
    return diffs


def v14_plan() -> dict:
    root = Path("artifacts/cache/v14")
    if not root.exists():
        return dict(present=False, built=False, scored=False,
                    reason="artifacts/cache/v14 отсутствует, кэш не строился")
    splits = []
    for split in ("val", "test"):
        cache = list(root.joinpath(split).glob("*.npz")) if root.joinpath(split).exists() else []
        pred = Path("artifacts/pred/v14") / WINNER / split
        preds = list(pred.glob("*.npz")) if pred.exists() else []
        if cache and preds:
            splits.append(split)
    if not splits:
        return dict(present=True, built=False, scored=False, splits=[],
                    reason="кэш v14 есть, предсказаний победителя нет; новый кэш не строился")
    return dict(present=True, built=False, scored=True, splits=splits,
                reason="считаем готовый кэш, новый не строился")


def render_section(report: dict) -> str:
    lines = [
        "",
        "## Сиды и смеси",
        "",
        "Вопрос: прирост ens_mean4 даёт обучение или случайные свёрточные признаки.",
        "Опубликованный `artifacts/eval/s3_mae_recon.pt` — init, `best_epoch=-1`.",
        "Первый запуск s1/s2 (строки MAE_S1_* и MAE_S2_* в 23:21) создавал сеть до сида:",
        "conv0 совпал, оба остановились на init. Эти чекпойнты `artifacts/ckpt/s3_mae_s1`",
        "и `s3_mae_s2` в таблицу не входят.",
        "Повтор: сид до `MaskedReconstructor`. s1 = 20261016 (`s3_mae_s1r`), s2 = 20261017 (`s3_mae_s2r`).",
        "s0 trained — то же опубликованное init, дообученное в `s3_mae_s0tr`;",
        "`s3_mae_recon.pt` и `s3_mae_train.json` не перезаписывались.",
        "Для каждого сида два состояния: init и лучшая эпоха `epoch≥0`.",
        "Смесь фиксирована заранее: (w × победитель + карта MAE) / (w + 1), w = 3, 4, 5.",
        "Среднее трёх init и среднее трёх trained считаются отдельно, оба в смеси 4:1.",
        "Порог FAR 1e-3 — фон всего v8 val, заморожен на test, v10 и v11.",
        "Бутстреп: 1000 выборок последовательностей с возвращением, микро-IoU дельты,",
        "интервал — перцентили 2.5 и 97.5. Вложенные k=5 и k=2 — фолды `i % k`",
        "по отсортированным именам v8 val, порог только на holdout (`eval_one`).",
        "",
        f"conv0 init: s0 {report['conv0']['s0_init']:.6f}, "
        f"s1 {report['conv0']['s1_init']:.6f}, s2 {report['conv0']['s2_init']:.6f}.",
        f"head_max init: {report['head_max']['s0_init']:.3g}, "
        f"{report['head_max']['s1_init']:.3g}, {report['head_max']['s2_init']:.3g}.",
        f"head_max trained: s0 {report['head_max']['s0_trained']:.4g} "
        f"(эпоха {report['trained_epoch']['s0_trained']}), "
        f"s1 {report['head_max']['s1_trained']:.4g} "
        f"(эпоха {report['trained_epoch']['s1_trained']}), "
        f"s2 {report['head_max']['s2_trained']:.4g} "
        f"(эпоха {report['trained_epoch']['s2_trained']}).",
        f"Макс. |Δ| карты init на одном файле val: "
        f"s1 {report['init_map_diff'].get('s1_init', float('nan')):.3e}, "
        f"s2 {report['init_map_diff'].get('s2_init', float('nan')):.3e}.",
        "Нулевая голова даёт hat=0, поэтому init-карта не зависит от свёрток и совпадает с CDF(|dT_space|).",
        f"Контроль s0 init 4:1 на val {report['s0_init_w4_val']:.6f} "
        f"(опубликовано {REF_S0_W4:.6f}).",
        f"v14: {report['v14']['reason']}.",
        "Реальные CSV не читались. `dts/infer.py` и выпуск по умолчанию не менялись.",
        "",
        "### Init",
        "",
        "| смесь | val IoU | Δ val | ДИ val | Δ test | ДИ test | Δ v10 | Δ v11 | Δ k5 | Δ k2 |",
        "|---|---:|---:|---|---:|---|---:|---:|---:|---:|",
    ]
    lines.extend(_rows(report, "init"))
    lines += [
        "",
        "### Trained (лучшая эпоха ≥ 0)",
        "",
        "| смесь | val IoU | Δ val | ДИ val | Δ test | ДИ test | Δ v10 | Δ v11 | Δ k5 | Δ k2 |",
        "|---|---:|---:|---|---:|---|---:|---:|---:|---:|",
    ]
    lines.extend(_rows(report, "trained"))
    extra = [sk for sk in report["candidates"] if sk.startswith("v14_")]
    if extra:
        lines += ["", "Колонки v14 есть в `artifacts/eval/s3_mae_seeds.json`.", ""]
    proposal = "да" if report["switch_proposed"] else "нет"
    lines += [
        "",
        f"ДИ val строго выше нуля: init {report['init_ci_pos']}/{report['n_init']}, "
        f"trained {report['trained_ci_pos']}/{report['n_trained']}.",
        "Предложение переключить выпуск смотрит только обученные смеси сид×вес",
        "(9 штук, без средних): все Δ val > 0 и нижняя граница ДИ > 0.",
        "Совпадение init с опубликованным фильтром само по себе переключение не предлагает:",
        "это не результат обучения.",
        f"Предлагать переключение: **{proposal}**. Файл `dts/infer.py` не менялся.",
        "",
    ]
    return "\n".join(lines)


def _rows(report: dict, kind: str) -> list[str]:
    lines = []
    for key, block in report["candidates"].items():
        if kind_of(key) != kind:
            continue
        if "v8_val" not in block or "v8_test" not in block:
            continue
        val, te = block["v8_val"], block["v8_test"]
        v10 = block.get("v10_test", {}).get("delta", float("nan"))
        v11 = block.get("v11_test", {}).get("delta", float("nan"))
        lines.append(
            f"| {key} | {val['iou']:.4f} | {val['delta']:+.4f} | "
            f"{val['ci']['lo']:+.4f}…{val['ci']['hi']:+.4f} | "
            f"{te['delta']:+.4f} | {te['ci']['lo']:+.4f}…{te['ci']['hi']:+.4f} | "
            f"{v10:+.4f} | {v11:+.4f} | "
            f"{val['nested5']['delta']:+.4f} | {val['nested2']['delta']:+.4f} |"
        )
    return lines


def _family(candidates, keys):
    deltas = [candidates[k]["v8_val"]["delta"] for k in keys]
    pos = sum(1 for k in keys if candidates[k]["v8_val"]["ci"]["positive"])
    return deltas, pos


def main() -> None:
    if has_tag("MAE_SEEDS_DONE"):
        raise SystemExit("MAE_SEEDS_DONE уже есть")
    for spec in SPECS:
        if not Path(spec["ckpt"]).exists():
            raise SystemExit(f"нет {spec['ckpt']}")
        if not Path(spec["meta"]).exists():
            raise SystemExit(f"нет {spec['meta']}")
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    conv0 = {spec["slot"]: conv0_of(Path(spec["ckpt"])) for spec in SPECS}
    head_max = {spec["slot"]: head_max_of(Path(spec["ckpt"])) for spec in SPECS}
    log(
        "conv0 "
        + " ".join(f"{k}={v:.6f}" for k, v in conv0.items())
    )
    if abs(conv0["s1_init"] - conv0["s2_init"]) < 1e-6:
        raise SystemExit("s1 и s2 init совпали: сид всё ещё после конструктора")
    if abs(conv0["s0_init"] - conv0["s1_init"]) < 1e-6:
        raise SystemExit("s1 init совпал с опубликованным s0")
    trained_epoch = {}
    for spec in SPECS:
        if spec["kind"] != "trained":
            continue
        meta = json.loads(Path(spec["meta"]).read_text(encoding="utf-8"))
        ep = int(meta.get("best_trained", {}).get("epoch", -1))
        if ep < 0:
            raise SystemExit(f"{spec['slot']}: нет эпохи epoch>=0")
        if head_max[spec["slot"]] <= 0.0:
            raise SystemExit(f"{spec['slot']}: голова обученной эпохи нулевая")
        trained_epoch[spec["slot"]] = ep
        log(f"{spec['slot']} epoch {ep} head_max {head_max[spec['slot']]:.4g}")
    labels = pd.read_csv("data/synth_meta/v8/labels.csv")
    seqs = pd.read_csv("data/synth_meta/v8/sequences.csv").set_index("seq_id")
    v14 = v14_plan()
    splits = [("v8", "val"), ("v8", "test"), ("v10", "test"), ("v11", "test")]
    for split in v14.get("splits") or []:
        splits.append(("v14", split))
    published = {
        "v8_val": WINNER_VAL,
        "v8_test": WINNER_TEST,
        "v10_test": WINNER_V10,
        "v11_test": WINNER_V11,
    }
    models = {}
    metas = {}
    t0 = time.time()
    if TMP.exists():
        shutil.rmtree(TMP)
    keys = candidate_keys()
    try:
        for spec in SPECS:
            model, knots, meta = load_model(spec)
            models[spec["slot"]] = (model, knots)
            metas[spec["slot"]] = dict(
                seed=spec["seed"], kind=spec["kind"], ckpt=spec["ckpt"],
                best_epoch=meta.get("best", {}).get("epoch"),
                best_trained=meta.get("best_trained"),
                init_probe=meta.get("init_probe"),
                seed_before_model=meta.get("seed_before_model"),
            )
        init_map_diff = maybe_alias_inits(models)
        shapes = {f"{ver}_{split}": shapes_of(ver, split) for ver, split in splits}
        for spec in SPECS:
            if spec["kind"] != "init":
                continue
            predict_split(models[spec["slot"]][0], models[spec["slot"]][1],
                          spec["slot"], "v8", "val", shapes["v8_val"])
        for spec in SPECS:
            if spec["kind"] != "trained":
                continue
            predict_split(models[spec["slot"]][0], models[spec["slot"]][1],
                          spec["slot"], "v8", "val", shapes["v8_val"])
        thr, win_thr, n_bg = thresholds_of(shapes["v8_val"])
        rows, win_rows = score_split(keys, shapes["v8_val"], "v8", "val", thr, win_thr)
        draws = {"v8_val": draw_index(len(win_rows), "v8_val")}
        win_point = iou_of(*np.asarray(win_rows, np.int64).sum(axis=0))
        if abs(win_point - WINNER_VAL) > 0.005:
            raise SystemExit(f"победитель val {win_point} далёк от публикации")
        s0 = summarize("s0_init_w4", rows["s0_init_w4"], win_rows, draws["v8_val"], None)
        if abs(s0["iou"] - REF_S0_W4) > 0.005:
            raise SystemExit(f"s0 init 4:1 val {s0['iou']} далёк от {REF_S0_W4}")
        base_frames = winner_frames(shapes["v8_val"], labels, seqs)
        nests = {}
        for key in keys:
            log(f"nested {key}")
            nests[key] = nested_pair(blend_frames(key, shapes["v8_val"], labels, seqs), base_frames)
        del base_frames
        split_rows = {"v8_val": (rows, win_rows)}
        drop_split("v8", "val")
        for ver, split in splits[1:]:
            key = f"{ver}_{split}"
            for spec in SPECS:
                predict_split(models[spec["slot"]][0], models[spec["slot"]][1],
                              spec["slot"], ver, split, shapes[key])
            rows_s, win_s = score_split(keys, shapes[key], ver, split, thr, win_thr)
            split_rows[key] = (rows_s, win_s)
            draws[key] = draw_index(len(win_s), key)
            got = iou_of(*np.asarray(win_s, np.int64).sum(axis=0))
            if key in published and abs(got - published[key]) > 0.005:
                raise SystemExit(f"победитель {key} {got} далёк от {published[key]}")
            log(f"winner {key} {got:.6f}")
            drop_split(ver, split)
        candidates = {}
        for key in keys:
            block = {}
            for sk, (rr, ww) in split_rows.items():
                nest = nests[key] if sk == "v8_val" else None
                block[sk] = summarize(f"{key} {sk}", rr[key], ww, draws[sk], nest)
            candidates[key] = block
    finally:
        if TMP.exists():
            shutil.rmtree(TMP, ignore_errors=True)
        for slot in list(models):
            del models[slot]

    init_keys = [k for k in keys if kind_of(k) == "init" and not k.startswith("mean3")]
    trained_keys = [k for k in keys if kind_of(k) == "trained" and not k.startswith("mean3")]
    init_deltas, init_pos = _family(candidates, init_keys)
    trained_deltas, trained_pos = _family(candidates, trained_keys)
    switch = all(
        candidates[k]["v8_val"]["delta"] > 0.0 and candidates[k]["v8_val"]["ci"]["positive"]
        for k in trained_keys
    )
    report = dict(
        name="MAE_SEEDS",
        question="gain from training or from the init convolutional filter",
        seeds=metas,
        conv0=conv0,
        head_max=head_max,
        trained_epoch=trained_epoch,
        init_map_diff=init_map_diff,
        init_aliased=dict(ALIASES),
        ratios=list(RATIOS),
        blend="(w * winner + mae) / (w + 1)",
        mean3_init="mean of three init CDF maps, then w=4",
        mean3_trained="mean of three trained CDF maps, then w=4",
        v14=v14,
        bootstrap=dict(n=N_BOOT, unit="sequence", interval="percentile 2.5 and 97.5",
                       statistic="micro IoU delta versus winner at the frozen v8-val threshold"),
        nested="s3_sel2_cv.eval_one, sequence_folds i%k, sorted v8 val stems",
        n_bg=n_bg,
        winner_threshold=win_thr,
        thresholds=thr,
        winner_recompute={
            sk: candidates["s0_init_w4"][sk]["winner_iou"] for sk in candidates["s0_init_w4"]
        },
        s0_init_w4_val=candidates["s0_init_w4"]["v8_val"]["iou"],
        s0_w4_published=REF_S0_W4,
        n_init=len(init_keys),
        n_trained=len(trained_keys),
        init_ci_pos=init_pos,
        trained_ci_pos=trained_pos,
        switch_rule="trained seed-by-ratio only: all 9 val deltas > 0 and val CI lower bound > 0",
        switch_proposed=bool(switch),
        infer_changed=False,
        candidates=candidates,
        seconds=round(time.time() - t0, 1),
        real_traces_used=False,
        discarded_runs="s3_mae_s1 and s3_mae_s2 at 23:21 shared init because the seed was applied after MaskedReconstructor",
    )
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(plain(report), indent=1, ensure_ascii=False), encoding="utf-8")
    section = render_section(report)
    old = DOC.read_text(encoding="utf-8") if DOC.exists() else ""
    if "## Сиды и смеси" in old:
        old = old.split("## Сиды и смеси")[0].rstrip() + "\n"
    DOC.write_text(old.rstrip() + "\n" + section, encoding="utf-8")

    def pack(group):
        return ",".join(f"{k.split('_w')[0].replace('_init','').replace('_trained','')}:{candidates[k]['v8_val']['delta']:+.4f}" for k in group)

    w4_init = [k for k in init_keys if k.endswith("_w4")]
    w4_trained = [k for k in trained_keys if k.endswith("_w4")]
    v14_flag = "scored" if v14.get("scored") else "absent"
    line = (
        f"MAE_SEEDS_DONE seeds=20261015,20261016,20261017 ratios=3,4,5 v14={v14_flag} "
        f"s0_init_w4_val={candidates['s0_init_w4']['v8_val']['iou']:.6f} "
        f"init_w4_val_d={pack(w4_init)} trained_w4_val_d={pack(w4_trained)} "
        f"init_dmin={min(init_deltas):+.4f} init_dmax={max(init_deltas):+.4f} "
        f"trained_dmin={min(trained_deltas):+.4f} trained_dmax={max(trained_deltas):+.4f} "
        f"init_ci_pos={init_pos}/{len(init_keys)} trained_ci_pos={trained_pos}/{len(trained_keys)} "
        f"mean3_init_w4_d={candidates['mean3_init_w4']['v8_val']['delta']:+.4f} "
        f"mean3_trained_w4_d={candidates['mean3_trained_w4']['v8_val']['delta']:+.4f} "
        f"trained_epoch=s0:{trained_epoch['s0_trained']},s1:{trained_epoch['s1_trained']},s2:{trained_epoch['s2_trained']} "
        f"conv0=s0:{conv0['s0_init']:.4f},s1:{conv0['s1_init']:.4f},s2:{conv0['s2_init']:.4f} "
        f"init_map_diff={max(init_map_diff.values()):.2e} "
        f"switch_proposed={int(switch)} beats=0 real_train=0"
    )
    elog(line)
    log(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
