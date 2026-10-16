"""Оценка MAEFT: FAR 1e-3 по фону v8 val, фиксированные смеси, вложенные k=2 и k=5.

Кандидаты перечислены в s3_maeft_net.CANDIDATES до чтения val.
Порог v10/v11 не пересчитывается. Карты только в /tmp и удаляются.
Реальные CSV не читаются и порог не двигают.
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
from s3_maeft_net import (
    ARMS, BLENDS, CANDIDATES, NESTED_GAIN, WIDTH, WINNER_TEST, WINNER_V10, WINNER_V11, WINNER_VAL,
    blend_scores, build_arm, choose_candidate, predict_map, split_candidate,
)

WINNER = "ens_u2ft_tcn_lgbm"
TMP = Path("/tmp/s3_maeft_maps")
TRAIN_JSON = Path("artifacts/eval/s3_maeft_train.json")
OUT_JSON = Path("artifacts/eval/s3_maeft_v8.json")
DOC = Path("docs/S3_MAEFT.md")
LOG = Path("/workspace/grok_logs/_events.log")
TILE = (256, 512)
PUB_N_BG = 65095696


def log(msg: str) -> None:
    print(time.strftime("%H:%M", time.gmtime()) + " " + msg, flush=True)


def elog(msg: str) -> None:
    log(msg)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(time.strftime("%H:%M", time.gmtime()) + " " + msg + "\n")


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


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(plain(obj), indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def shapes_of(ver, split) -> dict:
    out = {}
    pred = Path("artifacts/pred") / ver / WINNER / split
    for path in data.list_cache(f"artifacts/cache/{ver}", split):
        with np.load(pred / path.name) as z:
            out[path.stem] = tuple(z["s"].shape)
    return out


def open_map(arm, ver, split, stem, kind, shape, dtype):
    folder = TMP / arm / ver / split
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{stem}_{kind}.mmap"
    return np.memmap(path, dtype=dtype, mode="w+", shape=shape)


def read_map(arm, ver, split, stem, kind, shape, dtype):
    path = TMP / arm / ver / split / f"{stem}_{kind}.mmap"
    return np.memmap(path, dtype=dtype, mode="r", shape=shape)


def load_winner(ver, split, stem):
    with np.load(Path("artifacts/pred") / ver / WINNER / split / f"{stem}.npz") as z:
        return np.asarray(z["s"], np.float32), np.asarray(z["h"], np.float32)


def bg_mask(ver, split, stem):
    with np.load(Path("artifacts/cache") / ver / split / f"{stem}.npz") as meta:
        evm = np.asarray(meta["ev"])
        valid = np.asarray(meta["valid"], bool)
        nu = np.asarray(meta["nu"])
    return evm, valid, nu, valid & (evm == 0) & (nu == 0)


def load_arm(arm: str) -> torch.nn.Module:
    path = Path(f"artifacts/eval/s3_maeft_{arm}.pt")
    if not path.exists():
        raise SystemExit(f"нет {path}")
    model = build_arm(arm, 0)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    log(f"веса {arm} {path.stat().st_size} байт ширина {WIDTH}")
    return model


def predict_split(model, arm, ver, split, shapes) -> None:
    t0 = time.time()
    files = data.list_cache(f"artifacts/cache/{ver}", split)
    for i, path in enumerate(files, 1):
        shape = shapes[path.stem]
        with np.load(path) as z:
            image = np.asarray(z["X"], np.float32)
        if image.shape[1:] != shape:
            raise SystemExit(f"{ver} {path.name}: X {image.shape[1:]} != {shape}")
        score, sign = predict_map(model, image, tile=TILE, overlap=32, batch=2)
        if not np.isfinite(score).all() or not np.isfinite(sign).all():
            raise SystemExit(f"non-finite {arm} {path.name}")
        sm = open_map(arm, ver, split, path.stem, "s", shape, np.float32)
        hm = open_map(arm, ver, split, path.stem, "h", shape, np.float16)
        sm[:] = score
        hm[:] = sign.astype(np.float16)
        sm.flush()
        hm.flush()
        del sm, hm, image, score, sign
        if i == 1 or i % 5 == 0 or i == len(files):
            log(f"pred {arm} {ver} {split} {i}/{len(files)} {time.time() - t0:.0f}s")


def concat_bg(ver, split, shapes, score_of) -> tuple[np.ndarray, int]:
    parts = []
    n_bg = 0
    for stem, shape in shapes.items():
        _evm, _valid, _nu, bg = bg_mask(ver, split, stem)
        score = score_of(stem, shape)
        parts.append(np.asarray(score[bg], np.float32))
        n_bg += int(bg.sum())
        del score
    return np.concatenate(parts), n_bg


def score_split(ver, split, shapes, arms, thr) -> dict:
    wanted = [WINNER]
    for arm in arms:
        wanted.append(arm)
        for kind in BLENDS:
            wanted.append(f"{kind}_{arm}")
    names = [name for name in wanted if name in thr]
    tot = {name: np.zeros(3, np.int64) for name in names}
    far_n = {name: 0 for name in names}
    far_d = {name: 0 for name in names}
    for stem, shape in shapes.items():
        evm, valid, _nu, bg = bg_mask(ver, split, stem)
        truth = evm > 0
        ws, _wh = load_winner(ver, split, stem)
        packed = {WINNER: ws}
        for arm in arms:
            score = np.asarray(read_map(arm, ver, split, stem, "s", shape, np.float32))
            if arm in names:
                packed[arm] = score
            for kind in BLENDS:
                name = f"{kind}_{arm}"
                if name in names:
                    packed[name] = blend_scores(score, ws, kind)
        for name in names:
            ps = packed[name]
            pred = (ps >= thr[name]) & valid
            tot[name] += np.asarray(M.confusion(pred, truth, valid), np.int64)
            far_n[name] += int((pred & bg).sum())
            far_d[name] += int(bg.sum())
        del ws
    report = {}
    for name in names:
        row = dict(threshold=float(thr[name]), far=float(far_n[name] / max(far_d[name], 1)), **M.prf(*tot[name]))
        report[name] = row
        if name in CANDIDATES or name in ARMS or name == WINNER:
            log(f"{ver} {split} {name} IoU {row['iou']:.4f} FAR {row['far']:.5f}")
    return report


def thresholds_val(shapes) -> tuple[dict, int]:
    """Пороги по фону v8 val. Смеси по одной, чтобы не держать несколько копий фона."""
    win, n_bg = concat_bg("v8", "val", shapes, lambda stem, shape: load_winner("v8", "val", stem)[0])
    thr = {WINNER: float(M.far_threshold(win, 1e-3))}
    log(f"порог победителя {thr[WINNER]:.6f} n_bg {n_bg}")
    for arm in ARMS:
        alone, n_arm = concat_bg(
            "v8", "val", shapes,
            lambda stem, shape, arm=arm: read_map(arm, "v8", "val", stem, "s", shape, np.float32),
        )
        if n_arm != n_bg:
            raise SystemExit(f"n_bg {arm} {n_arm} != {n_bg}")
        thr[arm] = float(M.far_threshold(alone, 1e-3))
        log(f"порог {arm} {thr[arm]:.6f}")
        for kind in BLENDS:
            blended = blend_scores(alone, win, kind)
            name = f"{kind}_{arm}"
            thr[name] = float(M.far_threshold(blended, 1e-3))
            log(f"порог {name} {thr[name]:.6f}")
            del blended
        del alone
    del win
    return thr, n_bg


def frame_of(stem, score, ev_map, valid, nu, x, t, labels, seqs):
    ev_idx, nu_idx, bg_idx, n_bg = sev._indices(ev_map, valid, nu, stem)
    event, nuisance, bg = sev._take(score, ev_idx, nu_idx, bg_idx)
    cooling, _rows = sev._events_for(stem, ev_map, x, t, labels, seqs, ev_idx)
    return dict(
        stem=stem,
        ev_m=event.astype(np.float16)[None],
        nu_m=nuisance.astype(np.float16)[None],
        bg_m=bg.astype(np.float16)[None],
        n_bg=n_bg,
        cooling=cooling,
    )


def nested_ks(shapes, selected: str) -> dict:
    labels = pd.read_csv("data/synth_meta/v8/labels.csv")
    seqs = pd.read_csv("data/synth_meta/v8/sequences.csv").set_index("seq_id")
    kind, arm = split_candidate(selected)
    files = data.list_cache("artifacts/cache/v8", "val")
    winner_f, sel_f = [], []
    for path in files:
        stem = path.stem
        shape = shapes[stem]
        with np.load(path) as z:
            ev_map = np.asarray(z["ev"])
            valid = np.asarray(z["valid"])
            nu = np.asarray(z["nu"])
            x = np.asarray(z["x"], np.float64)
            t = np.asarray(z["t"], np.float64)
        score = np.asarray(read_map(arm, "v8", "val", stem, "s", shape, np.float32))
        ws, _wh = load_winner("v8", "val", stem)
        blend = blend_scores(score, ws, kind)
        winner_f.append(frame_of(stem, ws, ev_map, valid, nu, x, t, labels, seqs))
        sel_f.append(frame_of(stem, blend, ev_map, valid, nu, x, t, labels, seqs))
        del score, ws, blend
    order = np.argsort([fr["stem"] for fr in winner_f])
    winner_f = [winner_f[i] for i in order]
    sel_f = [sel_f[i] for i in order]
    out = {}
    for k in (2, 5):
        folds = sel.sequence_folds(len(winner_f), k)
        parts_w, parts_b, rows = [], [], []
        for fold, idx in enumerate(folds):
            hold_w = s2.eval_one(s2.take(winner_f, idx))
            hold_b = s2.eval_one(s2.take(sel_f, idx))
            parts_w.append(hold_w)
            parts_b.append(hold_b)
            rows.append(dict(
                fold=fold, winner=float(hold_w["iou"]), selected=float(hold_b["iou"]),
                delta=float(hold_b["iou"] - hold_w["iou"]),
                tp=hold_b["tp"], fp=hold_b["fp"], fn=hold_b["fn"],
            ))
            log(f"nested k={k} fold {fold} winner {hold_w['iou']:.4f} {selected} {hold_b['iou']:.4f} d {hold_b['iou'] - hold_w['iou']:+.4f}")
        iou_w = float(sel.micro_iou(parts_w))
        iou_b = float(sel.micro_iou(parts_b))
        out[str(k)] = dict(winner_iou=iou_w, selected_iou=iou_b, delta=float(iou_b - iou_w), folds=rows)
        log(f"nested k={k} micro {selected} {iou_b:.4f} winner {iou_w:.4f} delta {iou_b - iou_w:+.4f}")
    out["procedure"] = "s3_sel2_cv.eval_one, sequence_folds i%k, micro_iou; gain vs winner, not absolute bar 0.3897"
    out["gain_min"] = float(min(out["2"]["delta"], out["5"]["delta"]))
    return out


def event_recall(shapes, selected: str, thr: dict) -> dict:
    labels = pd.read_csv("data/synth_meta/v8/labels.csv")
    kind, arm = split_candidate(selected)
    names = (selected, arm, WINNER)
    counts = {name: {"n": 0, "hit": 0, "n_weak": 0, "hit_weak": 0} for name in names}
    for stem, shape in shapes.items():
        with np.load(Path("artifacts/cache/v8/val") / f"{stem}.npz") as z:
            ev_map = np.asarray(z["ev"])
            x = np.asarray(z["x"], np.float64)
            t = np.asarray(z["t"], np.float64)
        score = np.asarray(read_map(arm, "v8", "val", stem, "s", shape, np.float32))
        ws, _wh = load_winner("v8", "val", stem)
        maps = {arm: score, selected: blend_scores(score, ws, kind), WINNER: ws}
        for row in M.label_boxes(labels, stem).itertuples():
            if row.kind != "event":
                continue
            px = M.event_pixels(row, x, t, ev_map)
            if not px.any():
                continue
            weak = float(row.amp_obs_C) <= 2.0
            for name in names:
                hit = float((maps[name][px] >= thr[name]).mean()) >= 0.1
                counts[name]["n"] += 1
                counts[name]["hit"] += int(hit)
                counts[name]["n_weak"] += int(weak)
                counts[name]["hit_weak"] += int(hit and weak)
        del score, ws
    out = {}
    for name, c in counts.items():
        out[name] = dict(
            n=c["n"], recall=float(c["hit"] / max(c["n"], 1)),
            n_weak=c["n_weak"], recall_le2c=float(c["hit_weak"] / max(c["n_weak"], 1)),
            hit_frac=0.1, used_for_selection=False,
        )
        log(f"события {name} recall {out[name]['recall']:.3f} ≤2C {out[name]['recall_le2c']:.3f}")
    return out


def render_md(rep: dict) -> str:
    train = rep["train"]
    best = train.get("best", {})
    val = rep["val"]
    selected = rep["selected"]
    kind, arm = split_candidate(selected)
    lines = [
        "# S3 MAEFT",
        "",
        "Предобучение энкодера — маскированный реконструктор `artifacts/eval/s3_mae_recon.pt`.",
        "Его ствол 4×5×3 и ширина 16 не совпадают с DTSUNet (`unet_v810ft`: ствол 3×7×3, ширины 16-32-64-128).",
        "Совместимые веса энкодера (все тензоры e1/e2/e3, включая BatchNorm) перенесены в U-Net той же ширины.",
        "Декодер этого U-Net случайный: голова сегментации на 3 класса, пропуски складываются, как у MAE.",
        "Рядом дообучается тот же опубликованный DTSUNet из `artifacts/stage1/unet_v810ft.pt` и копия SegUNet без весов MAE.",
        "Обучение только на синтетике v8+v10 train. Официальный v8 val в выбор эпохи не входит.",
        "Прокси эпохи — IoU@FAR 1e-3 на тайлах train, у которых crc32 имени делится на 10.",
        "Смеси зафиксированы до val: ens_mean = (s+победитель)/2, ens_max = max, ens_mean4 = (4×победитель+s)/5.",
        "Выбор — максимум IoU этих смесей на v8 val. Ничья отдаёт ens_mean4 и руке MAE.",
        "Вложенные k=2 и k=5 — те же фолды, что `scripts/s3_sel2_cv.py` (sequence_folds i%k, eval_one, micro IoU).",
        "Прирост считается к победителю на этих фолдах. Абсолютная планка 0.3897 не используется: у самого победителя вложенный micro IoU ниже неё.",
        "Порог FAR 1e-3 — квантиль фона всего v8 val. На v10 и v11 он заморожен.",
        "Реальные CSV, карточки и спектр шума не читаются.",
        "",
        f"Решение: **{'принят' if rep['accept'] else 'отклонён'}**.",
        "",
        rep["decision"],
        "",
        "## Обучение",
        "",
        f"Остановка `{train.get('stopped')}`, {train.get('seconds')} с, шаг {train.get('step')}.",
        f"Скопировано тензоров энкодера MAE: {train.get('mae_copied')}, max|Δ| в момент копии {train.get('mae_copy_max_abs')}.",
        f"Лучшие эпохи по прокси: mae {best.get('mae')}, random {best.get('random')}, v810 {best.get('v810')}.",
        f"Сдвиг лучшего чекпойнта относительно старта: {train.get('encoder_max_dw')}.",
        "Потеря focal+tversky_ce, AdamW, lr энкодера 1e-5, lr декодера 1e-4, weight decay 0.",
        "Тайл 128×256, batch 8, 2 потока. Три руки делают одни и те же шаги.",
        "",
        "## Официальный v8 val",
        "",
        f"Фон {rep['n_bg']} клеток. Пересчёт победителя val {val[WINNER]['iou']:.6f} (опубликован {WINNER_VAL:.6f}).",
        f"Выбран `{selected}`.",
        "",
        "| кандидат | порог | IoU | F1 | FAR |",
        "|---|---:|---:|---:|---:|",
    ]
    show = [WINNER, selected, arm]
    for name in list(ARMS) + list(CANDIDATES):
        if name not in show:
            show.append(name)
    for name in show:
        row = val[name]
        lines.append(f"| {name} | {row['threshold']:.6f} | {row['iou']:.4f} | {row['f1']:.4f} | {row['far']:.5f} |")
    lines += [
        "",
        "## Test и перенос",
        "",
        "| сплит | выбранный | его U-Net | победитель |",
        "|---|---:|---:|---:|",
    ]
    for split in ("test", "v10", "v11"):
        block = rep[split]
        lines.append(
            f"| {split} | {block[selected]['iou']:.4f} | {block[arm]['iou']:.4f} | {block[WINNER]['iou']:.4f} |"
        )
    nest = rep["nested"]
    lines += [
        "",
        "## Вложенная проверка",
        "",
        nest["procedure"],
        "",
        f"k=5: выбранный {nest['5']['selected_iou']:.4f}, победитель {nest['5']['winner_iou']:.4f}, дельта {nest['5']['delta']:+.4f}.",
        f"k=2: выбранный {nest['2']['selected_iou']:.4f}, победитель {nest['2']['winner_iou']:.4f}, дельта {nest['2']['delta']:+.4f}.",
        f"Минимум дельт {nest['gain_min']:+.4f}. Порог принятия {NESTED_GAIN:.3f}.",
        "",
        "| k | фолд | победитель | выбранный | дельта |",
        "|---:|---:|---:|---:|---:|",
    ]
    for k in ("5", "2"):
        for row in nest[k]["folds"]:
            lines.append(
                f"| {k} | {row['fold']} | {row['winner']:.4f} | {row['selected']:.4f} | {row['delta']:+.4f} |"
            )
    ev = rep["events_val"]
    lines += [
        "",
        "## События v8 val",
        "",
        "Доля событий, у которых не меньше 10% пикселей выше порога. В выбор не входит.",
        "",
        f"Выбранный: recall {ev[selected]['recall']:.3f}, ≤2°C {ev[selected]['recall_le2c']:.3f} из {ev[selected]['n_weak']}.",
        f"Победитель: recall {ev[WINNER]['recall']:.3f}, ≤2°C {ev[WINNER]['recall_le2c']:.3f}.",
        "",
        f"Смесь, которая вошла в таблицу как выбранная, собрана как `{kind}` вокруг руки `{arm}`.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if LOG.exists() and "MAEFT_DONE " in LOG.read_text(encoding="utf-8"):
        log("MAEFT_DONE уже есть")
        return
    if not TRAIN_JSON.exists():
        raise SystemExit("нет artifacts/eval/s3_maeft_train.json")
    train = json.loads(TRAIN_JSON.read_text(encoding="utf-8"))
    if train.get("real_traces_used") or train.get("real_noise_used"):
        raise SystemExit("train json сообщает реальное обучение")
    t0 = time.time()
    shutil.rmtree(TMP, ignore_errors=True)
    try:
        models = {arm: load_arm(arm) for arm in ARMS}
        shapes = {
            "v8": {"val": shapes_of("v8", "val"), "test": shapes_of("v8", "test")},
            "v10": {"test": shapes_of("v10", "test")},
            "v11": {"test": shapes_of("v11", "test")},
        }
        for arm in ARMS:
            predict_split(models[arm], arm, "v8", "val", shapes["v8"]["val"])
        del models
        thr, n_bg = thresholds_val(shapes["v8"]["val"])
        val = score_split("v8", "val", shapes["v8"]["val"], ARMS, thr)
        selected = choose_candidate({name: val[name]["iou"] for name in CANDIDATES})
        log(f"выбран {selected} val {val[selected]['iou']:.4f}")
        nested = nested_ks(shapes["v8"]["val"], selected)
        events = event_recall(shapes["v8"]["val"], selected, thr)
        kind, arm = split_candidate(selected)
        # val-карты больше не нужны
        shutil.rmtree(TMP / "mae" / "v8" / "val", ignore_errors=True)
        shutil.rmtree(TMP / "random" / "v8" / "val", ignore_errors=True)
        shutil.rmtree(TMP / "v810" / "v8" / "val", ignore_errors=True)
        models = {arm: load_arm(arm)}
        extra = {}
        frozen = {name: thr[name] for name in (selected, arm, WINNER)}
        for ver, split in (("v8", "test"), ("v10", "test"), ("v11", "test")):
            predict_split(models[arm], arm, ver, split, shapes[ver][split])
            block = score_split(ver, split, shapes[ver][split], (arm,), frozen)
            # score_split также считает другие смеси этой руки с чужими порогами.
            # В отчёт кладём только выбранную смесь, саму руку и победителя.
            keep = {}
            for name in (arm, WINNER):
                keep[name] = block[name]
            # выбранная смесь посчитана, только если arm — её рука. Да.
            if selected not in block:
                raise SystemExit(f"нет {selected} в {ver}")
            keep[selected] = block[selected]
            extra[split if ver == "v8" else ver] = keep
            shutil.rmtree(TMP / arm / ver / split, ignore_errors=True)
            log(f"{ver} {split} готово")
        sanity = dict(
            val_abs=abs(float(val[WINNER]["iou"]) - WINNER_VAL),
            test_abs=abs(float(extra["test"][WINNER]["iou"]) - WINNER_TEST),
            v10_abs=abs(float(extra["v10"][WINNER]["iou"]) - WINNER_V10),
            v11_abs=abs(float(extra["v11"][WINNER]["iou"]) - WINNER_V11),
            n_bg=int(n_bg),
            n_bg_pub=PUB_N_BG,
        )
        if sanity["val_abs"] >= 0.005 or sanity["test_abs"] >= 0.005 or sanity["v10_abs"] >= 0.01 or sanity["v11_abs"] >= 0.01:
            raise SystemExit(f"протокол победителя не сошёлся {sanity}")
        gain = float(nested["gain_min"])
        val_iou = float(val[selected]["iou"])
        val_ok = bool(val_iou > WINNER_VAL and val_iou > float(val[WINNER]["iou"]))
        nested_ok = bool(gain + 1e-15 >= NESTED_GAIN)
        v10_ok = bool(float(extra["v10"][selected]["iou"]) + 1e-6 >= float(extra["v10"][WINNER]["iou"]))
        v11_ok = bool(float(extra["v11"][selected]["iou"]) + 1e-6 >= float(extra["v11"][WINNER]["iou"]))
        accept = bool(val_ok and nested_ok and v10_ok and v11_ok)
        beats = bool(accept and val_iou > WINNER_VAL + 1e-4)
        decision = (
            f"Выбран {selected}: val {val_iou:.4f} против победителя {val[WINNER]['iou']:.4f}. "
            f"Вложенный прирост min(k2,k5) {gain:+.4f} (нужно ≥ {NESTED_GAIN:.3f}). "
            f"v10 {extra['v10'][selected]['iou']:.4f} / {extra['v10'][WINNER]['iou']:.4f}, "
            f"v11 {extra['v11'][selected]['iou']:.4f} / {extra['v11'][WINNER]['iou']:.4f}. "
            f"Рука {arm} одна: val {val[arm]['iou']:.4f}, test {extra['test'][arm]['iou']:.4f}."
        )
        rep = dict(
            name="MAEFT",
            accept=accept,
            beats=beats,
            decision_code="accept" if accept else "reject",
            decision=decision,
            selected=selected,
            arm=arm,
            blend=kind,
            n_bg=int(n_bg),
            sanity=sanity,
            gates=dict(val_ok=val_ok, nested_ok=nested_ok, v10_ok=v10_ok, v11_ok=v11_ok),
            val=val,
            test=extra["test"],
            v10=extra["v10"],
            v11=extra["v11"],
            nested=nested,
            events_val=events,
            train=train,
            real_traces_used=False,
            eval_seconds=round(time.time() - t0, 1),
            candidates=list(CANDIDATES),
            published=dict(val=WINNER_VAL, test=WINNER_TEST, v10=WINNER_V10, v11=WINNER_V11),
        )
        dump_json(OUT_JSON, rep)
        DOC.parent.mkdir(parents=True, exist_ok=True)
        DOC.write_text(render_md(rep), encoding="utf-8")
        line = (
            f"MAEFT_DONE selected={selected} val={val_iou:.4f} test={extra['test'][selected]['iou']:.4f} "
            f"alone_val={val[arm]['iou']:.4f} winner_val={val[WINNER]['iou']:.6f} "
            f"nested5_delta={nested['5']['delta']:+.4f} nested2_delta={nested['2']['delta']:+.4f} "
            f"nested5_sel={nested['5']['selected_iou']:.4f} nested5_winner={nested['5']['winner_iou']:.4f} "
            f"v10={extra['v10'][selected]['iou']:.4f} v10_winner={extra['v10'][WINNER]['iou']:.4f} "
            f"v11={extra['v11'][selected]['iou']:.4f} v11_winner={extra['v11'][WINNER]['iou']:.4f} "
            f"accept={int(accept)} decision={'accept' if accept else 'reject'} "
            f"best_mae={train['best']['mae']['epoch']} best_random={train['best']['random']['epoch']} "
            f"best_v810={train['best']['v810']['epoch']} train_sec={train.get('seconds')} "
            f"eval_sec={rep['eval_seconds']:.0f} copied={train.get('mae_copied')} real_train=0"
        )
        elog(line)
        if beats:
            elog(f"BEATS_WINNER MAEFT {selected} val {val_iou:.4f} test {extra['test'][selected]['iou']:.4f}")
        log("eval готово")
    finally:
        shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
