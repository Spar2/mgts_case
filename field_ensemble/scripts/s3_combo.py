"""Смеси GEN2 с победителем и фильтром CDF(|dT_space|). Только оценка.

Веса зафиксированы до просмотра чисел и по val не подбираются.

(a) mean(GEN2, победитель) = GEN2 ens_blend = 0.5 * (gen2 + winner)
(b) (4 * (a) + CDF(|dT_space|)) / 5
(c) mean(GEN2, победитель, и тех из GEN2W24 / GEN2TCN, у кого есть DONE
    и чей тренер уже не пишет веса). Пока DONE нет, (c) не считается.

Порог FAR 1e-3 по чистому фону: valid и ev==0 и nu==0.
Полный v8 val включает четыре прокси-файла, по которым выбирался чекпойнт GEN2
(seq0006, seq0021, seq0047, seq0048). Это сверка с опубликованным ens_blend.
Честная val — остальные 41, порог заново по их фону.
test / v10 / v11 идут с порогом полного v8 val.
Бутстреп: 1000 перевыборок последовательностей, перцентили 2.5 и 97.5,
seed 20261016, одна и та же схема индексов на сплите для (a) и (b).
Вложенные k=5 и k=2: s3_sel2_cv.eval_one, sequence_folds i%k.
Основные фолды — 41 файл. Фолды на всех 45 помечены как прокси-загрязнённые.
Дельта (b)-(a) только описывает добавку фильтра и вес не выбирает.

Правило beats записано здесь, до прогона, и по результату не меняется.
Для каждой смеси отдельно beats=1 только если все пункты верны:
- IoU holdout-41 с порогом, подобранным на этих 41, > 0.38671235821914546 + 1e-4
- вложенная дельта k=5 на этих 41 >= 0.003
- IoU v10 и v11 не ниже опубликованных констант победителя
- нижняя граница ДИ holdout-41 (порог этих 41) > 0
- протокол сошёлся: победитель и (a) на полном val, test, v10, v11
  отличаются от опубликованных IoU не больше чем на 0.005
Иначе beats=0. Строка BEATS_WINNER пишется только при beats=1.
infer.py не меняется. Реальные CSV не читаются.

Карты GEN2 лежат в /tmp/s3_combo_maps и остаются, пока (c) не посчитана.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_k] = "2"

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)

from dts import data, metrics as M, models
from dts.mae_anom import apply_cdf
from s3_gen2_train import predict_batched

WINNER = "ens_u2ft_tcn_lgbm"
WINNER_VAL = 0.38671235821914546
WINNER_PUB = {
    "val": 0.38671235821914546,
    "test": 0.4052762065230473,
    "v10": 0.4038977883892501,
    "v11": 0.40390028295846503,
}
BLEND_PUB = {
    "val": 0.4106385680480887,
    "test": 0.4579198080515615,
    "v10": 0.4576271959761577,
    "v11": 0.4575285664488177,
}
PROXY = ("seq0006", "seq0021", "seq0047", "seq0048")
BOOT_N = 1000
BOOT_SEED = 20261016
MATCH_TOL = 0.005
NESTED_DELTA = 0.003
BEAT_GAP = 1e-4
MAPS = Path("/tmp/s3_combo_maps")
OUT = Path("artifacts/eval/s3_combo_report.json")
FAIL = Path("artifacts/eval/s3_combo_fail.json")
DOC = Path("docs/S3_COMBO.md")
LOG = Path("/workspace/run_logs/_events.log")
GEN2_PT = Path("artifacts/eval/s3_gen2_unet.pt")
PREV = Path("artifacts/eval/s3_gen2_v8.json")
CHECK = Path("artifacts/eval/s3_gen2_check.json")
KNOTS_JSON = Path("artifacts/eval/s3_mae_train.json")
KEYS = ("a", "b", "winner")


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S", time.gmtime()) + " " + msg, flush=True)


def elog(msg: str) -> None:
    log(msg)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(time.strftime("%H:%M", time.gmtime()) + " " + msg + "\n")


def has_tag(tag: str) -> bool:
    if not LOG.exists():
        return False
    for line in LOG.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split(" ", 1)
        if len(parts) == 2 and parts[1].startswith(tag + " "):
            return True
        if len(parts) == 2 and parts[1] == tag:
            return True
    return False


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


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(plain(obj), indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def iou_of(tp, fp, fn) -> float:
    return float(tp) / max(float(tp) + float(fp) + float(fn), 1.0)


def pack(rows):
    tp = int(sum(r[0] for r in rows))
    fp = int(sum(r[1] for r in rows))
    fn = int(sum(r[2] for r in rows))
    return dict(tp=tp, fp=fp, fn=fn, iou=iou_of(tp, fp, fn), n_seq=len(rows))


def bootstrap(new_rows, base_rows, n=BOOT_N, seed=BOOT_SEED):
    b = np.asarray(new_rows, np.float64)
    w = np.asarray(base_rows, np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(b), size=(n, len(b)))
    sb = b[idx].sum(axis=1)
    sw = w[idx].sum(axis=1)
    db = sb[:, 0] / np.maximum(sb.sum(axis=1), 1.0)
    dw = sw[:, 0] / np.maximum(sw.sum(axis=1), 1.0)
    delta = db - dw
    lo, mid, hi = np.percentile(delta, [2.5, 50, 97.5])
    point = pack(new_rows)["iou"] - pack(base_rows)["iou"]
    return dict(
        n=int(n), seed=int(seed), unit="sequence",
        point=float(point), mean=float(delta.mean()),
        p2_5=float(lo), p50=float(mid), p97_5=float(hi),
        frac_positive=float(np.mean(delta > 0)),
        covers_zero=bool(lo <= 0.0 <= hi),
    )


def split_block(new_rows, base_rows):
    pn, pb = pack(new_rows), pack(base_rows)
    ci = bootstrap(new_rows, base_rows)
    return dict(
        iou=pn["iou"], winner_iou=pb["iou"], delta=ci["point"],
        tp=pn["tp"], fp=pn["fp"], fn=pn["fn"],
        winner_tp=pb["tp"], winner_fp=pb["fp"], winner_fn=pb["fn"],
        n_seq=pn["n_seq"], ci=ci,
    )


def load_knots():
    body = json.loads(KNOTS_JSON.read_text(encoding="utf-8"))
    knots = np.asarray(body["cdf_knots"], np.float64).copy()
    if knots.size != 257:
        raise SystemExit(f"cdf_knots {knots.size} != 257")
    if np.any(np.diff(knots) <= 0):
        raise SystemExit("cdf_knots не строго возрастают")
    return knots


def load_gen2():
    model = models.build("unet", 16)
    state = torch.load(GEN2_PT, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def map_path(ver, split, stem) -> Path:
    return MAPS / f"{ver}_{split}" / f"{stem}.npy"


def map_ok(path: Path, shape) -> bool:
    if not path.exists():
        return False
    try:
        mm = np.load(path, mmap_mode="r")
        return tuple(mm.shape) == tuple(shape) and mm.dtype == np.float32
    except (OSError, ValueError):
        return False


def save_map(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial.npy")
    np.save(tmp, np.asarray(arr, np.float32))
    os.replace(tmp, path)


def cdf_channel(x1, valid, knots) -> np.ndarray:
    ch = np.nan_to_num(np.asarray(x1, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    v = np.asarray(valid, bool).astype(np.float32)
    return apply_cdf(np.abs(ch * v), knots)


def ensure_gen2(model, ver, split):
    files = data.list_cache(f"artifacts/cache/{ver}", split)
    t0 = time.time()
    for i, p in enumerate(files, 1):
        with np.load(p) as z:
            shape = tuple(np.asarray(z["ev"]).shape)
        out = map_path(ver, split, p.stem)
        if map_ok(out, shape):
            if i == 1 or i % 15 == 0 or i == len(files):
                log(f"map {ver} {split} {i}/{len(files)} есть {time.time() - t0:.0f}s")
            continue
        with np.load(p) as z:
            X = np.asarray(z["X"], np.float32)
        prob = predict_batched(model, X)
        s = (1.0 - prob[0]).astype(np.float32)
        del X, prob
        if tuple(s.shape) != shape:
            raise SystemExit(f"{p.stem} карта {s.shape} != {shape}")
        save_map(out, s)
        del s
        log(f"map {ver} {split} {i}/{len(files)} {p.stem} {time.time() - t0:.0f}s")
    return files


def load_seq(ver, split, path, knots, with_xy=False):
    stem = path.stem
    gpath = map_path(ver, split, stem)
    gen2 = np.load(gpath, mmap_mode="r")
    with np.load(path) as z:
        ev = np.asarray(z["ev"])
        valid = np.asarray(z["valid"], bool)
        nu = np.asarray(z["nu"])
        x1 = np.asarray(z["X"][1], np.float32)
        x = np.asarray(z["x"], np.float64) if with_xy else None
        t = np.asarray(z["t"], np.float64) if with_xy else None
    if tuple(gen2.shape) != tuple(ev.shape):
        raise SystemExit(f"{stem} gen2 {gen2.shape} != {ev.shape}")
    pred = Path("artifacts/pred") / ver / WINNER / split / path.name
    if not pred.exists():
        raise SystemExit(f"нет карты победителя {pred}")
    with np.load(pred) as pz:
        ws = np.asarray(pz["s"], np.float32)
    if ws.shape != ev.shape:
        raise SystemExit(f"{stem} winner {ws.shape} != {ev.shape}")
    g = np.asarray(gen2, np.float32)
    cdf = cdf_channel(x1, valid, knots)
    a = (0.5 * (g + ws)).astype(np.float32)
    b = ((4.0 * a + cdf) / 5.0).astype(np.float32)
    del g, cdf, x1, gen2
    bg = valid & (ev == 0) & (nu == 0)
    return dict(stem=stem, path=path, a=a, b=b, winner=ws, ev=ev, valid=valid, nu=nu, bg=bg, x=x, t=t)


def fit_thresholds(files, ver, split, knots):
    parts = {k: [] for k in KEYS}
    n_bg = 0
    for i, p in enumerate(files, 1):
        seq = load_seq(ver, split, p, knots)
        n_bg += int(seq["bg"].sum())
        for k in KEYS:
            parts[k].append(np.asarray(seq[k][seq["bg"]], np.float32).copy())
        del seq
        if i == 1 or i % 15 == 0 or i == len(files):
            log(f"порог {ver} {split} {i}/{len(files)} n_bg {n_bg}")
    out = {}
    for k in KEYS:
        bg = np.concatenate(parts[k])
        parts[k] = None
        out[k] = dict(threshold=float(M.far_threshold(bg, 1e-3)), n_bg=int(bg.size))
        del bg
        gc.collect()
    return out


def score_files(files, ver, split, knots, thr):
    rows = {k: [] for k in KEYS}
    for i, p in enumerate(files, 1):
        seq = load_seq(ver, split, p, knots)
        truth = seq["ev"] > 0
        valid = seq["valid"]
        for k in KEYS:
            rows[k].append(M.confusion((seq[k] >= thr[k]) & valid, truth, valid))
        del seq
        if i == 1 or i % 15 == 0 or i == len(files):
            log(f"счёт {ver} {split} {i}/{len(files)}")
    return rows


def check_close(got, exp, label) -> None:
    err = abs(float(got) - float(exp))
    log(f"сверка {label} {got:.6f} эталон {exp:.6f} |Δ| {err:.6g}")
    if err > MATCH_TOL:
        dump(FAIL, dict(label=label, got=float(got), expected=float(exp), abs_err=err))
        raise SystemExit(f"протокол {label}: {got} != {exp} (|Δ| {err} > {MATCH_TOL})")


def build_frames(files, ver, knots):
    import s3_sel2_cv as s2
    import s3_sel_eval as ev
    import pandas as pd

    labels = pd.read_csv("data/synth_meta/v8/labels.csv")
    seqs = pd.read_csv("data/synth_meta/v8/sequences.csv").set_index("seq_id")
    buckets = {k: [] for k in KEYS}
    for i, p in enumerate(files, 1):
        seq = load_seq(ver, "val", p, knots, with_xy=True)
        ev_idx, nu_idx, bg_idx, n_bg = ev._indices(seq["ev"], seq["valid"], seq["nu"], seq["stem"])
        cooling, _rows = ev._events_for(seq["stem"], seq["ev"], seq["x"], seq["t"], labels, seqs, ev_idx)

        def frame(arr, stem=seq["stem"], cooling=cooling, n_bg=n_bg, ev_idx=ev_idx, nu_idx=nu_idx, bg_idx=bg_idx):
            e, n, b = s2._cols(arr, ev_idx, nu_idx, bg_idx)
            return dict(ev_m=e[None], nu_m=n[None], bg_m=b[None], n_bg=n_bg, cooling=cooling, stem=stem)

        for k in KEYS:
            buckets[k].append(frame(seq[k]))
        del seq
        if i == 1 or i % 15 == 0 or i == len(files):
            log(f"кадры {i}/{len(files)}")
    order = np.argsort([fr["stem"] for fr in buckets["winner"]])
    return {k: [buckets[k][i] for i in order] for k in KEYS}


def nested_on(frames):
    import s3_sel2_cv as s2
    import s3_sel_core as sel

    winner = frames["winner"]
    out = {
        "stems": [fr["stem"] for fr in winner],
        "procedure": "s3_sel2_cv sequence_folds i%k, holdout FAR 1e-3, micro, фон до 200k на файл",
    }
    for name in ("a", "b"):
        out[name] = {}
        for k in (2, 5):
            folds = sel.sequence_folds(len(winner), k)
            parts_w, parts_m, rows = [], [], []
            for r, idx in enumerate(folds):
                hw = s2.eval_one(s2.take(winner, idx))
                hm = s2.eval_one(s2.take(frames[name], idx))
                parts_w.append(hw)
                parts_m.append(hm)
                rows.append(dict(
                    fold=r, n_seq=len(idx),
                    winner=float(hw["iou"]), selected=float(hm["iou"]),
                    delta=float(hm["iou"] - hw["iou"]),
                ))
            iw = float(sel.micro_iou(parts_w))
            im = float(sel.micro_iou(parts_m))
            out[name][str(k)] = dict(
                winner_iou=iw, selected_iou=im, delta=float(im - iw), folds=rows,
            )
            log(f"nested {name} k={k} {im:.4f} winner {iw:.4f} delta {im - iw:+.4f}")
    return out


def judge(name, hold, nested, v10, v11, protocol_ok):
    val_iou = float(hold["iou"])
    k5 = float(nested["5"]["delta"])
    lo = float(hold["ci"]["p2_5"])
    val_ok = bool(val_iou > WINNER_VAL + BEAT_GAP)
    nest_ok = bool(k5 + 1e-15 >= NESTED_DELTA)
    v10_ok = bool(float(v10["iou"]) + 1e-6 >= WINNER_PUB["v10"])
    v11_ok = bool(float(v11["iou"]) + 1e-6 >= WINNER_PUB["v11"])
    ci_ok = bool(lo > 0.0)
    beats = bool(protocol_ok and val_ok and nest_ok and v10_ok and v11_ok and ci_ok)
    return dict(
        name=name, val_ok=val_ok, nest_ok=nest_ok, v10_ok=v10_ok, v11_ok=v11_ok,
        ci_ok=ci_ok, beats=beats, protocol_ok=bool(protocol_ok),
        hold41_iou=val_iou, k5_delta=k5, ci_lo=lo,
        v10_iou=float(v10["iou"]), v11_iou=float(v11["iou"]),
    )


def f6(x) -> str:
    return f"{float(x):+.6f}"


def boot_txt(block) -> str:
    c = block["ci"]
    return f"d={f6(block['delta'])} ci={f6(c['p2_5'])},{f6(c['p97_5'])} z={int(c['covers_zero'])}"


def cell(block) -> str:
    c = block["ci"]
    return (
        f"{block['iou']:.6f} | {block['winner_iou']:.6f} | {block['delta']:+.5f} | "
        f"{c['p2_5']:+.5f} | {c['p97_5']:+.5f} | {int(c['covers_zero'])}"
    )


def render_doc(r) -> str:
    a, b = r["a"], r["b"]
    ja, jb = r["judge"]["a"], r["judge"]["b"]
    lines = [
        "# Смеси GEN2 и CDF(|dT_space|)",
        "",
        "Веса заданы до просмотра этих чисел и по val не подбирались.",
        "",
        "- (a) среднее GEN2 и победителя `ens_u2ft_tcn_lgbm`. Это GEN2 ens_blend: `0.5 * (gen2 + winner)`.",
        "- (b) `(4 * (a) + CDF(|dT_space|)) / 5`. Узлы CDF — 257 опубликованных значений из `artifacts/eval/s3_mae_train.json`, канал 1 кэша. Новые узлы не оценивались.",
        "- (c) среднее GEN2, победителя и готовых GEN2W24 / GEN2TCN. Сейчас тренеры без файла DONE, снимок весов в оценку не входил.",
        "",
        "Порог — FAR 1e-3 по чистому фону (`valid`, `ev==0`, `nu==0`), micro-IoU по сумме tp/fp/fn.",
        "Полный v8 val содержит четыре файла, по которым выбирался чекпойнт GEN2: "
        + ", ".join(r["proxy_files"])
        + ". Строка val45 сверяет (a) с опубликованным ens_blend. Честная val — 41 файл, порог подобран заново по их фону.",
        "test, v10 и v11 используют порог полного v8 val. В выбор чекпойнта они не входили.",
        "Бутстреп — 1000 перевыборок последовательностей, перцентили 2.5 и 97.5, seed 20261016. На одном сплите (a) и (b) делят одну схему индексов.",
        "Вложенные k=5 и k=2 — `scripts/s3_sel2_cv.py`: фолды `i % k` по отсортированным именам, порог только на holdout-фолде, фон до 200k клеток на файл.",
        "Дельта (b)−(a) описывает добавку фильтра поверх (a). Новый вес по ней не выбирается.",
        "",
        "Правило beats для каждой смеси отдельно, до прогона: IoU 41 файла с их порогом выше 0.38671235821914546 на 1e-4, вложенная дельта k=5 на этих 41 не меньше 0.003, v10 и v11 не ниже опубликованного победителя, нижняя граница ДИ этих 41 выше 0, и сверка победителя и (a) с публикацией в пределах 0.005. Иначе beats=0. `infer.py` не менялся. Реальные CSV не читались.",
        "",
        "## (a) mean(GEN2, победитель)",
        "",
        "| сплит | IoU | победитель | Δ | ДИ 2.5 | ДИ 97.5 | ДИ включает 0 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        f"| val45, порог полного val | {cell(a['val45'])} |",
        f"| val41, порог 41 | {cell(a['hold41_refit'])} |",
        f"| val41, порог полного val | {cell(a['hold41_frozen'])} |",
        f"| v8 test | {cell(a['test'])} |",
        f"| v10 test | {cell(a['v10'])} |",
        f"| v11 test | {cell(a['v11'])} |",
        "",
        f"Вложенные 41: k5 Δ {a['nested41']['5']['delta']:+.5f}, k2 Δ {a['nested41']['2']['delta']:+.5f}.",
        f"Вложенные 45 (в фолдах остаются четыре прокси-файла): k5 Δ {a['nested45']['5']['delta']:+.5f}, k2 Δ {a['nested45']['2']['delta']:+.5f}.",
        f"Пороги (a): полный val {r['thresholds_full']['a']['threshold']:.6f} (опубликован {r['frozen_published']['a']:.6f}), 41 файл {r['thresholds_hold41']['a']['threshold']:.6f}.",
        f"Пункты beats: val41 {int(ja['val_ok'])}, k5 {int(ja['nest_ok'])}, v10 {int(ja['v10_ok'])}, v11 {int(ja['v11_ok'])}, ДИ {int(ja['ci_ok'])}. beats={int(ja['beats'])}.",
        "",
        "## (b) (4×(a) + CDF(|dT_space|))/5",
        "",
        "| сплит | IoU | победитель | Δ | ДИ 2.5 | ДИ 97.5 | ДИ включает 0 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        f"| val45, порог полного val | {cell(b['val45'])} |",
        f"| val41, порог 41 | {cell(b['hold41_refit'])} |",
        f"| val41, порог полного val | {cell(b['hold41_frozen'])} |",
        f"| v8 test | {cell(b['test'])} |",
        f"| v10 test | {cell(b['v10'])} |",
        f"| v11 test | {cell(b['v11'])} |",
        "",
        f"Вложенные 41: k5 Δ {b['nested41']['5']['delta']:+.5f}, k2 Δ {b['nested41']['2']['delta']:+.5f}.",
        f"Вложенные 45 (прокси-файлы внутри фолдов): k5 Δ {b['nested45']['5']['delta']:+.5f}, k2 Δ {b['nested45']['2']['delta']:+.5f}.",
        f"Пороги (b): полный val {r['thresholds_full']['b']['threshold']:.6f}, 41 файл {r['thresholds_hold41']['b']['threshold']:.6f}.",
        f"Пункты beats: val41 {int(jb['val_ok'])}, k5 {int(jb['nest_ok'])}, v10 {int(jb['v10_ok'])}, v11 {int(jb['v11_ok'])}, ДИ {int(jb['ci_ok'])}. beats={int(jb['beats'])}.",
        "",
        "## (b) − (a), описание",
        "",
        "Обе смеси уже стоят на своём пороге FAR 1e-3. Интервал не используется как критерий выбора веса.",
        "",
        "| сплит | Δ IoU | ДИ 2.5 | ДИ 97.5 | ДИ включает 0 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for key, title in (
        ("val45", "val45"),
        ("hold41_refit", "val41, свой порог"),
        ("hold41_frozen", "val41, порог полного val"),
        ("test", "v8 test"),
        ("v10", "v10 test"),
        ("v11", "v11 test"),
    ):
        d = r["b_minus_a"][key]
        lines.append(
            f"| {title} | {d['point']:+.5f} | {d['p2_5']:+.5f} | {d['p97_5']:+.5f} | {int(d['covers_zero'])} |"
        )
    lines += [
        "",
        "## Что из этого следует",
        "",
        f"Сверка с публикацией прошла: победитель и (a) на val/test/v10/v11 в пределах {MATCH_TOL}. "
        f"beats={int(r['beats'])}. Релиз и `infer.py` не переключались. real_train=0.",
        "(c) ждёт `artifacts/ckpt/s3_gen2w24/DONE` и `artifacts/ckpt/s3_gen2tcn/DONE`. "
        "Текущие `artifacts/eval/s3_gen2w24_unet.pt` и `s3_gen2tcn_tcn.pt` — снимки идущего обучения.",
        "Полные счётчики и фолды: `artifacts/eval/s3_combo_report.json`.",
        "",
    ]
    return "\n".join(lines)


def event_line(r) -> str:
    a, b = r["a"], r["b"]
    ja, jb = r["judge"]["a"], r["judge"]["b"]
    passed = [name for name, j in (("a", ja), ("b", jb)) if j["beats"]]
    token = ""
    if passed:
        token = " BEATS_WINNER COMBO " + ",".join(passed)
    ba = r["b_minus_a"]["hold41_refit"]
    return (
        "COMBO1 c=pending n_boot=1000 seed=20261016 "
        f"proxy={','.join(r['proxy_files'])} "
        f"a_val45_iou={a['val45']['iou']:.6f} a_val45={boot_txt(a['val45'])} "
        f"a_hold41_iou={a['hold41_refit']['iou']:.6f} a_hold41={boot_txt(a['hold41_refit'])} "
        f"a_hold41_frozen={boot_txt(a['hold41_frozen'])} "
        f"a_test={boot_txt(a['test'])} a_v10={boot_txt(a['v10'])} a_v11={boot_txt(a['v11'])} "
        f"a_k5={a['nested41']['5']['delta']:+.6f} a_k2={a['nested41']['2']['delta']:+.6f} "
        f"a_k5_45={a['nested45']['5']['delta']:+.6f} a_k2_45={a['nested45']['2']['delta']:+.6f} "
        f"b_val45_iou={b['val45']['iou']:.6f} b_val45={boot_txt(b['val45'])} "
        f"b_hold41_iou={b['hold41_refit']['iou']:.6f} b_hold41={boot_txt(b['hold41_refit'])} "
        f"b_hold41_frozen={boot_txt(b['hold41_frozen'])} "
        f"b_test={boot_txt(b['test'])} b_v10={boot_txt(b['v10'])} b_v11={boot_txt(b['v11'])} "
        f"b_k5={b['nested41']['5']['delta']:+.6f} b_k2={b['nested41']['2']['delta']:+.6f} "
        f"b_k5_45={b['nested45']['5']['delta']:+.6f} b_k2_45={b['nested45']['2']['delta']:+.6f} "
        f"b_minus_a_hold41=d={f6(ba['point'])} ci={f6(ba['p2_5'])},{f6(ba['p97_5'])} "
        f"beats_a={int(ja['beats'])} beats_b={int(jb['beats'])} beats={int(r['beats'])} "
        f"real_train=0 infer=unchanged"
        + token
    )


def member_status() -> dict:
    out = {}
    for run, pt in (
        ("s3_gen2w24", "artifacts/eval/s3_gen2w24_unet.pt"),
        ("s3_gen2tcn", "artifacts/eval/s3_gen2tcn_tcn.pt"),
    ):
        done = Path("artifacts/ckpt") / run / "DONE"
        alive = False
        for proc in Path("/proc").iterdir():
            if not proc.name.isdigit():
                continue
            try:
                cmd = (proc / "cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="ignore")
            except OSError:
                continue
            if f"scripts/{run}_train.py" in cmd:
                alive = True
                break
        out[run] = dict(done=done.exists(), pt=Path(pt).exists(), trainer_alive=alive, scored=False)
    return out


def main():
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if has_tag("COMBO1") or has_tag("COMBO_DONE"):
        log("строка COMBO уже есть, повторно не пишу")
        return
    t0 = time.time()
    knots = load_knots()
    prev = json.loads(PREV.read_text(encoding="utf-8"))
    frozen_pub = {
        "a": float(prev["val"]["ens_blend"]["threshold"]),
        "winner": float(prev["val"][WINNER]["threshold"]),
    }
    if abs(frozen_pub["a"] - 0.8579896092424856) > 1e-12:
        raise SystemExit("порог ens_blend в s3_gen2_v8.json съехал")
    if abs(frozen_pub["winner"] - 0.8608087301264272) > 1e-12:
        raise SystemExit("порог победителя в s3_gen2_v8.json съехал")
    files = data.list_cache("artifacts/cache/v8", "val")
    if len(files) != 45 or tuple(p.stem for p in files[:4]) != PROXY:
        raise SystemExit(f"прокси { [p.stem for p in files[:4]] }")
    hold_files = files[4:]
    if len(hold_files) != 41:
        raise SystemExit("holdout != 41")
    log("proxy " + " ".join(PROXY))
    model = load_gen2()
    ensure_gen2(model, "v8", "val")
    del model
    gc.collect()

    fitted = fit_thresholds(files, "v8", "val", knots)
    # (a) и победитель на внешних сплитах и на val45 — по опубликованному порогу.
    # (b) — по порогу, оценённому на полном v8 val.
    thr_full = {
        "a": frozen_pub["a"],
        "b": fitted["b"]["threshold"],
        "winner": frozen_pub["winner"],
    }
    rows_full = score_files(files, "v8", "val", knots, thr_full)
    check_close(pack(rows_full["winner"])["iou"], WINNER_PUB["val"], "winner val45")
    check_close(pack(rows_full["a"])["iou"], BLEND_PUB["val"], "a val45")

    fitted_hold = fit_thresholds(hold_files, "v8", "val", knots)
    thr_hold = {k: fitted_hold[k]["threshold"] for k in KEYS}
    rows_hold = score_files(hold_files, "v8", "val", knots, thr_hold)
    rows_hold_frozen = score_files(hold_files, "v8", "val", knots, thr_full)
    rows_proxy = score_files(files[:4], "v8", "val", knots, thr_full)
    if CHECK.exists():
        chk = json.loads(CHECK.read_text(encoding="utf-8"))
        check_close(pack(rows_hold["a"])["iou"], chk["holdout_refit"]["ens_blend"]["iou"], "a hold41 vs GEN2_CHECK")
        check_close(pack(rows_hold["winner"])["iou"], chk["holdout_refit"][WINNER]["iou"], "winner hold41 vs GEN2_CHECK")
        expect_stems = chk["nested41"]["stems"]
    else:
        expect_stems = None

    frames41 = build_frames(hold_files, "v8", knots)
    if expect_stems is not None and [fr["stem"] for fr in frames41["winner"]] != expect_stems:
        raise SystemExit("стемы 41 не совпали с GEN2_CHECK")
    nested41 = nested_on(frames41)
    del frames41
    gc.collect()
    frames45 = build_frames(files, "v8", knots)
    nested45 = nested_on(frames45)
    del frames45
    gc.collect()

    model = load_gen2()
    external = {}
    for ver, split, key in (("v8", "test", "test"), ("v10", "test", "v10"), ("v11", "test", "v11")):
        ext_files = ensure_gen2(model, ver, split)
        rows = score_files(ext_files, ver, split, knots, thr_full)
        check_close(pack(rows["winner"])["iou"], WINNER_PUB[key], f"winner {key}")
        check_close(pack(rows["a"])["iou"], BLEND_PUB[key], f"a {key}")
        external[key] = rows
        del rows
        gc.collect()
    del model
    gc.collect()

    def mix(key):
        return dict(
            val45=split_block(rows_full[key], rows_full["winner"]),
            hold41_refit=split_block(rows_hold[key], rows_hold["winner"]),
            hold41_frozen=split_block(rows_hold_frozen[key], rows_hold_frozen["winner"]),
            test=split_block(external["test"][key], external["test"]["winner"]),
            v10=split_block(external["v10"][key], external["v10"]["winner"]),
            v11=split_block(external["v11"][key], external["v11"]["winner"]),
            nested41={k: {kk: vv for kk, vv in nested41[key][k].items()} for k in ("2", "5")},
            nested45={k: {kk: vv for kk, vv in nested45[key][k].items()} for k in ("2", "5")},
            proxy4=pack(rows_proxy[key]),
        )

    # Для (b)-(a) база — строки (a), не победитель. Победитель в split_block не нужен:
    # bootstrap сравнивает две уже посчитанные таблицы.
    b_minus_a = {
        "val45": bootstrap(rows_full["b"], rows_full["a"]),
        "hold41_refit": bootstrap(rows_hold["b"], rows_hold["a"]),
        "hold41_frozen": bootstrap(rows_hold_frozen["b"], rows_hold_frozen["a"]),
        "test": bootstrap(external["test"]["b"], external["test"]["a"]),
        "v10": bootstrap(external["v10"]["b"], external["v10"]["a"]),
        "v11": bootstrap(external["v11"]["b"], external["v11"]["a"]),
        "note": "Описание. Вес по этой дельте не выбирается.",
    }
    mixes = {k: mix(k) for k in ("a", "b")}
    # nested внутри mix скопирован без ссылки на живые кадры. Фолды остаются.
    protocol_ok = True
    judged = {
        "a": judge("a", mixes["a"]["hold41_refit"], mixes["a"]["nested41"], mixes["a"]["v10"], mixes["a"]["v11"], protocol_ok),
        "b": judge("b", mixes["b"]["hold41_refit"], mixes["b"]["nested41"], mixes["b"]["v10"], mixes["b"]["v11"], protocol_ok),
    }
    beats = int(judged["a"]["beats"] or judged["b"]["beats"])
    report = dict(
        name="COMBO1",
        formulas=dict(
            a="0.5 * (gen2 + winner)",
            b="(4 * a + CDF(|dT_space|)) / 5",
            c="mean(gen2, winner, finished GEN2W24 and/or GEN2TCN); not scored",
        ),
        weights_fixed_before_eval=True,
        proxy_files=list(PROXY),
        proxy_note="Чекпойнт GEN2 выбран по этим четырём файлам v8 val. val45 их содержит. val41 и nested41 — нет.",
        winner_published=WINNER_PUB,
        blend_published=BLEND_PUB,
        frozen_published=frozen_pub,
        thresholds_full=fitted,
        thresholds_used_on_val45_and_external=thr_full,
        thresholds_hold41=fitted_hold,
        n_bg_full=fitted["a"]["n_bg"],
        n_bg_hold41=fitted_hold["a"]["n_bg"],
        a=mixes["a"],
        b=mixes["b"],
        b_minus_a=b_minus_a,
        nested41_stems=nested41["stems"],
        nested45_stems=nested45["stems"],
        judge=judged,
        beats_rule=dict(
            hold41_iou_gt=WINNER_VAL + BEAT_GAP,
            nested41_k5_delta_ge=NESTED_DELTA,
            v10_v11_not_below_published_winner=True,
            hold41_refit_ci_lo_gt=0,
            protocol_abs_err_le=MATCH_TOL,
            mixtures=("a", "b"),
            b_minus_a_does_not_select_weight=True,
            nested45_not_in_the_rule=True,
        ),
        beats=beats,
        members=member_status(),
        c_scored=False,
        maps=str(MAPS),
        maps_kept=True,
        infer_changed=False,
        real_traces_used=False,
        real_train=0,
        seconds=round(time.time() - t0, 1),
    )
    # Фолды вложенной проверки лежат в nested*; в смеси оставлены те же числа.
    report["nested41"] = {k: nested41[k] for k in ("stems", "procedure", "a", "b")}
    report["nested45"] = {k: nested45[k] for k in ("stems", "procedure", "a", "b")}
    dump(OUT, report)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(render_doc(report), encoding="utf-8")
    line = event_line(report)
    if beats and "BEATS_WINNER" not in line:
        raise SystemExit("beats=1 без токена")
    if not beats and "BEATS_WINNER" in line:
        raise SystemExit("beats=0, а токен попал в строку")
    elog(line)
    log(f"записан {OUT} за {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
