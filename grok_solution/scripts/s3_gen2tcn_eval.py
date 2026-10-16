"""Оценка GEN2TCN. Порог только по фону полного v8 val. Test, v10 и v11 его не двигают.

Смеси зафиксированы до прогона, веса не подбираются:

- tcn_gen2 — дообученный TCN;
- ens_tcn — mean(tcn_gen2, winner);
- ens_trio — mean(gen2, tcn_gen2, winner);
- ens_gens — mean(gen2, tcn_gen2);
- ens_tcnm4 — (4 * winner + tcn_gen2) / 5.

Базы для сравнения, не кандидаты: winner и ens_blend = mean(gen2, winner).
Выбор — больший IoU кандидатов на полном v8 val. Вложенные k=2 и k=5 —
процедура scripts/s3_sel2_cv.py, против победителя и против ens_blend.
В выбор они не входят.
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

from dts import data, metrics as M, models
from s3_gen2_train import freeze_bn, predict_batched

WINNER = "ens_u2ft_tcn_lgbm"
WINNER_VAL = 0.38671235821914546
WINNER_TEST = 0.4052762065230473
WINNER_V10 = 0.4038977883892501
WINNER_V11 = 0.40390028295846503
GEN2_BLEND_VAL = 0.4106385680480887
BEAT_MARGIN = 1e-4
BEST = Path("artifacts/eval/s3_gen2tcn_tcn.pt")
TRAIN_JSON = Path("artifacts/eval/s3_gen2tcn_train.json")
GEN2_W = Path("artifacts/eval/s3_gen2_unet.pt")
OUT = Path("artifacts/eval/s3_gen2tcn_v8.json")
DOC = Path("docs/S3_GEN2TCN.md")
LOG = Path("/workspace/grok_logs/_events.log")

CANDIDATES = ("tcn_gen2", "ens_tcn", "ens_trio", "ens_gens", "ens_tcnm4")
BASELINES = ("ens_blend", WINNER)
NAMES = CANDIDATES + BASELINES
MIXTURES = dict(
    tcn_gen2="tcn_gen2",
    ens_tcn="mean(tcn_gen2, winner)",
    ens_trio="mean(gen2, tcn_gen2, winner)",
    ens_gens="mean(gen2, tcn_gen2)",
    ens_tcnm4="(4*winner + tcn_gen2)/5",
    ens_blend="mean(gen2, winner)",
    winner=WINNER,
)


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S", time.gmtime()) + " " + msg, flush=True)


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


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(plain(obj), indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def load_tcn(path: Path):
    model = models.build("tcn", 32)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state and "body.0.weight" not in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    freeze_bn(model)
    return model


def load_gen2():
    model = models.build("unet", 16)
    state = torch.load(GEN2_W, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state and not any(str(k).startswith("e1") for k in state):
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    freeze_bn(model)
    return model


def mix(name: str, st: np.ndarray, sg: np.ndarray, ws: np.ndarray) -> np.ndarray:
    """st — gen2tcn, sg — gen2 U-Net, ws — победитель. Веса не учатся."""
    if name == "tcn_gen2":
        return st
    if name == "ens_tcn":
        return (0.5 * (st + ws)).astype(np.float32)
    if name == "ens_trio":
        return ((st + sg + ws) / 3.0).astype(np.float32)
    if name == "ens_gens":
        return (0.5 * (st + sg)).astype(np.float32)
    if name == "ens_tcnm4":
        return ((4.0 * ws + st) / 5.0).astype(np.float32)
    if name == "ens_blend":
        return (0.5 * (sg + ws)).astype(np.float32)
    if name == WINNER:
        return ws
    raise KeyError(name)


def bg_mask(ev, nu, valid):
    return np.asarray(valid, bool) & (np.asarray(ev) == 0) & (np.asarray(nu) == 0)


def fit_thresholds(tmp, files, split, ver) -> dict:
    ens = Path("artifacts/pred") / ver / WINNER / split
    masks = []
    n_bg = 0
    for p in files:
        with np.load(p) as z:
            mask = bg_mask(z["ev"], z["nu"], z["valid"])
        masks.append(mask)
        n_bg += int(mask.sum())
    thr = {}
    for n in NAMES:
        parts = []
        for p, mask in zip(files, masks):
            st = np.load(tmp / f"{p.stem}_t_s.npy", mmap_mode="r")
            sg = np.load(tmp / f"{p.stem}_g_s.npy", mmap_mode="r")
            with np.load(ens / p.name) as pred:
                ws = np.asarray(pred["s"], np.float32)
            parts.append(np.asarray(mix(n, np.asarray(st), np.asarray(sg), ws)[mask], np.float32))
            del st, sg, ws
        thr[n] = float(M.far_threshold(np.concatenate(parts), 1e-3))
        log(f"порог {n} {thr[n]:.6f} n_bg {n_bg}")
        del parts
    return thr


def score_split(nets, ver, split, thr, make_pair: bool):
    files = data.list_cache(f"artifacts/cache/{ver}", split)
    tmp = Path(f"/tmp/s3_gen2tcn_{ver}_{split}_{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    ens = Path("artifacts/pred") / ver / WINNER / split
    if not ens.is_dir():
        raise SystemExit(f"нет карт победителя {ens}")
    tcn, gnet = nets
    t0 = time.time()
    pair = None
    try:
        for i, p in enumerate(files, 1):
            with np.load(p) as z:
                X = np.asarray(z["X"], np.float32)
            for key, net in (("t", tcn), ("g", gnet)):
                prob = predict_batched(net, X)
                np.save(tmp / f"{p.stem}_{key}_s.npy", (1.0 - prob[0]).astype(np.float32))
                np.save(tmp / f"{p.stem}_{key}_h.npy", (prob[1] - prob[2]).astype(np.float16))
                del prob
            del X
            if i == 1 or i % 5 == 0 or i == len(files):
                log(f"pred {ver} {split} {i}/{len(files)} {time.time() - t0:.0f}s")
        if thr is None:
            thr = fit_thresholds(tmp, files, split, ver)
        if make_pair:
            pair = compact_pair(tmp, files)
        report = {}
        tot = {n: np.zeros(3, np.int64) for n in NAMES}
        hot = {n: np.zeros(3, np.int64) for n in NAMES}
        cold = {n: np.zeros(3, np.int64) for n in NAMES}
        far_n = {n: 0 for n in NAMES}
        far_d = {n: 0 for n in NAMES}
        for p in files:
            with np.load(p) as z:
                ev = np.asarray(z["ev"])
                valid = np.asarray(z["valid"], bool)
                nu = np.asarray(z["nu"])
            st = np.load(tmp / f"{p.stem}_t_s.npy")
            sg = np.load(tmp / f"{p.stem}_g_s.npy")
            ht = np.load(tmp / f"{p.stem}_t_h.npy").astype(np.float32)
            hg = np.load(tmp / f"{p.stem}_g_h.npy").astype(np.float32)
            with np.load(ens / p.name) as pred:
                ws = np.asarray(pred["s"], np.float32)
                wh = np.asarray(pred["h"], np.float32)
            truth = ev > 0
            hot_t = (ev >= 1) & (ev <= 4)
            cold_t = ev >= 5
            bg = bg_mask(ev, nu, valid)
            for n in NAMES:
                ps = mix(n, st, sg, ws)
                ph = mix(n, ht, hg, wh)
                pred_m = (ps >= thr[n]) & valid
                tot[n] += np.asarray(M.confusion(pred_m, truth, valid), np.int64)
                hot[n] += np.asarray(M.confusion(pred_m & (ph > 0), hot_t, valid), np.int64)
                cold[n] += np.asarray(M.confusion(pred_m & (ph <= 0), cold_t, valid), np.int64)
                far_n[n] += int((pred_m & bg).sum())
                far_d[n] += int(bg.sum())
                del ps, ph
            del st, sg, ht, hg, ws, wh
        for n in NAMES:
            row = dict(
                threshold=float(thr[n]),
                far=float(far_n[n] / max(far_d[n], 1)),
                n_bg=int(far_d[n]),
                **M.prf(*map(int, tot[n])),
                hot=M.prf(*map(int, hot[n])),
                cold=M.prf(*map(int, cold[n])),
            )
            report[n] = row
            log(f"{ver} {split} {n} IoU {row['iou']:.4f} F1 {row['f1']:.4f} FAR {row['far']:.5f}")
        return report, pair, thr
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def compact_pair(tmp, files):
    import s3_sel2_cv as s2
    import s3_sel_eval as ev

    labels = pd.read_csv("data/synth_meta/v8/labels.csv")
    seqs = pd.read_csv("data/synth_meta/v8/sequences.csv").set_index("seq_id")
    buckets = {n: [] for n in NAMES}
    ens = Path("artifacts/pred/v8") / WINNER / "val"
    for p in files:
        with np.load(p) as z:
            ev_map = np.asarray(z["ev"])
            valid = np.asarray(z["valid"])
            nu = np.asarray(z["nu"])
            x = np.asarray(z["x"], np.float64)
            t = np.asarray(z["t"], np.float64)
        ev_idx, nu_idx, bg_idx, n_bg = ev._indices(ev_map, valid, nu, p.stem)
        st = np.asarray(np.load(tmp / f"{p.stem}_t_s.npy"), np.float32)
        sg = np.asarray(np.load(tmp / f"{p.stem}_g_s.npy"), np.float32)
        with np.load(ens / p.name) as pred:
            ws = np.asarray(pred["s"], np.float32)
        cooling, _rows = ev._events_for(p.stem, ev_map, x, t, labels, seqs, ev_idx)

        def frame(arr, stem=p.stem, cooling=cooling, n_bg=n_bg, ev_idx=ev_idx, nu_idx=nu_idx, bg_idx=bg_idx):
            e, n, b = s2._cols(arr, ev_idx, nu_idx, bg_idx)
            return dict(ev_m=e[None], nu_m=n[None], bg_m=b[None], n_bg=n_bg, cooling=cooling, stem=stem)

        for n in NAMES:
            buckets[n].append(frame(mix(n, st, sg, ws)))
        del st, sg, ws
    order = np.argsort([fr["stem"] for fr in buckets[WINNER]])
    return {n: [buckets[n][i] for i in order] for n in buckets}


def nested_ks(frames) -> dict:
    import s3_sel2_cv as s2
    import s3_sel_core as sel

    out = {
        "procedure": "scripts/s3_sel2_cv.py sequence_folds i%k, holdout FAR 1e-3, micro IoU",
        "stems": [fr["stem"] for fr in frames[WINNER]],
    }
    for name in CANDIDATES:
        out[name] = {}
        for base in ("winner", "ens_blend"):
            base_frames = frames[WINNER] if base == "winner" else frames["ens_blend"]
            out[name][base] = {}
            for k in (2, 5):
                folds = sel.sequence_folds(len(base_frames), k)
                parts_b, parts_s, rows = [], [], []
                for r, idx in enumerate(folds):
                    hb = s2.eval_one(s2.take(base_frames, idx))
                    hs = s2.eval_one(s2.take(frames[name], idx))
                    parts_b.append(hb)
                    parts_s.append(hs)
                    rows.append(dict(
                        fold=r, n_seq=len(idx),
                        baseline=float(hb["iou"]), selected=float(hs["iou"]),
                        delta=float(hs["iou"] - hb["iou"]),
                    ))
                    log(
                        f"nested {name} vs {base} k={k} fold {r} "
                        f"base {hb['iou']:.4f} sel {hs['iou']:.4f} d {hs['iou'] - hb['iou']:+.4f}"
                    )
                ib = float(sel.micro_iou(parts_b))
                iss = float(sel.micro_iou(parts_s))
                out[name][base][str(k)] = dict(
                    baseline_iou=ib, selected_iou=iss, delta=float(iss - ib), folds=rows,
                )
                log(f"nested {name} vs {base} k={k} micro {iss:.4f} base {ib:.4f} delta {iss - ib:+.4f}")
    return out


def render(rep: dict) -> str:
    tr = rep["train"]
    val, test = rep["val"], rep["test"]
    v10, v11 = rep["v10_test"], rep["v11_test"]
    lines = [
        "# S3 GEN2TCN",
        "",
        "Дообучение TCN участника (`artifacts/stage1/tcn.pt`, ширина 32) тем же генератором",
        "`dts/gen2.py`, что и GEN2 U-Net. Рецепт шага — `scripts/s3_gen2_train.py`:",
        "focal+tversky_ce, AdamW (тело 1e-5, голова 8e-5), weight decay 0, BatchNorm заморожен,",
        "тайл 128×256, batch 6, чередование v8 mmap и v10 npz, клип градиента 1, 2 потока.",
        "Потолок 50 минут. Реальные CSV, карточки и спектр шума в градиент и в порог не входили.",
        "",
        "Чекпойнт выбран по прокси IoU смеси mean(GEN2 U-Net, этот TCN) на всём v10 val",
        "при FAR 1e-3. Порог прокси — фон только этих файлов. v8 val в прокси не входит.",
        f"Лучший шаг {tr.get('best_step')}, прокси IoU {tr.get('best_proxy_iou')},",
        f"остановка по бюджету обучения, {tr.get('seconds')} с.",
        "",
    ]
    hist = tr.get("history") or []
    if hist:
        bits = ", ".join(
            f"шаг {h.get('step')} → {float(h.get('proxy_blend_iou')):.4f}"
            for h in hist
        )
        lines.append(f"Прокси на v10 val: {bits}.")
        if tr.get("best_step") == 0:
            lines.append(
                "Шаги после инициализации прокси ухудшили. В оценку пошли веса шага 0, "
                "совпадающие с `artifacts/stage1/tcn.pt`."
            )
        lines.append("")
    lines += [
        "Официальный порог каждого метода — свой квантиль чистого фона всего v8 val при FAR 1e-3.",
        "Test, v10 и v11 его не двигают. Смеси зафиксированы заранее:",
        "",
        "- `tcn_gen2` — один дообученный TCN;",
        "- `ens_tcn` — mean(tcn_gen2, победитель);",
        "- `ens_trio` — mean(GEN2 U-Net, tcn_gen2, победитель);",
        "- `ens_gens` — mean(GEN2 U-Net, tcn_gen2);",
        "- `ens_tcnm4` — (4×победитель + tcn_gen2) / 5.",
        "",
        "Базы, не кандидаты: победитель `ens_u2ft_tcn_lgbm` и GEN2 `ens_blend` = mean(GEN2 U-Net, победитель).",
        "Выбор — больший IoU кандидатов на полном v8 val. Вложенные k=2 и k=5 —",
        "`sequence_folds` (i % k) и `eval_one` из `scripts/s3_sel2_cv.py`, micro по сумме tp/fp/fn,",
        "против победителя и против пересчитанного `ens_blend`. В выбор они не входят.",
        "",
        "## Официальный FAR 1e-3",
        "",
        "| кандидат | порог | val IoU | test IoU | v10 IoU | v11 IoU | Δ val к победителю | Δ val к ens_blend |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    wval = val[WINNER]["iou"]
    bval = val["ens_blend"]["iou"]
    for n in NAMES:
        lines.append(
            f"| {n} | {val[n]['threshold']:.6f} | {val[n]['iou']:.4f} | {test[n]['iou']:.4f} | "
            f"{v10[n]['iou']:.4f} | {v11[n]['iou']:.4f} | {val[n]['iou'] - wval:+.4f} | "
            f"{val[n]['iou'] - bval:+.4f} |"
        )
    lines += [
        "",
        f"Пересчёт победителя: val {wval:.6f} (опубликовано {WINNER_VAL:.6f}), "
        f"test {test[WINNER]['iou']:.6f}, v10 {v10[WINNER]['iou']:.6f}, v11 {v11[WINNER]['iou']:.6f}.",
        f"Пересчёт GEN2 ens_blend: val {bval:.6f} (опубликовано {GEN2_BLEND_VAL:.6f}), "
        f"test {test['ens_blend']['iou']:.6f}, v10 {v10['ens_blend']['iou']:.6f}, "
        f"v11 {v11['ens_blend']['iou']:.6f}.",
        "",
        "| кандидат | P val | R val | FAR val | FAR test |",
        "|---|---:|---:|---:|---:|",
    ]
    for n in NAMES:
        lines.append(
            f"| {n} | {val[n]['precision']:.4f} | {val[n]['recall']:.4f} | "
            f"{val[n]['far']:.5f} | {test[n]['far']:.5f} |"
        )
    lines += ["", "## Вложенные фолды", ""]
    for name in CANDIDATES:
        lines.append(f"### {name}")
        lines.append("")
        lines.append("| база | k | база IoU | кандидат | Δ |")
        lines.append("|---|---:|---:|---:|---:|")
        block = rep["nested"][name]
        for base in ("winner", "ens_blend"):
            for k in ("2", "5"):
                row = block[base][k]
                lines.append(
                    f"| {base} | {k} | {row['baseline_iou']:.4f} | {row['selected_iou']:.4f} | "
                    f"{row['delta']:+.4f} |"
                )
        lines.append("")
        folds = block["winner"]["5"]["folds"]
        bits = ", ".join(f"{r['delta']:+.3f}" for r in folds)
        lines.append(f"Фолды k=5 против победителя: {bits}.")
        lines.append("")
    sel = rep["selected"]
    lines += [
        "## Итог",
        "",
        rep["decision"],
        f"Выбран `{sel}`: val {rep['selected_val_iou']:.6f}, test {rep['selected_test_iou']:.6f}, "
        f"v10 {v10[sel]['iou']:.6f}, v11 {v11[sel]['iou']:.6f}.",
        "Победитель по умолчанию не менялся. `models_release/` и `dts/gen2.py` не трогались. real_train=0.",
        f"BEATS_WINNER: {'да' if rep['beats_winner'] else 'нет'}. "
        f"Выше GEN2 ens_blend: {'да' if rep['beats_gen2_blend'] else 'нет'}.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    if not BEST.is_file():
        raise SystemExit(f"нет {BEST}")
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    t0 = time.time()
    train = json.loads(TRAIN_JSON.read_text()) if TRAIN_JSON.is_file() else {}
    nets = (load_tcn(BEST), load_gen2())
    val, pair, thr = score_split(nets, "v8", "val", None, True)
    if abs(val[WINNER]["iou"] - WINNER_VAL) > 0.005:
        dump(OUT, dict(protocol_ok=False, val=val, reason="winner val mismatch"))
        raise SystemExit(f"победитель val {val[WINNER]['iou']} != {WINNER_VAL}")
    nest = nested_ks(pair)
    test, _p, _t = score_split(nets, "v8", "test", thr, False)
    if abs(test[WINNER]["iou"] - WINNER_TEST) > 0.005:
        dump(OUT, dict(protocol_ok=False, val=val, test=test, nested=nest, reason="winner test mismatch"))
        raise SystemExit(f"победитель test {test[WINNER]['iou']} != {WINNER_TEST}")
    v10, _p, _t = score_split(nets, "v10", "test", thr, False)
    v11, _p, _t = score_split(nets, "v11", "test", thr, False)
    sel = max(CANDIDATES, key=lambda n: val[n]["iou"])
    sel_iou = float(val[sel]["iou"])
    blend_iou = float(val["ens_blend"]["iou"])
    protocol_ok = abs(val[WINNER]["iou"] - WINNER_VAL) < 2e-4
    beats_w = bool(protocol_ok and sel_iou > WINNER_VAL + BEAT_MARGIN)
    blend_ok = abs(blend_iou - GEN2_BLEND_VAL) < 2e-3
    beats_g = bool(blend_ok and sel_iou > blend_iou + BEAT_MARGIN)
    if beats_w and beats_g:
        decision = (
            f"Принято по val. {sel} {sel_iou:.4f} выше победителя {WINNER_VAL:.4f} "
            f"и выше пересчитанного GEN2 ens_blend {blend_iou:.4f}. Test в выбор не входил."
        )
    elif beats_w:
        decision = (
            f"Выше победителя, не выше GEN2. {sel} {sel_iou:.4f} против победителя {WINNER_VAL:.4f} "
            f"и ens_blend {blend_iou:.4f}. Test в выбор не входил."
        )
    else:
        decision = (
            f"Отклонено относительно победителя. Лучший кандидат {sel} {sel_iou:.4f}, "
            f"победитель {val[WINNER]['iou']:.4f}, GEN2 ens_blend {blend_iou:.4f}."
        )
    k5w = nest[sel]["winner"]["5"]["delta"]
    k2w = nest[sel]["winner"]["2"]["delta"]
    k5g = nest[sel]["ens_blend"]["5"]["delta"]
    k2g = nest[sel]["ens_blend"]["2"]["delta"]
    report = dict(
        name="GEN2TCN",
        protocol=(
            "Checkpoint by mean(gen2, gen2tcn) IoU at FAR 1e-3 on all v10 val, not v8 val. "
            "Official selection is full v8 val IoU at FAR 1e-3 among fixed mixtures: "
            "tcn_gen2, mean(tcn_gen2, winner), mean(gen2, tcn_gen2, winner), "
            "mean(gen2, tcn_gen2), (4*winner+tcn_gen2)/5. "
            "Nested k=2 and k=5 follow scripts/s3_sel2_cv.py versus the winner and versus "
            "recomputed GEN2 ens_blend. Nested does not select. Synthetic train only."
        ),
        mixtures=MIXTURES,
        decision=decision,
        accept=beats_w,
        beats_winner=beats_w,
        beats_gen2_blend=beats_g,
        selected=sel,
        selected_val_iou=sel_iou,
        selected_test_iou=float(test[sel]["iou"]),
        protocol_ok=protocol_ok,
        gen2_blend_match=blend_ok,
        published_winner=dict(val_iou=WINNER_VAL, test_iou=WINNER_TEST, v10_test=WINNER_V10, v11_test=WINNER_V11),
        published_gen2_ens_blend_val=GEN2_BLEND_VAL,
        winner_recomputed=dict(
            val=val[WINNER]["iou"], test=test[WINNER]["iou"],
            v10=v10[WINNER]["iou"], v11=v11[WINNER]["iou"],
            val_match=abs(val[WINNER]["iou"] - WINNER_VAL) < 1e-6,
            test_match=abs(test[WINNER]["iou"] - WINNER_TEST) < 1e-6,
        ),
        gen2_blend_recomputed=dict(
            val=blend_iou, test=test["ens_blend"]["iou"],
            v10=v10["ens_blend"]["iou"], v11=v11["ens_blend"]["iou"],
        ),
        train=dict(
            best_step=train.get("best_step"),
            best_proxy_iou=train.get("best_proxy_iou"),
            seconds=train.get("seconds"),
            step=train.get("step"),
            params=train.get("params"),
            history=train.get("history"),
            proxy_not_v8_val=True,
            val_used_for_train=False,
            real_traces_used=False,
        ),
        val=val, test=test, v10_test=v10, v11_test=v11, nested=nest,
        real_used_for_training=False,
        real_used_for_threshold=False,
        real_noise_used=False,
        seconds=round(time.time() - t0, 1),
    )
    dump(OUT, report)
    DOC.write_text(render(report), encoding="utf-8")
    elog(
        f"GEN2TCN_DONE selected={sel} val={sel_iou:.6f} test={test[sel]['iou']:.6f} "
        f"winner_val={val[WINNER]['iou']:.6f} gen2_blend_val={blend_iou:.6f} "
        f"delta_winner={sel_iou - WINNER_VAL:+.6f} delta_gen2={sel_iou - blend_iou:+.6f} "
        f"v10={v10[sel]['iou']:.6f} v11={v11[sel]['iou']:.6f} "
        f"nested_k5_vs_winner={k5w:+.6f} nested_k2_vs_winner={k2w:+.6f} "
        f"nested_k5_vs_gen2={k5g:+.6f} nested_k2_vs_gen2={k2g:+.6f} "
        f"thr={val[sel]['threshold']:.6f} "
        f"protocol_ok={int(protocol_ok)} accept={'yes' if beats_w else 'no'} "
        f"beats_gen2={'yes' if beats_g else 'no'} real_train=0"
    )
    if beats_w:
        elog(
            f"BEATS_WINNER GEN2TCN {sel} val {sel_iou:.6f} test {test[sel]['iou']:.6f} "
            f"winner_val {val[WINNER]['iou']:.6f}"
        )
    log(decision)
    log(f"записан {OUT} {DOC}")


if __name__ == "__main__":
    main()
