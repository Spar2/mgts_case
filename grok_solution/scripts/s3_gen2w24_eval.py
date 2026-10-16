"""Оценка GEN2W24. Порог только по фону полного v8 val. Test, v10 и v11 его не двигают.

Смеси зафиксированы до прогона, веса не подбираются:

- unet_gen2w24 — дообученная ширина 24;
- ens_w24 — mean(gen2w24, winner);
- ens_trio — mean(gen2, gen2w24, winner);
- ens_gens — mean(gen2, gen2w24);
- ens_w24m4 — (4 * winner + gen2w24) / 5.

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
WINNER_V10 = 0.403898
WINNER_V11 = 0.403900
GEN2_BLEND_VAL = 0.4106385680480887
BEAT_MARGIN = 1e-4
BEST = Path("artifacts/eval/s3_gen2w24_unet.pt")
GEN2_W = Path("artifacts/eval/s3_gen2_unet.pt")
OUT = Path("artifacts/eval/s3_gen2w24_v8.json")
LOG = Path("/workspace/grok_logs/_events.log")

# Порядок фиксирован. ens_blend и winner — базы, не участники выбора.
CANDIDATES = ("unet_gen2w24", "ens_w24", "ens_trio", "ens_gens", "ens_w24m4")
BASELINES = ("ens_blend", WINNER)
NAMES = CANDIDATES + BASELINES


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


def load_unet(path: Path, width: int):
    model = models.build("unet", width)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state and not any(str(k).startswith("e1") for k in state):
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    freeze_bn(model)
    return model


def mix(name: str, sw: np.ndarray, sg: np.ndarray, ws: np.ndarray) -> np.ndarray:
    """sw — gen2w24, sg — gen2, ws — победитель. Веса не учатся."""
    if name == "unet_gen2w24":
        return sw
    if name == "ens_w24":
        return (0.5 * (sw + ws)).astype(np.float32)
    if name == "ens_trio":
        return ((sw + sg + ws) / 3.0).astype(np.float32)
    if name == "ens_gens":
        return (0.5 * (sw + sg)).astype(np.float32)
    if name == "ens_w24m4":
        return ((4.0 * ws + sw) / 5.0).astype(np.float32)
    if name == "ens_blend":
        return (0.5 * (sg + ws)).astype(np.float32)
    if name == WINNER:
        return ws
    raise KeyError(name)


def mix_h(name: str, hw, hg, wh) -> np.ndarray:
    return mix(name, hw, hg, wh)


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
            sw = np.load(tmp / f"{p.stem}_w_s.npy", mmap_mode="r")
            sg = np.load(tmp / f"{p.stem}_g_s.npy", mmap_mode="r")
            with np.load(ens / p.name) as pred:
                ws = np.asarray(pred["s"], np.float32)
            parts.append(np.asarray(mix(n, np.asarray(sw), np.asarray(sg), ws)[mask], np.float32))
            del sw, sg, ws
        thr[n] = float(M.far_threshold(np.concatenate(parts), 1e-3))
        log(f"порог {n} {thr[n]:.6f} n_bg {n_bg}")
        del parts
    return thr


def score_split(nets, ver, split, thr, make_pair: bool):
    files = data.list_cache(f"artifacts/cache/{ver}", split)
    tmp = Path(f"/tmp/s3_gen2w24_{ver}_{split}_{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    ens = Path("artifacts/pred") / ver / WINNER / split
    if not ens.is_dir():
        raise SystemExit(f"нет карт победителя {ens}")
    w24, gnet = nets
    t0 = time.time()
    pair = None
    try:
        for i, p in enumerate(files, 1):
            with np.load(p) as z:
                X = np.asarray(z["X"], np.float32)
            for key, net in (("w", w24), ("g", gnet)):
                prob = predict_batched(net, X)
                np.save(tmp / f"{p.stem}_{key}_s.npy", (1.0 - prob[0]).astype(np.float32))
                np.save(tmp / f"{p.stem}_{key}_h.npy", (prob[1] - prob[2]).astype(np.float16))
                del prob
            del X
            if i == 1 or i % 10 == 0 or i == len(files):
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
            sw = np.load(tmp / f"{p.stem}_w_s.npy")
            sg = np.load(tmp / f"{p.stem}_g_s.npy")
            hw = np.load(tmp / f"{p.stem}_w_h.npy").astype(np.float32)
            hg = np.load(tmp / f"{p.stem}_g_h.npy").astype(np.float32)
            with np.load(ens / p.name) as pred:
                ws = np.asarray(pred["s"], np.float32)
                wh = np.asarray(pred["h"], np.float32)
            truth = ev > 0
            hot_t = (ev >= 1) & (ev <= 4)
            cold_t = ev >= 5
            bg = bg_mask(ev, nu, valid)
            for n in NAMES:
                ps = mix(n, sw, sg, ws)
                ph = mix_h(n, hw, hg, wh)
                pred_m = (ps >= thr[n]) & valid
                tot[n] += np.asarray(M.confusion(pred_m, truth, valid), np.int64)
                hot[n] += np.asarray(M.confusion(pred_m & (ph > 0), hot_t, valid), np.int64)
                cold[n] += np.asarray(M.confusion(pred_m & (ph <= 0), cold_t, valid), np.int64)
                far_n[n] += int((pred_m & bg).sum())
                far_d[n] += int(bg.sum())
                del ps, ph
            del sw, sg, hw, hg, ws, wh
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
        sw = np.asarray(np.load(tmp / f"{p.stem}_w_s.npy"), np.float32)
        sg = np.asarray(np.load(tmp / f"{p.stem}_g_s.npy"), np.float32)
        with np.load(ens / p.name) as pred:
            ws = np.asarray(pred["s"], np.float32)
        cooling, _rows = ev._events_for(p.stem, ev_map, x, t, labels, seqs, ev_idx)

        def frame(arr, stem=p.stem, cooling=cooling, n_bg=n_bg, ev_idx=ev_idx, nu_idx=nu_idx, bg_idx=bg_idx):
            e, n, b = s2._cols(arr, ev_idx, nu_idx, bg_idx)
            return dict(ev_m=e[None], nu_m=n[None], bg_m=b[None], n_bg=n_bg, cooling=cooling, stem=stem)

        for n in NAMES:
            buckets[n].append(frame(mix(n, sw, sg, ws)))
        del sw, sg, ws
    order = np.argsort([fr["stem"] for fr in buckets[WINNER]])
    return {n: [buckets[n][i] for i in order] for n in buckets}


def nested_ks(frames) -> dict:
    import s3_sel2_cv as s2
    import s3_sel_core as sel

    out = {"procedure": "scripts/s3_sel2_cv.py sequence_folds i%k, holdout FAR 1e-3, micro IoU",
           "stems": [fr["stem"] for fr in frames[WINNER]]}
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
                    log(f"nested {name} vs {base} k={k} fold {r} base {hb['iou']:.4f} sel {hs['iou']:.4f} d {hs['iou'] - hb['iou']:+.4f}")
                ib = float(sel.micro_iou(parts_b))
                iss = float(sel.micro_iou(parts_s))
                out[name][base][str(k)] = dict(
                    baseline_iou=ib, selected_iou=iss, delta=float(iss - ib), folds=rows,
                )
                log(f"nested {name} vs {base} k={k} micro {iss:.4f} base {ib:.4f} delta {iss - ib:+.4f}")
    return out


def main() -> None:
    if not BEST.is_file():
        raise SystemExit(f"нет {BEST}")
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    t0 = time.time()
    nets = (load_unet(BEST, 24), load_unet(GEN2_W, 16))
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
    beats_w = bool(sel_iou > WINNER_VAL + BEAT_MARGIN and abs(val[WINNER]["iou"] - WINNER_VAL) < 1e-6)
    beats_g = bool(sel_iou > blend_iou + BEAT_MARGIN)
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
        name="GEN2W24",
        protocol=(
            "Checkpoint by mean(gen2, gen2w24) IoU at FAR 1e-3 on all v10 val, not v8 val. "
            "Official selection is full v8 val IoU at FAR 1e-3 among fixed mixtures: "
            "unet_gen2w24, mean(gen2w24, winner), mean(gen2, gen2w24, winner), "
            "mean(gen2, gen2w24), (4*winner+gen2w24)/5. "
            "Nested k=2 and k=5 follow scripts/s3_sel2_cv.py versus the winner and versus "
            "recomputed GEN2 ens_blend. Nested does not select. Synthetic train only."
        ),
        mixtures=dict(
            unet_gen2w24="gen2w24",
            ens_w24="mean(gen2w24, winner)",
            ens_trio="mean(gen2, gen2w24, winner)",
            ens_gens="mean(gen2, gen2w24)",
            ens_w24m4="(4*winner + gen2w24)/5",
            ens_blend="mean(gen2, winner)",
            winner=WINNER,
        ),
        decision=decision,
        accept=beats_w,
        beats_winner=beats_w,
        beats_gen2_blend=beats_g,
        selected=sel,
        selected_val_iou=sel_iou,
        selected_test_iou=float(test[sel]["iou"]),
        protocol_ok=True,
        published_winner=dict(val_iou=WINNER_VAL, test_iou=WINNER_TEST, v10_test=WINNER_V10, v11_test=WINNER_V11),
        published_gen2_ens_blend_val=GEN2_BLEND_VAL,
        winner_recomputed=dict(
            val=val[WINNER]["iou"], test=test[WINNER]["iou"],
            v10=v10[WINNER]["iou"], v11=v11[WINNER]["iou"],
            val_match=abs(val[WINNER]["iou"] - WINNER_VAL) < 1e-9,
            test_match=abs(test[WINNER]["iou"] - WINNER_TEST) < 1e-9,
        ),
        gen2_blend_recomputed=dict(val=blend_iou, test=test["ens_blend"]["iou"],
                                   v10=v10["ens_blend"]["iou"], v11=v11["ens_blend"]["iou"]),
        val=val, test=test, v10_test=v10, v11_test=v11, nested=nest,
        real_used_for_training=False,
        real_used_for_threshold=False,
        real_noise_used=False,
        seconds=round(time.time() - t0, 1),
    )
    dump(OUT, report)
    elog(
        f"GEN2W24_DONE selected={sel} val={sel_iou:.6f} test={test[sel]['iou']:.6f} "
        f"winner_val={val[WINNER]['iou']:.6f} gen2_blend_val={blend_iou:.6f} "
        f"delta_winner={sel_iou - WINNER_VAL:+.6f} delta_gen2={sel_iou - blend_iou:+.6f} "
        f"v10={v10[sel]['iou']:.6f} v11={v11[sel]['iou']:.6f} "
        f"nested_k5_vs_winner={k5w:+.6f} nested_k2_vs_winner={k2w:+.6f} "
        f"nested_k5_vs_gen2={k5g:+.6f} nested_k2_vs_gen2={k2g:+.6f} "
        f"accept={'yes' if beats_w else 'no'} beats_gen2={'yes' if beats_g else 'no'} real_train=0"
    )
    log(decision)
    log(f"записан {OUT}")


if __name__ == "__main__":
    main()
