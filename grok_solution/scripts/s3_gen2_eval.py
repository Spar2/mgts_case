"""Оценка GEN2. Порог только по фону v8 val. Test, v10, v11 и real его не двигают.

Кандидаты: unet_gen2, ens_blend = (unet + победитель) / 2,
ens_mean4 = (4 * победитель + unet) / 5.
Вложенные k=2 и k=5 — процедура scripts/s3_sel2_cv.py на тех же фолдах i % k.
gen2_val и real в выбор не входят.
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

from dts import data, gen2, metrics as M, models
from dts.features import make_channels
from dts.real import load_csv
from s3_gen2_train import BEST, predict_batched, freeze_bn

WINNER = "ens_u2ft_tcn_lgbm"
WINNER_VAL = 0.38671235821914546
WINNER_TEST = 0.4052762065230473
WINNER_V10 = 0.403898
WINNER_V11 = 0.403900
BEAT_MARGIN = 1e-4
HIT = 0.30
OUT = Path("artifacts/eval/s3_gen2_v8.json")
INIT = Path("artifacts/stage1/unet_v810ft.pt")
NAMES = ("unet_gen2", "ens_blend", "ens_mean4", WINNER)


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S", time.gmtime()) + " " + msg, flush=True)


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


def blend_of(name, s, ws):
    if name == "unet_gen2":
        return s
    if name == "ens_blend":
        return (0.5 * (s + ws)).astype(np.float32)
    if name == "ens_mean4":
        return ((4.0 * ws + s) / 5.0).astype(np.float32)
    return ws


def bg_mask(ev, nu, valid):
    return np.asarray(valid, bool) & (np.asarray(ev) == 0) & (np.asarray(nu) == 0)


def fit_thresholds(tmp, files, split, ver) -> tuple[dict, int]:
    n_bg = 0
    masks = []
    for p in files:
        with np.load(p) as z:
            mask = bg_mask(z["ev"], z["nu"], z["valid"])
        masks.append(mask)
        n_bg += int(mask.sum())
    ens = Path("artifacts/pred") / ver / WINNER / split
    thr = {}
    for n in NAMES:
        parts = []
        for p, mask in zip(files, masks):
            s = np.load(tmp / f"{p.stem}_s.npy", mmap_mode="r")
            with np.load(ens / p.name) as pred:
                ws = np.asarray(pred["s"], np.float32)
            parts.append(np.asarray(blend_of(n, np.asarray(s), ws)[mask], np.float32))
            del s, ws
        bg = np.concatenate(parts)
        del parts
        thr[n] = float(M.far_threshold(bg, 1e-3))
        log(f"порог {n} {thr[n]:.6f} n_bg {n_bg}")
        del bg
    return thr, n_bg


def score_split(model, ver, split, thr):
    files = data.list_cache(f"artifacts/cache/{ver}", split)
    tmp = Path(f"/tmp/s3_gen2_{ver}_{split}_{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    ens = Path("artifacts/pred") / ver / WINNER / split
    t0 = time.time()
    pair = None
    try:
        for i, p in enumerate(files, 1):
            with np.load(p) as z:
                X = np.asarray(z["X"], np.float32)
            prob = predict_batched(model, X)
            np.save(tmp / f"{p.stem}_s.npy", (1.0 - prob[0]).astype(np.float32))
            np.save(tmp / f"{p.stem}_h.npy", (prob[1] - prob[2]).astype(np.float16))
            del X, prob
            if i == 1 or i % 15 == 0 or i == len(files):
                log(f"pred {ver} {split} {i}/{len(files)} {time.time() - t0:.0f}s")
        if thr is None:
            thr, _n = fit_thresholds(tmp, files, split, ver)
        if ver == "v8" and split == "val":
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
            s = np.load(tmp / f"{p.stem}_s.npy")
            h = np.load(tmp / f"{p.stem}_h.npy").astype(np.float32)
            with np.load(ens / p.name) as pred:
                ws = np.asarray(pred["s"], np.float32)
                wh = np.asarray(pred["h"], np.float32)
            packed = {
                "unet_gen2": (s, h),
                "ens_blend": (blend_of("ens_blend", s, ws), (h + wh) / 2),
                "ens_mean4": (blend_of("ens_mean4", s, ws), (h + 4 * wh) / 5),
                WINNER: (ws, wh),
            }
            truth = ev > 0
            hot_t = (ev >= 1) & (ev <= 4)
            cold_t = ev >= 5
            bg = bg_mask(ev, nu, valid)
            for n, (ps, ph) in packed.items():
                pred_m = (ps >= thr[n]) & valid
                tot[n] += np.asarray(M.confusion(pred_m, truth, valid), np.int64)
                hot[n] += np.asarray(M.confusion(pred_m & (ph > 0), hot_t, valid), np.int64)
                cold[n] += np.asarray(M.confusion(pred_m & (ph <= 0), cold_t, valid), np.int64)
                far_n[n] += int((pred_m & bg).sum())
                far_d[n] += int(bg.sum())
            del s, h, ws, wh
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
    buckets = {n: [] for n in ("ens_blend", "ens_mean4", WINNER)}
    for p in files:
        with np.load(p) as z:
            ev_map = np.asarray(z["ev"])
            valid = np.asarray(z["valid"])
            nu = np.asarray(z["nu"])
            x = np.asarray(z["x"], np.float64)
            t = np.asarray(z["t"], np.float64)
        ev_idx, nu_idx, bg_idx, n_bg = ev._indices(ev_map, valid, nu, p.stem)
        s = np.asarray(np.load(tmp / f"{p.stem}_s.npy"), np.float32)
        with np.load(Path("artifacts/pred/v8") / WINNER / "val" / p.name) as pred:
            ws = np.asarray(pred["s"], np.float32)
        cooling, _rows = ev._events_for(p.stem, ev_map, x, t, labels, seqs, ev_idx)

        def frame(arr, stem=p.stem, cooling=cooling, n_bg=n_bg, ev_idx=ev_idx, nu_idx=nu_idx, bg_idx=bg_idx):
            e, n, b = s2._cols(arr, ev_idx, nu_idx, bg_idx)
            return dict(ev_m=e[None], nu_m=n[None], bg_m=b[None], n_bg=n_bg, cooling=cooling, stem=stem)

        for n in buckets:
            buckets[n].append(frame(blend_of(n, s, ws) if n != WINNER else ws))
        del s, ws
    order = np.argsort([fr["stem"] for fr in buckets[WINNER]])
    return {n: [buckets[n][i] for i in order] for n in buckets}


def nested_ks(frames) -> dict:
    import s3_sel2_cv as s2
    import s3_sel_core as sel

    winner = frames[WINNER]
    out = {}
    for name in ("ens_blend", "ens_mean4"):
        out[name] = {}
        for k in (2, 5):
            folds = sel.sequence_folds(len(winner), k)
            parts_w, parts_b, rows = [], [], []
            for r, idx in enumerate(folds):
                hw = s2.eval_one(s2.take(winner, idx))
                hb = s2.eval_one(s2.take(frames[name], idx))
                parts_w.append(hw)
                parts_b.append(hb)
                rows.append(dict(
                    fold=r, n_seq=len(idx),
                    winner=float(hw["iou"]), selected=float(hb["iou"]),
                    delta=float(hb["iou"] - hw["iou"]),
                ))
                log(f"nested {name} k={k} fold {r} winner {hw['iou']:.4f} sel {hb['iou']:.4f} delta {hb['iou'] - hw['iou']:+.4f}")
            iw = float(sel.micro_iou(parts_w))
            ib = float(sel.micro_iou(parts_b))
            out[name][str(k)] = dict(winner_iou=iw, selected_iou=ib, delta=float(ib - iw), folds=rows)
            log(f"nested {name} k={k} micro {ib:.4f} winner {iw:.4f} delta {ib - iw:+.4f}")
    out["procedure"] = "scripts/s3_sel2_cv.py sequence_folds i%k, holdout FAR 1e-3, micro IoU"
    out["stems"] = [fr["stem"] for fr in winner]
    return out


def stored_threshold(name: str) -> float:
    files = data.list_cache("artifacts/cache/v8", "val")
    parts = []
    root = Path("artifacts/pred/v8") / name / "val"
    for p in files:
        with np.load(p) as z:
            mask = bg_mask(z["ev"], z["nu"], z["valid"])
        with np.load(root / p.name) as pred:
            s = np.asarray(pred["s"], np.float32)
        parts.append(s[mask])
        del s
    thr = float(M.far_threshold(np.concatenate(parts), 1e-3))
    log(f"порог сохранённый {name} {thr:.6f}")
    return thr


def _hit_stats(score, mask, thr):
    if not mask.any():
        return None
    pix = float((score[mask] >= thr).mean())
    return dict(pixel_recall=pix, hit=bool(pix >= HIT), n_pix=int(mask.sum()))


def gen2_val(model, init, thr_new, thr_init) -> dict:
    rows = []
    for seed in range(16):
        scene = gen2.build_scene(seed)
        sm = scene["info"]["static_mask"]
        am = scene["info"]["artifact_mask"]
        rec = dict(seed=seed, static_amp_c=scene["info"]["static"]["amp_c"],
                   static_width_m=scene["info"]["static"]["width_m"],
                   artifact_mode=scene["info"]["artifact"]["mode"],
                   hvac_period_min=scene["info"]["hvac"]["period_min"])
        for key, net, thr in (("unet_gen2", model, thr_new), ("unet_v810ft", init, thr_init)):
            prob = predict_batched(net, scene["X"])
            s = 1.0 - prob[0]
            st = _hit_stats(s, sm, thr)
            art = _hit_stats(s, am, thr)
            rec[key] = dict(
                static_pixel_recall=None if st is None else st["pixel_recall"],
                static_hit=None if st is None else st["hit"],
                artifact_pixel_fp=None if art is None else art["pixel_recall"],
                artifact_hit=None if art is None else art["hit"],
            )
            del prob, s
        rows.append(rec)
        del scene

    def pack(key, field, hit_field):
        vals = [r[key][field] for r in rows if r[key][field] is not None]
        hits = [r[key][hit_field] for r in rows if r[key][hit_field] is not None]
        return dict(
            mean=float(np.mean(vals)) if vals else None,
            n=len(vals),
            hits=int(sum(bool(h) for h in hits)),
        )

    summary = {}
    for key, thr in (("unet_gen2", thr_new), ("unet_v810ft", thr_init)):
        summary[key] = dict(
            threshold=float(thr),
            static_event_recall=pack(key, "static_pixel_recall", "static_hit"),
            artifact_fp=pack(key, "artifact_pixel_fp", "artifact_hit"),
        )
        st = summary[key]["static_event_recall"]
        af = summary[key]["artifact_fp"]
        log(
            f"gen2_val {key} static hits {st['hits']}/{st['n']} pix {st['mean']} "
            f"artifact fp pix {af['mean']} blobs {af['hits']}/{af['n']}"
        )
    return dict(
        n_scenes=16, sigma_c=0.30, dt_s=120, hit_frac=HIT,
        summary=summary, scenes=rows,
        used_for_selection=False,
        note="Свой шум N(0, 0.85) и σ=0.30 °C. Карточки не читались. Порог заморожен по v8 val.",
    )


def real_desc(model, init, thr_new, thr_init) -> dict:
    files = []
    for i in range(1, 7):
        path = Path(f"/workspace/mgts/MGTS_data/{i}.csv")
        try:
            rec = load_csv(path)
        except Exception as exc:
            files.append(dict(file=path.name, error=str(exc)))
            continue
        T = rec["T"]
        valid = rec["valid"]
        n = int(T.shape[0])
        height = 160
        if n <= height:
            starts = [0]
        else:
            starts = [0, int(n // 2)]
        windows = []
        for t0 in starts:
            ch, _aux = make_channels(T[t0:t0 + height], valid[t0:t0 + height])
            m = ch[2] > 0.5
            n_valid = int(m.sum())
            row = dict(t0=int(t0), n_valid=n_valid)
            if n_valid == 0:
                windows.append(row)
                continue
            for key, net, thr in (("unet_gen2", model, thr_new), ("unet_v810ft", init, thr_init)):
                prob = predict_batched(net, ch)
                s = 1.0 - prob[0]
                row[key] = float(np.mean(s[m] >= thr))
                del prob, s
            windows.append(row)
            del ch
        files.append(dict(
            file=path.name, sha256=rec["sha256"], shape=[n, int(T.shape[1])], windows=windows,
        ))
        log(f"real {path.name} windows {len(windows)}")
        del rec, T, valid
    return dict(
        used_for_selection=False,
        cards_read=False,
        threshold_source="v8 val background, not moved by these files",
        unet_gen2_threshold=float(thr_new),
        unet_v810ft_threshold=float(thr_init),
        files=files,
        note="Доля валидных клеток выше замороженного порога на двух окнах. Это не IoU и не подгон карточек.",
    )


def main() -> None:
    if not BEST.exists():
        raise SystemExit(f"нет {BEST}")
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    t0 = time.time()
    model = models.build("unet", 16)
    model.load_state_dict(torch.load(BEST, map_location="cpu", weights_only=True))
    model.eval()
    freeze_bn(model)
    val, pair, thr = score_split(model, "v8", "val", None)
    if abs(val[WINNER]["iou"] - WINNER_VAL) > 0.005:
        dump(OUT, dict(protocol_ok=False, val=val, reason="winner val mismatch"))
        raise SystemExit(f"победитель val {val[WINNER]['iou']} != {WINNER_VAL}")
    nest = nested_ks(pair)
    test, _p, _t = score_split(model, "v8", "test", thr)
    if abs(test[WINNER]["iou"] - WINNER_TEST) > 0.005:
        dump(OUT, dict(protocol_ok=False, val=val, test=test, nested=nest, reason="winner test mismatch"))
        raise SystemExit(f"победитель test {test[WINNER]['iou']} != {WINNER_TEST}")
    v10, _p, _t = score_split(model, "v10", "test", thr)
    v11, _p, _t = score_split(model, "v11", "test", thr)
    init = models.build("unet", 16)
    init.load_state_dict(torch.load(INIT, map_location="cpu", weights_only=True))
    init.eval()
    thr_init = stored_threshold("unet_v810ft")
    gval = gen2_val(model, init, thr["unet_gen2"], thr_init)
    real = real_desc(model, init, thr["unet_gen2"], thr_init)
    sel_name = max(("ens_blend", "ens_mean4"), key=lambda n: val[n]["iou"])
    sel_iou = float(val[sel_name]["iou"])
    protocol_ok = True
    beats = bool(sel_iou > WINNER_VAL + BEAT_MARGIN)
    if beats:
        decision = (
            f"Принято по val. {sel_name} IoU {sel_iou:.4f} выше победителя "
            f"{WINNER_VAL:.4f} больше чем на 1e-4. Test в выбор не входил."
        )
    else:
        decision = (
            f"Отклонено. Лучший из ens_blend и ens_mean4 на полном v8 val — {sel_name} "
            f"{sel_iou:.4f}, победитель {val[WINNER]['iou']:.4f}. "
            f"Зазор к публикации {sel_iou - WINNER_VAL:+.6f}."
        )
    report = dict(
        name="GEN2",
        protocol=(
            "v8 val IoU at FAR 1e-3, threshold from the full v8 val background. "
            "ens_blend is the mean of unet_gen2 and ens_u2ft_tcn_lgbm. "
            "ens_mean4 is (4 * winner + unet_gen2) / 5. "
            "Nested k=2 and k=5 use scripts/s3_sel2_cv.py on the same sorted folds. "
            "gen2_val and real do not select. Synthetic v8 and v10 train only."
        ),
        decision=decision,
        accept=beats,
        beats_winner=beats,
        selected=sel_name,
        selected_val_iou=sel_iou,
        selected_test_iou=float(test[sel_name]["iou"]),
        protocol_ok=protocol_ok,
        published_winner=dict(val_iou=WINNER_VAL, test_iou=WINNER_TEST, v10_test=WINNER_V10, v11_test=WINNER_V11),
        winner_recomputed=dict(
            val=val[WINNER]["iou"], test=test[WINNER]["iou"],
            v10=v10[WINNER]["iou"], v11=v11[WINNER]["iou"],
            val_match=abs(val[WINNER]["iou"] - WINNER_VAL) < 1e-9,
            test_match=abs(test[WINNER]["iou"] - WINNER_TEST) < 1e-9,
        ),
        val=val, test=test, v10_test=v10, v11_test=v11,
        nested=nest, gen2_val=gval, real=real,
        real_used_for_training=False,
        real_used_for_threshold=False,
        real_noise_used=False,
        cards_used_for_training=False,
        seconds=round(time.time() - t0, 1),
    )
    dump(OUT, report)
    log(decision)
    log(f"записан {OUT}")


if __name__ == "__main__":
    main()
