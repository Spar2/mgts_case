"""Оценка ученика дистилляции против ens_u2ft_tcn_lgbm.

Порог каждого метода — квантиль чистого фона v8 val при FAR 1e-3, затем заморожен
для v8 test, v10 test и v11 test. Вложенные фолды — sequence_folds / evaluate_frames /
micro_iou / nested_verdict из scripts/s3_sel_core.py (то же, что s3_sel2_cv.nested_fixed).
Прирост micro на k=5 должен быть ≥ 0.003, чтобы победить ансамбль. Реальные CSV не читаются.

  /workspace/repo/run/.venv/bin/python -u scripts/s3_distill_eval.py
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "NUMEXPR_MAX_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_k] = "2"

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import s3_sel_core as sel
from dts import data, distill, metrics as M, models

torch.set_num_threads(2)

WINNER = "ens_u2ft_tcn_lgbm"
PRED = Path("artifacts/ckpt/s3_distill/pred")
OUT = Path("artifacts/eval/s3_distill_v8.json")
CKPT = Path("artifacts/eval/s3_distill_unet.pt")
EVENTS = Path("/workspace/grok_logs/_events.log")


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S', time.gmtime())} {msg}", flush=True)


def event(msg: str) -> None:
    line = f"{time.strftime('%H:%M', time.gmtime())} {msg}"
    print(line, flush=True)
    with EVENTS.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


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


def dump_json(path: Path, obj) -> None:
    path = distill.assert_safe_output(path, ROOT)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(plain(obj), indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def load_student(path: Path, width: int):
    model = models.build("unet", width)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def save_map(path: Path, s, h) -> None:
    path = distill.assert_safe_output(path, ROOT)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, s=np.asarray(s, np.float32), h=np.asarray(h, np.float16))
    os.replace(tmp, path)


@torch.inference_mode()
def student_sh(model, cache_path: Path, ver: str, split: str):
    dst = PRED / ver / split / cache_path.name
    if dst.exists() and dst.stat().st_size > 0:
        with np.load(dst) as z:
            return np.asarray(z["s"], np.float32), np.asarray(z["h"], np.float32)
    X = np.asarray(np.load(cache_path)["X"], np.float32)
    pr = models.predict_full(model, X)
    s = (1.0 - pr[0]).astype(np.float32)
    h = (pr[1] - pr[2]).astype(np.float32)
    save_map(dst, s, h)
    return s, h


def winner_sh(ver: str, split: str, name: str):
    path = ROOT / "artifacts/pred" / ver / WINNER / split / name
    with np.load(path) as z:
        return np.asarray(z["s"], np.float32), np.asarray(z["h"], np.float32)


def groups_of(s, ev, nu, valid):
    s = np.asarray(s, np.float32)
    ev = np.asarray(ev)
    nu = np.asarray(nu)
    valid = np.asarray(valid, bool)
    m_ev = valid & (ev > 0)
    m_nu = valid & (ev == 0) & (nu > 0)
    m_bg = valid & (ev == 0) & (nu == 0)
    return dict(ev=s[m_ev], nu=s[m_nu], bg=s[m_bg])


def event_pack(s, ev, x, t, labels, stem):
    rows = []
    for r in M.label_boxes(labels, stem).itertuples():
        if r.kind != "event":
            continue
        px = M.event_pixels(r, x, t, ev)
        n = int(px.sum())
        if n == 0:
            continue
        sc = np.asarray(s[px], np.float32)
        rows.append(dict(amp=float(r.amp_obs_C), n=n, s=sc.astype(np.float16)))
    return rows


def summarize_events(rows, thr: float) -> dict:
    if not rows:
        return dict(n=0, recall=None, weak_n=0, weak_hit=0)
    hit = []
    weak_hit = 0
    weak_n = 0
    for rec in rows:
        ok = float((rec["s"].astype(np.float32) >= thr).mean()) >= 0.3
        hit.append(ok)
        if rec["amp"] <= 2.0:
            weak_n += 1
            weak_hit += int(ok)
    return dict(n=len(rows), recall=float(np.mean(hit)), weak_n=weak_n, weak_hit=weak_hit)


def polarity(s, h, ev, valid, thr: float) -> dict:
    pred = (np.asarray(s) >= thr) & np.asarray(valid, bool)
    valid = np.asarray(valid, bool)
    ev = np.asarray(ev)
    h = np.asarray(h)
    hot = M.confusion(pred & (h > 0), (ev >= 1) & (ev <= 4), valid)
    cold = M.confusion(pred & (h <= 0), ev >= 5, valid)
    return dict(hot=M.prf(*hot), cold=M.prf(*cold))


def sel_frame(fr):
    ev = np.asarray(fr["ev"], np.float32).ravel()
    nu = np.asarray(fr["nu"], np.float32).ravel()
    bg = np.asarray(fr["bg"], np.float32).ravel()
    return dict(
        ev_m=ev.reshape(1, -1),
        nu_m=nu.reshape(1, -1),
        bg_m=bg.reshape(1, -1),
        n_bg=int(fr["bg"].size),
        cooling=[],
    )


def nested_micro(frames, ks=(2, 5)) -> dict:
    made = [sel_frame(fr) for fr in frames]
    out = {}
    weight = np.array([1.0])
    for k in ks:
        folds = sel.sequence_folds(len(made), k)
        parts = []
        for hold_idx in folds:
            hold = [made[i] for i in hold_idx]
            parts.append(sel.evaluate_frames(hold, weight))
        micro = sel.micro_iou(parts)
        out[str(k)] = dict(micro_iou=float(micro), n_folds=len(folds))
    out["verdict"] = sel.nested_verdict({k: out[k]["micro_iou"] for k in out if k != "verdict"})
    return out


def pack_metrics(stats, events, pol) -> dict:
    out = dict(
        iou=float(stats["iou"]),
        precision=float(stats["precision"]),
        recall=float(stats["recall"]),
        f1=float(stats["f1"]),
        threshold=float(stats["threshold"]),
        far=float(stats["far"]),
        tp=int(stats["tp"]),
        fp=int(stats["fp"]),
        fn=int(stats["fn"]),
        n_bg=int(stats["n_bg"]),
    )
    if events is not None:
        out["events"] = events
    if pol is not None:
        out["hot_iou"] = float(pol["hot"]["iou"])
        out["cold_iou"] = float(pol["cold"]["iou"])
    return out


def stream_split(model, ver, split, thr: float, labels, with_events: bool):
    files = data.list_cache(f"artifacts/cache/{ver}", split)
    tp = fp = fn = far_n = far_d = 0
    hot = np.zeros(3)
    cold = np.zeros(3)
    ev_rows = []
    t0 = time.time()
    for i, path in enumerate(files, 1):
        with np.load(path) as z:
            ev = np.asarray(z["ev"])
            nu = np.asarray(z["nu"])
            valid = np.asarray(z["valid"])
            x = np.asarray(z["x"]) if with_events else None
            t = np.asarray(z["t"]) if with_events else None
        s, h = student_sh(model, path, ver, split)
        c = distill.pixel_counts(s, ev, nu, valid, thr)
        tp += c["tp"]
        fp += c["fp"]
        fn += c["fn"]
        far_n += c["far_n"]
        far_d += c["far_d"]
        pred = (s >= thr) & valid
        hot += M.confusion(pred & (h > 0), (ev >= 1) & (ev <= 4), valid)
        cold += M.confusion(pred & (h <= 0), ev >= 5, valid)
        if with_events:
            ev_rows.extend(event_pack(s, ev, x, t, labels, path.stem))
        if i == 1 or i % 10 == 0 or i == len(files):
            log(f"{ver} {split} {i}/{len(files)} {time.time() - t0:.0f}s")
        del s, h, ev, nu, valid
    stats = M.prf(tp, fp, fn)
    stats.update(threshold=float(thr), far=float(far_n / max(far_d, 1)), tp=tp, fp=fp, fn=fn, n_bg=int(far_d))
    events = summarize_events(ev_rows, thr) if with_events else None
    pol = dict(hot=M.prf(*hot), cold=M.prf(*cold))
    return pack_metrics(stats, events, pol)


def collect_val(model):
    labels = pd.read_csv(ROOT / "data/synth_meta/v8/labels.csv")
    files = data.list_cache("artifacts/cache/v8", "val")
    student, winner = [], []
    ev_s, ev_w = [], []
    t0 = time.time()
    for i, path in enumerate(files, 1):
        with np.load(path) as z:
            ev = np.asarray(z["ev"])
            nu = np.asarray(z["nu"])
            valid = np.asarray(z["valid"])
            x = np.asarray(z["x"])
            t = np.asarray(z["t"])
        s, h = student_sh(model, path, "v8", "val")
        ws, wh = winner_sh("v8", "val", path.name)
        if s.shape != ev.shape or ws.shape != ev.shape:
            raise SystemExit(f"форма {path.stem}: s {s.shape} winner {ws.shape} ev {ev.shape}")
        gs = groups_of(s, ev, nu, valid)
        gw = groups_of(ws, ev, nu, valid)
        gs["stem"] = path.stem
        gw["stem"] = path.stem
        student.append(gs)
        winner.append(gw)
        ev_s.extend(event_pack(s, ev, x, t, labels, path.stem))
        ev_w.extend(event_pack(ws, ev, x, t, labels, path.stem))
        del s, h, ws, wh
        if i == 1 or i % 5 == 0 or i == len(files):
            log(f"v8 val {i}/{len(files)} {time.time() - t0:.0f}s")
    student.sort(key=lambda fr: fr["stem"])
    winner.sort(key=lambda fr: fr["stem"])
    return student, winner, ev_s, ev_w, labels, time.time() - t0


def polarity_from_disk(ver, split, thr, who: str):
    files = data.list_cache(f"artifacts/cache/{ver}", split)
    hot = np.zeros(3)
    cold = np.zeros(3)
    for path in files:
        with np.load(path) as z:
            ev = np.asarray(z["ev"])
            valid = np.asarray(z["valid"])
        if who == "student":
            with np.load(PRED / ver / split / path.name) as q:
                s = np.asarray(q["s"], np.float32)
                h = np.asarray(q["h"], np.float32)
        else:
            s, h = winner_sh(ver, split, path.name)
        pred = (s >= thr) & valid
        hot += M.confusion(pred & (h > 0), (ev >= 1) & (ev <= 4), valid)
        cold += M.confusion(pred & (h <= 0), ev >= 5, valid)
    return dict(hot=M.prf(*hot), cold=M.prf(*cold))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(CKPT))
    ap.add_argument("--width", type=int, default=distill.STUDENT_WIDTH)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists() and not args.force:
        log(f"уже есть {out} (без --force не пересчитываю)")
        return
    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        raise SystemExit(f"нет весов ученика {ckpt}")
    t_all = time.time()
    model = load_student(ckpt, args.width)
    n_param = sum(p.numel() for p in model.parameters())
    log(f"student width={args.width} params={n_param}")

    student_fr, winner_fr, ev_s, ev_w, _labels, val_sec = collect_val(model)
    st_val = distill.iou_from_groups(student_fr)
    wn_val = distill.iou_from_groups(winner_fr)
    ev_s_sum = summarize_events(ev_s, st_val["threshold"])
    ev_w_sum = summarize_events(ev_w, wn_val["threshold"])
    log(f"val student IoU {st_val['iou']:.6f} thr {st_val['threshold']:.6f}")
    log(f"val winner  IoU {wn_val['iou']:.6f} thr {wn_val['threshold']:.6f}")

    nested_s = nested_micro(student_fr)
    nested_w = nested_micro(winner_fr)
    d5 = float(nested_s["5"]["micro_iou"] - nested_w["5"]["micro_iou"])
    d2 = float(nested_s["2"]["micro_iou"] - nested_w["2"]["micro_iou"])
    log(f"nested5 student {nested_s['5']['micro_iou']:.4f} winner {nested_w['5']['micro_iou']:.4f} delta {d5:.4f}")
    close = float(st_val["iou"]) >= distill.WINNER_VAL - distill.CLOSE_GAP
    log(f"close={int(close)} (gap {distill.CLOSE_GAP})")
    del student_fr, winner_fr, ev_s, ev_w

    pol_s = polarity_from_disk("v8", "val", st_val["threshold"], "student")
    pol_w = polarity_from_disk("v8", "val", wn_val["threshold"], "winner")

    labels_v8 = pd.read_csv(ROOT / "data/synth_meta/v8/labels.csv")
    st_test = stream_split(model, "v8", "test", st_val["threshold"], labels_v8, True)
    # winner test at its own val threshold, from saved maps
    wn_test = stream_winner("v8", "test", wn_val["threshold"], labels_v8)
    log(f"v8 test student {st_test['iou']:.4f} winner {wn_test['iou']:.4f}")

    st_v10 = stream_split(model, "v10", "test", st_val["threshold"], None, False)
    wn_v10 = stream_winner("v10", "test", wn_val["threshold"], None)
    st_v11 = stream_split(model, "v11", "test", st_val["threshold"], None, False)
    wn_v11 = stream_winner("v11", "test", wn_val["threshold"], None)
    log(f"v10 student {st_v10['iou']:.4f} winner {wn_v10['iou']:.4f}")
    log(f"v11 student {st_v11['iou']:.4f} winner {wn_v11['iou']:.4f}")

    decision = distill.promotion_decision(dict(
        student_val=st_val["iou"],
        winner_val_recomputed=wn_val["iou"],
        winner_test_recomputed=wn_test["iou"],
        winner_v10_recomputed=wn_v10["iou"],
        winner_v11_recomputed=wn_v11["iou"],
        nested_delta5=d5,
        v10=st_v10["iou"],
        v11=st_v11["iou"],
    ))
    # v10/v11 «не хуже» считается против пересчитанного победителя на том же коде,
    # а ворота promotion_decision — против опубликованных констант. Если пересчёт
    # победителя чуть ниже константы, ученик может быть не хуже пересчёта и всё же
    # не пройти константу. Оба числа остаются в JSON.
    decision["v10_vs_recomputed"] = float(st_v10["iou"] - wn_v10["iou"])
    decision["v11_vs_recomputed"] = float(st_v11["iou"] - wn_v11["iou"])
    decision["nested_bar_accept"] = bool(nested_s["verdict"]["accept"])
    decision["nested_delta2"] = d2

    report = dict(
        method="s3_distill",
        student="DTSUNet",
        width=args.width,
        n_param=n_param,
        ckpt=str(ckpt),
        teacher=WINNER,
        real_train=False,
        real_eval=False,
        protocol=dict(
            far=1e-3,
            threshold="v8 val background valid & ev==0 & nu==0",
            published_val=distill.WINNER_VAL,
            published_test=distill.WINNER_TEST,
            published_thr=distill.WINNER_THR,
            published_v10=distill.WINNER_V10,
            published_v11=distill.WINNER_V11,
            nested="s3_sel_core.sequence_folds + evaluate_frames + micro_iou, gain>=0.003 on k=5",
            close_gap=distill.CLOSE_GAP,
        ),
        val=dict(
            student=pack_metrics(st_val, ev_s_sum, pol_s),
            winner=pack_metrics(wn_val, ev_w_sum, pol_w),
        ),
        test=dict(student=st_test, winner=wn_test),
        v10_test=dict(student=st_v10, winner=wn_v10),
        v11_test=dict(student=st_v11, winner=wn_v11),
        nested=dict(student=nested_s, winner=nested_w, delta5=d5, delta2=d2),
        decision=decision,
        val_collect_sec=val_sec,
        elapsed_sec=time.time() - t_all,
    )
    dump_json(out, report)
    beats = bool(decision["beats"])
    tag = "DISTILL_DONE" if beats else "DISTILL_REJECT"
    line = (
        f"{tag} student_val={st_val['iou']:.6f} winner_val={wn_val['iou']:.6f} "
        f"dval={st_val['iou'] - wn_val['iou']:+.6f} test={st_test['iou']:.4f} "
        f"winner_test={wn_test['iou']:.4f} v10={st_v10['iou']:.4f} v10_winner={wn_v10['iou']:.4f} "
        f"v11={st_v11['iou']:.4f} v11_winner={wn_v11['iou']:.4f} "
        f"nested5_delta={d5:+.4f} nested2_delta={d2:+.4f} close={int(decision['close'])} "
        f"beats={int(beats)} reason={decision['reason']} thr={st_val['threshold']:.4f} "
        f"weak_val={ev_s_sum['weak_hit']}/{ev_s_sum['weak_n']} sec={time.time() - t_all:.0f} real_train=0"
    )
    event(line)
    if beats:
        event(
            f"BEATS_WINNER DISTILL val={st_val['iou']:.6f} dval={st_val['iou'] - distill.WINNER_VAL:+.6f} "
            f"nested5_delta={d5:+.4f} v10={st_v10['iou']:.4f} v11={st_v11['iou']:.4f}"
        )
    log(line)


def stream_winner(ver, split, thr, labels):
    files = data.list_cache(f"artifacts/cache/{ver}", split)
    tp = fp = fn = far_n = far_d = 0
    hot = np.zeros(3)
    cold = np.zeros(3)
    ev_rows = []
    t0 = time.time()
    for i, path in enumerate(files, 1):
        with np.load(path) as z:
            ev = np.asarray(z["ev"])
            nu = np.asarray(z["nu"])
            valid = np.asarray(z["valid"])
            x = np.asarray(z["x"]) if labels is not None else None
            t = np.asarray(z["t"]) if labels is not None else None
        s, h = winner_sh(ver, split, path.name)
        c = distill.pixel_counts(s, ev, nu, valid, thr)
        tp += c["tp"]
        fp += c["fp"]
        fn += c["fn"]
        far_n += c["far_n"]
        far_d += c["far_d"]
        pred = (s >= thr) & valid
        hot += M.confusion(pred & (h > 0), (ev >= 1) & (ev <= 4), valid)
        cold += M.confusion(pred & (h <= 0), ev >= 5, valid)
        if labels is not None:
            ev_rows.extend(event_pack(s, ev, x, t, labels, path.stem))
        if i == 1 or i % 15 == 0 or i == len(files):
            log(f"winner {ver} {split} {i}/{len(files)} {time.time() - t0:.0f}s")
        del s, h
    stats = M.prf(tp, fp, fn)
    stats.update(threshold=float(thr), far=float(far_n / max(far_d, 1)), tp=tp, fp=fp, fn=fn, n_bg=int(far_d))
    events = summarize_events(ev_rows, thr) if labels is not None else None
    pol = dict(hot=M.prf(*hot), cold=M.prf(*cold))
    return pack_metrics(stats, events, pol)


if __name__ == "__main__":
    main()
