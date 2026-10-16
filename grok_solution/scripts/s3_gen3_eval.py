"""Оценка GEN3 против выпущенного ens_gen2. v8 val в прокси обучения не входил.

Порог FAR 1e-3 только по 41 файлам v8 val (без seq0006, seq0021, seq0047, seq0048).
На v8 test, v10 test и v11 test порог заморожен.

  gen3    — U-Net одна
  mean_w  — 0.5 * (gen3 + победитель)
  mean_g  — 0.5 * (gen3 + ens_gen2), ens_gen2 = 0.5 * (GEN2 + победитель)
  ens_gen2 — выпуск, порог тоже снят с этих 41

Бутстреп 1000 — дельта против ens_gen2. Вложенные k=2 и k=5 — против ens_gen2.
Скрипт не пишет строку GEN3_DONE и не трогает отчёт и models_release.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_k] = "2"

import gc
import json
import shutil
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
from s3_gen2_train import predict_batched

WINNER = "ens_u2ft_tcn_lgbm"
WINNER_VAL = 0.38671235821914546
WINNER_THR = 0.8608087301264272
WINNER_41 = 0.3784001812423094
ENS_VAL = 0.4106385680480887
ENS_THR = 0.8579896092424856
HOLD_DROP = ("seq0006", "seq0021", "seq0047", "seq0048")
BOOT_N = 1000
BOOT_SEED = 20261019
BAR = 0.003
SEED1 = Path("artifacts/eval/s3_gen2_unet.pt")
GEN3 = Path("artifacts/eval/s3_gen3_unet.pt")
OUT = Path("artifacts/eval/s3_gen3.json")
CANDS = ("gen3", "mean_w", "mean_g")
NAMES = CANDS + ("ens_gen2", "winner")


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


def dump(obj) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(plain(obj), indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, OUT)


def iou_of(tp, fp, fn):
    return float(tp) / max(float(tp) + float(fp) + float(fn), 1.0)


def pack(rows):
    tp = int(sum(r[0] for r in rows))
    fp = int(sum(r[1] for r in rows))
    fn = int(sum(r[2] for r in rows))
    pr = M.prf(tp, fp, fn)
    return dict(
        tp=tp, fp=fp, fn=fn, iou=float(pr["iou"]),
        precision=float(pr["precision"]), recall=float(pr["recall"]),
        n_seq=len(rows),
    )


def blend_of(name, g3, s1, ws):
    g3 = np.asarray(g3, np.float32)
    s1 = np.asarray(s1, np.float32)
    ws = np.asarray(ws, np.float32)
    ens = (0.5 * (s1 + ws)).astype(np.float32)
    if name == "gen3":
        return g3
    if name == "mean_w":
        return (0.5 * (g3 + ws)).astype(np.float32)
    if name == "mean_g":
        return (0.5 * (g3 + ens)).astype(np.float32)
    if name == "ens_gen2":
        return ens
    if name == "winner":
        return ws
    raise KeyError(name)


def bootstrap(rows_a, rows_b, n=BOOT_N, seed=BOOT_SEED):
    a = np.asarray(rows_a, np.float64)
    b = np.asarray(rows_b, np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    sa = a[idx].sum(axis=1)
    sb = b[idx].sum(axis=1)
    da = sa[:, 0] / np.maximum(sa.sum(axis=1), 1.0)
    db = sb[:, 0] / np.maximum(sb.sum(axis=1), 1.0)
    delta = da - db
    lo, mid, hi = np.percentile(delta, [2.5, 50, 97.5])
    point = pack(rows_a)["iou"] - pack(rows_b)["iou"]
    return dict(
        n=int(n), seed=int(seed), unit="sequence",
        point=float(point), mean=float(delta.mean()),
        p2_5=float(lo), p50=float(mid), p97_5=float(hi),
        frac_positive=float(np.mean(delta > 0)),
        covers_zero=bool(lo <= 0.0 <= hi),
    )


def load_net(path):
    model = models.build("unet", 16)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def predict_split(m_gen3, m_gen2, ver, split, tmp: Path):
    files = data.list_cache(f"artifacts/cache/{ver}", split)
    ens = Path("artifacts/pred") / ver / WINNER / split
    t0 = time.time()
    for i, p in enumerate(files, 1):
        with np.load(p) as z:
            X = np.asarray(z["X"], np.float32)
            shape = z["ev"].shape
        g3 = (1.0 - predict_batched(m_gen3, X)[0]).astype(np.float32)
        s1 = (1.0 - predict_batched(m_gen2, X)[0]).astype(np.float32)
        del X
        with np.load(ens / p.name) as pred:
            if pred["s"].shape != shape:
                raise SystemExit(f"winner {ver} {split} {p.stem} {pred['s'].shape} != {shape}")
        np.save(tmp / f"{ver}_{split}_{p.stem}_g3.npy", g3)
        np.save(tmp / f"{ver}_{split}_{p.stem}_s1.npy", s1)
        del g3, s1
        if i == 1 or i % 15 == 0 or i == len(files):
            log(f"pred {ver} {split} {i}/{len(files)} {time.time() - t0:.0f}s")
    return files


def load_trio(tmp, ver, split, path):
    g3 = np.load(tmp / f"{ver}_{split}_{path.stem}_g3.npy")
    s1 = np.load(tmp / f"{ver}_{split}_{path.stem}_s1.npy")
    with np.load(path) as z:
        ev = np.asarray(z["ev"])
        valid = np.asarray(z["valid"], bool)
        nu = np.asarray(z["nu"])
    with np.load(Path("artifacts/pred") / ver / WINNER / split / path.name) as pred:
        ws = np.asarray(pred["s"], np.float32)
    return g3, s1, ws, ev, valid, nu


def clear_split(tmp, ver, split, files):
    for p in files:
        for tag in ("g3", "s1"):
            f = tmp / f"{ver}_{split}_{p.stem}_{tag}.npy"
            if f.exists():
                f.unlink()
    gc.collect()


def winner_full_val():
    files = data.list_cache("artifacts/cache/v8", "val")
    rows = []
    for p in files:
        with np.load(p) as z:
            ev = np.asarray(z["ev"])
            valid = np.asarray(z["valid"], bool)
        with np.load(Path("artifacts/pred/v8") / WINNER / "val" / p.name) as pred:
            ws = np.asarray(pred["s"], np.float32)
        rows.append(M.confusion((ws >= WINNER_THR) & valid, ev > 0, valid))
    iou = pack(rows)["iou"]
    if abs(iou - WINNER_VAL) > 0.005:
        raise SystemExit(f"победитель на 45 файлах {iou} != {WINNER_VAL}")
    log(f"winner full val {iou:.6f} match")
    return iou, files


def fit_thr(files, tmp, ver, split):
    thr = {}
    for name in NAMES:
        parts = []
        for p in files:
            g3, s1, ws, ev, valid, nu = load_trio(tmp, ver, split, p)
            score = blend_of(name, g3, s1, ws)
            bg = valid & (ev == 0) & (nu == 0)
            parts.append(np.asarray(score[bg], np.float32))
            del g3, s1, ws, score
        bg = np.concatenate(parts)
        thr[name] = float(M.far_threshold(bg, 1e-3))
        log(f"порог {name} {thr[name]:.6f} n_bg {bg.size}")
        del parts, bg
    return thr


def score_files(files, tmp, ver, split, thr):
    rows = {n: [] for n in NAMES}
    for p in files:
        g3, s1, ws, ev, valid, _nu = load_trio(tmp, ver, split, p)
        truth = ev > 0
        for name in NAMES:
            score = blend_of(name, g3, s1, ws)
            rows[name].append(M.confusion((score >= thr[name]) & valid, truth, valid))
            del score
        del g3, s1, ws
    summary = {n: pack(rows[n]) for n in NAMES}
    for n in NAMES:
        summary[n]["threshold"] = float(thr[n])
    return summary, rows


def score_at(files, tmp, ver, split, name, thr):
    rows = []
    for p in files:
        g3, s1, ws, ev, valid, _nu = load_trio(tmp, ver, split, p)
        score = blend_of(name, g3, s1, ws)
        rows.append(M.confusion((score >= thr) & valid, ev > 0, valid))
        del g3, s1, ws, score
    return pack(rows)


def nested_files(files, tmp, ver, split):
    import s3_sel2_cv as sel2
    import s3_sel_core as sel
    import s3_sel_eval as ev
    import pandas as pd

    labels = pd.read_csv(f"data/synth_meta/{ver}/labels.csv")
    seqs = pd.read_csv(f"data/synth_meta/{ver}/sequences.csv").set_index("seq_id")
    buckets = {n: [] for n in CANDS + ("ens_gen2",)}
    for p in files:
        g3, s1, ws, ev_map, valid, nu = load_trio(tmp, ver, split, p)
        with np.load(p) as z:
            x = np.asarray(z["x"], np.float64)
            t = np.asarray(z["t"], np.float64)
        ev_idx, nu_idx, bg_idx, n_bg = ev._indices(ev_map, valid, nu, p.stem)
        cooling, _rows = ev._events_for(p.stem, ev_map, x, t, labels, seqs, ev_idx)

        def frame(arr, stem=p.stem, cooling=cooling, n_bg=n_bg, ev_idx=ev_idx, nu_idx=nu_idx, bg_idx=bg_idx):
            e, n, b = sel2._cols(arr, ev_idx, nu_idx, bg_idx)
            return dict(ev_m=e[None], nu_m=n[None], bg_m=b[None], n_bg=n_bg, cooling=cooling, stem=stem)

        for name in buckets:
            buckets[name].append(frame(blend_of(name, g3, s1, ws)))
        del g3, s1, ws
    order = np.argsort([fr["stem"] for fr in buckets["ens_gen2"]])
    frames = {n: [buckets[n][i] for i in order] for n in buckets}
    base = frames["ens_gen2"]
    out = {
        "stems": [fr["stem"] for fr in base],
        "procedure": "s3_sel2_cv sequence_folds i%k, holdout FAR 1e-3, micro, bg cap 200k, baseline ens_gen2",
    }
    for name in CANDS:
        out[name] = {}
        for k in (2, 5):
            folds = sel.sequence_folds(len(base), k)
            parts_b, parts_s, fold_rows = [], [], []
            for r, idx in enumerate(folds):
                hb = sel2.eval_one(sel2.take(base, idx))
                hs = sel2.eval_one(sel2.take(frames[name], idx))
                parts_b.append(hb)
                parts_s.append(hs)
                fold_rows.append(dict(
                    fold=r, n_seq=len(idx),
                    ens_gen2=float(hb["iou"]), selected=float(hs["iou"]),
                    delta=float(hs["iou"] - hb["iou"]),
                ))
            ib = float(sel.micro_iou(parts_b))
            iss = float(sel.micro_iou(parts_s))
            out[name][str(k)] = dict(
                ens_gen2_iou=ib, selected_iou=iss, delta=float(iss - ib), folds=fold_rows,
            )
            log(f"nested {ver} {split} {name} k={k} {iss:.4f} ens_gen2 {ib:.4f} delta {iss - ib:+.4f}")
    return out


def eval_split(m3, m1, ver, split, thr, tmp):
    files = predict_split(m3, m1, ver, split, tmp)
    summary, rows = score_files(files, tmp, ver, split, thr)
    nest = nested_files(files, tmp, ver, split)
    boot = {}
    for name in CANDS:
        boot[name] = bootstrap(rows[name], rows["ens_gen2"])
        b = boot[name]
        log(
            f"{ver} {split} {name} {summary[name]['iou']:.4f} ens {summary['ens_gen2']['iou']:.4f} "
            f"delta {b['point']:+.4f} CI [{b['p2_5']:+.4f}, {b['p97_5']:+.4f}] covers0 {b['covers_zero']}"
        )
    clear_split(tmp, ver, split, files)
    return summary, boot, nest


def verdict(name, hold_sum, boot, nest, splits):
    point = float(hold_sum[name]["iou"] - hold_sum["ens_gen2"]["iou"])
    ci = boot["val41"][name]
    ci_pos = (not bool(ci["covers_zero"])) and float(ci["point"]) > 0.0
    k5 = float(nest["val41"][name]["5"]["delta"])
    k2 = float(nest["val41"][name]["2"]["delta"])
    v10_ok = float(splits["v10"][name]["iou"]) + 1e-4 >= float(splits["v10"]["ens_gen2"]["iou"])
    v11_ok = float(splits["v11"][name]["iou"]) + 1e-4 >= float(splits["v11"]["ens_gen2"]["iou"])
    beats = bool(point >= BAR and (ci_pos or k5 >= BAR) and v10_ok and v11_ok)
    return dict(
        hold41_delta=point,
        ci_excludes_0=ci_pos,
        nested_k5=k5,
        nested_k2=k2,
        v10_not_worse=v10_ok,
        v11_not_worse=v11_ok,
        beats_gen2=beats,
    )


def main():
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if not GEN3.is_file():
        raise SystemExit(f"нет {GEN3}")
    if GEN3.resolve() == SEED1.resolve():
        raise SystemExit("оценка читает чужой чекпойнт как GEN3")
    t0 = time.time()
    win_full, val_files = winner_full_val()
    if len(val_files) != 45:
        raise SystemExit(f"v8 val {len(val_files)}")
    dropped = tuple(p.stem for p in val_files[:4])
    if dropped != HOLD_DROP:
        raise SystemExit(f"первые 4 v8 val {dropped}")
    hold = val_files[4:]
    if len(hold) != 41:
        raise SystemExit(f"holdout {len(hold)}")
    tmp = Path(f"/tmp/s3_gen3_eval_{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        m3 = load_net(GEN3)
        m1 = load_net(SEED1)
        log("сети загружены")
        predict_split(m3, m1, "v8", "val", tmp)
        ens_full = score_at(val_files, tmp, "v8", "val", "ens_gen2", ENS_THR)
        log(f"ens_gen2 full val frozen {ens_full['iou']:.6f}")
        if abs(ens_full["iou"] - ENS_VAL) > 0.005:
            raise SystemExit(f"ens_gen2 на 45 файлах {ens_full['iou']} != {ENS_VAL}")
        thr = fit_thr(hold, tmp, "v8", "val")
        hold_sum, hold_rows = score_files(hold, tmp, "v8", "val", thr)
        win_frozen = score_at(hold, tmp, "v8", "val", "winner", WINNER_THR)["iou"]
        log(f"winner 41 frozen {win_frozen:.6f}")
        if abs(win_frozen - WINNER_41) > 1e-4:
            raise SystemExit(f"победитель на 41 при опубликованном пороге {win_frozen}")
        nest41 = nested_files(hold, tmp, "v8", "val")
        boot = {"val41": {name: bootstrap(hold_rows[name], hold_rows["ens_gen2"]) for name in CANDS}}
        for name in CANDS:
            b = boot["val41"][name]
            log(
                f"val41 {name} {hold_sum[name]['iou']:.4f} ens {hold_sum['ens_gen2']['iou']:.4f} "
                f"delta {b['point']:+.4f} CI [{b['p2_5']:+.4f}, {b['p97_5']:+.4f}] covers0 {b['covers_zero']}"
            )
        clear_split(tmp, "v8", "val", val_files)
        splits = {}
        nested = {"val41": nest41}
        for ver, split, key in (("v8", "test", "test"), ("v10", "test", "v10"), ("v11", "test", "v11")):
            summary, b, nest = eval_split(m3, m1, ver, split, thr, tmp)
            splits[key] = summary
            boot[key] = b
            nested[key] = nest
        decisions = {name: verdict(name, hold_sum, boot, nested, splits) for name in CANDS}
        winners = [name for name in CANDS if decisions[name]["beats_gen2"]]
        if winners:
            selected = max(winners, key=lambda n: decisions[n]["hold41_delta"])
        else:
            selected = max(CANDS, key=lambda n: hold_sum[n]["iou"])
        report = dict(
            name="GEN3",
            seed=20261019,
            init="artifacts/eval/s3_gen2_unet.pt",
            definitions={
                "gen3": "unet gen3",
                "mean_w": "0.5 * (gen3 + ens_u2ft_tcn_lgbm)",
                "mean_g": "0.5 * (gen3 + ens_gen2), ens_gen2 = 0.5 * (gen2 + ens_u2ft_tcn_lgbm)",
                "ens_gen2": "0.5 * (gen2 + ens_u2ft_tcn_lgbm)",
            },
            bar="hold41 delta>=0.003 vs ens_gen2 and (CI excludes 0 or nested k5>=0.003) and v10/v11 not worse",
            proxy="all artifacts/cache/v10/val, not v8 val",
            holdout_dropped=list(HOLD_DROP),
            holdout_n=41,
            threshold_source="FAR 1e-3 on the 41 v8 val backgrounds only; frozen for test, v10, v11",
            winner_full_val_iou=win_full,
            winner_match=abs(win_full - WINNER_VAL) < 1e-9,
            winner_41_at_published_threshold=win_frozen,
            ens_gen2_full_val_iou=ens_full["iou"],
            ens_gen2_full_val_match=abs(ens_full["iou"] - ENS_VAL) < 0.005,
            thresholds=thr,
            val41=hold_sum,
            splits=splits,
            bootstrap=boot,
            nested=nested,
            decisions=decisions,
            selected=selected,
            beats_gen2=bool(decisions[selected]["beats_gen2"]) if winners else False,
            bootstrap_definition="1000 resamples of sequences, micro IoU delta versus ens_gen2, percentile 2.5 and 97.5, seed 20261019",
            real_used_for_training=False,
            real_used_for_threshold=False,
            seconds=round(time.time() - t0, 1),
        )
        # beats_gen2 is true only when at least one candidate clears the bar.
        report["beats_gen2"] = bool(winners)
        dump(report)
        log(f"записан {OUT} selected={selected} beats={report['beats_gen2']}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
