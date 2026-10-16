"""Оценка FuseNet: FAR 1e-3 по фону v8 val, v8 test, v10/v11 test, вложенные k=2 и k=5.

Порог официальной метрики — квантиль чистого фона всего v8 val.
Вложенные фолды — те же, что scripts/s3_sel2_cv.py: sequence_folds i%k,
eval_one на отложенном фолде, micro по сумме tp/fp/fn.
Сравнение с победителем и с фиксированной смесью (4×победитель + CDF)/5.
v8 val порог не двигает test. Реальные трассы не читаются. Карты не пишутся.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "NUMEXPR_MAX_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_k] = "2"

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

import s3_sel2_cv as s2
import s3_sel_core as sel
import s3_sel_eval as sev
from dts import data
from dts import metrics as M
from s3_fusenet_net import FuseNet, blend4, fuse_input, n_params, predict_prob

WINNER = "ens_u2ft_tcn_lgbm"
WINNER_VAL = 0.38671235821914546
PT = Path("artifacts/eval/s3_fusenet.pt")
TRAIN_JSON = Path("artifacts/eval/s3_fusenet_train.json")
KNOTS_JSON = Path("artifacts/eval/s3_mae_train.json")
OUT = Path("artifacts/eval/s3_fusenet_v8.json")
DOC = Path("docs/S3_FUSENET.md")
LOG = Path("/workspace/run_logs/_events.log")
NAMES = ("fusenet", "ens_mean4", WINNER)


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def elog(msg: str) -> None:
    line = time.strftime("%H:%M") + " " + msg
    log(msg)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_model():
    model = FuseNet()
    model.load_state_dict(torch.load(PT, map_location="cpu", weights_only=True))
    model.eval()
    return model


def scores_of(model, X, winner_s, knots) -> dict:
    feat = fuse_input(winner_s, X, knots)
    return {
        "fusenet": predict_prob(model, feat),
        "ens_mean4": blend4(winner_s, X, knots),
        WINNER: np.nan_to_num(np.asarray(winner_s, np.float32), nan=0.0, posinf=0.0, neginf=0.0),
    }


def accumulate(prob, ev, valid, nu, thr, bucket) -> None:
    pred = prob >= thr
    truth = valid & (ev > 0)
    bg = valid & (ev == 0) & (nu == 0)
    nup = valid & (ev == 0) & (nu > 0)
    bucket["tp"] += int((pred & truth).sum())
    bucket["fn"] += int((~pred & truth).sum())
    bucket["fp_bg"] += int((pred & bg).sum())
    bucket["fp_nu"] += int((pred & nup).sum())
    bucket["n_bg"] += int(bg.sum())
    bucket["n_ev"] += int(truth.sum())
    bucket["n_nu"] += int(nup.sum())


def empty():
    return dict(tp=0, fn=0, fp_bg=0, fp_nu=0, n_bg=0, n_ev=0, n_nu=0)


def finish(bucket, thr: float) -> dict:
    tp, fn = bucket["tp"], bucket["fn"]
    fp = bucket["fp_bg"] + bucket["fp_nu"]
    row = M.prf(tp, fp, fn)
    row.update(
        tp=tp, fp=fp, fn=fn, fp_bg=bucket["fp_bg"], fp_nu=bucket["fp_nu"],
        n_bg=bucket["n_bg"], n_ev=bucket["n_ev"], n_nu=bucket["n_nu"],
        far=float(bucket["fp_bg"] / max(bucket["n_bg"], 1)),
        nu_fpr=float(bucket["fp_nu"] / max(bucket["n_nu"], 1)),
        threshold=float(thr),
    )
    return row


def split_files(ver: str, split: str):
    return data.list_cache(f"artifacts/cache/{ver}", split)


def load_pair(ver, split, path: Path):
    with np.load(path) as z:
        X = np.asarray(z["X"])
        ev = np.asarray(z["ev"])
        valid = np.asarray(z["valid"]) > 0.5
        nu = np.asarray(z["nu"])
    pred = Path("artifacts/pred") / ver / WINNER / split / f"{path.stem}.npz"
    with np.load(pred) as z:
        s = np.asarray(z["s"], np.float32)
    return X, ev, valid, nu, s


def official(model, knots) -> tuple[dict, dict]:
    """Возвращает метрики и компактные кадры v8 val для вложенной проверки."""
    bg = {n: [] for n in NAMES}
    frames = {n: [] for n in NAMES}
    log("v8 val pass 1 (порог)")
    files = split_files("v8", "val")
    for i, p in enumerate(files, 1):
        X, ev, valid, nu, s = load_pair("v8", "val", p)
        sc = scores_of(model, X, s, knots)
        mask = valid & (ev == 0) & (nu == 0)
        ev_i, nu_i, bg_i, n_bg = sev._indices(ev, valid, nu, p.stem)
        for n in NAMES:
            bg[n].append(sc[n][mask])
            e, nu_v, b = sev._take(sc[n], ev_i, nu_i, bg_i)
            frames[n].append(dict(
                stem=p.stem,
                ev_m=e.astype(np.float16)[None],
                nu_m=nu_v.astype(np.float16)[None],
                bg_m=b.astype(np.float16)[None],
                n_bg=n_bg,
                cooling=[],
            ))
        if i == 1 or i % 15 == 0 or i == len(files):
            log(f"  val {i}/{len(files)}")
        del X, sc
    thr = {}
    for n in NAMES:
        cat = np.concatenate(bg[n])
        thr[n] = M.far_threshold(cat, 1e-3)
        log(f"thr {n} {thr[n]:.6f} n_bg {cat.size}")
        del cat
    del bg
    report = {}
    for ver, split in (("v8", "val"), ("v8", "test"), ("v10", "test"), ("v11", "test")):
        buckets = {n: empty() for n in NAMES}
        files = split_files(ver, split)
        for i, p in enumerate(files, 1):
            X, ev, valid, nu, s = load_pair(ver, split, p)
            sc = scores_of(model, X, s, knots)
            for n in NAMES:
                accumulate(sc[n], ev, valid, nu, thr[n], buckets[n])
            if i == len(files) or i % 15 == 0:
                log(f"  {ver} {split} {i}/{len(files)}")
            del X, sc
        report[f"{ver}/{split}"] = {n: finish(buckets[n], thr[n]) for n in NAMES}
        fuse = report[f"{ver}/{split}"]["fusenet"]["iou"]
        win = report[f"{ver}/{split}"][WINNER]["iou"]
        log(f"{ver} {split} fuse {fuse:.4f} winner {win:.4f}")
    return report, frames, thr


def nested(frames) -> dict:
    order = np.argsort([f["stem"] for f in frames["fusenet"]])
    ordered = {n: [frames[n][i] for i in order] for n in NAMES}
    out = {}
    for k in (2, 5):
        folds = sel.sequence_folds(len(ordered["fusenet"]), k)
        parts = {n: [] for n in NAMES}
        rows = []
        for r, idx in enumerate(folds):
            hold = {}
            for n in NAMES:
                hold[n] = s2.eval_one(s2.take(ordered[n], idx))
                parts[n].append(hold[n])
            rows.append(dict(
                fold=r,
                fusenet=float(hold["fusenet"]["iou"]),
                ens_mean4=float(hold["ens_mean4"]["iou"]),
                winner=float(hold[WINNER]["iou"]),
                delta_winner=float(hold["fusenet"]["iou"] - hold[WINNER]["iou"]),
                delta_blend=float(hold["fusenet"]["iou"] - hold["ens_mean4"]["iou"]),
            ))
            log(
                f"nested k={k} fold {r} fuse {hold['fusenet']['iou']:.4f} "
                f"blend {hold['ens_mean4']['iou']:.4f} winner {hold[WINNER]['iou']:.4f}"
            )
        micros = {n: float(sel.micro_iou(parts[n])) for n in NAMES}
        out[str(k)] = dict(
            fusenet=micros["fusenet"],
            ens_mean4=micros["ens_mean4"],
            winner=micros[WINNER],
            delta_winner=float(micros["fusenet"] - micros[WINNER]),
            delta_blend=float(micros["fusenet"] - micros["ens_mean4"]),
            folds=rows,
        )
        log(f"nested k={k} micro fuse {micros['fusenet']:.4f} dW {out[str(k)]['delta_winner']:+.4f} dB {out[str(k)]['delta_blend']:+.4f}")
    return out


def render(rep: dict) -> str:
    tr = rep["train"]
    def cell(split, name, key, nd=4):
        v = rep["splits"][split][name][key]
        return f"{v:.{nd}f}"

    lines = [
        "# S3 FuseNet",
        "",
        "Обучаемое слияние вместо фиксированного 4:1. Маленькая сеть из свёрток 3×3",
        f"({tr['params']} параметров, лимит 5000) читает четыре канала:",
        "скор победителя ens_u2ft_tcn_lgbm, CDF(|dT_space|), |dT_time|/12, valid.",
        "Узлы CDF взяты из обучения MAE (`artifacts/eval/s3_mae_train.json`), не из v8 val.",
        "Сеть учится только на 60 последовательностях synth v10 train",
        f"({tr['n_train']} в лосс, {tr['n_hold']} — прокси IoU@FAR 1e-3 для ранней остановки).",
        "v8 val порог официальной метрики не двигает и в лосс не входит.",
        "Фиксированная смесь для сравнения: (4×победитель + CDF(|dT_space|)) / 5, те же узлы.",
        "Порог каждого метода — свой квантиль чистого фона всего v8 val при FAR 1e-3.",
        "Вложенные фолды — `sequence_folds` (i % k) и `eval_one` из scripts/s3_sel2_cv.py,",
        "micro по сумме tp/fp/fn. Реальные трассы не читались. Карты победителя на train",
        "лежали в /tmp и удалены после обучения.",
        "",
        f"Лучшая эпоха прокси {tr['best'].get('epoch')}, прокси IoU {tr['best'].get('iou')},",
        f"остановка {tr.get('stopped')}, {tr.get('seconds')} с, pos_weight {tr.get('pos_weight')}.",
        f"Совпадение генератора карт с artifacts/pred: max |Δ| {tr.get('winner_sanity_maxabs')}.",
        "",
        "## Официальный FAR 1e-3",
        "",
        "| сплит | fusenet | смесь 4:1 | победитель | Δ к победителю | Δ к 4:1 | порог fuse |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("v8/val", "v8/test", "v10/test", "v11/test"):
        f = rep["splits"][split]["fusenet"]["iou"]
        b = rep["splits"][split]["ens_mean4"]["iou"]
        w = rep["splits"][split][WINNER]["iou"]
        thr = rep["splits"][split]["fusenet"]["threshold"]
        lines.append(
            f"| {split} | {f:.4f} | {b:.4f} | {w:.4f} | {f - w:+.4f} | {f - b:+.4f} | {thr:.4f} |"
        )
    lines += [
        "",
        "Пересчёт победителя на v8 val должен совпасть с опубликованным 0.386712.",
        f"Здесь {rep['splits']['v8/val'][WINNER]['iou']:.6f}, смесь {rep['splits']['v8/val']['ens_mean4']['iou']:.6f}.",
        "",
        "| сплит | P fuse | R fuse | FAR fuse | nuFPR fuse | FAR победитель | nuFPR победитель | nuFPR 4:1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("v8/val", "v8/test", "v10/test", "v11/test"):
        f = rep["splits"][split]["fusenet"]
        w = rep["splits"][split][WINNER]
        b = rep["splits"][split]["ens_mean4"]
        lines.append(
            f"| {split} | {f['precision']:.4f} | {f['recall']:.4f} | {f['far']:.5f} | {f['nu_fpr']:.5f} | "
            f"{w['far']:.5f} | {w['nu_fpr']:.5f} | {b['nu_fpr']:.5f} |"
        )
    lines += ["", "## Вложенные фолды", ""]
    for k in ("2", "5"):
        block = rep["nested"][k]
        lines += [
            f"### k={k}",
            "",
            f"Micro: fusenet {block['fusenet']:.4f}, смесь {block['ens_mean4']:.4f}, "
            f"победитель {block['winner']:.4f}, Δ победитель {block['delta_winner']:+.4f}, "
            f"Δ смесь {block['delta_blend']:+.4f}.",
            "",
            "| фолд | fusenet | смесь 4:1 | победитель | Δ победитель | Δ смесь |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for r in block["folds"]:
            lines.append(
                f"| {r['fold']} | {r['fusenet']:.4f} | {r['ens_mean4']:.4f} | {r['winner']:.4f} | "
                f"{r['delta_winner']:+.4f} | {r['delta_blend']:+.4f} |"
            )
        lines.append("")
    beats = rep["beats_winner"]
    lines += [
        "## Итог",
        "",
        f"По val fusenet {rep['splits']['v8/val']['fusenet']['iou']:.4f} против победителя "
        f"{rep['splits']['v8/val'][WINNER]['iou']:.4f} и смеси "
        f"{rep['splits']['v8/val']['ens_mean4']['iou']:.4f}.",
        "Победитель по умолчанию не менялся. `models_release/` не трогался. real_train=0.",
        f"BEATS_WINNER: {'да' if beats else 'нет'}.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    torch.set_num_threads(2)
    t0 = time.time()
    train = json.loads(TRAIN_JSON.read_text())
    knots = np.asarray(json.loads(KNOTS_JSON.read_text())["cdf_knots"], np.float64).copy()
    model = load_model()
    splits, frames, thr = official(model, knots)
    nest = nested(frames)
    win_val = splits["v8/val"][WINNER]["iou"]
    fuse_val = splits["v8/val"]["fusenet"]["iou"]
    protocol_ok = abs(win_val - WINNER_VAL) < 2e-4
    beats = bool(protocol_ok and fuse_val > WINNER_VAL + 1e-4)
    rep = dict(
        name="FUSENET",
        params=n_params(model),
        train=dict(
            params=train.get("params"),
            best=train.get("best"),
            stopped=train.get("stopped"),
            seconds=train.get("seconds"),
            n_train=train.get("n_train"),
            n_hold=train.get("n_hold"),
            pos_weight=train.get("pos_weight"),
            winner_sanity_maxabs=train.get("winner_sanity_maxabs"),
            val_used_for_train=False,
            real_traces_used=False,
        ),
        thresholds=thr,
        splits=splits,
        nested=nest,
        nested_procedure="s3_sel2_cv.eval_one, sequence_folds i%k, micro_iou",
        protocol_ok=protocol_ok,
        beats_winner=beats,
        winner_val_published=WINNER_VAL,
        seconds=round(time.time() - t0, 1),
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    DOC.write_text(render(rep), encoding="utf-8")
    sv = splits["v8/val"]
    st = splits["v8/test"]
    s10 = splits["v10/test"]
    s11 = splits["v11/test"]
    elog(
        "FUSENET_DONE "
        f"params={n_params(model)} best_epoch={train.get('best', {}).get('epoch')} "
        f"val={sv['fusenet']['iou']:.6f} blend_val={sv['ens_mean4']['iou']:.6f} "
        f"winner_val={sv[WINNER]['iou']:.6f} "
        f"delta_val_winner={sv['fusenet']['iou'] - sv[WINNER]['iou']:+.6f} "
        f"delta_val_blend={sv['fusenet']['iou'] - sv['ens_mean4']['iou']:+.6f} "
        f"test={st['fusenet']['iou']:.6f} blend_test={st['ens_mean4']['iou']:.6f} "
        f"winner_test={st[WINNER]['iou']:.6f} "
        f"v10={s10['fusenet']['iou']:.6f} v10_blend={s10['ens_mean4']['iou']:.6f} "
        f"v10_winner={s10[WINNER]['iou']:.6f} "
        f"v11={s11['fusenet']['iou']:.6f} v11_blend={s11['ens_mean4']['iou']:.6f} "
        f"v11_winner={s11[WINNER]['iou']:.6f} "
        f"nested5_delta_winner={nest['5']['delta_winner']:+.6f} "
        f"nested5_delta_blend={nest['5']['delta_blend']:+.6f} "
        f"nested2_delta_winner={nest['2']['delta_winner']:+.6f} "
        f"nested2_delta_blend={nest['2']['delta_blend']:+.6f} "
        f"thr={sv['fusenet']['threshold']:.6f} "
        f"protocol_ok={int(protocol_ok)} beats={int(beats)} real_train=0"
    )
    if beats:
        elog(
            f"BEATS_WINNER FUSENET val {sv['fusenet']['iou']:.6f} test {st['fusenet']['iou']:.6f} "
            f"winner_val {sv[WINNER]['iou']:.6f}"
        )
    log(f"wrote {OUT} {DOC}")


if __name__ == "__main__":
    main()
