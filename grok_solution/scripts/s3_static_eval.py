"""Оценка StaticNet: порог FAR 1e-3 только по фону v8 val, выбор только по synth val.

Test не участвует в выборе. Реальные трассы здесь не читаются.
IoU — пиксельный, как в scripts/eval_stage1.py. Статичное событие засчитано,
если порог закрывает ≥ 30 % его пикселей.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("MKL_NUM_THREADS", "3")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "3")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dts import metrics as M
from dts.static_net import StaticNet, add_score_hist, fuse, predict_map, residual_from_cache, threshold_from_hist
from dts.static_synth import (
    HOT_AMP, HOT_WIDTH, SPOT_RANGE, apply_baseline, apply_profiles, apply_row_gaps,
    fixed_scene, seq_seed, unit_profile,
)

WINNER_PUB_VAL = 0.386712
WINNER_PUB_TEST = 0.405276
FAR = 1e-3
NBIN = 200_000
HIT_FRAC = 0.3
DVAL_MIN = -0.003
VARIANTS = (
    {"name": "winner", "mode": "identity", "scale": 1.0, "gate": 0.0},
    {"name": "branch", "mode": "branch", "scale": 1.0, "gate": 0.0},
    {"name": "max", "mode": "max", "scale": 1.0, "gate": 0.0},
    {"name": "max_g90", "mode": "max", "scale": 1.0, "gate": 0.90},
    {"name": "max_g95", "mode": "max", "scale": 1.0, "gate": 0.95},
    {"name": "max_g98", "mode": "max", "scale": 1.0, "gate": 0.98},
    {"name": "blend50", "mode": "blend", "scale": 0.5, "gate": 0.0},
)
LOG = Path("/workspace/grok_logs/_events.log")


def log(msg: str):
    line = time.strftime("%H:%M") + " " + msg
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def amp_bin(a: float) -> str:
    if a < 5:
        return "2-5C"
    if a < 8:
        return "5-8C"
    if a < 15:
        return "8-15C"
    return ">15C"


def load_model(path: Path) -> StaticNet:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = StaticNet(
        cin=int(ckpt.get("cin", 9)), width=int(ckpt.get("width", 16)), pool=int(ckpt.get("pool", 80)),
        max_rows=int(ckpt.get("max_rows", 64)), n_bins=int(ckpt.get("n_bins", 0)),
    )
    net.load_state_dict(ckpt["state"])
    net.eval()
    net._ckpt_step = int(ckpt.get("step", -1))
    return net


def pairs(split: str):
    cache = ROOT / "artifacts/cache/v8" / split
    pred = ROOT / "artifacts/pred/v8/ens_u2ft_tcn_lgbm" / split
    for p in sorted(cache.glob("seq*.npz")):
        q = pred / p.name
        if q.exists():
            yield p, q


def load_seq(cache_path: Path, pred_path: Path):
    z = np.load(cache_path)
    R, valid = residual_from_cache(z)
    s = np.asarray(np.load(pred_path)["s"], np.float32)
    if s.shape != valid.shape:
        raise RuntimeError(f"shape {cache_path.name}: score {s.shape} vs valid {valid.shape}")
    return dict(R=R, valid=valid, ev=np.asarray(z["ev"]), nu=np.asarray(z["nu"]),
                x=np.asarray(z["x"]), name=cache_path.stem, s=s)


def empty_counts():
    return dict(tot=np.zeros(3, np.int64), hot=np.zeros(3, np.int64), cold=np.zeros(3, np.int64),
                far_n=0, far_d=0, hist=np.zeros(NBIN, np.int64))


def accumulate(counts, fused, d, thr=None):
    bg = d["valid"] & (d["ev"] == 0) & (d["nu"] == 0)
    if thr is None:
        add_score_hist(counts["hist"], fused[bg])
        return
    pred = (fused >= thr) & d["valid"]
    truth = d["ev"] > 0
    counts["tot"] += M.confusion(pred, truth, d["valid"])
    counts["hot"] += M.confusion(pred & (d.get("h") is not None and False), truth, d["valid"])  # placeholder overwritten below
    # hot/cold считаются снаружи, если передана карта h. Здесь только общий IoU и FAR.
    counts["far_n"] += int((pred & bg).sum())
    counts["far_d"] += int(bg.sum())
    return pred


def event_rows(specs, fused, valid, thr):
    rows = []
    width = fused.shape[1]
    for sp in specs:
        if sp.get("role", "hot") != "hot":
            continue
        cols = unit_profile(width, sp) >= 0.5
        m = valid[:, cols] if cols.any() else np.zeros((valid.shape[0], 0), bool)
        # valid[:, cols] keeps time; need pixel mask
        if not cols.any():
            continue
        pix = valid & cols[None, :]
        if not pix.any():
            continue
        frac = float((fused[pix] >= thr).mean())
        rows.append(dict(amp=float(sp["amp"]), amp_bin=amp_bin(float(sp["amp"])),
                         width_m=float(sp["width_m"]), frac=frac, hit=frac >= HIT_FRAC))
    return rows


def summarize_events(rows):
    if not rows:
        return dict(n=0, recall=None, pixel_recall=None, by_amp={})
    hit = np.array([r["hit"] for r in rows])
    frac = np.array([r["frac"] for r in rows], float)
    out = dict(n=int(len(rows)), recall=float(hit.mean()), pixel_recall=float(frac.mean()))
    by = {}
    for key in ("2-5C", "5-8C", "8-15C", ">15C"):
        sub = [r for r in rows if r["amp_bin"] == key]
        if not sub:
            continue
        by[key] = dict(n=len(sub), recall=float(np.mean([r["hit"] for r in sub])),
                       pixel_recall=float(np.mean([r["frac"] for r in sub])))
    out["by_amp"] = by
    return out


def pack_counts(counts, thr):
    pr = M.prf(*counts["tot"])
    return dict(threshold=float(thr), far=float(counts["far_n"] / max(counts["far_d"], 1)), **pr)


def static_specs_for(d, seed_base: int):
    avoid = (d["ev"] > 0).mean(0) > 0.02
    return fixed_scene(d["R"].shape[1], seq_seed(seed_base, d["name"]), n_events=2, avoid=avoid)


def run_split(model, split: str, seed_base: int, thresholds: dict | None, save_specs: bool):
    """Если thresholds is None — только гистограммы фона и карты p не копятся.
    Иначе IoU и recall статичных событий.
    """
    counts = {v["name"]: empty_counts() for v in VARIANTS}
    events = {v["name"]: [] for v in VARIANTS}
    spec_dump = []
    n = 0
    for cache_p, pred_p in pairs(split):
        d = load_seq(cache_p, pred_p)
        pmap = predict_map(model, d["R"], d["valid"])
        if thresholds is None:
            for v in VARIANTS:
                fused = fuse(d["s"], pmap, v["mode"], v["scale"], v["gate"])
                accumulate(counts[v["name"]], fused, d, thr=None)
            n += 1
            if n % 10 == 0:
                print(f"  {split} {n}", flush=True)
            del d
            continue
        scene = static_specs_for(d, seed_base)
        specs = scene["specs"]
        if save_specs:
            spec_dump.append(dict(seq=d["name"], specs=specs, baseline=scene["baseline"], gap=scene["gap"]))
        Rp, _, _ = apply_profiles(d["R"], specs)
        Rp = apply_baseline(Rp, scene["baseline"])
        # Пропуски только на копии valid вклеенного пути. Чистый фон порога и IoU не трогаем.
        v_gap = apply_row_gaps(d["valid"], scene["gap"])
        p_static = predict_map(model, Rp, v_gap) if specs else pmap
        for v in VARIANTS:
            thr = thresholds[v["name"]]
            fused = fuse(d["s"], pmap, v["mode"], v["scale"], v["gate"])
            pred = (fused >= thr) & d["valid"]
            truth = d["ev"] > 0
            bg = d["valid"] & (d["ev"] == 0) & (d["nu"] == 0)
            counts[v["name"]]["tot"] += M.confusion(pred, truth, d["valid"])
            counts[v["name"]]["far_n"] += int((pred & bg).sum())
            counts[v["name"]]["far_d"] += int(bg.sum())
            fused_s = fuse(d["s"], p_static, v["mode"], v["scale"], v["gate"])
            events[v["name"]].extend(event_rows(specs, fused_s, v_gap, thr))
        n += 1
        if n % 10 == 0:
            print(f"  {split} {n}", flush=True)
        del d, Rp
    return counts, events, spec_dump


def val_pass(model, seed: int, save_specs: bool):
    counts_thr, _, _ = run_split(model, "val", seed, thresholds=None, save_specs=False)
    thresholds = {}
    for v in VARIANTS:
        thresholds[v["name"]] = threshold_from_hist(counts_thr[v["name"]]["hist"], FAR)
        print(f"thr {v['name']} {thresholds[v['name']]:.6f} n_bg {int(counts_thr[v['name']]['hist'].sum())}", flush=True)
    counts_val, events_val, spec_val = run_split(model, "val", seed, thresholds, save_specs=save_specs)
    report = {}
    for v in VARIANTS:
        row = pack_counts(counts_val[v["name"]], thresholds[v["name"]])
        row["static"] = summarize_events(events_val[v["name"]])
        row.update({k: v[k] for k in ("mode", "scale", "gate")})
        report[v["name"]] = row
        st = row["static"]
        print(v["name"], "val iou", round(row["iou"], 4), "far", round(row["far"], 5),
              "static", None if st.get("recall") is None else round(st["recall"], 4),
              f"{st.get('n')}", flush=True)
    return report, thresholds, spec_val


def choose(val_report, step: int):
    """Максимум static recall при dval >= -0.003 относительно победителя этого прогона."""
    winner_iou = val_report["winner"]["iou"]
    ranked = [(name, row) for name, row in val_report.items() if name != "winner"]
    ok = [r for r in ranked if r[1]["iou"] >= winner_iou + DVAL_MIN - 1e-12]
    pool = ok or ranked

    def key(r):
        return (
            r[1]["static"]["recall"] or 0.0,
            r[1]["iou"],
            1.0 if r[1]["mode"] == "max" and r[1]["gate"] == 0.0 else 0.0,
            step,
        )

    pool.sort(key=key, reverse=True)
    return pool[0][0], pool[0][1], bool(ok), winner_iou


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", default=["artifacts/static/static_net.pt"])
    ap.add_argument("--out", default="artifacts/static/s3_static.json")
    ap.add_argument("--seed", type=int, default=20261015)
    ap.add_argument("--tag", default="STATIC")
    ap.add_argument("--skip-test", action="store_true")
    ap.add_argument("--spec-prefix", default="synth_static")
    a = ap.parse_args()
    torch.set_num_threads(3)
    t0 = time.time()
    sweeps = []
    best = None
    spec_val = None
    for i, ckpt_s in enumerate(a.ckpt):
        model = load_model(ROOT / ckpt_s)
        step = int(getattr(model, "_ckpt_step", -1))
        log(f"{a.tag}_EVAL ckpt={ckpt_s} step={step} val")
        val_report, thresholds, specs = val_pass(model, a.seed, save_specs=(spec_val is None))
        if spec_val is None:
            spec_val = specs
        name, row, ok, winner_iou = choose(val_report, step)
        item = dict(ckpt=ckpt_s, step=step, selected=name, constraint_met=ok,
                    winner_iou=winner_iou, row=row, val=val_report, thresholds=thresholds)
        sweeps.append(item)
        log(f"{a.tag}_EVAL ckpt={ckpt_s} pick={name} val_iou={row['iou']:.4f} "
            f"dval={row['iou'] - winner_iou:+.4f} static_val={row['static']['recall']} "
            f"winner_static={val_report['winner']['static']['recall']} "
            f"branch_static={val_report['branch']['static']['recall']} "
            f"constraint={'ok' if ok else 'best_effort'}")
        rank = (
            row["static"]["recall"] or 0.0,
            row["iou"],
            1.0 if row["mode"] == "max" and row["gate"] == 0.0 else 0.0,
            step,
        )
        if best is None or rank > best["rank"]:
            best = dict(rank=rank, **item)
        del model
    chosen_name = best["selected"]
    chosen = best["row"]
    winner_iou = best["winner_iou"]
    val_report = best["val"]
    thresholds = best["thresholds"]
    beats = bool(chosen["iou"] > WINNER_PUB_VAL)
    test_report = None
    spec_te = None
    if not a.skip_test:
        model = load_model(ROOT / best["ckpt"])
        log(f"{a.tag}_EVAL test ckpt={best['ckpt']}")
        counts_te, events_te, spec_te = run_split(model, "test", a.seed + 17, thresholds, save_specs=True)
        test_report = {}
        for v in VARIANTS:
            row = pack_counts(counts_te[v["name"]], thresholds[v["name"]])
            row["static"] = summarize_events(events_te[v["name"]])
            test_report[v["name"]] = row
            print(v["name"], "test iou", round(row["iou"], 4),
                  "static", row["static"].get("recall"), flush=True)
    art = ROOT / "artifacts/static"
    art.mkdir(parents=True, exist_ok=True)
    if spec_val is not None and a.spec_prefix == "synth_static":
        (art / "synth_static_val.json").write_text(json.dumps(spec_val))
    elif spec_val is not None:
        (art / f"{a.spec_prefix}_val.json").write_text(json.dumps(spec_val))
    if spec_te is not None and a.spec_prefix == "synth_static":
        (art / "synth_static_test.json").write_text(json.dumps(spec_te))
    elif spec_te is not None:
        (art / f"{a.spec_prefix}_test.json").write_text(json.dumps(spec_te))

    def slim(item):
        return dict(ckpt=item["ckpt"], step=item["step"], selected=item["selected"],
                    constraint_met=item["constraint_met"],
                    val_iou=item["row"]["iou"],
                    dval=item["row"]["iou"] - item["winner_iou"],
                    static_val=item["row"]["static"]["recall"],
                    winner_static=item["val"]["winner"]["static"]["recall"],
                    branch_static=item["val"]["branch"]["static"]["recall"],
                    threshold=item["row"]["threshold"])

    out = dict(
        name=a.tag,
        far=FAR,
        dval_min=DVAL_MIN,
        ckpt=best["ckpt"],
        step=best["step"],
        published_winner=dict(val_iou=WINNER_PUB_VAL, test_iou=WINNER_PUB_TEST, threshold=0.86080873),
        winner_recompute=dict(val=val_report["winner"],
                              test=None if test_report is None else test_report["winner"]),
        branch_alone=dict(val=val_report["branch"],
                          test=None if test_report is None else test_report["branch"]),
        selected=chosen_name,
        constraint_met=best["constraint_met"],
        beats_winner=beats,
        sweeps=[slim(s) for s in sweeps],
        val=val_report,
        test=test_report,
        seconds=round(time.time() - t0, 1),
        generator=dict(hot_width=list(HOT_WIDTH), hot_amp=list(HOT_AMP), spot_m=list(SPOT_RANGE),
                       scene="fixed_scene: hots + neighbors + baseline + row gaps; event_rows skips role!=hot"),
        note="Порог — квантиль фона v8 val (valid & ev==0 & nu==0). "
             "События вклеены только на время подсчёта recall и не входят в фон порога. "
             "Знаменатель — горячие события fixed_scene, соседи не считаются. "
             "Выбор — максимальный static recall на val при dval >= -0.003. Test в выборе не участвовал. "
             "branch — карта одной ветви, без ансамбля.",
    )
    Path(ROOT / a.out).write_text(json.dumps(out, indent=1))
    sel_te = None if test_report is None else test_report[chosen_name]
    te_iou = None if sel_te is None else sel_te["iou"]
    te_st = None if sel_te is None else sel_te["static"]["recall"]
    w_te = None if test_report is None else test_report["winner"]["iou"]
    log(f"{a.tag}_EVAL_DONE selected={chosen_name} ckpt={best['ckpt']} step={best['step']} "
        f"val_iou={chosen['iou']:.4f} test_iou={te_iou} "
        f"dval={chosen['iou'] - winner_iou:+.4f} static_val={chosen['static']['recall']} "
        f"static_test={te_st} thr={chosen['threshold']:.4f} "
        f"winner_val={winner_iou:.4f} winner_test={w_te} "
        f"winner_static_val={val_report['winner']['static']['recall']} "
        f"branch_static_val={val_report['branch']['static']['recall']} "
        f"beats={int(beats)}")


if __name__ == "__main__":
    main()
