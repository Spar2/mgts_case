"""Оценка StaticNet6. Порог FAR 1e-3 только по фону v8 val.

Знаменатель static recall — опубликованные synth_static_val/test (генератор
STATIC2, без пятна). Сцена STATIC5 не используется. Test в выборе не участвует.
Вход — сырая T из кэша, без пространственной медианы.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

_NTHR = os.environ.get("OMP_NUM_THREADS", "3")
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, _NTHR)

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dts import metrics as M
from dts.static_net import (
    REF_DILATIONS, TRUNK_DILATIONS, StaticNet6, add_score_hist, fuse, predict_temperature,
    threshold_from_hist,
)
from dts.static_synth import apply_profiles, static2_view, unit_profile

WINNER_PUB_VAL = 0.386712
WINNER_PUB_TEST = 0.405276
S2_VAL_IOU = 0.386027
S2_HITS = 62
S2_N = 87
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
LOG = Path("/workspace/run_logs/_events.log")
SPEC = {
    "val": ROOT / "artifacts/static/synth_static_val.json",
    "test": ROOT / "artifacts/static/synth_static_test.json",
}


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


def load_model(path: Path) -> StaticNet6:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("arch") != "static6":
        raise SystemExit(f"{path} не StaticNet6")
    net = StaticNet6(
        cin=int(ckpt.get("cin", 5)),
        width=int(ckpt.get("width", 16)),
        max_rows=int(ckpt.get("max_rows", 256)),
        win_30=int(ckpt.get("win_30", 15)),
        win_2h=int(ckpt.get("win_2h", 60)),
        ref_dilations=tuple(ckpt.get("ref_dilations", REF_DILATIONS)),
        trunk_dilations=tuple(ckpt.get("trunk_dilations", TRUNK_DILATIONS)),
    )
    net.load_state_dict(ckpt["state"])
    net.eval()
    net._ckpt_step = int(ckpt.get("step", -1))
    return net


def spec_map(split: str) -> dict:
    rows = json.loads(SPEC[split].read_text())
    out = {}
    for row in rows:
        out[row["seq"]] = list(row.get("specs") or [])
    return out


def pairs(split: str):
    cache = ROOT / "artifacts/cache/v8" / split
    pred = ROOT / "artifacts/pred/v8/ens_u2ft_tcn_lgbm" / split
    for p in sorted(cache.glob("seq*.npz")):
        q = pred / p.name
        if q.exists():
            yield p, q


def load_seq(cache_path: Path, pred_path: Path):
    z = np.load(cache_path)
    if "T" not in z.files:
        raise RuntimeError(f"{cache_path} без сырой T")
    T = np.asarray(z["T"], np.float32)
    valid = np.asarray(z["valid"], bool)
    s = np.asarray(np.load(pred_path)["s"], np.float32)
    if s.shape != valid.shape or T.shape != valid.shape:
        raise RuntimeError(f"shape {cache_path.name}: T {T.shape} score {s.shape} valid {valid.shape}")
    return dict(T=T, valid=valid, ev=np.asarray(z["ev"]), nu=np.asarray(z["nu"]),
                name=cache_path.stem, s=s)


def empty_counts():
    return dict(tot=np.zeros(3, np.int64), far_n=0, far_d=0, hist=np.zeros(NBIN, np.int64))


def event_rows(specs, fused, valid, thr):
    rows = []
    width = fused.shape[1]
    for sp in specs:
        if sp.get("role", "hot") != "hot":
            continue
        cols = unit_profile(width, static2_view(sp)) >= 0.5
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
        return dict(n=0, hits=0, recall=None, pixel_recall=None, by_amp={})
    hit = np.array([r["hit"] for r in rows])
    frac = np.array([r["frac"] for r in rows], float)
    out = dict(n=int(len(rows)), hits=int(hit.sum()), recall=float(hit.mean()),
               pixel_recall=float(frac.mean()))
    by = {}
    for key in ("2-5C", "5-8C", "8-15C", ">15C"):
        sub = [r for r in rows if r["amp_bin"] == key]
        if not sub:
            continue
        by[key] = dict(n=len(sub), hits=int(sum(r["hit"] for r in sub)),
                       recall=float(np.mean([r["hit"] for r in sub])),
                       pixel_recall=float(np.mean([r["frac"] for r in sub])))
    out["by_amp"] = by
    return out


def pack_counts(counts, thr):
    pr = M.prf(*counts["tot"])
    return dict(threshold=float(thr), far=float(counts["far_n"] / max(counts["far_d"], 1)), **pr)


def run_split(model, split: str, specs_by_seq: dict, thresholds: dict | None):
    counts = {v["name"]: empty_counts() for v in VARIANTS}
    events = {v["name"]: [] for v in VARIANTS}
    n = 0
    missing = 0
    for cache_p, pred_p in pairs(split):
        d = load_seq(cache_p, pred_p)
        pmap = predict_temperature(model, d["T"], d["valid"])
        if thresholds is None:
            for v in VARIANTS:
                fused = fuse(d["s"], pmap, v["mode"], v["scale"], v["gate"])
                bg = d["valid"] & (d["ev"] == 0) & (d["nu"] == 0)
                add_score_hist(counts[v["name"]]["hist"], fused[bg])
            n += 1
            if n % 10 == 0:
                print(f"  {split} hist {n}", flush=True)
            del d
            continue
        specs = specs_by_seq.get(d["name"])
        if specs is None:
            missing += 1
            specs = []
        views = [static2_view(sp) for sp in specs]
        Tp, _, _ = apply_profiles(d["T"], views) if views else (d["T"], None, None)
        p_static = predict_temperature(model, Tp, d["valid"]) if views else pmap
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
            events[v["name"]].extend(event_rows(specs, fused_s, d["valid"], thr))
        n += 1
        if n % 10 == 0:
            print(f"  {split} {n}", flush=True)
        del d, Tp
    if missing:
        print(f"  {split} specs missing {missing}", flush=True)
    return counts, events


def val_pass(model, specs_by_seq):
    counts_thr, _ = run_split(model, "val", specs_by_seq, thresholds=None)
    thresholds = {}
    for v in VARIANTS:
        thresholds[v["name"]] = threshold_from_hist(counts_thr[v["name"]]["hist"], FAR)
        print(f"thr {v['name']} {thresholds[v['name']]:.6f} n_bg {int(counts_thr[v['name']]['hist'].sum())}",
              flush=True)
    counts_val, events_val = run_split(model, "val", specs_by_seq, thresholds)
    report = {}
    for v in VARIANTS:
        row = pack_counts(counts_val[v["name"]], thresholds[v["name"]])
        row["static"] = summarize_events(events_val[v["name"]])
        row.update({k: v[k] for k in ("mode", "scale", "gate")})
        report[v["name"]] = row
        st = row["static"]
        print(v["name"], "val iou", round(row["iou"], 4), "far", round(row["far"], 5),
              "static", None if st.get("recall") is None else round(st["recall"], 4),
              f"{st.get('hits')}/{st.get('n')}", flush=True)
    return report, thresholds


def choose(val_report, step: int):
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


def better_than_static2(row, constraint_met: bool) -> bool:
    st = row["static"]
    if not constraint_met or not st.get("n"):
        return False
    if st["n"] == S2_N:
        hits = int(st.get("hits") or 0)
        if hits > S2_HITS:
            return True
        if hits == S2_HITS and row["iou"] > S2_VAL_IOU + 1e-6:
            return True
        return False
    rec = st["recall"] or 0.0
    if rec > (S2_HITS / S2_N) + 1e-9:
        return True
    if abs(rec - S2_HITS / S2_N) <= 1e-4 and row["iou"] > S2_VAL_IOU + 1e-6:
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--out", default="artifacts/static/s3_static6.json")
    ap.add_argument("--tag", default="STATIC6")
    ap.add_argument("--skip-test", action="store_true")
    ap.add_argument("--resume", default="", help="игнорируется: оценка без оптимизатора")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "3")))
    t0 = time.time()
    specs_val = spec_map("val")
    specs_te = spec_map("test")
    sweeps = []
    best = None
    for ckpt_s in a.ckpt:
        path = ROOT / ckpt_s
        if not path.is_file():
            print(f"skip missing {ckpt_s}", flush=True)
            continue
        model = load_model(path)
        step = int(getattr(model, "_ckpt_step", -1))
        log(f"{a.tag}_EVAL ckpt={ckpt_s} step={step} val")
        val_report, thresholds = val_pass(model, specs_val)
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
    if best is None:
        raise SystemExit("нет чекпойнтов")
    chosen_name = best["selected"]
    chosen = best["row"]
    winner_iou = best["winner_iou"]
    val_report = best["val"]
    thresholds = best["thresholds"]
    beats = bool(chosen["iou"] > WINNER_PUB_VAL + 1e-4)
    better = better_than_static2(chosen, best["constraint_met"])
    test_report = None
    if not a.skip_test:
        model = load_model(ROOT / best["ckpt"])
        log(f"{a.tag}_EVAL test ckpt={best['ckpt']}")
        counts_te, events_te = run_split(model, "test", specs_te, thresholds)
        test_report = {}
        for v in VARIANTS:
            row = pack_counts(counts_te[v["name"]], thresholds[v["name"]])
            row["static"] = summarize_events(events_te[v["name"]])
            test_report[v["name"]] = row
            st = row["static"]
            print(v["name"], "test iou", round(row["iou"], 4),
                  "static", f"{st.get('hits')}/{st.get('n')}", flush=True)

    def slim(item):
        return dict(ckpt=item["ckpt"], step=item["step"], selected=item["selected"],
                    constraint_met=item["constraint_met"],
                    val_iou=item["row"]["iou"],
                    dval=item["row"]["iou"] - item["winner_iou"],
                    static_val=item["row"]["static"]["recall"],
                    static_hits=item["row"]["static"].get("hits"),
                    static_n=item["row"]["static"].get("n"),
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
        static2_reference=dict(val_iou=S2_VAL_IOU, static_hits=S2_HITS, static_n=S2_N,
                               threshold=0.865895, mode="max", step=800,
                               ckpt="artifacts/static/static_net2_s800.pt"),
        better_than_static2=better,
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
        generator=dict(
            name="static2",
            specs_val=str(SPEC["val"].relative_to(ROOT)),
            specs_test=str(SPEC["test"].relative_to(ROOT)),
            spot_m=0,
            note="Вклейка опубликованных спецификаций без пятна прибора. Генератор STATIC5 не вызывался.",
        ),
        input=dict(
            eval_field="raw T, one scalar median removed inside temperature_features, no spatial median",
            train_field="cache residual + STATIC2 smooth trend; train npz has no T",
            windows="15 and 60 rows on the linspace grid the model sees (30 min and 2 h at dt=120 s when rows<=256)",
        ),
        note="Порог — квантиль фона v8 val без вклеек. Выбор — максимальный static recall "
             "при dval >= -0.003. Test не выбирает. Сравнение с STATIC2 — 62/87 и val IoU 0.386027.",
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
        f"vs_static2={int(better)} beats={int(beats)}")


if __name__ == "__main__":
    main()
