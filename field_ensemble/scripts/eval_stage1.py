"""Оценка stage 1: CNN vs эталон haklolarp (median_x3+spatial_z |Z|≥3.5) при равном FAR.
FAR_ref и пороги — только по val; цифры — на test. Результат: artifacts/eval/stage1_<ver>.json"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dts import data, metrics as M

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="artifacts/cache/v8"); ap.add_argument("--labels", default="data/synth_meta/v8/labels.csv")
ap.add_argument("--sequences", default="data/synth_meta/v8/sequences.csv")
ap.add_argument("--methods", default="unet,tcn,dilated,matched_filter,pixel_lgbm")
ap.add_argument("--out", default=None); ap.add_argument("--fit-cache", default=None, help="кэш, по val которого калибруются пороги (по умолчанию тот же)")
ap.add_argument("--thresholds-from", default=None, help="json со старыми порогами (для оценки на новой версии синтетики без перекалибровки)")
ap.add_argument("--splits", default="val,test", help="какие сплиты грузить; без val нужен --thresholds-from")
ap.add_argument("--no-base", action="store_true", help="не считать эталон |Z| в этом прогоне")
a = ap.parse_args()
ver = Path(a.cache).name
out = Path(a.out or f"artifacts/eval/stage1_{ver}.json")
labels = pd.read_csv(a.labels); seqs = pd.read_csv(a.sequences).set_index("seq_id")
METH = [m for m in a.methods.split(",") if m]
SPLITS = [s.strip() for s in a.splits.split(",") if s.strip()]


def load_split(split):
    res = {}
    for p in data.list_cache(a.cache, split):
        z = np.load(p)
        d = dict(ev=z["ev"], nu=z["nu"], valid=z["valid"], x=z["x"], t=z["t"])
        if not a.no_base:
            d["bz"] = z["base_z"].astype(np.float32); d["bzone"] = z["base_zone"]
        for m in METH:
            q = np.load(Path("artifacts/pred") / ver / m / split / p.name)
            d["s_" + m] = q["s"].astype(np.float32); d["h_" + m] = q["h"].astype(np.float32)
        res[p.stem] = d
    return res

t0 = time.time()
val = load_split("val") if "val" in SPLITS else None
test = load_split("test") if "test" in SPLITS else None
print("loaded", SPLITS, round(time.time() - t0), flush=True)

def bg(d):
    return d["valid"] & (d["ev"] == 0) & (d["nu"] == 0)

methods = {m: (lambda d, m=m: (d["s_" + m], d["h_" + m])) for m in METH}
if not a.no_base:
    methods["base_median_x3_spatial_z"] = lambda d: (np.abs(d["bz"]), d["bz"])
if a.thresholds_from:
    prev = json.load(open(a.thresholds_from)); far_ref = prev["far_ref"]
    missing = [m for m in methods if m not in prev["thresholds"]]
    if missing:
        raise SystemExit(f"нет порога в {a.thresholds_from}: {missing}")
    thr = {m: prev["thresholds"][m] for m in methods}
elif val is None:
    raise SystemExit("без val нужен --thresholds-from: порог на этом кэше не подбирается")
else:
    far_ref = float(np.mean(np.concatenate([(np.abs(d["bz"][bg(d)]) >= 3.5) for d in val.values()])))
    thr = {}
    for m, f in methods.items():
        bs = np.concatenate([f(d)[0][bg(d)] for d in val.values()])
        thr[m] = {"far_ref": M.far_threshold(bs, far_ref), "far_1e-3": M.far_threshold(bs, 1e-3), "far_1e-4": M.far_threshold(bs, 1e-4), "far_1e-2": M.far_threshold(bs, 1e-2)}
        # гистерезис: зерно при FAR 1e-3, рост связной области до порога FAR 1e-2 (как apply_hysteresis_threshold)
        thr[m]["hyst_1e-3_1e-2"] = [thr[m]["far_1e-3"], thr[m]["far_1e-2"]]
    if "base_median_x3_spatial_z" in thr:
        thr["base_median_x3_spatial_z"]["frozen_3.5"] = 3.5
print("FAR_ref", far_ref, thr, flush=True)


def evaluate(split_data, m, th):
    f = methods[m]; tot = np.zeros(3); tot_ni = np.zeros(3); hot = np.zeros(3); cold = np.zeros(3); per_seq = {}; far_n = far_d = nu_n = nu_d = 0
    ev_rows = []
    for sid, d in split_data.items():
        s, h = f(d)
        if isinstance(th, (list, tuple)):
            from scipy import ndimage as _nd
            lab, _ = _nd.label((s >= th[1]) & d["valid"]); keep = np.unique(lab[(s >= th[0]) & d["valid"]]); keep = keep[keep > 0]
            pred = np.isin(lab, keep)
        else:
            pred = (s >= th) & d["valid"]
        truth = d["ev"] > 0
        c = M.confusion(pred, truth, d["valid"]); tot += c; per_seq[sid] = c
        tot_ni += M.confusion(pred, truth, d["valid"] & ~((d["nu"] > 0) & ~truth))
        hot += M.confusion(pred & (h > 0), (d["ev"] >= 1) & (d["ev"] <= 4), d["valid"])
        cold += M.confusion(pred & (h <= 0), d["ev"] >= 5, d["valid"])
        b = bg(d); far_n += (pred & b).sum(); far_d += b.sum()
        nm = d["valid"] & (d["nu"] > 0) & ~truth; nu_n += (pred & nm).sum(); nu_d += nm.sum()
        for r in M.label_boxes(labels, sid).itertuples():
            if r.kind != "event": continue
            px = M.event_pixels(r, d["x"], d["t"], d["ev"])
            if px.sum() == 0: continue
            frac = float(pred[px].mean())
            ev_rows.append(dict(seq_id=sid, label=r.label, amp=float(r.amp_obs_C), amp_bin=M.amp_bin(float(r.amp_obs_C)),
                                template=seqs.loc[sid, "template"], difficulty=r.difficulty, frac=frac, hit=frac >= 0.3,
                                tp=int((pred & px).sum()), fn=int((~pred & px).sum())))
    E = pd.DataFrame(ev_rows)
    res = dict(threshold=th if isinstance(th, list) else float(th), far=float(far_n / max(far_d, 1)), far_ratio=float(far_n / max(far_d, 1) / far_ref),
               nuisance_alarm_frac=float(nu_n / max(nu_d, 1)), **M.prf(*tot),
               hot=M.prf(*hot), cold=M.prf(*cold), nuisance_ignored=M.prf(*tot_ni), iou_ci95=M.bootstrap_ci(per_seq, sorted(per_seq)))
    if len(E):
        def agg(g):
            return dict(n=int(len(g)), recall=float(g.hit.mean()), pixel_recall=float(g.tp.sum() / max(g.tp.sum() + g.fn.sum(), 1)))
        res["events_all"] = agg(E)
        res["by_amp"] = {k: agg(g) for k, g in E.groupby("amp_bin")}
        res["by_label"] = {k: agg(g) for k, g in E.groupby("label")}
        res["by_template"] = {k: agg(g) for k, g in E.groupby("template")}
        # weak-event IoU: пиксельный IoU в окрестности слабых событий
        res["weak_events"] = agg(E[E.amp <= 2])
    # per-template pixel IoU
    tm = {}
    for sid, c in per_seq.items():
        tm.setdefault(seqs.loc[sid, "template"], np.zeros(3)); tm[seqs.loc[sid, "template"]] += c
    res["iou_by_template"] = {k: M.prf(*v)["iou"] for k, v in tm.items()}
    return res

report = dict(version=ver, far_ref=far_ref, thresholds=thr, test={}, val={}, threshold_free={}, test_far_matched={},
              splits=SPLITS, thresholds_from=a.thresholds_from)
from sklearn.metrics import average_precision_score
from dts import events as EV
rng = np.random.default_rng(0)
# Порог, подобранный по фону test, — только диагностика. На переносе (порог заморожен, val этого кэша не читается) его не считаем.
do_refit = test is not None and not (a.thresholds_from and val is None) and not a.no_base
if test is not None:
    samp = {}
    for sid, d in test.items():
        idx = np.flatnonzero(d["valid"]); samp[sid] = rng.choice(idx, min(len(idx), 40000), replace=False)
    for m, f in methods.items():
        ys = np.concatenate([(test[s]["ev"].ravel()[samp[s]] > 0) for s in test]); ss = np.concatenate([f(test[s])[0].ravel()[samp[s]] for s in test])
        wk = []
        for sid, d in test.items():
            for r in M.label_boxes(labels, sid).itertuples():
                if r.kind == "event" and float(r.amp_obs_C) <= 2:
                    px = M.event_pixels(r, d["x"], d["t"], d["ev"])
                    if px.any(): wk.append(float(np.max(f(d)[0][px])))
        report["threshold_free"][m] = dict(pixel_average_precision=float(average_precision_score(ys, ss)),
                                           weak_event_max_score_median=float(np.median(wk)) if wk else None)
if do_refit:
    tb = np.concatenate([(np.abs(d["bz"][bg(d)]) >= 3.5) for d in test.values()]).mean()
    report["base_test_far"] = float(tb)
    for m, f in methods.items():
        bs = np.concatenate([f(d)[0][bg(d)] for d in test.values()])
        report["test_far_matched"][m] = evaluate(test, m, M.far_threshold(bs, tb))
elif a.thresholds_from and val is None:
    report["refit"] = "skipped: thresholds frozen from v8, not fit on this cache"


def false_zones(split_data, m, th):
    f = methods[m]; nz = nf = 0; kmh = 0.0
    for d in split_data.values():
        s, h = f(d); zs = EV.zones_from_prob(s, h, d["valid"], d["x"], d["t"], th)
        for z in zs:
            sl = (slice(z["t0"], z["t1"]), slice(z["x0"], z["x1"])); comp = (s[sl] >= th) & d["valid"][sl]
            nz += 1; nf += int((d["ev"][sl][comp] > 0).mean() < 0.1) if comp.any() else 1
        kmh += (d["x"][-1] - d["x"][0]) / 1000 * (d["t"][-1] - d["t"][0]) / 3600
    return dict(n_zones=nz, n_false=nf, false_zones_per_km_h=nf / max(kmh, 1e-9), false_frac=nf / max(nz, 1))
for m in methods:
    for op, th in thr[m].items():
        if test is not None:
            report["test"][f"{m}@{op}"] = evaluate(test, m, th)
            if not isinstance(th, list): report["test"][f"{m}@{op}"]["zones"] = false_zones(test, m, th)
        if val is not None:
            report["val"][f"{m}@{op}"] = {k: v for k, v in evaluate(val, m, th).items() if k in ("far", "iou", "f1", "far_ratio")}
# официальный выход пайплайна haklolarp (зоны после фильтров) — фиксированная рабочая точка
def zone_eval(split_data):
    tot = np.zeros(3); far_n = far_d = 0
    for d in split_data.values():
        pred = (d["bzone"] > 0) & d["valid"]; tot += M.confusion(pred, d["ev"] > 0, d["valid"])
        b = bg(d); far_n += (pred & b).sum(); far_d += b.sum()
    return dict(far=float(far_n / far_d), **M.prf(*tot))
if test is not None and test and "bzone" in next(iter(test.values())):
    report["test"]["base_pipeline_zones"] = zone_eval(test)
out.parent.mkdir(parents=True, exist_ok=True); json.dump(report, open(out, "w"), indent=1, ensure_ascii=False)
for k, v in report["test"].items():
    print(k, {kk: (round(vv, 4) if isinstance(vv, float) else None) for kk, vv in v.items() if kk in ("far", "precision", "recall", "f1", "iou")},
          "weak", v.get("weak_events"), flush=True)
