"""Многомасштабный прогон победителя. Порог только по v8 val. Test и real — отчёт.

Пишет artifacts/eval/s2_ms_v8.json, s2_ms_v10.json, s2_ms_real.json.
Исходный масштаб синтетики берётся из уже посчитанных карт ансамбля.
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dts import data, events as E, grid, metrics as M
from experiments.multiscale.multiscale import (
    DIAG_SCALES, EV_BINS, FUSIONS, NB, add_hist, ensemble_score, hist_ge, hist_threshold,
    multiscale_maps, prf_from_hists,
)

def log(msg):
    print(msg, flush=True)


def event(msg):
    line = time.strftime("%H:%M", time.gmtime()) + " " + msg + "\n"
    with open("/workspace/grok_logs/_events.log", "a") as f:
        f.write(line)
    log(line.strip())


def load_members(threads):
    import torch
    torch.set_num_threads(threads)
    from dts import models
    art = Path(os.environ.get("DTS_MODELS", ROOT / "models_release"))
    th = json.loads((art / "stage1_thresholds.json").read_text())
    members = []
    for m in th["members"]:
        f = art / m["file"]
        if m["kind"] == "pixel_lgbm":
            import joblib
            mdl = joblib.load(f)
            if hasattr(mdl, "set_params"):
                mdl.set_params(n_jobs=threads)
            members.append(("pixel_lgbm", mdl))
        else:
            net = models.build(m["kind"], m.get("width"))
            net.load_state_dict(torch.load(f, map_location="cpu"))
            net.eval()
            members.append(("net", net))
    return members, th


def empty_acc():
    keys = ("bg", "pos", "neg", "hot_truth", "hot_h", "hot_both", "cold_truth", "cold_h", "cold_both")
    return {k: np.zeros(NB, np.int64) for k in keys}, []


def summarize(variant_acc, thr):
    from experiments.multiscale.multiscale import hist_ge as ge
    acc, events = variant_acc
    pixel = prf_from_hists(acc["pos"], acc["neg"], thr)
    far_n = ge(acc["bg"], thr)
    far_d = int(acc["bg"].sum())
    hot_tp = ge(acc["hot_both"], thr)
    hot_fp = ge(acc["hot_h"], thr) - hot_tp
    hot_fn = int(acc["hot_truth"].sum()) - hot_tp
    cold_tp = ge(acc["cold_both"], thr)
    cold_fp = ge(acc["cold_h"], thr) - cold_tp
    cold_fn = int(acc["cold_truth"].sum()) - cold_tp
    rows = []
    for e in events:
        ge_n = ge(e["hist"], thr)
        n = int(e["n"])
        frac = ge_n / max(n, 1)
        rows.append(dict(label=e["label"], amp=e["amp"], template=e["template"],
                         frac=frac, hit=frac >= 0.3, tp=ge_n, fn=n - ge_n))
    Edf = pd.DataFrame(rows)
    def agg(g):
        if len(g) == 0:
            return dict(n=0, recall=None, pixel_recall=None)
        return dict(n=int(len(g)), recall=float(g.hit.mean()),
                    pixel_recall=float(g.tp.sum() / max(int(g.tp.sum() + g.fn.sum()), 1)))
    out = dict(threshold=float(thr), far=float(far_n / max(far_d, 1)), **{k: pixel[k] for k in ("precision", "recall", "f1", "iou")},
               hot=M.prf(hot_tp, hot_fp, hot_fn), cold=M.prf(cold_tp, cold_fp, cold_fn))
    if len(Edf):
        out["events_all"] = agg(Edf)
        out["weak_events"] = agg(Edf[Edf.amp <= 2])
        out["by_label"] = {k: agg(g) for k, g in Edf.groupby("label")}
        out["by_template"] = {k: agg(g) for k, g in Edf.groupby("template")}
    return out


def variant_names():
    return list(FUSIONS) + ["max_" + s for s in DIAG_SCALES]


def run_split(members, cache, pred_root, split, labels, seqs):
    """Один проход: гистограммы всех слияний. Карты на диск не пишутся."""
    names = variant_names()
    store = {n: empty_acc() for n in names}
    paths = data.list_cache(cache, split)
    log(f"split {cache} {split} n={len(paths)}")
    for i, p in enumerate(paths, 1):
        t0 = time.time()
        z = np.load(p)
        T = z["T"].astype(np.float32)
        valid = np.asarray(z["valid"], bool)
        ev = np.asarray(z["ev"])
        nu = np.asarray(z["nu"])
        x = np.asarray(z["x"], np.float64)
        t = np.asarray(z["t"], np.float64)
        dx = float(np.median(np.diff(x))) if len(x) > 1 else 0.25
        qp = Path(pred_root) / split / p.name
        if not qp.exists():
            raise SystemExit(f"нет исходной карты {qp}")
        q = np.load(qp)
        native = (q["s"].astype(np.float32), q["h"].astype(np.float32))
        maps = multiscale_maps(members, T, valid, dx, native=native)
        boxes = M.label_boxes(labels, p.stem)
        box_px = []
        for r in boxes.itertuples():
            if r.kind != "event":
                continue
            px = M.event_pixels(r, x, t, ev)
            if not px.any():
                continue
            box_px.append((r, px))
        template = str(seqs.loc[p.stem, "template"]) if p.stem in seqs.index else ""
        for name in names:
            s, h = maps[name]
            acc, _ev = store[name]
            v = valid
            truth = ev > 0
            add_hist(acc["bg"], s[v & ~truth & (nu == 0)])
            add_hist(acc["pos"], s[v & truth])
            add_hist(acc["neg"], s[v & ~truth])
            hot_t = v & (ev >= 1) & (ev <= 4)
            cold_t = v & (ev >= 5)
            hp = h > 0
            add_hist(acc["hot_truth"], s[hot_t])
            add_hist(acc["hot_h"], s[v & hp])
            add_hist(acc["hot_both"], s[hot_t & hp])
            add_hist(acc["cold_truth"], s[cold_t])
            add_hist(acc["cold_h"], s[v & ~hp])
            add_hist(acc["cold_both"], s[cold_t & ~hp])
            for r, px in box_px:
                hist = np.zeros(EV_BINS, np.int64)
                add_hist(hist, s[px])
                store[name][1].append(dict(label=str(r.label), amp=float(r.amp_obs_C), template=template,
                                           n=int(px.sum()), hist=hist))
        log(f"  {p.stem} {i}/{len(paths)} {round(time.time()-t0,1)}s")
        del T, maps, q, z
    return store


def thresholds_from(store, far=1e-3):
    return {name: hist_threshold(acc["bg"], far) for name, (acc, _e) in store.items()}


def report_split(store, thr):
    return {name: summarize((acc, ev), thr[name]) for name, (acc, ev) in store.items()}


def choose(val_report):
    """Максимум val IoU среди native/max/mean/noise. Ничья — более ранний в FUSIONS."""
    best = None
    for name in FUSIONS:
        iou = float(val_report[name]["iou"])
        if best is None or iou > best[0] + 1e-12:
            best = (iou, name)
    return best[1]


def load_eventnet():
    import torch
    from dts import models
    art = Path(os.environ.get("DTS_MODELS", ROOT / "models_release"))
    ck = torch.load(art / "eventnet.pt", map_location="cpu", weights_only=False)
    mu = np.asarray(ck["mu"], np.float32)
    sd = np.asarray(ck["sd"], np.float32)
    enet = models.EventNet(len(E.CLASSES), len(mu), cin=int(ck.get("cin", 3)), late_fusion=bool(ck.get("late_fusion", False)))
    enet.load_state_dict(ck["state"])
    enet.eval()
    lgbm, lgbm_kind = None, "legacy"
    try:
        import joblib
        obj = joblib.load(art / "lgbm.joblib")
        if isinstance(obj, dict) and "model" in obj:
            lgbm = obj["seeds"] if obj.get("seeds") else obj["model"]
            lgbm_kind = obj.get("kind", "full")
        else:
            lgbm = obj
    except Exception:
        lgbm = None
    return enet, mu, sd, ck, lgbm, lgbm_kind


def eval_real(members, fusion, thr, real_root, cards_paths):
    from dts import features, real
    from dts.infer import predict_zone_probs
    from dts.domain import presence_tally
    enet, mu, sd, ck, lgbm, lgbm_kind = load_eventnet()
    cards = []
    for c in cards_paths:
        cards += [json.loads(line) for line in open(c) if line.strip()]
    files = {}
    card_rows = []
    for i in range(1, 7):
        path = Path(real_root) / f"{i}.csv"
        t0 = time.time()
        rec = real.load_csv(path)
        p = grid.plan(rec["x"], rec["t"])
        Tc, v = grid.resample(rec["T"], rec["valid"], p, "mean")
        del rec["T"]
        maps = multiscale_maps(members, Tc, v, p["dx"], native=None)
        s, h = maps[fusion]
        zones = E.zones_from_prob(s, h, v, p["x"], p["t"], thr)
        X, aux = features.make_channels(Tc, v, dx=p["dx"])
        probs = predict_zone_probs(
            enet, mu, sd, X, aux["sigma_x"], p["x"], p["t"], zones,
            lgbm=lgbm, lgbm_kind=lgbm_kind,
            temperature=float(ck.get("temperature", 1.0)),
            lgbm_temperature=float(ck.get("lgbm_temperature", 1.0)),
            cnn_weight=float(ck.get("cnn_weight", 0.5)),
            scalar_names=list(ck["scalar_names"]) if ck.get("scalar_names") else None,
            two_scale=bool(ck.get("two_scale", False)),
            stack_coef=None if ck.get("stack_coef") is None else np.asarray(ck["stack_coef"], np.float32),
            stack_intercept=None if ck.get("stack_intercept") is None else np.asarray(ck["stack_intercept"], np.float32),
        )
        for z, pr in zip(zones, probs):
            z["class"] = E.CLASSES[int(np.argmax(pr))]
            z["alarm"] = z["class"] not in ("HVAC_Diurnal", "Sensor_Artifact")
        n_alarm = sum(1 for z in zones if z["alarm"])
        files[rec["name"]] = dict(n_zones=len(zones), n_alarm=n_alarm, seconds=round(time.time() - t0, 1))
        name = rec["name"]
        for c in cards:
            if str(c.get("record")) != name:
                continue
            E0 = pd.Timestamp("1970-01-01", tz="UTC")
            ts = [(pd.Timestamp(s).tz_convert("UTC") if pd.Timestamp(s).tzinfo else pd.Timestamp(s, tz="UTC")) for s in c["time"]]
            ta, tb = [(ss - E0).total_seconds() for ss in ts]
            xa, xb = c["meters"]
            ix = (p["x"] >= xa) & (p["x"] <= xb)
            it = (p["t"] >= ta) & (p["t"] <= tb)
            if not ix.any() or not it.any():
                card_rows.append(dict(id=c.get("id"), record=name, presence=c.get("presence"), status="box_outside_record",
                                       hit_alarm=False, hit_zone=False))
                continue
            overl = [z for z in zones if z["end_m"] >= xa and z["start_m"] <= xb and z["t_end_s"] >= ta and z["t_start_s"] <= tb]
            alarm = [z for z in overl if z["alarm"]]
            card_rows.append(dict(id=c.get("id"), record=name, presence=c.get("presence"), status="ok",
                                   hit_alarm=bool(alarm), hit_zone=bool(overl), n_alarm=len(alarm), n_zones=len(overl)))
        log(f"real {name} zones {len(zones)} alarm {n_alarm} {files[rec['name']]['seconds']}s")
        del maps, Tc, X
    return dict(files=files, cards=card_rows,
                presence_alarm=presence_tally(card_rows, "hit_alarm"),
                presence_zone=presence_tally(card_rows, "hit_zone"),
                n_zones=int(sum(v["n_zones"] for v in files.values())),
                n_alarm=int(sum(v["n_alarm"] for v in files.values())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--cache-v8", default="artifacts/cache/v8")
    ap.add_argument("--cache-v10", default="artifacts/cache/v10")
    ap.add_argument("--pred-v8", default="artifacts/pred/v8/ens_u2ft_tcn_lgbm")
    ap.add_argument("--pred-v10", default="artifacts/pred/v10/ens_u2ft_tcn_lgbm")
    ap.add_argument("--labels-v8", default="data/synth_meta/v8/labels.csv")
    ap.add_argument("--seq-v8", default="data/synth_meta/v8/sequences.csv")
    ap.add_argument("--labels-v10", default="data/synth_meta/v10/labels.csv")
    ap.add_argument("--seq-v10", default="data/synth_meta/v10/sequences.csv")
    ap.add_argument("--real-root", default="/workspace/mgts/MGTS_data")
    ap.add_argument("--cards", nargs="*", default=["data/operator_cards/cards_batch1.jsonl", "data/operator_cards/cards_batch2.jsonl"])
    ap.add_argument("--out-dir", default="artifacts/eval")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    v8_path = out / "s2_ms_v8.json"
    v10_path = out / "s2_ms_v10.json"
    real_path = out / "s2_ms_real.json"
    members, th = load_members(a.threads)
    method = th.get("method", "")
    log(f"members {method} n={len(members)}")

    if v8_path.exists():
        v8 = json.loads(v8_path.read_text())
        log(f"reuse {v8_path} selected {v8['selected']}")
    else:
        labels = pd.read_csv(a.labels_v8)
        seqs = pd.read_csv(a.seq_v8).set_index("seq_id")
        val = run_split(members, a.cache_v8, a.pred_v8, "val", labels, seqs)
        thr = thresholds_from(val)
        val_rep = report_split(val, thr)
        selected = choose(val_rep)
        test = run_split(members, a.cache_v8, a.pred_v8, "test", labels, seqs)
        test_rep = report_split(test, thr)
        v8 = dict(method=method, far=1e-3, scales=["native", "t2", "t4", "x2", "x4"],
                  weights=dict(native=1, t2=2, t4=4, x2=2, x4=4),
                  fusions=list(FUSIONS), selected=selected, selection="v8 val IoU @ far_1e-3",
                  thresholds=thr, val=val_rep, test=test_rep)
        v8_path.write_text(json.dumps(v8, indent=1, ensure_ascii=False))
        event(f"s2 multiscale v8: selected {selected} val IoU {val_rep[selected]['iou']:.4f} "
              f"(native {val_rep['native']['iou']:.4f}) test IoU {test_rep[selected]['iou']:.4f}")
        del val, test

    if v10_path.exists():
        log(f"reuse {v10_path}")
    else:
        labels = pd.read_csv(a.labels_v10)
        seqs = pd.read_csv(a.seq_v10).set_index("seq_id")
        test = run_split(members, a.cache_v10, a.pred_v10, "test", labels, seqs)
        test_rep = report_split(test, v8["thresholds"])
        v10 = dict(method=method, selected=v8["selected"], thresholds_from="v8 val",
                   thresholds=v8["thresholds"], test=test_rep)
        v10_path.write_text(json.dumps(v10, indent=1, ensure_ascii=False))
        event(f"s2 multiscale v10 test {v8['selected']} IoU {test_rep[v8['selected']]['iou']:.4f} "
              f"native {test_rep['native']['iou']:.4f} thr frozen from v8")
        del test

    if real_path.exists():
        log(f"reuse {real_path}")
    else:
        real = eval_real(members, v8["selected"], v8["thresholds"][v8["selected"]], a.real_root, a.cards)
        real.update(method=method, fusion=v8["selected"], threshold=v8["thresholds"][v8["selected"]],
                    note="порог с v8 val, карточки только оценка, box15 не применяется")
        real_path.write_text(json.dumps(real, indent=1, ensure_ascii=False))
        pa = real["presence_alarm"]
        event(f"s2 multiscale real {v8['selected']}: zones {real['n_zones']} alarm {real['n_alarm']} "
              f"cards yes {pa.get('yes_hit_alarm')}/{pa.get('yes_n')} no {pa.get('no_hit_alarm')}/{pa.get('no_n')}")
    log("MULTISCALE_DONE")


if __name__ == "__main__":
    main()
