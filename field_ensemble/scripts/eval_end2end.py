"""Сквозная оценка test: карта детектора -> зоны -> класс этапа 2 -> сопоставление с метками.

Зона с IoU бокса < порога (по умолчанию 0.1) — ложная. macro-F1 считается по пулу TP/FP/FN
семи классов: верная пара даёт TP, пара с чужим классом — FP предсказанного и FN истинного,
зона без пары — FP, метка без пары — FN.

event_recall — доля истинных событий (kind==event, без nuisance) с парной зоной.
alarm_precision — среди зон, которым этап 2 поставил тревогу (не HVAC_Diurnal и не
Sensor_Artifact, как в infer.Analyzer.run), доля тех, что совпали с истинным событием.
false_nuisance_frac — доля ложных зон, отсеянных как нетревоги.

Порог детектора берётся из artifacts/eval/stage1_<ver>.json (thresholds.<method>.<op>)
или из --thr. Пока json этапа 1 нет, передайте --thr.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dts import events as E, metrics as M, models
from dts.infer import predict_zone_probs


def resolve_threshold(stage1_json, method, op, thr):
    if thr is not None:
        return float(thr), "arg"
    path = Path(stage1_json)
    if not path.exists():
        raise SystemExit(f"нет порога: {path} не найден, передайте --thr")
    data = json.load(open(path))
    try:
        block = data["thresholds"][method]
    except KeyError as exc:
        keys = list(data.get("thresholds", {}))
        raise SystemExit(f"в {path} нет thresholds.{method}; есть {keys}") from exc
    if isinstance(block, dict):
        if op not in block:
            raise SystemExit(f"в {path} нет thresholds.{method}.{op}; есть {list(block)}")
        return float(block[op]), f"{path}:{method}.{op}"
    return float(block), f"{path}:{method}"


# Тревога совпадает с dts.infer.Analyzer.run.
NON_ALARM = ("HVAC_Diurnal", "Sensor_Artifact")


def gt_boxes(labels, seq_id, x, t):
    """Истинные боксы, обрезанные по сетке записи. Класс — таксономия этапа 2.

    kind отсутствует в старых csv — бокс считается событием, не nuisance.
    """
    boxes, cls, is_event = [], [], []
    x0, x1 = float(x[0]), float(x[-1])
    t0, t1 = float(t[0]), float(t[-1])
    for r in M.label_boxes(labels, seq_id).itertuples():
        if r.label not in E.SYNTH_TO_CLASS:
            continue
        xa, xb = max(float(r.dist_start_m), x0), min(float(r.dist_end_m), x1)
        ta, tb = max(float(r.ts), t0), min(float(r.te), t1)
        if xb <= xa or tb <= ta:
            continue
        boxes.append((xa, xb, ta, tb))
        cls.append(E.CLASSES.index(E.SYNTH_TO_CLASS[r.label]))
        is_event.append(str(getattr(r, "kind", "event")) == "event")
    return boxes, np.asarray(cls, np.int64), np.asarray(is_event, bool)


def zone_alarm_stats(pred_cls, is_event, matches, n_classes):
    """Счётчики одного кадра: ложные классы, recall событий, alarm precision.

    Совпадение с событием — жадная пара IoU, у которой метка имеет kind==event.
    Класс пары для alarm precision не требуется.
    """
    pred_cls = np.asarray(pred_cls, np.int64)
    is_event = np.asarray(is_event, bool)
    nuis = {E.CLASSES.index(c) for c in NON_ALARM}
    false_cls = np.zeros(n_classes, np.int64)
    matched_p = {}
    matched_g = set()
    for i, j, _ in matches:
        matched_p[int(i)] = int(j)
        matched_g.add(int(j))
    n_event = int(is_event.sum()) if len(is_event) else 0
    n_event_hit = int(sum(1 for j in matched_g if is_event[j]))
    n_alarm = n_alarm_hit = 0
    for i, c in enumerate(pred_cls):
        c = int(c)
        if i not in matched_p:
            false_cls[c] += 1
        if c not in nuis:
            n_alarm += 1
            j = matched_p.get(i)
            if j is not None and bool(is_event[j]):
                n_alarm_hit += 1
    return false_cls, n_event, n_event_hit, n_alarm, n_alarm_hit


def load_stage2(path, lgbm_path, use_lgbm):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    enet = models.EventNet(len(E.CLASSES), len(ck["mu"]), cin=int(ck.get("cin", 3)),
                           late_fusion=bool(ck.get("late_fusion", False)))
    enet.load_state_dict(ck["state"]); enet.eval()
    lgbm, kind = None, "legacy"
    if use_lgbm and lgbm_path and Path(lgbm_path).exists():
        import joblib
        obj = joblib.load(lgbm_path)
        if isinstance(obj, dict) and "model" in obj:
            kind = obj.get("kind", "full")
            lgbm = obj["seeds"] if obj.get("seeds") else obj["model"]
        else:
            lgbm = obj
    return enet, ck, lgbm, kind


def classify_zones(enet, ck, lgbm, kind, X, sigma, x, t, zones):
    return predict_zone_probs(
        enet, ck["mu"], ck["sd"], X, sigma, x, t, zones, lgbm=lgbm, lgbm_kind=kind,
        temperature=float(ck.get("temperature", 1.0)), lgbm_temperature=float(ck.get("lgbm_temperature", 1.0)),
        cnn_weight=float(ck.get("cnn_weight", 0.5)) if lgbm is not None else 1.0,
        scalar_names=ck.get("scalar_names"), two_scale=bool(ck.get("two_scale", False)),
        stack_coef=ck.get("stack_coef"), stack_intercept=ck.get("stack_intercept"))


def evaluate(cache, pred, split, labels, enet, ck, lgbm, kind, thr, iou=0.1, min_px=12):
    K = len(E.CLASSES)
    tp = np.zeros(K); fp = np.zeros(K); fn = np.zeros(K)
    false_cls = np.zeros(K, np.int64)
    n_zones = n_gt = n_matched = n_false = n_correct = 0
    n_event = n_event_hit = n_alarm = n_alarm_hit = 0
    per_seq = {}
    files = sorted(Path(cache, split).glob("seq*.npz"))
    if not files:
        raise SystemExit(f"нет кэша в {cache}/{split}")
    for p in files:
        q = Path(pred, split, p.name)
        if not q.exists():
            raise SystemExit(f"нет карты детектора {q}")
        z = np.load(p); d = np.load(q)
        zones = E.zones_from_prob(d["s"], d["h"], z["valid"], z["x"], z["t"], thr, min_px=min_px)
        boxes, gcls, is_event = gt_boxes(labels, p.stem, z["x"], z["t"])
        if zones:
            prob = np.asarray(classify_zones(enet, ck, lgbm, kind, z["X"], z["sigma"], z["x"], z["t"], zones))
            pcls = prob.argmax(1)
            pboxes = [(zz["start_m"], zz["end_m"], zz["t_start_s"], zz["t_end_s"]) for zz in zones]
        else:
            pcls = np.zeros(0, np.int64)
            pboxes = []
        matches, unmatched_p, _ = E.greedy_match(pboxes, boxes, thr=iou)
        ti, fi, ni, correct = E.e2e_counts(pcls, gcls, matches, K)
        fc, ne, nh, na, nah = zone_alarm_stats(pcls, is_event, matches, K)
        tp += ti; fp += fi; fn += ni
        false_cls += fc
        n_zones += len(pboxes); n_gt += len(boxes); n_matched += len(matches)
        n_false += len(unmatched_p); n_correct += correct
        n_event += ne; n_event_hit += nh; n_alarm += na; n_alarm_hit += nah
        per_seq[p.stem] = (ti, fi, ni, ne, nh, na, nah)
    macro, per = E.f1_from_counts(tp, fp, fn, E.CLASSES)
    nuisance = int(sum(false_cls[E.CLASSES.index(c)] for c in NON_ALARM))
    counts = {E.CLASSES[k]: int(false_cls[k]) for k in range(K)}
    frac = {name: (counts[name] / n_false if n_false else 0.0) for name in E.CLASSES}
    return dict(macro_f1=macro, per_class=per, n_seq=len(files), n_zones=int(n_zones), n_gt=int(n_gt),
                n_matched=int(n_matched), n_false_zones=int(n_false), n_correct=int(n_correct),
                false_zone_frac=float(n_false / n_zones) if n_zones else 0.0,
                matched_class_accuracy=float(n_correct / n_matched) if n_matched else 0.0,
                n_event=int(n_event), n_event_hit=int(n_event_hit),
                event_recall=float(n_event_hit / n_event) if n_event else 0.0,
                n_alarm=int(n_alarm), n_alarm_hit=int(n_alarm_hit),
                alarm_precision=float(n_alarm_hit / n_alarm) if n_alarm else 0.0,
                false_zone_class_counts=counts, false_zone_class_frac=frac,
                false_nuisance_count=nuisance,
                false_nuisance_frac=float(nuisance / n_false) if n_false else 0.0,
                _per_seq=per_seq)


def bootstrap_macro(per_seq, n=300, seed=0):
    rng = np.random.default_rng(seed)
    keys = list(per_seq)
    if not keys:
        return [0.0, 0.0]
    vals = []
    idx = np.arange(len(keys))
    for _ in range(n):
        pick = rng.choice(idx, len(idx))
        tp = np.zeros(len(E.CLASSES)); fp = np.zeros(len(E.CLASSES)); fn = np.zeros(len(E.CLASSES))
        for i in pick:
            row = per_seq[keys[i]]
            tp += row[0]; fp += row[1]; fn += row[2]
        vals.append(E.f1_from_counts(tp, fp, fn, E.CLASSES)[0])
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def bootstrap_ratio(per_seq, num_i, den_i, n=300, seed=0):
    rng = np.random.default_rng(seed)
    keys = list(per_seq)
    if not keys:
        return [0.0, 0.0]
    vals = []
    idx = np.arange(len(keys))
    for _ in range(n):
        num = den = 0.0
        for i in rng.choice(idx, len(idx)):
            row = per_seq[keys[i]]
            num += row[num_i]
            den += row[den_i]
        vals.append(num / den if den else 0.0)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main(argv=None):
    import torch
    torch.set_num_threads(1)
    ap = argparse.ArgumentParser(description="End-to-end macro-F1 и доля ложных зон на test")
    ap.add_argument("--pred", default="artifacts/pred/v8/matched_filter")
    ap.add_argument("--cache", default="artifacts/cache/v8")
    ap.add_argument("--split", default="test")
    ap.add_argument("--labels", default="data/synth_meta/v8/labels.csv")
    ap.add_argument("--stage1-json", default="artifacts/eval/stage1_v8.json")
    ap.add_argument("--method", default=None)
    ap.add_argument("--op", default="far_ref")
    ap.add_argument("--thr", type=float, default=None)
    ap.add_argument("--stage2", default="artifacts/stage2/eventnet.pt")
    ap.add_argument("--lgbm", default="artifacts/stage2/lgbm.joblib")
    ap.add_argument("--no-lgbm", action="store_true")
    ap.add_argument("--iou", type=float, default=0.1)
    ap.add_argument("--min-px", type=int, default=12)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    method = a.method or Path(a.pred).name
    thr, src = resolve_threshold(a.stage1_json, method, a.op, a.thr)
    if not Path(a.stage2).exists():
        raise SystemExit(f"нет модели этапа 2: {a.stage2}")
    labels_path = Path(a.labels)
    if not labels_path.exists():
        alt = Path("data/synth_meta") / Path(a.cache).name / "labels.csv"
        if alt.exists():
            print("labels fallback", alt, flush=True)
            labels_path = alt
        else:
            raise SystemExit(f"нет меток: {a.labels}")
    import pandas as pd
    labels = pd.read_csv(labels_path)
    t0 = time.time()
    enet, ck, lgbm, kind = load_stage2(a.stage2, a.lgbm, not a.no_lgbm)
    rep = evaluate(a.cache, a.pred, a.split, labels, enet, ck, lgbm, kind, thr, iou=a.iou, min_px=a.min_px)
    per_seq = rep.pop("_per_seq")
    rep["macro_f1_ci95"] = bootstrap_macro(per_seq)
    rep["event_recall_ci95"] = bootstrap_ratio(per_seq, 4, 3)
    rep["alarm_precision_ci95"] = bootstrap_ratio(per_seq, 6, 5)
    rep.update(method=method, split=a.split, op=a.op, threshold=thr, threshold_source=src, iou=a.iou,
               stage2=str(a.stage2), lgbm=None if a.no_lgbm else str(a.lgbm), seconds=round(time.time() - t0, 1))
    text = json.dumps(rep, indent=1, ensure_ascii=False)
    print(text)
    out = Path(a.out) if a.out else Path("artifacts/stage2_scratch") / f"e2e_{method}_{a.split}_{a.op}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print("wrote", out, flush=True)
    return rep


if __name__ == "__main__":
    main()
