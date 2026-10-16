"""Раздельные пороги hot/cold для карт ens_u2ft_tcn_lgbm. Без обучения.

На v8 val перебирается доля бюджета FAR 1e-3, которая достаётся горячей
полярности (h > 0). Остаток бюджета — холодная полярность (h <= 0).
Пара (thr_hot, thr_cold) с наибольшим val IoU замораживается для v8 test,
v10 test и v11 test. Реальные CSV не читаются.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dts import data, metrics as M

ROOT = Path(__file__).resolve().parents[1]
METHOD = "ens_u2ft_tcn_lgbm"
FAR = 1e-3
# Горячий хвост фона для FAR 1e-3 лежит около 0.86. Холодный балл ниже,
# поэтому холодный фон хранится целиком, горячий — от этого порога.
HOT_GATE = 0.25
ALPHAS = np.round(np.linspace(0.0, 1.0, 41), 5)
WINNER_JSON = ROOT / "artifacts/eval/r7_ens_u2ft_tcn_lgbm.json"
WINNER_MARGIN = 0.003


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


def load_pair(ver, split, name):
    cache = np.load(ROOT / "artifacts/cache" / ver / split / name)
    pred = np.load(ROOT / "artifacts/pred" / ver / METHOD / split / name)
    s = np.asarray(pred["s"], np.float32)
    h = np.asarray(pred["h"], np.float32)
    if s.shape != cache["ev"].shape or h.shape != s.shape:
        raise RuntimeError(f"shape {ver} {split} {name}")
    return cache, s, h


def alarm_budget(n, far=FAR):
    """Сколько фоновых клеток должно пройти порог при том же правиле, что far_threshold."""
    if n <= 0:
        return 0
    k = int(np.ceil((1.0 - far) * n)) - 1
    k = int(np.clip(k, 0, n - 1))
    return int(n - k - 1)


def thr_for_alarms(scores, n_low, n_total, n_alarm):
    """Порог, при котором около n_alarm хранимых баллов строго выше s[k].

    scores — баллы >= gate, n_low — сколько баллов фона этой полярности ниже gate.
    None, если квантиль попала в отброшенный низ.
    """
    scores = np.asarray(scores, np.float32)
    if n_alarm <= 0:
        top = float(scores.max()) if scores.size else 0.0
        return top + 1.0
    if n_alarm >= n_total:
        return -1.0
    k = n_total - int(n_alarm) - 1
    if k < n_low:
        return None
    kt = k - n_low
    if kt >= scores.size:
        return None
    part = np.partition(scores, int(kt))
    return float(part[int(kt)]) + 1e-12


def n_ge(scores, thr):
    if scores.size == 0 or thr > 1e8:
        return 0
    return int(np.count_nonzero(scores >= thr))


def collect_val():
    """Хвосты фона и все пиксели событий/помех на v8 val. Для сетки порогов."""
    hot_bg, cold_bg = [], []
    pos_hot, pos_cold = [], []
    nu_hot, nu_cold = [], []
    hot_tp, cold_tp = [], []
    hot_fp_ev, cold_fp_ev = [], []
    n_hot_low = n_bg = n_hot_bg = n_cold_bg = 0
    n_pos = n_hot_truth = n_cold_truth = 0
    paths = data.list_cache(ROOT / "artifacts/cache/v8", "val")
    t0 = time.time()
    for i, p in enumerate(paths):
        z, s, h = load_pair("v8", "val", p.name)
        valid = np.asarray(z["valid"], bool)
        ev = z["ev"]
        nu = z["nu"]
        hot = h > 0
        bg = valid & (ev == 0) & (nu == 0)
        nui = valid & (ev == 0) & (nu > 0)
        pos = valid & (ev > 0)
        hot_t = valid & (ev >= 1) & (ev <= 4)
        cold_t = valid & (ev >= 5)
        n_bg += int(bg.sum())
        n_hot_bg += int((bg & hot).sum())
        n_cold_bg += int((bg & ~hot).sum())
        n_pos += int(pos.sum())
        n_hot_truth += int(hot_t.sum())
        n_cold_truth += int(cold_t.sum())
        hb = s[bg & hot]
        keep = hb >= HOT_GATE
        n_hot_low += int((~keep).sum())
        if keep.any():
            hot_bg.append(hb[keep].astype(np.float32, copy=False))
        cb = s[bg & ~hot]
        if cb.size:
            cold_bg.append(cb.astype(np.float32, copy=False))
        if (pos & hot).any():
            pos_hot.append(s[pos & hot])
        if (pos & ~hot).any():
            pos_cold.append(s[pos & ~hot])
        if (nui & hot).any():
            nu_hot.append(s[nui & hot])
        if (nui & ~hot).any():
            nu_cold.append(s[nui & ~hot])
        if (hot_t & hot).any():
            hot_tp.append(s[hot_t & hot])
        # горячий FP среди чужих событий (не фон): ev>=5 и h>0
        if ((ev >= 5) & valid & hot).any():
            hot_fp_ev.append(s[(ev >= 5) & valid & hot])
        if (cold_t & ~hot).any():
            cold_tp.append(s[cold_t & ~hot])
        if (hot_t & ~hot).any():
            cold_fp_ev.append(s[hot_t & ~hot])
        if i % 10 == 0:
            print(f"collect {i} {p.stem} {time.time() - t0:.0f}s", flush=True)
    bag = dict(
        hot_bg=_cat(hot_bg), cold_bg=_cat(cold_bg),
        pos_hot=_cat(pos_hot), pos_cold=_cat(pos_cold),
        nu_hot=_cat(nu_hot), nu_cold=_cat(nu_cold),
        hot_tp=_cat(hot_tp), cold_tp=_cat(cold_tp),
        hot_fp_ev=_cat(hot_fp_ev), cold_fp_ev=_cat(cold_fp_ev),
        n_hot_low=int(n_hot_low), n_bg=int(n_bg),
        n_hot_bg=int(n_hot_bg), n_cold_bg=int(n_cold_bg),
        n_pos=int(n_pos), n_hot_truth=int(n_hot_truth), n_cold_truth=int(n_cold_truth),
    )
    print(
        f"bag bg {bag['n_bg']} hot {bag['n_hot_bg']} cold {bag['n_cold_bg']} "
        f"hot_tail {bag['hot_bg'].size} cold_all {bag['cold_bg'].size} "
        f"pos {bag['n_pos']} nu {bag['nu_hot'].size + bag['nu_cold'].size}",
        flush=True,
    )
    return bag


def _cat(chunks):
    if not chunks:
        return np.empty(0, np.float32)
    if len(chunks) == 1:
        return np.asarray(chunks[0], np.float32)
    return np.concatenate([np.asarray(c, np.float32) for c in chunks])


def iou_of(tp, fp, fn):
    return float(tp / max(tp + fp + fn, 1))


def score_pair(bag, thr_hot, thr_cold):
    tp = n_ge(bag["pos_hot"], thr_hot) + n_ge(bag["pos_cold"], thr_cold)
    fp_bg = n_ge(bag["hot_bg"], thr_hot) + n_ge(bag["cold_bg"], thr_cold)
    # низ горячего фона не хранится: если порог ниже gate, эти клетки тоже сработают
    if thr_hot <= HOT_GATE:
        fp_bg += bag["n_hot_low"]
    fp_nu = n_ge(bag["nu_hot"], thr_hot) + n_ge(bag["nu_cold"], thr_cold)
    fp = fp_bg + fp_nu
    fn = bag["n_pos"] - tp
    far = float(fp_bg / max(bag["n_bg"], 1))
    hot_tp = n_ge(bag["hot_tp"], thr_hot)
    hot_fp = n_ge(bag["hot_bg"], thr_hot) + n_ge(bag["nu_hot"], thr_hot) + n_ge(bag["hot_fp_ev"], thr_hot)
    if thr_hot <= HOT_GATE:
        hot_fp += bag["n_hot_low"]
    hot_fn = bag["n_hot_truth"] - hot_tp
    cold_tp = n_ge(bag["cold_tp"], thr_cold)
    cold_fp = n_ge(bag["cold_bg"], thr_cold) + n_ge(bag["nu_cold"], thr_cold) + n_ge(bag["cold_fp_ev"], thr_cold)
    cold_fn = bag["n_cold_truth"] - cold_tp
    return dict(
        thr_hot=float(thr_hot), thr_cold=float(thr_cold),
        iou=iou_of(tp, fp, fn), f1=M.prf(tp, fp, fn)["f1"],
        precision=M.prf(tp, fp, fn)["precision"], recall=M.prf(tp, fp, fn)["recall"],
        far=far, tp=int(tp), fp=int(fp), fn=int(fn),
        hot_iou=iou_of(hot_tp, hot_fp, hot_fn), cold_iou=iou_of(cold_tp, cold_fp, cold_fn),
    )


def single_threshold(bag):
    """Общий порог по смешанному фону, как metrics.far_threshold."""
    n = bag["n_bg"]
    k = int(np.ceil((1.0 - FAR) * n)) - 1
    k = int(np.clip(k, 0, n - 1))
    # отброшенный низ горячего фона идёт раньше любых холодных, которые мы храним целиком,
    # только если холодные баллы тоже могут быть ниже gate. Холод хранится целиком,
    # горячий низ — отдельным счётчиком. Собираем пул >= gate.
    hot = bag["hot_bg"]
    cold_hi = bag["cold_bg"][bag["cold_bg"] >= HOT_GATE]
    n_low = bag["n_hot_low"] + int(np.count_nonzero(bag["cold_bg"] < HOT_GATE))
    tail = _cat([hot, cold_hi])
    if k < n_low:
        raise RuntimeError(f"общий порог ниже gate {HOT_GATE}: k={k} n_low={n_low}")
    kt = k - n_low
    part = np.partition(tail, int(kt))
    return float(part[int(kt)]) + 1e-12


def build_grid(bag):
    k_budget = alarm_budget(bag["n_bg"], FAR)
    rows = []
    for alpha in ALPHAS:
        k_hot = int(round(float(alpha) * k_budget))
        k_cold = int(k_budget - k_hot)
        thr_h = thr_for_alarms(bag["hot_bg"], bag["n_hot_low"], bag["n_hot_bg"], k_hot)
        thr_c = thr_for_alarms(bag["cold_bg"], 0, bag["n_cold_bg"], k_cold)
        if thr_h is None or thr_c is None:
            raise RuntimeError(f"порог ушёл ниже сохранённого хвоста, alpha={alpha}")
        row = score_pair(bag, thr_h, thr_c)
        row["alpha_hot"] = float(alpha)
        row["k_hot"] = int(k_hot)
        row["k_cold"] = int(k_cold)
        row["k_budget"] = int(k_budget)
        rows.append(row)
        print(
            f"alpha {alpha:.2f} thr {thr_h:.4f}/{thr_c:.4f} iou {row['iou']:.4f} "
            f"far {row['far']:.6f} hot {row['hot_iou']:.4f} cold {row['cold_iou']:.4f}",
            flush=True,
        )
    return rows


def select(rows):
    """Максимум val IoU. Ничья: ближе к FAR 1e-3, затем больший F1, затем alpha ближе к 0.5."""
    def key(r):
        return (r["iou"], -abs(r["far"] - FAR), r["f1"], -abs(r["alpha_hot"] - 0.5))
    return max(rows, key=key)


def evaluate_split(ver, split, thr_hot, thr_cold, thr_single, labels, seqs):
    paths = data.list_cache(ROOT / "artifacts/cache" / ver, split)
    by_seq = {sid: g for sid, g in labels.groupby("seq_id")} if labels is not None else {}
    acc = {name: _empty() for name in ("pol", "single")}
    t0 = time.time()
    for i, p in enumerate(paths):
        z, s, h = load_pair(ver, split, p.name)
        valid = np.asarray(z["valid"], bool)
        ev = z["ev"]
        nu = z["nu"]
        x, t = z["x"], z["t"]
        pred_pol = valid & (((h > 0) & (s >= thr_hot)) | ((h <= 0) & (s >= thr_cold)))
        pred_one = valid & (s >= thr_single)
        sid = p.stem
        template = "?"
        if seqs is not None and sid in seqs.index:
            template = seqs.loc[sid, "template"]
            if not np.ndim(template) == 0:
                template = template.iloc[0]
        lab = by_seq.get(sid)
        _update(acc["pol"], sid, pred_pol, s, h, valid, ev, nu, x, t, lab, str(template), thr_hot)
        _update(acc["single"], sid, pred_one, s, h, valid, ev, nu, x, t, lab, str(template), thr_single)
        if i % 10 == 0:
            print(f"eval {ver} {split} {i} {sid} {time.time() - t0:.0f}s", flush=True)
    return {name: _pack(a, thr_hot if name == "pol" else thr_single) for name, a in acc.items()}


def _empty():
    return dict(
        tot=np.zeros(3, np.int64), hot=np.zeros(3, np.int64), cold=np.zeros(3, np.int64),
        per={}, far_n=0, far_d=0, ev_rows=[],
    )


def _update(acc, sid, pred, s, h, valid, ev, nu, x, t, labels_seq, template, thr_for_events):
    truth = ev > 0
    c = np.array(M.confusion(pred, truth, valid), np.int64)
    acc["tot"] += c
    acc["per"][sid] = tuple(int(v) for v in c)
    acc["hot"] += np.array(M.confusion(pred & (h > 0), (ev >= 1) & (ev <= 4), valid), np.int64)
    acc["cold"] += np.array(M.confusion(pred & (h <= 0), ev >= 5, valid), np.int64)
    bg = valid & (ev == 0) & (nu == 0)
    acc["far_n"] += int((pred & bg).sum())
    acc["far_d"] += int(bg.sum())
    if labels_seq is None:
        return
    # событийные метрики — по уже готовой маске, без второго чтения
    for r in M.label_boxes(labels_seq, sid).itertuples():
        if r.kind != "event":
            continue
        px = M.event_pixels(r, x, t, ev)
        if px.sum() == 0:
            continue
        frac = float(pred[px].mean())
        acc["ev_rows"].append(dict(
            label=r.label, amp=float(r.amp_obs_C), amp_bin=M.amp_bin(float(r.amp_obs_C)),
            template=template, frac=frac, hit=bool(frac >= 0.3),
            tp=int((pred & px).sum()), fn=int((~pred & px).sum()),
        ))
    del thr_for_events, s


def _pack(acc, thr):
    def agg(rows):
        if not rows:
            return dict(n=0, recall=None, pixel_recall=None)
        tp = sum(r["tp"] for r in rows)
        fn = sum(r["fn"] for r in rows)
        hit = sum(1 for r in rows if r["hit"])
        return dict(n=len(rows), recall=hit / len(rows), pixel_recall=tp / max(tp + fn, 1))

    E = acc["ev_rows"]
    res = dict(
        threshold=float(thr) if np.ndim(thr) == 0 else plain(thr),
        far=float(acc["far_n"] / max(acc["far_d"], 1)),
        **M.prf(*[int(v) for v in acc["tot"]]),
        hot=M.prf(*[int(v) for v in acc["hot"]]),
        cold=M.prf(*[int(v) for v in acc["cold"]]),
        n_bg=int(acc["far_d"]),
        n_seq=len(acc["per"]),
        iou_ci95=M.bootstrap_ci(acc["per"], sorted(acc["per"])) if acc["per"] else None,
    )
    if E:
        res["events_all"] = agg(E)
        res["weak_events"] = agg([r for r in E if r["amp"] <= 2])
        res["by_amp"] = {k: agg([r for r in E if r["amp_bin"] == k]) for k in sorted({r["amp_bin"] for r in E})}
        res["by_label"] = {k: agg([r for r in E if r["label"] == k]) for k in sorted({r["label"] for r in E})}
    return res


def meta(ver):
    labels = pd.read_csv(ROOT / f"data/synth_meta/{ver}/labels.csv")
    seqs = pd.read_csv(ROOT / f"data/synth_meta/{ver}/sequences.csv").set_index("seq_id")
    return labels, seqs


def slim_grid(rows):
    keep = ("alpha_hot", "thr_hot", "thr_cold", "iou", "f1", "precision", "recall", "far", "hot_iou", "cold_iou", "k_hot", "k_cold")
    return [{k: plain(r[k]) for k in keep if k in r} for r in rows]


def main():
    t0 = time.time()
    ref = json.load(open(WINNER_JSON))
    published_thr = float(ref["thresholds"][METHOD]["far_1e-3"])
    published = dict(
        val_iou=float(ref["val"][f"{METHOD}@far_1e-3"]["iou"]),
        test_iou=float(ref["test"][f"{METHOD}@far_1e-3"]["iou"]),
        thr=published_thr,
    )
    bag = collect_val()
    own_thr = single_threshold(bag)
    base_row = score_pair(bag, own_thr, own_thr)
    base_row["alpha_hot"] = None
    print(f"single thr {own_thr:.6f} published {published_thr:.6f} bag_iou {base_row['iou']:.4f}", flush=True)
    rows = build_grid(bag)
    best = select(rows)
    print(
        f"best alpha {best['alpha_hot']:.2f} thr {best['thr_hot']:.4f}/{best['thr_cold']:.4f} "
        f"bag_iou {best['iou']:.4f} far {best['far']:.6f}",
        flush=True,
    )
    # точные маски. Общий порог — опубликованный, чтобы строка совпала с r7.
    labels, seqs = meta("v8")
    v8_val = evaluate_split("v8", "val", best["thr_hot"], best["thr_cold"], published_thr, labels, seqs)
    v8_test = evaluate_split("v8", "test", best["thr_hot"], best["thr_cold"], published_thr, labels, seqs)
    out_versions = {"v8": {"val": v8_val, "test": v8_test}}
    for ver in ("v10", "v11"):
        lab, sq = meta(ver)
        out_versions[ver] = {
            "test": evaluate_split(ver, "test", best["thr_hot"], best["thr_cold"], published_thr, lab, sq)
        }
    val_iou = float(v8_val["pol"]["iou"])
    test_iou = float(v8_test["pol"]["iou"])
    beats = val_iou >= published["val_iou"] + WINNER_MARGIN
    print(
        f"exact val {val_iou:.4f} test {test_iou:.4f} "
        f"single {float(v8_val['single']['iou']):.4f}/{float(v8_test['single']['iou']):.4f} "
        f"v10 {float(out_versions['v10']['test']['pol']['iou']):.4f} "
        f"v11 {float(out_versions['v11']['test']['pol']['iou']):.4f}",
        flush=True,
    )
    single_val = float(v8_val["single"]["iou"])
    if abs(single_val - published["val_iou"]) > 2e-3:
        print(f"WARNING single val IoU {single_val:.4f} vs published {published['val_iou']:.4f}", flush=True)
    report = dict(
        name="polthr",
        method=METHOD,
        far_target=FAR,
        hot_gate=HOT_GATE,
        published=published,
        single_recomputed_thr=own_thr,
        bag_single=slim_grid([base_row])[0],
        grid=slim_grid(rows),
        selected=slim_grid([best])[0],
        exact={
            "v8_val": plain(v8_val),
            "v8_test": plain(v8_test),
            "v10_test": plain(out_versions["v10"]["test"]),
            "v11_test": plain(out_versions["v11"]["test"]),
        },
        beats_winner_val=bool(beats),
        winner_margin=WINNER_MARGIN,
        seconds=round(time.time() - t0, 1),
        note="Пороги только по фону v8 val. Суммарный бюджет тревог как у far_threshold(1e-3). Реальные файлы не использовались.",
    )
    path = ROOT / "artifacts/eval/s3_polthr_v8.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(plain(report), open(path, "w"), indent=1, ensure_ascii=False)
    v10_iou = float(out_versions["v10"]["test"]["pol"]["iou"])
    v11_iou = float(out_versions["v11"]["test"]["pol"]["iou"])
    line = (
        f"POLTHR_DONE val_iou={val_iou:.4f} test_iou={test_iou:.4f} "
        f"v10_test={v10_iou:.4f} v11_test={v11_iou:.4f} "
        f"thr_hot={best['thr_hot']:.4f} thr_cold={best['thr_cold']:.4f} "
        f"alpha_hot={best['alpha_hot']:.2f} val_far={v8_val['pol']['far']:.6f} "
        f"test_far={v8_test['pol']['far']:.6f} "
        f"single_val={single_val:.4f} single_test={float(v8_test['single']['iou']):.4f} "
        f"beats_winner={str(beats).lower()} winner_val={published['val_iou']:.4f}"
    )
    elog = Path("/workspace/run_logs/_events.log")
    elog.parent.mkdir(parents=True, exist_ok=True)
    with elog.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
        if beats:
            f.write(f"BEATS_WINNER polthr val {val_iou:.4f} test {test_iou:.4f}\n")
    print(line, flush=True)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
