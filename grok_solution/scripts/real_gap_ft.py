"""Промахи и топ-зоны выпуска ens_u2ft_tcn_lgbm на MGTS.

Ждёт строку FINAL в _events.log: до неё real_v8.json ещё от прошлого ансамбля.
Порог — релизный far 1e-3, карточки его не выбирают. Старые карты
domain_cache/*_score.npz перезаписываются на месте.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_k] = "2"

import csv
import gc
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "artifacts/eval/domain_cache"
CARDS = [
    ROOT / "data/operator_cards/cards_batch1.jsonl",
    ROOT / "data/operator_cards/cards_batch2.jsonl",
]
UNSUP = ROOT / "artifacts/unsup"
REAL_JSON = ROOT / "artifacts/eval/real_v8.json"
ZONES_CSV = ROOT / "artifacts/eval/real_zones.csv"
OUT = ROOT / "artifacts/eval/real_gap_ft.json"
EVENTS = Path("/workspace/grok_logs/_events.log")
METHOD = "ens_u2ft_tcn_lgbm"
FINAL_MARK = "FINAL released ens_u2ft_tcn_lgbm"


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S', time.gmtime())} {msg}", flush=True)


def wait_final() -> None:
    for _ in range(90):
        if EVENTS.exists() and FINAL_MARK in EVENTS.read_text():
            log("FINAL есть")
            return
        log("жду FINAL")
        time.sleep(20)
    raise SystemExit("нет строки FINAL")


def cap_blas() -> None:
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=2)
    except Exception as exc:
        log(f"threadpoolctl: {exc}")


def mem_available_mb() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    return 0


def wait_ram(need_mb: int, what: str) -> None:
    for _ in range(60):
        avail = mem_available_mb()
        if avail >= need_mb:
            return
        log(f"жду ОЗУ для {what}: {avail} МБ")
        time.sleep(20)
        gc.collect()
    log(f"ОЗУ {mem_available_mb()} МБ, продолжаю {what}")


def write_json(path: Path, obj) -> None:
    def conv(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        raise TypeError(type(o))

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=conv))
    os.replace(tmp, path)


def save_score(name: str, score: np.ndarray, hot: np.ndarray) -> None:
    path = CACHE / f"{name}_score.npz"
    tmp = CACHE / f".{name}_score.partial.npz"
    np.savez_compressed(tmp, score=np.asarray(score, np.float32), hot=np.asarray(hot, np.float16),
                        method=np.array(METHOD))
    os.replace(tmp, path)


def score_is_current(name: str) -> bool:
    path = CACHE / f"{name}_score.npz"
    if not path.exists():
        return False
    with np.load(path) as z:
        if "method" not in z.files:
            return False
        return str(z["method"]) == METHOD


def load_ch(name: str) -> dict:
    with np.load(CACHE / f"{name}_ch.npz") as z:
        return dict(X=np.array(z["X"]), valid=np.array(z["valid"]).astype(bool),
                    x=np.array(z["x"]), t=np.array(z["t"]), sigma=np.array(z["sigma"]),
                    dT=np.array(z["dT"]))


def load_score(name: str):
    with np.load(CACHE / f"{name}_score.npz") as z:
        return np.array(z["score"]), np.array(z["hot"]).astype(np.float32)


def load_base(name: str) -> list:
    return json.loads((CACHE / f"{name}_base.json").read_text())


def load_cards() -> list:
    rows = []
    for path in CARDS:
        rows += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows


def load_analyzer():
    wait_ram(1600, "Analyzer")
    import torch
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except Exception as exc:
        log(f"interop: {exc}")
    from dts import infer
    analyzer = infer.Analyzer(op="far_1e-3")
    for kind, mdl in analyzer.members:
        if kind != "net" and hasattr(mdl, "set_params"):
            mdl.set_params(n_jobs=2)
            log(f"lgbm n_jobs={mdl.n_jobs}")
    raw = analyzer.classify

    def classify_batched(X, sigma, x, t, zones, batch=256):
        if len(zones) <= batch:
            return raw(X, sigma, x, t, zones)
        return np.concatenate([raw(X, sigma, x, t, zones[i:i + batch]) for i in range(0, len(zones), batch)], 0)

    analyzer.classify = classify_batched
    log(f"members {len(analyzer.members)} thr {analyzer.thr:.6f}")
    return analyzer


def ensure_scores(names, analyzer) -> None:
    from dts import domain
    for name in names:
        if score_is_current(name):
            log(f"{name}: score {METHOD} уже есть")
            continue
        wait_ram(1200, f"файла {name}")
        field = load_ch(name)
        X = field["X"].astype(np.float32)
        t0 = time.time()
        score, hot = domain.ensemble_maps(analyzer, X, log=f"ft {name}")
        save_score(name, score, hot)
        log(f"{name}: {score.shape} за {time.time() - t0:.0f} с")
        del field, X, score, hot
        gc.collect()


def box_stats(card, x, t, valid, score, hot, thrs: dict) -> dict:
    from dts import domain
    xa, xb, ta, tb = domain.card_bounds(card)
    ix = (x >= xa) & (x <= xb)
    it = (t >= ta) & (t <= tb)
    width_m = abs(xb - xa)
    dur_h = abs(tb - ta) / 3600.0
    if not ix.any() or not it.any():
        return dict(width_m=width_m, dur_h=dur_h, status="box_outside_record", max_score=None)
    sl = np.ix_(it, ix)
    vb = valid[sl]
    sc = np.asarray(score[sl], np.float64)
    hh = np.asarray(hot[sl], np.float64)
    if not vb.any():
        return dict(width_m=width_m, dur_h=dur_h, status="no_valid", max_score=None, n_valid=0)
    scv, hhv = sc[vb], hh[vb]
    k = int(np.argmax(scv))
    out = dict(width_m=round(width_m, 2), dur_h=round(dur_h, 2), status="ok", n_valid=int(vb.sum()),
               max_score=float(scv[k]), hot_at_max=float(hhv[k]))
    for key, thr in thrs.items():
        above = scv >= thr
        out[f"frac_{key}"] = float(above.mean())
        out[f"n_above_{key}"] = int(above.sum())
    return out


def explain(shape: str, st: dict, thr_ref: float, thr_1e2: float, thr_1e3: float) -> str:
    cooling = "cool" in (shape or "").lower()
    heating = "heat" in (shape or "").lower()
    hot = st.get("hot_at_max")
    mx = st.get("max_score")
    parts = []
    if mx is None:
        return "рамка вне записи"
    if mx < thr_ref:
        parts.append("фон: пик ниже far_ref")
    elif mx < thr_1e2:
        parts.append("порог: пик между far_ref и far 1e-2")
    elif mx < thr_1e3:
        parts.append("порог: пик между far 1e-2 и far 1e-3")
    if cooling and hot is not None and hot > 0.05:
        parts.append("полярность: карточка cooling, пик горячий")
    elif heating and hot is not None and hot < -0.05:
        parts.append("полярность: карточка heating, пик холодный")
    elif hot is not None and abs(hot) >= 0.05:
        parts.append("знак пика совпадает с карточкой" if (cooling or heating) else "пик со знаком")
    if st["width_m"] >= 80 or st["dur_h"] >= 8:
        parts.append("рамка широкая или длинная")
    n_rel = st.get("n_above_far_1e-3", 0)
    if 0 < n_rel < 12:
        parts.append(f"выше порога {n_rel} пикс., меньше минимума зоны 12")
    if not parts:
        parts.append("зона при пороге релиза не пересекла рамку")
    return "; ".join(parts)


def unsup_as_zones(name: str) -> list:
    import pandas as pd
    path = UNSUP / f"zones_{name}.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    E0 = pd.Timestamp("1970-01-01", tz="UTC")
    out = []
    for r in df.itertuples(index=False):
        t0 = pd.Timestamp(r.t_start_utc)
        t1 = pd.Timestamp(r.t_end_utc)
        t0 = t0.tz_convert("UTC") if t0.tzinfo else t0.tz_localize("UTC")
        t1 = t1.tz_convert("UTC") if t1.tzinfo else t1.tz_localize("UTC")
        out.append(dict(start_m=float(r.x_start_m), end_m=float(r.x_end_m),
                        t_start_s=float((t0 - E0).total_seconds()),
                        t_end_s=float((t1 - E0).total_seconds()),
                        polarity=r.polarity, alarm=True))
    return out


def load_supervised_zones() -> list:
    import pandas as pd
    E0 = pd.Timestamp("1970-01-01", tz="UTC")
    rows = []
    with open(ZONES_CSV, newline="") as f:
        for r in csv.DictReader(f):
            t0 = pd.Timestamp(r["t_start_utc"])
            t1 = pd.Timestamp(r["t_end_utc"])
            t0 = t0.tz_convert("UTC") if t0.tzinfo else t0.tz_localize("UTC")
            t1 = t1.tz_convert("UTC") if t1.tzinfo else t1.tz_localize("UTC")
            rows.append(dict(
                file=r["source"].replace(".csv", ""),
                zone_id=r["zone_id"],
                start_m=float(r["x_start_m"]), end_m=float(r["x_end_m"]),
                t_start_s=float((t0 - E0).total_seconds()),
                t_end_s=float((t1 - E0).total_seconds()),
                t_start_utc=r["t_start_utc"], t_end_utc=r["t_end_utc"],
                polarity=r["polarity"], klass=r["class"],
                score=float(r["score"]), peak_dT_C=float(r["peak_dT_C"]),
                n_px=int(float(r["n_px"])), is_alarm=int(float(r["is_alarm"])),
            ))
    return rows


def main():
    os.chdir(ROOT)
    cap_blas()
    import sys
    sys.path.insert(0, str(ROOT))
    wait_final()
    published = json.loads(REAL_JSON.read_text())
    if abs(float(published["threshold"]) - 0.8608087301264272) > 1e-6:
        log(f"SANITY_FAIL threshold real_v8={published.get('threshold')}")
    release = json.loads((ROOT / "models_release/stage1_thresholds.json").read_text())
    if release.get("method") != METHOD:
        raise SystemExit(f"релиз {release.get('method')}")
    cnn = release["thresholds"]["cnn_unet"]
    thr_ref = float(cnn["far_ref"])
    thr_1e3 = float(cnn["far_1e-3"])
    thr_1e2 = float(cnn["far_1e-2"])
    thrs = {"far_1e-3": thr_1e3, "far_1e-2": thr_1e2, "far_ref": thr_ref}

    from dts import domain
    names = [str(i) for i in range(1, 7)]
    analyzer = load_analyzer()
    if abs(analyzer.thr - thr_1e3) > 1e-6:
        log(f"SANITY_FAIL analyzer thr {analyzer.thr}")
    cap_blas()
    ensure_scores(names, analyzer)

    cards = load_cards()
    pub = {c["id"]: c for c in published["cards"]}
    diffs = []
    misses = []
    hits_odd = []
    tallies = {"yes": 0, "yes_n": 0, "no": 0, "no_n": 0}
    file_alarm = {}

    for name in names:
        field = load_ch(name)
        score, hot = load_score(name)
        X = field["X"].astype(np.float32)
        out = dict(grid=dict(x=field["x"], t=field["t"]), X=X, valid=field["valid"], score=score,
                   hotness=hot, sigma=field["sigma"].astype(np.float32), dT=field["dT"].astype(np.float32))
        got = domain.rethreshold(analyzer, out, thr_1e3)
        n_alarm = int(sum(bool(z.get("alarm")) for z in got["zones"]))
        file_alarm[name] = n_alarm
        pub_n = published["files"][name]["n_alarm_zones_cnn"]
        if n_alarm != pub_n:
            log(f"SANITY_FAIL file {name} zones {n_alarm} vs real_v8 {pub_n}")
        base = load_base(name)
        for card in cards:
            if str(card.get("record")) != name or card.get("presence") not in ("yes", "no"):
                continue
            ev = domain.score_card(card, field["x"], field["t"], field["valid"], score, got["zones"], thr_1e3, base)
            st = box_stats(card, field["x"], field["t"], field["valid"], score, hot, thrs)
            p = pub.get(card["id"], {})
            if p.get("cnn_max_score") is not None and st.get("max_score") is not None:
                diffs.append(abs(float(p["cnn_max_score"]) - float(st["max_score"])))
            if card["presence"] == "yes":
                tallies["yes_n"] += 1
                tallies["yes"] += int(bool(ev["hit_alarm"]))
            else:
                tallies["no_n"] += 1
                tallies["no"] += int(bool(ev["hit_alarm"]))
            if card["presence"] != "yes" or ev["hit_alarm"]:
                shape = card.get("shape") or ""
                hotv = st.get("hot_at_max")
                wrong = (("cool" in shape.lower() and hotv is not None and hotv > 0.05)
                         or ("heat" in shape.lower() and hotv is not None and hotv < -0.05))
                if card["presence"] == "yes" and wrong:
                    hits_odd.append(dict(id=card["id"], record=name, shape=shape,
                                         max_score=st.get("max_score"), hot_at_max=hotv,
                                         cnn_class=ev.get("cnn_dominant_class"),
                                         base_n=ev.get("base_n_zones"),
                                         notes=(card.get("notes") or "")[:180]))
                continue
            xa, xb, ta, tb = domain.card_bounds(card)
            bz = domain.overlapping_zones(base, xa, xb, ta, tb, False)
            misses.append(dict(
                id=card["id"], record=name, shape=card.get("shape") or "",
                notes=(card.get("notes") or "")[:220],
                base_n=len(bz), base_polarity=sorted({z.get("polarity") for z in bz}),
                reason=explain(card.get("shape") or "", st, thr_ref, thr_1e2, thr_1e3),
                published_max=p.get("cnn_max_score"),
                published_frac=p.get("cnn_frac_flagged"),
                published_base=p.get("base_n_zones"),
                **st,
            ))
        log(f"{name} alarm {n_alarm}")
        del field, X, score, hot, out
        gc.collect()

    if diffs:
        log(f"сверка max score median {np.median(diffs):.6f} max {np.max(diffs):.6f}")
        if float(np.max(diffs)) > 0.01:
            log("SANITY_FAIL max score")

    sup = load_supervised_zones()
    sup.sort(key=lambda z: -z["score"])
    unsup_cache = {n: unsup_as_zones(n) for n in names}
    base_cache = {n: load_base(n) for n in names}
    top = []
    for z in sup[:20]:
        dur_h = (z["t_end_s"] - z["t_start_s"]) / 3600.0
        width_m = z["end_m"] - z["start_m"]
        ua = domain.overlapping_zones(unsup_cache[z["file"]], z["start_m"], z["end_m"], z["t_start_s"], z["t_end_s"], False)
        ba = domain.overlapping_zones(base_cache[z["file"]], z["start_m"], z["end_m"], z["t_start_s"], z["t_end_s"], False)
        top.append(dict(
            zone_id=z["zone_id"], file=z["file"], klass=z["klass"], polarity=z["polarity"],
            score=z["score"], peak_dT_C=z["peak_dT_C"], dur_h=round(dur_h, 2),
            width_m=round(width_m, 2), n_px=z["n_px"],
            t_start_utc=z["t_start_utc"], t_end_utc=z["t_end_utc"],
            start_m=z["start_m"], end_m=z["end_m"],
            unsup_n=len(ua), unsup_same_sign=sum(u.get("polarity") == z["polarity"] for u in ua),
            base_n=len(ba), base_same_sign=sum(b.get("polarity") == z["polarity"] for b in ba),
        ))
    all_phys = []
    for z in sup:
        ua = domain.overlapping_zones(unsup_cache[z["file"]], z["start_m"], z["end_m"], z["t_start_s"], z["t_end_s"], False)
        ba = domain.overlapping_zones(base_cache[z["file"]], z["start_m"], z["end_m"], z["t_start_s"], z["t_end_s"], False)
        all_phys.append(dict(
            unsup_n=len(ua), base_n=len(ba),
            unsup_same=sum(u.get("polarity") == z["polarity"] for u in ua),
            base_same=sum(b.get("polarity") == z["polarity"] for b in ba),
            dur_h=(z["t_end_s"] - z["t_start_s"]) / 3600.0,
            width_m=z["end_m"] - z["start_m"],
            peak=z["peak_dT_C"], polarity=z["polarity"], klass=z["klass"], score=z["score"],
        ))
    classes = {}
    for z in all_phys:
        classes[z["klass"]] = classes.get(z["klass"], 0) + 1
    summary = dict(
        n=len(all_phys),
        n_hot=sum(z["polarity"] == "hot" for z in all_phys),
        n_cold=sum(z["polarity"] == "cold" for z in all_phys),
        n_alarm=sum(z["is_alarm"] for z in sup),
        classes=classes,
        dur_p50=float(np.median([z["dur_h"] for z in all_phys])) if all_phys else None,
        width_p50=float(np.median([z["width_m"] for z in all_phys])) if all_phys else None,
        peak_abs_p50=float(np.median([abs(z["peak"]) for z in all_phys])) if all_phys else None,
        weak_dt=sum(abs(z["peak"]) < 0.4 for z in all_phys),
        wide=sum(z["width_m"] >= 80 for z in all_phys),
        long=sum(z["dur_h"] >= 6 for z in all_phys),
        both=sum(z["unsup_n"] and z["base_n"] for z in all_phys),
        only_base=sum(z["base_n"] and not z["unsup_n"] for z in all_phys),
        only_unsup=sum(z["unsup_n"] and not z["base_n"] for z in all_phys),
        neither=sum(not z["unsup_n"] and not z["base_n"] for z in all_phys),
        both_same_sign=sum(z["unsup_same"] and z["base_same"] for z in all_phys),
    )
    payload = dict(
        method=METHOD,
        threshold=thr_1e3,
        thresholds=thrs,
        note="Порог релиза far 1e-3. Карточки не выбирали порог.",
        tally=tallies,
        file_alarm=file_alarm,
        published_files={n: published["files"][n]["n_alarm_zones_cnn"] for n in names},
        misses=misses,
        hits_odd_sign=hits_odd,
        top_zones=top,
        zone_summary=summary,
        score_check=dict(n=len(diffs), median_abs=float(np.median(diffs)) if diffs else None,
                         max_abs=float(np.max(diffs)) if diffs else None),
    )
    write_json(OUT, payload)
    log(f"yes {tallies['yes']}/{tallies['yes_n']} no {tallies['no']}/{tallies['no_n']} misses {len(misses)}")
    log("REAL_GAP_FT_DONE")


if __name__ == "__main__":
    main()
