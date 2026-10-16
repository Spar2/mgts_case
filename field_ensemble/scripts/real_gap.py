"""Разрыв ens_u2_tcn_lgbm на MGTS: промахи, кривая рабочих точек, топ-зоны.

Порог не выбирается по карточкам. Карточки читаются после того, как пороги
зафиксированы: глобальные — из релиза или квантиля фона val, пофайловые —
квантиль валидных клеток этого файла.

Старые карты domain_cache/*_score.npz от предыдущего ансамбля перезаписываются
на месте, вторая копия не хранится.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_k] = "2"

import gc
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "artifacts/eval/domain_cache"
PRED = ROOT / "artifacts/pred/v8/ens_u2_tcn_lgbm/val"
SYN = ROOT / "artifacts/cache/v8/val"
CARDS = [
    ROOT / "data/operator_cards/cards_batch1.jsonl",
    ROOT / "data/operator_cards/cards_batch2.jsonl",
]
UNSUP = ROOT / "artifacts/unsup"
REAL_JSON = ROOT / "artifacts/eval/real_v8.json"
ZONES_CSV = ROOT / "artifacts/eval/real_zones.csv"
OUT = ROOT / "artifacts/eval/real_gap.json"
METHOD = "ens_u2_tcn_lgbm"


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S', time.gmtime())} {msg}", flush=True)


def cap_blas() -> None:
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=2)
    except Exception as exc:
        log(f"threadpoolctl: {exc}")


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


def mem_available_mb() -> int:
    avail = 0
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            avail = int(line.split()[1]) // 1024
    return avail


def wait_ram(need_mb: int, what: str, tries: int = 90) -> None:
    """Ждём, пока чужие процессы отпустят память. Свой пик не должен их вытеснить."""
    for _ in range(tries):
        avail = mem_available_mb()
        if avail >= need_mb:
            return
        log(f"жду ОЗУ для {what}: доступно {avail} МБ, нужно {need_mb}")
        time.sleep(20)
        gc.collect()
    log(f"ОЗУ всё ещё {mem_available_mb()} МБ, продолжаю {what}")


def val_thresholds(fars: tuple[float, ...]) -> tuple[dict, int]:
    """Один проход по фону val. Тот же far_threshold, что eval_stage1.bg."""
    from dts import metrics
    parts = []
    for path in sorted(SYN.glob("seq*.npz")):
        with np.load(path) as z:
            valid = np.array(z["valid"]).astype(bool)
            ev = np.array(z["ev"])
            nu = np.array(z["nu"])
        with np.load(PRED / f"{path.stem}.npz") as z:
            score = np.array(z["s"])
        mask = valid & (ev == 0) & (nu == 0)
        parts.append(np.asarray(score[mask], np.float32))
        del score, valid, ev, nu, mask
    bg = np.concatenate(parts)
    del parts
    out = {far: metrics.far_threshold(bg, far) for far in fars}
    n = int(bg.size)
    del bg
    gc.collect()
    return out, n


def load_analyzer():
    wait_ram(1600, "загрузки Analyzer")
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
    raw_classify = analyzer.classify

    def classify_batched(X, sigma, x, t, zones, batch=256):
        if len(zones) <= batch:
            return raw_classify(X, sigma, x, t, zones)
        parts = [raw_classify(X, sigma, x, t, zones[i:i + batch]) for i in range(0, len(zones), batch)]
        return np.concatenate(parts, 0)

    analyzer.classify = classify_batched
    log(f"members {len(analyzer.members)} thr {analyzer.thr:.6f} torch {torch.get_num_threads()}")
    return analyzer


def ensure_scores(names, analyzer) -> None:
    from dts import domain
    for name in names:
        if score_is_current(name):
            log(f"{name}: score {METHOD} уже есть")
            continue
        wait_ram(1400, f"инференса файла {name}")
        field = load_ch(name)
        X = field["X"].astype(np.float32)
        t0 = time.time()
        score, hot = domain.ensemble_maps(analyzer, X, log=f"real {name}")
        save_score(name, score, hot)
        log(f"{name}: score {score.shape} за {time.time() - t0:.0f} с")
        del field, X, score, hot


def exposure(field) -> tuple[float, float]:
    x, t, valid = field["x"], field["t"], field["valid"]
    hours = float((t[-1] - t[0]) / 3600.0) if len(t) > 1 else 0.0
    cols = valid.any(0)
    km = float(np.ptp(x[cols]) / 1000.0) if cols.any() else 0.0
    return hours, km


def zone_brief(zones) -> dict:
    classes = {}
    for z in zones:
        classes[z.get("class", "?")] = classes.get(z.get("class", "?"), 0) + 1
    return {"n_zones": len(zones), "n_alarm": int(sum(bool(z.get("alarm")) for z in zones)), "classes": classes}


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
    scv = sc[vb]
    hhv = hh[vb]
    k = int(np.argmax(scv))
    top = scv >= np.quantile(scv, 0.95)
    out = dict(width_m=round(width_m, 2), dur_h=round(dur_h, 2), status="ok", n_valid=int(vb.sum()),
               max_score=float(scv[k]), hot_at_max=float(hhv[k]),
               hot_top5=float(np.mean(hhv[top])) if top.any() else float("nan"),
               n_px=int(vb.sum()))
    for key, thr in thrs.items():
        above = scv >= thr
        out[f"frac_{key}"] = float(above.mean())
        out[f"n_above_{key}"] = int(above.sum())
    return out


def explain(shape: str, st: dict, thr_ref: float, thr_1e2: float, thr_1e3: float) -> str:
    cooling = "cool" in (shape or "").lower()
    hot = st.get("hot_at_max")
    mx = st.get("max_score")
    parts = []
    if mx is None:
        return "рамка вне записи"
    if mx < thr_ref:
        parts.append("порог: пик ниже far_ref, в рамке score как у фона")
    elif mx < thr_1e2:
        parts.append("порог: пик между far_ref и far 1e-2")
    elif mx < thr_1e3:
        parts.append("порог: пик между far 1e-2 и far 1e-3, компонент при релизе не собирается")
    if cooling and hot is not None and hot > 0.05:
        parts.append("полярность: карточка cooling, пик score горячий")
    elif cooling and hot is not None and hot < -0.05:
        parts.append("знак холодный и совпадает с карточкой")
    elif (not cooling) and hot is not None and hot < -0.05 and "heat" in (shape or "").lower():
        parts.append("полярность: карточка heating, пик score холодный")
    if st["width_m"] >= 80 or st["dur_h"] >= 8:
        parts.append("протяжённость: широкая или длинная рамка, плато снимается вычитанием профиля")
    if st["width_m"] <= 6 and st["dur_h"] <= 1.5 and mx < thr_1e3:
        parts.append("малый размер рамки")
    n_rel = st.get("n_above_far_1e-3", 0)
    if 0 < n_rel < 12:
        parts.append(f"выше порога только {n_rel} пикс., меньше минимума зоны 12")
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
                        polarity=r.polarity, alarm=True, score=float(r.score),
                        peak_dT_C=float(r.peak_dT_C)))
    return out


def load_supervised_zones() -> list:
    import csv
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
                n_px=int(r["n_px"]), is_alarm=int(r["is_alarm"]),
                class_prob=float(r["class_prob"]) if r["class_prob"] else None,
            ))
    return rows


def zone_physics(z) -> dict:
    dur_h = (z["t_end_s"] - z["t_start_s"]) / 3600.0
    width_m = z["end_m"] - z["start_m"]
    return dict(zone_id=z["zone_id"], file=z["file"], klass=z["klass"], polarity=z["polarity"],
                score=z["score"], peak_dT_C=z["peak_dT_C"], dur_h=round(dur_h, 2),
                width_m=round(width_m, 2), n_px=z["n_px"], class_prob=z["class_prob"],
                t_start_utc=z["t_start_utc"], t_end_utc=z["t_end_utc"],
                start_m=z["start_m"], end_m=z["end_m"])


def main():
    os.chdir(ROOT)
    cap_blas()
    import sys
    sys.path.insert(0, str(ROOT))
    from dts import domain

    release = json.loads((ROOT / "models_release/stage1_thresholds.json").read_text())
    cnn = release["thresholds"]["cnn_unet"]
    thr_ref = float(cnn["far_ref"])
    thr_1e3 = float(cnn["far_1e-3"])
    thr_1e2 = float(cnn["far_1e-2"])
    far_ref_rate = float(release["far_ref"])
    log("квантили фона val одним проходом")
    got, n_bg = val_thresholds((1e-3, 3e-3, 1e-2))
    chk_1e3, thr_3e3, chk_1e2 = got[1e-3], got[3e-3], got[1e-2]
    log(f"val n_bg={n_bg} far_1e-3 recomputed {chk_1e3:.6f} release {thr_1e3:.6f}")
    log(f"val far_1e-2 recomputed {chk_1e2:.6f} release {thr_1e2:.6f} far_3e-3 {thr_3e3:.6f}")
    if abs(chk_1e3 - thr_1e3) > 0.01 or abs(chk_1e2 - thr_1e2) > 0.01:
        log("THRESHOLD_MISMATCH val preds vs release")

    global_ops = [
        ("global_far_1e-3", thr_1e3, None),
        ("global_far_3e-3", thr_3e3, None),
        ("global_far_1e-2", thr_1e2, None),
        ("global_far_ref", thr_ref, None),
    ]
    per_rates = [
        ("per_file_1e-3", 1e-3),
        ("per_file_3e-3", 3e-3),
        ("per_file_1e-2", 1e-2),
        ("per_file_far_ref", far_ref_rate),
    ]

    names = [str(i) for i in range(1, 7)]
    analyzer = load_analyzer()
    cap_blas()
    ensure_scores(names, analyzer)

    published = json.loads(REAL_JSON.read_text())
    cards = load_cards()
    card_rows = []
    file_ops = {}
    misses = []
    exposures = {}
    max_diffs = []

    for name in names:
        field = load_ch(name)
        score, hot = load_score(name)
        hours, km = exposure(field)
        exposures[name] = dict(hours=round(hours, 3), km=round(km, 3), km_h=round(hours * km, 4))
        X = field["X"].astype(np.float32)
        out = dict(grid=dict(x=field["x"], t=field["t"]), X=X, valid=field["valid"], score=score,
                   hotness=hot, sigma=field["sigma"].astype(np.float32), dT=field["dT"].astype(np.float32))
        base = load_base(name)
        ops = {}
        thrs_box = {"far_1e-3": thr_1e3, "far_3e-3": thr_3e3, "far_1e-2": thr_1e2, "far_ref": thr_ref}
        for op_name, thr, _ in global_ops:
            got = domain.rethreshold(analyzer, out, thr)
            brief = zone_brief(got["zones"])
            brief["threshold"] = float(thr)
            brief["flag_fraction"] = domain.flag_fraction(score, field["valid"], thr)
            ops[op_name] = brief
            for card in cards:
                if str(card.get("record")) != name:
                    continue
                ev = domain.score_card(card, field["x"], field["t"], field["valid"], score, got["zones"], thr, base)
                ev["op"] = op_name
                ev["shape"] = card.get("shape", "")
                card_rows.append(ev)
        for op_name, far in per_rates:
            thr = float(domain.flag_threshold(score, field["valid"], far))
            got = domain.rethreshold(analyzer, out, thr)
            brief = zone_brief(got["zones"])
            brief["threshold"] = thr
            brief["flag_fraction"] = domain.flag_fraction(score, field["valid"], thr)
            ops[op_name] = brief
            for card in cards:
                if str(card.get("record")) != name:
                    continue
                ev = domain.score_card(card, field["x"], field["t"], field["valid"], score, got["zones"], thr, base)
                ev["op"] = op_name
                ev["shape"] = card.get("shape", "")
                card_rows.append(ev)
        ops["n_zones_base"] = len(base)
        file_ops[name] = ops
        pub_cards = {c["id"]: c for c in published["cards"] if str(c.get("record")) == name}
        for card in cards:
            if str(card.get("record")) != name or card.get("presence") != "yes":
                continue
            st = box_stats(card, field["x"], field["t"], field["valid"], score, hot, thrs_box)
            hit = next(r for r in card_rows if r["id"] == card["id"] and r["op"] == "global_far_1e-3")
            pub = pub_cards.get(card["id"])
            if pub and pub.get("cnn_max_score") is not None and st.get("max_score") is not None:
                max_diffs.append(abs(float(pub["cnn_max_score"]) - float(st["max_score"])))
            if not hit["hit_alarm"]:
                xa, xb, ta, tb = domain.card_bounds(card)
                bz = domain.overlapping_zones(base, xa, xb, ta, tb, False)
                pols = sorted({z.get("polarity") for z in bz})
                misses.append(dict(
                    id=card["id"], record=name, shape=card.get("shape", ""),
                    notes=(card.get("notes") or "")[:180],
                    base_n=len(bz), base_polarity=pols,
                    reason=explain(card.get("shape", ""), st, thr_ref, thr_1e2, thr_1e3),
                    **st,
                ))
        log(name + " " + ", ".join(
            f"{op} z={file_ops[name][op]['n_alarm']}" for op, _, _ in global_ops))
        del field, X, score, hot, out
        gc.collect()

    if max_diffs:
        log(f"сверка max score с real_v8: median {np.median(max_diffs):.4f} max {np.max(max_diffs):.4f}")

    tallies = {}
    op_names = [n for n, _, _ in global_ops] + [n for n, _ in per_rates]
    for op_name in op_names:
        subset = [r for r in card_rows if r["op"] == op_name]
        tallies[op_name] = {key: domain.presence_tally(subset, key) for key in ("hit_alarm", "hit_base")}

    curve = []
    for op_name in op_names:
        t = tallies[op_name]["hit_alarm"]
        tb = tallies[op_name]["hit_base"]
        n_alarm = 0
        km_h = 0.0
        per = {}
        for name, ops in file_ops.items():
            exp = exposures[name]["km_h"]
            na = ops[op_name]["n_alarm"]
            n_alarm += na
            km_h += exp
            per[name] = dict(n_alarm=na, threshold=ops[op_name]["threshold"],
                             flag_fraction=ops[op_name]["flag_fraction"],
                             zones_per_km_h=(na / exp) if exp else None,
                             classes=ops[op_name]["classes"])
        curve.append(dict(
            op=op_name,
            yes=t["yes_hit_alarm"], yes_n=t["yes_n"],
            no=t["no_hit_alarm"], no_n=t["no_n"],
            base_yes=tb["yes_hit_base"], base_no=tb["no_hit_base"],
            n_alarm=n_alarm, km_h=round(km_h, 3),
            zones_per_km_h=(n_alarm / km_h) if km_h else None,
            files=per,
        ))

    # топ зоны релизного CSV и пересечение с unsup / эталоном
    sup = load_supervised_zones()
    sup.sort(key=lambda z: -z["score"])
    top = []
    unsup_cache = {n: unsup_as_zones(n) for n in names}
    base_cache = {n: load_base(n) for n in names}
    for z in sup[:20]:
        ph = zone_physics(z)
        ua = domain.overlapping_zones(unsup_cache[z["file"]], z["start_m"], z["end_m"], z["t_start_s"], z["t_end_s"], False)
        ba = domain.overlapping_zones(base_cache[z["file"]], z["start_m"], z["end_m"], z["t_start_s"], z["t_end_s"], False)
        same_u = [u for u in ua if u.get("polarity") == z["polarity"]]
        same_b = [b for b in ba if b.get("polarity") == z["polarity"]]
        ph["unsup_n"] = len(ua)
        ph["unsup_same_sign"] = len(same_u)
        ph["base_n"] = len(ba)
        ph["base_same_sign"] = len(same_b)
        top.append(ph)

    def agree(zs):
        both = sum(1 for z in zs if z["unsup_n"] and z["base_n"])
        only_b = sum(1 for z in zs if z["base_n"] and not z["unsup_n"])
        only_u = sum(1 for z in zs if z["unsup_n"] and not z["base_n"])
        none = sum(1 for z in zs if not z["unsup_n"] and not z["base_n"])
        return dict(both=both, only_base=only_b, only_unsup=only_u, neither=none, n=len(zs))

    all_phys = []
    for z in sup:
        ua = domain.overlapping_zones(unsup_cache[z["file"]], z["start_m"], z["end_m"], z["t_start_s"], z["t_end_s"], False)
        ba = domain.overlapping_zones(base_cache[z["file"]], z["start_m"], z["end_m"], z["t_start_s"], z["t_end_s"], False)
        all_phys.append(dict(unsup_n=len(ua), base_n=len(ba),
                             dur_h=(z["t_end_s"] - z["t_start_s"]) / 3600.0,
                             width_m=z["end_m"] - z["start_m"],
                             peak=z["peak_dT_C"], polarity=z["polarity"], klass=z["klass"],
                             score=z["score"]))
    summary = dict(
        n=len(all_phys),
        n_hot=sum(1 for z in all_phys if z["polarity"] == "hot"),
        n_cold=sum(1 for z in all_phys if z["polarity"] == "cold"),
        classes={},
        dur_p50=float(np.median([z["dur_h"] for z in all_phys])) if all_phys else None,
        width_p50=float(np.median([z["width_m"] for z in all_phys])) if all_phys else None,
        peak_abs_p50=float(np.median([abs(z["peak"]) for z in all_phys])) if all_phys else None,
        weak_dt=sum(1 for z in all_phys if abs(z["peak"]) < 0.4),
        wide=sum(1 for z in all_phys if z["width_m"] >= 80),
        long=sum(1 for z in all_phys if z["dur_h"] >= 6),
        agree=agree(all_phys),
    )
    for z in all_phys:
        summary["classes"][z["klass"]] = summary["classes"].get(z["klass"], 0) + 1

    payload = dict(
        method=METHOD,
        note="Пороги зафиксированы до чтения карточек. Карточки только в столбцах yes/no.",
        thresholds={
            "far_1e-3": thr_1e3, "far_3e-3": thr_3e3, "far_1e-2": thr_1e2, "far_ref": thr_ref,
            "far_ref_rate": far_ref_rate,
            "val_recomputed_far_1e-3": chk_1e3, "val_recomputed_far_1e-2": chk_1e2,
            "val_n_bg": n_bg,
        },
        exposures=exposures,
        curve=curve,
        misses=misses,
        tallies=tallies,
        files=file_ops,
        top_zones=top,
        zone_summary=summary,
        score_check=dict(n=len(max_diffs), median_abs=float(np.median(max_diffs)) if max_diffs else None,
                         max_abs=float(np.max(max_diffs)) if max_diffs else None),
    )
    # classes inside file_ops are fine; drop nothing else large
    write_json(OUT, payload)
    log(f"yes global {curve[0]['yes']}/{curve[0]['yes_n']} misses {len(misses)}")
    log("REAL_GAP_DONE")


if __name__ == "__main__":
    main()
