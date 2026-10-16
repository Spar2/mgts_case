"""Реальные MGTS_data/1–6.csv: инференс и сравнение с эталоном haklolarp + карточки оператора.

Порог по умолчанию берётся из релиза (--calib global). --calib per_file ставит квантиль
score этого файла так, чтобы доля флагов равнялась --far. Карточки и метки в порог не входят.
На этих данных ничего не обучается."""
import argparse, json, os, sys, glob
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dts import real, grid, baseline, infer, events as E, export

_nt = os.environ.get("OMP_NUM_THREADS")
if _nt:
    import torch
    torch.set_num_threads(max(1, int(_nt)))

ap = argparse.ArgumentParser()
ap.add_argument("--real-root", default="/workspace/mgts/MGTS_data"); ap.add_argument("--cards", nargs="*", default=[])
ap.add_argument("--out", default="artifacts/eval/real_v8.json"); ap.add_argument("--fig", default="docs/figures")
ap.add_argument("--op", default="far_1e-3"); ap.add_argument("--zones-csv", default="artifacts/eval/real_zones.csv")
ap.add_argument("--zones", default="cc", choices=("cc", "pdrs_hybrid"),
                help="cc — порог, замыкание и компоненты (победитель); pdrs_hybrid — гибрид PDRS из pdrs_hybrid_best_params.json")
ap.add_argument("--calib", default="global", choices=("global", "per_file"),
                help="global — порог из stage1_thresholds; per_file — квантиль score этого файла, без меток и без карточек")
ap.add_argument("--far", type=float, default=1e-3, help="доля валидных клеток выше порога при --calib per_file")
ap.add_argument("--cv-annot", action="store_true",
                help="OpenCV-аннотация зон (контур, фронт, период) и рисунок real_cv_<file>.png. Порог не меняет.")
ap.add_argument("--detector", default="winner", choices=("winner", "dom"),
                help="winner — выпуск models_release; dom — смесь ens_dom из models_release_dom, models_release только читается")
ap.add_argument("--static", default="off", choices=("off", "on"),
                help="off — скор победителя без StaticNet; on — слияние выбранной ветви STATIC2, порог из s3_static2.json")
ap.add_argument("--mae", default="off", choices=("off", "on"),
                help="off — победитель models_release без изменений; on — член mae веса 1/5, порог ens_mean4 из models_release_mae, отдельные real_mae.json/.csv")
ap.add_argument("--gen2", default="off", choices=("off", "on"),
                help="off — победитель без U-Net GEN2; on — ens_blend 0.5*(победитель+GEN2), порог из models_release_gen2, отдельные real_gen2.json/.csv")
a = ap.parse_args()
_root = Path(__file__).resolve().parents[1]
_OWNED = (
    "artifacts/eval/real_v8.json",
    "artifacts/eval/real_zones.csv",
    "docs/figures",
    "artifacts/eval/real_pdrs_hybrid.json",
    "artifacts/eval/real_pdrs_hybrid.csv",
    "artifacts/eval/real_static.json",
    "artifacts/eval/real_dom.json",
    "artifacts/eval/real_dom.csv",
    "artifacts/eval/real_dom_fig",
)
_CARDS2 = Path("/home/box/agent-data/agents/7620c11d-77b1-4929-aaa8-c80e192177e7/attachments/66eacdf5a4b0ab8649ecf57a4b35554ad67790401bcf97c035e546eff4f764ff.jsonl")
_REVIEW2 = _root / "artifacts/eval/cards2_review.csv"


def _refuse_owned(paths, label):
    forbidden = {(_root / p).resolve() for p in _OWNED}
    hits = [str(Path(p).resolve()) for p in paths if Path(p).resolve() in forbidden]
    if hits:
        sys.exit(f"{label} не пишет " + ", ".join(hits))


def _narrow_static(review_csv):
    """Узкие статичные пропуски победителя из CARDS_REVIEW_2. Порог по ним не выбирается."""
    import csv
    meta = {}
    with open(review_csv, newline="") as f:
        for row in csv.DictReader(f):
            user = (row.get("user") or "").strip()
            if not user.startswith("yes"):
                continue
            try:
                winner = int(float(row["winner"]))
                dT = float(row["dT_hi"])
                width = float(row["x1"]) - float(row["x0"])
            except (TypeError, ValueError):
                continue
            if winner != 0 or width > 12.0 or dT < 3.0:
                continue
            meta[row["id"]] = dict(dT_hi=dT, width_m=round(width, 3), winner=0)
    return meta


def _card_window(card):
    import pandas as pd
    E0 = pd.Timestamp("1970-01-01", tz="UTC")
    ts = []
    for s in card["time"]:
        t = pd.Timestamp(s)
        t = t.tz_convert("UTC") if t.tzinfo else t.tz_localize("UTC")
        ts.append((t - E0).total_seconds())
    return float(card["meters"][0]), float(card["meters"][1]), ts[0], ts[1]


def _alarm_hits(zones, xa, xb, ta, tb):
    return [z for z in zones if z.get("alarm") and z["end_m"] >= xa and z["start_m"] <= xb and z["t_end_s"] >= ta and z["t_start_s"] <= tb]


if a.static == "on":
    if a.detector != "winner":
        sys.exit("static=on слит с ансамблем победителя; --detector dom калиброван отдельно")
    if a.zones != "cc":
        sys.exit("static=on режет слитую карту зонами cc; pdrs_hybrid запускайте отдельно")
    if a.calib != "global":
        sys.exit("static=on держит порог synth v8 val; --calib per_file на реальных CSV его двигает")
    if a.out == "artifacts/eval/real_v8.json":
        a.out = "artifacts/eval/real_static_on.json"
    if a.zones_csv == "artifacts/eval/real_zones.csv":
        a.zones_csv = "artifacts/eval/real_static_on.csv"
    if a.fig == "docs/figures":
        a.fig = "artifacts/eval/fig_static_on"
    _refuse_owned((a.out, a.zones_csv, a.fig), "static=on")
    if not a.cards:
        if not _CARDS2.is_file():
            sys.exit(f"нет карточек {_CARDS2}")
        a.cards = [str(_CARDS2)]
if a.mae == "on":
    if a.detector != "winner":
        sys.exit("mae=on — член ансамбля победителя; --detector dom калиброван отдельно")
    if a.static != "off":
        sys.exit("mae=on и static=on вместе не калибровались")
    if a.zones != "cc":
        sys.exit("mae=on режет ens_mean4 зонами cc; pdrs_hybrid запускайте отдельно")
    if a.calib != "global":
        sys.exit("mae=on держит порог synth v8 val; --calib per_file на реальных CSV его двигает")
    if a.out == "artifacts/eval/real_v8.json":
        a.out = "artifacts/eval/real_mae.json"
    if a.zones_csv == "artifacts/eval/real_zones.csv":
        a.zones_csv = "artifacts/eval/real_mae.csv"
    if a.fig == "docs/figures":
        a.fig = "artifacts/eval/real_mae_fig"
    _refuse_owned((a.out, a.zones_csv, a.fig), "mae=on")
    if not a.cards:
        a.cards = [
            str(_root / "data/operator_cards/cards_batch1.jsonl"),
            str(_root / "data/operator_cards/cards_batch2.jsonl"),
        ]
if a.gen2 == "on":
    if a.detector != "winner":
        sys.exit("gen2=on — смесь с ансамблем победителя; --detector dom калиброван отдельно")
    if a.static != "off":
        sys.exit("gen2=on и static=on вместе не калибровались")
    if a.mae != "off":
        sys.exit("gen2=on и mae=on вместе не калибровались")
    if a.zones != "cc":
        sys.exit("gen2=on режет ens_blend зонами cc; pdrs_hybrid запускайте отдельно")
    if a.calib != "global":
        sys.exit("gen2=on держит порог synth v8 val; --calib per_file на реальных CSV его двигает")
    if a.out == "artifacts/eval/real_v8.json":
        a.out = "artifacts/eval/real_gen2.json"
    if a.zones_csv == "artifacts/eval/real_zones.csv":
        a.zones_csv = "artifacts/eval/real_gen2.csv"
    if a.fig == "docs/figures":
        a.fig = "artifacts/eval/real_gen2_fig"
    _refuse_owned((a.out, a.zones_csv, a.fig), "gen2=on")
    if not a.cards:
        a.cards = [
            str(_root / "data/operator_cards/cards_batch1.jsonl"),
            str(_root / "data/operator_cards/cards_batch2.jsonl"),
        ]
if a.detector == "dom":
    if a.out == "artifacts/eval/real_v8.json":
        a.out = "artifacts/eval/real_dom.json"
    if a.zones_csv == "artifacts/eval/real_zones.csv":
        a.zones_csv = "artifacts/eval/real_dom.csv"
    if a.fig == "docs/figures":
        a.fig = "artifacts/eval/real_dom_fig"
    _forbidden = {
        (_root / "artifacts/eval/real_v8.json").resolve(),
        (_root / "artifacts/eval/real_zones.csv").resolve(),
        (_root / "docs/figures").resolve(),
    }
    _hits = [Path(p).resolve() for p in (a.out, a.zones_csv, a.fig)]
    if any(p in _forbidden for p in _hits):
        sys.exit("ens_dom не пишет real_v8.json, real_zones.csv и docs/figures")
if a.zones != "cc":
    root = Path(__file__).resolve().parents[1]
    if Path(a.out).resolve() == (root / "artifacts/eval/real_v8.json").resolve():
        sys.exit("pdrs_hybrid не пишет artifacts/eval/real_v8.json — укажите другой --out")
    if Path(a.zones_csv).resolve() == (root / "artifacts/eval/real_zones.csv").resolve():
        sys.exit("pdrs_hybrid не пишет artifacts/eval/real_zones.csv — укажите другой --zones-csv")
    if Path(a.fig).resolve() == (root / "docs/figures").resolve():
        sys.exit("pdrs_hybrid не пишет docs/figures — укажите другой --fig")
dom_unet = None
dom_spec = None
if a.detector == "dom":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from s3_dom_real import load_dom, run_blend
    A, dom_unet, dom_spec = load_dom(a.op, a.zones)
else:
    A = infer.Analyzer(op=a.op, zones=a.zones, static=a.static, mae=a.mae, gen2=a.gen2)
COL = {"Local_Heating": "red", "Extended_Heating": "orange", "Contact_Trace": "magenta", "Local_Cooling": "blue",
       "Extended_Cooling": "cyan", "HVAC_Diurnal": "green", "Sensor_Artifact": "gray"}
cards = []
for c in a.cards:
    cards += [json.loads(l) for l in open(c) if l.strip()]
res = {"op": a.op, "threshold": A.thr, "calib": a.calib, "far": a.far, "zones_mode": a.zones,
      "static": a.static, "mae": a.mae, "gen2": a.gen2, "files": {}, "cards": []}
if a.static == "on":
    spec = A.static_spec
    res["static_branch"] = {
        "selected": spec["name"],
        "mode": spec["mode"],
        "scale": float(spec["scale"]),
        "gate": float(spec["gate"]),
        "threshold": float(spec["threshold"]),
        "ckpt": spec["ckpt"],
        "step": int(spec["step"]),
        "beats_winner": False,
        "note": "Порог и режим заморожены на synth v8 val (artifacts/static/s3_static2.json). Карточки и реальные CSV порог не двигают и в обучение не входят.",
    }
if a.mae == "on":
    res["detector"] = "ens_mean4"
    res["mae_branch"] = {
        "definition": A.mae_spec["definition"],
        "release": "models_release_mae",
        "winner_release_read_only": "models_release",
        "threshold": float(A.mae_spec["threshold"]),
        "mae_weight": float(A.mae_spec["mae_weight"]),
        "ckpt": A.mae_spec["ckpt"],
        "real_traces_used_for_train": False,
        "cards_move_threshold": False,
        "note": "Порог ens_mean4 заморожен на synth v8 val. Карточки и реальные CSV его не двигают и в обучение не входят.",
    }
if a.gen2 == "on":
    res["detector"] = "ens_blend"
    res["gen2_branch"] = {
        "definition": A.gen2_spec["definition"],
        "release": "models_release_gen2",
        "winner_release_read_only": "models_release",
        "threshold": float(A.gen2_spec["threshold"]),
        "ckpt": A.gen2_spec["ckpt"],
        "real_traces_used_for_train": False,
        "cards_move_threshold": False,
        "note": "Порог ens_blend заморожен на synth v8 val. Карточки и реальные CSV его не двигают и в обучение не входят.",
    }
if dom_spec is not None:
    res["detector"] = "ens_dom"
    res["ens_dom"] = {
        "definition": dom_spec.get("definition"),
        "release": "models_release_dom",
        "winner_release_read_only": "models_release",
        "threshold_source": dom_spec.get("threshold_source"),
    }
all_rows = []
Path(a.fig).mkdir(parents=True, exist_ok=True)
if a.static == "on":
    print(f"static=on selected={A.static_spec['name']} thr={A.thr:.6f} ckpt={A.static_spec['ckpt']}", flush=True)
for f in sorted(glob.glob(f"{a.real_root}/*.csv")):
    r = real.load_csv(f); name = r["name"]
    if dom_unet is None:
        o = A.run(r["T"], r["valid"], r["x"], r["t"], zones=a.zones)
    else:
        o = run_blend(A, dom_unet, r["T"], r["valid"], r["x"], r["t"], zones=a.zones)
    p = o["grid"]
    thr = A.thr
    if a.calib == "per_file":
        from dts import domain
        thr = domain.flag_threshold(o["score"], o["valid"], a.far)
        if a.zones == "cc":
            o = domain.rethreshold(A, o, thr)
        else:
            o = infer.rezone(A, o, thr, zones=a.zones)
    b = baseline.run(r["T"], r["x"], r["t"], valid=r["valid"])
    bz, _ = grid.resample(b["zone"], np.ones_like(r["valid"]), p, "nearest")
    cnn_mask = np.zeros_like(o["valid"])
    for z in o["zones"]: cnn_mask[z["t0"]:z["t1"], z["x0"]:z["x1"]] |= (o["score"][z["t0"]:z["t1"], z["x0"]:z["x1"]] >= thr)
    base_mask = (bz > 0) & o["valid"]
    inter = (cnn_mask & base_mask).sum(); uni = (cnn_mask | base_mask).sum()
    dead = int((cnn_mask & ~o["valid"]).sum())
    hours = (r["t"][-1] - r["t"][0]) / 3600; km = float(np.ptp(r["x"][r["valid"].any(0)])) / 1000
    cls = {}
    for z in o["zones"]: cls[z["class"]] = cls.get(z["class"], 0) + 1
    if a.cv_annot:
        from dts import zone_cv
        mask = (o["score"] >= thr) & o["valid"]
        anns = zone_cv.annotate(mask, o["dT"], o["valid"], p["x"], p["t"], o["zones"])
        fig_cv, ax_cv = plt.subplots(figsize=(12, 6))
        zone_cv.show_dT(ax_cv, o["dT"], o["valid"], p["x"], p["t"])
        zone_cv.overlay(fig_cv, anns, p["x"], p["t"], ax=ax_cv)
        ax_cv.set_title(f"{name}.csv: контуры зон и линия фронта (Хаф)")
        fig_cv.tight_layout()
        fig_cv.savefig(f"{a.fig}/real_cv_{name}.png", dpi=110)
        plt.close(fig_cv)
    res["files"][name] = dict(sha256=r["sha256"], tz_missing=r["tz_missing"], threshold=float(thr),
                              shape=list(r["T"].shape), grid_shape=list(o["valid"].shape),
                              hours=round(hours, 2), km=round(km, 3), n_zones_cnn=len(o["zones"]), n_alarm_zones_cnn=sum(z["alarm"] for z in o["zones"]),
                              n_zones_base=len(b["zones"]), alarm_fraction_cnn=float(cnn_mask[o["valid"]].mean()),
                              alarm_fraction_base=float(base_mask[o["valid"]].mean()), iou_cnn_vs_base=float(inter / max(uni, 1)),
                              dead_alarms=dead, classes=cls,
                              zones=[{k: v for k, v in z.items() if k not in ("t0", "t1", "x0", "x1")} for z in o["zones"]])
    all_rows += list(export.zone_rows(o["zones"], f"{name}.csv", r["sha256"]))
    # рисунок
    fig, ax = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    ext = [p["x"][0], p["x"][-1], (p["t"][-1] - p["t"][0]) / 3600, 0]
    dT = np.where(o["valid"], o["dT"], np.nan)
    v = np.nanpercentile(np.abs(dT), 99.5) if np.isfinite(dT).any() else 1
    ax[0].imshow(dT, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v, extent=ext, interpolation="nearest")
    ax[0].set_title(f"{name}.csv: ΔT (без синфазной и статической составляющих), °C; рамки — CNN")
    ax[1].imshow(o["score"], aspect="auto", cmap="magma", vmin=0, vmax=1, extent=ext, interpolation="nearest")
    ax[1].set_title("P(аномалия) U-Net; белые рамки — эталон median_x3+spatial_z")
    t0 = p["t"][0]
    for z in o["zones"]:
        ax[0].add_patch(Rectangle((z["start_m"], (z["t_start_s"] - t0) / 3600), max(z["end_m"] - z["start_m"], 0.5), (z["t_end_s"] - z["t_start_s"]) / 3600 + 0.03,
                                  fill=False, ec=COL[z["class"]], lw=1.2 if z["alarm"] else 0.6, ls="-" if z["alarm"] else ":"))
    for z in b["zones"]:
        ax[1].add_patch(Rectangle((z["start_m"], (z["t_start_s"] - t0) / 3600), z["end_m"] - z["start_m"], (z["t_end_s"] - z["t_start_s"]) / 3600, fill=False, ec="white", lw=0.6))
    for k, c in COL.items(): ax[0].plot([], [], color=c, label=k)
    ax[0].legend(fontsize=7, loc="lower right"); ax[0].set_xlabel("x, м"); ax[1].set_xlabel("x, м"); ax[0].set_ylabel("время от начала, ч")
    plt.tight_layout(); plt.savefig(f"{a.fig}/real_{name}.png", dpi=110); plt.close()
    # карточки оператора по этому файлу
    for c in [c for c in cards if str(c.get("record")) == name]:
        import pandas as pd
        E0 = pd.Timestamp("1970-01-01", tz="UTC")
        ts = [(pd.Timestamp(s).tz_convert("UTC") if pd.Timestamp(s).tzinfo else pd.Timestamp(s, tz="UTC")) for s in c["time"]]
        ta, tb = [(s - E0).total_seconds() for s in ts]
        if r["tz_missing"]:
            # наивное время файла трактуется как UTC; карточка — в UTC (может быть сдвиг TZ — отмечаем)
            pass
        xa, xb = c["meters"]
        ix = (p["x"] >= xa) & (p["x"] <= xb); it = (p["t"] >= ta) & (p["t"] <= tb)
        box = np.ix_(it, ix)
        if not ix.any() or not it.any():
            res["cards"].append(dict(id=c["id"], record=name, status="box_outside_record")); continue
        vb = o["valid"][box]
        overl = [z for z in o["zones"] if z["end_m"] >= xa and z["start_m"] <= xb and z["t_end_s"] >= ta and z["t_start_s"] <= tb]
        overl_alarm = [z for z in overl if z["alarm"]]
        bo = [z for z in b["zones"] if z["end_m"] >= xa and z["start_m"] <= xb and z["t_end_s"] >= ta and z["t_start_s"] <= tb]
        dom = None
        if overl:
            dom = max(overl, key=lambda z: z["n_px"])
        res["cards"].append(dict(id=c["id"], record=name, presence=c["presence"], shape=c.get("shape", ""), notes=c.get("notes", ""),
                                 cnn_max_score=float(o["score"][box][vb].max()) if vb.any() else None,
                                 cnn_frac_flagged=float((o["score"][box] >= thr)[vb].mean()) if vb.any() else None,
                                 cnn_n_zones=len(overl), cnn_n_alarm_zones=len(overl_alarm), cnn_dominant_class=dom["class"] if dom else None,
                                 cnn_dominant_polarity=dom["polarity"] if dom else None, base_n_zones=len(bo),
                                 base_polarities=sorted({z["polarity"] for z in bo})))
    print(name, {k: v for k, v in res["files"][name].items() if k != "zones"}, flush=True)

def _presence_hits(presence):
    rows = [c for c in res["cards"] if c.get("presence") == presence and c.get("status") != "box_outside_record"]
    return {
        "n": len(rows),
        "alarm_hit": sum(1 for c in rows if c.get("cnn_n_alarm_zones")),
        "any_zone_hit": sum(1 for c in rows if c.get("cnn_n_zones")),
    }

res["cards_descriptive"] = {
    "yes": _presence_hits("yes"),
    "no": _presence_hits("no"),
    "unclear": _presence_hits("unclear"),
    "note": "Карточки описывают попадание зоны в рамку оператора. Порог и параметры гибрида по карточкам не выбирались и на реальных файлах не калибровались.",
}
if a.detector == "dom" or a.mae == "on" or a.gen2 == "on":
    res["cards_descriptive"]["winner_published"] = {"yes_alarm": "8/15", "no_alarm": "0/5"}
    res["cards_descriptive"]["note"] += " Сравнение с выпуском ens_u2ft_tcn_lgbm (8/15, 0/5) описательное и порог не двигает."
if a.mae == "on" or a.gen2 == "on":
    yes_h, no_h = res["cards_descriptive"]["yes"], res["cards_descriptive"]["no"]
    res["cards_descriptive"]["batch1_batch2"] = {
        "yes_alarm": f"{yes_h['alarm_hit']}/{yes_h['n']}",
        "no_alarm": f"{no_h['alarm_hit']}/{no_h['n']}",
        "winner_yes_alarm": "8/15",
        "winner_no_alarm": "0/5",
    }
    review_cards = []
    if _CARDS2.is_file():
        review_cards = [json.loads(l) for l in open(_CARDS2) if l.strip()]
    review_rows = []
    for c in review_cards:
        name = str(c.get("record"))
        zones = (res["files"].get(name) or {}).get("zones") or []
        try:
            xa, xb, ta, tb = _card_window(c)
        except (KeyError, TypeError, ValueError):
            review_rows.append(dict(id=c.get("id"), record=name, presence=c.get("presence"), status="bad_window"))
            continue
        hits = _alarm_hits(zones, xa, xb, ta, tb)
        review_rows.append(dict(
            id=c.get("id"), record=name, presence=c.get("presence"),
            alarm_hit=bool(hits), n_alarm=len(hits), status="ok",
        ))
    def _rev(presence):
        rows = [r for r in review_rows if r.get("presence") == presence and r.get("status") == "ok"]
        return {"n": len(rows), "alarm_hit": sum(1 for r in rows if r["alarm_hit"])}
    narrow_rows = []
    if _REVIEW2.is_file() and review_rows:
        import csv
        rev_csv = {}
        with open(_REVIEW2, newline="") as f:
            for row in csv.DictReader(f):
                rev_csv[row["id"]] = row
        by_id = {r["id"]: r for r in review_rows}
        for cid, info in _narrow_static(_REVIEW2).items():
            hit = by_id.get(cid)
            alarm = bool(hit and hit.get("alarm_hit"))
            narrow_rows.append(dict(id=cid, alarm_hit=alarm, dT_hi=info["dT_hi"], width_m=info["width_m"], winner=0))
        # победитель на том же наборе — колонка winner в cards2_review, не пересчёт
        def _csv_hits(presence):
            rows = [row for row in rev_csv.values() if (row.get("user") or "").strip() == presence]
            return {"n": len(rows), "winner_alarm": sum(1 for row in rows if str(row.get("winner")) in ("1", "1.0"))}
        res["cards_review2"] = {
            "source": str(_CARDS2),
            "table": "artifacts/eval/cards2_review.csv",
            "yes": _rev("yes"),
            "no": _rev("no"),
            "unclear": _rev("unclear"),
            "winner_from_table": {
                "yes": _csv_hits("yes"),
                "no": _csv_hits("no"),
                "unclear": _csv_hits("unclear"),
            },
            "narrow": {
                "definition": "presence=yes, winner=0, ширина ≤ 12 м, dT_hi ≥ 3 °C. Попадание — тревожная зона этого прогона пересекает рамку. Порог не подбирался.",
                "n": len(narrow_rows),
                "alarm_hit": sum(1 for r in narrow_rows if r["alarm_hit"]),
                "ids": narrow_rows,
            },
            "note": "CARDS_REVIEW_2 описательный. Порог этого прогона по этим рамкам не двигался. Рядом — колонка winner из cards2_review.csv.",
        }
        res["cards_descriptive"]["review2_yes"] = f"{res['cards_review2']['yes']['alarm_hit']}/{res['cards_review2']['yes']['n']}"
        res["cards_descriptive"]["review2_no"] = f"{res['cards_review2']['no']['alarm_hit']}/{res['cards_review2']['no']['n']}"
        res["cards_descriptive"]["review2_unclear"] = f"{res['cards_review2']['unclear']['alarm_hit']}/{res['cards_review2']['unclear']['n']}"
        res["cards_descriptive"]["review2_narrow"] = f"{res['cards_review2']['narrow']['alarm_hit']}/{res['cards_review2']['narrow']['n']}"
        wyes = res["cards_review2"]["winner_from_table"]["yes"]
        wno = res["cards_review2"]["winner_from_table"]["no"]
        wun = res["cards_review2"]["winner_from_table"]["unclear"]
        res["cards_descriptive"]["review2_winner"] = {
            "yes": f"{wyes['winner_alarm']}/{wyes['n']}",
            "no": f"{wno['winner_alarm']}/{wno['n']}",
            "unclear": f"{wun['winner_alarm']}/{wun['n']}",
        }
        res["cards_descriptive"]["note"] += " CARDS_REVIEW_2 (cards_review2.py / cards2_review.csv) посчитан по тем же рамкам рядом с колонкой winner, порог не двигался."
    elif a.mae == "on" or a.gen2 == "on":
        res["cards_review2"] = {"missing": True, "note": "нет jsonl CARDS_REVIEW_2, набор не считался"}
if a.static == "on" and _REVIEW2.is_file():
    meta = _narrow_static(_REVIEW2)
    by_id = {c.get("id"): c for c in res["cards"]}
    rows = []
    for cid, info in meta.items():
        hit = by_id.get(cid)
        alarm = int(hit.get("cnn_n_alarm_zones") or 0) if hit else 0
        score = None if hit is None else hit.get("cnn_max_score")
        thr = float(A.static_spec["threshold"])
        rows.append(dict(
            id=cid, dT_hi=info["dT_hi"], width_m=info["width_m"],
            alarm_zones=alarm,
            alarm_hit=bool(alarm),
            cnn_max_score=score,
            score_ge_thr=bool(score is not None and score >= thr),
            status=None if hit is None else hit.get("status"),
        ))
    res["narrow_static"] = {
        "definition": "presence=yes, winner=0 в cards2_review, ширина рамки ≤ 12 м, dT_hi ≥ 3 °C. Попадание — тревожная зона этого прогона пересекает рамку. Порог synth не подбирался по карточкам.",
        "n": len(rows),
        "alarm_hit": sum(1 for r in rows if r["alarm_hit"]),
        "score_ge_thr": sum(1 for r in rows if r["score_ge_thr"]),
        "ids": rows,
    }
    res["cards_descriptive"]["narrow_static"] = f"{res['narrow_static']['alarm_hit']}/{res['narrow_static']['n']}"
    res["cards_descriptive"]["note"] += " Узкий срез CARDS_REVIEW_2 описательный: порог StaticNet по нему не двигался."
if a.zones == "pdrs_hybrid":
    spec = infer.hybrid_spec()
    res["pdrs_hybrid"] = {
        "mode": spec["mode"],
        "floor": float(spec["floor"]),
        "r_saddle": float(spec["r_saddle"]),
        "region_median": float(spec["region_median"]),
        "fallback": bool(spec["fallback"]),
        "thr": float(A.thr),
        "seed": float(A.thr) if spec["mode"] == "seed" else float(spec.get("seed", A.thr)),
        "note": "Параметры заморожены на v8 val (artifacts/eval/pdrs_hybrid_best_params.json). По iou_all гибрид победителя не заменяет.",
    }
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
print("zones csv:", export.write_csv(a.zones_csv, all_rows), "rows ->", a.zones_csv)
json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False, default=float)
