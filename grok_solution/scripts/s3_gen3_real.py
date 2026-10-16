"""Описательные карточки GEN3. Порог — synth hold41 из s3_gen3.json.

Реальные CSV не двигают порог и не входят в обучение.
Не пишет docs/REAL_EXPERT_REPORT.md, real_v8.json, real_zones.csv и docs/figures.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "2"

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from dts import features, grid, models, real
from dts.infer import Analyzer, pipeline_zones
from dts import postproc as PP

EVAL = Path("artifacts/eval/s3_gen3.json")
OUT = Path("artifacts/eval/real_gen3.json")
GEN2_PT = Path("artifacts/eval/s3_gen2_unet.pt")
GEN3_PT = Path("artifacts/eval/s3_gen3_unet.pt")
BANNED = {
    (ROOT / "docs/REAL_EXPERT_REPORT.md").resolve(),
    (ROOT / "artifacts/eval/real_v8.json").resolve(),
    (ROOT / "artifacts/eval/real_zones.csv").resolve(),
    (ROOT / "docs/figures").resolve(),
}
CARDS = [
    ROOT / "data/operator_cards/cards_batch1.jsonl",
    ROOT / "data/operator_cards/cards_batch2.jsonl",
]
REAL_ROOT = Path("/workspace/mgts/MGTS_data")


def log(msg: str) -> None:
    import time
    print(time.strftime("%H:%M:%S", time.gmtime()) + " " + msg, flush=True)


def load_unet(path: Path):
    net = models.build("unet", 16)
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    net.eval()
    return net


def member_maps(az, X):
    from dts import classic
    ss, hh = [], []
    for kind, mdl in az.members:
        if kind == "net":
            prob = models.predict_full(mdl, X)
        else:
            prob = np.moveaxis(
                mdl.predict_proba(classic.pixel_features(X)).reshape(X.shape[1], X.shape[2], 3),
                -1, 0,
            )
        ss.append(1.0 - prob[0])
        hh.append(prob[1] - prob[2])
    return np.mean(ss, 0).astype(np.float32), np.mean(hh, 0).astype(np.float32)


def unet_maps(net, X):
    prob = models.predict_full(net, X)
    return (1.0 - prob[0]).astype(np.float32), (prob[1] - prob[2]).astype(np.float32)


def blend(name, winner_s, winner_h, g2s, g2h, g3s, g3h):
    ens_s = 0.5 * (g2s + winner_s)
    ens_h = 0.5 * (g2h + winner_h)
    if name == "gen3":
        return g3s, g3h
    if name == "mean_w":
        return 0.5 * (g3s + winner_s), 0.5 * (g3h + winner_h)
    if name == "mean_g":
        return 0.5 * (g3s + ens_s), 0.5 * (g3h + ens_h)
    if name == "ens_gen2":
        return ens_s, ens_h
    raise KeyError(name)


def card_window(card):
    e0 = pd.Timestamp("1970-01-01", tz="UTC")
    ts = []
    for s in card["time"]:
        t = pd.Timestamp(s)
        t = t.tz_convert("UTC") if t.tzinfo else t.tz_localize("UTC")
        ts.append(float((t - e0).total_seconds()))
    return float(card["meters"][0]), float(card["meters"][1]), ts[0], ts[1]


def alarm_hits(zones, xa, xb, ta, tb):
    return [
        z for z in zones
        if z.get("alarm") and z["end_m"] >= xa and z["start_m"] <= xb
        and z["t_end_s"] >= ta and z["t_start_s"] <= tb
    ]


def main() -> None:
    if OUT.resolve() in BANNED:
        raise SystemExit("отказ писать запертый отчёт или real_v8")
    if not EVAL.is_file():
        raise SystemExit(f"нет {EVAL}")
    report = json.loads(EVAL.read_text())
    name = str(report["selected"])
    thr = float(report["thresholds"][name])
    torch.set_num_threads(2)
    az = Analyzer(op="far_1e-3", gen2="off", static="off", mae="off")
    if az.gen2 != "off" or abs(az.thr - 0.8608087301264272) > 1e-12:
        raise SystemExit("Analyzer по умолчанию сдвинул выпуск")
    g2 = load_unet(GEN2_PT)
    g3 = load_unet(GEN3_PT)
    cards = []
    for path in CARDS:
        for line in path.read_text().splitlines():
            if line.strip():
                cards.append(json.loads(line))
    by_file = {}
    files = sorted(REAL_ROOT.glob("*.csv"))
    if len(files) != 6:
        raise SystemExit(f"ожидались 6 CSV, найдено {len(files)}")
    log(f"real descriptive {name} thr {thr:.6f} n_csv {len(files)}")
    for path in files:
        raw = real.load_csv(path)
        p = grid.plan(raw["x"], raw["t"])
        tc, valid = grid.resample(raw["T"], raw["valid"], p, "mean")
        X, aux = features.make_channels(tc, valid)
        ws, wh = member_maps(az, X)
        g2s, g2h = unet_maps(g2, X)
        g3s, g3h = unet_maps(g3, X)
        score, hot = blend(name, ws, wh, g2s, g2h, g3s, g3h)
        score = np.asarray(score, np.float32)
        hot = np.asarray(hot, np.float32)
        if PP.resolve_enabled(None, "far_1e-3"):
            score = PP.score_for_analyzer(score, valid, thr, True)
        zones = pipeline_zones(score, hot, valid, p["x"], p["t"], thr, "cc")
        probs = az.classify(X, aux["sigma_x"], p["x"], p["t"], zones)
        from dts import events as E
        for z, pr in zip(zones, probs):
            z["class"] = E.CLASSES[int(pr.argmax())]
            z["alarm"] = z["class"] not in ("HVAC_Diurnal", "Sensor_Artifact")
        by_file[raw["name"]] = zones
        log(f"file {raw['name']} zones {len(zones)} alarm {sum(bool(z['alarm']) for z in zones)}")
        del X, score, hot, ws, g2s, g3s
    rows = []
    for card in cards:
        rec = str(card.get("record"))
        zones = by_file.get(rec)
        if zones is None:
            rows.append(dict(id=card.get("id"), record=rec, presence=card.get("presence"), status="missing_file"))
            continue
        xa, xb, ta, tb = card_window(card)
        hits = alarm_hits(zones, xa, xb, ta, tb)
        rows.append(dict(
            id=card.get("id"), record=rec, presence=card.get("presence"),
            status="ok", alarm_hit=bool(hits), n_alarm=len(hits),
        ))
    def count(presence):
        subset = [r for r in rows if r.get("presence") == presence and r.get("status") == "ok"]
        return dict(n=len(subset), alarm_hit=sum(1 for r in subset if r["alarm_hit"]))
    out = dict(
        name="GEN3_REAL",
        candidate=name,
        threshold=thr,
        threshold_source="synth hold41 FAR 1e-3 from artifacts/eval/s3_gen3.json",
        real_used_for_training=False,
        real_used_for_threshold=False,
        cards_moved_threshold=False,
        files={
            k: dict(n_zones=len(v), n_alarm=sum(bool(z["alarm"]) for z in v))
            for k, v in by_file.items()
        },
        batch=dict(yes=count("yes"), no=count("no"), unclear=count("unclear")),
        published_gen2_batch="yes 7/15 no 0/5",
        published_winner_batch="yes 8/15 no 0/5",
        rows=rows,
    )
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, OUT)
    yes, no = out["batch"]["yes"], out["batch"]["no"]
    log(f"записан {OUT} yes {yes['alarm_hit']}/{yes['n']} no {no['alarm_hit']}/{no['n']}")


if __name__ == "__main__":
    main()
