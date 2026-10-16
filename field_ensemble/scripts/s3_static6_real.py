"""Описательный прогон StaticNet6 по MGTS. Порог не двигает.

Читает чекпойнт и порог из artifacts/static/s3_static6.json.
Вход — сырая температура после канонической сетки, без медианного фильтра по x.
Карточки в выбор не входят.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

_NTHR = os.environ.get("OMP_NUM_THREADS", "3")
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, _NTHR)

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dts import grid, real
from dts.static_net import (
    REF_DILATIONS, TRUNK_DILATIONS, StaticNet6, predict_temperature, temperature_features,
)

LOG = Path("/workspace/run_logs/_events.log")
REAL_ROOT = Path("/workspace/mgts/MGTS_data")
ZONES = ROOT / "artifacts/eval/real_zones.csv"
CARDS = Path(
    "/home/box/agent-data/agents/7620c11d-77b1-4929-aaa8-c80e192177e7/attachments/"
    "66eacdf5a4b0ab8649ecf57a4b35554ad67790401bcf97c035e546eff4f764ff.jsonl"
)
REVIEW = ROOT / "artifacts/eval/cards2_review.csv"
SRC = ROOT / "artifacts/static/s3_static6.json"
OUT = ROOT / "artifacts/static/s3_static6_real.json"


def log(msg: str):
    line = time.strftime("%H:%M") + " " + msg
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def qdict(a):
    a = np.asarray(a, np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None
    qs = (0.5, 0.9, 0.99)
    v = np.quantile(a, qs)
    out = {lab: round(float(x), 4) for lab, x in zip(("p50", "p90", "p99"), v)}
    out["n"] = int(a.size)
    out["max"] = round(float(a.max()), 4)
    return out


def load_alarm_spans(path: Path):
    spans = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("is_alarm") not in ("1", "True", "true"):
                continue
            src = row["source"].replace(".csv", "")
            spans.setdefault(src, []).append((float(row["x_start_m"]), float(row["x_end_m"])))
    return spans


def narrow_ids(path: Path):
    ids = []
    with path.open(newline="") as f:
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
            ids.append(row["id"])
    return ids


def card_box(card):
    import pandas as pd
    xa, xb = card["meters"]
    E0 = pd.Timestamp("1970-01-01", tz="UTC")
    ts = []
    for s in card["time"]:
        t = pd.Timestamp(s)
        t = t.tz_convert("UTC") if t.tzinfo else t.tz_localize("UTC")
        ts.append((t - E0).total_seconds())
    return float(xa), float(xb), float(ts[0]), float(ts[1])


def load_model(path: Path) -> StaticNet6:
    import torch
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
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
    return net


def file_report(rec, spans, model, cards_by_id, wanted, thr):
    raw = real.load_csv(REAL_ROOT / f"{rec}.csv")
    x0, t0, T0, v0 = raw["x"], raw["t"], raw["T"], raw["valid"]
    p = grid.plan(x0, t0)
    Tc, vc = grid.resample(T0, v0, p, "mean")
    s = predict_temperature(model, Tc, vc)
    _, good = temperature_features(
        Tc, vc, max_rows=int(model.max_rows), win_30=int(model.win_30), win_2h=int(model.win_2h),
    )
    xc = p["x"]
    t_used = p["t"]
    inside = np.zeros(len(xc), bool)
    for a, b in spans.get(rec, []):
        inside |= (xc >= a) & (xc <= b)
    outside = ~inside
    boxes = []
    for cid in wanted:
        c = cards_by_id.get(cid)
        if c is None or str(c["record"]) != rec:
            continue
        xa, xb, ta, tb = card_box(c)
        ix = (xc >= xa) & (xc <= xb)
        it = (t_used >= ta) & (t_used <= tb)
        box = dict(
            id=cid,
            width_m=round(xb - xa, 3),
            n_cols=int(ix.sum()),
            n_rows_in_window=int(it.sum()),
            s_static_max=round(float(s[ix].max()), 4) if ix.any() else None,
            s_static_med=round(float(np.median(s[ix])), 4) if ix.any() else None,
            ge_thr=bool(ix.any() and float(s[ix].max()) >= thr),
        )
        if it.any():
            box["time_frac_of_file"] = round(float(it.mean()), 3)
        boxes.append(box)
    rep = dict(
        rows=int(Tc.shape[0]),
        cols=int(Tc.shape[1]),
        s_outside=qdict(s[outside & good]),
        s_inside=qdict(s[inside & good]),
        frac_ge_thr_outside=round(float((s[outside & good] >= thr).mean()), 5) if (outside & good).any() else None,
        frac_ge_thr_inside=round(float((s[inside & good] >= thr).mean()), 5) if (inside & good).any() else None,
        narrow_boxes=boxes,
    )
    print(rec, "rows", Tc.shape[0], "s_out_max",
          None if rep["s_outside"] is None else rep["s_outside"]["max"],
          "boxes", len(boxes), flush=True)
    del raw, T0, Tc
    return rep


def main():
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "3")))
    report = json.loads(SRC.read_text())
    ckpt = ROOT / report["ckpt"]
    thr = float(report["val"][report["selected"]]["threshold"])
    log(f"STATIC6_REAL ckpt={report['ckpt']} thr={thr:.6f} selected={report['selected']} descriptive_only")
    t0 = time.time()
    model = load_model(ckpt)
    spans = load_alarm_spans(ZONES)
    wanted = narrow_ids(REVIEW)
    cards = {}
    if CARDS.is_file():
        for line in CARDS.read_text().splitlines():
            if not line.strip():
                continue
            c = json.loads(line)
            if c.get("id") in wanted:
                cards[c["id"]] = c
    files = {}
    for rec in ("1", "2", "3", "4", "5", "6"):
        files[rec] = file_report(rec, spans, model, cards, wanted, thr)
    boxes = [b for f in files.values() for b in f["narrow_boxes"]]
    ge = [b for b in boxes if b.get("ge_thr")]
    out = dict(
        note="Описание. Порог взят из synth val и не двигался. Вход — сырая T без пространственного детренда. "
             "Карточки в генератор и в выбор не входили.",
        ckpt=report["ckpt"],
        step=report.get("step"),
        selected=report.get("selected"),
        threshold=thr,
        val_iou=report["val"][report["selected"]]["iou"],
        static_val=report["val"][report["selected"]]["static"],
        better_than_static2=report.get("better_than_static2"),
        files=files,
        narrow_descriptive=dict(
            n=len(boxes),
            s_static_ge_thr=len(ge),
            s_static_max_of_boxes=None if not boxes else max(b["s_static_max"] or 0 for b in boxes),
            ids_ge_thr=[b["id"] for b in ge],
            definition="16 рамок: yes, winner=0, ширина ≤ 12 м, dT_hi ≥ 3 °C. Столбец ветви на всём файле.",
        ),
        seconds=round(time.time() - t0, 1),
    )
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("wrote", OUT, f"{len(ge)}/{len(boxes)}", "max",
          out["narrow_descriptive"]["s_static_max_of_boxes"], flush=True)


if __name__ == "__main__":
    main()
