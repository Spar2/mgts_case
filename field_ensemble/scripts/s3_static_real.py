"""Описательный прогон статичного канала по MGTS 1–6.csv.

Порог, гейт и режим слияния берутся из artifacts/static/s3_static.json
(калибровка только на synth v8 val). Карточки и сами CSV не двигают порог,
не входят в обучение и не используются как метки. Сравнение с разбором
CARDS_REVIEW_2 — только текст и счётчики попаданий в уже размеченные рамки.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("MKL_NUM_THREADS", "3")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dts import grid, real
from dts.static_net import StaticNet, predict_map, residual_from_temperature


def _load(path: Path) -> StaticNet:
    import torch
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = StaticNet(
        cin=int(ckpt.get("cin", 9)), width=int(ckpt.get("width", 16)), pool=int(ckpt.get("pool", 80)),
        max_rows=int(ckpt.get("max_rows", 64)), n_bins=int(ckpt.get("n_bins", 0)),
    )
    net.load_state_dict(ckpt["state"])
    net.eval()
    return net


def subsample_rows(T, valid, t, n=160):
    if len(t) <= n:
        return T, valid, t
    idx = np.linspace(0, len(t) - 1, n).astype(np.int64)
    return T[idx], valid[idx], t[idx]


def on_grid(T, valid, x, t):
    """Каноническая сетка dx=0.25. Время уже прорежено, повторный dt только выравнивает ось."""
    p = grid.plan(x, t)
    Tc, v = grid.resample(T, valid, p, "mean")
    return Tc, v, p["x"], p["t"]


def column_score(model, T, valid, x, t, max_rows=160):
    Ts, vs, ts = subsample_rows(T, valid, t, max_rows)
    Tc, vc, xc, tc = on_grid(Ts, vs, x, ts)
    resid, rv = residual_from_temperature(Tc, vc, dx=float(grid.DX_M), max_rows=max_rows)
    p = predict_map(model, resid, rv)
    return p, xc, tc


def peaks(p, x, thr):
    if len(p) < 3:
        return []
    d = np.diff(p)
    idx = np.where((d[:-1] > 0) & (d[1:] <= 0) & (p[1:-1] >= thr))[0] + 1
    out = []
    for i in idx:
        out.append(dict(x_m=round(float(x[i]), 2), p=round(float(p[i]), 3)))
    out.sort(key=lambda z: -z["p"])
    return out[:30]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-root", default="/workspace/mgts/MGTS_data")
    ap.add_argument("--ckpt", default="artifacts/static/static_net.pt")
    ap.add_argument("--metrics", default="artifacts/static/s3_static.json")
    ap.add_argument("--cards", default="/home/box/agent-data/agents/7620c11d-77b1-4929-aaa8-c80e192177e7/attachments/66eacdf5a4b0ab8649ecf57a4b35554ad67790401bcf97c035e546eff4f764ff.jsonl")
    ap.add_argument("--review", default="artifacts/eval/cards2_review.csv")
    ap.add_argument("--out", default="artifacts/eval/real_static.json")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(3)
    metrics = json.loads((ROOT / a.metrics).read_text())
    name = metrics["selected"]
    spec = metrics["val"][name]
    thr = float(spec["threshold"])
    mode, scale, gate = spec["mode"], float(spec["scale"]), float(spec["gate"])
    model = _load(ROOT / a.ckpt)
    review = pd.read_csv(ROOT / a.review)
    cards = [json.loads(l) for l in open(a.cards) if l.strip()]
    by_id = {c["id"]: c for c in cards}
    rev_by_id = {r.id: r for r in review.itertuples()}
    out = dict(
        note="Только описание. Порог заморожен с synth v8 val, карточки не калибруют модель.",
        selected=name, threshold=thr, mode=mode, scale=scale, gate=gate,
        files={}, cards=[],
    )
    t0 = time.time()
    for rec in sorted({str(c["record"]) for c in cards}):
        path = Path(a.real_root) / f"{rec}.csv"
        print("load", path.name, flush=True)
        raw = real.load_csv(path)
        p_all, x_all, _ = column_score(model, raw["T"], raw["valid"], raw["x"], raw["t"], max_rows=180)
        # гейт как при слиянии: слабый p не считается срабатыванием канала
        p_use = np.where(p_all >= gate, p_all * scale, 0.0) if gate > 0 and mode == "max" else p_all * (scale if mode == "max" else 1.0)
        if mode == "blend":
            # без карты ансамбля blend не определён; для описания берём сам p
            p_use = p_all
        flagged = p_use >= thr
        file_cards = []
        for c in cards:
            if str(c["record"]) != rec:
                continue
            rv = rev_by_id.get(c["id"])
            xa, xb = c["meters"]
            E0 = pd.Timestamp("1970-01-01", tz="UTC")
            ts = []
            for s in c["time"]:
                t = pd.Timestamp(s)
                t = t.tz_convert("UTC") if t.tzinfo else t.tz_localize("UTC")
                ts.append((t - E0).total_seconds())
            ta, tb = ts
            it = (raw["t"] >= ta) & (raw["t"] <= tb)
            if it.sum() < 4:
                # окно короче четырёх кадров — берём ближайшие
                mid = 0.5 * (ta + tb)
                order = np.argsort(np.abs(raw["t"] - mid))[:24]
                it = np.zeros(len(raw["t"]), bool)
                it[order] = True
            p_win, x_win, _ = column_score(model, raw["T"][it], raw["valid"][it], raw["x"], raw["t"][it], max_rows=96)
            if gate > 0 and mode == "max":
                p_win_use = np.where(p_win >= gate, p_win * scale, 0.0)
            elif mode == "max":
                p_win_use = p_win * scale
            else:
                p_win_use = p_win
            ix = (x_win >= xa) & (x_win <= xb)
            width = float(xb - xa)
            row = dict(
                id=c["id"], record=rec, presence=c.get("presence"), shape=c.get("shape") or "",
                width_m=round(width, 2),
                winner=None if rv is None else int(rv.winner),
                baseline=None if rv is None else int(rv.baseline),
                dT_hi=None if rv is None else float(rv.dT_hi) if pd.notna(rv.dT_hi) else None,
                p_window_max=float(p_win_use[ix].max()) if ix.any() else None,
                p_file_max=float(p_use[(x_all >= xa) & (x_all <= xb)].max()) if ((x_all >= xa) & (x_all <= xb)).any() else None,
            )
            row["static_hit_window"] = bool(row["p_window_max"] is not None and row["p_window_max"] >= thr)
            file_cards.append(row)
            out["cards"].append(row)
        n_flag_cols = int(flagged.sum())
        out["files"][rec] = dict(
            n_rows=int(len(raw["t"])), n_x=int(len(x_all)),
            frac_columns_ge_thr=float(flagged.mean()) if len(p_use) else None,
            n_columns_ge_thr=n_flag_cols,
            p_max=float(p_use.max()) if len(p_use) else None,
            peaks=peaks(p_use, x_all, thr),
            n_cards=len(file_cards),
        )
        print(rec, "cols", n_flag_cols, "pmax", round(float(p_use.max()), 3), "cards", len(file_cards), flush=True)
        del raw
    # узкие яркие пропуски победителя — тот самый изъян из CARDS_REVIEW_2
    narrow = []
    for row in out["cards"]:
        if row["presence"] != "yes" or row["winner"] != 0:
            continue
        if row["width_m"] is None or row["width_m"] > 12:
            continue
        if row["dT_hi"] is None or row["dT_hi"] < 3:
            continue
        narrow.append(row)
    hit = sum(1 for r in narrow if r["static_hit_window"])
    out["narrow_hot_missed_by_winner"] = dict(
        definition="yes, winner=0, ширина рамки ≤ 12 м, dT_hi ≥ 3 °C. Попадание — max p в рамке на её же времени ≥ порога synth.",
        n=len(narrow),
        static_hit=hit,
        ids=[dict(id=r["id"], p=r["p_window_max"], dT_hi=r["dT_hi"], width_m=r["width_m"], baseline=r["baseline"]) for r in narrow],
    )
    out["seconds"] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    # путь относительно корня, если out относительный
    dest = Path(a.out)
    if not dest.is_absolute():
        dest = ROOT / dest
    dest.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("wrote", dest, "narrow", f"{hit}/{len(narrow)}", flush=True)


if __name__ == "__main__":
    main()
