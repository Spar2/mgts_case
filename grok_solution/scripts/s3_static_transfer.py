"""Диагностика переноса synth → real для ветви STATIC. Порог не двигает.

Считает s_static выбранной сети STATIC2 на MGTS 1–6.csv и сверяет генератор
с допущениями ТЗ кейса (+3…+24 °C, длина 3–5 м) и с пятном прибора ~1 м.
Карточки только описывают уже названный срез 16 рамок. Их ширины и амплитуды
в константы генератора не подставляются.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("MKL_NUM_THREADS", "3")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dts import grid, real
from dts.features import robust_sigma_x
from dts.infer import load_static_net, static_ckpt_path, static_spec
from dts.static_net import residual_from_cache, residual_from_temperature, temporal_features, predict_map
from dts.static_synth import DX, HOT_AMP, HOT_WIDTH, hot_spec, sample_train_specs, unit_profile, fwhm_m

# Допущения кейса из ТЗ и физика прибора из docs/NOTE.md. Не оценки по карточкам.
TZ_AMP = (3.0, 24.0)
TZ_WIDTH = (3.0, 5.0)
SPOT_M = 1.0

LOG = Path("/workspace/grok_logs/_events.log")
REAL_ROOT = Path("/workspace/mgts/MGTS_data")
ZONES = ROOT / "artifacts/eval/real_zones.csv"
CARDS = Path("/home/box/agent-data/agents/7620c11d-77b1-4929-aaa8-c80e192177e7/attachments/66eacdf5a4b0ab8649ecf57a4b35554ad67790401bcf97c035e546eff4f764ff.jsonl")
REVIEW = ROOT / "artifacts/eval/cards2_review.csv"
OUT = ROOT / "artifacts/static/s3_static4.json"
QS = (0.5, 0.9, 0.99, 0.999)


def log(msg: str):
    line = time.strftime("%H:%M") + " " + msg
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def qdict(a, qs=QS):
    a = np.asarray(a, np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None
    v = np.quantile(a, qs)
    labels = {0.5: "p50", 0.9: "p90", 0.99: "p99", 0.999: "p999"}
    out = {labels.get(q, str(q)): round(float(x), 4) for q, x in zip(qs, v)}
    out["n"] = int(a.size)
    out["max"] = round(float(a.max()), 4)
    out["min"] = round(float(a.min()), 4)
    return out


def runs(mask: np.ndarray):
    """Длины непрерывных True-отрезков."""
    if mask.size == 0 or not mask.any():
        return np.zeros(0, np.int64)
    d = np.diff(mask.astype(np.int8))
    starts = np.flatnonzero(d == 1) + 1
    ends = np.flatnonzero(d == -1) + 1
    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, mask.size]
    return ends - starts


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


def generator_audit(n=2000, seed=20261015):
    rng = np.random.default_rng(seed)
    widths, amps, fwhms, kinds = [], [], [], []
    spikes, broads = [], []
    for _ in range(n):
        sp, _ = hot_spec(rng, 4000, DX, None)
        widths.append(sp["width_m"])
        amps.append(sp["amp"])
        kinds.append(sp["kind"])
        y = unit_profile(4000, sp)
        fwhms.append(fwhm_m(y, DX))
        specs = sample_train_specs(rng, 4000, DX, None)
        for s in specs:
            if s["role"] != "hot" and s["kind"] == "broad":
                broads.append(s["width_m"])
            elif s["role"] != "hot":
                spikes.append(s["width_m"])
    w = np.asarray(widths)
    a = np.asarray(amps)
    fw = np.asarray(fwhms)
    return {
        "hot_width_code": list(HOT_WIDTH),
        "hot_amp_code": list(HOT_AMP),
        "tz_width_m": list(TZ_WIDTH),
        "tz_amp_C": list(TZ_AMP),
        "spot_m": SPOT_M,
        "n": n,
        "width_min": round(float(w.min()), 3),
        "width_max": round(float(w.max()), 3),
        "width_mean": round(float(w.mean()), 3),
        "frac_width_inside_tz": round(float(((w >= TZ_WIDTH[0]) & (w <= TZ_WIDTH[1])).mean()), 3),
        "amp_min": round(float(a.min()), 3),
        "amp_max": round(float(a.max()), 3),
        "amp_median": round(float(np.median(a)), 3),
        "frac_amp_inside_tz": round(float(((a >= TZ_AMP[0]) & (a <= TZ_AMP[1])).mean()), 3),
        "frac_gaussian": round(float(np.mean([k == "gaussian" for k in kinds])), 3),
        "fwhm_minus_width_p50": round(float(np.median(fw - w)), 3),
        "frac_fwhm_below_spot": round(float((fw < SPOT_M).mean()), 3),
        "spike_width_min": None if not spikes else round(float(np.min(spikes)), 3),
        "spike_width_max": None if not spikes else round(float(np.max(spikes)), 3),
        "frac_spike_narrower_than_spot": None if not spikes else round(float(np.mean(np.asarray(spikes) < SPOT_M)), 3),
        "broad_width_min": None if not broads else round(float(np.min(broads)), 3),
        "note": "Аудит кода генератора, на котором обучена выбранная сеть. Карточки в эти доли не входят.",
    }


def synth_residual_sample(n_files=8):
    files = sorted((ROOT / "artifacts/cache/v8/val").glob("*.npz"))[:n_files]
    meds = []
    sig = []
    for p in files:
        z = np.load(p)
        R, valid = residual_from_cache(z)
        sig.append(np.asarray(z["sigma"], np.float32))
        # медиана по времени, как первый признак ветви, без деления на 10
        feats, good = temporal_features(R, valid, max_rows=256, n_bins=8)
        med = feats[0] * 10.0
        meds.append(med[good])
        z.close()
    cat = np.concatenate(meds) if meds else np.zeros(0)
    sig_c = np.concatenate(sig) if sig else np.zeros(0)
    return {
        "n_files": len(files),
        "column_median_C": qdict(cat),
        "sigma_C": qdict(sig_c),
        "clip_note": "Остаток обучения — канал 1, обрезанный на ±12σ и умноженный обратно на σ.",
    }


def known_bump_transfer():
    """Нагретый участок 4 м, +10 °C на гладком склоне. Карточек здесь нет."""
    rng = np.random.default_rng(0)
    rows, width = 80, 800
    x = np.arange(width) * DX
    slope = (0.02 * (x - x.mean())).astype(np.float32)
    noise = rng.normal(0, 0.4, size=(rows, width)).astype(np.float32)
    T = 20.0 + slope[None, :] + noise
    center = 400
    half = int(round(2.0 / DX))
    T[:, center - half:center + half + 1] += 10.0
    valid = np.ones_like(T, bool)
    resid, rv = residual_from_temperature(T, valid, dx=DX, max_rows=rows)
    col = resid[:, center]
    return {
        "planted_width_m": 4.0,
        "planted_amp_C": 10.0,
        "residual_center_median_C": round(float(np.median(col)), 3),
        "residual_center_p10_C": round(float(np.quantile(col, 0.1)), 3),
        "kept_fraction_of_amp": round(float(np.median(col) / 10.0), 3),
    }


def file_report(rec, spans, model, cards_by_id, wanted):
    raw = real.load_csv(REAL_ROOT / f"{rec}.csv")
    x0, t0, T0, v0 = raw["x"], raw["t"], raw["T"], raw["valid"]
    dx_native = float(np.median(np.diff(x0))) if len(x0) > 1 else None
    dt_native = float(np.median(np.diff(t0))) if len(t0) > 1 else None
    time_gaps = np.diff(t0)
    col_valid = v0.mean(axis=0)
    invalid_cols = col_valid < 0.2
    gap_px = runs(invalid_cols)
    p = grid.plan(x0, t0)
    Tc, vc = grid.resample(T0, v0, p, "mean")
    # σ по времени на канонической сетке: шум растёт к дальнему концу, это не метка
    sig = robust_sigma_x(np.where(vc, Tc, np.nan), vc)
    # Тот же прореженный набор кадров, что у слияния: residual_from_temperature режет до max_rows.
    max_rows = int(model.max_rows)
    if Tc.shape[0] > max_rows:
        idx = np.linspace(0, Tc.shape[0] - 1, max_rows).astype(np.int64)
    else:
        idx = np.arange(Tc.shape[0])
    resid, rv = residual_from_temperature(Tc[idx], vc[idx], dx=float(p["dx"]), max_rows=max_rows)
    s = predict_map(model, resid, rv)
    xc = p["x"]
    t_used = p["t"][idx]
    inside = np.zeros(len(xc), bool)
    for a, b in spans.get(rec, []):
        inside |= (xc >= a) & (xc <= b)
    outside = ~inside
    feats, good = temporal_features(resid, rv, max_rows=int(model.max_rows), n_bins=int(model.n_bins))
    med_c = feats[0] * 10.0
    # тренд: медиана T по времени минус остаток ветви. Крупный ход по x, не событие.
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tmed = np.nanmedian(np.where(vc, Tc, np.nan), axis=0)
    trend = np.where(np.isfinite(tmed), tmed, np.nan)
    trend_span = None if not np.isfinite(trend).any() else round(float(np.nanquantile(trend, 0.99) - np.nanquantile(trend, 0.01)), 3)
    # насколько тренд круче 40 м: разность соседних метров у сглаженного профиля
    finite_t = np.isfinite(trend)
    slope = np.diff(trend[finite_t]) / DX if finite_t.sum() > 2 else np.zeros(0)
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
            valid_frac=round(float(rv[it][:, ix].mean()), 3) if ix.any() and it.any() else None,
            s_static_max=round(float(s[ix].max()), 4) if ix.any() else None,
            s_static_med=round(float(np.median(s[ix])), 4) if ix.any() else None,
            column_median_full_C=round(float(np.median(med_c[ix])), 3) if ix.any() else None,
            ge_thr_cols=int((s[ix] >= 0.865895).sum()) if ix.any() else 0,
        )
        if ix.any() and it.any():
            window = resid[it][:, ix]
            wvalid = rv[it][:, ix]
            if wvalid.any():
                box["residual_window_median_C"] = round(float(np.median(window[wvalid])), 3)
            else:
                box["residual_window_median_C"] = None
            # доля кадров файла, где медиана по x рамки выше 3 °C — долго ли пятно стоит
            col_med_t = np.nanmedian(np.where(rv[:, ix], resid[:, ix], np.nan), axis=1)
            box["frac_rows_median_ge_3C"] = round(float(np.nanmean(col_med_t >= 3.0)), 3)
            box["time_frac_of_file"] = round(float(it.mean()), 3)
        boxes.append(box)
    rep = dict(
        n_rows_raw=int(len(t0)),
        n_x_raw=int(len(x0)),
        n_rows_grid=int(Tc.shape[0]),
        n_x_grid=int(Tc.shape[1]),
        dx_native_m=None if dx_native is None else round(dx_native, 4),
        dt_native_s=None if dt_native is None else round(dt_native, 1),
        dx_grid_m=float(p["dx"]),
        dt_grid_s=float(p["dt"]),
        valid_frac_raw=round(float(v0.mean()), 4),
        valid_frac_grid=round(float(vc.mean()), 4),
        row_ok_frac=round(float(p["row_ok"].mean()), 4),
        invalid_col_frac=round(float(invalid_cols.mean()), 4),
        gap_run_px=qdict(gap_px.astype(np.float64)) if gap_px.size else {"n": 0},
        time_gap_gt_300s=int((time_gaps > 300).sum()) if time_gaps.size else 0,
        sigma_C=qdict(sig),
        sigma_far_over_near=None,
        trend_p01_p99_C=trend_span,
        trend_slope_C_per_m=qdict(slope),
        s_all=qdict(s[good]),
        s_inside_alarm=qdict(s[inside & good]),
        s_outside_alarm=qdict(s[outside & good]),
        frac_s_ge_0_5_outside=round(float((s[outside & good] >= 0.5).mean()), 5) if (outside & good).any() else None,
        frac_s_ge_thr_outside=round(float((s[outside & good] >= 0.865895).mean()), 5) if (outside & good).any() else None,
        frac_s_ge_thr_inside=round(float((s[inside & good] >= 0.865895).mean()), 5) if (inside & good).any() else None,
        n_alarm_cols=int((inside & good).sum()),
        n_outside_cols=int((outside & good).sum()),
        column_median_outside_C=qdict(med_c[outside & good]),
        column_median_inside_C=qdict(med_c[inside & good]),
        narrow_boxes=boxes,
    )
    near = sig[: max(1, len(sig) // 10)]
    far = sig[-max(1, len(sig) // 10):]
    if np.median(near) > 0:
        rep["sigma_far_over_near"] = round(float(np.median(far) / np.median(near)), 3)
    print(rec, "s_out", rep["s_outside_alarm"], "boxes", len(boxes), flush=True)
    del raw, T0, Tc, resid
    return rep


def main():
    import torch
    torch.set_num_threads(3)
    t0 = time.time()
    spec = static_spec()
    log(f"STATIC4_START ckpt={spec['ckpt']} thr={spec['threshold']:.6f} mode={spec['mode']}")
    model = load_static_net(static_ckpt_path(spec))
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
        files[rec] = file_report(rec, spans, model, cards, wanted)
    out = dict(
        note="Диагностика. Порог 0.865895 заморожен с synth v8 val. Карточки и CSV его не двигают и в генератор не входят.",
        threshold=float(spec["threshold"]),
        ckpt=spec["ckpt"],
        tz=dict(amp_C=list(TZ_AMP), width_m=list(TZ_WIDTH), spot_m=SPOT_M),
        generator=generator_audit(),
        synth_residual=synth_residual_sample(),
        planted_bump=known_bump_transfer(),
        files=files,
        seconds=round(time.time() - t0, 1),
    )
    # сводка по 16 рамкам: только описание, не подгонка
    boxes = [b for f in files.values() for b in f["narrow_boxes"]]
    ge = [b for b in boxes if (b.get("s_static_max") or 0) >= float(spec["threshold"])]
    out["narrow_descriptive"] = dict(
        n=len(boxes),
        s_static_ge_thr=len(ge),
        s_static_max_of_boxes=None if not boxes else max(b["s_static_max"] or 0 for b in boxes),
        definition="16 рамок CARDS_REVIEW_2: yes, winner=0, ширина ≤ 12 м, dT_hi ≥ 3 °C. s_static — столбец ветви на всём файле.",
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("wrote", OUT, "narrow s_static", f"{len(ge)}/{len(boxes)}", "sec", out["seconds"], flush=True)


if __name__ == "__main__":
    main()
