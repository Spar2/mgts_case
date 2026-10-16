"""Диагностика сдвига synth→real и пофайловый порог без меток.

Карточки оператора и сырые CSV МГТС сюда не входят: функции считают статистики
поля и пересобирают зоны по уже полученной карте score. Калибровка порога
использует только валидные клетки того же файла.
"""
from __future__ import annotations

import numpy as np

from . import events as E
from . import features, grid, metrics

FARS = (1e-3, 3e-3, 1e-2)
GLOBAL_OP = "far_1e-3"


def flag_threshold(score: np.ndarray, valid: np.ndarray, far: float) -> float:
    """Квантиль score на валидных клетках: доля флагов равна far.

    Меток нет, поэтому «фон» файла — все валидные клетки. Редкие очаги
    чуть завышают порог; это и есть пофайловая CFAR-калибровка.
    """
    s = np.asarray(score, np.float64)[np.asarray(valid, bool)]
    s = s[np.isfinite(s)]
    if s.size == 0:
        return 1.0
    return metrics.far_threshold(s, float(far))


def flag_fraction(score: np.ndarray, valid: np.ndarray, thr: float) -> float:
    v = np.asarray(valid, bool) & np.isfinite(score)
    if not v.any():
        return 0.0
    return float((score[v] >= thr).mean())


def annotate_zones(analyzer, X, sigma, x, t, zones, dT):
    """Те же поля зоны, что пишет Analyzer.run после classify."""
    P = analyzer.classify(X, sigma, x, t, zones)
    for z, pr in zip(zones, P):
        z["class"] = E.CLASSES[int(np.argmax(pr))]
        z["class_prob"] = {c: round(float(q), 3) for c, q in zip(E.CLASSES, pr)}
        z["alarm"] = z["class"] not in ("HVAC_Diurnal", "Sensor_Artifact")
        sl = (slice(z["t0"], z["t1"]), slice(z["x0"], z["x1"]))
        block = np.abs(dT[sl])
        z["peak_dT_C"] = float(np.max(block)) if block.size else 0.0
    return zones


def rethreshold(analyzer, out: dict, thr: float) -> dict:
    """Новые зоны при другом пороге. Карта score не пересчитывается."""
    p = out["grid"]
    zones = E.zones_from_prob(out["score"], out["hotness"], out["valid"], p["x"], p["t"], thr)
    annotate_zones(analyzer, out["X"], out["sigma"], p["x"], p["t"], zones, out["dT"])
    new = dict(out)
    new["zones"] = zones
    new["threshold"] = float(thr)
    return new


def ensemble_maps(analyzer, X: np.ndarray, log: str | None = None):
    """Среднее score/hotness членов Analyzer. Тот же цикл, что Analyzer.run."""
    import time
    ss, hh = [], []
    for i, (kind, mdl) in enumerate(analyzer.members):
        t0 = time.time()
        label = type(mdl).__name__ if kind == "net" else kind
        if log:
            print(f"{log} member {i} {label} start", flush=True)
        if kind == "net":
            from . import models
            prob = models.predict_full(mdl, X)
        else:
            from . import classic
            prob = np.moveaxis(mdl.predict_proba(classic.pixel_features(X)).reshape(X.shape[1], X.shape[2], 3), -1, 0)
        if log:
            print(f"{log} member {i} {label} {time.time() - t0:.1f}s", flush=True)
        ss.append(1.0 - prob[0])
        hh.append(prob[1] - prob[2])
    return np.mean(ss, 0).astype(np.float32), np.mean(hh, 0).astype(np.float32)


def infer_field(analyzer, T, valid, x, t, sigma_scale: float = 1.0, log: str | None = None) -> dict:
    """Инференс релиза. sigma_scale > 1 уменьшает каналы: σ_used = σ(x) * scale."""
    p = grid.plan(x, t)
    Tc, v = grid.resample(T, valid, p, "mean")
    X, aux = features.make_channels(Tc, v, dx=float(p["dx"]))
    scale = float(sigma_scale)
    if scale != 1.0:
        X = np.array(X, copy=True)
        X[:2] /= np.float32(scale)
    score, hot = ensemble_maps(analyzer, X, log=log)
    return dict(grid=p, X=X, valid=v, score=score, hotness=hot, sigma=aux["sigma_x"] * scale,
                dT=aux["dT_time"], sigma_scale=scale, channels_unscaled=scale == 1.0)


def sigma_scale_factor(real_sigma: np.ndarray, synth_sigma: np.ndarray) -> float:
    """Во сколько раз медианный шум real громче synth. 1 — масштабы совпали."""
    a = np.asarray(real_sigma, np.float64)
    b = np.asarray(synth_sigma, np.float64)
    a = a[np.isfinite(a) & (a > 0)]
    b = b[np.isfinite(b) & (b > 0)]
    if a.size == 0 or b.size == 0:
        return 1.0
    return float(np.median(a) / np.median(b))


def channel_mad(X: np.ndarray, valid: np.ndarray, chan: int = 0) -> float:
    v = np.asarray(valid, bool)
    z = np.asarray(X[chan], np.float64)[v]
    z = z[np.isfinite(z)]
    if z.size == 0:
        return float("nan")
    med = np.median(z)
    return float(1.4826 * np.median(np.abs(z - med)))


def quantile_summary(a: np.ndarray, qs=(50, 90, 99, 99.9)) -> dict:
    x = np.asarray(a, np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    pct = np.percentile(x, qs)
    keys = {50: "p50", 90: "p90", 99: "p99", 99.9: "p999"}
    return {"n": int(x.size), **{keys[q]: float(v) for q, v in zip(qs, pct)},
            "mean": float(x.mean()), "std": float(x.std())}


def _finite_traces(field: np.ndarray, valid: np.ndarray, axis: int, n_traces: int, min_len: int, rng: np.random.Generator):
    field = np.asarray(field, np.float64)
    valid = np.asarray(valid, bool) & np.isfinite(field)
    n = field.shape[1 - axis]
    take = min(n_traces, n)
    if take <= 0:
        return []
    idx = rng.choice(n, size=take, replace=False) if take < n else np.arange(n)
    out = []
    for i in idx:
        if axis == 0:
            col, m = field[:, i], valid[:, i]
        else:
            col, m = field[i], valid[i]
        if int(m.sum()) < min_len:
            continue
        # дырки не интерполируем: берём самую длинную сплошную серию
        seg = _longest_true_run(col, m)
        if seg is not None and len(seg) >= min_len:
            out.append(seg)
    return out


def _longest_true_run(values, mask):
    best = None
    start = None
    for i, ok in enumerate(mask):
        if ok and start is None:
            start = i
        if (not ok or i == len(mask) - 1) and start is not None:
            end = i + 1 if ok else i
            seg = values[start:end]
            if best is None or len(seg) > len(best):
                best = seg
            start = None
    return best


def acf_1d(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    if x.size < max_lag + 2:
        return np.full(max_lag + 1, np.nan)
    x = x - x.mean()
    nfft = 1 << int(np.ceil(np.log2(2 * x.size - 1)))
    spec = np.fft.rfft(x, nfft)
    ac = np.fft.irfft(spec * np.conj(spec), nfft)[: max_lag + 1]
    if not np.isfinite(ac[0]) or ac[0] <= 0:
        return np.full(max_lag + 1, np.nan)
    return (ac / ac[0]).astype(np.float64)


def mean_acf(field, valid, axis: int, max_lag: int, n_traces: int, rng: np.random.Generator, min_len: int = 32) -> np.ndarray:
    rows = []
    for seg in _finite_traces(field, valid, axis, n_traces, max(min_len, max_lag + 2), rng):
        a = acf_1d(seg, max_lag)
        if np.isfinite(a).all():
            rows.append(a)
    if not rows:
        return np.full(max_lag + 1, np.nan)
    return np.mean(rows, 0)


def mean_psd(field, valid, axis: int, n_bins: int, n_traces: int, rng: np.random.Generator, min_len: int = 32) -> np.ndarray:
    """Средний периодограмм, нормированный на свою энергию, на сетке частот 0…0.5."""
    acc = []
    grid = np.linspace(0.0, 0.5, n_bins)
    for seg in _finite_traces(field, valid, axis, n_traces, min_len, rng):
        seg = seg - seg.mean()
        power = np.abs(np.fft.rfft(seg)) ** 2
        energy = power.sum()
        if energy <= 0:
            continue
        freq = np.fft.rfftfreq(len(seg))
        acc.append(np.interp(grid, freq, power / energy))
    if not acc:
        return np.full(n_bins, np.nan)
    return np.mean(acc, 0)


def reservoir_update(hold: np.ndarray, n_seen: int, chunk: np.ndarray, rng: np.random.Generator) -> int:
    """Algorithm R. hold — буфер фиксированной длины, n_seen — сколько точек уже прошло."""
    cap = len(hold)
    chunk = np.asarray(chunk, np.float32).ravel()
    chunk = chunk[np.isfinite(chunk)]
    if chunk.size == 0:
        return n_seen
    if n_seen < cap:
        take = min(cap - n_seen, chunk.size)
        hold[n_seen:n_seen + take] = chunk[:take]
        n_seen += take
        chunk = chunk[take:]
    if chunk.size == 0:
        return n_seen
    idx = np.arange(n_seen + 1, n_seen + chunk.size + 1)
    chosen = rng.random(chunk.size) < (cap / idx)
    n_take = int(chosen.sum())
    if n_take:
        hold[rng.integers(0, cap, size=n_take)] = chunk[chosen]
    return n_seen + int(chunk.size)


def overlapping_zones(zones, xa, xb, ta, tb, alarm_only: bool):
    hit = []
    for z in zones:
        if alarm_only and not z.get("alarm", True):
            continue
        if z["end_m"] >= xa and z["start_m"] <= xb and z["t_end_s"] >= ta and z["t_start_s"] <= tb:
            hit.append(z)
    return hit


def card_bounds(card: dict):
    import pandas as pd
    E0 = pd.Timestamp("1970-01-01", tz="UTC")
    ts = []
    for s in card["time"]:
        t = pd.Timestamp(s)
        t = t.tz_convert("UTC") if t.tzinfo else t.tz_localize("UTC")
        ts.append(t)
    ta, tb = [(s - E0).total_seconds() for s in ts]
    xa, xb = card["meters"]
    return float(min(xa, xb)), float(max(xa, xb)), float(min(ta, tb)), float(max(ta, tb))


def score_card(card, x, t, valid, score, zones, thr, base_zones):
    xa, xb, ta, tb = card_bounds(card)
    ix = (x >= xa) & (x <= xb)
    it = (t >= ta) & (t <= tb)
    if not ix.any() or not it.any():
        return dict(id=card.get("id"), record=str(card.get("record")), presence=card.get("presence"),
                    status="box_outside_record", hit_alarm=False, hit_zone=False, hit_pixel=False, hit_base=False)
    box = np.ix_(it, ix)
    vb = valid[box]
    pix = float((score[box][vb] >= thr).mean()) if vb.any() else 0.0
    alarm = overlapping_zones(zones, xa, xb, ta, tb, True)
    any_z = overlapping_zones(zones, xa, xb, ta, tb, False)
    base = overlapping_zones(base_zones, xa, xb, ta, tb, False)
    return dict(id=card.get("id"), record=str(card.get("record")), presence=card.get("presence"),
                status="ok", cnn_frac_flagged=pix, cnn_n_alarm_zones=len(alarm), cnn_n_zones=len(any_z),
                base_n_zones=len(base), hit_alarm=bool(alarm), hit_zone=bool(any_z), hit_pixel=bool(pix > 0),
                hit_base=bool(base),
                cnn_dominant_class=(max(alarm or any_z, key=lambda z: z.get("n_px", 0)).get("class") if (alarm or any_z) else None))


def presence_tally(rows, key: str) -> dict:
    yes = [r for r in rows if r.get("presence") == "yes" and r.get("status") == "ok"]
    no = [r for r in rows if r.get("presence") == "no" and r.get("status") == "ok"]
    return {f"yes_{key}": int(sum(bool(r.get(key)) for r in yes)), "yes_n": len(yes),
            f"no_{key}": int(sum(bool(r.get(key)) for r in no)), "no_n": len(no)}


def augmentation_recipe(real_row: dict, synth_row: dict) -> dict:
    """Параметры noise-matched аугментации для агента ae.

    Сеть ae учится на каналах ΔT/σ. Масштаб σ(x) уже снимается в make_channels,
    поэтому рецепт добивает остаток: во сколько раз шум real громче, и какая
    корреляция остаётся на сетке 120 с × 0,25 м.
    """
    rs = float(real_row["sigma_p50"])
    ss = float(synth_row["sigma_p50"])
    rc = float(real_row["ch0_mad"])
    sc = float(synth_row["ch0_mad"])
    ratio = rs / ss if ss > 0 else 1.0
    ch_ratio = rc / sc if sc > 0 else 1.0
    phi_t = float(real_row["acf_t_lag1"])
    phi_x = float(real_row["acf_x_lag1"])
    phi_t_s = float(synth_row["acf_t_lag1"])
    phi_x_s = float(synth_row["acf_x_lag1"])
    # добавка в единицах уже нормированного канала, чтобы MAD(ch0) догнал real
    extra = float(np.sqrt(max(ch_ratio ** 2 - 1.0, 0.0)))
    return {
        "grid_dt_s": 120.0,
        "grid_dx_m": 0.25,
        "physical_sigma_C_real_p50": rs,
        "physical_sigma_C_synth_p50": ss,
        "physical_sigma_ratio_real_over_synth": ratio,
        "channel0_mad_real": rc,
        "channel0_mad_synth_bg": sc,
        "channel_mad_ratio": ch_ratio,
        "ar1_time_phi_real": phi_t,
        "ar1_time_phi_synth": phi_t_s,
        "ar1_space_phi_real": phi_x,
        "ar1_space_phi_synth": phi_x_s,
        "extra_channel_noise_std": extra,
        "apply_before_make_channels": (
            f"К белому шуму синтетики (после события, до make_channels) добавить "
            f"окрашенный остаток с σ_°C = {max(rs - ss, 0.0):.4f} "
            f"(разница медианных σ; если отрицательная — не добавлять, а умножить шум генератора на {ratio:.3f}). "
            f"Окраска: AR(1) по времени с φ={phi_t:.3f} на шаге 120 с и AR(1) по кабелю с φ={phi_x:.3f} на шаге 0.25 м. "
            f"Независимо, на уже нормированном тайле ae (каналы 0 и 1) добавить такой же AR(1) "
            f"с std={extra:.3f} в единицах σ, φ_t={phi_t:.3f}, φ_x={phi_x:.3f}, канал valid не трогать."
        ),
    }
