"""Кривая recall слабых событий (≤2 °C) против FAR. Без обучения.

Пороги — квантиль фона v8 val, тем же правилом, что dts.metrics.far_threshold.
Дальше порог заморожен и применяется к test v8, v10 и v11.
SNR = amp_obs_C / медиана σ(x) по столбцам события. Уровень 50 % — пересечение
бинов by_amp по медиане этого SNR.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_weak_lib as W
from dts import data, metrics as M

ROOT = Path(__file__).resolve().parents[1]
FARS = np.unique(np.round(np.concatenate([
    np.geomspace(1e-4, 3e-2, 12),
    np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]),
]), decimals=10))
METHODS = ("ens_u2ft_tcn_lgbm", "base_median_x3_spatial_z")
VERSIONS = ("v8", "v10", "v11")
# Хвост, в котором ещё лежит квантиль FAR=3e-2 (проверено на v8 val).
CUTOFF = {"ens_u2ft_tcn_lgbm": 0.05, "base_median_x3_spatial_z": 2.0}
SNR_EDGES = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0, 40.0, 80.0])


class ExactTail:
    """Квантили фона без усечения динамического диапазона |Z|."""

    def __init__(self, cutoff: float):
        self.cutoff = float(cutoff)
        self.total = 0
        self.chunks: list[np.ndarray] = []

    def add(self, scores) -> None:
        s = np.asarray(scores, np.float32).ravel()
        s = s[np.isfinite(s)]
        self.total += int(s.size)
        if s.size == 0:
            return
        hi = s[s > self.cutoff]
        if hi.size:
            self.chunks.append(np.ascontiguousarray(hi))

    def thresholds(self, fars) -> list[float]:
        if self.total <= 0:
            raise SystemExit("пустой фон")
        kept = np.concatenate(self.chunks) if self.chunks else np.empty(0, np.float32)
        kept.sort()
        n_below = self.total - int(kept.size)
        out = []
        for far in fars:
            k = int(np.ceil((1.0 - float(far)) * self.total)) - 1
            k = int(np.clip(k, 0, self.total - 1))
            if k < n_below:
                raise SystemExit(f"cutoff {self.cutoff} выше квантиля FAR={far}")
            out.append(float(kept[k - n_below]) + 1e-12)
        return out


def scores_of(method: str, cache_path: Path, ens_dir: Path):
    z = np.load(cache_path)
    if method == "ens_u2ft_tcn_lgbm":
        s = np.asarray(np.load(ens_dir / cache_path.name)["s"], np.float32)
    else:
        s = np.abs(np.asarray(z["base_z"], np.float32))
    return z, s


def fit_thresholds(fars):
    tails = {m: ExactTail(CUTOFF[m]) for m in METHODS}
    files = data.list_cache(ROOT / "artifacts/cache/v8", "val")
    ens = ROOT / "artifacts/pred/v8/ens_u2ft_tcn_lgbm/val"
    t0 = time.time()
    for i, p in enumerate(files):
        z = np.load(p)
        bg = W.bg_mask(z["ev"], z["valid"], z["nu"])
        tails["ens_u2ft_tcn_lgbm"].add(np.asarray(np.load(ens / p.name)["s"], np.float32)[bg])
        tails["base_median_x3_spatial_z"].add(np.abs(np.asarray(z["base_z"], np.float32))[bg])
        if (i + 1) % 15 == 0 or i + 1 == len(files):
            print("val", i + 1, "/", len(files), "sec", round(time.time() - t0), flush=True)
    thr = {m: tails[m].thresholds(fars) for m in METHODS}
    return thr, {m: tails[m].total for m in METHODS}


def event_sigma(x, sigma, row) -> float:
    ix = (x >= row.dist_start_m - 0.25) & (x <= row.dist_end_m + 0.25)
    if not np.any(ix):
        return float(np.median(sigma))
    return float(np.median(np.asarray(sigma, np.float32)[ix]))


def eval_split(ver: str, thr: dict, fars):
    labels = pd.read_csv(ROOT / "data/synth_meta" / ver / "labels.csv")
    files = data.list_cache(ROOT / "artifacts/cache" / ver, "test")
    ens = ROOT / "artifacts/pred" / ver / "ens_u2ft_tcn_lgbm" / "test"
    n_far = len(fars)
    far_n = {m: np.zeros(n_far, np.int64) for m in METHODS}
    far_d = {m: 0 for m in METHODS}
    weak_tp = {m: np.zeros(n_far, np.int64) for m in METHODS}
    weak_fn = {m: np.zeros(n_far, np.int64) for m in METHODS}
    weak_hit = {m: np.zeros(n_far, np.int64) for m in METHODS}
    weak_n = 0
    # все события, для SNR и by_amp при каждом FAR
    events = []
    t0 = time.time()
    for i, p in enumerate(files):
        z, s_ens = scores_of("ens_u2ft_tcn_lgbm", p, ens)
        s_base = np.abs(np.asarray(z["base_z"], np.float32))
        smap = {"ens_u2ft_tcn_lgbm": s_ens, "base_median_x3_spatial_z": s_base}
        ev, valid, nu = z["ev"], z["valid"], z["nu"]
        x, t, sigma = z["x"], z["t"], z["sigma"]
        bg = W.bg_mask(ev, valid, nu)
        nbg = int(bg.sum())
        for m in METHODS:
            sb = smap[m][bg]
            far_d[m] += nbg
            for j, thj in enumerate(thr[m]):
                far_n[m][j] += int((sb >= thj).sum())
        rows = [r for r in M.label_boxes(labels, p.stem).itertuples() if r.kind == "event"]
        for r in rows:
            px = M.event_pixels(r, x, t, ev)
            n = int(px.sum())
            if n == 0:
                continue
            amp = float(r.amp_obs_C)
            sig = event_sigma(x, sigma, r)
            snr = amp / max(sig, 1e-3)
            rec = dict(amp=amp, sigma=sig, snr=snr, label=str(r.label), n=n,
                       weak=bool(amp <= 2.0), hit={}, pix={})
            for m in METHODS:
                vals = smap[m][px]
                hit = np.empty(n_far, np.uint8)
                pix = np.empty(n_far, np.float32)
                for j, thj in enumerate(thr[m]):
                    tp = int((vals >= thj).sum())
                    frac = tp / n
                    hit[j] = frac >= 0.3
                    pix[j] = frac
                    if amp <= 2.0:
                        weak_tp[m][j] += tp
                        weak_fn[m][j] += n - tp
                        weak_hit[m][j] += int(frac >= 0.3)
                rec["hit"][m] = hit
                rec["pix"][m] = pix
            if amp <= 2.0:
                weak_n += 1
            events.append(rec)
        if (i + 1) % 15 == 0 or i + 1 == len(files):
            print(ver, "test", i + 1, "/", len(files), "sec", round(time.time() - t0), flush=True)
    curves = {}
    for m in METHODS:
        curves[m] = dict(
            event_recall=(weak_hit[m] / max(weak_n, 1)).tolist(),
            pixel_recall=(weak_tp[m] / np.maximum(weak_tp[m] + weak_fn[m], 1)).tolist(),
            achieved_far=(far_n[m] / max(far_d[m], 1)).tolist(),
            n_weak=int(weak_n),
            n_events=int(len(events)),
        )
    return curves, events


def _bin_table(events, method, j, key):
    """key='amp' группирует by_amp, key='snr' — фиксированные края SNR."""
    rows = []
    if key == "amp":
        groups = [("<=2C", 0.0, 2.0), ("2-5C", 2.0, 5.0), (">5C", 5.0, 1e9)]
        for name, a, b in groups:
            sel = [e for e in events if a < e["amp"] <= b or (name == "<=2C" and e["amp"] <= 2.0)]
            if name != "<=2C":
                sel = [e for e in events if a < e["amp"] <= b]
            else:
                sel = [e for e in events if e["amp"] <= 2.0]
            if not sel:
                rows.append(dict(bin=name, n=0, event_recall=None, pixel_recall=None,
                                 median_amp=None, median_sigma=None, median_snr=None))
                continue
            hits = np.array([e["hit"][method][j] for e in sel], float)
            pix = np.array([e["pix"][method][j] for e in sel], float)
            # pixel recall — по пикселям, не среднее долей
            tp = sum(e["pix"][method][j] * e["n"] for e in sel)
            nn = sum(e["n"] for e in sel)
            rows.append(dict(
                bin=name, n=len(sel),
                event_recall=float(hits.mean()),
                pixel_recall=float(tp / max(nn, 1)),
                median_amp=float(np.median([e["amp"] for e in sel])),
                median_sigma=float(np.median([e["sigma"] for e in sel])),
                median_snr=float(np.median([e["snr"] for e in sel])),
                mean_frac=float(pix.mean()),
            ))
        return rows
    out = []
    for a, b in zip(SNR_EDGES[:-1], SNR_EDGES[1:]):
        sel = [e for e in events if a <= e["snr"] < b]
        if len(sel) < 5:
            continue
        hits = np.array([int(e["hit"][method][j]) for e in sel], float)
        tp = sum(e["pix"][method][j] * e["n"] for e in sel)
        nn = sum(e["n"] for e in sel)
        center = float(np.sqrt(a * b)) if a > 0 else float(b / 2)
        out.append(dict(snr_lo=float(a), snr_hi=float(b), snr_center=center, n=len(sel),
                        event_recall=float(hits.mean()),
                        pixel_recall=float(tp / max(nn, 1))))
    return out


def _ordered_crossing(rows, xkey):
    """Бины by_amp с положительным x, по возрастанию x. Пересечение recall=0.5 и пометка края."""
    pts = [(r[xkey], r["event_recall"]) for r in rows
           if r["n"] and r[xkey] is not None and r[xkey] > 0 and r["event_recall"] is not None]
    pts.sort(key=lambda p: p[0])
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    level = W.crossing_level(xs, ys, 0.5)
    if level is not None:
        note = "log-interpolated between by_amp bins"
    elif not ys:
        note = "no bins"
    elif ys[0] >= 0.5:
        note = "recall already ≥50% at the lowest by_amp bin"
    elif ys[-1] < 0.5:
        note = "recall stays <50% through the highest by_amp bin"
    else:
        note = "no upward crossing"
    return level, note, xs, ys


def snr_block(events, method, fars):
    """50 % event recall по медиане SNR бинов by_amp, плюс проверка мелкими бинами SNR."""
    per_far = []
    for j, far in enumerate(fars):
        by_amp = _bin_table(events, method, j, "amp")
        fine = _bin_table(events, method, j, "snr")
        snr50, note, _, _ = _ordered_crossing(by_amp, "median_snr")
        amp50, amp_note, _, _ = _ordered_crossing(by_amp, "median_amp")
        fine_x = [r["snr_center"] for r in fine]
        fine_y = [r["event_recall"] for r in fine]
        snr50_fine = W.crossing_level(fine_x, fine_y, 0.5)
        per_far.append(dict(
            far=float(far),
            by_amp=by_amp,
            snr_bins=fine,
            snr_50=snr50,
            snr_50_note=note,
            amp_50=amp50,
            amp_50_note=amp_note,
            snr_50_fine=snr50_fine,
        ))
    return per_far


def plot(fars, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"v8": "#1f4e79", "v10": "#c47b00", "v11": "#2a7f62"}
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.15), sharex=True)
    specs = [
        (axes[0], "event_recall", "event recall, ≤2 °C"),
        (axes[1], "pixel_recall", "pixel recall, ≤2 °C"),
    ]
    for ax, key, title in specs:
        for ver in VERSIONS:
            c = curves[ver]
            ax.plot(fars, c["ens_u2ft_tcn_lgbm"][key], color=colors[ver], lw=1.8,
                    label=f"ens {ver}")
            ax.plot(fars, c["base_median_x3_spatial_z"][key], color=colors[ver], lw=1.4,
                    ls="--", label=f"|Z| {ver}")
        ax.axvline(1e-3, color="0.45", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xlim(1e-4, 3e-2)
        ax.set_ylim(0, 1)
        ax.set_xlabel("FAR, порог с v8 val")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, which="both", lw=0.4, alpha=0.5)
    axes[1].legend(loc="upper left", fontsize=8, frameon=False, ncol=2)
    fig.suptitle("Слабые события ≤2 °C: recall против FAR", fontsize=12)
    fig.tight_layout()
    out = ROOT / "docs/figures/weak_far_curve.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def far_index(fars, target: float = 1e-3) -> int:
    arr = np.asarray(fars, np.float64)
    j = int(np.argmin(np.abs(arr - target)))
    if abs(float(arr[j]) - target) > 1e-9:
        raise SystemExit(f"FAR {target} нет в сетке")
    return j


def main():
    os.chdir(ROOT)
    t0 = time.time()
    fars = [float(v) for v in FARS]
    j = far_index(fars, 1e-3)
    thr, n_bg = fit_thresholds(fars)
    print("thr far=1e-3", {m: thr[m][j] for m in METHODS}, flush=True)
    curves = {}
    snr = {}
    for ver in VERSIONS:
        curves[ver], events = eval_split(ver, thr, fars)
        snr[ver] = {m: snr_block(events, m, fars) for m in METHODS}
        del events
    # контрольная точка опубликованного порога ансамбля
    report = {
        "description": "Weak ≤2°C recall vs FAR. Thresholds from v8 val background, frozen on test. No training.",
        "far": fars,
        "thresholds": {m: thr[m] for m in METHODS},
        "n_background_v8_val": n_bg,
        "published_ens_thr_far_1e-3": 0.8608087301264272,
        "ens_thr_at_1e-3": thr["ens_u2ft_tcn_lgbm"][j],
        "hit_rule": "event hit if ≥30% of its pixels are above the threshold",
        "snr_def": "amp_obs_C / median sigma(x) on the event columns",
        "snr_50_def": "log-x interpolation of event recall across by_amp bins ordered by median SNR; null if recall never crosses 50%",
        "amp_50_def": "same interpolation on median amp_obs_C of the by_amp bins",
        "curves": curves,
        "snr": snr,
        "real_used": False,
        "seconds": round(time.time() - t0),
    }
    fig = plot(fars, curves)
    out = ROOT / "artifacts/eval/s3_weak_far.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out, "w"), indent=1)
    print("wrote", out, fig, "sec", report["seconds"], flush=True)
    for ver in VERSIONS:
        for m in METHODS:
            c = curves[ver][m]
            s50 = snr[ver][m][j]["snr_50"]
            print(ver, m, "weak@", f"{c['event_recall'][j]:.3f}", "pix", f"{c['pixel_recall'][j]:.3f}",
                  "n", c["n_weak"], "snr50", s50, flush=True)


if __name__ == "__main__":
    main()
