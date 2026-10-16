"""Диагностика сдвига synth v8 val → MGTS и пофайловый порог на неразмеченном score.

Карточки оператора читаются только после того, как порог файла уже зафиксирован.
Порог — квантиль всех валидных клеток этого файла. Метки в него не входят.

Запуск из /workspace/dts_solution, не больше 2 потоков:
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \\
  /workspace/repo/run/.venv/bin/python -u scripts/domain_shift.py
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_k] = "2"

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE_SYNTH = ROOT / "artifacts/cache/v8/val"
PRED_SYNTH = ROOT / "artifacts/pred/v8/ens_unet_tcn_lgbm/val"
REAL_ROOT = Path("/workspace/mgts/MGTS_data")
CARD_PATHS = [
    ROOT / "data/operator_cards/cards_batch1.jsonl",
    ROOT / "data/operator_cards/cards_batch2.jsonl",
]
FIG = ROOT / "docs/figures"
EVAL = ROOT / "artifacts/eval"
CACHE = EVAL / "domain_cache"
SECTION = "<!-- lit-domain-shift:start -->"
SECTION_END = "<!-- lit-domain-shift:end -->"

MAX_LAG = 48
N_TRACES = 12
N_BINS = 33
CHAN_HOLD = 300_000
SCORE_EDGES = np.linspace(0.0, 1.0, 2001)
CH_EDGES = np.linspace(-12.0, 12.0, 121)
OPS = (
    ("global_far_1e-3", None),
    ("per_file_1e-3", 1e-3),
    ("per_file_3e-3", 3e-3),
    ("per_file_1e-2", 1e-2),
)


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S', time.gmtime())} {msg}", flush=True)


def cap_blas() -> None:
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=2)
    except Exception as exc:
        log(f"threadpoolctl: {exc}")


def save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.stem}.partial.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    def conv(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        raise TypeError(type(o))

    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=conv))
    os.replace(tmp, path)


def finite(a) -> np.ndarray:
    x = np.asarray(a, np.float64).ravel()
    return x[np.isfinite(x)]


def mad_of(v: np.ndarray) -> float:
    v = finite(v)
    if v.size == 0:
        return float("nan")
    med = np.median(v)
    return float(1.4826 * np.median(np.abs(v - med)))


def sigma_columns(sigma, mask) -> np.ndarray:
    frac = np.asarray(mask, bool).mean(axis=0)
    s = np.asarray(sigma, np.float64)
    keep = (frac >= 0.5) & np.isfinite(s) & (s > 0)
    return s[keep]


def sample_masked(field, mask, n, rng) -> np.ndarray:
    m = np.asarray(mask, bool) & np.isfinite(field)
    idx = np.flatnonzero(m.ravel())
    if idx.size == 0:
        return np.zeros(0, np.float32)
    take = rng.choice(idx, size=min(n, idx.size), replace=False)
    return np.asarray(field, np.float32).ravel()[take]


def hist_quantile(counts: np.ndarray, edges: np.ndarray, q: float) -> float:
    total = float(counts.sum())
    if total <= 0:
        return float("nan")
    cdf = np.cumsum(counts)
    i = int(np.searchsorted(cdf, q / 100.0 * total, side="left"))
    i = min(max(i, 0), len(edges) - 2)
    return float(edges[i + 1])


def add_hist(counts: np.ndarray, edges: np.ndarray, values: np.ndarray) -> None:
    v = finite(values)
    if v.size == 0:
        return
    counts += np.histogram(v, bins=edges)[0]


def field_curves(X, mask, rng):
    from dts import domain
    acf_t = domain.mean_acf(X[0], mask, 0, MAX_LAG, N_TRACES, rng)
    acf_x = domain.mean_acf(X[0], mask, 1, MAX_LAG, N_TRACES, rng)
    psd_t = domain.mean_psd(X[0], mask, 0, N_BINS, N_TRACES, rng)
    psd_x = domain.mean_psd(X[0], mask, 1, N_BINS, N_TRACES, rng)
    return acf_t, acf_x, psd_t, psd_x


def row_from_arrays(X, valid_mask, sigma, rng, score=None, global_thr=None) -> dict:
    from dts import domain
    sig = sigma_columns(sigma, valid_mask)
    qs = domain.quantile_summary(sig, (50, 90))
    ch0 = np.asarray(X[0], np.float32)[np.asarray(valid_mask, bool)]
    ch1 = np.asarray(X[1], np.float32)[np.asarray(valid_mask, bool)]
    acf_t, acf_x, psd_t, psd_x = field_curves(X, valid_mask, rng)
    n = int(np.isfinite(ch0).sum())
    row = {
        "n": n,
        "sigma_n": int(sig.size),
        "sigma_p50": qs.get("p50", float("nan")),
        "sigma_p90": qs.get("p90", float("nan")),
        "ch0_mad": mad_of(ch0),
        "ch1_mad": mad_of(ch1),
        "ch0_abs_p99": float(np.percentile(np.abs(finite(ch0)), 99)) if n else float("nan"),
        "ch0_clip_frac": float((np.abs(finite(ch0)) >= 11.9).mean()) if n else float("nan"),
        "acf_t_lag1": float(acf_t[1]) if np.isfinite(acf_t[1]) else float("nan"),
        "acf_x_lag1": float(acf_x[1]) if np.isfinite(acf_x[1]) else float("nan"),
    }
    if score is not None:
        s = np.asarray(score, np.float64)[np.asarray(valid_mask, bool)]
        s = s[np.isfinite(s)]
        pq = domain.quantile_summary(s, (50, 99, 99.9))
        row["score_p50"] = pq.get("p50", float("nan"))
        row["score_p99"] = pq.get("p99", float("nan"))
        row["score_p999"] = pq.get("p999", float("nan"))
        if global_thr is not None and s.size:
            row["frac_above_global_thr"] = float((s >= global_thr).mean())
        else:
            row["frac_above_global_thr"] = float("nan")
    return row, acf_t, acf_x, psd_t, psd_x, ch0, ch1


def nanmean_curves(curves):
    stack = [c for c in curves if c is not None and np.isfinite(c).all()]
    if not stack:
        return None
    return np.mean(stack, 0)


def build_real_channels(names: list[str]) -> None:
    from dts import baseline, features, grid, real
    for name in names:
        ch_path = CACHE / f"{name}_ch.npz"
        meta_path = CACHE / f"{name}_meta.json"
        base_path = CACHE / f"{name}_base.json"
        if ch_path.exists() and meta_path.exists() and base_path.exists():
            log(f"real {name}: кэш каналов есть")
            continue
        csv_path = REAL_ROOT / f"{name}.csv"
        t0 = time.time()
        log(f"real {name}: читаю {csv_path.name}")
        rec = real.load_csv(csv_path)
        log(f"real {name}: csv {tuple(rec['T'].shape)} за {time.time() - t0:.0f} с")
        p = grid.plan(rec["x"], rec["t"])
        Tc, v = grid.resample(rec["T"], rec["valid"], p, "mean")
        X, aux = features.make_channels(Tc, v, dx=float(p["dx"]))
        log(f"real {name}: каналы {tuple(X.shape)} за {time.time() - t0:.0f} с")
        if not base_path.exists():
            b = baseline.run(rec["T"], rec["x"], rec["t"], valid=rec["valid"])
            zones = []
            for z in b["zones"]:
                zones.append(dict(start_m=float(z["start_m"]), end_m=float(z["end_m"]),
                                  t_start_s=float(z["t_start_s"]), t_end_s=float(z["t_end_s"]),
                                  polarity=z.get("polarity"), alarm=True))
            base_path.write_text(json.dumps(zones))
            log(f"real {name}: эталон {len(zones)} зон за {time.time() - t0:.0f} с")
        save_npz(ch_path, X=X.astype(np.float16), valid=v.astype(np.uint8),
                 x=np.asarray(p["x"], np.float64), t=np.asarray(p["t"], np.float64),
                 sigma=np.asarray(aux["sigma_x"], np.float32),
                 dT=np.asarray(aux["dT_time"], np.float16))
        meta = dict(sha256=rec["sha256"], tz_missing=bool(rec["tz_missing"]),
                    raw_shape=list(rec["T"].shape), grid_shape=list(v.shape),
                    channel_id=rec.get("channel_id"))
        meta_path.write_text(json.dumps(meta))
        del rec, Tc, X, aux
        log(f"real {name}: каналы записаны")


def load_ch(name: str) -> dict:
    with np.load(CACHE / f"{name}_ch.npz") as z:
        return dict(X=np.array(z["X"]), valid=np.array(z["valid"]).astype(bool),
                    x=np.array(z["x"]), t=np.array(z["t"]), sigma=np.array(z["sigma"]),
                    dT=np.array(z["dT"]))


def load_meta(name: str) -> dict:
    return json.loads((CACHE / f"{name}_meta.json").read_text())


def load_base(name: str) -> list:
    return json.loads((CACHE / f"{name}_base.json").read_text())


def diagnose_synth(limit: int, rng_seed: int, global_thr: float):
    from dts import domain
    paths = sorted(CACHE_SYNTH.glob("seq*.npz"))
    if limit:
        paths = paths[:limit]
    log(f"synth файлов {len(paths)}")
    rows = []
    acf_t_all, acf_x_all, psd_t_all, psd_x_all = [], [], [], []
    sig_parts = []
    score_hist = np.zeros(len(SCORE_EDGES) - 1, np.int64)
    ch0_hist = np.zeros(len(CH_EDGES) - 1, np.int64)
    ch1_hist = np.zeros(len(CH_EDGES) - 1, np.int64)
    hold0 = np.empty(CHAN_HOLD, np.float32)
    hold1 = np.empty(CHAN_HOLD, np.float32)
    n0 = n1 = 0
    n_bg = n_above = 0
    rng = np.random.default_rng(rng_seed)
    for i, path in enumerate(paths):
        pred_path = PRED_SYNTH / f"{path.stem}.npz"
        with np.load(path) as z:
            X = np.array(z["X"])
            valid = np.array(z["valid"]).astype(bool)
            ev = np.array(z["ev"])
            nu = np.array(z["nu"])
            sigma = np.array(z["sigma"])
        with np.load(pred_path) as z:
            score = np.array(z["s"])
        if score.shape != valid.shape:
            raise RuntimeError(f"{path.name}: score {score.shape} != valid {valid.shape}")
        mask = valid & (ev == 0) & (nu == 0)
        file_rng = np.random.default_rng(rng_seed + 1 + i)
        row, acf_t, acf_x, psd_t, psd_x, ch0, ch1 = row_from_arrays(
            X, mask, sigma, file_rng, score=score, global_thr=global_thr)
        row["file"] = path.stem
        rows.append({k: row[k] for k in ("file", "n", "sigma_p50", "ch0_mad", "acf_t_lag1", "acf_x_lag1",
                                         "frac_above_global_thr")})
        acf_t_all.append(acf_t)
        acf_x_all.append(acf_x)
        psd_t_all.append(psd_t)
        psd_x_all.append(psd_x)
        sig_parts.append(sigma_columns(sigma, mask))
        add_hist(score_hist, SCORE_EDGES, np.asarray(score, np.float32)[mask])
        add_hist(ch0_hist, CH_EDGES, ch0)
        add_hist(ch1_hist, CH_EDGES, ch1)
        n0 = domain.reservoir_update(hold0, n0, finite(ch0).astype(np.float32), rng)
        n1 = domain.reservoir_update(hold1, n1, finite(ch1).astype(np.float32), rng)
        s = np.asarray(score, np.float32)[mask]
        s = s[np.isfinite(s)]
        n_bg += int(s.size)
        n_above += int((s >= global_thr).sum())
        if i % 5 == 0 or i + 1 == len(paths):
            log(f"synth {i + 1}/{len(paths)} {path.stem} n_bg={n_bg}")
        del X, score, ch0, ch1
    sig = np.concatenate(sig_parts) if sig_parts else np.zeros(0)
    sq = domain.quantile_summary(sig, (50, 90))
    acf_t = nanmean_curves(acf_t_all)
    acf_x = nanmean_curves(acf_x_all)
    pooled = {
        "n": int(n_bg),
        "sigma_n": int(sig.size),
        "sigma_p50": sq.get("p50", float("nan")),
        "sigma_p90": sq.get("p90", float("nan")),
        "ch0_mad": mad_of(hold0[:n0]),
        "ch1_mad": mad_of(hold1[:n1]),
        "ch0_abs_p99": float(np.percentile(np.abs(hold0[:n0]), 99)) if n0 else float("nan"),
        "acf_t_lag1": float(acf_t[1]) if acf_t is not None else float("nan"),
        "acf_x_lag1": float(acf_x[1]) if acf_x is not None else float("nan"),
        "score_p50": hist_quantile(score_hist, SCORE_EDGES, 50),
        "score_p99": hist_quantile(score_hist, SCORE_EDGES, 99),
        "score_p999": hist_quantile(score_hist, SCORE_EDGES, 99.9),
        "frac_above_global_thr": float(n_above / n_bg) if n_bg else float("nan"),
        "n_files": len(paths),
    }
    return dict(files=rows, pooled=pooled, sigma=sig.astype(np.float32),
                acf_t=acf_t, acf_x=acf_x, psd_t=nanmean_curves(psd_t_all), psd_x=nanmean_curves(psd_x_all),
                score_hist=score_hist, ch0_hist=ch0_hist, ch1_hist=ch1_hist)


def diagnose_real(names, rng_seed: int, global_thr: float, with_scores: bool):
    from dts import domain
    rows = []
    acf_t_all, acf_x_all, psd_t_all, psd_x_all = [], [], [], []
    sig_parts = []
    per_sigma = {}
    score_hist = np.zeros(len(SCORE_EDGES) - 1, np.int64)
    ch0_hist = np.zeros(len(CH_EDGES) - 1, np.int64)
    ch1_hist = np.zeros(len(CH_EDGES) - 1, np.int64)
    per_score_hist = {}
    hold0 = np.empty(CHAN_HOLD, np.float32)
    hold1 = np.empty(CHAN_HOLD, np.float32)
    n0 = n1 = 0
    rng = np.random.default_rng(rng_seed)
    for i, name in enumerate(names):
        field = load_ch(name)
        mask = field["valid"]
        score = None
        score_path = CACHE / f"{name}_score.npz"
        if with_scores and score_path.exists():
            with np.load(score_path) as z:
                score = np.array(z["score"])
        file_rng = np.random.default_rng(rng_seed + 1 + i)
        row, acf_t, acf_x, psd_t, psd_x, ch0, ch1 = row_from_arrays(
            field["X"], mask, field["sigma"], file_rng, score=score, global_thr=global_thr)
        row["file"] = name
        meta = load_meta(name)
        row["sha256"] = meta.get("sha256")
        row["tz_missing"] = meta.get("tz_missing")
        row["grid_shape"] = meta.get("grid_shape")
        rows.append(row)
        acf_t_all.append(acf_t)
        acf_x_all.append(acf_x)
        psd_t_all.append(psd_t)
        psd_x_all.append(psd_x)
        sig = sigma_columns(field["sigma"], mask)
        sig_parts.append(sig)
        per_sigma[name] = sig.astype(np.float32)
        add_hist(ch0_hist, CH_EDGES, ch0)
        add_hist(ch1_hist, CH_EDGES, ch1)
        n0 = domain.reservoir_update(hold0, n0, finite(ch0).astype(np.float32), rng)
        n1 = domain.reservoir_update(hold1, n1, finite(ch1).astype(np.float32), rng)
        if score is not None:
            h = np.zeros_like(score_hist)
            add_hist(h, SCORE_EDGES, np.asarray(score, np.float32)[mask])
            per_score_hist[name] = h
            score_hist += h
        log(f"real {name}: σ p50={row['sigma_p50']:.4f} MAD ch0={row['ch0_mad']:.3f} φt={row['acf_t_lag1']:.3f} φx={row['acf_x_lag1']:.3f}")
        del field, ch0, ch1
    from dts import domain as domain_mod
    sig = np.concatenate(sig_parts) if sig_parts else np.zeros(0)
    sq = domain_mod.quantile_summary(sig, (50, 90))
    acf_t = nanmean_curves(acf_t_all)
    acf_x = nanmean_curves(acf_x_all)
    pooled = {
        "n": int(sum(r["n"] for r in rows)),
        "sigma_n": int(sig.size),
        "sigma_p50": sq.get("p50", float("nan")),
        "sigma_p90": sq.get("p90", float("nan")),
        "ch0_mad": mad_of(hold0[:n0]),
        "ch1_mad": mad_of(hold1[:n1]),
        "ch0_abs_p99": float(np.percentile(np.abs(hold0[:n0]), 99)) if n0 else float("nan"),
        "acf_t_lag1": float(acf_t[1]) if acf_t is not None else float("nan"),
        "acf_x_lag1": float(acf_x[1]) if acf_x is not None else float("nan"),
    }
    scored = [r for r in rows if "frac_above_global_thr" in r]
    if scored:
        n_tot = sum(r["n"] for r in scored)
        n_above = sum(float(r["frac_above_global_thr"]) * r["n"] for r in scored)
        pooled["frac_above_global_thr"] = float(n_above / n_tot) if n_tot else float("nan")
        pooled["score_p50"] = hist_quantile(score_hist, SCORE_EDGES, 50)
        pooled["score_p99"] = hist_quantile(score_hist, SCORE_EDGES, 99)
        pooled["score_p999"] = hist_quantile(score_hist, SCORE_EDGES, 99.9)
    return dict(files=rows, pooled=pooled, sigma=sig.astype(np.float32), per_sigma=per_sigma,
                acf_t=acf_t, acf_x=acf_x, psd_t=nanmean_curves(psd_t_all), psd_x=nanmean_curves(psd_x_all),
                score_hist=score_hist, per_score_hist=per_score_hist, ch0_hist=ch0_hist, ch1_hist=ch1_hist)


def load_analyzer():
    log("load_analyzer: import torch")
    import torch
    log(f"load_analyzer: torch {torch.__version__}")
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except Exception as exc:
        log(f"set_num_interop_threads: {exc}")
    log("load_analyzer: import infer")
    from dts import infer
    log("load_analyzer: Analyzer")
    analyzer = infer.Analyzer(op="far_1e-3")
    for kind, mdl in analyzer.members:
        if kind == "net":
            continue
        if hasattr(mdl, "set_params"):
            mdl.set_params(n_jobs=2)
        log(f"pixel_lgbm n_jobs={getattr(mdl, 'n_jobs', None)}")
    log(f"torch threads {torch.get_num_threads()} global thr {analyzer.thr:.6f}")
    return analyzer


def ensure_scores(names, analyzer, scale: float, tag: str) -> None:
    from dts import domain
    for name in names:
        out_path = CACHE / (f"{name}_score.npz" if tag == "raw" else f"{name}_score_{tag}.npz")
        if out_path.exists():
            log(f"real {name}: score {tag} уже есть")
            continue
        field = load_ch(name)
        X = field["X"].astype(np.float32)
        if scale != 1.0:
            X = X.copy()
            X[:2] /= np.float32(scale)
        t0 = time.time()
        score, hot = domain.ensemble_maps(analyzer, X, log=f"real {name} x{scale:.3f}")
        save_npz(out_path, score=score.astype(np.float32), hot=hot.astype(np.float16))
        log(f"real {name}: score {tag} {score.shape} за {time.time() - t0:.0f} с, p99={np.percentile(score[field['valid']], 99):.3f}")
        del field, X, score, hot


def zone_brief(zones) -> dict:
    classes = {}
    for z in zones:
        classes[z.get("class", "?")] = classes.get(z.get("class", "?"), 0) + 1
    return {"n_zones": len(zones), "n_alarm": int(sum(bool(z.get("alarm")) for z in zones)), "classes": classes}


def compact_zone(z) -> dict:
    keys = ("start_m", "end_m", "t_start_s", "t_end_s", "n_px", "polarity", "class", "alarm", "score")
    return {k: z[k] for k in keys if k in z}


def thresholds_and_cards(names, global_thr: float, score_tag: str, analyzer) -> dict:
    from dts import domain
    cards = []
    for path in CARD_PATHS:
        cards += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    per_file = {}
    card_rows = []
    for name in names:
        field = load_ch(name)
        score_path = CACHE / (f"{name}_score.npz" if score_tag == "raw" else f"{name}_score_{score_tag}.npz")
        with np.load(score_path) as z:
            score = np.array(z["score"])
            hot = np.array(z["hot"]).astype(np.float32)
        scale = 1.0 if score_tag == "raw" else float(score_tag.split("k")[-1])
        X = field["X"].astype(np.float32)
        if scale != 1.0:
            X = X.copy()
            X[:2] /= np.float32(scale)
        out = dict(grid=dict(x=field["x"], t=field["t"]), X=X, valid=field["valid"], score=score,
                   hotness=hot, sigma=field["sigma"].astype(np.float32) * scale, dT=field["dT"].astype(np.float32))
        base = load_base(name)
        meta = load_meta(name)
        file_ops = {}
        for op_name, far in OPS:
            if far is None:
                thr = float(global_thr)
            else:
                thr = float(domain.flag_threshold(score, field["valid"], far))
            got = domain.rethreshold(analyzer, out, thr)
            brief = zone_brief(got["zones"])
            brief["threshold"] = thr
            brief["flag_fraction"] = domain.flag_fraction(score, field["valid"], thr)
            brief["zones"] = [compact_zone(z) for z in got["zones"]]
            file_ops[op_name] = brief
            for card in cards:
                if str(card.get("record")) != name:
                    continue
                ev = domain.score_card(card, field["x"], field["t"], field["valid"], score, got["zones"], thr, base)
                ev["op"] = op_name
                ev["score_tag"] = score_tag
                ev["sha_ok"] = card.get("source_sha256") == meta.get("sha256")
                ev["threshold"] = thr
                card_rows.append(ev)
        file_ops["n_zones_base"] = len(base)
        file_ops["sha256"] = meta.get("sha256")
        file_ops["tz_missing"] = meta.get("tz_missing")
        per_file[name] = file_ops
        log(f"real {name} {score_tag}: " + ", ".join(
            f"{op} thr={file_ops[op]['threshold']:.3f} alarm={file_ops[op]['n_alarm']}/{file_ops[op]['n_zones']}"
            for op, _ in OPS))
        del field, X, score, out
    tallies = {}
    for op_name, _ in OPS:
        subset = [r for r in card_rows if r["op"] == op_name]
        tallies[op_name] = {key: domain.presence_tally(subset, key)
                            for key in ("hit_alarm", "hit_zone", "hit_pixel", "hit_base")}
    return {"files": per_file, "cards": card_rows, "tallies": tallies, "score_tag": score_tag}


def shift_reasons(real_row: dict, synth_row: dict) -> list[str]:
    reasons = []
    ratio = real_row["sigma_p50"] / synth_row["sigma_p50"]
    ch = real_row["ch0_mad"] / synth_row["ch0_mad"]
    if not (0.8 <= ratio <= 1.25):
        reasons.append(f"медианная σ °C real/synth = {ratio:.3f}")
    if not (0.9 <= ch <= 1.1):
        reasons.append(f"MAD канала 0 real/synth = {ch:.3f}")
    dt = abs(real_row["acf_t_lag1"] - synth_row["acf_t_lag1"])
    dx = abs(real_row["acf_x_lag1"] - synth_row["acf_x_lag1"])
    if dt > 0.1:
        reasons.append(f"сдвиг φ_t на лаге 1 = {dt:.3f} (real {real_row['acf_t_lag1']:.3f}, synth {synth_row['acf_t_lag1']:.3f})")
    if dx > 0.1:
        reasons.append(f"сдвиг φ_x на лаге 1 = {dx:.3f} (real {real_row['acf_x_lag1']:.3f}, synth {synth_row['acf_x_lag1']:.3f})")
    rp, sp = real_row.get("ch0_abs_p99"), synth_row.get("ch0_abs_p99")
    if rp and sp and np.isfinite(rp) and np.isfinite(sp) and sp > 0:
        tail = rp / sp
        if not (0.85 <= tail <= 1.18):
            reasons.append(f"|ch0| p99 real/synth = {tail:.3f}")
    return reasons


def plot_all(synth, real) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    paths = []

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    sig_s, sig_r = finite(synth["sigma"]), finite(real["sigma"])
    lo = float(min(np.percentile(sig_s, 1), np.percentile(sig_r, 1)))
    hi = float(max(np.percentile(sig_s, 99), np.percentile(sig_r, 99)))
    bins = np.linspace(lo, hi, 40)
    ax[0].hist(sig_s, bins=bins, density=True, histtype="step", lw=1.6, label="synth v8 val, фон")
    ax[0].hist(sig_r, bins=bins, density=True, histtype="step", lw=1.6, label="real MGTS")
    ax[0].set_xlabel("σ(x), °C")
    ax[0].set_ylabel("плотность")
    ax[0].set_title("Шум σ(x) = 1.4826·MAD(Δt)/√2")
    ax[0].legend(fontsize=8)
    labels, data = [], []
    for name, sig in real["per_sigma"].items():
        labels.append(name)
        data.append(finite(sig))
    ax[1].boxplot(data, showfliers=False)
    ax[1].set_xticks(np.arange(1, len(labels) + 1))
    ax[1].set_xticklabels(labels)
    ax[1].axhline(float(np.median(sig_s)), color="C0", ls="--", label="медиана synth")
    ax[1].set_xlabel("файл MGTS")
    ax[1].set_ylabel("σ(x), °C")
    ax[1].set_title("σ по файлам")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    p = FIG / "domain_sigma.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(str(p))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    lags = np.arange(MAX_LAG + 1)
    for axis, key, title in ((0, "acf_t", "по времени, шаг 120 с"), (1, "acf_x", "по кабелю, шаг 0.25 м")):
        if synth[key] is not None:
            ax[axis].plot(lags, synth[key], label="synth v8 val, фон")
        if real[key] is not None:
            ax[axis].plot(lags, real[key], label="real MGTS")
        ax[axis].set_xlabel("лаг, шаги сетки")
        ax[axis].set_title(f"АКФ канала dT_time/σ {title}")
        ax[axis].legend(fontsize=8)
    ax[0].set_ylabel("АКФ")
    fig.tight_layout()
    p = FIG / "domain_acf.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(str(p))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    freq = np.linspace(0.0, 0.5, N_BINS)
    for axis, key, title in ((0, "psd_t", "по времени"), (1, "psd_x", "по кабелю")):
        if synth[key] is not None:
            ax[axis].semilogy(freq, np.clip(synth[key], 1e-8, None), label="synth v8 val, фон")
        if real[key] is not None:
            ax[axis].semilogy(freq, np.clip(real[key], 1e-8, None), label="real MGTS")
        ax[axis].set_xlabel("частота, циклы/шаг")
        ax[axis].set_title(f"Спектр канала dT_time/σ {title}")
        ax[axis].legend(fontsize=8)
    ax[0].set_ylabel("доля энергии")
    fig.tight_layout()
    p = FIG / "domain_spectrum.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(str(p))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    centers = 0.5 * (CH_EDGES[:-1] + CH_EDGES[1:])
    for axis, key, title in ((0, "ch0_hist", "канал 0, dT_time/σ"), (1, "ch1_hist", "канал 1, dT_space/σ")):
        for pack, label in ((synth, "synth v8 val, фон"), (real, "real MGTS")):
            c = pack[key].astype(np.float64)
            dens = c / max(c.sum(), 1.0) / np.diff(CH_EDGES)
            ax[axis].semilogy(centers, np.clip(dens, 1e-6, None), label=label)
        ax[axis].set_xlabel("значение канала")
        ax[axis].set_title(title)
        ax[axis].legend(fontsize=8)
    ax[0].set_ylabel("плотность")
    fig.tight_layout()
    p = FIG / "domain_channels.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(str(p))

    if synth["score_hist"].sum() and real["score_hist"].sum():
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        edges = SCORE_EDGES[:-1]

        def surv(counts):
            total = counts.sum()
            rev = np.cumsum(counts[::-1])[::-1]
            return rev / max(float(total), 1.0)

        ax.semilogy(edges, np.clip(surv(synth["score_hist"]), 1e-6, None), label="synth v8 val, фон")
        ax.semilogy(edges, np.clip(surv(real["score_hist"]), 1e-6, None), label="real, все валидные клетки")
        for name, h in real.get("per_score_hist", {}).items():
            ax.semilogy(edges, np.clip(surv(h), 1e-6, None), lw=0.7, alpha=0.7, label=f"файл {name}")
        ax.axvline(float(synth.get("global_thr", 0.844)), color="k", ls="--", lw=1, label="глобальный порог far 1e-3")
        ax.set_xlabel("score ансамбля")
        ax.set_ylabel("доля клеток ≥ score")
        ax.set_title("Хвост score: synth фон и real")
        ax.set_ylim(1e-5, 1)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        p = FIG / "domain_scores.png"
        fig.savefig(p, dpi=110)
        plt.close(fig)
        paths.append(str(p))
    return paths


def fmt(x, nd=3):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "н/д"
    if not np.isfinite(v):
        return "н/д"
    return f"{v:.{nd}f}"


def tally_text(tallies, op, key):
    t = tallies[op][key]
    return t[f"yes_{key}"], t["yes_n"], t[f"no_{key}"], t["no_n"]


def render_section(payload) -> str:
    synth = payload["synth_pooled"]
    real = payload["real_pooled"]
    lines = ["## Доменный сдвиг и пофайловый порог (lit)", ""]
    lines.append(
        "Сравнение фона synth v8 val (валидные клетки без события и без nuisance) с шестью файлами MGTS. "
        "Каналы CNN уже поделены на σ(x) файла: σ(x) = 1.4826·MAD(Δt)/√2, медианный фильтр 41, пол 0.03 °C. "
        "АКФ и спектр сняты с канала dT_time/σ, среднее по файлам, внутри файла — до 12 длинных трасс. "
        "Глобальный порог ансамбля ens_unet_tcn_lgbm — квантиль far 1e-3 с релиза, "
        f"{fmt(payload['global_thr'], 4)}. Пофайловый порог — квантиль всех валидных клеток этого файла "
        "(карточки и метки не участвуют). Попадание в карточку — пересечение тревожной зоны с рамкой, "
        "как в `scripts/run_real.py`: класс HVAC_Diurnal и Sensor_Artifact не тревога."
    )
    lines.append("")
    lines.append("| домен | n | σ p50, °C | σ p90 | MAD ch0 | MAD ch1 | φ_t | φ_x | score p50 | score p99 | score p999 | доля ≥ глобального порога |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    def table_row(label, row):
        return "| " + " | ".join([
            label, str(row.get("n", "")), fmt(row.get("sigma_p50"), 4), fmt(row.get("sigma_p90"), 4),
            fmt(row.get("ch0_mad")), fmt(row.get("ch1_mad")), fmt(row.get("acf_t_lag1")), fmt(row.get("acf_x_lag1")),
            fmt(row.get("score_p50")), fmt(row.get("score_p99")), fmt(row.get("score_p999")),
            fmt(row.get("frac_above_global_thr"), 5),
        ]) + " |"

    lines.append(table_row("synth v8 val, фон", synth))
    lines.append(table_row("real, все файлы", real))
    for row in payload["real_files"]:
        lines.append(table_row(f"файл {row['file']}", row))
    lines.append("")
    reasons = payload["shift_reasons"]
    if reasons:
        lines.append("Сдвиг шума явный: " + "; ".join(reasons) + ".")
    else:
        lines.append("Пороги σ, MAD канала и φ на лаге 1 лежат близко к synth. Отдельный рецепт аугментации не нужен.")
    lines.append("")
    recipe = payload.get("recipe")
    if recipe:
        lines.append("Рецепт noise-matched аугментации для агента ae (числа с этого прогона):")
        lines.append("")
        lines.append(
            f"Физическая медиана σ: real {fmt(recipe['physical_sigma_C_real_p50'], 4)} °C, "
            f"synth {fmt(recipe['physical_sigma_C_synth_p50'], 4)} °C, "
            f"отношение {fmt(recipe['physical_sigma_ratio_real_over_synth'])}. "
            f"MAD канала 0: real {fmt(recipe['channel0_mad_real'])}, synth {fmt(recipe['channel0_mad_synth_bg'])}, "
            f"отношение {fmt(recipe['channel_mad_ratio'])}. "
            f"AR(1) real: φ_t = {fmt(recipe['ar1_time_phi_real'])} на шаге 120 с, "
            f"φ_x = {fmt(recipe['ar1_space_phi_real'])} на шаге 0.25 м "
            f"(synth φ_t = {fmt(recipe['ar1_time_phi_synth'])}, φ_x = {fmt(recipe['ar1_space_phi_synth'])}). "
            f"Добавка на уже нормированный тайл: std = {fmt(recipe['extra_channel_noise_std'])}."
        )
        lines.append("")
        lines.append(recipe["apply_before_make_channels"])
        lines.append("")
    ch_ratio = payload.get("channel_mad_ratio")
    if payload.get("matched_ran"):
        lines.append(
            f"Каналы real поделены на {fmt(ch_ratio)} (отношение MAD) и прогнаны тем же ансамблем. "
            "Порог на этих score снова квантиль файла."
        )
        lines.append("")
    elif ch_ratio is not None:
        lines.append(
            f"Отношение MAD канала 0 равно {fmt(ch_ratio)} и лежит в [0.85, 1.18], "
            "поэтому второй прогон с делением каналов не запускался: make_channels уже делит на σ(x). "
            f"Отношение медианных σ °C равно {fmt(payload.get('physical_sigma_ratio'))}."
        )
        lines.append("")
    lines.append(
        "Калибровка `--calib per_file` в `scripts/run_real.py` (по умолчанию остаётся `--calib global`). "
        "Пример: `python scripts/run_real.py --calib per_file --far 1e-3`. "
        "Доля флагов считается по валидным клеткам файла, без карточек."
    )
    lines.append("")
    lines.append("| режим | presence=yes, тревожная зона | presence=no, ложная тревога | yes, любая зона |")
    lines.append("|---|---:|---:|---:|")
    cal = payload["calib"]
    labels = {
        "global_far_1e-3": "глобальный far 1e-3",
        "per_file_1e-3": "пофайловый 1e-3",
        "per_file_3e-3": "пофайловый 3e-3",
        "per_file_1e-2": "пофайловый 1e-2",
    }
    for op, label in labels.items():
        y, yn, n, nn = tally_text(cal["tallies"], op, "hit_alarm")
        yz, _, _, _ = tally_text(cal["tallies"], op, "hit_zone")
        lines.append(f"| {label} | {y}/{yn} | {n}/{nn} | {yz}/{yn} |")
    yb, ynb, nb, nnb = tally_text(cal["tallies"], "global_far_1e-3", "hit_base")
    lines.append(f"| эталон haklolarp median_x3 + spatial Z | {yb}/{ynb} | {nb}/{nnb} | {yb}/{ynb} |")
    matched = payload.get("matched_calib")
    if matched:
        y, yn, n, nn = tally_text(matched["tallies"], "per_file_1e-3", "hit_alarm")
        lines.append(f"| σ-matched, пофайловый 1e-3 | {y}/{yn} | {n}/{nn} | |")
    lines.append("")
    lines.append(
        "В журнале прогресса разрыв был 7/15 у детектора против 11/15 у эталона. "
        "Числа в таблице пересчитаны на этом прогоне тем же правилом пересечения рамки."
    )
    lines.append("")
    lines.append("| файл | σ p50 | MAD ch0 | доля global | зоны global (трев.) | порог 1e-3 | зоны 1e-3 | порог 3e-3 | зоны 3e-3 | порог 1e-2 | зоны 1e-2 | зоны эталона |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    by_file = {r["file"]: r for r in payload["real_files"]}
    for name, ops in cal["files"].items():
        row = by_file.get(name, {})
        def cell(op):
            b = ops[op]
            return f"{b['n_zones']} ({b['n_alarm']})"
        lines.append("| " + " | ".join([
            name,
            fmt(row.get("sigma_p50"), 4),
            fmt(row.get("ch0_mad")),
            fmt(row.get("frac_above_global_thr"), 5),
            cell("global_far_1e-3"),
            fmt(ops["per_file_1e-3"]["threshold"], 3),
            cell("per_file_1e-3"),
            fmt(ops["per_file_3e-3"]["threshold"], 3),
            cell("per_file_3e-3"),
            fmt(ops["per_file_1e-2"]["threshold"], 3),
            cell("per_file_1e-2"),
            str(ops["n_zones_base"]),
        ]) + " |")
    lines.append("")
    lines.append("Графики: `docs/figures/domain_sigma.png`, `domain_acf.png`, `domain_spectrum.png`, `domain_channels.png`, `domain_scores.png`.")
    lines.append("")
    return "\n".join(lines)


def publish(section: str, event: str) -> None:
    docs = ROOT / "docs/EXPERIMENTS_LATE.md"
    block = f"{SECTION}\n{section.rstrip()}\n{SECTION_END}\n"
    if docs.exists():
        old = docs.read_text()
        if SECTION in old and SECTION_END in old:
            pre, rest = old.split(SECTION, 1)
            _, post = rest.split(SECTION_END, 1)
            if post.startswith("\n"):
                post = post[1:]
            docs.write_text(pre + block + ("\n" if post and not post.startswith("\n") else "") + post)
        else:
            if old and not old.endswith("\n"):
                old += "\n"
            docs.write_text(old + "\n" + block)
    else:
        docs.write_text("# Поздние эксперименты\n\n" + block)
    events = Path("/workspace/grok_logs/_events.log")
    with events.open("a") as f:
        f.write(event if event.endswith("\n") else event + "\n")


def slim_calib(cal: dict) -> dict:
    files = {}
    for name, ops in cal["files"].items():
        files[name] = {}
        for key, val in ops.items():
            if key in ("n_zones_base", "sha256", "tz_missing"):
                files[name][key] = val
                continue
            files[name][key] = {k: val[k] for k in ("n_zones", "n_alarm", "classes", "threshold", "flag_fraction", "zones")}
    return {"files": files, "tallies": cal["tallies"], "cards": cal["cards"], "score_tag": cal["score_tag"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-synth", type=int, default=0)
    ap.add_argument("--only", default="", help="номера файлов MGTS через запятую, пусто = все")
    ap.add_argument("--no-infer", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    os.chdir(ROOT)
    cap_blas()
    import sys
    sys.path.insert(0, str(ROOT))
    from dts import domain

    names = [s for s in args.only.split(",") if s] or [str(i) for i in range(1, 7)]
    publish_ok = not args.tag and args.limit_synth == 0 and not args.no_infer and not args.only
    thr_doc = json.loads((ROOT / "models_release/stage1_thresholds.json").read_text())
    global_thr = float(thr_doc["thresholds"]["cnn_unet"]["far_1e-3"])
    log(f"файлы {names}, global thr {global_thr:.6f}, publish={publish_ok}")

    CACHE.mkdir(parents=True, exist_ok=True)
    build_real_channels(names)
    synth = diagnose_synth(args.limit_synth, 17, global_thr)
    real = diagnose_real(names, 90, global_thr, with_scores=False)
    synth["global_thr"] = global_thr
    reasons = shift_reasons(real["pooled"], synth["pooled"])
    log("сдвиг: " + ("; ".join(reasons) if reasons else "в допусках"))
    figures = plot_all(synth, real)
    log("графики без score: " + ", ".join(Path(p).name for p in figures))

    payload = {
        "stage": "diag",
        "global_thr": global_thr,
        "method": thr_doc.get("method"),
        "synth_pooled": synth["pooled"],
        "real_pooled": {k: v for k, v in real["pooled"].items()},
        "real_files": [{k: v for k, v in row.items() if k != "sha256"} | {"sha256": row.get("sha256"),
                        "tz_missing": row.get("tz_missing"), "grid_shape": row.get("grid_shape")}
                       for row in real["files"]],
        "synth_files_brief": synth["files"],
        "shift_reasons": reasons,
        "figures": figures,
    }
    out_main = EVAL / ("domain_shift.json" if not args.tag else f"domain_shift_{args.tag}.json")
    write_json(out_main, payload)

    if args.no_infer:
        log("DOMAIN_SHIFT_DONE diag-only")
        return

    analyzer = load_analyzer()
    cap_blas()
    ensure_scores(names, analyzer, 1.0, "raw")
    real_sc = diagnose_real(names, 90, global_thr, with_scores=True)
    synth["global_thr"] = global_thr
    real_sc["global_thr"] = global_thr
    figures = plot_all(synth, real_sc)
    for row in real_sc["files"]:
        log(f"score файл {row['file']}: p99={row.get('score_p99')} доля≥thr={row.get('frac_above_global_thr')}")

    cal = thresholds_and_cards(names, global_thr, "raw", analyzer)
    ch_ratio = real_sc["pooled"]["ch0_mad"] / synth["pooled"]["ch0_mad"]
    phys_ratio = real_sc["pooled"]["sigma_p50"] / synth["pooled"]["sigma_p50"]
    matched_cal = None
    matched_ran = False
    if np.isfinite(ch_ratio) and not (0.85 <= ch_ratio <= 1.18):
        log(f"второй прогон, деление каналов на {ch_ratio:.4f}")
        tag = f"k{ch_ratio:.4f}"
        ensure_scores(names, analyzer, float(ch_ratio), tag)
        matched_cal = thresholds_and_cards(names, global_thr, tag, analyzer)
        matched_ran = True
    else:
        log(f"деление каналов не нужно, MAD ratio {ch_ratio:.4f}")

    recipe = None
    recipe_keys = ("sigma_p50", "ch0_mad", "acf_t_lag1", "acf_x_lag1")
    recipe_ok = all(np.isfinite(real_sc["pooled"].get(k, np.nan)) and np.isfinite(synth["pooled"].get(k, np.nan))
                    for k in recipe_keys)
    if reasons and recipe_ok:
        recipe = domain.augmentation_recipe(real_sc["pooled"], synth["pooled"])

    payload.update({
        "stage": "done",
        "real_pooled": real_sc["pooled"],
        "real_files": real_sc["files"],
        "shift_reasons": shift_reasons(real_sc["pooled"], synth["pooled"]),
        "channel_mad_ratio": ch_ratio,
        "physical_sigma_ratio": phys_ratio,
        "matched_ran": matched_ran,
        "recipe": recipe,
        "calib": slim_calib(cal),
        "figures": figures,
        "cli": "python scripts/run_real.py --calib per_file --far 1e-3",
    })
    if matched_cal is not None:
        payload["matched_calib"] = slim_calib(matched_cal)
    write_json(out_main, payload)
    write_json(EVAL / ("calib_per_file.json" if not args.tag else f"calib_per_file_{args.tag}.json"), slim_calib(cal))
    section = render_section(payload)
    (EVAL / "domain_shift_section.md").write_text(section)
    if publish_ok:
        y, yn, n, nn = tally_text(cal["tallies"], "global_far_1e-3", "hit_alarm")
        yp, ypn, np_, npn = tally_text(cal["tallies"], "per_file_1e-3", "hit_alarm")
        yb, ybn, nb, nbn = tally_text(cal["tallies"], "global_far_1e-3", "hit_base")
        event = (
            f"{time.strftime('%H:%M', time.gmtime())} lit domain-shift: "
            f"sigma_p50 real/synth={phys_ratio:.3f} ch0_MAD ratio={ch_ratio:.3f} "
            f"phi_t {synth['pooled']['acf_t_lag1']:.3f}->{real_sc['pooled']['acf_t_lag1']:.3f} "
            f"phi_x {synth['pooled']['acf_x_lag1']:.3f}->{real_sc['pooled']['acf_x_lag1']:.3f}; "
            f"cards global {y}/{yn} fp {n}/{nn}, per_file_1e-3 {yp}/{ypn} fp {np_}/{npn}, "
            f"baseline {yb}/{ybn} fp {nb}/{nbn}; recipe={'yes' if recipe else 'no'}; "
            f"sigma_match={'yes' if matched_ran else 'no'}"
        )
        publish(section, event)
        log(event)
    log("DOMAIN_SHIFT_DONE")


if __name__ == "__main__":
    main()
