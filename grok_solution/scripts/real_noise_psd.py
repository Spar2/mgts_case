"""Спектр и автокорреляция остатка на реальном фоне МГТС.

Остаток r = dT_time / σ(x) без обрезки ±12. Фон — валидные клетки вне
прямоугольников зон модели (artifacts/eval/real_zones.csv, выпуск
ens_u2ft_tcn_lgbm) и эталона haklolarp (domain_cache/{n}_base.json).
Каждый прямоугольник расширен на 0.25 м и 120 с, чтобы плечо события
не входило в фон. Карточки оператора не читаются.

Запуск из /workspace/dts_solution, не больше 2 потоков:
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \\
  /workspace/repo/run/.venv/bin/python -u scripts/real_noise_psd.py
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_k] = "2"

import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CACHE = ROOT / "artifacts/eval/domain_cache"
ZONES_CSV = ROOT / "artifacts/eval/real_zones.csv"
OUT = ROOT / "artifacts/real_noise_psd.npz"

METHOD = "ens_u2ft_tcn_lgbm"
DT_S = 120.0
DX_M = 0.25
PAD_S = 120.0
PAD_M = 0.25
MAX_LAG = 48
N_BINS = 64
N_TRACES = 48
MIN_LEN = 64
TILE_T = 64
TILE_X = 128
TILE_MIN_FRAC = 0.90
TILE_CAP = 40
STRIDE_T = 16
STRIDE_X = 32
FT2_N = 17
FX2_N = 33
FILES = (1, 2, 3, 4, 5, 6)
SEED_BASE = 20261015


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S', time.gmtime())} {msg}", flush=True)


def cap_blas() -> None:
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=2)
    except Exception as exc:
        log(f"threadpoolctl: {exc}")


def rng_for(file_id: int, kind: int, mask_tag: int) -> np.random.Generator:
    """kind: 0 acf_t, 1 acf_x, 2 psd_t, 3 psd_x, 4 tiles. mask_tag: 0 фон, 1 все валидные."""
    return np.random.default_rng(SEED_BASE + int(file_id) * 100 + int(kind) * 10 + int(mask_tag))


def unix_utc(text: str) -> float:
    s = str(text).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return float(dt.timestamp())


def load_model_zones(path: Path) -> dict[int, list[dict]]:
    out = {i: [] for i in FILES}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            name = Path(row["source"]).name
            fid = int(name.split(".", 1)[0])
            if fid not in out:
                raise SystemExit(f"зона не из файлов 1–6: {name}")
            out[fid].append(dict(
                x0=float(row["x_start_m"]), x1=float(row["x_end_m"]),
                t0=unix_utc(row["t_start_utc"]), t1=unix_utc(row["t_end_utc"]),
                alarm=int(float(row["is_alarm"])),
            ))
    return out


def load_base_zones(fid: int) -> list[dict]:
    raw = json.loads((CACHE / f"{fid}_base.json").read_text())
    return [dict(x0=float(z["start_m"]), x1=float(z["end_m"]),
                 t0=float(z["t_start_s"]), t1=float(z["t_end_s"]),
                 alarm=bool(z.get("alarm", True))) for z in raw]


def paint_out(mask: np.ndarray, x: np.ndarray, t: np.ndarray, boxes: list[dict]) -> int:
    n = 0
    for b in boxes:
        ti = np.flatnonzero((t >= b["t0"] - PAD_S) & (t <= b["t1"] + PAD_S))
        xi = np.flatnonzero((x >= b["x0"] - PAD_M) & (x <= b["x1"] + PAD_M))
        if ti.size and xi.size:
            mask[np.ix_(ti, xi)] = False
            n += 1
    return n


def resid_stats(r: np.ndarray, mask: np.ndarray) -> dict:
    v = np.asarray(r, np.float64)[np.asarray(mask, bool)]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return dict(n=0, mean=np.nan, std=np.nan, mad=np.nan, frac_abs_gt_12=np.nan)
    med = float(np.median(v))
    return dict(
        n=int(v.size),
        mean=float(v.mean()),
        std=float(v.std()),
        mad=float(1.4826 * np.median(np.abs(v - med))),
        frac_abs_gt_12=float(np.mean(np.abs(v) > 12.0)),
    )


def curves(field, mask, fid: int, mask_tag: int):
    from dts import domain
    acf_t = domain.mean_acf(field, mask, 0, MAX_LAG, N_TRACES, rng_for(fid, 0, mask_tag), MIN_LEN)
    acf_x = domain.mean_acf(field, mask, 1, MAX_LAG, N_TRACES, rng_for(fid, 1, mask_tag), MIN_LEN)
    psd_t = domain.mean_psd(field, mask, 0, N_BINS, N_TRACES, rng_for(fid, 2, mask_tag), MIN_LEN)
    psd_x = domain.mean_psd(field, mask, 1, N_BINS, N_TRACES, rng_for(fid, 3, mask_tag), MIN_LEN)
    n_t = _n_kept(field, mask, 0, rng_for(fid, 0, mask_tag))
    n_x = _n_kept(field, mask, 1, rng_for(fid, 1, mask_tag))
    n_pt = _n_kept(field, mask, 0, rng_for(fid, 2, mask_tag), psd=True)
    n_px = _n_kept(field, mask, 1, rng_for(fid, 3, mask_tag), psd=True)
    return acf_t, acf_x, psd_t, psd_x, n_t, n_x, n_pt, n_px


def _n_kept(field, mask, axis: int, rng, psd: bool = False) -> int:
    from dts import domain
    need = max(MIN_LEN, MAX_LAG + 2) if not psd else MIN_LEN
    segs = domain._finite_traces(field, mask, axis, N_TRACES, need, rng)
    if psd:
        n = 0
        for seg in segs:
            seg = np.asarray(seg, np.float64)
            seg = seg - seg.mean()
            if float(np.sum(np.abs(np.fft.rfft(seg)) ** 2)) > 0:
                n += 1
        return n
    n = 0
    for seg in segs:
        a = domain.acf_1d(seg, MAX_LAG)
        if np.isfinite(a).all():
            n += 1
    return n


def _tile_candidates(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = np.asarray(mask, np.float64)
    nt, nx = m.shape
    if nt < TILE_T or nx < TILE_X:
        return np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.float64)
    ii = np.pad(m, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    ts = np.arange(0, nt - TILE_T + 1, STRIDE_T, dtype=np.int32)
    xs = np.arange(0, nx - TILE_X + 1, STRIDE_X, dtype=np.int32)
    area = float(TILE_T * TILE_X)
    tt, xx = np.meshgrid(ts, xs, indexing="ij")
    t0 = tt.ravel()
    x0 = xx.ravel()
    t1 = t0 + TILE_T
    x1 = x0 + TILE_X
    count = ii[t1, x1] - ii[t0, x1] - ii[t1, x0] + ii[t0, x0]
    frac = count / area
    keep = frac >= TILE_MIN_FRAC
    return t0[keep], x0[keep], frac[keep]


def _hann2() -> np.ndarray:
    wt = np.hanning(TILE_T).astype(np.float64)
    wx = np.hanning(TILE_X).astype(np.float64)
    return wt[:, None] * wx[None, :]


def _fold_periodogram(tile: np.ndarray) -> np.ndarray | None:
    """Односторонний периодограмм rfft по x, отрицательные частоты t сложены на положительные."""
    spec = np.fft.rfft2(tile)
    power = np.abs(spec) ** 2
    nt = tile.shape[0]
    pos = power[: nt // 2 + 1].copy()
    if nt > 2:
        neg = power[nt // 2 + 1:][::-1]
        pos[1: nt // 2] += neg
    energy = float(pos.sum())
    if energy <= 0 or not np.isfinite(energy):
        return None
    return pos / energy


def _resample_2d(power: np.ndarray, ft_src, fx_src, ft_dst, fx_dst) -> np.ndarray:
    tmp = np.empty((power.shape[0], fx_dst.size), np.float64)
    for i in range(power.shape[0]):
        tmp[i] = np.interp(fx_dst, fx_src, power[i])
    out = np.empty((ft_dst.size, fx_dst.size), np.float64)
    for j in range(fx_dst.size):
        out[:, j] = np.interp(ft_dst, ft_src, tmp[:, j])
    s = float(out.sum())
    if s <= 0 or not np.isfinite(s):
        return np.full_like(out, np.nan)
    return out / s


def spectrum_2d(resid: np.ndarray, mask: np.ndarray, fid: int):
    t0s, x0s, fracs = _tile_candidates(mask)
    n_cand = int(t0s.size)
    ft_src = np.fft.rfftfreq(TILE_T)
    fx_src = np.fft.rfftfreq(TILE_X)
    ft_dst = np.linspace(0.0, 0.5, FT2_N)
    fx_dst = np.linspace(0.0, 0.5, FX2_N)
    empty = np.full((FT2_N, FX2_N), np.nan)
    if n_cand == 0:
        return empty, 0, n_cand, np.nan
    rng = rng_for(fid, 4, 0)
    take = n_cand
    if n_cand > TILE_CAP:
        sel = rng.choice(n_cand, size=TILE_CAP, replace=False)
        t0s, x0s, fracs = t0s[sel], x0s[sel], fracs[sel]
        take = TILE_CAP
    win = _hann2()
    acc = []
    used_frac = []
    for t0, x0, fr in zip(t0s, x0s, fracs):
        sl = resid[t0: t0 + TILE_T, x0: x0 + TILE_X]
        m = mask[t0: t0 + TILE_T, x0: x0 + TILE_X]
        if int(m.sum()) < 8:
            continue
        tile = np.array(sl, np.float64, copy=True)
        mu = float(tile[m].mean())
        tile -= mu
        tile[~m] = 0.0
        tile *= win
        folded = _fold_periodogram(tile)
        if folded is None:
            continue
        acc.append(_resample_2d(folded, ft_src, fx_src, ft_dst, fx_dst))
        used_frac.append(float(fr))
    if not acc:
        return empty, 0, n_cand, np.nan
    return np.mean(acc, 0), len(acc), n_cand, float(np.mean(used_frac))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    if den <= 0 or not np.isfinite(den):
        return float("nan")
    return float(np.sum(a * b) / den)


def separability(psd_2d, psd_t, psd_x, freq, ft2, fx2) -> tuple[float, float]:
    """Корреляция psd_2d с внешним произведением 1D-спектров.

    Второй коэффициент считается без нулевой частоты: у 1D-периодограммы
    бина DC нет, потому что трасса центрирована, а окно Ханна на тайле
    возвращает энергию в нулевой бин 2D-спектра.
    """
    if not np.isfinite(psd_2d).all() or not np.isfinite(psd_t).all() or not np.isfinite(psd_x).all():
        return float("nan"), float("nan")
    pt = np.interp(ft2, freq, psd_t)
    px = np.interp(fx2, freq, psd_x)
    outer = np.outer(pt, px)
    if float(outer.sum()) <= 0:
        full = float("nan")
    else:
        full = _corr(psd_2d, outer / outer.sum())
    pt2 = np.interp(ft2[1:], freq, psd_t)
    px2 = np.interp(fx2[1:], freq, psd_x)
    outer2 = np.outer(pt2, px2)
    block = psd_2d[1:, 1:]
    if float(outer2.sum()) <= 0 or float(block.sum()) <= 0:
        nodc = float("nan")
    else:
        nodc = _corr(block / block.sum(), outer2 / outer2.sum())
    return full, nodc


def nanmean_stack(rows: list[np.ndarray]) -> np.ndarray:
    a = np.stack(rows, 0)
    with np.errstate(all="ignore"):
        return np.nanmean(a, 0)


def main() -> None:
    cap_blas()
    if not ZONES_CSV.exists():
        raise SystemExit(f"нет {ZONES_CSV}")
    zones_sha = hashlib.sha256(ZONES_CSV.read_bytes()).hexdigest()
    model = load_model_zones(ZONES_CSV)
    log(f"зоны модели { {k: len(v) for k, v in model.items()} } sha {zones_sha[:12]}")

    lag = np.arange(MAX_LAG + 1, dtype=np.int32)
    freq = np.linspace(0.0, 0.5, N_BINS)
    ft2 = np.linspace(0.0, 0.5, FT2_N)
    fx2 = np.linspace(0.0, 0.5, FX2_N)

    acf_t = np.full((6, MAX_LAG + 1), np.nan)
    acf_x = np.full((6, MAX_LAG + 1), np.nan)
    psd_t = np.full((6, N_BINS), np.nan)
    psd_x = np.full((6, N_BINS), np.nan)
    acf_t_all = np.full((6, MAX_LAG + 1), np.nan)
    acf_x_all = np.full((6, MAX_LAG + 1), np.nan)
    psd_2d = np.full((6, FT2_N, FX2_N), np.nan)
    phi_t = np.full(6, np.nan)
    phi_x = np.full(6, np.nan)
    phi_t_all = np.full(6, np.nan)
    phi_x_all = np.full(6, np.nan)
    sep = np.full(6, np.nan)
    sep_nodc = np.full(6, np.nan)
    resid_std = np.full(6, np.nan)
    resid_mad = np.full(6, np.nan)
    resid_mean = np.full(6, np.nan)
    frac_clip = np.full(6, np.nan)
    n_bg = np.zeros(6, np.int64)
    n_valid = np.zeros(6, np.int64)
    n_traces_t = np.zeros(6, np.int32)
    n_traces_x = np.zeros(6, np.int32)
    n_psd_t = np.zeros(6, np.int32)
    n_psd_x = np.zeros(6, np.int32)
    n_tiles = np.zeros(6, np.int32)
    n_tile_cand = np.zeros(6, np.int32)
    n_model = np.zeros(6, np.int32)
    n_base = np.zeros(6, np.int32)
    tile_frac = np.full(6, np.nan)

    for i, fid in enumerate(FILES):
        t0 = time.time()
        with np.load(CACHE / f"{fid}_ch.npz") as z:
            valid = np.asarray(z["valid"], np.uint8).astype(bool)
            x = np.asarray(z["x"], np.float64)
            t = np.asarray(z["t"], np.float64)
            sigma = np.asarray(z["sigma"], np.float64)
            dT = np.asarray(z["dT"], np.float64)
        if dT.shape != valid.shape or sigma.shape != (valid.shape[1],):
            raise SystemExit(f"файл {fid}: формы dT {dT.shape} valid {valid.shape} sigma {sigma.shape}")
        if x.size != valid.shape[1] or t.size != valid.shape[0]:
            raise SystemExit(f"файл {fid}: оси x {x.size} t {t.size} поле {valid.shape}")
        base = load_base_zones(fid)
        n_model[i] = len(model[fid])
        n_base[i] = len(base)
        if model[fid]:
            gap = abs(float(np.median(t)) - float(np.median([b["t0"] for b in model[fid]])))
            if gap > 7 * 86400:
                raise SystemExit(f"файл {fid}: ось t не в тех же секундах, что зоны (разрыв {gap:.0f} с)")
        sig_ok = np.isfinite(sigma) & (sigma >= 1e-6)
        resid = np.zeros(valid.shape, np.float64)
        resid[:, sig_ok] = dT[:, sig_ok] / sigma[sig_ok]
        finite = valid & np.isfinite(resid) & sig_ok[None, :]
        bg = finite.copy()
        paint_out(bg, x, t, model[fid])
        paint_out(bg, x, t, base)
        st = resid_stats(resid, bg)
        st_all = resid_stats(resid, finite)
        n_valid[i] = st_all["n"]
        n_bg[i] = st["n"]
        resid_std[i] = st["std"]
        resid_mad[i] = st["mad"]
        resid_mean[i] = st["mean"]
        frac_clip[i] = st["frac_abs_gt_12"]
        acf_t[i], acf_x[i], psd_t[i], psd_x[i], n_traces_t[i], n_traces_x[i], n_psd_t[i], n_psd_x[i] = curves(
            resid, bg, fid, 0)
        acf_t_all[i], acf_x_all[i], _, _, _, _, _, _ = curves(resid, finite, fid, 1)
        phi_t[i] = acf_t[i, 1]
        phi_x[i] = acf_x[i, 1]
        phi_t_all[i] = acf_t_all[i, 1]
        phi_x_all[i] = acf_x_all[i, 1]
        psd_2d[i], n_tiles[i], n_tile_cand[i], tile_frac[i] = spectrum_2d(resid, bg, fid)
        sep[i], sep_nodc[i] = separability(psd_2d[i], psd_t[i], psd_x[i], freq, ft2, fx2)
        frac_bg = (st["n"] / st_all["n"]) if st_all["n"] else float("nan")
        log(
            f"file {fid}: valid {st_all['n']} bg {st['n']} ({frac_bg:.3f}) "
            f"std {st['std']:.3f} mad {st['mad']:.3f} |r|>12 {st['frac_abs_gt_12']:.4f} "
            f"phi_t {phi_t[i]:.3f} (all {phi_t_all[i]:.3f}) phi_x {phi_x[i]:.3f} (all {phi_x_all[i]:.3f}) "
            f"traces {n_traces_t[i]}/{n_traces_x[i]} tiles {n_tiles[i]}/{n_tile_cand[i]} "
            f"sep {sep[i]:.3f}/{sep_nodc[i]:.3f} zones model {n_model[i]} base {n_base[i]} "
            f"{time.time() - t0:.1f}s"
        )
        del resid, bg, finite, dT

    pooled_acf_t = nanmean_stack(list(acf_t))
    pooled_acf_x = nanmean_stack(list(acf_x))
    pooled_psd_t = nanmean_stack(list(psd_t))
    pooled_psd_x = nanmean_stack(list(psd_x))
    pooled_psd_2d = nanmean_stack(list(psd_2d))
    pooled_sep, pooled_sep_nodc = separability(pooled_psd_2d, pooled_psd_t, pooled_psd_x, freq, ft2, fx2)
    note = (
        "Background residual r = unclipped dT_time/sigma_x from domain_cache/{1..6}_ch.npz. "
        "Mask = valid AND finite r AND sigma>=1e-6, outside model rectangles in "
        "artifacts/eval/real_zones.csv (method ens_u2ft_tcn_lgbm) and baseline rectangles in "
        "domain_cache/{n}_base.json. Each rectangle is padded by pad_m=0.25 and pad_s=120. "
        "Operator cards are not used. "
        "1D ACF/PSD: dts.domain.mean_acf and mean_psd, longest contiguous run, holes not interpolated, "
        f"max_lag={MAX_LAG}, psd bins={N_BINS} on linspace(0,0.5), min_len={MIN_LEN}, "
        f"up to {N_TRACES} traces, seeds {SEED_BASE}+file*100+kind*10+mask_tag. "
        "2D: tiles 64x128 with background fraction>=0.90, stride 16x32, at most 40 tiles/file, "
        "demeaned on valid cells, holes set to 0, separable Hann, rfft2, negative ft folded onto "
        "positive, each tile divided by its energy, interpolated to ft2 x fx2 and renormalized. "
        "separability correlates psd_2d with outer(psd_t, psd_x); separability_nodc drops the "
        "zero-frequency row and column, because 1D traces are demeaned (DC bin ~0) while the "
        "Hann window puts energy back into the 2D DC bin. "
        "Pooled curves are the equal-weight nanmean across the six files. "
        "acf_*_allvalid is the same residual on all valid cells, for the mask effect only. "
        "Frequencies are cycles per sample (dt=120 s, dx=0.25 m)."
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.parent / f".{OUT.stem}.partial.npz"
    np.savez_compressed(
        tmp,
        method=np.array(METHOD),
        file_id=np.array(FILES, np.int32),
        dt_s=np.float64(DT_S),
        dx_m=np.float64(DX_M),
        pad_s=np.float64(PAD_S),
        pad_m=np.float64(PAD_M),
        zones_csv_sha256=np.array(zones_sha),
        lag=lag,
        freq=freq,
        ft2=ft2,
        fx2=fx2,
        acf_t=acf_t,
        acf_x=acf_x,
        psd_t=psd_t,
        psd_x=psd_x,
        psd_2d=psd_2d,
        acf_t_allvalid=acf_t_all,
        acf_x_allvalid=acf_x_all,
        phi_t=phi_t,
        phi_x=phi_x,
        phi_t_allvalid=phi_t_all,
        phi_x_allvalid=phi_x_all,
        pooled_acf_t=pooled_acf_t,
        pooled_acf_x=pooled_acf_x,
        pooled_psd_t=pooled_psd_t,
        pooled_psd_x=pooled_psd_x,
        pooled_psd_2d=pooled_psd_2d,
        pooled_phi_t=np.float64(pooled_acf_t[1]),
        pooled_phi_x=np.float64(pooled_acf_x[1]),
        pooled_phi_t_allvalid=np.float64(nanmean_stack(list(acf_t_all))[1]),
        pooled_phi_x_allvalid=np.float64(nanmean_stack(list(acf_x_all))[1]),
        separability=sep,
        separability_nodc=sep_nodc,
        pooled_separability=np.float64(pooled_sep),
        pooled_separability_nodc=np.float64(pooled_sep_nodc),
        resid_std=resid_std,
        resid_mad=resid_mad,
        resid_mean=resid_mean,
        frac_abs_gt_12=frac_clip,
        n_bg=n_bg,
        n_valid=n_valid,
        frac_bg=np.divide(n_bg, np.maximum(n_valid, 1)),
        n_traces_t=n_traces_t,
        n_traces_x=n_traces_x,
        n_psd_t=n_psd_t,
        n_psd_x=n_psd_x,
        n_tiles=n_tiles,
        n_tile_cand=n_tile_cand,
        tile_frac_mean=tile_frac,
        n_model_zones=n_model,
        n_base_zones=n_base,
        note=np.array(note),
    )
    os.replace(tmp, OUT)
    log(
        f"pooled phi_t {pooled_acf_t[1]:.3f} (allvalid {nanmean_stack(list(acf_t_all))[1]:.3f}) "
        f"phi_x {pooled_acf_x[1]:.3f} (allvalid {nanmean_stack(list(acf_x_all))[1]:.3f}) "
        f"sep {pooled_sep:.3f} nodc {pooled_sep_nodc:.3f} -> {OUT}"
    )
    log("REAL_NOISE_PSD_DONE")


if __name__ == "__main__":
    main()
