"""Реалистичный шум фона по статистике artifacts/real_noise_psd.npz.

Спектр и АКФ сняты с фона шести файлов МГТС (без карточек и без меток).
Сами трассы здесь не читаются: в обучение и в подбор порога они не входят.
Генератор — разделимый линейный фильтр (свёртка через FFT). Веса фильтра
задаёт измеренная АКФ, короткое уточнение логарифма амплитуды — по белому
шуму, без реальных отсчётов. Дообучение U-Net вызывает apply_tile: шум
в единицах σ добавляется к остатку до make_channels.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from dts import features
from dts.augment import feature_delta, swap_hot_cold

CLIP = features.CLIP


def default_psd_path() -> Path:
    return Path(__file__).resolve().parents[1] / "artifacts" / "real_noise_psd.npz"


def _fill_nonfinite(rows: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    out = np.array(rows, np.float64, copy=True)
    if out.ndim == 1:
        out = out[None]
    for i in range(out.shape[0]):
        bad = ~np.isfinite(out[i])
        if not bad.any():
            continue
        if fallback is not None and np.isfinite(fallback).all():
            out[i, bad] = fallback[bad]
        else:
            good = np.isfinite(out[i])
            if good.any():
                out[i, bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(good), out[i, good])
            else:
                out[i] = 0.0
    return out


def _renorm_acf(acf: np.ndarray) -> np.ndarray:
    out = _fill_nonfinite(acf)
    for i in range(out.shape[0]):
        scale = out[i, 0]
        if not np.isfinite(scale) or abs(scale) < 1e-8:
            out[i] = 0.0
            out[i, 0] = 1.0
        else:
            out[i] /= scale
            out[i, 0] = 1.0
    return out


class Bank:
    """Набор спектров: по одному на файл и сводный (последняя строка)."""

    def __init__(self, freq, psd_t, psd_x, acf_t, acf_x, names, sha256: str, path: str):
        self.freq = np.asarray(freq, np.float64)
        self.psd_t = np.asarray(psd_t, np.float64)
        self.psd_x = np.asarray(psd_x, np.float64)
        self.acf_t = np.asarray(acf_t, np.float64)
        self.acf_x = np.asarray(acf_x, np.float64)
        self.names = list(names)
        self.sha256 = sha256
        self.path = path

    @property
    def n_src(self) -> int:
        return int(self.psd_t.shape[0])

    def phi(self, src: int) -> tuple[float, float]:
        return float(self.acf_t[src, 1]), float(self.acf_x[src, 1])

    def as_meta(self) -> dict:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "n_src": self.n_src,
            "names": self.names,
            "phi_t": [float(v) for v in self.acf_t[:, 1]],
            "phi_x": [float(v) for v in self.acf_x[:, 1]],
        }


def load_bank(path: str | Path | None = None) -> Bank:
    path = Path(path) if path else default_psd_path()
    raw = np.load(path)
    freq = np.asarray(raw["freq"], np.float64)
    pooled_t = np.asarray(raw["pooled_psd_t"], np.float64)
    pooled_x = np.asarray(raw["pooled_psd_x"], np.float64)
    psd_t = _fill_nonfinite(raw["psd_t"], pooled_t)
    psd_x = _fill_nonfinite(raw["psd_x"], pooled_x)
    pooled_t = _fill_nonfinite(pooled_t[None])[0]
    pooled_x = _fill_nonfinite(pooled_x[None])[0]
    acf_t = _renorm_acf(np.vstack([raw["acf_t"], raw["pooled_acf_t"]]))
    acf_x = _renorm_acf(np.vstack([raw["acf_x"], raw["pooled_acf_x"]]))
    psd_t = np.vstack([psd_t, pooled_t])
    psd_x = np.vstack([psd_x, pooled_x])
    psd_t = np.maximum(psd_t, 0.0)
    psd_x = np.maximum(psd_x, 0.0)
    names = [f"file{int(i)}" for i in raw["file_id"]] + ["pooled"]
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return Bank(freq, psd_t, psd_x, acf_t, acf_x, names, sha, str(path))


def synthetic_bank(phi_t: float = 0.55, phi_x: float = 0.4, n_freq: int = 64, n_lag: int = 32) -> Bank:
    """АР(1)-банк для тестов. Реальные файлы не читает."""
    freq = np.linspace(0.0, 0.5, n_freq)
    lags = np.arange(n_lag + 1, dtype=np.float64)

    def psd(phi):
        w = 2 * np.pi * freq
        p = (1 - phi * phi) / (1 - 2 * phi * np.cos(w) + phi * phi)
        return p

    def acf(phi):
        return phi ** lags

    return Bank(
        freq,
        np.stack([psd(phi_t), psd(phi_t * 0.5)]),
        np.stack([psd(phi_x), psd(phi_x * 0.5)]),
        np.stack([acf(phi_t), acf(phi_t * 0.5)]),
        np.stack([acf(phi_x), acf(phi_x * 0.5)]),
        ["ar", "ar_half"],
        sha256="synthetic",
        path="synthetic",
    )


def acf_embed_amplitude(acf: np.ndarray, n: int) -> np.ndarray:
    """Амплитуда rFFT, чья АКФ совпадает с acf до лага len(acf)-1 (обрезание дальше)."""
    acf = np.asarray(acf, np.float64)
    if acf[0] == 0:
        raise ValueError("acf[0] = 0")
    acf = acf / acf[0]
    r = np.zeros(n, np.float64)
    m = min(acf.size - 1, n // 2)
    r[0] = 1.0
    r[1:m + 1] = acf[1:m + 1]
    if m:
        r[-m:] = acf[1:m + 1][::-1]
    power = np.fft.rfft(r).real
    return np.sqrt(np.maximum(power, 0.0)).astype(np.float64)


def psd_amplitude(psd: np.ndarray, freq: np.ndarray, n: int) -> np.ndarray:
    f = np.fft.rfftfreq(n)
    p = np.interp(f, np.asarray(freq, np.float64), np.maximum(np.asarray(psd, np.float64), 0.0))
    return np.sqrt(p).astype(np.float64)


def choose_amplitude(acf, psd, freq, n: int) -> tuple[np.ndarray, str]:
    amp = acf_embed_amplitude(acf, n)
    clipped = float(np.mean(amp <= 1e-8))
    if clipped > 0.3 or not np.isfinite(amp).all() or float(amp.sum()) <= 0:
        return psd_amplitude(psd, freq, n), "psd"
    return amp, "acf"


def mean_acf_2d(field: np.ndarray, max_lag: int, axis: int) -> np.ndarray:
    """Средняя линейная АКФ по второй оси. Лаг 0 равен 1."""
    x = np.asarray(field, np.float64)
    x = x - np.mean(x, axis=axis, keepdims=True)
    out = np.ones(max_lag + 1, np.float64)
    if axis == 0:
        energy = np.maximum(np.sum(x * x, axis=0), 1e-12)
        for k in range(1, max_lag + 1):
            out[k] = float(np.mean(np.sum(x[:-k] * x[k:], axis=0) / energy))
    elif axis == 1:
        energy = np.maximum(np.sum(x * x, axis=1), 1e-12)
        for k in range(1, max_lag + 1):
            out[k] = float(np.mean(np.sum(x[:, :-k] * x[:, k:], axis=1) / energy))
    else:
        raise ValueError(axis)
    return out


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    if den <= 0 or not np.isfinite(den):
        return float("nan")
    return float(np.sum(a * b) / den)


def acf_match(field: np.ndarray, acf_t: np.ndarray, acf_x: np.ndarray, max_lag: int | None = None) -> dict:
    lt = min(len(acf_t) - 1, field.shape[0] // 4)
    lx = min(len(acf_x) - 1, field.shape[1] // 4)
    if max_lag is not None:
        lt = min(lt, max_lag)
        lx = min(lx, max_lag)
    got_t = mean_acf_2d(field, lt, 0)
    got_x = mean_acf_2d(field, lx, 1)
    return {
        "phi_t": float(got_t[1]) if lt >= 1 else float("nan"),
        "phi_x": float(got_x[1]) if lx >= 1 else float("nan"),
        "target_phi_t": float(acf_t[1]),
        "target_phi_x": float(acf_x[1]),
        "corr_t": _pearson(got_t[1:], acf_t[1:lt + 1]),
        "corr_x": _pearson(got_x[1:], acf_x[1:lx + 1]),
        "lag8_t": float(got_t[min(8, lt)]),
        "target_lag8_t": float(acf_t[min(8, len(acf_t) - 1)]),
    }


class SpectralShaper(nn.Module):
    """Разделимый фильтр: rFFT по времени и по x, амплитуда из АКФ банка.

    log_delta_* — поправка на сетке freq банка. Ноль оставляет измеренную АКФ.
    """

    def __init__(self, bank: Bank):
        super().__init__()
        self.n_src = bank.n_src
        self.register_buffer("freq", torch.from_numpy(np.asarray(bank.freq, np.float32)))
        self.register_buffer("acf_t", torch.from_numpy(np.asarray(bank.acf_t, np.float32)))
        self.register_buffer("acf_x", torch.from_numpy(np.asarray(bank.acf_x, np.float32)))
        self.register_buffer("psd_t", torch.from_numpy(np.asarray(bank.psd_t, np.float32)))
        self.register_buffer("psd_x", torch.from_numpy(np.asarray(bank.psd_x, np.float32)))
        self.log_delta_t = nn.Parameter(torch.zeros(bank.n_src, bank.freq.size))
        self.log_delta_x = nn.Parameter(torch.zeros(bank.n_src, bank.freq.size))
        self._amp_cache: dict[tuple[int, int, int], torch.Tensor] = {}

    def base_amplitude(self, src: int, n: int, axis: int) -> torch.Tensor:
        key = (int(src), int(n), int(axis))
        hit = self._amp_cache.get(key)
        if hit is not None:
            return hit
        if axis == 0:
            amp, _ = choose_amplitude(
                self.acf_t[src].detach().cpu().numpy(),
                self.psd_t[src].detach().cpu().numpy(),
                self.freq.detach().cpu().numpy(),
                n,
            )
        else:
            amp, _ = choose_amplitude(
                self.acf_x[src].detach().cpu().numpy(),
                self.psd_x[src].detach().cpu().numpy(),
                self.freq.detach().cpu().numpy(),
                n,
            )
        t = torch.from_numpy(amp.astype(np.float32))
        self._amp_cache[key] = t
        return t

    def _gain(self, log_delta_row: torch.Tensor, n: int) -> torch.Tensor:
        dst = torch.from_numpy(np.fft.rfftfreq(n).astype(np.float32))
        src = self.freq
        idx = torch.searchsorted(src, dst.clamp(max=float(src[-1])))
        idx = idx.clamp(1, src.numel() - 1)
        f0, f1 = src[idx - 1], src[idx]
        w = (dst - f0) / (f1 - f0).clamp(min=1e-8)
        d0, d1 = log_delta_row[idx - 1], log_delta_row[idx]
        return torch.exp((d0 + w * (d1 - d0)).clamp(-1.5, 1.5))

    def forward(self, white: torch.Tensor, src: int) -> torch.Tensor:
        """white (N, 1, T, W) -> поле той же формы."""
        n, _, t, w = white.shape
        at = self.base_amplitude(src, t, 0).to(white.dtype) * self._gain(self.log_delta_t[src], t).to(white.dtype)
        ax = self.base_amplitude(src, w, 1).to(white.dtype) * self._gain(self.log_delta_x[src], w).to(white.dtype)
        ft = torch.fft.rfft(white, dim=-2)
        y = torch.fft.irfft(ft * at.view(1, 1, -1, 1), n=t, dim=-2)
        fx = torch.fft.rfft(y, dim=-1)
        return torch.fft.irfft(fx * ax.view(1, 1, 1, -1), n=w, dim=-1)

    def sample(self, rng: np.random.Generator, shape, src: int, std: float, valid: np.ndarray | None = None) -> np.ndarray:
        """Поле shape (T, W) со СКО std на валидных клетках. std<=0 → нули."""
        shape = tuple(int(v) for v in shape)
        if std is None or std <= 0:
            return np.zeros(shape, np.float32)
        white = rng.standard_normal(shape).astype(np.float32)
        wt = torch.from_numpy(white)[None, None]
        with torch.no_grad():
            field = self.forward(wt, int(src))[0, 0].numpy()
        m = np.ones(shape, bool) if valid is None else np.asarray(valid, bool)
        sel = field[m]
        if sel.size == 0:
            return np.zeros(shape, np.float32)
        field = field - float(sel.mean())
        sel = field[m]
        scale = float(sel.std())
        if scale < 1e-8:
            return np.zeros(shape, np.float32)
        field = (field * np.float32(std / scale)).astype(np.float32)
        if valid is not None:
            field = np.where(m, field, 0.0).astype(np.float32)
        return field


def _acf_loss(field: torch.Tensor, tgt_t: torch.Tensor, tgt_x: torch.Tensor, max_lag: int) -> torch.Tensor:
    """field (B, T, W). MSE АКФ по лагам 1..max_lag."""

    def one(axis: int, tgt: torch.Tensor) -> torch.Tensor:
        x = field - field.mean(dim=axis, keepdim=True)
        L = min(max_lag, tgt.numel() - 1)
        if axis == -2:
            energy = (x * x).sum(-2).clamp(min=1e-6)
            pred = []
            for k in range(1, L + 1):
                pred.append(((x[:, :-k] * x[:, k:]).sum(-2) / energy).mean(-1))
        else:
            energy = (x * x).sum(-1).clamp(min=1e-6)
            pred = []
            for k in range(1, L + 1):
                pred.append(((x[:, :, :-k] * x[:, :, k:]).sum(-1) / energy).mean(-1))
        got = torch.stack(pred, -1)
        return (got - tgt[1:L + 1]).pow(2).mean()

    return one(-2, tgt_t) + one(-1, tgt_x)


def fit_shaper(bank: Bank, steps: int = 20, seed: int = 0, tile=(160, 256), max_lag: int = 16) -> tuple[SpectralShaper, dict]:
    """Короткая подгонка log-поправки. Цель — АКФ банка, вход — белый шум.

    Реальные трассы не используются. Если поправка не поднимает корреляцию
    на фиксированном пробнике, остаются нулевые delta (чистая измеренная АКФ).
    """
    torch.manual_seed(seed)
    shaper = SpectralShaper(bank)
    rng = np.random.default_rng(seed + 11)
    probe = [rng.standard_normal(tile).astype(np.float32) for _ in range(3)]
    src_probe = min(bank.n_src - 1, bank.n_src)  # сводный, если он есть

    def probe_corr(module: SpectralShaper) -> float:
        module.eval()
        corrs = []
        for white in probe:
            with torch.no_grad():
                field = module.forward(torch.from_numpy(white)[None, None], src_probe)[0, 0].numpy()
            m = acf_match(field, bank.acf_t[src_probe], bank.acf_x[src_probe], max_lag=max_lag)
            corrs.append(0.5 * (m["corr_t"] + m["corr_x"]))
        return float(np.nanmean(corrs))

    base_corr = probe_corr(shaper)
    info = {"steps": int(steps), "base_acf_corr": base_corr, "best_acf_corr": base_corr, "accepted_fit": False,
            "probe_src": bank.names[src_probe], "max_lag": int(max_lag)}
    if steps <= 0:
        shaper.eval()
        for p in shaper.parameters():
            p.requires_grad_(False)
        return shaper, info

    opt = torch.optim.Adam([shaper.log_delta_t, shaper.log_delta_x], lr=0.05)
    best_state = {k: v.detach().cpu().clone() for k, v in shaper.state_dict().items()}
    best = base_corr
    for _ in range(int(steps)):
        shaper.train()
        src = int(rng.integers(0, bank.n_src))
        white = torch.from_numpy(rng.standard_normal((2, 1, tile[0], tile[1])).astype(np.float32))
        pred = shaper.forward(white, src)[:, 0]
        loss = _acf_loss(pred, shaper.acf_t[src], shaper.acf_x[src], max_lag)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([shaper.log_delta_t, shaper.log_delta_x], 1.0)
        opt.step()
        with torch.no_grad():
            shaper.log_delta_t.clamp_(-1.5, 1.5)
            shaper.log_delta_x.clamp_(-1.5, 1.5)
        shaper._amp_cache.clear()
        corr = probe_corr(shaper)
        if np.isfinite(corr) and corr > best + 1e-4:
            best = corr
            best_state = {k: v.detach().cpu().clone() for k, v in shaper.state_dict().items()}
    shaper.load_state_dict(best_state)
    shaper._amp_cache.clear()
    shaper.eval()
    for p in shaper.parameters():
        p.requires_grad_(False)
    info["best_acf_corr"] = float(probe_corr(shaper))
    info["accepted_fit"] = bool(info["best_acf_corr"] > base_corr + 1e-3)
    if not info["accepted_fit"]:
        with torch.no_grad():
            shaper.log_delta_t.zero_()
            shaper.log_delta_x.zero_()
        shaper._amp_cache.clear()
        info["best_acf_corr"] = float(probe_corr(shaper))
    info["delta_abs_mean"] = float(0.5 * (shaper.log_delta_t.abs().mean() + shaper.log_delta_x.abs().mean()))
    return shaper, info


class AugSpec:
    def __init__(self, std_lo: float = 0.25, std_hi: float = 0.70, p_apply: float = 0.7, sign_p: float = 0.0):
        self.std_lo = float(std_lo)
        self.std_hi = float(std_hi)
        self.p_apply = float(p_apply)
        self.sign_p = float(sign_p)

    def as_dict(self) -> dict:
        return dict(std_lo=self.std_lo, std_hi=self.std_hi, p_apply=self.p_apply, sign_p=self.sign_p,
                    std_units="sigma_x", note="шум в единицах σ остатка, затем make_channels")


def apply_tile(xt: np.ndarray, yt: np.ndarray, sigma, rng: np.random.Generator,
               shaper: SpectralShaper, spec: AugSpec, dx: float = 0.25):
    """Тайл (3, T, W). std — доля σ, не градусы. p_apply=0 или std<=0 оставляют каналы.

    Знак (sign_p) отражает остаток и меняет классы 1↔2. Канал valid не меняется.
    """
    xt = np.array(xt, np.float32, copy=True)
    yt = np.array(yt, copy=True)
    if spec.p_apply <= 0 or float(rng.random()) > spec.p_apply:
        return xt, yt
    valid = xt[2] > 0.5
    sig = np.maximum(np.asarray(sigma, np.float32), 0.03)
    if sig.shape[-1] != xt.shape[-1]:
        raise ValueError(f"sigma {sig.shape[-1]} != tile {xt.shape[-1]}")
    T = xt[0] * sig[None, :]
    if spec.sign_p > 0 and float(rng.random()) < spec.sign_p:
        T = -T
        xt[0] *= -1
        xt[1] *= -1
        yt = swap_hot_cold(yt)
    std = float(rng.uniform(spec.std_lo, spec.std_hi))
    if std <= 0:
        xt[:2] *= valid.astype(np.float32)
        return xt, yt
    src = int(rng.integers(0, shaper.n_src))
    noise_n = shaper.sample(rng, T.shape, src, std, valid)
    noise_c = noise_n * sig[None, :]
    delta = feature_delta(T, noise_c, valid, dx)
    xt[0] = np.clip(xt[0] + delta[0], -CLIP, CLIP)
    xt[1] = np.clip(xt[1] + delta[1], -CLIP, CLIP)
    xt[:2] *= valid.astype(np.float32)
    return xt, yt


def spectrum_gap(field: np.ndarray, valid: np.ndarray, bank: Bank, src: int = -1, max_lag: int = 24) -> dict:
    """Корреляция АКФ поля с выбранным источником банка. Поле — остаток в единицах σ."""
    m = np.asarray(valid, bool)
    work = np.where(m, np.asarray(field, np.float32), 0.0)
    # дыры не интерполируем: в АКФ они дают нулевую добавку только если поле занулено,
    # поэтому для диагностики берём самый плотный прямоугольник не нужно — считаем по всем строкам,
    # где валидных точек больше половины.
    keep_t = m.mean(axis=1) > 0.5
    keep_x = m.mean(axis=0) > 0.5
    if int(keep_t.sum()) < 16 or int(keep_x.sum()) < 16:
        sub = work
    else:
        sub = work[np.ix_(keep_t, keep_x)]
    return acf_match(sub, bank.acf_t[src], bank.acf_x[src], max_lag=max_lag)
