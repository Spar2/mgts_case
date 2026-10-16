"""Фреше-расстояние на bottleneck обученной DTSUNet (FBD, аналог FID).

Признак патча t×x — глобальный средний пул выхода энкодера (блок e4 после TimeGate),
веса заморожены (models_release/det0_unet.pt или det1). Реальные записи МГТС
допустимы только как выборка для оценки: на них ничего не обучается и не
калибруется, метки событий им не приписываются.

Рядом — вариант ТЗ: KL гистограмм «зона против тихих профилей того же сегмента»
по полю, которое выдала сеть. Зоны передаются снаружи (фиксированный эталон
или разметка синтетики), а не из оцениваемой карты.
"""
from __future__ import annotations

import zlib

import numpy as np
from scipy.linalg import sqrtm
from scipy.ndimage import find_objects, label

_EPS_SCHEDULE = (1e-6, 1e-4, 1e-2, 1e-1)


def stable_seed(base: int, *parts) -> int:
    """Целый seed, стабильный между процессами (не встроенный hash())."""
    chunks = [int(base) & 0xFFFFFFFF]
    for part in parts:
        chunks.append(zlib.crc32(str(part).encode()) & 0xFFFFFFFF)
    return int(np.random.SeedSequence(chunks).generate_state(1)[0])


def segment_ids(time_s: np.ndarray, max_gap_s: float = 300.0) -> np.ndarray:
    """Номер непрерывного отрезка времени. Разрыв > max_gap_s — граница сегмента."""
    t = np.asarray(time_s, np.float64)
    if t.ndim != 1 or len(t) == 0:
        raise ValueError("time_s must be a non-empty 1d array")
    sid = np.zeros(len(t), np.int32)
    if len(t) == 1:
        return sid
    cuts = np.r_[0, np.flatnonzero(np.diff(t) > max_gap_s) + 1, len(t)]
    for k, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
        sid[a:b] = k
    return sid


def _integral(mask: np.ndarray) -> np.ndarray:
    acc = np.cumsum(np.cumsum(mask.astype(np.int32), axis=0), axis=1)
    out = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), np.int32)
    out[1:, 1:] = acc
    return out


def _rect(ii: np.ndarray, t0: int, w0: int, th: int, tw: int) -> int:
    return int(ii[t0 + th, w0 + tw] - ii[t0, w0 + tw] - ii[t0 + th, w0] + ii[t0, w0])


def sample_patches(X, valid, n: int, patch: tuple[int, int], seed: int, ban=None, min_valid: float = 0.8,
                   max_tries: int | None = None, return_origins: bool = False):
    """Случайные патчи (N,C,th,tw). ban=True — клетка запрещена (зона): такой патч не берётся.

    N может быть меньше n, если валидных окон не хватило. Один и тот же seed даёт те же окна.
    return_origins=True добавляет массив (N, 2) левых верхних углов (t0, w0).
    """
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError("X must be (C,T,W)")
    valid = np.asarray(valid, bool)
    _, T, W = X.shape
    th, tw = int(patch[0]), int(patch[1])
    empty = np.zeros((0, X.shape[0], th, tw), np.float32)
    no_origin = np.zeros((0, 2), np.int32)

    def finish_empty():
        return (empty, no_origin) if return_origins else empty

    if n <= 0 or T < th or W < tw or th < 8 or tw < 8:
        return finish_empty()
    vii = _integral(valid)
    bii = _integral(ban) if ban is not None else None
    need = int(np.ceil(min_valid * th * tw))
    rng = np.random.default_rng(int(seed))
    tries_left = int(max_tries if max_tries is not None else max(1000, n * 120))
    origins = []
    seen = set()
    while len(origins) < n and tries_left > 0:
        tries_left -= 1
        t0 = int(rng.integers(0, T - th + 1))
        w0 = int(rng.integers(0, W - tw + 1))
        key = (t0, w0)
        if key in seen:
            continue
        if _rect(vii, t0, w0, th, tw) < need:
            continue
        if bii is not None and _rect(bii, t0, w0, th, tw) > 0:
            continue
        seen.add(key)
        origins.append(key)
    if not origins:
        return finish_empty()
    out = np.empty((len(origins), X.shape[0], th, tw), np.float32)
    for i, (t0, w0) in enumerate(origins):
        out[i] = np.asarray(X[:, t0:t0 + th, w0:w0 + tw], np.float32)
    if return_origins:
        return out, np.asarray(origins, np.int32)
    return out


def boxes_to_mask(t, x, zones) -> np.ndarray:
    """Прямоугольники зон (метры и секунды) на сетке (t, x). Это маска исключения, не метки классов."""
    t = np.asarray(t, np.float64)
    x = np.asarray(x, np.float64)
    mask = np.zeros((len(t), len(x)), bool)
    for z in zones:
        i0 = int(np.searchsorted(x, float(z["start_m"]), side="left"))
        i1 = int(np.searchsorted(x, float(z["end_m"]), side="right"))
        j0 = int(np.searchsorted(t, float(z["t_start_s"]), side="left"))
        j1 = int(np.searchsorted(t, float(z["t_end_s"]), side="right"))
        i0 = max(i0, 0)
        j0 = max(j0, 0)
        i1 = min(i1, len(x))
        j1 = min(j1, len(t))
        if i1 > i0 and j1 > j0:
            mask[j0:j1, i0:i1] = True
    return mask


class BottleneckEncoder:
    """Замороженный энкодер DTSUNet: bottleneck e4+TimeGate, global average pool."""

    def __init__(self, model, width: int, path: str | None = None):
        self.model = model
        self.model.eval()
        self.width = int(width)
        self.dim = int(width) * 8
        self.path = path

    @classmethod
    def load(cls, path, map_location: str = "cpu") -> "BottleneckEncoder":
        import torch
        from dts.models import DTSUNet

        sd = torch.load(path, map_location=map_location, weights_only=True)
        width = int(sd["e1.0.weight"].shape[0])
        model = DTSUNet(w=width)
        model.load_state_dict(sd)
        model.eval()
        return cls(model, width, str(path))

    def embed(self, patches: np.ndarray, batch: int = 8) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        patches = np.asarray(patches, np.float32)
        if patches.size == 0:
            return np.zeros((0, self.dim), np.float32)
        if patches.ndim != 4:
            raise ValueError("patches must be (N,C,H,W)")
        outs = []
        model = self.model
        with torch.inference_mode():
            for i in range(0, len(patches), int(batch)):
                x = torch.from_numpy(np.ascontiguousarray(patches[i:i + int(batch)]))
                e1 = model.e1(x)
                e2 = model.e2(F.max_pool2d(e1, 2))
                e3 = model.e3(F.max_pool2d(e2, 2))
                e4 = model.gate(model.e4(F.max_pool2d(e3, 2)))
                outs.append(e4.mean(dim=(2, 3)).numpy())
        feat = np.concatenate(outs, 0).astype(np.float32, copy=False)
        if not np.isfinite(feat).all():
            raise RuntimeError("encoder produced non-finite features")
        return feat


def _gaussian_stats(feats: np.ndarray, shrink: bool):
    X = np.asarray(feats, np.float64)
    if X.ndim != 2 or X.shape[0] < 2:
        raise ValueError("need at least 2 feature rows")
    shrinkage = None
    if shrink:
        from sklearn.covariance import LedoitWolf

        lw = LedoitWolf().fit(X)
        mu = np.asarray(lw.location_, np.float64)
        cov = np.asarray(lw.covariance_, np.float64)
        shrinkage = float(lw.shrinkage_)
    else:
        mu = X.mean(axis=0)
        cov = np.cov(X, rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]])
    cov = 0.5 * (cov + cov.T)
    return mu, cov, shrinkage


def _covmean(s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    d = s1.shape[0]
    eye = np.eye(d, dtype=np.float64)
    last = None
    for eps in _EPS_SCHEDULE:
        a = s1 + eps * eye
        b = s2 + eps * eye
        prod = a @ b
        cm = sqrtm(prod)
        last = cm
        if not np.isfinite(cm).all():
            continue
        real = np.real(cm)
        imag = float(np.max(np.abs(np.imag(cm)))) if np.iscomplexobj(cm) else 0.0
        scale = float(np.max(np.abs(real))) + 1.0
        if imag <= 1e-3 * scale and np.isfinite(real).all():
            return 0.5 * (real + real.T)
    if last is None:
        return eye
    real = np.real(last)
    real = np.where(np.isfinite(real), real, 0.0)
    return 0.5 * (real + real.T)


def frechet_from_stats(mu1, cov1, mu2, cov2) -> float:
    """||μ1−μ2||² + Tr(Σ1 + Σ2 − 2(Σ1 Σ2)^{1/2}). Отрицательный шум sqrtm обрезается в 0."""
    mu1 = np.asarray(mu1, np.float64)
    mu2 = np.asarray(mu2, np.float64)
    s1 = np.asarray(cov1, np.float64)
    s2 = np.asarray(cov2, np.float64)
    diff = mu1 - mu2
    cm = _covmean(s1, s2)
    # След считаем по тем же матрицам, к которым подобрался рабочий ridge внутри _covmean:
    # для устойчивости берём след произведения через уже посчитанный covmean и сырые Σ.
    val = float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2.0 * np.trace(cm))
    if not np.isfinite(val):
        return float("nan")
    return float(max(val, 0.0))


def frechet_diag(feats_a, feats_b) -> float:
    """Фреше при диагональных ковариациях: устойчивая оценка при n ≈ d."""
    a = np.asarray(feats_a, np.float64)
    b = np.asarray(feats_b, np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] < 2 or b.shape[0] < 2:
        raise ValueError("need at least 2 rows in each set")
    mu1, mu2 = a.mean(0), b.mean(0)
    v1 = np.maximum(a.var(0, ddof=1), 1e-12)
    v2 = np.maximum(b.var(0, ddof=1), 1e-12)
    return float(np.sum((mu1 - mu2) ** 2) + np.sum((np.sqrt(v1) - np.sqrt(v2)) ** 2))


def frechet_distance(feats_a, feats_b, shrink: bool = True) -> dict:
    """FBD двух облаков признаков. shrink=True — ковариация Ледуа–Вольфа (устойчива при n~d)."""
    mu1, c1, sh1 = _gaussian_stats(feats_a, shrink)
    mu2, c2, sh2 = _gaussian_stats(feats_b, shrink)
    return {
        "fbd": frechet_from_stats(mu1, c1, mu2, c2),
        "fbd_diag": frechet_diag(feats_a, feats_b),
        "n_a": int(len(feats_a)),
        "n_b": int(len(feats_b)),
        "feature_dim": int(np.asarray(feats_a).shape[1]),
        "shrinkage_a": sh1,
        "shrinkage_b": sh2,
    }


def bootstrap_fbd(feats_a, feats_b, n_boot: int = 200, seed: int = 0, shrink: bool = True) -> dict:
    """Точечный FBD и процентильный 95% ДИ по бутстрепу строк признаков. Seed фиксирует ДИ."""
    a = np.asarray(feats_a, np.float64)
    b = np.asarray(feats_b, np.float64)
    point = frechet_distance(a, b, shrink=shrink)
    rng = np.random.default_rng(int(seed))
    vals = np.empty(int(n_boot), np.float64)
    na, nb = len(a), len(b)
    for i in range(int(n_boot)):
        aa = a[rng.integers(0, na, na)]
        bb = b[rng.integers(0, nb, nb)]
        vals[i] = frechet_distance(aa, bb, shrink=shrink)["fbd"]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    out = dict(point)
    out.update(ci95=[float(lo), float(hi)], boot_mean=float(np.mean(vals)), n_boot=int(n_boot), seed=int(seed))
    return out


def sample_zone_patches(X, valid, zone, n: int, patch: tuple[int, int], seed: int, min_valid: float = 0.8,
                        min_sep: int = 8, return_origins: bool = False):
    """Патчи, каждый накрывает хотя бы одну клетку зоны. Якорь — случайная клетка зоны.

    Сначала окна разносятся на min_sep, затем добираются без зазора, если зон мало.
    return_origins=True добавляет массив (N, 2) левых верхних углов (t0, w0).
    """
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError("X must be (C,T,W)")
    valid = np.asarray(valid, bool)
    zone = np.asarray(zone, bool) & valid
    _, T, W = X.shape
    th, tw = int(patch[0]), int(patch[1])
    empty = np.zeros((0, X.shape[0], th, tw), np.float32)
    no_origin = np.zeros((0, 2), np.int32)

    def finish_empty():
        return (empty, no_origin) if return_origins else empty

    if n <= 0 or T < th or W < tw or th < 8 or tw < 8:
        return finish_empty()
    anchors = np.argwhere(zone)
    if len(anchors) == 0:
        return finish_empty()
    vii = _integral(valid)
    need = int(np.ceil(min_valid * th * tw))
    rng = np.random.default_rng(int(seed))
    origins: list[tuple[int, int]] = []
    seen = set()

    def accept(sep: int, tries: int) -> None:
        left = tries
        while len(origins) < n and left > 0:
            left -= 1
            ta, wa = (int(v) for v in anchors[int(rng.integers(0, len(anchors)))])
            t0 = ta - int(rng.integers(0, th))
            w0 = wa - int(rng.integers(0, tw))
            if t0 < 0 or w0 < 0 or t0 + th > T or w0 + tw > W:
                continue
            key = (t0, w0)
            if key in seen:
                continue
            if _rect(vii, t0, w0, th, tw) < need:
                continue
            if sep > 0 and any(max(abs(t0 - t), abs(w0 - w)) < sep for t, w in origins):
                continue
            seen.add(key)
            origins.append(key)

    accept(int(min_sep), max(1500, n * 60))
    if len(origins) < n:
        accept(0, max(1500, n * 40))
    if not origins:
        return finish_empty()
    out = np.empty((len(origins), X.shape[0], th, tw), np.float32)
    for i, (t0, w0) in enumerate(origins):
        out[i] = np.asarray(X[:, t0:t0 + th, w0:w0 + tw], np.float32)
    if return_origins:
        return out, np.asarray(origins, np.int32)
    return out


def crop_patches(X, origins, patch: tuple[int, int]) -> np.ndarray:
    """Те же окна, другой канал-тензор (C,T,W). Пустой список углов даёт (0,C,th,tw)."""
    X = np.asarray(X)
    th, tw = int(patch[0]), int(patch[1])
    origins = np.asarray(origins, np.int32).reshape(-1, 2)
    out = np.empty((len(origins), X.shape[0], th, tw), np.float32)
    for i, (t0, w0) in enumerate(origins):
        out[i] = np.asarray(X[:, int(t0):int(t0) + th, int(w0):int(w0) + tw], np.float32)
    return out


def channels_with_colored_noise(T, valid, seed: int, phi_t: float, phi_x: float, std: float,
                                dx: float = 0.25) -> np.ndarray:
    """AR(1) в градусах на всём поле, затем make_channels. std<=0 возвращает каналы исходного T."""
    from dts.augment import add_colored_noise
    from dts.features import make_channels

    valid = np.asarray(valid, bool)
    T = np.asarray(T, np.float32)
    if std is None or float(std) <= 0:
        ch, _aux = make_channels(T, valid, dx)
        return ch
    rng = np.random.default_rng(int(seed))
    noisy = add_colored_noise(T, valid, rng, float(phi_t), float(phi_x), float(std))
    ch, _aux = make_channels(noisy, valid, dx)
    return ch


def bn_running_snapshot(model) -> list[dict]:
    """Копия running mean/var всех BatchNorm2d. Это статистики обучения, не параметры γ, β."""
    import torch.nn as nn

    rows = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.BatchNorm2d):
            rows.append({
                "name": name,
                "mean": mod.running_mean.detach().cpu().numpy().copy(),
                "var": mod.running_var.detach().cpu().numpy().copy(),
            })
    return rows


def bn_stats_max_abs(before: list[dict], after: list[dict]) -> float:
    """Максимальный |Δ| running mean и var. 0 — статистики не сдвинулись."""
    if len(before) != len(after):
        return float("inf")
    delta = 0.0
    for a, b in zip(before, after):
        if a["name"] != b["name"]:
            return float("inf")
        delta = max(delta, float(np.max(np.abs(a["mean"] - b["mean"]))))
        delta = max(delta, float(np.max(np.abs(a["var"] - b["var"]))))
    return delta


class SynthBNSession:
    """Накопление running-статистик BatchNorm по потоку синтетических патчей.

    Реальные записи сюда не передаются: domain обязан быть «synth». γ, β и свёртки
    не шагаются. В буфере держится не больше одного батча, хвост из одной строки
    не уходит в BN отдельным шагом.
    """

    def __init__(self, model, batch: int = 8, domain: str = "synth"):
        import torch.nn as nn

        if domain != "synth":
            raise ValueError("BN statistics are not fit on real recordings")
        self.model = model
        self.batch = max(int(batch), 2)
        self._hold: list[np.ndarray] = []
        self.n_forwarded = 0
        self._closed = False
        model.train()
        for mod in model.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.reset_running_stats()
                mod.momentum = None

    def feed(self, patches) -> None:
        if self._closed:
            raise RuntimeError("BN session is closed")
        patches = np.asarray(patches, np.float32)
        if patches.size == 0:
            return
        if patches.ndim != 4:
            raise ValueError("patches must be (N,C,H,W)")
        self._hold.append(patches)
        self._drain(final=False)

    def close(self):
        if self._closed:
            return self
        self._drain(final=True)
        import torch.nn as nn

        for mod in self.model.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.momentum = 0.1
        self.model.eval()
        self._closed = True
        if self.n_forwarded < 2:
            raise RuntimeError("BN session saw fewer than 2 synth patches")
        return self

    def _drain(self, final: bool) -> None:
        if not self._hold:
            return
        cat = self._hold[0] if len(self._hold) == 1 else np.concatenate(self._hold, 0)
        self._hold = []
        n = len(cat)
        batch = self.batch
        if not final:
            n_full = n // batch
            leftover = n % batch
            if leftover == 1 and n_full >= 1:
                n_full -= 1
                leftover = batch + 1
            emit = n_full * batch
            if emit:
                self._forward(cat[:emit])
            if n - emit:
                self._hold = [cat[emit:]]
            return
        if n >= 2:
            self._forward(cat)

    def _forward(self, patches: np.ndarray) -> None:
        import torch

        n = len(patches)
        if n < 2:
            return
        batch = self.batch
        starts = list(range(0, n, batch))
        if n - starts[-1] < 2 and len(starts) >= 2:
            starts.pop()
        with torch.no_grad():
            for i, start in enumerate(starts):
                stop = n if i + 1 == len(starts) else starts[i + 1]
                chunk = patches[start:stop]
                x = torch.from_numpy(np.ascontiguousarray(chunk))
                self.model(x)
                self.n_forwarded += int(len(chunk))


def refresh_bn_synth(model, patches, batch: int = 8, domain: str = "synth"):
    """AdaBN только на синтетических патчах: заново накопить running mean/var."""
    patches = np.asarray(patches, np.float32)
    if patches.ndim != 4 or len(patches) < 2:
        raise ValueError("need at least 2 synth patches (N,C,H,W)")
    session = SynthBNSession(model, batch=batch, domain=domain)
    session.feed(patches)
    session.close()
    return model


def polynomial_kernel(X, Y, degree: int = 3, coef0: float = 1.0) -> np.ndarray:
    """k(x, y) = (x·y / d + coef0)^degree. Так задано ядро KID (Bińkowski et al., 2018)."""
    X = np.asarray(X, np.float64)
    Y = np.asarray(Y, np.float64)
    gamma = 1.0 / X.shape[1]
    return (gamma * (X @ Y.T) + coef0) ** int(degree)


def kid_mmd(feats_a, feats_b, degree: int = 3) -> float:
    """Несмещённая оценка MMD² с полиномиальным ядром. Равные длины не обязательны формуле, но протокол их выравнивает."""
    X = np.asarray(feats_a, np.float64)
    Y = np.asarray(feats_b, np.float64)
    m, n = len(X), len(Y)
    if m < 2 or n < 2:
        raise ValueError("KID needs at least 2 rows on each side")
    kxx = polynomial_kernel(X, X, degree)
    kyy = polynomial_kernel(Y, Y, degree)
    kxy = polynomial_kernel(X, Y, degree)
    np.fill_diagonal(kxx, 0.0)
    np.fill_diagonal(kyy, 0.0)
    return float(kxx.sum() / (m * (m - 1)) + kyy.sum() / (n * (n - 1)) - 2.0 * kxy.mean())


def _r2(x, y, intercept, slope) -> float:
    pred = intercept + slope * x
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 0:
        return 1.0 if ss_res <= 1e-18 else 0.0
    return float(1.0 - ss_res / ss_tot)


def fd_infinity(feats_a, feats_b, seed: int = 0, n_sizes: int = 6, n_repeats: int = 6, shrink: bool = False) -> dict:
    """FD_∞: линейная экстраполяция FID(N) = FD_∞ + slope / N (Chong & Forsyth, CVPR 2020).

    На каждой длине N обе выборки режутся до одного N без возвращения. Ковариация — выборочная,
    как в статье: смещение, которое снимает экстраполяция, относится к обычному FID.
    """
    a = np.asarray(feats_a, np.float64)
    b = np.asarray(feats_b, np.float64)
    n = min(len(a), len(b))
    d = int(a.shape[1])
    if n < d + 8:
        point = frechet_distance(a[:n], b[:n], shrink=True)
        return {"fd_inf": None, "slope": None, "r2": None, "sizes": [], "fid_mean": [],
                "reason": f"n={n} не больше размерности {d}", "fbd_at_n": point["fbd"], "n": n, "feature_dim": d}
    lo = max(d + 8, int(round(0.5 * n)))
    if lo >= n:
        lo = max(d + 2, n // 2)
    sizes = np.unique(np.linspace(lo, n, int(max(n_sizes, 2))).astype(int))
    sizes = sizes[(sizes >= 2) & (sizes <= n)]
    if len(sizes) < 2:
        point = frechet_distance(a[:n], b[:n], shrink=True)
        return {"fd_inf": None, "slope": None, "r2": None, "sizes": [int(s) for s in sizes], "fid_mean": [],
                "reason": "меньше двух длин подвыборки", "fbd_at_n": point["fbd"], "n": int(n), "feature_dim": d}
    rng = np.random.default_rng(int(seed))
    means = []
    kept = []
    for size in sizes:
        vals = []
        for _ in range(int(n_repeats)):
            ia = rng.choice(len(a), int(size), replace=False)
            ib = rng.choice(len(b), int(size), replace=False)
            vals.append(frechet_distance(a[ia], b[ib], shrink=shrink)["fbd"])
        vals = [v for v in vals if np.isfinite(v)]
        if not vals:
            continue
        kept.append(int(size))
        means.append(float(np.mean(vals)))
    if len(means) < 2:
        return {"fd_inf": None, "slope": None, "r2": None, "sizes": kept, "fid_mean": means,
                "reason": "подвыборки Фреше не сошлись", "n": int(n), "feature_dim": d,
                "n_repeats": int(n_repeats), "shrink": bool(shrink)}
    inv = 1.0 / np.asarray(kept, np.float64)
    y = np.asarray(means, np.float64)
    slope, intercept = (float(v) for v in np.polyfit(inv, y, 1))
    return {
        "fd_inf": float(intercept),
        "slope": float(slope),
        "r2": _r2(inv, y, float(intercept), float(slope)),
        "sizes": kept,
        "fid_mean": [float(v) for v in means],
        "n": int(n),
        "feature_dim": d,
        "n_repeats": int(n_repeats),
        "shrink": bool(shrink),
    }


def domain_probe_auc(feats_a, groups_a, feats_b, groups_b, n_splits: int = 5, seed: int = 0) -> dict:
    """Логрег на эмбеддингах, AUC synth-против-real. GroupKFold по файлам: тестовый кабель модель не видела.

    Веса зонда никуда не переносятся и порог детектора не калибруют. Метки — только имя домена, не класс события.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    A = np.asarray(feats_a, np.float64)
    B = np.asarray(feats_b, np.float64)
    if len(A) < 4 or len(B) < 4:
        return {"auc_oof": None, "separation": None, "n_folds": 0, "reason": "too few rows"}
    y = np.concatenate([np.zeros(len(A), np.int64), np.ones(len(B), np.int64)])
    groups = np.concatenate([
        np.array([f"a:{g}" for g in groups_a], dtype=object),
        np.array([f"b:{g}" for g in groups_b], dtype=object),
    ])
    X = np.concatenate([A, B], 0)
    n_groups = len(set(groups.tolist()))
    splits = int(min(n_splits, n_groups))
    if splits < 2:
        return {"auc_oof": None, "separation": None, "n_folds": 0, "n_groups": n_groups, "reason": "too few groups"}
    oof = np.full(len(y), np.nan, np.float64)
    fold_aucs = []
    for train, test in GroupKFold(n_splits=splits).split(X, y, groups):
        if len(np.unique(y[train])) < 2 or len(np.unique(y[test])) < 2:
            continue
        scaler = StandardScaler()
        xt = scaler.fit_transform(X[train])
        xs = scaler.transform(X[test])
        clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs", random_state=int(seed))
        clf.fit(xt, y[train])
        prob = clf.predict_proba(xs)[:, 1]
        oof[test] = prob
        fold_aucs.append(float(roc_auc_score(y[test], prob)))
    mask = np.isfinite(oof)
    pooled = None
    if int(mask.sum()) >= 4 and len(np.unique(y[mask])) == 2:
        pooled = float(roc_auc_score(y[mask], oof[mask]))
    separation = None if pooled is None else float(max(pooled, 1.0 - pooled))
    return {
        "auc_oof": pooled,
        "auc_folds_mean": float(np.mean(fold_aucs)) if fold_aucs else None,
        "auc_folds_std": float(np.std(fold_aucs)) if fold_aucs else None,
        "separation": separation,
        "n_folds": len(fold_aucs),
        "n_splits": splits,
        "n_a": int(len(A)),
        "n_b": int(len(B)),
        "n_groups": n_groups,
        "seed": int(seed),
    }


def file_domain_probe(feats_a, groups_a, feats_b, n_chunks: int = 4, n_splits: int = 4, seed: int = 0) -> dict:
    """Зонд одного кабеля против синтетики.

    Патчи кабеля делятся на n_chunks групп по порядку выборки. GroupKFold держит кусок
    кабеля и файлы синтетики вне обучения своего теста. Веса зонда не калибруют детектор.
    """
    b = np.asarray(feats_b)
    chunks = int(max(2, min(int(n_chunks), len(b))))
    groups_b = [f"chunk{i % chunks}" for i in range(len(b))]
    out = domain_probe_auc(feats_a, groups_a, b, groups_b, n_splits=min(int(n_splits), chunks), seed=seed)
    out["n_chunks"] = chunks
    return out


def _take_even(chunks, ids, n: int, seed: int):
    rng = np.random.default_rng(int(seed))
    usable = [(np.asarray(chunk, np.float64), str(fid)) for chunk, fid in zip(chunks, ids) if len(chunk)]
    if n <= 0 or not usable:
        return np.zeros((0, 1), np.float64), []
    order = [rng.permutation(len(chunk)) for chunk, _ in usable]
    ptr = np.zeros(len(usable), np.int64)
    feats, groups = [], []
    while len(feats) < n:
        progressed = False
        for i in rng.permutation(len(usable)):
            if ptr[i] >= len(order[i]):
                continue
            feats.append(usable[i][0][int(order[i][ptr[i]])])
            groups.append(usable[i][1])
            ptr[i] += 1
            progressed = True
            if len(feats) >= n:
                break
        if not progressed:
            break
    if not feats:
        return np.zeros((0, usable[0][0].shape[1]), np.float64), []
    return np.stack(feats), groups


def assemble_balanced(pack_a: dict, pack_b: dict, seed: int = 0):
    """Одинаковое число зонных и тихих патчей с обеих сторон. Доля зон совпадает по построению.

    Внутри стороны файлы чередуются, чтобы один кабель не забил облако.
    """
    nz = min(sum(len(a) for a in pack_a["zone"]), sum(len(a) for a in pack_b["zone"]))
    nq = min(sum(len(a) for a in pack_a["quiet"]), sum(len(a) for a in pack_b["quiet"]))
    if nz < 2 or nq < 2:
        return None
    parts = {}
    for side, pack in (("a", pack_a), ("b", pack_b)):
        fz, gz = _take_even(pack["zone"], pack["ids"], nz, stable_seed(seed, side, "zone"))
        fq, gq = _take_even(pack["quiet"], pack["ids"], nq, stable_seed(seed, side, "quiet"))
        parts[side] = (np.concatenate([fz, fq]), gz + gq)
    info = {"n_zone": int(nz), "n_quiet": int(nq), "n": int(nz + nq),
            "zone_fraction": float(nz / (nz + nq))}
    fa, ga = parts["a"]
    fb, gb = parts["b"]
    return fa, ga, fb, gb, info


def kid_balanced(pack_a: dict, pack_b: dict, seed: int = 0) -> dict:
    """KID после выравнивания числа зонных и тихих патчей. Доля зон на обеих сторонах одна."""
    assembled = assemble_balanced(pack_a, pack_b, seed=seed)
    if assembled is None:
        return {"kid": None, "n_zone": 0, "n_quiet": 0, "n": 0, "zone_fraction": None}
    fa, _ga, fb, _gb, info = assembled
    info = dict(info)
    info["kid"] = kid_mmd(fa, fb)
    return info


def gap_report(feats_a, groups_a, feats_b, groups_b, seed: int = 0, n_sizes: int = 6, n_repeats: int = 6) -> dict:
    """KID, FD_∞ и AUC зонда на уже выровненных облаках одной и той же длины."""
    kid = kid_mmd(feats_a, feats_b)
    fd = fd_infinity(feats_a, feats_b, seed=seed, n_sizes=n_sizes, n_repeats=n_repeats, shrink=False)
    probe = domain_probe_auc(feats_a, groups_a, feats_b, groups_b, n_splits=5, seed=seed)
    empirical = frechet_distance(feats_a, feats_b, shrink=False)
    shrunk = frechet_distance(feats_a, feats_b, shrink=True)
    return {
        "n_a": int(len(feats_a)),
        "n_b": int(len(feats_b)),
        "feature_dim": int(np.asarray(feats_a).shape[1]),
        "kid": kid,
        "fd_inf": fd.get("fd_inf"),
        "fd_inf_detail": fd,
        "fbd_empirical": empirical["fbd"],
        "fbd_shrink": shrunk["fbd"],
        "probe": probe,
    }


def far_threshold_hist(hist: np.ndarray, edges: np.ndarray, far: float) -> float:
    """Порог по гистограмме фона: доля клеток ≥ порога ≈ far. Середина бина порядковой статистики."""
    hist = np.asarray(hist)
    total = int(hist.sum())
    if total <= 0:
        return 1.0
    k = int(np.ceil((1.0 - float(far)) * total)) - 1
    k = int(np.clip(k, 0, total - 1))
    cdf = np.cumsum(hist)
    idx = int(np.searchsorted(cdf, k + 1, side="left"))
    idx = min(max(idx, 0), len(hist) - 1)
    return float(0.5 * (edges[idx] + edges[idx + 1]))


def _components(mask: np.ndarray, min_cells: int):
    lab, n = label(mask)
    if n == 0:
        return
    for i, sl in enumerate(find_objects(lab), 1):
        if sl is None:
            continue
        comp = lab[sl] == i
        if int(comp.sum()) < min_cells:
            continue
        yield sl, comp


def zone_specs(ev, valid, t, nu=None, min_cells: int = 12, min_quiet_profiles: int = 10, max_gap_s: float = 300.0):
    """Фиксированные зоны синтетики: hot ev 1–4, cold ev 5–6.

    Для каждой связной зоны — индексы клеток P и тихих клеток Q тех же столбцов x
    в том же временном сегменте. Оцениваемая карта сюда не входит.
    """
    ev = np.asarray(ev)
    valid = np.asarray(valid, bool)
    hot = valid & (ev >= 1) & (ev <= 4)
    cold = valid & (ev >= 5) & (ev <= 6)
    covered = hot | cold
    quiet = valid & ~covered
    if nu is not None:
        quiet = quiet & (np.asarray(nu) == 0)
    sid = segment_ids(t, max_gap_s)
    specs = []
    for polarity, mask in (("hot", hot), ("cold", cold)):
        for sl, comp in _components(mask, min_cells):
            rows = sl[0].start + np.flatnonzero(comp.any(1))
            cols = sl[1].start + np.flatnonzero(comp.any(0))
            seg = int(np.bincount(sid[rows]).argmax())
            seg_rows = np.flatnonzero(sid == seg)
            if len(cols) == 0 or len(seg_rows) == 0:
                specs.append({"skip": True, "polarity": polarity})
                continue
            row_has_q = quiet[np.ix_(seg_rows, cols)].any(1)
            qrows = seg_rows[row_has_q]
            if len(qrows) < min_quiet_profiles:
                specs.append({"skip": True, "polarity": polarity})
                continue
            rr, cc = np.nonzero(comp)
            pr = rr + sl[0].start
            pc = cc + sl[1].start
            keep = sid[pr] == seg
            pr, pc = pr[keep], pc[keep]
            if len(pr) < min_cells:
                specs.append({"skip": True, "polarity": polarity})
                continue
            qr = np.repeat(qrows, len(cols))
            qc = np.tile(cols, len(qrows))
            qok = quiet[qr, qc]
            qr, qc = qr[qok], qc[qok]
            if len(qr) == 0:
                specs.append({"skip": True, "polarity": polarity})
                continue
            specs.append({"skip": False, "polarity": polarity, "pr": pr, "pc": pc, "qr": qr, "qc": qc})
    return specs


def hist_kl(p_values, q_values, bins: np.ndarray, eps: float = 1e-3):
    """KL(P‖Q) по общей гистограмме. Сглаживание в пространстве вероятностей: одинаковые формы дают 0."""
    pv = np.asarray(p_values, np.float64)
    qv = np.asarray(q_values, np.float64)
    pv = pv[np.isfinite(pv)]
    qv = qv[np.isfinite(qv)]
    if pv.size == 0 or qv.size == 0:
        return None
    lo, hi = float(bins[0]), float(bins[-1])
    pv = np.clip(pv, lo, hi)
    qv = np.clip(qv, lo, hi)
    hp, _ = np.histogram(pv, bins=bins)
    hq, _ = np.histogram(qv, bins=bins)
    if hp.sum() == 0 or hq.sum() == 0:
        return None
    p = hp.astype(np.float64) / hp.sum()
    q = hq.astype(np.float64) / hq.sum()
    p = p + float(eps)
    q = q + float(eps)
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def kl_zone_quiet(field, specs, n_bins: int = 64, eps: float = 1e-3, value_range=(0.0, 1.0)) -> dict:
    """Средний KL по зонам. field — выход сети (вероятность), specs — фиксированные зоны."""
    bins = np.linspace(float(value_range[0]), float(value_range[1]), int(n_bins) + 1)
    field = np.asarray(field)
    hot, cold, skipped = [], [], 0
    for sp in specs:
        if sp.get("skip"):
            skipped += 1
            continue
        kl = hist_kl(field[sp["pr"], sp["pc"]], field[sp["qr"], sp["qc"]], bins, eps=eps)
        if kl is None or not np.isfinite(kl):
            skipped += 1
            continue
        (hot if sp["polarity"] == "hot" else cold).append(kl)
    return {"hot": hot, "cold": cold, "n_skipped": int(skipped)}


def kl_summary(values) -> dict:
    if not values:
        return {"mean": None, "median": None, "n": 0}
    a = np.asarray(values, np.float64)
    return {"mean": float(a.mean()), "median": float(np.median(a)), "n": int(a.size)}


def spearman_rho(x, y) -> dict:
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < 3:
        return {"rho": None, "pvalue": None, "n": n}
    from scipy.stats import spearmanr

    result = spearmanr(x[mask], y[mask])
    rho = getattr(result, "statistic", None)
    if rho is None:
        rho = result.correlation
    p = float(result.pvalue)
    rho = float(rho)
    if not np.isfinite(rho):
        return {"rho": None, "pvalue": None, "n": n}
    return {"rho": rho, "pvalue": p, "n": n}


def selector_verdict(rho_hot, rho_cold, threshold: float = 0.6) -> str:
    """Селектор только если обе ранговые корреляции с IoU не ниже порога. Иначе диагностика."""
    if rho_hot is None or rho_cold is None:
        return "diagnostics"
    if not (np.isfinite(rho_hot) and np.isfinite(rho_cold)):
        return "diagnostics"
    if float(rho_hot) >= threshold and float(rho_cold) >= threshold:
        return "selector"
    return "diagnostics"
