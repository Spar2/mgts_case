"""Синтетика статичных и медленно дрейфующих узких горячих точек.

Событие уже включено на первом кадре окна: внутри окна нет фронта включения.
Кейс ТЗ по амплитуде: избыток +3…+24 °C. Длина участка в STATIC5 — физический
диапазон 2–8 м (кейс 3–5 м внутри него). Пятно чувствительности DTS порядка
1–2 м (docs/NOTE.md называет около 1 м): профиль сворачивается с этим пятном.
Шум измерения в профиль не добавляется — он уже сидит в поле-носителе, и к
дальнему концу волокна σ(x) выше. Существующие кэши не переписываются.

Поле, куда клеится горб, — пространственный остаток в градусах, а не канал
dT_time победителя: тот вычитает временной профиль и съедает статичную точку.

Базовая линия и пропуски строк — отдельные функции. `apply_profiles` их не
делает: этап 2 клеит только профили и проверяет, что горб не включается внутри
окна. `fixed_hot_specs` — только горячие события. Соседи, базовая линия и
пропуски живут в `fixed_scene` и в обучающей выборке ветви.
"""
from __future__ import annotations

import numpy as np

DX = 0.25
# Длина — физический диапазон вокруг кейса 3–5 м. Амплитуда — интервал ТЗ.
# Ни то ни другое не подгонялось под ширины и амплитуды карточек.
HOT_WIDTH = (2.0, 8.0)
HOT_AMP = (3.0, 24.0)
SPOT_M = 1.0  # нижняя граница FWHM пятна, м
SPOT_RANGE = (1.0, 2.0)
EDGE_RANGE = (0.5, 2.0)
# Множитель амплитуды во времени. Нижняя граница держит min/amp ≥ 0.8
# (проверка этапа 2) и не даёт событию погаснуть внутри окна.
ENV_LO = 0.82
ENV_HI = 1.22


def _log_uniform(rng, a, b) -> float:
    return float(np.exp(rng.uniform(np.log(a), np.log(b))))


def _spot_kernel(dx: float, spot_m: float = SPOT_M) -> np.ndarray:
    """Гауссово пятно прибора. На шаге dx уже самого пятна ядро не уже одного пикселя."""
    sigma = float(spot_m) / 2.355
    if sigma < dx:
        return np.ones(1, np.float64)
    half = int(np.ceil(3.0 * sigma / dx))
    xs = np.arange(-half, half + 1, dtype=np.float64) * dx
    k = np.exp(-0.5 * (xs / sigma) ** 2)
    k /= k.sum()
    return k


def _cosine_segment(x: np.ndarray, width_m: float, edge_m: float) -> np.ndarray:
    """Плоский участок с косинусными краями. edge_m съедает плоскую вершину, не добавляется снаружи."""
    half = width_m / 2.0
    edge = min(max(float(edge_m), 0.0), half)
    ax = np.abs(x)
    y = np.zeros(x.shape, np.float64)
    if edge <= 1e-6:
        y[ax <= half] = 1.0
        return y
    y[ax <= half - edge] = 1.0
    ramp = (ax > half - edge) & (ax <= half)
    if ramp.any():
        y[ramp] = 0.5 * (1.0 + np.cos(np.pi * (ax[ramp] - (half - edge)) / edge))
    return y


def _smooth_step(x: np.ndarray, width_m: float) -> np.ndarray:
    """Ступенька 0→1. width_m — ширина перехода, не половина кабеля."""
    half = max(float(width_m), 1e-3) / 2.0
    y = np.zeros(x.shape, np.float64)
    y[x >= half] = 1.0
    ramp = (x > -half) & (x < half)
    if ramp.any():
        y[ramp] = 0.5 * (1.0 - np.cos(np.pi * (x[ramp] + half) / (2.0 * half)))
    return y


def unit_profile(n_px: int, spec: dict, dx: float = DX) -> np.ndarray:
    """Профиль с наблюдаемым пиком 1 после пятна прибора.

    segment и dip — участок длины width_m с плавными краями. gaussian и
    raised_cosine — гладкие формы. broad — широкое плато. step — гладкая
    ступенька. Все сворачиваются с пятном spec['spot_m'] (по умолчанию 1 м)
    и нормируются так, чтобы пик был 1: amp — наблюдаемые градусы.
    """
    x = (np.arange(n_px, dtype=np.float64) - float(spec["center"])) * dx
    w = max(float(spec["width_m"]), dx)
    kind = spec.get("kind", "gaussian")
    if kind in ("segment", "dip"):
        y = _cosine_segment(x, w, float(spec.get("edge_m", 0.0)))
    elif kind == "raised_cosine":
        y = np.where(np.abs(x) <= w, 0.5 * (1.0 + np.cos(np.pi * x / w)), 0.0)
    elif kind == "broad":
        half = w / 2.0
        edge = min(4.0, half)
        ax = np.abs(x)
        y = np.zeros(n_px, np.float64)
        y[ax <= half - edge] = 1.0
        ramp = (ax > half - edge) & (ax < half)
        if edge > 0 and ramp.any():
            y[ramp] = 0.5 * (1.0 + np.cos(np.pi * (ax[ramp] - (half - edge)) / edge))
    elif kind == "step":
        y = _smooth_step(x, w)
    else:
        sigma = w / 2.355
        y = np.exp(-0.5 * (x / max(sigma, 1e-3)) ** 2)
    k = _spot_kernel(dx, float(spec.get("spot_m", SPOT_M)))
    if k.size > 1:
        y = np.convolve(y, k, mode="same")
    peak = float(y.max()) if y.size else 0.0
    if peak > 0.0:
        y = y / peak
    return y.astype(np.float32)


def fwhm_m(profile: np.ndarray, dx: float = DX) -> float:
    i = int(np.argmax(profile))
    if profile[i] <= 0:
        return 0.0
    thr = 0.5 * float(profile[i])
    left = i
    while left > 0 and profile[left] > thr:
        left -= 1
    right = i
    n = len(profile)
    while right < n - 1 and profile[right] > thr:
        right += 1
    return float((right - left) * dx)


def _occupy(avoid: np.ndarray | None, n_px: int, center: int, width_m: float, dx: float) -> np.ndarray:
    half = int(np.ceil(width_m / dx)) + 2
    out = np.zeros(n_px, bool) if avoid is None else np.array(avoid, copy=True)
    a, b = max(0, center - half), min(n_px, center + half + 1)
    out[a:b] = True
    return out


def _place_center(rng, n_px: int, width_m: float, dx: float, avoid: np.ndarray | None):
    half = int(np.ceil((width_m / 2.0) / dx)) + 2
    lo, hi = half, n_px - half
    if hi <= lo:
        return None
    for _ in range(48):
        c = int(rng.integers(lo, hi))
        sl = slice(max(0, c - half), min(n_px, c + half + 1))
        if avoid is not None and avoid[sl].any():
            continue
        return c
    return int(rng.integers(lo, hi))


def _fluct_draw(rng, n_lo: int, n_hi: int, amp_hi: float) -> list[dict]:
    n = int(rng.integers(n_lo, n_hi))
    out = []
    for _ in range(n):
        out.append(dict(
            amp=float(rng.uniform(0.0, amp_hi)),
            cycles=float(rng.uniform(0.4, 1.5)),
            phase=float(rng.uniform(0.0, 2.0 * np.pi)),
        ))
    return out


def _spec(rng, n_px, dx, avoid, *, role, kind, width, amp, drift,
          spot_m=SPOT_M, edge_m=0.0, fluct=None, mask="fwhm"):
    center = _place_center(rng, n_px, width, dx, avoid)
    if center is None:
        return None, avoid
    waves = []
    for w in fluct or []:
        waves.append(dict(amp=float(w["amp"]), cycles=float(w["cycles"]), phase=float(w["phase"])))
    sp = dict(
        center=int(center), width_m=float(width), amp=float(amp), drift=float(drift),
        kind=str(kind), role=str(role), dx=float(dx),
        spot_m=float(spot_m), edge_m=float(edge_m), fluct=waves, mask=str(mask),
    )
    return sp, _occupy(avoid, n_px, center, width, dx)


def hot_spec(rng, n_px: int, dx: float = DX, avoid: np.ndarray | None = None):
    """Нагретый участок: 2–8 м, +3…+24 °C, пятно 1–2 м, плавные края, медленный дрейф.

    Амплитуда равномерна по интервалу ТЗ. Дрейф ±20 % и одна-две медленные
    синусоиды не гасят участок: множитель времени зажат в [ENV_LO, ENV_HI].
    """
    width = float(rng.uniform(*HOT_WIDTH))
    amp = float(rng.uniform(*HOT_AMP))
    drift = float(rng.uniform(-0.20, 0.20))
    spot_m = float(rng.uniform(*SPOT_RANGE))
    edge_m = float(rng.uniform(*EDGE_RANGE))
    fluct = _fluct_draw(rng, 1, 3, 0.12)
    return _spec(
        rng, n_px, dx, avoid, role="hot", kind="segment", width=width, amp=amp, drift=drift,
        spot_m=spot_m, edge_m=edge_m, fluct=fluct, mask="fwhm",
    )


def neighbor_spec(rng, n_px: int, dx: float = DX, avoid: np.ndarray | None = None):
    """Соседняя структура, не второй нагретый участок. Роль отрицательная.

    Широкое плато, неразрешённый контакт уже кейса, гладкая ступенька
    (маска только на переходе) или холодная яма. Амплитуды — единицы–полтора
    десятка градусов, не плотность и не пики реального файла.
    """
    spot_m = float(rng.uniform(*SPOT_RANGE))
    roll = float(rng.random())
    if roll < 0.35:
        return _spec(
            rng, n_px, dx, avoid, role="neg", kind="broad",
            width=float(rng.uniform(18.0, 90.0)),
            amp=_log_uniform(rng, 3.0, 15.0),
            drift=float(rng.uniform(-0.10, 0.10)),
            spot_m=spot_m, edge_m=0.0, mask="fwhm",
        )
    if roll < 0.60:
        # Контакт неразрешим: не уже пятна и уже участка 2 м.
        return _spec(
            rng, n_px, dx, avoid, role="neg", kind="gaussian",
            width=float(rng.uniform(SPOT_M, 1.8)),
            amp=_log_uniform(rng, 3.0, 15.0),
            drift=float(rng.uniform(-0.05, 0.05)),
            spot_m=spot_m, edge_m=0.0, mask="fwhm",
        )
    if roll < 0.80:
        return _spec(
            rng, n_px, dx, avoid, role="neg", kind="step",
            width=float(rng.uniform(4.0, 24.0)),
            amp=float(rng.uniform(2.0, 12.0)),
            drift=float(rng.uniform(-0.05, 0.05)),
            spot_m=spot_m, edge_m=0.0, mask="edge",
        )
    return _spec(
        rng, n_px, dx, avoid, role="neg", kind="dip",
        width=float(rng.uniform(*HOT_WIDTH)),
        amp=-float(rng.uniform(3.0, 15.0)),
        drift=float(rng.uniform(-0.10, 0.10)),
        spot_m=spot_m, edge_m=float(rng.uniform(*EDGE_RANGE)), mask="fwhm",
    )


def _copy_avoid(avoid, n_px: int) -> np.ndarray:
    if avoid is None:
        return np.zeros(n_px, bool)
    return np.array(avoid, copy=True)


def sample_train_specs(rng, n_px: int, dx: float = DX, avoid: np.ndarray | None = None) -> list[dict]:
    """Горячие участки и 0–3 соседние структуры (роль neg). Без базовой линии и пропусков."""
    specs = []
    used = _copy_avoid(avoid, n_px)
    n_hot = 1 if rng.random() < 0.82 else int(rng.integers(0, 3))
    for _ in range(n_hot):
        sp, used = hot_spec(rng, n_px, dx, used)
        if sp:
            specs.append(sp)
    for _ in range(int(rng.integers(0, 4))):
        sp, used = neighbor_spec(rng, n_px, dx, used)
        if sp:
            specs.append(sp)
    return specs


def fixed_hot_specs(n_px: int, seed: int, n_events: int = 2, dx: float = DX,
                    avoid: np.ndarray | None = None) -> list[dict]:
    """Только горячие события. Seed задаёт параметры. Соседей здесь нет: их читает этап 2."""
    rng = np.random.default_rng(int(seed))
    specs = []
    used = _copy_avoid(avoid, n_px)
    for _ in range(int(n_events)):
        sp, used = hot_spec(rng, n_px, dx, used)
        if sp:
            specs.append(sp)
    return specs


def sample_baseline(rng) -> dict:
    """Крупномасштабная базовая линия: сдвиг, наклон и 1–3 синуса. Не горячая точка."""
    n_waves = int(rng.integers(1, 4))
    waves = []
    for _ in range(n_waves):
        waves.append(dict(
            lam=float(rng.uniform(40.0, 300.0)),
            amp=float(rng.uniform(0.3, 6.0)),
            phase=float(rng.uniform(0.0, 2.0 * np.pi)),
        ))
    return dict(
        offset=float(rng.uniform(-1.5, 1.5)),
        slope=float(rng.uniform(-0.015, 0.015)),
        waves=waves,
    )


def sample_gap(rng) -> dict:
    """Доля пропущенных строк. Параметры не зависят от длины окна."""
    return dict(
        frac=float(rng.uniform(0.0, 0.35)),
        n_blocks=int(rng.integers(1, 3)),
        scatter=float(rng.uniform(0.0, 0.04)),
        seed=int(rng.integers(0, 2 ** 31 - 1)),
    )


def fixed_scene(n_px: int, seed: int, n_events: int = 2, dx: float = DX,
                avoid: np.ndarray | None = None) -> dict:
    """Два горячих события, 0–3 соседа, одна базовая линия и одни пропуски.

    Горячие спецификации совпадают с `fixed_hot_specs` того же seed: соседи
    добираются после них. Сцена детерминирована seed.
    """
    rng = np.random.default_rng(int(seed))
    specs = []
    used = _copy_avoid(avoid, n_px)
    for _ in range(int(n_events)):
        sp, used = hot_spec(rng, n_px, dx, used)
        if sp:
            specs.append(sp)
    for _ in range(int(rng.integers(0, 4))):
        sp, used = neighbor_spec(rng, n_px, dx, used)
        if sp:
            specs.append(sp)
    return dict(specs=specs, baseline=sample_baseline(rng), gap=sample_gap(rng))


def time_multiplier(n_rows: int, spec: dict) -> np.ndarray:
    """Медленный дрейф и флуктуации. Результат зажат, событие не включается и не гаснет."""
    n = int(n_rows)
    if n <= 0:
        return np.zeros(0, np.float32)
    u = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    m = 1.0 + float(spec.get("drift", 0.0)) * u
    t = np.linspace(0.0, 1.0, n, dtype=np.float64)
    for w in spec.get("fluct") or []:
        phase = 2.0 * np.pi * float(w["cycles"]) * t + float(w["phase"])
        m = m + float(w["amp"]) * np.sin(phase)
    return np.clip(m, ENV_LO, ENV_HI).astype(np.float32)


def _mask_cols(y: np.ndarray, mode: str) -> np.ndarray:
    if mode == "edge":
        return (y >= 0.3) & (y <= 0.7)
    if mode == "none":
        return np.zeros(y.shape, bool)
    return y >= 0.5


def apply_profiles(R: np.ndarray, specs: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Добавить профили ко всем строкам времени. Включения внутри окна нет.

    Множитель строки — дрейф и медленные синусоиды, зажатые в [ENV_LO, ENV_HI].
    Базовую линию и пропуски эта функция не добавляет.
    Возвращает поле, маску горячих столбцов и маску отвлекающих столбцов.
    """
    out = np.array(R, np.float32, copy=True)
    _, width = out.shape
    pos = np.zeros(width, bool)
    hard = np.zeros(width, bool)
    if out.shape[0] == 0 or not specs:
        return out, pos, hard
    for sp in specs:
        y = unit_profile(width, sp, dx=float(sp.get("dx", DX)))
        amp = np.float32(sp["amp"]) * time_multiplier(out.shape[0], sp)
        out += amp[:, None] * y[None, :]
        cols = _mask_cols(y, str(sp.get("mask", "fwhm")))
        if sp.get("role", "hot") == "hot":
            pos |= cols
        else:
            hard |= cols
    hard &= ~pos
    return out, pos, hard


def apply_baseline(R: np.ndarray, baseline: dict | None, dx: float = DX) -> np.ndarray:
    """Прибавить гладкую базовую линию. None оставляет копию поля."""
    out = np.array(R, np.float32, copy=True)
    if not baseline or out.size == 0:
        return out
    width = out.shape[1]
    x = (np.arange(width, dtype=np.float64) - 0.5 * (width - 1)) * dx
    y = np.full(width, float(baseline.get("offset", 0.0)), np.float64)
    y += float(baseline.get("slope", 0.0)) * x
    for w in baseline.get("waves") or []:
        y += float(w["amp"]) * np.sin(2.0 * np.pi * x / float(w["lam"]) + float(w["phase"]))
    out += y.astype(np.float32)[None, :]
    return out


def row_gap_mask(n_rows: int, gap: dict | None) -> np.ndarray:
    """True — строка наблюдается. Пропуски: 1–2 блока и редкая россыпь, не больше 40 % строк."""
    n = int(n_rows)
    keep = np.ones(n, dtype=bool)
    if not gap or n < 8:
        return keep
    frac = max(0.0, float(gap.get("frac", 0.0)))
    scatter = max(0.0, float(gap.get("scatter", 0.0)))
    total_frac = min(0.40, frac + scatter)
    n_drop = int(np.floor(total_frac * n + 1e-9))
    if n_drop <= 0:
        return keep
    rng = np.random.default_rng(int(gap.get("seed", 0)) % (2 ** 31 - 1))
    n_block = int(np.floor(min(frac, 0.40) * n + 1e-9))
    n_block = min(n_block, n_drop)
    n_blocks = max(1, min(int(gap.get("n_blocks", 1)), 2))
    if n_block > 0:
        n_blocks = min(n_blocks, n_block)
        if n_blocks > 1:
            cuts = np.sort(rng.choice(n_block - 1, size=n_blocks - 1, replace=False) + 1)
            edges = np.concatenate([np.array([0]), cuts, np.array([n_block])])
        else:
            edges = np.array([0, n_block])
        lengths = [int(edges[i + 1] - edges[i]) for i in range(len(edges) - 1)]
        free = n - n_block
        bins = len(lengths) + 1
        extra = np.zeros(bins, np.int64)
        if free > 0:
            picks = rng.integers(0, bins, size=int(free))
            for p in picks:
                extra[int(p)] += 1
        cursor = 0
        for i, length in enumerate(lengths):
            cursor += int(extra[i])
            keep[cursor:cursor + length] = False
            cursor += length
    n_scatter = n_drop - int((~keep).sum())
    if n_scatter > 0:
        cand = np.flatnonzero(keep)
        if cand.size:
            take = rng.choice(cand, size=min(n_scatter, int(cand.size)), replace=False)
            keep[take] = False
    min_keep = int(np.ceil(0.60 * n - 1e-9))
    if int(keep.sum()) < min_keep:
        dropped = np.flatnonzero(~keep)
        need = min_keep - int(keep.sum())
        if dropped.size and need > 0:
            restore = rng.choice(dropped, size=min(int(dropped.size), need), replace=False)
            keep[restore] = True
    return keep


def apply_row_gaps(valid: np.ndarray, gap: dict | None) -> np.ndarray:
    """Снять наблюдение со строк. Температуру не обнуляет: это не включение события."""
    out = np.array(valid, copy=True)
    if out.ndim != 2 or not gap:
        return out
    keep = row_gap_mask(out.shape[0], gap)
    out[~keep, :] = False
    return out


def add_smooth_trend(R: np.ndarray, rng, dx: float = DX, n_waves: int = 2) -> np.ndarray:
    """Старый гладкий тренд. Обучение ветви STATIC5 его не вызывает: базовая линия отдельно."""
    width = R.shape[1]
    x = np.arange(width, dtype=np.float64) * dx
    y = np.zeros(width, np.float32)
    for _ in range(n_waves):
        lam = float(rng.uniform(50.0, 250.0))
        y += np.float32(rng.uniform(0.4, 5.0)) * np.sin(
            2 * np.pi * x / lam + float(rng.uniform(0, 2 * np.pi))
        ).astype(np.float32)
    return np.array(R, np.float32) + y[None, :]


def seq_seed(base: int, seq_id: str) -> int:
    import zlib
    return int((int(base) + zlib.adler32(seq_id.encode("utf-8"))) % (2 ** 31 - 1))


# Генератор STATIC2. Константы HOT_* / SPOT_* выше остаются генератором STATIC5:
# их читают этап 2 и уже записанные тесты. Здесь ширина 2–6 м, амплитуда
# +2…+25 °C лог-равномерно, 70 % гаусс / 30 % raised cosine, дрейф ±15 %,
# без пятна прибора, без краёв, без флуктуаций и без базовой линии STATIC5.
S2_WIDTH = (2.0, 6.0)
S2_AMP = (2.0, 25.0)


def static2_view(spec: dict) -> dict:
    """Копия спецификации для вклейки STATIC2: пятно прибора выключено.

    Опубликованные synth_static_*.json не содержат spot_m. Текущий
    unit_profile без этого ключа сворачивает профиль с пятном 1 м.
    Нулевое пятно этот шаг пропускает. Флуктуации STATIC5 тоже снимаются:
    у STATIC2 их не было, остаётся только линейный дрейф.
    """
    out = dict(spec)
    out["spot_m"] = 0.0
    out["edge_m"] = 0.0
    out["fluct"] = []
    return out


def static2_hot_spec(rng, n_px: int, dx: float = DX, avoid: np.ndarray | None = None):
    """Горячая точка STATIC2. Не вызывает hot_spec генератора STATIC5."""
    kind = "gaussian" if float(rng.random()) < 0.7 else "raised_cosine"
    width = float(rng.uniform(*S2_WIDTH))
    amp = _log_uniform(rng, *S2_AMP)
    drift = float(rng.uniform(-0.15, 0.15))
    return _spec(
        rng, n_px, dx, avoid, role="hot", kind=kind, width=width, amp=amp, drift=drift,
        spot_m=0.0, edge_m=0.0, fluct=[], mask="fwhm",
    )


def static2_train_specs(rng, n_px: int, dx: float = DX, avoid: np.ndarray | None = None) -> list[dict]:
    """Горячие точки STATIC2 и прежние отвлечения: плато 60 %, спайк 45 %.

    Соседей, базовую линию и пропуски STATIC5 эта функция не добавляет.
    """
    specs = []
    used = _copy_avoid(avoid, n_px)
    n_hot = 1 if float(rng.random()) < 0.82 else int(rng.integers(0, 3))
    for _ in range(n_hot):
        sp, used = static2_hot_spec(rng, n_px, dx, used)
        if sp:
            specs.append(sp)
    if float(rng.random()) < 0.6:
        sp, used = _spec(
            rng, n_px, dx, used, role="neg", kind="broad",
            width=float(rng.uniform(18.0, 90.0)),
            amp=_log_uniform(rng, 3.0, 18.0),
            drift=float(rng.uniform(-0.1, 0.1)),
            spot_m=0.0, edge_m=0.0, fluct=[], mask="fwhm",
        )
        if sp:
            specs.append(sp)
    if float(rng.random()) < 0.45:
        sp, used = _spec(
            rng, n_px, dx, used, role="neg", kind="gaussian",
            width=float(rng.uniform(0.35, 1.3)),
            amp=_log_uniform(rng, 4.0, 18.0),
            drift=float(rng.uniform(-0.05, 0.05)),
            spot_m=0.0, edge_m=0.0, fluct=[], mask="fwhm",
        )
        if sp:
            specs.append(sp)
    return specs


def static2_fixed_hots(n_px: int, seed: int, n_events: int = 2, dx: float = DX,
                       avoid: np.ndarray | None = None) -> list[dict]:
    """Два горячих события STATIC2. Seed задаёт параметры. Соседей нет."""
    rng = np.random.default_rng(int(seed))
    specs = []
    used = _copy_avoid(avoid, n_px)
    for _ in range(int(n_events)):
        sp, used = static2_hot_spec(rng, n_px, dx, used)
        if sp:
            specs.append(sp)
    return specs
