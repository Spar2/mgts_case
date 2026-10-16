"""Веса ансамбля карт нейросетей. Порог — квантиль фона, не пространственная z-статистика.

Отбор на val: среди кандидатов с IoU не ниже максимума более чем на IOU_BAND
берётся больший recall событий cooling (доля пикселей события ≥ HIT_FRAC),
затем IoU. Сумма iou + cooling_recall пишется рядом и сама по себе победителя
не назначает: скачок 1/6 события больше типичного разрыва IoU.
"""
from __future__ import annotations

import numpy as np

FAR = 1e-3
HIT_FRAC = 0.3
IOU_BAND = 0.005


def far_threshold(bg_scores, far: float = FAR) -> float:
    """Тот же порядковый статистик, что dts.metrics.far_threshold."""
    s = np.sort(np.asarray(bg_scores, np.float64).ravel())
    if s.size == 0:
        return 1.0
    k = int(np.ceil((1.0 - float(far)) * s.size)) - 1
    k = min(max(k, 0), s.size - 1)
    return float(s[k]) + 1e-12


def threshold_from_hist(counts, edges, far: float = FAR) -> float:
    """Порог по гистограмме. Внутри корзины — линейная доля, как у порядковой статистики."""
    counts = np.asarray(counts, np.int64)
    edges = np.asarray(edges, np.float64)
    total = int(counts.sum())
    if total <= 0:
        return 1.0
    k = int(np.ceil((1.0 - float(far)) * total)) - 1
    k = min(max(k, 0), total - 1)
    seen = 0
    for i, n in enumerate(counts):
        n = int(n)
        if n <= 0:
            continue
        if seen + n > k:
            frac = (k - seen) / n
            return float(edges[i] + frac * (edges[i + 1] - edges[i])) + 1e-12
        seen += n
    return float(edges[-2]) + 1e-12


def combined_score(iou: float, cooling_recall: float) -> float:
    return float(iou) + float(cooling_recall)


def prf_from_counts(tp, fp, fn) -> dict:
    tp = float(tp)
    fp = float(fp)
    fn = float(fn)
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    iou = tp / max(tp + fp + fn, 1.0)
    return dict(precision=precision, recall=recall, f1=f1, iou=iou)


# Планка вложенной проверки: опубликованный val IoU победителя плюс 0.003.
# Число 0.3867 — то, с которым сравнивали в журнале (четыре знака 0.386712…).
NESTED_BASE = 0.3867
NESTED_MARGIN = 0.003
NESTED_BAR = NESTED_BASE + NESTED_MARGIN


def sequence_folds(n: int, k: int) -> list[list[int]]:
    """Фолды по порядку уже отсортированных последовательностей: i % k.

    Списки попарно не пересекаются и покрывают range(n).
    """
    n = int(n)
    k = int(k)
    if k < 2 or n < k:
        raise ValueError("для фолдов нужно k >= 2 и не меньше k последовательностей")
    folds = [[] for _ in range(k)]
    for i in range(n):
        folds[i % k].append(i)
    return folds


def seeded_folds(n: int, k: int, seed: int) -> list[list[int]]:
    """То же разбиение i % k, но после перестановки последовательностей с данным seed."""
    n = int(n)
    k = int(k)
    if k < 2 or n < k:
        raise ValueError("для фолдов нужно k >= 2 и не меньше k последовательностей")
    order = np.random.default_rng(int(seed)).permutation(n)
    folds = [[] for _ in range(k)]
    for pos, idx in enumerate(order.tolist()):
        folds[pos % k].append(int(idx))
    return folds


def delta_summary(deltas) -> dict:
    """Среднее и доля строго положительных дельт. Ноль не считается победой."""
    d = np.asarray(list(deltas), np.float64).ravel()
    if d.size == 0:
        raise ValueError("нет дельт")
    n_pos = int((d > 0).sum())
    return dict(
        n=int(d.size),
        mean=float(d.mean()),
        n_positive=n_pos,
        frac_positive=float(n_pos / d.size),
    )


def micro_iou(parts) -> float:
    """Один IoU по сумме tp/fp/fn. Каждый кусок уже посчитан со своим порогом и весами."""
    tp = 0.0
    fp = 0.0
    fn = 0.0
    for p in parts:
        tp += float(p["tp"])
        fp += float(p["fp"])
        fn += float(p["fn"])
    return float(tp / max(tp + fp + fn, 1.0))


def nested_clears_bar(iou: float, bar: float = NESTED_BAR) -> bool:
    """Строго больше планки. Равенство планке — отказ."""
    return float(iou) > float(bar)


def nested_verdict(iou_by_k: dict, bar: float = NESTED_BAR) -> dict:
    """Принять метод, только если каждая вложенная оценка (2 и 5) строго выше планки.

    Оценка метода — минимум по k. Он и сравнивается с планкой.
    """
    if not iou_by_k:
        raise ValueError("нет вложенных оценок")
    scores = {str(k): float(v) for k, v in iou_by_k.items()}
    worst = min(scores.values())
    return dict(
        bar=float(bar),
        iou_by_k=scores,
        worst=float(worst),
        accept=bool(nested_clears_bar(worst, bar)),
    )


def clip_normalize(weights) -> np.ndarray:
    w = np.clip(np.asarray(weights, np.float64), 0.0, None)
    if not np.isfinite(w).all() or w.sum() <= 0:
        w = np.ones(w.shape if w.ndim else (1,), np.float64)
    return w / w.sum()


def weights_from_array(names, values) -> dict:
    w = clip_normalize(values)
    if len(w) != len(names):
        raise ValueError("длина весов не равна числу членов")
    return {str(n): float(v) for n, v in zip(names, w)}


def as_vector(names, weights: dict) -> np.ndarray:
    return clip_normalize([float(weights.get(n, 0.0)) for n in names])


def softmax_weights(names, scores, temperature: float) -> dict:
    s = np.asarray(scores, np.float64)
    z = s / max(float(temperature), 1e-6)
    z = z - np.max(z)
    return weights_from_array(names, np.exp(z))


def recipe_weights(names, active) -> dict:
    active = set(active)
    return weights_from_array(names, [1.0 if n in active else 0.0 for n in names])


def dirichlet_bank(n_members: int, n: int, seed: int, concentration: float = 0.7) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return rng.dirichlet(np.full(int(n_members), float(concentration)), size=int(n))


def mix_stack(stack, weights, mode: str = "mean") -> np.ndarray:
    """stack: (M, N) → оценка длины N. mean — выпуклая смесь; logit — смесь логитов; geomean — взвешенное геом. среднее."""
    if stack.size == 0:
        return np.zeros(0, np.float32)
    w = clip_normalize(weights).astype(np.float32)
    p = np.asarray(stack, np.float32)
    if p.ndim != 2 or p.shape[0] != w.shape[0]:
        raise ValueError("stack (M, N) не совпал с весами")
    if mode == "mean":
        return np.tensordot(w, p, axes=(0, 0))
    if mode == "logit":
        pc = np.clip(p, 1e-4, 1.0 - 1e-4)
        logit = np.log(pc) - np.log1p(-pc)
        z = np.tensordot(w, logit, axes=(0, 0))
        return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)
    if mode == "geomean":
        pc = np.clip(p, 1e-4, 1.0)
        return np.exp(np.tensordot(w, np.log(pc), axes=(0, 0))).astype(np.float32)
    raise ValueError(mode)


def background_threshold(frames, weights, far: float = FAR, mode: str = "mean") -> float:
    """Квантиль FAR только по чистому фону этих кадров. События в порог не входят."""
    w = np.asarray(weights, np.float64)
    parts = []
    for fr in frames:
        bg = mix_stack(fr["bg_m"], w, mode)
        if bg.size:
            parts.append(np.asarray(bg, np.float64))
    if not parts:
        return 1.0
    return far_threshold(np.concatenate(parts), far)


def evaluate_frames(frames, weights, far: float = FAR, mode: str = "mean", hit_frac: float = HIT_FRAC, thr=None) -> dict:
    """Компактные кадры: ev_m/nu_m/bg_m (M, N), n_bg, cooling — индексы в ось события.

    Фон может быть подвыборкой: ложные на чистом фоне масштабируются n_bg / n_sample.
    Помеха (nu) хранится целиком. Порог — только по чистому фону этих кадров,
    если thr не передан. Переданный thr не пересчитывается.
    """
    w = np.asarray(weights, np.float64)
    bg_parts = []
    mixes = []
    for fr in frames:
        ev = mix_stack(fr["ev_m"], w, mode)
        nu = mix_stack(fr["nu_m"], w, mode)
        bg = mix_stack(fr["bg_m"], w, mode)
        mixes.append((ev, nu, bg))
        if bg.size:
            bg_parts.append(bg)
    if thr is None:
        thr = far_threshold(np.concatenate(bg_parts), far) if bg_parts else 1.0
    else:
        thr = float(thr)
    tp = 0
    fn = 0
    fp_nu = 0
    fp_bg = 0.0
    cool_hit = 0
    cool_n = 0
    for fr, (ev, nu, bg) in zip(frames, mixes):
        tp += int((ev >= thr).sum())
        fn += int((ev < thr).sum())
        if nu.size:
            fp_nu += int((nu >= thr).sum())
        n_s = int(bg.size)
        if n_s:
            fp_bg += float((bg >= thr).sum()) * (float(fr["n_bg"]) / float(n_s))
        for idx in fr["cooling"]:
            cool_n += 1
            if len(idx) == 0:
                continue
            frac = float((ev[np.asarray(idx)] >= thr).mean())
            cool_hit += int(frac >= hit_frac)
    fp = fp_nu + fp_bg
    stats = prf_from_counts(tp, fp, fn)
    crec = (cool_hit / cool_n) if cool_n else 0.0
    far_emp = 0.0
    if bg_parts:
        cat = np.concatenate(bg_parts)
        far_emp = float((cat >= thr).mean())
    return dict(
        threshold=float(thr),
        far=far_emp,
        tp=int(tp),
        fp=float(fp),
        fn=int(fn),
        cooling_hits=int(cool_hit),
        cooling_n=int(cool_n),
        cooling_recall=float(crec),
        combined=combined_score(stats["iou"], crec),
        mode=mode,
        **stats,
    )


def pick_band(rows, tol: float = IOU_BAND):
    """Внутри полосы IoU — выше cooling recall, затем IoU, затем сумма, затем имя."""
    if not rows:
        raise ValueError("нет кандидатов")
    best_iou = max(float(r["iou"]) for r in rows)
    band = [r for r in rows if float(r["iou"]) >= best_iou - float(tol)]
    band.sort(key=lambda r: (
        -float(r["cooling_recall"]),
        -float(r["iou"]),
        -float(r.get("combined", combined_score(r["iou"], r["cooling_recall"]))),
        str(r.get("name", "")),
    ))
    return band[0], best_iou, band


def coordinate_ascent(eval_fn, names, start: dict, steps=(0.0, 0.05, 0.15, 0.4), rounds: int = 2):
    """Подъём по симплексу. eval_fn(dict) → число, больше лучше. Веса < 0 отсекаются."""
    best_w = weights_from_array(names, [start.get(n, 0.0) for n in names])
    best = float(eval_fn(best_w))
    for _ in range(int(rounds)):
        improved = False
        for i, name in enumerate(names):
            for a in steps:
                arr = np.array([best_w[n] for n in names], np.float64)
                arr[i] = float(a)
                trial = weights_from_array(names, arr)
                sc = float(eval_fn(trial))
                if sc > best + 1e-6:
                    best = sc
                    best_w = trial
                    improved = True
        if not improved:
            break
    return best_w, best
