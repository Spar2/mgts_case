"""Многомасштабный инференс без обучения.

Поле T усредняется неперекрывающимися блоками по времени или по кабелю,
каналы и σ пересчитываются на прореженной сетке, score возвращается
на исходную сетку повторением блока. Слияние с исходным масштабом:
max, среднее и среднее с весом N (дисперсия среднего N отсчётов ~ 1/N,
поэтому σ меньше примерно в √N).
"""
from __future__ import annotations
import numpy as np

# (имя, ось 0=время / 1=кабель, фактор, вес = число усреднённых отсчётов)
SCALE_SPEC = (
    ("t2", 0, 2, 2.0),
    ("t4", 0, 4, 4.0),
    ("x2", 1, 2, 2.0),
    ("x4", 1, 4, 4.0),
)
FUSIONS = ("native", "max", "mean", "noise")
# Диагностика: max(исходный, один грубый масштаб). В выбор победителя не входит.
DIAG_SCALES = ("t2", "t4", "x2", "x4")
NB = 100001
EV_BINS = 2001


def block_mean(a, valid, axis, factor):
    """Среднее по неперекрывающимся блокам. Хвост короче фактора сохраняется."""
    a = np.asarray(a, np.float64)
    valid = np.asarray(valid, bool)
    factor = int(factor)
    if factor < 1:
        raise ValueError("factor")
    n = a.shape[axis]
    full = (n // factor) * factor
    chunks, oks = [], []
    if full:
        sl = [slice(None)] * a.ndim
        sl[axis] = slice(0, full)
        mask = valid[tuple(sl)]
        block = np.where(mask, a[tuple(sl)], 0.0)
        shape = list(block.shape)
        shape[axis] = full // factor
        shape.insert(axis + 1, factor)
        cnt = mask.reshape(shape).sum(axis=axis + 1)
        sm = block.reshape(shape).sum(axis=axis + 1)
        mean = np.divide(sm, cnt, out=np.zeros(cnt.shape, np.float64), where=cnt > 0)
        chunks.append(mean)
        oks.append(cnt > 0)
    if full < n:
        sl = [slice(None)] * a.ndim
        sl[axis] = slice(full, n)
        mask = valid[tuple(sl)]
        block = np.where(mask, a[tuple(sl)], 0.0)
        cnt = mask.sum(axis=axis)
        sm = block.sum(axis=axis)
        mean = np.divide(sm, cnt, out=np.zeros(cnt.shape, np.float64), where=cnt > 0)
        chunks.append(np.expand_dims(mean, axis))
        oks.append(np.expand_dims(cnt > 0, axis))
    if not chunks:
        return np.zeros_like(a, np.float32), np.zeros(a.shape, bool)
    return np.concatenate(chunks, axis).astype(np.float32), np.concatenate(oks, axis)


def upsample_axis(coarse, n, axis, factor):
    """Повтор значения блока на исходную длину. Хвост обрезается."""
    reps = np.repeat(np.asarray(coarse), int(factor), axis=axis)
    sl = [slice(None)] * reps.ndim
    sl[axis] = slice(0, int(n))
    return reps[tuple(sl)]


def _mask(s, h, valid):
    v = np.asarray(valid, bool)
    return np.where(v, s, 0).astype(np.float32), np.where(v, h, 0).astype(np.float32)


def fuse(pairs, weights, how):
    """pairs: список (score, hotness) одной формы. how: max | mean | noise."""
    S = np.stack([np.asarray(p[0], np.float32) for p in pairs], 0)
    H = np.stack([np.asarray(p[1], np.float32) for p in pairs], 0)
    if how == "max":
        idx = np.argmax(S, 0)
        s = np.max(S, 0)
        h = np.take_along_axis(H, idx[None], 0)[0]
    elif how == "mean":
        s = S.mean(0)
        h = H.mean(0)
    elif how == "noise":
        w = np.asarray(weights, np.float32).reshape(-1, *([1] * (S.ndim - 1)))
        z = float(w.sum())
        s = (S * w).sum(0) / z
        h = (H * w).sum(0) / z
    else:
        raise ValueError(how)
    return s.astype(np.float32), h.astype(np.float32)


def ensemble_score(members, T, valid, dx):
    """Среднее score/hotness участников на поле T. Каналы считаются заново."""
    from dts import classic, features, models
    valid = np.asarray(valid, bool)
    if not np.any(valid):
        z = np.zeros(np.shape(T), np.float32)
        return z, z
    X, _aux = features.make_channels(np.asarray(T, np.float32), valid, dx=float(dx))
    ss, hh = [], []
    for kind, mdl in members:
        if kind == "net":
            prob = models.predict_full(mdl, X)
        else:
            pr = mdl.predict_proba(classic.pixel_features(X))
            prob = np.moveaxis(pr.reshape(X.shape[1], X.shape[2], -1), -1, 0)
        ss.append(1.0 - prob[0])
        hh.append(prob[1] - prob[2])
    s = np.mean(np.stack(ss, 0), 0)
    h = np.mean(np.stack(hh, 0), 0)
    return _mask(s, h, valid)


def multiscale_maps(members, T, valid, dx, native=None):
    """Исходный масштаб и четыре прореженных. Ключи: native, max, mean, noise, max_<scale>."""
    T = np.asarray(T, np.float32)
    valid = np.asarray(valid, bool)
    if native is None:
        native_pair = ensemble_score(members, T, valid, dx)
    else:
        native_pair = _mask(native[0], native[1], valid)
    scales = {"native": native_pair}
    weight = {"native": 1.0}
    for name, axis, factor, w in SCALE_SPEC:
        Tc, vc = block_mean(T, valid, axis, factor)
        dx_c = float(dx) * (factor if axis == 1 else 1.0)
        sc, hc = ensemble_score(members, Tc, vc, dx_c)
        n = T.shape[axis]
        scales[name] = _mask(upsample_axis(sc, n, axis, factor), upsample_axis(hc, n, axis, factor), valid)
        weight[name] = float(w)
    order = ["native", "t2", "t4", "x2", "x4"]
    packed = [scales[k] for k in order]
    w = [weight[k] for k in order]
    out = {
        "native": scales["native"],
        "max": _mask(*fuse(packed, w, "max"), valid),
        "mean": _mask(*fuse(packed, w, "mean"), valid),
        "noise": _mask(*fuse(packed, w, "noise"), valid),
    }
    for name in DIAG_SCALES:
        out["max_" + name] = _mask(*fuse([scales["native"], scales[name]], [1.0, weight[name]], "max"), valid)
    return out


def score_index(scores, nb):
    x = np.clip(np.asarray(scores, np.float64).ravel(), 0.0, 1.0)
    if x.size == 0:
        return np.zeros(0, np.int32)
    return np.minimum((x * (nb - 1)).astype(np.int32), nb - 1)


def add_hist(hist, scores):
    idx = score_index(scores, len(hist))
    if idx.size == 0:
        return hist
    hist += np.bincount(idx, minlength=len(hist))
    return hist


def hist_threshold(hist, far):
    """Тот же порядковый порог, что metrics.far_threshold, по левому краю бина."""
    total = int(np.asarray(hist).sum())
    if total <= 0:
        return 1.0
    k = int(np.ceil((1.0 - float(far)) * total)) - 1
    k = int(np.clip(k, 0, total - 1))
    c = 0
    nb = len(hist)
    for i, h in enumerate(hist):
        c += int(h)
        if c > k:
            return float(i) / (nb - 1) + 1e-12
    return 1.0


def hist_ge(hist, thr):
    """Число отсчётов, попавших в бины с левым краем >= thr."""
    nb = len(hist)
    i0 = int(np.ceil(float(thr) * (nb - 1) - 1e-9))
    i0 = min(max(i0, 0), nb)
    return int(np.asarray(hist)[i0:].sum())


def prf_from_hists(pos, neg, thr):
    from dts import metrics as M
    tp = hist_ge(pos, thr)
    fp = hist_ge(neg, thr)
    fn = int(np.asarray(pos).sum()) - tp
    return dict(tp=tp, fp=fp, fn=fn, **M.prf(tp, fp, fn))
