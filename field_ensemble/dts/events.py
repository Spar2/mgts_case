"""Stage 2: вырезки событий, скалярные признаки, таксономия классов, зоны из карты вероятностей."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy import ndimage

# Таксономия по кейсу: длительное/обширное vs кратковременное/точечное, нагрев/охлаждение,
# «естественные» циклы помещения и артефакты прибора. Маппинг синтетических меток — по генератору gen_synth.py.
CLASSES = ["Local_Heating", "Extended_Heating", "Contact_Trace", "Local_Cooling", "Extended_Cooling", "HVAC_Diurnal", "Sensor_Artifact"]
SYNTH_TO_CLASS = {"hotspot": "Local_Heating", "section_heating": "Extended_Heating", "slow_warming": "Extended_Heating",
                  "tracing": "Contact_Trace", "coldspot": "Local_Cooling", "cooling": "Extended_Cooling",
                  "diurnal_section": "HVAC_Diurnal", "common_mode_jump": "Sensor_Artifact", "connector_step": "Sensor_Artifact"}
COARSE = {"Local_Heating": "heating", "Extended_Heating": "heating", "Contact_Trace": "heating", "Local_Cooling": "cooling",
          "Extended_Cooling": "cooling", "HVAC_Diurnal": "periodic_natural", "Sensor_Artifact": "artifact"}
RULES_TO_CLASS = {"Local_Heating": "Local_Heating", "Local_Cooling": "Local_Cooling", "Periodic_Thermal": "HVAC_Diurnal",
                  "Broad_Shift": "Sensor_Artifact", "Sensor_Fault": "Sensor_Artifact", "Background": "Sensor_Artifact"}
CROP = (32, 64)
SCALAR_NAMES = ["log_dur_h", "log_width_m", "width_frac", "amp_med", "amp_peak", "amp_min", "sign", "space_peak",
                "rise_frac", "fall_frac", "plateau_frac", "n_sign_changes", "acf_period", "acf_peak", "edge_ratio",
                "row_frac_cable", "valid_frac", "t_frac_start", "t_frac_end"]


def box_indices(x, t, xa, xb, ta, tb, margin=0.25, min_px=4):
    w = max(xb - xa, 0.5); d = max(tb - ta, 240.0)
    ix = np.where((x >= xa - margin * w - min_px * 0.25) & (x <= xb + margin * w + min_px * 0.25))[0]
    it = np.where((t >= ta - margin * d - 240) & (t <= tb + margin * d + 240))[0]
    if len(ix) == 0: ix = np.array([int(np.argmin(np.abs(x - 0.5 * (xa + xb))))])
    if len(it) == 0: it = np.array([int(np.argmin(np.abs(t - 0.5 * (ta + tb))))])
    return slice(it[0], it[-1] + 1), slice(ix[0], ix[-1] + 1)


def _resize(a, shape):
    import torch, torch.nn.functional as F
    ten = torch.as_tensor(np.ascontiguousarray(a, dtype=np.float32))[None]
    return F.adaptive_avg_pool2d(F.interpolate(ten, size=(max(shape[0], a.shape[1]), max(shape[1], a.shape[2])), mode="nearest"), shape)[0].numpy()


def crop_and_features(X, sigma, x, t, xa, xb, ta, tb):
    st, sx = box_indices(x, t, xa, xb, ta, tb)
    c = X[:, st, sx].astype(np.float32)
    img = _resize(c, CROP)
    # внутренний бокс (без полей)
    ix = (x[sx] >= xa - 0.25) & (x[sx] <= xb + 0.25); it = (t[st] >= ta - 120) & (t[st] <= tb + 120)
    if not ix.any(): ix[:] = True
    if not it.any(): it[:] = True
    sg = sigma[sx][ix]
    inner = c[0][np.ix_(it, ix)] * sg[None]; space = c[1][np.ix_(it, ix)] * sg[None]
    val = c[2][np.ix_(it, ix)]
    prof = np.nanmedian(np.where(val > 0, inner, np.nan), axis=1) if inner.size else np.zeros(1)
    prof = np.nan_to_num(prof)
    med = float(np.median(inner)) if inner.size else 0.0
    sign = 1.0 if med >= 0 else -1.0
    sp = prof * sign
    pk = float(sp.max()) if sp.size else 0.0
    half = sp >= 0.5 * pk if pk > 0 else np.zeros_like(sp, bool)
    n = len(sp)
    first = int(np.argmax(half)) if half.any() else 0; last = n - 1 - int(np.argmax(half[::-1])) if half.any() else n - 1
    d = np.diff(sp) if n > 1 else np.zeros(1)
    nsc = int(np.sum(np.diff(np.sign(d[np.abs(d) > 1e-6])) != 0)) if n > 2 else 0
    pr = sp - sp.mean()
    if n > 6 and pr.std() > 1e-6:
        ac = np.correlate(pr, pr, "full")[n - 1:] / (pr.var() * n)
        lag = int(np.argmax(ac[3:n // 2])) + 3 if n // 2 > 3 else 0
        acp = float(ac[lag]) if lag else 0.0
    else:
        lag, acp = 0, 0.0
    # контраст границ: внутри vs поля
    outer = np.abs(c[0]).mean() if c.size else 0.0
    edge = float(np.abs(c[0][np.ix_(it, ix)]).mean() / (outer + 1e-6)) if inner.size else 1.0
    T0, T1 = t[0], t[-1]
    f = [np.log1p((tb - ta) / 3600), np.log1p(xb - xa), (xb - xa) / max(x[-1] - x[0], 1), med, float(np.percentile(inner, 99)) if inner.size else 0,
         float(np.percentile(inner, 1)) if inner.size else 0, sign, float(np.percentile(np.abs(space), 95)) if space.size else 0,
         first / max(n, 1), (n - 1 - last) / max(n, 1), float(half.mean()) if n else 0, nsc / max(n, 1), lag / max(n, 1), acp, edge,
         (xb - xa) / max(x[-1] - x[0], 1), float(val.mean()) if val.size else 0, (ta - T0) / max(T1 - T0, 1), (tb - T0) / max(T1 - T0, 1)]
    return img.astype(np.float32), np.nan_to_num(np.array(f, np.float32))


def rules_zone(feat, xa, xb, ta, tb):
    """Зона в формате haklolarp для rules.classify_zones (oracle-бокс)."""
    sign = feat[SCALAR_NAMES.index("sign")]
    return {"polarity": "hot" if sign > 0 else "cold", "start_m": float(xa), "end_m": float(xb), "width_m": float(xb - xa),
            "t_start_s": float(ta), "t_end_s": float(tb), "peak_contrast_c": float(feat[SCALAR_NAMES.index("space_peak")]),
            "median_contrast_c": float(feat[SCALAR_NAMES.index("amp_med")]), "time_segment": 0}


# Контекст соседей по той же последовательности (геометрия боксов, без их классов)
# и форма уже собранной вырезки. К базовым SCALAR_NAMES не приклеиваются, чтобы
# crop_and_features и старые чекпоинты не меняли размерность.
SHAPE_NAMES = ["sp_half", "sp_peakmean", "tp_half", "tp_peakmean", "sp_center_edge", "sp_d2", "c1_over_c0", "sign_agree"]
CONTEXT_NAMES = ["log_n_other", "n_x_touch", "n_t_overlap", "n_chain", "log_nearest_dx_m", "log_nearest_gap_h", "n_periodic_x", "n_parallel"]
# legacy LGBM: SCALAR + image_bank; full: + shape + context
IMAGE_BANK_DIM = 24


def image_bank(img):
    """Сжатая статистика вырезки, как в исходном LGBM (24 числа)."""
    img = np.asarray(img, np.float32)
    return np.concatenate([
        img.mean((2, 3)), img.std((2, 3)), np.abs(img[:, :2]).max((2, 3)),
        img[:, 0].mean(2)[:, ::4], img[:, 0].mean(1)[:, ::8]], 1).astype(np.float32)


def shape_features(img):
    """Форма вырезки: пиковость по x/t, кривизна, доля знака. Различает локальный пик и плато."""
    img = np.asarray(img, np.float32)
    ch0, ch1 = img[:, 0], img[:, 1]
    sp, tp = ch0.mean(1), ch0.mean(2)

    def half_frac(v):
        a = np.abs(v)
        m = np.maximum(a.max(1, keepdims=True), 1e-6)
        return (a >= 0.5 * m).mean(1)

    sp_a = np.abs(sp)
    w = sp.shape[1]
    a, b = w // 3, 2 * w // 3
    center = sp_a[:, a:b].mean(1)
    edge = np.concatenate([sp_a[:, :a], sp_a[:, b:]], 1).mean(1)
    d2 = np.abs(np.diff(sp, n=2, axis=1)).mean(1) if sp.shape[1] > 2 else np.zeros(len(sp), np.float32)
    c0 = np.abs(ch0).mean((1, 2))
    c1 = np.abs(ch1).mean((1, 2))
    med = np.median(ch0, axis=(1, 2))
    agree = (np.sign(ch0) == np.sign(med)[:, None, None]).mean((1, 2))
    feats = np.stack([
        half_frac(sp), sp_a.max(1) / (sp_a.mean(1) + 1e-6),
        half_frac(tp), np.abs(tp).max(1) / (np.abs(tp).mean(1) + 1e-6),
        center / (edge + 1e-6), d2, c1 / (c0 + 1e-6), agree], 1)
    return np.nan_to_num(feats.astype(np.float32))


def context_features(xa, xb, ta, tb):
    """Признаки соседних боксов одной последовательности. xa/xb в метрах, ta/tb в секундах.

    n_chain — соседи в 5..150 м и не дальше 20 мин (шаг Contact_Trace).
    n_periodic_x — то же место кабеля, пауза 6..30 ч.
    n_parallel — другое место, почти то же время (секции HVAC / синфазный скачок).
    Одиночный бокс получает большие log-расстояния, а не нули.
    """
    xa = np.asarray(xa, np.float64).reshape(-1)
    xb = np.asarray(xb, np.float64).reshape(-1)
    ta = np.asarray(ta, np.float64).reshape(-1)
    tb = np.asarray(tb, np.float64).reshape(-1)
    n = len(xa)
    out = np.zeros((n, len(CONTEXT_NAMES)), np.float32)
    lonely_dx, lonely_h = 1e5, 1e3
    if n == 0:
        return out
    out[:, 4] = np.log1p(lonely_dx)
    out[:, 5] = np.log1p(lonely_h)
    if n == 1:
        return out
    cx = 0.5 * (xa + xb)
    dx = np.abs(cx[:, None] - cx[None, :])
    gap_t = np.maximum(ta[None, :] - tb[:, None], ta[:, None] - tb[None, :])
    gap_x = np.maximum(xa[None, :] - xb[:, None], xa[:, None] - xb[None, :])
    gap_t = np.maximum(gap_t, 0.0)
    gap_x = np.maximum(gap_x, 0.0)
    eye = np.eye(n, dtype=bool)
    dx[eye] = np.inf
    gap_t[eye] = np.inf
    gap_x[eye] = np.inf
    near_dx = np.min(dx, 1)
    near_dt = np.min(gap_t, 1)
    near_dx = np.where(np.isfinite(near_dx), near_dx, lonely_dx)
    near_dt = np.where(np.isfinite(near_dt), near_dt, lonely_h * 3600)
    out[:, 0] = np.log1p(n - 1)
    out[:, 1] = (gap_x <= 5.0).sum(1)
    out[:, 2] = (gap_t <= 1.0).sum(1)
    out[:, 3] = ((gap_t <= 1200.0) & (dx >= 5.0) & (dx <= 150.0)).sum(1)
    out[:, 4] = np.log1p(near_dx)
    out[:, 5] = np.log1p(near_dt / 3600.0)
    out[:, 6] = ((gap_x <= 2.0) & (gap_t >= 6 * 3600.0) & (gap_t <= 30 * 3600.0)).sum(1)
    out[:, 7] = ((gap_t <= 60.0) & (dx >= 30.0)).sum(1)
    return out


def context_from_zones(zones):
    if not zones:
        return np.zeros((0, len(CONTEXT_NAMES)), np.float32)
    return context_features(
        [z["start_m"] for z in zones], [z["end_m"] for z in zones],
        [z["t_start_s"] for z in zones], [z["t_end_s"] for z in zones])


def dataset_context(cache_dir, labels_path, expect_y=None, expect_groups=None):
    """Контекст в том же порядке, в каком train_stage2 собирает dataset_*.npz."""
    import pandas as pd
    from . import metrics as M
    labels = pd.read_csv(labels_path)
    rows, ys, groups = [], [], []
    for p in sorted(Path(cache_dir).glob("*/seq*.npz")):
        x = np.load(p)["x"]
        B = M.label_boxes(labels, p.stem)
        kept = []
        for r in B.itertuples():
            if r.label not in SYNTH_TO_CLASS:
                continue
            xa, xb = max(float(r.dist_start_m), float(x[0])), min(float(r.dist_end_m), float(x[-1]))
            if xb < xa:
                continue
            kept.append((xa, xb, float(r.ts), float(r.te), CLASSES.index(SYNTH_TO_CLASS[r.label])))
        if not kept:
            continue
        xa, xb, ta, tb, yy = (np.array(v) for v in zip(*kept))
        rows.append(context_features(xa, xb, ta, tb))
        ys.append(yy.astype(np.int64))
        groups.append(np.array([p.stem] * len(yy)))
    ctx = np.concatenate(rows).astype(np.float32)
    y = np.concatenate(ys)
    g = np.concatenate(groups).astype(str)
    if expect_y is not None and (len(y) != len(expect_y) or not np.array_equal(y, np.asarray(expect_y))):
        raise RuntimeError(f"context alignment: y {y.shape} vs dataset {np.shape(expect_y)}")
    if expect_groups is not None and not np.array_equal(g, np.asarray(expect_groups).astype(str)):
        raise RuntimeError("context alignment: seq_id order differs from dataset groups")
    return ctx


def named_scalars(base, img, zones, names):
    """Собрать матрицу скаляров в порядке names (чекпоинт хранит этот список)."""
    base = np.asarray(base, np.float32)
    cat = {n: base[:, i] for i, n in enumerate(SCALAR_NAMES)}
    if any(n in names for n in SHAPE_NAMES):
        sh = shape_features(img)
        cat.update({n: sh[:, i] for i, n in enumerate(SHAPE_NAMES)})
    if any(n in names for n in CONTEXT_NAMES):
        ctx = context_from_zones(zones)
        cat.update({n: ctx[:, i] for i, n in enumerate(CONTEXT_NAMES)})
    missing = [n for n in names if n not in cat]
    if missing:
        raise KeyError("unknown scalar features: " + ",".join(missing))
    return np.column_stack([cat[n] for n in names]).astype(np.float32)


def lgbm_matrix(base_scal, img, context=None, kind="legacy"):
    """Признаки LGBM. legacy совместим с уже обученным lgbm.joblib (43 колонки)."""
    base_scal = np.asarray(base_scal, np.float32)
    parts = [base_scal, image_bank(img)]
    if kind == "shape":
        parts.append(shape_features(img))
    elif kind == "context":
        if context is None:
            raise ValueError("context kind needs a context matrix")
        parts.append(np.asarray(context, np.float32))
    elif kind == "full":
        if context is None:
            raise ValueError("full kind needs a context matrix")
        parts.append(shape_features(img))
        parts.append(np.asarray(context, np.float32))
    elif kind != "legacy":
        raise ValueError(kind)
    return np.nan_to_num(np.concatenate(parts, 1).astype(np.float32))


def box_iou(a, b):
    """IoU осевых боксов (x0, x1, t0, t1). Масштаб осей сокращается."""
    x0, x1 = max(float(a[0]), float(b[0])), min(float(a[1]), float(b[1]))
    t0, t1 = max(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    inter = max(0.0, x1 - x0) * max(0.0, t1 - t0)
    aa = max(0.0, float(a[1]) - float(a[0])) * max(0.0, float(a[3]) - float(a[2]))
    bb = max(0.0, float(b[1]) - float(b[0])) * max(0.0, float(b[3]) - float(b[2]))
    return inter / max(aa + bb - inter, 1e-12)


def greedy_match(pred_boxes, gt_boxes, thr=0.1):
    """Жадное 1-1 сопоставление по IoU. Возвращает matches (i, j, iou), индексы без пары."""
    pred_boxes = list(pred_boxes)
    gt_boxes = list(gt_boxes)
    p, g = len(pred_boxes), len(gt_boxes)
    if p == 0 or g == 0:
        return [], list(range(p)), list(range(g))
    pairs = []
    for i, pb in enumerate(pred_boxes):
        for j, gb in enumerate(gt_boxes):
            v = box_iou(pb, gb)
            if v >= thr:
                pairs.append((v, i, j))
    pairs.sort(key=lambda z: (-z[0], z[1], z[2]))
    used_p, used_g, matches = set(), set(), []
    for v, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        matches.append((i, j, float(v)))
    return matches, [i for i in range(p) if i not in used_p], [j for j in range(g) if j not in used_g]


def e2e_counts(pred_cls, gt_cls, matches, n_classes):
    """TP/FP/FN детекции+класса. Совпавшая пара с разным классом — FP предсказанного и FN истинного.
    Зона без пары — FP, метка без пары — FN. Возвращает также число верных пар."""
    tp = np.zeros(n_classes, np.float64)
    fp = np.zeros(n_classes, np.float64)
    fn = np.zeros(n_classes, np.float64)
    matched_p, matched_g, correct = set(), set(), 0
    pred_cls = np.asarray(pred_cls, np.int64)
    gt_cls = np.asarray(gt_cls, np.int64)
    for i, j, _ in matches:
        matched_p.add(i)
        matched_g.add(j)
        pc, gc = int(pred_cls[i]), int(gt_cls[j])
        if pc == gc:
            tp[pc] += 1
            correct += 1
        else:
            fp[pc] += 1
            fn[gc] += 1
    for i, c in enumerate(pred_cls):
        if i not in matched_p:
            fp[int(c)] += 1
    for j, c in enumerate(gt_cls):
        if j not in matched_g:
            fn[int(c)] += 1
    return tp, fp, fn, correct


def f1_from_counts(tp, fp, fn, names):
    per, f1s = {}, []
    for k, name in enumerate(names):
        p = float(tp[k] / (tp[k] + fp[k])) if tp[k] + fp[k] else 0.0
        r = float(tp[k] / (tp[k] + fn[k])) if tp[k] + fn[k] else 0.0
        f = float(2 * p * r / (p + r)) if p + r else 0.0
        per[name] = dict(precision=p, recall=r, f1=f, tp=int(tp[k]), fp=int(fp[k]), fn=int(fn[k]))
        f1s.append(f)
    return float(np.mean(f1s)) if f1s else 0.0, per


def expected_calibration_error(prob, yt, bins=10):
    """ECE по уверенности max-класса. Та же корзина, что в train_stage2 (сравнение с 0.069/0.036)."""
    prob = np.asarray(prob, np.float64)
    yt = np.asarray(yt)
    conf, pred, e = prob.max(1), prob.argmax(1), 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any():
            e += m.mean() * abs(float((pred[m] == yt[m]).mean()) - float(conf[m].mean()))
    return float(e)


def fit_temperature(logits, y):
    """Один скаляр T: softmax(logits / T). Аргмакс не меняется."""
    import torch
    import torch.nn.functional as F
    t = torch.nn.Parameter(torch.ones(()))
    opt = torch.optim.LBFGS([t], lr=0.25, max_iter=40)
    lo = torch.as_tensor(np.asarray(logits, np.float32))
    yt = torch.as_tensor(np.asarray(y, np.int64))

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lo / t.clamp(min=1e-2), yt)
        loss.backward()
        return loss

    opt.step(closure)
    return float(t.detach().clamp(min=1e-2))


def temper_proba(prob, temperature):
    if temperature is None or abs(float(temperature) - 1.0) < 1e-6:
        return np.asarray(prob, np.float32)
    logp = np.log(np.clip(np.asarray(prob, np.float64), 1e-8, 1.0))
    z = logp / float(temperature)
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(1, keepdims=True)).astype(np.float32)


def zones_from_prob(score, hotness, valid, x, t, thr, min_px=12, close=(3, 3)):
    """Порог + морфологическое замыкание + связные компоненты -> список зон."""
    m = (score >= thr) & valid
    if close: m = ndimage.binary_closing(m, structure=np.ones(close)) & valid
    lab, n = ndimage.label(m)
    zones = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        comp = lab[sl] == i
        npx = int(comp.sum())
        if npx < min_px: continue
        h = float(np.mean(hotness[sl][comp])); s = float(np.max(score[sl][comp]))
        zones.append(dict(id=len(zones), t0=int(sl[0].start), t1=int(sl[0].stop), x0=int(sl[1].start), x1=int(sl[1].stop),
                          start_m=float(x[sl[1].start]), end_m=float(x[sl[1].stop - 1]), t_start_s=float(t[sl[0].start]),
                          t_end_s=float(t[sl[0].stop - 1]), n_px=npx, polarity="hot" if h > 0 else "cold", score=s))
    return zones
