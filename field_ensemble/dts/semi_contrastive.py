"""Полуобучение классов событий DTS в стиле TS-TCC / CA-TCC.

Окно события — то же, что у классификатора этапа 2 (``events.box_indices`` /
``crop_and_features``): медиана зоны поперёк волокна (временной ряд) и медиана
по времени (поперечный профиль). Самообучение контрастирует две аугментации
этого окна. Аугментации только маскирование отсчётов и частично перекрывающаяся
обрезка с возвратом на исходную сетку. Амплитуду не масштабируем и отсчёты не
переставляем: величина и знак перегрева остаются в ряде.

Реальные трассы MGTS в этот модуль не подаются: ни как метки, ни как
неразмеченные окна. ``ENABLED`` по умолчанию выключен и не подменяет EventNet.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from . import events as E

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
    TORCH_OK = True
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    F = None
    TORCH_OK = False


# В pipeline этапа 2 модуль не включается, пока при бюджете 5% macro-F1
# не выше и EventNet, и LGBM на том же GroupKFold. Решение пишет бенч.
ENABLED = False

LENGTH = 64
N_CH = 3
AMP_DIM = 4
# Каналы кэша уже обрезаны в [-CLIP, CLIP] (dts.features.CLIP). Делим на эту
# константу, а не на статистики выборки и не на случайный коэффициент.
AMP_SCALE = 12.0
PSEUDO_THRESHOLD = 0.9
CLASSES = E.CLASSES

_FORBIDDEN = ("MGTS_data", "real_zones", "/mgts/")


def assert_not_real(path) -> None:
    """Реальные измерения нельзя ни учить, ни использовать как неразмеченный SSL."""
    text = str(path).replace("\\", "/")
    if any(token in text for token in _FORBIDDEN):
        raise RuntimeError(f"real traces are forbidden for CATCC training: {path}")


def decision_enabled(catcc_f1: float, eventnet_f1: float, lgbm_f1: float) -> bool:
    """Включать модуль только при строгом выигрыше у обоих текущих классификаторов."""
    return bool(catcc_f1 > eventnet_f1 and catcc_f1 > lgbm_f1)


def _require_torch():
    if not TORCH_OK:
        raise ImportError("dts.semi_contrastive needs torch")


def set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    if TORCH_OK:
        torch.manual_seed(int(seed))


# --- аугментации: маска и перекрывающаяся обрезка, без масштаба амплитуды ---

def overlap_len(start_a: int, len_a: int, start_b: int, len_b: int) -> int:
    return max(0, min(start_a + len_a, start_b + len_b) - max(start_a, start_b))


def sample_overlapping_starts(length: int, len_a: int, len_b: int, min_overlap: float, rng: np.random.Generator):
    """Два окна одной серии. Перекрытие не меньше min_overlap от более короткого."""
    need = int(np.ceil(float(min_overlap) * min(len_a, len_b)))
    need = min(need, min(len_a, len_b))
    hi_a = length - len_a
    hi_b = length - len_b
    for _ in range(40):
        sa = int(rng.integers(0, hi_a + 1))
        sb = int(rng.integers(0, hi_b + 1))
        if overlap_len(sa, len_a, sb, len_b) >= need:
            return sa, sb
    return 0, 0


def paired_crop_batch(batch: int, length: int, frac_a: float, frac_b: float, min_overlap: float, rng: np.random.Generator):
    len_a = min(length, max(2, int(round(length * float(frac_a)))))
    len_b = min(length, max(2, int(round(length * float(frac_b)))))
    starts_a = np.empty(batch, np.int64)
    starts_b = np.empty(batch, np.int64)
    for i in range(batch):
        starts_a[i], starts_b[i] = sample_overlapping_starts(length, len_a, len_b, min_overlap, rng)
    return (starts_a, len_a), (starts_b, len_b)


def mask_time(x, ratio: float, rng: np.random.Generator):
    """Обнулить долю отсчётов. Оставшиеся значения не меняются (нет масштаба)."""
    _require_torch()
    ratio = float(ratio)
    if ratio <= 0:
        return x
    keep = rng.random((x.shape[0], x.shape[-1])) >= ratio
    gate = torch.from_numpy(np.ascontiguousarray(keep, dtype=np.float32)).to(x.device)
    return x * gate[:, None, :]


def crop_resize(x, starts, crop_len: int):
    """Вырезать отрезок и линейно вернуть на исходную длину. Веса интерполяции суммируются в 1."""
    _require_torch()
    length = int(x.shape[-1])
    crop_len = int(crop_len)
    if crop_len >= length:
        return x
    # align_corners=True: крайние точки совпадают с краями отрезка, смесь выпуклая.
    grid = torch.linspace(0, crop_len - 1, length)
    pos = torch.as_tensor(starts, dtype=torch.float32).view(-1, 1) + grid.view(1, -1)
    pos = pos.clamp(0, length - 1)
    i0 = pos.long()
    i1 = (i0 + 1).clamp(max=length - 1)
    w = (pos - i0.float()).to(dtype=x.dtype)
    i0 = i0[:, None, :].expand(-1, x.shape[1], -1)
    i1 = i1[:, None, :].expand(-1, x.shape[1], -1)
    x0 = torch.gather(x, 2, i0)
    x1 = torch.gather(x, 2, i1)
    return x0 * (1.0 - w[:, None, :]) + x1 * w[:, None, :]


def make_views(temporal, spatial, rng: np.random.Generator, *, weak_mask=0.10, strong_mask=0.35,
               weak_frac=0.85, strong_frac=0.65, min_overlap=0.5):
    """Слабый и сильный вид. Обрезка времени и профиля независимы, каждая пара перекрывается."""
    batch, _, length = temporal.shape
    (sw, cw), (ss, cs) = paired_crop_batch(batch, length, weak_frac, strong_frac, min_overlap, rng)
    tw = mask_time(crop_resize(temporal, sw, cw), weak_mask, rng)
    ts = mask_time(crop_resize(temporal, ss, cs), strong_mask, rng)
    (sw, cw), (ss, cs) = paired_crop_batch(batch, length, weak_frac, strong_frac, min_overlap, rng)
    sw_ = mask_time(crop_resize(spatial, sw, cw), weak_mask, rng)
    ss_ = mask_time(crop_resize(spatial, ss, cs), strong_mask, rng)
    return tw, sw_, ts, ss_


# --- окно: средний ряд зоны + поперечный профиль ---

def resample_series(series: np.ndarray, length: int) -> np.ndarray:
    """series (C, n) -> (C, length). Линейная интерполяция, без коэффициента усиления."""
    series = np.asarray(series, np.float32)
    channels, n = series.shape
    length = int(length)
    if n == length:
        return series.copy()
    if n <= 1:
        return np.repeat(series, length, axis=1).astype(np.float32)
    src = np.linspace(0.0, 1.0, n)
    dst = np.linspace(0.0, 1.0, length)
    out = np.empty((channels, length), np.float32)
    for c in range(channels):
        out[c] = np.interp(dst, src, series[c])
    return out


def _masked_median(values: np.ndarray, mask: np.ndarray, axis: int) -> np.ndarray:
    filled = np.where(mask, values, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(filled, axis=axis)
    return np.nan_to_num(med, nan=0.0).astype(np.float32)


def series_pair(X, x, t, xa, xb, ta, tb, length: int = LENGTH):
    """Медиана поперёк зоны (время) и медиана по времени (расстояние), длина ``length``.

    Бокс совпадает с ``events.box_indices``: то же окно, что у EventNet, с полями.
    Каналы: dT_time, dT_space, доля валидных узлов свёрнутой оси. Единицы — σ, как в кэше.
    """
    st, sx = E.box_indices(x, t, float(xa), float(xb), float(ta), float(tb))
    c0 = np.asarray(X[0, st, sx], np.float32)
    c1 = np.asarray(X[1, st, sx], np.float32)
    valid = np.asarray(X[2, st, sx], np.float32) > 0.5
    if not valid.any():
        valid = np.ones(c0.shape, dtype=bool)
    temporal = np.stack([
        _masked_median(c0, valid, 1),
        _masked_median(c1, valid, 1),
        valid.mean(1).astype(np.float32),
    ], 0)
    spatial = np.stack([
        _masked_median(c0, valid, 0),
        _masked_median(c1, valid, 0),
        valid.mean(0).astype(np.float32),
    ], 0)
    return resample_series(temporal, length), resample_series(spatial, length)


def amplitude_from_series(temporal: np.ndarray) -> np.ndarray:
    """Знак и величина перегрева. Деление на фиксированный CLIP, не на случайный масштаб."""
    trace = np.asarray(temporal[0], np.float32)
    raw = np.array([np.median(trace), np.max(trace), np.min(trace), np.max(np.abs(trace))], np.float32)
    return raw / np.float32(AMP_SCALE)


def build_event_window(X, sigma, x, t, xa, xb, ta, tb, length: int = LENGTH):
    temporal, spatial = series_pair(X, x, t, xa, xb, ta, tb, length)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        img, scal = E.crop_and_features(X, sigma, x, t, float(xa), float(xb), float(ta), float(tb))
    amp = amplitude_from_series(temporal)
    return temporal, spatial, img.astype(np.float32), np.asarray(scal, np.float32), amp


def budget_indices(y: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Стратифицированная доля меток. В каждом присутствующем классе хотя бы один пример."""
    y = np.asarray(y)
    rng = np.random.default_rng(int(seed))
    chosen = []
    n_cls = int(y.max()) + 1 if len(y) else 0
    for c in range(n_cls):
        pool = np.flatnonzero(y == c)
        if len(pool) == 0:
            continue
        n = int(round(len(pool) * float(frac)))
        n = min(len(pool), max(1, n))
        chosen.append(rng.choice(pool, size=n, replace=False))
    if not chosen:
        return np.zeros(0, np.int64)
    return np.sort(np.concatenate(chosen).astype(np.int64))


def episode_folds(groups, n_splits: int = 5, mask=None):
    """GroupKFold по эпизодам. Индексы возвращаются в исходную длину ``groups``."""
    from sklearn.model_selection import GroupKFold
    groups = np.asarray(groups)
    if mask is None:
        mask = np.ones(len(groups), dtype=bool)
    dev = np.flatnonzero(np.asarray(mask, bool))
    n_groups = len(np.unique(groups[dev]))
    if n_groups < n_splits:
        raise ValueError(f"need at least {n_splits} episodes, got {n_groups}")
    folds = []
    splitter = GroupKFold(n_splits)
    for tr, va in splitter.split(np.zeros((len(dev), 1)), groups=groups[dev]):
        tr_i = dev[tr]
        va_i = dev[va]
        if set(groups[tr_i]) & set(groups[va_i]):
            raise RuntimeError("GroupKFold leaked an episode into both sides")
        folds.append((tr_i, va_i))
    return folds


def filter_pseudo(prob: np.ndarray, threshold: float = PSEUDO_THRESHOLD):
    """Псевдометки с порогом уверенности. Равенство порогу проходит (conf ≥ threshold)."""
    prob = np.asarray(prob, np.float64)
    if len(prob) == 0:
        empty_i = np.zeros(0, np.int64)
        empty_b = np.zeros(0, bool)
        return empty_i, empty_b, np.zeros(0, np.float64)
    conf = prob.max(axis=1)
    pred = prob.argmax(axis=1).astype(np.int64)
    keep = conf >= float(threshold)
    return pred, keep, conf


def score_probs(y, prob, class_names=None, reject_below: float = PSEUDO_THRESHOLD) -> dict:
    from sklearn.metrics import confusion_matrix, f1_score
    y = np.asarray(y, np.int64)
    prob = np.asarray(prob, np.float64)
    names = list(class_names or CLASSES)
    k = len(names)
    pred = prob.argmax(1) if len(prob) else np.zeros(0, np.int64)
    labels = list(range(k))
    per = f1_score(y, pred, average=None, labels=labels, zero_division=0) if len(y) else np.zeros(k)
    conf = prob.max(1) if len(prob) else np.zeros(0)
    reject = conf < float(reject_below) if len(conf) else np.zeros(0, bool)
    out = {
        "macro_f1": float(f1_score(y, pred, average="macro", labels=labels, zero_division=0)) if len(y) else 0.0,
        "per_class": {names[i]: float(per[i]) for i in labels},
        "accuracy": float((pred == y).mean()) if len(y) else 0.0,
        "reject_rate": float(reject.mean()) if len(y) else 0.0,
        "n": int(len(y)),
        "n_accepted": int((~reject).sum()) if len(y) else 0,
        "confusion": confusion_matrix(y, pred, labels=labels).tolist() if len(y) else [[0] * k for _ in range(k)],
    }
    if len(y) and (~reject).any():
        out["macro_f1_accepted"] = float(f1_score(y[~reject], pred[~reject], average="macro", labels=labels, zero_division=0))
    else:
        out["macro_f1_accepted"] = None
    return out


def lgbm_feature_matrix(img, scal, ctx) -> np.ndarray:
    """59 колонок текущего LGBM: скаляры + банк вырезки + форма + контекст соседей."""
    return E.lgbm_matrix(np.asarray(scal, np.float32), np.asarray(img, np.float32), np.asarray(ctx, np.float32), "full")


def collect_synth_windows(root, versions=("v8", "v10"), splits=("train", "val", "test"),
                          length: int = LENGTH, limit_files: int | None = None) -> dict:
    """Окна синтетики. Путь с MGTS / real_zones отвергается до чтения."""
    import pandas as pd
    from . import metrics as M

    root = Path(root)
    temporal, spatial, imgs, scals, amps = [], [], [], [], []
    ys, groups, split_name, version, xa, xb, ta, tb = [], [], [], [], [], [], [], []
    n_files = 0
    for ver in versions:
        if str(ver).startswith("real"):
            raise RuntimeError(f"real version is forbidden: {ver}")
        cache = root / "artifacts" / "cache" / ver
        labels_path = root / "data" / "synth_meta" / ver / "labels.csv"
        assert_not_real(cache)
        assert_not_real(labels_path)
        if not labels_path.exists():
            raise FileNotFoundError(labels_path)
        labels = pd.read_csv(labels_path)
        for split in splits:
            folder = cache / split
            if not folder.is_dir():
                continue
            files = sorted(folder.glob("seq*.npz"))
            if limit_files is not None:
                files = files[:int(limit_files)]
            for path in files:
                assert_not_real(path)
                z = np.load(path)
                X = z["X"]
                x = z["x"]
                t = z["t"]
                sigma = z["sigma"] if "sigma" in z.files else np.ones(len(x), np.float32)
                boxes = M.label_boxes(labels, path.stem)
                for r in boxes.itertuples(index=False):
                    if r.label not in E.SYNTH_TO_CLASS:
                        continue
                    x0 = max(float(r.dist_start_m), float(x[0]))
                    x1 = min(float(r.dist_end_m), float(x[-1]))
                    if x1 < x0:
                        continue
                    ser_t, ser_x, img, scal, amp = build_event_window(
                        X, sigma, x, t, x0, x1, float(r.ts), float(r.te), length)
                    temporal.append(ser_t)
                    spatial.append(ser_x)
                    imgs.append(img)
                    scals.append(scal)
                    amps.append(amp)
                    ys.append(E.CLASSES.index(E.SYNTH_TO_CLASS[r.label]))
                    groups.append(path.stem)
                    split_name.append(split)
                    version.append(ver)
                    xa.append(x0)
                    xb.append(x1)
                    ta.append(float(r.ts))
                    tb.append(float(r.te))
                n_files += 1
                del X, z
                if n_files % 40 == 0:
                    print(f"collect {n_files} files, windows {len(ys)}", flush=True)
    if not ys:
        raise RuntimeError("no synthetic windows collected")
    n = len(ys)
    groups_a = np.asarray(groups)
    version_a = np.asarray(version)
    ctx = np.zeros((n, len(E.CONTEXT_NAMES)), np.float32)
    buckets: dict[tuple, list] = {}
    xa_a = np.asarray(xa, np.float64)
    xb_a = np.asarray(xb, np.float64)
    ta_a = np.asarray(ta, np.float64)
    tb_a = np.asarray(tb, np.float64)
    for i, key in enumerate(zip(version_a.tolist(), groups_a.tolist())):
        buckets.setdefault(key, []).append(i)
    for ix in buckets.values():
        ix_a = np.asarray(ix, np.int64)
        ctx[ix_a] = E.context_features(xa_a[ix_a], xb_a[ix_a], ta_a[ix_a], tb_a[ix_a])
    ds = {
        "temporal": np.stack(temporal).astype(np.float32),
        "spatial": np.stack(spatial).astype(np.float32),
        "img": np.stack(imgs).astype(np.float32),
        "scal": np.stack(scals).astype(np.float32),
        "amp": np.stack(amps).astype(np.float32),
        "ctx": ctx,
        "y": np.asarray(ys, np.int64),
        "groups": groups_a.astype(str),
        "split": np.asarray(split_name).astype(str),
        "version": version_a.astype(str),
        "format": np.asarray("catcc-windows-v1"),
    }
    print(f"collected {n} windows from {n_files} files", {c: int((ds['y'] == i).sum()) for i, c in enumerate(CLASSES)}, flush=True)
    return ds


def save_windows(path, ds) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(ds)
    payload["img"] = np.asarray(ds["img"], np.float16)
    payload["temporal"] = np.asarray(ds["temporal"], np.float16)
    payload["spatial"] = np.asarray(ds["spatial"], np.float16)
    np.savez(path, **payload)


def load_windows(path) -> dict:
    z = np.load(path, allow_pickle=False)
    if "format" not in z.files or str(z["format"]) != "catcc-windows-v1":
        raise RuntimeError(f"unexpected window cache: {path}")
    ds = {k: z[k] for k in z.files}
    ds["img"] = np.asarray(ds["img"], np.float32)
    ds["temporal"] = np.asarray(ds["temporal"], np.float32)
    ds["spatial"] = np.asarray(ds["spatial"], np.float32)
    ds["groups"] = np.asarray(ds["groups"]).astype(str)
    ds["split"] = np.asarray(ds["split"]).astype(str)
    ds["version"] = np.asarray(ds["version"]).astype(str)
    return ds


# --- модель ---

def _ntxent(z1, z2, temperature: float):
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = z1 @ z2.transpose(0, 1) / float(temperature)
    labels = torch.arange(z1.shape[0], device=z1.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.transpose(0, 1), labels))


if TORCH_OK:
    class _Tower(nn.Module):
        def __init__(self, cin: int, width: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(cin, 32, 5, padding=2), nn.ReLU(inplace=True), nn.MaxPool1d(2),
                nn.Conv1d(32, 48, 5, padding=2), nn.ReLU(inplace=True), nn.MaxPool1d(2),
                nn.Conv1d(48, width, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool1d(2),
            )

        def forward(self, x):
            return self.net(x)

    class CATCC(nn.Module):
        """Энкодер временного ряда и поперечного профиля, кросс-внимание, контраст TS-TCC.

        Временной контраст: контекст одного вида предсказывает будущие шаги другого.
        Контекстный контраст: NT-Xent по вектору (временной контекст + профиль).
        BatchNorm нет: он стирает величину перегрева. Вход делим на фиксированный CLIP.
        """

        def __init__(self, width: int = 64, pred_steps: int = 2, temperature: float = 0.2, n_ch: int = N_CH):
            super().__init__()
            if int(width) % 4 != 0:
                raise ValueError("width must be divisible by 4")
            self.width = int(width)
            self.pred_steps = int(pred_steps)
            self.temperature = float(temperature)
            self.lambda_tc = 1.0
            self.lambda_cc = 0.7
            self.embed_dim = 2 * int(width)
            self.enc_t = _Tower(n_ch, width)
            self.enc_x = _Tower(n_ch, width)
            self.wq = nn.Linear(width, width)
            self.wk = nn.Linear(width, width)
            self.wv = nn.Linear(width, width)
            self.pos = nn.Parameter(torch.zeros(1, 16, width))
            nn.init.normal_(self.pos, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=width, nhead=4, dim_feedforward=width * 2, dropout=0.1,
                batch_first=True, norm_first=True, activation="gelu")
            self.transformer = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
            self.pred = nn.ModuleList([nn.Linear(width, width) for _ in range(self.pred_steps)])
            self.proj = nn.Sequential(nn.Linear(self.embed_dim, width), nn.ReLU(inplace=True), nn.Linear(width, width))
            scale = torch.tensor([1.0 / AMP_SCALE, 1.0 / AMP_SCALE, 1.0], dtype=torch.float32).view(1, n_ch, 1)
            self.register_buffer("in_scale", scale, persistent=False)

        def maps(self, temporal, spatial):
            ft = self.enc_t(temporal * self.in_scale)
            fx = self.enc_x(spatial * self.in_scale)
            q = self.wq(ft.transpose(1, 2))
            k = self.wk(fx.transpose(1, 2))
            v = self.wv(fx.transpose(1, 2))
            attn = torch.softmax(q @ k.transpose(1, 2) / (q.shape[-1] ** 0.5), dim=-1)
            ft = ft + (attn @ v).transpose(1, 2)
            return ft, fx

        def _pool(self, ft, fx):
            tok = ft.transpose(1, 2) + self.pos[:, :ft.shape[-1]]
            h = self.transformer(tok).mean(dim=1)
            return torch.cat([h, fx.mean(dim=-1)], dim=-1)

        def embed(self, temporal, spatial):
            ft, fx = self.maps(temporal, spatial)
            return self._pool(ft, fx)

        def _temporal_contrast(self, ft_from, ft_to):
            steps = ft_from.shape[-1]
            k = self.pred_steps
            if steps < k + 3:
                return ft_from.new_zeros(())
            hi = steps - k
            t_ctx = int(torch.randint(3, hi + 1, (1,)).item())
            tok = ft_from.transpose(1, 2)[:, :t_ctx] + self.pos[:, :t_ctx]
            context = self.transformer(tok)[:, -1]
            future = ft_to[:, :, t_ctx:t_ctx + k].detach()
            loss = ft_from.new_zeros(())
            for i in range(k):
                loss = loss + _ntxent(self.pred[i](context), future[:, :, i], self.temperature)
            return loss / k

        def ssl_loss(self, tw, sw, ts, ss):
            ft_w, fx_w = self.maps(tw, sw)
            ft_s, fx_s = self.maps(ts, ss)
            loss_tc = 0.5 * (self._temporal_contrast(ft_w, ft_s) + self._temporal_contrast(ft_s, ft_w))
            loss_cc = _ntxent(self.proj(self._pool(ft_w, fx_w)), self.proj(self._pool(ft_s, fx_s)), self.temperature)
            return self.lambda_tc * loss_tc + self.lambda_cc * loss_cc, loss_tc, loss_cc

    class Head(nn.Module):
        def __init__(self, embed_dim: int, amp_dim: int, n_classes: int, hidden: int = 64, dropout: float = 0.3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(embed_dim + amp_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
                nn.Linear(hidden, n_classes))
            nn.init.normal_(self.net[-1].weight, std=0.01)
            nn.init.zeros_(self.net[-1].bias)

        def forward(self, embedding, amp):
            return self.net(torch.cat([F.normalize(embedding, dim=-1), amp], dim=-1))
else:  # pragma: no cover
    class CATCC:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("dts.semi_contrastive needs torch")

    class Head:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("dts.semi_contrastive needs torch")


def _as_tensor(array, idx=None):
    _require_torch()
    data = array if idx is None else array[idx]
    return torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32))


def pretrain_encoder(model, temporal, spatial, idx, *, epochs: int, batch_size: int, seed: int,
                     lr: float = 3e-4, weight_decay: float = 3e-4, betas=(0.9, 0.99)):
    """Контрастивное самообучение без меток. ``idx`` — окна обучающих эпизодов фолда.

    Шаг 3e-4, weight decay 3e-4, betas (0.9, 0.99), температура NT-Xent 0.2 на модели.
    """
    _require_torch()
    idx = np.asarray(idx, np.int64)
    if len(idx) < 4:
        raise ValueError("SSL needs at least 4 windows")
    set_seed(seed)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay), betas=tuple(betas))
    rng = np.random.default_rng(int(seed))
    batch_size = max(4, min(int(batch_size), len(idx)))
    last = float("nan")
    loss_tc = loss_cc = None
    for ep in range(int(epochs)):
        order = rng.permutation(len(idx))
        total, seen = 0.0, 0
        for start in range(0, len(order) - batch_size + 1, batch_size):
            j = idx[order[start:start + batch_size]]
            tw, sw, ts, ss = make_views(_as_tensor(temporal, j), _as_tensor(spatial, j), rng)
            loss, loss_tc, loss_cc = model.ssl_loss(tw, sw, ts, ss)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite SSL loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach())
            seen += 1
        last = total / max(seen, 1)
        tc_v = float("nan") if loss_tc is None else float(loss_tc.detach())
        cc_v = float("nan") if loss_cc is None else float(loss_cc.detach())
        print(f"  ssl epoch {ep + 1}/{epochs} loss {last:.4f} tc {tc_v:.3f} cc {cc_v:.3f}", flush=True)
    return last


def _predict_proba(model, head, temporal, spatial, amp, idx, batch_size: int = 256) -> np.ndarray:
    _require_torch()
    idx = np.asarray(idx, np.int64)
    n_classes = int(head.net[-1].out_features)
    if len(idx) == 0:
        return np.zeros((0, n_classes), np.float32)
    was_model, was_head = model.training, head.training
    model.eval()
    head.eval()
    out = []
    with torch.inference_mode():
        for start in range(0, len(idx), batch_size):
            j = idx[start:start + batch_size]
            logits = head(model.embed(_as_tensor(temporal, j), _as_tensor(spatial, j)), _as_tensor(amp, j))
            out.append(torch.softmax(logits, 1).cpu().numpy())
    model.train(was_model)
    head.train(was_head)
    return np.concatenate(out).astype(np.float32)


def _class_weights(y: np.ndarray, n_classes: int):
    counts = np.bincount(np.asarray(y, np.int64), minlength=n_classes).astype(np.float32)
    weights = np.zeros(n_classes, np.float32)
    present = counts > 0
    weights[present] = counts[present].sum() / (present.sum() * counts[present])
    np.clip(weights, 0, 10, out=weights)
    return weights


def _ssl_param_ids(model):
    return {id(p) for p in list(model.proj.parameters()) + list(model.pred.parameters())}


def _supervised_epochs(model, head, temporal, spatial, amp, y_all, idx, y_idx, is_pseudo,
                       *, epochs: int, batch_size: int, lr: float, encoder_lr: float,
                       train_encoder: bool, seed: int, n_classes: int, pseudo_weight: float):
    _require_torch()
    if epochs <= 0 or len(idx) == 0:
        return
    # Проекционная голова и предикторы будущего нужны только SSL. Энкодер
    # включается и выключается явно: после заморозки requires_grad сам не вернётся.
    frozen = _ssl_param_ids(model)
    for p in model.parameters():
        p.requires_grad_(train_encoder and id(p) not in frozen)
    model.train(bool(train_encoder))
    head.train()
    groups = [{"params": list(head.parameters()), "lr": float(lr)}]
    if train_encoder:
        enc = [p for p in model.parameters() if p.requires_grad]
        if enc:
            groups.insert(0, {"params": enc, "lr": float(encoder_lr)})
    opt = torch.optim.AdamW(groups, weight_decay=1e-3)
    class_w = torch.from_numpy(_class_weights(y_idx, n_classes))
    rng = np.random.default_rng(int(seed))
    batch_size = max(1, min(int(batch_size), len(idx)))
    y_t = np.asarray(y_idx, np.int64)
    pseudo = np.asarray(is_pseudo, bool)
    for _ in range(int(epochs)):
        order = rng.permutation(len(idx))
        for start in range(0, len(order), batch_size):
            b = order[start:start + batch_size]
            j = idx[b]
            logits = head(model.embed(_as_tensor(temporal, j), _as_tensor(spatial, j)), _as_tensor(amp, j))
            target = torch.from_numpy(np.ascontiguousarray(y_t[b]))
            loss = F.cross_entropy(logits, target, reduction="none")
            w = class_w[target]
            if pseudo[b].any():
                w = w * torch.from_numpy(np.where(pseudo[b], np.float32(pseudo_weight), np.float32(1.0)))
            loss = (loss * w).sum() / w.sum().clamp(min=1e-6)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            if train_encoder:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()


def finetune_predict(model, head, temporal, spatial, amp, y, labeled_idx, unlabeled_idx, predict_idx, *,
                     epochs_sup: int = 12, epochs_pseudo: int = 8, batch_size: int = 32, seed: int = 0,
                     lr: float = 1e-3, encoder_lr: float = 1e-4, pseudo_threshold: float = PSEUDO_THRESHOLD,
                     pseudo_weight: float = 0.5, n_classes: int | None = None):
    """Дообучение на бюджете меток, затем псевдометки с conf ≥ threshold только на train-эпизодах."""
    _require_torch()
    n_classes = int(n_classes or len(CLASSES))
    labeled_idx = np.asarray(labeled_idx, np.int64)
    unlabeled_idx = np.asarray(unlabeled_idx, np.int64)
    y = np.asarray(y, np.int64)
    set_seed(seed)
    freeze_epochs = int(epochs_sup) // 2
    tune_epochs = int(epochs_sup) - freeze_epochs
    _supervised_epochs(
        model, head, temporal, spatial, amp, y, labeled_idx, y[labeled_idx], np.zeros(len(labeled_idx), bool),
        epochs=freeze_epochs, batch_size=batch_size, lr=lr, encoder_lr=encoder_lr, train_encoder=False,
        seed=seed + 1, n_classes=n_classes, pseudo_weight=1.0)
    _supervised_epochs(
        model, head, temporal, spatial, amp, y, labeled_idx, y[labeled_idx], np.zeros(len(labeled_idx), bool),
        epochs=tune_epochs, batch_size=batch_size, lr=lr, encoder_lr=encoder_lr, train_encoder=True,
        seed=seed + 2, n_classes=n_classes, pseudo_weight=1.0)
    stats = {"threshold": float(pseudo_threshold), "n_unlabeled": int(len(unlabeled_idx)), "n_pseudo": 0,
             "frac_pseudo": 0.0, "pseudo_accuracy_diagnostic": None}
    if len(unlabeled_idx):
        prob_u = _predict_proba(model, head, temporal, spatial, amp, unlabeled_idx, batch_size=256)
        pred, keep, conf = filter_pseudo(prob_u, pseudo_threshold)
        if keep.any():
            pseudo_idx = unlabeled_idx[keep]
            pseudo_y = pred[keep]
            stats["n_pseudo"] = int(len(pseudo_idx))
            stats["frac_pseudo"] = float(len(pseudo_idx) / max(len(unlabeled_idx), 1))
            stats["pseudo_accuracy_diagnostic"] = float((pseudo_y == y[pseudo_idx]).mean())
            both = np.concatenate([labeled_idx, pseudo_idx])
            both_y = np.concatenate([y[labeled_idx], pseudo_y])
            both_pseudo = np.concatenate([np.zeros(len(labeled_idx), bool), np.ones(len(pseudo_idx), bool)])
            _supervised_epochs(
                model, head, temporal, spatial, amp, y, both, both_y, both_pseudo,
                epochs=int(epochs_pseudo), batch_size=batch_size, lr=lr, encoder_lr=encoder_lr,
                train_encoder=False, seed=seed + 3, n_classes=n_classes, pseudo_weight=pseudo_weight)
    prob = _predict_proba(model, head, temporal, spatial, amp, predict_idx, batch_size=256)
    return prob, stats


def train_eventnet_classifier(img, scal, y, labeled_idx, scale_idx, predict_idx, *,
                              epochs: int = 40, seed: int = 0, n_classes: int | None = None, batch_size: int = 64):
    """Текущий EventNet, рецепт scripts/train_stage2.py, только на бюджете меток."""
    _require_torch()
    from . import models
    n_classes = int(n_classes or len(CLASSES))
    labeled_idx = np.asarray(labeled_idx, np.int64)
    scale_idx = np.asarray(scale_idx, np.int64)
    y = np.asarray(y, np.int64)
    set_seed(seed)
    mu = scal[scale_idx].mean(0).astype(np.float32)
    sd = (scal[scale_idx].std(0) + 1e-6).astype(np.float32)
    net = models.EventNet(n_classes, scal.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
    counts = np.bincount(y[labeled_idx], minlength=n_classes)
    cw = torch.tensor(len(labeled_idx) / (n_classes * np.maximum(counts, 1)), dtype=torch.float32).clamp(max=10)
    xi = torch.from_numpy(np.ascontiguousarray(img[labeled_idx], np.float32))
    si = torch.from_numpy(((scal[labeled_idx] - mu) / sd).astype(np.float32))
    yi = torch.from_numpy(y[labeled_idx].astype(np.int64))
    rng = np.random.default_rng(int(seed))
    net.train()
    for _ in range(int(epochs)):
        order = rng.permutation(len(labeled_idx))
        for start in range(0, len(labeled_idx), batch_size):
            b = order[start:start + batch_size]
            xb = xi[b]
            if rng.random() < 0.5:
                xb = torch.flip(xb, [3])
            loss = F.cross_entropy(net(xb, si[b]), yi[b], weight=cw, label_smoothing=0.05)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    net.eval()
    predict_idx = np.asarray(predict_idx, np.int64)
    if len(predict_idx) == 0:
        return np.zeros((0, n_classes), np.float32)
    s = torch.from_numpy(((scal[predict_idx] - mu) / sd).astype(np.float32))
    xt = torch.from_numpy(np.ascontiguousarray(img[predict_idx], np.float32))
    with torch.no_grad():
        lo = net(xt, s)
        lo = 0.5 * (lo + net(torch.flip(xt, [3]), s))
    return torch.softmax(lo, 1).numpy().astype(np.float32)


def train_lgbm_classifier(feat, y, labeled_idx, predict_idx, *, seed: int = 0, n_classes: int | None = None,
                          n_estimators: int = 200):
    """Текущий LGBM на полной матрице из 59 признаков, только бюджет меток."""
    import lightgbm as lgb
    n_classes = int(n_classes or len(CLASSES))
    labeled_idx = np.asarray(labeled_idx, np.int64)
    predict_idx = np.asarray(predict_idx, np.int64)
    y = np.asarray(y, np.int64)
    if len(np.unique(y[labeled_idx])) < 2:
        return np.full((len(predict_idx), n_classes), 1.0 / n_classes, np.float32)
    model = lgb.LGBMClassifier(
        n_estimators=int(n_estimators), learning_rate=0.05, num_leaves=15, min_child_samples=1,
        class_weight="balanced", random_state=int(seed), verbose=-1, n_jobs=4)
    model.fit(np.asarray(feat[labeled_idx], np.float32), y[labeled_idx])
    out = np.zeros((len(predict_idx), n_classes), np.float32)
    if len(predict_idx) == 0:
        return out
    raw = model.predict_proba(np.asarray(feat[predict_idx], np.float32))
    for col, cls in enumerate(model.classes_):
        out[:, int(cls)] = raw[:, col]
    return out


def _clone_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def run_group_benchmark(ds, *, budgets=(0.01, 0.05, 0.10), n_splits: int = 5, seed: int = 0,
                        ssl_epochs: int = 10, sup_epochs: int = 12, pseudo_epochs: int = 8,
                        eventnet_epochs: int = 40, batch_size: int = 64, pseudo_threshold: float = PSEUDO_THRESHOLD,
                        do_test: bool = True, width: int = 64, threads: int = 6, verbose: bool = True) -> dict:
    """GroupKFold по seq_id. v8 и v10 одного seq_id — один эпизод (парные генерации)."""
    import time
    _require_torch()
    torch.set_num_threads(int(threads))
    y = np.asarray(ds["y"], np.int64)
    groups = np.asarray(ds["groups"]).astype(str)
    split = np.asarray(ds["split"]).astype(str)
    version = np.asarray(ds["version"]).astype(str)
    dev = split != "test"
    test = np.flatnonzero(~dev)
    dev_idx = np.flatnonzero(dev)
    folds = episode_folds(groups, n_splits=n_splits, mask=dev)
    feat = lgbm_feature_matrix(ds["img"], ds["scal"], ds["ctx"])
    k = len(CLASSES)
    budget_probs = {
        f"{b:.2f}": {name: np.full((len(y), k), np.nan, np.float64) for name in ("catcc", "eventnet", "lgbm")}
        for b in budgets
    }
    pseudo_stats = {f"{b:.2f}": [] for b in budgets}
    fold_labeled = {f"{b:.2f}": [] for b in budgets}
    t0 = time.time()
    for fold_id, (tr, va) in enumerate(folds):
        if verbose:
            print(f"fold {fold_id + 1}/{len(folds)} train {len(tr)} val {len(va)} episodes {len(np.unique(groups[tr]))}", flush=True)
        set_seed(seed + 100 * fold_id)
        model = CATCC(width=width)
        pretrain_encoder(model, ds["temporal"], ds["spatial"], tr, epochs=ssl_epochs, batch_size=batch_size,
                         seed=seed + 100 * fold_id)
        base = _clone_state(model)
        for b_i, frac in enumerate(budgets):
            key = f"{float(frac):.2f}"
            rel = budget_indices(y[tr], float(frac), seed + 17 * fold_id + int(round(float(frac) * 1000)))
            labeled = tr[rel]
            unlabeled = np.setdiff1d(tr, labeled, assume_unique=False)
            fold_labeled[key].append({"fold": fold_id, "n_labeled": int(len(labeled)), "n_train": int(len(tr)),
                                      "n_per_class": np.bincount(y[labeled], minlength=k).tolist()})
            model.load_state_dict(base)
            head = Head(model.embed_dim, AMP_DIM, k)
            prob, stats = finetune_predict(
                model, head, ds["temporal"], ds["spatial"], ds["amp"], y, labeled, unlabeled, va,
                epochs_sup=sup_epochs, epochs_pseudo=pseudo_epochs, batch_size=min(32, max(4, len(labeled))),
                seed=seed + fold_id + b_i, pseudo_threshold=pseudo_threshold)
            budget_probs[key]["catcc"][va] = prob
            pseudo_stats[key].append(stats)
            budget_probs[key]["eventnet"][va] = train_eventnet_classifier(
                ds["img"], ds["scal"], y, labeled, tr, va, epochs=eventnet_epochs, seed=seed + fold_id,
                n_classes=k)
            budget_probs[key]["lgbm"][va] = train_lgbm_classifier(
                feat, y, labeled, va, seed=seed + fold_id, n_classes=k)
            if verbose:
                sc = score_probs(y[va], prob)
                se = score_probs(y[va], budget_probs[key]["eventnet"][va])
                sl = score_probs(y[va], budget_probs[key]["lgbm"][va])
                print(f"  budget {key} fold {fold_id} catcc {sc['macro_f1']:.3f} eventnet {se['macro_f1']:.3f} "
                      f"lgbm {sl['macro_f1']:.3f} pseudo {stats['n_pseudo']}/{stats['n_unlabeled']}", flush=True)
        del model, base
    by_budget = {}
    for frac in budgets:
        key = f"{float(frac):.2f}"
        block = {"per_fold": {}, "pseudo": _mean_pseudo(pseudo_stats[key]), "labeled": fold_labeled[key]}
        for name in ("catcc", "eventnet", "lgbm"):
            prob = budget_probs[key][name][dev_idx]
            if not np.isfinite(prob).all():
                raise RuntimeError(f"non-finite OOF probabilities: {name} budget {key}")
            block[name] = _pack_scores(y[dev_idx], prob, version[dev_idx])
            block[name]["per_fold"] = [score_probs(y[va], budget_probs[key][name][va])["macro_f1"] for _, va in folds]
        block["enabled_vs_current"] = decision_enabled(block["catcc"]["macro_f1"], block["eventnet"]["macro_f1"], block["lgbm"]["macro_f1"])
        by_budget[key] = block
    test_block = {}
    if do_test and len(test):
        if verbose:
            print(f"test fit on {len(dev_idx)} dev windows -> {len(test)} test", flush=True)
        set_seed(seed + 999)
        model = CATCC(width=width)
        pretrain_encoder(model, ds["temporal"], ds["spatial"], dev_idx, epochs=ssl_epochs, batch_size=batch_size, seed=seed + 999)
        base = _clone_state(model)
        for b_i, frac in enumerate(budgets):
            key = f"{float(frac):.2f}"
            rel = budget_indices(y[dev_idx], float(frac), seed + 5000 + int(round(float(frac) * 1000)))
            labeled = dev_idx[rel]
            unlabeled = np.setdiff1d(dev_idx, labeled, assume_unique=False)
            model.load_state_dict(base)
            head = Head(model.embed_dim, AMP_DIM, k)
            prob, stats = finetune_predict(
                model, head, ds["temporal"], ds["spatial"], ds["amp"], y, labeled, unlabeled, test,
                epochs_sup=sup_epochs, epochs_pseudo=pseudo_epochs, batch_size=min(32, max(4, len(labeled))),
                seed=seed + 50 + b_i, pseudo_threshold=pseudo_threshold)
            en = train_eventnet_classifier(ds["img"], ds["scal"], y, labeled, dev_idx, test, epochs=eventnet_epochs, seed=seed, n_classes=k)
            lg = train_lgbm_classifier(feat, y, labeled, test, seed=seed, n_classes=k)
            test_block[key] = {
                "catcc": score_probs(y[test], prob),
                "eventnet": score_probs(y[test], en),
                "lgbm": score_probs(y[test], lg),
                "pseudo": stats,
                "n_labeled": int(len(labeled)),
            }
            test_block[key]["enabled_vs_current"] = decision_enabled(
                test_block[key]["catcc"]["macro_f1"], test_block[key]["eventnet"]["macro_f1"], test_block[key]["lgbm"]["macro_f1"])
    focus = by_budget.get("0.05") or next(iter(by_budget.values()))
    enabled = bool(focus["enabled_vs_current"])
    return {
        "name": "CATCC",
        "method": "CA-TCC/TS-TCC",
        "enabled": enabled,
        "real_data_used": False,
        "criterion": "OOF macro-F1 at 5% labels strictly above both EventNet and LGBM on the same GroupKFold indices",
        "protocol": {
            "group": "seq_id (v8 and v10 of one sequence stay in one fold)",
            "splits": "GroupKFold on split!=test; official test fit separately, never used to choose the recipe",
            "budgets": list(budgets),
            "budget_sampling": "stratified by class inside the training fold, at least one label per present class",
            "augmentations": ["time mask", "overlapping crop resized back"],
            "amplitude_scaling": False,
            "permutation": False,
            "pseudo_threshold": pseudo_threshold,
            "ssl_epochs": ssl_epochs,
            "sup_epochs": sup_epochs,
            "pseudo_epochs": pseudo_epochs,
            "eventnet_epochs": eventnet_epochs,
            "seed": seed,
            "n_splits": n_splits,
            "width": width,
            "length": LENGTH,
        },
        "classes": list(CLASSES),
        "n": {
            "windows": int(len(y)),
            "dev": int(dev.sum()),
            "test": int((~dev).sum()),
            "episodes_dev": int(len(np.unique(groups[dev]))),
            "episodes_test": int(len(np.unique(groups[~dev]))) if (~dev).any() else 0,
            "by_version": {ver: int((version == ver).sum()) for ver in sorted(set(version.tolist()))},
            "by_split": {sp: int((split == sp).sum()) for sp in sorted(set(split.tolist()))},
            "class_counts": {CLASSES[i]: int((y == i).sum()) for i in range(k)},
        },
        "budgets": by_budget,
        "test": test_block,
        "seconds": round(time.time() - t0, 1),
    }


def _mean_pseudo(rows: list) -> dict:
    if not rows:
        return {}
    acc = [r["pseudo_accuracy_diagnostic"] for r in rows if r["pseudo_accuracy_diagnostic"] is not None]
    return {
        "n_pseudo_mean": float(np.mean([r["n_pseudo"] for r in rows])),
        "frac_pseudo_mean": float(np.mean([r["frac_pseudo"] for r in rows])),
        "pseudo_accuracy_diagnostic_mean": None if not acc else float(np.mean(acc)),
        "folds": rows,
    }


def resolve_pseudo(prob, threshold: float = PSEUDO_THRESHOLD, min_frac: float = 0.2):
    """Псевдометки conf ≥ threshold только если такая доля неразмеченных ≥ min_frac.

    Иначе маска пустая: голову оставляем на истинных метках бюджета.
    """
    pred, keep, conf = filter_pseudo(prob, threshold)
    frac = float(keep.mean()) if len(keep) else 0.0
    used = bool(frac >= float(min_frac) and keep.any())
    if not used:
        keep = np.zeros(len(keep), dtype=bool)
    return pred, keep, conf, frac, used


def embedding_collapse_stats(h) -> dict:
    """Среднее std по координатам. Малый mean_std — коллапс представления."""
    h = np.asarray(h, np.float64)
    if h.ndim != 2 or h.shape[0] == 0:
        return {"n": 0, "dim": 0, "mean_std": 0.0, "min_std": 0.0, "max_std": 0.0, "feature_std": 0.0, "collapsed": True}
    per = h.std(axis=0)
    mean_std = float(per.mean())
    return {
        "n": int(h.shape[0]),
        "dim": int(h.shape[1]),
        "mean_std": mean_std,
        "min_std": float(per.min()),
        "max_std": float(per.max()),
        "feature_std": float(h.std()),
        "collapsed": bool(mean_std < 1e-3),
    }


def _aligned_proba(model, features, n_classes: int) -> np.ndarray:
    raw = np.asarray(model.predict_proba(features), np.float64)
    out = np.zeros((len(features), n_classes), np.float32)
    for col, cls in enumerate(model.classes_):
        out[:, int(cls)] = raw[:, col]
    total = out.sum(axis=1, keepdims=True)
    total[total <= 0] = 1.0
    return (out / total).astype(np.float32)


def _make_probe(kind: str, seed: int):
    if kind == "linear":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=600, class_weight="balanced", C=1.0, solver="lbfgs", random_state=int(seed))
    if kind == "embed_lgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            n_estimators=180, learning_rate=0.05, num_leaves=15, min_child_samples=1,
            class_weight="balanced", random_state=int(seed), verbose=-1, n_jobs=2)
    raise ValueError(kind)


def fit_frozen_head(kind, features, y, labeled, unlabeled, predict, *, seed: int = 0, n_classes: int | None = None,
                    threshold: float = PSEUDO_THRESHOLD, min_frac: float = 0.2, pseudo_weight: float = 0.3,
                    standardize: bool = False):
    """Голова на замороженных признаках. Псевдометки — только при доле conf ≥ threshold не меньше min_frac."""
    n_classes = int(n_classes or len(CLASSES))
    labeled = np.asarray(labeled, np.int64)
    unlabeled = np.asarray(unlabeled, np.int64)
    predict = np.asarray(predict, np.int64)
    y = np.asarray(y, np.int64)
    stats = {
        "kind": kind, "threshold": float(threshold), "min_frac": float(min_frac), "used_pseudo": False,
        "frac_confident": 0.0, "frac_pseudo": 0.0, "n_pseudo": 0, "n_unlabeled": int(len(unlabeled)),
        "pseudo_accuracy_diagnostic": None,
    }
    if len(predict) == 0:
        return np.zeros((0, n_classes), np.float32), stats
    if len(labeled) == 0 or len(np.unique(y[labeled])) < 2:
        return np.full((len(predict), n_classes), 1.0 / n_classes, np.float32), stats
    train_rows = np.concatenate([labeled, unlabeled]) if len(unlabeled) else labeled
    feat = np.asarray(features, np.float32)
    if standardize:
        mu = feat[train_rows].mean(0)
        sd = feat[train_rows].std(0) + 1e-6
        feat = ((feat - mu) / sd).astype(np.float32)
    model = _make_probe(kind, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(feat[labeled], y[labeled])
        if len(unlabeled):
            pred, keep, _conf, frac, used = resolve_pseudo(_aligned_proba(model, feat[unlabeled], n_classes), threshold, min_frac)
            stats["frac_confident"] = frac
            if used:
                pseudo_idx = unlabeled[keep]
                both = np.concatenate([labeled, pseudo_idx])
                both_y = np.concatenate([y[labeled], pred[keep]])
                sw = np.concatenate([np.ones(len(labeled), np.float32), np.full(len(pseudo_idx), np.float32(pseudo_weight))])
                model.fit(feat[both], both_y, sample_weight=sw)
                stats["used_pseudo"] = True
                stats["n_pseudo"] = int(len(pseudo_idx))
                stats["frac_pseudo"] = float(len(pseudo_idx) / max(len(unlabeled), 1))
                stats["pseudo_accuracy_diagnostic"] = float((pred[keep] == y[pseudo_idx]).mean())
        prob = _aligned_proba(model, feat[predict], n_classes)
    return prob, stats


def embed_all(model, temporal, spatial, batch_size: int = 256) -> np.ndarray:
    """Эмбеддинги замороженного энкодера, без аугментаций."""
    _require_torch()
    model.eval()
    parts = []
    with torch.inference_mode():
        for start in range(0, len(temporal), int(batch_size)):
            stop = start + int(batch_size)
            parts.append(model.embed(_as_tensor(temporal[start:stop]), _as_tensor(spatial[start:stop])).cpu().numpy())
    return np.concatenate(parts).astype(np.float32)


def run_probe_benchmark(ds, *, budgets=(0.01, 0.05, 0.10), n_splits: int = 5, seed: int = 0,
                        ssl_epochs: int = 8, batch_size: int = 64, width: int = 32, threads: int = 3,
                        ssl_lr: float = 3e-4, ssl_wd: float = 3e-4, pseudo_min_frac: float = 0.2,
                        pseudo_weight: float = 0.3, temperature: float = 0.2) -> dict:
    """SSL + линейный зонд и LGBM на замороженных эмбеддингах. Энкодер после SSL не дообучается."""
    import time
    _require_torch()
    torch.set_num_threads(int(threads))
    y = np.asarray(ds["y"], np.int64)
    groups = np.asarray(ds["groups"]).astype(str)
    split = np.asarray(ds["split"]).astype(str)
    version = np.asarray(ds["version"]).astype(str)
    amp = np.asarray(ds["amp"], np.float32)
    dev = split != "test"
    dev_idx = np.flatnonzero(dev)
    folds = episode_folds(groups, n_splits=n_splits, mask=dev)
    k = len(CLASSES)
    names = ("linear", "embed_lgbm")
    probs = {f"{b:.2f}": {name: np.full((len(y), k), np.nan, np.float64) for name in names} for b in budgets}
    pseudo_rows = {f"{b:.2f}": {name: [] for name in names} for b in budgets}
    collapse_rows = []
    t0 = time.time()
    for fold_id, (tr, va) in enumerate(folds):
        print(f"fold {fold_id + 1}/{len(folds)} train {len(tr)} val {len(va)}", flush=True)
        set_seed(seed + 100 * fold_id)
        model = CATCC(width=width, temperature=temperature)
        pretrain_encoder(
            model, ds["temporal"], ds["spatial"], tr, epochs=ssl_epochs, batch_size=batch_size,
            seed=seed + 100 * fold_id, lr=ssl_lr, weight_decay=ssl_wd, betas=(0.9, 0.99))
        hidden = embed_all(model, ds["temporal"], ds["spatial"])
        stats_h = embedding_collapse_stats(hidden[tr])
        stats_h["fold"] = fold_id
        collapse_rows.append(stats_h)
        print(f"  embed mean_std {stats_h['mean_std']:.4f} feature_std {stats_h['feature_std']:.4f} collapsed {stats_h['collapsed']}", flush=True)
        features = np.concatenate([hidden, amp], axis=1).astype(np.float32)
        del model, hidden
        for b_i, frac in enumerate(budgets):
            key = f"{float(frac):.2f}"
            rel = budget_indices(y[tr], float(frac), seed + 17 * fold_id + int(round(float(frac) * 1000)))
            labeled = tr[rel]
            unlabeled = np.setdiff1d(tr, labeled, assume_unique=False)
            for kind, stdize in (("linear", True), ("embed_lgbm", False)):
                prob, st = fit_frozen_head(
                    kind, features, y, labeled, unlabeled, va, seed=seed + fold_id + b_i, n_classes=k,
                    min_frac=pseudo_min_frac, pseudo_weight=pseudo_weight, standardize=stdize)
                probs[key][kind][va] = prob
                st["fold"] = fold_id
                pseudo_rows[key][kind].append(st)
            print(
                f"  budget {key} fold {fold_id} linear {score_probs(y[va], probs[key]['linear'][va])['macro_f1']:.3f} "
                f"emb_lgbm {score_probs(y[va], probs[key]['embed_lgbm'][va])['macro_f1']:.3f} "
                f"pseudo_lin {pseudo_rows[key]['linear'][-1]['used_pseudo']} "
                f"frac_lin {pseudo_rows[key]['linear'][-1]['frac_confident']:.3f}",
                flush=True)
    by_budget = {}
    for frac in budgets:
        key = f"{float(frac):.2f}"
        block = {}
        for name in names:
            prob = probs[key][name][dev_idx]
            if not np.isfinite(prob).all():
                raise RuntimeError(f"non-finite probe probabilities: {name} {key}")
            packed = _pack_scores(y[dev_idx], prob, version[dev_idx])
            packed["per_fold"] = [score_probs(y[va], probs[key][name][va])["macro_f1"] for _, va in folds]
            packed["pseudo"] = _mean_pseudo([
                {"n_pseudo": r["n_pseudo"], "frac_pseudo": r["frac_pseudo"],
                 "pseudo_accuracy_diagnostic": r["pseudo_accuracy_diagnostic"]}
                for r in pseudo_rows[key][name]
            ])
            packed["pseudo"]["used_folds"] = int(sum(r["used_pseudo"] for r in pseudo_rows[key][name]))
            packed["pseudo"]["frac_confident_mean"] = float(np.mean([r["frac_confident"] for r in pseudo_rows[key][name]]))
            block[name] = packed
        by_budget[key] = block
    mean_std = float(np.mean([r["mean_std"] for r in collapse_rows]))
    return {
        "name": "CATCC2",
        "method": "frozen SSL embedding, linear probe and LGBM",
        "real_data_used": False,
        "protocol": {
            "ssl_lr": ssl_lr, "ssl_wd": ssl_wd, "betas": [0.9, 0.99], "temperature": temperature,
            "lambda_tc": 1.0, "lambda_cc": 0.7, "batch_size": batch_size, "ssl_epochs": ssl_epochs,
            "width": width, "pseudo_threshold": PSEUDO_THRESHOLD, "pseudo_min_frac": pseudo_min_frac,
            "pseudo_weight": pseudo_weight, "head": "frozen", "features": "embedding+amplitude",
            "seed": seed, "n_splits": n_splits,
        },
        "collapse": {"mean_std": mean_std, "collapsed": bool(mean_std < 1e-3), "folds": collapse_rows},
        "budgets": by_budget,
        "seconds": round(time.time() - t0, 1),
    }


def _pack_scores(y, prob, version) -> dict:
    packed = score_probs(y, prob)
    packed["by_version"] = {}
    version = np.asarray(version)
    for ver in sorted(set(version.tolist())):
        m = version == ver
        packed["by_version"][str(ver)] = score_probs(y[m], prob[m])
    return packed
