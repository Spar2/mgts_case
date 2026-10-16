"""Test-time augmentation ансамбля: отражение x, отражение t, симметрия знака.

Сети едят каналы (dT_time, dT_space, valid), не сырое T. Отражение оси
переставляет вход и тем же образом возвращает вероятности на место.
Симметрия знака меняет знак каналов 0 и 1 и меняет местами классы
нагрев/охлаждение. Фон остаётся фоном: score = 1 − p(фон) не меняет
определения, hotness = p(нагрев) − p(охлаждение) меняет знак.

Порог FAR на этих картах не считается. Функции детерминированы.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import numpy as np

from dts import metrics as M

# Имена видов. id — опубликованная карта, без нового прохода сети.
VIEW_SPEC = {
    "flip_x": {"flip_t": False, "flip_x": True, "sign": False},
    "flip_t": {"flip_t": True, "flip_x": False, "sign": False},
    "sign": {"flip_t": False, "flip_x": False, "sign": True},
}

# Линейные смеси видов. Веса на видах; id всегда есть.
# Смесь на уровне ансамбля совпадает со смесью внутри каждого члена,
# если в каждый вид добавлен один и тот же вклад LightGBM (см. тест).
LINEAR = {
    "mean_x": {"id": 1.0, "flip_x": 1.0},
    "mean_t": {"id": 1.0, "flip_t": 1.0},
    "mean_xt": {"id": 1.0, "flip_x": 1.0, "flip_t": 1.0},
    "wsign_0.25": {"id": 0.75, "sign": 0.25},
    "wsign_0.50": {"id": 0.5, "sign": 0.5},
    "wsign_0.75": {"id": 0.25, "sign": 0.75},
    "mean_x_sign": {"id": 1.0, "flip_x": 1.0, "sign": 1.0},
    "mean_full": {"id": 1.0, "flip_x": 1.0, "flip_t": 1.0, "sign": 1.0},
    # Только знаковый вид: члены без этого прохода остаются при своём id.
    "sign_member": {"sign": 1.0},
}
NONLINEAR = ("logit_full", "maxpol", "shrink_full", "geomean_sign")
PRIMARY = "mean_full"
ALL_VARIANTS = tuple(LINEAR) + NONLINEAR


def views_required(name: str) -> set[str]:
    """Какие проходы сверх id нужны, чтобы собрать вариант."""
    if name in LINEAR:
        return {k for k in LINEAR[name] if k != "id"}
    if name in ("logit_full", "shrink_full"):
        return {"flip_x", "flip_t", "sign"}
    if name in ("maxpol", "geomean_sign"):
        return {"sign"}
    raise KeyError(name)


def transform_input(X, flip_t=False, flip_x=False, sign=False):
    """X: (3, T, W) float. Знак только у остатков, valid не трогается."""
    Y = np.array(X, dtype=np.float32, copy=True)
    if sign:
        Y[0] *= np.float32(-1)
        Y[1] *= np.float32(-1)
    if flip_t:
        Y = np.flip(Y, 1)
    if flip_x:
        Y = np.flip(Y, 2)
    return np.ascontiguousarray(Y, dtype=np.float32)


def align_prob(prob, flip_t=False, flip_x=False, sign=False):
    """Вернуть вероятности (3, T, W) в координаты и полярность исходного поля.

    Оси снимаются в обратном порядке к transform_input. Знак классов
    коммутирует с отражениями осей.
    """
    P = prob
    if flip_x:
        P = np.flip(P, 2)
    if flip_t:
        P = np.flip(P, 1)
    if sign:
        P = np.stack([P[0], P[2], P[1]], 0)
    return np.ascontiguousarray(P, dtype=np.float32)


def sh_from_prob(p):
    """p(3,T,W) -> score=1−p0, hotness=p1−p2. Оба float32."""
    p = np.asarray(p, np.float32)
    s = np.subtract(np.float32(1), p[0], dtype=np.float32)
    h = np.subtract(p[1], p[2], dtype=np.float32)
    return s, h


def prob_from_sh(s, h):
    """Обратное к sh_from_prob. p0+p1+p2 = 1, если |h| ≤ s и s ∈ [0, 1]."""
    s = np.asarray(s, np.float32)
    h = np.asarray(h, np.float32)
    half = np.float32(0.5)
    p1 = half * (s + h)
    p2 = half * (s - h)
    p0 = np.float32(1) - s
    return np.stack([p0, p1, p2], 0).astype(np.float32, copy=False)


def linear_fuse(ens_views: dict, weights: dict):
    """Взвешенное среднее уже собранных видов ансамбля. Веса нормируются."""
    wsum = float(sum(weights.values()))
    if wsum <= 0:
        raise ValueError("weights")
    s = None
    h = None
    for k, w in weights.items():
        sk, hk = ens_views[k]
        c = np.float32(w / wsum)
        s = c * sk if s is None else s + c * sk
        h = c * hk if h is None else h + c * hk
    return s.astype(np.float32, copy=False), h.astype(np.float32, copy=False)


def member_delta(ens_s, ens_h, old_s, old_h, new_s, new_h, n: int = 4):
    """Заменить один из n равных членов: ens + (new − old) / n.

    Так знак одного члена выражается через уже опубликованный ансамбль,
    без повторной загрузки остальных карт.
    """
    if int(n) <= 0:
        raise ValueError("n")
    scale = np.float32(1.0 / int(n))
    s = np.asarray(ens_s, np.float32) + (np.asarray(new_s, np.float32) - np.asarray(old_s, np.float32)) * scale
    h = np.asarray(ens_h, np.float32) + (np.asarray(new_h, np.float32) - np.asarray(old_h, np.float32)) * scale
    return np.ascontiguousarray(s, dtype=np.float32), np.ascontiguousarray(h, dtype=np.float32)


def logit_avg(probs):
    """Среднее логитов, затем softmax. probs — список (3, T, W)."""
    acc = None
    n = 0
    for p in probs:
        lg = np.log(np.clip(np.asarray(p, np.float32), np.float32(1e-6), np.float32(1)))
        acc = lg if acc is None else acc + lg
        n += 1
    acc = acc / np.float32(n)
    acc = acc - acc.max(axis=0, keepdims=True)
    e = np.exp(acc)
    return (e / e.sum(axis=0, keepdims=True)).astype(np.float32)


def maxpol(id_sh, sign_sh):
    """Уверенность полярности: max p(класс) между id и знаком, min p(фон), нормировка.

    После симметрии знака p1 — нагрев исходного поля. max поднимает ту
    полярность, которую хотя бы один вид видит уверенно.
    """
    a = prob_from_sh(*id_sh)
    b = prob_from_sh(*sign_sh)
    p1 = np.maximum(a[1], b[1])
    p2 = np.maximum(a[2], b[2])
    p0 = np.minimum(a[0], b[0])
    tot = np.maximum(p0 + p1 + p2, np.float32(1e-6))
    return sh_from_prob(np.stack([p0 / tot, p1 / tot, p2 / tot], 0))


def nonlinear_member(mv: dict, name: str):
    """Один член ансамбля, все нужные виды уже в координатах id."""
    if name == "logit_full":
        keys = ("id", "flip_x", "flip_t", "sign")
        return sh_from_prob(logit_avg([prob_from_sh(*mv[k]) for k in keys]))
    if name == "maxpol":
        return maxpol(mv["id"], mv["sign"])
    if name == "shrink_full":
        keys = ("id", "flip_x", "flip_t", "sign")
        ss = np.stack([mv[k][0] for k in keys], 0)
        hh = np.stack([mv[k][1] for k in keys], 0)
        s = ss.mean(0) - np.float32(0.5) * ss.std(0)
        return s.astype(np.float32, copy=False), hh.mean(0).astype(np.float32, copy=False)
    if name == "geomean_sign":
        eps = np.float32(1e-6)
        a, b = mv["id"][0], mv["sign"][0]
        s = np.exp(np.float32(0.5) * (np.log(a + eps) + np.log(b + eps)))
        h = np.float32(0.5) * (mv["id"][1] + mv["sign"][1])
        return s.astype(np.float32, copy=False), h.astype(np.float32, copy=False)
    raise KeyError(name)


class Acc:
    """Сумма карт score/hotness. mean() делит на число add."""

    def __init__(self):
        self.s = None
        self.h = None
        self.n = 0

    def add(self, s, h):
        s = np.array(s, dtype=np.float32, copy=True)
        h = np.array(h, dtype=np.float32, copy=True)
        if self.s is None:
            self.s = s
            self.h = h
        else:
            self.s += s
            self.h += h
        self.n += 1

    def mean(self):
        if not self.n:
            raise RuntimeError("empty acc")
        inv = np.float32(1.0 / self.n)
        return (self.s * inv).astype(np.float32, copy=False), (self.h * inv).astype(np.float32, copy=False)


def _cat(chunks):
    if not chunks:
        return np.empty(0, np.float32)
    if len(chunks) == 1:
        return chunks[0]
    return np.concatenate(chunks)


def _n_ge(chunks, thr: float) -> int:
    thr = float(thr)
    return int(sum(int(np.count_nonzero(c >= thr)) for c in chunks))


def _n_items(chunks) -> int:
    return int(sum(int(c.size) for c in chunks))


class ScoreBag:
    """Хвост баллов фона и все баллы событий: точный порог FAR и пиксельный IoU.

    Порог far_threshold — порядковая статистика фона. Квантиль FAR 1e-3
    лежит в верхнем хвосте, поэтому баллы ниже gate не хранятся. Если
    найденный порог ниже gate, metrics вернёт None.
    """

    def __init__(self, gate: float = 0.25):
        self.gate = float(gate)
        self.seqs: dict[str, dict] = {}

    def add(self, sid: str, s, h, valid, ev, nu):
        s = np.asarray(s, np.float32)
        h = np.asarray(h, np.float32)
        valid = np.asarray(valid, bool)
        ev = np.asarray(ev)
        nu = np.asarray(nu)
        if s.shape != valid.shape or h.shape != s.shape:
            raise ValueError(f"shape {s.shape} {h.shape} {valid.shape}")
        pos = valid & (ev > 0)
        neg = valid & (ev == 0)
        bg = neg & (nu == 0)
        hot_t = valid & (ev >= 1) & (ev <= 4)
        cold_t = valid & (ev >= 5)
        rec = self.seqs.get(sid)
        if rec is None:
            rec = {
                "pos": [], "neg": [], "bg": [],
                "hot_tp": [], "hot_fp": [], "cold_tp": [], "cold_fp": [],
                "n_pos": 0, "n_neg": 0, "n_bg": 0, "n_bg_low": 0, "n_neg_low": 0,
                "n_hot": 0, "n_cold": 0,
                "n_hot_fp_low": 0, "n_cold_fp_low": 0,
            }
            self.seqs[sid] = rec
        rec["n_pos"] += int(pos.sum())
        rec["n_neg"] += int(neg.sum())
        rec["n_bg"] += int(bg.sum())
        rec["n_hot"] += int(hot_t.sum())
        rec["n_cold"] += int(cold_t.sum())
        g = self.gate
        if pos.any():
            rec["pos"].append(np.array(s[pos], dtype=np.float32, copy=True))
        self._tail(rec["bg"], s, bg, g, rec, "n_bg_low")
        self._tail(rec["neg"], s, neg, g, rec, "n_neg_low")
        hp = hot_t & (h > 0)
        if hp.any():
            rec["hot_tp"].append(np.array(s[hp], dtype=np.float32, copy=True))
        self._tail(rec["hot_fp"], s, (~hot_t) & valid & (h > 0), g, rec, "n_hot_fp_low")
        cp = cold_t & (h <= 0)
        if cp.any():
            rec["cold_tp"].append(np.array(s[cp], dtype=np.float32, copy=True))
        self._tail(rec["cold_fp"], s, (~cold_t) & valid & (h <= 0), g, rec, "n_cold_fp_low")

    @staticmethod
    def _tail(bucket, s, mask, gate, rec, low_key):
        if not np.any(mask):
            return
        b = s[mask]
        keep = b >= gate
        rec[low_key] += int((~keep).sum())
        if np.any(keep):
            bucket.append(np.array(b[keep], dtype=np.float32, copy=True))

    def _bg_parts(self):
        n = 0
        n_low = 0
        chunks = []
        for rec in self.seqs.values():
            n += rec["n_bg"]
            n_low += rec["n_bg_low"]
            chunks.extend(rec["bg"])
            if _n_items(rec["bg"]) + rec["n_bg_low"] != rec["n_bg"]:
                raise RuntimeError("bg tail count diverged from n_bg")
        return n, n_low, chunks

    def threshold(self, far: float):
        """Порог как metrics.far_threshold, или None, если квантиль ниже gate."""
        n, n_low, chunks = self._bg_parts()
        if n <= 0:
            raise RuntimeError("no background")
        k = int(np.ceil((1.0 - far) * n)) - 1
        k = int(np.clip(k, 0, n - 1))
        if k < n_low:
            return None
        tail = _cat(chunks)
        kt = k - n_low
        if kt < 0 or kt >= tail.size:
            return None
        part = np.partition(tail, kt)
        return float(part[kt]) + 1e-12

    def counts(self, thr: float):
        """tp, fp, fn, far, hot/cold counts. None, если хвост не покрывает порог."""
        if thr is None or not np.isfinite(thr):
            return None
        thr = float(thr)
        if thr < self.gate:
            return None
        tp = fp = fn = 0
        far_n = n_bg = 0
        hot = [0, 0, 0]
        cold = [0, 0, 0]
        per = {}
        for sid, rec in self.seqs.items():
            tpi = _n_ge(rec["pos"], thr)
            fpi = _n_ge(rec["neg"], thr)
            fni = rec["n_pos"] - tpi
            tp += tpi
            fp += fpi
            fn += fni
            per[sid] = (tpi, fpi, fni)
            bgi = _n_ge(rec["bg"], thr)
            far_n += bgi
            n_bg += rec["n_bg"]
            ht = _n_ge(rec["hot_tp"], thr)
            hot[0] += ht
            hot[2] += rec["n_hot"] - ht
            hot[1] += _n_ge(rec["hot_fp"], thr)
            ct = _n_ge(rec["cold_tp"], thr)
            cold[0] += ct
            cold[2] += rec["n_cold"] - ct
            cold[1] += _n_ge(rec["cold_fp"], thr)
        return dict(
            tp=tp, fp=fp, fn=fn, per_seq=per,
            far=float(far_n / max(n_bg, 1)), n_bg=int(n_bg),
            hot=tuple(hot), cold=tuple(cold), threshold=thr,
        )

    def report(self, thr: float, far_ref: float | None = None):
        c = self.counts(thr)
        if c is None:
            return None
        out = dict(
            threshold=float(c["threshold"]),
            far=c["far"],
            tp=int(c["tp"]), fp=int(c["fp"]), fn=int(c["fn"]),
            **M.prf(c["tp"], c["fp"], c["fn"]),
            hot=M.prf(*c["hot"]),
            cold=M.prf(*c["cold"]),
            n_bg=c["n_bg"],
            n_seq=len(c["per_seq"]),
        )
        if far_ref:
            out["far_ratio"] = float(c["far"] / far_ref)
        keys = sorted(c["per_seq"])
        if keys:
            out["iou_ci95"] = M.bootstrap_ci(c["per_seq"], keys)
        return out

    def nbytes(self) -> int:
        n = 0
        for rec in self.seqs.values():
            for k in ("pos", "neg", "bg", "hot_tp", "hot_fp", "cold_tp", "cold_fp"):
                n += sum(int(c.nbytes) for c in rec[k])
        return n
