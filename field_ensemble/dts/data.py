"""Загрузка кэша и семплирование тайлов для обучения."""
from __future__ import annotations
from pathlib import Path
import numpy as np

HOT = np.array([0, 1, 1, 1, 1, 2, 2], np.int64)  # mask_event -> {0 фон, 1 нагрев, 2 охлаждение}


def list_cache(cache_root, split):
    return sorted(Path(cache_root).glob(f"{split}/seq*.npz"))


def load(path, keys=("X", "ev", "valid")):
    z = np.load(path)
    return {k: z[k] for k in keys}


def target(ev, valid):
    y = HOT[np.minimum(ev, 6)].astype(np.int8)
    y[~valid] = -1
    return y


def mmap_items(cache_root, split, mm_root):
    """npz -> пара .npy (X float16, y int8), открываем через mmap: не держим всё в RAM."""
    out = []
    for p in list_cache(cache_root, split):
        d = Path(mm_root) / split; d.mkdir(parents=True, exist_ok=True)
        fx, fy = d / f"{p.stem}_X.npy", d / f"{p.stem}_y.npy"
        if not fy.exists():
            z = np.load(p); np.save(fx, z["X"]); np.save(fy, target(z["ev"], z["valid"]))
        out.append(dict(X=np.load(fx, mmap_mode="r"), y=np.load(fy, mmap_mode="r"), name=p.stem, src=p))
    return out


def add_weak_weights(items, labels, mm_root, split, amp_max=2.0):
    """Карта весов: пиксели слабых событий (amp_obs ≤ amp_max °C) помечаются 1 (uint8), остальные 0."""
    from . import metrics as M
    for it in items:
        fw = Path(mm_root) / split / f"{it['name']}_weak.npy"
        if not fw.exists():
            z = np.load(it["src"]); w = np.zeros(z["ev"].shape, np.uint8)
            for r in M.label_boxes(labels, it["name"]).itertuples():
                if r.kind == "event" and float(r.amp_obs_C) <= amp_max:
                    w |= M.event_pixels(r, z["x"], z["t"], z["ev"]).astype(np.uint8)
            np.save(fw, w)
        it["weak"] = np.load(fw, mmap_mode="r")
    return items


class TileSampler:
    def __init__(self, items, tile=(128, 256), p_event=0.6, seed=0, p_weak=0.0, aug=None):
        self.items = items; self.tile = tile; self.p_event = p_event; self.p_weak = p_weak
        self.aug = aug
        self.rng = np.random.default_rng(seed)
        self.ev_idx = []
        for it in items:
            idx = np.argwhere(np.asarray(it["y"]) > 0)
            if len(idx) > 5000: idx = idx[self.rng.choice(len(idx), 5000, replace=False)]
            self.ev_idx.append(idx)
        self.weak_idx = []
        for it in items:
            if "weak" in it:
                idx = np.argwhere(np.asarray(it["weak"]) > 0)
                if len(idx) > 2000: idx = idx[self.rng.choice(len(idx), 2000, replace=False)]
            else:
                idx = np.zeros((0, 2), int)
            self.weak_idx.append(idx)
        self.has_weak = [k for k, w in enumerate(self.weak_idx) if len(w)]
        self.weights = np.array([it["y"].size for it in items], float); self.weights /= self.weights.sum()

    def sample(self, n):
        th, tw = self.tile
        xs, ys, ws = [], [], []
        for _ in range(n):
            weak_pick = self.p_weak > 0 and self.has_weak and self.rng.random() < self.p_weak
            k = int(self.rng.choice(self.has_weak)) if weak_pick else self.rng.choice(len(self.items), p=self.weights)
            it = self.items[k]; X, y = it["X"], it["y"]
            T, W = y.shape
            if weak_pick:
                c = self.weak_idx[k][self.rng.integers(len(self.weak_idx[k]))]
                t0 = int(c[0]) - self.rng.integers(th); w0 = int(c[1]) - self.rng.integers(tw)
            elif self.rng.random() < self.p_event and len(self.ev_idx[k]):
                c = self.ev_idx[k][self.rng.integers(len(self.ev_idx[k]))]
                t0 = int(c[0]) - self.rng.integers(th); w0 = int(c[1]) - self.rng.integers(tw)
            else:
                t0 = self.rng.integers(max(T - th, 0) + 1); w0 = self.rng.integers(max(W - tw, 0) + 1)
            t0 = int(np.clip(t0, 0, max(T - th, 0))); w0 = int(np.clip(w0, 0, max(W - tw, 0)))
            xt = np.zeros((3, th, tw), np.float32); yt = np.full((th, tw), -1, np.int64)
            a = X[:, t0:t0 + th, w0:w0 + tw].astype(np.float32); b = y[t0:t0 + th, w0:w0 + tw]
            xt[:, :a.shape[1], :a.shape[2]] = a; yt[:b.shape[0], :b.shape[1]] = b
            wt = np.zeros((th, tw), np.float32)
            if "weak" in it:
                c_ = it["weak"][t0:t0 + th, w0:w0 + tw]; wt[:c_.shape[0], :c_.shape[1]] = c_
            if self.aug is not None:
                xt, yt = self.aug(xt, yt, it, t0, w0)
            if self.rng.random() < 0.5:                       # зеркало по x: физика симметрична
                xt = xt[:, :, ::-1]; yt = yt[:, ::-1]; wt = wt[:, ::-1]
            g = self.rng.uniform(0.75, 1.3)                      # рассогласование оценки шума синт/реал
            xt[:2] = np.clip(xt[:2] * g, -12, 12)
            xs.append(xt.copy()); ys.append(yt.copy()); ws.append(wt.copy())
        self.last_weak = np.stack(ws)
        return np.stack(xs), np.stack(ys)
