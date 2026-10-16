"""ST2: классификатор типа события на нейросетевых эмбеддингах.

Энкодеры U-Net (замороженные чекпойнты этапа 1) дают пространственные эмбеддинги
вырезки. Рядом обучается свёртка EventNet и MLP по геометрии зоны (длительность,
ширина, форма вырезки, соседние боксы). Голова сливает три ветки. Отбор — GroupKFold
по seq_id на dev (train+val) синтетики v8. Test, v11 и v12 в обучение не входят.
Реальные записи меток семи классов не имеют и в обучение не попадают.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def iter_group_folds(idx: np.ndarray, groups: np.ndarray, n_splits: int = 5):
    """Индексы GroupKFold. Возвращает пары (train_idx, val_idx) в исходной нумерации."""
    from sklearn.model_selection import GroupKFold

    idx = np.asarray(idx)
    groups = np.asarray(groups)
    if len(idx) != len(groups):
        raise ValueError("idx and groups length differ")
    out = []
    for tr, va in GroupKFold(n_splits).split(idx, groups=groups):
        tr_i, va_i = idx[tr], idx[va]
        if set(groups[tr]) & set(groups[va]):
            raise RuntimeError("group leak in GroupKFold")
        out.append((tr_i, va_i))
    return out


class EmbedFusion(nn.Module):
    """Свёртка вырезки + MLP скаляров + проекция эмбеддинга U-Net."""

    def __init__(self, n_scalar: int, d_unet: int, n_classes: int = 7, w: int = 24):
        super().__init__()
        self.d_unet = int(d_unet)
        self.w = int(w)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, w, 3, padding=1), nn.BatchNorm2d(w), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(w, 2 * w, 3, padding=1), nn.BatchNorm2d(2 * w), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(2 * w, 4 * w, 3, padding=1), nn.BatchNorm2d(4 * w), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten())
        self.mlp = nn.Sequential(nn.Linear(n_scalar, 48), nn.ReLU(inplace=True), nn.Dropout(0.1))
        hid = 16 * w + 48
        if self.d_unet:
            self.unet_proj = nn.Sequential(nn.Linear(self.d_unet, 64), nn.ReLU(inplace=True), nn.Dropout(0.2))
            hid += 64
        self.head = nn.Sequential(
            nn.Dropout(0.25), nn.Linear(hid, 96), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(96, n_classes))

    def forward(self, img, scal, unet=None):
        parts = [self.cnn(img), self.mlp(scal)]
        if self.d_unet:
            if unet is None:
                raise ValueError("unet embedding is required")
            parts.append(self.unet_proj(unet))
        return self.head(torch.cat(parts, 1))


def unet_embed(net: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Мультимасштабный эмбеддинг энкодера DTSUNet: среднее, std и центр-минус-край."""
    e1 = net.e1(x)
    e2 = net.e2(F.max_pool2d(e1, 2))
    e3 = net.e3(F.max_pool2d(e2, 2))
    e4 = net.gate(net.e4(F.max_pool2d(e3, 2)))

    def pool(e: torch.Tensor) -> torch.Tensor:
        gap = e.mean((2, 3))
        std = e.std((2, 3), unbiased=False)
        w = e.shape[-1]
        a, b = w // 3, (2 * w) // 3
        if b <= a:
            diff = torch.zeros_like(gap)
        else:
            center = e[..., a:b].mean((2, 3))
            chunks = []
            if a > 0:
                chunks.append(e[..., :a].mean((2, 3)))
            if b < w:
                chunks.append(e[..., b:].mean((2, 3)))
            edge = sum(chunks) / len(chunks)
            diff = center - edge
        return torch.cat([gap, std, diff], 1)

    return torch.cat([pool(e2), pool(e3), pool(e4)], 1)


@torch.no_grad()
def embed_images(net: nn.Module, img: np.ndarray, batch: int = 64) -> np.ndarray:
    net.eval()
    parts = []
    for i in range(0, len(img), batch):
        x = torch.from_numpy(np.ascontiguousarray(img[i:i + batch], dtype=np.float32))
        parts.append(unet_embed(net, x).numpy())
    return np.concatenate(parts, 0).astype(np.float32) if parts else np.zeros((0, 1), np.float32)


def load_unet(path: str | Path) -> nn.Module:
    from dts import models

    sd = torch.load(resolve(path), map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if isinstance(sd, dict) and "e1.0.weight" not in sd and "state" in sd:
        sd = sd["state"]
    w = int(sd["e1.0.weight"].shape[0])
    net = models.DTSUNet(w=w)
    net.load_state_dict(sd)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def pack_scalars(base: np.ndarray, img: np.ndarray, ctx: np.ndarray) -> np.ndarray:
    """19 скаляров зоны + форма вырезки + контекст соседей. Без z-карты детектора."""
    from dts import events as E

    base = np.asarray(base, np.float32)
    sh = E.shape_features(img)
    ctx = np.asarray(ctx, np.float32)
    if len(base) != len(sh) or len(base) != len(ctx):
        raise ValueError(f"pack length base {len(base)} shape {len(sh)} ctx {len(ctx)}")
    return np.nan_to_num(np.concatenate([base, sh, ctx], 1).astype(np.float32))


def zfit(a: np.ndarray, idx: np.ndarray):
    mu = a[idx].mean(0).astype(np.float32)
    sd = (a[idx].std(0) + 1e-6).astype(np.float32)
    return mu, sd


def zapply(a: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return ((a - mu) / sd).astype(np.float32)


def _class_weight(y: np.ndarray, n_classes: int) -> torch.Tensor:
    cnt = np.bincount(y, minlength=n_classes).astype(np.float32)
    w = len(y) / (n_classes * np.maximum(cnt, 1.0))
    return torch.tensor(np.clip(w, 0.25, 10.0), dtype=torch.float32)


def _inner_split(idx: np.ndarray, groups: np.ndarray, seed: int, frac: float = 0.15):
    ug = np.unique(groups[idx])
    if len(ug) < 6:
        return idx, None
    rng = np.random.default_rng(seed)
    order = rng.permutation(ug)
    n_inner = max(1, int(round(frac * len(order))))
    hold = set(order[:n_inner].tolist())
    g = groups[idx]
    inner = idx[np.array([gi in hold for gi in g])]
    fit = idx[np.array([gi not in hold for gi in g])]
    if len(fit) < 16 or len(inner) < 4:
        return idx, None
    return fit, inner


def fit_embed_fusion(img, scal, unet, unet_flip, y, groups, idx, *,
                     seed: int = 0, epochs: int = 32, patience: int = 6, min_epochs: int = 10,
                     batch: int = 64, lr: float = 2e-3, n_classes: int = 7, use_inner: bool = True):
    """Обучить EmbedFusion на idx. Нормы считаются по всему idx, ранняя остановка — по внутренним группам."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    idx = np.asarray(idx)
    d_unet = 0 if unet is None else int(unet.shape[1])
    mu_s, sd_s = zfit(scal, idx)
    if d_unet:
        mu_u, sd_u = zfit(unet, idx)
        u_all = zapply(unet, mu_u, sd_u)
        uf_all = zapply(unet_flip, mu_u, sd_u)
    else:
        mu_u = sd_u = np.zeros(1, np.float32)
        u_all = uf_all = None
    s_all = zapply(scal, mu_s, sd_s)
    fit_idx, inner = _inner_split(idx, groups, seed) if use_inner else (idx, None)
    net = EmbedFusion(scal.shape[1], d_unet, n_classes=n_classes)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-3)
    cw = _class_weight(y[fit_idx], n_classes)
    Xi = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32))
    Si = torch.from_numpy(s_all)
    Yi = torch.from_numpy(np.asarray(y, np.int64))
    Ui = torch.from_numpy(u_all) if u_all is not None else None
    Uf = torch.from_numpy(uf_all) if uf_all is not None else None

    best_state, best_f1, stall = None, -1.0, 0
    hist = []
    for ep in range(epochs):
        net.train()
        perm = rng.permutation(fit_idx)
        losses = []
        for b in range(0, len(perm), batch):
            j = perm[b:b + batch]
            if len(j) < 2:
                continue
            xb = Xi[j]
            ub = Ui[j] if Ui is not None else None
            take = torch.from_numpy(rng.random(len(j)) < 0.5)
            if bool(take.any()):
                xb = xb.clone()
                xb[take] = torch.flip(xb[take], [3])
                if ub is not None:
                    ub = ub.clone()
                    ub[take] = Uf[j][take]
            opt.zero_grad()
            loss = F.cross_entropy(net(xb, Si[j], ub), Yi[j], weight=cw, label_smoothing=0.05)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        rec = dict(epoch=ep, loss=float(np.mean(losses)) if losses else None)
        if inner is not None:
            prob = predict_probs(net, img, scal, unet, unet_flip, mu_s, sd_s, mu_u, sd_u, inner)
            f1 = _macro(y[inner], prob.argmax(1), n_classes)
            rec["inner_f1"] = f1
            if f1 > best_f1 + 1e-4:
                best_f1, stall = f1, 0
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            else:
                stall += 1
            hist.append(rec)
            if ep + 1 >= min_epochs and stall >= patience:
                break
        else:
            hist.append(rec)
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    info = dict(epochs_ran=len(hist), best_inner_f1=None if best_f1 < 0 else float(best_f1),
                last_loss=None if not hist else hist[-1].get("loss"),
                n_fit=int(len(fit_idx)), n_inner=0 if inner is None else int(len(inner)))
    return net, mu_s, sd_s, mu_u, sd_u, info


@torch.no_grad()
def predict_probs(net, img, scal, unet, unet_flip, mu_s, sd_s, mu_u, sd_u, idx=None) -> np.ndarray:
    net.eval()
    if idx is None:
        idx = np.arange(len(img))
    idx = np.asarray(idx)
    if len(idx) == 0:
        return np.zeros((0, net.head[-1].out_features), np.float32)
    s = torch.from_numpy(zapply(scal, mu_s, sd_s)[idx])
    xb = torch.from_numpy(np.ascontiguousarray(img[idx], dtype=np.float32))
    if net.d_unet:
        u = torch.from_numpy(zapply(unet, mu_u, sd_u)[idx])
        uf = torch.from_numpy(zapply(unet_flip, mu_u, sd_u)[idx])
        lo = net(xb, s, u)
        lo = 0.5 * (lo + net(torch.flip(xb, [3]), s, uf))
    else:
        lo = net(xb, s, None)
        lo = 0.5 * (lo + net(torch.flip(xb, [3]), s, None))
    return torch.softmax(lo, 1).numpy().astype(np.float32)


def _macro(yt, yp, n_classes: int) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(yt, yp, average="macro", labels=list(range(n_classes)), zero_division=0))


def score_predictions(yt, yp, prob, groups, class_names) -> dict:
    from sklearn.metrics import confusion_matrix, f1_score
    from dts import events as E

    yt = np.asarray(yt)
    yp = np.asarray(yp)
    n_classes = len(class_names)
    per = f1_score(yt, yp, average=None, labels=list(range(n_classes)), zero_division=0)
    coarse_t = [E.COARSE[class_names[i]] for i in yt]
    coarse_p = [E.COARSE[class_names[i]] for i in yp]
    r = dict(
        macro_f1=_macro(yt, yp, n_classes),
        per_class={class_names[k]: float(v) for k, v in enumerate(per)},
        coarse_macro_f1=float(f1_score(coarse_t, coarse_p, average="macro", zero_division=0)),
        accuracy=float((yt == yp).mean()) if len(yt) else 0.0,
        confusion=confusion_matrix(yt, yp, labels=list(range(n_classes))).tolist(),
        n=int(len(yt)),
        class_counts=np.bincount(yt, minlength=n_classes).tolist() if len(yt) else [0] * n_classes,
    )
    if prob is not None:
        r["ece"] = float(E.expected_calibration_error(np.asarray(prob), yt))
    if groups is not None and len(yt):
        rng = np.random.default_rng(0)
        ug = np.unique(groups)
        gi = {g: np.where(groups == g)[0] for g in ug}
        vals = []
        for _ in range(300):
            s = np.concatenate([gi[g] for g in rng.choice(ug, len(ug))])
            vals.append(_macro(yt[s], yp[s], n_classes))
        r["macro_f1_ci95"] = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
    return r


def present_macro(yt, yp) -> float:
    from sklearn.metrics import f1_score
    labels = sorted(set(np.asarray(yt).tolist()))
    if not labels:
        return 0.0
    return float(f1_score(yt, yp, average="macro", labels=labels, zero_division=0))


def crops_from_cache(cache_root, labels_path, splits) -> dict:
    """Оракульные вырезки из кэша. Порядок — как в train_stage2: sorted seq по сплитам."""
    from dts import events as E
    from dts import metrics as M
    import pandas as pd

    labels = pd.read_csv(resolve(labels_path))
    want = set(splits)
    imgs, scal, ys, groups, spl, templates = [], [], [], [], [], []
    ctx_parts = []
    seq_template = {}
    seq_csv = Path(resolve(labels_path)).parent / "sequences.csv"
    if seq_csv.exists():
        sdf = pd.read_csv(seq_csv)
        seq_template = dict(zip(sdf.seq_id.astype(str), sdf.template.astype(str)))
    files = []
    root = resolve(cache_root)
    for sp in ("train", "val", "test"):
        if sp not in want:
            continue
        files.extend(sorted((root / sp).glob("seq*.npz")))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        for p in files:
            z = np.load(p)
            X = np.asarray(z["X"], np.float32)
            x = z["x"]
            t = z["t"]
            sig = z["sigma"]
            B = M.label_boxes(labels, p.stem)
            kept = []
            ims = []
            for r in B.itertuples():
                if r.label not in E.SYNTH_TO_CLASS:
                    continue
                xa, xb = max(float(r.dist_start_m), float(x[0])), min(float(r.dist_end_m), float(x[-1]))
                if xb < xa:
                    continue
                im, f = E.crop_and_features(X, sig, x, t, xa, xb, float(r.ts), float(r.te))
                ims.append(im)
                kept.append((xa, xb, float(r.ts), float(r.te), E.CLASSES.index(E.SYNTH_TO_CLASS[r.label]), f))
            if not kept:
                continue
            xa, xb, ta, tb, yy, ff = (np.array(v) for v in zip(*kept))
            ctx = E.context_features(xa, xb, ta, tb)
            imgs.append(np.stack(ims).astype(np.float32))
            scal.append(np.stack(ff).astype(np.float32))
            ctx_parts.append(ctx.astype(np.float32))
            ys.append(yy.astype(np.int64))
            groups.append(np.array([p.stem] * len(yy)))
            spl.append(np.array([p.parent.name] * len(yy)))
            templates.append(np.array([seq_template.get(p.stem, "?")] * len(yy)))
    if not imgs:
        raise RuntimeError(f"no crops in {cache_root} splits={splits}")
    return dict(
        img=np.concatenate(imgs),
        base=np.concatenate(scal),
        ctx=np.concatenate(ctx_parts),
        y=np.concatenate(ys),
        groups=np.concatenate(groups).astype(str),
        split=np.concatenate(spl).astype(str),
        template=np.concatenate(templates).astype(str),
    )


def crops_from_synth(synth_root, splits) -> dict:
    """Вырезки напрямую из h5, без записи кэша полей. Для v12, где кэша нет."""
    from dts import events as E
    from dts import features, grid, metrics as M, synth

    info = synth.discover(resolve(synth_root))
    labels = info["labels"]
    seq_template = {}
    if info["sequences"] is not None:
        sdf = info["sequences"]
        seq_template = dict(zip(sdf.seq_id.astype(str), sdf.template.astype(str)))
    want = set(splits)
    imgs, bases, ctxs, ys, groups, spl, templates = [], [], [], [], [], [], []
    seqs = [s for s in info["seqs"] if s["split"] in want]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        for i, s in enumerate(seqs):
            d = synth.read_seq(s["path"])
            p = grid.plan(d["x"], d["t"])
            T, v = grid.resample(d["noisy"], d["valid"], p, "mean")
            ch, aux = features.make_channels(np.asarray(T, np.float32), np.asarray(v, bool))
            x, t, sig = p["x"], p["t"], aux["sigma_x"]
            B = M.label_boxes(labels, s["seq_id"])
            kept, ims = [], []
            for r in B.itertuples():
                if r.label not in E.SYNTH_TO_CLASS:
                    continue
                xa, xb = max(float(r.dist_start_m), float(x[0])), min(float(r.dist_end_m), float(x[-1]))
                if xb < xa:
                    continue
                im, f = E.crop_and_features(ch, sig, x, t, xa, xb, float(r.ts), float(r.te))
                ims.append(im)
                kept.append((xa, xb, float(r.ts), float(r.te), E.CLASSES.index(E.SYNTH_TO_CLASS[r.label]), f))
            del d, T, ch
            if not kept:
                continue
            xa, xb, ta, tb, yy, ff = (np.array(v) for v in zip(*kept))
            imgs.append(np.stack(ims).astype(np.float32))
            bases.append(np.stack(ff).astype(np.float32))
            ctxs.append(E.context_features(xa, xb, ta, tb).astype(np.float32))
            ys.append(yy.astype(np.int64))
            groups.append(np.array([s["seq_id"]] * len(yy)))
            spl.append(np.array([s["split"]] * len(yy)))
            templates.append(np.array([seq_template.get(s["seq_id"], "?")] * len(yy)))
            if i % 15 == 0:
                print("synth crops", i, len(seqs), s["seq_id"], flush=True)
    if not imgs:
        raise RuntimeError(f"no crops in {synth_root}")
    return dict(
        img=np.concatenate(imgs), base=np.concatenate(bases), ctx=np.concatenate(ctxs),
        y=np.concatenate(ys), groups=np.concatenate(groups).astype(str),
        split=np.concatenate(spl).astype(str), template=np.concatenate(templates).astype(str),
    )


def _lgbm_oof(base, img, ctx, y, fold_pairs, seed: int, n_classes: int):
    import lightgbm as lgb
    from dts import events as E

    oof = np.zeros((len(y), n_classes), np.float32)
    for f, (tr, va) in enumerate(fold_pairs):
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, class_weight="balanced",
                               random_state=seed + f, verbose=-1, n_jobs=2)
        m.fit(E.lgbm_matrix(base[tr], img[tr], ctx[tr], "full"), y[tr])
        raw = m.predict_proba(E.lgbm_matrix(base[va], img[va], ctx[va], "full"))
        for i, c in enumerate(m.classes_):
            oof[va, int(c)] = raw[:, i]
    return oof


def _subset_score(block, mask, class_names, prob, pred):
    if int(mask.sum()) == 0:
        return dict(n=0)
    yt, yp = block["y"][mask], pred[mask]
    r = score_predictions(yt, yp, prob[mask], block["groups"][mask], class_names)
    r["macro_f1_present"] = present_macro(yt, yp)
    return r


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def dump_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(_jsonable(obj), f, indent=1, ensure_ascii=False)
        f.write("\n")


def build_unet_matrix(img: np.ndarray, ckpts: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    used = []
    parts, parts_f = [], []
    flipped = np.ascontiguousarray(np.flip(img, axis=-1))
    for ck in ckpts:
        path = resolve(ck)
        if not path.exists():
            print("skip missing ckpt", path, flush=True)
            continue
        net = load_unet(path)
        parts.append(embed_images(net, img))
        parts_f.append(embed_images(net, flipped))
        used.append(str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path))
        del net
    if not parts:
        raise RuntimeError("no U-Net checkpoints")
    return np.concatenate(parts, 1), np.concatenate(parts_f, 1), used


def oof_model(img, scal, unet, unet_flip, y, groups, fold_pairs, *, seeds, epochs, patience, min_epochs, tag, use_inner=False, batch=128):
    acc = np.zeros((len(y), 7), np.float32)
    infos = []
    t0 = time.time()
    for seed in seeds:
        torch.manual_seed(seed)
        oof = np.zeros((len(y), 7), np.float32)
        for f, (tr, va) in enumerate(fold_pairs):
            net, mu_s, sd_s, mu_u, sd_u, info = fit_embed_fusion(
                img, scal, unet, unet_flip, y, groups, tr,
                seed=seed + 17 * f, epochs=epochs, patience=patience, min_epochs=min_epochs,
                n_classes=7, use_inner=use_inner, batch=batch)
            oof[va] = predict_probs(net, img, scal, unet, unet_flip, mu_s, sd_s, mu_u, sd_u, va)
            info.update(fold=f, seed=seed, tag=tag, sec=round(time.time() - t0, 1))
            infos.append(info)
            print(tag, "seed", seed, "fold", f, "loss", None if info["last_loss"] is None else round(info["last_loss"], 3),
                  "inner", None if info["best_inner_f1"] is None else round(info["best_inner_f1"], 3),
                  "ep", info["epochs_ran"], "sec", info["sec"], flush=True)
            del net
        acc += oof
    acc /= max(len(seeds), 1)
    return acc, infos


def refit_predict(img_src, scal_src, unet_src, unet_flip_src, y_src, groups_src, idx,
                  img_tgt, scal_tgt, unet_tgt, unet_flip_tgt, *, seeds, epochs, patience, min_epochs):
    acc = np.zeros((len(img_tgt), 7), np.float32)
    infos = []
    for seed in seeds:
        net, mu_s, sd_s, mu_u, sd_u, info = fit_embed_fusion(
            img_src, scal_src, unet_src, unet_flip_src, y_src, groups_src, idx,
            seed=seed, epochs=epochs, patience=patience, min_epochs=min_epochs, n_classes=7)
        # нормы источника нельзя применять вслепую: предсказание цели считает zscore
        # по mu/sd источника. Для этого подменяем статистику через конкатенацию фиктивно —
        # predict_probs считает zfit уже сохранённый. Цель — отдельные массивы.
        acc += _predict_transfer(net, scal_tgt, unet_tgt, unet_flip_tgt, img_tgt, mu_s, sd_s, mu_u, sd_u)
        infos.append(info)
        del net
    acc /= max(len(seeds), 1)
    return acc, infos


def _predict_transfer(net, scal, unet, unet_flip, img, mu_s, sd_s, mu_u, sd_u):
    return predict_probs(net, img, scal, unet, unet_flip, mu_s, sd_s, mu_u, sd_u, None)


def slice_score(y, pred, prob, groups, names, template, split):
    out = dict(all=score_predictions(y, pred, prob, groups, names))
    by_split = {}
    for sp in sorted(set(split.tolist())):
        m = split == sp
        by_split[sp] = score_predictions(y[m], pred[m], prob[m], groups[m], names)
    out["by_split"] = by_split
    by_t = {}
    for tname in sorted(set(template.tolist())):
        m = template == tname
        by_t[tname] = dict(
            n=int(m.sum()),
            macro_f1=_macro(y[m], pred[m], len(names)) if m.any() else None,
            macro_f1_present=present_macro(y[m], pred[m]) if m.any() else None,
            class_counts=np.bincount(y[m], minlength=len(names)).tolist() if m.any() else [],
        )
    out["by_template"] = by_t
    m5 = template == "F5"
    rest = template != "F5"
    out["F5"] = dict(
        n=int(m5.sum()),
        macro_f1=_macro(y[m5], pred[m5], len(names)) if m5.any() else None,
        macro_f1_present=present_macro(y[m5], pred[m5]) if m5.any() else None,
        class_counts=np.bincount(y[m5], minlength=len(names)).tolist() if m5.any() else [],
    )
    out["not_F5"] = dict(
        n=int(rest.sum()),
        macro_f1=_macro(y[rest], pred[rest], len(names)) if rest.any() else None,
        macro_f1_present=present_macro(y[rest], pred[rest]) if rest.any() else None,
    )
    return out


def load_v8_table(labels_path: str):
    from dts import events as E
    import pandas as pd

    ds = np.load(resolve("artifacts/stage2/dataset_v8.npz"), allow_pickle=True)
    img = np.asarray(ds["img"], np.float32)
    base = np.asarray(ds["scal"], np.float32)
    y = np.asarray(ds["y"], np.int64)
    groups = np.asarray(ds["groups"]).astype(str)
    split = np.asarray(ds["split"]).astype(str)
    rules = np.asarray(ds["rules"], np.int64)
    ctx = E.dataset_context(resolve("artifacts/cache/v8"), resolve(labels_path), y, groups)
    scal = pack_scalars(base, img, ctx)
    seq_template = {}
    seq_csv = resolve(labels_path).parent / "sequences.csv"
    if seq_csv.exists():
        sdf = pd.read_csv(seq_csv)
        seq_template = dict(zip(sdf.seq_id.astype(str), sdf.template.astype(str)))
    template = np.array([seq_template.get(g, "?") for g in groups])
    return dict(img=img, base=base, ctx=ctx, scal=scal, y=y, groups=groups, split=split,
                rules=rules, template=template)


def prepare_external(block, unet_ckpts: list[str]):
    from dts import events as E
    scal = pack_scalars(block["base"], block["img"], block["ctx"])
    # shape_features уже внутри pack; ctx собран по последовательности
    assert scal.shape[1] == len(E.SCALAR_NAMES) + len(E.SHAPE_NAMES) + len(E.CONTEXT_NAMES)
    u, uf, used = build_unet_matrix(block["img"], unet_ckpts)
    block = dict(block)
    block["scal"] = scal
    block["unet"] = u
    block["unet_flip"] = uf
    block["unet_ckpts"] = used
    return block


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/synth_meta/v8/labels.csv")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--min-epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--skip-robust", action="store_true")
    ap.add_argument("--skip-nounet", action="store_true")
    ap.add_argument("--skip-lgbm", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=40.0)
    ap.add_argument("--unet", action="append", default=None)
    a = ap.parse_args()
    if a.smoke:
        a.epochs, a.min_epochs, a.seeds, a.folds = 1, 1, 1, 1
        a.skip_robust = a.skip_nounet = a.skip_lgbm = True
        a.patience = 1
    torch.set_num_threads(a.threads)
    t_start = time.time()
    from dts import events as E

    ckpts = a.unet or ["artifacts/stage1/unet_v810ft.pt", "artifacts/stage1/unet_w24.pt"]
    print("load v8", flush=True)
    v8 = load_v8_table(a.labels)
    print("unet embed", v8["img"].shape, flush=True)
    unet, unet_flip, used = build_unet_matrix(v8["img"], ckpts)
    print("embed", unet.shape, used, "sec", round(time.time() - t_start, 1), flush=True)
    dev = np.where(v8["split"] != "test")[0]
    test = np.where(v8["split"] == "test")[0]
    folds = iter_group_folds(dev, v8["groups"][dev], n_splits=5)
    if a.folds < len(folds):
        folds = folds[:a.folds]
    seeds = list(range(a.seeds))
    names = E.CLASSES
    bar = 0.9545288956704258

    rules_dev = score_predictions(v8["y"][dev], v8["rules"][dev], None, v8["groups"][dev], names)
    print("rules dev", round(rules_dev["macro_f1"], 4), "n", rules_dev["n"], flush=True)

    fusion_oof, fusion_info = oof_model(
        v8["img"], v8["scal"], unet, unet_flip, v8["y"], v8["groups"], folds,
        seeds=seeds, epochs=a.epochs, patience=a.patience, min_epochs=a.min_epochs, tag="fusion", batch=a.batch)
    # при неполном числе фолдов оцениваем только покрытые val-строки
    covered = np.concatenate([va for _, va in folds])
    fusion_gkf = score_predictions(v8["y"][covered], fusion_oof[covered].argmax(1), fusion_oof[covered],
                                   v8["groups"][covered], names)
    fusion_gkf["per_fold"] = []
    for tr, va in folds:
        fusion_gkf["per_fold"].append(_macro(v8["y"][va], fusion_oof[va].argmax(1), 7))
    print("fusion gkf", round(fusion_gkf["macro_f1"], 4), fusion_gkf.get("macro_f1_ci95"), flush=True)

    nounet_gkf = None
    nounet_info = []
    if not a.skip_nounet and (time.time() - t_start) < a.max_minutes * 60:
        nounet_oof, nounet_info = oof_model(
            v8["img"], v8["scal"], None, None, v8["y"], v8["groups"], folds,
            seeds=[0], epochs=a.epochs, patience=a.patience, min_epochs=a.min_epochs, tag="nounet", batch=a.batch)
        nounet_gkf = score_predictions(v8["y"][covered], nounet_oof[covered].argmax(1), nounet_oof[covered],
                                       v8["groups"][covered], names)
        nounet_gkf["per_fold"] = [_macro(v8["y"][va], nounet_oof[va].argmax(1), 7) for _, va in folds]
        print("nounet gkf", round(nounet_gkf["macro_f1"], 4), flush=True)
    else:
        nounet_oof = None

    lgbm_gkf = None
    blend_gkf = None
    if not a.skip_lgbm:
        lgbm_oof = _lgbm_oof(v8["base"], v8["img"], v8["ctx"], v8["y"], folds, seed=0, n_classes=7)
        lgbm_gkf = score_predictions(v8["y"][covered], lgbm_oof[covered].argmax(1), lgbm_oof[covered],
                                     v8["groups"][covered], names)
        lgbm_gkf["per_fold"] = [_macro(v8["y"][va], lgbm_oof[va].argmax(1), 7) for _, va in folds]
        blend = 0.5 * fusion_oof + 0.5 * lgbm_oof
        blend_gkf = score_predictions(v8["y"][covered], blend[covered].argmax(1), blend[covered],
                                      v8["groups"][covered], names)
        blend_gkf["per_fold"] = [_macro(v8["y"][va], blend[va].argmax(1), 7) for _, va in folds]
        print("lgbm gkf", round(lgbm_gkf["macro_f1"], 4), "blend", round(blend_gkf["macro_f1"], 4), flush=True)

    # разрез OOF: строки train-сплита против val-сплита (без дообучения)
    def _masked(mask):
        m = np.intersect1d(covered, np.where(mask)[0])
        return score_predictions(v8["y"][m], fusion_oof[m].argmax(1), fusion_oof[m], v8["groups"][m], names)

    split_view = dict(train=_masked(v8["split"] == "train"), val=_masked(v8["split"] == "val"))

    report = dict(
        protocol=dict(
            selection="GroupKFold(5) by seq_id on split!=test, oracle boxes",
            bar_groupkfold_macro_f1=bar,
            bar_name="published equal_weight_raw EventNet+LGBM full",
            unet_ckpts=used,
            unet_role="frozen stage-1 encoders; 3-class segmentation, not 7-way labels",
            unet_leakage=("encoders trained on v8 train (v810ft also v10 train) with early stop on v8 val. "
                          "7-class head of each fold does not see its val groups. Test, v11, v12 never train the head."),
            scalars="SCALAR_NAMES + SHAPE_NAMES + CONTEXT_NAMES",
            real="not used: MGTS files have no 7-class labels; none were invented",
            seeds=seeds,
            epochs=a.epochs,
            min_epochs=a.min_epochs,
            patience=a.patience,
            batch=a.batch,
            threads=a.threads,
            n_dev=int(len(dev)),
            n_test=int(len(test)),
            n_covered=int(len(covered)),
            embed_dim=int(unet.shape[1]),
            scalar_dim=int(v8["scal"].shape[1]),
        ),
        groupkfold=dict(
            embed_fusion=fusion_gkf,
            rules_haklolarp=rules_dev,
        ),
        oof_by_source_split=split_view,
        fit_info=dict(fusion=fusion_info, nounet=nounet_info),
    )
    if nounet_gkf is not None:
        report["groupkfold"]["embed_nounet"] = nounet_gkf
    if lgbm_gkf is not None:
        report["groupkfold"]["lgbm_full_rerun"] = lgbm_gkf
        report["groupkfold"]["blend_fusion_lgbm"] = blend_gkf
    report["beats_published_bar"] = bool(fusion_gkf["macro_f1"] > bar + 1e-4)
    if blend_gkf is not None:
        report["blend_beats_published_bar"] = bool(blend_gkf["macro_f1"] > bar + 1e-4)

    out_v8 = resolve("artifacts/eval/s3_st2_v8.json")
    if not a.smoke:
        dump_json(out_v8, report)
        print("wrote", out_v8, flush=True)

    # финальное дообучение на всём dev и перенос
    if a.smoke:
        print("smoke done", round(time.time() - t_start, 1), flush=True)
        return report

    # Число эпох — медиана ранней остановки OOF. Дообучение идёт по всем dev-группам,
    # без второго прохода: те же веса считаются на test и на v11/v12.
    ran = [int(info["epochs_ran"]) for info in fusion_info if info.get("epochs_ran")]
    refit_epochs = int(np.median(ran)) if ran else a.epochs
    refit_epochs = max(refit_epochs, 8)
    print("refit dev epochs", refit_epochs, flush=True)
    packed_models = []
    for seed in seeds:
        net, mu_s, sd_s, mu_u, sd_u, info = fit_embed_fusion(
            v8["img"], v8["scal"], unet, unet_flip, v8["y"], v8["groups"], dev,
            seed=seed + 100, epochs=refit_epochs, patience=refit_epochs, min_epochs=refit_epochs,
            n_classes=7, use_inner=False, batch=a.batch)
        packed_models.append((net, mu_s, sd_s, mu_u, sd_u, info))
        print("refit", seed, "ep", info["epochs_ran"], flush=True)
    test_prob = np.zeros((len(test), 7), np.float32)
    for net, mu_s, sd_s, mu_u, sd_u, info in packed_models:
        test_prob += _predict_transfer(
            net, v8["scal"][test], unet[test], unet_flip[test], v8["img"][test], mu_s, sd_s, mu_u, sd_u)
    test_prob /= len(packed_models)
    test_score = score_predictions(v8["y"][test], test_prob.argmax(1), test_prob, v8["groups"][test], names)
    print("v8 test", round(test_score["macro_f1"], 4), flush=True)
    report["test"] = dict(embed_fusion=test_score, refit_epochs=refit_epochs)
    report["fit_info"]["refit_dev"] = [info for *_, info in packed_models]
    report["elapsed_sec"] = round(time.time() - t_start, 1)
    dump_json(out_v8, report)

    if a.skip_robust:
        print("skip robust", flush=True)
        return report

    robust = {}
    targets = [
        ("v11", "cache", "artifacts/cache/v11", "data/synth_meta/v11/labels.csv", ("val", "test")),
        ("v12", "synth", "/workspace/synth_v12/d", None, ("val", "test")),
    ]
    for name, kind, src, labels, splits in targets:
        if (time.time() - t_start) > 80 * 60 and name == "v12":
            print("time budget, skip", name, flush=True)
            robust[name] = dict(skipped="time budget")
            continue
        print("crops", name, flush=True)
        if kind == "cache":
            block = crops_from_cache(src, labels, splits)
        else:
            block = crops_from_synth(src, splits)
        block = prepare_external(block, ckpts)
        acc = np.zeros((len(block["y"]), 7), np.float32)
        for net, mu_s, sd_s, mu_u, sd_u, info in packed_models:
            acc += _predict_transfer(net, block["scal"], block["unet"], block["unet_flip"], block["img"],
                                     mu_s, sd_s, mu_u, sd_u)
        acc /= len(packed_models)
        pred = acc.argmax(1)
        robust[name] = slice_score(block["y"], pred, acc, block["groups"], names, block["template"], block["split"])
        robust[name]["n"] = int(len(block["y"]))
        robust[name]["unet_ckpts"] = block["unet_ckpts"]
        print(name, "macro", round(robust[name]["all"]["macro_f1"], 4),
              "F5", robust[name]["F5"].get("macro_f1"), "n", robust[name]["n"], flush=True)
        del block

    for net, *_rest in packed_models:
        del net
    report["robust"] = robust
    report["elapsed_sec"] = round(time.time() - t_start, 1)
    dump_json(out_v8, report)
    dump_json(resolve("artifacts/eval/s3_st2_robust.json"), dict(
        bar_groupkfold_macro_f1=bar,
        v8_groupkfold_macro_f1=fusion_gkf["macro_f1"],
        v8_test=test_score,
        robust=robust,
        elapsed_sec=report["elapsed_sec"],
        note="Heads fit on v8 dev only. v11/v12 labels used solely as evaluation. Real MGTS not scored.",
    ))
    print("DONE", round(report["elapsed_sec"], 1), flush=True)
    return report


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    main()
