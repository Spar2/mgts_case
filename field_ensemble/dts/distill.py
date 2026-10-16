"""Дистилляция ансамбля ens_u2ft_tcn_lgbm в один DTSUNet.

Учитель — равное среднее четырёх карт (unet_w24, unet_v810ft, tcn, pixel_lgbm):
s = 1 − p(фон), h = p(нагрев) − p(охлаждение). Ученик учится на синтетике.
Реальные CSV сюда не входят. Порог и отбор — synth v8 val, FAR 1e-3.
"""
from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Опубликованная точка ens_u2ft_tcn_lgbm (artifacts/eval/r7_ens_u2ft_tcn_lgbm.json).
WINNER_VAL = 0.38671235821914546
WINNER_TEST = 0.4052762065230473
WINNER_THR = 0.8608087301264272
WINNER_V10 = 0.4038977883892501
WINNER_V11 = 0.40390028295846503

STUDENT_WIDTH = 16
WIDTH_MAX = 24
BEAT_EPS = 1e-4
PROTOCOL_TOL = 1e-4
NESTED_GAIN = 0.003
CLOSE_GAP = 0.015

TEACHER_NAMES = ("unet_w24", "unet_v810ft", "tcn", "pixel_lgbm")


def is_proxy(ver: str, stem: str, mod: int = 10) -> bool:
    """Стабильный holdout обучения: не val и не test."""
    key = f"{ver}:{stem}".encode()
    return (zlib.crc32(key) & 0xFFFFFFFF) % int(mod) == 0


def select_teacher_rows(rows, k: int, max_pixels: int = 2_200_000):
    """Круг по (версия, шаблон). Слишком длинные кабели не берём: LightGBM на них не укладывается в бюджет."""
    buckets: dict[tuple, list] = {}
    for row in rows:
        if int(row["n_pix"]) > int(max_pixels):
            continue
        buckets.setdefault((row["ver"], row["template"]), []).append(row)
    for bucket in buckets.values():
        bucket.sort(key=lambda r: r["stem"])
    keys = sorted(buckets)
    out = []
    while len(out) < int(k):
        moved = False
        for key in keys:
            if not buckets[key]:
                continue
            out.append(buckets[key].pop(0))
            moved = True
            if len(out) >= int(k):
                break
        if not moved:
            break
    return out


def split_train_proxy(rows, min_proxy: int = 2, min_train: int = 4):
    """Отделяет proxy-файлы. Если их мало, забирает хвост по имени из train."""
    train, proxy = [], []
    for row in rows:
        if row.get("proxy"):
            proxy.append(row)
        else:
            train.append(row)
    if len(proxy) < min_proxy and len(train) > min_train:
        order = sorted(range(len(train)), key=lambda i: (train[i]["ver"], train[i]["stem"]))
        move = []
        for i in reversed(order):
            if len(proxy) + len(move) >= min_proxy:
                break
            if len(train) - len(move) <= min_train:
                break
            move.append(i)
        for i in sorted(move, reverse=True):
            proxy.append(train.pop(i))
    return train, proxy


def mean_scores(parts):
    """parts: список (s, h). Равное среднее, как scripts/ensemble_preds.py."""
    if not parts:
        raise ValueError("нет карт учителя")
    s = np.mean([np.asarray(a, np.float32) for a, _ in parts], axis=0)
    h = np.mean([np.asarray(b, np.float32) for _, b in parts], axis=0)
    return s.astype(np.float32), h.astype(np.float32)


def probs_from_sh(s, h):
    """Восстанавливает симплекс {фон, нагрев, охлаждение} из s и h.

    s = 1 − p_фон, h = p_нагрев − p_охлаждение, p_нагрев + p_охлаждение = s.
    Ось классов — 1 для карты (B, H, W) и 0 для более коротких тензоров.
    Отрицательные доли от float16 обнуляются и перенормируются.
    """
    s = s.clamp(0, 1)
    p_bg = 1.0 - s
    p_hot = 0.5 * (s + h)
    p_cold = 0.5 * (s - h)
    axis = 1 if s.ndim >= 3 else 0
    p = torch.stack([p_bg, p_hot, p_cold], dim=axis)
    p = p.clamp(min=0)
    denom = p.sum(dim=axis, keepdim=True).clamp(min=1e-6)
    return p / denom


def distill_loss(
    logits,
    teacher_p,
    hard_y,
    temperature: float = 2.0,
    w_kl: float = 1.0,
    w_bce: float = 1.0,
    w_hard: float = 0.25,
):
    """KL к смягчённому учителю, BCE по s (это оценка на FAR) и лёгкий CE по жёстким меткам.

    hard_y: long, −1 не входит. Веса классов 1, 3, 3 — как в dts.losses.
    Температура применяется к KL. BCE считается при T = 1: на валидации s = 1 − softmax(z)_фон.
    """
    valid = hard_y >= 0
    if int(valid.sum().item()) == 0:
        zero = logits.sum() * 0
        return zero, dict(kl=0.0, bce=0.0, hard=0.0)
    mask = valid.to(dtype=logits.dtype)
    denom = mask.sum().clamp(min=1)
    temperature = float(temperature)
    log_t = teacher_p.clamp(min=1e-6).log()
    t_soft = torch.softmax(log_t / temperature, dim=1)
    log_s = F.log_softmax(logits / temperature, dim=1)
    kl_map = (t_soft * (t_soft.clamp(min=1e-6).log() - log_s)).sum(dim=1)
    kl = (kl_map * mask).sum() / denom * (temperature ** 2)

    s_hat = (1.0 - torch.softmax(logits, dim=1)[:, 0]).clamp(1e-6, 1.0 - 1e-6)
    s_teacher = (1.0 - teacher_p[:, 0]).clamp(0, 1)
    bce_map = F.binary_cross_entropy(s_hat, s_teacher, reduction="none")
    bce = (bce_map * mask).sum() / denom

    weight = logits.new_tensor([1.0, 3.0, 3.0])
    hard = F.cross_entropy(logits, hard_y, weight=weight, ignore_index=-1)
    total = w_kl * kl + w_bce * bce + w_hard * hard
    return total, dict(kl=float(kl.detach()), bce=float(bce.detach()), hard=float(hard.detach()))


def pixel_counts(s, ev, nu, valid, thr: float) -> dict:
    """tp/fp/fn как dts.metrics.confusion: истина — ev > 0 на valid. FAR — только чистый фон."""
    s = np.asarray(s)
    ev = np.asarray(ev)
    nu = np.asarray(nu)
    valid = np.asarray(valid, bool)
    pred = (s >= thr) & valid
    truth = (ev > 0) & valid
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    bg = valid & (ev == 0) & (nu == 0)
    return dict(
        tp=tp,
        fp=fp,
        fn=fn,
        far_n=int((pred & bg).sum()),
        far_d=int(bg.sum()),
    )


def iou_from_groups(frames, far: float = 1e-3) -> dict:
    """Полный (не компактный) IoU. Порог — квантиль всего чистого фона этих кадров."""
    from dts import metrics as M

    bg = np.concatenate([np.asarray(fr["bg"], np.float64).ravel() for fr in frames if len(fr["bg"])])
    thr = M.far_threshold(bg, far)
    tp = fp = fn = 0
    far_n = 0
    for fr in frames:
        ev = np.asarray(fr["ev"], np.float64).ravel()
        nu = np.asarray(fr["nu"], np.float64).ravel()
        b = np.asarray(fr["bg"], np.float64).ravel()
        tp += int((ev >= thr).sum())
        fn += int((ev < thr).sum())
        fp += int((nu >= thr).sum()) + int((b >= thr).sum())
        far_n += int((b >= thr).sum())
    stats = M.prf(tp, fp, fn)
    stats.update(threshold=float(thr), far=float(far_n / max(bg.size, 1)), tp=tp, fp=fp, fn=fn, n_bg=int(bg.size))
    return stats


def promotion_decision(rec: dict) -> dict:
    """BEATS_WINNER только если протокол сошёлся и пройдены все ворота.

    Ворота: val IoU строго выше опубликованного более чем на 1e-4,
    прирост micro-IoU на 5 фолдах (как s3_sel2_cv / s3_sel_core) не меньше 0.003,
    v10 и v11 не хуже опубликованных. Одно проваленное ворота — отказ.
    """
    winner_val = float(rec["winner_val_recomputed"])
    student_val = float(rec["student_val"])
    protocol_ok = abs(winner_val - WINNER_VAL) <= PROTOCOL_TOL
    for key, published, tol in (
        ("winner_test_recomputed", WINNER_TEST, 5e-4),
        ("winner_v10_recomputed", WINNER_V10, 5e-4),
        ("winner_v11_recomputed", WINNER_V11, 5e-4),
    ):
        if rec.get(key) is None:
            continue
        if abs(float(rec[key]) - published) > tol:
            protocol_ok = False
    val_win = student_val > WINNER_VAL + BEAT_EPS
    delta5 = rec.get("nested_delta5")
    nested_win = delta5 is not None and float(delta5) >= NESTED_GAIN
    transfer_ok = (
        float(rec["v10"]) >= WINNER_V10 - BEAT_EPS
        and float(rec["v11"]) >= WINNER_V11 - BEAT_EPS
    )
    close = student_val >= WINNER_VAL - CLOSE_GAP
    beats = bool(protocol_ok and val_win and nested_win and transfer_ok)
    reasons = []
    if not protocol_ok:
        reasons.append("protocol")
    if not val_win:
        reasons.append("val")
    if not nested_win:
        reasons.append("nested")
    if not transfer_ok:
        reasons.append("transfer")
    if not close:
        reasons.append("not_close")
    return dict(
        beats=beats,
        close=bool(close),
        protocol_ok=bool(protocol_ok),
        val_win=bool(val_win),
        nested_win=bool(nested_win),
        transfer_ok=bool(transfer_ok),
        reason="ok" if beats else ",".join(reasons),
    )


def assert_safe_output(path, root) -> Path:
    """Запрещает запись в веса учителя, релиз и каталог его карт.

    Пути под /tmp разрешены: туда можно класть короткий смоук, не официальный ckpt.
    """
    root = Path(root).resolve()
    p = Path(path)
    if not p.is_absolute():
        p = root / p
    p = p.resolve()
    bases = [root]
    tmp = Path("/tmp").resolve()
    if tmp not in bases:
        bases.append(tmp)
    rel = None
    base = None
    for cand in bases:
        try:
            rel = p.relative_to(cand)
            base = cand
            break
        except ValueError:
            continue
    if rel is None:
        raise RuntimeError(f"путь вне рабочей папки: {p}")
    if base == root:
        parts = rel.parts
        if parts[:2] == ("artifacts", "stage1"):
            raise RuntimeError(f"отказ писать в веса учителя: {p}")
        if parts and parts[0] in ("models_release", "models_release_dom"):
            raise RuntimeError(f"отказ писать в релиз: {p}")
        if "ens_u2ft_tcn_lgbm" in parts:
            raise RuntimeError(f"отказ писать в карты победителя: {p}")
    return p
