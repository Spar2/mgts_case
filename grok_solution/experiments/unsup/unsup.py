"""Безучительный слепой прогнозист ΔT для Raman-DTS.

Сеть предсказывает канал ΔT/σ в точке (t, x) только по соседям за пределами
полосы blind_x (41 точка ≈ 10 м) и только по их прошлому/настоящему.
Собственный столбец x не читается ни в какой момент: длинное узкое событие
нельзя скопировать из своей же истории. Бутылочное горлышко узкое, чтобы
не выучить восстановление очага по краю (так провалился synth-AE с дыркой 8×16).

Скор — робастно нормированный остаток по строке: |r| / (1.4826 · MAD_x).
"""
from __future__ import annotations

import time
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dts import events as E

# 15 строк × 120 с = 30 мин, 41 точка × 0.25 м ≈ 10.3 м.
# Медиана hotspot на v8 — около 6 строк и 37 точек: окно шире типичного очага.
BLIND_T = 15
BLIND_X = 41
WIDTH = 8
BOTTLENECK = 4
HUBER_DELTA = 1.5
Z_FLOOR = 1e-3


def delay_time(x: torch.Tensor, lag: int) -> torch.Tensor:
    """delayed[i] = x[i - lag]; сверху нули. Будущее не подмешивается."""
    if lag <= 0:
        return x
    return F.pad(x, (0, 0, lag, 0))[:, :, :-lag, :]


def causal_time(x: torch.Tensor, conv: nn.Conv2d) -> torch.Tensor:
    """Свёртка (k, 1) с выравниванием: выход i видит только строки ≤ i."""
    k = conv.kernel_size[0]
    dil = conv.dilation[0]
    span = (k - 1) * dil
    y = conv(F.pad(x, (0, 0, span, 0)))
    if y.shape[-2] != x.shape[-2]:
        raise RuntimeError(f"time align {tuple(y.shape)} vs {tuple(x.shape)}")
    return y


def left_view(x: torch.Tensor, conv: nn.Conv2d, margin: int) -> torch.Tensor:
    """Свёртка (1, k): выход j видит только столбцы ≤ j - margin."""
    k = conv.kernel_size[1]
    dil = conv.dilation[1]
    span = (k - 1) * dil
    y = conv(F.pad(x, (span + margin, 0, 0, 0)))
    if y.shape[-1] < x.shape[-1]:
        raise RuntimeError(f"space align {tuple(y.shape)} vs {tuple(x.shape)} margin={margin}")
    return y[..., : x.shape[-1]]


def right_view(x: torch.Tensor, conv: nn.Conv2d, margin: int) -> torch.Tensor:
    """Зеркало left_view: выход j видит только столбцы ≥ j + margin."""
    return left_view(x.flip(-1), conv, margin).flip(-1)


def blind_box(i: int, j: int, blind_t: int = BLIND_T, blind_x: int = BLIND_X):
    """Полуинтервал [r0, r1) × [c0, c1) — 15×41 вокруг (i, j), включая сам пиксель.

    Модель слепа строже: невидима вся полоса столбцов [c0, c1) на всех временах.
    """
    half = blind_x // 2
    return i - blind_t + 1, i + 1, j - half, j + half + 1


class BlindSpotForecaster(nn.Module):
    """Прогноз одного канала по прошлому и настоящему соседей слева и справа.

    Ближнее кольцо начинается сразу за дыркой (margin = blind_x//2+1).
    Дальнее — ещё на 40 точек (10 м), чтобы плато шире дырки не задавало ответ целиком.
    """

    def __init__(self, cin=3, width=WIDTH, bottleneck=BOTTLENECK, blind_t=BLIND_T, blind_x=BLIND_X,
                 k_time=5, k_space=7, far_extra=40):
        super().__init__()
        if blind_t < 1 or blind_x % 2 == 0 or blind_x < 3:
            raise ValueError("blind_t >= 1 и blind_x нечётное >= 3")
        self.cin = int(cin)
        self.width = int(width)
        self.bottleneck = int(bottleneck)
        self.blind_t = int(blind_t)
        self.blind_x = int(blind_x)
        self.k_time = int(k_time)
        self.k_space = int(k_space)
        near = blind_x // 2 + 1
        self.margins = (near, near + int(far_extra))
        self.past_time = nn.ModuleList([
            nn.Conv2d(cin if i == 0 else width, width, (k_time, 1)) for i in range(2)])
        self.now_time = nn.Conv2d(cin, width, (3, 1))

        def bank():
            return nn.ModuleList([nn.Conv2d(width, width, (1, k_space)) for _ in self.margins])

        self.past_left, self.past_right = bank(), bank()
        self.now_left, self.now_right = bank(), bank()
        n_feat = 4 * len(self.margins) * width
        self.head = nn.Sequential(
            nn.Conv2d(n_feat, bottleneck, 1), nn.GELU(),
            nn.Conv2d(bottleneck, 1, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = delay_time(x, self.blind_t)
        for conv in self.past_time:
            h = F.gelu(causal_time(h, conv))
        n = F.gelu(causal_time(x, self.now_time))
        parts = []
        for conv, m in zip(self.past_left, self.margins):
            parts.append(F.gelu(left_view(h, conv, m)))
        for conv, m in zip(self.past_right, self.margins):
            parts.append(F.gelu(right_view(h, conv, m)))
        for conv, m in zip(self.now_left, self.margins):
            parts.append(F.gelu(left_view(n, conv, m)))
        for conv, m in zip(self.now_right, self.margins):
            parts.append(F.gelu(right_view(n, conv, m)))
        return self.head(torch.cat(parts, 1))


def huber_masked(pred, target, valid, delta=HUBER_DELTA):
    w = (valid > 0.5).to(dtype=pred.dtype)
    r = pred - target
    a = r.abs()
    hub = torch.where(a <= delta, 0.5 * r * r, delta * (a - 0.5 * delta))
    return (hub * w).sum() / w.sum().clamp(min=1.0)


def robust_abs_z(resid: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """|r| / (1.4826 · MAD_x). MAD — медиана |r − med_x(r)| по валидным x.

    В числителе |r|, а не |r−med|: общий сдвиг строки (скачок режима) не вычитается
    и даёт большой z по всей ширине. Вырожденный MAD заменяется медианой масштабов.
    """
    valid = np.asarray(valid, bool)
    r = np.where(valid, np.asarray(resid, np.float64), np.nan)
    with np.errstate(all="ignore"):
        med = np.nanmedian(r, axis=1)
        mad = np.nanmedian(np.abs(r - med[:, None]), axis=1)
    counts = valid.sum(axis=1)
    scale = 1.4826 * mad
    good = (counts >= 8) & np.isfinite(scale) & (scale > 1e-8)
    finite = scale[good]
    fallback = float(np.median(finite)) if finite.size else 1.0
    fallback = max(fallback, Z_FLOOR)
    scale = np.where(good, scale, fallback)
    z = np.abs(np.nan_to_num(r, nan=0.0)) / scale[:, None]
    return np.where(valid, z, 0.0).astype(np.float32)


def anomaly_from_residual(resid: np.ndarray, valid: np.ndarray):
    """Скор и hotness (плюс = наблюдение горячее прогноза)."""
    valid = np.asarray(valid, bool)
    resid = np.asarray(resid, np.float32)
    z = robust_abs_z(resid, valid)
    hot = np.sign(np.where(valid, resid, 0.0)).astype(np.float32)
    return z, hot


@torch.no_grad()
def predict_residual(model: BlindSpotForecaster, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """X (3, T, W) → pred, resid=obs−pred, valid. Канал 0 — цель."""
    model.eval()
    X = np.ascontiguousarray(X, dtype=np.float32)
    pred = model(torch.from_numpy(X)[None])[0, 0].numpy()
    valid = X[2] > 0.5
    resid = np.where(valid, X[0] - pred, 0.0).astype(np.float32)
    return pred.astype(np.float32), resid, valid


def score_field(model: BlindSpotForecaster, X: np.ndarray):
    _pred, resid, valid = predict_residual(model, X)
    return anomaly_from_residual(resid, valid)


def _crop(X: np.ndarray, th: int, tw: int, rng: np.random.Generator) -> np.ndarray:
    _c, T, W = X.shape
    t0 = 0 if T <= th else int(rng.integers(0, T - th + 1))
    w0 = 0 if W <= tw else int(rng.integers(0, W - tw + 1))
    src = np.array(X[:, t0:t0 + th, w0:w0 + tw], dtype=np.float32, copy=True)
    if rng.random() < 0.5:
        src = src[:, :, ::-1].copy()
    out = np.zeros((X.shape[0], th, tw), np.float32)
    out[:, :src.shape[1], :src.shape[2]] = src
    return out


def _fixed_probe(X: np.ndarray, blind_t: int, blind_x: int):
    """Постоянный контроль: фон как есть и тот же кроп с невидимым горбом +8σ."""
    _c, T, W = X.shape
    th, tw = min(T, 96), min(W, 192)
    t0 = max(0, (T - th) // 2)
    w0 = max(0, (W - tw) // 2)
    base = np.array(X[:, t0:t0 + th, w0:w0 + tw], dtype=np.float32, copy=True)
    bump = base.copy()
    i, j = th // 2, tw // 2
    half_x = min(4, blind_x // 4)
    r0, r1 = max(0, i - 2), min(th, i + 3)
    c0, c1 = max(0, j - half_x), min(tw, j + half_x + 1)
    # горб уже дырки и по времени, и по x
    assert (r1 - r0) < blind_t and (c1 - c0) < blind_x
    bump[0, r0:r1, c0:c1] += 8.0
    return base, bump, (r0, r1, c0, c1)


@torch.no_grad()
def _probe_maes(model, base, bump, box):
    model.eval()
    def resid(arr):
        pred = model(torch.from_numpy(arr)[None])[0, 0].numpy()
        return np.abs(arr[0] - pred)
    rb, re = resid(base), resid(bump)
    r0, r1, c0, c1 = box
    return float(rb.mean()), float(re[r0:r1, c0:c1].mean())


def train_forecaster(fields: Iterable[np.ndarray], minutes: float = 8.0, seed: int = 0, width: int = WIDTH,
                     bottleneck: int = BOTTLENECK, blind_t: int = BLIND_T, blind_x: int = BLIND_X,
                     batch: int = 8, tile_t: int = 80, tile_x: int = 160, lr: float = 2e-3,
                     max_steps: int = 10**9, probe_every: int = 40, log_every: int = 20, tag: str = ""):
    """Обучение на списке полей (3, T, W). Метки не используются.

    Чекпоинт — состояние с наименьшей ошибкой фона среди тех, где синтетический
    горб внутри дырки не восстановился (остаток горба > 4 при амплитуде 8).
    """
    fields = [np.ascontiguousarray(a, dtype=np.float32) for a in fields]
    if not fields:
        raise ValueError("empty training fields")
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = BlindSpotForecaster(width=width, bottleneck=bottleneck, blind_t=blind_t, blind_x=blind_x)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    areas = np.array([a.shape[1] * a.shape[2] for a in fields], np.float64)
    prob = areas / areas.sum()
    base, bump, box = _fixed_probe(fields[int(np.argmax(areas))], blind_t, blind_x)
    hist = []
    best_bg, best_state = float("inf"), None
    t0 = time.time()
    budget = float(minutes) * 60.0
    step = 0
    while step <= max_steps and (time.time() - t0) < budget:
        do_probe = step == 0 or (probe_every and step % probe_every == 0)
        if do_probe or step == 0:
            bg, ev = _probe_maes(model, base, bump, box)
            rec = dict(step=step, bg=round(bg, 4), event=round(ev, 4), sec=round(time.time() - t0, 1))
            if ev > 4.0 and bg < best_bg:
                best_bg = bg
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                rec["saved"] = True
            hist.append(rec)
            if step == 0 or step % log_every == 0:
                print(f"{tag}step {step} bg {bg:.3f} event {ev:.3f} best_bg {best_bg:.3f} "
                      f"sec {rec['sec']:.0f}", flush=True)
            if step == max_steps or (time.time() - t0) >= budget:
                break
        model.train()
        xb = np.stack([_crop(fields[int(rng.choice(len(fields), p=prob))], tile_t, tile_x, rng) for _ in range(batch)])
        xt = torch.from_numpy(xb)
        opt.zero_grad(set_to_none=True)
        loss = huber_masked(model(xt), xt[:, 0:1], xt[:, 2:3])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        frac = min(1.0, (time.time() - t0) / budget) if budget > 0 else 1.0
        for g in opt.param_groups:
            g["lr"] = lr * (1.0 - 0.85 * frac)
        opt.step()
        step += 1
        if log_every and step % log_every == 0:
            hist.append(dict(step=step, loss=round(float(loss.detach()), 5), sec=round(time.time() - t0, 1)))
            print(f"{tag}step {step} loss {float(loss.detach()):.4f} sec {time.time() - t0:.0f}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    bg, ev = _probe_maes(model, base, bump, box)
    hist.append(dict(step=step, bg=round(bg, 4), event=round(ev, 4), sec=round(time.time() - t0, 1), final=True))
    print(f"{tag}final step {step} bg {bg:.3f} event {ev:.3f} sec {time.time() - t0:.0f}", flush=True)
    return model, hist


def assemble_zones(score, hotness, valid, x, t, dT, thr, min_px=12):
    """Порог → зоны dts.events.zones_from_prob плюс пик ΔT в °C и пустой класс."""
    zones = E.zones_from_prob(score, hotness, valid, x, t, float(thr), min_px=min_px)
    valid = np.asarray(valid, bool)
    dT = np.asarray(dT, np.float32)
    score = np.asarray(score, np.float32)
    for z in zones:
        sl = (slice(z["t0"], z["t1"]), slice(z["x0"], z["x1"]))
        m = (score[sl] >= thr) & valid[sl]
        vals = dT[sl][m]
        if vals.size:
            z["peak_dT_C"] = float(vals.max() if z["polarity"] == "hot" else vals.min())
            z["median_dT_C"] = float(np.median(vals))
        else:
            z["peak_dT_C"] = 0.0
            z["median_dT_C"] = 0.0
        z["alarm"] = True
        z["class"] = ""
        z["class_prob"] = {"": 0.0}
    return zones


def zone_geometry(z, score, valid, x, t, thr) -> dict:
    """Длительность, протяжённость, доля кабеля под флагом в те же сроки."""
    t0, t1, x0, x1 = z["t0"], z["t1"], z["x0"], z["x1"]
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 120.0
    dx = float(np.median(np.diff(x))) if len(x) > 1 else 0.25
    dur_h = (t1 - t0) * dt / 3600.0
    width_m = (x1 - x0) * dx
    occupied = np.asarray(valid, bool).any(0)
    if occupied.any():
        cable_m = float(x[occupied][-1] - x[occupied][0]) + dx
    else:
        cable_m = float(x[-1] - x[0]) + dx
    vv = np.asarray(valid, bool)[t0:t1]
    flagged = (np.asarray(score)[t0:t1] >= thr) & vv
    row_flag_frac = float(flagged.sum() / max(int(vv.sum()), 1))
    return dict(dur_h=dur_h, width_m=width_m, width_frac=width_m / max(cable_m, 1.0),
                row_flag_frac=row_flag_frac, cable_m=cable_m,
                touches_end=bool(x0 <= 2 or x1 >= len(x) - 2))


def judge_zone(stats: dict) -> tuple[str, list]:
    """Грубое физическое суждение. Это не метка и не эталон."""
    reasons = []
    peak = float(stats.get("peak_dT_C", 0.0))
    amp = abs(float(stats.get("abs_peak_C", abs(peak))))
    if stats["polarity"] == "hot":
        reasons.append(f"знак: нагрев, пик {peak:+.2f} °C")
    else:
        reasons.append(f"знак: охлаждение, пик {peak:+.2f} °C")
    reasons.append(f"длительность {stats['dur_h']:.2f} ч, протяжённость {stats['width_m']:.1f} м, "
                   f"доля кабеля в эти сроки под флагом {stats['row_flag_frac']:.3f}")
    if stats["row_flag_frac"] >= 0.25 or stats["width_frac"] >= 0.35:
        reasons.append("в те же сроки помечена большая доля трассы — похоже на скачок общего режима, а не на локальный очаг")
        verdict = "похоже на артефакт режима"
    elif stats["width_m"] <= 2.0 and stats["dur_h"] >= 2.0:
        reasons.append("узкая полоса держится часами — так выглядят коннектор или стабильный точечный контакт")
        verdict = "возможно коннектор или узкий контакт"
    elif stats["dur_h"] >= 8.0 and stats["width_m"] >= 30.0:
        reasons.append("многочасовой широкий ход согласуется с суточным или секционным дрейфом помещения")
        verdict = "похоже на суточный ход"
    elif amp < 0.25:
        reasons.append("амплитуда меньше 0.25 °C — уровень шума или остатка вычитания фона")
        verdict = "слабо, возможно шум"
    elif 0.4 <= stats["width_m"] <= 80.0 and 0.08 <= stats["dur_h"] <= 18.0 and amp >= 0.4:
        reasons.append("размер и амплитуда похожи на локальный прогрев или остывание участка, а не на сбой всей трассы")
        verdict = "правдоподобно как тепловое событие"
    else:
        reasons.append("форма не совпала с узкими шаблонами очага, коннектора и суточного хода")
        verdict = "неоднозначно"
    if stats.get("touches_end"):
        reasons.append("зона упирается в конец размеченной трассы — край волокна часто даёт ступеньку")
    return verdict, reasons


def describe_zone(z, score, valid, x, t, thr) -> dict:
    geo = zone_geometry(z, score, valid, x, t, thr)
    stats = dict(geo)
    stats["polarity"] = z["polarity"]
    stats["peak_dT_C"] = float(z.get("peak_dT_C", 0.0))
    stats["abs_peak_C"] = abs(stats["peak_dT_C"])
    stats["median_dT_C"] = float(z.get("median_dT_C", 0.0))
    stats["score"] = float(z["score"])
    stats["n_px"] = int(z["n_px"])
    verdict, reasons = judge_zone(stats)
    return dict(id=z["id"], t_start_s=z["t_start_s"], t_end_s=z["t_end_s"], start_m=z["start_m"], end_m=z["end_m"],
                polarity=z["polarity"], score=float(z["score"]), n_px=int(z["n_px"]),
                peak_dT_C=stats["peak_dT_C"], median_dT_C=stats["median_dT_C"],
                dur_h=round(geo["dur_h"], 3), width_m=round(geo["width_m"], 2),
                width_frac=round(geo["width_frac"], 4), row_flag_frac=round(geo["row_flag_frac"], 4),
                touches_end=geo["touches_end"], verdict=verdict, reasons=reasons)


def save_model(model: BlindSpotForecaster, path, **meta) -> None:
    path = __import__("pathlib").Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=model.state_dict(), config=dict(
        cin=model.cin, width=model.width, bottleneck=model.bottleneck, blind_t=model.blind_t,
        blind_x=model.blind_x, k_time=model.k_time, k_space=model.k_space, margins=list(model.margins),
    ), **meta), path)


def load_model(path) -> BlindSpotForecaster:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    far = cfg.get("margins", [None, None])
    far_extra = 40
    if len(far) >= 2 and far[0] is not None:
        far_extra = int(far[1] - far[0])
    model = BlindSpotForecaster(
        cin=cfg.get("cin", 3), width=cfg.get("width", WIDTH), bottleneck=cfg.get("bottleneck", BOTTLENECK),
        blind_t=cfg.get("blind_t", BLIND_T), blind_x=cfg.get("blind_x", BLIND_X),
        k_time=cfg.get("k_time", 5), k_space=cfg.get("k_space", 7), far_extra=far_extra)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model
