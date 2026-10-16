#!/usr/bin/env python3
"""Рисунки docs/figures/*.png для пояснительной записки. Без обучения.

1. 3–4 окна test-записей синтетики v8 (разные шаблоны F1–F6, со слабыми
   событиями ≤2 °C): ΔT/σ, истинная маска, карта U-Net с контуром порога
   FAR=10⁻³, маска эталона |Z|≥3.5.
2. Пиксельные кривые FAR→recall на всём test: каждый метод из
   artifacts/pred/v8/<method>/test и эталон по |base_z|.
3. Матрицы ошибок этапа 2 (test) из artifacts/eval/stage2_v8.json.

Подписи на русском. Ссылки пишутся в docs/NOTE.md между маркерами
<!-- FIGURES:stage1 --> и <!-- FIGURES:stage2 -->.
Один процесс, потоки BLAS/OpenMP = 1.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import deque
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

FAMILY_PRIORITY = ["F1", "F2", "F5", "F6", "F4", "F3"]
N_EXAMPLES = 4
PAD_M = 80.0
PAD_S = 2 * 3600.0
# Рамка должна занимать заметную долю кадра. Иначе след короче метра
# (типичный tracing на F6) теряется на фоне 80 м × 2 ч.
MIN_FRAC_X = 0.06
MIN_FRAC_T = 0.12
BASE_Z = 3.5
PROB_METHODS = {"unet", "tcn", "dilated", "pixel_lgbm"}
METHOD_RU = {
    "unet": "U-Net",
    "tcn": "TCN",
    "dilated": "Dilated FCN",
    "matched_filter": "согласованный фильтр",
    "pixel_lgbm": "LightGBM по пикселям",
    "base": "эталон |Z|",
}
METHOD_ORDER = ["unet", "tcn", "dilated", "matched_filter", "pixel_lgbm", "base"]
CLASS_RU = {
    "Local_Heating": "лок. нагрев",
    "Extended_Heating": "протяж. нагрев",
    "Contact_Trace": "трассировка",
    "Local_Cooling": "лок. охлаждение",
    "Extended_Cooling": "протяж. охлаждение",
    "HVAC_Diurnal": "суточный ход",
    "Sensor_Artifact": "артефакт",
}
MODEL_RU = {
    "cnn_eventnet": "EventNet",
    "cnn_image_only": "CNN только по кропу",
    "lgbm_features": "LightGBM",
    "ensemble_cnn_lgbm": "ансамбль",
    "rules_haklolarp": "правила",
}
MODEL_ORDER = ["cnn_eventnet", "cnn_image_only", "lgbm_features", "ensemble_cnn_lgbm", "rules_haklolarp"]
META_CANDIDATES = [
    Path("data/synth_meta/v8"),
    Path("/workspace/synth_v8/d"),
]
STAGE1_BLOCK = re.compile(r"<!-- FIGURES:stage1 -->.*?<!-- /FIGURES:stage1 -->", re.S)
STAGE2_BLOCK = re.compile(r"<!-- FIGURES:stage2 -->.*?<!-- /FIGURES:stage2 -->", re.S)


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def family_of(template: str) -> str:
    text = str(template)
    if text.startswith("F1"):
        return "F1"
    return text


def discover_meta(explicit: str | None) -> Path | None:
    if explicit:
        root = Path(explicit)
        if (root / "sequences.csv").is_file() and (root / "labels.csv").is_file():
            return root
        parent = root.parent
        if (parent / "sequences.csv").is_file():
            return parent
        return None
    for root in META_CANDIDATES:
        if (root / "sequences.csv").is_file() and (root / "labels.csv").is_file():
            return root
    return None


def to_unix(series: pd.Series) -> np.ndarray:
    ts = pd.to_datetime(series, utc=True, format="mixed")
    origin = pd.Timestamp("1970-01-01", tz="UTC")
    return (ts - origin).dt.total_seconds().to_numpy(dtype=np.float64)


def list_methods(pred_root: Path) -> list[str]:
    found = []
    if not pred_root.is_dir():
        return found
    for path in sorted(pred_root.iterdir()):
        if path.is_dir() and (path / "test").is_dir() and any((path / "test").glob("*.npz")):
            found.append(path.name)
    return sorted(found, key=lambda name: (METHOD_ORDER.index(name) if name in METHOD_ORDER else 99, name))


def load_thresholds(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"пороги не прочитаны: {exc}", flush=True)
        return {}
    thr = data.get("thresholds")
    out = {"far_ref": data.get("far_ref"), "raw": thr if isinstance(thr, dict) else {}}
    return out


def unet_far_1e3(thr: dict) -> float | None:
    raw = thr.get("raw") or {}
    node = raw.get("unet")
    if isinstance(node, dict) and "far_1e-3" in node:
        try:
            return float(node["far_1e-3"])
        except (TypeError, ValueError):
            return None
    return None


def method_thresholds(thr: dict, method: str) -> list[tuple[str, float]]:
    raw = thr.get("raw") or {}
    key = "base_median_x3_spatial_z" if method == "base" else method
    node = raw.get(key)
    if not isinstance(node, dict):
        return []
    out = []
    for op in ("far_ref", "far_1e-3", "far_1e-4"):
        if op in node:
            try:
                out.append((op, float(node[op])))
            except (TypeError, ValueError):
                continue
    return out


def select_examples(cache: Path, meta_root: Path | None, pred: Path, k: int) -> list[dict]:
    files = sorted(cache.glob("*.npz"))
    if not files:
        return []
    templates = {}
    weak = {}
    if meta_root is not None:
        seq = pd.read_csv(meta_root / "sequences.csv")
        templates = dict(zip(seq.seq_id.astype(str), seq.template.astype(str)))
        lab = pd.read_csv(meta_root / "labels.csv")
        ev = lab[lab.kind.astype(str) == "event"].copy()
        ev["amp"] = pd.to_numeric(ev.amp_obs_C, errors="coerce")
        weak_rows = ev[ev.amp <= 2]
        for sid, g in weak_rows.groupby(weak_rows.seq_id.astype(str)):
            weak[str(sid)] = g
    records = []
    for path in files:
        sid = path.stem
        if not (pred / "unet" / "test" / path.name).is_file():
            continue
        with np.load(path, mmap_mode="r") as z:
            n_ev = int(np.count_nonzero(z["ev"] > 0))
            n_pix = int(np.prod(z["ev"].shape))
        w = weak.get(sid)
        n_weak = 0 if w is None else int(len(w))
        min_amp = float(w.amp.min()) if n_weak else 99.0
        template = templates.get(sid, "неизвестен")
        records.append(dict(seq=sid, path=path, template=template, family=family_of(template),
                            n_ev=n_ev, n_pix=n_pix, n_weak=n_weak, min_amp=min_amp, weak=w))
    chosen = []
    used = set()
    for fam in FAMILY_PRIORITY:
        cands = [r for r in records if r["family"] == fam and r["n_ev"] > 0 and r["n_weak"] > 0 and r["seq"] not in used]
        if not cands:
            continue
        cands.sort(key=lambda r: (r["min_amp"], -min(r["n_ev"], 80000), r["n_pix"], r["seq"]))
        chosen.append(cands[0])
        used.add(cands[0]["seq"])
        if len(chosen) >= k:
            return chosen
    for fam in FAMILY_PRIORITY:
        if any(r["family"] == fam for r in chosen):
            continue
        cands = [r for r in records if r["family"] == fam and r["n_ev"] > 0 and r["seq"] not in used]
        if not cands:
            continue
        cands.sort(key=lambda r: (-min(r["n_ev"], 80000), r["n_pix"], r["seq"]))
        chosen.append(cands[0])
        used.add(cands[0]["seq"])
        if len(chosen) >= k:
            break
    return chosen


def weak_boxes(frame: pd.DataFrame | None) -> list[dict]:
    if frame is None or len(frame) == 0:
        return []
    out = []
    order = frame.sort_values("amp")
    ts = to_unix(order.t_start)
    te = to_unix(order.t_end)
    for i, row in enumerate(order.itertuples(index=False)):
        mask = int(row.mask_value) if hasattr(row, "mask_value") else 0
        out.append(dict(xa=float(row.dist_start_m), xb=float(row.dist_end_m),
                        ta=float(ts[i]), tb=float(te[i]), amp=float(row.amp),
                        label=str(row.label), mask=mask))
    return out


def slice_covering(axis: np.ndarray, a: float, b: float, pad: float) -> slice:
    lo, hi = min(a, b) - pad, max(a, b) + pad
    idx = np.flatnonzero((axis >= lo) & (axis <= hi))
    if idx.size == 0:
        j = int(np.argmin(np.abs(axis - 0.5 * (a + b))))
        return slice(j, j + 1)
    return slice(int(idx[0]), int(idx[-1]) + 1)


def is_substantial(box: dict) -> bool:
    """Протяжённое слабое событие: его имеет смысл показывать без сильного зума."""
    return box["width_m"] >= 4.0 or box["dur_min"] >= 5.0 or box["n_px"] >= 30


def attach_raster(x, t, ev, boxes: list[dict]):
    """Границы рамки — клетки маски того же mask_value, что и бокс разметки.

    Ближайшая передискретизация на сетку 0.25 м размазывает короткий tracing
    на несколько клеток шире подписи. Рамка обводит эти клетки, с запасом 2 м
    и 10 мин, чтобы не склеить соседнее событие того же класса.
    """
    dx = float(np.median(np.diff(x))) if x.size > 1 else 0.25
    dt = float(np.median(np.diff(t))) if t.size > 1 else 120.0
    for box in boxes:
        box["n_px"] = 0
        box["width_m"] = abs(box["xb"] - box["xa"])
        box["dur_min"] = max(abs(box["tb"] - box["ta"]), dt) / 60.0
        box["xa_draw"], box["xb_draw"] = box["xa"], box["xb"]
        box["ta_draw"], box["tb_draw"] = box["ta"], box["tb"]
        mv = int(box.get("mask", 0))
        ix = np.flatnonzero((x >= box["xa"] - 2.0) & (x <= box["xb"] + 2.0))
        it = np.flatnonzero((t >= box["ta"] - 600.0) & (t <= box["tb"] + 600.0))
        if ix.size == 0 or it.size == 0 or mv == 0:
            continue
        crop = np.asarray(ev[np.ix_(it, ix)])
        xs = x[ix][None, :]
        ts = t[it][:, None]
        seed = ((crop == mv) & (xs >= box["xa"] - dx) & (xs <= box["xb"] + dx)
                & (ts >= box["ta"] - dt) & (ts <= box["tb"] + dt))
        if not seed.any():
            continue
        vis = np.zeros(crop.shape, dtype=bool)
        qi, qj = np.where(seed)
        q = deque(zip(qi.tolist(), qj.tolist()))
        vis[qi, qj] = True
        h, w = crop.shape
        while q:
            i, j = q.popleft()
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ni, nj = i + di, j + dj
                if ni < 0 or nj < 0 or ni >= h or nj >= w or vis[ni, nj]:
                    continue
                if crop[ni, nj] != mv:
                    continue
                vis[ni, nj] = True
                q.append((ni, nj))
        ii, jj = np.where(vis)
        cells_x = x[ix[jj]]
        cells_t = t[it[ii]]
        box["n_px"] = int(ii.size)
        box["xa_draw"] = float(cells_x.min() - dx / 2)
        box["xb_draw"] = float(cells_x.max() + dx / 2)
        box["ta_draw"] = float(cells_t.min() - dt / 2)
        box["tb_draw"] = float(cells_t.max() + dt / 2)
        box["width_m"] = box["xb_draw"] - box["xa_draw"]
        box["dur_min"] = (box["tb_draw"] - box["ta_draw"]) / 60.0


def choose_focus(boxes: list[dict], x, t, ev) -> dict | None:
    """Самое слабое событие, которое есть на маске.

    Если рядом есть протяжённое слабое событие (≥4 м, ≥5 мин или ≥30 клеток),
    рамка ставится на самое слабое из протяжённых.
    """
    attach_raster(x, t, ev, boxes)
    hit = [b for b in boxes if b["n_px"] >= 1]
    visible = [b for b in hit if is_substantial(b)]
    pool = visible or hit
    if not pool:
        return None
    return min(pool, key=lambda b: (b["amp"], -b["n_px"], b["xa"]))


def pads_for(box: dict) -> tuple[float, float]:
    width = max(float(box.get("width_m", 0.0)), 0.25)
    dur = max(float(box.get("dur_min", 0.0)) * 60.0, 120.0)
    pad_m, pad_s = PAD_M, PAD_S
    if width / (width + 2.0 * pad_m) < MIN_FRAC_X:
        pad_m = max(1.0, width * (1.0 / MIN_FRAC_X - 1.0) / 2.0)
    if dur / (dur + 2.0 * pad_s) < MIN_FRAC_T:
        pad_s = max(180.0, dur * (1.0 / MIN_FRAC_T - 1.0) / 2.0)
    return pad_m, pad_s


def choose_window(x, t, focus: dict | None, pad_m: float, pad_s: float) -> tuple[slice, slice]:
    if focus is None:
        return slice(None), slice(None)
    st = slice_covering(t, focus["ta_draw"], focus["tb_draw"], pad_s)
    sx = slice_covering(x, focus["xa_draw"], focus["xb_draw"], pad_m)
    return st, sx


def hours_of(t: np.ndarray) -> np.ndarray:
    return (t - t[0]) / 3600.0


def draw_boxes(ax, boxes, t0, hours_span, with_label=False):
    drawn = False
    for box in boxes:
        if box.get("n_px", 1) <= 0:
            continue
        y0 = (box["ta_draw"] - t0) / 3600.0
        height = max((box["tb_draw"] - box["ta_draw"]) / 3600.0, 120.0 / 3600.0)
        if y0 > hours_span[1] + 0.05 or y0 + height < hours_span[0] - 0.05:
            continue
        kw = dict(fill=False, ec="#39d353", lw=1.8, zorder=5)
        if with_label and not drawn:
            kw["label"] = "слабое ≤2 °C"
        ax.add_patch(Rectangle((box["xa_draw"], y0), max(box["xb_draw"] - box["xa_draw"], 0.25), height, **kw))
        drawn = True
    return drawn


def plot_example(rec: dict, pred: Path, thr: float | None, out_dir: Path) -> dict:
    z = np.load(rec["path"])
    x = z["x"].astype(np.float64)
    t = z["t"].astype(np.float64)
    valid = z["valid"].astype(bool)
    ev = z["ev"]
    base = np.abs(z["base_z"].astype(np.float32))
    dT = z["X"][0].astype(np.float32)
    score_path = pred / "unet" / "test" / rec["path"].name
    score = np.load(score_path)["s"].astype(np.float32) if score_path.is_file() else None
    boxes = weak_boxes(rec["weak"])
    focus = choose_focus(boxes, x, t, ev)
    pad_m, pad_s = pads_for(focus) if focus is not None else (PAD_M, PAD_S)
    st, sx = choose_window(x, t, focus, pad_m, pad_s)
    xw, tw = x[sx], t[st]
    hours_full0 = float(t[0])
    hw = (tw - hours_full0) / 3600.0
    dTw = np.where(valid[st, sx], dT[st, sx], np.nan)
    evw = np.where(valid[st, sx], (ev[st, sx] > 0).astype(np.float32), np.nan)
    basew = np.where(valid[st, sx], (base[st, sx] >= BASE_Z).astype(np.float32), np.nan)
    if score is not None:
        scw = np.where(valid[st, sx], score[st, sx], np.nan)
    else:
        scw = np.full(dTw.shape, np.nan, np.float32)
    finite = dTw[np.isfinite(dTw)]
    if finite.size:
        scale = float(np.clip(np.percentile(np.abs(finite), 99), 2.0, 6.0))
    else:
        scale = 3.0
    dx = float(np.median(np.diff(xw))) if xw.size > 1 else 0.25
    dt_h = (float(np.median(np.diff(tw))) if tw.size > 1 else 120.0) / 3600.0
    # Границы пикселя, а не координаты центров: рамка совпадает с клетками imshow.
    ext = [float(xw[0] - dx / 2), float(xw[-1] + dx / 2),
           float(hw[-1] + dt_h / 2), float(hw[0] - dt_h / 2)]
    fig, axes = plt.subplots(4, 1, figsize=(11.5, 9.2), sharex=True, sharey=True)
    cmap_dt = plt.get_cmap("RdBu_r").copy()
    cmap_dt.set_bad("#c8c8c8")
    cmap_ev = ListedColormap(["#f4f4f4", "#b2182b"])
    cmap_ev.set_bad("#c8c8c8")
    cmap_base = ListedColormap(["#f4f4f4", "#2166ac"])
    cmap_base.set_bad("#c8c8c8")
    cmap_s = plt.get_cmap("magma").copy()
    cmap_s.set_bad("#c8c8c8")
    im0 = axes[0].imshow(dTw, extent=ext, origin="upper", aspect="auto", interpolation="nearest",
                         cmap=cmap_dt, vmin=-scale, vmax=scale)
    im1 = axes[1].imshow(evw, extent=ext, origin="upper", aspect="auto", interpolation="nearest",
                         cmap=cmap_ev, vmin=0, vmax=1)
    im2 = axes[2].imshow(scw, extent=ext, origin="upper", aspect="auto", interpolation="nearest",
                         cmap=cmap_s, vmin=0, vmax=1)
    im3 = axes[3].imshow(basew, extent=ext, origin="upper", aspect="auto", interpolation="nearest",
                         cmap=cmap_base, vmin=0, vmax=1)
    if thr is not None and np.isfinite(scw).any():
        cont = np.where(np.isfinite(scw), scw, 0.0)
        axes[2].contour(xw, hw, cont, levels=[thr], colors="#7FFFD4", linewidths=0.9)
        axes[2].legend(handles=[Line2D([0], [0], color="#7FFFD4", lw=1.4, label=f"порог FAR = 10⁻³ ({thr:.3f})")],
                       loc="upper right", fontsize=8, frameon=True, framealpha=0.92)
    span = (float(hw[0]), float(hw[-1]))
    has_box = False
    for i, ax in enumerate(axes):
        has_box = draw_boxes(ax, boxes, hours_full0, span, with_label=(i == 0)) or has_box
    if has_box:
        axes[0].legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.92)
    if thr is not None:
        finite_s = scw[np.isfinite(scw)]
        if finite_s.size and float(finite_s.max()) < thr:
            axes[2].text(0.01, 0.06, "в этом окне порог не достигнут", transform=axes[2].transAxes,
                         fontsize=8, color="white",
                         bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.55, ec="none"))
    for ax in axes:
        ax.set_ylabel("время, ч")
        ax.set_ylim(ext[2], ext[3])
        ax.set_xlim(ext[0], ext[1])
    axes[3].set_xlabel("x, м")
    axes[0].set_title("ΔT по времени / σ (канал 0). Серое — нет измерения")
    axes[1].set_title("Истинная маска события (ev > 0)")
    axes[2].set_title("U-Net, вероятность аномалии")
    axes[3].set_title(f"Эталон: |Z| ≥ {BASE_Z:.1f}")
    fig.colorbar(im0, ax=axes[0], fraction=0.02, pad=0.01).set_label("σ")
    fig.colorbar(im2, ax=axes[2], fraction=0.02, pad=0.01).set_label("P")
    if focus is not None:
        title = (f"{rec['seq']}, шаблон {rec['template']}. Рамка: {focus['label']}, "
                 f"{focus['amp']:.2f} °C, {focus['width_m']:.2f} м × {focus['dur_min']:.0f} мин")
    else:
        title = f"{rec['seq']}, шаблон {rec['template']}"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    name = f"example_{rec['seq']}_{rec['template']}.png"
    fig.savefig(out_dir / name, dpi=120)
    plt.close(fig)
    full = (st == slice(None) or (st.start in (0, None) and st.stop in (len(t), None))) and (
        sx == slice(None) or (sx.start in (0, None) and sx.stop in (len(x), None)))
    n_weak = rec["n_weak"]
    n_smaller = 0
    if focus is not None and is_substantial(focus):
        n_smaller = sum(1 for b in boxes if b["amp"] < focus["amp"] and b.get("n_px", 0) >= 1 and not is_substantial(b))
    info = dict(file=name, seq=rec["seq"], template=rec["template"], family=rec["family"],
                n_weak=n_weak, min_amp=None if not n_weak else rec["min_amp"],
                x0=float(xw[0]), x1=float(xw[-1]), h0=float(hw[0]), h1=float(hw[-1]),
                full=bool(full), length_m=float(x[-1] - x[0]), duration_h=float((t[-1] - t[0]) / 3600.0),
                focus_amp=None if focus is None else focus["amp"],
                focus_label=None if focus is None else focus["label"],
                focus_width_m=None if focus is None else focus["width_m"],
                focus_dur_min=None if focus is None else focus["dur_min"],
                focus_n_px=None if focus is None else focus["n_px"],
                n_smaller=n_smaller)
    print(f"пример {name} окно x {info['x0']:.1f}–{info['x1']:.1f} м, t {info['h0']:.2f}–{info['h1']:.2f} ч, "
          f"рамка {info['focus_label']} {info['focus_amp']} °C, {info['focus_n_px']} кл.", flush=True)
    return info


def edges_for(method: str, probe: np.ndarray | None) -> np.ndarray:
    if method in PROB_METHODS:
        return np.linspace(0.0, 1.0, 2001)
    if probe is not None and probe.size and float(np.nanmin(probe)) >= -0.05 and float(np.nanmax(probe)) <= 1.05:
        return np.linspace(0.0, 1.0, 2001)
    return np.linspace(0.0, 40.0, 4001)


def accumulate(hist: np.ndarray, ends: np.ndarray, values: np.ndarray, edges: np.ndarray):
    """ends: [under, over]. Гистограмма значений внутри edges."""
    if values.size == 0:
        return
    under = int(np.count_nonzero(values < edges[0]))
    over = int(np.count_nonzero(values > edges[-1]))
    ends[0] += under
    ends[1] += over
    mid = values[(values >= edges[0]) & (values <= edges[-1])]
    if mid.size:
        hist += np.histogram(mid, bins=edges)[0]


def curve_from_hist(hist: np.ndarray, ends: np.ndarray, edges: np.ndarray, total: int):
    if total <= 0:
        return np.array([]), np.array([])
    # доля score >= edges[i]
    tail = np.cumsum(hist[::-1])[::-1] + ends[1]
    frac = tail / total
    return edges[:-1], frac


def plot_curves(cache: Path, pred: Path, methods: list[str], thr: dict, out_dir: Path) -> Path | None:
    files = sorted(cache.glob("*.npz"))
    if not files:
        print("кривые: нет test-кэша", flush=True)
        return None
    names = list(methods) + ["base"]
    probe = {}
    sample = files[0]
    for method in methods:
        sp = pred / method / "test" / sample.name
        if sp.is_file():
            probe[method] = np.load(sp)["s"]
    edges = {m: edges_for(m, probe.get(m)) for m in methods}
    edges["base"] = np.linspace(0.0, 40.0, 4001)
    hist_bg = {m: np.zeros(len(edges[m]) - 1, np.int64) for m in names}
    hist_ev = {m: np.zeros(len(edges[m]) - 1, np.int64) for m in names}
    ends_bg = {m: np.zeros(2, np.int64) for m in names}
    ends_ev = {m: np.zeros(2, np.int64) for m in names}
    total_bg = {m: 0 for m in names}
    total_ev = {m: 0 for m in names}
    ops = {m: method_thresholds(thr, m) for m in names}
    exact = {m: {op: [0, 0] for op, _ in pairs} for m, pairs in ops.items()}
    used_nu = True
    for i, path in enumerate(files, 1):
        z = np.load(path)
        valid = np.asarray(z["valid"], dtype=bool)
        ev = np.asarray(z["ev"])
        event = valid & (ev > 0)
        if "nu" in z.files:
            bg = valid & (ev == 0) & (np.asarray(z["nu"]) == 0)
        else:
            used_nu = False
            bg = valid & (ev == 0)
        bz = np.abs(np.asarray(z["base_z"], dtype=np.float32))
        accumulate(hist_bg["base"], ends_bg["base"], bz[bg], edges["base"])
        accumulate(hist_ev["base"], ends_ev["base"], bz[event], edges["base"])
        total_bg["base"] += int(bg.sum())
        total_ev["base"] += int(event.sum())
        for op, level in ops.get("base", []):
            exact["base"][op][0] += int(np.count_nonzero(bz[bg] >= level))
            exact["base"][op][1] += int(np.count_nonzero(bz[event] >= level))
        del bz
        for method in methods:
            sp = pred / method / "test" / path.name
            if not sp.is_file():
                continue
            s = np.asarray(np.load(sp)["s"], dtype=np.float32)
            if s.shape != valid.shape:
                print(f"пропуск {method} {path.name}: форма {s.shape} != {valid.shape}", flush=True)
                continue
            accumulate(hist_bg[method], ends_bg[method], s[bg], edges[method])
            accumulate(hist_ev[method], ends_ev[method], s[event], edges[method])
            total_bg[method] += int(bg.sum())
            total_ev[method] += int(event.sum())
            for op, level in ops.get(method, []):
                exact[method][op][0] += int(np.count_nonzero(s[bg] >= level))
                exact[method][op][1] += int(np.count_nonzero(s[event] >= level))
            del s
        if i % 10 == 0 or i == len(files):
            print(f"кривые {i}/{len(files)}", flush=True)
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    colors = {
        "unet": "#d95f02",
        "tcn": "#1b9e77",
        "dilated": "#7570b3",
        "matched_filter": "#e7298a",
        "pixel_lgbm": "#66a61e",
        "base": "#000000",
    }
    for method in names:
        if total_ev[method] == 0 or total_bg[method] == 0:
            continue
        thr_x, far = curve_from_hist(hist_bg[method], ends_bg[method], edges[method], total_bg[method])
        _, rec = curve_from_hist(hist_ev[method], ends_ev[method], edges[method], total_ev[method])
        # одинаковая сетка порогов: обе гистограммы на одних edges
        order = np.argsort(far)
        far_s, rec_s = far[order], rec[order]
        # монотонная оболочка по recall при росте FAR (порог снижается)
        keep = np.ones(len(far_s), dtype=bool)
        if len(far_s) > 400:
            keep[:: max(len(far_s) // 400, 1)] = True
            keep[-1] = True
        ax.plot(np.clip(far_s[keep], 1e-6, 1), rec_s[keep], color=colors.get(method, None),
                lw=1.8 if method != "base" else 1.4, label=METHOD_RU.get(method, method))
        if ends_bg[method][1]:
            print(f"предупреждение: {method} имеет {int(ends_bg[method][1])} фоновых значений выше сетки гистограммы", flush=True)
    far_ref = thr.get("far_ref")
    if isinstance(far_ref, (int, float)) and far_ref > 0:
        ax.axvline(float(far_ref), color="0.25", ls="-.", lw=0.8, label=f"FAR эталона {float(far_ref):.3f}")
    ax.axvline(1e-3, color="0.45", ls="--", lw=0.8, label="FAR = 10⁻³")
    ax.axvline(1e-4, color="0.65", ls=":", lw=0.8, label="FAR = 10⁻⁴")
    ax.set_xscale("log")
    ax.set_xlim(1e-5, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("FAR на фоне test (валидные клетки без события и без помехи)")
    ax.set_ylabel("Recall по пикселям события")
    nu_txt = "Помеха исключена из фона." if used_nu else "В кэше нет маски помех, фон — все валидные клетки без события."
    ax.set_title("Test синтетики v8: пиксельные FAR и recall")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=8, loc="lower right")
    fig.text(0.01, 0.01, nu_txt, fontsize=8, color="0.3")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    dest = out_dir / "far_recall_test.png"
    fig.savefig(dest, dpi=140)
    plt.close(fig)
    print("кривая", dest.name, flush=True)
    for method in names:
        if total_bg[method] == 0:
            continue
        bits = []
        for op, _level in ops.get(method, []):
            hit_bg, hit_ev = exact[method][op]
            bits.append(f"{op}: FAR {hit_bg / total_bg[method]:.4g}, recall {hit_ev / max(total_ev[method], 1):.3f}")
        if bits:
            print(method, "; ".join(bits), flush=True)
    return dest


def plot_confusion(path: Path, out_dir: Path) -> Path | None:
    if not path.is_file():
        print("матрица: нет", path, flush=True)
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    test = data.get("test") if isinstance(data.get("test"), dict) else {}
    classes = data.get("classes") if isinstance(data.get("classes"), list) else []
    names = [m for m in MODEL_ORDER if m in test and isinstance(test[m], dict) and test[m].get("confusion")]
    names += [m for m in test if m not in names and isinstance(test[m], dict) and test[m].get("confusion")]
    if not names:
        print("матрица: в test нет confusion", flush=True)
        return None
    n = len(names)
    cols = 2 if n > 1 else 1
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 5.2 * rows), squeeze=False)
    for i, name in enumerate(names):
        ax = axes[i // cols][i % cols]
        cm = np.asarray(test[name]["confusion"], dtype=float)
        labels = classes[: cm.shape[0]] if classes else [str(k) for k in range(cm.shape[0])]
        if len(labels) < cm.shape[0]:
            labels = labels + [f"класс {k}" for k in range(len(labels), cm.shape[0])]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(cm.shape[1]))
        ax.set_yticks(range(cm.shape[0]))
        nice = [CLASS_RU.get(lb, lb) for lb in labels]
        ax.set_xticklabels(nice, rotation=35, ha="right", fontsize=8)
        ax.set_yticklabels(nice, fontsize=8)
        vmax = cm.max() if cm.size else 1
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                val = cm[r, c]
                ax.text(c, r, f"{int(round(val))}", ha="center", va="center", fontsize=7,
                        color="white" if val > 0.55 * vmax else "black")
        f1 = test[name].get("macro_f1")
        f1_txt = f", macro-F1 {float(f1):.3f}" if isinstance(f1, (int, float)) else ""
        ax.set_title(MODEL_RU.get(name, name) + f1_txt)
        ax.set_xlabel("предсказанный класс")
        ax.set_ylabel("истинный класс")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle("Этап 2, отложенный test. Числа — объекты.", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    dest = out_dir / "stage2_confusion_test.png"
    fig.savefig(dest, dpi=140)
    plt.close(fig)
    print("матрица", dest.name, flush=True)
    return dest


def md_examples(infos: list[dict], thr: float | None) -> str:
    if not infos and thr is None:
        return "Рисунки этапа 1 не построены: нет test-кэша или карт U-Net."
    thr_txt = f"{thr:.3f}" if thr is not None else "н/д"
    n_txt = {1: "Одна запись", 2: "Две записи", 3: "Три записи", 4: "Четыре записи"}.get(
        len(infos), f"{len(infos)} записей")
    lines = [
        "### Примеры test и кривая FAR–recall",
        "",
        f"{n_txt} отложенного test синтетики v8, разные шаблоны. "
        "Зелёная рамка — слабое событие с амплитудой не выше 2 °C. "
        "Если в записи есть слабое событие протяжённее 4 м или длиннее 5 минут, рамка стоит на самом слабом из таких. "
        "Иначе рамка — самое слабое событие, которое есть на маске. "
        "Запас вокруг бокса — 80 м по кабелю и 2 часа по времени. "
        "Если при этом рамка занимает меньше 6% ширины кадра или меньше 12% его высоты, запас уменьшается. "
        "Серое поле — клетка без измерения. "
        f"Бирюзовый контур на карте U-Net — порог точки FAR = 10⁻³, значение {thr_txt} "
        "из `artifacts/eval/stage1_v8_early.json` (`thresholds.unet`).",
        "",
    ]
    for info in infos:
        if info.get("focus_amp") is None:
            cap = (f"{info['seq']}, шаблон {info['template']}. "
                   f"Окно {info['x0']:.0f}–{info['x1']:.0f} м, {info['h0']:.2f}–{info['h1']:.2f} ч.")
        else:
            cap = (f"{info['seq']}, шаблон {info['template']}: рамка — {info['focus_label']}, "
                   f"{info['focus_amp']:.2f} °C, {info['focus_width_m']:.2f} м × {info['focus_dur_min']:.1f} мин, "
                   f"{info['focus_n_px']} клеток маски. "
                   f"Окно {info['x0']:.1f}–{info['x1']:.1f} м, {info['h0']:.2f}–{info['h1']:.2f} ч от начала записи "
                   f"(вся запись {info['length_m']:.0f} м и {info['duration_h']:.1f} ч). "
                   f"Слабых событий ≤2 °C в записи: {info['n_weak']}.")
            if info.get("n_smaller"):
                cap += (f" В записи есть ещё {info['n_smaller']} более слабых следов короче 4 м; "
                        "рамка поставлена на протяжённое событие.")
        lines.append(f"![{cap}](figures/{info['file']})")
        lines.append("")
        lines.append(cap)
        lines.append("")
    lines.append(
        "Кривые считаются по всему test, не по окнам. По горизонтали — доля фоновых пикселей выше порога "
        "(валидная клетка, нет события и нет помехи). По вертикали — доля пикселей события выше того же порога. "
        "Линии FAR эталона, 10⁻³ и 10⁻⁴ нанесены для чтения, порог каждой модели на test не подгонялся."
    )
    lines.append("")
    lines.append("![Пиксельные FAR и recall на test синтетики v8](figures/far_recall_test.png)")
    lines.append("")
    return "\n".join(lines).rstrip()


def md_stage2(built: bool) -> str:
    if not built:
        return "Рисунок матрицы ошибок не построен: в `artifacts/eval/stage2_v8.json` нет блока test."
    return "\n".join([
        "![Матрицы ошибок этапа 2 на test](figures/stage2_confusion_test.png)",
        "",
        "На рисунке — отложенный test. Строка — истинный класс, столбец — предсказанный, в клетке — число объектов. "
        "Те же матрицы и перекрёстная проверка — в `docs/RESULTS.md`.",
    ])


def update_note(note: Path, stage1_md: str, stage2_md: str):
    if not note.is_file():
        print("нет записки, ссылки не встроены", flush=True)
        return
    text = note.read_text(encoding="utf-8")
    changed = False
    if STAGE1_BLOCK.search(text):
        text2 = STAGE1_BLOCK.sub("<!-- FIGURES:stage1 -->\n" + stage1_md + "\n<!-- /FIGURES:stage1 -->", text, count=1)
        changed = changed or text2 != text
        text = text2
    else:
        print("в записке нет маркера FIGURES:stage1", flush=True)
    if STAGE2_BLOCK.search(text):
        text2 = STAGE2_BLOCK.sub("<!-- FIGURES:stage2 -->\n" + stage2_md + "\n<!-- /FIGURES:stage2 -->", text, count=1)
        changed = changed or text2 != text
        text = text2
    else:
        print("в записке нет маркера FIGURES:stage2", flush=True)
    if changed:
        note.write_text(text, encoding="utf-8")
        print("обновлены ссылки в", note, flush=True)


def parse_args(argv):
    p = argparse.ArgumentParser(description="Рисунки этапа 1 и матрица этапа 2")
    p.add_argument("--cache", default="artifacts/cache/v8/test")
    p.add_argument("--pred", default="artifacts/pred/v8")
    p.add_argument("--stage1", default="artifacts/eval/stage1_v8_early.json")
    p.add_argument("--stage2", default="artifacts/eval/stage2_v8.json")
    p.add_argument("--meta", default=None, help="каталог с sequences.csv и labels.csv")
    p.add_argument("--out", default="docs/figures")
    p.add_argument("--note", default="docs/NOTE.md")
    p.add_argument("--no-note", action="store_true")
    p.add_argument("--examples", type=int, default=N_EXAMPLES)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv or sys.argv[1:])
    cache = resolve(args.cache)
    pred = resolve(args.pred)
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = discover_meta(args.meta)
    print("метаданные", meta if meta else "не найдены", flush=True)
    thr = load_thresholds(resolve(args.stage1))
    level = unet_far_1e3(thr)
    print("порог U-Net FAR 1e-3", level, flush=True)
    infos = []
    if meta is None:
        print("без labels.csv окна слабых событий не выбрать", flush=True)
    else:
        chosen = select_examples(cache, meta, pred, max(1, args.examples))
        print("выбраны", [(r["seq"], r["template"], r["n_weak"], round(r["min_amp"], 3)) for r in chosen], flush=True)
        for rec in chosen:
            infos.append(plot_example(rec, pred, level, out))
    methods = list_methods(pred)
    print("методы", methods, flush=True)
    plot_curves(cache, pred, methods, thr, out)
    cm_path = plot_confusion(resolve(args.stage2), out)
    if not args.no_note:
        update_note(resolve(args.note), md_examples(infos, level), md_stage2(cm_path is not None))
    return 0


if __name__ == "__main__":
    sys.exit(main())
