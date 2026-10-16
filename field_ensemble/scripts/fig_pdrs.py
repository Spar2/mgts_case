#!/usr/bin/env python3
"""Один пример v8 test: вероятность ансамбля, зоны CC, зоны PDRS, истинная маска.

Параметры PDRS — из artifacts/eval/pdrs_best_params.json (выбор только по v8 val).
Порог CC — far 1e-3 выпуска ens_u2ft_tcn_lgbm. Реальные файлы не читаются.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dts.pdrs import extract_zones_cc, extract_zones_pdrs

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

HOT = (1, 2, 3, 4)
COLD = (5, 6)
CC_THR = 0.8608087301264272  # far 1e-3, models_release/stage1_thresholds.json
PRED = ROOT / "artifacts/pred/v8/ens_u2ft_tcn_lgbm/test"
CACHE = ROOT / "artifacts/cache/v8/test"
OUT = ROOT / "docs/figures/pdrs_example.png"
PARAMS = ROOT / "artifacts/eval/pdrs_best_params.json"


def _components(mask, min_px=400):
    lab, n = ndi.label(mask, structure=np.ones((3, 3)))
    out = []
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None:
            continue
        npx = int((lab[sl] == i).sum())
        if npx >= min_px:
            out.append((npx, sl))
    return out


def _union(sl_a, sl_b, shape, pad_t=25, pad_x=40):
    r0 = max(0, min(sl_a[0].start, sl_b[0].start) - pad_t)
    r1 = min(shape[0], max(sl_a[0].stop, sl_b[0].stop) + pad_t)
    c0 = max(0, min(sl_a[1].start, sl_b[1].start) - pad_x)
    c1 = min(shape[1], max(sl_a[1].stop, sl_b[1].stop) + pad_x)
    return slice(r0, r1), slice(c0, c1)


def _pick_sequence(params_floor):
    """Окно одной записи v8 test: соседние hot и cold, оба видны на карте s."""
    best = None
    for path in sorted(CACHE.glob("*.npz")):
        pred_path = PRED / path.name
        if not pred_path.is_file():
            continue
        with np.load(path) as blob:
            ev = blob["ev"]
            valid = blob["valid"]
        score = np.load(pred_path)["s"]
        hot = np.isin(ev, HOT) & valid
        cold = np.isin(ev, COLD) & valid
        visible = score >= min(0.5, params_floor)
        hots = _components(hot & visible)
        colds = _components(cold & visible)
        for nh, sh in hots:
            for nc, sc in colds:
                st, sx = _union(sh, sc, ev.shape)
                area = (st.stop - st.start) * (sx.stop - sx.start)
                # близкая пара заполняет окно, а не растягивает его пустотой
                rank = min(nh, nc) / area
                if best is None or rank > best[0]:
                    best = (rank, path, st, sx)
    if best is None:
        raise SystemExit("нет окна v8 test, где hot и cold оба выше пола вероятности")
    return best[1], best[2], best[3]


def _polarity_image(lab, zones):
    img = np.zeros(lab.shape, np.float32)
    for z in zones:
        img[lab == z["zone"]] = 1.0 if z["polarity"] == "hot" else 2.0
    return img


def main():
    if not PARAMS.is_file():
        raise SystemExit(f"нет {PARAMS}: сетка val ещё не дописана")
    params = json.loads(PARAMS.read_text())
    path, st, sx = _pick_sequence(float(params.get("floor", 0.5)))
    data = np.load(path)
    pred = np.load(PRED / path.name)
    score = pred["s"]
    sign = pred["h"]
    valid = data["valid"]
    ev = data["ev"]
    time_s = data["t"]
    dist_m = data["x"]
    lab_cc, zones_cc = extract_zones_cc(score, valid, sign, thr=CC_THR)
    lab_p, zones_p = extract_zones_pdrs(score, valid, sign, time_s, dist_m, **params)
    gt = np.zeros(ev.shape, np.float32)
    gt[np.isin(ev, HOT) & valid] = 1.0
    gt[np.isin(ev, COLD) & valid] = 2.0
    x = dist_m[sx].astype(np.float64)
    t = time_s[st].astype(np.float64)
    hours0 = float(time_s[0])
    hours = (t - hours0) / 3600.0
    dx = float(np.median(np.diff(x))) if x.size > 1 else 0.25
    dt_h = (float(np.median(np.diff(t))) if t.size > 1 else 120.0) / 3600.0
    extent = [
        float(x[0] - dx / 2), float(x[-1] + dx / 2),
        float(hours[-1] + dt_h / 2), float(hours[0] - dt_h / 2),
    ]

    def show(img):
        out = img[st, sx].astype(np.float32)
        return np.where(valid[st, sx], out, np.nan)

    cmap_s = plt.get_cmap("magma").copy()
    cmap_s.set_bad("#d0d0d0")
    cmap_z = ListedColormap(["#f7f7f7", "#b2182b", "#2166ac"])
    cmap_z.set_bad("#d0d0d0")
    panels = [
        (show(score), cmap_s, 0.0, 1.0, "вероятность ансамбля s"),
        (show(gt), cmap_z, 0.0, 2.0, "истинная маска (hot / cold)"),
        (show(_polarity_image(lab_cc, zones_cc)), cmap_z, 0.0, 2.0,
         f"порог + CC, far 1e-3 ({len(zones_cc)} зон на записи)"),
        (show(_polarity_image(lab_p, zones_p)), cmap_z, 0.0, 2.0,
         f"PDRS ({len(zones_p)} зон на записи)"),
    ]
    legend = [
        Patch(facecolor="#f7f7f7", edgecolor="#bbbbbb", label="фон"),
        Patch(facecolor="#b2182b", label="hot"),
        Patch(facecolor="#2166ac", label="cold"),
        Patch(facecolor="#d0d0d0", label="нет данных"),
    ]
    fig, axes = plt.subplots(4, 1, figsize=(11.2, 9.0), sharex=True, sharey=True)
    for ax, (img, cmap, vmin, vmax, title) in zip(axes, panels):
        im = ax.imshow(img, extent=extent, origin="upper", aspect="auto",
                       interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylabel("часы от начала")
        if vmax == 1.0:
            fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    axes[1].legend(handles=legend, loc="upper right", fontsize=8, framealpha=0.92, ncol=4)
    axes[-1].set_xlabel("расстояние, м")
    param_txt = ", ".join(f"{k}={params[k]}" for k in ("seed", "floor", "r_saddle", "region_median"))
    fig.suptitle(
        f"v8 test {path.name}: CC и PDRS на карте ens_u2ft_tcn_lgbm\nпараметры PDRS с val: {param_txt}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=120)
    plt.close(fig)
    print(f"wrote {OUT} from {path.name} window t[{st.start}:{st.stop}] x[{sx.start}:{sx.stop}] "
          f"cc={len(zones_cc)} pdrs={len(zones_p)}")


if __name__ == "__main__":
    main()
