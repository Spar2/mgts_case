"""Кластеры зон реальных записей МГТС. Только инференс.

Порог детектора — готовая рабочая точка far_1e-3 из models_release (ансамбль).
На этих записях ничего не обучается и порог не подбирается. Стандартизация и число
кластеров относятся только к описанию уже найденных зон.
"""
from __future__ import annotations
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dts import events as E

# Вход последнего Linear слоя head: Dropout -> Linear -> ReLU -> Linear.
EMBED_NOTE = "вход последнего слоя EventNet.head (ReLU после первого Linear, усреднение с отражением по x)"
PHYS_KEYS = ("dur_h", "width_m", "amp_med", "amp_peak", "sign", "acf_peak", "acf_period", "rise_frac", "fall_frac", "plateau_frac")


def head_input(enet, img, scal):
    """Активация на входе последнего слоя head. img/scal — батч torch."""
    h = torch_cat_embed(enet, img, scal)
    return enet.head[:-1](h)


def torch_cat_embed(enet, img, scal):
    import torch
    return torch.cat([enet.cnn(img), enet.mlp(scal)], 1)


def eventnet_embedding(enet, imgs, scal, mu, sd):
    """Эмбеддинг зон. Скаляры нормируются так же, как в classify. Отражение по x усредняется."""
    import torch
    enet.eval()
    mu = np.asarray(mu, np.float32)
    sd = np.asarray(sd, np.float32)
    S = ((np.asarray(scal, np.float32) - mu) / sd).astype(np.float32)
    xt = torch.from_numpy(np.asarray(imgs, np.float32))
    St = torch.from_numpy(S)
    with torch.no_grad():
        a = head_input(enet, xt, St)
        b = head_input(enet, torch.flip(xt, [3]), St)
        emb = (0.5 * (a + b)).cpu().numpy().astype(np.float32)
    return emb


def physical_row(scal):
    """Физические величины одной зоны из вектора SCALAR_NAMES."""
    s = {n: float(scal[i]) for i, n in enumerate(E.SCALAR_NAMES)}
    return dict(
        dur_h=float(np.expm1(s["log_dur_h"])),
        width_m=float(np.expm1(s["log_width_m"])),
        amp_med=s["amp_med"],
        amp_peak=s["amp_peak"],
        sign=s["sign"],
        acf_peak=s["acf_peak"],
        acf_period=s["acf_period"],
        rise_frac=s["rise_frac"],
        fall_frac=s["fall_frac"],
        plateau_frac=s["plateau_frac"],
    )


def name_cluster(med):
    """Короткое имя по медианам. Пороги фиксированы, не по меткам классификатора.

    Длительность: <1 ч кратко, <6 ч многочасовой, иначе длительный.
    Ширина: <8 м узкий, <25 м умеренный, иначе обширный.
    Знак медианы: >0.25 нагрев, <-0.25 охлаждение, иначе знакопеременное.
    """
    dur, width, sign = float(med["dur_h"]), float(med["width_m"]), float(med["sign"])
    if dur < 1.0:
        dur_key = "short"
    elif dur < 6.0:
        dur_key = "mid"
    else:
        dur_key = "long"
    if width < 8.0:
        w_key = "narrow"
    elif width < 25.0:
        w_key = "midw"
    else:
        w_key = "wide"
    forms = {
        "short": ("кратковременный", "кратковременное"),
        "mid": ("многочасовой", "многочасовое"),
        "long": ("длительный", "длительное"),
        "narrow": ("узкий", "узкое"),
        "midw": ("умеренный", "умеренное"),
        "wide": ("обширный", "обширное"),
    }
    if sign > 0.25:
        g, noun = 0, "нагрев"
    elif sign < -0.25:
        g, noun = 1, "охлаждение"
    else:
        g, noun = 1, "знакопеременное"
    name = f"{forms[dur_key][g]} {forms[w_key][g]} {noun}"
    extra = []
    if float(med["acf_peak"]) >= 0.4:
        extra.append("с периодичностью")
    rise, fall, plat = float(med["rise_frac"]), float(med["fall_frac"]), float(med["plateau_frac"])
    # Нулевые фронты при плато на весь профиль значат, что спад не попал в бокс, а не что фронт резкий.
    if rise <= 0.1 and fall <= 0.1 and plat >= 0.8:
        extra.append("без спада внутри бокса")
    elif rise <= 0.1 and fall <= 0.1:
        extra.append("резкие фронты")
    elif rise >= 0.4:
        extra.append("медленный подъём")
    if extra:
        name += ", " + ", ".join(extra)
    return name


def _limit_threads(analyzer):
    for kind, mdl in analyzer.members:
        if kind == "pixel_lgbm" and hasattr(mdl, "set_params"):
            try:
                mdl.set_params(n_jobs=2)
            except Exception:
                pass
    if analyzer.lgbm is not None:
        models = analyzer.lgbm if isinstance(analyzer.lgbm, (list, tuple)) else [analyzer.lgbm]
        for m in models:
            if hasattr(m, "set_params"):
                try:
                    m.set_params(n_jobs=2)
                except Exception:
                    pass


def collect_records(analyzer, paths):
    """Инференс по файлам. Возвращает списки метаданных, скаляров (N, 19) и картинок."""
    from dts import real
    metas, scals, imgs = [], [], []
    per_file = []
    for path in paths:
        t0 = time.time()
        print("infer", path, flush=True)
        rec = real.load_csv(path)
        out = analyzer.run(rec["T"], rec["valid"], rec["x"], rec["t"])
        n = _append_zones(metas, scals, imgs, rec["name"], out["zones"], out["X"], out["sigma"], out["grid"])
        per_file.append(dict(record=rec["name"], n_zones=n, seconds=round(time.time() - t0, 1)))
        print("  zones", n, "sec", per_file[-1]["seconds"], flush=True)
    return _stack(metas, scals, imgs), per_file


def _append_zones(metas, scals, imgs, record, zones, X, sigma, grid):
    n = 0
    for z in zones:
        im, f = E.crop_and_features(X, sigma, grid["x"], grid["t"],
                                    z["start_m"], z["end_m"], z["t_start_s"], z["t_end_s"])
        imgs.append(im)
        scals.append(f)
        metas.append(dict(record=record, zone_class=z["class"], alarm=bool(z["alarm"]),
                          score=float(z["score"]), peak_dT_C=float(z.get("peak_dT_C", 0.0)),
                          start_m=float(z["start_m"]), end_m=float(z["end_m"]),
                          t_start_s=float(z["t_start_s"]), t_end_s=float(z["t_end_s"])))
        n += 1
    return n


def _stack(metas, scals, imgs):
    scal = np.stack(scals).astype(np.float32) if scals else np.zeros((0, len(E.SCALAR_NAMES)), np.float32)
    img = np.stack(imgs).astype(np.float32) if imgs else np.zeros((0, 3) + E.CROP, np.float32)
    return metas, scal, img


def _unix_utc(text):
    from datetime import datetime, timezone
    s = str(text).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return float(dt.timestamp())


def load_eventnet():
    """Только EventNet и порог релиза. Детектор stage-1 не грузится."""
    import json
    import torch
    from dts import models
    art = Path(os.environ.get("DTS_MODELS", ROOT / "models_release"))
    ck = torch.load(art / "eventnet.pt", map_location="cpu", weights_only=False)
    mu = np.asarray(ck["mu"], np.float32)
    sd = np.asarray(ck["sd"], np.float32)
    enet = models.EventNet(len(E.CLASSES), len(mu), cin=int(ck.get("cin", 3)), late_fusion=bool(ck.get("late_fusion", False)))
    enet.load_state_dict(ck["state"])
    enet.eval()
    th = json.loads((art / "stage1_thresholds.json").read_text())
    return enet, mu, sd, th


def collect_from_zones_csv(zones_csv, real_root):
    """Зоны из опубликованного csv. Каналы те же, что у детектора, без повторного прохода сети."""
    import csv as csvmod
    from collections import OrderedDict
    from dts import real, grid, features
    by_source = OrderedDict()
    with open(zones_csv, newline="") as f:
        for row in csvmod.DictReader(f):
            by_source.setdefault(row["source"], []).append(row)
    metas, scals, imgs, per_file = [], [], [], []
    root = Path(real_root)
    for source, rows in by_source.items():
        path = root / source
        if not path.exists():
            path = root / Path(source).name
        if not path.exists():
            raise SystemExit(f"нет записи для зон {source}: {root}")
        t0 = time.time()
        print("channels", path, "zones", len(rows), flush=True)
        rec = real.load_csv(path)
        sha = {r["source_sha256"] for r in rows}
        if sha != {rec["sha256"]}:
            raise SystemExit(f"sha {path.name} не совпал с csv: файл {rec['sha256']}, csv {sorted(sha)}")
        p = grid.plan(rec["x"], rec["t"])
        Tc, v = grid.resample(rec["T"], rec["valid"], p, "mean")
        X, aux = features.make_channels(Tc, v)
        zones = []
        for r in rows:
            zones.append(dict(
                start_m=float(r["x_start_m"]), end_m=float(r["x_end_m"]),
                t_start_s=_unix_utc(r["t_start_utc"]), t_end_s=_unix_utc(r["t_end_utc"]),
                **{"class": r["class"]}, alarm=bool(int(float(r["is_alarm"]))),
                score=float(r["score"]), peak_dT_C=float(r["peak_dT_C"]),
            ))
        n = _append_zones(metas, scals, imgs, rec["name"], zones, X, aux["sigma_x"], p)
        per_file.append(dict(record=rec["name"], n_zones=n, seconds=round(time.time() - t0, 1)))
        print("  crops", n, "sec", per_file[-1]["seconds"], flush=True)
        del Tc, v, X, rec
    return _stack(metas, scals, imgs), per_file


def cluster_matrix(scal, emb):
    """Стандартизованные скаляры и эмбеддинг. Скалер считается по этим зонам, без меток."""
    from sklearn.preprocessing import StandardScaler
    raw = np.nan_to_num(np.hstack([np.asarray(scal, np.float32), np.asarray(emb, np.float32)]).astype(np.float64))
    return StandardScaler().fit_transform(raw)


def select_kmeans(X, ks=range(2, 9), seed=0):
    """KMeans k=2..8. Выбор по silhouette. Возвращает (best_k, labels, rows)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    rows, best = [], None
    for k in ks:
        if k >= len(X):
            continue
        lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
        if len(set(lab)) < 2:
            continue
        try:
            sil = float(silhouette_score(X, lab, metric="euclidean"))
        except ValueError:
            continue
        rows.append(dict(k=int(k), silhouette=sil))
        if best is None or sil > best[0]:
            best = (sil, int(k), lab)
    if best is None:
        raise RuntimeError("не удалось выбрать k: слишком мало зон")
    return best[1], best[2], rows


def agglomerative_labels(X, k):
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    lab = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
    sil = float(silhouette_score(X, lab, metric="euclidean")) if len(set(lab)) > 1 else 0.0
    return lab, sil


def cluster_summaries(metas, phys, labels):
    """Медианы физических признаков и имя каждого кластера."""
    labels = np.asarray(labels)
    phys = list(phys)
    out = []
    for c in sorted(set(labels.tolist())):
        block = [phys[i] for i in np.flatnonzero(labels == c)]
        med = {k: float(np.median([row[k] for row in block])) for k in PHYS_KEYS}
        med["acf_share"] = float(np.mean([row["acf_peak"] >= 0.4 for row in block]))
        classes = {}
        for i in np.flatnonzero(labels == c):
            cls = metas[i]["zone_class"]
            classes[cls] = classes.get(cls, 0) + 1
        dominant = max(classes, key=classes.get)
        n = int((labels == c).sum())
        out.append(dict(id=int(c), n=n, medians=med, name=name_cluster(med),
                        class_counts=classes, dominant_class=dominant,
                        dominant_frac=classes[dominant] / n))
    # одинаковые имена различаем номером
    seen = {}
    for row in out:
        seen[row["name"]] = seen.get(row["name"], 0) + 1
    if any(v > 1 for v in seen.values()):
        for row in out:
            if seen[row["name"]] > 1:
                row["name"] = f"{row['name']} (кластер {row['id']})"
    return out


def contingency(metas, labels):
    labels = np.asarray(labels)
    counts = np.zeros((len(set(labels.tolist())), len(E.CLASSES)), int)
    order = sorted(set(labels.tolist()))
    idx = {c: i for i, c in enumerate(order)}
    for meta, lab in zip(metas, labels):
        counts[idx[int(lab)], E.CLASSES.index(meta["zone_class"])] += 1
    return order, counts


def adjusted_rand(a, b):
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(a, b))


def plot_pca(X, labels, classes, summaries, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=0)
    xy = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    cmap = plt.get_cmap("tab10")
    markers = ["o", "s", "^", "v", "D", "P", "X"]
    class_marker = {c: markers[i % len(markers)] for i, c in enumerate(E.CLASSES)}
    for c in sorted(set(labels.tolist())):
        m = labels == c
        # точки одного кластера рисуем по классам разными маркерами
        for cls in E.CLASSES:
            mm = m & (classes == cls)
            if not np.any(mm):
                continue
            ax.scatter(xy[mm, 0], xy[mm, 1], s=28, color=cmap(c % 10), marker=class_marker[cls],
                       linewidths=0.4, edgecolors="k", label=None)
    for row in summaries:
        m = labels == row["id"]
        ax.annotate(f"{row['id']}: {row['name']}", xy[m].mean(0), fontsize=8,
                    ha="center", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85, lw=0))
    # легенда маркеров классов
    for cls, mk in class_marker.items():
        ax.scatter([], [], color="0.4", marker=mk, label=cls)
    ax.legend(fontsize=7, loc="best", title="класс классификатора")
    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PCA 1 ({100 * var[0]:.0f}%)")
    ax.set_ylabel(f"PCA 2 ({100 * var[1]:.0f}%)")
    ax.set_title("Зоны реальных записей: цвет — кластер KMeans, маркер — класс")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return [float(v) for v in var]


def plot_profiles(summaries, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = ["dur_h", "width_m", "amp_med", "sign", "acf_peak", "rise_frac", "fall_frac", "plateau_frac"]
    labels_ru = ["длительность, ч", "ширина, м", "амплитуда, °C", "знак", "пик АКФ", "фронт подъёма", "фронт спада", "доля плато"]
    mat = np.array([[row["medians"][k] for k in keys] for row in summaries], float)
    show = mat.copy()
    scale = np.std(show, 0)
    scale[scale < 1e-9] = 1.0
    show = (show - np.mean(show, 0)) / scale
    fig, ax = plt.subplots(figsize=(9.2, 0.55 * len(summaries) + 2.2))
    im = ax.imshow(show, aspect="auto", cmap="RdBu_r", vmin=-2.2, vmax=2.2)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels_ru, rotation=30, ha="right")
    ax.set_yticks(range(len(summaries)))
    ax.set_yticklabels([f"{row['id']}: {row['name']} (n={row['n']})" for row in summaries], fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7, color="k")
    fig.colorbar(im, ax=ax, fraction=0.03, label="медиана, z между кластерами")
    ax.set_title("Медианы физических признаков кластеров")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def render_markdown(rep, fig_pca, fig_prof):
    lines = []
    lines.append("# Кластеры зон на реальных записях МГТС")
    lines.append("")
    if rep.get("zones_csv"):
        lines.append("Шесть файлов `/workspace/mgts/MGTS_data/1.csv`–`6.csv`. Зоны взяты из опубликованной таблицы, детектор в этом прогоне не запускался. Порог в шапке — текущий выпуск `models_release/` и по этим записям не подбирался. Классификатор этапа 2 не дообучался. Стандартизация и выбор числа кластеров описывают уже найденные зоны.")
        lines.append("")
        lines.append(f"Файл зон `{rep['zones_csv']}`. Каналы вырезок — `features.make_channels` на сетке детектора. Рабочая точка `{rep['op']}`, порог выпуска {rep['threshold']:.4f}, метод `{rep['method']}`. Зон: **{rep['n_zones']}**.")
    else:
        lines.append("Шесть файлов `/workspace/mgts/MGTS_data/1.csv`–`6.csv`. Детектор — `dts.infer.Analyzer(op='far_1e-3')`, модели `models_release/`. Порог взят из выпуска и по этим записям не подбирался. Классификатор этапа 2 не дообучался. Стандартизация и выбор числа кластеров описывают уже найденные зоны и не меняют детектор.")
        lines.append("")
        lines.append(f"Рабочая точка `{rep['op']}`, порог {rep['threshold']:.4f}, метод выпуска `{rep['method']}`. Зон: **{rep['n_zones']}**.")
    lines.append("")
    lines.append("## Признаки")
    lines.append("")
    lines.append("На каждую зону — вырезка `crop_and_features` и 19 скаляров `SCALAR_NAMES`, плюс эмбеддинг: " + EMBED_NOTE + f" (размер {rep['embed_dim']}). В кластеризацию идёт конкатенация, затем `StandardScaler` по этим зонам. Класс классификатора в признаки не входит.")
    lines.append("")
    lines.append("Скаляры, как их считает `crop_and_features`:")
    lines.append("")
    lines.append("- `log_dur_h`, `log_width_m` — log1p длительности в часах и ширины в метрах.")
    lines.append("- `width_frac`, `row_frac_cable` — доля ширины зоны от длины кабеля (две одинаковые колонки).")
    lines.append("- `amp_med`, `amp_peak`, `amp_min` — медиана, 99-й и 1-й перцентиль ΔT внутри бокса, °C.")
    lines.append("- `sign` — +1, если медиана неотрицательна, иначе −1.")
    lines.append("- `space_peak` — 95-й перцентиль модуля пространственного канала.")
    lines.append("- `rise_frac`, `fall_frac` — доля профиля до выхода на полумаксимум и после ухода с него. Малые значения — резкий фронт.")
    lines.append("- `plateau_frac` — доля отсчётов выше полумаксимума.")
    lines.append("- `n_sign_changes` — число смен знака производной, делённое на длину профиля.")
    lines.append("- `acf_period`, `acf_peak` — лаг пика автокорреляции (доля длины) и высота этого пика. По ним судим периодичность.")
    lines.append("- `edge_ratio` — контраст модуля ΔT внутри бокса к полю вырезки.")
    lines.append("- `valid_frac` — доля валидных пикселей.")
    lines.append("- `t_frac_start`, `t_frac_end` — положение начала и конца зоны внутри записи.")
    lines.append("")
    lines.append("Имя кластера собирается только из медиан длительности, ширины, знака, пика АКФ и фронтов. Пороги: кратко < 1 ч, многочасовой < 6 ч, иначе длительный; узкий < 8 м, умеренный < 25 м, иначе обширный; знак > 0.25 нагрев, < −0.25 охлаждение.")
    lines.append("")
    lines.append("## Записи")
    lines.append("")
    time_col = "секунд каналов" if rep.get("zones_csv") else "секунд инференса"
    lines.append(f"| запись | зон | {time_col} |")
    lines.append("|---|---:|---:|")
    for row in rep["per_file"]:
        lines.append(f"| {row['record']} | {row['n_zones']} | {row['seconds']} |")
    lines.append("")
    lines.append("## Выбор числа кластеров")
    lines.append("")
    lines.append("KMeans, `n_init=10`, `random_state=0`, евклидово расстояние после стандартизации. Критерий — silhouette.")
    lines.append("")
    lines.append("| k | silhouette |")
    lines.append("|---:|---:|")
    for row in rep["silhouette"]:
        mark = " ← выбран" if row["k"] == rep["k"] else ""
        lines.append(f"| {row['k']} | {row['silhouette']:.3f}{mark} |")
    lines.append("")
    lines.append(f"Выбран k = **{rep['k']}**, silhouette KMeans {rep['kmeans_silhouette']:.3f}. Агломеративная кластеризация (Ward) при том же k: silhouette {rep['agg_silhouette']:.3f}, ARI с KMeans {rep['ari_kmeans_vs_agg']:.3f}.")
    lines.append("")
    lines.append("## Кластеры KMeans")
    lines.append("")
    lines.append("| id | имя | n | ч | м | °C мед. | знак | АКФ | доля АКФ≥0.4 | подъём | спад | плато | частый класс |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in rep["clusters"]:
        m = row["medians"]
        lines.append(
            f"| {row['id']} | {row['name']} | {row['n']} | {m['dur_h']:.2f} | {m['width_m']:.2f} | {m['amp_med']:.2f} | {m['sign']:.2f} | {m['acf_peak']:.2f} | {m['acf_share']:.2f} | {m['rise_frac']:.2f} | {m['fall_frac']:.2f} | {m['plateau_frac']:.2f} | {row['dominant_class']} ({row['dominant_frac']:.0%}) |"
        )
    lines.append("")
    lines.append("## Сопоставление с классификатором")
    lines.append("")
    lines.append(f"ARI кластеров KMeans и классов классификатора: **{rep['ari_vs_class']:.3f}**. ARI агломеративных кластеров и тех же классов: {rep['ari_agg_vs_class']:.3f}. Единица была бы при полном совпадении разбиений, ноль — при случайном согласии.")
    lines.append("")
    header = "| кластер | " + " | ".join(E.CLASSES) + " |"
    sep = "|---:|" + "|".join(["---:"] * len(E.CLASSES)) + "|"
    lines.append(header)
    lines.append(sep)
    for cid, counts in zip(rep["cluster_order"], rep["contingency"]):
        lines.append("| " + str(cid) + " | " + " | ".join(str(int(v)) for v in counts) + " |")
    lines.append("")
    lines.append(f"![PCA]({fig_pca})")
    lines.append("")
    lines.append(f"Доля дисперсии PCA: {rep['pca_var'][0]:.0%} и {rep['pca_var'][1]:.0%}. Подписи — имена кластеров в их центроидах. Маркер точки — класс классификатора, цвет — кластер.")
    lines.append("")
    lines.append(f"![профили]({fig_prof})")
    lines.append("")
    lines.append("## Выводы")
    lines.append("")
    for line in rep["conclusions"]:
        lines.append(f"- {line}")
    lines.append("")
    return "\n".join(lines)


def conclusions(rep):
    """Короткие выводы из уже посчитанных таблиц. Без подгонки порога."""
    lines = []
    lines.append(
        f"На {rep['n_zones']} зонах шести записей silhouette максимален при k={rep['k']} ({rep['kmeans_silhouette']:.3f}). "
        f"Ward-кластеризация при том же k согласована с KMeans на ARI {rep['ari_kmeans_vs_agg']:.3f} (silhouette {rep['agg_silhouette']:.3f})."
    )
    lines.append(
        f"Согласие с классами классификатора умеренное либо слабое: ARI {rep['ari_vs_class']:.3f}. "
        "Кластеры собраны по форме вырезки и эмбеддингу, а не по голосу классификатора, поэтому разбиение не обязано повторять семь имён."
    )
    # самый крупный и самый «периодический»
    biggest = max(rep["clusters"], key=lambda r: r["n"])
    periodic = max(rep["clusters"], key=lambda r: r["medians"]["acf_peak"])
    lines.append(
        f"Крупнейший кластер {biggest['id']} «{biggest['name']}»: {biggest['n']} зон, медиана {biggest['medians']['dur_h']:.2f} ч и {biggest['medians']['width_m']:.1f} м, "
        f"чаще всего классификатор ставит {biggest['dominant_class']} ({biggest['dominant_frac']:.0%})."
    )
    if periodic["id"] != biggest["id"] and periodic["medians"]["acf_peak"] >= 0.4:
        lines.append(
            f"Выше других по пику АКФ кластер {periodic['id']} «{periodic['name']}»: медиана acf_peak {periodic['medians']['acf_peak']:.2f}, "
            f"доля зон с АКФ ≥ 0.4 равна {periodic['medians']['acf_share']:.0%}."
        )
    cold = [r for r in rep["clusters"] if r["medians"]["sign"] < -0.25]
    hot = [r for r in rep["clusters"] if r["medians"]["sign"] > 0.25]
    lines.append(
        f"По знаку медианы нагрев у {sum(r['n'] for r in hot)} зон ({_ru_clusters(len(hot))}), "
        f"охлаждение у {sum(r['n'] for r in cold)} зон ({_ru_clusters(len(cold))})."
    )
    acfs = [r["medians"]["acf_peak"] for r in rep["clusters"]]
    if acfs and max(acfs) < 0.4:
        lines.append(
            "Ни один кластер не набирает медиану пика АКФ 0.4: устойчивой периодичности в типичной зоне нет. "
            + ". ".join(f"Кластер {r['id']} — {r['medians']['acf_peak']:.2f}" for r in rep["clusters"]) + "."
        )
    return lines


def _ru_clusters(n):
    n = int(n)
    if n % 10 == 1 and n % 100 != 11:
        word = "кластер"
    elif n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
        word = "кластера"
    else:
        word = "кластеров"
    return f"{n} {word}"


def main(argv=None):
    import torch
    torch.set_num_threads(2)
    ap = argparse.ArgumentParser(description="Кластеры зон реальных записей, только инференс")
    ap.add_argument("--real-root", default="/workspace/mgts/MGTS_data")
    ap.add_argument("--zones", default=None, help="csv опубликованных зон; без повторного детектора")
    ap.add_argument("--out", default=str(ROOT / "docs" / "REAL_CLUSTERS.md"))
    ap.add_argument("--fig", default=str(ROOT / "docs" / "figures"))
    a = ap.parse_args(argv)
    zones_csv = None
    if a.zones:
        enet, mu, sd, th = load_eventnet()
        packed, per_file = collect_from_zones_csv(a.zones, a.real_root)
        metas, scal, img = packed
        thr = float(th["thresholds"]["cnn_unet"]["far_1e-3"])
        method = th.get("method", "")
        zones_csv = str(a.zones)
    else:
        from dts.infer import Analyzer
        analyzer = Analyzer(op="far_1e-3")
        _limit_threads(analyzer)
        paths = [str(Path(a.real_root) / f"{i}.csv") for i in range(1, 7)]
        missing = [p for p in paths if not Path(p).exists()]
        if missing:
            raise SystemExit("нет файлов: " + ", ".join(missing))
        packed, per_file = collect_records(analyzer, paths)
        metas, scal, img = packed
        enet, mu, sd = analyzer.enet, analyzer.mu, analyzer.sd
        thr = float(analyzer.thr)
        method = None
    if len(metas) < 3:
        raise SystemExit(f"слишком мало зон для кластеров: {len(metas)}")
    emb = eventnet_embedding(enet, img, scal, mu, sd)
    if emb.shape[1] != enet.head[-1].in_features:
        raise RuntimeError("эмбеддинг не совпал со входом последнего слоя head")
    X = cluster_matrix(scal, emb)
    k, labels, rows = select_kmeans(X)
    agg, agg_sil = agglomerative_labels(X, k)
    phys = [physical_row(s) for s in scal]
    summaries = cluster_summaries(metas, phys, labels)
    order, table = contingency(metas, labels)
    classes = np.array([m["zone_class"] for m in metas])
    fig_dir = Path(a.fig)
    fig_dir.mkdir(parents=True, exist_ok=True)
    pca_path = fig_dir / "real_clusters_pca.png"
    prof_path = fig_dir / "real_clusters_profiles.png"
    pca_var = plot_pca(X, labels, classes, summaries, pca_path)
    plot_profiles(summaries, prof_path)
    import json
    if method is None:
        th_path = Path(os.environ.get("DTS_MODELS", ROOT / "models_release")) / "stage1_thresholds.json"
        method = json.loads(th_path.read_text()).get("method", "")
    rel_pca = os.path.relpath(pca_path, Path(a.out).resolve().parent)
    rel_prof = os.path.relpath(prof_path, Path(a.out).resolve().parent)
    rep = dict(
        op="far_1e-3", threshold=float(thr), method=method, zones_csv=zones_csv,
        n_zones=len(metas), per_file=per_file, embed_dim=int(emb.shape[1]),
        silhouette=rows, k=int(k), kmeans_silhouette=float(next(r["silhouette"] for r in rows if r["k"] == k)),
        agg_silhouette=float(agg_sil),
        ari_kmeans_vs_agg=adjusted_rand(labels, agg),
        ari_vs_class=adjusted_rand(classes, labels),
        ari_agg_vs_class=adjusted_rand(classes, agg),
        clusters=summaries, cluster_order=order, contingency=table.tolist(),
        pca_var=pca_var,
    )
    rep["conclusions"] = conclusions(rep)
    text = render_markdown(rep, rel_pca, rel_prof)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print("wrote", out, "k", k, "ARI", round(rep["ari_vs_class"], 3), flush=True)
    return rep


if __name__ == "__main__":
    main()
