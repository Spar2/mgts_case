#!/usr/bin/env python3
"""Сборка docs/RESULTS.md и подстановка цифр в docs/NOTE.md.

Читает (любого из файлов может ещё не быть):
  artifacts/eval/stage1_v8_*.json, r5_*.json, r6_*.json, r7_*.json — схема eval_stage1.py
    несколько --stage1 сливаются: поздний файл перекрывает тот же ключ
  artifacts/eval/v10_*.json, stage1_v9_test.json — тот же порог, другой набор
  artifacts/eval/stage1_{v810ft,nm}_{val,v8,v10}.json — дообучение, свой pooled val
  artifacts/eval/cv_v8.json, cv_v10.json, cv_cards.json — классический CV
  artifacts/eval/stage2b_postproc_v8.json, stage2b_tta_v8.json — пост-обработка
  artifacts/eval/domain_shift.json — доменный сдвиг
  artifacts/eval/real_gap.json — кривая порога прежнего выпуска
  artifacts/eval/real_gap_ft.json — промахи текущего выпуска, если файл есть
  docs/REAL_CLUSTERS.md — кластеры уже найденных зон
  models_release/stage1_thresholds.json — состав выпуска
  artifacts/eval/stage2_v8.json  — схема scripts/train_stage2.py
  artifacts/eval/real_v8.json    — схема scripts/run_real.py
  artifacts/eval/unsup_real.json — безучительный прогон, если файл есть
  artifacts/stage2_scratch/e2e_*.json — сквозная оценка
  artifacts/stage1/train_log*.json — лучший val loss, если файл есть
  artifacts/eval/v11_ens_u2ft_tcn_lgbm.json — test v11, порог из файла
  artifacts/eval/s2_ens_cn*_v8full.json, s2_ens_cn*_real.json — цветной шум
  artifacts/eval/s2_ms_v8.json — многомасштабный инференс
  artifacts/eval/realnet_*.json, realnet_bc_*.json — конвейер realnet
  artifacts/eval/real_cv_zones.csv — геометрия OpenCV уже найденных зон

Победитель этапа 1 и рабочая точка выбираются только по IoU в блоке val
на точке far_1e-3 слитого этапа 1. Цифры test, v9, v10, дообучения,
CV, карточек, домена, промахов, кластеров, v11, цветного шума,
многомасштабного инференса и realnet в этот выбор не входят.

Отсутствующий файл, битый объект и недостающий ключ не роняют таблицы:
в ячейке стоит «н/д». Повреждённый JSON (файл есть, но не читается) даёт
код выхода 1, отчёт при этом всё равно пишется.

Маркеры в записке: <!-- AUTO:имя -->значение<!-- /AUTO:имя -->.
Повторный запуск заменяет только значение между метками. Список имён
собирает build_auto().

Вычисление лёгкое (только JSON), один процесс, без torch.
Запуск из каталога grok_solution: python scripts/make_report.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]

FAR_MATCH = 1.1  # метод «на равном FAR», если фактический FAR не выше 1.1 × FAR_ref

METHOD_ORDER = [
    "unet",
    "unet_weak",
    "unet_ms",
    "tcn",
    "dilated",
    "matched_filter",
    "pixel_lgbm",
    "ssl_ae",
    "ens_cnn3",
    "ens_cnn3_lgbm",
    "ens_unet_lgbm",
    "ens_unet_tcn_lgbm",
    "unet_w24",
    "unet_s2",
    "ens_u2",
    "ens_u2_tcn_lgbm",
    "ens_u3_tcn_lgbm",
    "ens_u3_tcn_lgbm_w",
    "ens_all6",
    "ens_u2ft_tcn_lgbm",
    "ens_u3ft_tcn_lgbm",
    "base_median_x3_spatial_z",
]
METHOD_RU = {
    "unet": "U-Net",
    "unet_ms": "U-Net с интегрирующим входом",
    "unet_weak": "U-Net, вес слабых",
    "tcn": "TCN",
    "dilated": "Dilated FCN",
    "matched_filter": "согласованный фильтр",
    "pixel_lgbm": "LightGBM по пикселям",
    "ssl_ae": "SSL-автоэнкодер",
    "ens_cnn3": "ансамбль U-Net + unet_ms + TCN",
    "ens_cnn3_lgbm": "ансамбль U-Net + unet_ms + TCN + LightGBM",
    "ens_unet_lgbm": "ансамбль U-Net + LightGBM",
    "ens_unet_tcn_lgbm": "ансамбль U-Net + TCN + LightGBM",
    "unet_w24": "U-Net, ширина 24",
    "unet_s2": "U-Net, второй прогон ширины 16",
    "ens_u2": "ансамбль ens_u2",
    "ens_u2_tcn_lgbm": "ансамбль U-Net w16 + U-Net w24 + TCN + LightGBM",
    "ens_u3_tcn_lgbm": "ансамбль трёх U-Net + TCN + LightGBM",
    "ens_u3_tcn_lgbm_w": "ансамбль трёх U-Net + TCN + LightGBM, веса 1.5 у TCN и LightGBM",
    "ens_all6": "ансамбль шести карт",
    "ens_u2ft_tcn_lgbm": "ансамбль U-Net w24 + U-Net v8+v10 + TCN + LightGBM",
    "ens_u3ft_tcn_lgbm": "ансамбль U-Net + U-Net w24 + U-Net v8+v10 + TCN + LightGBM",
    "unet_v810ft": "U-Net, дообучение focal+Tversky на v8+v10",
    "unet_nm": "U-Net, шум AR(1) на v8+v10",
    "cv_ref": "классический CV",
    "cv_pipe": "CV со спайком и Hough",
    "unet_cv_post": "U-Net после маски CV",
    "base_median_x3_spatial_z": "эталон |Z|, пиксели",
    "base_pipeline_zones": "эталон, зоны",
    "cnn_eventnet": "EventNet",
    "cnn_image_only": "CNN только по кропу",
    "lgbm_features": "LightGBM по признакам",
    "ensemble_cnn_lgbm": "ансамбль CNN + LightGBM",
    "lgbm_full": "LightGBM, полный набор",
    "equal_weight_raw": "равный вес CNN и LightGBM",
    "deployed": "выпущенная смесь",
    "stack_blend": "смесь с весом по фолдам",
    "stack_logreg": "лог-регрессия поверх CNN и LightGBM",
    "rules_haklolarp": "правила haklolarp",
}
OP_ORDER = ["far_ref", "far_1e-2", "far_1e-3", "far_1e-4", "hyst_1e-3_1e-2", "frozen_3.5"]
OP_RU = {
    "far_ref": "FAR эталона",
    "far_1e-2": "FAR = 1e-2",
    "far_1e-3": "FAR = 1e-3",
    "far_1e-4": "FAR = 1e-4",
    "hyst_1e-3_1e-2": "гистерезис 1e-3→1e-2",
    "frozen_3.5": "|Z| = 3.5",
}
SELECT_OP = "far_1e-3"
# Без --stage1 читаются эти файлы, ранний первым: поздний перекрывает тот же ключ.
DEFAULT_STAGE1 = [
    "artifacts/eval/stage1_v8_early.json",
    "artifacts/eval/stage1_v8_ens.json",
    "artifacts/eval/stage1_v8_round1.json",
    "artifacts/eval/stage1_v8_round2.json",
    "artifacts/eval/stage1_v8_round3.json",
    "artifacts/eval/stage1_v8_round4.json",
    "artifacts/eval/r5_*.json",
    "artifacts/eval/r6_*.json",
    "artifacts/eval/r7_*.json",
]
# Полоса «лидеры почти рядом». Только для текста. Выбор — строго максимум val IoU.
LEAD_GAP = 0.005
# Состав среднего score. Источник — scripts раундов и release_models, не догадка по имени.
MEMBERS = {
    "ens_cnn3": "среднее карт U-Net, unet_ms и TCN",
    "ens_cnn3_lgbm": "среднее карт U-Net, unet_ms, TCN и пиксельного LightGBM",
    "ens_unet_lgbm": "среднее карт U-Net и пиксельного LightGBM",
    "ens_unet_tcn_lgbm": "среднее карт U-Net, TCN и пиксельного LightGBM",
    "ens_u2_tcn_lgbm": "среднее карт U-Net ширины 16, U-Net ширины 24, TCN и пиксельного LightGBM",
    "ens_u3_tcn_lgbm": "среднее карт U-Net ширины 16, U-Net ширины 24, второго U-Net ширины 16, TCN и пиксельного LightGBM",
    "ens_u3_tcn_lgbm_w": "те же пять карт, веса TCN и LightGBM 1.5, у трёх U-Net вес 1",
    "ens_all6": "среднее шести карт: U-Net ширины 16, U-Net ширины 24, второй U-Net ширины 16, unet_ms, TCN и пиксельный LightGBM",
    "ens_u2ft_tcn_lgbm": "среднее карт U-Net ширины 24, дообученного U-Net ширины 16 (focal+Tversky на v8+v10), TCN и пиксельного LightGBM",
    "ens_u3ft_tcn_lgbm": "среднее карт U-Net ширины 16, U-Net ширины 24, дообученного U-Net ширины 16 (focal+Tversky на v8+v10), TCN и пиксельного LightGBM",
}
DEFAULT_FT = (
    ("unet_v810ft", "artifacts/eval/stage1_v810ft_val.json",
     "artifacts/eval/stage1_v810ft_v8.json", "artifacts/eval/stage1_v810ft_v10.json",
     "artifacts/stage1/train_log_unet_v810ft.json"),
    ("unet_nm", "artifacts/eval/stage1_nm_val.json",
     "artifacts/eval/stage1_nm_v8.json", "artifacts/eval/stage1_nm_v10.json",
     "artifacts/stage1/train_log_unet_nm.json"),
)
DEFAULT_CV_V8 = "artifacts/eval/cv_v8.json"
DEFAULT_CV_V10 = "artifacts/eval/cv_v10.json"
DEFAULT_CV_CARDS = "artifacts/eval/cv_cards.json"
DEFAULT_POST = "artifacts/eval/stage2b_postproc_v8.json"
DEFAULT_TTA = "artifacts/eval/stage2b_tta_v8.json"
DEFAULT_DOMAIN = "artifacts/eval/domain_shift.json"
DEFAULT_GAP_PREV = "artifacts/eval/real_gap.json"
DEFAULT_GAP_FT = "artifacts/eval/real_gap_ft.json"
DEFAULT_CLUSTERS = "docs/REAL_CLUSTERS.md"
DEFAULT_RELEASE = "models_release/stage1_thresholds.json"
DEFAULT_V9 = "artifacts/eval/stage1_v9_test.json"
DEFAULT_V10_GLOB = "artifacts/eval/v10_*.json"
DEFAULT_UNSUP = "artifacts/eval/unsup_real.json"
DEFAULT_V11 = "artifacts/eval/v11_ens_u2ft_tcn_lgbm.json"
DEFAULT_CN = "artifacts/eval/s2_ens_cn_v8full.json"
DEFAULT_CNSYM = "artifacts/eval/s2_ens_cnsym_v8full.json"
DEFAULT_CN_REAL = "artifacts/eval/s2_ens_cn_real.json"
DEFAULT_CNSYM_REAL = "artifacts/eval/s2_ens_cnsym_real.json"
DEFAULT_MS = "artifacts/eval/s2_ms_v8.json"
DEFAULT_RN_SYNTH = "artifacts/eval/realnet_synth.json"
DEFAULT_RN_REAL = "artifacts/eval/realnet_real.json"
DEFAULT_RN_ABL = "artifacts/eval/realnet_ablation.json"
DEFAULT_RN_BC_SYNTH = "artifacts/eval/realnet_bc_synth.json"
DEFAULT_RN_BC_REAL = "artifacts/eval/realnet_bc_real.json"
DEFAULT_CV_ZONES = "artifacts/eval/real_cv_zones.csv"
DEFAULT_S2_DOC = "docs/EXPERIMENTS_S2.md"
# Короткие суффиксы маркеров AUTO. Имена стабильны между пересборками.
METHOD_SHORT = {
    "unet": "unet",
    "unet_weak": "unet_weak",
    "unet_ms": "unet_ms",
    "tcn": "tcn",
    "dilated": "dilated",
    "matched_filter": "mf",
    "pixel_lgbm": "lgbm",
    "ssl_ae": "ssl",
    "ens_cnn3": "ens_cnn3",
    "ens_cnn3_lgbm": "ens_cnn3_lgbm",
    "ens_unet_lgbm": "ens_unet_lgbm",
    "ens_unet_tcn_lgbm": "ens_unet_tcn_lgbm",
    "unet_w24": "unet_w24",
    "unet_s2": "unet_s2",
    "ens_u2": "ens_u2",
    "ens_u2_tcn_lgbm": "ens_u2_tcn_lgbm",
    "ens_u3_tcn_lgbm": "ens_u3_tcn_lgbm",
    "ens_u3_tcn_lgbm_w": "ens_u3_tcn_lgbm_w",
    "ens_all6": "ens_all6",
    "ens_u2ft_tcn_lgbm": "ens_u2ft_tcn_lgbm",
    "ens_u3ft_tcn_lgbm": "ens_u3ft_tcn_lgbm",
    "base_median_x3_spatial_z": "base",
}
TEMPLATE_ORDER = ["F1_dt30", "F1_dt240", "F1", "F2", "F3", "F4", "F5", "F6"]
TEMPLATE_RU = {
    "F1_dt30": "F1, dt≈30 с",
    "F1_dt240": "F1, dt≈240 с",
    "F1": "F1",
    "F2": "F2",
    "F3": "F3",
    "F4": "F4",
    "F5": "F5",
    "F6": "F6",
}
LABEL_ORDER = ["hotspot", "section_heating", "slow_warming", "tracing", "coldspot", "cooling"]
LABEL_RU = {
    "hotspot": "точечный нагрев",
    "section_heating": "нагрев участка",
    "slow_warming": "медленный прогрев",
    "tracing": "трассировка",
    "coldspot": "точечное охлаждение",
    "cooling": "охлаждение участка",
}
AMP_ORDER = ["<=2C", "2-5C", ">5C"]
AMP_RU = {"<=2C": "≤2 °C", "2-5C": "2–5 °C", ">5C": ">5 °C"}
CLASS_ORDER = [
    "Local_Heating",
    "Extended_Heating",
    "Contact_Trace",
    "Local_Cooling",
    "Extended_Cooling",
    "HVAC_Diurnal",
    "Sensor_Artifact",
]
CLASS_SHORT = {
    "Local_Heating": "LH",
    "Extended_Heating": "EH",
    "Contact_Trace": "CT",
    "Local_Cooling": "LC",
    "Extended_Cooling": "EC",
    "HVAC_Diurnal": "HVAC",
    "Sensor_Artifact": "SA",
}
MODEL_ORDER = [
    "cnn_eventnet",
    "cnn_image_only",
    "lgbm_features",
    "lgbm_full",
    "ensemble_cnn_lgbm",
    "equal_weight_raw",
    "deployed",
    "stack_blend",
    "stack_logreg",
    "rules_haklolarp",
]
STAGE2_ALIASES = {
    "cnn_eventnet": ("cnn_eventnet",),
    "cnn_image_only": ("cnn_image_only",),
    "lgbm_features": ("lgbm_features", "lgbm_full"),
    "ensemble_cnn_lgbm": ("ensemble_cnn_lgbm", "deployed", "equal_weight_raw"),
    "rules_haklolarp": ("rules_haklolarp",),
}
ALIASES = {
    "iou": ("iou", "IoU"),
    "f1": ("f1", "F1"),
    "precision": ("precision", "prec"),
    "recall": ("recall", "rec"),
    "far": ("far", "FAR"),
    "far_ratio": ("far_ratio",),
    "threshold": ("threshold", "thr"),
    "nuisance_alarm_frac": ("nuisance_alarm_frac", "nuisance"),
    "iou_ci95": ("iou_ci95", "iou_ci"),
    "macro_f1": ("macro_f1", "macroF1"),
    "coarse_macro_f1": ("coarse_macro_f1", "coarse_f1"),
    "ece": ("ece", "ECE"),
    "accuracy": ("accuracy", "acc"),
    "macro_f1_ci95": ("macro_f1_ci95", "macro_f1_ci", "ci95"),
    "confusion": ("confusion", "confusion_matrix"),
    "per_class": ("per_class", "f1_per_class"),
    "n": ("n", "count"),
    "pixel_recall": ("pixel_recall",),
}

NA = "н/д"
SPAN_RE = re.compile(
    r"<!--\s*AUTO:([A-Za-z0-9_.\-]+)\s*-->.*?<!--\s*/AUTO:\1\s*-->",
    re.DOTALL,
)
BARE_RE = re.compile(r"<!--\s*AUTO:([A-Za-z0-9_.\-]+)\s*-->")


def num(x):
    if isinstance(x, bool) or x is None:
        return None
    if isinstance(x, (int, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, str):
        try:
            v = float(x.strip())
        except ValueError:
            return None
        return v if math.isfinite(v) else None
    return None


def pick(d, name):
    if not isinstance(d, dict):
        return None
    for key in ALIASES.get(name, (name,)):
        if key in d and d[key] is not None:
            return d[key]
    return None


def fmt_metric(x, digits=3):
    v = num(x)
    if v is None:
        return NA
    return f"{v:.{digits}f}"


def fmt_far(x):
    v = num(x)
    if v is None:
        return NA
    ax = abs(v)
    if ax == 0:
        return "0"
    if ax < 1e-3:
        return f"{v:.2e}"
    return f"{v:.4f}"


def fmt_thr(x):
    if isinstance(x, (list, tuple)):
        parts = [fmt_thr(v) for v in x]
        if not parts or any(p == NA for p in parts):
            return NA
        return " / ".join(parts)
    v = num(x)
    if v is None:
        return NA
    ax = abs(v)
    if ax >= 100:
        return f"{v:.1f}"
    if ax >= 1:
        return f"{v:.3f}"
    return f"{v:.4g}"


def fmt_count(x):
    v = num(x)
    if v is None:
        return NA
    if abs(v - round(v)) < 1e-6:
        return str(int(round(v)))
    return f"{v:.3f}"


def fmt_ci(x):
    if not isinstance(x, (list, tuple)) or len(x) < 2:
        return NA
    a, b = fmt_metric(x[0]), fmt_metric(x[1])
    if a == NA or b == NA:
        return NA
    return f"[{a}; {b}]"


def fmt_delta(a, b):
    if a is None or b is None:
        return NA
    d = a - b
    sign = "+" if d >= 0 else "−"
    return f"{sign}{abs(d):.3f}"


def fmt_pct(x):
    v = num(x)
    if v is None:
        return NA
    return f"{100.0 * v:.2f}"


def cell(x):
    s = NA if x is None else str(x)
    s = s.replace("\r", " ").replace("\n", " ").replace("|", "/")
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > 180:
        s = s[:177] + "…"
    return s or NA


def md_table(headers, rows):
    if not rows:
        return "_Нет строк._"
    head = "| " + " | ".join(cell(h) for h in headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(cell(c) for c in row) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def method_ru(name):
    return METHOD_RU.get(name, str(name))


def op_ru(name):
    if not name:
        return "без точки"
    return OP_RU.get(name, str(name))


def rank_in(order, key):
    try:
        return order.index(key)
    except ValueError:
        return len(order)


def natural_key(text):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(text))]


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def rel(path: Path) -> str:
    """Путь от каталога проекта. Symlink artifacts/ не разворачивается:
    resolve() увёл бы ссылку в общий каталог данных."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def mtime(path: Path) -> str:
    if not path.is_file():
        return NA
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def load_json(path: Path):
    """Вернуть (dict | None, статус): ok / missing / invalid:..."""
    if not path.is_file():
        return None, "missing"
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:  # noqa: BLE001 — любой битый JSON должен стать «н/д», не падением
        return None, f"invalid: {exc.__class__.__name__}: {exc}"
    if not isinstance(data, dict):
        return None, "invalid: корень JSON не объект"
    return data, "ok"


def load_csv_rows(path: Path):
    """Строки CSV. Нет файла — missing. Битый файл — invalid, отчёт всё равно пишется."""
    if not path.is_file():
        return None, "missing"
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle)), "ok"
    except Exception as exc:  # noqa: BLE001 — битый CSV не должен ронять сборку
        return None, f"invalid: {exc.__class__.__name__}: {exc}"


def status_word(st: str) -> str:
    if st == "ok":
        return "есть"
    if st == "missing":
        return "нет файла"
    return "повреждён"


def dict_of(obj, key):
    if not isinstance(obj, dict):
        return {}
    val = obj.get(key)
    return val if isinstance(val, dict) else {}


def list_of(obj, key):
    if not isinstance(obj, dict):
        return []
    val = obj.get(key)
    return val if isinstance(val, list) else []


def weak_block(block):
    if not isinstance(block, dict):
        return None
    w = block.get("weak_events")
    if isinstance(w, dict) and (pick(w, "recall") is not None or pick(w, "n") is not None):
        return w
    by_amp = block.get("by_amp")
    if isinstance(by_amp, dict):
        for key in ("<=2C", "<=2", "<=2°C"):
            if isinstance(by_amp.get(key), dict):
                return by_amp[key]
    return None


def sub_metric(block, sub, name):
    if not isinstance(block, dict):
        return None
    node = block.get(sub)
    if isinstance(node, dict):
        return pick(node, name)
    return None


def stage1_rows(test):
    rows = []
    special = []
    if not isinstance(test, dict):
        return []
    for key, block in test.items():
        if not isinstance(block, dict):
            continue
        if "@" in str(key):
            method, op = str(key).split("@", 1)
            rows.append((method, op, str(key), block))
        else:
            special.append((str(key), "", str(key), block))
    rows.sort(key=lambda r: (rank_in(METHOD_ORDER, r[0]), r[0], rank_in(OP_ORDER, r[1]), r[1]))
    special.sort(key=lambda r: (0 if r[0] == "base_pipeline_zones" else 1, r[0]))
    return rows + special


def split_key(key):
    text = str(key)
    if "@" not in text:
        return text, ""
    method, op = text.split("@", 1)
    return method, op


def block_at(node, method, op):
    if not isinstance(node, dict):
        return None
    block = node.get(f"{method}@{op}")
    return block if isinstance(block, dict) else None


def methods_present(*nodes):
    found = []
    seen = set()
    for node in nodes:
        if not isinstance(node, dict):
            continue
        for key in node:
            method, _op = split_key(key)
            if method in {"base_pipeline_zones", ""} or method in seen:
                continue
            seen.add(method)
            found.append(method)
    found.sort(key=lambda name: (rank_in(METHOD_ORDER, name), name))
    return found


def winner_from_val(val):
    """Метод с максимальным IoU на val в точке SELECT_OP. None, если блока нет.

    Равенство IoU решает порядок METHOD_ORDER: раньше в списке — выше.
    Test сюда не передаётся.
    """
    best = None
    if not isinstance(val, dict):
        return None
    for key, block in val.items():
        method, op = split_key(key)
        if op != SELECT_OP or not isinstance(block, dict):
            continue
        iou = num(pick(block, "iou"))
        if iou is None:
            continue
        rank = rank_in(METHOD_ORDER, method)
        cand = (iou, -rank, method)
        if best is None or cand > best:
            best = cand
    if best is None:
        return None
    return best[2], best[0]


def metric_at(node, method, op, name):
    block = block_at(node, method, op)
    if not block:
        return NA
    return fmt_metric(pick(block, name))


def weak_at(node, method, op):
    block = block_at(node, method, op)
    weak = weak_block(block) if block else None
    if not weak:
        return NA
    return fmt_metric(pick(weak, "recall"))


def weak_span(test, op):
    vals = []
    for method in methods_present(test):
        block = block_at(test, method, op)
        weak = weak_block(block) if block else None
        value = num(pick(weak or {}, "recall"))
        if value is not None:
            vals.append(value)
    if not vals:
        return NA, NA
    return fmt_metric(min(vals)), fmt_metric(max(vals))


def ap_of(s1, method):
    node = dict_of(s1, "threshold_free").get(method)
    if isinstance(node, dict):
        return fmt_metric(pick(node, "pixel_average_precision"))
    return NA


def merge_stage1(loaded):
    """Слить несколько JSON этапа 1. Поздний файл перекрывает тот же ключ."""
    merged = {}
    any_ok = False
    for data, status in loaded:
        if status != "ok" or not isinstance(data, dict):
            continue
        any_ok = True
        for key in ("version", "far_ref", "base_test_far"):
            if data.get(key) is not None:
                merged[key] = data[key]
        for key in ("val", "test", "threshold_free", "test_far_matched", "thresholds"):
            node = data.get(key)
            if isinstance(node, dict):
                merged.setdefault(key, {}).update(node)
    return merged if any_ok else None


def stage1_status(statuses):
    if any(item == "ok" for item in statuses):
        return "ok"
    for item in statuses:
        if str(item).startswith("invalid"):
            return item
    return "missing"


def expand_inputs(items):
    """Строки путей. Шаблон с * ? [ раскрывается по имени файла, без копий кэша."""
    out = []
    for item in items:
        text = str(item)
        if any(ch in text for ch in "*?["):
            pat = resolve(text)
            found = sorted(pat.parent.glob(pat.name), key=lambda p: p.name) if pat.parent.is_dir() else []
            out.extend(found)
        else:
            out.append(resolve(text))
    return out


def is_candidate_source(path: Path) -> bool:
    name = path.name
    return name == "stage1_v8_round4.json" or name.startswith(("r5_", "r6_", "r7_"))


def refit_ru(value):
    if value is None or value == "":
        return NA
    text = str(value).strip()
    known = {
        "skipped: thresholds frozen from v8, not fit on this cache":
            "пропущен: порог заморожен с v8, на этом кэше не подгонялся",
    }
    return known.get(text, text)


def close(a, b, tol=1e-9):
    va, vb = num(a), num(b)
    if va is None or vb is None:
        return False
    return abs(va - vb) <= tol


def ratio_text(hit, total):
    if total is None:
        return NA
    if hit is None:
        return NA
    return f"{int(hit)}/{int(total)}"


def card_buckets(cards):
    buckets = {key: {"n": 0, "cnn": 0, "base": 0} for key in ("yes", "no", "unclear")}
    for card in cards:
        if not isinstance(card, dict):
            continue
        pres = str(card.get("presence") or "").strip().lower()
        if pres not in buckets:
            continue
        buckets[pres]["n"] += 1
        cnn = num(card.get("cnn_n_alarm_zones"))
        base = num(card.get("base_n_zones"))
        if cnn is not None and cnn > 0:
            buckets[pres]["cnn"] += 1
        if base is not None and base > 0:
            buckets[pres]["base"] += 1
    return buckets


def zones_per_file(files):
    bits = []
    for name in sorted(files, key=natural_key):
        rec = files[name]
        if not isinstance(rec, dict):
            continue
        bits.append(
            f"{name}: CNN {fmt_count(rec.get('n_zones_cnn'))} / эталон {fmt_count(rec.get('n_zones_base'))}"
        )
    return "; ".join(bits) if bits else NA


def thresholds_match(data, v8_thr):
    node = data.get("thresholds") if isinstance(data, dict) else None
    if not isinstance(node, dict) or not isinstance(v8_thr, dict):
        return NA
    compared = bad = 0
    for method, ops in node.items():
        if not isinstance(ops, dict):
            continue
        other = v8_thr.get(method)
        if not isinstance(other, dict):
            continue
        if num(ops.get(SELECT_OP)) is None or num(other.get(SELECT_OP)) is None:
            continue
        compared += 1
        if not close(ops.get(SELECT_OP), other.get(SELECT_OP)):
            bad += 1
    if compared == 0:
        return NA
    return "да" if bad == 0 else "нет"


def baseline_iou_stable(loaded):
    vals = []
    for data, status in loaded:
        if status != "ok" or not isinstance(data, dict):
            continue
        block = block_at(dict_of(data, "test"), "base_median_x3_spatial_z", SELECT_OP)
        value = num(pick(block or {}, "iou"))
        if value is not None:
            vals.append(value)
    if not vals:
        return NA
    return "да" if max(vals) - min(vals) <= 1e-12 else "нет"


def scored_val(val):
    rows = []
    for method in methods_present(val):
        if method == "base_median_x3_spatial_z":
            continue
        iou = num(pick(block_at(val, method, SELECT_OP) or {}, "iou"))
        if iou is None:
            continue
        rows.append((iou, rank_in(METHOD_ORDER, method), method))
    rows.sort(key=lambda item: (-item[0], item[1], item[2]))
    return rows


def pack_leaders(val, out):
    rows = scored_val(val)
    if not rows:
        for key in ("s1_lead_n", "s1_lead_val_min", "s1_lead_val_max", "s1_lead_names",
                    "s1_lead_second", "s1_lead_second_iou", "s1_lead_gap"):
            out[key] = NA
        return
    best = rows[0][0]
    leaders = [item for item in rows if item[0] >= best - LEAD_GAP]
    out["s1_lead_n"] = str(len(leaders))
    out["s1_lead_val_min"] = fmt_metric(min(item[0] for item in leaders))
    out["s1_lead_val_max"] = fmt_metric(max(item[0] for item in leaders))
    out["s1_lead_names"] = "; ".join(method_ru(item[2]) for item in leaders)
    if len(rows) >= 2:
        out["s1_lead_second"] = method_ru(rows[1][2])
        out["s1_lead_second_iou"] = fmt_metric(rows[1][0])
        out["s1_lead_gap"] = fmt_metric(rows[0][0] - rows[1][0])
    else:
        out["s1_lead_second"] = NA
        out["s1_lead_second_iou"] = NA
        out["s1_lead_gap"] = NA


def shift_records(loaded_ok):
    """Поздний файл перекрывает тот же метод. Только точка FAR = 1e-3 и строка зон эталона."""
    store = {}
    for path, data in loaded_ok:
        if not isinstance(data, dict):
            continue
        test = data.get("test")
        if not isinstance(test, dict):
            continue
        meta_from = data.get("thresholds_from")
        meta_refit = data.get("refit")
        val = data.get("val") if isinstance(data.get("val"), dict) else {}
        for key, block in test.items():
            if not isinstance(block, dict):
                continue
            method, op = split_key(key)
            if op != SELECT_OP and method != "base_pipeline_zones":
                continue
            store[(method, op)] = (path, block, val, meta_from, meta_refit)
    keys = sorted(store, key=lambda item: (rank_in(METHOD_ORDER, item[0]), item[0], 0 if item[1] == SELECT_OP else 1))
    return [(method, op, *store[(method, op)]) for method, op in keys]


def shift_table(loaded_ok, v8_thr=None):
    records = shift_records(loaded_ok)
    if not records:
        return "_В этих JSON нет точки FAR = 1e-3._"
    body = []
    for method, op, path, block, val, src, refit in records:
        vb = block_at(val, method, op) if op else None
        weak = weak_block(block) or {}
        events = block.get("events_all") if isinstance(block.get("events_all"), dict) else {}
        same = NA
        if isinstance(v8_thr, dict) and op == SELECT_OP:
            node = v8_thr.get(method)
            if isinstance(node, dict) and num(node.get(SELECT_OP)) is not None and num(pick(block, "threshold")) is not None:
                same = "да" if close(node.get(SELECT_OP), pick(block, "threshold")) else "нет"
        body.append([
            method_ru(method),
            "зоны эталона" if method == "base_pipeline_zones" else op_ru(op),
            fmt_metric(pick(vb or {}, "iou")) if vb else NA,
            fmt_metric(pick(block, "iou")),
            fmt_metric(pick(block, "f1")),
            fmt_metric(pick(block, "precision")),
            fmt_metric(pick(block, "recall")),
            fmt_far(pick(block, "far")),
            fmt_metric(pick(weak, "recall")),
            fmt_metric(pick(events, "recall")),
            fmt_thr(pick(block, "threshold")),
            same,
            str(src) if src else NA,
            refit_ru(refit),
            rel(path),
        ])
    return md_table(
        ["Метод", "Точка", "IoU val", "IoU", "F1", "Precision", "Recall", "FAR",
         "Recall ≤2 °C", "Recall событий", "Порог", "Порог = v8", "Откуда порог", "refit", "Файл"],
        body,
    )


def shift_summary(loaded_ok):
    bits = []
    for method, op, _path, block, *_rest in shift_records(loaded_ok):
        if op != SELECT_OP:
            continue
        bits.append(f"{method_ru(method)} IoU {fmt_metric(pick(block, 'iou'))}")
    return "; ".join(bits) if bits else NA


def method_on_shift(loaded_ok, method):
    for name, op, _path, block, *_rest in shift_records(loaded_ok):
        if name == method and op == SELECT_OP:
            return block
    return None


def e2e_label(path, data):
    method = str(data.get("method") or path.stem)
    name = method_ru(method)
    if "legacy" in path.name:
        name += ", прежний классификатор"
    return name


def load_e2e(pattern):
    """Файлы e2e_*.json. Возвращает (ряды (path, dict), список битых)."""
    pat = Path(pattern)
    if not pat.is_absolute():
        pat = ROOT / pat
    rows, bad = [], []
    if not pat.parent.is_dir():
        return rows, bad
    for path in sorted(pat.parent.glob(pat.name)):
        data, status = load_json(path)
        if status == "ok":
            rows.append((path, data))
        else:
            bad.append((path, status))
    rows.sort(key=lambda item: (rank_in(METHOD_ORDER, str(item[1].get("method") or "")), item[0].name))
    return rows, bad


def all_methods_table(s1, winner_key):
    val = dict_of(s1, "val")
    test = dict_of(s1, "test")
    threshold_free = dict_of(s1, "threshold_free")
    methods = methods_present(val, test, threshold_free)
    body = []
    for method in methods:
        vb = block_at(val, method, SELECT_OP) or {}
        tb = block_at(test, method, SELECT_OP) or {}
        weak = weak_block(tb) or {}
        ap = threshold_free.get(method) if isinstance(threshold_free.get(method), dict) else {}
        body.append([
            method_ru(method),
            fmt_metric(pick(vb, "iou")),
            "да" if method == winner_key else "нет",
            fmt_far(pick(tb, "far")),
            fmt_metric(pick(tb, "precision")),
            fmt_metric(pick(tb, "recall")),
            fmt_metric(pick(tb, "f1")),
            fmt_metric(pick(tb, "iou")),
            fmt_metric(pick(weak, "recall")),
            fmt_metric(pick(ap, "pixel_average_precision")),
        ])
    if not body:
        return "_В блоках val и test нет методов._"
    return md_table(
        ["Метод", "IoU val", "Победитель", "FAR test", "Precision", "Recall", "F1", "IoU test", "Recall ≤2 °C", "AP пикселей"],
        body,
    )


def hyst_table(s1):
    val = dict_of(s1, "val")
    test = dict_of(s1, "test")
    methods = []
    for method in methods_present(val, test):
        if any(block_at(node, method, op) for node in (val, test) for op in ("hyst_1e-3_1e-2", "far_1e-2")):
            methods.append(method)
    if not methods:
        return "_Точек гистерезиса и FAR = 1e-2 в JSON нет._"
    body = []
    for method in methods:
        body.append([
            method_ru(method),
            metric_at(val, method, SELECT_OP, "iou"),
            metric_at(val, method, "hyst_1e-3_1e-2", "iou"),
            metric_at(val, method, "far_1e-2", "iou"),
            metric_at(test, method, SELECT_OP, "iou"),
            metric_at(test, method, "hyst_1e-3_1e-2", "iou"),
            metric_at(test, method, "far_1e-2", "iou"),
        ])
    return md_table(
        ["Метод", "IoU val, 1e-3", "IoU val, гистерезис", "IoU val, 1e-2", "IoU test, 1e-3", "IoU test, гистерезис", "IoU test, 1e-2"],
        body,
    )


def weak_methods_table(s1):
    test = dict_of(s1, "test")
    methods = methods_present(dict_of(s1, "val"), test, dict_of(s1, "threshold_free"))
    body = []
    for method in methods:
        ref = block_at(test, method, "far_ref")
        strict = block_at(test, method, SELECT_OP)
        if ref is None and strict is None:
            continue
        body.append([
            method_ru(method),
            weak_at(test, method, "far_ref"),
            weak_at(test, method, SELECT_OP),
            fmt_count(pick(weak_block(strict or ref) or {}, "n")),
        ])
    if not body:
        return "_Recall слабых событий в JSON нет._"
    return md_table(
        ["Метод", "Recall ≤2 °C, FAR эталона", "Recall ≤2 °C, FAR = 1e-3", "Слабых событий"],
        body,
    )


def train_table(logs):
    """logs: list of (label, path, data or None)."""
    if not logs:
        return "_Журналы обучения не запрошены._"
    body = []
    for label, path, data in logs:
        if not isinstance(data, dict):
            body.append([label, NA, NA, rel(path)])
            continue
        hist = data.get("history")
        epochs = len(hist) if isinstance(hist, list) else None
        body.append([label, fmt_metric(data.get("best_val")), fmt_count(epochs), rel(path)])
    return md_table(["Модель", "Лучший val loss", "Эпох в журнале", "Файл"], body)


def e2e_tables(rows):
    if not rows:
        return ["_Файлов `e2e_*.json` нет._", ""]
    body = []
    class_rows = []
    classes = list(CLASS_ORDER)
    for path, data in rows:
        per = data.get("per_class") if isinstance(data.get("per_class"), dict) else {}
        for name in per:
            if name not in classes:
                classes.append(name)
        body.append([
            e2e_label(path, data),
            fmt_metric(data.get("macro_f1")),
            fmt_ci(data.get("macro_f1_ci95")),
            fmt_metric(data.get("event_recall")),
            fmt_ci(data.get("event_recall_ci95")),
            fmt_metric(data.get("alarm_precision")),
            fmt_ci(data.get("alarm_precision_ci95")),
            fmt_count(data.get("n_zones")),
            fmt_count(data.get("n_false_zones")),
            fmt_metric(data.get("false_zone_frac")),
            fmt_metric(data.get("matched_class_accuracy")),
            fmt_metric(data.get("false_nuisance_frac")),
            fmt_thr(data.get("threshold")),
            rel(path),
        ])
        class_rows.append([
            e2e_label(path, data)] + [
            fmt_metric(per.get(name).get("f1") if isinstance(per.get(name), dict) else per.get(name))
            for name in classes
        ])
    lines = [
        md_table(
            ["Метод", "macro-F1", "95% ДИ", "Recall событий", "95% ДИ recall", "Precision тревоги", "95% ДИ precision", "Зон", "Ложных зон", "Доля ложных", "Точность класса на совпавших", "Доля ложных, снятых как нетревога", "Порог", "Файл"],
            body,
        ),
        "",
        "F1 по классам на совпадении зон с метками. Класс без пары в знаменателе macro-F1 остаётся.",
        "",
        md_table(["Метод"] + [CLASS_SHORT.get(name, name) for name in classes], class_rows),
        "",
    ]
    return lines


def pack_selection(s1, out):
    val = dict_of(s1, "val")
    test = dict_of(s1, "test")
    won = winner_from_val(val)
    if won:
        method, iou = won
        out["s1_winner_name"] = method_ru(method)
        out["s1_winner_key"] = method
        out["s1_winner_val_iou"] = fmt_metric(iou)
        pack_detector(block_at(test, method, SELECT_OP), "s1_winner", out)
    else:
        out["s1_winner_name"] = NA
        out["s1_winner_key"] = NA
        out["s1_winner_val_iou"] = NA
        pack_detector(None, "s1_winner", out)
    for method, short in METHOD_SHORT.items():
        out[f"s1_val_iou_{short}"] = metric_at(val, method, SELECT_OP, "iou")
        out[f"s1_test_iou_{short}_1e3"] = metric_at(test, method, SELECT_OP, "iou")
        out[f"s1_val_iou_{short}_hyst"] = metric_at(val, method, "hyst_1e-3_1e-2", "iou")
        out[f"s1_test_iou_{short}_hyst"] = metric_at(test, method, "hyst_1e-3_1e-2", "iou")
        out[f"s1_val_iou_{short}_1e2"] = metric_at(val, method, "far_1e-2", "iou")
        out[f"s1_test_iou_{short}_1e2"] = metric_at(test, method, "far_1e-2", "iou")
        out[f"s1_ap_{short}"] = ap_of(s1, method)
        out[f"s1_weak_ref_{short}"] = weak_at(test, method, "far_ref")
        out[f"s1_weak_1e3_{short}"] = weak_at(test, method, SELECT_OP)
    lo, hi = weak_span(test, SELECT_OP)
    out["s1_weak_1e3_min"] = lo
    out["s1_weak_1e3_max"] = hi


def pack_train(logs, out):
    """logs: dict short label -> parsed json or None. Метки: unet, unet_ms, unet_weak."""
    logs = logs or {}
    for label, prefix in (("unet", "s1_train_best_unet"), ("unet_ms", "s1_train_best_unet_ms")):
        data = logs.get(label)
        hist = data.get("history") if isinstance(data, dict) else None
        out[prefix] = fmt_metric(data.get("best_val")) if isinstance(data, dict) else NA
        epoch_key = "s1_train_epochs_unet" if label == "unet" else "s1_train_epochs_unet_ms"
        out[epoch_key] = fmt_count(len(hist)) if isinstance(hist, list) and hist else NA
    weak = logs.get("unet_weak")
    args = weak.get("args") if isinstance(weak, dict) and isinstance(weak.get("args"), dict) else {}
    out["s1_train_weak_weight"] = fmt_count(args.get("weak_weight")) if num(args.get("weak_weight")) is not None else NA


def pack_e2e(rows, out):
    bits = []
    seen = set()
    for path, data in rows or []:
        method = str(data.get("method") or "")
        if not method:
            continue
        seen.add(method)
        short = METHOD_SHORT.get(method, re.sub(r"[^A-Za-z0-9_]", "_", method))
        prefix = f"e2e_{short}_legacy" if "legacy" in path.name else f"e2e_{short}"
        out[f"{prefix}_macro_f1"] = fmt_metric(data.get("macro_f1"))
        out[f"{prefix}_ci"] = fmt_ci(data.get("macro_f1_ci95"))
        out[f"{prefix}_event_recall"] = fmt_metric(data.get("event_recall"))
        out[f"{prefix}_alarm_precision"] = fmt_metric(data.get("alarm_precision"))
        out[f"{prefix}_false_frac"] = fmt_metric(data.get("false_zone_frac"))
        out[f"{prefix}_n_zones"] = fmt_count(data.get("n_zones"))
        out[f"{prefix}_n_false"] = fmt_count(data.get("n_false_zones"))
        out[f"{prefix}_matched_acc"] = fmt_metric(data.get("matched_class_accuracy"))
        out[f"{prefix}_nuisance_frac"] = fmt_metric(data.get("false_nuisance_frac"))
        bits.append(f"{e2e_label(path, data)} macro-F1 {out[f'{prefix}_macro_f1']}")
    for method, short in METHOD_SHORT.items():
        if method in seen:
            continue
        prefix = f"e2e_{short}"
        for suffix in ("macro_f1", "ci", "event_recall", "alarm_precision", "false_frac", "n_zones", "n_false", "matched_acc", "nuisance_frac"):
            out.setdefault(f"{prefix}_{suffix}", NA)
    out["e2e_summary"] = "; ".join(bits) if bits else NA
    out["e2e_n_files"] = str(len(rows)) if rows else ("0" if rows is not None else NA)


def collect_keys(rows, field, order):
    found = []
    seen = set()
    for *_, block in rows:
        node = block.get(field)
        if isinstance(node, dict):
            for key in node:
                if key not in seen:
                    seen.add(key)
                    found.append(key)
    return sorted(found, key=lambda k: (rank_in(order, k), str(k)))


def event_cell(node):
    if not isinstance(node, dict):
        return NA
    recall, n = pick(node, "recall"), pick(node, "n")
    pixel = pick(node, "pixel_recall")
    if recall is None and n is None:
        return NA
    text = fmt_metric(recall)
    if pixel is not None:
        text += f"; пикс. {fmt_metric(pixel)}"
    if n is not None:
        text += f" (n={fmt_count(n)})"
    return text


def weighted_recall(block, keys):
    node = block.get("by_template") if isinstance(block, dict) else None
    if not isinstance(node, dict):
        return None
    num_s = den = 0.0
    for key in keys:
        item = node.get(key)
        if not isinstance(item, dict):
            return None
        n, r = num(pick(item, "n")), num(pick(item, "recall"))
        if n is None or r is None:
            return None
        den += n
        num_s += n * r
    if den <= 0:
        return None
    return num_s / den


def iou_cell(value):
    if isinstance(value, dict):
        return fmt_metric(pick(value, "iou"))
    return fmt_metric(value)


def pack_detector(block, prefix, out):
    """Плоские AUTO-поля одного детектора. block может быть None."""
    b = block if isinstance(block, dict) else {}
    weak = weak_block(b) or {}
    raw_iou = num(pick(b, "iou"))
    raw_f1 = num(pick(b, "f1"))
    raw_weak = num(pick(weak, "recall"))
    out[f"{prefix}_threshold"] = fmt_thr(pick(b, "threshold"))
    out[f"{prefix}_far"] = fmt_far(pick(b, "far"))
    out[f"{prefix}_far_ratio"] = fmt_metric(pick(b, "far_ratio"))
    out[f"{prefix}_precision"] = fmt_metric(pick(b, "precision"))
    out[f"{prefix}_recall"] = fmt_metric(pick(b, "recall"))
    out[f"{prefix}_f1"] = fmt_metric(raw_f1)
    out[f"{prefix}_iou"] = fmt_metric(raw_iou)
    out[f"{prefix}_iou_ci"] = fmt_ci(pick(b, "iou_ci95"))
    out[f"{prefix}_hot_iou"] = fmt_metric(sub_metric(b, "hot", "iou"))
    out[f"{prefix}_cold_iou"] = fmt_metric(sub_metric(b, "cold", "iou"))
    out[f"{prefix}_weak_recall"] = fmt_metric(raw_weak)
    out[f"{prefix}_weak_pixel_recall"] = fmt_metric(pick(weak, "pixel_recall"))
    out[f"{prefix}_weak_n"] = fmt_count(pick(weak, "n"))
    out[f"{prefix}_event_recall"] = fmt_metric(sub_metric(b, "events_all", "recall"))
    out[f"{prefix}_nuisance"] = fmt_metric(pick(b, "nuisance_alarm_frac"))
    return raw_iou, raw_f1, raw_weak


def per_class_bits(block):
    node = pick(block, "per_class") if isinstance(block, dict) else None
    if not isinstance(node, dict):
        node = {}
    keys = [k for k in CLASS_ORDER if k in node] + [k for k in node if k not in CLASS_ORDER]
    pairs = []
    for key in keys:
        v = num(node.get(key))
        if v is None:
            continue
        pairs.append((key, v))
    return pairs


def extremum(pairs, how):
    if not pairs:
        return NA, NA
    if how == "min":
        key, val = min(pairs, key=lambda kv: (kv[1], rank_in(CLASS_ORDER, kv[0]), kv[0]))
    else:
        key, val = max(pairs, key=lambda kv: (kv[1], -rank_in(CLASS_ORDER, kv[0]), kv[0]))
    return str(key), fmt_metric(val)


def pack_classifier(block, prefix, out):
    b = block if isinstance(block, dict) else {}
    pairs = per_class_bits(b)
    worst_c, worst_f = extremum(pairs, "min")
    best_c, best_f = extremum(pairs, "max")
    text = "; ".join(f"{k} {fmt_metric(v)}" for k, v in pairs) or NA
    out[f"{prefix}_macro_f1"] = fmt_metric(pick(b, "macro_f1"))
    out[f"{prefix}_ci"] = fmt_ci(pick(b, "macro_f1_ci95"))
    out[f"{prefix}_coarse"] = fmt_metric(pick(b, "coarse_macro_f1"))
    out[f"{prefix}_ece"] = fmt_metric(pick(b, "ece"))
    out[f"{prefix}_accuracy"] = fmt_metric(pick(b, "accuracy"))
    out[f"{prefix}_per_class"] = text
    out[f"{prefix}_worst_class"] = worst_c
    out[f"{prefix}_worst_f1"] = worst_f
    out[f"{prefix}_best_class"] = best_c
    out[f"{prefix}_best_f1"] = best_f


def class_count_str(rep):
    classes = list_of(rep, "classes")
    counts = list_of(rep, "class_counts")
    if not classes:
        return NA
    bits = []
    for i, name in enumerate(classes):
        n = counts[i] if i < len(counts) else None
        bits.append(f"{name} — {fmt_count(n)}")
    return "; ".join(bits)


def presence_flag(value):
    if isinstance(value, bool):
        return value
    v = num(value)
    if v is not None and v in (0.0, 1.0) and not isinstance(value, str):
        return bool(int(v))
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"1", "true", "yes", "да", "anomaly", "present", "есть", "event"}:
            return True
        if s in {"0", "false", "no", "нет", "normal", "absent", "фон", "background", "none"}:
            return False
    return None


def pack_block_metrics(block, prefix, out):
    b = block if isinstance(block, dict) else {}
    weak = weak_block(b) or {}
    out[f"{prefix}_iou"] = fmt_metric(pick(b, "iou"))
    out[f"{prefix}_f1"] = fmt_metric(pick(b, "f1"))
    out[f"{prefix}_precision"] = fmt_metric(pick(b, "precision"))
    out[f"{prefix}_recall"] = fmt_metric(pick(b, "recall"))
    out[f"{prefix}_far"] = fmt_far(pick(b, "far"))
    out[f"{prefix}_weak"] = fmt_metric(pick(weak, "recall"))


def pack_shift_auto(out, winner_key, s1, v10_ok, v10_status, v9, v9_status, unsup, unsup_status):
    out["v10_status"] = status_word(v10_status)
    out["v10_n_files"] = str(len(v10_ok)) if v10_status == "ok" else ("0" if v10_status == "ok" else NA)
    if v10_status != "ok":
        out["v10_n_files"] = "0" if v10_status == "missing" else NA
    refits = []
    for _path, data in v10_ok:
        if isinstance(data, dict) and data.get("refit"):
            refits.append(str(data.get("refit")))
    uniq = list(dict.fromkeys(refit_ru(item) for item in refits))
    out["v10_refit"] = uniq[0] if len(uniq) == 1 else ("; ".join(uniq) if uniq else NA)
    out["v10_summary"] = shift_summary(v10_ok)
    pack_block_metrics(method_on_shift(v10_ok, winner_key), "v10_winner", out)
    base = method_on_shift(v10_ok, "base_median_x3_spatial_z")
    out["v10_base_iou"] = fmt_metric(pick(base or {}, "iou"))
    out["v9_status"] = status_word(v9_status)
    v9_ok = [(Path(DEFAULT_V9), v9)] if isinstance(v9, dict) else []
    out["v9_summary"] = shift_summary(v9_ok)
    out["v9_thr_match"] = thresholds_match(v9, dict_of(s1, "thresholds")) if isinstance(v9, dict) else NA
    pack_block_metrics(method_on_shift(v9_ok, winner_key), "v9_winner", out)
    out["unsup_status"] = status_word(unsup_status)
    presence = dict_of(unsup, "presence")
    synth = dict_of(dict_of(unsup, "synth_v8"), "test_iou_at_val_far")
    out["unsup_yes"] = ratio_text(num(presence.get("yes_hit_alarm")), num(presence.get("yes_n"))) if presence else NA
    out["unsup_no"] = ratio_text(num(presence.get("no_hit_alarm")), num(presence.get("no_n"))) if presence else NA
    out["unsup_synth_iou"] = fmt_metric(synth.get("iou")) if synth else NA


def pooled_op(data, split, op=SELECT_OP):
    """Схема stage1_*_val.json: val.{pooled,v8,v10}.{far_1e-3: блок}."""
    node = dict_of(dict_of(data, "val"), split)
    block = node.get(op) if isinstance(node, dict) else None
    return block if isinstance(block, dict) else None


def test_op(data, method, op=SELECT_OP):
    return block_at(dict_of(data, "test"), method, op)


def weak_count_text(block):
    if not isinstance(block, dict):
        return NA
    weak = weak_block(block)
    if weak:
        n = num(pick(weak, "n"))
        recall = num(pick(weak, "recall"))
    else:
        n = num(block.get("weak_n"))
        recall = num(block.get("weak_recall"))
    if n is None or recall is None:
        return NA
    hits = recall * n
    if abs(hits - round(hits)) < 1e-6:
        return f"{int(round(hits))}/{int(n)}"
    return f"{fmt_metric(recall)} из {int(n)}"


def train_facts(data):
    if not isinstance(data, dict):
        return {}
    hist = data.get("history") if isinstance(data.get("history"), list) else []
    args = data.get("args") if isinstance(data.get("args"), dict) else {}
    sec = None
    if hist and isinstance(hist[-1], dict):
        sec = num(hist[-1].get("sec"))
    return {
        "best": num(data.get("best_val")),
        "epochs": len(hist) if hist else None,
        "sec": sec,
        "loss": args.get("loss"),
        "sets": args.get("sets"),
        "noise_std": num(args.get("noise_std")),
        "phi_t": num(args.get("noise_phi_t")),
        "phi_x": num(args.get("noise_phi_x")),
    }


def ft_records(side):
    """Список (method, val, v8, v10, train) только из разобранных объектов."""
    rows = []
    for item in (side or {}).get("ft") or []:
        method, val, v8, v10, train = item
        rows.append((method, val if isinstance(val, dict) else None,
                     v8 if isinstance(v8, dict) else None,
                     v10 if isinstance(v10, dict) else None,
                     train if isinstance(train, dict) else None))
    return rows


def ft_choice(rows):
    scored = []
    for method, val, _v8, _v10, _train in rows:
        iou = num(pick(pooled_op(val, "pooled") or {}, "iou"))
        if iou is not None:
            scored.append((iou, method))
    if not scored:
        return None, None
    iou, method = max(scored)
    return method, iou


def fmt_noise(facts):
    std, pt, px = facts.get("noise_std"), facts.get("phi_t"), facts.get("phi_x")
    if std is None and pt is None and px is None:
        return NA
    if (std or 0) == 0 and (pt or 0) == 0 and (px or 0) == 0:
        return "нет"
    return f"std {fmt_metric(std)}, φ_t {fmt_metric(pt)}, φ_x {fmt_metric(px)}"


def presence_zone_buckets(cards, zone_key="n_zones"):
    buckets = {key: {"n": 0, "hit": 0} for key in ("yes", "no", "unclear")}
    for card in cards or []:
        if not isinstance(card, dict):
            continue
        flag = presence_flag(card.get("presence"))
        key = "unclear" if flag is None else ("yes" if flag else "no")
        raw = str(card.get("presence") or "").strip().lower()
        if flag is None and raw not in {"unclear", "неясно", "?"}:
            continue
        buckets[key]["n"] += 1
        n_zones = num(card.get(zone_key))
        if n_zones is not None and n_zones > 0:
            buckets[key]["hit"] += 1
    return buckets


def alarm_tally(data, branch, op):
    tallies = dict_of(dict_of(data, branch), "tallies")
    return dict_of(dict_of(tallies, op), "hit_alarm")


def curve_row(data, op):
    for row in list_of(data, "curve"):
        if isinstance(row, dict) and row.get("op") == op:
            return row
    return {}


def postproc_flag():
    path = ROOT / "dts" / "postproc.py"
    if not path.is_file():
        return NA
    match = re.search(r"^ENABLED\s*=\s*(True|False)\b", path.read_text(encoding="utf-8"), re.M)
    if not match:
        return NA
    return "да" if match.group(1) == "True" else "нет"


def parse_clusters(text):
    out = {}
    if not text:
        return out
    head = re.search(
        r"порог(?: выпуска)? ([0-9]+(?:\.[0-9]+)?), метод (?:выпуска )?`([^`]+)`\. Зон: \*\*(\d+)\*\*",
        text,
    )
    if head:
        out["thr"] = head.group(1)
        out["method"] = head.group(2)
        out["n"] = head.group(3)
    chosen = re.search(
        r"Выбран k = \*\*(\d+)\*\*, silhouette KMeans ([0-9.]+)\. "
        r"Агломеративная кластеризация \(Ward\) при том же k: silhouette ([0-9.]+), ARI с KMeans ([0-9.]+)\.",
        text,
    )
    if chosen:
        out["k"] = chosen.group(1)
        out["sil"] = chosen.group(2)
        out["sil_ward"] = chosen.group(3)
        out["ari_ward"] = chosen.group(4)
    ari = re.search(r"ARI кластеров KMeans и классов классификатора: \*\*([0-9.]+)\*\*", text)
    if ari:
        out["ari_class"] = ari.group(1)
    files_block = text.split("## Записи", 1)[-1].split("## ", 1)[0] if "## Записи" in text else ""
    files = re.findall(r"^\| (\d+) \| (\d+) \| [0-9.]+ \|$", files_block, re.M)
    if files:
        out["files"] = "; ".join(f"{name}: {n}" for name, n in files)
    cluster_block = text.split("## Кластеры KMeans", 1)[-1].split("## ", 1)[0] if "## Кластеры KMeans" in text else ""
    clusters = re.findall(
        r"^\| (\d+) \| ([^|]+?) \| (\d+) \| ([0-9.]+) \| ([0-9.]+) \| ([^|]+?) \| ([^|]+?) \| ([0-9.]+) \|",
        cluster_block,
        re.M,
    )
    bits = []
    for cid, name, n, hours, meters, _amp, sign, acf in clusters:
        sign_v = num(sign)
        kind = "нагрев" if sign_v is not None and sign_v > 0.25 else (
            "охлаждение" if sign_v is not None and sign_v < -0.25 else "без ясного знака"
        )
        bits.append(
            f"{cid} «{name.strip()}» ({kind}): {n} зон, медиана {hours} ч и {meters} м, АКФ {acf}"
        )
    if bits:
        out["summary"] = "; ".join(bits)
    return out


def release_members_text(data):
    if not isinstance(data, dict):
        return NA
    bits = []
    for member in list_of(data, "members"):
        if not isinstance(member, dict):
            continue
        kind = str(member.get("kind") or "")
        width = member.get("width")
        if kind == "unet" and num(width) is not None:
            bits.append(f"U-Net ширины {int(num(width))}")
        elif kind == "tcn":
            bits.append("TCN")
        elif kind == "pixel_lgbm":
            bits.append("пиксельный LightGBM")
        elif kind:
            bits.append(kind)
    return "; ".join(bits) if bits else NA


def unsup_file_text(unsup):
    files = dict_of(unsup, "files")
    if not files:
        return NA
    bits = []
    for name in sorted(files, key=natural_key):
        rec = files[name]
        if not isinstance(rec, dict):
            continue
        bits.append(f"{name}: {fmt_count(rec.get('n_zones'))} зон, порог {fmt_thr(rec.get('threshold'))}")
    return "; ".join(bits) if bits else NA


def unsup_expert_text(unsup):
    """Вердикты из json. Это поле отчёта, не разметка кабеля."""
    files = dict_of(unsup, "files")
    if not files:
        return NA
    bits = []
    for name in sorted(files, key=natural_key):
        rec = files[name]
        if not isinstance(rec, dict):
            continue
        tops = [z for z in list_of(rec, "zones_top") if isinstance(z, dict)]
        counts = {}
        for zone in tops:
            verdict = str(zone.get("verdict") or NA)
            counts[verdict] = counts.get(verdict, 0) + 1
        verdicts = ", ".join(f"{label} {n}" for label, n in counts.items()) or NA
        peak = None
        if tops:
            peak = max(tops, key=lambda zone: abs(num(zone.get("peak_dT_C")) or 0))
        if peak:
            sign = "+" if (num(peak.get("peak_dT_C")) or 0) >= 0 else ""
            peak_bit = (
                f"сильнейший пик id {peak.get('id')} {sign}{fmt_metric(peak.get('peak_dT_C'))} °C, "
                f"{fmt_metric(peak.get('start_m'))}–{fmt_metric(peak.get('end_m'))} м, "
                f"{fmt_metric(peak.get('dur_h'))} ч, «{peak.get('verdict') or NA}»"
            )
        else:
            peak_bit = "топ пуст"
        bits.append(f"файл {name}: {verdicts}; {peak_bit}")
    return "; ".join(bits) if bits else NA


def pack_side(out, side, s1, unsup):
    side = side or {}
    rows = ft_records(side)
    chosen, _chosen_iou = ft_choice(rows)
    out["ft_choice"] = method_ru(chosen) if chosen else NA
    out["ft_sets"] = NA
    out["ft_train"] = NA
    out["ft_thr"] = NA
    for key in (
        "ft_v8_iou", "ft_v8_f1", "ft_v8_far", "ft_v8_weak",
        "ft_v10_iou", "ft_v10_f1", "ft_v10_far", "ft_v10_weak",
        "ft_nm_pooled", "ft_nm_v8_iou", "ft_nm_v10_iou", "ft_pooled",
    ):
        out[key] = NA
    train_bits = []
    for method, val, v8, v10, train in rows:
        facts = train_facts(train)
        if method == "unet_v810ft" and facts.get("sets"):
            out["ft_sets"] = str(facts["sets"])
        if facts:
            train_bits.append(
                f"{method_ru(method)}: потеря {facts.get('loss') or NA}, эпох {fmt_count(facts.get('epochs'))}, "
                f"лучший val loss {fmt_metric(facts.get('best'))}, секунд в журнале {fmt_count(facts.get('sec'))}, "
                f"шум {fmt_noise(facts)}"
            )
        pooled = num(pick(pooled_op(val, "pooled") or {}, "iou"))
        v8_block = test_op(v8, method)
        v10_block = test_op(v10, method)
        if method == chosen:
            out["ft_pooled"] = fmt_metric(pooled)
            out["ft_thr"] = fmt_thr(pick(v8_block or {}, "threshold"))
            out["ft_v8_iou"] = fmt_metric(pick(v8_block or {}, "iou"))
            out["ft_v8_f1"] = fmt_metric(pick(v8_block or {}, "f1"))
            out["ft_v8_far"] = fmt_far(pick(v8_block or {}, "far"))
            out["ft_v8_weak"] = weak_count_text(v8_block)
            out["ft_v10_iou"] = fmt_metric(pick(v10_block or {}, "iou"))
            out["ft_v10_f1"] = fmt_metric(pick(v10_block or {}, "f1"))
            out["ft_v10_far"] = fmt_far(pick(v10_block or {}, "far"))
            out["ft_v10_weak"] = weak_count_text(v10_block)
        if method == "unet_nm":
            out["ft_nm_pooled"] = fmt_metric(pooled)
            out["ft_nm_v8_iou"] = fmt_metric(pick(v8_block or {}, "iou"))
            out["ft_nm_v10_iou"] = fmt_metric(pick(v10_block or {}, "iou"))
    out["ft_train"] = "; ".join(train_bits) if train_bits else NA

    def cv_metric(data, method, split, name):
        block = block_at(dict_of(data, split), method, SELECT_OP)
        return fmt_metric(pick(block or {}, name))

    cv8 = side.get("cv_v8") if isinstance(side.get("cv_v8"), dict) else None
    cv10 = side.get("cv_v10") if isinstance(side.get("cv_v10"), dict) else None
    out["cv_val_iou"] = cv_metric(cv8, "cv_ref", "val", "iou")
    out["cv_pipe_val_iou"] = cv_metric(cv8, "cv_pipe", "val", "iou")
    out["cv_unet_val_iou"] = cv_metric(cv8, "unet", "val", "iou")
    out["cv_post_val_iou"] = cv_metric(cv8, "unet_cv_post", "val", "iou")
    out["cv_base_val_iou"] = cv_metric(cv8, "base_median_x3_spatial_z", "val", "iou")
    ref_test = block_at(dict_of(cv8, "test"), "cv_ref", SELECT_OP) or {}
    out["cv_test_iou"] = fmt_metric(pick(ref_test, "iou"))
    out["cv_test_f1"] = fmt_metric(pick(ref_test, "f1"))
    out["cv_test_precision"] = fmt_metric(pick(ref_test, "precision"))
    out["cv_test_recall"] = fmt_metric(pick(ref_test, "recall"))
    out["cv_test_far"] = fmt_far(pick(ref_test, "far"))
    out["cv_test_event"] = fmt_metric(sub_metric(ref_test, "events_all", "recall"))
    out["cv_test_weak"] = fmt_metric(pick(weak_block(ref_test) or {}, "recall"))
    out["cv_test_ci"] = fmt_ci(pick(ref_test, "iou_ci95"))
    out["cv_thr"] = fmt_thr(pick(ref_test, "threshold"))
    out["cv_unet_test_iou"] = cv_metric(cv8, "unet", "test", "iou")
    out["cv_post_test_iou"] = cv_metric(cv8, "unet_cv_post", "test", "iou")
    out["cv_ap"] = fmt_metric(pick(dict_of(dict_of(cv8, "threshold_free"), "cv_ref"), "pixel_average_precision"))
    out["cv_unet_ap"] = fmt_metric(pick(dict_of(dict_of(cv8, "threshold_free"), "unet"), "pixel_average_precision"))
    out["cv10_val_iou"] = cv_metric(cv10, "cv_ref", "val", "iou")
    out["cv10_pipe_val_iou"] = cv_metric(cv10, "cv_pipe", "val", "iou")
    ref10 = block_at(dict_of(cv10, "test"), "cv_ref", SELECT_OP) or {}
    out["cv10_test_iou"] = fmt_metric(pick(ref10, "iou"))
    out["cv10_test_f1"] = fmt_metric(pick(ref10, "f1"))
    out["cv10_test_far"] = fmt_far(pick(ref10, "far"))
    out["cv10_test_event"] = fmt_metric(sub_metric(ref10, "events_all", "recall"))
    out["cv10_thr"] = fmt_thr(pick(ref10, "threshold"))
    out["cv10_test_ci"] = fmt_ci(pick(ref10, "iou_ci95"))
    cards = side.get("cv_cards") if isinstance(side.get("cv_cards"), dict) else None
    ref_cards = dict_of(dict_of(cards, "methods"), "cv_ref")
    buckets = presence_zone_buckets(list_of(ref_cards, "cards"))
    out["cv_yes"] = ratio_text(buckets["yes"]["hit"], buckets["yes"]["n"]) if buckets["yes"]["n"] else NA
    out["cv_no"] = ratio_text(buckets["no"]["hit"], buckets["no"]["n"]) if buckets["no"]["n"] else NA
    file_bits = []
    for name in sorted(dict_of(ref_cards, "files"), key=natural_key):
        rec = dict_of(ref_cards, "files").get(name)
        if isinstance(rec, dict):
            file_bits.append(f"{name}: {fmt_count(rec.get('n_zones'))}")
    out["cv_zones"] = "; ".join(file_bits) if file_bits else NA

    post = side.get("post") if isinstance(side.get("post"), dict) else None
    selection = dict_of(post, "selection")
    post_bits = []
    for method in ("unet", "ens_unet_tcn_lgbm"):
        node = selection.get(method) if isinstance(selection.get(method), dict) else None
        if not node:
            continue
        ident = node.get("identity") if isinstance(node.get("identity"), dict) else {}
        val = node.get("val") if isinstance(node.get("val"), dict) else {}
        cfar = dict_of(dict_of(dict_of(post, "val"), method), "cfar_x")
        post_bits.append(
            f"{method_ru(method)}: выбран {node.get('variant') or NA}, "
            f"val IoU {fmt_metric(val.get('iou'))} против identity {fmt_metric(ident.get('iou'))}, "
            f"слабые {weak_count_text(val)}; per-x CFAR val IoU {fmt_metric(cfar.get('iou'))}, "
            f"слабые {weak_count_text(cfar)}"
        )
    out["post_summary"] = "; ".join(post_bits) if post_bits else NA
    out["post_enabled"] = postproc_flag()
    tta = side.get("tta") if isinstance(side.get("tta"), dict) else None
    out["tta_val_iou"] = fmt_metric(pick(dict_of(tta, "val"), "iou"))
    out["tta_test_iou"] = fmt_metric(pick(dict_of(tta, "test"), "iou"))
    out["tta_val_events"] = fmt_metric(pick(dict_of(tta, "val"), "event_recall"))
    out["tta_test_events"] = fmt_metric(pick(dict_of(tta, "test"), "event_recall"))

    domain = side.get("domain") if isinstance(side.get("domain"), dict) else None
    out["dom_method"] = str(domain.get("method")) if isinstance(domain, dict) and domain.get("method") else NA
    out["dom_thr"] = fmt_thr(domain.get("global_thr") if isinstance(domain, dict) else None)
    out["dom_sigma"] = fmt_metric(domain.get("physical_sigma_ratio") if isinstance(domain, dict) else None)
    out["dom_mad"] = fmt_metric(domain.get("channel_mad_ratio") if isinstance(domain, dict) else None)
    synth = dict_of(domain, "synth_pooled")
    real_p = dict_of(domain, "real_pooled")
    out["dom_phi_t_real"] = fmt_metric(real_p.get("acf_t_lag1"))
    out["dom_phi_t_synth"] = fmt_metric(synth.get("acf_t_lag1"))
    out["dom_phi_x_real"] = fmt_metric(real_p.get("acf_x_lag1"))
    out["dom_phi_x_synth"] = fmt_metric(synth.get("acf_x_lag1"))
    out["dom_sigma_real"] = fmt_metric(real_p.get("sigma_p50"))
    out["dom_sigma_synth"] = fmt_metric(synth.get("sigma_p50"))
    glob = alarm_tally(domain, "calib", "global_far_1e-3")
    per = alarm_tally(domain, "calib", "per_file_1e-3")
    matched = alarm_tally(domain, "matched_calib", "global_far_1e-3")
    out["dom_yes"] = ratio_text(num(glob.get("yes_hit_alarm")), num(glob.get("yes_n"))) if glob else NA
    out["dom_no"] = ratio_text(num(glob.get("no_hit_alarm")), num(glob.get("no_n"))) if glob else NA
    out["dom_pf_yes"] = ratio_text(num(per.get("yes_hit_alarm")), num(per.get("yes_n"))) if per else NA
    out["dom_pf_no"] = ratio_text(num(per.get("no_hit_alarm")), num(per.get("no_n"))) if per else NA
    out["dom_matched_yes"] = ratio_text(num(matched.get("yes_hit_alarm")), num(matched.get("yes_n"))) if matched else NA
    out["dom_matched_no"] = ratio_text(num(matched.get("no_hit_alarm")), num(matched.get("no_n"))) if matched else NA

    prev = side.get("gap_prev") if isinstance(side.get("gap_prev"), dict) else None
    out["gap_prev_method"] = str(prev.get("method")) if isinstance(prev, dict) and prev.get("method") else NA
    g1 = curve_row(prev, "global_far_1e-3")
    g2 = curve_row(prev, "global_far_1e-2")
    gp = curve_row(prev, "per_file_1e-3")
    out["gap_prev_yes"] = ratio_text(num(g1.get("yes")), num(g1.get("yes_n"))) if g1 else NA
    out["gap_prev_no"] = ratio_text(num(g1.get("no")), num(g1.get("no_n"))) if g1 else NA
    out["gap_prev_zones"] = fmt_count(g1.get("n_alarm")) if g1 else NA
    out["gap_prev_zpk"] = fmt_metric(g1.get("zones_per_km_h")) if g1 else NA
    out["gap_prev_kmh"] = fmt_metric(g1.get("km_h")) if g1 else NA
    out["gap_prev_yes_1e2"] = ratio_text(num(g2.get("yes")), num(g2.get("yes_n"))) if g2 else NA
    out["gap_prev_no_1e2"] = ratio_text(num(g2.get("no")), num(g2.get("no_n"))) if g2 else NA
    out["gap_prev_pf_yes"] = ratio_text(num(gp.get("yes")), num(gp.get("yes_n"))) if gp else NA
    out["gap_prev_pf_no"] = ratio_text(num(gp.get("no")), num(gp.get("no_n"))) if gp else NA
    out["gap_prev_thr"] = fmt_thr(dict_of(prev, "thresholds").get("far_1e-3"))

    gap = side.get("gap_ft") if isinstance(side.get("gap_ft"), dict) else None
    out["gap_ft_method"] = str(gap.get("method")) if isinstance(gap, dict) and gap.get("method") else NA
    out["gap_ft_thr"] = fmt_thr(gap.get("threshold") if isinstance(gap, dict) else None)
    tally = dict_of(gap, "tally")
    out["gap_ft_yes"] = ratio_text(num(tally.get("yes")), num(tally.get("yes_n"))) if tally else NA
    out["gap_ft_no"] = ratio_text(num(tally.get("no")), num(tally.get("no_n"))) if tally else NA
    misses = [m for m in list_of(gap, "misses") if isinstance(m, dict)]
    out["gap_ft_n_miss"] = str(len(misses)) if isinstance(gap, dict) else NA
    miss_bits = []
    for miss in misses:
        miss_bits.append(
            f"{miss.get('id')} файл {miss.get('record')}: {miss.get('reason') or NA}, "
            f"max {fmt_thr(miss.get('max_score'))}"
        )
    out["gap_ft_misses"] = "; ".join(miss_bits) if miss_bits else (NA if not isinstance(gap, dict) else "нет")
    summary = dict_of(gap, "zone_summary")
    out["gap_ft_n"] = fmt_count(summary.get("n"))
    out["gap_ft_hot"] = fmt_count(summary.get("n_hot"))
    out["gap_ft_cold"] = fmt_count(summary.get("n_cold"))
    out["gap_ft_dur"] = fmt_metric(summary.get("dur_p50"))
    out["gap_ft_width"] = fmt_metric(summary.get("width_p50"))
    out["gap_ft_peak"] = fmt_metric(summary.get("peak_abs_p50"))
    out["gap_ft_both"] = fmt_count(summary.get("both"))
    out["gap_ft_base"] = fmt_count(summary.get("only_base"))
    out["gap_ft_unsup"] = fmt_count(summary.get("only_unsup"))
    out["gap_ft_neither"] = fmt_count(summary.get("neither"))
    classes = summary.get("classes") if isinstance(summary.get("classes"), dict) else {}
    out["gap_ft_classes"] = "; ".join(f"{name} {fmt_count(classes[name])}" for name in classes) if classes else NA
    check = dict_of(gap, "score_check")
    out["gap_ft_dmed"] = fmt_far(check.get("median_abs"))
    out["gap_ft_dmax"] = fmt_far(check.get("max_abs"))
    alarms = dict_of(gap, "file_alarm")
    out["gap_ft_files"] = "; ".join(
        f"{name}: {fmt_count(alarms[name])}" for name in sorted(alarms, key=natural_key)
    ) if alarms else NA

    clusters = parse_clusters(side.get("clusters_text") or "")
    out["cl_method"] = clusters.get("method", NA)
    out["cl_thr"] = fmt_thr(clusters.get("thr"))
    out["cl_n"] = clusters.get("n", NA)
    out["cl_k"] = clusters.get("k", NA)
    out["cl_sil"] = fmt_metric(clusters.get("sil"))
    out["cl_ari_ward"] = fmt_metric(clusters.get("ari_ward"))
    out["cl_ari_class"] = fmt_metric(clusters.get("ari_class"))
    out["cl_summary"] = clusters.get("summary", NA)
    out["cl_files"] = clusters.get("files", NA)

    release = side.get("release") if isinstance(side.get("release"), dict) else None
    out["release_method"] = str(release.get("method")) if isinstance(release, dict) and release.get("method") else NA
    out["release_members"] = release_members_text(release)
    out["release_thr"] = fmt_thr(dict_of(dict_of(release, "thresholds"), "cnn_unet").get(SELECT_OP))
    out["unsup_files"] = unsup_file_text(unsup)
    out["unsup_expert"] = unsup_expert_text(unsup)
    winner_key = out.get("s1_winner_key")
    same = NA
    if isinstance(gap, dict) and winner_key and winner_key != NA:
        same = "да" if gap.get("method") == winner_key else "нет"
    out["gap_ft_is_winner"] = same
    pack_session2(out, side, s1)


def plain_metrics(node):
    """Блок, где iou лежит прямо в словаре, а не под ключом far_1e-3."""
    return node if isinstance(node, dict) and num(node.get("iou")) is not None else None


def cooling_pair(block):
    if not isinstance(block, dict):
        return None, None
    node = block.get("by_label_cooling")
    if not isinstance(node, dict):
        labels = block.get("by_label")
        node = labels.get("cooling") if isinstance(labels, dict) else None
    if not isinstance(node, dict):
        return None, None
    return num(node.get("recall")), num(node.get("n"))


def fmt_cool(block):
    recall, n = cooling_pair(block)
    if recall is None:
        return NA
    if n is None:
        return fmt_metric(recall)
    return f"{fmt_metric(recall)} (n={int(n)})"


def fmt_weight(value):
    v = num(value)
    if v is None:
        return NA
    text = f"{v:.4f}".rstrip("0").rstrip(".")
    return text or "0"


def weights_text(weights):
    if isinstance(weights, (list, tuple)) and len(weights) >= 3:
        return " / ".join(fmt_weight(part) for part in weights[:3])
    if not isinstance(weights, dict):
        return NA
    return " / ".join(fmt_weight(weights.get(key)) for key in ("seg", "mask", "wide"))


def ablation_row(data, name):
    for row in list_of(data, "rows"):
        if isinstance(row, dict) and row.get("name") == name:
            return row
    chosen = data.get("chosen") if isinstance(data, dict) else None
    if isinstance(chosen, dict) and chosen.get("name") == name:
        return chosen
    return {}


def nested_iou(node, *path):
    cur = node
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    if isinstance(cur, dict):
        return num(cur.get("iou"))
    return num(cur)


def threshold_span(node):
    """Один порог или размах пофайловых порогов. Доля тревоги и число файлов сюда не входят."""
    if not isinstance(node, dict):
        return NA
    if num(node.get("threshold")) is not None:
        return fmt_thr(node.get("threshold"))
    summary = node.get("val_threshold_summary")
    if not isinstance(summary, dict):
        summary = node.get("test_threshold_summary")
    if isinstance(summary, dict):
        lo, hi = num(summary.get("threshold_min")), num(summary.get("threshold_max"))
        med = num(summary.get("threshold_median"))
        if lo is not None and hi is not None:
            span = f"{fmt_thr(lo)}–{fmt_thr(hi)}"
            if med is not None:
                return f"по файлу {span}, медиана {fmt_thr(med)}"
            return f"по файлу {span}"
    rule = node.get("threshold_rule")
    return str(rule) if rule else NA


def presence_ratios(node):
    """yes/no по уже посчитанному tally hit_alarm. Unclear в эти поля не входит."""
    if not isinstance(node, dict):
        return NA, NA
    alarm = node.get("alarm") if isinstance(node.get("alarm"), dict) else node
    yes = ratio_text(num(alarm.get("yes_hit_alarm")), num(alarm.get("yes_n")))
    no = ratio_text(num(alarm.get("no_hit_alarm")), num(alarm.get("no_n")))
    if yes == NA and no == NA and isinstance(node.get("cards"), list):
        buckets = card_buckets(node["cards"])
        yes = ratio_text(buckets["yes"]["cnn"], buckets["yes"]["n"]) if buckets["yes"]["n"] else NA
        no = ratio_text(buckets["no"]["cnn"], buckets["no"]["n"]) if buckets["no"]["n"] else NA
    return yes, no


def real_run_cards(data):
    """Карточки прогона run_real: тревожная зона CNN. Эталон сюда не подмешивается."""
    if not isinstance(data, dict):
        return NA, NA, NA
    cards = data.get("cards") if isinstance(data.get("cards"), list) else []
    if cards and any(isinstance(card, dict) and "cnn_n_alarm_zones" in card for card in cards):
        buckets = card_buckets(cards)
        yes = ratio_text(buckets["yes"]["cnn"], buckets["yes"]["n"]) if buckets["yes"]["n"] else NA
        no = ratio_text(buckets["no"]["cnn"], buckets["no"]["n"]) if buckets["no"]["n"] else NA
    else:
        yes, no = presence_ratios(data.get("presence_alarm") if isinstance(data.get("presence_alarm"), dict) else data.get("presence"))
    files = data.get("files") if isinstance(data.get("files"), dict) else {}
    bits = []
    for name in sorted(files, key=natural_key):
        rec = files[name]
        if not isinstance(rec, dict):
            continue
        n_alarm = rec.get("n_alarm")
        if n_alarm is None:
            n_alarm = rec.get("n_zones_cnn")
        bits.append(f"{name}: {fmt_count(n_alarm)}")
    return yes, no, ("; ".join(bits) if bits else NA)


def median_num(values):
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2


def pack_cv_zones(out, rows):
    if not rows:
        out["s2_cvz_n"] = out["s2_cvz_geom"] = out["s2_cvz_sol"] = NA
        out["s2_cvz_ecc"] = out["s2_cvz_drift"] = out["s2_cvz_period"] = NA
        return
    geom, sol, ecc = [], [], []
    n_drift = n_period = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("cv_area_px") or "").strip():
            geom.append(1)
        sol.append(num(row.get("cv_solidity")))
        ecc.append(num(row.get("cv_eccentricity")))
        if str(row.get("cv_drift_m_per_h") or "").strip():
            n_drift += 1
        if str(row.get("cv_period_h") or "").strip():
            n_period += 1
    out["s2_cvz_n"] = str(len(rows))
    out["s2_cvz_geom"] = str(len(geom))
    out["s2_cvz_sol"] = fmt_metric(median_num(sol))
    out["s2_cvz_ecc"] = fmt_metric(median_num(ecc))
    out["s2_cvz_drift"] = str(n_drift)
    out["s2_cvz_period"] = str(n_period)


def pack_noise_method(out, data, prefix):
    val = plain_metrics(dict_of(data, "val"))
    test = plain_metrics(dict_of(data, "test"))
    out[f"{prefix}_val_iou"] = fmt_metric(pick(val or {}, "iou"))
    out[f"{prefix}_test_iou"] = fmt_metric(pick(test or {}, "iou"))
    out[f"{prefix}_test_f1"] = fmt_metric(pick(test or {}, "f1"))
    out[f"{prefix}_test_far"] = fmt_far(pick(test or {}, "far"))
    out[f"{prefix}_cool_val"] = fmt_cool(val)
    out[f"{prefix}_cool_test"] = fmt_cool(test)
    return num(pick(val or {}, "iou"))


def pack_session2(out, side, s1):
    """Сессия 2. Ни один из этих файлов не подмешивается в val победителя."""
    side = side or {}
    v11 = side.get("v11") if isinstance(side.get("v11"), dict) else None
    v11_block = block_at(dict_of(v11, "test"), "ens_u2ft_tcn_lgbm", SELECT_OP) if v11 else None
    v11_base = block_at(dict_of(v11, "test"), "base_median_x3_spatial_z", SELECT_OP) if v11 else None
    out["s2_v11_iou"] = fmt_metric(pick(v11_block or {}, "iou"))
    out["s2_v11_f1"] = fmt_metric(pick(v11_block or {}, "f1"))
    out["s2_v11_p"] = fmt_metric(pick(v11_block or {}, "precision"))
    out["s2_v11_r"] = fmt_metric(pick(v11_block or {}, "recall"))
    out["s2_v11_far"] = fmt_far(pick(v11_block or {}, "far"))
    out["s2_v11_base_iou"] = fmt_metric(pick(v11_base or {}, "iou"))
    winner_key = out.get("s1_winner_key")
    winner_test = block_at(dict_of(s1, "test"), winner_key, SELECT_OP) if winner_key and winner_key != NA else None
    winner_val = block_at(dict_of(s1, "val"), winner_key, SELECT_OP) if winner_key and winner_key != NA else None
    win_thr = pick(winner_test or {}, "threshold")
    v11_thr = pick(v11_block or {}, "threshold")
    if num(win_thr) is None or num(v11_thr) is None:
        out["s2_v11_thr_match"] = NA
    else:
        out["s2_v11_thr_match"] = "да" if close(win_thr, v11_thr) else "нет"
    out["s2_win_cool"] = fmt_cool(winner_test)

    cn_val = pack_noise_method(out, side.get("cn"), "s2_cn")
    cns_val = pack_noise_method(out, side.get("cnsym"), "s2_cnsym")
    win_iou = num(pick(winner_val or {}, "iou"))
    flags = []
    for iou in (cn_val, cns_val):
        if iou is None or win_iou is None:
            flags.append(None)
        else:
            flags.append(iou < win_iou)
    if any(flag is None for flag in flags):
        out["s2_cn_below"] = NA
    else:
        out["s2_cn_below"] = "да" if all(flags) else "нет"
    cn_yes, cn_no, cn_files = real_run_cards(side.get("cn_real"))
    cns_yes, cns_no, cns_files = real_run_cards(side.get("cnsym_real"))
    out["s2_cn_yes"] = cn_yes
    out["s2_cn_no"] = cn_no
    out["s2_cn_files"] = cn_files
    out["s2_cnsym_yes"] = cns_yes
    out["s2_cnsym_no"] = cns_no
    out["s2_cnsym_files"] = cns_files

    ms = side.get("ms") if isinstance(side.get("ms"), dict) else None
    ms_val = dict_of(ms, "val")
    out["s2_ms_selected"] = str(ms.get("selected")) if isinstance(ms, dict) and ms.get("selected") else NA
    out["s2_ms_native"] = fmt_metric(pick(plain_metrics(ms_val.get("native")) or {}, "iou"))
    out["s2_ms_max"] = fmt_metric(pick(plain_metrics(ms_val.get("max")) or {}, "iou"))
    out["s2_ms_mean"] = fmt_metric(pick(plain_metrics(ms_val.get("mean")) or {}, "iou"))
    out["s2_ms_noise"] = fmt_metric(pick(plain_metrics(ms_val.get("noise")) or {}, "iou"))
    native_iou = num(pick(plain_metrics(ms_val.get("native")) or {}, "iou"))
    max_iou = num(pick(plain_metrics(ms_val.get("max")) or {}, "iou"))
    mean_iou = num(pick(plain_metrics(ms_val.get("mean")) or {}, "iou"))
    if None in (native_iou, max_iou, mean_iou):
        out["s2_ms_native_wins"] = NA
    else:
        out["s2_ms_native_wins"] = "да" if native_iou > max_iou and native_iou > mean_iou else "нет"

    abl = side.get("rn_abl") if isinstance(side.get("rn_abl"), dict) else None
    out["s2_ab_unet"] = fmt_metric(ablation_row(abl, "a_unet").get("iou"))
    out["s2_ab_adabn"] = fmt_metric(ablation_row(abl, "b_adabn").get("iou"))
    out["s2_ab_input"] = fmt_metric(ablation_row(abl, "c_baseline_input").get("iou"))

    synth = side.get("rn_synth") if isinstance(side.get("rn_synth"), dict) else None
    real_a = side.get("rn_real") if isinstance(side.get("rn_real"), dict) else None
    out["s2_rn_a_w"] = weights_text(synth.get("weights") if isinstance(synth, dict) else None)
    out["s2_rn_a_thr"] = fmt_thr(synth.get("threshold") if isinstance(synth, dict) else None)
    out["s2_rn_a_val"] = fmt_metric(nested_iou(synth, "val", "pooled"))
    out["s2_rn_a_v8"] = fmt_metric(nested_iou(synth, "test", "v8"))
    out["s2_rn_a_v10"] = fmt_metric(nested_iou(synth, "test", "v10"))
    yes_a, no_a = presence_ratios(real_a.get("presence_alarm") if isinstance(real_a, dict) else None)
    if yes_a == NA and isinstance(real_a, dict):
        yes_a, no_a = presence_ratios(real_a)
    out["s2_rn_a_yes"] = yes_a
    out["s2_rn_a_no"] = no_a

    bc_synth = side.get("rn_bc_synth") if isinstance(side.get("rn_bc_synth"), dict) else None
    bc_real = side.get("rn_bc_real") if isinstance(side.get("rn_bc_real"), dict) else None
    presence = dict_of(bc_real, "presence")
    for label, key in (("b", "B"), ("c", "C")):
        node = bc_synth.get(key) if isinstance(bc_synth, dict) else None
        out[f"s2_rn_{label}_w"] = weights_text(node.get("weights") if isinstance(node, dict) else None)
        out[f"s2_rn_{label}_thr"] = threshold_span(node)
        out[f"s2_rn_{label}_val"] = fmt_metric(nested_iou(node, "val", "pooled"))
        out[f"s2_rn_{label}_v8"] = fmt_metric(nested_iou(node, "test", "v8"))
        out[f"s2_rn_{label}_v10"] = fmt_metric(nested_iou(node, "test", "v10"))
        yes, no = presence_ratios(presence.get(key))
        out[f"s2_rn_{label}_yes"] = yes
        out[f"s2_rn_{label}_no"] = no
    pack_cv_zones(out, side.get("cv_zones") if isinstance(side.get("cv_zones"), list) else None)
    fig_dir = ROOT / "docs" / "figures"
    n_fig = len(list(fig_dir.glob("real_cv_*.png"))) if fig_dir.is_dir() else 0
    out["s2_cvz_fig"] = str(n_fig) if fig_dir.is_dir() else NA


def session2_sections(auto, side):
    auto = auto or {}
    lines = [
        "## Сессия 2",
        "",
        "Эти прогоны не выбирают победителя этапа 1. В выбор по-прежнему входит только максимум val IoU при FAR = 1e-3 в слитых файлах раундов. Test, карточки и real сюда не голосуют.",
        "",
        "### v11, тот же детектор",
        "",
        f"Файл `v11_ens_u2ft_tcn_lgbm.json`. Порог far 1e-3 совпадает с порогом победителя на v8: {auto.get('s2_v11_thr_match', NA)}.",
        "",
        md_table(
            ["Метод", "IoU", "F1", "Precision", "Recall", "FAR"],
            [[
                "выпущенный ансамбль",
                auto.get("s2_v11_iou", NA),
                auto.get("s2_v11_f1", NA),
                auto.get("s2_v11_p", NA),
                auto.get("s2_v11_r", NA),
                auto.get("s2_v11_far", NA),
            ], [
                "эталон Z",
                auto.get("s2_v11_base_iou", NA),
                NA, NA, NA, NA,
            ]],
        ),
        "",
        "### Цветной шум",
        "",
        "Порог каждого ансамбля снят с фона v8 val. Test и recall охлаждения победителя не назначают.",
        f"Оба val IoU строго ниже текущего победителя: {auto.get('s2_cn_below', NA)}. "
        f"Recall охлаждения победителя на test v8: {auto.get('s2_win_cool', NA)}.",
        "",
        md_table(
            ["Метод", "val IoU", "test IoU", "test F1", "test FAR", "охлаждение val", "охлаждение test", "карточки yes", "карточки no"],
            [
                ["ens_cn", auto.get("s2_cn_val_iou", NA), auto.get("s2_cn_test_iou", NA), auto.get("s2_cn_test_f1", NA),
                 auto.get("s2_cn_test_far", NA), auto.get("s2_cn_cool_val", NA), auto.get("s2_cn_cool_test", NA),
                 auto.get("s2_cn_yes", NA), auto.get("s2_cn_no", NA)],
                ["ens_cnsym", auto.get("s2_cnsym_val_iou", NA), auto.get("s2_cnsym_test_iou", NA), auto.get("s2_cnsym_test_f1", NA),
                 auto.get("s2_cnsym_test_far", NA), auto.get("s2_cnsym_cool_val", NA), auto.get("s2_cnsym_cool_test", NA),
                 auto.get("s2_cnsym_yes", NA), auto.get("s2_cnsym_no", NA)],
            ],
        ),
        "",
        f"Зоны real ens_cn: {auto.get('s2_cn_files', NA)}. ens_cnsym: {auto.get('s2_cnsym_files', NA)}.",
        "",
        "### Многомасштабный инференс",
        "",
        f"Выбран `{auto.get('s2_ms_selected', NA)}`. native выше max и mean по val IoU: {auto.get('s2_ms_native_wins', NA)}.",
        "",
        md_table(
            ["Слияние", "val IoU"],
            [
                ["native", auto.get("s2_ms_native", NA)],
                ["max", auto.get("s2_ms_max", NA)],
                ["mean", auto.get("s2_ms_mean", NA)],
                ["noise", auto.get("s2_ms_noise", NA)],
            ],
        ),
        "",
        "### Realnet",
        "",
        "Архитектура и экспертное чтение зон — в `docs/REAL_PIPELINE.md`. Это экспертное суждение, не эталон. Карточки вариант не выбирают.",
        f"Абляция v8 val, IoU: исходный U-Net {auto.get('s2_ab_unet', NA)}, AdaBN {auto.get('s2_ab_adabn', NA)}, замена входа {auto.get('s2_ab_input', NA)}.",
        "",
        md_table(
            ["Вариант", "Веса seg/mask/wide", "Порог", "val IoU", "test v8", "test v10", "yes", "no"],
            [
                ["A", auto.get("s2_rn_a_w", NA), auto.get("s2_rn_a_thr", NA), auto.get("s2_rn_a_val", NA),
                 auto.get("s2_rn_a_v8", NA), auto.get("s2_rn_a_v10", NA), auto.get("s2_rn_a_yes", NA), auto.get("s2_rn_a_no", NA)],
                ["B", auto.get("s2_rn_b_w", NA), auto.get("s2_rn_b_thr", NA), auto.get("s2_rn_b_val", NA),
                 auto.get("s2_rn_b_v8", NA), auto.get("s2_rn_b_v10", NA), auto.get("s2_rn_b_yes", NA), auto.get("s2_rn_b_no", NA)],
                ["C", auto.get("s2_rn_c_w", NA), auto.get("s2_rn_c_thr", NA), auto.get("s2_rn_c_val", NA),
                 auto.get("s2_rn_c_v8", NA), auto.get("s2_rn_c_v10", NA), auto.get("s2_rn_c_yes", NA), auto.get("s2_rn_c_no", NA)],
            ],
        ),
        "",
        "### Аннотации OpenCV",
        "",
        "`dts/zone_cv.py` не ищет зоны и не двигает порог. Геометрия вешается на уже найденные зоны выпуска.",
        f"Строк в `real_cv_zones.csv`: {auto.get('s2_cvz_n', NA)}, с площадью контура {auto.get('s2_cvz_geom', NA)}. "
        f"Медиана solidity {auto.get('s2_cvz_sol', NA)}, эксцентриситета {auto.get('s2_cvz_ecc', NA)}. "
        f"Ненулевой дрейф у {auto.get('s2_cvz_drift', NA)} зон, период у {auto.get('s2_cvz_period', NA)}. "
        f"Рисунков `real_cv_*.png`: {auto.get('s2_cvz_fig', NA)}.",
        "",
        "### Сравнение на карточках (только описательно)",
        "",
        "Одинаковое правило попадания внутри своего json. Строка не выбирает детектор и не двигает порог.",
        "",
        md_table(
            ["Источник", "yes", "no"],
            [
                ["победитель, выпуск", auto.get("real_yes_cnn", NA), auto.get("real_no_cnn", NA)],
                ["эталон haklolarp", auto.get("real_yes_base", NA), auto.get("real_no_base", NA)],
                ["realnet A", auto.get("s2_rn_a_yes", NA), auto.get("s2_rn_a_no", NA)],
                ["realnet B", auto.get("s2_rn_b_yes", NA), auto.get("s2_rn_b_no", NA)],
                ["realnet C", auto.get("s2_rn_c_yes", NA), auto.get("s2_rn_c_no", NA)],
                ["ens_cn", auto.get("s2_cn_yes", NA), auto.get("s2_cn_no", NA)],
                ["ens_cnsym", auto.get("s2_cnsym_yes", NA), auto.get("s2_cnsym_no", NA)],
                ["безучительный прогноз", auto.get("unsup_yes", NA), auto.get("unsup_no", NA)],
                ["классический CV", auto.get("cv_yes", NA), auto.get("cv_no", NA)],
            ],
        ),
        "",
    ]
    return lines


def ft_table(side):
    rows = ft_records(side)
    chosen, _iou = ft_choice(rows)
    if not rows:
        return "_Файлов дообучения нет._"
    body = []
    for method, val, v8, v10, _train in rows:
        v8_block = test_op(v8, method) or {}
        v10_block = test_op(v10, method) or {}
        body.append([
            method_ru(method),
            fmt_metric(pick(pooled_op(val, "pooled") or {}, "iou")),
            fmt_metric(pick(pooled_op(val, "v8") or {}, "iou")),
            fmt_metric(pick(pooled_op(val, "v10") or {}, "iou")),
            fmt_thr(pick(v8_block, "threshold")),
            fmt_metric(pick(v8_block, "iou")),
            fmt_metric(pick(v8_block, "f1")),
            fmt_far(pick(v8_block, "far")),
            weak_count_text(v8_block),
            fmt_metric(pick(v10_block, "iou")),
            fmt_metric(pick(v10_block, "f1")),
            fmt_far(pick(v10_block, "far")),
            weak_count_text(v10_block),
            "да" if method == chosen else "нет",
        ])
    return md_table(
        ["Метод", "IoU pooled val", "IoU val v8", "IoU val v10", "Порог",
         "IoU test v8", "F1 v8", "FAR v8", "Слабые v8",
         "IoU test v10", "F1 v10", "FAR v10", "Слабые v10", "Выбор дообучения"],
        body,
    )


def cv_split_table(data, title_methods):
    if not isinstance(data, dict):
        return "_JSON нет._"
    body = []
    for method in title_methods:
        val = block_at(dict_of(data, "val"), method, SELECT_OP)
        test = block_at(dict_of(data, "test"), method, SELECT_OP)
        if val is None and test is None:
            continue
        test = test or {}
        body.append([
            method_ru(method),
            fmt_metric(pick(val or {}, "iou")),
            fmt_metric(pick(test, "iou")),
            fmt_metric(pick(test, "f1")),
            fmt_metric(pick(test, "precision")),
            fmt_metric(pick(test, "recall")),
            fmt_far(pick(test, "far")),
            fmt_metric(pick(weak_block(test) or {}, "recall")),
            fmt_metric(sub_metric(test, "events_all", "recall")),
            fmt_metric(pick(dict_of(dict_of(data, "threshold_free"), method), "pixel_average_precision")),
            fmt_ci(pick(test, "iou_ci95")),
            fmt_thr(pick(test, "threshold")),
        ])
    if not body:
        return "_В JSON нет точки FAR = 1e-3._"
    return md_table(
        ["Метод", "IoU val", "IoU test", "F1", "Precision", "Recall", "FAR",
         "Recall ≤2 °C", "Recall событий", "AP", "95% ДИ IoU", "Порог"],
        body,
    )


def postproc_table(post, tta):
    lines = []
    if not isinstance(post, dict):
        lines.append("_JSON пост-обработки нет._")
    else:
        body = []
        val = dict_of(post, "val")
        test = dict_of(post, "test")
        selection = dict_of(post, "selection")
        for method in ("unet", "ens_unet_tcn_lgbm"):
            chosen = selection.get(method) if isinstance(selection.get(method), dict) else {}
            variant = chosen.get("variant")
            for name in ("identity", "box15", "cfar_x", "cfar_2d"):
                vb = dict_of(dict_of(val, method), name)
                tb = dict_of(dict_of(test, method), name)
                if not vb and not tb:
                    continue
                body.append([
                    method_ru(method),
                    name,
                    "да" if name == variant else "нет",
                    fmt_metric(vb.get("iou")),
                    fmt_metric(vb.get("f1")),
                    weak_count_text(vb),
                    fmt_metric(tb.get("iou")),
                    fmt_metric(tb.get("event_recall")),
                    weak_count_text(tb),
                ])
        lines.append(md_table(
            ["Метод", "Вариант", "Выбран по val", "IoU val", "F1 val", "Слабые val",
             "IoU test", "Recall событий test", "Слабые test"],
            body,
        ))
    lines.append("")
    if isinstance(tta, dict):
        lines.append(
            f"TTA U-Net ({tta.get('augment') or NA}): val IoU {fmt_metric(pick(dict_of(tta, 'val'), 'iou'))}, "
            f"test IoU {fmt_metric(pick(dict_of(tta, 'test'), 'iou'))}. "
            f"В выбор варианта пост-обработки test не входил."
        )
    else:
        lines.append("JSON TTA нет.")
    return lines


def domain_table(domain):
    if not isinstance(domain, dict):
        return ["_JSON домена нет._", ""]
    synth = dict_of(domain, "synth_pooled")
    real_p = dict_of(domain, "real_pooled")
    body = []
    for label, node in (("синтетика", synth), ("кабель", real_p)):
        if not node:
            continue
        body.append([
            label,
            fmt_metric(node.get("sigma_p50")),
            fmt_metric(node.get("sigma_p90")),
            fmt_metric(node.get("ch0_mad")),
            fmt_metric(node.get("ch1_mad")),
            fmt_metric(node.get("acf_t_lag1")),
            fmt_metric(node.get("acf_x_lag1")),
            fmt_far(node.get("frac_above_global_thr")),
            fmt_metric(node.get("score_p50")),
        ])
    lines = [
        f"Метод файла: `{domain.get('method', NA)}`, глобальный порог {fmt_thr(domain.get('global_thr'))}. "
        f"Отношение σ p50 real/synth {fmt_metric(domain.get('physical_sigma_ratio'))}, "
        f"отношение MAD канала 0 {fmt_metric(domain.get('channel_mad_ratio'))}.",
        "",
        md_table(
            ["Домен", "σ p50", "σ p90", "MAD ch0", "MAD ch1", "φ_t", "φ_x", "Доля ≥ порога", "score p50"],
            body,
        ),
        "",
    ]
    reasons = domain.get("shift_reasons")
    if isinstance(reasons, list) and reasons:
        lines.append("Причины, как их записал прогон: " + "; ".join(str(item) for item in reasons) + ".")
        lines.append("")
    return lines


def gap_prev_table(data):
    if not isinstance(data, dict):
        return ["_JSON кривой прежнего выпуска нет._", ""]
    body = []
    for row in list_of(data, "curve"):
        if not isinstance(row, dict):
            continue
        op_name = str(row.get("op") or "")
        thr_key = op_name.replace("global_", "") if op_name.startswith("global_") else None
        body.append([
            op_name or NA,
            fmt_thr(dict_of(data, "thresholds").get(thr_key)) if thr_key else NA,
            ratio_text(num(row.get("yes")), num(row.get("yes_n"))),
            ratio_text(num(row.get("no")), num(row.get("no_n"))),
            fmt_count(row.get("n_alarm")),
            fmt_metric(row.get("zones_per_km_h")),
            fmt_metric(row.get("km_h")),
        ])
    thr = dict_of(data, "thresholds")
    lines = [
        f"Метод `{data.get('method', NA)}`. {data.get('note') or ''}".strip(),
        f"Пороги: far 1e-3 {fmt_thr(thr.get('far_1e-3'))}, far 3e-3 {fmt_thr(thr.get('far_3e-3'))}, "
        f"far 1e-2 {fmt_thr(thr.get('far_1e-2'))}, far_ref {fmt_thr(thr.get('far_ref'))}.",
        "",
        md_table(
            ["Точка", "Порог, если ключ есть", "yes", "no", "Тревожных зон", "Зон на км·ч", "км·ч"],
            body,
        ),
        "",
    ]
    misses = [m for m in list_of(data, "misses") if isinstance(m, dict)]
    if misses:
        lines.append("Промахи этого же прежнего выпуска:")
        lines.append("")
        lines.append(md_table(
            ["Карточка", "Файл", "Причина", "max score", "м", "ч"],
            [[m.get("id"), m.get("record"), m.get("reason") or NA, fmt_thr(m.get("max_score")),
              fmt_metric(m.get("width_m")), fmt_metric(m.get("dur_h"))] for m in misses],
        ))
        lines.append("")
    return lines


def gap_ft_tables(data):
    if not isinstance(data, dict):
        return ["_JSON промахов текущего выпуска нет._", ""]
    summary = dict_of(data, "zone_summary")
    lines = [
        f"Метод `{data.get('method', NA)}`, порог {fmt_thr(data.get('threshold'))}. {data.get('note') or ''}".strip(),
        f"Зон {fmt_count(summary.get('n'))}, горячих {fmt_count(summary.get('n_hot'))}, "
        f"холодных {fmt_count(summary.get('n_cold'))}. "
        f"Медианы: {fmt_metric(summary.get('dur_p50'))} ч, {fmt_metric(summary.get('width_p50'))} м, "
        f"|ΔT| {fmt_metric(summary.get('peak_abs_p50'))} °C. "
        f"Пересечение с unsup и эталоном {fmt_count(summary.get('both'))}, только эталон {fmt_count(summary.get('only_base'))}, "
        f"только unsup {fmt_count(summary.get('only_unsup'))}, ни с кем {fmt_count(summary.get('neither'))}.",
        "",
    ]
    misses = [m for m in list_of(data, "misses") if isinstance(m, dict)]
    lines.append(md_table(
        ["Карточка", "Файл", "Причина", "max score", "м", "ч", "Эталон, зон"],
        [[m.get("id"), m.get("record"), m.get("reason") or NA, fmt_thr(m.get("max_score")),
          fmt_metric(m.get("width_m")), fmt_metric(m.get("dur_h")), fmt_count(m.get("base_n"))] for m in misses],
    ))
    lines.append("")
    odd = [m for m in list_of(data, "hits_odd_sign") if isinstance(m, dict)]
    if odd:
        lines.append("Попадания, где знак пика не совпал с карточкой:")
        lines.append("")
        lines.append(md_table(
            ["Карточка", "Файл", "Форма", "max score", "hotness пика", "Класс CNN"],
            [[m.get("id"), m.get("record"), m.get("shape") or NA, fmt_thr(m.get("max_score")),
              fmt_metric(m.get("hot_at_max")), m.get("cnn_class") or NA] for m in odd],
        ))
        lines.append("")
    tops = [z for z in list_of(data, "top_zones") if isinstance(z, dict)][:8]
    if tops:
        lines.append("Верх списка по score. Это измерение зон, экспертное суждение, не эталон.")
        lines.append("")
        lines.append(md_table(
            ["Зона", "Файл", "Класс", "Знак", "score", "|ΔT|", "ч", "м", "unsup", "эталон"],
            [[z.get("zone_id"), z.get("file"), z.get("klass"), z.get("polarity"),
              fmt_thr(z.get("score")), fmt_metric(abs(num(z.get("peak_dT_C")) or 0) if num(z.get("peak_dT_C")) is not None else None),
              fmt_metric(z.get("dur_h")), fmt_metric(z.get("width_m")),
              fmt_count(z.get("unsup_n")), fmt_count(z.get("base_n"))] for z in tops],
        ))
        lines.append("")
    return lines


def build_auto(s1, s2, real, st1, st2, st_real, trains=None, e2e_rows=None,
               cand=None, cand_loaded=None, v10_ok=None, v10_status="missing",
               v9=None, v9_status="missing", unsup=None, unsup_status="missing",
               side=None):
    out = {
        "s1_status": status_word(st1),
        "s1_version": (str(s1.get("version")) if isinstance(s1, dict) and s1.get("version") is not None else NA),
        "s1_far_ref": fmt_far(s1.get("far_ref") if isinstance(s1, dict) else None),
        "s2_status": status_word(st2),
        "s2_version": (str(s2.get("version")) if isinstance(s2, dict) and s2.get("version") is not None else NA),
        "s2_n_dev": fmt_count(s2.get("n_dev") if isinstance(s2, dict) else None),
        "s2_n_test": fmt_count(s2.get("n_test") if isinstance(s2, dict) else None),
        "s2_class_counts": class_count_str(s2),
        "s2_deployed_winner": NA,
        "s2_deployed_cnn_weight": NA,
        "real_status": status_word(st_real),
        "real_op": NA,
        "real_threshold": NA,
    }
    test = dict_of(s1, "test")
    unet = test.get("unet@far_ref")
    base = test.get("base_median_x3_spatial_z@far_ref")
    zones = test.get("base_pipeline_zones")
    ui, uf, uw = pack_detector(unet, "s1_unet", out)
    bi, bf, bw = pack_detector(base, "s1_base", out)
    pack_detector(zones, "s1_zones", out)
    out["s1_unet_delta_iou"] = fmt_delta(ui, bi)
    out["s1_unet_delta_f1"] = fmt_delta(uf, bf)
    out["s1_unet_delta_weak"] = fmt_delta(uw, bw)

    best_name, best_iou, best_weak, best_ratio = NA, NA, NA, NA
    pool = []
    for method, op, _key, block in stage1_rows(test):
        if op != "far_ref":
            continue
        iou = num(pick(block, "iou"))
        ratio = num(pick(block, "far_ratio"))
        if iou is None or ratio is None or ratio > FAR_MATCH:
            continue
        weak = num(pick(weak_block(block) or {}, "recall"))
        pool.append((iou, -rank_in(METHOD_ORDER, method), method, weak, ratio))
    if pool:
        iou, _r, method, weak, ratio = max(pool)
        best_name, best_iou = method_ru(method), fmt_metric(iou)
        best_weak, best_ratio = fmt_metric(weak), fmt_metric(ratio)
    out["s1_best_matched_name"] = best_name
    out["s1_best_matched_iou"] = best_iou
    out["s1_best_matched_weak"] = best_weak
    out["s1_best_matched_far_ratio"] = best_ratio
    pack_selection(s1, out)
    winner_key = out.get("s1_winner_key")
    out["s1_winner_members"] = MEMBERS.get(winner_key, NA) if winner_key and winner_key != NA else NA
    lead_src = cand if isinstance(cand, dict) else s1
    pack_leaders(dict_of(lead_src, "val"), out)
    out["s1_base_stable"] = baseline_iou_stable(cand_loaded or [])
    pack_train(trains, out)
    pack_e2e(e2e_rows if e2e_rows is not None else [], out)
    pack_shift_auto(out, winner_key, s1, v10_ok or [], v10_status, v9, v9_status, unsup, unsup_status)

    gkf = dict_of(s2, "groupkfold")
    te = dict_of(s2, "test")
    for src, prefix_root, names in (
        (te, "s2_test", ("cnn_eventnet", "lgbm_features", "ensemble_cnn_lgbm", "rules_haklolarp")),
        (gkf, "s2_gkf", ("cnn_eventnet", "cnn_image_only", "lgbm_features", "rules_haklolarp")),
    ):
        short = {
            "cnn_eventnet": "eventnet",
            "cnn_image_only": "image",
            "lgbm_features": "lgbm",
            "ensemble_cnn_lgbm": "ensemble",
            "rules_haklolarp": "rules",
        }
        for name in names:
            block = None
            if isinstance(src, dict):
                for alias in STAGE2_ALIASES.get(name, (name,)):
                    candidate = src.get(alias)
                    if isinstance(candidate, dict) and pick(candidate, "macro_f1") is not None:
                        block = candidate
                        break
            pack_classifier(block, f"{prefix_root}_{short[name]}", out)

    deployed = dict_of(s2, "deployed")
    if deployed.get("winner") is not None:
        out["s2_deployed_winner"] = str(deployed.get("winner"))
    out["s2_deployed_cnn_weight"] = fmt_metric(deployed.get("cnn_weight"))

    files = dict_of(real, "files")
    cards = list_of(real, "cards")
    if isinstance(real, dict):
        if real.get("op") is not None:
            out["real_op"] = str(real.get("op"))
        out["real_threshold"] = fmt_thr(real.get("threshold"))
    ious = []
    sums = {k: 0.0 for k in ("zones_cnn", "zones_alarm", "zones_base", "dead", "hours", "km")}
    seen = {k: 0 for k in sums}
    for name, rec in files.items():
        if not isinstance(rec, dict):
            continue
        mapping = {
            "zones_cnn": rec.get("n_zones_cnn"),
            "zones_alarm": rec.get("n_alarm_zones_cnn"),
            "zones_base": rec.get("n_zones_base"),
            "dead": rec.get("dead_alarms"),
            "hours": rec.get("hours"),
            "km": rec.get("km"),
        }
        for key, val in mapping.items():
            v = num(val)
            if v is None:
                continue
            sums[key] += v
            seen[key] += 1
        iv = num(rec.get("iou_cnn_vs_base"))
        if iv is not None:
            ious.append((iv, str(name)))
    def summed(key, formatter):
        return formatter(sums[key]) if seen[key] else NA
    n_files = sum(isinstance(v, dict) for v in files.values())
    out["real_n_files"] = str(n_files) if st_real == "ok" or n_files else NA
    out["real_hours"] = summed("hours", lambda v: f"{v:.1f}")
    out["real_km"] = summed("km", lambda v: f"{v:.2f}")
    out["real_zones_cnn"] = summed("zones_cnn", fmt_count)
    out["real_zones_alarm"] = summed("zones_alarm", fmt_count)
    out["real_zones_base"] = summed("zones_base", fmt_count)
    out["real_dead_alarms"] = summed("dead", fmt_count)
    if ious:
        mean = sum(v for v, _ in ious) / len(ious)
        lo = min(ious, key=lambda it: (it[0], natural_key(it[1])))
        hi = max(ious, key=lambda it: (it[0], natural_key(it[1])))
        out["real_iou_mean"] = fmt_metric(mean)
        out["real_iou_min"] = fmt_metric(lo[0])
        out["real_iou_min_file"] = lo[1]
        out["real_iou_max"] = fmt_metric(hi[0])
        out["real_iou_max_file"] = hi[1]
    else:
        out["real_iou_mean"] = out["real_iou_min"] = out["real_iou_max"] = NA
        out["real_iou_min_file"] = out["real_iou_max_file"] = NA
    n_cards = len(cards) if st_real == "ok" or cards else (0 if st_real == "ok" else None)
    outside = alarm = comparable = agree = 0
    for card in cards:
        if not isinstance(card, dict):
            continue
        if card.get("status") == "box_outside_record":
            outside += 1
            continue
        n_alarm = num(card.get("cnn_n_alarm_zones"))
        flagged = None if n_alarm is None else n_alarm > 0
        if flagged:
            alarm += 1
        flag = presence_flag(card.get("presence"))
        if flag is not None and flagged is not None:
            comparable += 1
            if flag == flagged:
                agree += 1
    out["real_n_cards"] = fmt_count(n_cards) if n_cards is not None else NA
    out["real_cards_alarm"] = fmt_count(alarm) if cards else ( "0" if st_real == "ok" else NA)
    out["real_cards_outside"] = fmt_count(outside) if cards else ("0" if st_real == "ok" else NA)
    out["real_cards_comparable"] = fmt_count(comparable) if cards else ("0" if st_real == "ok" else NA)
    out["real_cards_agree"] = f"{agree} из {comparable}" if comparable else NA
    buckets = card_buckets(cards) if cards else None
    for key, prefix in (("yes", "real_yes"), ("no", "real_no"), ("unclear", "real_unclear")):
        if not buckets:
            out[f"{prefix}_cnn"] = out[f"{prefix}_base"] = out[f"{prefix}_n"] = NA
            continue
        node = buckets[key]
        out[f"{prefix}_n"] = str(node["n"]) if st_real == "ok" or cards else NA
        out[f"{prefix}_cnn"] = ratio_text(node["cnn"], node["n"]) if node["n"] else ("0/0" if st_real == "ok" else NA)
        out[f"{prefix}_base"] = ratio_text(node["base"], node["n"]) if node["n"] else ("0/0" if st_real == "ok" else NA)
    out["real_zones_per_file"] = zones_per_file(files) if files else NA
    winner_thr = None
    if isinstance(s1, dict):
        winner_thr = pick(block_at(dict_of(s1, "test"), out.get("s1_winner_key"), SELECT_OP) or {}, "threshold")
    real_thr = real.get("threshold") if isinstance(real, dict) else None
    if num(winner_thr) is None or num(real_thr) is None:
        out["real_thr_match"] = NA
    else:
        out["real_thr_match"] = "да" if close(winner_thr, real_thr) else "нет"
    pack_side(out, side, s1, unsup)
    return {k: clean_auto(v) for k, v in out.items()}


def clean_auto(value):
    s = str(value).replace("\r", " ").replace("\n", " ")
    s = s.replace("<!--", "").replace("-->", "")
    return re.sub(r"\s+", " ", s).strip() or NA


def fill_note(text, values):
    text = SPAN_RE.sub(lambda m: f"<!-- AUTO:{m.group(1)} -->", text)

    def wrap(match):
        key = match.group(1)
        val = values.get(key, NA)
        return f"<!-- AUTO:{key} -->{clean_auto(val)}<!-- /AUTO:{key} -->"

    text, n = BARE_RE.subn(wrap, text)
    filled_na = sum(
        1 for key in re.findall(r"<!-- AUTO:([A-Za-z0-9_.\-]+) -->", text) if values.get(key, NA) == NA
    )
    return text, n, filled_na


def banner(title, path: Path, st: str):
    lines = [f"## {title}", ""]
    when = f", изменён {mtime(path)}" if path.is_file() else ""
    lines.append(f"Источник: `{rel(path)}` ({status_word(st)}{when}).")
    if st == "missing":
        lines.append("")
        lines.append("Файла ещё нет. Таблицы ниже появятся после прогона оценки. Повторный `scripts/make_report.py` их заполнит.")
    elif st.startswith("invalid"):
        lines.append("")
        lines.append(f"Файл не разобран ({st}). Цифры не выдуманы и показаны как «н/д».")
    return lines


def detector_tables(rows):
    main, extra = [], []
    for method, op, _key, block in rows:
        name = method_ru(method)
        point = "зоны после фильтров" if method == "base_pipeline_zones" else op_ru(op)
        thr = "3.5 внутри эталона" if method == "base_pipeline_zones" else fmt_thr(pick(block, "threshold"))
        weak = weak_block(block) or {}
        main.append([
            name, point, thr, fmt_far(pick(block, "far")), fmt_metric(pick(block, "far_ratio")),
            fmt_metric(pick(block, "precision")), fmt_metric(pick(block, "recall")),
            fmt_metric(pick(block, "f1")), fmt_metric(pick(block, "iou")), fmt_ci(pick(block, "iou_ci95")),
        ])
        extra.append([
            name, point,
            fmt_metric(sub_metric(block, "hot", "iou")),
            fmt_metric(sub_metric(block, "cold", "iou")),
            fmt_metric(pick(weak, "recall")),
            fmt_count(pick(weak, "n")),
            fmt_metric(pick(weak, "pixel_recall")),
            fmt_metric(sub_metric(block, "events_all", "recall")),
            fmt_metric(pick(block, "nuisance_alarm_frac")),
        ])
    t1 = md_table(
        ["Метод", "Рабочая точка", "Порог", "FAR", "FAR/FAR_ref", "Precision", "Recall", "F1", "IoU", "IoU, 95% ДИ"],
        main,
    )
    t2 = md_table(
        ["Метод", "Рабочая точка", "IoU нагрев", "IoU охлаждение", "Recall событий ≤2 °C",
         "n ≤2 °C", "Пикс. recall ≤2 °C", "Recall всех событий", "Доля помех в тревоге"],
        extra,
    )
    return t1, t2


def nuisance_ignored_cells(block):
    node = block.get("nuisance_ignored") if isinstance(block, dict) else None
    if not isinstance(node, dict):
        return NA, NA
    return fmt_metric(pick(node, "precision")), fmt_metric(pick(node, "iou"))


def ignored_and_zones_table(rows):
    body = []
    for method, op, _key, block in rows:
        name = method_ru(method)
        point = "зоны после фильтров" if method == "base_pipeline_zones" else op_ru(op)
        prec, iou = nuisance_ignored_cells(block)
        zones = block.get("zones") if isinstance(block, dict) else None
        zones = zones if isinstance(zones, dict) else {}
        body.append([
            name, point, prec, iou,
            fmt_metric(pick(zones, "false_zones_per_km_h")),
            fmt_count(pick(zones, "n_zones")),
            fmt_count(pick(zones, "n_false")),
        ])
    return md_table(
        ["Метод", "Рабочая точка", "Precision, помехи вне счёта", "IoU, помехи вне счёта",
         "Ложные зоны на км·ч", "Зон", "Ложных зон"],
        body,
    )


def ap_table(threshold_free):
    if not isinstance(threshold_free, dict) or not threshold_free:
        return "_Блока threshold_free в JSON нет._"
    keys = sorted(threshold_free, key=lambda name: (rank_in(METHOD_ORDER, name), str(name)))
    body = []
    for name in keys:
        node = threshold_free.get(name)
        if not isinstance(node, dict):
            body.append([method_ru(name), NA])
            continue
        body.append([method_ru(name), fmt_metric(pick(node, "pixel_average_precision"))])
    return md_table(["Метод", "Средняя точность пикселей"], body)


def matched_table(matched):
    if not isinstance(matched, dict) or not matched:
        return "_Блока test_far_matched в JSON нет._"
    keys = sorted(matched, key=lambda name: (rank_in(METHOD_ORDER, name), str(name)))
    body = []
    for name in keys:
        block = matched.get(name)
        if not isinstance(block, dict):
            continue
        prec, iou = nuisance_ignored_cells(block)
        body.append([
            method_ru(name),
            fmt_thr(pick(block, "threshold")),
            fmt_far(pick(block, "far")),
            fmt_metric(pick(block, "precision")),
            fmt_metric(pick(block, "recall")),
            fmt_metric(pick(block, "f1")),
            fmt_metric(pick(block, "iou")),
            prec,
            iou,
        ])
    return md_table(
        ["Метод", "Порог на test", "FAR", "Precision", "Recall", "F1", "IoU",
         "Precision, помехи вне счёта", "IoU, помехи вне счёта"],
        body,
    )


def delta_table(rows):
    base = None
    for method, op, _key, block in rows:
        if method == "base_median_x3_spatial_z" and op == "far_ref":
            base = block
            break
    if base is None:
        return "_Эталона на точке «FAR эталона» в JSON нет, разности не посчитаны._"
    bi = num(pick(base, "iou"))
    bw = num(pick(weak_block(base) or {}, "recall"))
    body = []
    for method, op, _key, block in rows:
        if op != "far_ref":
            continue
        iou = num(pick(block, "iou"))
        weak = num(pick(weak_block(block) or {}, "recall"))
        body.append([
            method_ru(method),
            fmt_metric(iou),
            fmt_delta(iou, bi),
            fmt_metric(weak),
            fmt_delta(weak, bw),
            fmt_metric(pick(block, "far_ratio")),
            fmt_far(pick(block, "far")),
        ])
    return md_table(
        ["Метод", "IoU", "IoU − эталон", "Recall ≤2 °C", "Δ recall ≤2 °C", "FAR/FAR_ref", "FAR"],
        body,
    )


def val_table(val, rows):
    if not isinstance(val, dict) or not val:
        return "_Блока val в JSON нет. По протоколу он должен содержать только far, f1, iou и far_ratio._"
    body = []
    seen = set()
    ordered = [(m, o, k, b) for m, o, k, b in rows if o]
    for method, op, key, block in ordered:
        seen.add(key)
        vb = val.get(key) if isinstance(val.get(key), dict) else {}
        body.append([
            method_ru(method), op_ru(op),
            fmt_far(pick(vb, "far")), fmt_metric(pick(vb, "f1")), fmt_metric(pick(vb, "iou")),
            fmt_metric(pick(vb, "far_ratio")),
            fmt_far(pick(block, "far")), fmt_metric(pick(block, "iou")),
        ])
    for key, vb in val.items():
        if key in seen or not isinstance(vb, dict):
            continue
        method, op = (str(key).split("@", 1) + [""])[:2] if "@" in str(key) else (str(key), "")
        body.append([
            method_ru(method), op_ru(op) if op else "без точки",
            fmt_far(pick(vb, "far")), fmt_metric(pick(vb, "f1")), fmt_metric(pick(vb, "iou")),
            fmt_metric(pick(vb, "far_ratio")), NA, NA,
        ])
    return md_table(
        ["Метод", "Рабочая точка", "FAR val", "F1 val", "IoU val", "FAR/FAR_ref val", "FAR test", "IoU test"],
        body,
    )


def threshold_table(thr):
    if not isinstance(thr, dict) or not thr:
        return "_Порогов в JSON нет._"
    ops = []
    for node in thr.values():
        if isinstance(node, dict):
            for op in node:
                if op not in ops:
                    ops.append(op)
    ops = sorted(ops, key=lambda o: (rank_in(OP_ORDER, o), o))
    methods = sorted(thr.keys(), key=lambda m: (rank_in(METHOD_ORDER, m), m))
    body = []
    for method in methods:
        node = thr.get(method)
        if not isinstance(node, dict):
            continue
        body.append([method_ru(method)] + [fmt_thr(node.get(op)) for op in ops])
    return md_table(["Метод"] + [op_ru(o) for o in ops], body)


def grouped_by_op(rows):
    """Все методы одной рабочей точки — в одной таблице, в порядке методов."""
    buckets = {}
    for row in rows:
        if row[0] == "base_pipeline_zones":
            continue
        buckets.setdefault(row[1], []).append(row)
    ops = sorted(buckets, key=lambda op: (rank_in(OP_ORDER, op), str(op)))
    return [(op, buckets[op]) for op in ops]


def template_sections(rows):
    lines = []
    templates = collect_keys(rows, "by_template", TEMPLATE_ORDER)
    ious = collect_keys(rows, "iou_by_template", TEMPLATE_ORDER)
    both = [k for k in ("F1_dt30", "F1_dt240") if k in templates]
    if not templates and not ious:
        return ["Разбивки по шаблонам в JSON нет.", ""]
    lines.append("Шаблоны F1–F6 повторяют геометрию файлов 1–6.csv. F1 разрезан на два режима шага по времени: dt≈30 с и dt≈240 с.")
    lines.append("В ячейке событийной таблицы: доля событий, у которых порог покрыл не меньше 30% пикселей маски; через «пикс.» — доля пикселей маски; в скобках — число событий.")
    if len(both) == 2:
        lines.append("Колонка «F1, оба режима» — тот же событийный recall, взвешенный по числу событий двух режимов F1. Это не пиксельный F1 и не IoU.")
    lines.append("Пиксельный IoU ниже — IoU по всем валидным клеткам записей этого шаблона, а не только внутри бокса события.")
    lines.append("")
    for op, group in grouped_by_op(rows):
        lines.append(f"### Шаблоны, точка «{op_ru(op)}»")
        lines.append("")
        if templates:
            headers = ["Метод"] + [TEMPLATE_RU.get(t, t) for t in templates]
            if len(both) == 2:
                headers.append("F1, оба режима")
            body = []
            for method, _op, _k, block in group:
                node = block.get("by_template") if isinstance(block.get("by_template"), dict) else {}
                row = [method_ru(method)] + [event_cell(node.get(t)) for t in templates]
                if len(both) == 2:
                    row.append(fmt_metric(weighted_recall(block, both)))
                body.append(row)
            lines.append(md_table(headers, body))
            lines.append("")
        if ious:
            headers = ["Метод"] + [TEMPLATE_RU.get(t, t) for t in ious]
            node_rows = []
            for method, _op, _k, block in group:
                node = block.get("iou_by_template") if isinstance(block.get("iou_by_template"), dict) else {}
                node_rows.append([method_ru(method)] + [iou_cell(node.get(t)) for t in ious])
            lines.append("Пиксельный IoU по записям шаблона.")
            lines.append("")
            lines.append(md_table(headers, node_rows))
            lines.append("")
    return lines


def label_sections(rows):
    labels = collect_keys(rows, "by_label", LABEL_ORDER)
    lines = ["### Метки генератора", ""]
    if not labels:
        lines.append("Разбивки по меткам в JSON нет.")
        lines.append("")
        return lines
    lines.append("Событийный recall по метке `labels.csv` (только kind=event; бокс без пикселей маски в оценку не входит).")
    gloss = ", ".join(f"`{k}` — {LABEL_RU.get(k, k)}" for k in labels)
    lines.append(gloss + ".")
    lines.append("")
    for op, group in grouped_by_op(rows):
        lines.append(f"#### Точка «{op_ru(op)}»")
        lines.append("")
        headers = ["Метод"] + [str(l) for l in labels]
        body = []
        for method, _op, _k, block in group:
            node = block.get("by_label") if isinstance(block.get("by_label"), dict) else {}
            body.append([method_ru(method)] + [event_cell(node.get(l)) for l in labels])
        lines.append(md_table(headers, body))
        lines.append("")
    return lines


def amp_sections(rows):
    keys = collect_keys(rows, "by_amp", AMP_ORDER)
    lines = ["### Амплитуда события", ""]
    if not keys:
        lines.append("Разбивки по амплитуде в JSON нет. Recall событий с амплитудой ≤2 °C, если он есть, стоит в общей таблице слабых событий.")
        lines.append("")
        return lines
    lines.append("Амплитуда — `amp_obs_C` после аппаратной функции и усреднения трассы. Бин `≤2 °C` совпадает с определением слабого события в `eval_stage1.py`.")
    lines.append("")
    for op, group in grouped_by_op(rows):
        lines.append(f"#### Точка «{op_ru(op)}»")
        lines.append("")
        headers = ["Метод"] + [AMP_RU.get(k, k) for k in keys]
        body = []
        for method, _op, _k, block in group:
            node = block.get("by_amp") if isinstance(block.get("by_amp"), dict) else {}
            body.append([method_ru(method)] + [event_cell(node.get(k)) for k in keys])
        lines.append(md_table(headers, body))
        lines.append("")
    return lines


def model_names(part):
    if not isinstance(part, dict):
        return []
    known = [m for m in MODEL_ORDER if m in part]
    rest = sorted(k for k in part if k not in known and isinstance(part.get(k), dict))
    return known + rest


def summary_classifier(part):
    body = []
    for name in model_names(part):
        block = part.get(name)
        if not isinstance(block, dict):
            continue
        body.append([
            method_ru(name),
            fmt_metric(pick(block, "macro_f1")),
            fmt_ci(pick(block, "macro_f1_ci95")),
            fmt_metric(pick(block, "coarse_macro_f1")),
            fmt_metric(pick(block, "ece")),
            fmt_metric(pick(block, "accuracy")),
        ])
    return md_table(
        ["Модель", "macro-F1", "macro-F1, 95% ДИ", "coarse macro-F1", "ECE", "accuracy"],
        body,
    )


def per_class_table(part, classes):
    names = model_names(part)
    if not names:
        return "_Нет моделей._"
    labels = list(classes) if classes else []
    if not labels:
        for name in names:
            node = pick(part.get(name), "per_class")
            if isinstance(node, dict):
                labels = list(node.keys())
                break
    support = None
    for name in names:
        cm = pick(part.get(name), "confusion")
        if isinstance(cm, list) and cm and len(cm) == len(labels):
            support = [fmt_count(sum(num(v) or 0 for v in row)) if isinstance(row, list) else NA for row in cm]
            break
    headers = ["Класс"] + (["объектов"] if support else []) + [method_ru(n) for n in names]
    body = []
    for i, label in enumerate(labels):
        row = [str(label)]
        if support:
            row.append(support[i] if i < len(support) else NA)
        for name in names:
            node = pick(part.get(name), "per_class")
            val = node.get(label) if isinstance(node, dict) else None
            row.append(fmt_metric(val))
        body.append(row)
    return md_table(headers, body)


def confusion_tables(part, classes):
    lines = []
    labels = list(classes) if classes else []
    lines.append("Строки — истинный класс, столбцы — предсказанный. Коды: "
                 + ", ".join(f"{CLASS_SHORT.get(c, c)} = {c}" for c in (labels or CLASS_ORDER))
                 + ".")
    lines.append("")
    any_cm = False
    for name in model_names(part):
        cm = pick(part.get(name), "confusion")
        if not isinstance(cm, list) or not cm:
            continue
        any_cm = True
        n = max(len(labels), len(cm), max((len(r) for r in cm if isinstance(r, list)), default=0))
        heads = []
        for i in range(n):
            raw = labels[i] if i < len(labels) else f"class_{i}"
            heads.append(CLASS_SHORT.get(raw, raw))
        body = []
        col_sum = [0.0] * n
        col_ok = [False] * n
        for i in range(n):
            raw = labels[i] if i < len(labels) else f"class_{i}"
            src = cm[i] if i < len(cm) and isinstance(cm[i], list) else []
            vals = []
            total = 0.0
            total_ok = False
            for j in range(n):
                v = num(src[j]) if j < len(src) else None
                vals.append(NA if v is None else fmt_count(v))
                if v is not None:
                    total += v
                    total_ok = True
                    col_sum[j] += v
                    col_ok[j] = True
            body.append([CLASS_SHORT.get(raw, raw)] + vals + [fmt_count(total) if total_ok else NA])
        grand = sum(col_sum) if any(col_ok) else None
        body.append(["сумма"] + [fmt_count(col_sum[j]) if col_ok[j] else NA for j in range(n)] + [fmt_count(grand) if grand is not None else NA])
        f1 = fmt_metric(pick(part.get(name), "macro_f1"))
        lines.append(f"#### {method_ru(name)} (macro-F1 {f1})")
        lines.append("")
        lines.append(md_table(["истина \\ предск."] + heads + ["сумма"], body))
        lines.append("")
    if not any_cm:
        lines.append("Матриц ошибок в этом блоке нет.")
        lines.append("")
    return lines


def fmt_time(ts):
    v = num(ts)
    if v is None:
        return NA
    if v > 1_000_000_000:
        return datetime.fromtimestamp(v, timezone.utc).strftime("%Y-%m-%d %H:%M")
    return f"{v:.0f}"


def fmt_dur(a, b):
    aa, bb = num(a), num(b)
    if aa is None or bb is None:
        return NA
    sec = abs(bb - aa)
    if sec < 90:
        return f"{sec:.0f} с"
    if sec < 3 * 3600:
        return f"{sec / 60:.0f} мин"
    return f"{sec / 3600:.2f} ч"


def fmt_meters(a, b):
    aa, bb = num(a), num(b)
    if aa is None or bb is None:
        return NA
    return f"{aa:.1f}–{bb:.1f}"


def class_hist(rec):
    node = rec.get("classes") if isinstance(rec, dict) else None
    if not isinstance(node, dict) or not node:
        return NA
    keys = [k for k in CLASS_ORDER if k in node] + [k for k in node if k not in CLASS_ORDER]
    return ", ".join(f"{k} × {fmt_count(node[k])}" for k in keys)


def real_section(real, auto=None):
    lines = []
    files = dict_of(real, "files")
    cards = list_of(real, "cards")
    skipped = [name for name, rec in files.items() if not isinstance(rec, dict)]
    files = {name: rec for name, rec in files.items() if isinstance(rec, dict)}
    if not isinstance(real, dict):
        lines.append("Сравнивать с эталоном и выводить карточки не из чего.")
        lines.append("")
        return lines
    op = real.get("op", NA)
    thr = fmt_thr(real.get("threshold"))
    lines.append(f"Рабочая точка инференса: `{op}`, порог score {thr}. Порог взят из выпуска модели, на этих файлах не подбирался.")
    lines.append("IoU — пересечение маски зон CNN и маски зон эталона на валидных клетках канонической сетки. Это согласие двух алгоритмов, не совпадение с ручной разметкой и не совпадение имён классов.")
    lines.append("Маска CNN — пиксели внутри найденных зон, где score не ниже порога. Маска эталона — зоны haklolarp, перенесённые на ту же сетку. `dead_alarms` — пиксели маски CNN вне маски valid; ненулевое значение значит, что тревога встала на обрыв или −300.")
    lines.append("Зона с классом HVAC_Diurnal или Sensor_Artifact в счётчик тревожных зон не входит: она показана оператору, но флаг alarm снят.")
    if auto:
        lines.append(
            f"Карточки presence=yes: тревога CNN {auto.get('real_yes_cnn', NA)}, эталон {auto.get('real_yes_base', NA)}. "
            f"presence=no: CNN {auto.get('real_no_cnn', NA)}, эталон {auto.get('real_no_base', NA)}. "
            f"presence=unclear в эти доли не входит: карточек {auto.get('real_unclear_n', NA)}, "
            f"с тревогой CNN {auto.get('real_unclear_cnn', NA)}, с зоной эталона {auto.get('real_unclear_base', NA)}. "
            "Попадание — число зон в заранее указанном окне больше нуля. У CNN считаются только тревожные зоны."
        )
        lines.append(f"Зоны по файлам, CNN / эталон: {auto.get('real_zones_per_file', NA)}.")
        lines.append(
            f"Порог файла совпадает с порогом победителя на точке FAR = 1e-3: {auto.get('real_thr_match', NA)}."
        )
    if skipped:
        lines.append("В `files` пропущены записи, которые не являются объектом: " + ", ".join(str(s) for s in sorted(skipped, key=natural_key)) + ".")
    lines.append("")
    body = []
    names = sorted(files, key=natural_key)
    tz_any = False
    for name in names:
        rec = files[name] if isinstance(files[name], dict) else {}
        tz = rec.get("tz_missing")
        if tz is True:
            tz_any = True
        body.append([
            name,
            fmt_count(rec.get("hours")),
            fmt_metric(rec.get("km")),
            "да" if tz is True else ("нет" if tz is False else NA),
            fmt_count(rec.get("n_zones_cnn")),
            fmt_count(rec.get("n_alarm_zones_cnn")),
            fmt_count(rec.get("n_zones_base")),
            fmt_pct(rec.get("alarm_fraction_cnn")),
            fmt_pct(rec.get("alarm_fraction_base")),
            fmt_metric(rec.get("iou_cnn_vs_base")),
            fmt_count(rec.get("dead_alarms")),
            class_hist(rec),
        ])
    lines.append(md_table(
        ["Файл", "часы", "км", "пояс не задан", "зоны CNN", "из них тревога", "зоны эталона",
         "доля тревоги CNN, %", "доля тревоги эталона, %", "IoU масок", "dead_alarms", "классы CNN"],
        body,
    ))
    lines.append("")
    if tz_any:
        lines.append("У части файлов в метке времени нет пояса. Загрузчик читает такие метки как UTC, поэтому абсолютные часы зон могут быть сдвинуты относительно карточки оператора.")
        lines.append("")
    for name in names:
        rec = files[name] if isinstance(files[name], dict) else {}
        zones = rec.get("zones")
        lines.append(f"### Зоны {name}")
        lines.append("")
        sha = rec.get("sha256")
        shape = rec.get("shape")
        grid_shape = rec.get("grid_shape")
        bits = []
        if isinstance(sha, str) and sha:
            bits.append(f"sha256 `{sha[:12]}`")
        if isinstance(shape, list):
            bits.append("сырой размер " + "×".join(fmt_count(v) for v in shape))
        if isinstance(grid_shape, list):
            bits.append("сетка " + "×".join(fmt_count(v) for v in grid_shape))
        if bits:
            lines.append(", ".join(bits) + ".")
            lines.append("")
        if not isinstance(zones, list):
            lines.append("Список зон в JSON не записан. Счётчики в таблице выше взяты из полей файла, если они были.")
            lines.append("")
            continue
        if not zones:
            lines.append("Зон нет.")
            lines.append("")
            continue
        ordered = sorted(
            (z for z in zones if isinstance(z, dict)),
            key=lambda z: (num(z.get("start_m")) or 0, num(z.get("t_start_s")) or 0),
        )
        zrows = []
        for i, z in enumerate(ordered, 1):
            pol = z.get("polarity")
            pol_ru = {"hot": "нагрев", "cold": "охлаждение"}.get(pol, pol if pol is not None else NA)
            alarm = z.get("alarm")
            alarm_ru = "да" if alarm is True else ("нет" if alarm is False else NA)
            zrows.append([
                str(i),
                z.get("class", NA),
                alarm_ru,
                pol_ru,
                fmt_meters(z.get("start_m"), z.get("end_m")),
                fmt_time(z.get("t_start_s")),
                fmt_dur(z.get("t_start_s"), z.get("t_end_s")),
                fmt_metric(z.get("score")),
                fmt_metric(z.get("peak_dT_C"), 2),
                fmt_count(z.get("n_px")),
            ])
        lines.append(md_table(
            ["#", "класс", "тревога", "полярность", "x, м", "начало UTC", "длительность", "score", "пик |ΔT|, °C", "пикселей"],
            zrows,
        ))
        lines.append("")
    lines.append("### Карточки оператора")
    lines.append("")
    if not cards:
        lines.append("Карточек в JSON нет. `run_real.py` дописывает их, если при запуске переданы jsonl-файлы окон.")
        lines.append("")
        return lines
    lines.append("Карточка — заранее указанное окно (файл, интервал метров и времени). Модель его не выбирала и на нём не училась. Колонка «карточка: событие есть» заполняется только если поле presence читается как да/нет.")
    lines.append("")
    crows = []
    for card in cards:
        if not isinstance(card, dict):
            continue
        outside = card.get("status") == "box_outside_record"
        flag = presence_flag(card.get("presence"))
        flag_s = {True: "да", False: "нет"}.get(flag, NA)
        pols = card.get("base_polarities")
        if isinstance(pols, list):
            pols_s = ", ".join(str(p) for p in pols) or "—"
        else:
            pols_s = NA if pols is None else str(pols)
        crows.append([
            card.get("id", NA),
            card.get("record", NA),
            "да" if outside else "нет",
            flag_s,
            NA if outside else fmt_metric(card.get("cnn_max_score")),
            NA if outside else fmt_pct(card.get("cnn_frac_flagged")),
            NA if outside else fmt_count(card.get("cnn_n_zones")),
            NA if outside else fmt_count(card.get("cnn_n_alarm_zones")),
            NA if outside else (card.get("cnn_dominant_class") or "—"),
            NA if outside else (card.get("cnn_dominant_polarity") or "—"),
            NA if outside else fmt_count(card.get("base_n_zones")),
            NA if outside else pols_s,
            card.get("shape", ""),
            card.get("notes", ""),
        ])
    lines.append(md_table(
        ["id", "файл", "окно вне записи", "карточка: событие есть", "max score CNN", "доля флага CNN, %",
         "зоны CNN", "тревожных", "класс CNN", "полярность", "зоны эталона", "полярности эталона", "форма", "заметка"],
        crows,
    ))
    lines.append("")
    return lines


def headline(auto):
    lines = ["## Главные цифры", ""]
    if auto.get("s1_status") != "есть":
        lines.append(f"- Этап 1: {auto.get('s1_status', NA)}.")
    else:
        lines.append(
            f"- Этап 1. Победитель по val IoU на точке FAR = 1e-3: {auto.get('s1_winner_name', NA)} "
            f"(val IoU {auto.get('s1_winner_val_iou', NA)}). "
            f"На test той же точки: IoU {auto.get('s1_winner_iou', NA)}, "
            f"F1 {auto.get('s1_winner_f1', NA)}, precision {auto.get('s1_winner_precision', NA)}, "
            f"recall {auto.get('s1_winner_recall', NA)}, FAR {auto.get('s1_winner_far', NA)}, "
            f"recall ≤2 °C {auto.get('s1_winner_weak_recall', NA)}. "
            f"Test в выборе не участвует."
        )
        lead_n = auto.get("s1_lead_n", NA)
        lead_word = "метод" if lead_n == "1" else "методов"
        lines.append(
            f"- Лидеры val в полосе {LEAD_GAP:.3f} от максимума: "
            f"{auto.get('s1_lead_val_min', NA)}–{auto.get('s1_lead_val_max', NA)} "
            f"({lead_n} {lead_word}). "
            f"Второй — {auto.get('s1_lead_second', NA)} "
            f"(val IoU {auto.get('s1_lead_second_iou', NA)}), разрыв {auto.get('s1_lead_gap', NA)}. "
            f"95% интервал IoU победителя на test {auto.get('s1_winner_iou_ci', NA)} "
            f"описывает test, не разность val."
        )
        if auto.get("v10_status") == "есть":
            if auto.get("v10_winner_iou", NA) == NA:
                lines.append(
                    f"- v10 test, порог с val v8, без перекалибровки ({auto.get('v10_refit', NA)}). "
                    f"Для текущего победителя файла v10 нет. Эталон |Z|: IoU {auto.get('v10_base_iou', NA)}."
                )
            else:
                lines.append(
                    f"- v10 test, порог с val v8, без перекалибровки ({auto.get('v10_refit', NA)}). "
                    f"Тот же детектор: IoU {auto.get('v10_winner_iou', NA)}, "
                    f"F1 {auto.get('v10_winner_f1', NA)}. Эталон |Z|: IoU {auto.get('v10_base_iou', NA)}."
                )
        else:
            lines.append(f"- v10 test: {auto.get('v10_status', NA)}.")
        if auto.get("v9_status") == "есть":
            lines.append(
                f"- v9 test, пороги файла. Совпадение порога FAR = 1e-3 с v8: {auto.get('v9_thr_match', NA)}. "
                f"{auto.get('v9_summary', NA)}."
            )
        else:
            lines.append(f"- v9 test: {auto.get('v9_status', NA)}.")
        lines.append(
            f"- Эталон |Z| на test при FAR = 1e-3: IoU {auto.get('s1_test_iou_base_1e3', NA)}. "
            f"Диагностика точки «FAR эталона» (FAR_ref = {auto.get('s1_far_ref', NA)}), "
            f"лучший IoU на test среди методов с FAR ≤ {FAR_MATCH:.1f}×FAR_ref: "
            f"{auto.get('s1_best_matched_name')} (IoU {auto.get('s1_best_matched_iou')}). "
            f"Эта строка победителя не назначает."
        )
    if auto.get("s2_status") != "есть":
        lines.append(f"- Этап 2: {auto.get('s2_status', NA)}.")
    else:
        lines.append(
            f"- Этап 2, test (объектов {auto.get('s2_n_test')}). "
            f"EventNet macro-F1 {auto.get('s2_test_eventnet_macro_f1')} "
            f"(95% ДИ {auto.get('s2_test_eventnet_ci')}), "
            f"coarse {auto.get('s2_test_eventnet_coarse')}, ECE {auto.get('s2_test_eventnet_ece')}. "
            f"LightGBM {auto.get('s2_test_lgbm_macro_f1')}, "
            f"ансамбль {auto.get('s2_test_ensemble_macro_f1')}, "
            f"правила {auto.get('s2_test_rules_macro_f1')}."
        )
    if auto.get("real_status") != "есть":
        lines.append("- Реальные файлы: " + auto.get("real_status", NA) + ".")
    else:
        lines.append(
            f"- Реальные файлы: {auto.get('real_n_files')} шт., "
            f"среднее IoU масок с эталоном {auto.get('real_iou_mean')} "
            f"(мин. {auto.get('real_iou_min')} на {auto.get('real_iou_min_file')}, "
            f"макс. {auto.get('real_iou_max')} на {auto.get('real_iou_max_file')}). "
            f"Тревожных зон CNN {auto.get('real_zones_alarm')}, зон эталона {auto.get('real_zones_base')}. "
            f"Карточки yes: CNN {auto.get('real_yes_cnn')}, эталон {auto.get('real_yes_base')}; "
            f"no: CNN {auto.get('real_no_cnn')}, эталон {auto.get('real_no_base')}."
        )
    lines.append("")
    return lines


def how_to_read():
    return [
        "## Как читать метрики",
        "",
        "FAR — доля клеток фона, которые метод пометил тревогой. Фон на синтетике: валидная клетка, без события и без помехи. Precision — среди тревожных клеток доля клеток события. Recall — доля клеток события, попавших в тревогу. F1 — их гармоническое среднее. IoU — площадь пересечения масок, делённая на площадь объединения.",
        "",
        "IoU нагрева считает предсказание горячим, если канал «нагрев минус охлаждение» положительный, и сравнивает его с событиями маски 1–4. IoU охлаждения — противоположный знак против маски 5–6. Это пиксельное перекрытие по знаку, а не F1 классификатора этапа 2.",
        "",
        "Recall событий — доля размеченных событий, у которых порог покрыл не меньше 30% пикселей маски. Слабое событие: наблюдаемая амплитуда ≤2 °C. Доля помех в тревоге — доля клеток помехи (не события), которые всё же получили тревогу.",
        "",
        "Precision и IoU «помехи вне счёта» (`nuisance_ignored`) считаются после того, как клетки помехи вынуты из подсчёта: помеха не становится ни верной тревогой, ни ложной. Обычные precision и IoU этого не делают.",
        "",
        "Ложная зона — связная зона этапа 1, в которой меньше 10% тревожных валидных клеток пересекаются с маской события. «Ложные зоны на км·ч» — число таких зон, делённое на длину записи в километрах, умноженную на длительность в часах.",
        "",
        "Средняя точность пикселей (`threshold_free.pixel_average_precision`) — average precision score против маски события. Порог для неё не выбирается. В оценке она считается по подвыборке до 40 000 валидных клеток на каждую запись test.",
        "",
        "Таблица «FAR, совпавший с эталоном на test» (`test_far_matched`) подбирает порог уже на test, чтобы FAR совпал с долей фона test, где |Z| ≥ 3,5. Это диагностика одинакового FAR. Модель по ней не выбирают: рабочий порог по-прежнему ставится на val.",
        "",
        "Таблица «все методы» сравнивает детекторы только в точке FAR = 1e-3. Победитель — максимум IoU в блоке val на этой точке. При равенстве берётся метод раньше в списке отчёта. Столбцы test — итоговые цифры уже выбранной точки. Гистерезис (зерно при пороге FAR = 1e-3, рост связной области до порога FAR = 1e-2) и сама точка FAR = 1e-2 в выбор не входят.",
        "",
        "Сквозная оценка берёт зоны детектора на test, отдаёт их этапу 2 и сверяет боксы с метками при IoU не ниже 0.1. macro-F1 считается по семи классам. Precision тревоги смотрит зоны, чей класс не HVAC_Diurnal и не Sensor_Artifact.",
        "",
        "95% интервал IoU — бутстрэп целых записей (в коде метрик 1000 повторов, seed 0). Интервал macro-F1 этапа 2 — такой же бутстрэп записей, 300 повторов.",
        "",
        "macro-F1 — среднее F1 по классам, редкий класс весит столько же, сколько частый. coarse macro-F1 — то же на четырёх грубых группах: нагрев, охлаждение, естественный цикл, артефакт прибора. ECE — среднее расхождение уверенности и доли верных ответов по 10 корзинам максимальной вероятности. Меньше — лучше калибровка.",
        "",
        "Десятичная точка, как в исходных JSON. «н/д» значит, что ключа в файле не было.",
        "",
    ]


def stage1_banner(items):
    lines = ["## Этап 1. Бинарная детекция", ""]
    if len(items) <= 1:
        path, status = items[0]
        when = f", изменён {mtime(path)}" if path.is_file() else ""
        lines.append(f"Источник: `{rel(path)}` ({status_word(status)}{when}).")
    else:
        lines.append("Источники этапа 1 в порядке чтения. Одинаковый ключ метода берётся из более позднего файла.")
        lines.append("")
        for path, status in items:
            when = f", изменён {mtime(path)}" if path.is_file() else ""
            lines.append(f"- `{rel(path)}` ({status_word(status)}{when})")
    if all(status == "missing" for _path, status in items):
        lines.append("")
        lines.append("Файла ещё нет. Таблицы ниже появятся после прогона оценки. Повторный `scripts/make_report.py` их заполнит.")
    elif any(str(status).startswith("invalid") for _path, status in items):
        lines.append("")
        lines.append("Часть файлов не разобрана. Цифры из них не подставлены.")
    return lines


def candidate_section(auto, cand, cand_items):
    lines = ["### Кандидаты раундов 4–7", ""]
    if cand_items:
        lines.append("Источники таблицы, в порядке чтения. Поздний файл перекрывает тот же ключ. В выбор победителя по-прежнему входит весь слитый этап 1, не только эта таблица.")
        lines.append("")
        for path, status in cand_items:
            when = f", изменён {mtime(path)}" if path.is_file() else ""
            lines.append(f"- `{rel(path)}` ({status_word(status)}{when})")
        lines.append("")
    else:
        lines.append("Отдельных файлов раунда 4 и `r5_`/`r6_`/`r7_` в этом запуске нет. Ниже та же слитая таблица, что и в разделе всех методов.")
        lines.append("")
    lead_n = auto.get("s1_lead_n", NA)
    lead_word = "метод" if lead_n == "1" else "методов"
    lines.append(
        f"Методы с val IoU не ниже максимума более чем на {LEAD_GAP:.3f}: "
        f"{lead_n} {lead_word} ({auto.get('s1_lead_names', NA)}), "
        f"от {auto.get('s1_lead_val_min', NA)} до {auto.get('s1_lead_val_max', NA)}. "
        f"Второй по val — {auto.get('s1_lead_second', NA)} "
        f"({auto.get('s1_lead_second_iou', NA)}), разрыв {auto.get('s1_lead_gap', NA)}. "
        f"95% интервал IoU победителя на test {auto.get('s1_winner_iou_ci', NA)} "
        f"считает разброс test по записям и не является интервалом разности на val. "
        f"Отрыв лидеров на фоне этого интервала мал. Правило выбора порог значимости не добавляет: берётся максимум val IoU."
    )
    lines.append(
        f"Эталон median_x3 + spatial Z в этих файлах на test при FAR = 1e-3 даёт один и тот же IoU: {auto.get('s1_base_stable', NA)}."
    )
    lines.append("")
    lines.append(all_methods_table(cand if isinstance(cand, dict) else {}, auto.get("s1_winner_key")))
    lines.append("")
    return lines


def shift_banner(title, loaded, status):
    lines = [f"## {title}", ""]
    if not loaded and status == "missing":
        lines.append("Файла ещё нет. Таблица появится после прогона. Повторный `scripts/make_report.py` её заполнит.")
        lines.append("")
        return lines
    if status != "ok" and not any(item[1] == "ok" or (len(item) > 2 and item[2] == "ok") for item in loaded):
        lines.append(f"Состояние: {status_word(status)}.")
        lines.append("")
    return lines


def side_sections(auto, side):
    side = side or {}
    lines = ["## Смежные проверки", ""]
    lines.append("Эти разделы не выбирают победителя этапа 1. В выбор входит только максимум val IoU при FAR = 1e-3 в слитом этапе 1. Дообучение сравнивает свой объединённый val v8+v10 и в тот список не подмешивается.")
    lines.append("")
    lines.append("### Дообучение U-Net, focal+Tversky")
    lines.append("")
    lines.append("Порог снят с фона объединённого val и на test не пересчитывался. Столбец «Выбор дообучения» — максимум pooled val IoU. Более высокий test эту строку не назначает.")
    lines.append(f"Журналы: {auto.get('ft_train', NA)}.")
    lines.append("")
    lines.append(ft_table(side))
    lines.append("")
    lines.append("### Классический CV")
    lines.append("")
    lines.append("Отдельные json. Их val не сливается с ансамблями раундов 4–7.")
    lines.append("")
    lines.append("v8:")
    lines.append("")
    lines.append(cv_split_table(side.get("cv_v8"), ["cv_ref", "cv_pipe", "unet", "unet_cv_post", "base_median_x3_spatial_z"]))
    lines.append("")
    lines.append("v10, порог со своего val, U-Net в этом файле может отсутствовать:")
    lines.append("")
    lines.append(cv_split_table(side.get("cv_v10"), ["cv_ref", "cv_pipe", "base_median_x3_spatial_z"]))
    lines.append("")
    lines.append(
        f"Карточки классического CV (порог с val, не с МГТС): yes {auto.get('cv_yes', NA)}, "
        f"no {auto.get('cv_no', NA)}. Зоны по файлам: {auto.get('cv_zones', NA)}."
    )
    lines.append("")
    lines.append("### Пост-обработка score")
    lines.append("")
    lines.append(f"Флаг `dts.postproc.ENABLED`: {auto.get('post_enabled', NA)}. Выбор варианта — val IoU при FAR = 1e-3, до чтения test.")
    lines.append("")
    lines += postproc_table(side.get("post"), side.get("tta"))
    lines.append("")
    lines.append("### Доменный сдвиг")
    lines.append("")
    lines.append("Прогон описывает метод, который записан в его json. Если это не текущий победитель, цифры к новому выпуску не переносятся.")
    lines.append("")
    lines += domain_table(side.get("domain"))
    lines.append(
        f"Карточки global far 1e-3: yes {auto.get('dom_yes', NA)}, no {auto.get('dom_no', NA)}. "
        f"Пофайловый 1e-3: yes {auto.get('dom_pf_yes', NA)}, no {auto.get('dom_pf_no', NA)}. "
        f"После согласования шума, global: yes {auto.get('dom_matched_yes', NA)}, no {auto.get('dom_matched_no', NA)}."
    )
    lines.append("")
    lines.append("### Кривая порога прежнего выпуска")
    lines.append("")
    lines.append("Файл `real_gap.json`. Карточки порог не выбирали. Если метод ниже не совпадает с победителем, это предыдущий кабель, не текущий выпуск.")
    lines.append("")
    lines += gap_prev_table(side.get("gap_prev"))
    lines.append("### Промахи текущего выпуска")
    lines.append("")
    lines.append("Файл `real_gap_ft.json`, если он уже лежит. Совпадение метода с победителем по val: " + str(auto.get("gap_ft_is_winner", NA)) + ".")
    lines.append("")
    lines += gap_ft_tables(side.get("gap_ft"))
    lines.append("### Безучительный прогноз, экспертное суждение, не эталон")
    lines.append("")
    lines.append(
        f"Карточки {auto.get('unsup_yes', NA)} и {auto.get('unsup_no', NA)}, "
        f"IoU на синтетике {auto.get('unsup_synth_iou', NA)}. "
        f"Зоны: {auto.get('unsup_files', NA)}."
    )
    lines.append(f"Вердикты топа по файлам: {auto.get('unsup_expert', NA)}.")
    lines.append("Это чтение остатка и чисел зоны. Класса модель не ставит. Полные абзацы по файлам — в `docs/UNSUP.md`.")
    lines.append("")
    lines.append("### Кластеры зон")
    lines.append("")
    lines.append(
        f"Метод `{auto.get('cl_method', NA)}`, порог {auto.get('cl_thr', NA)}, зон {auto.get('cl_n', NA)}, "
        f"k {auto.get('cl_k', NA)}, silhouette {auto.get('cl_sil', NA)}, "
        f"ARI Ward {auto.get('cl_ari_ward', NA)}, ARI с классами {auto.get('cl_ari_class', NA)}. "
        f"По файлам: {auto.get('cl_files', NA)}."
    )
    lines.append(auto.get("cl_summary", NA) + ".")
    lines.append("Кластеризация не двигает порог и не меняет победителя по val IoU. Текст источника — `docs/REAL_CLUSTERS.md`.")
    lines.append("")
    lines.append(
        f"Выпуск в `models_release`: метод `{auto.get('release_method', NA)}`, "
        f"состав {auto.get('release_members', NA)}, порог far 1e-3 {auto.get('release_thr', NA)}."
    )
    lines.append("")
    return lines


def build_md(s1, s2, real, auto, paths, statuses, stage1_items=None, e2e_rows=None, e2e_bad=None, train_logs=None,
             cand=None, cand_items=None, v10_loaded=None, v10_status="missing", v9=None, v9_path=None, v9_status="missing",
             side=None):
    _p1, p2, p3 = paths
    st1, st2, st3 = statuses
    stage1_items = stage1_items or [(paths[0], st1)]
    e2e_rows = e2e_rows or []
    e2e_bad = e2e_bad or []
    train_logs = train_logs or []
    cand_items = cand_items or []
    v10_loaded = v10_loaded or []
    v9_path = v9_path or resolve(DEFAULT_V9)
    lines = [
        "# Результаты: детекция и типизация аномалий Raman-DTS",
        "",
        "Кейс МГТС. Этап 1 — есть ли аномалия на поле температура–время–расстояние. Этап 2 — какого она типа. Реальные файлы 1–6.csv в обучении и в подборе порога не участвуют.",
        "",
        "Документ собирает `scripts/make_report.py`. Пояснение протокола и физики — в `docs/NOTE.md`. Числа между маркерами `AUTO` в записке берутся отсюда же.",
        "",
    ]
    lines += headline(auto)
    lines += how_to_read()
    lines += stage1_banner(stage1_items)
    lines.append("")
    if isinstance(s1, dict):
        ver = s1.get("version", NA)
        lines.append(f"Версия синтетики в файле: `{ver}`. FAR_ref (доля фона val с |Z| ≥ 3.5) = {auto.get('s1_far_ref')}.")
        lines.append("Пороги всех методов, кроме строки «эталон, зоны» и отдельной диагностики `test_far_matched`, выбраны на val. Цифры этого раздела, если не подписано «val» или «совпавший FAR на test», посчитаны на test.")
        lines.append("Главная рабочая точка для чтения и для выбора победителя — FAR = 1e-3, критерий только IoU в блоке val. Точка «FAR эталона» грубее: пиксельный FAR на синтетике поднимают неразмеченные структуры фона, и FAR_ref оказывается около 5%.")
        lines.append("")
        lines.append("### Все методы, точка FAR = 1e-3")
        lines.append("")
        lines.append("Победитель — строка с наибольшим IoU val. Столбцы после «Победитель» посчитаны на test и в выбор не входят.")
        lines.append(f"Состав победителя: {auto.get('s1_winner_members', NA)}.")
        lines.append("")
        lines.append(f"Победитель: {auto.get('s1_winner_name', NA)}, val IoU {auto.get('s1_winner_val_iou', NA)}.")
        lines.append("")
        lines.append(all_methods_table(s1, auto.get("s1_winner_key")))
        lines.append("")
        lines += candidate_section(auto, cand, cand_items)
        lines.append("")
        lines.append("### Гистерезис и точка FAR = 1e-2")
        lines.append("")
        lines.append("Гистерезис берёт зерно при пороге FAR = 1e-3 и растит связную область до порога FAR = 1e-2. Эти точки в выбор победителя не входят.")
        lines.append("")
        lines.append(hyst_table(s1))
        lines.append("")
        lines.append("### Слабые события ≤2 °C")
        lines.append("")
        lines.append(
            f"Событийный recall на test. На точке FAR = 1e-3 он лежит между {auto.get('s1_weak_1e3_min', NA)} "
            f"и {auto.get('s1_weak_1e3_max', NA)}. На точке FAR эталона у части методов он выше."
        )
        lines.append("")
        lines.append(weak_methods_table(s1))
        lines.append("")
        lines.append("### Журналы обучения, лучший val loss")
        lines.append("")
        lines.append("best_val — минимум кросс-энтропии плюс Dice на val за прогон. Это не IoU. Расписание прогонов разное, число эпох стоит рядом.")
        lines.append("")
        lines.append(train_table(train_logs))
        lines.append("")
        rows = stage1_rows(dict_of(s1, "test"))
        lines.append("### Test: качество детекции")
        lines.append("")
        t1, t2 = detector_tables(rows)
        lines.append(t1)
        lines.append("")
        lines.append("Нагрев, охлаждение, слабые события и помехи на том же test.")
        lines.append("")
        lines.append(t2)
        lines.append("")
        lines.append("### Test: помехи вне счёта и ложные зоны")
        lines.append("")
        lines.append("Столбцы precision и IoU здесь — из `nuisance_ignored`: клетка помехи не входит ни в верные, ни в ложные тревоги. Если ключа в JSON нет, в ячейке «н/д». Ложные зоны — из `zones.false_zones_per_km_h`.")
        lines.append("")
        lines.append(ignored_and_zones_table(rows))
        lines.append("")
        lines.append("### Test: средняя точность без порога")
        lines.append("")
        lines.append("Один ряд на метод, из `threshold_free.pixel_average_precision`. Порог не выбирается.")
        lines.append("")
        lines.append(ap_table(dict_of(s1, "threshold_free")))
        lines.append("")
        lines.append("### Test: диагностика при FAR, совпавшем с эталоном на test")
        lines.append("")
        lines.append(
            "Порог каждого метода подобран на фоне test так, чтобы FAR совпал с долей фона test, где |Z| ≥ 3,5. "
            f"Эта доля (`base_test_far`) = {fmt_far(s1.get('base_test_far'))}. "
            "Таблица не выбирает модель. Рабочая точка по-прежнему ставится на val, главная из них — FAR = 1e-3."
        )
        lines.append("")
        lines.append(matched_table(dict_of(s1, "test_far_matched")))
        lines.append("")
        lines.append("### Test: разность с эталоном на точке равного FAR")
        lines.append("")
        lines.append("Эталон строки — пиксельный |Z| на точке «FAR эталона». Терпимость протокола к уезду FAR: отношение не выше 1.1. Разность «+» значит, что метод выше эталона.")
        lines.append("")
        lines.append(delta_table(rows))
        lines.append("")
        lines.append("### Пороги, снятые с val")
        lines.append("")
        lines.append(threshold_table(s1.get("thresholds") if isinstance(s1.get("thresholds"), dict) else {}))
        lines.append("")
        lines.append("### Val против test")
        lines.append("")
        lines.append("Val использован только чтобы поставить порог. Совпадение FAR val с целью и спокойный перенос на test — проверка, что точка не подгонялась по test.")
        lines.append("")
        lines.append(val_table(s1.get("val") if isinstance(s1.get("val"), dict) else {}, rows))
        lines.append("")
        lines += template_sections(rows)
        lines += label_sections(rows)
        lines += amp_sections(rows)
        lines.append("Строка «эталон, зоны» в первой таблице — не порог по score, а готовая маска зон haklolarp (фильтры ширины и длительности поверх |Z| ≥ 3.5). У неё нет отдельных горячих и холодных IoU в этом JSON.")
        lines.append("")
    v8_thr = dict_of(s1, "thresholds")
    lines += shift_banner("Синтетика v10, пороги с val v8", v10_loaded, v10_status)
    lines.append("Сплит test. Порог каждой строки взят из val v8 и на v10 не пересчитывался. Val v10 в этих файлах пуст, поэтому столбец IoU val здесь «н/д» и в выбор победителя не входит.")
    lines.append(f"Пометка refit: {auto.get('v10_refit', NA)}. Файлов: {auto.get('v10_n_files', NA)}.")
    lines.append(
        f"Победитель по val на v10 (если для него есть json): IoU {auto.get('v10_winner_iou', NA)}, "
        f"F1 {auto.get('v10_winner_f1', NA)}, precision {auto.get('v10_winner_precision', NA)}, "
        f"recall {auto.get('v10_winner_recall', NA)}, FAR {auto.get('v10_winner_far', NA)}, "
        f"recall ≤2 °C {auto.get('v10_winner_weak', NA)}."
    )
    lines.append("")
    v10_ok = [(path, data) for path, data, status in v10_loaded if status == "ok"]
    bad_v10 = [(path, status) for path, _data, status in v10_loaded if str(status).startswith("invalid")]
    if bad_v10:
        lines.append("Не разобраны: " + ", ".join(f"`{rel(path)}`" for path, _st in bad_v10) + ".")
        lines.append("")
    lines.append(shift_table(v10_ok, v8_thr))
    lines.append("")
    lines += shift_banner("Синтетика v9", [(v9_path, v9_status)], v9_status)
    when = f", изменён {mtime(v9_path)}" if v9_path.is_file() else ""
    lines.append(f"Источник: `{rel(v9_path)}` ({status_word(v9_status)}{when}).")
    lines.append(
        "Это отдельный набор. Его val не участвует в выборе победителя v8. "
        f"Порог точки FAR = 1e-3 совпадает с порогом того же метода в слитом v8: {auto.get('v9_thr_match', NA)}."
    )
    lines.append(
        "Выпущенного детектора в этом файле может не быть: тогда строка победителя в записке остаётся «н/д», а таблица показывает только методы json."
    )
    lines.append("")
    if isinstance(v9, dict):
        lines.append(shift_table([(v9_path, v9)], v8_thr))
    else:
        lines.append("_JSON v9 не подставлен._")
    lines.append("")
    lines += banner("Этап 2. Тип события", p2, st2)
    lines.append("")
    if isinstance(s2, dict):
        classes = list_of(s2, "classes") or CLASS_ORDER
        lines.append(f"Версия `{s2.get('version', NA)}`. Объектов dev (train+val): {fmt_count(s2.get('n_dev'))}. Объектов test: {fmt_count(s2.get('n_test'))}.")
        lines.append("Состав выборки (все сплиты, уже после слияния меток генератора в классы): " + class_count_str(s2) + ".")
        lines.append("")
        lines.append("GroupKFold — out-of-fold по `seq_id` на dev: запись не попадает в обучение того фолда, который её оценивает. Test — модели, дообученные на всём dev; test при обучении не открывался. Ансамбль (среднее вероятностей EventNet и LightGBM) считается только на test. CNN только по кропу — абляция, она есть в group k-fold и не выпускается отдельной test-моделью.")
        lines.append("")
        lines.append("### GroupKFold, dev")
        lines.append("")
        lines.append(summary_classifier(dict_of(s2, "groupkfold")))
        lines.append("")
        lines.append(per_class_table(dict_of(s2, "groupkfold"), classes))
        lines.append("")
        lines += confusion_tables(dict_of(s2, "groupkfold"), classes)
        lines.append("### Test")
        lines.append("")
        lines.append(summary_classifier(dict_of(s2, "test")))
        lines.append("")
        lines.append(per_class_table(dict_of(s2, "test"), classes))
        lines.append("")
        lines += confusion_tables(dict_of(s2, "test"), classes)
    lines.append("## Сквозная оценка, FAR = 1e-3")
    lines.append("")
    lines.append("Зоны детектора на test отдаются этапу 2. Совпадение с меткой — IoU боксов не ниже 0.1. В таблицу входят файлы `e2e_*.json`, в том числе прогон согласованного фильтра и его прежний классификатор. Победитель этапа 1 попадает сюда, если для него такой файл уже лежит.")
    lines.append("")
    if e2e_bad:
        bad_bits = ", ".join(f"`{rel(path)}` ({status_word(status)})" for path, status in e2e_bad)
        lines.append("Не разобраны: " + bad_bits + ".")
        lines.append("")
    lines += e2e_tables(e2e_rows)
    lines += banner("Реальные файлы 1–6.csv", p3, st3)
    lines.append("")
    if isinstance(real, dict):
        lines += real_section(real, auto)
    lines += side_sections(auto, side)
    lines += session2_sections(auto, side)
    lines.append("## Чего в этой сборке нет")
    lines.append("")
    missing = []
    if st1 != "ok":
        missing.append("этап 1")
    if st2 != "ok":
        missing.append("этап 2 (`artifacts/eval/stage2_v8.json`)")
    if st3 != "ok":
        missing.append("реальные файлы (`artifacts/eval/real_v8.json`)")
    if not e2e_rows:
        missing.append("сквозная оценка (`artifacts/stage2_scratch/e2e_*.json`)")
    if v10_status != "ok":
        missing.append("v10 (`artifacts/eval/v10_*.json`)")
    if v9_status != "ok":
        missing.append("v9 (`artifacts/eval/stage1_v9_test.json`)")
    if missing:
        lines.append("Не подставлены: " + "; ".join(missing) + ".")
    else:
        lines.append("Файлы оценки прочитаны. Пустые ячейки «н/д» внутри таблиц — отсутствующие ключи, а не нули.")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv):
    p = argparse.ArgumentParser(description="Собрать docs/RESULTS.md и заполнить маркеры docs/NOTE.md")
    p.add_argument("--stage1", action="append", default=None,
                   help="JSON этапа 1. Флаг можно повторять: поздний файл перекрывает тот же ключ. Шаблон * раскрывается.")
    p.add_argument("--stage2", default="artifacts/eval/stage2_v8.json")
    p.add_argument("--real", default="artifacts/eval/real_v8.json")
    p.add_argument("--v9", default=DEFAULT_V9)
    p.add_argument("--v10-glob", default=DEFAULT_V10_GLOB)
    p.add_argument("--unsup", default=DEFAULT_UNSUP)
    p.add_argument("--cv-v8", default=DEFAULT_CV_V8)
    p.add_argument("--cv-v10", default=DEFAULT_CV_V10)
    p.add_argument("--cv-cards", default=DEFAULT_CV_CARDS)
    p.add_argument("--postproc", default=DEFAULT_POST)
    p.add_argument("--tta", default=DEFAULT_TTA)
    p.add_argument("--domain", default=DEFAULT_DOMAIN)
    p.add_argument("--gap-prev", default=DEFAULT_GAP_PREV)
    p.add_argument("--gap-ft", default=DEFAULT_GAP_FT)
    p.add_argument("--clusters", default=DEFAULT_CLUSTERS)
    p.add_argument("--release", default=DEFAULT_RELEASE)
    p.add_argument("--v11", default=DEFAULT_V11)
    p.add_argument("--cn", default=DEFAULT_CN)
    p.add_argument("--cnsym", default=DEFAULT_CNSYM)
    p.add_argument("--cn-real", default=DEFAULT_CN_REAL)
    p.add_argument("--cnsym-real", default=DEFAULT_CNSYM_REAL)
    p.add_argument("--ms", default=DEFAULT_MS)
    p.add_argument("--realnet-synth", default=DEFAULT_RN_SYNTH)
    p.add_argument("--realnet-real", default=DEFAULT_RN_REAL)
    p.add_argument("--realnet-ablation", default=DEFAULT_RN_ABL)
    p.add_argument("--realnet-bc-synth", default=DEFAULT_RN_BC_SYNTH)
    p.add_argument("--realnet-bc-real", default=DEFAULT_RN_BC_REAL)
    p.add_argument("--cv-zones", default=DEFAULT_CV_ZONES)
    p.add_argument("--experiments-s2", default=DEFAULT_S2_DOC)
    p.add_argument("--e2e-glob", default="artifacts/stage2_scratch/e2e_*.json")
    p.add_argument("--train-unet", default="artifacts/stage1/train_log.json")
    p.add_argument("--train-unet-ms", default="artifacts/stage1/train_log_unet_ms.json")
    p.add_argument("--train-unet-weak", default="artifacts/stage1/train_log_unet_weak.json")
    p.add_argument("--results", default="docs/RESULTS.md")
    p.add_argument("--note", default="docs/NOTE.md")
    p.add_argument("--no-note", action="store_true", help="не трогать пояснительную записку")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv or sys.argv[1:])
    stage1_args = args.stage1 or list(DEFAULT_STAGE1)
    stage1_paths = expand_inputs(stage1_args)
    stage1_loaded = [load_json(path) for path in stage1_paths]
    stage1_items = list(zip(stage1_paths, [item[1] for item in stage1_loaded]))
    s1 = merge_stage1(stage1_loaded)
    st1 = stage1_status([item[1] for item in stage1_loaded])
    cand_idx = [i for i, path in enumerate(stage1_paths) if is_candidate_source(path)]
    cand_loaded = [stage1_loaded[i] for i in cand_idx] if cand_idx else stage1_loaded
    cand_items = [stage1_items[i] for i in cand_idx] if cand_idx else []
    cand = merge_stage1(cand_loaded)
    p2, p3 = resolve(args.stage2), resolve(args.real)
    v9_path = resolve(args.v9)
    v9, v9_status = load_json(v9_path)
    v10_paths = expand_inputs([args.v10_glob])
    v10_loaded = []
    for path in v10_paths:
        data, status = load_json(path)
        v10_loaded.append((path, data, status))
    v10_ok = [(path, data) for path, data, status in v10_loaded if status == "ok"]
    v10_status = stage1_status([status for _p, _d, status in v10_loaded]) if v10_loaded else "missing"
    unsup_path = resolve(args.unsup)
    unsup, unsup_status = load_json(unsup_path)
    s2, st2 = load_json(p2)
    real, st3 = load_json(p3)
    e2e_rows, e2e_bad = load_e2e(args.e2e_glob)
    train_specs = [
        ("U-Net", resolve(args.train_unet), "unet"),
        ("unet_ms", resolve(args.train_unet_ms), "unet_ms"),
        ("unet_weak", resolve(args.train_unet_weak), "unet_weak"),
    ]
    train_logs = []
    trains = {}
    train_bad = []
    for label, path, key in train_specs:
        data, status = load_json(path)
        trains[key] = data if status == "ok" else None
        train_logs.append((label, path, trains[key]))
        if str(status).startswith("invalid"):
            train_bad.append((path, status))
    paths = [stage1_paths[0], p2, p3]
    statuses = [st1, st2, st3]
    side_specs = {
        "cv_v8": args.cv_v8,
        "cv_v10": args.cv_v10,
        "cv_cards": args.cv_cards,
        "post": args.postproc,
        "tta": args.tta,
        "domain": args.domain,
        "gap_prev": args.gap_prev,
        "gap_ft": args.gap_ft,
        "release": args.release,
        "v11": args.v11,
        "cn": args.cn,
        "cnsym": args.cnsym,
        "cn_real": args.cn_real,
        "cnsym_real": args.cnsym_real,
        "ms": args.ms,
        "rn_synth": args.realnet_synth,
        "rn_real": args.realnet_real,
        "rn_abl": args.realnet_ablation,
        "rn_bc_synth": args.realnet_bc_synth,
        "rn_bc_real": args.realnet_bc_real,
    }
    side = {}
    side_bad = []
    for key, raw in side_specs.items():
        data, status = load_json(resolve(raw))
        side[key] = data if status == "ok" else None
        if str(status).startswith("invalid"):
            side_bad.append((resolve(raw), status))
    ft_rows = []
    for method, val_p, v8_p, v10_p, train_p in DEFAULT_FT:
        loaded = []
        for raw in (val_p, v8_p, v10_p, train_p):
            data, status = load_json(resolve(raw))
            loaded.append(data if status == "ok" else None)
            if str(status).startswith("invalid"):
                side_bad.append((resolve(raw), status))
        ft_rows.append((method, *loaded))
    side["ft"] = ft_rows
    cv_rows, cv_status = load_csv_rows(resolve(args.cv_zones))
    side["cv_zones"] = cv_rows if cv_status == "ok" else None
    if str(cv_status).startswith("invalid"):
        side_bad.append((resolve(args.cv_zones), cv_status))
    cluster_path = resolve(args.clusters)
    if cluster_path.is_file():
        side["clusters_text"] = cluster_path.read_text(encoding="utf-8")
    else:
        side["clusters_text"] = ""
    auto = build_auto(
        s1, s2, real, st1, st2, st3, trains=trains, e2e_rows=e2e_rows,
        cand=cand, cand_loaded=cand_loaded, v10_ok=v10_ok, v10_status=v10_status,
        v9=v9, v9_status=v9_status, unsup=unsup, unsup_status=unsup_status,
        side=side,
    )
    text = build_md(
        s1, s2, real, auto, paths, statuses,
        stage1_items=stage1_items, e2e_rows=e2e_rows, e2e_bad=e2e_bad, train_logs=train_logs,
        cand=cand, cand_items=cand_items, v10_loaded=v10_loaded, v10_status=v10_status,
        v9=v9, v9_path=v9_path, v9_status=v9_status, side=side,
    )
    out = resolve(args.results)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    note_msg = "записка не запрошена"
    if not args.no_note:
        note = resolve(args.note)
        if not note.is_file():
            note_msg = f"нет {rel(note)}, маркеры не обновлялись"
        else:
            raw = note.read_text(encoding="utf-8")
            filled, n, n_na = fill_note(raw, auto)
            if filled != raw:
                note.write_text(filled, encoding="utf-8")
            note_msg = f"маркеров {n}, из них «н/д» {n_na}, файл {rel(note)}"
        s2_doc = resolve(args.experiments_s2)
        if s2_doc.is_file() and "<!-- AUTO:" in s2_doc.read_text(encoding="utf-8"):
            raw_s2 = s2_doc.read_text(encoding="utf-8")
            filled_s2, n_s2, n_s2_na = fill_note(raw_s2, auto)
            if filled_s2 != raw_s2:
                s2_doc.write_text(filled_s2, encoding="utf-8")
            note_msg += f"; сессия 2: маркеров {n_s2}, «н/д» {n_s2_na}"
    stage1_msg = ", ".join(item[1] for item in stage1_loaded)
    print(f"stage1: {stage1_msg}; stage2: {st2}; real: {st3}; e2e: {len(e2e_rows)}; v9: {v9_status}; v10: {v10_status}; unsup: {unsup_status}")
    print(f"записан {rel(out)}")
    print(note_msg)
    bad = [item[1] for item in stage1_loaded if str(item[1]).startswith("invalid")]
    bad += [st for st in (st2, st3) if str(st).startswith("invalid")]
    bad += [status for _path, status in e2e_bad if str(status).startswith("invalid")]
    bad += [status for _path, status in train_bad if str(status).startswith("invalid")]
    bad += [status for _path, _data, status in v10_loaded if str(status).startswith("invalid")]
    if str(v9_status).startswith("invalid"):
        bad.append(v9_status)
    if str(unsup_status).startswith("invalid"):
        bad.append(unsup_status)
    bad += [status for _path, status in side_bad]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
