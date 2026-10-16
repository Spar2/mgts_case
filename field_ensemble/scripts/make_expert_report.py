"""Экспертный отчёт по реальным MGTS CSV: прогон и текст одной командой.

Только оценка. Порог берётся из релиза, на реальных записях и карточках
ничего не обучается и не калибруется. Метки не выдумываются.

  python scripts/make_expert_report.py --release ens_u2ft_tcn_lgbm --static off --out-prefix draft
  python scripts/make_expert_report.py --release ens_u2ft_tcn_lgbm --static on --out-prefix final
  python scripts/make_expert_report.py --release ens_mean4 --out-prefix final
  python scripts/make_expert_report.py --release ens_gen2 --out-prefix draft
  python scripts/make_expert_report.py --release ens_u2ft_tcn_lgbm --out-prefix draft --report-only

`--static on` только вместе с ens_u2ft_tcn_lgbm. ens_mean4 включает `--mae on`
и со `--static on` останавливается. ens_gen2 включает `--gen2 on` и со
`--static on` тоже останавливается. Голый `run_real.py --gen2 on` пишет
`artifacts/eval/real_gen2.json`, `.csv` и рисунки в `artifacts/eval/real_gen2_fig`.
Этот скрипт передаёт свои пути префикса и читает их, а не `real_v8.json`.

Рисунки пишутся во временный каталог и переносятся как
docs/figures/real_<prefix>_<file>.png. docs/figures/real_1.png … real_6.png,
real_cv_*.png, real_v8.json и real_zones.csv не трогаются.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = Path("/workspace/repo/run/.venv/bin/python")
EVENTS = Path("/workspace/run_logs/_events.log")
CARDS_BATCH = [
    ROOT / "data/operator_cards/cards_batch1.jsonl",
    ROOT / "data/operator_cards/cards_batch2.jsonl",
]
CARDS2 = Path(
    "/home/box/agent-data/agents/7620c11d-77b1-4929-aaa8-c80e192177e7"
    "/attachments/66eacdf5a4b0ab8649ecf57a4b35554ad67790401bcf97c035e546eff4f764ff.jsonl"
)
REVIEW2_CSV = ROOT / "artifacts/eval/cards2_review.csv"
REPORT = ROOT / "docs/REAL_EXPERT_REPORT.md"
FIG_DIR = ROOT / "docs/figures"

# Чужие опубликованные пути. Прогон их не открывает на запись; в конце сверяем mtime.
OWNED = [
    "artifacts/eval/real_v8.json",
    "artifacts/eval/real_zones.csv",
    *[f"docs/figures/real_{i}.png" for i in range(1, 7)],
    *[f"docs/figures/real_cv_{i}.png" for i in range(1, 7)],
]
# Победитель опубликован в real_v8. Голый run_real --gen2 on пишет real_gen2.*.
# Оба пути этот скрипт только читает. Свой прогон идёт в real_<prefix>.*
WINNER_JSON = ROOT / "artifacts/eval/real_v8.json"
GEN2_JSON = ROOT / "artifacts/eval/real_gen2.json"
GEN2_CSV = ROOT / "artifacts/eval/real_gen2.csv"
GEN2_FIG = ROOT / "artifacts/eval/real_gen2_fig"
SYNTH_GEN2_V8 = ROOT / "artifacts/eval/s3_gen2_v8.json"
SYNTH_GEN2_CHECK = ROOT / "artifacts/eval/s3_gen2_check.json"
SYNTH_COMBO = ROOT / "artifacts/eval/s3_combo_report.json"
SYNTH_V10 = ROOT / "artifacts/eval/v10_ens_u2ft_tcn_lgbm.json"
SYNTH_V11 = ROOT / "artifacts/eval/v11_ens_u2ft_tcn_lgbm.json"


def _gen2_raw() -> dict:
    path = ROOT / "models_release_gen2" / "stage1_thresholds.json"
    if not path.is_file():
        raise SystemExit(f"нет {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _gen2_blurb() -> str:
    """Текст релиза из models_release_gen2/stage1_thresholds.json, без подгонки по real."""
    raw = _gen2_raw()
    thr = raw["thresholds"]["cnn_unet"]["far_1e-3"]
    definition = raw.get("definition") or (
        "score = 0.5 * (ens_u2ft_tcn_lgbm + unet_gen2); "
        "hotness = 0.5 * (winner_hot + unet_gen2_hot)"
    )
    width = int(raw.get("width", 16))
    source = raw.get("threshold_source") or "полный фон v8 val, FAR 1e-3"
    return (
        "Смесь ens_blend из `models_release_gen2` (`--gen2 on`): "
        f"{definition}. "
        f"U-Net ширины {width}, дообученная на синтетике с физическим генератором: "
        "узкие статичные точки, широкие сдвиги, периодика HVAC, "
        "приборные провалы −20°C как негативы. "
        f"Порог FAR 1e-3 равен {thr}. Источник порога: {source} "
        "Реальные CSV и карточки порог не двигают, метки не назначаются. "
        "`models_release` только читается. Со `--static on` этот релиз запрещён."
    )
HEADINGS = [
    "## Краткое заключение",
    "## Качество модели на синтетике",
    "## Релиз и флаги",
    "## Обзор по файлам",
    "## Значимые зоны",
    "## Физическая интерпретация",
    "## Согласие с карточками оператора",
    "## Ограничения",
    "## Как воспроизвести",
]

# Имена, которые скрипт умеет прогнать за один вызов. Неизвестное имя — выход,
# а не тихий откат на победителя.
RELEASES = {
    "ens_u2ft_tcn_lgbm": {
        "blurb": (
            "Выпуск `models_release`: две сети U-Net (ширина 24 и 16), "
            "временная свёртка TCN и пиксельный LightGBM. Порог — квантиль фона "
            "синтетической валидации v8 при доле ложных около 1e-3."
        ),
        "argv": [
            "--detector", "winner",
            "--static", "off",
            "--mae", "off",
            "--calib", "global",
            "--op", "far_1e-3",
            "--zones", "cc",
        ],
    },
    "ens_mean4": {
        "blurb": (
            "Смесь выпуска и MAE-реконструктора с весом 1/5 (`--mae on`). "
            "Порог ens_mean4 заморожен на синтетической валидации "
            "(`models_release_mae`). Карточки и реальные CSV в обучение и в порог не входят."
        ),
        "argv": [
            "--detector", "winner",
            "--static", "off",
            "--mae", "on",
            "--calib", "global",
            "--op", "far_1e-3",
            "--zones", "cc",
        ],
    },
    "ens_dom": {
        "blurb": (
            "Смесь ens_dom из `models_release_dom` (`--detector dom`). "
            "Порог свой, `models_release` только читается. На реальных CSV порог не калибруется."
        ),
        "argv": [
            "--detector", "dom",
            "--static", "off",
            "--mae", "off",
            "--calib", "global",
            "--op", "far_1e-3",
            "--zones", "cc",
        ],
    },
    "ens_gen2": {
        "blurb": _gen2_blurb(),
        "argv": [
            "--detector", "winner",
            "--static", "off",
            "--mae", "off",
            "--calib", "global",
            "--op", "far_1e-3",
            "--zones", "cc",
            "--gen2", "on",
        ],
    },
}


def log(msg: str) -> None:
    print(time.strftime("%H:%M", time.gmtime()) + " " + msg, flush=True)


def elog(msg: str) -> None:
    log(msg)
    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    with EVENTS.open("a", encoding="utf-8") as fh:
        fh.write(time.strftime("%H:%M", time.gmtime()) + " " + msg + "\n")


def snapshot(rel_paths):
    out = {}
    for rel in rel_paths:
        path = ROOT / rel
        if path.exists():
            st = path.stat()
            out[rel] = (st.st_mtime_ns, st.st_size)
        else:
            out[rel] = None
    return out


def file_key(name: str):
    return (0, int(name)) if str(name).isdigit() else (1, str(name))


def fmt_dur(sec) -> str:
    sec = int(round(max(0.0, float(sec))))
    hours, rem = divmod(sec, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours and minutes:
        return f"{hours} ч {minutes} мин"
    if hours:
        return f"{hours} ч"
    if minutes:
        return f"{minutes} мин"
    return f"{seconds} с"


def fmt_utc(ts) -> str:
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def fmt_m(value) -> str:
    return f"{float(value):.1f}"


def md_cell(value) -> str:
    text = str(value).replace("|", "/").replace("\n", " ").strip()
    return text


def class_prob(zone) -> float:
    probs = zone.get("class_prob") or {}
    name = zone.get("class") or ""
    if name in probs:
        return float(probs[name])
    if probs:
        return float(max(probs.values()))
    return 0.0


def is_alarm(zone) -> bool:
    return bool(zone.get("alarm", True))


def polarity_ru(polarity: str) -> str:
    if polarity == "hot":
        return "нагрев (hot)"
    if polarity == "cold":
        return "охлаждение (cold)"
    return polarity or "—"


# Рамановский DTS: температура из отношения антистокс/стокс. Пространственное
# разрешение прибора порядка 1 м (шаг отсчёта в файлах МГТС 0.1–1 м может быть мельче).
# Сетка модели — 0.25 м и 120 с (dts/grid.py). Сырые записи идут с шагом 10–240 с.
SPATIAL_RES_M = 1.0
GRID_DT_S = 120.0
TARGET_N = 16
MIN_N = 10
MAX_N = 18
MAX_PER_FILE = 4


def signed_peak(zone) -> float:
    """В json пик — max |ΔT|. Знак берётся из полярности."""
    amp = abs(float(zone.get("peak_dT_C") or 0.0))
    if (zone.get("polarity") or "") == "cold":
        return -amp
    return amp


def zone_metrics(name, zone, info):
    x0 = float(zone["start_m"])
    x1 = float(zone["end_m"])
    if x1 < x0:
        x0, x1 = x1, x0
    width = max(0.0, x1 - x0)
    dur_s = max(0.0, float(zone["t_end_s"]) - float(zone["t_start_s"]))
    length_m = float(info.get("km") or 0.0) * 1000.0
    polarity = zone.get("polarity") or ""
    return {
        "file": str(name),
        "id": int(zone.get("id") or 0),
        "zone": zone,
        "info": info,
        "x0": x0,
        "x1": x1,
        "width": width,
        "dur_s": dur_s,
        "dur_h": dur_s / 3600.0,
        "amp": abs(float(zone.get("peak_dT_C") or 0.0)),
        "polarity": polarity,
        "cls": zone.get("class") or "",
        "prob": class_prob(zone),
        "length_m": length_m,
        "single_bin": dur_s < 1.0,
        "sub_res": width < SPATIAL_RES_M,
        "near_start": x0 <= 15.0,
        "far": bool(length_m) and x1 >= 0.85 * length_m,
        "reason": "",
    }


def _interest(m) -> float:
    if m["single_bin"]:
        dur_factor = 0.45
    else:
        dur_factor = 0.35 + min(m["dur_h"], 8.0) / 3.0
    return m["amp"] * dur_factor * (0.30 + m["prob"])


def _same_focus(a, b) -> bool:
    if a["file"] != b["file"] or a["polarity"] != b["polarity"]:
        return False
    if a["x1"] >= b["x0"] and b["x1"] >= a["x0"]:
        return True
    gap = b["x0"] - a["x1"] if a["x0"] <= b["x0"] else a["x0"] - b["x1"]
    return gap < 4.0


def _dedup(rows):
    rows = sorted(rows, key=lambda m: (-_interest(m), file_key(m["file"]), m["x0"], m["id"]))
    kept = []
    for row in rows:
        if any(_same_focus(row, prev) for prev in kept):
            continue
        kept.append(row)
    return kept


def _file_count(picked, name) -> int:
    return sum(1 for row in picked if row["file"] == name)


def _take(pool, picked, pred, reason) -> None:
    for row in pool:
        if any(row["file"] == prev["file"] and row["id"] == prev["id"] for prev in picked):
            continue
        if _file_count(picked, row["file"]) >= MAX_PER_FILE:
            continue
        if pred(row):
            row["reason"] = reason
            picked.append(row)
            return


def select_significant(files):
    """10–20 тревог: амплитуда, длительность, уверенность, разные файлы и разные физические случаи.

    Соседние рамки одного файла и одной полярности ближе 4 м считаются одним очагом.
    """
    rows = []
    for name, info in files.items():
        for zone in info.get("zones") or []:
            if is_alarm(zone):
                rows.append(zone_metrics(name, zone, info))
    pool = _dedup(rows)
    if not pool:
        return []
    if len(pool) <= MAX_N and len(rows) < MIN_N:
        for row in pool:
            row["reason"] = row["reason"] or "все тревоги файла, их мало"
        return sorted(pool, key=lambda m: (file_key(m["file"]), m["x0"], m["id"]))
    picked = []
    _take(pool, picked, lambda m: True, "амплитуда, длительность и уверенность")
    _take(pool, picked, lambda m: m["polarity"] == "cold" and m["amp"] >= 4 and m["dur_h"] >= 1,
          "устойчивое охлаждение")
    _take(pool, picked, lambda m: m["polarity"] == "hot" and m["width"] >= 15 and m["dur_h"] >= 2,
          "протяжённый нагрев")
    _take(pool, picked, lambda m: m["sub_res"] and m["amp"] >= 8, "уже разрешения ~1 м")
    _take(pool, picked, lambda m: m["near_start"] and m["amp"] >= 3 and not m["single_bin"],
          "начало линии")
    _take(pool, picked, lambda m: m["single_bin"] and m["amp"] >= 3, "одна временная ячейка")
    _take(
        pool, picked,
        lambda m: m["far"] and m["amp"] >= 4 and m["dur_h"] < 0.5 and not m["single_bin"] and m["width"] >= 1,
        "дальняя часть линии",
    )
    for name in sorted({m["file"] for m in pool}, key=file_key):
        if any(m["file"] == name for m in picked):
            continue
        _take(pool, picked, lambda m, name=name: m["file"] == name and m["amp"] >= 2,
              "сильнейшая зона файла")
    while len(picked) < TARGET_N:
        progressed = False
        n_sub = sum(1 for m in picked if m["sub_res"])
        n_single = sum(1 for m in picked if m["single_bin"])
        for row in pool:
            if any(row["file"] == prev["file"] and row["id"] == prev["id"] for prev in picked):
                continue
            if _file_count(picked, row["file"]) >= MAX_PER_FILE:
                continue
            if row["sub_res"] and n_sub >= 2:
                continue
            if row["single_bin"] and n_single >= 2:
                continue
            row["reason"] = "амплитуда, длительность и уверенность"
            picked.append(row)
            progressed = True
            break
        if not progressed:
            break
    if len(picked) < MIN_N:
        for row in pool:
            if len(picked) >= MIN_N:
                break
            if any(row["file"] == prev["file"] and row["id"] == prev["id"] for prev in picked):
                continue
            row["reason"] = row["reason"] or "добор до десятка"
            picked.append(row)
    picked = picked[:MAX_N]
    return sorted(picked, key=lambda m: (file_key(m["file"]), m["x0"], m["id"]))


def _t0(m) -> float:
    return float(m["zone"]["t_start_s"])


def _t1(m) -> float:
    return float(m["zone"]["t_end_s"])


def _gap_m(a, b) -> float:
    if a["x1"] >= b["x0"] and b["x1"] >= a["x0"]:
        return 0.0
    if a["x0"] <= b["x0"]:
        return b["x0"] - a["x1"]
    return a["x0"] - b["x1"]


def _overlap_s(a, b) -> float:
    return max(0.0, min(_t1(a), _t1(b)) - max(_t0(a), _t0(b)))


def _short_frac(a, b) -> float:
    short = min(a["dur_s"], b["dur_s"])
    if short < 1.0:
        return 1.0 if abs(_t0(a) - _t0(b)) < 1.0 else 0.0
    return _overlap_s(a, b) / short


def _zid(m) -> str:
    return f"{m['file']}-{int(m['id']):04d}"


def _along(m) -> str:
    if not m["length_m"]:
        return ""
    pct = 100.0 * m["x1"] / m["length_m"]
    return f" Это {pct:.0f}% длины кабеля ({m['length_m']:.0f} м)."


def _dur_phrase(m) -> str:
    if m["single_bin"]:
        return (
            "метки начала и конца совпали: одна временная ячейка. "
            "На сетке модели шаг 120 с, в сырых CSV шаг 10–240 с, то есть одно усреднение, а не развитый процесс"
        )
    n_bins = max(1, int(round(m["dur_s"] / GRID_DT_S)))
    dur = fmt_dur(m["dur_s"])
    if m["dur_s"] < 2 * GRID_DT_S:
        return f"длительность {dur}, около {n_bins} ячеек по {GRID_DT_S:.0f} с: по времени событие не развито"
    if m["dur_s"] < 30 * 60:
        return f"длительность {dur}, около {n_bins} усреднений по {GRID_DT_S:.0f} с: короткое пятно"
    return f"длительность {dur}, около {n_bins} усреднений по {GRID_DT_S:.0f} с"


def _facts(m, zone) -> str:
    n_res = m["width"] / SPATIAL_RES_M
    where = f"{fmt_m(m['x0'])}–{fmt_m(m['x1'])} м"
    return (
        f"Рамка {where}, ширина {m['width']:.1f} м ({n_res:.1f}× разрешения ~1 м), {_dur_phrase(m)}. "
        f"Пик ΔT {signed_peak(zone):+.1f} °C — max |ΔT| со знаком полярности, уже после вычитания "
        f"синфазной составляющей и статического профиля. Класс {m['cls'] or '—'}, P={m['prob']:.2f}."
    )


def _amp_climate(amp: float) -> str:
    if amp >= 15:
        return (
            f"Амплитуда около {amp:.0f} °C велика для вентиляции помещения: ближе к солнцу на фасаде, "
            "выводу кабеля наружу или длинному контакту с горячей трубой."
        )
    if amp >= 8:
        return (
            f"Амплитуда {amp:.1f} °C выше обычных 2–4 °C вентиляции и ниже десятков градусов на открытом солнце. "
            "Подходят фасад с окнами, тёплый лоток и техническое помещение."
        )
    return (
        f"Амплитуда {amp:.1f} °C — обычный масштаб климата помещения и вентиляции "
        "после снятия синфазной составляющей."
    )


def _peer_notes(m, peers) -> tuple[str, str]:
    """Связь с другими зонами короткого списка того же файла. Не двигает порог."""
    others = [p for p in (peers or []) if p["id"] != m["id"] and p.get("zone")]
    if not others:
        return "", ""
    bits = []
    check = ""
    cohort = [
        p for p in others
        if p["polarity"] == m["polarity"] and m["dur_h"] >= 1 and p["dur_h"] >= 1
        and abs(_t0(p) - _t0(m)) <= 20 * 60
    ]
    if len(cohort) >= 2 or (len(cohort) == 1 and m["dur_h"] >= 2):
        ids = ", ".join(_zid(p) for p in sorted(cohort, key=lambda p: p["x0"]))
        bits.append(
            f"По времени рядом (старт в пределах 20 мин, та же полярность) стоят {ids}. "
            "Общее окно — режим здания: отопление, вентиляция или ночь. "
            "Разнесённые по кабелю рамки — точки, где волокно этот режим чувствует, а не отдельные аварии."
        )
        check = (
            " Обойти эти точки одним проходом: общая труба, шахта или режим вентиляции "
            "важнее, чем отдельный наряд на каждую рамку."
        )
    ranked = sorted(others, key=lambda p: (_gap_m(m, p), p["id"]))
    near = ranked[0]
    gap = _gap_m(m, near)
    if gap <= 40 and _short_frac(m, near) >= 0.4 and near not in cohort:
        wider = m if m["width"] >= near["width"] else near
        narrower = near if wider is m else m
        linked = narrower["width"] <= 8 or narrower["sub_res"] or wider["width"] >= 2.5 * max(narrower["width"], 0.5)
        if linked and (narrower["width"] <= 8 or narrower["sub_res"]):
            bits.append(
                f"Ближайшая зона списка — {_zid(near)} в {gap:.0f} м "
                f"({near['polarity']}, {signed_peak(near['zone']):+.1f} °C, ширина {near['width']:.1f} м), "
                "время пересекается. Узкая рамка на краю широкой читается как точечный контакт "
                "или склейка внутри общего участка, широкая — как климат этого участка."
            )
            if not check:
                check = " На месте смотреть обе рамки вместе: сначала широкий участок, потом узкий пик внутри или рядом."
        elif linked:
            bits.append(
                f"В {gap:.0f} м стоит {_zid(near)} "
                f"({signed_peak(near['zone']):+.1f} °C, ширина {near['width']:.0f} м), время пересекается. "
                "Два пятна одного режима, а не две независимые неисправности."
            )
            if not check:
                check = " Пройти оба участка одним заходом: если покрытие одно (окна, лоток, вентканал), это один объект."
    for p in others:
        if p["polarity"] == m["polarity"] or p["polarity"] not in ("hot", "cold") or m["polarity"] not in ("hot", "cold"):
            continue
        gap_time = min(abs(_t0(m) - _t1(p)), abs(_t0(p) - _t1(m)))
        if gap_time <= 40 * 60 and 4 <= _gap_m(m, p) <= 80:
            bits.append(
                f"По часам стыкуется с {_zid(p)} ({p['polarity']} {signed_peak(p['zone']):+.1f} °C): "
                "одно заканчивается около того часа, когда другое уже идёт. "
                "Похоже на смену режима помещения, а не на два независимых дефекта."
            )
            break
    return " ".join(bits), check


def file_lead(rows) -> str:
    """Одна фраза на файл, если короткий список — один эпизод, а не набор аварий."""
    if len(rows) < 2:
        return ""
    notes = []
    used = set()
    for anchor in rows:
        if anchor["id"] in used or anchor["dur_h"] < 1:
            continue
        group = [anchor]
        for other in rows:
            if other["id"] == anchor["id"] or other["id"] in used:
                continue
            if (
                other["polarity"] == anchor["polarity"]
                and other["dur_h"] >= 1
                and abs(_t0(other) - _t0(anchor)) <= 20 * 60
            ):
                group.append(other)
        if len(group) >= 3:
            for item in group:
                used.add(item["id"])
            ids = ", ".join(_zid(item) for item in sorted(group, key=lambda item: item["x0"]))
            kind = "нагрева" if anchor["polarity"] == "hot" else "охлаждения" if anchor["polarity"] == "cold" else "одного знака"
            notes.append(
                f"Зоны {ids} начинаются почти вместе и стоят на разных метрах. "
                f"Это один эпизод {kind}: общий режим трассы и несколько точек контакта. Выезд один."
            )
    best = None
    for i, left in enumerate(rows):
        for right in rows[i + 1:]:
            gap = _gap_m(left, right)
            if gap > 25 or _short_frac(left, right) < 0.8:
                continue
            wide = max(left["width"], right["width"])
            narrow = max(min(left["width"], right["width"]), 0.25)
            if wide < 2.5 * narrow:
                continue
            if best is None or gap < best[0]:
                best = (gap, left, right)
    if best is not None:
        gap, left, right = best
        narrow = left if left["width"] <= right["width"] else right
        wide = right if narrow is left else left
        if narrow["width"] <= 8 or narrow["sub_res"]:
            notes.append(
                f"{_zid(narrow)} по времени сидит внутри {_zid(wide)} и уже её в разы. "
                "Сначала смотреть протяжённый участок; узкий пик — вероятная муфта или склейка на его краю, а не второй источник."
            )
        else:
            notes.append(
                f"{_zid(narrow)} и {_zid(wide)} держатся в одни часы, между ними {gap:.0f} м. "
                "Это один участок с двумя пятнами, не две аварии."
            )
    return " ".join(notes)


def interpret(zone, info=None, peers=None):
    """Суждение по физике Рамановского DTS. Не метка и не повод двигать порог."""
    info = info or {}
    peers = list(peers or [])
    file_name = next((str(p.get("file")) for p in peers if p.get("file")), "?")
    m = zone_metrics(file_name, zone, info)
    width, amp, prob = m["width"], m["amp"], m["prob"]
    polarity = m["polarity"]
    cls = m["cls"]
    where = f"{fmt_m(m['x0'])}–{fmt_m(m['x1'])} м"
    heat = cls in ("Local_Heating", "Extended_Heating", "Contact_Trace")
    cool = cls in ("Local_Cooling", "Extended_Cooling")
    sign_clash = (heat and polarity == "cold") or (cool and polarity == "hot")
    facts = _facts(m, zone)
    along = _along(m)
    tag = "уточнить на месте"
    if m["single_bin"]:
        tag = "одна ячейка усреднения"
        cause = (
            f"{facts} Температура в Рамановском DTS — отношение антистокса к стоксу. "
            "Антистокс слабый, одно усреднение легко даёт выброс, и тем сильнее, чем дальше точка по волокну. "
            f"Пятно {width:.1f} м может быть реальным источником, размазанным разрешением ~1 м, "
            "но по времени оно не подтверждено."
            f"{along}"
        )
        conf = "низкая"
        check = (
            f"Повторяется ли отметка {where} на соседних записях. Одна ячейка в наряд не ставится. "
            "Если повтор есть, смотреть муфту и нет ли ступеньки на сырой температуре."
        )
    elif m["near_start"] and amp >= 3 and width <= 20:
        tag = "начало линии, прибор"
        cause = (
            f"{facts} Первые метры волокна — пигтейл прибора, первый коннектор, стартовый участок. "
            "Здесь обычны ступенька калибровки и шум запуска, даже когда трасса дальше спокойная. "
            f"Ширина {width:.1f} м и пик {amp:.1f} °C: на муфту у кросса это похоже только во вторую очередь."
            f"{along}"
        )
        conf = "низкая" if amp < 6 else "средняя"
        check = (
            f"Кросс и пигтейл, первая склейка у прибора на {where}. "
            "Сравнить сырую температуру: ступенька около 1 м — коннектор, гладкий горб шире 3 м — уже трасса."
        )
    elif width < SPATIAL_RES_M and amp >= 5:
        tag = "склейка или коннектор"
        trace = ""
        if cls == "Contact_Trace":
            trace = (
                " Класс Contact_Trace означает цепочку близких точек. "
                "При ширине меньше 1 м цепочка не разрешена: прибор видит одну клетку."
            )
        held = (
            f"Держится {fmt_dur(m['dur_s'])}: для точечного контакта с трубой это коротко, "
            "для склейки — как раз масштаб сбоя поправки потерь."
            if m["dur_h"] < 1
            else "Держится часами: кроме склейки правдоподобен и точечный хомут к трубе, его прибор тоже размазывает в одну клетку."
        )
        cause = (
            f"{facts} Рамка уже пространственного разрешения (~1 м). Прибор не отделяет источник тоньше этой длины: "
            "отсчёт 0.1–1 м только дискретизация. Узкий и сильный пик типичен для склейки, коннектора "
            "или макроизгиба: они ломают поправку потерь, и отношение антистокс/стокс выглядит как температура. "
            f"{held}{trace}{along}"
        )
        conf = "средняя" if m["dur_h"] >= 0.3 else "низкая"
        check = (
            f"Муфта, коннектор или хомут в пределах 1 м от {where}. "
            "На сырой температуре ступенька — склейка; гладкий горб шире 2–3 м — реальный нагрев, рамка его обрезала."
        )
    elif polarity == "cold" and amp >= 15:
        tag = "провал, склейка или улица"
        cause = (
            f"{facts} Провал {amp:.0f} °C для воздуха в помещении неправдоподобен. "
            "Волокно так «остывает» в данных, когда срывается поправка потерь (склейка, коннектор, изгиб) "
            f"или кабель выходит на улицу либо в колодец.{along}"
        )
        conf = "средняя"
        check = (
            f"Муфта и коннектор на {where}, выход кабеля из здания, колодец. "
            "Совпадает ли ступенька с соседним каналом: общая ступенька — прибор, локальная — муфта или контакт."
        )
    elif polarity == "hot" and amp >= 12 and width >= 8 and m["dur_h"] >= 0.5:
        tag = "солнце, окна или вывод наружу"
        cause = (
            f"{facts} Одна склейка даёт масштаб около 1 м, одиночный шум — одну ячейку по 120 с. "
            f"Здесь {width:.0f} м и около {amp:.0f} °C: оболочка прогрета на протяжённом куске, "
            "либо на этом куске другой тепловой режим. "
            "Правдоподобны солнце на фасаде и окнах, кабельный вывод из здания, длинный контакт с тёплой трубой. "
            f"Вентиляция обычно даёт единицы градусов, не {amp:.0f}.{along}"
        )
        conf = "высокая" if prob >= 0.5 else "средняя"
        check = (
            f"Пройти {where}: окна и солнце в эти часы, трубы, ввод в здание, открытый участок. "
            "Сырая температура плавная по метрам — тепло; узкая ступенька внутри рамки — склейка."
        )
    elif polarity == "hot" and width <= 8 and m["dur_h"] >= 1 and amp >= 4:
        tag = "муфта, ввод или труба"
        if width < 4:
            shape = (
                f"Ширина {width:.1f} м — несколько клеток разрешения ~1 м. "
                "Так размазываются муфта, кабельный ввод и хомут к горячей трубе."
            )
        else:
            shape = (
                f"Ширина {width:.1f} м шире одной муфты. "
                "Скорее ввод или короткий участок вдоль трубы, а не одиночная склейка."
            )
        if prob < 0.6:
            shape += f" P={prob:.2f} невысокая: форма для модели неоднозначна, на месте решать по сырой температуре."
        elif prob >= 0.85:
            shape += f" P={prob:.2f}: форма пятна устойчивая, это не шум одной ячейки."
        cause = (
            f"{facts} {shape} "
            "Часы усреднений отделяют пятно от шума. Статический профиль постоянный горб частично вычитает, "
            f"в ΔT остаётся меняющаяся часть.{along}"
        )
        conf = "высокая" if amp >= 6 and prob >= 0.55 else "средняя"
        check = (
            f"Муфта, ввод и хомут к трубе на {where}. Потрогать кабель и соседние трубы. "
            "Ступенька ~1 м на сырой T — склейка; гладкий горб в те же часы — реальный нагрев."
        )
    elif polarity == "cold" and m["dur_h"] >= 1 and amp >= 4 and width <= 15:
        tag = "труба, шахта или сквозняк"
        if width < 3.5:
            shape = (
                f"Ширина {width:.1f} м — точечный контакт: холодная труба или устье шахты. "
                "Склейка обычно ещё уже, около 1 м, и на сырой T выглядит ступенькой, а не ночным провалом."
            )
        else:
            shape = (
                f"Ширина {width:.1f} м уже не одна муфта. "
                "Короткий участок в холодном колодце, вдоль холодной трубы или на сквозняке."
            )
        if amp >= 12:
            depth = f"Провал {amp:.1f} °C глубок для комнаты: кроме трубы проверить выход на улицу и срыв поправки потерь."
        else:
            depth = (
                f"Провал {amp:.1f} °C для холодной трубы или уличного воздуха правдоподобен. "
                "Для комнатного воздуха это много, но не тот провал −15…−25 °C, который рисует срыв поправки."
            )
        cause = f"{facts} {shape} {depth}{along}"
        conf = "высокая" if prob >= 0.8 and amp >= 6 else "средняя"
        check = (
            f"Трубы, тяга воздуха и колодец на {where}. "
            "Сырая температура: плавный провал — контакт или климат, ступенька ~1 м — склейка."
        )
    elif polarity == "hot" and width >= 15:
        tag = "климат, солнце или вентиляция"
        cause = (
            f"{facts} Протяжённый нагрев на {width:.0f} м. Одна муфта такой ширины не собирает: "
            "точечный источник размазывается примерно на 1 м. "
            f"{_amp_climate(amp)}{along}"
        )
        conf = "высокая" if amp >= 8 and prob >= 0.5 else "средняя"
        check = (
            f"Чем закрыт участок {where}: окна, лоток, вентканал. "
            "На рисунке сравнить день и ночь: суточный ход — солнце или климат помещения."
        )
    elif polarity == "hot" and width >= 8 and m["dur_h"] >= 2 and amp >= 3:
        tag = "климат, солнце или вентиляция"
        cause = (
            f"{facts} Нагрев на {width:.0f} м дольше двух часов. Одна муфта так широко не греет. "
            f"{_amp_climate(amp)}{along}"
        )
        conf = "высокая" if amp >= 8 and prob >= 0.5 else "средняя"
        check = (
            f"Чем закрыт участок {where}: окна, лоток, вентканал, вывод из здания. "
            "На рисунке сравнить день и ночь."
        )
    elif polarity == "cold" and width >= 15:
        tag = "климат или широкая тень"
        cause = (
            f"{facts} Широкое охлаждение на {width:.0f} м. Правдоподобны тень, смена режима вентиляции, "
            "участок в другом помещении. Общий сдвиг частично снимается вычитанием синфазной составляющей, "
            f"рамка может быть короче пятна на сырой T.{along}"
        )
        conf = "средняя" if amp >= 3 else "низкая"
        check = f"Вентиляция и покрытие трассы на {where}. Сравнить сырую температуру соседних десятков метров."
    elif m["far"] and (m["dur_h"] < 0.5 or amp < 6):
        tag = "дальняя часть линии"
        n_bins = max(1, int(round(m["dur_s"] / GRID_DT_S)))
        cause = (
            f"{facts} Точка на {fmt_m(m['x1'])} м при длине записи около {m['length_m']:.0f} м — дальняя часть волокна. "
            f"Антистокс слабеет с дистанцией, шум температуры растёт. Около {n_bins} ячеек по 120 с — "
            "слишком мало, чтобы отличить муфту от шума конца. Повторяющееся пятно уже смотреть как муфту."
        )
        conf = "низкая" if m["dur_h"] < 0.5 else "средняя"
        check = (
            f"Повторяется ли {where} на других записях. Если нет — не выезжать. "
            "Если да, последняя муфта и нет ли роста шума к концу волокна на сырой T."
        )
    elif amp <= 2.5:
        tag = "около порога FAR 1e-3"
        cause = (
            f"{facts} Пик {amp:.1f} °C. При пороге FAR 1e-3 на синтетике такие события на грани шума. "
            f"Правдоподобны остаток после вычитания синфазной составляющей, слабый сквозняк, цикл вентиляции.{along}"
        )
        conf = "низкая"
        check = f"Есть ли повтор {where} на соседних сутках и совпадает ли он с вентиляцией. Без повтора в наряд не ставить."
    else:
        tag = "локальный источник"
        cause = (
            f"{facts} По ширине это несколько длин разрешения. Правдоподобны муфта, кабельный ввод, "
            f"контакт с трубой или короткое солнечное пятно у окна. Вентиляция и климат дают более широкие рамки.{along}"
        )
        conf = "средняя" if prob >= 0.45 and amp >= 4 else "низкая"
        check = f"Осмотр {where}: муфта, ввод, труба, окно. Сырая T рядом с рисунком ΔT."

    peer_cause, peer_check = _peer_notes(m, peers)
    if peer_cause:
        cause = f"{cause} {peer_cause}"
        check = f"{check}{peer_check}"
    if m["far"] and tag not in ("дальняя часть линии", "начало линии, прибор"):
        if m["dur_h"] >= 1 and width >= 4 and m["length_m"]:
            cause += (
                f" Отметка на {100.0 * m['x1'] / m['length_m']:.0f}% длины. "
                "Шум антистокса к концу выше, но часы ровного пятна — не шум одной ячейки. "
                "Амплитуду читать осторожнее, чем в начале линии; причину искать как на середине трассы."
            )
        else:
            cause += (
                f" Отметка близка к дальнему концу ({m['length_m']:.0f} м): шум антистокса здесь выше, чем в начале."
            )
            if conf == "высокая":
                conf = "средняя"
    if info.get("tz_missing"):
        cause += " У файла нет пояса времени, метки прочитаны как UTC и могут быть сдвинуты относительно карточек."
        if conf == "высокая":
            conf = "средняя"
    if sign_clash:
        cause += " Знак полярности и имя класса расходятся: ориентир — полярность и рисунок ΔT."
        conf = "низкая"
        tag = "знак и класс расходятся"
    elif prob < 0.40 and conf == "высокая":
        conf = "средняя"
    if cls == "Sensor_Artifact":
        cause += " Классификатор сам поставил приборный артефакт."
        conf = "низкая"
        tag = "приборный артефакт"
    return cause, conf, check, tag


def _names_in(text: str):
    """Имена релизов в тексте: раньше по строке, при равенстве позиции — более длинное."""
    found = []
    for name in RELEASES:
        match = re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", text)
        if match:
            found.append((match.start(), -len(name), name))
    found.sort()
    return found


def release_in_decision_line(line: str):
    """Имя релиза в строке RELEASE_DECISION.

    Имя может стоять среди других слов. Если в хвосте после маркера несколько
    известных имён, берётся первое: строка «RELEASE_DECISION ens_gen2 … ens_u2ft_tcn_lgbm»
    — это ens_gen2, смесь в пояснении победителя решение не перебивает.
    """
    if "RELEASE_DECISION" not in line:
        return None
    tail = line.split("RELEASE_DECISION", 1)[1]
    found = _names_in(tail) or _names_in(line)
    if not found:
        return None
    return found[0][2]


def last_release_name(events: Path):
    if not events.is_file():
        return None
    last = None
    for line in events.read_text(encoding="utf-8", errors="replace").splitlines():
        if "RELEASE_DECISION" in line:
            last = line
    if last is None:
        return None
    name = release_in_decision_line(last)
    if name is None:
        raise SystemExit(f"строка RELEASE_DECISION не из известных релизов ({', '.join(RELEASES)}): {last}")
    return name


def resolve_release(explicit: str | None) -> str:
    if explicit:
        if explicit not in RELEASES:
            known = ", ".join(RELEASES)
            raise SystemExit(f"неизвестный --release {explicit}. Известны: {known}")
        return explicit
    found = last_release_name(EVENTS)
    return found or "ens_u2ft_tcn_lgbm"


def dest_png(prefix: str, name: str, kind: str) -> Path:
    if prefix not in ("draft", "final"):
        raise SystemExit(prefix)
    if not re.fullmatch(r"[0-9A-Za-z_-]+", str(name)):
        raise SystemExit(f"странное имя записи {name}")
    if kind == "main":
        base = f"real_{prefix}_{name}.png"
    elif kind == "cv":
        base = f"real_{prefix}_cv_{name}.png"
    else:
        raise SystemExit(kind)
    banned = {f"real_{i}.png" for i in range(1, 10)} | {f"real_cv_{i}.png" for i in range(1, 10)}
    if base in banned or base.startswith("real_cv_"):
        raise SystemExit(f"отказ писать {base}")
    return FIG_DIR / base


def build_argv(release: str, static: str):
    """Флаги run_real. static=on только у ens_u2ft_tcn_lgbm; ens_mean4 несёт mae=on, ens_gen2 — gen2=on."""
    if static not in ("off", "on"):
        raise SystemExit("--static должен быть off или on")
    if release not in RELEASES:
        raise SystemExit(f"неизвестный релиз {release}")
    if static == "on" and release != "ens_u2ft_tcn_lgbm":
        raise SystemExit(
            "static=on только с релизом ens_u2ft_tcn_lgbm. "
            "ens_mean4 (--mae on), ens_dom и ens_gen2 (--gen2 on) со static вместе не калибровались, "
            "run_real такой запуск отвергает."
        )
    argv = []
    pairs = list(zip(RELEASES[release]["argv"][0::2], RELEASES[release]["argv"][1::2]))
    seen_static = False
    for key, value in pairs:
        if key == "--static":
            value = static
            seen_static = True
        argv.extend([key, value])
    if not seen_static:
        argv.extend(["--static", static])
    if release == "ens_mean4":
        mae_at = argv.index("--mae") + 1
        if argv[mae_at] != "on":
            raise SystemExit("ens_mean4 обязан идти с --mae on")
    if release == "ens_gen2":
        gen2_at = argv.index("--gen2") + 1
        if argv[gen2_at] != "on" or argv[argv.index("--static") + 1] != "off":
            raise SystemExit("ens_gen2 обязан идти с --gen2 on и --static off")
        if argv[argv.index("--mae") + 1] != "off" or argv[argv.index("--detector") + 1] != "winner":
            raise SystemExit("ens_gen2 обязан идти с --detector winner и --mae off")
    return argv


def run_real(release: str, prefix: str, real_root: str, tmp: Path, static: str) -> None:
    out = ROOT / "artifacts/eval" / f"real_{prefix}.json"
    csv_path = ROOT / "artifacts/eval" / f"real_{prefix}.csv"
    owned_resolved = {(ROOT / rel).resolve() for rel in OWNED}
    for path in (out, csv_path, tmp):
        if path.resolve() in owned_resolved or path.resolve() == FIG_DIR.resolve():
            raise SystemExit(f"отказ писать в занятый путь {path}")
    if not all(p.is_file() for p in CARDS_BATCH):
        raise SystemExit("нет data/operator_cards/cards_batch1.jsonl или cards_batch2.jsonl")
    if not PY.is_file():
        raise SystemExit(f"нет интерпретатора {PY}")
    cmd = [
        str(PY), "-u", str(ROOT / "scripts/run_real.py"),
        "--real-root", real_root,
        "--cv-annot",
        "--cards", *[str(p) for p in CARDS_BATCH],
        "--out", str(out),
        "--zones-csv", str(csv_path),
        "--fig", str(tmp),
        *build_argv(release, static),
    ]
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "NUMEXPR_NUM_THREADS": "2",
        "VECLIB_MAXIMUM_THREADS": "2",
        "MPLBACKEND": "Agg",
    })
    log("run_real " + " ".join(cmd[2:]))
    proc = subprocess.Popen(
        cmd, cwd=str(ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    code = proc.wait()
    if code != 0:
        raise SystemExit(f"run_real завершился с кодом {code}")


def relocate(tmp: Path, prefix: str, names) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        src = tmp / f"real_{name}.png"
        cv = tmp / f"real_cv_{name}.png"
        if not src.is_file():
            raise SystemExit(f"нет рисунка {src}")
        if not cv.is_file():
            raise SystemExit(f"нет cv-рисунка {cv} (--cv-annot)")
        for path in (src, cv):
            with path.open("rb") as fh:
                sig = fh.read(8)
            if sig != b"\x89PNG\r\n\x1a\n" or path.stat().st_size < 5000:
                raise SystemExit(f"пустой или битый рисунок {path}")
        os.replace(src, dest_png(prefix, name, "main"))
        os.replace(cv, dest_png(prefix, name, "cv"))


def load_jsonl(path: Path):
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def card_window(card):
    stamps = []
    for raw in card["time"]:
        stamp = datetime.fromisoformat(str(raw))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        stamps.append(stamp.astimezone(timezone.utc).timestamp())
    x0, x1 = float(card["meters"][0]), float(card["meters"][1])
    return min(x0, x1), max(x0, x1), min(stamps), max(stamps)


def alarm_hits(zones, x0, x1, t0, t1):
    hits = []
    for zone in zones:
        if not is_alarm(zone):
            continue
        if zone["end_m"] >= x0 and zone["start_m"] <= x1 and zone["t_end_s"] >= t0 and zone["t_start_s"] <= t1:
            hits.append(zone)
    return hits


def read_review_csv(path: Path):
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        return {row["id"]: row for row in csv.DictReader(fh)}


def score_review2(files):
    cards = load_jsonl(CARDS2)
    table = read_review_csv(REVIEW2_CSV)
    rows = []
    for card in cards:
        name = str(card.get("record"))
        zones = (files.get(name) or {}).get("zones") or []
        try:
            x0, x1, t0, t1 = card_window(card)
        except (KeyError, TypeError, ValueError):
            rows.append({"id": card.get("id"), "record": name, "presence": card.get("presence"), "status": "bad_window", "hit": False})
            continue
        hits = alarm_hits(zones, x0, x1, t0, t1)
        rows.append({
            "id": card.get("id"),
            "record": name,
            "presence": card.get("presence") or "",
            "shape": card.get("shape") or "",
            "status": "ok",
            "hit": bool(hits),
            "n_alarm": len(hits),
        })
    return rows, table


def presence_counts(rows, presence):
    subset = [row for row in rows if row.get("presence") == presence and row.get("status") == "ok"]
    return len(subset), sum(1 for row in subset if row.get("hit"))


def published_winner_counts(table, presence):
    subset = [row for row in table.values() if (row.get("user") or "").strip() == presence]
    hits = sum(1 for row in subset if str(row.get("winner")) in ("1", "1.0"))
    return len(subset), hits


def narrow_static(table, by_id):
    """Узкие статичные пропуски из определения CARDS_REVIEW_2. Порог по ним не выбирается."""
    found = []
    for cid, row in table.items():
        user = (row.get("user") or "").strip()
        if not user.startswith("yes"):
            continue
        try:
            winner = int(float(row["winner"]))
            dT = float(row["dT_hi"])
            width = float(row["x1"]) - float(row["x0"])
        except (TypeError, ValueError, KeyError):
            continue
        if winner != 0 or width > 12.0 or dT < 3.0:
            continue
        hit = by_id.get(cid)
        found.append({
            "id": cid,
            "record": row.get("rec") or "",
            "width_m": round(width, 2),
            "dT_hi": dT,
            "hit": bool(hit and hit.get("hit")),
        })
    return found


def zone_rows_ok(csv_path: Path, files) -> int:
    with csv_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    expected = sum(len(info.get("zones") or []) for info in files.values())
    if len(rows) != expected:
        raise SystemExit(f"csv строк {len(rows)}, зон в json {expected}")
    return len(rows)


def _heat_cold(name: str) -> str:
    text = name or ""
    if "Cooling" in text:
        return "cold"
    if "Heating" in text or "Contact" in text:
        return "hot"
    return ""


def batch_reading(cards) -> str:
    """Описательное чтение партий 1–2. Порог не двигает."""
    usable = [c for c in cards if c.get("status") != "box_outside_record"]
    yes = [c for c in usable if c.get("presence") == "yes"]
    missed = [c for c in yes if not c.get("cnn_n_alarm_zones")]
    clashes = []
    for card in usable:
        if not card.get("cnn_n_alarm_zones"):
            continue
        op = _heat_cold(card.get("shape") or "")
        dom = _heat_cold(card.get("cnn_dominant_class") or "")
        if op and dom and op != dom:
            clashes.append(str(card.get("id")))
    parts = []
    if yes:
        parts.append(
            f"Из {len(yes)} рамок с ответом yes без тревоги модели остались {len(missed)}. "
            "Счёт идёт по всем тревогам файла, не по короткому списку выше."
        )
    if clashes:
        shown = ", ".join(clashes[:4])
        tail = f" и ещё {len(clashes) - 4}" if len(clashes) > 4 else ""
        n_clash = len(clashes)
        clash_word = "рамке" if n_clash % 10 == 1 and n_clash % 100 != 11 else "рамках"
        parts.append(
            f"Знак не совпал в {n_clash} {clash_word} ({shown}{tail}): оператор отметил один знак, "
            "главный тип модели — другой. Рамка пересеклась, физика нагрева и охлаждения — нет. "
            "Карточка смотрит сырой график, модель — ΔT после вычитания профиля; остаток может сменить знак."
        )
    nos = [c for c in usable if c.get("presence") == "no"]
    if nos:
        no_hit = sum(1 for c in nos if c.get("cnn_n_alarm_zones"))
        parts.append(
            f"Ответы no: тревога попала в {no_hit} из {len(nos)}. "
            "На этих рамках модель почти не рисует лишнего. Порог из-за этого не двигался."
        )
    return " ".join(parts)


def _count_zones(files) -> dict:
    out = {}
    for name, info in (files or {}).items():
        zones = info.get("zones") or []
        nz = info.get("n_zones_cnn")
        na = info.get("n_alarm_zones_cnn")
        if nz is None:
            nz = len(zones)
        if na is None:
            na = sum(1 for zone in zones if is_alarm(zone))
        out[str(name)] = (int(nz), int(na))
    return out


def _presence_alarm(cards, presence: str):
    usable = [
        card for card in (cards or [])
        if card.get("status") != "box_outside_record" and card.get("presence") == presence
    ]
    hit = [card for card in usable if card.get("cnn_n_alarm_zones")]
    return usable, hit


def _time_hit(a, b) -> bool:
    if _overlap_s(a, b) > 0:
        return True
    return bool(a["dur_s"] < 1.0 and b["dur_s"] < 1.0 and abs(_t0(a) - _t0(b)) < 1.0)


def _zones_share(a, b) -> bool:
    """Один очаг: тот же файл, та же полярность, рамки пересекаются по метрам и по времени."""
    if a["file"] != b["file"] or a["polarity"] != b["polarity"]:
        return False
    if not (a["x1"] >= b["x0"] and b["x1"] >= a["x0"]):
        return False
    return _time_hit(a, b)


def _match_zones(left, right):
    used = set()
    shared = []
    only_left = []
    for row in left:
        best_i = None
        best = -1.0
        for i, other in enumerate(right):
            if i in used or not _zones_share(row, other):
                continue
            score = _overlap_s(row, other) + 1.0
            width = min(row["x1"], other["x1"]) - max(row["x0"], other["x0"])
            score += max(0.0, width)
            if score > best:
                best = score
                best_i = i
        if best_i is None:
            only_left.append(row)
        else:
            used.add(best_i)
            shared.append((row, right[best_i]))
    only_right = [row for i, row in enumerate(right) if i not in used]
    return shared, only_left, only_right


def _zone_brief(row) -> str:
    return f"{_zid(row)} ({fmt_m(row['x0'])}–{fmt_m(row['x1'])} м, {polarity_ru(row['polarity'])})"


def comparison_pair(release: str, prefix: str, res: dict):
    """Победитель и ens_gen2, если оба прогона есть. Файлы только читаются."""
    this_is_gen2 = release == "ens_gen2" or res.get("gen2") == "on" or res.get("detector") == "ens_blend"
    this_is_winner = (
        release == "ens_u2ft_tcn_lgbm"
        and res.get("gen2") != "on"
        and res.get("mae") != "on"
        and res.get("static") != "on"
        and res.get("detector") not in ("ens_blend", "ens_dom", "ens_mean4")
    )
    if this_is_gen2:
        gen2_res = res
        gen2_src = f"artifacts/eval/real_{prefix}.json"
    else:
        gen2_res = json.loads(GEN2_JSON.read_text(encoding="utf-8")) if GEN2_JSON.is_file() else None
        gen2_src = "artifacts/eval/real_gen2.json"
    if this_is_winner:
        winner_res = res
        winner_src = f"artifacts/eval/real_{prefix}.json"
    else:
        winner_res = json.loads(WINNER_JSON.read_text(encoding="utf-8")) if WINNER_JSON.is_file() else None
        winner_src = "artifacts/eval/real_v8.json"
    if not gen2_res or not winner_res:
        return None
    if not gen2_res.get("files") or not winner_res.get("files"):
        return None
    return {
        "winner": winner_res,
        "gen2": gen2_res,
        "winner_src": winner_src,
        "gen2_src": gen2_src,
    }


def comparison_lines(pair: dict) -> list:
    winner = pair["winner"]
    gen2 = pair["gen2"]
    w_counts = _count_zones(winner.get("files"))
    g_counts = _count_zones(gen2.get("files"))
    names = sorted(set(w_counts) | set(g_counts), key=file_key)
    w_sig = select_significant(winner.get("files") or {})
    g_sig = select_significant(gen2.get("files") or {})
    shared, only_w, only_g = _match_zones(w_sig, g_sig)
    lines = [
        "## Сравнение победителя и ens_gen2",
        "",
        "Оба прогона уже есть, поэтому здесь они стоят рядом. "
        "Победитель прочитан из `" + pair["winner_src"] + "`, ens_gen2 — из `" + pair["gen2_src"] + "`. "
        "Голый `run_real.py --gen2 on` кладёт свой json в `artifacts/eval/real_gen2.json`, "
        "csv в `artifacts/eval/real_gen2.csv`, рисунки в `artifacts/eval/real_gen2_fig`. "
        "Эти пути не подменяют `real_v8.json` и `docs/figures/real_1.png` … `real_6.png`. "
        f"Порог победителя {winner.get('threshold')}, порог ens_gen2 {gen2.get('threshold')}. "
        "Оба заморожены на синтетике. Этот раздел описание: по нему порог не двигался.",
        "",
        "| файл | зоны победителя | тревоги победителя | зоны ens_gen2 | тревоги ens_gen2 |",
        "|---|---:|---:|---:|---:|",
    ]
    sum_w = [0, 0]
    sum_g = [0, 0]
    for name in names:
        wz, wa = w_counts.get(name, (0, 0))
        gz, ga = g_counts.get(name, (0, 0))
        sum_w[0] += wz
        sum_w[1] += wa
        sum_g[0] += gz
        sum_g[1] += ga
        lines.append(f"| {name} | {wz} | {wa} | {gz} | {ga} |")
    lines.append(f"| всего | {sum_w[0]} | {sum_w[1]} | {sum_g[0]} | {sum_g[1]} |")
    lines.append("")
    lines.append(
        f"Значимые зоны — тот же короткий список, что в разделе выше "
        f"(у победителя {len(w_sig)}, у ens_gen2 {len(g_sig)}). "
        "Общая зона — тот же файл, та же полярность и пересечение рамок по метрам и по времени. "
        "Сосед без пересечения остаётся зоной только одного детектора."
    )
    lines.append("")

    def _join(rows, empty):
        if not rows:
            return empty
        return "; ".join(_zone_brief(row) for row in rows)

    lines.append("Общие значимые зоны:")
    lines.append("")
    if shared:
        for left, right in shared:
            lines.append(f"- {_zone_brief(left)} и {_zone_brief(right)}")
    else:
        lines.append("- пересечений нет")
    lines.append("")
    lines.append("Только у победителя: " + _join(only_w, "нет") + ".")
    lines.append("")
    lines.append("Только у ens_gen2: " + _join(only_g, "нет") + ".")
    lines.append("")
    lines += [
        "Карточки партий 1 и 2. Попадание — тревожная зона прогона пересекает рамку. "
        "Ответ оператора остаётся мнением по графику. Порог из-за этих долей не менялся.",
        "",
        "| ответ | рамок у победителя | с тревогой | рамок у ens_gen2 | с тревогой |",
        "|---|---:|---:|---:|---:|",
    ]
    for presence in ("yes", "no"):
        w_all, w_hit = _presence_alarm(winner.get("cards"), presence)
        g_all, g_hit = _presence_alarm(gen2.get("cards"), presence)
        lines.append(f"| {presence} | {len(w_all)} | {len(w_hit)} | {len(g_all)} | {len(g_hit)} |")
    lines.append("")
    w_yes = {str(card.get("id")) for card in _presence_alarm(winner.get("cards"), "yes")[1]}
    g_yes = {str(card.get("id")) for card in _presence_alarm(gen2.get("cards"), "yes")[1]}
    w_no = {str(card.get("id")) for card in _presence_alarm(winner.get("cards"), "no")[1]}
    g_no = {str(card.get("id")) for card in _presence_alarm(gen2.get("cards"), "no")[1]}

    def _id_list(ids):
        shown = sorted(ids)
        if not shown:
            return "нет"
        if len(shown) <= 12:
            return ", ".join(shown)
        return ", ".join(shown[:12]) + f" и ещё {len(shown) - 12}"

    lines.append(
        "Yes с тревогой у обоих: " + _id_list(w_yes & g_yes) + ". "
        "Yes с тревогой только у победителя: " + _id_list(w_yes - g_yes) + ". "
        "Yes с тревогой только у ens_gen2: " + _id_list(g_yes - w_yes) + "."
    )
    lines.append("")
    lines.append(
        "No с тревогой у обоих: " + _id_list(w_no & g_no) + ". "
        "No с тревогой только у победителя: " + _id_list(w_no - g_no) + ". "
        "No с тревогой только у ens_gen2: " + _id_list(g_no - w_no) + "."
    )
    lines.append("")
    if CARDS2.is_file():
        w_rows, _table = score_review2(winner.get("files") or {})
        g_rows, _table = score_review2(gen2.get("files") or {})
        if w_rows and g_rows:
            lines.append(
                "Те же 150 рамок `CARDS_REVIEW_2`, пересчитанные по зонам каждого прогона. "
                "Колонка прошлого `winner` в csv не подгонялась."
            )
            lines.append("")
            lines.append("| ответ | рамок | тревога победителя | тревога ens_gen2 |")
            lines.append("|---|---:|---:|---:|")
            for presence in ("yes", "no", "unclear"):
                wn, wh = presence_counts(w_rows, presence)
                gn, gh = presence_counts(g_rows, presence)
                lines.append(f"| {presence} | {wn} | {wh} | {gh} |")
            lines.append("")
            lines.append("Счёт описательный. Порог по этим рамкам не выбирался.")
            lines.append("")
    if GEN2_JSON.is_file() and pair["gen2_src"] != "artifacts/eval/real_gen2.json":
        prior = json.loads(GEN2_JSON.read_text(encoding="utf-8"))
        prior_counts = _count_zones(prior.get("files"))
        prior_zones = sum(nz for nz, _na in prior_counts.values())
        prior_alarms = sum(na for _nz, na in prior_counts.values())
        same = prior_zones == sum_g[0] and prior_alarms == sum_g[1] and float(prior.get("threshold") or -1) == float(gen2.get("threshold") or -2)
        if same:
            lines.append(
                f"Ранее записанный `artifacts/eval/real_gen2.json` даёт те же {prior_zones} зон и {prior_alarms} тревог "
                f"при пороге {prior.get('threshold')}. Этот черновик его не переписывал."
            )
        else:
            lines.append(
                f"Ранее записанный `artifacts/eval/real_gen2.json`: {prior_zones} зон, {prior_alarms} тревог, "
                f"порог {prior.get('threshold')}. В таблице выше счёт ens_gen2 взят из `{pair['gen2_src']}`. "
                "Файл `real_gen2.json` этот скрипт не переписывал."
            )
        lines.append("")
    return lines


def _read_json(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _dig(obj, *keys):
    cur = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _num(value):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        got = float(value)
        if got == got:
            return got
    return None


def _first_num(specs):
    """Первое число по списку (файл, ключи). Нет ключа — дальше по списку."""
    cache = getattr(_first_num, "_cache", None)
    if cache is None:
        cache = {}
        _first_num._cache = cache
    for spec in specs:
        path, *keys = spec
        token = str(path)
        if token not in cache:
            cache[token] = _read_json(path)
        got = _num(_dig(cache[token], *keys))
        if got is not None:
            return got
    return None


def _fmt_num(value, digits=4, signed=False) -> str:
    if value is None:
        return "н/д"
    if signed:
        return f"{value:+.{digits}f}"
    return f"{value:.{digits}f}"


def _winner_iou(split: str):
    """IoU победителя @FAR1e-3. split: val, test, v10, v11."""
    if split == "val":
        return _first_num([
            (SYNTH_GEN2_V8, "val", "ens_u2ft_tcn_lgbm", "iou"),
            (SYNTH_GEN2_V8, "winner_recomputed", "val"),
            (SYNTH_GEN2_V8, "published_winner", "val_iou"),
            (SYNTH_COMBO, "winner_published", "val"),
            (SYNTH_GEN2_CHECK, "winner_full_val_iou"),
        ])
    if split == "test":
        return _first_num([
            (SYNTH_GEN2_V8, "test", "ens_u2ft_tcn_lgbm", "iou"),
            (SYNTH_GEN2_V8, "winner_recomputed", "test"),
            (SYNTH_GEN2_V8, "published_winner", "test_iou"),
            (SYNTH_COMBO, "winner_published", "test"),
            (SYNTH_GEN2_CHECK, "bootstrap", "test", "iou", "ens_u2ft_tcn_lgbm"),
        ])
    if split == "v10":
        return _first_num([
            (SYNTH_V10, "test", "ens_u2ft_tcn_lgbm@far_1e-3", "iou"),
            (SYNTH_GEN2_CHECK, "bootstrap", "v10", "iou", "ens_u2ft_tcn_lgbm"),
            (SYNTH_COMBO, "winner_published", "v10"),
            (SYNTH_GEN2_V8, "winner_recomputed", "v10"),
        ])
    if split == "v11":
        return _first_num([
            (SYNTH_V11, "test", "ens_u2ft_tcn_lgbm@far_1e-3", "iou"),
            (SYNTH_GEN2_CHECK, "bootstrap", "v11", "iou", "ens_u2ft_tcn_lgbm"),
            (SYNTH_COMBO, "winner_published", "v11"),
            (SYNTH_GEN2_V8, "winner_recomputed", "v11"),
        ])
    return None


def _gen2_iou(split: str):
    """IoU ens_blend (релиз ens_gen2) @FAR1e-3."""
    if split == "val":
        return _first_num([
            (SYNTH_GEN2_V8, "val", "ens_blend", "iou"),
            (SYNTH_GEN2_V8, "selected_val_iou"),
            (SYNTH_COMBO, "blend_published", "val"),
            (SYNTH_COMBO, "a", "val45", "iou"),
            (SYNTH_GEN2_CHECK, "published_ens_blend", "val"),
        ])
    if split == "test":
        return _first_num([
            (SYNTH_GEN2_V8, "test", "ens_blend", "iou"),
            (SYNTH_GEN2_V8, "selected_test_iou"),
            (SYNTH_COMBO, "blend_published", "test"),
            (SYNTH_GEN2_CHECK, "published_ens_blend", "test"),
            (SYNTH_GEN2_CHECK, "bootstrap", "test", "iou", "ens_blend"),
        ])
    if split == "v10":
        return _first_num([
            (SYNTH_GEN2_CHECK, "bootstrap", "v10", "iou", "ens_blend"),
            (SYNTH_GEN2_CHECK, "published_ens_blend", "v10"),
            (SYNTH_COMBO, "blend_published", "v10"),
        ])
    if split == "v11":
        return _first_num([
            (SYNTH_GEN2_CHECK, "bootstrap", "v11", "iou", "ens_blend"),
            (SYNTH_GEN2_CHECK, "published_ens_blend", "v11"),
            (SYNTH_COMBO, "blend_published", "v11"),
        ])
    return None


def _nested_block(k: str):
    data = _read_json(SYNTH_GEN2_CHECK)
    block = _dig(data, "nested41", "ens_blend", k)
    if not isinstance(block, dict):
        return None, None, None
    winner = _num(block.get("winner_iou"))
    blend = _num(block.get("selected_iou"))
    delta = _num(block.get("delta"))
    if delta is None and winner is not None and blend is not None:
        delta = blend - winner
    return winner, blend, delta


def _holdout_pair(kind: str):
    """kind: frozen | refit. Возвращает iou победителя, iou смеси, дельту."""
    data = _read_json(SYNTH_GEN2_CHECK)
    key = "holdout_at_frozen_threshold" if kind == "frozen" else "holdout_refit"
    winner = _num(_dig(data, key, "ens_u2ft_tcn_lgbm", "iou"))
    blend = _num(_dig(data, key, "ens_blend", "iou"))
    if winner is None or blend is None:
        combo_key = "hold41_frozen" if kind == "frozen" else "hold41_refit"
        combo = _dig(_read_json(SYNTH_COMBO), "a", combo_key)
        if isinstance(combo, dict):
            winner = winner if winner is not None else _num(combo.get("winner_iou"))
            blend = blend if blend is not None else _num(combo.get("iou"))
    delta = None if winner is None or blend is None else blend - winner
    return winner, blend, delta


def _boot_block(key: str):
    return _dig(_read_json(SYNTH_GEN2_CHECK), "bootstrap", key)


def _ci_cell(block) -> str:
    if not isinstance(block, dict):
        return "н/д"
    point = _num(block.get("point"))
    lo = _num(block.get("p2_5"))
    hi = _num(block.get("p97_5"))
    if point is None or lo is None or hi is None:
        return "н/д"
    return f"{point:+.4f} [{lo:+.4f}; {hi:+.4f}]"


def synth_quality_lines() -> list:
    rows = [
        ("v8 val", "val"),
        ("v8 test", "test"),
        ("v10 test", "v10"),
        ("v11 test", "v11"),
    ]
    lines = [
        "## Качество модели на синтетике",
        "",
        "IoU при FAR 1e-3 — доля пикселей, где модель и синтетическая разметка вместе отметили аномалию, "
        "среди всех пикселей, которые отметила хотя бы одна из сторон. "
        "Порог ставится так, чтобы на спокойном фоне синтетики ложных пикселей было около одной тысячной. "
        "Реальные CSV в эту таблицу не входят.",
        "",
        "| сплит | ens_u2ft_tcn_lgbm | ens_gen2 (ens_blend) |",
        "|---|---:|---:|",
    ]
    for label, split in rows:
        lines.append(f"| {label} | {_fmt_num(_winner_iou(split))} | {_fmt_num(_gen2_iou(split))} |")
    lines.append("")
    lines.append(
        "Дальше — проверка, что прирост не держится на четырёх файлах, по которым выбирали чекпойнт. "
        "Held-out 41 — остальные последовательности v8 val. Вложенные k=5 и k=2 считают порог на обучающей части фолда, "
        "а IoU — на отложенной. Дельта — IoU смеси минус IoU победителя. "
        "95% ДИ — процентили 2.5 и 97.5 бутстрэпа по последовательностям. Нет ключа в json — «н/д»."
    )
    lines.append("")
    lines.append("| проверка | победитель | ens_gen2 | дельта |")
    lines.append("|---|---:|---:|---:|")
    for label, kind in (
        ("held-out 41, порог полного v8 val", "frozen"),
        ("held-out 41, порог заново на этих 41", "refit"),
    ):
        winner, blend, delta = _holdout_pair(kind)
        lines.append(
            f"| {label} | {_fmt_num(winner)} | {_fmt_num(blend)} | {_fmt_num(delta, signed=True)} |"
        )
    for label, k in (("вложенная k=5", "5"), ("вложенная k=2", "2")):
        winner, blend, delta = _nested_block(k)
        lines.append(
            f"| {label} | {_fmt_num(winner)} | {_fmt_num(blend)} | {_fmt_num(delta, signed=True)} |"
        )
    lines.append("")
    lines.append("| бутстрэп дельты | точка и 95% ДИ |")
    lines.append("|---|---|")
    boot_rows = (
        ("held-out 41, порог полного v8 val", "val_hold41_frozen_thr"),
        ("held-out 41, порог заново на этих 41", "val_hold41_refit"),
        ("v8 test", "test"),
        ("v10 test", "v10"),
        ("v11 test", "v11"),
    )
    notes = []
    for label, key in boot_rows:
        block = _boot_block(key)
        lines.append(f"| {label} | {_ci_cell(block)} |")
        if isinstance(block, dict) and isinstance(block.get("covers_zero"), bool):
            word = "включает ноль" if block["covers_zero"] else "не включает ноль"
            notes.append(f"{label} — интервал {word}")
    lines.append("")
    if notes:
        lines.append("По флагу `covers_zero` в json: " + "; ".join(notes) + ".")
    else:
        lines.append("Флага `covers_zero` в json нет: н/д.")
    lines.append("")
    lines.append(
        "Источники: `artifacts/eval/s3_gen2_v8.json`, `artifacts/eval/s3_gen2_check.json`, "
        "`artifacts/eval/s3_combo_report.json`, `artifacts/eval/v10_ens_u2ft_tcn_lgbm.json`, "
        "`artifacts/eval/v11_ens_u2ft_tcn_lgbm.json`."
    )
    lines.append("")
    return lines


def _decision_blurb(release: str) -> str:
    """Последняя строка RELEASE_DECISION, если в ней есть имя релиза. Иначе формулировка по умолчанию."""
    last = None
    if EVENTS.is_file():
        for line in EVENTS.read_text(encoding="utf-8", errors="replace").splitlines():
            if "RELEASE_DECISION" in line:
                last = line.strip()
    named = release_in_decision_line(last) if last else None
    if named:
        text = last
        if len(text) > 400:
            text = text[:400].rstrip() + "…"
        return f"Этот текст собран с релиза `{release}`. Журнал называет решение `{named}`: {text}"
    return (
        f"Этот текст собран с релиза `{release}`. "
        "Релиз по умолчанию ens_u2ft_tcn_lgbm, решение в 04:15."
    )


def _why_phrase(m, tag: str) -> str:
    dur = "одна ячейка времени" if m["single_bin"] else fmt_dur(m["dur_s"])
    if tag == "муфта, ввод или труба":
        return f"нагрев {m['width']:.1f} м держится {dur}, это шире шума одной клетки"
    if tag in ("климат, солнце или вентиляция", "солнце, окна или вывод наружу"):
        return f"пятно {m['width']:.0f} м и {dur}: одна муфта так широко не греет"
    if tag == "труба, шахта или сквозняк":
        return f"охлаждение {m['amp']:.1f} °C на {m['width']:.1f} м держится {dur}"
    if tag == "провал, склейка или улица":
        return f"провал около {m['amp']:.0f} °C для воздуха в комнате слишком глубок"
    if tag == "склейка или коннектор":
        return "рамка уже разрешения прибора около 1 м"
    if tag == "начало линии, прибор":
        return "первые метры — пигтейл и коннектор, муфта здесь во вторую очередь"
    if tag == "одна ячейка усреднения":
        return "по времени пятно не подтверждено, одну ячейку в наряд не ставят"
    if tag == "дальняя часть линии":
        return "к концу волокна шум выше, короткий пик легко случаен"
    if tag == "около порога FAR 1e-3":
        return "амплитуда на уровне шума при этом пороге"
    if "расходятся" in tag:
        return "знак полярности и имя класса расходятся, ориентир — рисунок ΔT"
    return f"{tag}, ширина {m['width']:.1f} м, {dur}"


def _conf_rank(conf: str, single_bin: bool) -> int:
    rank = {"высокая": 2, "средняя": 1, "низкая": 0}.get(conf, 0)
    if single_bin:
        return min(rank, 0)
    return rank


def _top_visit_lines(files, limit=5) -> list:
    rows = []
    for name, info in (files or {}).items():
        for zone in info.get("zones") or []:
            if is_alarm(zone):
                rows.append(zone_metrics(str(name), zone, info))
    pool = _dedup(rows)
    ranked = []
    for row in pool:
        _cause, conf, _check, tag = interpret(row["zone"], row["info"], [row])
        ranked.append((row, conf, tag))
    ranked.sort(key=lambda item: (_conf_rank(item[1], item[0]["single_bin"]), _interest(item[0])), reverse=True)
    out = []
    for i, (row, conf, tag) in enumerate(ranked[:limit], 1):
        zone = row["zone"]
        t0 = fmt_utc(zone["t_start_s"])
        t1 = fmt_utc(zone["t_end_s"])
        out.append(
            f"{i}. Файл {row['file']}, {fmt_m(row['x0'])}–{fmt_m(row['x1'])} м, "
            f"{t0}–{t1} UTC, ΔT {signed_peak(zone):+.2f} °C. "
            f"Причина: {tag}. Уверенность: {conf}. Почему: {_why_phrase(row, tag)}."
        )
    return out


def summary_lines(release: str, files, picked) -> list:
    names = sorted(files or {}, key=file_key)
    bits = []
    n_alarm = 0
    for name in names:
        info = files[name]
        zones = info.get("zones") or []
        na = info.get("n_alarm_zones_cnn")
        if na is None:
            na = sum(1 for zone in zones if is_alarm(zone))
        na = int(na)
        n_alarm += na
        ns = sum(1 for row in picked if row["file"] == str(name))
        bits.append(f"{name}: {na}/{ns}")
    per_file = ", ".join(bits) if bits else "файлов нет"
    lines = [
        "## Краткое заключение",
        "",
        "Шесть записей МГТС, файлы 1.csv–6.csv: температура оптического волокна вдоль кабеля и по времени. "
        "Это оценка готовой модели. На этих файлах метки не ставились и порог не подбирался.",
        "",
        _decision_blurb(release),
        "",
        f"Тревог {n_alarm}, в короткий список для разбора {len(picked)}. "
        f"По файлам тревоги/в списке: {per_file}.",
        "",
        "До пяти мест для выезда. Выше в списке — сильнее по амплитуде, длительности и уверенности суждения. "
        "Это не журнал дефектов.",
        "",
    ]
    visits = _top_visit_lines(files)
    if not visits:
        lines.append("Тревожных зон для выезда нет.")
    else:
        lines.extend(visits)
    lines.append("")
    lines.append(
        "Надёжность. Модель хорошо видит локальный нагрев от 3–5 °C, который держится часами. "
        "Плохо видит слабое охлаждение до 2 °C, постоянные узкие горячие точки и перенос с синтетики на реальную трассу."
    )
    if len(lines) > 25:
        raise SystemExit(f"краткое заключение {len(lines)} строк, лимит 25")
    return lines


def render(release, prefix, res, review_rows, review_table, wall_s, reused=False, compare=None) -> str:
    spec = RELEASES[release]
    files = res.get("files") or {}
    names = sorted(files, key=file_key)
    n_zones = sum(len(info.get("zones") or []) for info in files.values())
    n_alarm = sum(1 for info in files.values() for zone in (info.get("zones") or []) if is_alarm(zone))
    picked = select_significant(files)
    picked_by = {}
    for row in picked:
        picked_by.setdefault(row["file"], []).append(row)
    thr = res.get("threshold")
    when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    minutes = wall_s / 60.0
    if prefix == "draft":
        banner = (
            "> **ЧЕРНОВИК.** Отчёт снят до строки `RELEASE_DECISION`, с релиза "
            f"`{release}` и флагов по умолчанию. В 04:15 UTC та же команда с "
            "`--out-prefix final` перезапишет этот файл итогом. Числа ниже — этого прогона."
        )
        title_note = "Черновик"
    else:
        banner = (
            "> **Итоговый отчёт** по релизу, который передан в `--release` "
            "(или по последней строке `RELEASE_DECISION`, если флаг не задан). "
            "Числа ниже — этого прогона."
        )
        title_note = "Итоговый отчёт"
    if reused:
        time_note = "Текст пересобран из готового json, csv и рисунков, без нового инференса."
    elif wall_s <= 20 * 60:
        time_note = f"Прогон занял {minutes:.1f} мин. Это укладывается в окно 20 минут."
    else:
        time_note = (
            f"Прогон занял {minutes:.1f} мин. Это дольше окна 20 минут: "
            "узкое место — инференс шести CSV, кэш заново не строился."
        )
    lines = [
        f"# Экспертный отчёт по реальным записям МГТС ({title_note})",
        "",
        banner,
        "",
        f"Сгенерирован {when}. Релиз `{release}`. {time_note}",
        "",
    ]
    lines += summary_lines(release, files, picked)
    lines.append("")
    lines += synth_quality_lines()
    lines += [
        "Рисунок каждой записи — поле ΔT (температура минус синфазная и статическая составляющие) "
        "и карта вероятности аномалии. Цвет рамки — тип зоны. Сплошная рамка — тревога, пунктир — зона без тревоги. "
        "Белые рамки на правой панели — классический детектор median_x3+spatial_z, ориентир сравнения, не ответ оператора и не вход модели. "
        "Рядом лежит рисунок контуров `real_"
        + prefix
        + "_cv_<файл>.png`: та же ΔT, контур зоны и линия фронта. Порог он не меняет.",
        "",
        "## Релиз и флаги",
        "",
        spec["blurb"],
        "",
        "Флаги `scripts/run_real.py` в этом прогоне:",
        "",
        "| флаг | значение |",
        "|---|---|",
    ]
    flags = {
        "--detector": "winner",
        "--static": "off",
        "--mae": "off",
        "--calib": "global",
        "--op": "far_1e-3",
        "--zones": "cc",
        "--gen2": "off",
        "--cv-annot": "да",
    }
    argv = spec["argv"]
    for i in range(0, len(argv), 2):
        flags[argv[i]] = argv[i + 1]
    if res.get("static"):
        flags["--static"] = res["static"]
    if res.get("mae"):
        flags["--mae"] = res["mae"]
    if res.get("gen2"):
        flags["--gen2"] = res["gen2"]
    for key in ("--detector", "--static", "--mae", "--gen2", "--calib", "--op", "--zones", "--cv-annot"):
        lines.append(f"| `{key}` | {flags[key]} |")
    lines += [
        "",
        f"Порог в json этого прогона: **{thr}**. Калибровка `{res.get('calib')}`, рабочая точка `{res.get('op')}`, режим зон `{res.get('zones_mode')}`. "
        f"static=`{res.get('static')}`, mae=`{res.get('mae')}`, gen2=`{res.get('gen2', 'off')}`.",
        "",
        "Порог заморожен на синтетике. Карточки ниже только показывают, попала ли уже найденная тревога в рамку оператора.",
        "",
    ]
    if res.get("mae_branch"):
        lines.append("Ветка MAE в json: " + md_cell(res["mae_branch"].get("note", "")))
        lines.append("")
    if res.get("ens_dom"):
        lines.append("Ветка ens_dom в json: порог из `" + md_cell(res["ens_dom"].get("threshold_source", "")) + "`.")
        lines.append("")
    if res.get("gen2_branch"):
        branch = res["gen2_branch"]
        lines.append(
            "Ветка GEN2 в json: "
            + md_cell(branch.get("definition", ""))
            + ". "
            + md_cell(branch.get("note", ""))
        )
        lines.append("")

    lines += [
        "## Обзор по файлам",
        "",
        "Длина — по валидным метрам кабеля. Длительность — от первой до последней метки времени. "
        "«Тревога» — зона, чей тип не HVAC_Diurnal и не Sensor_Artifact. "
        "IoU с классикой — пересечение масок на сетке, не согласие с оператором.",
        "",
        "| файл | часы | км | порог | зоны | тревоги | в разборе | зоны классики | IoU с классикой | без пояса |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name in names:
        info = files[name]
        lines.append(
            "| {name} | {hours} | {km} | {thr:.4f} | {nz} | {na} | {ns} | {nb} | {iou:.3f} | {tz} |".format(
                name=name,
                hours=info.get("hours"),
                km=info.get("km"),
                thr=float(info.get("threshold") or 0),
                nz=info.get("n_zones_cnn"),
                na=info.get("n_alarm_zones_cnn"),
                ns=len(picked_by.get(name, [])),
                nb=info.get("n_zones_base"),
                iou=float(info.get("iou_cnn_vs_base") or 0),
                tz="да" if info.get("tz_missing") else "нет",
            )
        )
    lines.append("")
    lines.append(
        f"Всего зон {n_zones}, тревог {n_alarm}. "
        f"В экспертный разбор ниже взято {len(picked)} из них. Остальные тревоги остаются в csv."
    )
    lines.append("")
    for name in names:
        info = files[name]
        classes = info.get("classes") or {}
        bits = ", ".join(f"{cls} {classes[cls]}" for cls in sorted(classes, key=lambda k: (-classes[k], k))) or "нет"
        tz = ""
        if info.get("tz_missing"):
            tz = " Время в файле без пояса и прочитано как UTC: рамка карточки в UTC может съехать."
        iou = float(info.get("iou_cnn_vs_base") or 0)
        n_base = int(info.get("n_zones_base") or 0)
        n_alarm_file = int(info.get("n_alarm_zones_cnn") or 0)
        mask_note = ""
        if n_base >= 20 and n_alarm_file > 0 and iou < 0.02 and n_base > 3 * n_alarm_file:
            mask_note = (
                " Классика ставит много зон при почти нулевом IoU: median_x3+spatial_z на этой записи берёт шум, "
                "нейросеть при FAR 1e-3 его в тревогу не поднимает."
            )
        elif n_alarm_file >= 20 and n_base > 0 and iou < 0.02 and n_alarm_file > 3 * n_base:
            mask_note = (
                " Нейросеть ставит гораздо больше зон, чем классика, и маски почти не совпадают: "
                "при FAR 1e-3 здесь остаются слабые пятна, которые spatial_z не держит."
            )
        elif iou >= 0.3:
            mask_note = f" Маски нейросети и классики здесь ближе, чем на остальных файлах (IoU {iou:.2f})."
        lines += [
            f"### Файл {name}",
            "",
            f"Длительность {info.get('hours')} ч, длина {info.get('km')} км, "
            f"исходная матрица {info.get('shape')}, сетка модели {info.get('grid_shape')}. "
            f"Зон по типам: {bits}. Доля пикселей тревоги {float(info.get('alarm_fraction_cnn') or 0):.4f}, "
            f"у классики {float(info.get('alarm_fraction_base') or 0):.4f}. "
            f"Мёртвых тревог вне маски {info.get('dead_alarms')}.{tz}{mask_note} "
            f"В разбор: {', '.join(f'{name}-{row['id']:04d}' for row in picked_by.get(name, [])) or '—'}.",
            "",
            f"![Файл {name}: ΔT и зоны](figures/real_{prefix}_{name}.png)",
            "",
            f"Контуры и фронт: [real_{prefix}_cv_{name}.png](figures/real_{prefix}_cv_{name}.png).",
            "",
        ]

    lines += [
        "## Значимые зоны",
        "",
        "Значимая зона здесь — не каждая тревога модели, а короткий список для выезда. "
        f"Из {n_alarm} тревог оставлены {len(picked)} (не меньше {MIN_N} и не больше {MAX_N}, если тревог достаточно). "
        "Счёт идёт по амплитуде |ΔT|, длительности и вероятности класса. "
        "Рядом стоящие рамки одного файла и одной полярности ближе 4 м сливаются в один очаг: в список берётся более сильный. "
        "Дальше список добирается так, чтобы были протяжённый нагрев, устойчивое охлаждение, "
        "объект уже разрешения ~1 м, начало линии, одиночная временная ячейка и хотя бы одна зона с каждого файла, где есть тревога. "
        "С одного файла не больше четырёх. Полный перечень тревог — в csv этого прогона, порог по короткому списку не двигался.",
        "",
        "Пик в таблице подписан знаком полярности. В json поле `peak_dT_C` хранит max |ΔT| без знака. "
        "Время — UTC. P(класс) — вероятность назначенного типа.",
        "",
        "Как устроен измеритель. Это Рамановский DTS: температура волокна получается из отношения антистоксова "
        "и стоксова рассеяния. Пространственное разрешение прибора около 1 м; шаг отсчёта в CSV (0.1–1 м) и клетка сетки модели "
        f"(0.25 м) бывают мельче и не создают нового разрешения. По времени модель видит ячейки по {GRID_DT_S:.0f} с "
        "(в сырых файлах шаг 10–240 с): каждая точка уже усреднение. К дальнему концу линии антистокс слабеет, шум растёт. "
        "Склейка, коннектор и изгиб портят поправку потерь и выглядят как ступенька температуры.",
        "",
    ]
    if not picked:
        lines.append("Тревожных зон в этом прогоне нет.")
        lines.append("")
    else:
        lines.append("| зона | файл | x, м | начало UTC | конец UTC | длительность | пик ΔT, °C | тип | P | зачем в списке |")
        lines.append("|---|---|---|---|---|---:|---:|---|---:|---|")
        for row in picked:
            zone = row["zone"]
            zid = f"{row['file']}-{row['id']:04d}"
            dur = "1 ячейка" if row["single_bin"] else fmt_dur(row["dur_s"])
            lines.append(
                "| {zid} | {file} | {x0}–{x1} | {t0} | {t1} | {dur} | {peak:+.2f} | {cls} | {prob:.2f} | {why} |".format(
                    zid=zid,
                    file=row["file"],
                    x0=fmt_m(row["x0"]),
                    x1=fmt_m(row["x1"]),
                    t0=fmt_utc(zone["t_start_s"]),
                    t1=fmt_utc(zone["t_end_s"]),
                    dur=dur,
                    peak=signed_peak(zone),
                    cls=md_cell(row["cls"]),
                    prob=row["prob"],
                    why=md_cell(row["reason"]),
                )
            )
        lines.append("")

    lines += [
        "## Физическая интерпретация",
        "",
        "Ниже по каждой зоне из короткого списка: причина, которая согласуется с шириной, длительностью, знаком и амплитудой "
        "(муфта или кабельный ввод, климат и вентиляция, солнце и окна, контакт с трубой, "
        "приборный провал, шум концов линии, склейка), уровень уверенности и проверка на месте. "
        "Как устроен измеритель, сказано выше. Уверенность суждения — не вероятность класса: "
        "низкая P при долгой узкой рамке всё ещё может быть муфтой, а высокая P у одной ячейки усреднения наряд не оправдывает. "
        "Это не эталон и порог не двигает.",
        "",
    ]
    current = None
    for row in picked:
        if row["file"] != current:
            current = row["file"]
            lines.append(f"### Файл {current}")
            lines.append("")
            lead = file_lead(picked_by.get(current, []))
            if lead:
                lines.append(lead)
                lines.append("")
        zid = f"{row['file']}-{row['id']:04d}"
        cause, conf, check, tag = interpret(row["zone"], row["info"], picked_by.get(row["file"], []))
        lines.append(f"**{zid}** ({tag}). {cause}")
        lines.append("")
        lines.append(f"Уверенность суждения: {conf}. На месте: {check}")
        lines.append("")
    if not picked:
        lines.append("Тревог нет, интерпретировать нечего.")
        lines.append("")

    lines += [
        "## Согласие с карточками оператора",
        "",
        "Только описание. Попадание — хотя бы одна тревожная зона этого прогона (любая, не только короткий список) пересекает рамку карточки по метрам и по времени. "
        "Порог из-за этих рамок не менялся. Ответ оператора — быстрое мнение по графику, не эталон.",
        "",
        "### Партии 1 и 2",
        "",
        "Файлы `data/operator_cards/cards_batch1.jsonl` и `cards_batch2.jsonl`. "
        "Счёт ниже снят с поля `cards` json этого прогона (тот же проход, что и зоны).",
        "",
    ]
    cards = [c for c in (res.get("cards") or []) if c.get("status") != "box_outside_record"]
    outside = [c for c in (res.get("cards") or []) if c.get("status") == "box_outside_record"]

    def batch_hit(presence):
        subset = [c for c in cards if c.get("presence") == presence]
        hit = sum(1 for c in subset if c.get("cnn_n_alarm_zones"))
        return len(subset), hit

    lines.append("| ответ оператора | рамок | рамок с тревогой модели |")
    lines.append("|---|---:|---:|")
    for presence in ("yes", "no", "unclear"):
        n, hit = batch_hit(presence)
        lines.append(f"| {presence} | {n} | {hit} |")
    lines.append("")
    reading = batch_reading(res.get("cards") or [])
    if reading:
        lines.append(reading)
        lines.append("")
    if outside:
        lines.append(f"Рамок вне записи: {len(outside)} (в счёт не входят).")
        lines.append("")
    lines.append("| id | файл | ответ | тип у оператора | тревог в рамке | главный тип модели |")
    lines.append("|---|---|---|---|---:|---|")
    for card in sorted(res.get("cards") or [], key=lambda c: (file_key(str(c.get("record"))), str(c.get("id")))):
        lines.append(
            "| {id} | {rec} | {pres} | {shape} | {n} | {dom} |".format(
                id=md_cell(card.get("id")),
                rec=md_cell(card.get("record")),
                pres=md_cell(card.get("status") or card.get("presence") or ""),
                shape=md_cell(card.get("shape") or "—"),
                n=card.get("cnn_n_alarm_zones") if card.get("status") != "box_outside_record" else "—",
                dom=md_cell(card.get("cnn_dominant_class") or "—"),
            )
        )
    lines += [
        "",
        "### CARDS_REVIEW_2",
        "",
        "150 рамок из jsonl разбора `docs/CARDS_REVIEW_2.md` (тот же файл, что читает `run_real` для mae/static). "
        "Колонка `winner` в `artifacts/eval/cards2_review.csv` — прошлый описательный подсчёт выпуска. "
        "Этот прогон её не подгоняет и порог по ней не двигает.",
        "",
    ]
    if not review_rows:
        lines.append("Jsonl карточек партии 2 на диске нет, свой пересчёт не сделан. Опубликованная таблица в `docs/CARDS_REVIEW_2.md` остаётся прошлым описанием (у выпуска там 14/91 yes, 3/19 no, 17/40 unclear).")
        lines.append("")
    else:
        lines.append("| ответ | рамок в этом прогоне | с тревогой | колонка winner (прошлый подсчёт) |")
        lines.append("|---|---:|---:|---|")
        same = True
        for presence in ("yes", "no", "unclear"):
            n, hit = presence_counts(review_rows, presence)
            pn, ph = published_winner_counts(review_table, presence) if review_table else (0, 0)
            pub = f"{ph}/{pn}" if review_table else "—"
            if review_table and (n != pn or hit != ph):
                same = False
            lines.append(f"| {presence} | {n} | {hit} | {pub} |")
        lines.append("")
        if review_table:
            if same:
                lines.append("Счёт этого прогона совпал с колонкой `winner`. Порог для этого не подбирался: так и должно быть, если детектор и зоны те же.")
            else:
                lines.append("Счёт этого прогона отличается от колонки `winner`. В отчёт записан свежий пересчёт. Порог из-за расхождения не двигался.")
            lines.append("")
        # типы среди yes
        lines.append("Среди ответов yes, если тип указан (рамка может нести несколько типов):")
        lines.append("")
        lines.append("| тип в карточке | рамок | с тревогой |")
        lines.append("|---|---:|---:|")
        yes_rows = [r for r in review_rows if r.get("presence") == "yes" and r.get("status") == "ok"]
        for token in ("Local_Heating", "Local_Cooling", "Periodic_Thermal", "Broad_Shift"):
            subset = [r for r in yes_rows if token in (r.get("shape") or "")]
            hit = sum(1 for r in subset if r.get("hit"))
            lines.append(f"| {token} | {len(subset)} | {hit} |")
        lines.append("")
        by_id = {r["id"]: r for r in review_rows}
        narrow = narrow_static(review_table, by_id) if review_table else []
        if narrow:
            nh = sum(1 for r in narrow if r["hit"])
            lines.append(
                f"Узкий срез прошлого разбора (yes, колонка winner = 0, ширина ≤ 12 м, dT_hi ≥ 3 °C): "
                f"тревога этого прогона попала в {nh} из {len(narrow)}. "
                "Определение взято из `docs/CARDS_REVIEW_2.md`. Порог по этому списку не выбирался."
            )
            lines.append("")
            lines.append("| id | файл | ширина, м | dT_hi, °C | тревога сейчас |")
            lines.append("|---|---|---:|---:|---|")
            for row in sorted(narrow, key=lambda r: (file_key(str(r["record"])), r["id"])):
                lines.append(
                    f"| {md_cell(row['id'])} | {md_cell(row['record'])} | {row['width_m']:.1f} | {row['dT_hi']:.2f} | {'да' if row['hit'] else 'нет'} |"
                )
            lines.append("")
        by_shape = {}
        for token in ("Local_Heating", "Local_Cooling", "Periodic_Thermal", "Broad_Shift"):
            subset = [r for r in yes_rows if token in (r.get("shape") or "")]
            by_shape[token] = (len(subset), sum(1 for r in subset if r.get("hit")))
        heat_n, heat_h = by_shape["Local_Heating"]
        cool_n, cool_h = by_shape["Local_Cooling"]
        lines.append(
            "Смысл таблиц тот же, что в `docs/CARDS_REVIEW_2.md`, и это описание, не новая разметка. "
            f"Local_Cooling {cool_h} из {cool_n}: слабое охлаждение около 1–2.5 °C сидит на уровне шума при FAR 1e-3 "
            "и часто уходит в статический профиль. "
            f"Local_Heating {heat_h} из {heat_n} виден лучше, но узкий долгий горб "
            f"(таблица узкого среза: {nh if narrow else 0} из {len(narrow)}) вычитается вместе с профилем: "
            "оператор отмечает его на сырой температуре, в ΔT его уже нет. "
            "Periodic_Thermal и Broad_Shift попадают в тревогу только там, где пятно не снято синфазной составляющей. "
            "Опубликованный разбор называет те же места: запись 1 около 417, 425, 491, 803, 812, 880–894 м "
            "и узкие точки записи 5. Порог по этим долям не подбирался."
        )
        lines.append("")

    if compare:
        lines += comparison_lines(compare)
    lines += [
        "## Ограничения",
        "",
        "- Рамановский DTS меряет температуру волокна по отношению антистокс/стокс, а не воздух в комнате. Оболочка кабеля отстаёт от воздуха, а склейка и изгиб меняют потери и рисуют ложную ступеньку.",
        "- Оверлей классического детектора (белые рамки) показан только как ориентир сравнения и в решениях модели не используется.",
        f"- Пространственное разрешение около {SPATIAL_RES_M:.0f} м. Шаг CSV 0.1–1 м и клетка модели 0.25 м мельче разрешения: рамка уже 1 м — это одна ячейка прибора (склейка, коннектор, шум), а не измеренная длина объекта.",
        f"- Время усреднения. Сетка модели — {GRID_DT_S:.0f} с, сырой шаг файлов 10–240 с. Зона с совпавшими метками начала и конца — одно усреднение. Такое пятно не подтверждено во времени.",
        "- Концы линии. У прибора (первые метры) обычны пигтейл, коннектор и ступенька калибровки. К дальнему концу антистокс слабеет, шум растёт. Короткое пятно там прежде всего шум конца, а не дефект трассы.",
        "- Слабые события с |ΔT| примерно до 2 °C при пороге FAR 1e-3 почти не видны. Порог ставится так, чтобы на фоне синтетики ложных пикселей было около одной тысячной.",
        "- Узкие статичные точки (несколько метров, много часов: муфта, которая греет постоянно) вычитаются вместе со статическим профилем. Детектор видит появление и исчезновение. Постоянный горб в ΔT часто уже пустой.",
        "- Доменный разрыв synth→real: модель училась на синтетике. Шум, профиль и климат трассы МГТС другие. Тип зоны — подсказка формы, не диагноз с объекта.",
        "- Широкий сдвиг, общий для длинного куска кабеля, частично снимается вычитанием синфазной составляющей. Рамка может быть короче пятна на сырой температуре.",
        "- Карточки оператора порог не двигали и в обучение не входили. Ответ «yes» на всю рамку и пиксели модели внутри рамки — разные вопросы.",
        "- Глубокие провалы около −15…−25 °C для воздуха в помещении неправдоподобны. Пока осмотр не покажет иное, это склейка, коннектор, изгиб или участок вне помещения.",
        "- Если у файла нет пояса, наивное время читается как UTC. Сдвиг часов относительно карточки возможен и помечен у файла.",
        "- Реальных меток семи классов на этих CSV нет, и этот отчёт их не назначает. Короткий список зон — не журнал дефектов.",
        "",
        "## Как воспроизвести",
        "",
        "Один скрипт. Два потока, без дообучения. Каталог рисунков прогона — временный, опубликованные `docs/figures/real_1.png` … `real_6.png` и `real_cv_*.png` он не затирает.",
        "",
        "Финал текущего выпуска, static выключен (значение по умолчанию):",
        "",
        "```bash",
        "cd /workspace/dts_solution",
        "OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \\",
        "  /workspace/repo/run/.venv/bin/python scripts/make_expert_report.py \\",
        "  --release ens_u2ft_tcn_lgbm --static off --out-prefix final",
        "```",
        "",
        "Финал со StaticNet (`--static on`). Только вместе с `ens_u2ft_tcn_lgbm`: "
        "порог ветви заморожен на синтетике, карточки его не двигают.",
        "",
        "```bash",
        "/workspace/repo/run/.venv/bin/python scripts/make_expert_report.py \\",
        "  --release ens_u2ft_tcn_lgbm --static on --out-prefix final",
        "```",
        "",
        "Финал смеси с MAE (`ens_mean4`, флаг `--mae on`). Со `--static on` скрипт останавливается: "
        "эти две ветви вместе не калибровались.",
        "",
        "```bash",
        "/workspace/repo/run/.venv/bin/python scripts/make_expert_report.py \\",
        "  --release ens_mean4 --out-prefix final",
        "```",
        "",
        "Финал смеси GEN2 (`ens_gen2`, флаг `--gen2 on`, детектор победителя, зоны `cc`). "
        "Со `--static on` скрипт останавливается: ветви вместе не калибровались. "
        "Порог берётся из `models_release_gen2/stage1_thresholds.json`.",
        "",
        "```bash",
        "/workspace/repo/run/.venv/bin/python scripts/make_expert_report.py \\",
        "  --release ens_gen2 --out-prefix final",
        "```",
        "",
        "Голый прогон без префикса отчёта пишет `artifacts/eval/real_gen2.json`, "
        "`artifacts/eval/real_gen2.csv` и `artifacts/eval/real_gen2_fig`. "
        "Сравнение в этом отчёте читает эти файлы, когда текущий релиз сам не ens_gen2. "
        "Прогон с `--out-prefix` пишет `artifacts/eval/real_<prefix>.json` и рисунки "
        "`docs/figures/real_<prefix>_*.png`, а `real_v8.json` и `real_gen2.json` не затирает.",
        "",
        "Пересобрать текст из уже посчитанного json, без нового инференса:",
        "",
        "```bash",
        "/workspace/repo/run/.venv/bin/python scripts/make_expert_report.py \\",
        f"  --release {release} --out-prefix {prefix} --report-only",
        "```",
        "",
        "Если `--release` не передан, скрипт берёт последнее известное имя из строки `RELEASE_DECISION` "
        f"в `/workspace/run_logs/_events.log`. Пока такой строки нет, остаётся `ens_u2ft_tcn_lgbm`. "
        f"Неизвестное имя — останов, а не тихий прогон победителя. Сейчас известны: {', '.join(RELEASES)}. "
        "`--static` по умолчанию `off`.",
        "",
        "Выходы:",
        "",
        f"- `artifacts/eval/real_{prefix}.json`",
        f"- `artifacts/eval/real_{prefix}.csv`",
        f"- `docs/figures/real_{prefix}_<файл>.png` и `docs/figures/real_{prefix}_cv_<файл>.png`",
        "- `docs/REAL_EXPERT_REPORT.md` (этот файл; префикс `draft` помечает его как черновик, `final` — как итог)",
        "",
        f"Внутренний вызов — `scripts/run_real.py` с флагами из раздела «Релиз и флаги», плюс `--cv-annot` и обе партии карточек. "
        + (
            "Инференс заново не измерялся: текст собран из json и рисунков этого префикса."
            if reused
            else f"Последний измеренный прогон: {minutes:.1f} мин."
        ),
        "",
    ]
    text = "\n".join(lines)
    if prefix == "draft":
        head = "\n".join(text.splitlines()[:12])
        if "ЧЕРНОВИК" not in head:
            raise SystemExit("черновик потерял пометку ЧЕРНОВИК")
    else:
        head = "\n".join(text.splitlines()[:12])
        if "ЧЕРНОВИК" in head:
            raise SystemExit("итоговый отчёт помечен как черновик")
    for heading in HEADINGS:
        if heading not in text:
            raise SystemExit(f"нет раздела {heading}")
    return text


def verify(prefix, owned_before, res) -> None:
    now = snapshot(OWNED)
    for rel, before in owned_before.items():
        if now.get(rel) != before:
            raise SystemExit(f"занятый файл изменился: {rel}")
    json_path = ROOT / "artifacts/eval" / f"real_{prefix}.json"
    csv_path = ROOT / "artifacts/eval" / f"real_{prefix}.csv"
    if not json_path.is_file() or not csv_path.is_file() or not REPORT.is_file():
        raise SystemExit("нет json, csv или отчёта")
    names = list((res.get("files") or {}).keys())
    if len(names) < 1:
        raise SystemExit("в json нет файлов")
    for name in names:
        for kind in ("main", "cv"):
            path = dest_png(prefix, name, kind)
            if not path.is_file():
                raise SystemExit(f"нет {path}")
    zone_rows_ok(csv_path, res["files"])
    report = REPORT.read_text(encoding="utf-8")
    for heading in HEADINGS:
        if heading not in report:
            raise SystemExit(f"в отчёте нет {heading}")
    if prefix == "draft" and "ЧЕРНОВИК" not in "\n".join(report.splitlines()[:12]):
        raise SystemExit("отчёт на диске без пометки ЧЕРНОВИК")


def self_check() -> None:
    zone = {
        "id": 3,
        "start_m": 417.2,
        "end_m": 421.0,
        "t_start_s": 1_700_000_000,
        "t_end_s": 1_700_000_000 + 5 * 3600,
        "polarity": "hot",
        "class": "Local_Heating",
        "class_prob": {"Local_Heating": 0.62},
        "alarm": True,
        "peak_dT_C": 6.4,
        "n_px": 40,
        "score": 0.9,
    }
    weak = dict(zone, id=4, peak_dT_C=1.2, class_prob={"Local_Heating": 0.4})
    cold = dict(zone, id=5, polarity="cold", peak_dT_C=-18.0,
                class_prob={"Local_Cooling": 0.5}, end_m=419.0)
    cold["class"] = "Local_Cooling"
    cause, conf, check, _tag = interpret(zone)
    if "муфт" not in cause.lower() and "муфт" not in check.lower():
        raise SystemExit(f"интерпретация узкого нагрева пустая: {cause}")
    if interpret(weak)[1] != "низкая":
        raise SystemExit("слабый пик должен давать низкую уверенность")
    if "склейк" not in interpret(cold)[0]:
        raise SystemExit("глубокий провал должен указывать на склейку: " + interpret(cold)[0])
    mean4 = build_argv("ens_mean4", "off")
    if mean4[mean4.index("--mae") + 1] != "on" or mean4[mean4.index("--static") + 1] != "off":
        raise SystemExit(f"ens_mean4 собран неверно: {mean4}")
    winner_on = build_argv("ens_u2ft_tcn_lgbm", "on")
    if winner_on[winner_on.index("--static") + 1] != "on":
        raise SystemExit("static=on не попал в флаги победителя")
    try:
        build_argv("ens_mean4", "on")
    except SystemExit as exc:
        if "static" not in str(exc.code):
            raise
    else:
        raise SystemExit("ens_mean4 со static=on не отвергнут")
    gen2_argv = build_argv("ens_gen2", "off")
    if gen2_argv[gen2_argv.index("--gen2") + 1] != "on":
        raise SystemExit(f"ens_gen2 без --gen2 on: {gen2_argv}")
    if gen2_argv[gen2_argv.index("--static") + 1] != "off":
        raise SystemExit("ens_gen2 собрался со static не off")
    if gen2_argv[gen2_argv.index("--detector") + 1] != "winner":
        raise SystemExit("ens_gen2 собрался не с detector winner")
    try:
        build_argv("ens_gen2", "on")
    except SystemExit as exc:
        if "static" not in str(exc.code):
            raise
    else:
        raise SystemExit("ens_gen2 со static=on не отвергнут")
    decision = "02:40 RELEASE_DECISION ens_gen2, смесь 0.5*(ens_u2ft_tcn_lgbm + unet)"
    if release_in_decision_line(decision) != "ens_gen2":
        raise SystemExit(f"RELEASE_DECISION не узнал ens_gen2: {release_in_decision_line(decision)}")
    if release_in_decision_line("02:40 RELEASE_DECISION ens_u2ft_tcn_lgbm") != "ens_u2ft_tcn_lgbm":
        raise SystemExit("RELEASE_DECISION не узнал победителя")
    raw_thr = _gen2_raw()["thresholds"]["cnn_unet"]["far_1e-3"]
    if "0.8579896092424856" not in RELEASES["ens_gen2"]["blurb"] and str(raw_thr) not in RELEASES["ens_gen2"]["blurb"]:
        raise SystemExit("описание ens_gen2 без порога из stage1_thresholds.json")
    if "−20" not in RELEASES["ens_gen2"]["blurb"] and "-20" not in RELEASES["ens_gen2"]["blurb"]:
        raise SystemExit("описание ens_gen2 без провалов −20°C")
    res = {
        "op": "far_1e-3",
        "threshold": 0.86,
        "calib": "global",
        "far": 0.001,
        "zones_mode": "cc",
        "static": "off",
        "mae": "off",
        "files": {
            "1": {
                "sha256": "abc",
                "tz_missing": False,
                "threshold": 0.86,
                "shape": [10, 20],
                "grid_shape": [8, 16],
                "hours": 1.5,
                "km": 0.4,
                "n_zones_cnn": 2,
                "n_alarm_zones_cnn": 2,
                "n_zones_base": 1,
                "alarm_fraction_cnn": 0.01,
                "alarm_fraction_base": 0.02,
                "iou_cnn_vs_base": 0.1,
                "dead_alarms": 0,
                "classes": {"Local_Heating": 2},
                "zones": [zone, weak],
            }
        },
        "cards": [
            {"id": "c1", "record": "1", "presence": "yes", "shape": "Local_Heating",
             "cnn_n_alarm_zones": 1, "cnn_dominant_class": "Local_Heating"},
            {"id": "c2", "record": "1", "presence": "no", "shape": "",
             "cnn_n_alarm_zones": 0, "cnn_dominant_class": None},
        ],
    }
    review_rows = [
        {"id": "ask-1", "record": "1", "presence": "yes", "shape": "Local_Heating", "status": "ok", "hit": True, "n_alarm": 1},
        {"id": "ask-2", "record": "1", "presence": "no", "shape": "", "status": "ok", "hit": False, "n_alarm": 0},
    ]
    text = render("ens_u2ft_tcn_lgbm", "draft", res, review_rows, {}, 90.0)
    if "1-0003" not in text or "ЧЕРНОВИК" not in text.splitlines()[2]:
        raise SystemExit("самопроверка отчёта не собрала зону")
    if "## Сравнение победителя и ens_gen2" in text:
        raise SystemExit("сравнение попало в отчёт без второго прогона")
    if "## Краткое заключение" not in text or "## Качество модели на синтетике" not in text:
        raise SystemExit("нет краткого заключения или качества на синтетике")
    if "в решениях модели не используется" not in text:
        raise SystemExit("в ограничениях нет пункта про белые рамки")
    if "релиз по умолчанию ens_u2ft_tcn_lgbm, решение в 04:15" not in text and "RELEASE_DECISION" not in text:
        raise SystemExit("заключение не говорит, откуда взят релиз")
    brief = []
    capture = False
    for line in text.splitlines():
        if line == "## Краткое заключение":
            capture = True
        elif capture and line.startswith("## "):
            break
        if capture:
            brief.append(line)
    if len(brief) > 25:
        raise SystemExit(f"краткое заключение {len(brief)} строк")
    if "3–5 °C" not in text or "синтетики на реальную" not in text:
        raise SystemExit("в заключении нет оценки надёжности")
    if (ROOT / "artifacts/eval/s3_gen2_v8.json").is_file():
        raw = json.loads((ROOT / "artifacts/eval/s3_gen2_v8.json").read_text(encoding="utf-8"))
        shown = f"{float(raw['val']['ens_u2ft_tcn_lgbm']['iou']):.4f}"
        if shown not in text:
            raise SystemExit(f"таблица синтетики не показала IoU победителя {shown}")
    other = {
        "op": "far_1e-3",
        "threshold": 0.8579896092424856,
        "calib": "global",
        "gen2": "on",
        "detector": "ens_blend",
        "files": {
            "1": {
                "threshold": 0.8579896092424856,
                "km": 0.4,
                "n_zones_cnn": 1,
                "n_alarm_zones_cnn": 1,
                "zones": [dict(zone, id=9, start_m=418.0, end_m=422.0)],
            }
        },
        "cards": [
            {"id": "c1", "record": "1", "presence": "yes", "cnn_n_alarm_zones": 1},
            {"id": "c2", "record": "1", "presence": "no", "cnn_n_alarm_zones": 1},
        ],
    }
    compared = render(
        "ens_gen2", "draft", other, review_rows, {}, 90.0,
        compare={
            "winner": res,
            "gen2": other,
            "winner_src": "artifacts/eval/real_v8.json",
            "gen2_src": "artifacts/eval/real_draft.json",
        },
    )
    if "## Сравнение победителя и ens_gen2" not in compared:
        raise SystemExit("сравнение двух прогонов не собралось")
    if "1-0003" not in compared or "1-0009" not in compared:
        raise SystemExit("сравнение потеряло значимые зоны")
    if "| yes |" not in compared or "c1" not in compared:
        raise SystemExit("сравнение потеряло карточки yes/no")
    final = render("ens_u2ft_tcn_lgbm", "final", res, review_rows, {}, 90.0)
    if "ЧЕРНОВИК" in "\n".join(final.splitlines()[:12]):
        raise SystemExit("итог помечен черновиком")
    try:
        dest_png("draft", "1", "main")
        dest_png("draft", "1", "cv")
    except SystemExit as exc:
        raise SystemExit(f"своё имя рисунка отвергнуто: {exc}")
    for bad_prefix in ("", "v8"):
        try:
            dest_png(bad_prefix, "1", "main")
        except SystemExit:
            pass
        else:
            raise SystemExit("пустой префикс не отвергнут")
    log("self-check ok")


def main():
    ap = argparse.ArgumentParser(description="Прогон run_real и экспертный отчёт по MGTS 1–6.csv")
    ap.add_argument("--release", default=None, help="имя релиза; пусто — последняя RELEASE_DECISION или ens_u2ft_tcn_lgbm")
    ap.add_argument("--out-prefix", choices=("draft", "final"), default=None)
    ap.add_argument("--static", choices=("off", "on"), default=None,
                    help="off по умолчанию; on только с ens_u2ft_tcn_lgbm; с ens_gen2 запрещён")
    ap.add_argument("--real-root", default="/workspace/mgts/MGTS_data")
    ap.add_argument("--report-only", action="store_true",
                    help="пересобрать markdown из уже посчитанного json, без инференса")
    ap.add_argument("--self-check", action="store_true", help="проверить текст отчёта на искусственной зоне и выйти")
    args = ap.parse_args()
    if args.self_check:
        self_check()
        return
    if args.out_prefix not in ("draft", "final"):
        raise SystemExit("нужен --out-prefix draft или final")
    release = resolve_release(args.release)
    prefix = args.out_prefix
    static = args.static or "off"
    if not args.report_only:
        build_argv(release, static)
    owned_before = snapshot(OWNED)
    json_path = ROOT / "artifacts/eval" / f"real_{prefix}.json"
    started = time.time()
    if args.report_only:
        if not json_path.is_file():
            raise SystemExit(f"нет {json_path}, нужен полный прогон без --report-only")
        res = json.loads(json_path.read_text(encoding="utf-8"))
        recorded = res.get("static") or "off"
        if args.static is not None and args.static != recorded:
            raise SystemExit(
                f"json этого прогона static={recorded}, запрошен --static {args.static}. "
                "Другой флаг — полный прогон, не --report-only."
            )
        review_rows, review_table = score_review2(res["files"])
        wall = time.time() - started
        report = render(
            release, prefix, res, review_rows, review_table, wall, reused=True,
            compare=comparison_pair(release, prefix, res),
        )
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(report, encoding="utf-8")
        verify(prefix, owned_before, res)
        n_shown = len(select_significant(res["files"]))
        n_files = len(res["files"])
        n_zones = sum(len(info.get("zones") or []) for info in res["files"].values())
        n_alarm = sum(1 for info in res["files"].values() for zone in info.get("zones") or [] if is_alarm(zone))
        if prefix != "draft":
            log(f"report-only prefix={prefix}: строку EXPERT_DONE не пишу, свежего прогона не было")
            return
        elog(
            f"EXPERT_DRAFT2_DONE release={release} prefix=draft static={recorded} "
            f"n_files={n_files} n_zones={n_zones} n_alarm={n_alarm} n_shown={n_shown} "
            f"sec={wall:.1f} json=artifacts/eval/real_{prefix}.json "
            f"report=docs/REAL_EXPERT_REPORT.md real_train=0"
        )
        return
    free = shutil.disk_usage(ROOT).free
    if free < 2 * 1024 ** 3:
        raise SystemExit(f"мало места: {free} байт свободно")
    tmp = Path(f"/tmp/expert_fig_{prefix}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        run_real(release, prefix, args.real_root, tmp, static)
        res = json.loads(json_path.read_text(encoding="utf-8"))
        names = sorted((res.get("files") or {}), key=file_key)
        if not names:
            raise SystemExit("run_real не записал ни одного файла")
        relocate(tmp, prefix, names)
        if release == "ens_gen2":
            if res.get("gen2") != "on" or res.get("detector") != "ens_blend":
                raise SystemExit(
                    f"ens_gen2 записал gen2={res.get('gen2')} detector={res.get('detector')}"
                )
            if res.get("static") != "off" or res.get("mae") != "off":
                raise SystemExit("ens_gen2 записал static или mae не off")
            expect = float(_gen2_raw()["thresholds"]["cnn_unet"]["far_1e-3"])
            if abs(float(res.get("threshold")) - expect) > 1e-9:
                raise SystemExit(f"порог прогона {res.get('threshold')} не равен порогу релиза {expect}")
        review_rows, review_table = score_review2(res["files"])
        wall = time.time() - started
        report = render(
            release, prefix, res, review_rows, review_table, wall,
            compare=comparison_pair(release, prefix, res),
        )
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(report, encoding="utf-8")
        verify(prefix, owned_before, res)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
    wall = time.time() - started
    n_files = len(res["files"])
    n_zones = sum(len(info.get("zones") or []) for info in res["files"].values())
    n_alarm = sum(1 for info in res["files"].values() for zone in info.get("zones") or [] if is_alarm(zone))
    n_shown = len(select_significant(res["files"]))
    token = "EXPERT_DRAFT_DONE" if prefix == "draft" else "EXPERT_DONE"
    elog(
        f"{token} release={release} prefix={prefix} static={static} n_files={n_files} n_zones={n_zones} "
        f"n_alarm={n_alarm} n_shown={n_shown} sec={wall:.1f} json=artifacts/eval/real_{prefix}.json "
        f"csv=artifacts/eval/real_{prefix}.csv fig=docs/figures/real_{prefix}_<file>.png "
        f"report=docs/REAL_EXPERT_REPORT.md real_train=0"
    )


if __name__ == "__main__":
    main()
