"""Build a portable local research pack exclusively from the six real CSVs.

Usage: python scripts/prepare_real_data.py
Outputs are ignored artifacts; small aggregate evidence is copied separately.
No training is performed and no expert labels are manufactured.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import html
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fiber.grid import segments, sample_widths
from fiber.io import Record, file_sha256
from fiber.real_data import (RULE_CONFIG, crop_partition, diagnose_record,
                             diversity_key, median, select_review,
                             temporal_partition, weak_evidence, quantiles)


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def plot_card(path, record, raw, clean, valid, x, t, bounds, core_rows, signals):
    # All rows are rendered. Coloured cells do not bridge temporal gaps.
    t_min = (t-t[0])/60
    masked = np.where(valid, raw, np.nan)
    residual = np.where(valid, raw-clean, np.nan)
    finite = masked[np.isfinite(masked)]
    limits = np.quantile(finite, [.01, .99]) if finite.size else [0, 1]
    if limits[0] == limits[1]:
        limits = [limits[0]-.1, limits[1]+.1]
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.2), layout="constrained")
    for ax, field, title, cmap, vlim in (
        (axes[0], masked, "Raw temperature (real measurements)", "inferno", limits),
        (axes[1], residual, "Raw minus median_x3 (filter change, not error)", "RdBu_r", [-1, 1]),
    ):
        dx = sample_widths(x)
        xe = np.r_[x-dx/2, x[-1]+dx[-1]/2]
        # Native axes are regular in each file; midpoint edges also handle short crops.
        xe[1:-1] = (x[:-1]+x[1:])/2
        for seg in segments(t):
            y = t_min[seg]
            dy = sample_widths(t[seg])/60 if seg.stop-seg.start > 1 else np.array([1/60])
            ye = np.r_[y-dy/2, y[-1]+dy[-1]/2]
            ye[1:-1] = (y[:-1]+y[1:])/2
            image = ax.pcolormesh(xe, ye, field[seg], cmap=cmap, vmin=vlim[0], vmax=vlim[1], rasterized=True)
        ax.axvline(bounds[0], color="#21abca", linewidth=1)
        ax.axvline(bounds[1], color="#21abca", linewidth=1)
        ax.axhline(t_min[core_rows[0]], color="#21abca", linewidth=1)
        ax.axhline(t_min[core_rows[1]-1], color="#21abca", linewidth=1)
        ax.set(title=title, xlabel="Distance, m", ylabel="Time from crop start, min")
        fig.colorbar(image, ax=ax, label="°C", fraction=.025, pad=.02)
    for name, color, label in (("core_c", "#b0471d", "Raw core median"),
                                ("reference_c", "#176c83", "Raw flank median")):
        for i, seg in enumerate(segments(t)):
            axes[2].plot(t_min[seg], signals[name][seg], color=color, label=label if i == 0 else None)
    core = (x >= bounds[0]) & (x <= bounds[1])
    clean_core = median(np.where(valid[:, core], clean[:, core], np.nan), axis=1)
    for i, seg in enumerate(segments(t)):
        axes[2].plot(t_min[seg], clean_core[seg], linestyle="--", color="#7459a8", linewidth=1,
                     label="Median_x3 core" if i == 0 else None)
    axes[2].set(xlabel="Time from crop start, min", ylabel="Temperature, °C")
    axes[2].legend(loc="best", fontsize=8)
    axes[2].grid(alpha=.2)
    fig.suptitle(f"{record} · observed crop · no expert label\nHeatmap colours clipped at raw 1% / 99%; crop NPZ preserves all values", fontsize=10)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def build_html(out, rows, shortlist):
    ids = set(shortlist)
    # Blind cards expose raw data and coverage, never rule votes or proposal hints.
    cards = []
    for row in rows:
        key = html.escape(row["id"])
        x0, x1 = row["x_m_inclusive"]
        cards.append(f'<article id="{key}"><h2>{key}</h2><p>Файл {row["record"]}; ядро {x0:.2f}–{x1:.2f} м; строки [{row["rows_half_open"][0]}, {row["rows_half_open"][1]}). Экспертная метка отсутствует.</p><img loading="lazy" src="cards/{key}.png" alt="Исходная температура, изменение фильтра и графики"><p><a href="patches/{key}.npz">Полные массивы NPZ</a></p></article>')
    style = '<style>body{font:16px/1.5 system-ui;max-width:1120px;margin:32px auto;padding:0 20px;color:#192e3a}img{width:100%;height:auto}article{border-top:1px solid #ccd5dc;margin-top:36px}a{color:#176c83}li{margin:8px 0}h2{font-size:20px}</style>'
    intro = '<h1>Реальные измерения: материал для проверки</h1><p>Синтетические сцены не используются. Это черновые наблюдения, не подтверждённые события или диагнозы. Голосование правил скрыто от первичного просмотра. Голубые границы обозначают ядро примера; окружение служит для сравнения.</p><p>Отсутствие локального контраста не исключает широкое температурное изменение. Цветовая шкала может скрывать экстремумы: массивы доступны полностью.</p><p><a href="review_template.jsonl">Шаблон ответов</a> · <a href="shortlist.html">Короткая подборка для разработки</a> · <a href="README.md">Инструкция</a></p>'
    prefix = '<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Реальные измерения</title>'+style+'<body>'
    (out/"index.html").write_text(prefix+intro+''.join(cards)+"</body></html>", encoding="utf-8")
    chosen = [card for row, card in zip(rows, cards, strict=True) if row["id"] in ids]
    (out/"shortlist.html").write_text(prefix+intro+'<p>Подборка вопросов разработки; не репрезентативный тест точности. Примеры внутреннего holdout сюда не включены.</p>'+''.join(chosen)+"</body></html>", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT/"artifacts/real-data-v2")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    out = args.out
    if (out/"manifest.json").exists():
        raise SystemExit("Use a new --out directory: completed packs are immutable.")
    for name in ("cards", "patches", "splits"):
        (out/name).mkdir(parents=True, exist_ok=True)
    proposals_path = ROOT/"artifacts/hardening/annotation/proposals.jsonl"
    proposals = [json.loads(line) for line in proposals_path.read_text(encoding="utf-8").splitlines()]
    rows, audits, sources = [], [], {}
    for name in map(str, range(1, 7)):
        rec = Record.load(ROOT/"artifacts/records"/name)
        # Validate the current source file, not just an inherited metadata assertion.
        if file_sha256(Path(rec.source_csv)) != rec.source_sha256:
            raise ValueError(f"CSV changed: {name}")
        clean_path = ROOT/"artifacts/denoise"/name/"median_x3.npy"
        clean = np.load(clean_path, mmap_mode="r", allow_pickle=False)
        if clean.shape != rec.raw_c.shape:
            raise ValueError("clean dimensions disagree")
        sources[name] = {"csv_sha256": rec.source_sha256, "raw_sha256": file_sha256(ROOT/"artifacts/records"/name/"raw_c.npy"),
                         "clean_sha256": file_sha256(clean_path), "quality_sha256": file_sha256(ROOT/"artifacts/records"/name/"quality_codes.npy")}
        audits.append({"record": name, "source_sha256": rec.source_sha256,
                       **diagnose_record(rec.raw_c, clean, rec.valid, rec.x_m, rec.time_s)})
        split, split_info = temporal_partition(rec.time_s, RULE_CONFIG["split_guard_s"])
        np.save(out/"splits"/f"{name}.npy", split)
        write_json(out/"splits"/f"{name}.json", split_info)
        native_segments = segments(rec.time_s)
        candidates = [r for r in proposals if r["record"] == name]
        # Stratified, detector-independent windows: two time bands x six spatial bands.
        # These are not negative examples. Endpoints use actual measurement coordinates.
        live_x = np.flatnonzero(rec.valid.any(axis=0))
        for ti, tf in enumerate((.25, .65)):
            a = min(rec.n_profiles-1, int(rec.n_profiles*tf))
            seg = next(s for s in native_segments if s.start <= a < s.stop)
            b = min(seg.stop, int(np.searchsorted(rec.time_s, rec.time_s[a]+1800, side="right")))
            for xi, xf in enumerate(np.linspace(.08, .92, 6)):
                anchor = live_x[min(len(live_x)-1, int(len(live_x)*xf))]
                candidates.append({"id": f"{name}-stratum-{ti}-{xi}", "record": name,
                    "source_sha256": rec.source_sha256, "origin": "stratified_review",
                    "rows_half_open": [a, b], "x_m_inclusive": [max(float(rec.x_m[0]), float(rec.x_m[anchor])-20),
                    min(float(rec.x_m[-1]), float(rec.x_m[anchor])+20)]})
        for candidate in candidates:
            a, b = candidate["rows_half_open"]
            left, right = candidate["x_m_inclusive"]
            if candidate["source_sha256"] != rec.source_sha256 or not (0 <= a < b <= rec.n_profiles):
                raise ValueError("invalid proposal source/rows")
            # Context padding is 180 seconds, bounded by the core's containing segment.
            first = next(s for s in native_segments if s.start <= a < s.stop)
            last = next(s for s in native_segments if s.start <= b-1 < s.stop)
            ca = max(first.start, int(np.searchsorted(rec.time_s, rec.time_s[a]-180)))
            cb = min(last.stop, int(np.searchsorted(rec.time_s, rec.time_s[b-1]+180, side="right")))
            xa = int(np.searchsorted(rec.x_m, left-40))
            xb = int(np.searchsorted(rec.x_m, right+40, side="right"))
            if xa == xb:
                raise ValueError("empty spatial crop")
            x, t = rec.x_m[xa:xb], rec.time_s[ca:cb]
            raw, valid, filtered = rec.raw_c[ca:cb, xa:xb], rec.valid[ca:cb, xa:xb], clean[ca:cb, xa:xb]
            evidence, core_signals = weak_evidence(raw[a-ca:b-ca], valid[a-ca:b-ca], x, rec.time_s[a:b], [left, right])
            clean_evidence, clean_signals = weak_evidence(filtered[a-ca:b-ca], valid[a-ca:b-ca], x, rec.time_s[a:b], [left, right])
            finite_contrast = np.isfinite(core_signals["contrast_c"]) & np.isfinite(clean_signals["contrast_c"])
            raw_contrast = core_signals["contrast_c"][finite_contrast]
            clean_contrast = clean_signals["contrast_c"][finite_contrast]
            evidence["filter_effect"] = {
                "paired_observed_rows": int(finite_contrast.sum()),
                "abs_contrast_change_c": quantiles(np.abs(raw_contrast-clean_contrast)),
                "raw_abs_peak_contrast_c": float(np.max(np.abs(raw_contrast))) if len(raw_contrast) else None,
                "clean_abs_peak_contrast_c": float(np.max(np.abs(clean_contrast))) if len(clean_contrast) else None,
                "raw_hot_runs": evidence["hot_runs_half_open"], "clean_hot_runs": clean_evidence["hot_runs_half_open"],
                "raw_cold_runs": evidence["cold_runs_half_open"], "clean_cold_runs": clean_evidence["cold_runs_half_open"],
                "interpretation": "changes in observable contrast and threshold runs, not error against expert event boundaries",
            }
            _, signals = weak_evidence(raw, valid, x, t, [left, right])
            # Presence disagreement uses detector selection only as a review question.
            if candidate["origin"] == "detector_proposal" and evidence["weak_presence_label"] == 0:
                evidence["review_reasons"].append("detector_proposal_vs_low_raw_local_contrast")
            if clean_evidence["weak_presence_label"] != evidence["weak_presence_label"]:
                evidence["review_reasons"].append("raw_vs_filtered_rule_presence_changes")
            blind_id = candidate.get("blind_id", f"stratum-{name}-{candidate['id'].split('stratum-')[-1]}")
            row = {"id": blind_id, "record": name, "source_sha256": rec.source_sha256,
                   "origin": candidate["origin"], "rows_half_open": [a, b], "x_m_inclusive": [left, right],
                   "context_rows_half_open": [ca, cb], "context_columns_half_open": [xa, xb],
                   "timestamp_bounds": [rec.time_iso[a], rec.time_iso[b-1]], "time_basis": "wall_clock_timezone_unknown" if rec.tz_missing else "utc_epoch",
                   "split_code": crop_partition(split[ca:cb]), "weak_evidence": evidence,
                   "expert_label": None, "expert_confirmed": False}
            path = out/"patches"/f"{blind_id}.npz"
            np.savez_compressed(path, raw_c=raw, median_x3_c=filtered, valid=valid,
                                quality_codes=rec.quality_codes[ca:cb, xa:xb], x_m=x, time_s=t,
                                source_rows=np.arange(ca, cb), source_columns=np.arange(xa, xb),
                                split_code=split[ca:cb], core_rows_half_open=np.array([a-ca, b-ca]),
                                core_x_m_inclusive=np.array([left, right]), **signals)
            row["patch_sha256"] = file_sha256(path)
            row["patch_bytes"] = path.stat().st_size
            rows.append(row)
            if not args.no_plots:
                plot_card(out/"cards"/f"{blind_id}.png", blind_id, raw, filtered, valid, x, t, [left, right], [a-ca, b-ca], signals)
            if len(rows) % 20 == 0:
                print(f"Prepared {len(rows)} real crops", flush=True)
        print(f"Record {name}: diagnostics and {len(candidates)} crops", flush=True)
    shortlist = select_review(rows)
    write_json(out/"noise_audit.json", {"schema": "real_noise_diagnostics_v1", "records": audits,
        "clean_ground_truth_available": False, "noise_independence_verified": False,
        "all_measurements_seen_during_development": True})
    (out/"weak_labels.jsonl").write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n' for r in rows), encoding="utf-8")
    # Independent answers stay empty. The browser view never includes rule suggestions.
    template = [{"id": r["id"], "source_sha256": r["source_sha256"], "patch_sha256": r["patch_sha256"],
                 "presence": None, "shape": None, "reviewer": None, "reviewer_type": None,
                 "notes": "", "expert_confirmed": False} for r in rows]
    (out/"review_template.jsonl").write_text(''.join(json.dumps(r)+'\n' for r in template), encoding="utf-8")
    write_json(out/"shortlist.json", {"ids": shortlist, "purpose": "development questions, not accuracy estimation", "per_record_max": 4})
    groups = Counter(str(diversity_key(r)) for r in rows)
    manifest = {"schema": "real_data_pack_v1", "synthetic_training_used": False, "models_trained": 0,
                "expert_labels_created": 0, "n_crops": len(rows), "n_shortlist": len(shortlist),
                "sources": sources, "rule_config": RULE_CONFIG,
                "origin_counts": dict(Counter(r["origin"] for r in rows)),
                "weak_presence_counts": dict(Counter(str(r["weak_evidence"]["weak_presence_label"]) for r in rows)),
                "split_counts": dict(Counter(str(r["split_code"]) for r in rows)),
                "n_diversity_groups": len(groups), "patch_bytes": sum(r["patch_bytes"] for r in rows),
                "expert_labels_are_not_weak_labels": True, "independent_unseen_test": False,
                "plots_rendered": not args.no_plots, "proposal_queue_sha256": file_sha256(proposals_path),
                "code_sha256": {p: file_sha256(ROOT/p) for p in ("fiber/real_data.py", "scripts/prepare_real_data.py")}}
    (out/"README.md").write_text("""# Реальные данные для Захара

Откройте `shortlist.html`: до четырёх вопросов разработки на файл. В полном
`index.html` — все предложения, случайные и стратифицированные участки.
Изображения и ссылки работают локально, интернет и сервер не требуются.
Скопировать можно весь каталог; абсолютных путей в пакете нет.

Для каждого примера в `patches/ID.npz`: исходное измерение, median_x3,
валидность, причины качества, координаты, время, исходные номера строк и
колонок, границы ядра, временной split и графики core/reference. Открытие:
`a = numpy.load(path, allow_pickle=False)`.

`review_template.jsonl` — пустой шаблон независимых ответов. Сначала смотрите
измерения, потом `weak_labels.jsonl`: там только предварительные правила,
их отказы и конфликты. Не копируйте их в expert_label.

Оцените присутствие различимого локального изменения: yes/no/uncertain.
Если можете, добавьте форму, поправку границ и краткое объяснение. Причина
и неисправность оборудования не следуют из температурного графика.
Отсутствие контраста с окружением не исключает общего или широкого нагрева.

`noise_audit.json` измеряет изменчивость и корреляции; истинный шум и чистый
сигнал неизвестны. Raw-minus-median — изменение фильтром, не ошибка очистки.

Split-коды: -1 исключён/пересекает границу; 0 train; 1 development;
2 internal holdout. Guard по времени 1800 с. Карточки с пересечением split
не должны идти в supervised обучение. Все шесть CSV уже исследовались:
internal holdout не является независимым испытанием на новых измерениях.
Просмотр holdout и настройка по нему дополнительно лишают его тестового статуса.

Модель здесь не обучается. Синтетические сцены и веса не используются.
Порог контраста 1°C, coverage 80% и параметры повторений — предположения
для разработки. Предварительные метки не доказывают промышленную точность.
""", encoding="utf-8")
    if not args.no_plots:
        build_html(out, rows, shortlist)
    write_json(out/"manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
