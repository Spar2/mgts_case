"""Durable local workspace and verified adapter for the real-data research pack."""

import copy
import html
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .profile_changes import changes_html
from .vendor.data import sha256


class Workspace:
    def __init__(self, path):
        self.path = str(path)
        with self.connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS documents (kind TEXT, id TEXT, payload TEXT, PRIMARY KEY(kind,id))"
            )
        for job in self.list("run"):
            if job["status"] in ("queued", "running", "cancel_requested"):
                self.put(
                    "run",
                    job["id"],
                    {
                        **job,
                        "status": "interrupted",
                        "error": "Server restarted before completion",
                    },
                )

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        try:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=FULL")
            with db:
                yield db
        finally:
            db.close()

    def put(self, kind, key, value):
        value = {**value, "updated_at": time.time()}
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO documents VALUES (?,?,?)",
                (kind, key, json.dumps(value, allow_nan=False)),
            )
            if kind == "run":
                total = (
                    db.execute(
                        "SELECT SUM(length(payload)) FROM documents WHERE kind='run'"
                    ).fetchone()[0]
                    or 0
                )
                if total > 32 * 1024 * 1024:
                    for old_key, payload in db.execute(
                        "SELECT id,payload FROM documents WHERE kind='run' AND id<>? ORDER BY rowid",
                        (key,),
                    ).fetchall():
                        old = json.loads(payload)
                        if "result" not in old:
                            continue
                        old.pop("result")
                        old.update(
                            result_retention="expired above 32 MiB workspace budget",
                            result_bytes=0,
                        )
                        reduced = json.dumps(old, allow_nan=False)
                        db.execute(
                            "UPDATE documents SET payload=? WHERE kind='run' AND id=?",
                            (reduced, old_key),
                        )
                        total -= len(payload) - len(reduced)
                        if total <= 32 * 1024 * 1024:
                            break
        return value

    def list(self, kind):
        with self.connect() as db:
            return [
                json.loads(p)
                for (p,) in db.execute(
                    "SELECT payload FROM documents WHERE kind=? ORDER BY rowid DESC LIMIT 500",
                    (kind,),
                )
            ]

    def get(self, kind, key):
        with self.connect() as db:
            row = db.execute(
                "SELECT payload FROM documents WHERE kind=? AND id=?", (kind, key)
            ).fetchone()
        if row is None:
            raise KeyError("Unknown workspace item")
        return json.loads(row[0])

    def delete(self, kind, key):
        with self.connect() as db:
            db.execute("DELETE FROM documents WHERE kind=? AND id=?", (kind, key))


def adapt_real_pack(root, store, target, answers):
    """Compare all pack cells to native observations; never import weak answers."""
    root, target = Path(root), Path(target)
    receipt_path = target / "verification.json"
    fingerprint = {
        "adapter_version": 2,
        "manifest": sha256(root / "manifest.json"),
        "proposals": sha256(root / "weak_labels.jsonl"),
        "shortlist": sha256(root / "shortlist.json"),
        "sources": {key: meta["source_sha256"] for key, meta in store.meta.items()},
        "patches": {
            p.name: sha256(p) for p in sorted((root / "patches").glob("*.npz"))
        },
    }
    if receipt_path.exists() and (target / "manifest.json").exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("fingerprint") == fingerprint and receipt.get(
            "queue_sha256"
        ) == sha256(target / "queue.json"):
            return receipt["info"]
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "real_data_pack_v1":
        raise ValueError("Unsupported research pack schema")
    evidence = [
        json.loads(line)
        for line in (root / "weak_labels.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    shortlist = json.loads((root / "shortlist.json").read_text(encoding="utf-8"))["ids"]
    if len(evidence) != manifest["n_crops"] or len({c["id"] for c in evidence}) != len(
        evidence
    ):
        raise ValueError("Research pack cardinality mismatch")
    if len(set(shortlist)) != len(shortlist) or not set(shortlist).issubset(
        {c["id"] for c in evidence}
    ):
        raise ValueError("Invalid research shortlist")
    target.mkdir(parents=True, exist_ok=True)
    cards = []
    for item in evidence:
        key = item["id"]
        if Path(key).name != key:
            raise ValueError("Invalid research card ID")
        path = root / "patches" / (key + ".npz")
        if sha256(path) != item["patch_sha256"]:
            raise ValueError("Research patch digest mismatch")
        record = store.get(item["record"])
        if record.manifest["source_sha256"] != item["source_sha256"]:
            raise ValueError("Research source mismatch")
        with np.load(path, allow_pickle=False) as f:
            rows, columns = f["source_rows"], f["source_columns"]
            if not (
                np.issubdtype(rows.dtype, np.integer)
                and np.issubdtype(columns.dtype, np.integer)
            ):
                raise ValueError("Integer native indices required")
            if (
                len(rows) < 1
                or len(columns) < 2
                or rows[0] < 0
                or columns[0] < 0
                or rows[-1] >= record.raw_c.shape[0]
                or columns[-1] >= record.raw_c.shape[1]
                or not np.all(np.diff(rows) == 1)
                or not np.all(np.diff(columns) == 1)
            ):
                raise ValueError("Contiguous bounded native context required")
            raw = record.raw_c[np.ix_(rows, columns)]
            reasons = record.invalid_reason[np.ix_(rows, columns)]
            if (
                raw.size > 1_000_000
                or not np.array_equal(raw.astype("f4"), f["raw_c"], equal_nan=True)
                or not np.array_equal(reasons == 0, f["valid"])
                or (
                    "quality_codes" in f
                    and not np.array_equal(
                        np.where(reasons == 1, 2, np.where(reasons == 2, 1, reasons)),
                        f["quality_codes"],
                    )
                )
                or f["median_x3_c"].shape != raw.shape
                or not np.isfinite(f["median_x3_c"][reasons == 0]).all()
                or not np.array_equal(record.x_m[columns], f["x_m"])
                or not np.allclose(
                    np.diff(record.elapsed_s[rows]),
                    np.diff(f["time_s"]),
                    atol=1e-6,
                    rtol=0,
                )
            ):
                raise ValueError("Research pack differs from native measurements")
            # Preserve Float64 raw and native mask; pack Float32 is a comparison artifact.
            artifact = target / (key + ".npz")
            np.savez_compressed(
                artifact,
                raw_c=raw,
                invalid_reason=reasons,
                x_m=record.x_m[columns],
                time_s=record.elapsed_s[rows],
                signed=np.empty((0, *raw.shape), dtype="i1"),
                binary=np.empty((0, *raw.shape), dtype=bool),
                median_x3_c=f["median_x3_c"],
            )
            cards.append(
                {
                    "review_id": key,
                    "source": item["record"] + ".csv",
                    "artifact": artifact.name,
                    "artifact_sha256": sha256(artifact),
                    "evidence": {
                        "source_sha256": item["source_sha256"],
                        "research_pack_sha256": sha256(root / "manifest.json"),
                        "original_patch_sha256": item["patch_sha256"],
                    },
                    "native_shape": list(raw.shape),
                    "row_bounds": [int(rows[0]), int(rows[-1]) + 1],
                    "column_bounds": [int(columns[0]), int(columns[-1]) + 1],
                    "x_bounds_m": [
                        float(record.x_m[columns[0]]),
                        float(record.x_m[columns[-1]]),
                    ],
                    "time_bounds_s": [
                        float(record.elapsed_s[rows[0]]),
                        float(record.elapsed_s[rows[-1]]),
                    ],
                    "timestamps": [
                        record.profiles[int(rows[0])]["timestamp"],
                        record.profiles[int(rows[-1])]["timestamp"],
                    ],
                    "methods": [],
                    "shortlist": key in shortlist,
                    "proposal_bounds": {
                        "row_bounds": item.get("rows_half_open"),
                        "x_bounds_m": item.get("x_m_inclusive"),
                    },
                }
            )
    (target / "queue.json").write_text(
        json.dumps({"cards": cards, "allowed_answers": answers}, ensure_ascii=False),
        encoding="utf-8",
    )
    (target / "manifest.json").write_text(
        json.dumps(
            {
                "queue_sha256": sha256(target / "queue.json"),
                "cards": len(cards),
                "research_pack_sha256": sha256(root / "manifest.json"),
            }
        ),
        encoding="utf-8",
    )
    info = {
        "cards": len(cards),
        "shortlist": shortlist,
        "source_manifest_sha256": sha256(root / "manifest.json"),
        "verified_native_cells": sum(np.prod(c["native_shape"]).item() for c in cards),
    }
    receipt_path.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "info": info,
                "queue_sha256": sha256(target / "queue.json"),
            }
        ),
        encoding="utf-8",
    )
    return info


def report_html(evidence):
    """Standalone SVG heatmap and numeric evidence; no JavaScript or invented labels."""
    evidence = copy.deepcopy(evidence)
    from .analysis_results import analysis_html

    analysis_section = analysis_html(evidence)
    values, valid = evidence["preview"], evidence["preview_valid"]
    low, high = evidence["scale"]
    palette = json.loads((Path(__file__).parent / "cividis.json").read_text())
    x_cells, t_cells = evidence["preview_x_cells"], evidence["preview_time_cells"]
    x_min, x_max = x_cells[0][0], x_cells[-1][1]
    t_min, t_max = t_cells[0][0], t_cells[-1][1]
    rectangles = []
    for r, row in enumerate(values):
        for c, value in enumerate(row):
            k = min(1, max(0, (value - low) / (high - low))) if valid[r][c] else 0
            rgb = palette[round(k * 255) * 4 : round(k * 255) * 4 + 3]
            color = f"rgb({rgb[0]},{rgb[1]},{rgb[2]})" if valid[r][c] else "#d2d2d2"
            x0, x1 = x_cells[c]
            t0, t1 = t_cells[r]
            rectangles.append(
                f'<rect x="{(x0 - x_min) / (x_max - x_min) * 1000}" y="{(t0 - t_min) / (t_max - t_min) * 320}" width="{(x1 - x0) / (x_max - x_min) * 1000}" height="{(t1 - t0) / (t_max - t_min) * 320}" fill="{color}"/>'
            )
    profile = evidence["profile"]
    clean = profile.get("processed_temperature_c")
    finite = [v for v in profile["temperature_c"] + (clean or []) if v is not None]
    lines = []

    def curve(points, color):
        if len(points) == 1:
            x, y = points[0].split(",")
            return f'<circle cx="{x}" cy="{y}" r="3" fill="{color}"/>'
        return f'<polyline fill="none" stroke="{color}" stroke-width="1.5" points="{" ".join(points)}"/>'

    if finite:
        lo, hi = min(finite) - 0.1, max(finite) + 0.1
        axes = profile["x_m"]
        points = []
        for x, v in zip(axes, profile["temperature_c"]):
            if v is None:
                if points:
                    lines.append(curve(points, "#0f62fe"))
                    points = []
            else:
                points.append(
                    f"{(x - axes[0]) / (axes[-1] - axes[0]) * 1000 if len(axes) > 1 else 500},{180 - (v - lo) / (hi - lo) * 180}"
                )
        if points:
            lines.append(curve(points, "#0f62fe"))
        if clean:
            points = []
            for x, v in zip(axes, clean):
                if v is None:
                    if points:
                        lines.append(curve(points, "#198038"))
                        points = []
                else:
                    points.append(
                        f"{(x - axes[0]) / (axes[-1] - axes[0]) * 1000 if len(axes) > 1 else 500},{180 - (v - lo) / (hi - lo) * 180}"
                    )
            if points:
                lines.append(curve(points, "#198038"))
    profile_axes = (
        f'<text x="-10" y="8" text-anchor="end">{hi:.2f}</text><text x="-10" y="180" text-anchor="end">{lo:.2f}</text><text x="0" y="210">{axes[0]:.3f}</text><text x="1000" y="210" text-anchor="end">{axes[-1]:.3f}</text><text x="500" y="238" text-anchor="middle">Расстояние по волокну, м · температура, °C</text>'
        if finite
        else '<text x="20" y="90">Нет валидных измерений профиля</text>'
    )
    profile_svg = (
        '<h2>Точный профиль</h2><p>Синий — исходные измерения; зелёный — доступная обработка. Общая шкала °C, пропуски не соединяются.</p><svg role="img" aria-label="Профиль температуры" viewBox="-70 -15 1100 260">'
        + "".join(lines)
        + profile_axes
        + "</svg>"
    )
    for name in (
        "preview_time_s",
        "preview_x_m",
        "preview_time_cells",
        "preview_x_cells",
    ):
        evidence.pop(name, None)
    for run in evidence.get("profile_runs", []):
        run.pop("result", None)
    evidence["profile"] = {
        "row": profile["row"],
        "points": len(profile["x_m"]),
        "processing_run_id": profile.get("processing_run_id"),
        "exact_values": "Available in JSON report",
    }
    raw = html.escape(json.dumps(evidence, ensure_ascii=False, indent=2))
    heat_axes = f'<text x="-10" y="8" text-anchor="end">{t_min:.1f}</text><text x="-10" y="320" text-anchor="end">{t_max:.1f}</text><text x="0" y="352">{x_min:.3f}</text><text x="1000" y="352" text-anchor="end">{x_max:.3f}</text><text x="500" y="380" text-anchor="middle">Расстояние, м · вертикальная ось: время от начала, с</text>'
    answer = evidence.get("expert_answer")
    assessment = (
        "<h2>Самостоятельный ответ</h2><p>"
        + html.escape(str(answer.get("temperature_episode", "unknown")))
        + " · "
        + html.escape(str(answer.get("reviewer", "")))
        + "</p><p>"
        + html.escape(str(answer.get("notes", "")))
        + "</p>"
        if answer
        else "<h2>Самостоятельный ответ</h2><p>Для этого участка подтверждённый ответ не выбран.</p>"
    )
    source = html.escape(str(evidence["source"]))
    source_sha = html.escape(str(evidence["source_sha256"]))
    timestamps = html.escape(" → ".join(evidence["timestamps"]))
    coverage = evidence["valid_fraction"] * 100
    gaps = html.escape(str(evidence["gaps_over_300s"]))
    return f"""<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>DTS — отчёт участка</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:32px auto;padding:16px;color:#161616}}svg{{width:100%;height:auto;background:#f4f4f4}}svg text{{font:18px system-ui;fill:#161616}}code{{overflow-wrap:anywhere}}.facts{{padding:20px;background:#f4f4f4;border-left:4px solid #0f62fe}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f4f4;padding:24px}}summary{{cursor:pointer;padding:16px 0}}</style>
<h1>Исследование температурного участка · {source}</h1><p>{"РЕПЕТИЦИЯ: ответ не является экспертной разметкой и не включён в рабочий журнал." if evidence.get("rehearsal") else "Исследовательский отчёт."}</p><p>Наблюдения и гипотезы. Физическая причина неизвестна; точность модели не установлена.</p>
<div class="facts"><p>{timestamps}</p><p>Валидное покрытие: {coverage:.2f}%. Разрывы более 300 с: {gaps}.</p><p>SHA источника: <code>{source_sha}</code></p></div>
<h2>Температурное поле</h2><p>Шкала источника: {low:.2f}–{high:.2f} °C, Cividis. Обзор прорежен по индексам; каждая показанная ячейка сохраняет исходные физические границы. Пустые промежутки не заполнены. Серый — нет валидного измерения.</p>
<svg role="img" aria-label="Температурные наблюдения" viewBox="-80 -20 1100 410">{"".join(rectangles)}{heat_axes}</svg>
{profile_svg}{changes_html(evidence.get("processing_changes"))}{analysis_section}{assessment}<p>Ответ относится к показанному окну; обучающие метки автоматически не создаются. Точные массивы доступны в JSON-экспорте.</p><details><summary>Полное происхождение, параметры и ограничения</summary><pre>{raw}</pre></details></html>"""
