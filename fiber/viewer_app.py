"""Minimal operational viewer. The dense heatmap is not POST /detect/field."""

from __future__ import annotations

import json

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from fiber.viewer import example_hotspot, expert_windows, view_field

app = FastAPI(title="haklolarp-viewer", version="0.1.0")

PAGE = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Операционный разбор примера</title>
<style>
  body { font-family: sans-serif; margin: 24px; max-width: 980px; }
  canvas { border: 1px solid #ccc; image-rendering: pixelated; }
  table { border-collapse: collapse; margin-top: 16px; }
  td, th { border: 1px solid #ccc; padding: 4px 8px; text-align: left; }
  .note { color: #333; }
</style>
</head>
<body>
<h1>Операционный путь: median_x3, затем spatial_z_halo</h1>
<p class="note">Горизонталь — метры кабеля, слева направо. Вертикаль — время, снизу вверх, шаг 60 с. Цвет — температура после median_x3, в °C. Это встроенный пример, не файлы 1–6. shape_hypothesis — гипотеза формы, не экспертная метка. expert_confirmed на зонах false.</p>
<canvas id="map" width="800" height="220"></canvas>
<p id="scale"></p>
<table id="zones"></table>
<h2>Окна research рядом с baseline</h2>
<p class="note">Таблица читает сохранённые отчёты serve=research, operational false, и промер spatial_z для прежних 15. Ask expert seeds идут следом: класс эксперта, класс контекстной сети и три числа context evidence из отчёта контекстной сети. Это не операционный детектор и не точность. confirmed — класс сети совпал с классом эксперта. Опровергнутое окно остаётся опровергнутым. Рядом contrast baseline в °C либо фраза baseline kept no zone. next-0010 входит в таблицу. Медиана прямоугольника, far cable и доля halo не ставят anomaly_status. Отсутствующая медиана far cable пишется absent и не копируется из прямоугольника.</p>
<table id="windows">__WINDOWS__</table>
<h2>Широкий канал, serve=research, operational false</h2>
<p class="note">Флаг wide_channel и кандидат Broad_Shift только при флаге true. Кандидат не заменяет экспертный класс. expert null остаётся null. Флаг false остаётся false.</p>
<table id="wide">__WIDE__</table>
<script>
function measurement(value) {
  return value === null || value === undefined ? "absent" : String(value);
}
function fillWindows(payload) {
  const table = document.getElementById("windows");
  table.innerHTML = "<tr><th>окно</th><th>эксперт</th><th>сеть research</th><th>verdict</th><th>baseline</th><th>медиана прямоугольника</th><th>far cable</th><th>доля halo</th></tr>";
  for (const row of payload.windows) {
    const tr = document.createElement("tr");
    const cells = [
      row.id,
      row.expert_class,
      row.network_class,
      row.verdict,
      row.baseline_text,
      "rectangle_median_c " + measurement(row.rectangle_median_c),
      "far_cable_median_c " + measurement(row.far_cable_median_c),
      "halo_inside_fraction " + measurement(row.halo_inside_fraction)
    ];
    for (const value of cells) {
      const td = document.createElement("td");
      td.textContent = value;
      tr.appendChild(td);
    }
    table.appendChild(tr);
  }
  const wide = document.getElementById("wide");
  wide.innerHTML = "<tr><th>окно</th><th>флаг</th><th>кандидат</th><th>эксперт</th><th>медиана прямоугольника</th><th>far cable</th><th>доля halo</th><th>разрыв медиан</th></tr>";
  for (const row of payload.wide_channels) {
    const tr = document.createElement("tr");
    const flag = row.wide_channel ? "wide_channel true" : "wide_channel false";
    const candidate = row.wide_channel_candidate === null || row.wide_channel_candidate === undefined
      ? "candidate absent"
      : "candidate " + row.wide_channel_candidate;
    const expert = row.expert_class === null || row.expert_class === undefined
      ? "expert null"
      : "expert " + row.expert_class;
    const cells = [
      row.id,
      flag,
      candidate,
      expert,
      "rectangle_median_c " + measurement(row.rectangle_median_c),
      "far_cable_median_c " + measurement(row.far_cable_median_c),
      "halo_inside_fraction " + measurement(row.halo_inside_fraction),
      "far_gap_c " + measurement(row.far_gap_c)
    ];
    for (const value of cells) {
      const td = document.createElement("td");
      td.textContent = value;
      tr.appendChild(td);
    }
    wide.appendChild(tr);
  }
}
async function main() {
  const body = await (await fetch("/view/example")).json();
  const heat = body.heatmap;
  const rows = heat.length, cols = heat[0].length;
  const canvas = document.getElementById("map");
  const ctx = canvas.getContext("2d");
  let lo = Infinity, hi = -Infinity;
  for (const row of heat) for (const v of row) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  const dw = canvas.width / cols, dh = canvas.height / rows;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const t = hi === lo ? 0.5 : (heat[r][c] - lo) / (hi - lo);
      const red = Math.round(255 * t), blue = Math.round(255 * (1 - t));
      ctx.fillStyle = "rgb(" + red + ",40," + blue + ")";
      ctx.fillRect(c * dw, (rows - 1 - r) * dh, Math.ceil(dw), Math.ceil(dh));
    }
  }
  document.getElementById("scale").textContent =
    "аномалия: " + body.anomaly_status +
    "; температура " + lo.toFixed(2) + "…" + hi.toFixed(2) + " °C" +
    "; метров " + body.x_m[0] + "…" + body.x_m[body.x_m.length - 1];
  const table = document.getElementById("zones");
  table.innerHTML = "<tr><th>метры</th><th>shape_hypothesis</th><th>contrast °C</th><th>width_m</th><th>observed_duration_s</th><th>support</th><th>evidence</th></tr>";
  for (const zone of body.zones) {
    const tr = document.createElement("tr");
    const evidence = zone.evidence.spatial.view + " + " + zone.evidence.temporal.view
      + " (" + zone.evidence.temporal.score_state + ")";
    tr.innerHTML = "<td>" + zone.start_m + "–" + zone.end_m + "</td><td>" + zone.shape_hypothesis
      + "</td><td>" + zone.contrast + "</td><td>" + zone.width_m + "</td><td>"
      + zone.observed_duration_s + "</td><td>" + zone.support + "</td><td>" + evidence + "</td>";
    table.appendChild(tr);
  }
  fillWindows(await (await fetch("/view/windows")).json());
}
main();
</script>
</body>
</html>
"""


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "haklolarp-viewer", "path": "operational"}


def _html_text(value) -> str:
    if value is None:
        text = "absent"
    elif isinstance(value, float):
        text = json.dumps(value)
    else:
        text = str(value)
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _research_tables(payload: dict) -> tuple[str, str]:
    """Server-rendered research rows. The numbers are the stored JSON values."""
    window_rows = [
        "<tr><th>окно</th><th>эксперт</th><th>сеть research</th><th>verdict</th><th>baseline</th>"
        "<th>медиана прямоугольника</th><th>far cable</th><th>доля halo</th></tr>"
    ]
    for row in payload["windows"]:
        window_rows.append(
            "<tr><td>"
            + "</td><td>".join(
                _html_text(value)
                for value in (
                    row["id"],
                    row["expert_class"],
                    row["network_class"],
                    row["verdict"],
                    row["baseline_text"],
                    "rectangle_median_c " + _html_text(row["rectangle_median_c"]),
                    "far_cable_median_c " + _html_text(row["far_cable_median_c"]),
                    "halo_inside_fraction " + _html_text(row["halo_inside_fraction"]),
                )
            )
            + "</td></tr>"
        )
    wide_rows = [
        "<tr><th>окно</th><th>флаг</th><th>кандидат</th><th>эксперт</th>"
        "<th>медиана прямоугольника</th><th>far cable</th><th>доля halo</th><th>разрыв медиан</th></tr>"
    ]
    for row in payload["wide_channels"]:
        flag = "wide_channel true" if row["wide_channel"] else "wide_channel false"
        candidate = (
            "candidate absent"
            if row["wide_channel_candidate"] is None
            else "candidate " + str(row["wide_channel_candidate"])
        )
        expert = "expert null" if row["expert_class"] is None else "expert " + str(row["expert_class"])
        wide_rows.append(
            "<tr><td>"
            + "</td><td>".join(
                _html_text(value)
                for value in (
                    row["id"],
                    flag,
                    candidate,
                    expert,
                    "rectangle_median_c " + _html_text(row["rectangle_median_c"]),
                    "far_cable_median_c " + _html_text(row["far_cable_median_c"]),
                    "halo_inside_fraction " + _html_text(row["halo_inside_fraction"]),
                    "far_gap_c " + _html_text(row["far_gap_c"]),
                )
            )
            + "</td></tr>"
        )
    return "".join(window_rows), "".join(wide_rows)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    windows, wide = _research_tables(expert_windows())
    return PAGE.replace("__WINDOWS__", windows).replace("__WIDE__", wide)


@app.get("/view/example")
def view_example() -> dict:
    raw, x_m, time_s = example_hotspot()
    return view_field(raw, x_m, time_s)


@app.get("/view/windows")
def view_windows() -> dict:
    return expert_windows()
