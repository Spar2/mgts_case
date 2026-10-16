"""One-card review page for the next shortlist.

Opens in a browser. Zakhar moves card by card, writes what he sees, and
downloads the answers. The page does not write expert_confirmed into
shortlist.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fiber.denoise.dsp import median_x
from fiber.io import Record
from fiber.shortlist import evidence_paths

QUESTIONS = {
    "wide": "Доля кольца не ниже 0.6 и разрыв с дальним кабелем не ниже 1.5 °C. Это один широкий участок или несколько местных?",
    "hard negative": "В окне нет зоны spatial_z или тревожных клеток мало, но контраст к флангу или дальнему кабелю заметен. В прямоугольнике есть событие?",
    "cooling": "В голубом прямоугольнике остывание? Если нет, напишите, что видно.",
    "residual-versus-context": "Residual и context назвали разные классы. Что видно в прямоугольнике?",
    "local": "Локальная зона. Что видно: нагрев, остывание, фон или несколько участков?",
}

_PAGE = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Короткий список для Захара</title>
<style>
  body { font-family: Segoe UI, sans-serif; margin: 24px auto; max-width: 960px; color: #1a1a1a; }
  canvas { width: 100%; height: auto; border: 1px solid #ccc; background: #f4f4f4; }
  button { font: inherit; margin: 0 8px 8px 0; padding: 8px 14px; }
  textarea { width: 100%; min-height: 72px; font: inherit; }
  .row { margin: 12px 0; }
  .muted { color: #444; }
  .q { font-size: 1.25rem; }
  .on { outline: 2px solid #0b6; }
</style>
</head>
<body>
<p>Одна карточка на экране. Выберите «да», «нет» или «не разобрать», напишите, что видно, и нажмите «Сохранить и дальше». В конце скачайте файл ответов. <code>shortlist.json</code> в блокноте править не нужно.</p>
<p class="muted" id="progress"></p>
<h1 id="title"></h1>
<p class="q" id="question"></p>
<p id="reason"></p>
<p id="where"></p>
<canvas id="map" width="900" height="280"></canvas>
<p class="muted" id="scale"></p>
<canvas id="profile" width="900" height="150"></canvas>
<p class="muted">Нижний график — медиана температуры по времени, °C. Голубая полоса — окно вопроса. Горизонталь — метры, слева направо. На карте время снизу вверх.</p>
<div class="row">
  <div>Есть изменение относительно соседнего кабеля?</div>
  <button type="button" data-presence="yes">да</button>
  <button type="button" data-presence="no">нет</button>
  <button type="button" data-presence="unclear">не разобрать</button>
</div>
<div class="row">
  <div>Что видно, своими словами</div>
  <button type="button" data-insert="Local_Heating">Local_Heating</button>
  <button type="button" data-insert="Local_Cooling">Local_Cooling</button>
  <button type="button" data-insert="Periodic_Thermal">Periodic_Thermal</button>
  <button type="button" data-insert="Broad_Shift">Broad_Shift</button>
  <button type="button" data-insert="несколько участков">несколько участков</button>
  <textarea id="shape"></textarea>
</div>
<div class="row">
  <div>Заметки: метры, °C, один участок или несколько</div>
  <textarea id="notes"></textarea>
</div>
<div class="row">
  <button type="button" id="prev">Назад</button>
  <button type="button" id="save" data-action="save-and-next">Сохранить и дальше</button>
  <button type="button" id="download">Скачать ответы</button>
</div>
<p id="status"></p>
<script>
const CARDS = __CARDS__;
const KEY = "haklolarp-next-shortlist-answers-v1";
let index = 0;
let answers = {};
try { answers = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { answers = {}; }

function finite(heat) {
  const vals = [];
  for (const row of heat) for (const v of row) if (v !== null && Number.isFinite(v)) vals.push(v);
  vals.sort((a, b) => a - b);
  if (!vals.length) return [0, 1];
  const q = (p) => vals[Math.min(vals.length - 1, Math.floor(p * (vals.length - 1)))];
  const lo = q(0.05), hi = q(0.95);
  return lo === hi ? [lo - 0.5, hi + 0.5] : [lo, hi];
}

function drawMap(card) {
  const heat = card.heat;
  const rows = heat.length, cols = heat[0].length;
  const canvas = document.getElementById("map");
  const ctx = canvas.getContext("2d");
  const [lo, hi] = finite(heat);
  const dw = canvas.width / cols, dh = canvas.height / rows;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const v = heat[r][c];
      const y = (rows - 1 - r) * dh;
      if (v === null || !Number.isFinite(v)) {
        ctx.fillStyle = "#d5d5d5";
      } else {
        const t = Math.min(1, Math.max(0, (v - lo) / (hi - lo)));
        ctx.fillStyle = "rgb(" + Math.round(40 + 215 * t) + "," + Math.round(70 + 40 * (1 - Math.abs(t - 0.5) * 2)) + "," + Math.round(200 * (1 - t)) + ")";
      }
      ctx.fillRect(c * dw, y, Math.ceil(dw), Math.ceil(dh));
    }
  }
  const [c0, c1] = card.core_cols, [r0, r1] = card.core_rows;
  ctx.strokeStyle = "#00e5ff";
  ctx.lineWidth = 2;
  ctx.strokeRect(c0 * dw + 1, (rows - r1) * dh + 1, Math.max(2, (c1 - c0) * dw - 2), Math.max(2, (r1 - r0) * dh - 2));
  document.getElementById("scale").textContent =
    "Цвет " + lo.toFixed(2) + "…" + hi.toFixed(2) + " °C (5–95%). Серое — нет измерения. " +
    "В окне медиана " + card.core_median_c + " °C, рядом " + card.side_median_c + " °C.";
}

function drawProfile(card) {
  const canvas = document.getElementById("profile");
  const ctx = canvas.getContext("2d");
  const xs = card.profile_x, ys = card.profile_c;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const finiteY = ys.filter((v) => v !== null && Number.isFinite(v));
  if (!finiteY.length) return;
  const y0 = Math.min.apply(null, finiteY), y1 = Math.max.apply(null, finiteY);
  const pad = y0 === y1 ? 0.5 : (y1 - y0) * 0.08;
  const lo = y0 - pad, hi = y1 + pad;
  const x0 = xs[0], x1 = xs[xs.length - 1];
  const X = (x) => (x - x0) / (x1 - x0 || 1) * (canvas.width - 16) + 8;
  const Y = (y) => (1 - (y - lo) / (hi - lo)) * (canvas.height - 16) + 8;
  ctx.fillStyle = "rgba(0,229,255,0.25)";
  ctx.fillRect(X(card.core_x[0]), 8, Math.max(2, X(card.core_x[1]) - X(card.core_x[0])), canvas.height - 16);
  ctx.beginPath();
  let started = false;
  for (let i = 0; i < xs.length; i++) {
    if (ys[i] === null || !Number.isFinite(ys[i])) { started = false; continue; }
    const px = X(xs[i]), py = Y(ys[i]);
    if (!started) { ctx.moveTo(px, py); started = true; } else ctx.lineTo(px, py);
  }
  ctx.strokeStyle = "#b00020";
  ctx.lineWidth = 2;
  ctx.stroke();
}

function show() {
  const card = CARDS[index];
  const done = Object.keys(answers).length;
  document.getElementById("progress").textContent = "Карточка " + (index + 1) + " из " + CARDS.length + ". Отвечено " + done + ".";
  document.getElementById("title").textContent = card.id + " · файл " + card.record;
  document.getElementById("question").textContent = card.question;
  document.getElementById("reason").textContent = card.reason || "";
  document.getElementById("where").textContent =
    card.meters[0] + "–" + card.meters[1] + " м, " + card.time[0] + " – " + card.time[1] +
    (card.tz_missing ? ". У этого файла время без часового пояса." : "");
  drawMap(card);
  drawProfile(card);
  const saved = answers[card.id] || {};
  document.getElementById("shape").value = saved.shape || "";
  document.getElementById("notes").value = saved.notes || "";
  for (const button of document.querySelectorAll("[data-presence]")) {
    button.classList.toggle("on", button.dataset.presence === saved.presence);
  }
  document.getElementById("status").textContent = saved.presence ? "Ответ этой карточки уже записан в браузере." : "";
}

function currentPresence() {
  const on = document.querySelector("[data-presence].on");
  return on ? on.dataset.presence : "";
}

function remember() {
  const card = CARDS[index];
  const presence = currentPresence();
  if (!presence) return false;
  answers[card.id] = {
    id: card.id,
    record: card.record,
    source_sha256: card.source_sha256,
    meters: card.meters,
    time: card.time,
    presence: presence,
    shape: document.getElementById("shape").value.trim(),
    reviewer: "Захар",
    reviewer_type: "human",
    notes: document.getElementById("notes").value.trim()
  };
  localStorage.setItem(KEY, JSON.stringify(answers));
  return true;
}

document.querySelectorAll("[data-presence]").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll("[data-presence]").forEach((other) => other.classList.remove("on"));
    button.classList.add("on");
  });
});
document.querySelectorAll("[data-insert]").forEach((button) => {
  button.addEventListener("click", () => {
    const box = document.getElementById("shape");
    box.value = box.value ? box.value + ", " + button.dataset.insert : button.dataset.insert;
  });
});
document.getElementById("prev").addEventListener("click", () => {
  if (currentPresence()) remember();
  index = Math.max(0, index - 1);
  show();
});
document.getElementById("save").addEventListener("click", () => {
  if (!remember()) {
    document.getElementById("status").textContent = "Сначала выберите: да, нет или не разобрать.";
    return;
  }
  document.getElementById("status").textContent = "Записано в браузере. В конце нажмите «Скачать ответы».";
  if (index < CARDS.length - 1) index += 1;
  show();
});
document.getElementById("download").addEventListener("click", () => {
  if (currentPresence()) remember();
  const lines = CARDS.map((card) => answers[card.id]).filter(Boolean).map((row) => JSON.stringify(row)).join("\\n");
  const blob = new Blob([lines ? lines + "\\n" : ""], {type: "application/json"});
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "zakhar_next_shortlist_answers.jsonl";
  link.click();
  document.getElementById("status").textContent = "Файл скачан. Пришлите zakhar_next_shortlist_answers.jsonl обратно. В shortlist.json ничего писать не нужно.";
});
show();
</script>
</body>
</html>
"""


def preview_window(clean, valid, x_m, time_iso, meters, time_bounds, *, rows_half_open=None, pad_m=40.0, max_rows=120, max_cols=160):
    """Downsample one review window. Time on the heatmap runs bottom to top."""
    field = np.asarray(clean, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(field)
    x = np.asarray(x_m, dtype=np.float64)
    if field.ndim != 2 or field.shape != mask.shape or field.shape[1] != x.size:
        raise ValueError("clean, valid, and x_m must share the cable grid")
    if rows_half_open is None:
        r0, r1 = _time_bounds(time_iso, time_bounds[0], time_bounds[1])
    else:
        r0, r1 = int(rows_half_open[0]), int(rows_half_open[1])
        r0 = max(0, min(field.shape[0], r0))
        r1 = max(r0 + 1, min(field.shape[0], r1))
    core_r0, core_r1 = r0, r1
    if r1 - r0 < 8:
        r0 = max(0, r0 - 4)
        r1 = min(field.shape[0], r1 + 4)
    x0, x1 = float(meters[0]), float(meters[1])
    c0 = int(np.searchsorted(x, x0 - pad_m, side="left"))
    c1 = int(np.searchsorted(x, x1 + pad_m, side="right"))
    c0 = max(0, min(x.size - 1, c0))
    c1 = max(c0 + 1, min(x.size, c1))
    core_c0 = int(np.searchsorted(x, x0, side="left"))
    core_c1 = int(np.searchsorted(x, x1, side="right"))
    core_c0 = min(max(core_c0, c0), c1 - 1)
    core_c1 = max(core_c0 + 1, min(core_c1, c1))
    view = field[r0:r1, c0:c1]
    view_valid = mask[r0:r1, c0:c1]
    heat, br, bc = _block_median(view, view_valid, max_rows=max_rows, max_cols=max_cols)
    local_r0, local_r1 = core_r0 - r0, core_r1 - r0
    local_c0, local_c1 = core_c0 - c0, core_c1 - c0
    core = view[local_r0:local_r1, local_c0:local_c1]
    core_ok = view_valid[local_r0:local_r1, local_c0:local_c1]
    side_ok = view_valid[local_r0:local_r1].copy()
    side_ok[:, local_c0:local_c1] = False
    side = view[local_r0:local_r1]
    profile = _column_median(view[local_r0:local_r1], view_valid[local_r0:local_r1])
    step = max(1, int(np.ceil(profile.size / 200)))
    return {
        "heat": heat,
        "core_rows": [local_r0 // br, int(np.ceil(local_r1 / br))],
        "core_cols": [local_c0 // bc, int(np.ceil(local_c1 / bc))],
        "core_x": [float(x[core_c0]), float(x[core_c1 - 1])],
        "x_edges": [float(x[c0]), float(x[c1 - 1])],
        "time_shown": [str(time_iso[r0]), str(time_iso[r1 - 1])],
        "profile_x": [float(v) for v in x[c0:c1:step]],
        "profile_c": [None if not np.isfinite(v) else round(float(v), 3) for v in profile[::step]],
        "core_median_c": _median_text(core, core_ok),
        "side_median_c": _median_text(side, side_ok),
    }


def render_page(cards) -> str:
    payload = json.dumps(cards, ensure_ascii=False).replace("<", "\\u003c")
    return _PAGE.replace("__CARDS__", payload)


def write_review_page(root=None) -> Path:
    """Build shortlist.html from the committed cards and the local records."""
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    shortlist = json.loads(evidence_paths(root)["shortlist"].read_text(encoding="utf-8"))
    records = {}
    views = []
    for card in shortlist["cards"]:
        name = str(card["record"])
        if name not in records:
            rec = Record.load(root / "artifacts" / "records" / name)
            clean = median_x(rec.raw_c, rec.valid)
            records[name] = (rec, clean)
        rec, clean = records[name]
        view = preview_window(
            clean, rec.valid, rec.x_m, rec.time_iso, card["meters"], card["time"],
            rows_half_open=card.get("rows_half_open"),
        )
        views.append({
            "id": card["id"],
            "record": name,
            "source_sha256": card["source_sha256"],
            "question": QUESTIONS[card["bucket"]],
            "reason": card["reason"],
            "meters": card["meters"],
            "time": card["time"],
            "tz_missing": bool(rec.tz_missing),
            **view,
        })
    out = root / "shortlist.html"
    out.write_text(render_page(views), encoding="utf-8", newline="\n")
    return out


def _time_bounds(time_iso, t0, t1):
    stamps = [str(value) for value in time_iso]
    start = 0
    for index, stamp in enumerate(stamps):
        if stamp >= t0:
            start = index
            break
    stop = len(stamps)
    for index in range(start, len(stamps)):
        if stamps[index] > t1:
            stop = index
            break
    if stop <= start:
        stop = min(len(stamps), start + 1)
    return start, stop


def _block_median(values, valid, *, max_rows, max_cols):
    n_rows, n_cols = values.shape
    row_step = max(1, int(np.ceil(n_rows / max_rows)))
    col_step = max(1, int(np.ceil(n_cols / max_cols)))
    heat = []
    for r0 in range(0, n_rows, row_step):
        row = []
        for c0 in range(0, n_cols, col_step):
            block = values[r0:r0 + row_step, c0:c0 + col_step]
            mask = valid[r0:r0 + row_step, c0:c0 + col_step]
            if not np.any(mask):
                row.append(None)
            else:
                row.append(round(float(np.median(block[mask])), 2))
        heat.append(row)
    return heat, row_step, col_step


def _column_median(values, valid):
    out = np.full(values.shape[1], np.nan, dtype=np.float64)
    for index in range(values.shape[1]):
        mask = valid[:, index]
        if np.any(mask):
            out[index] = float(np.median(values[:, index][mask]))
    return out


def _median_text(values, valid):
    if not np.any(valid):
        return None
    return round(float(np.median(values[valid])), 3)


def main():
    path = write_review_page()
    print(path)


if __name__ == "__main__":
    main()
