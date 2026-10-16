import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Button, InlineNotification, Select, SelectItem } from "@carbon/react";
import { Link, useSearchParams } from "react-router-dom";
import { control, useAccount } from "./access";

type Point = {
  threshold: number;
  hot_iou: number | null;
  cold_iou: number | null;
  far: number;
};
type FileMetric = {
  file: string;
  source_sha256: string;
  present: boolean;
  zone_count: number;
  median_zone_count: number;
  alarm_fraction: number;
  score_coverage: number;
  overlap_vs_median: number | null;
};
type Row = {
  method: string;
  scale: string;
  threshold: number;
  target_far: number;
  far_holdout: number;
  hot_iou: number;
  cold_iou: number;
  weak_hot_iou: number;
  weak_cold_iou: number;
  threshold_std: number;
  witness_passes: number | null;
  decision: string;
  failed: string[];
  curve_holdout: Point[];
  real_files: FileMetric[];
};
type Hybrid = {
  mode: string;
  hot_iou: number;
  cold_iou: number;
  median_hot_iou: number;
  median_cold_iou: number;
  peak_error: number;
  median_peak_error: number;
  decision: string;
  failed: string[];
  witness_passes: number | null;
  median_witness_passes: number | null;
  files: {
    file: string;
    zone_count: number;
    boundary_shift_m: { median: number | null; max: number | null };
  }[];
};
type Pack = {
  source_commit: string;
  sources: { path: string; sha256: string }[];
  operational_pair: { denoiser: string; detector: string; threshold: number };
  calibration: {
    manifest: {
      calib_seeds: number[];
      holdout_seeds: number[];
      fit_input_sha256: string;
    };
    device_declared: string;
    target_far: number;
    rows: Row[];
  };
  hybrid: { modes: Hybrid[] };
  audit: {
    limitations: string[];
    baseline_cpu_reproduced: boolean;
    neural_cuda_reproduced: boolean;
  };
};
const scales: Record<string, string> = {
  mad_profile: "MAD по профилю",
  mad_residual_global: "Глобальный MAD остатка",
  mad_raw_residual: "MAD исходного остатка",
};
const number = (value: number | null | undefined) =>
  value == null ? "Нет оценки" : value.toFixed(6);
const percentage = (value: number | null | undefined) =>
  value == null ? "Нет оценки" : (value * 100).toFixed(3) + "%";

function Curve({
  title,
  baseline,
  selected,
  metric,
  farX = false,
}: {
  title: string;
  baseline: Row;
  selected: Row;
  metric: "hot_iou" | "cold_iou" | "far";
  farX?: boolean;
}) {
  const [index, setIndex] = useState(0);
  const w = 640,
    h = 300,
    left = 110,
    top = 16,
    right = 560,
    bottom = 218;
  const rows = [baseline, selected];
  const points = rows.map((row) =>
    row.curve_holdout
      .map((p) => ({ x: farX ? p.far : p.threshold, y: p[metric] }))
      .filter((p): p is { x: number; y: number } => p.y !== null),
  );
  const xmax = farX ? Math.max(0.001, ...points.flat().map((p) => p.x)) : 8;
  const xmin = farX ? 0 : 2;
  const ymax =
    metric === "far" ? Math.max(0.001, ...points.flat().map((p) => p.y)) : 1;
  const px = (x: number) =>
      left + ((x - xmin) / (xmax - xmin)) * (right - left),
    py = (y: number) => bottom - (y / ymax) * (bottom - top);
  const chosen =
    selected.curve_holdout[Math.min(index, selected.curve_holdout.length - 1)];
  const base = baseline.curve_holdout.find(
    (p) => p.threshold === chosen?.threshold,
  );
  return (
    <figure className="experiment-chart">
      <h3>{title}</h3>
      <p>
        Синий: median_x3 / MAD по профилю. Красный: {selected.method} /{" "}
        {scales[selected.scale]}.
      </p>
      <svg
        viewBox={`0 0 ${w} ${h}`}
        role="img"
        aria-label={`${title}. Синтетический holdout; точные значения доступны ползунком и численно.`}
      >
        {[0, 0.25, 0.5, 0.75, 1].map((v) => (
          <g key={v}>
            <line
              x1={left}
              x2={right}
              y1={py(v * ymax)}
              y2={py(v * ymax)}
              stroke="var(--dts-chart-grid)"
            />
            <text x={left - 8} y={py(v * ymax) + 4} textAnchor="end">
              {metric === "far" ? percentage(v * ymax) : (v * ymax).toFixed(2)}
            </text>
          </g>
        ))}
        {[0, 0.25, 0.5, 0.75, 1].map((v) => (
          <text
            key={v}
            x={px(xmin + v * (xmax - xmin))}
            y={bottom + 20}
            textAnchor="middle"
          >
            {farX
              ? percentage(xmin + v * (xmax - xmin))
              : (xmin + v * (xmax - xmin)).toFixed(2)}
          </text>
        ))}
        {points.map((series, i) => (
          <polyline
            key={i}
            fill="none"
            stroke={i === 0 ? "var(--dts-chart-series)" : "var(--dts-chart-error)"}
            strokeWidth="2"
            strokeDasharray={i === 0 ? undefined : "5 3"}
            points={[...series]
              .sort((a, b) => a.x - b.x)
              .map((p) => `${px(p.x)},${py(p.y)}`)
              .join(" ")}
          />
        ))}
        {!farX && (
          <>
            <line
              x1={px(baseline.threshold)}
              x2={px(baseline.threshold)}
              y1={top}
              y2={bottom}
              stroke="var(--dts-chart-series)"
              strokeDasharray="2 3"
            />
            <line
              x1={px(selected.threshold)}
              x2={px(selected.threshold)}
              y1={top}
              y2={bottom}
              stroke="var(--dts-chart-error)"
              strokeDasharray="2 3"
            />
          </>
        )}
        <text x={(left + right) / 2} y={h - 8} textAnchor="middle">
          {farX ? "FAR: доля фоновых ячеек" : "Порог |score|"}
        </text>
      </svg>
      <label>
        Точная точка опубликованной кривой
        <input
          aria-label={`${title}: точка кривой`}
          type="range"
          min={0}
          max={Math.max(0, selected.curve_holdout.length - 1)}
          value={index}
          onChange={(e) => setIndex(Number(e.target.value))}
        />
      </label>
      {chosen && (
        <p className="mono">
          Порог {chosen.threshold} · baseline {number(base?.[metric])} ·
          выбранный метод {number(chosen[metric])} · FAR метода{" "}
          {percentage(chosen.far)}
        </p>
      )}
      <figcaption>
        Кривая рассчитана с шагом порога 0.25. Метрики выбранного порога в
        сводке рассчитаны отдельно; интерполяция не выдаётся за измерение.
      </figcaption>
    </figure>
  );
}

function RealComparison({
  baseline,
  selected,
  metric,
}: {
  baseline: Row;
  selected: Row;
  metric: "zone_count" | "alarm_fraction";
}) {
  const entries = selected.real_files.filter((file) => file.present);
  const max = Math.max(
    1e-6,
    ...entries.flatMap((file) => [
      file[metric],
      baseline.real_files.find((b) => b.file === file.file)?.[metric] ?? 0,
    ]),
  );
  const label = (value: number) =>
    metric === "alarm_fraction" ? percentage(value) : String(value);
  return (
    <figure className="experiment-chart experiment-real-chart">
      <h3>
        {metric === "zone_count"
          ? "Количество предложенных зон по записям"
          : "Доля отмеченных валидных ячеек по записям"}
      </h3>
      <p>
        Серый: baseline. Синий: {selected.method}. Это диагностические
        результаты на реальных CSV, без экспертного эталона.
      </p>
      <svg
        viewBox="0 0 640 340"
        role="img"
        aria-label="Сравнение по шести реальным записям; точные числа в выбранной записи"
      >
        {entries.map((file, index) => {
          const y = 20 + index * 50,
            b =
              baseline.real_files.find((f) => f.file === file.file)?.[metric] ??
              0;
          return (
            <g key={file.file}>
              <text x="8" y={y + 20}>
                {file.file}
              </text>
              <rect
                x="68"
                y={y}
                width={(b / max) * 400}
                height="12"
                fill="var(--dts-chart-muted)"
              />
              <rect
                x="68"
                y={y + 16}
                width={(file[metric] / max) * 400}
                height="12"
                fill="var(--dts-chart-series)"
              />
              <text x="480" y={y + 10}>
                {label(b)}
              </text>
              <text x="480" y={y + 28}>
                {label(file[metric])}
              </text>
            </g>
          );
        })}
      </svg>
      <div className="experiment-real-values">
        {entries.map((file) => (
          <dl key={file.file}>
            <dt>{file.file} · baseline</dt>
            <dd>
              {label(
                baseline.real_files.find((b) => b.file === file.file)?.[
                  metric
                ] ?? 0,
              )}
            </dd>
            <dt>{selected.method}</dt>
            <dd>{label(file[metric])}</dd>
          </dl>
        ))}
      </div>
    </figure>
  );
}

export default function Experiments() {
  const account = useAccount();
  const [params, setParams] = useSearchParams();
  const pack = useQuery({
    queryKey: ["ml-experiments", account?.project_id, account?.id],
    queryFn: ({ signal }) =>
      control<Pack>("/ml-experiments", undefined, signal),
    retry: false,
    enabled: !!account,
  });
  const data = pack.data;
  const [file, setFile] = useState("1.csv");
  if (!data)
    return (
      <main className="page">
        <h1>Эксперименты ML</h1>
        {pack.error ? (
          <InlineNotification
            kind="error"
            title="Исследовательские результаты недоступны"
            subtitle={String(pack.error)}
            hideCloseButton
          />
        ) : (
          <p role="status">Загрузка опубликованных результатов…</p>
        )}
        <Link to="/analysis">Вернуться к реальным измерениям</Link>
      </main>
    );
  const baseline = data.calibration.rows.find(
    (r) => r.method === "median_x3" && r.scale === "mad_profile",
  )!;
  const method = params.get("method") ?? "n2n_tsr",
    scale = params.get("scale") ?? "mad_profile";
  const row =
    data.calibration.rows.find(
      (r) => r.method === method && r.scale === scale,
    ) ?? baseline;
  const real = row.real_files.find((r) => r.file === file);
  return (
    <main className="page experiments-page">
      <h1>Эксперименты ML: пороги и границы</h1>
      <p>
        Опубликованные результаты ML-команды. Рабочая пара:{" "}
        {data.operational_pair.denoiser} + {data.operational_pair.detector}, |Z|
        ≥ {data.operational_pair.threshold}. Выбор здесь не меняет рабочий
        алгоритм и не запускает обучение.
      </p>
      <Link to="/analysis">Исследовать реальные измерения →</Link>
      <p className="mono">
        Снимок ML-ветки: {data.source_commit.slice(0, 7)}. Источник —
        сохранённый исследовательский отчёт, не текущий расчёт оборудования.
      </p>
      <InlineNotification
        kind="info"
        title="Синтетический протокол и реальные диагностические проверки"
        subtitle="IoU/FAR измерены на planted events. Данные шести CSV показывают поведение методов; экспертная точность аварий не установлена."
        hideCloseButton
      />
      <div className="analysis-controls">
        <Select
          id="experiment-method"
          labelText="Метод"
          value={row.method}
          onChange={(e) =>
            setParams({ method: e.target.value, scale: row.scale })
          }
        >
          {[...new Set(data.calibration.rows.map((r) => r.method))].map((m) => (
            <SelectItem key={m} value={m} text={m} />
          ))}
        </Select>
        <Select
          id="experiment-scale"
          labelText="Нормировка score"
          value={row.scale}
          onChange={(e) =>
            setParams({ method: row.method, scale: e.target.value })
          }
        >
          {Object.entries(scales).map(([v, label]) => (
            <SelectItem key={v} value={v} text={label} />
          ))}
        </Select>
      </div>
      <section className="experiment-summary">
        <h2>{row.method}: результат исходного протокола</h2>
        <p>
          <strong>
            {row.decision === "baseline"
              ? "Референс для сравнения"
              : row.decision}
          </strong>
          . Статус относится к этому эксперименту.
        </p>
        <dl>
          <div>
            <dt>Подобранный порог</dt>
            <dd>{row.threshold}</dd>
          </div>
          <div>
            <dt>Hot IoU · синтетика</dt>
            <dd>{number(row.hot_iou)}</dd>
          </div>
          <div>
            <dt>Cold IoU · синтетика</dt>
            <dd>{number(row.cold_iou)}</dd>
          </div>
          <div>
            <dt>FAR на holdout</dt>
            <dd>{percentage(row.far_holdout)}</dd>
          </div>
          <div>
            <dt>Целевой FAR</dt>
            <dd>{percentage(row.target_far)}</dd>
          </div>
          <div>
            <dt>Агентские regression witnesses</dt>
            <dd>
              {row.witness_passes ?? "Нет оценки"} / 8 — не человеческие метки
            </dd>
          </div>
        </dl>
        {row.failed.length > 0 && (
          <details>
            <summary>Почему метод не принят</summary>
            <ul>
              {row.failed.map((reason) => (
                <li key={reason}>{reason}</li>
              ))}
            </ul>
          </details>
        )}
      </section>
      <h2>Сравнение при пороге и достигнутом FAR</h2>
      <div className="experiment-grid">
        <Curve
          title="Горячие события: IoU и порог"
          baseline={baseline}
          selected={row}
          metric="hot_iou"
        />
        <Curve
          title="Холодные события: IoU и порог"
          baseline={baseline}
          selected={row}
          metric="cold_iou"
        />
        <Curve
          title="Фоновые отметки: FAR и порог"
          baseline={baseline}
          selected={row}
          metric="far"
        />
        <Curve
          title="Горячие события: IoU при достигнутом FAR"
          baseline={baseline}
          selected={row}
          metric="hot_iou"
          farX
        />
      </div>
      <h2>Поведение на шести реальных CSV</h2>
      <div className="experiment-grid">
        <RealComparison
          baseline={baseline}
          selected={row}
          metric="zone_count"
        />
        <RealComparison
          baseline={baseline}
          selected={row}
          metric="alarm_fraction"
        />
      </div>
      <p>
        Количество зон и пересечение с median_x3 — диагностика изменения
        результата. Это не precision, recall или подтверждённые аварии.
      </p>
      <Select
        id="experiment-record"
        labelText="Реальная запись"
        value={file}
        onChange={(e) => setFile(e.target.value)}
      >
        {row.real_files.map((f) => (
          <SelectItem key={f.file} value={f.file} text={f.file} />
        ))}
      </Select>
      {real && (
        <section className="experiment-summary">
          <h3>
            {real.file} / {row.method}
          </h3>
          <dl>
            <div>
              <dt>Предложенных зон</dt>
              <dd>{real.zone_count}</dd>
            </div>
            <div>
              <dt>Зон baseline</dt>
              <dd>{real.median_zone_count}</dd>
            </div>
            <div>
              <dt>Отмеченные валидные ячейки</dt>
              <dd>{percentage(real.alarm_fraction)}</dd>
            </div>
            <div>
              <dt>Доступность score среди валидных</dt>
              <dd>{percentage(real.score_coverage)}</dd>
            </div>
            <div>
              <dt>Пересечение с baseline</dt>
              <dd>{percentage(real.overlap_vs_median)}</dd>
            </div>
          </dl>
          <p className="mono">Исходник SHA: {real.source_sha256}</p>
          <Link to={`/analysis?record_id=${real.file.replace(/\.csv$/, "")}`}>
            Открыть каталог расчётов этой записи →
          </Link>
        </section>
      )}
      <h2>Уточнение границ: исследовательский результат</h2>
      <p>
        Рабочее уточнение отключено. В исходной реализации выявлены проблемы
        временных разрывов, округления границ и подсчёта исчезнувших масок. Эти
        цифры сохраняют статус авторского отчёта.
      </p>
      <div className="experiment-grid">
        {data.hybrid.modes.map((m) => (
          <section className="experiment-summary" key={m.mode}>
            <h3>
              {m.mode} · {m.decision}
            </h3>
            <dl>
              <div>
                <dt>Hot IoU: своя база → уточнение</dt>
                <dd>
                  {number(m.median_hot_iou)} → {number(m.hot_iou)}
                </dd>
              </div>
              <div>
                <dt>Применённый порог score</dt>
                <dd>{row.threshold}</dd>
              </div>
              <div>
                <dt>Cold IoU: своя база → уточнение</dt>
                <dd>
                  {number(m.median_cold_iou)} → {number(m.cold_iou)}
                </dd>
              </div>
              <div>
                <dt>Ошибка пика, °C: база → уточнение</dt>
                <dd>
                  {number(m.median_peak_error)} → {number(m.peak_error)}
                </dd>
              </div>
            </dl>
            <p>
              База здесь — прямоугольники зон, в таблице порогов — threshold
              support. Их IoU напрямую не сопоставляется.
            </p>
            <details>
              <summary>Условия отказа</summary>
              <ul>
                {m.failed.map((f) => (
                  <li key={f}>{f}</li>
                ))}
              </ul>
            </details>
          </section>
        ))}
      </div>
      <details>
        <summary>Протокол, происхождение и ограничения</summary>
        <p>
          Calibration: {data.calibration.manifest.calib_seeds[0]}–
          {data.calibration.manifest.calib_seeds.at(-1)}. Holdout:{" "}
          {data.calibration.manifest.holdout_seeds[0]}–
          {data.calibration.manifest.holdout_seeds.at(-1)}. Устройство исходного
          отчёта: {data.calibration.device_declared}.{" "}
          {data.audit.baseline_cpu_reproduced
            ? "Baseline независимо повторён на CPU."
            : "Независимый повтор baseline не зафиксирован."}{" "}
          {data.audit.neural_cuda_reproduced
            ? "Нейросетевой CUDA-прогон независимо повторён."
            : "Нейросетевой CUDA-прогон независимо не повторён."}
        </p>
        <ul>
          {data.audit.limitations.map((text) => (
            <li key={text}>{text}</li>
          ))}
        </ul>
        <p className="mono">ML commit: {data.source_commit}</p>
        {data.sources.map((source) => (
          <p className="mono" key={source.path}>
            {source.path} · SHA {source.sha256}
          </p>
        ))}
      </details>
      <Button
        kind="tertiary"
        onClick={() => {
          const blob = new Blob([JSON.stringify(data, null, 2)], {
            type: "application/json",
          });
          const url = URL.createObjectURL(blob);
          const a = document.createElement("a");
          a.href = url;
          a.download = "dts-ml-experiments-237168f.json";
          a.click();
          setTimeout(() => URL.revokeObjectURL(url), 1000);
        }}
      >
        Скачать исследовательскую сводку JSON
      </Button>
      <p>
        Это сводка эксперимента, не экспертный кейс и не пакет исходных
        температур.
      </p>
    </main>
  );
}
