import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Button,
  InlineNotification,
  Select,
  SelectItem,
  TextInput,
} from "@carbon/react";
import { Link, useSearchParams } from "react-router-dom";
import { control } from "./access";
import { CoordinateInput } from "./coordinate-input";
import { Heatmap, LineChart } from "./charts";
import { CaseForm } from "./cases";
import {
  parseAnalysisContext,
  analysisParams,
  type AnalysisContext,
} from "./analysis-context";

type Region = {
  record_id: string;
  row_start: number;
  row_end: number;
  column_start: number;
  column_end: number;
};
export type AnalysisRun = {
  id: string;
  region: Region;
  source_sha256: string;
  method: string;
  method_version: string;
  layers: string[];
  training_origin: string;
  research_status: string;
  limitations: string;
  parameters: object;
  calculated_at: string;
  time_kind?: string;
  timestamps?: string[];
  scales?: Record<string, [number, number]>;
  score?: {
    name: string;
    unit: string;
    meaning: string;
    threshold?: number;
    comparison?: string;
  };
  zones: (Region & { id: string; shape_hypothesis?: string })[];
};
type SliceLayer = {
  profile: (number | null)[];
  trend: (number | null)[];
  coverage: (number | null)[];
  zone?: {
    p10: (number | null)[];
    median: (number | null)[];
    p90: (number | null)[];
    valid_count: (number | null)[];
  };
};
type Slices = {
  run_id: string;
  row: number;
  column: number;
  source_sha256: string;
  x_m: number[];
  time_s: number[];
  native_time_s: number[];
  layers: Record<string, SliceLayer>;
  source_invalid_reason_profile: number[];
};
const labels: Record<string, string> = {
  raw: "Исходное поле",
  processed_temperature_c: "Обработанное поле",
  background_temperature_c: "Оценка фона",
  difference: "Разность: обработка − исходник",
  score: "Оценка аномальности",
};
function extent(
  values: (number | null)[],
  fallback: [number, number],
): [number, number] {
  let lo = Infinity,
    hi = -Infinity;
  for (const value of values)
    if (value !== null) {
      lo = Math.min(lo, value);
      hi = Math.max(hi, value);
    }
  return Number.isFinite(lo) ? [lo - 0.1, hi + 0.1] : fallback;
}
function RunView({
  run,
  comparison,
}: {
  run: AnalysisRun;
  comparison?: AnalysisRun;
}) {
  const [params, setParams] = useSearchParams();
  const parsed = parseAnalysisContext(params, run);
  if (!parsed.ok)
    return (
      <section>
        <InlineNotification
          kind="error"
          title="Контекст исследования несовместим"
          subtitle={parsed.message}
          hideCloseButton
        />
        <Button onClick={() => setParams({ run_id: run.id })}>
          Открыть покрытие расчёта заново
        </Button>
      </section>
    );
  return (
    <AnalysisWindow
      run={run}
      comparison={comparison}
      context={parsed.context}
    />
  );
}
export function AnalysisWindow({
  run,
  comparison,
  context,
  onContextChange,
  onSavedCase,
}: {
  run: AnalysisRun;
  comparison?: AnalysisRun;
  context: AnalysisContext;
  onContextChange?: (context: AnalysisContext) => void;
  onSavedCase?: (id: string) => void;
}) {
  const [params, setParams] = useSearchParams();
  const r = run.region;
  const { row, column, layer, zone, region: boundaries } = context;
  const update = (changes: Partial<AnalysisContext>, replace = true) => {
    if (onContextChange) onContextChange({ ...context, ...changes });
    else
      setParams(analysisParams(run, { ...context, ...changes }, params), {
        replace,
      });
  };
  const setRow = (row: number) => update({ row });
  const setColumn = (column: number) => update({ column });
  const setLayer = (layer: string) => update({ layer });
  const [title, setTitle] = useState("Разбор реального ML-результата"),
    [notes, setNotes] = useState("");
  const selected = run.zones.find((z) => z.id === zone);
  const chooseRegion = (region: Region, zone = "") =>
    update(
      {
        region,
        zone,
        row: Math.min(Math.max(row, region.row_start), region.row_end - 1),
        column: Math.min(
          Math.max(column, region.column_start),
          region.column_end - 1,
        ),
      },
      false,
    );
  const checked = (value: Slices, target: AnalysisRun) => {
    if (
      value.run_id !== target.id ||
      value.source_sha256 !== target.source_sha256 ||
      value.row !== row ||
      value.column !== column ||
      value.x_m.length !== boundaries.column_end - boundaries.column_start ||
      value.native_time_s.length !==
        boundaries.row_end - boundaries.row_start ||
      !value.layers.raw ||
      Object.values(value.layers).some(
        (l) =>
          l.profile.length !== value.x_m.length ||
          l.trend.length !== value.time_s.length ||
          l.coverage.length !== value.time_s.length,
      )
    )
      throw new Error(
        "Срез не соответствует выбранному источнику, окну или точке",
      );
    return value;
  };
  const windowQuery = {
    row_start: String(boundaries.row_start),
    row_end: String(boundaries.row_end),
    column_start: String(boundaries.column_start),
    column_end: String(boundaries.column_end),
  };
  const slice = useQuery({
    queryKey: [
      "analysis-slice",
      run.id,
      run.source_sha256,
      boundaries,
      row,
      column,
      zone,
    ],
    queryFn: ({ signal }) =>
      control<Slices>(
        `/result-runs/${run.id}/slices?${new URLSearchParams({ ...windowQuery, row: String(row), column: String(column), ...(zone ? { zone_id: zone } : {}) })}`,
        undefined,
        signal,
      ).then((value) => checked(value, run)),
    retry: false,
    gcTime: 10000,
  });
  const other = useQuery({
    queryKey: [
      "analysis-compare",
      comparison?.id,
      comparison?.source_sha256,
      boundaries,
      row,
      column,
    ],
    queryFn: ({ signal }) =>
      control<Slices>(
        `/result-runs/${comparison!.id}/slices?${new URLSearchParams({ ...windowQuery, row: String(row), column: String(column) })}`,
        undefined,
        signal,
      ).then((value) => checked(value, comparison!)),
    enabled: !!comparison,
    gcTime: 10000,
    retry: false,
  });
  const s = slice.data;
  const zoneGeometry = useQuery({
    queryKey: ["analysis-zone-geometry", run.id],
    queryFn: ({ signal }) =>
      control<
        {
          id: string;
          segment: number;
          x0: number;
          x1: number;
          t0: number;
          t1: number;
          label: string;
        }[]
      >(`/result-runs/${run.id}/zones`, undefined, signal),
    retry: false,
  });
  const external = useQuery({
    queryKey: ["analysis-external", run.id],
    queryFn: ({ signal }) =>
      control<
        {
          id: string;
          review_id: string;
          declared_reviewer: string;
          notes: string;
          source_commit: string;
        }[]
      >(`/result-runs/${run.id}/external-reviews`, undefined, signal),
    retry: false,
  });
  const layers = [
    "raw",
    ...run.layers,
    ...(run.layers.includes("processed_temperature_c") ? ["difference"] : []),
  ];

  const tile = `/api/v2/result-runs/${run.id}/tiles?${new URLSearchParams({ layer, width: matchMedia("(max-width: 672px)").matches ? "128" : "256", height: "256", ...(s ? { x0: String(s.x_m[0]), x1: String(s.x_m.at(-1)), t0: String(s.native_time_s[0]), t1: String(s.native_time_s.at(-1)) } : {}) })}`;
  const tempValues = s
    ? [
        s.layers.raw.profile,
        s.layers.processed_temperature_c?.profile ?? [],
        s.layers.background_temperature_c?.profile ?? [],
        other.data?.layers.processed_temperature_c?.profile ?? [],
      ].flat()
    : [];
  const tempScale = extent(tempValues, [0, 1]);
  return (
    <section className="analysis-workspace">
      <h2>
        {run.method} / {run.method_version}
      </h2>
      <p>
        Запись {r.record_id} · {run.research_status} · обучение:{" "}
        {run.training_origin}
      </p>
      <p className="mono">
        Run: {run.id} · SHA: {run.source_sha256}
      </p>
      <p>{run.limitations}</p>
      {!onContextChange && (
        <Link
          className="text-link"
          to={
            "/works?" +
            new URLSearchParams({
              record: r.record_id,
              source_sha256: run.source_sha256,
              row: String(row),
              column: String(column),
              r0: String(boundaries.row_start),
              r1: String(boundaries.row_end),
              c0: String(boundaries.column_start),
              c1: String(boundaries.column_end),
              run_id: run.id,
              layer,
            })
          }
        >
          Начать работу с этим результатом →
        </Link>
      )}
      <details>
        <summary>Параметры и происхождение</summary>
        <pre>
          {JSON.stringify(
            {
              parameters: run.parameters,
              calculated_at: run.calculated_at,
              time_kind: run.time_kind,
              timestamps: run.timestamps,
              score: run.score,
            },
            null,
            2,
          )}
        </pre>
      </details>
      <div className="analysis-controls">
        <Select
          id="analysis-layer"
          labelText="Слой"
          value={layer}
          onChange={(e) => setLayer(e.target.value)}
        >
          {layers.map((name) => (
            <SelectItem key={name} value={name} text={labels[name] ?? name} />
          ))}
        </Select>
        <Select
          id="analysis-zone"
          labelText="Зона алгоритма — гипотеза"
          value={zone}
          onChange={(e) => {
            const z = run.zones.find((item) => item.id === e.target.value);
            chooseRegion(
              z ? { ...z, record_id: r.record_id } : r,
              e.target.value,
            );
          }}
        >
          <SelectItem
            value=""
            text={
              JSON.stringify(boundaries) === JSON.stringify(r)
                ? "Весь участок расчёта"
                : "Собственное окно"
            }
          />
          {run.zones.map((z) => (
            <SelectItem
              key={z.id}
              value={z.id}
              text={`${z.id} · ${z.shape_hypothesis ?? "класс не указан"}`}
            />
          ))}
        </Select>
        <CoordinateInput
          id="analysis-row"
          label="Профиль — исходный индекс"
          integer
          min={boundaries.row_start}
          max={boundaries.row_end - 1}
          value={row}
          onCommit={setRow}
          onPrevious={() => setRow(Math.max(boundaries.row_start, row - 1))}
          onNext={() => setRow(Math.min(boundaries.row_end - 1, row + 1))}
        />
        <CoordinateInput
          id="analysis-column"
          label="Точка — исходный индекс"
          integer
          min={boundaries.column_start}
          max={boundaries.column_end - 1}
          value={column}
          onCommit={setColumn}
          onPrevious={() =>
            setColumn(Math.max(boundaries.column_start, column - 1))
          }
          onNext={() =>
            setColumn(Math.min(boundaries.column_end - 1, column + 1))
          }
        />
      </div>
      <WindowEditor
        key={JSON.stringify(boundaries)}
        region={boundaries}
        run={run}
        onApply={chooseRegion}
      />
      <p className="mono">
        Окно: строки [{boundaries.row_start}, {boundaries.row_end}), столбцы [
        {boundaries.column_start}, {boundaries.column_end}). Ссылка сохраняет
        окно, слой и точку.
      </p>
      {slice.isFetching && <p role="status">Загрузка выбранного окна…</p>}
      {other.error && (
        <InlineNotification
          kind="error"
          title="Сравнение не загружено"
          subtitle={String(other.error)}
          hideCloseButton
        />
      )}
      {slice.error && (
        <InlineNotification
          kind="error"
          title="Срез не загружен"
          subtitle={String(slice.error)}
          hideCloseButton
        />
      )}
      <h2>{labels[layer]}</h2>
      {layer === "score" && run.score && (
        <p>
          {run.score.meaning} · порог: {run.score.threshold ?? "не задан"} ·
          правило: {run.score.comparison ?? "не задано"}. Это оценка алгоритма,
          не вероятность аварии.
        </p>
      )}
      {s && (
        <Heatmap
          key={tile}
          path={tile}
          versionKey={run.id}
          expectedSourceSha={run.source_sha256}
          zones={zoneGeometry.data}
          selectedZoneId={zone}
          cursor={
            s
              ? [
                  s.x_m[column - boundaries.column_start],
                  s.native_time_s[row - boundaries.row_start],
                ]
              : undefined
          }
          onSelect={(x, t) => {
            if (!s) return;
            const nearest = (a: number[], v: number) =>
              a.reduce(
                (best, value, i) =>
                  Math.abs(value - v) < Math.abs(a[best] - v) ? i : best,
                0,
              );
            update({
              row: boundaries.row_start + nearest(s.native_time_s, t),
              column: boundaries.column_start + nearest(s.x_m, x),
            });
          }}
        />
      )}
      {run.zones.length > 0 && (
        <p>
          Штриховой контур — прямоугольник предложения алгоритма, не экспертная
          граница и не точная маска события. Синий контур соответствует
          выбранной зоне. Временные разрывы разделяют отображение.
        </p>
      )}
      {zoneGeometry.error && (
        <InlineNotification
          kind="warning"
          title="Контуры зон не загружены"
          subtitle="Исходное поле и точные срезы остаются доступными."
          hideCloseButton
        />
      )}
      {(zoneGeometry.data?.length ?? 0) > 200 && (
        <p>
          На поле показаны первые 200 контуров. Все зоны доступны в списке
          выбора; выберите конкретную зону для подробного просмотра.
        </p>
      )}
      {comparison && s && (
        <section>
          <h2>
            Второй запуск: {comparison.method} / {comparison.method_version}
          </h2>
          {layer === "raw" ||
          comparison.layers.includes(layer) ||
          (layer === "difference" &&
            comparison.layers.includes("processed_temperature_c")) ? (
            <Heatmap
              key={tile.replace(run.id, comparison.id)}
              path={tile.replace(run.id, comparison.id)}
              versionKey={comparison.id}
              expectedSourceSha={comparison.source_sha256}
              cursor={
                s
                  ? [
                      s.x_m[column - boundaries.column_start],
                      s.native_time_s[row - boundaries.row_start],
                    ]
                  : undefined
              }
            />
          ) : (
            <p>
              Во втором запуске этот слой отсутствует. Другие значения не
              подставляются.
            </p>
          )}
          <p>
            У температурных полей общая шкала источника. Шкала score относится к
            паспорту своего метода; значения разных протоколов не являются общей
            метрикой качества.
          </p>
        </section>
      )}
      {s && (
        <>
          <p>
            Точка: {s.x_m[column - boundaries.column_start]} м · время от
            начала: {s.native_time_s[row - boundaries.row_start]} с. Профиль и
            временной ряд относятся к этой точке.
          </p>
          <div className="line-grid">
            {[
              "raw",
              "processed_temperature_c",
              "background_temperature_c",
              "difference",
              "score",
            ]
              .filter((name) => s.layers[name])
              .map((name) => (
                <LineChart
                  key={name}
                  title={`${labels[name]} · пространственный профиль`}
                  x={s.x_m}
                  y={s.layers[name].profile}
                  xLabel="Расстояние, м"
                  yLimits={
                    name === "difference" || name === "score"
                      ? run.scales?.[name]
                      : tempScale
                  }
                  yLabel={name === "score" ? run.score?.unit : "°C"}
                  syncKey={`analysis-x-${run.id}`}
                />
              ))}
            <aside className="analysis-point">
              <h3>Точные значения выбранной точки</h3>
              <dl>
                <dt>Расстояние</dt>
                <dd>{s.x_m[column - boundaries.column_start]} м</dd>
                <dt>Причина качества исходника</dt>
                <dd>
                  {s.source_invalid_reason_profile?.[
                    column - boundaries.column_start
                  ] ?? "не передана"}
                </dd>
                {Object.entries(s.layers).map(([name, data]) => (
                  <div key={name}>
                    <dt>{labels[name]}</dt>
                    <dd>
                      {data.profile[column - boundaries.column_start] ??
                        "Нет данных"}{" "}
                      {name === "score" ? run.score?.unit : "°C"}
                    </dd>
                  </div>
                ))}
              </dl>
              <p>Цвет поля агрегирован. Эти значения взяты из точного среза.</p>
            </aside>
          </div>
          <div className="line-grid">
            {["raw", "processed_temperature_c", "difference", "score"]
              .filter((name) => s.layers[name])
              .map((name) => (
                <LineChart
                  key={name}
                  title={`${labels[name]} · временной ряд точки`}
                  x={s.time_s}
                  y={s.layers[name].trend}
                  xLabel="Время от начала, с"
                  yLabel={name === "score" ? run.score?.unit : "°C"}
                  yLimits={
                    name === "difference" || name === "score"
                      ? run.scales?.[name]
                      : undefined
                  }
                  threshold={
                    name === "score" ? run.score?.threshold : undefined
                  }
                  thresholdMode={
                    run.score?.comparison === "abs_ge" ? "abs_ge" : "ge"
                  }
                  syncKey={`analysis-t-${run.id}`}
                />
              ))}
          </div>
          {other.data?.layers.processed_temperature_c && (
            <LineChart
              title={`Сравнение: ${comparison!.method} / ${comparison!.method_version}`}
              x={other.data.x_m}
              y={other.data.layers.processed_temperature_c.profile}
              xLabel="Расстояние, м"
              yLimits={tempScale}
              syncKey={`analysis-x-${run.id}`}
            />
          )}
          {other.data && !other.data.layers.processed_temperature_c && (
            <p>
              Во втором запуске температурная обработка не передана; исходник не
              подставляется вместо неё.
            </p>
          )}
          {selected && (
            <>
              <h2>Выбранная зона: {selected.id}</h2>
              <p>
                Медиана и p10–p90 по валидным пространственным точкам зоны; это
                разброс измерений, не доверительный интервал модели.
              </p>
              <div className="line-grid">
                {["raw", "processed_temperature_c"]
                  .filter((name) => s.layers[name]?.zone)
                  .flatMap((name) =>
                    ["p10", "median", "p90"].map((q) => (
                      <LineChart
                        key={name + q}
                        title={`${labels[name]} · ${q}`}
                        x={s.time_s}
                        y={s.layers[name].zone![q as "p10" | "median" | "p90"]}
                        xLabel="Время от начала, с"
                        syncKey={`analysis-t-${run.id}`}
                      />
                    )),
                  )}
              </div>
            </>
          )}
          <h2>Покрытие</h2>
          <div className="line-grid">
            {Object.entries(s.layers)
              .filter(([name]) => name !== "difference")
              .map(([name, data]) => (
                <LineChart
                  key={name}
                  title={`${labels[name]} · валидные наблюдения`}
                  x={s.time_s}
                  y={data.coverage.map((v) => (v === null ? null : v * 100))}
                  xLabel="Время от начала, с"
                  yLabel="%"
                  yLimits={[0, 100]}
                  syncKey={`analysis-t-${run.id}`}
                />
              ))}
          </div>
          <NumericValues
            key={run.id + ":" + row + ":" + JSON.stringify(boundaries)}
            slices={s}
          />
          <TextInput
            id="analysis-title"
            labelText="Название кейса"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
          />
          <TextInput
            id="analysis-notes"
            labelText="Наблюдение и ограничения"
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
          />
          {(boundaries.row_end - boundaries.row_start) *
            (boundaries.column_end - boundaries.column_start) >
          1000000 ? (
            <InlineNotification
              kind="warning"
              title="Уменьшите окно для кейса"
              subtitle="Кейс принимает до 1 000 000 исходных ячеек и снимок до 4 MiB. Данные не усекаются. Исследование большого окна остаётся доступным."
              hideCloseButton
            />
          ) : (
            <CaseForm
              region={boundaries}
              profile={row}
              sha={run.source_sha256}
              runId={run.id}
              title={title}
              notes={notes}
              analysisColumn={column}
              analysisZoneId={zone || undefined}
              analysisLayer={layer}
              onSaved={onSavedCase}
            />
          )}
        </>
      )}
      {!run.layers.includes("processed_temperature_c") && (
        <p>
          Обработка температуры не передана. Score и зоны не заменяют
          температурный результат.
        </p>
      )}
      {!run.layers.includes("score") && <p>Оценка аномальности не передана.</p>}
      {!!external.data?.length && (
        <details>
          <summary>Внешние материалы просмотра: {external.data.length}</summary>
          <p>
            Исходник и фрагменты сверены. Авторство из файла заявлено; это не
            подтверждённые ответы серверного журнала. Для рабочего заключения
            автор сохраняет ответ через свою учётную запись.
          </p>
          {external.data.map((material) => (
            <article key={material.id}>
              <h3>
                {material.review_id} · {material.declared_reviewer}
              </h3>
              <p>{material.notes}</p>
              <p className="mono">Происхождение: {material.source_commit}</p>
            </article>
          ))}
        </details>
      )}
      <Link
        to={`/research?${new URLSearchParams({ record: r.record_id, row: String(row), column: String(column), r0: String(boundaries.row_start), r1: String(boundaries.row_end), c0: String(boundaries.column_start), c1: String(boundaries.column_end), source_sha256: run.source_sha256 })}`}
      >
        Открыть исходное исследование
      </Link>
    </section>
  );
}
function WindowEditor({
  region,
  run,
  onApply,
}: {
  region: Region;
  run: AnalysisRun;
  onApply: (r: Region) => void;
}) {
  const [draft, setDraft] = useState(
    [
      region.row_start,
      region.row_end,
      region.column_start,
      region.column_end,
    ].map(String),
  );
  const [error, setError] = useState("");
  return (
    <section aria-label="Собственное окно">
      <h2>Собственное окно</h2>
      <p>
        Исходные индексы. Начало включено, конец исключён. Изменения применяются
        вместе.
      </p>
      <div className="analysis-controls">
        {[
          "Начальная строка",
          "Конечная строка",
          "Начальный столбец",
          "Конечный столбец",
        ].map((label, i) => (
          <TextInput
            key={i}
            id={`analysis-bound-${i}`}
            labelText={label}
            inputMode="numeric"
            value={draft[i]}
            onChange={(e) =>
              setDraft(draft.map((v, j) => (j === i ? e.target.value : v)))
            }
          />
        ))}
      </div>
      {error && (
        <InlineNotification
          kind="error"
          title="Окно не применено"
          subtitle={error}
          hideCloseButton
        />
      )}
      <Button
        size="sm"
        onClick={() => {
          const p = new URLSearchParams();
          ["r0", "r1", "c0", "c1"].forEach((k, i) => p.set(k, draft[i]));
          const parsed = parseAnalysisContext(p, run);
          if (!parsed.ok) {
            setError(parsed.message);
            return;
          }
          setError("");
          onApply(parsed.context.region);
        }}
      >
        Применить окно
      </Button>{" "}
      <Button size="sm" kind="tertiary" onClick={() => onApply(run.region)}>
        Всё покрытие расчёта
      </Button>
    </section>
  );
}
function NumericValues({ slices: s }: { slices: Slices }) {
  const [page, setPage] = useState(0);
  const start = page * 100,
    names = Object.keys(s.layers),
    coordinates = s.x_m.slice(start, start + 100);
  return (
    <details>
      <summary>Точные значения выбранного профиля</summary>
      <p>
        Точки {start + 1}–{start + coordinates.length} из {s.x_m.length}.
        Пропуски не заменены нулями.
      </p>
      <div className="analysis-table analysis-desktop-values">
        <table>
          <thead>
            <tr>
              <th>Расстояние, м</th>
              {names.map((name) => (
                <th key={name}>{labels[name]}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {coordinates.map((x, i) => (
              <tr key={i}>
                <td>{x}</td>
                {names.map((name) => (
                  <td key={name}>
                    {s.layers[name].profile[start + i] ?? "Нет данных"}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="analysis-mobile-values">
        {coordinates.map((x, i) => (
          <dl key={i}>
            <dt>Расстояние, м</dt>
            <dd>{x}</dd>
            {names.map((name) => (
              <div key={name}>
                <dt>{labels[name]}</dt>
                <dd>{s.layers[name].profile[start + i] ?? "Нет данных"}</dd>
              </div>
            ))}
          </dl>
        ))}
      </div>
      <Button
        size="sm"
        kind="tertiary"
        disabled={page === 0}
        onClick={() => setPage(page - 1)}
      >
        Предыдущие точки
      </Button>
      <Button
        size="sm"
        kind="tertiary"
        disabled={start + 100 >= s.x_m.length}
        onClick={() => setPage(page + 1)}
      >
        Следующие точки
      </Button>
    </details>
  );
}
export default function Analysis() {
  const [params, setParams] = useSearchParams();
  const [record, setRecord] = useState(params.get("record_id") ?? ""),
    [method, setMethod] = useState(""),
    [version, setVersion] = useState(""),
    [status, setStatus] = useState("");
  const catalog = useQuery({
    queryKey: ["analysis-catalog", record, method, version, status],
    queryFn: ({ signal }) =>
      control<AnalysisRun[]>(
        `/result-runs?${new URLSearchParams({ record_id: record, method, method_version: version, research_status: status })}`,
        undefined,
        signal,
      ),
    retry: false,
  });
  const id = params.get("run_id") ?? "",
    compare = params.get("compare") ?? "";
  const detail = useQuery({
    queryKey: ["analysis-run", id],
    queryFn: ({ signal }) =>
      control<AnalysisRun>(`/result-runs/${id}`, undefined, signal),
    enabled: !!id,
    retry: false,
  });
  const other = useQuery({
    queryKey: ["analysis-run", compare],
    queryFn: ({ signal }) =>
      control<AnalysisRun>(`/result-runs/${compare}`, undefined, signal),
    enabled: !!compare,
    retry: false,
  });
  const compatible =
    !other.data ||
    !detail.data ||
    (other.data.source_sha256 === detail.data.source_sha256 &&
      (() => {
        const parsed = parseAnalysisContext(params, detail.data);
        if (!parsed.ok) return false;
        const r = parsed.context.region,
          b = other.data.region;
        return (
          b.record_id === r.record_id &&
          b.row_start <= r.row_start &&
          r.row_end <= b.row_end &&
          b.column_start <= r.column_start &&
          r.column_end <= b.column_end
        );
      })());
  return (
    <main className="page">
      <h1>Результаты анализа реальных данных</h1>
      <p>
        <Link to="/experiments">
          Пороги, сравнения методов и ограничения ML-экспериментов →
        </Link>
      </p>
      <p>
        Выбирайте сохранённый расчёт. Графики описывают сигнал; качество
        обнаружения без независимого эталона не установлено.
      </p>
      <div className="analysis-controls">
        {[
          ["Запись", record, setRecord],
          ["Метод", method, setMethod],
          ["Версия", version, setVersion],
          ["Статус", status, setStatus],
        ].map(([label, value, setter], i) => (
          <TextInput
            key={i}
            id={`analysis-filter-${i}`}
            labelText={label as string}
            value={value as string}
            onChange={(e) => (setter as (v: string) => void)(e.target.value)}
          />
        ))}
      </div>
      <Select
        id="analysis-run"
        labelText="Выбранный запуск"
        value={id}
        onChange={(e) => setParams({ run_id: e.target.value })}
      >
        <SelectItem value="" text="Выберите расчёт" />
        {catalog.data?.map((run) => (
          <SelectItem
            key={run.id}
            value={run.id}
            text={`${run.region.record_id} · ${run.method} · ${run.method_version} · ${run.id.slice(0, 8)}`}
          />
        ))}
      </Select>
      {detail.data && (
        <Select
          id="analysis-compare"
          labelText="Второй совместимый запуск"
          value={compare}
          onChange={(e) =>
            setParams(() => {
              const next = new URLSearchParams(params);
              if (e.target.value) next.set("compare", e.target.value);
              else next.delete("compare");
              return next;
            })
          }
        >
          <SelectItem value="" text="Без сравнения" />
          {catalog.data
            ?.filter(
              (run) =>
                run.id !== id &&
                run.source_sha256 === detail.data.source_sha256 &&
                JSON.stringify(run.region) ===
                  JSON.stringify(detail.data.region),
            )
            .map((run) => (
              <SelectItem
                key={run.id}
                value={run.id}
                text={`${run.method} / ${run.method_version}`}
              />
            ))}
        </Select>
      )}
      {(catalog.error || detail.error || other.error) && (
        <InlineNotification
          kind="error"
          title="Результат недоступен"
          subtitle={String(catalog.error ?? detail.error ?? other.error)}
          hideCloseButton
        />
      )}
      {catalog.data?.length === 0 && (
        <p>
          Пакеты полей ещё не импортированы. Одиночные профили доступны в
          разделе «Запуски».
        </p>
      )}
      {!compatible && (
        <InlineNotification
          kind="warning"
          title="Второй запуск несовместим"
          subtitle="Источник или покрытие отличаются. Сравнение не показано; основной контекст сохранён."
          hideCloseButton
        />
      )}
      {detail.data && (
        <RunView
          key={detail.data.id}
          run={detail.data}
          comparison={compatible ? other.data : undefined}
        />
      )}
    </main>
  );
}
