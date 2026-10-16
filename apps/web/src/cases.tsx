import { ThemeSwitch, chartColors, useAppTheme } from "./theme";
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Button,
  Checkbox,
  InlineNotification,
  Loading,
  Tag,
} from "@carbon/react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { control, useAccount } from "./access";
import { api, ApiError, type Queue } from "./api";
import { profileChanges, type ProfileChanges } from "./profile-changes";
import { LineChart } from "./charts";
import { palette } from "./palette";
import { analysisParams } from "./analysis-context";
import type { components } from "./api-schema";

type Region = components["schemas"]["Region"];
type Request = components["schemas"]["EvidenceCaseRequest"];
type Summary = components["schemas"]["EvidenceCaseSummary"];
type Evidence = {
  source: string;
  source_sha256: string;
  scale: [number, number];
  time_kind: string;
  mask_counts: Record<string, number>;
  valid_fraction: number;
  region: Region;
  timestamps: string[];
  preview: number[][];
  preview_valid: boolean[][];
  preview_x_cells: [number, number][];
  preview_time_cells: [number, number][];
  profile: {
    row: number;
    x_m: number[];
    temperature_c: (number | null)[];
    processed_temperature_c?: (number | null)[];
  };
  processing_changes?: ProfileChanges;
  analysis_view?: {
    run_id: string;
    source_sha256: string;
    region: Region;
    row: number;
    column: number;
    layer: string;
    zone_id?: string | null;
  };
  analysis?: {
    run_id: string;
    column?: number;
    zone_id?: string | null;
    time_s?: number[];
    raw_trend?: (number | null)[];
    distance_m?: number;
    score?: { unit: string; meaning: string } | null;
    scales?: Record<string, [number, number]>;
    layers: Record<
      string,
      {
        profile: (number | null)[];
        trend?: (number | null)[];
        preview: number[][];
        preview_valid: boolean[][];
      }
    >;
  };
  expert_answer: {
    reviewer: string;
    notes: string;
    revision: number;
    temperature_episode: string;
  } | null;
};
type Detail = Summary & {
  result: {
    evidence: Evidence;
    run: components["schemas"]["RunDetail"] | null;
  };
  source_available: boolean;
  research_available: boolean;
};

export function CaseForm({
  region,
  profile,
  sha,
  runId,
  title,
  notes,
  proposal,
  analysisColumn,
  analysisZoneId,
  analysisLayer,
  onSaved,
}: {
  region: Region;
  profile: number;
  sha: string;
  runId?: string;
  title: string;
  notes: string;
  proposal?: string;
  analysisColumn?: number;
  analysisZoneId?: string;
  analysisLayer?: string;
  onSaved?: (id:string)=>void;
}) {
  const account = useAccount();
  const [include, setInclude] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [saved, setSaved] = useState("");
  const [pending, setPending] = useState<{ body: Request; key: string }>();
  const pendingRef = useRef<typeof pending>(undefined);
  const sendingRef = useRef(false);
  const [preview, setPreview] = useState<{
    key: string;
    value: { method: string | null; expert_answer: Evidence["expert_answer"] };
  }>();
  const errorRef = useRef<HTMLDivElement>(null);
  const packageId = proposal?.split(":")[0] ?? "general";
  const expert = account && ["expert", "admin"].includes(account.role);
  const queue = useQuery({
    queryKey: ["case-own-queue", packageId, account?.id],
    queryFn: ({ signal }) =>
      api<Queue>(
        `/queue?${new URLSearchParams({ package: packageId, independent: "true", reviewer: account!.display_name })}`,
        signal,
      ),
    enabled: !!expert,
    retry: false,
  });
  const card = queue.data?.cards.find(
    (c) =>
      c.source_sha256 === sha &&
      c.source.replace(/\.csv$/, "") === region.record_id &&
      c.row_bounds[0] === region.row_start &&
      c.row_bounds[1] === region.row_end &&
      c.column_bounds[0] === region.column_start &&
      c.column_bounds[1] === region.column_end,
  );
  const annotation = card ? queue.data?.annotations[card.review_id] : undefined;
  const body: Request = {
    title,
    notes,
    region,
    profile,
    source_sha256: sha,
    run_id: runId ?? null,
    ...(analysisColumn === undefined
      ? {}
      : { analysis_column: analysisColumn }),
    ...(analysisZoneId === undefined
      ? {}
      : { analysis_zone_id: analysisZoneId }),
    ...(analysisLayer === undefined ? {} : { analysis_layer: analysisLayer }),
    review:
      include && expert && card && queue.data
        ? {
            package: packageId,
            review_id: card.review_id,
            queue_sha256: queue.data.queue_sha256,
            expected_revision: annotation?.revision ?? 0,
          }
        : null,
  };
  const key = JSON.stringify(body);
  async function execute(save: boolean) {
    if (sendingRef.current || saved) return;
    sendingRef.current = true;
    setBusy(true);
    setError("");
    try {
      if (save) {
        const attempt = pendingRef.current ?? {
          body: structuredClone(body),
          key: crypto.randomUUID(),
        };
        pendingRef.current = attempt;
        setPending(attempt);
        const result = await control<Detail>(
          "/cases",
          attempt.body,
          undefined,
          attempt.key,
        );
        setSaved(result.id);
        onSaved?.(result.id);
        pendingRef.current = undefined;
        setPending(undefined);
      } else {
        const value = await control<{
          method: string | null;
          expert_answer: Evidence["expert_answer"];
        }>("/cases/preview", body);
        setPreview({ key, value });
      }
    } catch (e) {
      setError(String(e));
      if (
        !save ||
        (e instanceof ApiError && e.status >= 400 && e.status < 500)
      ) {
        pendingRef.current = undefined;
        setPending(undefined);
        setPreview(undefined);
        if (expert) await queue.refetch();
      }
      requestAnimationFrame(() => errorRef.current?.focus());
    } finally {
      sendingRef.current = false;
      setBusy(false);
    }
  }
  if (!account || account.role==='viewer') return null;
  return (
    <section className="case-form" aria-labelledby="case-form-title">
      <h3 id="case-form-title">Зафиксировать кейс</h3>
      <p>
        Сохраняется неизменяемый снимок окна, выбранной обработки и выбранного
        ответа. Новые расчёты и правки журнала не изменят кейс.
      </p>
      <dl className="case-facts">
        <div>
          <dt>Источник</dt>
          <dd>
            {region.record_id}.csv · <code>{sha}</code>
          </dd>
        </div>
        <div>
          <dt>Окно и профиль</dt>
          <dd>
            Строки [{region.row_start}, {region.row_end}), координаты [
            {region.column_start}, {region.column_end}); профиль {profile}
          </dd>
        </div>
        <div>
          <dt>Расчёт</dt>
          <dd>{runId ?? "Обработка не выбрана"}</dd>
        </div>
        {analysisLayer && (
          <div>
            <dt>Контекст просмотра</dt>
            <dd>
              Слой: {analysisLayer}; точка {analysisColumn};{" "}
              {analysisZoneId
                ? `зона ${analysisZoneId}`
                : "собственное окно или полное покрытие"}
              . Он будет сохранён для возврата из кейса.
            </dd>
          </div>
        )}
      </dl>
      {expert && (
        <Checkbox
          id="case-include-answer"
          labelText="Включить мой сохранённый ответ для этого окна"
          checked={include}
          disabled={busy || !!pending || !!saved}
          onChange={(_, { checked }) => {
            setInclude(checked);
            setPreview(undefined);
          }}
        />
      )}
      {queue.error && include && !pending && !saved && (
        <p role="alert">
          Не удалось проверить свой ответ. Повторите загрузку либо отключите его
          включение.
        </p>
      )}
      {preview?.key === key && (
        <div className="case-preview" role="status">
          <p>Метод: {preview.value.method ?? "Обработка не выбрана"}</p>
          {preview.value.expert_answer ? (
            <>
              <p>
                Автор: {preview.value.expert_answer.reviewer} · ревизия{" "}
                {preview.value.expert_answer.revision}
              </p>
              <p>
                {preview.value.expert_answer.temperature_episode} ·{" "}
                {preview.value.expert_answer.notes}
              </p>
            </>
          ) : (
            <p>Эксперт ещё не проверил — ответ в кейс не включён.</p>
          )}
          <p>Метрики качества и обучающие метки не создаются.</p>
        </div>
      )}
      {error && (
        <div ref={errorRef} tabIndex={-1} role="alert">
          <InlineNotification
            kind="error"
            title={
              pending
                ? "Подтверждение сохранения не получено"
                : "Кейс не сохранён"
            }
            subtitle={error}
            hideCloseButton
          />
        </div>
      )}
      <div className="action-row">
        <Button
          kind="tertiary"
          disabled={
            busy ||
            !!pending ||
            !!saved ||
            !title.trim() ||
            (!!expert && include && (queue.isPending || !!queue.error))
          }
          onClick={() => void execute(false)}
        >
          Проверить состав кейса
        </Button>
        <Button
          disabled={busy || !!pending || !!saved || preview?.key !== key}
          onClick={() => void execute(true)}
        >
          {busy ? "Обрабатываем…" : "Зафиксировать кейс"}
        </Button>
        {expert && queue.error && (
          <Button kind="ghost" onClick={() => void queue.refetch()}>
            Повторить загрузку ответа
          </Button>
        )}
      </div>
      {pending && !busy && (
        <div className="case-pending" role="status">
          <p>
            Сервер мог сохранить кейс, но подтверждение не получено. Повтор
            использует те же параметры и ключ и не создаёт второй кейс. До смены
            участка или обновления страницы проверьте результат.
          </p>
          <p>
            Отправленная попытка: {pending.body.title} · профиль{" "}
            {pending.body.profile} · расчёт {pending.body.run_id ?? "не выбран"}
            . {pending.body.notes}
          </p>
          <Button onClick={() => void execute(true)}>
            Повторить сохранение
          </Button>
        </div>
      )}
      {saved && (
        <p role="status">
          <Link to={`/cases/${saved}`}>Кейс сохранён — открыть →</Link>
        </p>
      )}
    </section>
  );
}

function SnapshotField({
  evidence: e,
  unit = "°C",
  diverging = false,
}: {
  evidence: Evidence;
  unit?: string;
  diverging?: boolean;
}) {
  const { theme } = useAppTheme();
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current!;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const xs = e.preview_x_cells[0][0],
      xe = e.preview_x_cells.at(-1)![1];
    const ts = e.preview_time_cells[0][0],
      te = e.preview_time_cells.at(-1)![1];
    const [low, high] = e.scale;
    const draw = () => {
      const width = Math.max(160, canvas.clientWidth),
        height = 340;
      const dpr = Math.min(devicePixelRatio || 1, 2);
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const plotWidth = width - 65,
        plotHeight = 250,
        left = 50;
      ctx.fillStyle = chartColors().background;
      ctx.fillRect(0, 0, width, height);
      e.preview.forEach((row, r) =>
        row.forEach((value, c) => {
          const color =
            Math.round(
              Math.max(0, Math.min(1, (value - low) / (high - low || 1))) * 255,
            ) * 4;
          const fraction = Math.max(
            0,
            Math.min(1, (value - low) / (high - low || 1)),
          );
          const signedColor =
            fraction < 0.5
              ? `rgb(${Math.round(255 * fraction * 2)},${Math.round(255 * fraction * 2)},255)`
              : `rgb(255,${Math.round(255 * (1 - fraction) * 2)},${Math.round(255 * (1 - fraction) * 2)})`;
          ctx.fillStyle = e.preview_valid[r][c]
            ? diverging
              ? signedColor
              : `rgb(${palette[color]},${palette[color + 1]},${palette[color + 2]})`
            : "#8d8d8d";
          const [x0, x1] = e.preview_x_cells[c],
            [t0, t1] = e.preview_time_cells[r];
          ctx.fillRect(
            left + ((x0 - xs) / (xe - xs)) * plotWidth,
            20 + ((t0 - ts) / (te - ts)) * plotHeight,
            ((x1 - x0) / (xe - xs)) * plotWidth,
            ((t1 - t0) / (te - ts)) * plotHeight,
          );
        }),
      );
      ctx.fillStyle = chartColors().axis;
      ctx.font = "12px sans-serif";
      ctx.fillText(`${ts.toFixed(0)} с`, 0, 30);
      ctx.fillText(`${te.toFixed(0)} с`, 0, 270);
      ctx.fillText(`${xs.toFixed(2)} м`, left, 295);
      ctx.textAlign = "right";
      ctx.fillText(`${xe.toFixed(2)} м`, width - 15, 295);
      ctx.textAlign = "left";
      ctx.fillText(
        `Шкала: ${low.toFixed(2)}–${high.toFixed(2)} ${unit} · ${diverging ? "знаковая" : "Cividis"}`,
        left,
        325,
      );
    };
    const observer = new ResizeObserver(draw);
    observer.observe(canvas);
    draw();
    return () => observer.disconnect();
  }, [e, unit, diverging, theme]);
  return (
    <figure>
      <canvas
        ref={ref}
        width={1000}
        height={400}
        className="case-field"
        role="img"
        aria-label="Сохранённый обзор температурного окна с физическими координатами"
      />
      <figcaption>
        Обзор прорежен по индексам; это не полное поле. Разрывы времени не
        заполнены. Серый — невалидное измерение. Точные значения профиля
        доступны ниже.
      </figcaption>
    </figure>
  );
}

const caseSteps = [
  "Измерение",
  "Наблюдение",
  "Обработка",
  "Экспертная проверка",
  "Передача пакета",
];

function ProfileComparison({
  c,
  comparison = true,
}: {
  c: Detail;
  comparison?: boolean;
}) {
  const e = c.result.evidence,
    p = e.profile;
  const values = [
    ...p.temperature_c,
    ...(comparison ? (p.processed_temperature_c ?? []) : []),
  ].filter((v): v is number => v !== null);
  const scale: [number, number] | undefined = values.length
    ? [Math.min(...values) - 0.1, Math.max(...values) + 0.1]
    : undefined;
  const delta = p.processed_temperature_c?.map((v, i) =>
    v !== null && p.temperature_c[i] !== null ? v - p.temperature_c[i]! : null,
  );
  const extent = Math.max(
    0.1,
    ...(delta ?? []).filter((v): v is number => v !== null).map(Math.abs),
  );
  return (
    <div className="chart-grid">
      <LineChart
        title="Исходный профиль · °C"
        x={p.x_m}
        y={p.temperature_c}
        xLabel="Расстояние, м"
        yLimits={scale}
        syncKey={c.id}
      />
      {comparison && p.processed_temperature_c && (
        <>
          <LineChart
            title="Выбранная обработка · °C"
            x={p.x_m}
            y={p.processed_temperature_c}
            xLabel="Расстояние, м"
            yLimits={scale}
            syncKey={c.id}
          />
          <LineChart
            title="Обработка − исходный сигнал · °C"
            x={p.x_m}
            y={delta!}
            xLabel="Расстояние, м"
            yLimits={[-extent, extent]}
            syncKey={c.id}
          />
        </>
      )}
      {comparison &&
        e.analysis &&
        Object.entries(e.analysis.layers)
          .filter(
            ([name]) => name === "score" || name === "background_temperature_c",
          )
          .map(([name, data]) => (
            <LineChart
              key={name}
              title={
                name === "score"
                  ? "Сохранённая оценка аномальности"
                  : "Сохранённый фон"
              }
              x={p.x_m}
              y={data.profile}
              xLabel="Расстояние, м"
              yLabel={name === "score" ? "score" : "°C"}
              syncKey={c.id}
            />
          ))}
      {comparison && e.analysis && (
        <details className="case-analysis-fields">
          <summary>Сохранённые поля обработки и score</summary>
          <p>
            Подвыборка исходных ячеек. Точные профили сохранены в JSON; полнота
            поля не заявляется.
          </p>
          {Object.entries(e.analysis.layers).map(([name, data]) => (
            <section key={name}>
              <h3>{name}</h3>
              <SnapshotField
                evidence={{
                  ...e,
                  preview: data.preview,
                  preview_valid: data.preview_valid,
                  scale:
                    name === "score" || name === "difference"
                      ? (e.analysis?.scales?.[name] ?? e.scale)
                      : e.scale,
                }}
                unit={
                  name === "score" ? (e.analysis?.score?.unit ?? "score") : "°C"
                }
                diverging={
                  (name === "score" || name === "difference") &&
                  (e.analysis?.scales?.[name]?.[0] ?? 0) < 0
                }
              />
            </section>
          ))}
        </details>
      )}
      {comparison && e.analysis?.time_s && e.analysis.raw_trend && (
        <>
          <LineChart
            title={`Сохранённый временной ряд точки ${e.analysis.distance_m} м`}
            x={e.analysis.time_s}
            y={e.analysis.raw_trend}
            xLabel="Время от начала, с"
            syncKey={`case-time-${c.id}`}
          />
          {Object.entries(e.analysis.layers)
            .filter(([, data]) => data.trend)
            .map(([name, data]) => (
              <LineChart
                key={name}
                title={`${name} · сохранённый временной ряд`}
                x={e.analysis!.time_s!}
                y={data.trend!}
                xLabel="Время от начала, с"
                yLabel={
                  name === "score" ? (e.analysis?.score?.unit ?? "score") : "°C"
                }
                syncKey={`case-time-${c.id}`}
              />
            ))}
        </>
      )}
    </div>
  );
}

function ProcessingChanges({ e }: { e: Evidence }) {
  const changes = e.processing_changes ?? profileChanges(e.profile);
  if (!changes) return null;
  return (
    <section className="processing-changes" aria-label="Что изменила обработка">
      <h3>Что изменила обработка</h3>
      {changes.paired_points ? (
        <>
          <dl className="case-facts">
            <div>
              <dt>Сопоставимые точки</dt>
              <dd>
                {changes.paired_points} из {changes.total_points}
              </dd>
            </div>
            <div>
              <dt>Изменились больше 0,01 °C</dt>
              <dd>{changes.changed_points}</dd>
            </div>
            <div>
              <dt>Максимальное абсолютное изменение, °C</dt>
              <dd>{changes.max_absolute_change_c}</dd>
            </div>
            <div>
              <dt>Среднее абсолютное изменение, °C</dt>
              <dd>{changes.mean_absolute_change_c}</dd>
            </div>
          </dl>
          <div className="case-change-table">
            <table>
              <caption>До пяти наибольших изменений</caption>
              <thead>
                <tr>
                  <th>Расстояние, м</th>
                  <th>Исходник, °C</th>
                  <th>Обработка, °C</th>
                  <th>Разность, °C</th>
                </tr>
              </thead>
              <tbody>
                {changes.top_points.map((p) => (
                  <tr key={p.index}>
                    <td>{p.x_m}</td>
                    <td>{p.original_c}</td>
                    <td>{p.processed_c}</td>
                    <td>{p.delta_c}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div
            className="case-change-cards"
            aria-label="До пяти наибольших изменений"
          >
            {changes.top_points.map((p) => (
              <dl key={p.index}>
                <dt>Расстояние, м</dt>
                <dd>{p.x_m}</dd>
                <dt>Исходник, °C</dt>
                <dd>{p.original_c}</dd>
                <dt>Обработка, °C</dt>
                <dd>{p.processed_c}</dd>
                <dt>Разность, °C</dt>
                <dd>{p.delta_c}</dd>
              </dl>
            ))}
          </div>
        </>
      ) : (
        <p>Сопоставимых точек нет.</p>
      )}
      <p>
        Пропуски исходника: {changes.missing_original}; пропуски обработки среди
        валидных исходных точек: {changes.missing_processed}.
      </p>
      <p>
        Порог 0,01 °C — правило отображения, не точность датчика. Изменения
        сигнала не подтверждают качество очистки.
      </p>
      {!e.processing_changes && (
        <p>
          Показатели вычислены из старого снимка только для интерфейса. Его SHA
          и автономный пакет не изменяются.
        </p>
      )}
    </section>
  );
}

function ExpertSnapshot({ e }: { e: Evidence }) {
  return (
    <section>
      <h2>Экспертная проверка</h2>
      {e.expert_answer ? (
        <>
          <p>
            {e.expert_answer.reviewer} · ревизия {e.expert_answer.revision} ·{" "}
            {e.expert_answer.temperature_episode}
          </p>
          <p>{e.expert_answer.notes}</p>
        </>
      ) : (
        <p>Эксперт ещё не проверил.</p>
      )}
      <p>
        Ответ относится к сохранённому окну. Физическая причина и точность
        модели не установлены; обучающие метки не создаются.
      </p>
    </section>
  );
}

function ProfileValues({ e }: { e: Evidence }) {
  return (
    <details>
      <summary>Точные значения профиля</summary>
      <div className="case-table">
        <table>
          <thead>
            <tr>
              <th>Расстояние, м</th>
              <th>Исходник, °C</th>
              <th>Обработка, °C</th>
            </tr>
          </thead>
          <tbody>
            {e.profile.x_m.map((x, i) => (
              <tr key={i}>
                <td>{x}</td>
                <td>{e.profile.temperature_c[i] ?? "Пропуск"}</td>
                <td>
                  {e.profile.processed_temperature_c
                    ? (e.profile.processed_temperature_c[i] ?? "Пропуск")
                    : "Не выбрана"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  );
}

function CasePresentation({
  c,
  link,
  error,
  busy,
  download,
  reload,
}: {
  c: Detail;
  link: string;
  error: string;
  busy: boolean;
  download: () => Promise<void>;
  reload: () => void;
}) {
  const [params, setParams] = useSearchParams();
  const rawStep = params.get("step") ?? "0";
  const step = /^\d+$/.test(rawStep) ? Math.min(4, Number(rawStep)) : 0;
  const e = c.result.evidence;
  const root = useRef<HTMLElement>(null),
    heading = useRef<HTMLHeadingElement>(null);
  const [fullscreen, setFullscreen] = useState(false),
    [screenError, setScreenError] = useState("");
  function go(next: number) {
    const query = new URLSearchParams(params);
    query.set("present", "1");
    query.set("step", String(Math.max(0, Math.min(4, next))));
    setParams(query);
  }
  useLayoutEffect(() => {
    const handler = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (
        event.altKey ||
        event.ctrlKey ||
        event.metaKey ||
        target?.closest(
          'input,textarea,select,[contenteditable="true"],[role="textbox"]',
        )
      )
        return;
      if (event.key === "ArrowRight" || event.key === "ArrowLeft") {
        event.preventDefault();
        const next = new URLSearchParams(params);
        next.set("present", "1");
        next.set(
          "step",
          String(
            Math.max(
              0,
              Math.min(4, step + (event.key === "ArrowRight" ? 1 : -1)),
            ),
          ),
        );
        setParams(next);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [params, setParams, step]);
  useLayoutEffect(() => {
    heading.current?.focus();
  }, [step]);
  useEffect(() => {
    const changed = () =>
      setFullscreen(document.fullscreenElement === root.current);
    document.addEventListener("fullscreenchange", changed);
    return () => document.removeEventListener("fullscreenchange", changed);
  }, []);
  async function toggleFullscreen() {
    setScreenError("");
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else {
        if (!root.current?.requestFullscreen) throw new Error("Unsupported");
        await root.current.requestFullscreen();
      }
    } catch {
      setScreenError(
        "Полноэкранный режим недоступен. Показ можно продолжить в обычном окне.",
      );
    }
  }
  return (
    <main
      ref={root}
      className="page case-page case-presentation"
      aria-label="Показ сохранённого кейса"
    >
      <div className="section-title">
        <div>
          <Tag type="blue">Неизменяемый снимок</Tag>
          <h1>{c.title}</h1>
        </div>
        <Link to={`/cases/${c.id}`}>Вернуться к кейсу →</Link>
      </div>
      <div className="action-row">
        <Button kind="tertiary" onClick={() => void toggleFullscreen()}>
          {fullscreen ? "Выйти из полного экрана" : "На весь экран"}
        </Button>
        <ThemeSwitch />
        <Button kind="ghost" onClick={reload}>
          Повторить загрузку
        </Button>
        {c.research_available && <Link to={link}>Открыть исследование →</Link>}
      </div>
      {(error || screenError) && (
        <InlineNotification
          kind="error"
          title="Показ можно продолжить"
          subtitle={error || screenError}
          hideCloseButton
        />
      )}
      <nav aria-label="Шаги защиты">
        <ol className="case-steps">
          {caseSteps.map((title, i) => (
            <li key={title}>
              <button
                type="button"
                aria-current={i === step ? "step" : undefined}
                onClick={() => go(i)}
              >
                {i + 1}. {title}
              </button>
            </li>
          ))}
        </ol>
      </nav>
      <section className="case-slide" aria-labelledby="case-step-heading">
        <p className="case-step-count">Шаг {step + 1} из 5</p>
        <h2 id="case-step-heading" ref={heading} tabIndex={-1}>
          {caseSteps[step]}
        </h2>
        {step === 0 && (
          <>
            <p>
              Распределённое измерение температуры вдоль волокна. Показ
              использует сохранённые значения и маски, а не текущий расчёт.
            </p>
            <dl className="case-facts">
              <div>
                <dt>Источник / SHA</dt>
                <dd>
                  {e.source}
                  <br />
                  <code>{c.source_sha256}</code>
                </dd>
              </div>
              <div>
                <dt>Окно и профиль</dt>
                <dd>
                  Строки [{c.region.row_start}, {c.region.row_end}), координаты
                  [{c.region.column_start}, {c.region.column_end}); профиль{" "}
                  {c.profile}
                </dd>
              </div>
              <div>
                <dt>Время измерения</dt>
                <dd>
                  {e.timestamps.join(" → ")}
                  <br />
                  {e.time_kind}; неизвестный часовой пояс не добавляется.
                </dd>
              </div>
              <div>
                <dt>Валидные наблюдения окна</dt>
                <dd>{(e.valid_fraction * 100).toFixed(1)} %</dd>
              </div>
            </dl>
            <details>
              <summary>Причины невалидности и происхождение</summary>
              <pre>{JSON.stringify(e.mask_counts, null, 2)}</pre>
              <p>
                Кейс {c.id} · снимок <code>{c.snapshot_sha256}</code>. Пропуски
                не превращаются в температуру.
              </p>
            </details>
            <SnapshotField evidence={e} />
          </>
        )}
        {step === 1 && (
          <>
            <p>{c.notes || "Описание наблюдения в этом кейсе не задано."}</p>
            <p>
              Это описание автора кейса. Наличие изменения сигнала не
              устанавливает его физическую причину.
            </p>
            <SnapshotField evidence={e} />
            <ProfileComparison c={c} comparison={false} />
            <ProfileValues e={e} />
          </>
        )}
        {step === 2 && (
          <>
            {e.profile.processed_temperature_c ? (
              <>
                <p>
                  {c.method} · {c.result.run?.method_version} · запуск{" "}
                  <code>{c.run_id}</code>
                </p>
                <p>
                  Происхождение обучения: {c.training_origin}.{" "}
                  {c.method?.startsWith("median_") &&
                    "Референсный медианный фильтр; не подтверждённая финальная модель Захара."}
                </p>
                <p>{c.result.run?.limitations}</p>
                <details>
                  <summary>Параметры обработки</summary>
                  <pre>{JSON.stringify(c.result.run?.parameters, null, 2)}</pre>
                </details>
                <p>
                  Исходник и обработка используют общую шкалу; разность имеет
                  симметричную шкалу относительно нуля. Курсоры связаны.
                  Изменение значения не доказывает удаление шума.
                </p>
              </>
            ) : (
              <p>
                Обработка не выбрана. Показан исходный сигнал; разность не
                вычисляется.
              </p>
            )}
            <ProfileComparison c={c} />
            <ProcessingChanges e={e} />
            <ProfileValues e={e} />
          </>
        )}
        {step === 3 && (
          <>
            <ExpertSnapshot e={e} />
            {c.review && (
              <p>
                SHA очереди: <code>{String(c.review.queue_sha256)}</code>
              </p>
            )}
            <p>
              Показ читает зафиксированную ревизию. Последующая правка журнала
              не изменяет этот кейс.
            </p>
          </>
        )}
        {step === 4 && (
          <>
            <p>
              Передайте специалисту автономный пакет, чтобы он мог открыть
              разбор без сервера и сверить значения с доверенным подготовленным
              исходником.
            </p>
            <ul>
              <li>evidence.html — автономный разбор</li>
              <li>evidence.json — значения, маски и происхождение</li>
              {c.run_id && <li>run.json — выбранный запуск {c.run_id}</li>}
              <li>manifest.json — идентификаторы и SHA файлов</li>
              <li>README.txt — инструкция проверки</li>
            </ul>
            <p>
              Кейс <code>{c.id}</code> · снимок <code>{c.snapshot_sha256}</code>
            </p>
            <p>
              SHA подтверждают целостность относительно manifest, а не авторство
              или точность модели. Полный CSV, пароли и сессии в пакет не
              включаются.
            </p>
            <pre>
              python scripts/verify-evidence.py --record
              /path/to/prepared-record --bundle dts-case-{c.id}.zip
            </pre>
            <Button disabled={busy} onClick={() => void download()}>
              {busy ? "Готовим пакет…" : "Скачать пакет"}
            </Button>
            <p>
              Скачайте пакет до выступления: при отсутствии связи используйте
              уже сохранённый evidence.html.
            </p>
          </>
        )}
      </section>
      <div className="action-row case-slide-navigation">
        <Button
          kind="secondary"
          disabled={step === 0}
          onClick={() => go(step - 1)}
        >
          Назад
        </Button>
        <span aria-live="polite">{step + 1} / 5</span>
        <Button disabled={step === 4} onClick={() => go(step + 1)}>
          Далее
        </Button>
      </div>
      <p className="case-keyboard-hint">
        ← → переключают шаги вне полей ввода. Ссылка сохраняет текущий шаг.
      </p>
    </main>
  );
}

export default function Cases() {
  const { id } = useParams();
  const [params] = useSearchParams();
  const account = useAccount();
  const list = useQuery({
    queryKey: ["cases", account?.id],
    queryFn: ({ signal }) => control<Summary[]>("/cases", undefined, signal),
    enabled: !id,
    retry: false,
  });
  const detail = useQuery({
    queryKey: ["case", id, account?.id],
    queryFn: ({ signal }) => control<Detail>(`/cases/${id}`, undefined, signal),
    enabled: !!id,
    retry: false,
  });
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const c = detail.data,
    e = c?.result.evidence;
  const link = c
    ? c.result.evidence.analysis_view || c.result.evidence.analysis
      ? `/analysis?${analysisParams({ id: c.run_id!, source_sha256: c.source_sha256 }, { region: c.region, row: c.profile, column: c.result.evidence.analysis_view?.column ?? c.result.evidence.analysis?.column ?? c.region.column_start, layer: c.result.evidence.analysis_view?.layer ?? "raw", zone: c.result.evidence.analysis_view?.zone_id ?? c.result.evidence.analysis?.zone_id ?? "" })}`
      : `/research?${new URLSearchParams({ record: c.region.record_id, row: String(c.profile), column: String(c.region.column_start), r0: String(c.region.row_start), r1: String(c.region.row_end), c0: String(c.region.column_start), c1: String(c.region.column_end), source_sha256: c.source_sha256, ...(c.run_id ? { run_id: c.run_id } : {}) })}`
    : "";
  async function download() {
    setBusy(true);
    setError("");
    try {
      const response = await fetch(`/api/v2/cases/${id}/bundle`);
      if (response.status === 401)
        window.dispatchEvent(new Event("dts-auth-expired"));
      if (!response.ok) throw new Error((await response.json()).detail);
      const url = URL.createObjectURL(await response.blob()),
        anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `dts-case-${id}.zip`;
      anchor.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(false);
    }
  }
  if (c && params.get("present") === "1")
    return (
      <CasePresentation
        c={c}
        link={link}
        error={String(detail.error || error || "")}
        busy={busy}
        download={download}
        reload={() => void detail.refetch()}
      />
    );
  return (
    <main className="page case-page">
      <div className="section-title">
        <h1>{c?.title ?? "Проверяемые кейсы"}</h1>
        {id && <Link to="/cases">Все кейсы →</Link>}
      </div>
      {(id ? detail.isPending : list.isPending) && (
        <Loading
          description="Открываем сохранённые данные"
          withOverlay={false}
        />
      )}
      {(detail.error || list.error || error) && (
        <InlineNotification
          kind="error"
          title="Не удалось получить данные"
          subtitle={String(detail.error || list.error || error)}
          hideCloseButton
        />
      )}
      <div className="action-row">
        <Button
          kind="ghost"
          onClick={() => void (id ? detail.refetch() : list.refetch())}
        >
          Повторить загрузку
        </Button>
      </div>
      {!id && (
        <>
          <p>
            Кейсы фиксируют выбранный расчёт и экспертную ревизию. Чужой
            экспертный ответ доступен после самостоятельного ответа на то же
            окно.
          </p>
          <div className="run-list">
            {list.data?.map((item) => (
              <article key={item.id}>
                <h2>
                  <Link to={`/cases/${item.id}`}>{item.title}</Link>
                </h2>
                <p>
                  {item.region.record_id}.csv · профиль {item.profile} ·{" "}
                  {item.method ?? "Обработка не выбрана"}
                </p>
                <p>
                  {item.review
                    ? `Ответ · ревизия ${String(item.review.revision)}`
                    : "Эксперт ещё не проверил"}
                </p>
              </article>
            ))}
          </div>
          {list.data?.length === 0 && (
            <p>
              Кейсов ещё нет. Откройте участок в исследовании и зафиксируйте его
              после проверки состава.
            </p>
          )}
        </>
      )}
      {c && e && (
        <>
          <p>{c.notes}</p>
          <Tag type="blue">Неизменяемый снимок</Tag>
          <p>
            Снимок сохранён {new Date(c.created_at * 1000).toLocaleString("ru")}{" "}
            · {e.source} · профиль {c.profile}
          </p>
          <p>
            Измерение: {e.timestamps.join(" → ")} · интерпретация времени:{" "}
            {e.time_kind}. Часовой пояс не добавляется, если его нет в
            источнике.
          </p>
          <dl className="case-facts">
            <div>
              <dt>Источник / SHA</dt>
              <dd>
                <code>{c.source_sha256}</code>
              </dd>
            </div>
            <div>
              <dt>Расчёт</dt>
              <dd>{c.run_id ?? "Обработка не выбрана"}</dd>
            </div>
            <div>
              <dt>Метод / обучение</dt>
              <dd>
                {c.method ?? "Исходный сигнал"} ·{" "}
                {c.training_origin ?? "не применяется"}
              </dd>
            </div>
            <div>
              <dt>Снимок / SHA</dt>
              <dd>
                <code>{c.snapshot_sha256}</code>
              </dd>
            </div>
          </dl>
          <div className="action-row">
            <Link className="text-link" to={`/cases/${c.id}?present=1&step=0`}>
              Показать на защите →
            </Link>
            {c.research_available && (
              <Link className="text-link" to={link}>
                Открыть исследование →
              </Link>
            )}
            <Button disabled={busy} onClick={() => void download()}>
              {busy ? "Готовим пакет…" : "Скачать пакет"}
            </Button>
          </div>
          {!c.research_available && (
            <p role="status">
              Исходник или выбранный расчёт изменился либо недоступен. Снимок и
              автономный пакет доступны; другой источник не подставляется.
            </p>
          )}
          <SnapshotField evidence={e} />
          <ProfileComparison c={c} />
          <ProcessingChanges e={e} />
          <ExpertSnapshot e={e} />
          {c.result.run && (
            <details>
              <summary>Метод, версия и ограничения</summary>
              <p>{c.result.run.method_version}</p>
              <p>{c.result.run.limitations}</p>
              <pre>{JSON.stringify(c.result.run.parameters, null, 2)}</pre>
            </details>
          )}
          <ProfileValues e={e} />
        </>
      )}
    </main>
  );
}
