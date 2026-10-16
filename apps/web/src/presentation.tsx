import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Button,
  InlineNotification,
  Select,
  SelectItem,
  TextArea,
  Tag,
  RadioButtonGroup,
  RadioButton,
} from "@carbon/react";
import { Link, useSearchParams } from "react-router-dom";
import { api, mutate, download, type RecordDetail, type Slices } from "./api";
import { Heatmap, LineChart } from "./charts";
import { CoordinateInput } from "./coordinate-input";
import { binary } from "./cache";
import { regionLink, type Region } from "./workbench";
import { rehearsalChanged } from "./demo-guard";
import {
  compareProfile,
  qualityLabel,
  signalSummary,
  answerLabels,
  breakTimeGaps,
} from "./presentation-model";
import { ProfileOverlay } from "./presentation-comparison";
import "./presentation.scss";

type Processing = {
  row: number;
  source_sha256: string;
  method: string;
  method_version: string;
  id?: string;
  parameters?: Record<string, unknown>;
  limitations: string;
  training_origin: string;
  result: { denoised?: (number | null)[]; zones?: unknown[] };
};
type Session = {
  id: string;
  region: Region;
  source_sha256: string;
  queue_sha256: string;
  scale: [number, number];
  package: string;
  candidate: string;
  proposal_bounds?: { x_bounds_m?: number[]; row_bounds?: number[] } | null;
  expert_answer: { temperature_episode: string; notes: string } | null;
  processing: Processing | null;
};
type Candidate = { id: string; region: Region; source_sha256: string };
type Evidence = {
  native_shape: number[];
  valid_fraction: number;
  temperature_c: Record<string, number | null>;
  time_kind: string;
  gaps_over_300s: number[];
  mask_counts: Record<string, number>;
  profile_invalid_reason: number[];
  profile_segment_id: number;
  selection: Slices;
};
type Preflight = {
  ml_available: boolean;
  prepared_results: number;
  stream_permitted: boolean;
  services: Record<string, boolean | number | null>;
  source_sha256: string;
};
type Stream = {
  job: { id: string; status: string; error?: string } | null;
  snapshot: {
    accepted: number;
    persisted: number;
    analyzed: number;
    verified_persisted: number;
    verified_cells: number;
    total_profiles: number;
    status: string;
  } | null;
};
const steps = ["Измерение", "Наблюдение", "Сравнение", "Проверка", "Отчёт"];
const descriptions = [
  "Проверьте, откуда взяты данные, как читать температурное поле и какие измерения доступны.",
  "Выберите точку и сопоставьте изменение по расстоянию и времени. Числа и срезы помогают проверить наблюдение.",
  "Подключите фактическую обработку и проверьте, какие значения изменились. Общие оси и разность сохраняют масштаб сравнения.",
  "Сформулируйте вывод и обоснование по исходному сигналу. Сохраните ответ отдельно от рабочего экспертного журнала.",
  "Проверьте состав разбора и скачайте автономный отчёт с исходными числами, обработкой и сохранённым наблюдением.",
];
function numberParam(value: string | null, fallback: number) {
  return value !== null && /^\d+$/.test(value) ? Number(value) : fallback;
}

export default function Presentation() {
  const [params] = useSearchParams();
  return <Rehearsal key={params.get("session") ?? "start"} />;
}
function Rehearsal() {
  const client = useQueryClient();
  const [params, setParams] = useSearchParams();
  const key = params.get("session");
  const step = Math.min(4, numberParam(params.get("step"), 0));
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [answer, setAnswer] = useState("unknown");
  const [notes, setNotes] = useState("");
  const [selectedRun, setSelectedRun] = useState("");
  const [method, setMethod] = useState("median_x3");
  const [ready, setReady] = useState(false);
  const creationKey = useRef(crypto.randomUUID());
  const restored = useRef<string | null>(null);
  const [chosen, setChosen] = useState("");
  const [lastEvidence, setLastEvidence] = useState<Evidence | null>(null);
  const [copied, setCopied] = useState(false);
  const [streamExpanded, setStreamExpanded] = useState(false);
  const workspace = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    // Change the slide inside its viewport; never scroll the document.
    workspace.current?.scrollTo({ top: 0, left: 0, behavior: "instant" });
  }, [step]);
  const packages = useQuery({
    queryKey: ["packages"],
    queryFn: ({ signal }) => api<{ id: string }[]>("/packages", signal),
  });
  const packageId =
    params.get("package") ??
    (packages.data?.some((p) => p.id === "real-data-v2")
      ? "real-data-v2"
      : "general");
  const candidates = useQuery({
    queryKey: ["presentation-candidates", packageId],
    queryFn: ({ signal }) =>
      api<Candidate[]>(
        `/candidates?package=${packageId}&shortlist=${packageId === "real-data-v2"}`,
        signal,
      ),
    enabled: !!packages.data && !key,
  });
  const requestedCandidate = chosen || params.get("candidate");
  const preferred =
    candidates.data?.find((c) => c.id === requestedCandidate)?.id ??
    candidates.data?.find(
      (c) => c.region.record_id === "5" && c.id === "review-00018",
    )?.id ??
    candidates.data?.find((c) => c.region.record_id === "5")?.id ??
    candidates.data?.[0]?.id ??
    "";
  const session = useQuery({
    queryKey: ["demo-session", key],
    queryFn: ({ signal }) => api<Session>(`/demo/sessions/${key}`, signal),
    enabled: !!key,
    retry: false,
  });
  const ctx = session.data;
  const region = ctx?.region;
  const row = region
    ? Math.max(
        region.row_start,
        Math.min(
          region.row_end - 1,
          numberParam(
            params.get("row"),
            Math.floor((region.row_start + region.row_end - 1) / 2),
          ),
        ),
      )
    : 0;
  const column = region
    ? Math.max(
        region.column_start,
        Math.min(
          region.column_end - 1,
          numberParam(
            params.get("column"),
            Math.floor((region.column_start + region.column_end - 1) / 2),
          ),
        ),
      )
    : 0;
  const detail = useQuery({
    queryKey: ["demo-record", region?.record_id, ctx?.source_sha256],
    queryFn: ({ signal }) =>
      api<RecordDetail>(`/records/${region!.record_id}`, signal),
    enabled: !!ctx,
    retry: false,
  });
  const evidence = useQuery({
    queryKey: ["demo-evidence", key, row, column],
    queryFn: ({ signal }) =>
      api<Evidence>(
        `/demo/sessions/${key}/evidence?row=${row}&column=${column}`,
        signal,
      ),
    enabled: !!ctx,
    // Retain charts while another point loads, but never across sessions.
    placeholderData: (previous, query) =>
      query?.queryKey[1] === key ? previous : undefined,
    gcTime: 10000,
    retry: false,
  });
  const preflight = useQuery({
    queryKey: ["demo-preflight", key],
    queryFn: ({ signal }) =>
      api<Preflight>(`/demo/sessions/${key}/preflight`, signal),
    enabled: !!ctx,
    retry: false,
  });
  const results = useQuery({
    queryKey: ["demo-results", key],
    queryFn: ({ signal }) =>
      api<
        { id: string; method: string; method_version: string; region: Region }[]
      >(`/demo/sessions/${key}/results`, signal),
    enabled: !!ctx,
    retry: false,
  });
  const stream = useQuery({
    queryKey: ["demo-stream", key],
    queryFn: ({ signal }) =>
      api<Stream>(`/demo/sessions/${key}/stream`, signal),
    enabled: !!ctx && streamExpanded,
    refetchInterval: (query) =>
      ["queued", "running"].includes(query.state.data?.job?.status ?? "")
        ? 2000
        : false,
  });
  useEffect(() => {
    if (evidence.data && !evidence.isPlaceholderData)
      setLastEvidence(evidence.data);
  }, [evidence.data, evidence.isPlaceholderData]);
  const guard = useQuery({
    queryKey: ["rehearsal-state"],
    queryFn: ({ signal }) =>
      api<{ active: boolean; session_id: string | null }>(
        "/demo/state",
        signal,
      ),
    enabled: !!key,
    refetchInterval: 5000,
  });
  const d = detail.data,
    e = evidence.data ?? lastEvidence,
    s = e?.selection;
  const selectionReady =
    !!s &&
    s.row === row &&
    s.column === column &&
    s.source_sha256 === ctx?.source_sha256 &&
    !evidence.error &&
    !evidence.isPlaceholderData;
  const processing =
    ctx &&
    s &&
    ctx.processing?.row === s.row &&
    ctx.processing.source_sha256 === ctx.source_sha256
      ? ctx.processing
      : null;
  const clean = processing?.result.denoised;
  const x = useMemo(
    () =>
      s && region ? s.x_m.slice(region.column_start, region.column_end) : [],
    [s, region],
  );
  const raw = useMemo(
    () =>
      s && region
        ? s.profile.slice(region.column_start, region.column_end)
        : [],
    [s, region],
  );
  const comparison = useMemo(
    () =>
      compareProfile(
        x,
        raw,
        clean?.slice(region?.column_start, region?.column_end),
      ),
    [x, raw, clean, region],
  );
  const profileSummary = signalSummary(raw);
  const trendSummary = signalSummary(s?.trend ?? []);
  const trendLine = useMemo(
    () => breakTimeGaps(s?.time_s ?? [], s?.trend ?? []),
    [s],
  );
  const draftDirty =
    !!ctx &&
    (notes !== (ctx.expert_answer?.notes ?? "") ||
      answer !== (ctx.expert_answer?.temperature_episode ?? "unknown"));
  const canWrite = ready && guard.data?.active && guard.data.session_id === key;
  const canUseData = selectionReady && d?.source_sha256 === ctx?.source_sha256;
  const validSource = d?.source_sha256 === ctx?.source_sha256;
  const path =
    d && region && validSource
      ? `/records/${region.record_id}/tiles?x0=${d.x_m[region.column_start]}&x1=${d.x_m[region.column_end - 1]}&t0=${d.time_s[region.row_start]}&t1=${Math.max(d.time_s[region.row_end - 1], d.time_s[region.row_start] + 1)}&width=${innerWidth < 800 ? 96 : 128}&height=128`
      : "";

  useEffect(() => {
    if (!key) return;
    let active = true;
    void mutate<Session>(`/demo/sessions/${key}/activate`, {})
      .then(() => {
        if (active) {
          setReady(true);
          rehearsalChanged();
          void client.invalidateQueries({ queryKey: ["rehearsal-state"] });
        }
      })
      .catch((err) => {
        if (active) setError(String(err));
      });
    return () => {
      active = false;
    };
  }, [key, client]);
  useEffect(() => {
    if (!ctx || restored.current === ctx.id) return;
    restored.current = ctx.id;
    setAnswer(ctx.expert_answer?.temperature_episode ?? "unknown");
    setNotes(ctx.expert_answer?.notes ?? "");
  }, [ctx]);
  useEffect(() => {
    if (!path || !ctx) return;
    const abort = new AbortController();
    void binary(path, abort.signal, ctx.source_sha256).catch(() => {});
    return () => abort.abort();
  }, [path, ctx?.source_sha256]);
  function move(next: number) {
    const updated = new URLSearchParams(params);
    updated.set("step", String(Math.max(0, Math.min(4, next))));
    setParams(updated, { replace: true, preventScrollReset: true });
  }
  useEffect(() => {
    const navigate = (event: KeyboardEvent) => {
      if (
        event.ctrlKey ||
        event.altKey ||
        event.metaKey ||
        (event.target as HTMLElement)?.closest(
          "input,textarea,select,button,a,[contenteditable]",
        )
      )
        return;
      if (event.key === "ArrowRight" || event.key === "ArrowLeft") {
        event.preventDefault();
        move(step + (event.key === "ArrowRight" ? 1 : -1));
      }
    };
    window.addEventListener("keydown", navigate);
    return () => window.removeEventListener("keydown", navigate);
  });
  async function work(action: () => Promise<unknown>) {
    setError("");
    setBusy(true);
    try {
      await action();
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(false);
    }
  }
  function changeSelection(r: number, c: number) {
    if (
      busy ||
      !region ||
      r < region.row_start ||
      r >= region.row_end ||
      c < region.column_start ||
      c >= region.column_end
    )
      return;
    const next = new URLSearchParams(params);
    next.set("row", String(r));
    next.set("column", String(c));
    setParams(next, { replace: true, preventScrollReset: true });
  }
  const retry = () =>
    void work(async () => {
      await session.refetch();
      await detail.refetch();
      await evidence.refetch();
    });
  const focusPeak = () => {
    if (profileSummary.peak >= 0 && region)
      changeSelection(row, region.column_start + profileSummary.peak);
  };
  const notePoint = () => {
    if (!s || !canUseData) return;
    const observation = `${s.timestamp}; ${s.distance_m.toFixed(3)} м; ${s.temperature === null ? "нет измерения" : `${s.temperature.toFixed(4)} °C`}; ${qualityLabel(s.invalid_reason)}.`;
    setNotes((previous) =>
      previous ? `${previous}\n${observation}`.slice(0, 8000) : observation,
    );
  };
  const sourceParams = new URLSearchParams(
    region ? regionLink(region).split("?")[1] : "",
  );
  sourceParams.set("row", String(row));
  sourceParams.set("column", String(column));
  sourceParams.set("source_sha256", ctx?.source_sha256 ?? "");
  const sourceLink = `/research?${sourceParams}`;
  const pointReading = (
    <div className="demo-reading" aria-busy={evidence.isFetching}>
      <div>
        <span>Расстояние</span>
        <strong>{s ? `${s.distance_m.toFixed(3)} м` : "—"}</strong>
      </div>
      <div>
        <span>Время измерения</span>
        <strong>{s?.timestamp ?? "—"}</strong>
      </div>
      <div>
        <span>Температура</span>
        <strong>
          {s?.temperature != null
            ? `${s.temperature.toFixed(4)} °C`
            : "Нет измерения"}
        </strong>
      </div>
      <div>
        <span>Качество</span>
        <strong>{s ? qualityLabel(s.invalid_reason) : "—"}</strong>
      </div>
    </div>
  );
  const pointControls = region && (
    <div className="region-controls demo-coordinates">
      <CoordinateInput
        id="demo-row"
        label="Профиль (исходный индекс)"
        value={row}
        min={region.row_start}
        max={region.row_end - 1}
        integer
        disabled={busy}
        onCommit={(value) => changeSelection(value, column)}
        onPrevious={() => changeSelection(row - 1, column)}
        onNext={() => changeSelection(row + 1, column)}
      />
      <CoordinateInput
        id="demo-column"
        label="Координата (исходный индекс)"
        value={column}
        min={region.column_start}
        max={region.column_end - 1}
        integer
        disabled={busy}
        previousLabel="Предыдущая координата"
        nextLabel="Следующая координата"
        onCommit={(value) => changeSelection(row, value)}
        onPrevious={() => changeSelection(row, column - 1)}
        onNext={() => changeSelection(row, column + 1)}
      />
    </div>
  );
  const heatmap = path && ctx && region && (
    <Heatmap
      path={path}
      versionKey={ctx.source_sha256}
      expectedSourceSha={ctx.source_sha256}
      cursor={d ? [d.x_m[column], d.time_s[row]] : undefined}
      onSelect={(distance, time) => {
        if (!d || busy) return;
        const nearest = (
          axis: number[],
          value: number,
          start: number,
          end: number,
        ) => {
          let best = start;
          for (let i = start + 1; i < end; i++)
            if (Math.abs(axis[i] - value) < Math.abs(axis[best] - value))
              best = i;
          return best;
        };
        changeSelection(
          nearest(d.time_s, time, region.row_start, region.row_end),
          nearest(d.x_m, distance, region.column_start, region.column_end),
        );
      }}
    />
  );
  const sourceProfile =
    s && region ? (
      <LineChart
        title="Исходный профиль · точный срез"
        x={x}
        y={raw}
        xLabel="Расстояние, м"
        syncKey={`demo-${key}`}
      />
    ) : (
      <div className="demo-chart-loading">Загружаем точный профиль…</div>
    );
  const series = s ? (
    <>
      <LineChart
        title="Температура выбранной координаты во времени"
        x={trendLine.x}
        y={trendLine.y}
        xLabel="Время от начала, с"
      />
      <p className="demo-hint">
        Разрывы времени более 300 с показаны разрывом линии. Выбранная
        координата не усредняется с соседними.
      </p>
    </>
  ) : (
    <div className="demo-chart-loading">Загружаем временной ряд…</div>
  );
  const stepStatus = [
    validSource && !!e ? "Источник сверён" : "Проверка источника",
    notes.trim() ? "Есть заметки" : "Исследуйте сигнал",
    comparison ? "Обработка подключена" : "Не выбрана",
    ctx?.expert_answer
      ? draftDirty
        ? "Есть несохранённые изменения"
        : "Ответ сохранён"
      : "Ответ не сохранён",
    "HTML + JSON",
  ];
  if (!key)
    return (
      <main className="page presentation">
        <p className="eyebrow">Рабочая область / Демонстрация</p>
        <h1>От измерения к проверяемому выводу</h1>
        <p className="demo-lead">
          Пройдите полный разбор реального измерения: от качества исходника до
          автономного отчёта.
        </p>
        <div className="demo-start">
          <section className="demo-panel">
            <h2>Выберите исследовательское окно</h2>
            <p>
              Каждый участок привязан к записи и SHA. Репетиция не меняет
              рабочие экспертные ответы.
            </p>
            <Select
              id="demo-candidate"
              labelText="Реальный участок"
              value={preferred}
              disabled={busy || candidates.isPending}
              onChange={(event) => {
                setChosen(event.target.value);
                creationKey.current = crypto.randomUUID();
              }}
            >
              {!candidates.data?.length && (
                <SelectItem
                  value=""
                  text={
                    candidates.isPending
                      ? "Загружаем участки…"
                      : "Участки недоступны"
                  }
                />
              )}
              {(candidates.data ?? []).map((c) => (
                <SelectItem
                  key={c.id}
                  value={c.id}
                  text={`${c.region.record_id}.csv · ${c.id} · профили ${c.region.row_start}–${c.region.row_end - 1}`}
                />
              ))}
            </Select>
            <Button
              disabled={busy || !preferred}
              onClick={() =>
                void work(async () => {
                  const created = await mutate<Session>("/demo/sessions", {
                    client_key: creationKey.current,
                    package: packageId,
                    candidate: preferred,
                  });
                  setParams({ session: created.id, step: "0" });
                  rehearsalChanged();
                })
              }
            >
              {busy ? "Открываем участок…" : "Начать репетицию"}
            </Button>
            <p className="demo-hint">
              Пока репетиция активна, сервер защищает рабочий журнал во всех
              вкладках. Завершить её можно кнопкой в верхнем уведомлении.
            </p>
          </section>
          <ol className="demo-itinerary">
            {steps.map((name, i) => (
              <li key={name}>
                <span>{String(i + 1).padStart(2, "0")}</span>
                <div>
                  <h2>{name}</h2>
                  <p>{descriptions[i]}</p>
                </div>
              </li>
            ))}
          </ol>
        </div>
        {(error || packages.error || candidates.error) && (
          <InlineNotification
            kind="error"
            title="Маршрут недоступен"
            subtitle={error || String(packages.error ?? candidates.error)}
            hideCloseButton
          />
        )}
        {(packages.error || candidates.error) && (
          <Button
            kind="tertiary"
            onClick={() => {
              void packages.refetch();
              void candidates.refetch();
            }}
          >
            Повторить загрузку участков
          </Button>
        )}
      </main>
    );
  return (
    <main className="page presentation" id="presentation-main">
      <div className="page-title">
        <div>
          <p className="eyebrow">Демонстрация / Реальное измерение</p>
          <h1>От измерения к проверяемому выводу</h1>
        </div>
        <Tag type="blue">Репетиция · не разметка</Tag>
      </div>
      {(error ||
        session.error ||
        detail.error ||
        evidence.error ||
        evidence.fetchStatus === "paused") && (
        <div className="demo-error" role="alert">
          <InlineNotification
            kind="error"
            title="Не удалось обновить данные"
            subtitle={
              error ||
              (evidence.fetchStatus === "paused"
                ? "Нет соединения. Загруженные данные сохранены; обновление продолжится после восстановления связи."
                : String(session.error ?? detail.error ?? evidence.error))
            }
            hideCloseButton
          />
          <Button kind="tertiary" disabled={busy} onClick={retry}>
            Повторить подключение
          </Button>
          <p>
            Уже загруженный сигнал остаётся доступен. Необновлённые значения не
            используются для расчёта и отчёта.
          </p>
        </div>
      )}
      {!ctx && !session.error && (
        <p role="status">Открываем сохранённый маршрут…</p>
      )}
      {!ctx && session.error && <Link to="/demo">Выбрать новое окно</Link>}
      {ctx && region && (
        <>
          <nav
            className="presentation-steps"
            aria-label="Шаги демонстрации"
            onKeyDown={(event) => {
              if (event.ctrlKey || event.altKey || event.metaKey) return;
              if (
                !["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)
              )
                return;
              event.preventDefault();
              const next =
                event.key === "Home"
                  ? 0
                  : event.key === "End"
                    ? 4
                    : Math.max(
                        0,
                        Math.min(
                          4,
                          step + (event.key === "ArrowRight" ? 1 : -1),
                        ),
                      );
              move(next);
              const button =
                event.currentTarget.querySelectorAll("button")[next];
              button.focus({ preventScroll: true });
              if (
                event.currentTarget.scrollWidth >
                event.currentTarget.clientWidth
              )
                event.currentTarget.scrollLeft =
                  button.offsetLeft - event.currentTarget.offsetLeft;
            }}
          >
            {steps.map((name, i) => (
              <button
                key={name}
                aria-current={i === step ? "step" : undefined}
                onClick={() => move(i)}
              >
                <span>{String(i + 1).padStart(2, "0")}</span>
                <strong>{name}</strong>
                <small>{stepStatus[i]}</small>
              </button>
            ))}
          </nav>
          <section className="presentation-caption" aria-live="polite">
            <p className="eyebrow">
              Шаг {step + 1} из 5 · {region.record_id}.csv · {ctx.candidate}
            </p>
            <h2 tabIndex={-1}>{steps[step]}</h2>
            <div className="presentation-description">
              {descriptions.map((description, i) => (
                <p
                  key={i}
                  aria-hidden={i !== step}
                  style={{ visibility: i === step ? "visible" : "hidden" }}
                >
                  {description}
                </p>
              ))}
            </div>
          </section>
          {d && !validSource && (
            <p className="demo-warning" role="alert">
              SHA источника изменился. Расчёт и экспорт заблокированы: другой
              исходник не подставлен.
            </p>
          )}
          {!canWrite && !guard.isPending && (
            <div className="demo-warning">
              <p>
                Репетиция не активна. Просмотр доступен; для сохранения ответа
                или обработки возобновите защиту рабочего журнала.
              </p>
              <Button
                kind="tertiary"
                disabled={busy}
                onClick={() =>
                  void work(async () => {
                    await mutate(`/demo/sessions/${key}/activate`, {});
                    setReady(true);
                    rehearsalChanged();
                    await guard.refetch();
                  })
                }
              >
                Возобновить репетицию
              </Button>
            </div>
          )}
          <p className="demo-selection-status" role="status">
            {evidence.error || evidence.fetchStatus === "paused"
              ? "Выборка не обновлена. Ниже последний загруженный срез."
              : busy
                ? "Выполняем действие…"
                : !selectionReady && s
                  ? "Выбрана новая точка. До обновления показан предыдущий срез."
                  : evidence.isFetching
                    ? "Обновляем точный срез…"
                    : "Материалы шага прокручиваются внутри блока."}
          </p>
          <div
            ref={workspace}
            className={`presentation-workspace demo-step-${step}`}
            role="region"
            aria-label="Материалы выбранного шага"
            tabIndex={0}
          >
            <section
              className="presentation-signal demo-panel"
              aria-label={`Данные: ${steps[step]}`}
            >
              {step === 0 && (
                <>
                  <div className="demo-section-heading">
                    <h2>Исходный сигнал</h2>
                    <Tag type={validSource ? "green" : "gray"}>
                      {validSource ? "SHA совпадает" : "Проверяем источник"}
                    </Tag>
                  </div>
                  <p>
                    Нажмите на поле: по горизонтали — расстояние вдоль волокна,
                    по вертикали — время. Числа ниже взяты из исходной ячейки.
                  </p>
                  {heatmap}
                  {pointReading}
                  {pointControls}
                  <details className="demo-inline-details">
                    <summary>Как читать поле и пропуски</summary>
                    <p>
                      Цвет показывает температуру. Серый означает отсутствие
                      валидного измерения. Пропуски не заполняются; sentinel
                      −300 не считается температурой. Обзор может агрегировать
                      исходные ячейки — правило указано под полем, точный срез
                      доступен на следующем шаге.
                    </p>
                  </details>
                </>
              )}
              {step === 1 && (
                <>
                  <div className="demo-section-heading">
                    <h2>Исследуйте изменение</h2>
                    <Button
                      size="sm"
                      kind="tertiary"
                      disabled={busy || !canUseData || profileSummary.peak < 0}
                      onClick={focusPeak}
                    >
                      Максимум выбранного профиля
                    </Button>
                  </div>
                  <p>
                    Найдите изменение на поле, затем проверьте его по
                    пространственному профилю и временному ряду одной
                    координаты.
                  </p>
                  {heatmap}
                  {pointReading}
                  {pointControls}
                  {sourceProfile}
                  {series}
                </>
              )}
              {step === 2 && (
                <>
                  <h2>Что изменила обработка</h2>
                  <p>
                    Сравнение относится к профилю {s?.row ?? row}. Смена метода
                    в форме не запускает вычисление и не подменяет подключённый
                    результат.
                  </p>
                  {pointControls}
                  {comparison ? (
                    <>
                      <ProfileOverlay
                        x={x}
                        raw={comparison.original}
                        clean={comparison.clean}
                        limits={comparison.limits}
                        syncKey={`demo-${key}`}
                      />
                      <LineChart
                        title="Разность: обработка − исходник"
                        x={x}
                        y={comparison.delta}
                        xLabel="Расстояние, м"
                        yLimits={comparison.deltaLimits}
                        syncKey={`demo-${key}`}
                      />
                      <div className="demo-metrics">
                        <div>
                          <span>Сопоставимых точек</span>
                          <strong>
                            {comparison.stats.paired_points} /{" "}
                            {comparison.stats.total_points}
                          </strong>
                        </div>
                        <div>
                          <span>Изменений больше 0,01 °C</span>
                          <strong>{comparison.stats.changed_points}</strong>
                        </div>
                        <div>
                          <span>Среднее |изменение|</span>
                          <strong>
                            {formatC(comparison.stats.mean_absolute_change_c)}
                          </strong>
                        </div>
                        <div>
                          <span>Максимальное |изменение|</span>
                          <strong>
                            {formatC(comparison.stats.max_absolute_change_c)}
                          </strong>
                        </div>
                      </div>
                      <p className="demo-hint">
                        Разность описывает изменение сигнала, а не доказанный
                        шум. Порог 0,01 °C — правило отображения, не точность
                        датчика. Пропуски исключены из пар.
                      </p>
                      {comparison.stats.paired_points === 0 && (
                        <p>
                          Сопоставимых валидных пар нет. Оценить величину
                          изменений в этом профиле нельзя.
                        </p>
                      )}
                      {comparison.stats.top_points.length > 0 && (
                        <div className="numeric-scroll">
                          <table>
                            <caption>Пять наибольших изменений</caption>
                            <thead>
                              <tr>
                                <th>Расстояние, м</th>
                                <th>Исходник, °C</th>
                                <th>Обработка, °C</th>
                                <th>Разность, °C</th>
                              </tr>
                            </thead>
                            <tbody>
                              {comparison.stats.top_points.map((point) => (
                                <tr key={point.index}>
                                  <td>
                                    <button
                                      className="demo-text-button"
                                      disabled={busy}
                                      onClick={() =>
                                        changeSelection(
                                          row,
                                          region.column_start + point.index,
                                        )
                                      }
                                    >
                                      {point.x_m.toFixed(3)}
                                    </button>
                                  </td>
                                  <td>{point.original_c.toFixed(4)}</td>
                                  <td>{point.processed_c.toFixed(4)}</td>
                                  <td>
                                    {point.delta_c > 0 ? "+" : ""}
                                    {point.delta_c.toFixed(4)}
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      )}
                    </>
                  ) : (
                    <>
                      {sourceProfile}
                      <div className="demo-empty">
                        <h3>Обработка этого профиля не выбрана</h3>
                        <p>
                          Подключите совместимый сохранённый результат или явно
                          запустите референсную очистку. До этого показаны
                          только исходные значения.
                        </p>
                      </div>
                    </>
                  )}
                  {pointReading}
                </>
              )}
              {step === 3 && (
                <>
                  <h2>Самостоятельное наблюдение</h2>
                  <p>
                    Сверьте вывод с исходным сигналом. Эта репетиция не является
                    слепым исследованием: результат обработки мог быть показан
                    на предыдущем шаге.
                  </p>
                  {heatmap}
                  {pointReading}
                  {pointControls}
                  {sourceProfile}
                  {series}
                </>
              )}
              {step === 4 && (
                <>
                  <h2>Проверяемый итог</h2>
                  <p>
                    Сверьте состав перед передачей. Отчёт формируется сервером
                    для этой записи, окна и выбранного профиля.
                  </p>
                  <ol className="demo-report-checks">
                    <li>
                      <span>01</span>
                      <div>
                        <h3>Исходное измерение</h3>
                        <p>
                          {region.record_id}.csv · строки [{region.row_start},{" "}
                          {region.row_end}), столбцы [{region.column_start},{" "}
                          {region.column_end}).
                        </p>
                        <p>
                          Профиль {row} ·{" "}
                          {d?.timestamps?.[row] ??
                            s?.timestamp ??
                            "Время загружается"}
                        </p>
                        <Tag type={validSource ? "green" : "red"}>
                          {validSource
                            ? "Источник сверён по SHA"
                            : "Проверка не завершена"}
                        </Tag>
                      </div>
                    </li>
                    <li>
                      <span>02</span>
                      <div>
                        <h3>Результат обработки</h3>
                        <p>
                          {processing
                            ? `${processing.method} · ${processing.method_version}`
                            : "Обработка не выбрана. Отчёт сохранит исходный сигнал."}
                        </p>
                        {processing && <p>{processing.limitations}</p>}
                        <Button kind="ghost" size="sm" onClick={() => move(2)}>
                          Проверить сравнение
                        </Button>
                      </div>
                    </li>
                    <li>
                      <span>03</span>
                      <div>
                        <h3>Сохранённое наблюдение</h3>
                        <p>
                          {ctx.expert_answer
                            ? `${answerLabels[ctx.expert_answer.temperature_episode] ?? ctx.expert_answer.temperature_episode}. ${ctx.expert_answer.notes}`
                            : "Ответ ещё не сохранён. Его отсутствие будет обозначено явно."}
                        </p>
                        {draftDirty && (
                          <p className="demo-warning">
                            Есть несохранённый черновик. В отчёт войдёт только
                            последний сохранённый ответ.
                          </p>
                        )}
                        <Button kind="ghost" size="sm" onClick={() => move(3)}>
                          Проверить ответ
                        </Button>
                      </div>
                    </li>
                  </ol>
                  <details className="demo-inline-details">
                    <summary>Что можно проверить независимо</summary>
                    <p>
                      HTML открывается без сервера. JSON содержит исходный
                      профиль, координаты, маски, обработку при наличии, SHA и
                      ответ с пометкой «репетиция». SHA связывает данные с
                      источником, но не подтверждает правильность вывода или
                      авторство. Этот отчёт не является неизменяемым рабочим
                      кейсом; такой кейс фиксируется в рабочем исследовании.
                    </p>
                  </details>
                </>
              )}
            </section>
            <aside
              className="presentation-decision demo-panel"
              aria-label={`Действия: ${steps[step]}`}
            >
              {step === 0 && (
                <>
                  <h2>Паспорт измерения</h2>
                  <dl className="demo-passport">
                    <dt>Источник</dt>
                    <dd>{region.record_id}.csv</dd>
                    <dt>Размер окна</dt>
                    <dd>{e?.native_shape.join(" × ") ?? "…"} ячеек</dd>
                    <dt>Валидных наблюдений</dt>
                    <dd>
                      {e ? `${(e.valid_fraction * 100).toFixed(1)} %` : "…"}
                    </dd>
                    <dt>Время окна</dt>
                    <dd>
                      {d
                        ? `${d.timestamps[region.row_start]} — ${d.timestamps[region.row_end - 1]}`
                        : "…"}
                    </dd>
                    <dt>Часовой пояс</dt>
                    <dd>
                      {e?.time_kind === "local_unzoned"
                        ? "Не указан в источнике"
                        : (e?.time_kind ?? "…")}
                    </dd>
                  </dl>
                  <h3>Качество исходника</h3>
                  <ul className="demo-quality">
                    {Object.entries(e?.mask_counts ?? {}).map(
                      ([reason, count]) => (
                        <li key={reason}>
                          <span>{qualityLabel(Number(reason))}</span>
                          <strong>{count.toLocaleString("ru")}</strong>
                        </li>
                      ),
                    )}
                  </ul>
                  <p className="demo-hint">
                    Это исходные измерения, не синтетические данные. Границы
                    окна не означают подтверждённую аварию.
                  </p>
                  <Button kind="tertiary" onClick={() => move(1)}>
                    Исследовать изменение →
                  </Button>
                </>
              )}
              {step === 1 && (
                <>
                  <h2>Основание наблюдения</h2>
                  <p>
                    {ctx.proposal_bounds?.x_bounds_m
                      ? `Предложенное ядро: ${ctx.proposal_bounds.x_bounds_m.map((v) => v.toFixed(3)).join("–")} м. Границы остаются гипотезой.`
                      : "У этого окна нет подтверждённых границ события. Исследуйте локальные изменения и соседний контекст."}
                  </p>
                  <dl className="demo-passport">
                    <dt>Выбранный профиль: min / max</dt>
                    <dd>
                      {formatC(profileSummary.min)} /{" "}
                      {formatC(profileSummary.max)}
                    </dd>
                    <dt>Размах во времени в выбранной точке</dt>
                    <dd>{formatC(trendSummary.range)}</dd>
                    <dt>Временной ряд: валидных / пропусков</dt>
                    <dd>
                      {trendSummary.valid} / {trendSummary.missing}
                    </dd>
                    <dt>Разрывы более 300 с</dt>
                    <dd>{e?.gaps_over_300s.length ?? "…"}</dd>
                  </dl>
                  <p>
                    Один пик может быть локальным выбросом или частью
                    устойчивого изменения. Сравните соседние профили и
                    длительность; причина из этих чисел не следует.
                  </p>
                  <Button
                    kind="tertiary"
                    disabled={!canUseData}
                    onClick={notePoint}
                  >
                    Добавить точку в обоснование
                  </Button>
                  {notes && (
                    <p role="status">
                      Заметки доступны на шаге «Проверка». Черновик сохраняется
                      отдельным действием.
                    </p>
                  )}
                  <Link className="text-link" target="_blank" to={sourceLink}>
                    Открыть исходные срезы и соседний контекст →
                  </Link>
                </>
              )}
              {step === 2 && (
                <>
                  <h2>Сравнение обработки</h2>
                  <p>
                    Расчёт выполняется только по кнопке. Медиана — референсная
                    очистка профиля, не финальная модель обнаружения.
                  </p>
                  <Select
                    id="demo-method"
                    labelText="Обработка профиля"
                    value={method}
                    disabled={busy}
                    onChange={(event) => setMethod(event.target.value)}
                  >
                    <SelectItem value="median_x3" text="Медиана ×3" />
                    <SelectItem value="median_x5" text="Медиана ×5" />
                  </Select>
                  <Button
                    disabled={
                      busy ||
                      !canWrite ||
                      !canUseData ||
                      preflight.data?.ml_available !== true
                    }
                    onClick={() =>
                      void work(async () => {
                        await mutate(`/demo/sessions/${key}/process`, {
                          row,
                          method,
                        });
                        await session.refetch();
                      })
                    }
                  >
                    {busy ? "Выполняем обработку…" : "Получить обработку"}
                  </Button>
                  {preflight.data?.ml_available !== true && (
                    <p className="demo-hint">
                      {preflight.isPending
                        ? "Проверяем доступность ML API…"
                        : "ML API недоступен. Исходный сигнал и импортированные результаты доступны."}
                    </p>
                  )}
                  <Select
                    id="demo-result"
                    labelText="Подготовленный результат"
                    value={selectedRun}
                    disabled={busy}
                    onChange={(event) => setSelectedRun(event.target.value)}
                  >
                    <SelectItem
                      value=""
                      text="Выберите совместимый результат"
                    />
                    {(results.data ?? [])
                      .filter(
                        (run) =>
                          run.region.row_end - run.region.row_start !== 1 ||
                          run.region.row_start === row,
                      )
                      .map((run) => (
                        <SelectItem
                          key={run.id}
                          value={run.id}
                          text={`${run.method} · ${run.method_version}`}
                        />
                      ))}
                  </Select>
                  <Button
                    kind="tertiary"
                    disabled={
                      busy ||
                      !canWrite ||
                      !canUseData ||
                      !selectedRun ||
                      !(results.data ?? []).some(
                        (run) =>
                          run.id === selectedRun &&
                          (run.region.row_end - run.region.row_start !== 1 ||
                            run.region.row_start === row),
                      )
                    }
                    onClick={() =>
                      void work(async () => {
                        await mutate(`/demo/sessions/${key}/process`, {
                          row,
                          method,
                          run_id: selectedRun,
                        });
                        await session.refetch();
                      })
                    }
                  >
                    Подключить результат
                  </Button>
                  <Button
                    kind="ghost"
                    onClick={() => {
                      void results.refetch();
                      void preflight.refetch();
                    }}
                  >
                    Обновить список результатов
                  </Button>
                  {!results.data?.length && !results.isPending && (
                    <p className="demo-hint">
                      Совместимых профильных результатов для этого окна нет.
                      Поля из лаборатории не подставляются вместо профиля.
                    </p>
                  )}
                  {results.error && (
                    <p role="alert">
                      Список не обновлён: {String(results.error)}
                    </p>
                  )}
                  {processing && (
                    <div className="demo-processing-passport">
                      <h3>Подключённый результат</h3>
                      <p>
                        {processing.method} · профиль {processing.row}
                      </p>
                      <dl className="demo-passport">
                        <dt>Версия</dt>
                        <dd>{processing.method_version}</dd>
                        <dt>Происхождение обучения</dt>
                        <dd>
                          {processing.training_origin === "untrained"
                            ? "Без обучения"
                            : processing.training_origin}
                        </dd>
                        {processing.id && (
                          <>
                            <dt>ID запуска</dt>
                            <dd>
                              <code>{processing.id}</code>
                            </dd>
                          </>
                        )}
                      </dl>
                      <p>{processing.limitations}</p>
                      {!clean && (
                        <p>
                          В результате нет полного обработанного профиля. Линия
                          сравнения не построена.
                        </p>
                      )}
                    </div>
                  )}
                </>
              )}
              {step === 3 && (
                <form
                  onSubmit={(event) => {
                    event.preventDefault();
                    void work(async () => {
                      await mutate(`/demo/sessions/${key}/answer`, {
                        temperature_episode: answer,
                        notes,
                      });
                      await session.refetch();
                    });
                  }}
                >
                  <h2>Зафиксируйте наблюдение</h2>
                  <p>
                    Ответ относится ко всему окну, а не к одной ячейке. Опишите,
                    что видно и чего не хватает для вывода.
                  </p>
                  <RadioButtonGroup
                    name="demo-answer"
                    orientation="vertical"
                    legendText="Что видно в этом окне?"
                    valueSelected={answer}
                    onChange={(value) => setAnswer(String(value))}
                  >
                    {Object.entries(answerLabels).map(([value, label]) => (
                      <RadioButton
                        key={value}
                        id={`demo-answer-${value}`}
                        value={value}
                        labelText={label}
                      />
                    ))}
                  </RadioButtonGroup>
                  <TextArea
                    id="demo-notes"
                    labelText="Обоснование по сигналу"
                    value={notes}
                    maxLength={8000}
                    required
                    disabled={busy}
                    placeholder="Где и когда видно изменение? Есть ли пропуски или альтернативное объяснение?"
                    onChange={(event) => setNotes(event.target.value)}
                  />
                  <p className="demo-hint">
                    {notes.length} / 8000 ·{" "}
                    {draftDirty
                      ? "Черновик не сохранён"
                      : ctx.expert_answer
                        ? "Совпадает с сохранённым ответом"
                        : "Ответ ещё не сохранён"}
                  </p>
                  <Button
                    type="submit"
                    disabled={
                      busy ||
                      !canWrite ||
                      !canUseData ||
                      !notes.trim() ||
                      !draftDirty
                    }
                  >
                    {busy ? "Сохраняем…" : "Сохранить ответ репетиции"}
                  </Button>
                  {ctx.expert_answer && !draftDirty && (
                    <p className="demo-saved" role="status">
                      Ответ репетиции сохранён отдельно. Рабочая разметка не
                      изменена.
                    </p>
                  )}
                  <p className="demo-hint">
                    Авторство рабочего эксперта и обучающие метки здесь не
                    создаются.
                  </p>
                </form>
              )}
              {step === 4 && (
                <>
                  <h2>Передать разбор</h2>
                  <p>
                    Два формата одного серверного отчёта: HTML для просмотра,
                    JSON для числовой сверки.
                  </p>
                  <Button
                    disabled={busy || !canUseData}
                    onClick={() =>
                      void work(() =>
                        download(
                          `/demo/sessions/${key}/report?format=html&row=${row}`,
                          {},
                          "dts-rehearsal.html",
                        ),
                      )
                    }
                  >
                    Скачать автономный отчёт
                  </Button>
                  <Button
                    kind="tertiary"
                    disabled={busy || !canUseData}
                    onClick={() =>
                      void work(() =>
                        download(
                          `/demo/sessions/${key}/report?format=json&row=${row}`,
                          {},
                          "dts-rehearsal.json",
                        ),
                      )
                    }
                  >
                    Скачать точные данные JSON
                  </Button>
                  <Button
                    kind="ghost"
                    onClick={() =>
                      void work(async () => {
                        if (!navigator.clipboard)
                          throw new Error(
                            "Буфер обмена недоступен. Скопируйте ссылку из адресной строки.",
                          );
                        await navigator.clipboard.writeText(location.href);
                        setCopied(true);
                      })
                    }
                  >
                    Скопировать ссылку на этот шаг
                  </Button>
                  {copied && <p role="status">Ссылка скопирована.</p>}
                  <p className="demo-hint">
                    Ссылка требует входа владельца и доступна в течение срока
                    репетиции. Переданный HTML работает без сайта. Скачайте его
                    заранее для резервного показа.
                  </p>
                  <dl className="demo-passport">
                    <dt>Сессия</dt>
                    <dd>
                      <code>{ctx.id}</code>
                    </dd>
                    <dt>Источник / SHA-256</dt>
                    <dd>
                      <code>{ctx.source_sha256}</code>
                    </dd>
                    <dt>Очередь / SHA-256</dt>
                    <dd>
                      <code>{ctx.queue_sha256}</code>
                    </dd>
                  </dl>
                </>
              )}
            </aside>
          </div>
          <div className="presentation-navigation">
            <Button
              kind="tertiary"
              disabled={step === 0}
              onClick={() => move(step - 1)}
            >
              Назад
            </Button>
            <span>Шаг {step + 1} / 5 · стрелки ← → вне полей ввода</span>
            <Button disabled={step === 4} onClick={() => move(step + 1)}>
              Следующий шаг
            </Button>
          </div>
          <details className="presentation-details">
            <summary>Проверить точные значения и происхождение</summary>
            <p>
              SHA источника: <code>{ctx.source_sha256}</code>
            </p>
            <p>
              SHA очереди: <code>{ctx.queue_sha256}</code>
            </p>
            <p>
              Окно [{region.row_start}, {region.row_end}) × [
              {region.column_start}, {region.column_end}). Шкала поля{" "}
              {ctx.scale.join("–")} °C фиксирована для источника; значения могут
              выходить за p1/p99.
            </p>
            {s && e && (
              <div className="numeric-scroll">
                <table>
                  <caption>
                    Точный профиль {s.row}: первые {Math.min(x.length, 500)} из{" "}
                    {x.length} координат
                  </caption>
                  <thead>
                    <tr>
                      <th>Расстояние, м</th>
                      <th>Исходная T, °C</th>
                      <th>Качество</th>
                      <th>После обработки, °C</th>
                    </tr>
                  </thead>
                  <tbody>
                    {x.slice(0, 500).map((value, i) => (
                      <tr key={i}>
                        <td>{value.toFixed(6)}</td>
                        <td>{raw[i]?.toFixed(6) ?? "Нет измерения"}</td>
                        <td>{qualityLabel(e.profile_invalid_reason[i])}</td>
                        <td>{comparison?.clean[i]?.toFixed(6) ?? "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p>Полный профиль доступен в JSON на шаге «Отчёт».</p>
              </div>
            )}
          </details>
          <details
            className="presentation-details"
            onToggle={(event) => setStreamExpanded(event.currentTarget.open)}
          >
            <summary>Потоковый эпизод: архив → хранилище → обработка</summary>
            <p>
              Настоящая историческая запись, отдельная техническая сессия.
              Принято — Redis; сохранено — ClickHouse; обработано — сверенные
              исходные ячейки и median ×3. Это не подключение оборудования.
            </p>
            <div className="evidence-strip">
              {[
                ["Принято", stream.data?.snapshot?.accepted],
                ["Сохранено", stream.data?.snapshot?.persisted],
                ["Обработано", stream.data?.snapshot?.analyzed],
                ["Сверено ячеек", stream.data?.snapshot?.verified_cells],
              ].map(([label, value]) => (
                <div key={String(label)}>
                  <span className="label">{label}</span>
                  <strong>{value ?? 0}</strong>
                </div>
              ))}
            </div>
            <p role="status">
              {stream.data?.job
                ? `Задание: ${stream.data.job.status} · ${stream.data.job.id}`
                : "Техническая передача ещё не запущена."}
            </p>
            {stream.error && (
              <p role="alert">
                Связь потеряна; повторяем получение подтверждённого снимка.{" "}
                {String(stream.error)}
              </p>
            )}
            {stream.data?.job?.error && (
              <p role="alert">{stream.data.job.error}</p>
            )}
            <div className="action-row">
              <Button
                disabled={
                  busy ||
                  !canWrite ||
                  !!stream.data?.job ||
                  !preflight.data?.stream_permitted ||
                  !preflight.data.services.stream_worker ||
                  !preflight.data.services.ingestion ||
                  !preflight.data.services.clickhouse
                }
                onClick={() =>
                  void work(async () => {
                    await mutate(`/demo/sessions/${key}/stream`, {});
                    await stream.refetch();
                  })
                }
              >
                Начать техническую передачу
              </Button>
              <Button
                kind="tertiary"
                disabled={
                  busy ||
                  !stream.data?.job ||
                  !["queued", "running"].includes(stream.data.job.status)
                }
                onClick={() =>
                  void work(async () => {
                    await mutate(`/demo/sessions/${key}/stream/cancel`, {});
                    await stream.refetch();
                  })
                }
              >
                Прервать передачу
              </Button>
              <Button
                kind="tertiary"
                disabled={
                  busy ||
                  !stream.data?.job ||
                  !["failed", "cancelled"].includes(stream.data.job.status)
                }
                onClick={() =>
                  void work(async () => {
                    await mutate(`/demo/sessions/${key}/stream/resume`, {});
                    await stream.refetch();
                  })
                }
              >
                Возобновить с прежнего места
              </Button>
            </div>
            <p>
              Нужны права аналитика/администратора и потоковые сервисы. Архивное
              исследование работает и без них. Переключение страницы не
              останавливает запущенное задание.
            </p>
          </details>
        </>
      )}
    </main>
  );
}
function formatC(value: number | null | undefined) {
  return value == null ? "Нет данных" : `${value.toFixed(4)} °C`;
}
