import { useEffect, useMemo, useRef, useState } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import {
  Button,
  InlineNotification,
  Select,
  SelectItem,
  Tag,
} from "@carbon/react";
import { motion } from "motion/react";
import { Link, useSearchParams } from "react-router-dom";
import {
  api,
  mutate,
  type RecordMeta,
  type RecordDetail,
  type Slices,
  type Replay,
} from "./api";
import { Heatmap, LineChart } from "./charts";
import { CoordinateInput } from "./coordinate-input";
import { RegionPanel, regionLink } from "./workbench";
import type { components } from "./api-schema";
import { processedProfile } from "./processing";
import { AnalysisJob } from "./analysis-job";
import { control, useAccount } from "./access";
function nearest(axis: number[], value: number) {
  let a = 0,
    b = axis.length - 1;
  while (a < b) {
    const m = Math.floor((a + b) / 2);
    if (axis[m] < value) a = m + 1;
    else b = m;
  }
  return a > 0 && Math.abs(axis[a - 1] - value) < Math.abs(axis[a] - value)
    ? a - 1
    : a;
}
export default function Explorer() {
  const account=useAccount();
  const capabilities=useQuery({queryKey:['service-capabilities',account?.id],queryFn:({signal})=>control<{legacy_profile:{available:boolean;method_version:string|null}}>("/service-capabilities",undefined,signal),enabled:!!account,refetchInterval:10000});
  const [params, setParams] = useSearchParams();
  const id = params.get("record") ?? "5";
  const [regionOpen, setRegionOpen] = useState(params.has("r0"));
  const activeRecord = useRef(id);
  activeRecord.current = id;
  const [replayGeneration, setReplayGeneration] = useState(0);
  const [replayBusy, setReplayBusy] = useState(false);
  const [speed, setSpeed] = useState<1 | 10 | 60>(1);
  const scope = useRef(0);
  const [replayLost, setReplayLost] = useState(false);
  const pendingSeek = useRef<{ key: string; row: number } | undefined>(
    undefined,
  );
  const wasPlaying = useRef(false);
  const [row, setRow] = useState(0),
    [column, setColumn] = useState(0),
    [zoom, setZoom] = useState<{
      x0: number;
      x1: number;
      t0: number;
      t1: number;
    }>(),
    [fullscreen, setFullscreen] = useState(false);
  const [replay, setReplay] = useState<Replay>(),
    replayId = useRef<string | undefined>(undefined),
    [error, setError] = useState(""),
    [method, setMethod] = useState("median_x3"),
    [processing, setClean] = useState<{
      record: string;
      row: number;
      values: (number | null)[];
      runId: string;
      key: string;
    }>(),
    [busy, setBusy] = useState(false);
  const catalog = useQuery({
    queryKey: ["records"],
    queryFn: ({ signal }) => api<RecordMeta[]>("/records", signal),
  });
  const detail = useQuery({
    queryKey: ["record", id],
    queryFn: ({ signal }) => api<RecordDetail>("/records/" + id, signal),
  });
  const slice = useQuery({
    queryKey: ["slice", id, detail.data?.source_sha256, row, column],
    queryFn: ({ signal }) =>
      api<Slices>(
        `/records/${id}/slices?row=${row}&column=${column}&source_sha256=${detail.data!.source_sha256}`,
        signal,
      ),
    enabled:
      !!detail.data &&
      (!params.get("source_sha256") ||
        params.get("source_sha256") === detail.data.source_sha256),
    gcTime: 10000,
    placeholderData: (previous, query) =>
      query?.queryKey[1] === id &&
      query.queryKey[2] === detail.data?.source_sha256
        ? keepPreviousData(previous)
        : undefined,
  });
  const processingContext = JSON.stringify([
    id,
    row,
    column,
    method,
    zoom,
    detail.data?.source_sha256,
  ]);
  const activeProcessing = useRef(processingContext);
  activeProcessing.current = processingContext;
  const clean = processing?.key === processingContext ? processing : undefined;
  const d = detail.data,
    s = slice.data;
  useEffect(() => {
    if (!fullscreen) return;
    const close = (e: KeyboardEvent) => {
      if (e.key === "Escape") setFullscreen(false);
    };
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, [fullscreen]);
  useEffect(() => {
    setRow(Math.max(0, Number(params.get("row")) || 0));
    setColumn(Math.max(0, Number(params.get("column")) || 0));
    setZoom(undefined);
    setClean(undefined);
    setError("");
    setReplay(undefined);
    setReplayLost(false);
    setReplayBusy(false);
    setSpeed(1);
  }, [id]);
  useEffect(() => {
    if (!d) return;
    const r = Number(params.get("row")),
      c = Number(params.get("column"));
    if (params.has("row") && Number.isInteger(r) && r >= 0 && r < d.shape[0])
      setRow(r);
    else if (params.has("row")) {
      setRow(0);
      setError("Индекс профиля в ссылке вне записи. Открыт первый профиль.");
    }
    if (params.has("column") && Number.isInteger(c) && c >= 0 && c < d.shape[1])
      setColumn(c);
    else if (params.has("column")) {
      setColumn(0);
      setError(
        "Индекс координаты в ссылке вне записи. Открыта первая координата.",
      );
    }
    if (params.has("r0")) {
      const r0 = Number(params.get("r0")),
        r1 = Number(params.get("r1")),
        c0 = Number(params.get("c0")),
        c1 = Number(params.get("c1"));
      if (
        [r0, r1, c0, c1].every(Number.isInteger) &&
        r0 >= 0 &&
        r0 < r1 &&
        r1 <= d.shape[0] &&
        c0 >= 0 &&
        c0 + 1 < c1 &&
        c1 <= d.shape[1]
      ) {
        setZoom({
          x0: d.x_m[c0],
          x1: d.x_m[c1 - 1],
          t0: d.time_s[r0],
          t1: Math.max(d.time_s[r1 - 1], d.time_s[r0] + 1),
        });
        setRegionOpen(true);
      }
    }
  }, [d, params]);
  useEffect(() => {
    if (!d) return;
    scope.current++;
    const old = replayId.current;
    replayId.current = undefined;
    wasPlaying.current = false;
    pendingSeek.current = undefined;
    if (old)
      void mutate("/replays/" + old, undefined, undefined, "DELETE").catch(
        () => {},
      );
    const release = () => {
      scope.current++;
      const key = replayId.current;
      if (key) {
        void mutate("/replays/" + key, undefined, undefined, "DELETE").catch(
          () => {},
        );
        replayId.current = undefined;
      }
    };
    window.addEventListener("pagehide", release);
    return () => {
      window.removeEventListener("pagehide", release);
      release();
    };
  }, [id, d, replayGeneration]);
  useEffect(() => {
    if (!replay?.id) return;
    const events = new EventSource(`/api/v1/replays/${replay.id}/events`);
    events.addEventListener("cursor", (e) => {
      const state = JSON.parse((e as MessageEvent).data) as Replay;
      setReplay((prev) => ({ ...state, id: prev?.id }));
      if (
        (state.playing || wasPlaying.current) &&
        pendingSeek.current?.key !== replay.id
      )
        setRow(state.row);
      wasPlaying.current = state.playing;
    });
    events.onerror = () => {
      setError("Replay: связь прервана, выполняется повторное подключение.");
      void api<Replay>(`/replays/${replay.id}`).catch((e) => {
        if (e.status === 404 && replayId.current === replay.id) {
          events.close();
          setReplayLost(true);
          setError(
            "Сессия replay завершена или сервер перезапущен. Можно создать новую сессию.",
          );
        }
      });
    };
    events.onopen = () => setError("");
    return () => events.close();
  }, [replay?.id]);
  const viewport =
    zoom ??
    (d
      ? {
          x0: d.x_bounds[0],
          x1: d.x_bounds[1],
          t0: d.time_bounds[0],
          t1: d.time_bounds[1],
        }
      : undefined);
  const path = useMemo(
    () =>
      viewport
        ? `/records/${id}/tiles?${new URLSearchParams({ ...Object.fromEntries(Object.entries(viewport).map(([k, v]) => [k, String(v)])), width: matchMedia("(max-width: 672px)").matches ? "96" : "192", height: "128" })}`
        : "",
    [id, viewport?.x0, viewport?.x1, viewport?.t0, viewport?.t1],
  );
  async function command(action: string, value?: number) {
    let key = replayId.current;
    const currentScope = scope.current;
    if (replayLost || replayBusy) return;
    if (!key && action !== "play") return;
    try {
      if (!key) {
        setReplayBusy(true);
        const created = await mutate<Replay & { id: string }>("/replays", {
          record_id: id,
        });
        if (currentScope !== scope.current) {
          void mutate(
            "/replays/" + created.id,
            undefined,
            undefined,
            "DELETE",
          ).catch(() => {});
          return;
        }
        key = created.id;
        replayId.current = key;
        await mutate(`/replays/${key}/commands`, { action: "seek", row });
        await mutate(`/replays/${key}/commands`, { action: "speed", speed });
      }
      if (action === "seek") pendingSeek.current = { key, row: value! };
      const state = await mutate<Replay>(`/replays/${key}/commands`, {
        action,
        ...(action === "seek"
          ? { row: value }
          : action === "speed"
            ? { speed: value }
            : {}),
      });
      if (key !== replayId.current) return;
      if (action === "seek" && pendingSeek.current?.row !== value) return;
      setReplay({ ...state, id: key });
      setRow(state.row);
    } catch (e) {
      setError(String(e));
    } finally {
      if (currentScope === scope.current) setReplayBusy(false);
      if (
        action === "seek" &&
        pendingSeek.current &&
        pendingSeek.current?.key === key &&
        pendingSeek.current.row === value
      )
        pendingSeek.current = undefined;
    }
  }
  function chooseRow(value: number) {
    setRow(value);
    setClean(undefined);
    void command("seek", value);
  }
  if (
    d &&
    params.get("source_sha256") &&
    params.get("source_sha256") !== d.source_sha256
  )
    return (
      <main className="page">
        <h1>Версия источника изменилась</h1>
        <p>
          Эта ссылка относится к другому SHA. Исходный результат нельзя
          сопоставлять с заменённой записью.
        </p>
        <Link to="/demo">Вернуться к репетиции</Link>
      </main>
    );
  if (catalog.error || detail.error)
    return (
      <main className="page">
        <h1>Источник недоступен</h1>
        <InlineNotification
          kind="error"
          title="Не удалось открыть записи"
          subtitle={String(catalog.error ?? detail.error)}
          hideCloseButton
        />
        <Button
          onClick={() => {
            void catalog.refetch();
            void detail.refetch();
          }}
        >
          Повторить
        </Button>
      </main>
    );
  return (
    <main className="page">
      <div className="page-title">
        <div>
          <p className="eyebrow">Рабочая область / Исследование</p>
          <h1>Температурное поле</h1>
          {d&&<Link className="text-link" to={'/works?'+new URLSearchParams({record:id,source_sha256:d.source_sha256,row:String(row),column:String(column),r0:String(zoom?nearest(d.time_s,zoom.t0):Math.max(0,row-24)),r1:String(zoom?nearest(d.time_s,zoom.t1)+1:Math.min(d.shape[0],row+25)),c0:String(zoom?nearest(d.x_m,zoom.x0):Math.max(0,column-128)),c1:String(zoom?nearest(d.x_m,zoom.x1)+1:Math.min(d.shape[1],Math.max(2,column+129)))})}>Начать работу с этим участком →</Link>}
        </div>
        <Link to={`/review?source=${id}.csv`} className="text-link">
          Перейти к экспертной проверке →
        </Link>
      </div>
      <div className="record-strip">
        <Select
          id="record"
          labelText="Запись"
          value={id}
          onChange={(e) => setParams({ record: e.target.value })}
        >
          {(catalog.data ?? []).map((r) => (
            <SelectItem
              key={r.id}
              value={r.id}
              text={`${r.source} · ${r.shape[0].toLocaleString("ru")} × ${r.shape[1].toLocaleString("ru")}`}
            />
          ))}
        </Select>
        <div>
          <span className="label">Период наблюдения</span>
          <strong>
            {d?.first_timestamp} → {d?.last_timestamp}
          </strong>
        </div>
        <div>
          <span className="label">Качество исходных данных</span>
          <strong>
            {d
              ? `${(d.valid_fraction * 100).toFixed(1)}% валидных · ${d.segments} сегм.`
              : "Загрузка…"}
          </strong>
        </div>
        <Tag type="gray">
          {d?.time_kind === "local_unzoned"
            ? "Часовой пояс неизвестен"
            : "Смещение из источника"}
        </Tag>
      </div>
      {(slice.error || detail.error || catalog.error) && (
        <div role="group" aria-label="Восстановление данных">
          <InlineNotification
            kind="warning"
            title="Данные не обновлены"
            subtitle={String(slice.error ?? detail.error ?? catalog.error)}
            hideCloseButton
          />
          <Button
            kind="tertiary"
            onClick={() => {
              void detail.refetch();
              void catalog.refetch();
              void slice.refetch();
            }}
          >
            Обновить запись
          </Button>
        </div>
      )}
      {error && (
        <InlineNotification
          kind="warning"
          title="Состояние подключения"
          subtitle={error}
          onCloseButtonClick={() => setError("")}
        />
      )}
      <div className="workspace">
        <section className={fullscreen ? "plots expanded" : "plots"}>
          <div className="plot-toolbar">
            <span>{zoom ? "Приближённый участок" : "Обзор всей записи"}</span>
            <div>
              <Button
                kind="ghost"
                size="sm"
                disabled={!d}
                onClick={() => {
                  if (!d) return;
                  const x = d.x_m[column],
                    t = d.time_s[row];
                  setZoom({
                    x0: Math.max(d.x_bounds[0], x - 30),
                    x1: Math.min(d.x_bounds[1], x + 30),
                    t0: Math.max(0, t - 300),
                    t1: Math.min(d.time_bounds[1], t + 300),
                  });
                }}
              >
                Приблизить выбранную точку
              </Button>
              <Button
                kind="ghost"
                size="sm"
                disabled={!zoom}
                onClick={() => setZoom(undefined)}
              >
                Вся запись
              </Button>
              <Button
                kind="tertiary"
                size="sm"
                onClick={() => setFullscreen((v) => !v)}
              >
                {fullscreen ? "Вернуть панели" : "Развернуть поле"}
              </Button>
            </div>
          </div>
          {d && path && (
            <Heatmap
              path={path}
              versionKey={d.source_sha256}
              expectedSourceSha={d.source_sha256}
              cursor={[d.x_m[s?.column ?? column], d.time_s[s?.row ?? row]]}
              onSelect={(x, t) => {
                setColumn(nearest(d.x_m, x));
                chooseRow(nearest(d.time_s, t));
              }}
            />
          )}
          <div className="replay-bar" data-replay-id={replay?.id}>
            {replayLost && (
              <Button
                size="sm"
                kind="tertiary"
                onClick={() => {
                  setReplay(undefined);
                  setReplayLost(false);
                  setReplayGeneration((v) => v + 1);
                }}
              >
                Восстановить replay
              </Button>
            )}
            <Button
              kind="secondary"
              size="sm"
              disabled={!d || replayLost || replayBusy}
              onClick={() => void command(replay?.playing ? "pause" : "play")}
            >
              {replayBusy
                ? "Подготовка replay…"
                : replay?.playing
                  ? "Пауза"
                  : "Воспроизвести"}
            </Button>
            <Select
              id="speed"
              labelText="Скорость replay"
              size="sm"
              value={replay?.speed ?? speed}
              disabled={replayBusy}
              onChange={(e) => {
                const next = Number(e.target.value) as 1 | 10 | 60;
                setSpeed(next);
                void command("speed", next);
              }}
            >
              <SelectItem value={1} text="1×" />
              <SelectItem value={10} text="10×" />
              <SelectItem value={60} text="60×" />
            </Select>
            <span className="mono">{s?.timestamp ?? "—"}</span>
            <span>Историческая запись · полное поле остаётся видимым</span>
          </div>
          {s && (
            <div className="line-grid">
              <LineChart
                title={
                  clean?.record === id && clean.row === row
                    ? "Очищенный пространственный профиль"
                    : "Пространственный профиль"
                }
                x={s.x_m}
                y={
                  clean?.record === id && clean.row === row
                    ? clean.values
                    : s.profile
                }
                xLabel="Расстояние, м"
              />
              <LineChart
                title="Температура выбранной точки"
                x={s.time_s}
                y={s.trend}
                xLabel="Время от начала, с"
              />
            </div>
          )}
        </section>
        <motion.aside
          className="inspector"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.16 }}
        >
          <h2>Точный срез</h2>
          {slice.isPlaceholderData && (
            <p role="status">
              Обновляется выбранный срез… Пока показано последнее подтверждённое
              наблюдение.
            </p>
          )}
          <p>
            Выберите точку на поле или задайте профиль и расстояние. Значения
            ниже взяты из исходной записи.
          </p>
          <CoordinateInput
            key={id + "-row"}
            id="row"
            disabled={replayBusy}
            label="Профиль (индекс с 0)"
            integer
            min={0}
            max={(d?.shape[0] ?? 1) - 1}
            value={row}
            onCommit={chooseRow}
            onPrevious={() => chooseRow(Math.max(0, row - 1))}
            onNext={() => chooseRow(Math.min((d?.shape[0] ?? 1) - 1, row + 1))}
          />
          <CoordinateInput
            key={id + "-column"}
            id="column"
            label="Расстояние по волокну, м"
            min={d?.x_bounds[0] ?? 0}
            max={d?.x_bounds[1] ?? 0}
            value={d?.x_m[column] ?? 0}
            onCommit={(v) => {
              if (d) setColumn(nearest(d.x_m, v));
            }}
            onPrevious={() => setColumn(Math.max(0, column - 1))}
            onNext={() =>
              setColumn(Math.min((d?.shape[1] ?? 1) - 1, column + 1))
            }
          />
          <dl className="readings">
            <dt>Расстояние</dt>
            <dd>{s?.distance_m.toFixed(3)} м</dd>
            <dt>Температура</dt>
            <dd>
              {s?.temperature === null
                ? "Нет валидного измерения"
                : `${s?.temperature.toFixed(4) ?? "—"} °C`}
            </dd>
            <dt>Причина маски</dt>
            <dd>
              {s?.invalid_reason === 0
                ? "Валидное наблюдение"
                : s?.invalid_reason === 1
                  ? "Sentinel −300"
                  : s?.invalid_reason === 2
                    ? "Нечисловое значение"
                    : s?.invalid_reason === 4
                      ? "Отсутствующий хвост"
                      : "—"}
            </dd>
            <dt>Время источника</dt>
            <dd>{s?.timestamp}</dd>
          </dl>
          {slice.error && (
            <InlineNotification
              kind="error"
              title="Срез не загружен"
              subtitle={String(slice.error)}
              hideCloseButton
            />
          )}
          <details className="analysis-tools">
            <summary>Очистка и анализ участка</summary>
            <h2>Очистка профиля</h2>
            <p>Только по вашему действию. Исходная запись сохраняется.</p>
            {account && !capabilities.data?.legacy_profile.available && <InlineNotification kind="warning" title="Очистка профиля сейчас недоступна" subtitle="Архив и сохранённые результаты доступны. Состояние сервиса обновится автоматически." hideCloseButton />}
            <Select
              id="method"
              labelText="Метод ML API"
              value={method}
              onChange={(e) => {
                setMethod(e.target.value);
                setClean(undefined);
              }}
            >
              <SelectItem value="median_x3" text="Медиана по 3 точкам" />
              <SelectItem value="median_x5" text="Медиана по 5 точкам" />
            </Select>
            <Button
              size="sm"
              disabled={busy || !s || (!!account && !capabilities.data?.legacy_profile.available)}
              onClick={async () => {
                setBusy(true);
                try {
                  const requested = processingContext;
                  const r = await mutate<
                    components["schemas"]["DenoiseResult"]
                  >("/denoise", { record_id: id, row, method });
                  if (activeProcessing.current !== requested || !d || !s)
                    return;
                  const run = await api<components["schemas"]["RunDetail"]>(
                    "/runs/" + encodeURIComponent(r.run_id),
                  );
                  if (activeProcessing.current !== requested) return;
                  const values = processedProfile(
                    run,
                    {
                      record_id: id,
                      row_start: row,
                      row_end: row + 1,
                      column_start: 0,
                      column_end: s.profile.length,
                    },
                    row,
                    d.source_sha256,
                    s.profile,
                  );
                  setClean({
                    record: id,
                    row,
                    values,
                    runId: r.run_id,
                    key: requested,
                  });
                } catch (e) {
                  setError(String(e));
                } finally {
                  setBusy(false);
                }
              }}
            >
              {busy ? "Обработка…" : "Применить к профилю"}
            </Button>
            {clean && (
              <Button
                kind="ghost"
                size="sm"
                onClick={() => setClean(undefined)}
              >
                Показать исходный профиль
              </Button>
            )}
            {clean && s && (
              <p>
                <code>{clean.runId}</code>
                <br />
                <Link
                  className="text-link"
                  to={`${regionLink({ record_id: id, row_start: row, row_end: row + 1, column_start: 0, column_end: s.profile.length })}&run_id=${encodeURIComponent(clean.runId)}`}
                >
                  Открыть выбранную обработку и отчёт →
                </Link>
              </p>
            )}
            {d && (
              <AnalysisJob
                key={id + d.source_sha256}
                record={id}
                sha={d.source_sha256}
                shape={d.shape}
                row={row}
                column={column}
              />
            )}
          </details>
          <details className="provenance">
            <summary>Происхождение данных</summary>
            <p>{d?.source}</p>
            <code>{d?.source_sha256}</code>
            <p>
              Исходные координаты и время. Пропуски не интерполируются. Цвет
              ограничен p1/p99; точное значение выше может выходить за шкалу.
            </p>
          </details>
        </motion.aside>
      </div>
      <div className="action-row">
        <Button kind="tertiary" onClick={() => setRegionOpen((v) => !v)}>
          {regionOpen
            ? "Скрыть карточку участка"
            : "Исследовать и сохранить участок"}
        </Button>
        <Link className="text-link" to="/candidates">
          Открыть подготовленные кандидаты →
        </Link>
      </div>
      {regionOpen && d && (
        <RegionPanel
          key={id + params.get("r0")}
          initial={{
            record_id: id,
            row_start: Math.max(0, Number(params.get("r0")) || row - 30),
            row_end: Math.min(d.shape[0], Number(params.get("r1")) || row + 30),
            column_start: Math.max(0, Number(params.get("c0")) || column - 100),
            column_end: Math.min(
              d.shape[1],
              Number(params.get("c1")) || column + 100,
            ),
          }}
        />
      )}
    </main>
  );
}
