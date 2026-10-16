import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link, useSearchParams } from "react-router-dom";
import {
  Button,
  InlineNotification,
  Select,
  SelectItem,
  Tag,
} from "@carbon/react";
import { control, useAccount } from "./access";
import { api, type RecordMeta, type RecordDetail } from "./api";
import { StreamField } from "./control-plane";
import { LineChart } from "./charts";
import { AnalysisJob } from "./analysis-job";
import { type AnalysisRun } from "./analysis";

type Stream = {
  id: string;
  record_id: string;
  source_sha256: string;
  status: string;
  accepted: number;
  persisted: number;
  verified_persisted: number;
  analyzed: number;
  snapshot_version: number;
  analyzed_at?: number;
  updated_at: number;
  backlog_profiles?: number;
};
type Profile = {
  stream_id: string;
  snapshot_version: number;
  source_sha256: string;
  row: number;
  x_m: number[];
  raw_c: (number | null)[];
  processed_c: (number | null)[];
  elapsed_s: number;
  timestamp: string;
  verified_at: number;
};
function StreamView({ id }: { id: string }) {
  const [connection, setConnection] = useState("Подключаемся"),
    [now, setNow] = useState(Date.now());
  const snapshot = useQuery({
    queryKey: ["monitor-stream", id],
    queryFn: ({ signal }) =>
      control<Stream>("/stream/" + id, undefined, signal),
    refetchInterval: 5000,
  });
  const s = snapshot.data;
  const refresh = useRef(snapshot.refetch);
  refresh.current = snapshot.refetch;
  useEffect(() => {
    const stream = new EventSource("/api/v2/stream/" + id + "/events");
    const offline = () =>
      setConnection("Связь прервана; показан последний подтверждённый снимок");
    const online = () => {
      setConnection("Связь восстановлена; обновляем подтверждённый снимок");
      void refresh.current();
    };
    window.addEventListener("offline", offline);
    window.addEventListener("online", online);
    if (!navigator.onLine) offline();
    stream.addEventListener("snapshot", () => {
      if (!navigator.onLine) return;
      setConnection("Связь установлена");
      void refresh.current();
    });
    stream.onerror = () =>
      setConnection("Связь прервана; показан последний подтверждённый снимок");
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => {
      window.removeEventListener("offline", offline);
      window.removeEventListener("online", online);
      clearInterval(timer);
      stream.close();
    };
  }, [id]);
  const native = useQuery({
    queryKey: ["record", s?.record_id],
    queryFn: ({ signal }) =>
      api<RecordDetail>("/records/" + s!.record_id, signal),
    enabled: !!s,
  });
  const profile = useQuery({
    queryKey: ["monitor-profile", id, s?.snapshot_version],
    queryFn: ({ signal }) =>
      control<Profile>(
        `/stream/${id}/profile?snapshot_version=${s!.snapshot_version}`,
        undefined,
        signal,
      ),
    enabled: !!s && s.analyzed >= 0,
    retry: false,
    placeholderData: (previous) =>
      previous?.stream_id === id ? previous : undefined,
    gcTime: 10000,
  });
  const runs = useQuery({
    queryKey: ["monitor-results", s?.record_id],
    queryFn: ({ signal }) =>
      control<AnalysisRun[]>(
        "/result-runs?record_id=" + s!.record_id,
        undefined,
        signal,
      ),
    enabled: !!s,
    refetchInterval: 10000,
  });
  if (!s) return <p>Загружаем состояние воспроизведения…</p>;
  const d = native.data;
  const age = Math.max(
    0,
    Math.floor(now / 1000 - (s.analyzed_at ?? s.updated_at)),
  );
  const last = runs.data
    ?.filter(
      (r) =>
        r.source_sha256 === s.source_sha256 &&
        r.region.record_id === s.record_id &&
        r.region.row_end <= s.verified_persisted + 1,
    )
    .sort(
      (a, b) =>
        (Date.parse(b.calculated_at) || 0) - (Date.parse(a.calculated_at) || 0),
    )[0];
  const start = Math.max(0, s.verified_persisted - 49),
    end = s.verified_persisted + 1;
  const workLink = (run?: AnalysisRun) =>
    "/works?" +
    new URLSearchParams({
      record: s.record_id,
      source_sha256: s.source_sha256,
      row: String(run ? run.region.row_start : s.verified_persisted),
      column: String(run?.region.column_start ?? 0),
      r0: String(run?.region.row_start ?? start),
      r1: String(run?.region.row_end ?? end),
      c0: String(run?.region.column_start ?? 0),
      c1: String(run?.region.column_end ?? Math.min(d?.shape[1] ?? 2, 256)),
      origin: "monitoring",
      stream_id: s.id,
      ...(run ? { run_id: run.id } : {}),
    });
  const p = profile.data;
  return (
    <section className="monitor-stream">
      <h2>{s.record_id}.csv · воспроизведение записи</h2>
      <p role="status">
        {connection} · состояние: {s.status} · последний DSP-снимок {age} с
        назад
      </p>
      {snapshot.error && (
        <InlineNotification
          kind="warning"
          title="Обновление недоступно"
          subtitle="Ранее полученные данные остаются на экране."
          hideCloseButton
        />
      )}
      <dl className="monitor-stages">
        {[
          ["Принято", s.accepted],
          ["Записано", s.persisted],
          ["Проверено", s.verified_persisted],
          ["DSP профилей", s.analyzed],
        ].map(([label, index]) => (
          <div key={String(label)}>
            <dt>{label}</dt>
            <dd>{Math.max(0, Number(index) + 1)} профилей</dd>
          </div>
        ))}
      </dl>
      <p>
        DSP профиля — reference median_x3, не анализ поля и не классификация
        аварий.
      </p>
      <StreamField snapshot={s} />
      {p && (
        <>
          <p>
            Точный загруженный профиль: строка {p.row} · {p.elapsed_s} с от
            начала · {p.timestamp}.{" "}
            {p.snapshot_version !== s.snapshot_version
              ? "Снимок обновляется; ниже сохранён предыдущий подтверждённый профиль."
              : ""}
          </p>
          <div className="work-chart-pair">
            <LineChart
              title="Исходное измерение"
              x={p.x_m}
              y={p.raw_c}
              xLabel="Расстояние, м"
              syncKey={"monitor-" + id}
            />
            <LineChart
              title="DSP median_x3"
              x={p.x_m}
              y={p.processed_c}
              xLabel="Расстояние, м"
              syncKey={"monitor-" + id}
            />
          </div>
        </>
      )}
      {profile.error && (
        <p>
          Точный профиль временно не обновлён; неподтверждённые значения не
          показываются.
        </p>
      )}
      {d && s.verified_persisted >= 0 && (
        <>
          <p>
            <Link className="text-link" to={workLink()}>
              Начать работу с подтверждённым участком →
            </Link>
          </p>
          <AnalysisJob
            record={s.record_id}
            sha={s.source_sha256}
            shape={[s.verified_persisted + 1, d.shape[1]]}
            row={s.verified_persisted}
            column={Math.floor(d.shape[1] / 2)}
            streamId={s.id}
          />
        </>
      )}
      <section>
        <h3>Последний совместимый расчёт поля</h3>
        {last ? (
          <>
            <p>
              {last.method} · строки [{last.region.row_start},{" "}
              {last.region.row_end}) ·{" "}
              {last.region.row_end < end
                ? "Не покрывает последние поступившие профили"
                : "Покрывает текущий подтверждённый диапазон до своего конца"}
            </p>
            <p>Предложения алгоритма — гипотезы, не подтверждённые аварии.</p>
            <Link to={workLink(last)}>Начать разбор результата →</Link>
            <p>
              <Link to={"/analysis?run_id=" + last.id}>
                Открыть поле, score и зоны →
              </Link>
            </p>
          </>
        ) : (
          <p>
            Совместимый завершённый анализ поля отсутствует. Можно явно
            рассчитать подтверждённый интервал.
          </p>
        )}
      </section>
    </section>
  );
}

export default function Monitoring() {
  const account = useAccount();
  const [params, setParams] = useSearchParams(),
    id = params.get("stream_id") ?? "";
  const [record, setRecord] = useState("5"),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false);
  const attempt = useRef<{ body: object; key: string } | undefined>(undefined);
  const records = useQuery({
    queryKey: ["records"],
    queryFn: ({ signal }) => api<RecordMeta[]>("/records", signal),
  });
  const streams = useQuery({
    queryKey: ["monitor-streams", account?.id],
    queryFn: ({ signal }) =>
      control<{ items: Stream[] }>("/streams", undefined, signal),
    refetchInterval: 5000,
  });
  const caps = useQuery({
    queryKey: ["service-capabilities", account?.id],
    queryFn: ({ signal }) =>
      control<{ replay: { available: boolean } }>(
        "/service-capabilities",
        undefined,
        signal,
      ),
    refetchInterval: 5000,
  });
  async function start() {
    setBusy(true);
    setError("");
    if (!attempt.current)
      attempt.current = {
        body: { kind: "stream_replay", record_id: record, delay_s: 0.05 },
        key: crypto.randomUUID(),
      };
    try {
      const job = await control<{ id: string }>(
        "/jobs",
        attempt.current.body,
        undefined,
        attempt.current.key,
      );
      setParams({ stream_id: job.id });
      attempt.current = undefined;
      await streams.refetch();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <main className="page monitoring-page">
      <h1>Мониторинг измерений</h1>
      <Tag type="blue">Воспроизведение настоящих записей</Tag>
      <p>
        Источник оборудования не подключён. Приём в Redis не означает
        сохранение: поле показывается после подтверждения и числовой проверки
        ClickHouse.
      </p>
      <div className="lab-controls">
        <Select
          id="monitor-source"
          labelText="Запись для воспроизведения"
          value={record}
          onChange={(e) => {
            setRecord(e.target.value);
            attempt.current = undefined;
          }}
        >
          {records.data?.map((r) => (
            <SelectItem key={r.id} value={r.id} text={r.source} />
          ))}
        </Select>
        <Button
          disabled={
            busy ||
            !caps.data?.replay.available ||
            !account ||
            !["analyst", "admin"].includes(account.role)
          }
          onClick={() => void start()}
        >
          {attempt.current ? "Повторить отправку" : "Воспроизвести запись"}
        </Button>
      </div>
      {!caps.data?.replay.available && (
        <p>
          Потоковый контур сейчас недоступен. Сохранённые результаты и последний
          загруженный снимок остаются доступны.
        </p>
      )}
      {error && (
        <InlineNotification
          kind="error"
          title="Воспроизведение не запущено"
          subtitle={error}
          hideCloseButton
        />
      )}
      <Select
        id="monitor-stream"
        labelText="Сохранённая сессия"
        value={id}
        onChange={(e) => setParams({ stream_id: e.target.value })}
      >
        <SelectItem value="" text="Выберите сессию" />
        {streams.data?.items.map((s) => (
          <SelectItem
            key={s.id}
            value={s.id}
            text={`${s.record_id}.csv / ${s.status} / ${s.id.slice(0, 8)}`}
          />
        ))}
      </Select>
      {id && <StreamView key={id} id={id} />}
    </main>
  );
}
