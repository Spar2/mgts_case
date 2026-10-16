import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Button,
  Select,
  SelectItem,
  InlineNotification,
  Tag,
  Modal,
} from "@carbon/react";
import { Link } from "react-router-dom";
import { control, useAccount } from "./access";
import { InvitationsPanel } from "./account-security";
import { api, type RecordMeta, type RecordDetail } from "./api";
import { Heatmap } from "./charts";
import { analysisParams, type AnalysisRegion } from "./analysis-context";

type Job = {
  id: string;
  kind: string;
  status: string;
  attempts: number;
  error: string | null;
  payload: {
    record_id?: string;
    region?: AnalysisRegion;
    source_sha256?: string;
  };
  result?: { run_id?: string };
  updated_at: number;
};
type Stream = {
  id: string;
  record_id: string;
  accepted: number;
  persisted: number;
  verified_persisted: number;
  analyzed: number;
  status: string;
  source_sha256: string;
  snapshot_version: number;
  backlog_profiles?: number;
  last_accept_ms?: number;
  verified_cells?: number;
};
const roleNames: Record<string, string> = {
  viewer: "Наблюдатель",
  expert: "Эксперт",
  analyst: "Аналитик",
  admin: "Администратор",
};
const jobNames: Record<string, string> = {
  prepare_views: "Подготовка обзоров",
  stream_replay: "Передача записи",
  import_csv: "Импорт CSV",
  detect: "Анализ поля",
  analysis_field: "Расчёт поля и числовых слоёв",
};
const statusNames: Record<string, string> = {
  queued: "В очереди",
  running: "Выполняется",
  complete: "Завершено",
  failed: "Ошибка",
  cancelled: "Отменено",
  starting: "Запускается",
  interrupted: "Прервано",
  storage_lag: "Ожидает сохранения",
};
export function StreamField({ snapshot }: { snapshot: Stream }) {
  const detail = useQuery({
    queryKey: ["record", snapshot.record_id],
    queryFn: ({ signal }) =>
      api<RecordDetail>("/records/" + snapshot.record_id, signal),
  });
  const d = detail.data;
  if (!d || (snapshot.verified_persisted ?? -1) < 1)
    return <p>Ожидаем подтверждённые профили ClickHouse.</p>;
  const end = Math.min(snapshot.verified_persisted, d.shape[0] - 1),
    start = Math.max(0, end - 49);
  const path = `/records/${snapshot.record_id}/tiles?x0=${d.x_m[0]}&x1=${d.x_m.at(-1)}&t0=${d.time_s[start]}&t1=${d.time_s[end]}&width=128&height=128`;
  return (
    <>
      <p>
        Подтверждённое окно: профили {start}–{end}. Источник SHA:{" "}
        <code>{snapshot.source_sha256}</code>
      </p>
      <Heatmap
        path={path}
        versionKey={snapshot.source_sha256}
        expectedSourceSha={snapshot.source_sha256}
      />
      <Link
        className="text-link"
        to={`/research?record=${snapshot.record_id}&row=${end}&column=0&r0=${start}&r1=${end + 1}&c0=0&c1=${d.shape[1]}`}
      >
        Исследовать подтверждённый участок →
      </Link>
    </>
  );
}
export default function ControlPlane() {
  const submission = useRef<{ digest: string; key: string } | null>(null);
  const account = useAccount(),
    [error, setError] = useState(""),
    [record, setRecord] = useState("5"),
    [selected, setSelected] = useState(""),
    [offset, setOffset] = useState(0),
    [submitting, setSubmitting] = useState(false);
  const [actionJob, setActionJob] = useState(""),
    [accountBusy, setAccountBusy] = useState(false);
  const [accessChange, setAccessChange] = useState<{
    id: string;
    name: string;
    enabled: boolean;
  } | null>(null);
  const [streamConnection, setStreamConnection] = useState("");
  const records = useQuery({
    queryKey: ["records"],
    queryFn: ({ signal }) => api<RecordMeta[]>("/records", signal),
  });
  const status = useQuery({
    queryKey: ["platform-status"],
    queryFn: () =>
      control<{
        jobs: { status: string; count: number }[];
        data_path: string;
        capabilities: { stream_replay: boolean; prepare_views: boolean };
        services: Record<string, boolean | number | null>;
      }>("/status"),
    enabled: !!account,
    refetchInterval: 3000,
  });
  const jobs = useQuery({
    queryKey: ["platform-jobs", offset],
    queryFn: () =>
      control<{ items: Job[]; next_offset: number | null }>(
        `/jobs?limit=20&offset=${offset}`,
      ),
    enabled: !!account,
    refetchInterval: 3000,
  });
  const stream = useQuery({
    queryKey: ["stream", selected],
    queryFn: () => control<Stream>("/stream/" + selected),
    enabled: !!selected,
    retry: false,
    refetchInterval: selected ? 5000 : false,
  });
  const refresh = stream.refetch;
  useEffect(() => {
    if (!selected) return;
    setStreamConnection("Подключаемся к потоку…");
    const sse = new EventSource("/api/v2/stream/" + selected + "/events");
    sse.addEventListener("snapshot", () => {
      setStreamConnection("Связь с потоком установлена");
      void refresh();
    });
    sse.onerror = () =>
      setStreamConnection(
        "Связь с потоком прервана. Переподключаемся; показан последний подтверждённый снимок.",
      );
    return () => sse.close();
  }, [selected, refresh]);
  const users = useQuery({
    queryKey: ["platform-users"],
    queryFn: () =>
      control<
        {
          id: string;
          username: string;
          display_name: string;
          role: string;
          enabled: boolean;
        }[]
      >("/accounts"),
    enabled: account?.role === "admin",
  });
  const differences = useQuery({
    queryKey: ["disagreements"],
    queryFn: () =>
      control<{
        items: {
          review_id: string;
          source: string;
          disagreement_fraction: number;
          methods: string[];
        }[];
      }>("/disagreements?limit=20"),
    enabled: !!account && ["admin", "analyst"].includes(account.role),
    staleTime: 60000,
  });
  async function run(kind: string) {
    setError("");
    setSubmitting(true);
    try {
      const body = { kind, record_id: record },
        digest = JSON.stringify(body);
      if (submission.current?.digest !== digest)
        submission.current = { digest, key: crypto.randomUUID() };
      const job = await control<Job>(
        "/jobs",
        body,
        undefined,
        submission.current.key,
      );
      submission.current = null;
      if (kind === "stream_replay") setSelected(job.id);
      await jobs.refetch();
    } catch (e) {
      setError(String(e));
    } finally {
      setSubmitting(false);
    }
  }
  async function jobAction(id: string, action: "cancel" | "resume") {
    setActionJob(id);
    setError("");
    try {
      await control("/jobs/" + id + "/" + action, {});
      if (action === "resume") setSelected(id);
      await jobs.refetch();
    } catch (e) {
      setError(String(e));
    } finally {
      setActionJob("");
    }
  }
  if (!account)
    return (
      <main className="page">
        <h1>Система</h1>
        <p>
          Командные функции требуют PostgreSQL и отдельного worker. Запустите
          профиль из инструкции production-пилота.
        </p>
      </main>
    );
  return (
    <main className="page">
      <div className="page-heading">
        <h1>Система и путь измерения</h1>
        <Tag type="blue">Реальные CSV · воспроизведение</Tag>
      </div>
      <p>{status.data?.data_path}</p>
      <p>
        Передача измерений не является подключением оборудования. DSP не
        подтверждает физическую причину события.
      </p>
      {(error || status.error) && (
        <InlineNotification
          kind="error"
          title="Действие не выполнено"
          subtitle={error || String(status.error)}
          hideCloseButton
        />
      )}
      <section className="region-panel">
        <h2>Поток измерений</h2>
        <div className="action-row">
          <Select
            id="stream-record"
            labelText="Запись для передачи"
            value={record}
            onChange={(e) => setRecord(e.target.value)}
          >
            {records.data?.map((r) => (
              <SelectItem key={r.id} value={r.id} text={r.source} />
            ))}
          </Select>
          {["admin", "analyst"].includes(account.role) && (
            <>
              <Button
                disabled={
                  submitting || !status.data?.capabilities.stream_replay
                }
                onClick={() => void run("stream_replay")}
              >
                Передать запись через стек
              </Button>
              <Button
                disabled={
                  submitting || !status.data?.capabilities.prepare_views
                }
                kind="tertiary"
                onClick={() => void run("prepare_views")}
              >
                Подготовить обзоры
              </Button>
            </>
          )}
        </div>
        {selected && <p role="status">{streamConnection}</p>}
        {selected && !stream.data && (
          <p>
            Ожидаем первый снимок выбранной задачи. Он появится после запуска
            stream-worker.
          </p>
        )}
        {status.data && !status.data.capabilities.stream_replay && (
          <p role="status">
            Передача сейчас недоступна. Проверьте stream-worker, приёмник, Redis
            и ClickHouse. Архив можно исследовать независимо от передачи.
          </p>
        )}
        {status.data?.services && (
          <p>
            Вычисления:{" "}
            {status.data.services.compute_worker ? "доступны" : "недоступны"} ·
            Передача:{" "}
            {status.data.services.stream_worker
              ? "worker работает"
              : "worker недоступен"}{" "}
            · Redis: {status.data.services.redis ? "доступен" : "недоступен"} ·
            ClickHouse:{" "}
            {status.data.services.clickhouse ? "доступен" : "недоступен"}
          </p>
        )}
        {stream.data && (
          <>
            <div className="readings">
              <p>Принято: {stream.data.accepted + 1}</p>
              <p>Сохранено: {stream.data.persisted + 1}</p>
              <p>DSP: {stream.data.analyzed + 1}</p>
              <p>
                Состояние:{" "}
                {statusNames[stream.data.status] ?? stream.data.status}
              </p>
              <p>
                Ожидают записи:{" "}
                {stream.data.backlog_profiles ??
                  Math.max(0, stream.data.accepted - stream.data.persisted)}
              </p>
              <p>
                Последний приём: {stream.data.last_accept_ms?.toFixed(1) ?? "—"}{" "}
                мс
              </p>
              <p>
                Сверено ячеек:{" "}
                {stream.data.verified_cells?.toLocaleString("ru") ?? "—"}
              </p>
            </div>
            <StreamField snapshot={stream.data} />
          </>
        )}
      </section>
      <section className="region-panel">
        <h2>Фоновые задачи</h2>
        {jobs.isPending && <p role="status">Загружаем задачи…</p>}
        {jobs.isError && (
          <p role="alert">
            Не удалось загрузить задачи. Проверьте связь с сервером.
          </p>
        )}
        {jobs.data?.items.length === 0 && (
          <p>
            На этой странице пока нет задач. Запустите передачу записи или
            подготовку обзоров выше.
          </p>
        )}
        {jobs.data?.items.map((job) => (
          <article className="job-row" key={job.id}>
            <div>
              <strong>{jobNames[job.kind] ?? job.kind}</strong> ·{" "}
              {statusNames[job.status] ?? job.status} · попытка {job.attempts}
              <p>{job.error}</p>
              <code>{job.id}</code>
            </div>
            <div className="action-row">
              {job.kind === "analysis_field" &&
                job.payload.region &&
                job.payload.source_sha256 && (
                  <>
                    <Link
                      to={`/research?${new URLSearchParams({ record: job.payload.region.record_id, source_sha256: job.payload.source_sha256, job_id: job.id, row: String(job.payload.region.row_start), column: String(job.payload.region.column_start), r0: String(job.payload.region.row_start), r1: String(job.payload.region.row_end), c0: String(job.payload.region.column_start), c1: String(job.payload.region.column_end) })}`}
                    >
                      Открыть задачу расчёта
                    </Link>
                    {job.result?.run_id && (
                      <Link
                        to={`/analysis?${analysisParams({ id: job.result.run_id, source_sha256: job.payload.source_sha256 }, { region: job.payload.region, row: job.payload.region.row_start, column: job.payload.region.column_start, layer: "processed_temperature_c", zone: "" })}`}
                      >
                        Открыть числовой результат
                      </Link>
                    )}
                  </>
                )}
              {job.kind === "stream_replay" && (
                <Button kind="ghost" onClick={() => setSelected(job.id)}>
                  Открыть поток
                </Button>
              )}
              {["queued", "running"].includes(job.status) &&
                ["admin", "analyst"].includes(account.role) && (
                  <Button
                    kind="tertiary"
                    disabled={actionJob === job.id}
                    onClick={() => void jobAction(job.id, "cancel")}
                  >
                    Запросить отмену
                  </Button>
                )}
              {job.kind === "stream_replay" &&
                ["failed", "cancelled"].includes(job.status) &&
                ["admin", "analyst"].includes(account.role) && (
                  <Button
                    kind="tertiary"
                    disabled={actionJob === job.id}
                    onClick={() => void jobAction(job.id, "resume")}
                  >
                    Возобновить передачу
                  </Button>
                )}
            </div>
          </article>
        ))}
        <div className="action-row">
          <Button
            kind="ghost"
            disabled={!offset}
            onClick={() => setOffset(Math.max(0, offset - 20))}
          >
            Предыдущие
          </Button>
          <Button
            kind="ghost"
            disabled={jobs.data?.next_offset == null}
            onClick={() => setOffset(jobs.data!.next_offset!)}
          >
            Следующие
          </Button>
        </div>
      </section>
      <CaseQueue />
      {differences.data && (
        <section className="region-panel">
          <h2>Где методы расходятся</h2>
          <p>
            Приоритет исследования по различию гипотез. Это не калиброванная
            вероятность ошибки. В прежней очереди присутствуют исторические
            синтетические модели.
          </p>
          <div className="numeric-scroll">
            <table>
              <thead>
                <tr>
                  <th>Окно</th>
                  <th>Различие валидных ячеек</th>
                  <th>Действие</th>
                </tr>
              </thead>
              <tbody>
                {differences.data.items.map((c) => (
                  <tr key={c.review_id}>
                    <td>
                      {c.source} · {c.review_id}
                    </td>
                    <td>{(c.disagreement_fraction * 100).toFixed(2)}%</td>
                    <td>
                      <Link to={`/review?package=general&card=${c.review_id}`}>
                        Исследовать и проверить
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
      {account.role === "admin" && (
        <section className="region-panel">
          <h2>Участники проекта</h2>
          <p>
            Наблюдатель читает данные; эксперт отвечает; аналитик запускает
            расчёты; администратор управляет участниками.
          </p>
          <InvitationsPanel />
          {users.data?.map((u) => (
            <article className="job-row" key={u.id}>
              <p>
                {u.display_name} · {u.username} · {roleNames[u.role] ?? u.role}{" "}
                · {u.enabled ? "Доступ открыт" : "Доступ отключён"}
              </p>
              {u.id !== account.id && (
                <Button
                  kind="ghost"
                  onClick={() =>
                    setAccessChange({
                      id: u.id,
                      name: u.display_name,
                      enabled: u.enabled,
                    })
                  }
                >
                  {u.enabled ? "Отключить доступ" : "Восстановить доступ"}
                </Button>
              )}
            </article>
          ))}
          <Modal
            open={!!accessChange}
            modalHeading={
              accessChange?.enabled
                ? "Отключить доступ участника?"
                : "Восстановить доступ участника?"
            }
            primaryButtonText={
              accessChange?.enabled ? "Отключить доступ" : "Восстановить доступ"
            }
            secondaryButtonText="Отмена"
            primaryButtonDisabled={accountBusy}
            danger={!!accessChange?.enabled}
            onRequestClose={() => {
              if (!accountBusy) setAccessChange(null);
            }}
            onRequestSubmit={() => {
              if (!accessChange) return;
              setAccountBusy(true);
              setError("");
              void control(
                "/accounts/" +
                  accessChange.id +
                  (accessChange.enabled ? "/disable" : "/enable"),
                {},
              )
                .then(() => {
                  setAccessChange(null);
                  void users.refetch();
                })
                .catch((e) => setError(String(e)))
                .finally(() => setAccountBusy(false));
            }}
          >
            <p>
              {accessChange?.name}:{" "}
              {accessChange?.enabled
                ? "все текущие сессии будут отозваны. Сохранённые ответы останутся в журнале."
                : "участник сможет войти со своим паролем. Прежние сессии не восстанавливаются."}
            </p>
            {error && <p role="alert">{error}</p>}
          </Modal>
        </section>
      )}
    </main>
  );
}

function CaseQueue() {
  const account = useAccount();
  const [offset, setOffset] = useState(0),
    [error, setError] = useState(""),
    [assignee, setAssignee] = useState("");
  const packages = useQuery({
    queryKey: ["packages"],
    queryFn: ({ signal }) => api<{ id: string }[]>("/packages", signal),
  });
  const packageId = packages.data?.some((p) => p.id === "real-data-v2")
    ? "real-data-v2"
    : "general";
  type Case = {
    review_id: string;
    source: string;
    state: string;
    revision: number;
    assigned_to: string | null;
  };
  const cases = useQuery({
    queryKey: ["cases", packageId, offset],
    queryFn: () =>
      control<{ items: Case[]; total: number }>(
        `/case-assignments?package=${packageId}&limit=20&offset=${offset}`,
      ),
    enabled: !!account && account.role !== "viewer",
  });
  const experts = useQuery({
    queryKey: ["experts"],
    queryFn: () => control<{ id: string; display_name: string }[]>("/experts"),
    enabled: !!account && account.role !== "viewer",
  });
  if (!account || account.role === "viewer") return null;
  async function save(item: Case, state: string) {
    try {
      await control("/case-assignments", {
        package: packageId,
        review_id: item.review_id,
        state,
        assigned_to: assignee || item.assigned_to,
        expected_revision: item.revision,
        note: "",
      });
      await cases.refetch();
      setError("");
    } catch (e) {
      setError(String(e));
      await cases.refetch();
    }
  }
  return (
    <section className="region-panel">
      <h2>Работа с исследовательскими окнами</h2>
      <p>
        Назначение и обсуждение окна не создают обучающую метку. Независимые
        ответы остаются в журналах экспертов.
      </p>
      {["admin", "analyst"].includes(account.role) && (
        <Select
          id="case-expert"
          labelText="Эксперт для назначения"
          value={assignee}
          onChange={(e) => setAssignee(e.target.value)}
        >
          <SelectItem value="" text="Выберите эксперта" />
          {experts.data?.map((v) => (
            <SelectItem key={v.id} value={v.id} text={v.display_name} />
          ))}
        </Select>
      )}
      {error && <p role="alert">{error}</p>}
      {cases.data?.items.map((item) => (
        <article key={item.review_id} className="job-row">
          <p>
            <strong>
              {item.source} · {item.review_id}
            </strong>{" "}
            ·{" "}
            {(
              {
                new: "Новое",
                assigned: "Назначено",
                discuss: "На обсуждении",
                deferred: "Отложено",
                closed: "Закрыто",
              } as Record<string, string>
            )[item.state] ?? item.state}{" "}
            ·{" "}
            {experts.data?.find((e) => e.id === item.assigned_to)
              ?.display_name ?? "Не назначено"}
          </p>
          <div className="action-row">
            <Link
              className="text-link"
              to={`/review?package=${packageId}&card=${item.review_id}`}
            >
              Проверить окно →
            </Link>
            {["admin", "analyst"].includes(account.role) && (
              <>
                <Button
                  kind="ghost"
                  disabled={!assignee}
                  onClick={() => void save(item, "assigned")}
                >
                  Назначить
                </Button>
                <Button kind="ghost" onClick={() => void save(item, "discuss")}>
                  На обсуждение
                </Button>
                <Button
                  kind="ghost"
                  onClick={() => void save(item, "deferred")}
                >
                  Отложить
                </Button>
                <Button kind="ghost" onClick={() => void save(item, "closed")}>
                  Закрыть
                </Button>
              </>
            )}
          </div>
        </article>
      ))}
      <div className="action-row">
        <Button
          kind="ghost"
          disabled={!offset}
          onClick={() => setOffset(Math.max(0, offset - 20))}
        >
          Назад
        </Button>
        <Button
          kind="ghost"
          disabled={offset + 20 >= (cases.data?.total ?? 0)}
          onClick={() => setOffset(offset + 20)}
        >
          Далее
        </Button>
      </div>
    </section>
  );
}
