import { useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Button, InlineNotification, TextInput } from "@carbon/react";
import { Link, useSearchParams } from "react-router-dom";
import { control, useAccount } from "./access";
import { ApiError } from "./api";
import { analysisParams } from "./analysis-context";
import type { components } from "./api-schema";

type Request = components["schemas"]["AnalysisJobRequest"];
type Job = components["schemas"]["AnalysisJobView"];
const stages: Record<string, string> = {
  queued: "В очереди",
  computing: "Расчёт",
  validating: "Проверка результата",
  complete: "Готово",
  failed: "Не выполнено",
  cancelled: "Отменено",
};

export function AnalysisJob({
  record,
  sha,
  shape,
  row,
  column,
  streamId,
}: {
  record: string;
  sha: string;
  shape: [number, number];
  row: number;
  column: number;
  streamId?: string;
}) {
  const account = useAccount();
  const queryClient = useQueryClient();
  const [params, setParams] = useSearchParams();
  const jobId = params.get("job_id") ?? "";
  const around = [
    Math.max(0, row - 30),
    Math.min(shape[0], Math.max(0, row - 30) + 60),
    Math.max(0, column - 100),
    Math.min(shape[1], Math.max(0, column - 100) + 200),
  ];
  const defaults = ["r0", "r1", "c0", "c1"].map(
    (key, i) => params.get(key) ?? String(around[i]),
  );
  const defaultKey = JSON.stringify(defaults);
  const [draft, setDraft] = useState(defaults),
    [dirty, setDirty] = useState(false),
    [busy, setBusy] = useState(false),
    [error, setError] = useState("");
  const [pending, setPending] = useState<{ body: Request; key: string }>();
  useEffect(() => {
    if (!dirty && !pending && !jobId) setDraft(JSON.parse(defaultKey));
  }, [defaultKey, dirty, pending, jobId]);
  const capability = useQuery({
    queryKey: ["analysis-capability", account?.id],
    enabled: !!account,
    queryFn: ({ signal }) =>
      control<components["schemas"]["AnalysisCapability"]>(
        "/analysis-capability",
        undefined,
        signal,
      ),
    refetchInterval: 10000,
    retry: false,
  });
  const state = useQuery({
    queryKey: ["analysis-job", account?.id, jobId],
    queryFn: ({ signal }) =>
      control<Job>(`/analysis-jobs/${jobId}`, undefined, signal),
    enabled: !!jobId && !!account,
    retry: false,
    refetchInterval: (query) => {
      if (
        query.state.error instanceof ApiError &&
        query.state.error.status >= 400 &&
        query.state.error.status < 500
      )
        return false;
      return ["complete", "failed", "cancelled"].includes(
        query.state.data?.status ?? "",
      )
        ? false
        : document.hidden
          ? 10000
          : 2000;
    },
  });
  const job = state.data;
  useEffect(() => {
    if (job?.status === "complete")
      void queryClient.invalidateQueries({ queryKey: ["analysis-catalog"] });
  }, [job?.status, queryClient]);
  const compatible =
    (!params.has("source_sha256") || params.get("source_sha256") === sha) &&
    (!job || (job.region.record_id === record && job.source_sha256 === sha));
  const active = job && ["queued", "running"].includes(job.status);
  const allowed = account && ["analyst", "admin"].includes(account.role);
  const sending = useRef(false);
  const alive = useRef(true);
  const request = useRef<AbortController | undefined>(undefined);
  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
      request.current?.abort();
    };
  }, []);
  async function send(attempt: { body: Request; key: string }) {
    if (sending.current) return;
    sending.current = true;
    setBusy(true);
    setError("");
    request.current = new AbortController();
    try {
      const result = await control<Job>(
        "/analysis-jobs",
        attempt.body,
        request.current.signal,
        attempt.key,
      );
      if (!alive.current) return;
      queryClient.setQueryData(
        ["analysis-job", account?.id, result.id],
        result,
      );
      setParams((previous) => {
        const next = new URLSearchParams(previous);
        next.set("job_id", result.id);
        return next;
      });
      setPending(undefined);
    } catch (e) {
      if (!alive.current) return;
      if (e instanceof ApiError && e.status >= 400 && e.status < 500)
        setPending(undefined);
      setError(String(e));
    } finally {
      sending.current = false;
      if (alive.current) setBusy(false);
    }
  }
  return (
    <section className="analysis-job" aria-label="Расчёт реального поля">
      <h2>Расчёт реального участка</h2>
      <p>
        median_x3 + spatial_z_halo · |Z| ≥ 3.5. Референсная обработка без
        обучения; зоны — гипотезы.
      </p>
      {capability.data && (
        <p className="mono">Код расчёта: {capability.data.method_version}</p>
      )}
      <p>
        Исходные индексы: начало включено, конец исключён. До 1 000 000 ячеек;
        действует дополнительный предел halo-сетки.
      </p>
      <div className="analysis-controls">
        {[
          "Начальная строка расчёта",
          "Конечная строка расчёта",
          "Начальный столбец расчёта",
          "Конечный столбец расчёта",
        ].map((label, i) => (
          <TextInput
            key={i}
            id={`compute-bound-${i}`}
            labelText={label}
            inputMode="numeric"
            value={draft[i]}
            disabled={busy || !!pending || !!active}
            onChange={(e) => {
              setDirty(true);
              setDraft(draft.map((v, j) => (j === i ? e.target.value : v)));
            }}
          />
        ))}
      </div>
      {!capability.data?.ready && (
        <p role="status">
          Worker расчёта сейчас недоступен. Исходные данные и сохранённые
          результаты остаются доступны.
        </p>
      )}
      {!allowed && <p>Запуск доступен аналитику и администратору.</p>}
      {(error || state.error) && (
        <InlineNotification
          kind="error"
          title="Действие не завершено"
          subtitle={error || String(state.error)}
          hideCloseButton
        />
      )}
      {pending ? (
        <>
          <p>
            Повтор отправит прежние границы с прежним ключом. Ошибка связи не
            означает, что задача не была принята.
          </p>
          <Button size="sm" disabled={busy} onClick={() => void send(pending)}>
            Повторить отправку расчёта
          </Button>
        </>
      ) : (
        <Button
          size="sm"
          disabled={
            busy ||
            !!active ||
            !allowed ||
            !capability.data?.ready ||
            !compatible
          }
          onClick={() => {
            const [r0, r1, c0, c1] = draft.map((v) =>
              /^\d+$/.test(v) ? Number(v) : NaN,
            );
            if (
              ![r0, r1, c0, c1].every(Number.isSafeInteger) ||
              !(
                0 <= r0 &&
                r0 < r1 &&
                r1 <= shape[0] &&
                0 <= c0 &&
                c0 + 1 < c1 &&
                c1 <= shape[1]
              ) ||
              (r1 - r0) * (c1 - c0) > 1000000
            ) {
              setError(
                "Выберите допустимые исходные границы, не менее двух столбцов и до 1 000 000 ячеек.",
              );
              return;
            }
            const attempt = {
              key: crypto.randomUUID(),
              body: {
                region: {
                  record_id: record,
                  row_start: r0,
                  row_end: r1,
                  column_start: c0,
                  column_end: c1,
                },
                source_sha256: sha,
                ...(streamId?{stream_id:streamId}:{}),
              },
            };
            setPending(attempt);
            void send(attempt);
          }}
        >
          Запустить расчёт поля
        </Button>
      )}
      {job && compatible && (
        <div role="status" className="job-status">
          <h3>
            {stages[job.stage] ?? "Выполняется"}
            {job.cancel_requested && job.status === "running"
              ? " · отмена запрошена"
              : ""}
          </h3>
          <p>
            Строки [{job.region.row_start}, {job.region.row_end}), столбцы [
            {job.region.column_start}, {job.region.column_end}).
          </p>
          <p className="mono">
            Задача: {job.id} · ML: {job.method_version}
          </p>
          {job.error && <p>{job.error}</p>}
          {active && allowed && (
            <Button
              size="sm"
              kind="danger--tertiary"
              disabled={busy || job.cancel_requested}
              onClick={async () => {
                setBusy(true);
                try {
                  await control(`/jobs/${job.id}/cancel`, {});
                  await state.refetch();
                } catch (e) {
                  setError(String(e));
                } finally {
                  setBusy(false);
                }
              }}
            >
              Отменить расчёт
            </Button>
          )}
          {job.run_id && (
            <Link
              to={`/analysis?${analysisParams({ id: job.run_id, source_sha256: job.source_sha256 }, { region: job.region, row: Math.min(Math.max(row, job.region.row_start), job.region.row_end - 1), column: Math.min(Math.max(column, job.region.column_start), job.region.column_end - 1), layer: "processed_temperature_c", zone: "" })}`}
            >
              Открыть результат расчёта →
            </Link>
          )}
        </div>
      )}
      {!compatible && (
        <InlineNotification
          kind="warning"
          title="Задача или SHA относятся к другому источнику"
          subtitle="Откройте нужную задачу через список заданий; результат не подставляется в эту запись."
          hideCloseButton
        />
      )}
      {jobId && !active && (
        <Button
          size="sm"
          kind="ghost"
          onClick={() =>
            setParams((previous) => {
              const next = new URLSearchParams(previous);
              next.delete("job_id");
              return next;
            })
          }
        >
          Снять выбор задачи
        </Button>
      )}
      <p>
        <Link to="/system">Задания и состояние системы →</Link>
      </p>
    </section>
  );
}
