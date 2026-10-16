import { useRef, useState } from "react";
import {
  Link,
  useNavigate,
  useParams,
  useSearchParams,
} from "react-router-dom";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Button,
  InlineNotification,
  Select,
  SelectItem,
  TextArea,
  TextInput,
  Tag,
} from "@carbon/react";
import { control, useAccount } from "./access";
import { api, type RecordDetail, type RecordMeta, type Slices } from "./api";
import { Heatmap, LineChart } from "./charts";
import { AnalysisWindow, type AnalysisRun } from "./analysis";
import { CaseForm } from "./cases";
import type { AnalysisRegion } from "./analysis-context";

type Context = {
  region: AnalysisRegion;
  source_sha256: string;
  profile: number;
  column: number;
  run_ids: string[];
  layer?: string;
};
type Work = {
  id: string;
  title: string;
  notes: string;
  context: Context;
  origin: string;
  revision: number;
  author_name: string;
  created_at: number;
  updated_at: number;
  case_ids: string[];
  source_available: boolean;
};
type ReviewRequest = {
  id: string;
  package: string;
  review_id: string;
  assigned_to: string;
  status: string;
  context: Context;
};
const bounds = ["row_start", "row_end", "column_start", "column_end"] as const;
const boundLabels = [
  "Первая строка",
  "Конец строк, не включён",
  "Первый столбец",
  "Конец столбцов, не включён",
];

function RawSignal({
  context,
  onChange,
}: {
  context: Context;
  onChange: (change: Partial<Context>) => void;
}) {
  const r = context.region;
  const detail = useQuery({
    queryKey: ["record", r.record_id],
    queryFn: ({ signal }) =>
      api<RecordDetail>("/records/" + r.record_id, signal),
  });
  const d = detail.data;
  const valid =
    !!d &&
    d.source_sha256 === context.source_sha256 &&
    r.row_start >= 0 &&
    r.row_end <= d.shape[0] &&
    r.row_start <= context.profile &&
    context.profile < r.row_end &&
    r.column_start >= 0 &&
    r.column_end <= d.shape[1] &&
    r.column_start <= context.column &&
    context.column < r.column_end;
  const slices = useQuery({
    queryKey: ["work-slice", context],
    queryFn: ({ signal }) =>
      api<Slices>(
        `/records/${r.record_id}/slices?${new URLSearchParams({ row: String(context.profile), column: String(context.column), row_start: String(r.row_start), row_end: String(r.row_end), source_sha256: context.source_sha256 })}`,
        signal,
      ),
    enabled: valid,
    gcTime: 10000,
  });
  if (!valid)
    return <p>Проверьте сохранённые границы и доступность исходника.</p>;
  const path = `/records/${r.record_id}/tiles?${new URLSearchParams({ x0: String(d!.x_m[r.column_start]), x1: String(d!.x_m[r.column_end - 1]), t0: String(d!.time_s[r.row_start]), t1: String(Math.max(d!.time_s[r.row_start] + 1, d!.time_s[r.row_end - 1])), width: "256", height: "256" })}`;
  return (
    <section className="work-signal">
      <h2>Исходное измерение</h2>
      <Heatmap
        path={path}
        expectedSourceSha={context.source_sha256}
        versionKey={context.source_sha256}
        cursor={[d!.x_m[context.column], d!.time_s[context.profile]]}
        onSelect={(x, t) => {
          const nearest = (axis: number[], value: number) => {
            let lo = 0,
              hi = axis.length - 1;
            while (lo < hi) {
              const mid = Math.floor((lo + hi) / 2);
              if (axis[mid] < value) lo = mid + 1;
              else hi = mid;
            }
            return lo > 0 &&
              Math.abs(axis[lo - 1] - value) < Math.abs(axis[lo] - value)
              ? lo - 1
              : lo;
          };
          onChange({
            profile: Math.max(
              r.row_start,
              Math.min(r.row_end - 1, nearest(d!.time_s, t)),
            ),
            column: Math.max(
              r.column_start,
              Math.min(r.column_end - 1, nearest(d!.x_m, x)),
            ),
          });
        }}
      />
      {slices.error && (
        <InlineNotification
          kind="error"
          title="Срез не загружен"
          subtitle={String(slices.error)}
          hideCloseButton
        />
      )}
      {slices.data && (
        <>
          <p>
            Точка: {slices.data.distance_m.toFixed(3)} м ·{" "}
            {slices.data.elapsed_s} с от начала ·{" "}
            {slices.data.temperature === null
              ? "нет измерения"
              : slices.data.temperature.toFixed(3) + " °C"}
          </p>
          <div className="work-chart-pair">
            <LineChart
              title="Точный пространственный профиль"
              x={slices.data.x_m.slice(r.column_start, r.column_end)}
              y={slices.data.profile.slice(r.column_start, r.column_end)}
              xLabel="Расстояние, м"
              syncKey={context.source_sha256}
            />
            <LineChart
              title="Временной ряд выбранной точки"
              x={slices.data.time_s}
              y={slices.data.trend}
              xLabel="Время от начала, с"
            />
          </div>
        </>
      )}
    </section>
  );
}

function CreateWork() {
  const [params] = useSearchParams();
  const navigate = useNavigate();
  const account = useAccount();
  const [record, setRecord] = useState(params.get("record") ?? "5");
  const [title, setTitle] = useState("Разбор измерения");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const pending = useRef<{ body: object; key: string } | undefined>(undefined);
  const records = useQuery({
    queryKey: ["records"],
    queryFn: ({ signal }) => api<RecordMeta[]>("/records", signal),
  });
  const native = records.data?.find((r) => r.id === record);
  async function create() {
    if (!native) return;
    setBusy(true);
    setError("");
    if (
      params.get("source_sha256") &&
      params.get("source_sha256") !== native.source_sha256
    ) {
      setError(
        "Версия источника изменилась; ссылка не заменяется текущими данными",
      );
      setBusy(false);
      return;
    }
    if (!pending.current) {
      const row = Math.max(
        0,
        Math.min(native.shape[0] - 1, Number(params.get("row")) || 0),
      );
      const start = Math.max(0, Math.min(row, Number(params.get("r0")) || 0));
      const end = Math.min(
        native.shape[0],
        Math.max(
          row + 1,
          Number(params.get("r1")) || Math.min(native.shape[0], start + 48),
        ),
      );
      const cs = Math.max(0, Number(params.get("c0")) || 0);
      const ce = Math.min(
        native.shape[1],
        Math.max(
          cs + 2,
          Number(params.get("c1")) || Math.min(native.shape[1], cs + 256),
        ),
      );
      pending.current = {
        key: crypto.randomUUID(),
        body: {
          title,
          context: {
            region: {
              record_id: record,
              row_start: start,
              row_end: end,
              column_start: cs,
              column_end: ce,
            },
            source_sha256: native.source_sha256,
            profile: row,
            column: Math.max(
              cs,
              Math.min(ce - 1, Number(params.get("column")) || cs),
            ),
            run_ids: params.get("run_id") ? [params.get("run_id")!] : [],
            layer: params.get("layer") ?? "raw",
          },
          origin:
            params.get("origin") ??
            (params.get("run_id") ? "analysis" : "archive"),
          ...(params.get("stream_id")
            ? { stream_id: params.get("stream_id") }
            : {}),
        },
      };
    }
    try {
      const work = await control<Work>(
        "/investigations",
        pending.current.body,
        undefined,
        pending.current.key,
      );
      navigate("/works/" + work.id);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <section className="work-create">
      <h2>Начать работу</h2>
      <p>
        Закрепляется источник и явно заданное окно. После открытия можно
        уточнить контекст; это не экспертная метка.
      </p>
      <TextInput
        id="work-title-new"
        labelText="Название"
        value={title}
        onChange={(e) => {
          setTitle(e.target.value);
          pending.current = undefined;
        }}
      />
      <Select
        id="work-record-new"
        labelText="Запись"
        value={record}
        onChange={(e) => {
          setRecord(e.target.value);
          pending.current = undefined;
        }}
      >
        {records.data?.map((r) => (
          <SelectItem key={r.id} value={r.id} text={r.source} />
        ))}
      </Select>
      {error && (
        <InlineNotification
          kind="error"
          title="Работа не сохранена"
          subtitle={error}
          hideCloseButton
        />
      )}
      <Button
        disabled={
          busy ||
          !native ||
          !title.trim() ||
          !account ||
          !["analyst", "admin"].includes(account.role)
        }
        onClick={() => void create()}
      >
        {busy
          ? "Сохраняем…"
          : pending.current
            ? "Повторить ту же попытку"
            : "Создать работу"}
      </Button>
    </section>
  );
}

function WorkDetail({ work }: { work: Work }) {
  const account = useAccount();
  const client = useQueryClient();
  const [draft, setDraft] = useState({
    title: work.title,
    notes: work.notes,
    context: work.context,
  });
  const [tab, setTab] = useState("signal");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [savedMessage, setSavedMessage] = useState("");
  const [caseToLink, setCaseToLink] = useState("");
  async function linkCase(id: string) {
    setCaseToLink(id);
    try {
      const value = await control<Work>(
        "/investigations/" + work.id + "/cases",
        { case_id: id },
      );
      client.setQueryData(["work", work.id], value);
      setCaseToLink("");
      setSavedMessage("Кейс сохранён и связан с работой");
    } catch (e) {
      setError("Кейс сохранён, но ссылка не добавлена: " + String(e));
    }
  }
  const canEdit = !!account && ["analyst", "admin"].includes(account.role);
  const runs = useQuery({
    queryKey: ["result-runs", work.context.region.record_id, account?.id],
    queryFn: ({ signal }) =>
      control<AnalysisRun[]>(
        "/result-runs?record_id=" + work.context.region.record_id,
        undefined,
        signal,
      ),
  });
  const selected = runs.data?.find((r) => r.id === draft.context.run_ids[0]);
  const scope = draft.context.region;
  const selectedCompatible =
    !!selected &&
    selected.source_sha256 === work.context.source_sha256 &&
    selected.region.record_id === scope.record_id &&
    selected.region.row_start <= scope.row_start &&
    scope.row_start <= draft.context.profile &&
    draft.context.profile < scope.row_end &&
    scope.row_end <= selected.region.row_end &&
    selected.region.column_start <= scope.column_start &&
    scope.column_start <= draft.context.column &&
    draft.context.column < scope.column_end &&
    scope.column_end <= selected.region.column_end &&
    [
      "raw",
      ...selected.layers,
      ...(selected.layers.includes("processed_temperature_c")
        ? ["difference"]
        : []),
    ].includes(draft.context.layer ?? "raw");
  const comparison = runs.data?.find((r) => r.id === draft.context.run_ids[1]);
  const comparisonCompatible =
    !!comparison &&
    comparison.source_sha256 === work.context.source_sha256 &&
    comparison.region.record_id === scope.record_id &&
    comparison.region.row_start <= scope.row_start &&
    comparison.region.row_end >= scope.row_end &&
    comparison.region.column_start <= scope.column_start &&
    comparison.region.column_end >= scope.column_end;
  const history = useQuery({
    queryKey: ["work-history", work.id, work.revision],
    queryFn: ({ signal }) =>
      control<{
        items: {
          id: string;
          revision: number;
          author_name: string;
          action: string;
          at: number;
        }[];
      }>("/investigations/" + work.id + "/history", undefined, signal),
    enabled: tab === "history",
  });
  const requests = useQuery({
    queryKey: ["work-reviews", work.id, work.revision],
    queryFn: ({ signal }) =>
      control<ReviewRequest[]>(
        "/investigations/" + work.id + "/review-requests",
        undefined,
        signal,
      ),
    enabled: tab === "review",
  });
  const experts = useQuery({
    queryKey: ["assignable-experts"],
    queryFn: ({ signal }) =>
      control<{ id: string; display_name: string }[]>(
        "/review-experts",
        undefined,
        signal,
      ),
    enabled: tab === "review" && canEdit,
  });
  const [expert, setExpert] = useState("");
  const assignment = useRef<{ body: object; key: string } | undefined>(
    undefined,
  );
  const dirty =
    JSON.stringify(draft) !==
    JSON.stringify({
      title: work.title,
      notes: work.notes,
      context: work.context,
    });
  function context(change: Partial<Context>) {
    setDraft((p) => ({ ...p, context: { ...p.context, ...change } }));
    setSavedMessage("");
  }
  async function save() {
    setBusy(true);
    setError("");
    try {
      const value = await control<Work>(
        "/investigations/" + work.id,
        { ...draft, expected_revision: work.revision },
        undefined,
        undefined,
        "PATCH",
      );
      client.setQueryData(["work", work.id], value);
      setDraft({
        title: value.title,
        notes: value.notes,
        context: value.context,
      });
      setSavedMessage("Контекст сохранён · ревизия " + value.revision);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }
  async function assign() {
    setBusy(true);
    setError("");
    if (!assignment.current)
      assignment.current = {
        body: { expected_revision: work.revision, assigned_to: expert },
        key: crypto.randomUUID(),
      };
    try {
      await control(
        "/investigations/" + work.id + "/review-requests",
        assignment.current.body,
        undefined,
        assignment.current.key,
      );
      assignment.current = undefined;
      await client.invalidateQueries({ queryKey: ["work", work.id] });
      await requests.refetch();
      setSavedMessage("Эксперту отправлен неизменяемый снимок окна");
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <div className="work-heading">
        <div>
          <p>
            Работа · {work.author_name} · ревизия {work.revision}
          </p>
          <h1>{work.title}</h1>
          <p>
            {work.context.region.record_id}.csv ·{" "}
            <code>{work.context.source_sha256}</code>
          </p>
        </div>
        <Tag type={dirty ? "warm-gray" : "green"}>
          {dirty ? "Изменения не сохранены" : "Сохранённый контекст"}
        </Tag>
      </div>
      {error && (
        <InlineNotification
          kind="error"
          title="Изменения не сохранены"
          subtitle={error}
          hideCloseButton
        />
      )}
      {savedMessage && <p role="status">{savedMessage}</p>}
      <details className="work-context">
        <summary>Контекст и заметки</summary>
        <TextInput
          id="work-title"
          labelText="Название"
          value={draft.title}
          onChange={(e) => setDraft((p) => ({ ...p, title: e.target.value }))}
        />
        <TextArea
          id="work-notes"
          labelText="Наблюдения и ограничения"
          value={draft.notes}
          onChange={(e) => setDraft((p) => ({ ...p, notes: e.target.value }))}
        />
        <div className="work-coordinate-grid">
          {bounds.map((key, i) => (
            <TextInput
              key={key}
              id={"work-bound-" + i}
              labelText={boundLabels[i]}
              inputMode="numeric"
              value={String(draft.context.region[key])}
              onChange={(e) =>
                context({
                  region: {
                    ...draft.context.region,
                    [key]: Number(e.target.value),
                  },
                })
              }
            />
          ))}
          <TextInput
            id="work-profile"
            labelText="Профиль"
            inputMode="numeric"
            value={String(draft.context.profile)}
            onChange={(e) => context({ profile: Number(e.target.value) })}
          />
          <TextInput
            id="work-column"
            labelText="Точка / столбец"
            inputMode="numeric"
            value={String(draft.context.column)}
            onChange={(e) => context({ column: Number(e.target.value) })}
          />
        </div>
        <div className="work-results">
          {[0, 1].map((i) => (
            <Select
              key={i}
              id={"work-run-" + i}
              labelText={
                i ? "Второй совместимый результат" : "Выбранная обработка"
              }
              value={draft.context.run_ids[i] ?? ""}
              disabled={i === 1 && !draft.context.run_ids[0]}
              onChange={(e) => {
                if (i === 0 && !e.target.value) {
                  context({ run_ids: [], layer: "raw" });
                  return;
                }
                const ids = [...draft.context.run_ids];
                ids[i] = e.target.value;
                context({ run_ids: ids.filter(Boolean) });
              }}
            >
              <SelectItem
                value=""
                text={i ? "Не сравнивать" : "Обработка не выбрана"}
              />
              {runs.data?.map((r) => (
                <SelectItem
                  key={r.id}
                  value={r.id}
                  text={r.method + " / " + r.method_version.slice(0, 12)}
                />
              ))}
            </Select>
          ))}
        </div>
        <Button
          disabled={busy || !canEdit || !dirty}
          onClick={() => void save()}
        >
          Сохранить контекст
        </Button>
        <Button
          kind="secondary"
          onClick={() => {
            setDraft({
              title: work.title,
              notes: work.notes,
              context: work.context,
            });
            setError("");
          }}
        >
          Вернуть сохранённый
        </Button>
      </details>
      <nav className="work-tabs" aria-label="Разделы работы">
        {[
          ["signal", "Сигнал"],
          ["comparison", "Сравнение"],
          ["review", "Проверка"],
          ["history", "История"],
        ].map(([id, label]) => (
          <button
            key={id}
            aria-current={tab === id ? "page" : undefined}
            onClick={() => setTab(id)}
          >
            {label}
          </button>
        ))}
      </nav>
      {tab === "comparison" && comparison && !comparisonCompatible && (
        <InlineNotification
          kind="warning"
          title="Второй результат не покрывает выбранное окно"
          subtitle="Уменьшите окно до общей области или выберите совместимый запуск. Другой результат не подставляется."
          hideCloseButton
        />
      )}
      {(tab === "signal" || tab === "comparison") &&
        (selected && selectedCompatible ? (
          <AnalysisWindow
            run={selected}
            comparison={
              tab === "comparison" && comparisonCompatible
                ? comparison
                : undefined
            }
            context={{
              region: draft.context.region,
              row: draft.context.profile,
              column: draft.context.column,
              layer: draft.context.layer ?? "raw",
              zone: "",
            }}
            onContextChange={(c) =>
              context({
                region: c.region,
                profile: c.row,
                column: c.column,
                layer: c.layer,
              })
            }
            onSavedCase={(id) => void linkCase(id)}
          />
        ) : (
          <>
            <InlineNotification
              kind="info"
              title={
                selected
                  ? "Выбранный результат не покрывает текущий контекст"
                  : "Обработка не выбрана"
              }
              subtitle="Ниже показаны только исходные измерения. Другой результат автоматически не выбирается."
              hideCloseButton
            />
            <RawSignal context={draft.context} onChange={context} />
          </>
        ))}
      {tab === "review" && (
        <section>
          <h2>Независимая проверка</h2>
          <p>
            Задание использует сохранённый снимок. Будущие изменения окна не
            меняют вопрос эксперту.
          </p>
          {canEdit && (
            <>
              <Select
                id="work-expert"
                labelText="Эксперт команды"
                value={expert}
                onChange={(e) => {
                  setExpert(e.target.value);
                  assignment.current = undefined;
                }}
              >
                <SelectItem value="" text="Выберите эксперта" />
                {experts.data?.map((e) => (
                  <SelectItem key={e.id} value={e.id} text={e.display_name} />
                ))}
              </Select>
              <Button
                disabled={busy || dirty || !expert}
                onClick={() => void assign()}
              >
                {assignment.current
                  ? "Повторить отправку"
                  : "Передать сохранённое окно"}
              </Button>
            </>
          )}
          {requests.data?.map((r) => (
            <article key={r.id}>
              <h3>
                Зафиксированное окно [{r.context.region.row_start},{" "}
                {r.context.region.row_end})
              </h3>
              <Link
                to={
                  "/review?" +
                  new URLSearchParams({
                    package: r.package,
                    card: r.review_id,
                    work_id: work.id,
                  })
                }
              >
                Открыть независимую проверку →
              </Link>
            </article>
          ))}
        </section>
      )}
      {tab === "history" && (
        <section>
          <h2>История работы</h2>
          {history.data?.items.map((e) => (
            <p key={e.id}>
              Ревизия {e.revision} · {e.author_name} · {e.action} ·{" "}
              {new Date(e.at * 1000).toLocaleString("ru-RU")}
            </p>
          ))}
        </section>
      )}
      <section>
        <h2>Зафиксированные кейсы</h2>
        {caseToLink && (
          <Button onClick={() => void linkCase(caseToLink)}>
            Повторить связывание сохранённого кейса
          </Button>
        )}
        {work.case_ids.length ? (
          work.case_ids.map((id) => (
            <p key={id}>
              <Link to={"/cases/" + id}>Открыть кейс {id.slice(0, 8)} →</Link>
            </p>
          ))
        ) : (
          <p>
            Кейс ещё не связан. Сохранённый снимок доступен независимо от
            будущих изменений работы.
          </p>
        )}
        {!selected && (
          <CaseForm
            region={draft.context.region}
            profile={draft.context.profile}
            sha={work.context.source_sha256}
            title={work.title}
            notes={draft.notes}
            onSaved={(id) => void linkCase(id)}
          />
        )}
      </section>
    </>
  );
}

export default function Investigations() {
  const { id } = useParams();
  const account = useAccount();
  const catalog = useQuery({
    queryKey: ["works", account?.id],
    queryFn: ({ signal }) =>
      control<Work[]>("/investigations", undefined, signal),
    enabled: !id,
  });
  const work = useQuery({
    queryKey: ["work", id],
    queryFn: ({ signal }) =>
      control<Work>("/investigations/" + id, undefined, signal),
    enabled: !!id,
  });
  return (
    <main className="page work-page">
      {id ? (
        <>
          <Link to="/works">← Все работы</Link>
          {work.error && (
            <InlineNotification
              kind="error"
              title="Работа не открыта"
              subtitle={String(work.error)}
              hideCloseButton
            />
          )}
          {work.data && <WorkDetail key={work.data.id} work={work.data} />}
        </>
      ) : (
        <>
          <h1>Работы команды</h1>
          <p>
            Измерение, расчёты, независимая проверка и сохранённый кейс в одном
            контексте.
          </p>
          <CreateWork />
          {catalog.error && (
            <InlineNotification
              kind="error"
              title="Список не загружен"
              subtitle={String(catalog.error)}
              hideCloseButton
            />
          )}
          <div className="work-list">
            {catalog.data?.map((w) => (
              <article key={w.id}>
                <h2>
                  <Link to={"/works/" + w.id}>{w.title}</Link>
                </h2>
                <p>
                  {w.context.region.record_id}.csv · {w.author_name} · ревизия{" "}
                  {w.revision}
                </p>
                <p>
                  Строки [{w.context.region.row_start},{" "}
                  {w.context.region.row_end}) · {w.context.run_ids.length}{" "}
                  расчётов · {w.case_ids.length} кейсов
                </p>
              </article>
            ))}
          </div>
        </>
      )}
    </main>
  );
}
