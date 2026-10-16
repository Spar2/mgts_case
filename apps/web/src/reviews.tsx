import { useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useVirtualizer } from "@tanstack/react-virtual";
import {
  Button,
  InlineNotification,
  RadioButton,
  RadioButtonGroup,
  Select,
  SelectItem,
  TextArea,
  TextInput,
  Tag,
} from "@carbon/react";
import { motion } from "motion/react";
import { Link, useSearchParams } from "react-router-dom";
import {
  api,
  ApiError,
  mutate,
  type Annotation,
  type Queue,
  type ReviewBody,
  type Slices,
} from "./api";
import { Heatmap, LineChart } from "./charts";
import { readDraft, writeDraft, clearDrafts } from "./drafts";
import { control, useAccount } from "./access";
import { CaseForm } from "./cases";
const episodeLabels: Record<string, string> = {
  present: "Есть температурное событие",
  absent: "Нет температурного события",
  ambiguous: "Неоднозначно",
  unknown: "Недостаточно данных",
};
const phenomenonLabels: Record<string, string> = {
  local_change: "Локальное изменение",
  common_trend: "Общий тренд",
  dropout: "Пропуски сигнала",
  mixed: "Смешанное наблюдение",
  unknown: "Неизвестно",
};
type Draft = {
  reviewer: string;
  episode: ReviewBody["temperature_episode"] | "";
  phenomenon: ReviewBody["phenomenon"];
  notes: string;
  revision: number;
};
export default function Reviews() {
  const account = useAccount();
  const client = useQueryClient(),
    [params, setParams] = useSearchParams(),
    source = params.get("source") ?? "all",
    selected = params.get("card");
  const packageId = params.get("package") ?? "general";
  const workId = params.get("work_id");
  const [quick, setQuick] = useState(params.get("quick") === "1");
  const progress = useQuery({
    queryKey: ["review-progress", packageId, account?.id],
    queryFn: ({ signal }) =>
      control<{ items: Record<string, string> }>(
        "/review-progress?package=" + encodeURIComponent(packageId),
        undefined,
        signal,
      ),
    enabled: !!account,
  });
  const inbox = useQuery({
    queryKey: ["review-inbox", account?.id],
    queryFn: ({ signal }) =>
      control<{
        items: {
          id: string;
          investigation_id: string;
          package: string;
          review_id: string;
          region: { record_id: string; row_start: number; row_end: number };
          status: string;
        }[];
      }>("/review-inbox", undefined, signal),
    enabled: !!account && ["expert", "admin"].includes(account.role),
    refetchInterval: 15000,
  });
  const [identity, setIdentity] = useState(
    () => sessionStorage.getItem("dts-reviewer") ?? "",
  );
  const [trusted, setTrusted] = useState(
    () => localStorage.getItem("dts-trusted-drafts") === "yes",
  );
  const suffix = `package=${encodeURIComponent(packageId)}&independent=true&reviewer=${encodeURIComponent(identity)}`;
  const queueKey = ["queue", packageId, account?.id ?? identity];
  const query = useQuery({
      queryKey: queueKey,
      queryFn: ({ signal }) => api<Queue>("/queue?" + suffix, signal),
    }),
    q = query.data;
  const cards =
      q?.cards.filter((c) => source === "all" || c.source === source) ?? [],
    card = cards.find((c) => c.review_id === selected) ?? cards[0];
  const [draft, setDraft] = useState<Draft>({
      reviewer: "",
      episode: "",
      phenomenon: "unknown",
      notes: "",
      revision: 0,
    }),
    drafts = useRef(new Map<string, Draft>());
  const [saved, setSaved] = useState<Annotation>(),
    [conflict, setConflict] = useState<Annotation>(),
    [error, setError] = useState(""),
    [sending, setSending] = useState(false),
    [revealed, setRevealed] = useState(false),
    [overlay, setOverlay] = useState("");
  const assigned = useQuery({
    queryKey: ["review-work-context", workId, card?.review_id],
    queryFn: ({ signal }) =>
      control<
        { review_id: string; context: { run_ids: string[]; column: number } }[]
      >("/investigations/" + workId + "/review-requests", undefined, signal),
    enabled: !!workId && !!(saved || q?.annotations[card?.review_id ?? ""]),
  });
  const [mobileStep, setMobileStep] = useState<
    "tasks" | "signal" | "decision" | "saved"
  >(selected ? "signal" : "tasks");
  const idempotency = useRef<{ digest: string; key: string } | undefined>(
      undefined,
    ),
    list = useRef<HTMLDivElement>(null),
    active = useRef<string | undefined>(undefined);
  const virtual = useVirtualizer({
    count: cards.length,
    getScrollElement: () => list.current,
    estimateSize: () => 76,
    overscan: 4,
  });
  const draftKey =
    card && q
      ? `${q.queue_sha256}:${account?.id ?? identity}:${card.review_id}`
      : "";
  useEffect(() => {
    if (!card) return;
    active.current = draftKey;
    setSending(false);
    const known = q?.annotations[card.review_id];
    setDraft(
      drafts.current.get(draftKey) ?? {
        reviewer: identity || known?.reviewer || "",
        episode: known?.temperature_episode ?? "",
        phenomenon: known?.phenomenon ?? "unknown",
        notes: known?.notes ?? "",
        revision: known?.revision ?? 0,
      },
    );
    setSaved(known);
    setConflict(undefined);
    setError("");
    setRevealed(false);
    setOverlay("");
    idempotency.current = undefined;
    if (trusted && identity)
      void readDraft<Draft>(draftKey)
        .then((stored) => {
          if (
            stored &&
            active.current === draftKey &&
            !drafts.current.has(draftKey)
          ) {
            drafts.current.set(draftKey, stored);
            setDraft(stored);
            setSaved(
              known &&
                known.revision === stored.revision &&
                known.notes === stored.notes &&
                known.temperature_episode === stored.episode &&
                known.phenomenon === stored.phenomenon
                ? known
                : undefined,
            );
          }
        })
        .catch((e) => setError("Черновик устройства недоступен: " + String(e)));
  }, [draftKey]);
  const methods = useQuery({
    queryKey: [
      "methods",
      packageId,
      q?.queue_sha256,
      account?.id ?? identity,
      card?.review_id,
    ],
    queryFn: ({ signal }) =>
      api<{
        methods: string[];
        counts: number[];
        policy: string;
      }>(`/queue/${card!.review_id}/methods?${suffix}`, signal),
    enabled: !!card && revealed,
    retry: false,
    gcTime: 10000,
  });
  const [profileOffset, setProfileOffset] = useState(0);
  useEffect(() => setProfileOffset(0), [packageId, card?.review_id]);
  const slices = useQuery({
    queryKey: [
      "review-slice",
      packageId,
      q?.queue_sha256,
      card?.source_sha256,
      card?.review_id,
      profileOffset,
    ],
    queryFn: ({ signal }) =>
      api<Slices>(
        `/records/${card!.source.replace(".csv", "")}/slices?row=${card!.row_bounds[0] + profileOffset}&column=${Math.floor((card!.column_bounds[0] + card!.column_bounds[1]) / 2)}&source_sha256=${card!.source_sha256}`,
        signal,
      ),
    enabled: !!card,
    gcTime: 10000,
  });
  function update(p: Partial<Draft>) {
    setDraft((d) => {
      const next = { ...d, ...p };
      if (card) {
        drafts.current.set(draftKey, next);
        if (trusted && identity)
          void writeDraft(draftKey, next).catch((e) =>
            setError("Черновик не записан на устройство: " + String(e)),
          );
      }
      return next;
    });
    setSaved(undefined);
  }
  function nextCard() {
    if (!card || !cards.length) return;
    const next =
      cards[
        (cards.findIndex((c) => c.review_id === card.review_id) + 1) %
          cards.length
      ];
    const p = new URLSearchParams(params);
    p.set("package", packageId);
    p.set("card", next.review_id);
    setParams(p);
    setMobileStep("signal");
  }
  async function skip() {
    if (!card || !account) return;
    setError("");
    try {
      await control("/review-progress", {
        package: packageId,
        review_id: card.review_id,
        state: "skipped",
      });
      await progress.refetch();
      nextCard();
    } catch (e) {
      setError(String(e));
    }
  }
  async function save(advance = false) {
    if (!card || !q || !draft.episode) return;
    setSending(true);
    setError("");
    setConflict(undefined);
    const body: ReviewBody = {
      queue_sha256: q.queue_sha256,
      review_id: card.review_id,
      expected_revision: draft.revision,
      reviewer: identity,
      reviewer_type: "expert",
      temperature_episode: draft.episode,
      phenomenon: draft.phenomenon,
      notes: draft.notes,
    };
    const digest = JSON.stringify(body);
    if (idempotency.current?.digest !== digest)
      idempotency.current = { digest, key: crypto.randomUUID() };
    try {
      const result = await mutate<Annotation>(
        "/reviews?" + suffix,
        body,
        idempotency.current.key,
      );
      const next = { ...draft, revision: result.revision };
      drafts.current.set(draftKey, next);
      if (trusted)
        void writeDraft(draftKey, next).catch((e) => setError(String(e)));
      if (active.current === draftKey) {
        setSaved(result);
        setDraft(next);
        setMobileStep("saved");
      }
      client.setQueryData<Queue>(queueKey, (old) =>
        old
          ? {
              ...old,
              annotations: { ...old.annotations, [card.review_id]: result },
            }
          : old,
      );
      if (advance && active.current === draftKey) nextCard();
    } catch (e) {
      if (active.current !== draftKey) return;
      if (e instanceof ApiError && e.status === 409)
        setConflict(e.body.current);
      setError(String(e));
    } finally {
      if (active.current === draftKey) setSending(false);
    }
  }
  return (
    <main
      className="page"
      tabIndex={0}
      onKeyDown={(e) => {
        if (
          !quick ||
          e.target instanceof HTMLInputElement ||
          e.target instanceof HTMLTextAreaElement ||
          e.target instanceof HTMLSelectElement
        )
          return;
        const key = Number(e.key);
        if (key >= 1 && key <= 4) {
          e.preventDefault();
          update({
            episode: (["present", "absent", "ambiguous", "unknown"] as const)[
              key - 1
            ],
          });
        }
        if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
          e.preventDefault();
          void save(true);
        }
      }}
    >
      {inbox.data?.items.length ? (
        <section className="review-inbox">
          <h2>Задания вам</h2>
          {inbox.data.items.map((item) => (
            <p key={item.id}>
              <Link
                to={
                  "/review?" +
                  new URLSearchParams({
                    package: item.package,
                    card: item.review_id,
                    work_id: item.investigation_id,
                    quick: "1",
                  })
                }
              >
                {item.region.record_id}.csv / строки [{item.region.row_start},{" "}
                {item.region.row_end}) ·{" "}
                {item.status === "answered"
                  ? "Ответ сохранён"
                  : "Ожидает вашего ответа"}
              </Link>
            </p>
          ))}
        </section>
      ) : null}
      <div className="review-quick-controls">
        <label>
          <input
            type="checkbox"
            checked={quick}
            onChange={(e) => setQuick(e.target.checked)}
          />{" "}
          Короткая проверка: ответ и переход
        </label>
        {workId && <Link to={"/works/" + workId}>Вернуться к работе →</Link>}
        <p>Пропуск не создаёт ответа «неизвестно» и не является меткой.</p>
      </div>
      <div className="page-title">
        <div>
          <p className="eyebrow">Рабочая область / Экспертная проверка</p>
          <h1>Самостоятельный ответ по сигналу</h1>
        </div>
        <a className="text-link" href={"/api/v1/export?" + suffix} download>
          Экспорт ответов ↓
        </a>
      </div>
      <div className="review-policy">
        <Tag type="blue">Слепая проверка</Tag>
        <p>
          Сначала исходный сигнал и ваш ответ. Затем доступны результаты
          методов.{" "}
          {account
            ? "Ответ сохраняется от имени вашей учётной записи."
            : "Имя эксперта в локальной версии заявлено самостоятельно."}
          Ответ относится только к этому окну.
        </p>
      </div>
      {query.error && (
        <InlineNotification
          kind="error"
          title="Очередь недоступна"
          subtitle={String(query.error)}
          hideCloseButton
        />
      )}
      <nav className="review-jumps" aria-label="Шаги проверки на телефоне">
        {(["tasks", "signal", "decision", "saved"] as const).map(
          (step, index) => (
            <button
              key={step}
              type="button"
              aria-current={step === mobileStep ? "step" : undefined}
              disabled={step === "saved" && !saved}
              onClick={() => setMobileStep(step)}
            >
              {["Очередь", "1 · Сигнал", "2 · Ответ", "3 · Сохранено"][index]}
            </button>
          ),
        )}
      </nav>
      <div className={`review-workspace mobile-${mobileStep}`}>
        <aside className="queue-panel">
          <Select
            id="queue-source"
            labelText="Источник"
            value={source}
            onChange={(e) =>
              setParams({ package: packageId, source: e.target.value })
            }
          >
            <SelectItem value="all" text="Все записи" />
            {[...new Set(q?.cards.map((c) => c.source))].map((s) => (
              <SelectItem key={s} value={s} text={s} />
            ))}
          </Select>
          <div className="queue-count">
            {cards.length} окон · {Object.keys(q?.annotations ?? {}).length}{" "}
            ответов всего
          </div>
          <div
            ref={list}
            className="queue-list"
            role="list"
            aria-label="Окна экспертной проверки"
          >
            <div
              style={{ height: virtual.getTotalSize(), position: "relative" }}
            >
              {virtual.getVirtualItems().map((v) => {
                const c = cards[v.index];
                return (
                  <button
                    key={c.review_id}
                    className={`queue-item ${c.review_id === card?.review_id ? "selected" : ""}`}
                    style={{
                      position: "absolute",
                      top: 0,
                      left: 0,
                      width: "100%",
                      height: v.size,
                      transform: `translateY(${v.start}px)`,
                    }}
                    onClick={() => {
                      setMobileStep("signal");
                      setParams({
                        package: packageId,
                        source,
                        card: c.review_id,
                      });
                    }}
                    aria-current={
                      c.review_id === card?.review_id ? "true" : undefined
                    }
                  >
                    <strong>
                      {c.source} / {c.x_bounds_m[0].toFixed(1)}–
                      {c.x_bounds_m[1].toFixed(1)} м
                    </strong>
                    <span>
                      {c.timestamps[0]?.slice(11, 19)} ·{" "}
                      {c.native_shape.join(" × ")} ячеек
                    </span>
                    <span>
                      {q?.annotations[c.review_id]
                        ? "Ответ сохранён"
                        : progress.data?.items[c.review_id] === "skipped"
                          ? "Пропущено без ответа"
                          : "Ожидает проверки"}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>
        </aside>
        {card && (
          <section className="review-evidence" id="review-signal">
            <div className="section-title">
              <h2>
                {card.source} / {card.x_bounds_m[0].toFixed(1)}–
                {card.x_bounds_m[1].toFixed(1)} м
              </h2>
              <Link
                className="text-link"
                to={`/research?record=${card.source.replace(".csv", "")}`}
              >
                Открыть запись →
              </Link>
            </div>
            <p className="mono">
              {card.timestamps[0]} → {card.timestamps.at(-1)}
            </p>
            <Heatmap
              path={`/queue/${card.review_id}/window?package=${encodeURIComponent(packageId)}`}
              versionKey={q?.queue_sha256}
              expectedSourceSha={card.source_sha256}
            />
            <div className="native-controls">
              <Button
                kind="ghost"
                size="sm"
                disabled={profileOffset === 0}
                onClick={() => setProfileOffset((v) => v - 1)}
              >
                ← Предыдущий профиль
              </Button>
              <span>
                Профиль {profileOffset + 1} / {card.native_shape[0]}
              </span>
              <Button
                kind="ghost"
                size="sm"
                disabled={profileOffset >= card.native_shape[0] - 1}
                onClick={() => setProfileOffset((v) => v + 1)}
              >
                Следующий →
              </Button>
            </div>
            {slices.data && (
              <>
                <LineChart
                  title="Точный профиль экспертного окна"
                  x={slices.data.x_m.slice(
                    card.column_bounds[0],
                    card.column_bounds[1],
                  )}
                  y={slices.data.profile.slice(
                    card.column_bounds[0],
                    card.column_bounds[1],
                  )}
                  xLabel="Расстояние, м"
                />
                <details>
                  <summary>
                    Числовые измерения выбранного профиля ·{" "}
                    {slices.data.timestamp}
                  </summary>
                  <div className="numeric-scroll">
                    <table>
                      <thead>
                        <tr>
                          <th>Расстояние, м</th>
                          <th>Температура, °C</th>
                        </tr>
                      </thead>
                      <tbody>
                        {slices.data.x_m
                          .slice(card.column_bounds[0], card.column_bounds[1])
                          .map((x, i) => (
                            <tr key={i}>
                              <td>{x.toFixed(4)}</td>
                              <td>
                                {slices.data!.profile[
                                  card.column_bounds[0] + i
                                ]?.toFixed(6) ?? "Нет валидного наблюдения"}
                              </td>
                            </tr>
                          ))}
                      </tbody>
                    </table>
                  </div>
                </details>
              </>
            )}
            <div className="methods-reveal">
              <h2>Сопоставление с методами</h2>
              <p>
                Доступно после сохранения независимого ответа. Результаты —
                исследовательские гипотезы; качество на реальных данных не
                доказано.
              </p>
              <Button
                kind="tertiary"
                size="sm"
                disabled={!saved && !q?.annotations[card.review_id]}
                onClick={() => setRevealed(true)}
              >
                Раскрыть результаты методов
              </Button>
              {methods.error && <p role="alert">{String(methods.error)}</p>}
              {methods.data && (
                <>
                  {methods.data.methods.length === 0 ? (
                    <p>
                      В этой очереди нет подготовленных результатов методов.
                      Исходные данные и ответы доступны.
                    </p>
                  ) : (
                    <Select
                      id="method-overlay"
                      labelText="Гипотеза метода на исходной сетке"
                      value={overlay}
                      onChange={(e) => setOverlay(e.target.value)}
                    >
                      <SelectItem value="" text="Выберите метод" />
                      {methods.data.methods.map((name) => (
                        <SelectItem key={name} value={name} text={name} />
                      ))}
                    </Select>
                  )}
                  {overlay && (
                    <Heatmap
                      path={`/queue/${card.review_id}/methods/${overlay}/window?${suffix}`}
                      versionKey={q?.queue_sha256}
                      expectedSourceSha={card.source_sha256}
                    />
                  )}
                  <table>
                    <thead>
                      <tr>
                        <th>Метод</th>
                        <th>Кандидатные ячейки</th>
                        <th>Статус</th>
                      </tr>
                    </thead>
                    <tbody>
                      {methods.data.methods.map((name, i) => (
                        <tr key={name}>
                          <td>{name}</td>
                          <td>{methods.data!.counts[i]}</td>
                          <td>
                            {["tree_v2", "cnn_v2"].includes(name)
                              ? "Историческое обучение на синтетике"
                              : "Исследовательский результат"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </>
              )}
            </div>
          </section>
        )}
        {card && (
          <motion.aside
            className="review-form"
            id="review-decision"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.16 }}
          >
            <h2>Ваше решение</h2>
            <p className="mono">
              {card.source} ·{" "}
              {card.x_bounds_m.map((x) => x.toFixed(2)).join("–")} м ·{" "}
              {card.timestamps[0]} → {card.timestamps.at(-1)}
            </p>
            <form
              onSubmit={(e) => {
                e.preventDefault();
                void save();
              }}
            >
              <TextInput
                id="reviewer"
                labelText="Имя эксперта"
                value={draft.reviewer}
                maxLength={100}
                required={!account}
                readOnly={!!account}
                onChange={(e) => update({ reviewer: e.target.value })}
                onBlur={(e) => {
                  const name = e.target.value.trim();
                  if (name !== identity) {
                    sessionStorage.setItem("dts-reviewer", name);
                    setIdentity(name);
                  }
                }}
              />
              <RadioButtonGroup
                name="episode"
                legendText="Температурное событие в окне"
                orientation="vertical"
                valueSelected={draft.episode}
                onChange={(value) =>
                  update({ episode: value as Draft["episode"] })
                }
              >
                {Object.entries(episodeLabels).map(([value, label]) => (
                  <RadioButton
                    key={value}
                    id={"answer-" + value}
                    value={value}
                    labelText={label}
                  />
                ))}
              </RadioButtonGroup>
              <Select
                id="phenomenon"
                labelText="Характер наблюдения"
                value={draft.phenomenon}
                onChange={(e) =>
                  update({ phenomenon: e.target.value as Draft["phenomenon"] })
                }
              >
                {Object.entries(phenomenonLabels).map(([value, label]) => (
                  <SelectItem key={value} value={value} text={label} />
                ))}
              </Select>
              <TextArea
                id="notes"
                labelText={
                  account
                    ? "Пояснение по сигналу — необязательно"
                    : "Обоснование по сигналу"
                }
                helperText="Укажите наблюдаемые изменения, время, координаты или ограничения данных."
                rows={5}
                value={draft.notes}
                maxLength={8000}
                required={!account}
                onChange={(e) => update({ notes: e.target.value })}
              />
              <Button
                type="submit"
                disabled={
                  (!!account && !["expert", "admin"].includes(account.role)) ||
                  sending ||
                  !identity ||
                  !!saved ||
                  !draft.episode ||
                  (!account && !draft.notes.trim()) ||
                  !draft.reviewer.trim()
                }
              >
                {sending ? "Отправляется…" : "Сохранить ответ"}
              </Button>
              {quick &&
                account &&
                ["expert", "admin"].includes(account.role) && (
                  <>
                    <Button
                      type="button"
                      kind="secondary"
                      disabled={sending || !!saved || !draft.episode}
                      onClick={() => void save(true)}
                    >
                      Сохранить и дальше
                    </Button>
                    <Button
                      type="button"
                      kind="ghost"
                      disabled={sending}
                      onClick={() => void skip()}
                    >
                      Пропустить без ответа
                    </Button>
                  </>
                )}
              {account && !["expert", "admin"].includes(account.role) && (
                <p>
                  Для сохранения независимого ответа нужна роль эксперта. Сейчас
                  можно исследовать исходный сигнал.
                </p>
              )}
              {saved && (
                <>
                  <InlineNotification
                    kind="success"
                    title={`Сохранено · ревизия ${saved.revision}`}
                    subtitle={saved.saved_at}
                    hideCloseButton
                  />
                  <Button
                    kind="tertiary"
                    onClick={() => {
                      setMobileStep("decision");
                      update({ revision: saved.revision });
                    }}
                  >
                    Изменить ответ новой ревизией
                  </Button>
                  <Button
                    kind="secondary"
                    disabled={cards.length < 2}
                    onClick={() => {
                      const next =
                        cards[
                          (cards.findIndex(
                            (c) => c.review_id === card.review_id,
                          ) +
                            1) %
                            cards.length
                        ];
                      setMobileStep("signal");
                      setParams({
                        package: packageId,
                        source,
                        card: next.review_id,
                      });
                    }}
                  >
                    Следующее окно
                  </Button>
                </>
              )}
              {error && (
                <InlineNotification
                  kind="error"
                  title="Ответ не сохранён"
                  subtitle={error}
                  hideCloseButton
                />
              )}
              {conflict && (
                <div className="conflict">
                  <h3>Ответ изменён в другой вкладке</h3>
                  <p>Ваш текст сохранён в текущей форме.</p>
                  <p>
                    Сервер, ревизия {conflict.revision}:{" "}
                    {episodeLabels[conflict.temperature_episode]}
                  </p>
                  <blockquote>{conflict.notes}</blockquote>
                  <Button
                    kind="tertiary"
                    size="sm"
                    onClick={() => {
                      update({ revision: conflict.revision });
                      setConflict(undefined);
                      setError("");
                      idempotency.current = undefined;
                    }}
                  >
                    Подготовить мой вариант к повторной отправке
                  </Button>
                </div>
              )}
            </form>
            <p className="draft-policy">
              {account
                ? "Учётная запись определяет автора ответа."
                : "Имя заявлено локально, это не персональная авторизация."}{" "}
              У каждого имени отдельный журнал и собственное раскрытие методов.
              Смена имени переключает рабочий контекст.
            </p>
            <label className="draft-consent">
              <input
                type="checkbox"
                checked={trusted}
                onChange={(event) => {
                  const enabled = event.target.checked;
                  setTrusted(enabled);
                  localStorage.setItem(
                    "dts-trusted-drafts",
                    enabled ? "yes" : "no",
                  );
                  if (enabled && identity)
                    void writeDraft(draftKey, draft).catch((e) =>
                      setError(String(e)),
                    );
                }}
              />{" "}
              Это доверенное устройство: сохранять черновики после обновления
            </label>
            <p>
              Локальный черновик не является отправленным ответом. Отключение
              сохраняет ранее записанные черновики до явной очистки.
            </p>
            <Button
              kind="ghost"
              size="sm"
              onClick={() =>
                void clearDrafts()
                  .then(() => {
                    drafts.current.clear();
                    setError(
                      "Локальные черновики очищены. Текущая форма остаётся открытой.",
                    );
                  })
                  .catch((e) => setError(String(e)))
              }
            >
              Очистить черновики устройства
            </Button>
            <dl className="readings">
              <dt>Окно</dt>
              <dd className="mono">{card.review_id}</dd>
              <dt>Ревизия</dt>
              <dd>{draft.revision}</dd>
              <dt>Физическая причина</dt>
              <dd>Не установлена</dd>
              <dt>Обучающие метки</dt>
              <dd>Не создаются автоматически</dd>
            </dl>
          </motion.aside>
        )}
      </div>
      {card && account && (saved || q?.annotations[card.review_id]) && (
        <section className="review-case">
          <h2>Передать проверяемый кейс</h2>
          <CaseForm
            region={{
              record_id: card.source.replace(/\.csv$/, ""),
              row_start: card.row_bounds[0],
              row_end: card.row_bounds[1],
              column_start: card.column_bounds[0],
              column_end: card.column_bounds[1],
            }}
            profile={card.row_bounds[0] + profileOffset}
            sha={card.source_sha256}
            title={"Проверенное окно " + card.source}
            notes={draft.notes}
            proposal={packageId + ":" + card.review_id}
            runId={
              assigned.data?.find((r) => r.review_id === card.review_id)
                ?.context.run_ids[0]
            }
            onSaved={(id) => {
              if (workId)
                void control("/investigations/" + workId + "/cases", {
                  case_id: id,
                })
                  .then(() =>
                    client.invalidateQueries({ queryKey: ["work", workId] }),
                  )
                  .catch((e) =>
                    setError(
                      "Кейс сохранён; ссылку нужно повторить из работы: " +
                        String(e),
                    ),
                  );
            }}
          />
        </section>
      )}
    </main>
  );
}
