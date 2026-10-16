import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Button,
  InlineNotification,
  Select,
  SelectItem,
  TextInput,
  TextArea,
  Tag,
} from "@carbon/react";
import { Link, useSearchParams } from "react-router-dom";
import { api, mutate, download, type RecordDetail, type Slices } from "./api";
import { Heatmap, LineChart } from "./charts";
import type { components } from "./api-schema";
import { control, useAccount } from "./access";
import { processedProfile, processingKey } from "./processing";
import { CaseForm } from "./cases";

export type Region = components["schemas"]["Region"];
type Candidate = components["schemas"]["CandidateView"] & {
  proposal_bounds?: { row_bounds?: number[]; x_bounds_m?: number[] } | null;
};
type Package = components["schemas"]["PackageView"];
type Evidence = components["schemas"]["EvidenceView"];
type Run = components["schemas"]["RunDetail"];
export function regionLink(r: Region) {
  return `/research?${new URLSearchParams({ record: r.record_id, row: String(r.row_start), column: String(r.column_start), r0: String(r.row_start), r1: String(r.row_end), c0: String(r.column_start), c1: String(r.column_end) })}`;
}

export function RegionPanel({
  initial,
  proposal,
}: {
  initial: Region;
  proposal?: string;
}) {
  const account = useAccount();
  const [params] = useSearchParams();
  const passportId = params.get("passport");
  const passport = useQuery({
    queryKey: ["passport", passportId],
    queryFn: () =>
      control<{
        source_sha256: string;
        title: string;
        notes: string;
        region: Region;
        profile: number;
        method: string;
        scale: [number, number];
        run_id?: string | null;
      }>("/passports/" + passportId),
    enabled: !!account && !!passportId,
    retry: false,
  });
  const [region, setRegion] = useState(initial),
    [editing, setEditing] = useState(initial);
  const [title, setTitle] = useState("Исследованный участок"),
    [notes, setNotes] = useState(""),
    [passportLink, setPassportLink] = useState(""),
    [message, setMessage] = useState("");
  const [processing, setProcessing] = useState<{
      values: (number | null)[];
      runId: string;
      selectionKey: string;
      run: Run;
    }>(),
    [method, setMethod] = useState("median_x3"),
    [busy, setBusy] = useState(false);
  const [profile, setProfile] = useState(() => {
    const row = Number(params.get("row"));
    return params.has("row") &&
      Number.isInteger(row) &&
      row >= initial.row_start &&
      row < initial.row_end
      ? row
      : initial.row_start;
  });
  const detail = useQuery({
    queryKey: ["record", region.record_id],
    queryFn: ({ signal }) =>
      api<RecordDetail>(`/records/${region.record_id}`, signal),
  });
  const evidence = useQuery({
    queryKey: ["region-evidence", region],
    queryFn: ({ signal }) =>
      mutate<Evidence>("/regions/evidence", region, undefined, "POST", signal),
    retry: false,
    gcTime: 10000,
  });
  const slices = useQuery({
    queryKey: ["region-profile", region, detail.data?.source_sha256, profile],
    queryFn: ({ signal }) =>
      api<Slices>(
        `/records/${region.record_id}/slices?row=${profile}&column=${Math.floor((region.column_start + region.column_end - 1) / 2)}&row_start=${region.row_start}&row_end=${region.row_end}&source_sha256=${detail.data!.source_sha256}`,
        signal,
      ),
    gcTime: 10000,
    enabled: !!detail.data,
  });
  const generation = useRef(0);
  const selectionKey = processingKey(
    region,
    profile,
    method,
    detail.data?.source_sha256,
  );
  const currentSelection = useRef(selectionKey);
  currentSelection.current = selectionKey;
  const clean =
    processing?.selectionKey === selectionKey ? processing.values : undefined;
  const selectedRunId = clean ? processing?.runId : undefined;
  const prepared = useQuery({
    queryKey: ["prepared-profile-runs"],
    queryFn: ({ signal }) =>
      api<components["schemas"]["RunView"][]>("/runs?limit=100", signal),
    refetchOnWindowFocus: false,
  });
  const [pendingRun, setPendingRun] = useState(params.get("run_id") ?? "");
  const restoredRun = useRef(false);
  async function selectRun(runId: string) {
    const requested = selectionKey;
    const run = await api<Run>("/runs/" + encodeURIComponent(runId));
    if (currentSelection.current !== requested || !detail.data || !slices.data)
      return false;
    const values = processedProfile(
      run,
      region,
      profile,
      detail.data.source_sha256,
      slices.data.profile,
    );
    const key = processingKey(
      region,
      profile,
      run.method,
      detail.data.source_sha256,
    );
    setMethod(run.method);
    setProcessing({ values, runId: run.id, selectionKey: key, run });
  }
  useEffect(() => {
    const runId = params.get("run_id") ?? passport.data?.run_id;
    if (restoredRun.current || !runId || !detail.data || !slices.data) return;
    if (
      passportId &&
      (!passport.data ||
        profile !== passport.data.profile ||
        JSON.stringify(region) !== JSON.stringify(passport.data.region))
    )
      return;
    restoredRun.current = true;
    void selectRun(runId).catch((error) => setMessage(String(error)));
  }, [detail.data, slices.data, passport.data]);
  useEffect(() => {
    const saved = passport.data;
    if (!saved || !detail.data) return;
    if (saved.source_sha256 !== detail.data.source_sha256) {
      setMessage(
        "Версия источника паспорта недоступна; другой источник не подставлен.",
      );
      return;
    }
    setRegion(saved.region);
    setEditing(saved.region);
    setProfile(saved.profile);
    setMethod(saved.method);
    setTitle(saved.title);
    setNotes(saved.notes);
  }, [passport.data, detail.data]);
  useEffect(() => {
    generation.current++;
    setProcessing((value) =>
      value?.selectionKey === selectionKey ? value : undefined,
    );
  }, [selectionKey]);
  const d = detail.data,
    s = slices.data,
    e = evidence.data;
  const finite = s
    ? [
        ...s.profile.slice(region.column_start, region.column_end),
        ...(clean?.slice(region.column_start, region.column_end) ?? []),
      ].filter((v): v is number => v !== null && Number.isFinite(v))
    : [];
  const commonScale: [number, number] | undefined = finite.length
    ? [Math.min(...finite) - 0.1, Math.max(...finite) + 0.1]
    : undefined;
  const syncKey = `region-${region.record_id}-${region.row_start}-${region.column_start}`;
  const reportQuery = new URLSearchParams({
    profile: String(profile),
    method,
    package: proposal?.split(":")[0] ?? "general",
    review_id: proposal?.split(":")[1] ?? "",
    reviewer: sessionStorage.getItem("dts-reviewer") ?? "",
  });
  if (selectedRunId) reportQuery.set("run_id", selectedRunId);
  const path = d
    ? `/records/${region.record_id}/tiles?x0=${d.x_m[region.column_start]}&x1=${d.x_m[region.column_end - 1]}&t0=${d.time_s[region.row_start]}&t1=${Math.max(d.time_s[region.row_end - 1], d.time_s[region.row_start] + 1)}&width=128&height=128`
    : "";
  async function action(work: () => Promise<unknown>, success: string) {
    setMessage("");
    setBusy(true);
    try {
      const result = await work();
      if (result !== false) setMessage(success);
    } catch (error) {
      setMessage(String(error));
    } finally {
      setBusy(false);
    }
  }
  const limits: Record<string, number> = {
    row_start: (d?.shape[0] ?? 1) - 1,
    row_end: d?.shape[0] ?? 1,
    column_start: (d?.shape[1] ?? 2) - 2,
    column_end: d?.shape[1] ?? 2,
  };
  if (d && e && d.source_sha256 !== e.source_sha256)
    return (
      <section className="region-panel">
        <h2>Версия источника изменилась</h2>
        <p role="alert">
          Метаданные и исследовательское окно относятся к разным версиям.
          Обновите участок перед продолжением.
        </p>
        <Button
          onClick={() => {
            void detail.refetch();
            void evidence.refetch();
          }}
        >
          Обновить участок
        </Button>
      </section>
    );
  return (
    <section className="region-panel">
      <div className="section-title">
        <h2>Доказательства по участку</h2>
        <Tag type="gray">Физическая причина неизвестна</Tag>
      </div>
      <p>
        Границы по исходным индексам, конец не включён. Изменение границ создаёт
        ваш контекст исследования и не исправляет автоматически предложение
        метода.
      </p>
      <form
        className="region-controls"
        onSubmit={(event) => {
          event.preventDefault();
          if (
            editing.row_start >= editing.row_end ||
            editing.column_start + 1 >= editing.column_end
          ) {
            setMessage("Нужно минимум один профиль и две координаты.");
            return;
          }
          setRegion({ ...editing });
          setProfile(editing.row_start);
        }}
      >
        {(["row_start", "row_end", "column_start", "column_end"] as const).map(
          (key, i) => (
            <TextInput
              key={key}
              id={`region-${key}`}
              type="number"
              min={key === "column_end" ? 2 : key === "row_end" ? 1 : 0}
              max={limits[key]}
              step={1}
              labelText={
                [
                  "Первый профиль",
                  "Конец профилей",
                  "Первая координата (индекс)",
                  "Конец координат (индекс)",
                ][i]
              }
              value={editing[key]}
              onChange={(event) =>
                setEditing({ ...editing, [key]: Number(event.target.value) })
              }
              required
            />
          ),
        )}
        <Button type="submit" disabled={!d}>
          Исследовать участок
        </Button>
      </form>
      {evidence.error && (
        <InlineNotification
          kind="error"
          title="Участок недоступен"
          subtitle={String(evidence.error)}
          hideCloseButton
        />
      )}
      {e && (
        <div className="evidence-strip">
          <div>
            <span className="label">Измерения</span>
            <strong>{e.native_shape.join(" × ")}</strong>
          </div>
          <div>
            <span className="label">Валидное покрытие</span>
            <strong>{(e.valid_fraction * 100).toFixed(1)}%</strong>
          </div>
          <div>
            <span className="label">Температура min / median / max, °C</span>
            <strong>
              {Object.values(e.temperature_c)
                .map((v) => (v === null ? "—" : v.toFixed(3)))
                .join(" / ")}
            </strong>
          </div>
          <div>
            <span className="label">Разрывы более 300 с</span>
            <strong>{e.gaps_over_300s.length}</strong>
          </div>
        </div>
      )}
      {d && e && (
        <p className="mono">
          {d.x_m[region.column_start].toFixed(3)}–
          {d.x_m[region.column_end - 1].toFixed(3)} м ·{" "}
          {d.timestamps[region.row_start]} → {d.timestamps[region.row_end - 1]}{" "}
          · шкала {d.scale.map((v) => v.toFixed(2)).join("–")} °C
        </p>
      )}
      {path && e && (
        <Heatmap
          path={path}
          versionKey={e.source_sha256}
          expectedSourceSha={e.source_sha256}
          onSelect={(_x, t) => {
            if (d) {
              let nearest = region.row_start;
              for (let r = region.row_start; r < region.row_end; r++)
                if (Math.abs(d.time_s[r] - t) < Math.abs(d.time_s[nearest] - t))
                  nearest = r;
              setProfile(nearest);
            }
          }}
          cursor={
            d ? [d.x_m[region.column_start], d.time_s[profile]] : undefined
          }
        />
      )}
      <div className="region-controls">
        <TextInput
          id="comparison-row"
          labelText="Точный профиль (индекс)"
          type="number"
          min={region.row_start}
          max={region.row_end - 1}
          value={profile}
          onChange={(event) => {
            const value = Number(event.target.value);
            if (
              Number.isInteger(value) &&
              value >= region.row_start &&
              value < region.row_end
            )
              setProfile(value);
          }}
        />
        <Select
          id="comparison-method"
          labelText="Обработка профиля"
          value={method}
          onChange={(event) => {
            setMethod(event.target.value);
            setProcessing(undefined);
          }}
        >
          <SelectItem value="median_x3" text="Медиана ×3" />
          <SelectItem value="median_x5" text="Медиана ×5" />
          {!["median_x3", "median_x5"].includes(method) && (
            <SelectItem
              value={method}
              text={`${method} · сохранённый результат`}
            />
          )}
        </Select>
        <Button
          disabled={
            busy ||
            !s ||
            !e ||
            !detail.data ||
            !["median_x3", "median_x5"].includes(method)
          }
          onClick={() =>
            void action(async () => {
              const current = generation.current;
              const requestedSelection = selectionKey;
              const expectedSha = detail.data!.source_sha256;
              const result = await mutate<
                components["schemas"]["DenoiseResult"]
              >("/denoise", {
                record_id: region.record_id,
                row: profile,
                method,
              });
              if (
                current === generation.current &&
                currentSelection.current === requestedSelection
              ) {
                if (
                  result.source_sha256 !== expectedSha ||
                  result.row !== profile
                )
                  throw new Error(
                    "Обработка относится к другому источнику или профилю",
                  );
                const run = await api<Run>(
                  "/runs/" + encodeURIComponent(result.run_id),
                );
                if (currentSelection.current !== requestedSelection)
                  return false;
                const values = processedProfile(
                  run,
                  region,
                  profile,
                  expectedSha,
                  s!.profile,
                );
                setProcessing({
                  values,
                  runId: result.run_id,
                  selectionKey: requestedSelection,
                  run,
                });
              } else return false;
            }, "Обработка получена. Исходный сигнал сохранён.")
          }
        >
          Сравнить обработку
        </Button>
      </div>
      <div className="region-controls">
        <Select
          id="prepared-processing"
          labelText="Сохранённый результат профиля"
          value={pendingRun}
          onChange={(event) => setPendingRun(event.target.value)}
        >
          <SelectItem value="" text="Выберите расчёт" />
          {prepared.data
            ?.filter(
              (run) =>
                ["complete", "imported"].includes(run.status) &&
                run.source_sha256 === d?.source_sha256 &&
                run.region.record_id === region.record_id &&
                run.region.row_start === profile &&
                run.region.row_end === profile + 1 &&
                run.region.column_start <= region.column_start &&
                run.region.column_end >= region.column_end &&
                run.has_result,
            )
            .map((run) => (
              <SelectItem
                key={run.id}
                value={run.id}
                text={`${run.method} · ${run.method_version} · ${run.id.slice(0, 12)}`}
              />
            ))}
        </Select>
        <Button
          kind="secondary"
          disabled={busy || !pendingRun || !s || !d}
          onClick={() =>
            void action(
              () => selectRun(pendingRun),
              "Выбранный расчёт подключён к графику и отчёту",
            )
          }
        >
          Подключить результат
        </Button>
        <Button
          kind="ghost"
          disabled={!selectedRunId}
          onClick={() => setProcessing(undefined)}
        >
          Только исходный сигнал
        </Button>
        <Button kind="ghost" onClick={() => void prepared.refetch()}>
          Обновить результаты
        </Button>
        <Link className="text-link" to="/runs">
          Найти расчёт в истории запусков →
        </Link>
      </div>
      {prepared.error && (
        <p role="alert">
          Не удалось обновить список результатов. {String(prepared.error)}
        </p>
      )}
      {clean && processing && (
        <div className="evidence-strip" aria-label="Выбранная обработка">
          <div>
            <span className="label">Метод и версия</span>
            <strong>
              {processing.run.method} · {processing.run.method_version}
            </strong>
          </div>
          <div>
            <span className="label">Выбранный ID графика и отчёта</span>
            <code>{processing.runId}</code>
          </div>
          <div>
            <span className="label">Происхождение обучения</span>
            <strong>{processing.run.training_origin}</strong>
          </div>
          <div>
            <span className="label">Параметры и ограничения</span>
            <p>
              {JSON.stringify(processing.run.parameters)} ·{" "}
              {processing.run.limitations}
            </p>
          </div>
        </div>
      )}
      {s && e && (
        <div className="line-grid">
          <LineChart
            title="Исходный профиль"
            x={s.x_m.slice(region.column_start, region.column_end)}
            y={s.profile.slice(region.column_start, region.column_end)}
            xLabel="Расстояние, м"
            yLimits={commonScale}
            syncKey={syncKey}
          />
          {clean ? (
            <LineChart
              title="Обработанный профиль · та же область и шкала"
              x={s.x_m.slice(region.column_start, region.column_end)}
              y={clean.slice(region.column_start, region.column_end)}
              xLabel="Расстояние, м"
              yLimits={commonScale}
              syncKey={syncKey}
            />
          ) : (
            <LineChart
              title="Температура центральной точки · выбранное окно"
              x={s.time_s}
              y={s.trend}
              xLabel="Время от начала, с"
            />
          )}
        </div>
      )}
      {s && clean && (
        <LineChart
          title="Остаток raw − clean, °C · не доказанный шум"
          x={s.x_m.slice(region.column_start, region.column_end)}
          y={s.profile
            .slice(region.column_start, region.column_end)
            .map((v, i) =>
              v === null || clean[i + region.column_start] === null
                ? null
                : v - clean[i + region.column_start]!,
            )}
          xLabel="Расстояние, м"
        />
      )}
      {s && e && (
        <details>
          <summary>Точные измерения выбранного профиля</summary>
          <div className="numeric-scroll">
            <table>
              <thead>
                <tr>
                  <th>Расстояние, м</th>
                  <th>Исходная температура, °C</th>
                  <th>После обработки, °C</th>
                </tr>
              </thead>
              <tbody>
                {s.x_m
                  .slice(
                    region.column_start,
                    Math.min(region.column_end, region.column_start + 500),
                  )
                  .map((x, i) => (
                    <tr key={i}>
                      <td>{x.toFixed(4)}</td>
                      <td>
                        {s.profile[region.column_start + i]?.toFixed(6) ??
                          "Нет измерения"}
                      </td>
                      <td>
                        {clean?.[region.column_start + i]?.toFixed(6) ?? "—"}
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
          <p>
            Первые 500 координат участка. Полный профиль доступен через API
            точных срезов.
          </p>
        </details>
      )}
      <div className="region-controls">
        <TextInput
          id="bookmark-title"
          labelText="Название закладки"
          value={title}
          maxLength={150}
          onChange={(event) => setTitle(event.target.value)}
        />
        <TextArea
          id="bookmark-notes"
          labelText="Наблюдения и ограничения"
          value={notes}
          maxLength={8000}
          onChange={(event) => setNotes(event.target.value)}
        />
      </div>
      <div className="action-row">
        <Button
          disabled={busy || !e || !title.trim()}
          onClick={() =>
            void action(
              () =>
                mutate("/bookmarks", {
                  ...region,
                  title,
                  notes,
                  proposal_id: proposal ?? null,
                }),
              "Закладка сохранена сервером",
            )
          }
        >
          Сохранить участок
        </Button>
        {account && (
          <Button
            kind="secondary"
            disabled={busy || !d || !title.trim()}
            onClick={() =>
              void action(async () => {
                const saved = await control<{ id: string }>("/passports", {
                  title,
                  notes,
                  region,
                  profile,
                  method,
                  source_sha256: d!.source_sha256,
                  scale: d!.scale,
                  proposal_id: proposal ?? null,
                  run_id: selectedRunId ?? null,
                });
                const link = new URL(regionLink(region), location.origin);
                link.searchParams.set("passport", saved.id);
                link.searchParams.set("row", String(profile));
                if (selectedRunId)
                  link.searchParams.set("run_id", selectedRunId);
                setPassportLink(link.toString());
                await navigator.clipboard
                  ?.writeText(link.toString())
                  .catch(() => {});
              }, "Паспорт сохранён; ссылка скопирована")
            }
          >
            Сохранить паспорт и ссылку
          </Button>
        )}
        <Link className="text-link" to={regionLink(region)}>
          Открыть исследование по ссылке →
        </Link>
        <Button
          kind="tertiary"
          disabled={busy || !e}
          onClick={() =>
            void action(
              () =>
                download(
                  "/reports?format=html&" + reportQuery,
                  region,
                  "dts-evidence.html",
                ),
              "Отчёт скачан",
            )
          }
        >
          Отчёт HTML
        </Button>
        <Button
          kind="ghost"
          disabled={busy || !e}
          onClick={() =>
            void action(
              () =>
                download(
                  "/reports?" + reportQuery,
                  region,
                  "dts-evidence.json",
                ),
              "Данные отчёта скачаны",
            )
          }
        >
          JSON
        </Button>
      </div>
      {passportLink && (
        <p>
          <a className="text-link" href={passportLink}>
            Открыть сохранённый паспорт →
          </a>
        </p>
      )}
      {account && d && s && (
        <CaseForm
          key={selectionKey + (selectedRunId ?? "raw")}
          region={region}
          profile={profile}
          sha={d.source_sha256}
          runId={selectedRunId}
          title={title}
          notes={notes}
          proposal={proposal}
        />
      )}
      {message && <p role="status">{message}</p>}
      {e && (
        <details>
          <summary>Происхождение и ограничения</summary>
          <code>{e.source_sha256}</code>
          <p>
            {e.time_kind} · Обработка не доказывает наличие аномалии. Цвет
            фиксирован для источника; точные значения могут выходить за его
            пределы.
          </p>
        </details>
      )}
    </section>
  );
}

export function Candidates({ demo = false }: { demo?: boolean }) {
  const [params, setParams] = useSearchParams(),
    [playing, setPlaying] = useState(false),
    [error, setError] = useState("");
  const packages = useQuery({
    queryKey: ["packages"],
    queryFn: ({ signal }) => api<Package[]>("/packages", signal),
  });
  const packageId =
    params.get("package") ??
    (packages.data?.some((p) => p.id === "real-data-v2")
      ? "real-data-v2"
      : "general");
  const shortlist = demo || params.get("shortlist") !== "false";
  const candidates = useQuery({
    queryKey: ["candidates", packageId, shortlist],
    queryFn: ({ signal }) =>
      api<Candidate[]>(
        `/candidates?package=${packageId}&shortlist=${shortlist && packageId === "real-data-v2"}`,
        signal,
      ),
  });
  const cards = candidates.data ?? [],
    current = cards.find((c) => c.id === params.get("candidate")) ?? cards[0],
    index = cards.findIndex((c) => c.id === current?.id);
  const [bookmarks, setBookmarks] = useState<
    (Region & { id: string; title: string })[]
  >([]);
  useEffect(() => {
    void api<(Region & { id: string; title: string })[]>("/bookmarks")
      .then(setBookmarks)
      .catch((e) => setError(String(e)));
  }, []);
  useEffect(() => {
    if (!demo || !playing || !cards.length) return;
    const timer = setInterval(() => {
      const next = cards[(index + 1) % cards.length];
      setParams({ package: packageId, candidate: next.id });
      if (index + 1 === cards.length) setPlaying(false);
    }, 12000);
    return () => clearInterval(timer);
  }, [demo, playing, index, cards.length, packageId, setParams, cards]);
  function step(delta: number) {
    setPlaying(false);
    const next = cards[Math.max(0, Math.min(cards.length - 1, index + delta))];
    if (next)
      setParams({
        package: packageId,
        candidate: next.id,
        shortlist: String(shortlist),
      });
  }
  return (
    <main className="page">
      <div className="page-title">
        <div>
          <p className="eyebrow">
            Рабочая область / {demo ? "Демонстрация" : "Исследовательские окна"}
          </p>
          <h1>
            {demo
              ? "От измерения к проверяемому выводу"
              : "Кандидаты и доказательства"}
          </h1>
        </div>
        <Tag type="gray">
          {demo
            ? "Настоящие записи · офлайн-гипотезы"
            : "Без подтверждённых причин"}
        </Tag>
      </div>
      <p>
        {demo
          ? "Маршрут по реальным исследовательским окнам. Автопереход каждые 12 секунд переключает контекст; воспроизведение самой записи доступно в Исследовании. Это не подключение к оборудованию."
          : "Окна для исследования и проверки. Наличие окна в подборке не означает подтверждённую аномалию."}
      </p>
      <div className="region-controls">
        <Select
          id="research-package"
          labelText="Пакет данных"
          value={packageId}
          onChange={(event) => {
            setPlaying(false);
            setParams({ package: event.target.value });
          }}
        >
          {packages.data?.map((p) => (
            <SelectItem
              key={p.id}
              value={p.id}
              text={`${p.title} · ${p.cards} окон`}
            />
          ))}
        </Select>
        <Select
          id="candidate-select"
          labelText={`Окно ${index + 1} / ${cards.length}`}
          value={current?.id ?? ""}
          onChange={(event) => {
            setPlaying(false);
            setParams({
              package: packageId,
              candidate: event.target.value,
              shortlist: String(shortlist),
            });
          }}
        >
          {cards.map((c) => (
            <SelectItem
              key={c.id}
              value={c.id}
              text={`${c.id} · ${c.region.record_id}.csv · ${c.x_bounds_m.map((x) => x.toFixed(1)).join("–")} м`}
            />
          ))}
        </Select>
        {!demo && packageId === "real-data-v2" && (
          <Button
            kind="tertiary"
            onClick={() =>
              setParams({ package: packageId, shortlist: String(!shortlist) })
            }
          >
            {shortlist ? "Все 340 окон" : "Короткая подборка"}
          </Button>
        )}
      </div>
      <div className="action-row">
        <Button kind="secondary" disabled={index <= 0} onClick={() => step(-1)}>
          Предыдущее
        </Button>
        <Button disabled={index >= cards.length - 1} onClick={() => step(1)}>
          Следующее
        </Button>
        {demo && (
          <>
            <Button
              kind="tertiary"
              disabled={!cards.length}
              onClick={() => setPlaying((v) => !v)}
            >
              {playing ? "Пауза маршрута" : "Автопереход"}
            </Button>
            <Button
              kind="ghost"
              disabled={!cards.length}
              onClick={() => {
                setPlaying(false);
                setParams({ package: packageId, candidate: cards[0].id });
              }}
            >
              С начала
            </Button>
          </>
        )}
      </div>
      {demo && (
        <ol className="demo-steps">
          <li>Исходные измерения</li>
          <li>Покрытие и разрывы</li>
          <li>Точный профиль и обработка</li>
          <li>Самостоятельный ответ и отчёт</li>
        </ol>
      )}
      {(candidates.error || packages.error || error) && (
        <InlineNotification
          kind="error"
          title="Не удалось открыть контекст"
          subtitle={String(candidates.error ?? packages.error ?? error)}
          hideCloseButton
        />
      )}
      {current && (
        <>
          {current.proposal_bounds?.x_bounds_m && (
            <p>
              Предложенное ядро:{" "}
              {current.proposal_bounds.x_bounds_m
                .map((x) => x.toFixed(3))
                .join("–")}{" "}
              м; ниже показано окружение. Границы предложения не подтверждены
              экспертом.
            </p>
          )}
          <div
            onFocusCapture={() => setPlaying(false)}
            onPointerDownCapture={() => setPlaying(false)}
          >
            <RegionPanel
              key={`${packageId}-${current.id}`}
              initial={current.region}
              proposal={`${packageId}:${current.id}`}
            />
          </div>
          <div className="action-row">
            {!demo && (
              <Link
                className="text-link"
                to={
                  "/works?" +
                  new URLSearchParams({
                    record: current.region.record_id,
                    source_sha256: current.source_sha256,
                    r0: String(current.region.row_start),
                    r1: String(current.region.row_end),
                    c0: String(current.region.column_start),
                    c1: String(current.region.column_end),
                    row: String(current.region.row_start),
                    column: String(current.region.column_start),
                    origin: "review",
                  })
                }
              >
                Начать работу с этим окном →
              </Link>
            )}
            <Link
              className="text-link"
              to={`/review?package=${packageId}&card=${current.id}`}
            >
              Дать самостоятельный ответ →
            </Link>
            <Link className="text-link" to={regionLink(current.region)}>
              Открыть поле и исторический replay →
            </Link>
          </div>
        </>
      )}
      {!demo && (
        <details>
          <summary>Сохранённые участки ({bookmarks.length})</summary>
          {bookmarks.map((b) => (
            <p key={b.id}>
              <Link to={regionLink(b)}>
                {b.title} · {b.record_id}.csv
              </Link>
            </p>
          ))}
          <p>Новые закладки появятся после повторного открытия раздела.</p>
        </details>
      )}
    </main>
  );
}

export function Runs() {
  const [handoff, setHandoff] =
    useState<components["schemas"]["ResultImport"]>();
  const [importing, setImporting] = useState(false);
  const [imported, setImported] = useState<Run>();
  const methods = useQuery({
    queryKey: ["method-registry"],
    queryFn: ({ signal }) =>
      api<components["schemas"]["MethodView"][]>("/methods", signal),
    refetchInterval: 10000,
  });
  const [runOffset, setRunOffset] = useState(0);
  const [selectedRun, setSelectedRun] = useState<string>();
  const result = useQuery({
    queryKey: ["run-result", selectedRun],
    queryFn: ({ signal }) => api<Run>(`/runs/${selectedRun}`, signal),
    enabled: !!selectedRun,
    gcTime: 10000,
  });
  const runs = useQuery({
      queryKey: ["runs", runOffset],
      queryFn: ({ signal }) =>
        api<Run[]>(`/runs?limit=20&offset=${runOffset}`, signal),
      refetchInterval: 3000,
    }),
    [error, setError] = useState("");
  return (
    <main className="page">
      <p className="eyebrow">Рабочая область / Методы</p>
      <h1>Запуски и происхождение результатов</h1>
      <p>
        Сохранённые расчёты и результаты Захара. Перезапуск сервера не удаляет
        историю. Отмена выполняющегося HTTP-расчёта подтверждается после
        завершения вызова; результат отменённой задачи не публикуется.
      </p>
      <div className="action-row">
        <Button
          kind="ghost"
          disabled={!runOffset}
          onClick={() => setRunOffset(Math.max(0, runOffset - 20))}
        >
          Предыдущие запуски
        </Button>
        <Button
          kind="ghost"
          disabled={(runs.data?.length ?? 0) < 20}
          onClick={() => setRunOffset(runOffset + 20)}
        >
          Следующие запуски
        </Button>
      </div>
      <details>
        <summary>Методы и происхождение</summary>
        <div className="numeric-scroll">
          <table>
            <thead>
              <tr>
                <th>Метод</th>
                <th>Версия</th>
                <th>Обучение</th>
                <th>Доступность</th>
              </tr>
            </thead>
            <tbody>
              {methods.data?.map((m) => (
                <tr key={m.id + m.version}>
                  <td>{m.id}</td>
                  <td>{m.version}</td>
                  <td>{m.training_origin}</td>
                  <td>{m.availability}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {methods.error && <p role="alert">{String(methods.error)}</p>}
      </details>
      <label className="result-import">
        Подключить результат Захара (JSON, контракт dts-method-result-v1, до 240
        KiB)
        <input
          type="file"
          accept="application/json,.json"
          onChange={async (event) => {
            const file = event.target.files?.[0];
            if (!file) return;
            try {
              if (file.size > 240 * 1024)
                throw new Error(
                  "Файл превышает 240 KiB. Большие массивы подключаются серверным адаптером.",
                );
              const value = JSON.parse(
                await file.text(),
              ) as components["schemas"]["ResultImport"];
              if (
                value.schema_version !== "dts-method-result-v1" ||
                !value.region ||
                typeof value.method !== "string" ||
                typeof value.method_version !== "string" ||
                typeof value.source_sha256 !== "string" ||
                !value.result ||
                typeof value.result !== "object"
              )
                throw new Error(
                  "Нужен полный файл dts-method-result-v1. Проверьте упаковку результата.",
                );
              setHandoff(value);
              setImported(undefined);
              setError("");
            } catch (e) {
              setError(String(e));
            }
            event.target.value = "";
          }}
        />
      </label>
      {handoff && (
        <section
          className="case-preview"
          aria-labelledby="import-preview-title"
        >
          <h2 id="import-preview-title">Проверить перед импортом</h2>
          <dl className="case-facts">
            <div>
              <dt>Источник / SHA</dt>
              <dd>
                {handoff.region.record_id}.csv ·{" "}
                <code>{handoff.source_sha256}</code>
              </dd>
            </div>
            <div>
              <dt>Область</dt>
              <dd>
                Строки [{handoff.region.row_start}, {handoff.region.row_end}),
                координаты [{handoff.region.column_start},{" "}
                {handoff.region.column_end})
              </dd>
            </div>
            <div>
              <dt>Метод и версия</dt>
              <dd>
                {handoff.method} · {handoff.method_version}
              </dd>
            </div>
            <div>
              <dt>Обучение</dt>
              <dd>{handoff.training_origin}</dd>
            </div>
          </dl>
          <p>{handoff.limitations}</p>
          <p>
            {Array.isArray(handoff.result.denoised)
              ? `Температурный профиль: ${handoff.result.denoised.length} точек; полную исходную форму проверит сервер.`
              : "Температурного профиля нет. Результат будет доступен как сводка; подключение к профильному графику невозможно."}
          </p>
          <details>
            <summary>Параметры расчёта</summary>
            <pre>{JSON.stringify(handoff.parameters, null, 2)}</pre>
          </details>
          <div className="action-row">
            <Button
              disabled={importing}
              onClick={async () => {
                setImporting(true);
                setError("");
                try {
                  const saved = await mutate<Run>("/results", handoff);
                  setImported(saved);
                  setHandoff(undefined);
                  setRunOffset(0);
                  await runs.refetch();
                } catch (e) {
                  setError(String(e));
                } finally {
                  setImporting(false);
                }
              }}
            >
              {importing ? "Проверяем и импортируем…" : "Подтвердить импорт"}
            </Button>
            <Button
              kind="ghost"
              disabled={importing}
              onClick={() => setHandoff(undefined)}
            >
              Отменить импорт
            </Button>
          </div>
        </section>
      )}
      {imported && (
        <p role="status">
          Результат импортирован.{" "}
          <Link
            to={`${regionLink(imported.region)}&run_id=${encodeURIComponent(imported.id)}&source_sha256=${imported.source_sha256}`}
          >
            Исследовать выбранный результат →
          </Link>
        </p>
      )}
      {(error || runs.error) && (
        <InlineNotification
          kind="error"
          title="Результат не подключён"
          subtitle={error || String(runs.error)}
          hideCloseButton
        />
      )}
      {runs.data?.length === 0 && (
        <p>
          Расчётов пока нет. Запустите ограниченный анализ участка в
          Исследовании или подключите подготовленный результат.
        </p>
      )}
      <div className="run-list">
        {runs.data?.map((run) => (
          <article key={run.id}>
            <div className="section-title">
              <h2>{run.method}</h2>
              <Tag type="gray">{run.status}</Tag>
            </div>
            <p>
              {run.method_version} · происхождение обучения:{" "}
              {run.training_origin}
            </p>
            <p>
              {new Date(run.updated_at * 1000).toLocaleString("ru")} ·{" "}
              {run.region.record_id}.csv
            </p>
            <p>{run.limitations}</p>
            <code>{run.source_sha256}</code>
            <div className="action-row">
              <Link
                className="text-link"
                to={`${regionLink(run.region)}&run_id=${encodeURIComponent(run.id)}&source_sha256=${run.source_sha256}`}
              >
                Исследовать участок →
              </Link>
              {["running", "queued"].includes(run.status) && (
                <Button
                  kind="tertiary"
                  onClick={async () => {
                    try {
                      await mutate(`/jobs/${run.id}/cancel`, {});
                      await runs.refetch();
                    } catch (e) {
                      setError(String(e));
                    }
                  }}
                >
                  Запросить отмену
                </Button>
              )}
            </div>
            {run.error && <p role="alert">{run.error}</p>}
            {run.result_summary?.processed_points && (
              <p>
                Обработано{" "}
                {run.result_summary.processed_points.toLocaleString("ru")}{" "}
                координат профиля. Классификация событий не выполнялась.
              </p>
            )}
            {run.result_summary?.zone_count !== undefined &&
              run.result_summary?.zone_count !== null && (
                <p>
                  Предложенных зон: {run.result_summary.zone_count}. Это
                  гипотезы, без экспертного подтверждения.
                </p>
              )}
            {!!run.result_summary?.zones?.length && (
              <div className="numeric-scroll">
                <table>
                  <thead>
                    <tr>
                      <th>Расстояние, м</th>
                      <th>Длительность, с</th>
                      <th>Направление</th>
                      <th>Контраст, °C</th>
                    </tr>
                  </thead>
                  <tbody>
                    {run.result_summary.zones.map((zone, i) => (
                      <tr key={i}>
                        <td>
                          {String(zone.start_m ?? "—")}–
                          {String(zone.end_m ?? "—")}
                        </td>
                        <td>{String(zone.duration_s ?? "—")}</td>
                        <td>{String(zone.polarity ?? "—")}</td>
                        <td>{String(zone.median_contrast_c ?? "—")}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p>Первые 100 зон. Полный результат открывается отдельно.</p>
              </div>
            )}
            {run.has_result && (
              <Button
                kind="ghost"
                onClick={() =>
                  setSelectedRun(selectedRun === run.id ? undefined : run.id)
                }
              >
                {selectedRun === run.id
                  ? "Скрыть полный результат"
                  : "Открыть полный результат"}
              </Button>
            )}
            {selectedRun === run.id && result.data?.result !== undefined && (
              <details>
                <summary>Числовой результат метода</summary>
                <pre>{JSON.stringify(result.data.result, null, 2)}</pre>
              </details>
            )}
            {selectedRun === run.id && result.error && (
              <p role="alert">{String(result.error)}</p>
            )}
          </article>
        ))}
      </div>
    </main>
  );
}
