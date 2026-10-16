import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Button,
  InlineNotification,
  Select,
  SelectItem,
  TextInput,
} from "@carbon/react";
import { Link, useSearchParams } from "react-router-dom";
import { control, useAccount } from "./access";
import { ApiError, api, type RecordDetail, type RecordMeta, type Slices } from "./api";
import { analysisParams } from "./analysis-context";
import { CoordinateInput } from "./coordinate-input";
import { Heatmap, LineChart } from "./charts";
import { profileChanges } from "./profile-changes";
import { breakTimeGaps, qualityLabel } from "./presentation-model";
import {
  PIPELINE_BLOCKS,
  absentLayer,
  afterAbort,
  applyLateResult,
  applySavedSetting,
  attachJob,
  comparisonCaption,
  changedSettingsPatch,
  compareResults,
  editBoundField,
  editDraft,
  explicitSubmit,
  formatWindow,
  limitedWindow,
  parseHalfOpen,
  rememberRevision,
  resolvePipelineLink,
  resultDifference,
  resultName,
  savePrompt,
  sameBody,
  samePoint,
  sameScoreDefinition,
  sameSettingsDraft,
  sliceForPoint,
  sourceTimeBounds,
  stageText,
  submitBody,
  type Attempt,
  type BoundDraft,
  type CompareRun,
  type HalfOpenWindow,
  type PipelineBlock,
  type PipelineSession,
  type SettingsDraft,
} from "./pipeline-state";
import {
  STAGES,
  checkStages,
  readStoredSource,
  stageStatus,
  writeStoredSource,
  type StageError,
  type StageId,
} from "./pipeline-stages";
import { PipelineWorkbench } from "./pipeline-workbench";
import "./pipeline.scss";

type Job = {
  id: string;
  status: string;
  stage: string;
  region: HalfOpenWindow & { record_id: string };
  source_sha256: string;
  method: string;
  method_version: string;
  cancel_requested: boolean;
  run_id?: string | null;
  error?: string | null;
  created_at?: number | null;
  settings_id?: string | null;
  settings_revision?: number | null;
  settings_revision_after?: number | null;
  author_name?: string | null;
};

type PipelineSetting = {
  id: string;
  title: string;
  record_id: string;
  source_sha256: string;
  window: HalfOpenWindow;
  point: { row: number; column: number };
  method: string;
  method_version: string;
  parameters?: Record<string, unknown>;
  run_ids: string[];
  job_ids: string[];
  author_name: string;
  revision: number;
  created_at: number | null;
};

type PreflightReport = {
  accepted?: boolean;
  ready?: boolean;
  method?: string;
  method_version?: string;
  max_cells?: number;
  timeout_s?: number;
  warning?: string | null;
};

type HistoryRow = { id: string; job?: Job; error?: string };

type Run = {
  id: string;
  region: HalfOpenWindow & { record_id: string };
  source_sha256: string;
  method: string;
  method_version: string;
  layers: string[];
  training_origin: string;
  research_status: string;
  limitations: string;
  parameters: object;
  calculated_at?: string | null;
  time_kind?: string;
  scales?: Record<string, [number, number]>;
  score?: {
    name?: string;
    unit?: string;
    meaning?: string;
    threshold?: number | null;
    comparison?: string;
    kind?: string;
  } | null;
  zones: {
    id: string;
    row_start: number;
    row_end: number;
    column_start: number;
    column_end: number;
    shape_hypothesis?: string;
  }[];
};

type RunSlices = {
  run_id: string;
  source_sha256: string;
  row: number;
  column: number;
  x_m: number[];
  time_s: number[];
  native_time_s: number[];
  layers: Record<
    string,
    { profile: (number | null)[]; trend: (number | null)[]; coverage: (number | null)[] }
  >;
  source_invalid_reason_profile: number[];
};

type ZoneBox = {
  id: string;
  segment: number;
  x0: number;
  x1: number;
  t0: number;
  t1: number;
  label: string;
  source_sha256?: string;
  run_id?: string;
};

function replaceParams(
  setParams: ReturnType<typeof useSearchParams>[1],
  change: (params: URLSearchParams) => void,
) {
  setParams(
    (previous) => {
      const next = new URLSearchParams(previous);
      change(next);
      return next;
    },
    { replace: true, preventScrollReset: true },
  );
}

function placed(value: string | null, start: number, end: number) {
  if (value !== null && /^\d+$/.test(value)) {
    const number = Number(value);
    if (Number.isSafeInteger(number) && start <= number && number < end) return number;
  }
  return start;
}

function finiteExtent(values: (number | null)[]): [number, number] | undefined {
  let lo = Infinity;
  let hi = -Infinity;
  for (const value of values) {
    if (value === null) continue;
    lo = Math.min(lo, value);
    hi = Math.max(hi, value);
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return undefined;
  return lo === hi ? [lo - 1, hi + 1] : [lo, hi];
}

function symmetricScale(
  values: (number | null)[],
  server: [number, number] | undefined,
): [number, number] {
  if (
    server &&
    Number.isFinite(server[0]) &&
    Number.isFinite(server[1]) &&
    server[0] < 0 &&
    server[1] > 0
  ) {
    const span = Math.max(Math.abs(server[0]), Math.abs(server[1]));
    return [-span, span];
  }
  let span = 0.01;
  for (const value of values) if (value !== null) span = Math.max(span, Math.abs(value));
  return [-span, span];
}

function textNumber(value: number | null | undefined, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(value)) return "Нет данных";
  return value.toLocaleString("ru", { maximumFractionDigits: digits });
}

function nearest(axis: number[], value: number) {
  return axis.reduce(
    (best, item, index) =>
      Math.abs(item - value) < Math.abs(axis[best] - value) ? index : best,
    0,
  );
}

/** Heatmap `t` stays one sample per row: `nearest` returns a row index. The line uses `breakTimeGaps`. */
function sourceFieldAxes(
  distance: number[],
  time: number[],
  bounds: HalfOpenWindow,
): { x: number[]; t: number[]; timeGap: boolean } {
  const t = time.slice(bounds.row_start, bounds.row_end);
  const broken = breakTimeGaps(t, t.map((): number | null => null));
  return {
    x: distance.slice(bounds.column_start, bounds.column_end),
    t,
    timeGap: broken.x.length > t.length,
  };
}

function asNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function readSetting(value: unknown): PipelineSetting | null {
  if (!value || typeof value !== "object") return null;
  const item = value as Partial<PipelineSetting> & { window?: Partial<HalfOpenWindow>; point?: Partial<PipelineSetting["point"]> };
  const window = item.window;
  const point = item.point;
  if (
    typeof item.id !== "string" ||
    typeof item.revision !== "number" ||
    typeof item.record_id !== "string" ||
    typeof item.source_sha256 !== "string" ||
    !window ||
    asNumber(window.row_start) === null ||
    asNumber(window.row_end) === null ||
    asNumber(window.column_start) === null ||
    asNumber(window.column_end) === null ||
    !point ||
    asNumber(point.row) === null ||
    asNumber(point.column) === null
  )
    return null;
  const strings = (input: unknown) =>
    Array.isArray(input) ? input.filter((entry): entry is string => typeof entry === "string") : [];
  return {
    id: item.id,
    title: typeof item.title === "string" ? item.title : "",
    record_id: item.record_id,
    source_sha256: item.source_sha256,
    window: {
      row_start: window.row_start as number,
      row_end: window.row_end as number,
      column_start: window.column_start as number,
      column_end: window.column_end as number,
    },
    point: { row: point.row as number, column: point.column as number },
    method: typeof item.method === "string" ? item.method : "",
    method_version: typeof item.method_version === "string" ? item.method_version : "",
    parameters: item.parameters && typeof item.parameters === "object" ? item.parameters : {},
    run_ids: strings(item.run_ids).slice(0, 2),
    job_ids: strings(item.job_ids),
    author_name: typeof item.author_name === "string" ? item.author_name : "",
    revision: item.revision,
    created_at: typeof item.created_at === "number" ? item.created_at : null,
  };
}

function readSettingList(value: unknown): PipelineSetting[] {
  const source = Array.isArray(value)
    ? value
    : value && typeof value === "object" && Array.isArray((value as { items?: unknown }).items)
      ? (value as { items: unknown[] }).items
      : [];
  return source.flatMap((item) => {
    const setting = readSetting(item);
    return setting ? [setting] : [];
  });
}

function settingDraft(setting: PipelineSetting): SettingsDraft {
  return {
    title: setting.title,
    record_id: setting.record_id,
    source_sha256: setting.source_sha256,
    window: setting.window,
    point: setting.point,
    method: setting.method,
    method_version: setting.method_version,
    run_ids: setting.run_ids,
  };
}

function windowLabel(recordId: string, window: HalfOpenWindow) {
  return `запись ${recordId}, строки [${window.row_start}, ${window.row_end}), столбцы [${window.column_start}, ${window.column_end})`;
}

function settingLabel(item: PipelineSetting) {
  const title = item.title.trim() || "без названия";
  return `${title} · ${resultName({
    recordId: item.record_id,
    window: item.window,
    calculatedAt: item.created_at,
  })}`;
}

function conflictText(reason: unknown, fallback: string) {
  if (!(reason instanceof ApiError) || reason.status !== 409) return null;
  return reason.message.startsWith("Не удалось") ? fallback : reason.message;
}

function maskText(reasons: number[] | undefined) {
  if (!reasons) return "Срез не получен";
  const masked = reasons.filter((value) => value !== 0).length;
  return `${masked} из ${reasons.length} точек профиля с ненулевым кодом качества. Это маска источника, не оценка обнаружения.`;
}

function coverageText(values: (number | null)[] | undefined) {
  if (!values) return "Слой не передан";
  const numbers = values.filter((value): value is number => typeof value === "number");
  if (!numbers.length) return "Нет данных";
  const mean = numbers.reduce((sum, value) => sum + value, 0) / numbers.length;
  return `${(mean * 100).toFixed(1)}% по ${numbers.length.toLocaleString("ru")} строкам. Покрытие каждого результата отдельно.`;
}

function sameAxis(a: number[] | undefined, b: number[] | undefined) {
  return !!a && !!b && a.length === b.length && a.every((value, index) => value === b[index]);
}

function compareRun(run: Run): CompareRun {
  return {
    id: run.id,
    record_id: run.region.record_id,
    source_sha256: run.source_sha256,
    window: {
      row_start: run.region.row_start,
      row_end: run.region.row_end,
      column_start: run.region.column_start,
      column_end: run.region.column_end,
    },
  };
}

async function loadCompareSlice(
  run: Run,
  window: HalfOpenWindow,
  point: { row: number; column: number },
  signal: AbortSignal,
): Promise<RunSlices> {
  const value = await control<RunSlices>(
    `/result-runs/${encodeURIComponent(run.id)}/slices?${new URLSearchParams({
      row: String(point.row),
      column: String(point.column),
      row_start: String(window.row_start),
      row_end: String(window.row_end),
      column_start: String(window.column_start),
      column_end: String(window.column_end),
    })}`,
    undefined,
    signal,
  );
  if (value.run_id !== run.id || value.source_sha256 !== run.source_sha256)
    throw new Error("Срез сравнения не соответствует выбранному результату");
  return value;
}

type StageCheckReport = {
  ok?: boolean;
  stage_id?: StageId;
  line?: number;
  message?: string;
};

function shownStageMessage(message: string): string {
  const text = message.toLowerCase();
  if (text.includes("tab")) return "табуляция вместо отступа";
  if (text.includes("indent")) return "после условия ожидался отступ";
  if (text.includes("never closed") || text.includes("unmatched")) return "скобка не закрыта";
  return message;
}

async function interpretStages(sources: Record<StageId, string>): Promise<StageError | null> {
  try {
    const report = await control<StageCheckReport>("/pipeline-check", {
      stages: STAGES.map((stage) => ({ id: stage.id, source: sources[stage.id] ?? "" })),
    });
    if (report.ok === true) return null;
    const spec = STAGES.find((stage) => stage.id === report.stage_id);
    if (report.ok === false && spec && typeof report.line === "number" && typeof report.message === "string") {
      return {
        stageId: spec.id,
        label: spec.label,
        line: report.line,
        message: shownStageMessage(report.message),
      };
    }
    return checkStages(sources);
  } catch (reason) {
    if (reason instanceof ApiError && (reason.status === 404 || reason.status === 405))
      return checkStages(sources);
    throw reason;
  }
}

export default function Pipeline() {
  const account = useAccount();
  const queryClient = useQueryClient();
  const [params, setParams] = useSearchParams();
  // React Router's setter callback closes over its render's URL, unlike setState.
  // Async replies must use the latest URL rather than the URL at submission time.
  const navigation = useRef({ params, setParams });
  navigation.current = { params, setParams };
  const paramKey = params.toString();
  const catalog = useQuery({
    queryKey: ["pipeline-records", account?.id],
    enabled: !!account,
    queryFn: ({ signal }) => api<RecordMeta[]>("/records", signal),
    retry: false,
  });
  const recordId = params.get("record") ?? "";
  const detail = useQuery({
    queryKey: ["pipeline-record", account?.id, recordId],
    enabled: !!account && !!recordId,
    queryFn: ({ signal }) => api<RecordDetail>(`/records/${recordId}`, signal),
    retry: false,
  });
  const capability = useQuery({
    queryKey: ["analysis-capability", account?.id],
    enabled: !!account,
    queryFn: ({ signal }) =>
      control<{ ready: boolean; method: string; method_version: string; max_cells: number }>(
        "/analysis-capability",
        undefined,
        signal,
      ),
    refetchInterval: 10000,
    retry: false,
  });
  const jobId = params.get("job_id") ?? "";
  const jobQuery = useQuery({
    queryKey: ["pipeline-job", account?.id, jobId],
    enabled: !!account && !!jobId,
    retry: false,
    queryFn: ({ signal }) => control<Job>(`/analysis-jobs/${jobId}`, undefined, signal),
    refetchInterval: (query) => {
      if (
        query.state.error instanceof ApiError &&
        query.state.error.status >= 400 &&
        query.state.error.status < 500
      )
        return false;
      const status = query.state.data?.status ?? "";
      return ["complete", "failed", "cancelled"].includes(status)
        ? false
        : document.hidden
          ? 10000
          : 2000;
    },
  });
  const job = jobQuery.data;
  const block = PIPELINE_BLOCKS.some((item) => item.id === params.get("block"))
    ? (params.get("block") as PipelineBlock)
    : "source";
  const [draftText, setDraftText] = useState<BoundDraft>(["", "", "", ""]);
  const [draftDirty, setDraftDirty] = useState(false);
  const [title, setTitle] = useState("");
  const [compareA, setCompareA] = useState("");
  const [compareB, setCompareB] = useState("");
  const [preflight, setPreflight] = useState<PreflightReport | null>(null);
  const [settingsError, setSettingsError] = useState("");
  const saveAttempt = useRef<{ key: string; body: string } | null>(null);
  const draftKey = draftText.join("|");
  const [session, setSession] = useState<PipelineSession>({
    draft: { row_start: 0, row_end: 1, column_start: 0, column_end: 2 },
    submitted: null,
    attempt: null,
    jobId: null,
    shownRunId: null,
  });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [cleanView, setCleanView] = useState<"raw" | "processed_temperature_c" | "difference">(
    "processed_temperature_c",
  );
  const [scoreView, setScoreView] = useState<"background_temperature_c" | "score">("score");
  const [stageActive, setStageActive] = useState<StageId>("clean");
  const [stageSources, setStageSources] = useState<Record<StageId, string>>(() =>
    Object.fromEntries(STAGES.map((stage) => [stage.id, stage.source])) as Record<StageId, string>,
  );
  const [stageSaved, setStageSaved] = useState<Record<StageId, string | null>>(() =>
    Object.fromEntries(STAGES.map((stage) => [stage.id, null])) as Record<StageId, string | null>,
  );
  const [stageRemoteError, setStageRemoteError] = useState<StageError | null>(null);
  const [stageChecking, setStageChecking] = useState(false);
  const stageError = useMemo(
    () => stageRemoteError ?? checkStages(stageSources),
    [stageRemoteError, stageSources],
  );
  useEffect(() => {
    if (!recordId) return;
    const sources = {} as Record<StageId, string>;
    const saved = {} as Record<StageId, string | null>;
    for (const stage of STAGES) {
      let stored: string | null = null;
      try {
        stored = readStoredSource(recordId, stage.id, localStorage);
      } catch {
        stored = null;
      }
      sources[stage.id] = stored ?? stage.source;
      saved[stage.id] = stored;
    }
    setStageSources(sources);
    setStageSaved(saved);
    setStageRemoteError(null);
  }, [recordId]);
  const sending = useRef(false);
  const alive = useRef(true);
  const request = useRef<AbortController | undefined>(undefined);
  const flight = useRef(0);
  const tabs = useRef<Partial<Record<PipelineBlock, HTMLButtonElement | null>>>({});
  const panel = useRef<HTMLElement | null>(null);

  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
      request.current?.abort();
    };
  }, []);

  const settingsId = params.get("settings_id") ?? "";
  const settingsList = useQuery({
    queryKey: ["pipeline-settings", account?.id],
    enabled: !!account,
    retry: false,
    queryFn: async ({ signal }) => readSettingList(await control<unknown>("/pipeline-settings", undefined, signal)),
  });
  const settingQuery = useQuery({
    queryKey: ["pipeline-setting", account?.id, settingsId],
    enabled: !!account && !!settingsId,
    retry: false,
    queryFn: async ({ signal }) => readSetting(await control<unknown>(`/pipeline-settings/${encodeURIComponent(settingsId)}`, undefined, signal)),
  });
  const saved = settingQuery.data && settingQuery.data.id === settingsId ? settingQuery.data : null;
  const listedSettings = settingsList.data ?? [];
  const historyIds = useMemo(() => {
    const ids: string[] = [];
    for (const setting of listedSettings) {
      for (const id of setting.job_ids) if (!ids.includes(id)) ids.push(id);
    }
    return ids.slice(0, 30);
  }, [listedSettings]);
  const history = useQuery<HistoryRow[]>({
    queryKey: ["pipeline-history", account?.id, historyIds],
    enabled: !!account && historyIds.length > 0,
    retry: false,
    refetchInterval: (query) =>
      query.state.data?.some((item) => item.job?.status === "queued" || item.job?.status === "running")
        ? document.hidden
          ? 10000
          : 2000
        : false,
    queryFn: async ({ signal }) =>
      Promise.all(
        historyIds.map(async (id) => {
          try {
            return { id, job: await control<Job>(`/analysis-jobs/${encodeURIComponent(id)}`, undefined, signal) };
          } catch (reason) {
            return { id, error: reason instanceof Error ? reason.message : "Задача не прочитана" };
          }
        }),
      ),
  });

  useEffect(() => {
    if (!catalog.data?.length) return;
    const current = new URLSearchParams(paramKey);
    const recordParam = current.get("record");
    const record = recordParam ? recordParam : null;
    const shape = detail.data && record === detail.data.id ? detail.data.shape : null;
    const resolution = resolvePipelineLink({
      record,
      bounds: (["r0", "r1", "c0", "c1"] as const).map((key) =>
        current.has(key) ? current.get(key) : null,
      ) as [string | null, string | null, string | null, string | null],
      shape,
      knownRecords: catalog.data.map((item) => item.id),
      jobId,
      job: job
        ? { record_id: job.region.record_id, source_sha256: job.source_sha256, region: job.region }
        : null,
      settingPending: !!settingsId && !settingQuery.data && !settingQuery.isError,
      setting:
        saved && (!record || record === saved.record_id)
          ? {
              record_id: saved.record_id,
              source_sha256: saved.source_sha256,
              window: saved.window,
              point: saved.point,
            }
          : null,
      firstRecord: catalog.data[0].id,
    });
    if (resolution.action !== "fill") return;
    const next = new URLSearchParams(current);
    if (!record) next.set("record", resolution.record);
    if (resolution.source_sha256 && !current.get("source_sha256"))
      next.set("source_sha256", resolution.source_sha256);
    if (resolution.window && !["r0", "r1", "c0", "c1"].some((key) => current.has(key))) {
      formatWindow(resolution.window).forEach((value, index) => {
        next.set(["r0", "r1", "c0", "c1"][index], value);
      });
    }
    if (resolution.point) {
      if (!current.get("row")) next.set("row", String(resolution.point.row));
      if (!current.get("column")) next.set("column", String(resolution.point.column));
    }
    if (next.toString() === current.toString()) return;
    setParams(next, { replace: true, preventScrollReset: true });
  }, [catalog.data, detail.data, job, jobId, paramKey, saved, setParams, settingQuery.data, settingQuery.isError, settingsId]);

  const hydratedSetting = useRef("");
  useEffect(() => {
    if (!saved || hydratedSetting.current === saved.id) return;
    hydratedSetting.current = saved.id;
    setTitle((current) => (current.trim() ? current : saved.title));
    setCompareA((current) => current || saved.run_ids[0] || "");
    setCompareB((current) => current || saved.run_ids[1] || "");
    setSession((current) => rememberRevision(current, saved.revision));
  }, [saved]);
  useEffect(() => {
    if (!saved) return;
    setSession((current) =>
      current.savedRevision === saved.revision ? current : rememberRevision(current, saved.revision),
    );
  }, [saved]);

  const source = detail.data;
  const shaMismatch = !!(
    source &&
    params.get("source_sha256") &&
    params.get("source_sha256") !== source.source_sha256
  );
  const committed = useMemo(() => {
    if (!source) return null;
    return parseHalfOpen(
      [
        params.get("r0") ?? "",
        params.get("r1") ?? "",
        params.get("c0") ?? "",
        params.get("c1") ?? "",
      ],
      source.shape,
    );
  }, [source, paramKey, params]);

  useEffect(() => {
    if (!source || shaMismatch) return;
    const current = new URLSearchParams(paramKey);
    if (current.get("record") !== source.id) return;
    const next = new URLSearchParams(current);
    next.set("source_sha256", source.source_sha256);
    const existing = parseHalfOpen(
      [
        current.get("r0") ?? "",
        current.get("r1") ?? "",
        current.get("c0") ?? "",
        current.get("c1") ?? "",
      ],
      source.shape,
    );
    const hasBounds = ["r0", "r1", "c0", "c1"].some((key) => current.has(key));
    if (hasBounds && !existing) return;
    const chosen = existing ?? limitedWindow(source.shape);
    if (!chosen) return;
    if (!existing) {
      next.set("r0", String(chosen.row_start));
      next.set("r1", String(chosen.row_end));
      next.set("c0", String(chosen.column_start));
      next.set("c1", String(chosen.column_end));
    }
    if (!current.get("block")) next.set("block", "source");
    if (!current.get("row")) next.set("row", String(chosen.row_start));
    if (!current.get("column")) next.set("column", String(chosen.column_start));
    if (next.toString() === current.toString()) return;
    setParams(next, { replace: true, preventScrollReset: true });
  }, [source, shaMismatch, paramKey, setParams]);

  useEffect(() => {
    if (!committed) return;
    const formatted = formatWindow(committed);
    if (draftDirty && formatted.join("|") !== draftKey) return;
    if (formatted.join("|") !== draftKey) setDraftText(formatted);
    if (draftDirty) setDraftDirty(false);
    setSession((current) => editDraft(current, committed));
  }, [committed, draftDirty, draftKey]);

  const row = committed ? placed(params.get("row"), committed.row_start, committed.row_end) : 0;
  const column = committed ? placed(params.get("column"), committed.column_start, committed.column_end) : 0;
  const sourceRow = committed
    ? placed(String(row), committed.row_start, committed.row_end)
    : row;
  const sourceColumn = committed
    ? placed(String(column), committed.column_start, committed.column_end)
    : column;
  const allowed = !!account && ["analyst", "admin"].includes(account.role);
  const active = !!job && ["queued", "running"].includes(job.status);
  const jobForRecord =
    !!job &&
    !!source &&
    job.region.record_id === source.id &&
    job.source_sha256 === source.source_sha256;

  const sourceSlice = useQuery<Slices>({
    queryKey: [
      "pipeline-source-slice",
      account?.id,
      source?.source_sha256,
      committed,
      sourceRow,
      sourceColumn,
    ],
    enabled: block === "source" && !!source && !!committed && !shaMismatch,
    placeholderData: (previous, query) => {
      const key = query?.queryKey;
      if (
        key?.[1] !== account?.id ||
        previous?.source_sha256 !== source?.source_sha256 ||
        JSON.stringify(key?.[3]) !== JSON.stringify(committed) ||
        key?.[4] !== sourceRow ||
        key?.[5] !== sourceColumn
      )
        return undefined;
      return previous;
    },
    retry: false,
    queryFn: async ({ signal }) => {
      const value = await
      api<Slices>(
        `/records/${source!.id}/slices?${new URLSearchParams({
          row: String(sourceRow),
          column: String(sourceColumn),
          row_start: String(committed!.row_start),
          row_end: String(committed!.row_end),
          source_sha256: source!.source_sha256,
        })}`,
        signal,
      );
      if (!samePoint(value, source!.source_sha256, sourceRow, sourceColumn))
        throw new Error("Исходный срез не соответствует выбранной точке или SHA");
      return value;
    },
  });
  const sourceScope = JSON.stringify([account?.id, source?.id, source?.source_sha256, committed]);
  const [lastSource, setLastSource] = useState<{ scope: string; data: Slices }>();
  useEffect(() => {
    if (sourceSlice.data && !sourceSlice.isPlaceholderData)
      setLastSource({ scope: sourceScope, data: sourceSlice.data });
  }, [sourceSlice.data, sourceSlice.isPlaceholderData, sourceScope]);
  const sourceChoice = sliceForPoint(
    sourceSlice.data,
    !shaMismatch && lastSource?.scope === sourceScope ? lastSource.data : undefined,
    source?.source_sha256 ?? "",
    sourceRow,
    sourceColumn,
  );
  const visibleSource = shaMismatch ? undefined : sourceChoice.plot;

  const run = useQuery({
    queryKey: ["pipeline-run", account?.id, job?.run_id],
    enabled: !!account && !!job?.run_id && job.status === "complete",
    retry: false,
    queryFn: ({ signal }) => control<Run>(`/result-runs/${job!.run_id}`, undefined, signal),
  });
  const runData = run.data && run.data.id === job?.run_id && job &&
    sameBody(submitBody(run.data.region.record_id, run.data.source_sha256, run.data.region),
      submitBody(job.region.record_id, job.source_sha256, job.region)) ? run.data : undefined;
  const resultPoint = job
    ? {
        row: placed(params.get("row"), job.region.row_start, job.region.row_end),
        column: placed(params.get("column"), job.region.column_start, job.region.column_end),
      }
    : null;
  const runSlice = useQuery<RunSlices>({
    queryKey: [
      "pipeline-run-slice",
      account?.id,
      job?.run_id,
      resultPoint?.row,
      resultPoint?.column,
      block,
    ],
    enabled: !!job?.run_id && !!resultPoint && job.status === "complete",
    placeholderData: (previous, query) => {
      const key = query?.queryKey;
      if (
        key?.[1] !== account?.id ||
        previous?.run_id !== job?.run_id ||
        previous?.source_sha256 !== job?.source_sha256 ||
        key?.[3] !== resultPoint?.row ||
        key?.[4] !== resultPoint?.column
      )
        return undefined;
      return previous;
    },
    retry: false,
    queryFn: async ({ signal }) => {
      const value = await
      control<RunSlices>(
        `/result-runs/${job!.run_id}/slices?${new URLSearchParams({
          row: String(resultPoint!.row),
          column: String(resultPoint!.column),
          row_start: String(job!.region.row_start),
          row_end: String(job!.region.row_end),
          column_start: String(job!.region.column_start),
          column_end: String(job!.region.column_end),
        })}`,
        undefined,
        signal,
      );
      if (value.run_id !== job!.run_id ||
        !samePoint(value, job!.source_sha256, resultPoint!.row, resultPoint!.column))
        throw new Error("Срез результата не соответствует выбранному запуску или точке");
      return value;
    },
  });
  const resultScope = JSON.stringify([account?.id, job?.run_id, job?.source_sha256]);
  const [lastResult, setLastResult] = useState<{ scope: string; data: RunSlices }>();
  useEffect(() => {
    if (runSlice.data && !runSlice.isPlaceholderData)
      setLastResult({ scope: resultScope, data: runSlice.data });
  }, [runSlice.data, runSlice.isPlaceholderData, resultScope]);
  const resultChoice = sliceForPoint(
    runSlice.data,
    lastResult?.scope === resultScope ? lastResult.data : undefined,
    job?.source_sha256 ?? "",
    resultPoint?.row ?? -1,
    resultPoint?.column ?? -1,
  );
  const shownSlice = resultChoice.plot;
  const sliceCurrent = !!resultPoint && samePoint(shownSlice, job?.source_sha256 ?? "", resultPoint.row, resultPoint.column);
  const zones = useQuery({
    queryKey: ["pipeline-zones", account?.id, job?.run_id],
    enabled: (block === "zones" || !!params.get("zone_id")) && !!job?.run_id && job.status === "complete",
    retry: false,
    queryFn: ({ signal }) =>
      control<ZoneBox[]>(`/result-runs/${job!.run_id}/zones`, undefined, signal),
  });

  const runs = useQuery({
    queryKey: ["pipeline-result-runs", account?.id, source?.id],
    enabled: !!account && !!source?.id && !shaMismatch,
    retry: false,
    queryFn: ({ signal }) =>
      control<Run[]>(`/result-runs?${new URLSearchParams({ record_id: source!.id })}`, undefined, signal),
  });
  const selectedA = runs.data?.find((item) => item.id === compareA) ?? null;
  const selectedB = runs.data?.find((item) => item.id === compareB) ?? null;
  const comparison =
    compareA && compareB && runs.data
      ? compareResults(selectedA ? compareRun(selectedA) : null, selectedB ? compareRun(selectedB) : null)
      : null;
  const intersection = comparison && "window" in comparison ? comparison.window : null;
  const comparePoint = intersection
    ? {
        row: intersection.row_start <= row && row < intersection.row_end ? row : intersection.row_start,
        column:
          intersection.column_start <= column && column < intersection.column_end
            ? column
            : intersection.column_start,
      }
    : null;
  const compareSliceA = useQuery({
    queryKey: ["pipeline-compare-slice", account?.id, selectedA?.id, intersection, comparePoint],
    enabled: !!selectedA && !!intersection && !!comparePoint,
    retry: false,
    queryFn: ({ signal }) =>
      loadCompareSlice(selectedA!, intersection!, comparePoint!, signal),
  });
  const compareSliceB = useQuery({
    queryKey: ["pipeline-compare-slice", account?.id, selectedB?.id, intersection, comparePoint],
    enabled: !!selectedB && !!intersection && !!comparePoint,
    retry: false,
    queryFn: ({ signal }) =>
      loadCompareSlice(selectedB!, intersection!, comparePoint!, signal),
  });
  const sourceCurrent = samePoint(visibleSource, source?.source_sha256 ?? "", sourceRow, sourceColumn);
  const currentBody = source && committed ? submitBody(source.id, source.source_sha256, committed) : null;
  const jobMatchesWindow = !!job && !!currentBody && sameBody(currentBody,
    submitBody(job.region.record_id, job.source_sha256, job.region));

  function chooseBlock(next: PipelineBlock) {
    panel.current?.scrollTo({ top: 0, behavior: "instant" });
    replaceParams(setParams, (current) => current.set("block", next));
  }

  function commitDraft() {
    if (!source) return;
    const parsed = parseHalfOpen(draftText, source.shape);
    if (!parsed) {
      setError(
        "Окно задаётся полуоткрытыми индексами внутри записи, не меньше двух столбцов.",
      );
      return;
    }
    setError("");
    setSession((current) => editDraft(current, parsed));
    replaceParams(setParams, (current) => {
      current.set("r0", String(parsed.row_start));
      current.set("r1", String(parsed.row_end));
      current.set("c0", String(parsed.column_start));
      current.set("c1", String(parsed.column_end));
      const nextRow = placed(current.get("row"), parsed.row_start, parsed.row_end);
      const nextColumn = placed(
        current.get("column"),
        parsed.column_start,
        parsed.column_end,
      );
      current.set("row", String(nextRow));
      current.set("column", String(nextColumn));
    });
  }

  const activeDraft = useMemo(() => {
    if (!source || !capability.data) return null;
    const parsed = parseHalfOpen(draftText, source.shape);
    if (!parsed) return null;
    const point = {
      row: placed(String(row), parsed.row_start, parsed.row_end),
      column: placed(String(column), parsed.column_start, parsed.column_end),
    };
    const runIds = [compareA, compareB]
      .filter((id, index, all) => {
        if (!id || all.indexOf(id) !== index) return false;
        if (!runs.data) return true;
        return runs.data.some(
          (item) => item.id === id && item.region.record_id === source.id && item.source_sha256 === source.source_sha256,
        );
      })
      .slice(0, 2);
    const draft: SettingsDraft = {
      title: title.trim(),
      record_id: source.id,
      source_sha256: source.source_sha256,
      window: parsed,
      point,
      method: capability.data.method,
      method_version: capability.data.method_version,
      run_ids: runIds,
    };
    return draft;
  }, [capability.data, column, compareA, compareB, draftText, row, runs.data, source, title]);
  const draftMatches = !!(saved && activeDraft && sameSettingsDraft(settingDraft(saved), activeDraft));
  useEffect(() => {
    setPreflight(null);
  }, [compareA, compareB, draftKey, recordId, title]);

  function noteAcceptedRevision(id: string, revision: number, acceptedJobId: string) {
    if (!id) return;
    const apply = (current: PipelineSetting | null | undefined) =>
      current && current.id === id
        ? {
            ...current,
            revision,
            job_ids: current.job_ids.includes(acceptedJobId) ? current.job_ids : [...current.job_ids, acceptedJobId],
          }
        : current;
    queryClient.setQueryData<PipelineSetting | null>(["pipeline-setting", account?.id, id], (current) =>
      current && current.id === id ? apply(current) ?? current : current,
    );
    queryClient.setQueryData<PipelineSetting[]>(["pipeline-settings", account?.id], (current) =>
      Array.isArray(current) ? current.map((item) => apply(item) ?? item) : current,
    );
  }

  function storeSetting(setting: PipelineSetting) {
    hydratedSetting.current = setting.id;
    queryClient.setQueryData(["pipeline-setting", account?.id, setting.id], setting);
    queryClient.setQueryData<PipelineSetting[]>(["pipeline-settings", account?.id], (current) => [
      setting,
      ...(current ?? []).filter((item) => item.id !== setting.id),
    ]);
    setTitle(setting.title);
    setCompareA(setting.run_ids[0] ?? "");
    setCompareB(setting.run_ids[1] ?? "");
    setDraftDirty(false);
    setSession((current) => rememberRevision(current, setting.revision));
  }

  function writeSettingUrl(setting: PipelineSetting) {
    const latest = navigation.current;
    const next = new URLSearchParams(latest.params);
    next.set("settings_id", setting.id);
    next.set("record", setting.record_id);
    next.set("source_sha256", setting.source_sha256);
    formatWindow(setting.window).forEach((value, index) => next.set(["r0", "r1", "c0", "c1"][index], value));
    next.set("row", String(setting.point.row));
    next.set("column", String(setting.point.column));
    next.delete("zone_id");
    latest.setParams(next, { replace: true, preventScrollReset: true });
  }

  async function saveSetting() {
    if (!activeDraft) {
      setSettingsError("Окно или закреплённый метод ещё не готовы к сохранению.");
      return;
    }
    if (!activeDraft.title) {
      setSettingsError("Укажите название настройки.");
      return;
    }
    const patch = saved ? changedSettingsPatch(settingDraft(saved), saved.revision, activeDraft) : null;
    if (saved && !patch) return;
    setBusy(true);
    setSettingsError("");
    try {
      const created = {
        title: activeDraft.title,
        record_id: activeDraft.record_id,
        source_sha256: activeDraft.source_sha256,
        window: activeDraft.window,
        point: activeDraft.point,
        method: activeDraft.method,
        method_version: activeDraft.method_version,
        parameters: {},
        run_ids: [...activeDraft.run_ids],
      };
      let stored: PipelineSetting | null;
      if (!saved) {
        const serialized = JSON.stringify(created);
        if (!saveAttempt.current || saveAttempt.current.body !== serialized)
          saveAttempt.current = { key: crypto.randomUUID(), body: serialized };
        stored = readSetting(
          await control<unknown>("/pipeline-settings", created, undefined, saveAttempt.current.key),
        );
      } else {
        stored = readSetting(
          await control<unknown>(
            `/pipeline-settings/${encodeURIComponent(saved.id)}`,
            patch,
            undefined,
            undefined,
            "PATCH",
          ),
        );
      }
      if (!alive.current) return;
      if (!stored) {
        setSettingsError("Сервер не вернул сохранённую настройку.");
        return;
      }
      saveAttempt.current = null;
      const decision = applySavedSetting({
        urlRecord: navigation.current.params.get("record"),
        submittedRecord: stored.record_id,
      });
      if (!decision.write) {
        queryClient.setQueryData(["pipeline-setting", account?.id, stored.id], stored);
        queryClient.setQueryData<PipelineSetting[]>(["pipeline-settings", account?.id], (current) => [
          stored,
          ...(current ?? []).filter((item) => item.id !== stored.id),
        ]);
        return;
      }
      storeSetting(stored);
      writeSettingUrl(stored);
    } catch (reason) {
      if (!alive.current) return;
      const conflict = conflictText(
        reason,
        "Сохранённое окно уже изменилось. Запись на сервере не перезаписана.",
      );
      if (conflict) {
        setSettingsError(conflict);
        return;
      }
      setSettingsError(reason instanceof Error ? reason.message : "Не удалось сохранить настройки");
    } finally {
      if (alive.current) setBusy(false);
    }
  }

  async function checkDraft() {
    if (!source) return;
    const parsed = parseHalfOpen(draftText, source.shape);
    if (!parsed) {
      setSettingsError("Окно задаётся полуоткрытыми индексами внутри записи, не меньше двух столбцов.");
      return;
    }
    setBusy(true);
    setSettingsError("");
    setPreflight(null);
    try {
      const report = await control<PreflightReport>(
        "/analysis-preflight",
        submitBody(
          source.id,
          source.source_sha256,
          parsed,
          draftMatches && saved ? { id: saved.id, revision: saved.revision } : undefined,
        ),
      );
      if (!alive.current) return;
      setPreflight(report);
    } catch (reason) {
      if (!alive.current) return;
      setSettingsError(reason instanceof Error ? reason.message : "Проверка не выполнена");
    } finally {
      if (alive.current) setBusy(false);
    }
  }

  async function reloadSetting() {
    if (!settingsId) return;
    setBusy(true);
    setSettingsError("");
    try {
      const stored = readSetting(
        await control<unknown>(`/pipeline-settings/${encodeURIComponent(settingsId)}`),
      );
      if (!alive.current) return;
      if (!stored) {
        setSettingsError("Сервер не вернул сохранённое окно.");
        return;
      }
      const decision = applySavedSetting({
        urlRecord: navigation.current.params.get("record"),
        submittedRecord: stored.record_id,
      });
      if (!decision.write) return;
      storeSetting(stored);
      writeSettingUrl(stored);
    } catch (reason) {
      if (!alive.current) return;
      setSettingsError(reason instanceof Error ? reason.message : "Сохранённое окно не прочитано");
    } finally {
      if (alive.current) setBusy(false);
    }
  }

  async function send(attempt: Attempt) {
    if (sending.current) return;
    sending.current = true;
    setBusy(true);
    setError("");
    const controller = new AbortController();
    request.current = controller;
    const token = ++flight.current;
    const previousJobId = navigation.current.params.get("job_id");
    try {
      const result = await control<Job>(
        "/analysis-jobs",
        attempt.body,
        controller.signal,
        attempt.key,
      );
      if (!alive.current || token !== flight.current) return;
      const latest = navigation.current;
      const route = applyLateResult({
        urlRecord: latest.params.get("record"),
        submittedRecord: attempt.body.region.record_id,
        urlJobId: latest.params.get("job_id"),
        jobIdAtSend: previousJobId,
        resultId: result.id,
      });
      const acceptedSettings = result.settings_id ?? attempt.body.settings_id ?? "";
      if (typeof result.settings_revision_after === "number")
        noteAcceptedRevision(acceptedSettings, result.settings_revision_after, result.id);
      setSession((current) => {
        const base =
          typeof result.settings_revision_after === "number"
            ? rememberRevision(current, result.settings_revision_after)
            : current;
        if (route.jobId !== result.id) return base;
        return attachJob(base, attempt.body, {
          id: result.id,
          run_id: result.run_id ?? null,
          source_sha256: result.source_sha256,
          region: result.region,
        });
      });
      if (route.jobId === result.id && latest.params.get("job_id") !== result.id) {
        const next = new URLSearchParams(latest.params);
        next.set("job_id", result.id);
        next.delete("zone_id");
        latest.setParams(next, { replace: true, preventScrollReset: true });
      }
    } catch (reason) {
      if (!alive.current || token !== flight.current) return;
      if (controller.signal.aborted) {
        setSession((current) => afterAbort(current).session);
        return;
      }
      const conflict = conflictText(
        reason,
        "Сохранённое окно уже изменилось. Запись на сервере не перезаписана.",
      );
      if (conflict) {
        setSession((current) => ({ ...current, attempt: null }));
        setError(conflict);
        return;
      }
      if (reason instanceof ApiError && reason.status >= 400 && reason.status < 500)
        setSession((current) => ({ ...current, attempt: null }));
      setError(reason instanceof Error ? reason.message : "Не удалось отправить расчёт");
    } finally {
      sending.current = false;
      if (alive.current) setBusy(false);
    }
  }

  function launchCalculation() {
    if (!source) return;
    const parsed = parseHalfOpen(draftText, source.shape);
    if (!parsed) {
      setError("Окно задаётся полуоткрытыми индексами внутри записи, не меньше двух столбцов.");
      return;
    }
    setError("");
    const linked =
      saved &&
      activeDraft &&
      sameSettingsDraft(settingDraft(saved), {
        ...activeDraft,
        window: parsed,
        point: {
          row: placed(String(row), parsed.row_start, parsed.row_end),
          column: placed(String(column), parsed.column_start, parsed.column_end),
        },
      })
        ? { id: saved.id, revision: saved.revision }
        : undefined;
    const body = submitBody(source.id, source.source_sha256, parsed, linked);
    const decision = explicitSubmit(session, body, () => crypto.randomUUID());
    setSession(decision.session);
    replaceParams(setParams, (current) => {
      current.set("r0", String(parsed.row_start));
      current.set("r1", String(parsed.row_end));
      current.set("c0", String(parsed.column_start));
      current.set("c1", String(parsed.column_end));
    });
    void send(decision.post);
  }

  function saveActiveStage() {
    if (!recordId) return;
    const sourceText = stageSources[stageActive];
    try {
      writeStoredSource(recordId, stageActive, sourceText, localStorage);
    } catch {
      setError("Вкладка не сохранилась в этом браузере.");
      return;
    }
    setStageSaved((current) => ({ ...current, [stageActive]: sourceText }));
  }

  async function checkPipeline() {
    if (stageChecking) return;
    setStageChecking(true);
    setStageRemoteError(null);
    try {
      const found = await interpretStages(stageSources);
      if (!alive.current) return;
      if (found) {
        setStageRemoteError(found);
        setStageActive(found.stageId);
        return;
      }
      launchCalculation();
    } catch (reason) {
      if (!alive.current) return;
      setError(reason instanceof Error ? reason.message : "Проверка кода не выполнена");
    } finally {
      if (alive.current) setStageChecking(false);
    }
  }

  const sourceWindow = committed;
  const timeBounds = source && sourceWindow ? sourceTimeBounds(source.time_s, sourceWindow.row_start, sourceWindow.row_end) : null;
  const sourcePath =
    source && sourceWindow && !shaMismatch && timeBounds
      ? `/records/${source.id}/tiles?${new URLSearchParams({
          x0: String(source.x_m[sourceWindow.column_start]),
          x1: String(source.x_m[sourceWindow.column_end - 1]),
          t0: String(timeBounds[0]),
          t1: String(timeBounds[1]),
          width: matchMedia("(max-width: 672px)").matches ? "96" : "192",
          height: "128",
        })}`
      : "";
  const rawProfile = shownSlice?.layers.raw?.profile;
  const processedProfile = shownSlice?.layers.processed_temperature_c?.profile;
  const changes =
    shownSlice &&
    rawProfile &&
    processedProfile &&
    rawProfile.length === processedProfile.length &&
    rawProfile.length === shownSlice.x_m.length
      ? profileChanges({
          x_m: shownSlice.x_m,
          temperature_c: rawProfile,
          processed_temperature_c: processedProfile,
        })
      : null;
  const temperatureScale = shownSlice
    ? finiteExtent([
        ...(shownSlice.layers.raw?.profile ?? []),
        ...(shownSlice.layers.processed_temperature_c?.profile ?? []),
      ])
    : undefined;
  const analysisTo =
    job?.run_id && resultPoint
      ? `/analysis?${analysisParams(
          { id: job.run_id, source_sha256: job.source_sha256 },
          {
            region: job.region,
            row: resultPoint.row,
            column: resultPoint.column,
            layer: "processed_temperature_c",
            zone: params.get("zone_id") ?? "",
          },
        )}`
      : "";
  const relative = job && shownSlice
    ? {
        row: shownSlice.row - job.region.row_start,
        column: shownSlice.column - job.region.column_start,
      }
    : null;

  return (
    <main className="page pipeline">
      <h1>Пайплайн</h1>
      <p>
        Это один закреплённый расчёт выбранного окна исходной записи. Код вкладок проверяется интерпретатором и не заменяет метод сервера.
      </p>
      <p>Дальше: задайте отрезок и нажмите «Проверить пайплайн».</p>
      {catalog.isLoading && <p role="status">Загрузка записей…</p>}
      {catalog.isError && (
        <InlineNotification
          kind="error"
          title="Архив не прочитан"
          subtitle={String(catalog.error)}
          hideCloseButton
          lowContrast
        />
      )}
      {catalog.isError && <Button kind="tertiary" onClick={() => void catalog.refetch()}>Повторить загрузку архива</Button>}
      {catalog.data?.length === 0 && <p>В проекте нет доступных подготовленных записей.</p>}
      {catalog.data && (
        <Select
          id="pipeline-record"
          labelText={`Запись архива · доступно ${catalog.data.length}`}
          value={recordId}
          onChange={(event) => {
            const id = event.target.value;
            setDraftDirty(false);
            replaceParams(setParams, (current) => {
              current.set("record", id);
              current.delete("source_sha256");
              current.delete("r0");
              current.delete("r1");
              current.delete("c0");
              current.delete("c1");
              current.delete("row");
              current.delete("column");
              current.delete("zone_id");
              current.set("block", "source");
            });
          }}
        >
          {recordId && !catalog.data.some((item) => item.id === recordId) && <SelectItem value={recordId} text={`Недоступная запись: ${recordId}`} disabled />}
          {catalog.data.map((item) => (
            <SelectItem
              key={item.id}
              value={item.id}
              text={`${item.id} · ${item.source} · ${item.shape[0].toLocaleString("ru")} × ${item.shape[1].toLocaleString("ru")}`}
            />
          ))}
        </Select>
      )}
      {shaMismatch && (
        <InlineNotification
          kind="warning"
          title="SHA ссылки не совпадает с записью"
          subtitle="Окно и графики этой ссылки не сопоставляются с заменённым источником."
          hideCloseButton
          lowContrast
        />
      )}
      {source && !committed && ["r0", "r1", "c0", "c1"].some((key) => params.has(key)) && (
        <p role="alert">Границы ссылки недопустимы. Введите корректное окно; автоматической замены нет.</p>
      )}
      <PipelineWorkbench
        stages={STAGES.map((stage) => ({
          id: stage.id,
          label: stage.label,
          chip: stage.chip,
          source: stageSources[stage.id] ?? "",
          status: stageStatus({
            id: stage.id,
            source: stageSources[stage.id] ?? "",
            savedSource: stageSaved[stage.id] ?? null,
            error: stageError,
          }),
        }))}
        active={stageActive}
        error={stageError}
        recordLabel={
          source && committed
            ? `запись ${source.id}, строки [${committed.row_start}, ${committed.row_end}), столбцы [${committed.column_start}, ${committed.column_end})`
            : "отрезок ещё не задан"
        }
        canSave={!!recordId}
        canCheck={
          !busy &&
          !stageChecking &&
          !active &&
          allowed &&
          !!capability.data?.ready &&
          !capability.isError &&
          !!source &&
          !shaMismatch &&
          !session.attempt
        }
        canSegment={!!source && !shaMismatch}
        checking={stageChecking || busy}
        onActive={setStageActive}
        onSource={(value) => {
          setStageRemoteError(null);
          setStageSources((current) => ({ ...current, [stageActive]: value }));
        }}
        onSave={saveActiveStage}
        onCheck={() => void checkPipeline()}
        onSegment={commitDraft}
        segment={
          <>
            {(
              [
                "Начальная строка",
                "Конечная строка",
                "Начальный столбец",
                "Конечный столбец",
              ] as const
            ).map((label, index) => (
              <TextInput
                key={label}
                id={`pipeline-bound-${index}`}
                labelText={label}
                inputMode="numeric"
                value={draftText[index]}
                onChange={(event) => {
                  setDraftDirty(true);
                  setDraftText(editBoundField(draftText, index as 0 | 1 | 2 | 3, event.target.value));
                }}
                onKeyDown={(event) => {
                  if (event.key === "Enter") {
                    event.preventDefault();
                    commitDraft();
                  }
                }}
              />
            ))}
            <p className="pipeline-workbench-note">
              {draftDirty
                ? "Границы изменены. «Задать отрезок» обновит просмотр. «Проверить пайплайн» отправит введённые границы."
                : "Окно просмотра совпадает с полями. Конечный индекс в окно не входит."}
            </p>
          </>
        }
        original={
          sourcePath && source ? (
            <Heatmap
              path={sourcePath}
              versionKey={source.source_sha256}
              expectedSourceSha={source.source_sha256}
              cursor={
                source && sourceWindow && visibleSource
                  ? [
                      sourceFieldAxes(source.x_m, source.time_s, sourceWindow).x[
                        (visibleSource.column ?? sourceColumn) - sourceWindow.column_start
                      ],
                      sourceFieldAxes(source.x_m, source.time_s, sourceWindow).t[
                        (visibleSource.row ?? sourceRow) - sourceWindow.row_start
                      ],
                    ]
                  : undefined
              }
              onSelect={(x, t) => {
                if (!source || !sourceWindow) return;
                const axes = sourceFieldAxes(source.x_m, source.time_s, sourceWindow);
                replaceParams(setParams, (current) => {
                  current.set("row", String(sourceWindow.row_start + nearest(axes.t, t)));
                  current.set("column", String(sourceWindow.column_start + nearest(axes.x, x)));
                });
              }}
            />
          ) : (
            <p className="pipeline-workbench-note">Исходное поле появится, когда отрезок задан.</p>
          )
        }
        checked={
          !stageError &&
          job?.status === "complete" &&
          jobForRecord &&
          jobMatchesWindow &&
          runData?.layers.includes("processed_temperature_c") &&
          shownSlice &&
          sliceCurrent ? (
            <Heatmap
              path={tilePath(runData.id, "processed_temperature_c", shownSlice)}
              versionKey={runData.id}
              expectedSourceSha={runData.source_sha256}
              onSelect={(x, t) => {
                if (!job) return;
                replaceParams(setParams, (current) => {
                  current.set("row", String(job.region.row_start + nearest(shownSlice.native_time_s, t)));
                  current.set("column", String(job.region.column_start + nearest(shownSlice.x_m, x)));
                });
              }}
            />
          ) : (
            <p className="pipeline-workbench-note">
              {stageError
                ? "Поле после проверки не заменяется исходником, пока интерпретатор остановился."
                : job && ["queued", "running"].includes(job.status)
                  ? "Расчёт ещё идёт. Это поле не заполняется исходником."
                  : job?.status === "complete" && jobForRecord && jobMatchesWindow
                    ? "Загрузка поля после проверки…"
                    : "Поле появится после успешной проверки. Это не исходник."}
            </p>
          )
        }
      />
      <section className="pipeline-passport" aria-label="Паспорт записи">
        {!source && !detail.isError && <p role="status">Загрузка паспорта…</p>}
        {detail.isError && (
          <InlineNotification
            kind="error"
            title="Паспорт не загружен"
            subtitle={String(detail.error)}
            hideCloseButton
            lowContrast
          />
        )}
        {detail.isError && <Button kind="tertiary" onClick={() => void detail.refetch()}>Повторить загрузку паспорта</Button>}
        {source && (
          <dl>
            <dt>Запись</dt>
            <dd>{source.id}</dd>
            <dt>SHA источника</dt>
            <dd className="mono">{source.source_sha256}</dd>
            <dt>Координаты</dt>
            <dd>
              {textNumber(source.x_bounds[0])}–{textNumber(source.x_bounds[1])} м · индексы
              столбцов [0, {source.shape[1]})
            </dd>
            <dt>Время</dt>
            <dd>
              {source.first_timestamp} → {source.last_timestamp}. На графиках — секунды от
              начала записи.{" "}
              {source.time_kind === "local_unzoned"
                ? "Часовой пояс неизвестен."
                : "Смещение взято из источника."}
            </dd>
            <dt>Единицы</dt>
            <dd>Температура °C, расстояние м, время с.</dd>
            <dt>Качество</dt>
            <dd>
              {(source.valid_fraction * 100).toFixed(1)}% валидных ячеек · {source.segments}{" "}
              сегм. Пропуск не является нулём температуры.
            </dd>
          </dl>
        )}
      </section>
      {committed && source && (
        <div className="analysis-controls">
          <CoordinateInput
            id="pipeline-row"
            label="Строка профиля"
            integer
            min={committed.row_start}
            max={committed.row_end - 1}
            value={row}
            onCommit={(value) =>
              replaceParams(setParams, (current) => current.set("row", String(value)))
            }
            onPrevious={() =>
              replaceParams(setParams, (current) =>
                current.set("row", String(Math.max(committed.row_start, row - 1))),
              )
            }
            onNext={() =>
              replaceParams(setParams, (current) =>
                current.set("row", String(Math.min(committed.row_end - 1, row + 1))),
              )
            }
          />
          <CoordinateInput
            id="pipeline-column"
            label="Столбец точки"
            previousLabel="Предыдущая точка"
            nextLabel="Следующая точка"
            integer
            min={committed.column_start}
            max={committed.column_end - 1}
            value={column}
            onCommit={(value) =>
              replaceParams(setParams, (current) => current.set("column", String(value)))
            }
            onPrevious={() =>
              replaceParams(setParams, (current) =>
                current.set("column", String(Math.max(committed.column_start, column - 1))),
              )
            }
            onNext={() =>
              replaceParams(setParams, (current) =>
                current.set(
                  "column",
                  String(Math.min(committed.column_end - 1, column + 1)),
                ),
              )
            }
          />
        </div>
      )}
      <section className="pipeline-status" aria-label="Задача расчёта">
        <p>Метод один, его назначает сервер. Выбрать другой нельзя.</p>
        {capability.isLoading && <p role="status">Проверка доступности расчёта…</p>}
        {capability.isError && <p role="alert">Не удалось проверить доступность расчёта: {String(capability.error)}</p>}
        {capability.data && !capability.data.ready && (
          <p role="status">
            Расчётчик сейчас недоступен. Исходник и уже сохранённые результаты остаются доступны.
          </p>
        )}
        {!allowed && <p>Запуск доступен аналитику и администратору этого проекта.</p>}
        {error && (
          <InlineNotification
            kind="error"
            title="Действие не завершено"
            subtitle={error}
            hideCloseButton
            lowContrast
          />
        )}
        {jobQuery.isError && (
          <InlineNotification
            kind="error"
            title="Задача не прочитана"
            subtitle={String(jobQuery.error)}
            hideCloseButton
            lowContrast
          />
        )}
        <div className="pipeline-actions">
          {session.attempt ? (
            <Button
              kind="primary"
              size="lg"
              disabled={busy || !allowed}
              onClick={() => {
                const pending = session.attempt;
                if (!pending) return;
                const decision = explicitSubmit(
                  session,
                  pending.body,
                  () => crypto.randomUUID(),
                );
                setSession(decision.session);
                void send(decision.post);
              }}
            >
              Повторить отправку
            </Button>
          ) : null}
          <Button kind="tertiary" size="lg" disabled={!allowed || busy || !source || shaMismatch} onClick={() => void checkDraft()}>
            Проверить границы
          </Button>
          {job && active && allowed && (
            <Button
              kind="danger--tertiary"
              size="lg"
              disabled={busy || job.cancel_requested}
              onClick={() => {
                setBusy(true);
                setError("");
                void control(`/jobs/${job.id}/cancel`, {})
                  .then(() => jobQuery.refetch())
                  .catch((reason) =>
                    setError(reason instanceof Error ? reason.message : "Отмена не подтверждена"),
                  )
                  .finally(() => setBusy(false));
              }}
            >
              Отменить расчёт
            </Button>
          )}
          {analysisTo && <Link to={analysisTo}>Открыть этот результат в ML-графиках</Link>}
          {capability.isError && <Button kind="tertiary" onClick={() => void capability.refetch()}>Повторить проверку доступности</Button>}
          {jobQuery.isError && <Button kind="tertiary" onClick={() => void jobQuery.refetch()}>Повторить чтение задачи</Button>}
        </div>
        {preflight && (
          <div role="status">
            <p>
              Сервер {preflight.accepted ? "принимает это окно" : "не принимает это окно"}. Расчётчик {preflight.ready ? "готов" : "сейчас недоступен"}. Задача не создана.
            </p>
            {preflight.warning && <p>{preflight.warning}</p>}
          </div>
        )}
        {saved && !draftMatches && (
          <p className="pipeline-unsaved">
            Окно на экране отличается от сохранённого. «Проверить пайплайн» отправит то, что сейчас на экране, и не перепишет уже отправленную задачу.
          </p>
        )}
        {session.attempt && (
          <p>
            Повтор отправит то же окно записи {session.attempt.body.region.record_id}: строки [{session.attempt.body.region.row_start}, {session.attempt.body.region.row_end}), столбцы [{session.attempt.body.region.column_start}, {session.attempt.body.region.column_end}). Обрыв ответа не означает, что задача не принята, и не отменяет её на сервере.
          </p>
        )}
        {job && (
          <div role="status">
            <p>
              Статус сервера: {stageText(job.status)}{job.stage !== job.status ? `. Стадия: ${stageText(job.stage)}` : ""}.
              {job.cancel_requested ? " Отмена запрошена." : ""}
            </p>
            <p>
              {resultName({
                recordId: job.region.record_id,
                window: job.region,
                calculatedAt: job.created_at,
              })}
            </p>
            {job.error && <p>{job.error}</p>}
            {(!jobForRecord || !jobMatchesWindow) && source && (
              <p role="status">Показана задача другого окна или источника. Сохранённое окно её не меняет; результат относится только к указанному окну задачи.</p>
            )}
          </div>
        )}
      </section>
      <h2 className="pipeline-result-title">Разбор результата</h2>
      <p className="pipeline-result-note">Графики закреплённого расчёта. Это не код вкладок выше.</p>
      <div
        className="pipeline-chain"
        role="tablist"
        aria-label="Разбор результата"
        onKeyDown={(event) => {
          if (event.ctrlKey || event.metaKey || event.altKey) return;
          if (!["ArrowRight", "ArrowLeft", "Home", "End"].includes(event.key)) return;
          const current = PIPELINE_BLOCKS.findIndex((item) => item.id === block);
          const next =
            event.key === "Home"
              ? 0
              : event.key === "End"
                ? PIPELINE_BLOCKS.length - 1
                : event.key === "ArrowRight"
                  ? (current + 1) % PIPELINE_BLOCKS.length
                  : (current + PIPELINE_BLOCKS.length - 1) % PIPELINE_BLOCKS.length;
          event.preventDefault();
          chooseBlock(PIPELINE_BLOCKS[next].id);
          tabs.current[PIPELINE_BLOCKS[next].id]?.focus({ preventScroll: true });
        }}
      >
        {PIPELINE_BLOCKS.map((item) => (
          <Button
            key={item.id}
            ref={(node) => {
              tabs.current[item.id] = node;
            }}
            kind={item.id === block ? "secondary" : "ghost"}
            size="md"
            role="tab"
            aria-selected={item.id === block}
            aria-controls="pipeline-panel"
            tabIndex={item.id === block ? 0 : -1}
            id={`pipeline-tab-${item.id}`}
            onClick={() => chooseBlock(item.id)}
          >
            {item.label}
          </Button>
        ))}
      </div>
      {runData && (
        <details className="pipeline-provenance">
          <summary>Паспорт выбранного результата</summary>
          <dl>
            <dt>Имя</dt>
            <dd>{resultName({ recordId: runData.region.record_id, window: runData.region, calculatedAt: runData.calculated_at })}</dd>
            <dt>Источник</dt>
            <dd>запись {runData.region.record_id} · SHA {runData.source_sha256}</dd>
            <dt>Метод</dt>
            <dd>{runData.method}. Его назначает сервер.</dd>
            <dt>Обучение</dt>
            <dd>{runData.training_origin === "untrained" ? "Без обучения. Это не обученная модель." : runData.training_origin}</dd>
            <dt>Статус исследования</dt>
            <dd>{runData.research_status}</dd>
            <dt>Ограничения</dt>
            <dd>{runData.limitations}</dd>
            <dt>Параметры</dt>
            <dd>{Object.keys(runData.parameters).length ? <pre>{JSON.stringify(runData.parameters, null, 2)}</pre> : "Дополнительных параметров нет."}</dd>
          </dl>
        </details>
      )}
      {run.data && !runData && <p role="alert">Паспорт результата не соответствует задаче. Данные не подставлены.</p>}
      <section
        ref={panel}
        className="pipeline-panel"
        id="pipeline-panel"
        role="tabpanel"
        aria-labelledby={`pipeline-tab-${block}`}
      >
        {(block === "source" ? sourceSlice.isError : run.isError || runSlice.isError || zones.isError) && (
          <Button kind="tertiary" onClick={() => {
            if (block === "source") void sourceSlice.refetch();
            else {
              if (run.isError) void run.refetch();
              if (block === "zones") void zones.refetch();
              else void runSlice.refetch();
            }
          }}>Повторить чтение данных</Button>
        )}
        {block === "source" && (
          <SourceBlock
            path={sourcePath}
            sha={source?.source_sha256 ?? ""}
            window={sourceWindow}
            axes={
              source && sourceWindow
                ? sourceFieldAxes(source.x_m, source.time_s, sourceWindow)
                : null
            }
            row={sourceRow}
            column={sourceColumn}
            slice={visibleSource}
            retained={sourceChoice.retained}
            loading={sourceSlice.isFetching}
            current={sourceCurrent}
            error={sourceSlice.error}
            onPoint={(nextRow, nextColumn) =>
              replaceParams(setParams, (current) => {
                current.set("row", String(nextRow));
                current.set("column", String(nextColumn));
              })
            }
          />
        )}
        {block === "clean" && (
          <CleanBlock
            job={job}
            run={runData}
            slice={shownSlice}
            retained={resultChoice.retained}
            loading={run.isFetching || runSlice.isFetching}
            current={sliceCurrent}
            error={run.error ?? runSlice.error}
            view={cleanView}
            onView={setCleanView}
            changes={changes}
            temperatureScale={temperatureScale}
            index={relative}
            zones={zones.data}
            selectedZone={runData?.zones.find((zone) => zone.id === params.get("zone_id"))}
            onPoint={(nextRow, nextColumn) =>
              replaceParams(setParams, (current) => {
                current.set("row", String(nextRow));
                current.set("column", String(nextColumn));
              })
            }
          />
        )}
        {block === "score" && (
          <ScoreBlock
            job={job}
            run={runData}
            slice={shownSlice}
            retained={resultChoice.retained}
            loading={run.isFetching || runSlice.isFetching}
            current={sliceCurrent}
            error={run.error ?? runSlice.error}
            view={scoreView}
            onView={setScoreView}
            index={relative}
            onPoint={(nextRow, nextColumn) =>
              replaceParams(setParams, (current) => {
                current.set("row", String(nextRow));
                current.set("column", String(nextColumn));
              })
            }
          />
        )}
        {block === "zones" && (
          <ZonesBlock
            job={job}
            run={runData}
            zones={zones.data}
            loading={zones.isLoading || run.isLoading}
            error={zones.error ?? run.error}
            selected={params.get("zone_id") ?? ""}
            onSelect={(zone) =>
              replaceParams(setParams, (current) => {
                if (!job) return;
                current.set("record", job.region.record_id);
                current.set("source_sha256", job.source_sha256);
                ["r0", "r1", "c0", "c1"].forEach((key, index) => current.set(key, formatWindow(job.region)[index]));
                current.set("zone_id", zone.id);
                current.set("row", String(zone.row_start));
                current.set("column", String(zone.column_start));
                current.set("block", runData?.layers.includes("processed_temperature_c") ? "clean" : "score");
              })
            }
          />
        )}
      </section>
      <section className="pipeline-status" aria-label="Сохраняемые настройки">
        <h2>Сохранить это окно</h2>
        <p>
          Название нужно, чтобы вернуться к тому же окну записи. Автосохранения нет. Правка после запуска не меняет уже отправленный расчёт.
        </p>
        <TextInput
          id="pipeline-setting-title"
          labelText="Название окна"
          value={title}
          onChange={(event) => setTitle(event.target.value)}
        />
        <p className={saved && !draftMatches ? "pipeline-unsaved" : undefined} role="status">
          {savePrompt({
            savedTitle: saved ? saved.title : null,
            draftMatches,
            authorName: saved?.author_name,
            jobOnScreen: !!job,
            ownerTitles:
              saved || !job
                ? []
                : listedSettings
                    .filter((setting) => setting.job_ids.includes(job.id))
                    .map((setting) => setting.title),
            loadingSelection: !!settingsId && !saved && settingQuery.isLoading && !settingQuery.isError,
          })}
        </p>
        <div className="pipeline-actions">
          <Button kind="secondary" size="lg" disabled={!allowed || busy || !source || shaMismatch || !capability.data} onClick={() => void saveSetting()}>
            Сохранить окно
          </Button>
          {saved && !draftMatches && (
            <Button kind="ghost" size="lg" disabled={busy || !settingsId} onClick={() => void reloadSetting()}>
              Вернуть сохранённое окно
            </Button>
          )}
        </div>
        {settingsList.data && (
          <Select
            id="pipeline-setting"
            labelText="Сохранённое окно"
            value={settingsId}
            onChange={(event) => {
              const id = event.target.value;
              const chosen = listedSettings.find((item) => item.id === id);
              if (!chosen) {
                replaceParams(setParams, (current) => {
                  if (id) current.set("settings_id", id);
                  else current.delete("settings_id");
                });
                return;
              }
              storeSetting(chosen);
              writeSettingUrl(chosen);
            }}
          >
            <SelectItem value="" text="Не выбрано" />
            {settingsId && !listedSettings.some((item) => item.id === settingsId) && (
              <SelectItem value={settingsId} text={saved ? settingLabel(saved) : settingsId} />
            )}
            {listedSettings.map((item) => (
              <SelectItem key={item.id} value={item.id} text={settingLabel(item)} />
            ))}
          </Select>
        )}
        {settingsList.isError && (
          <InlineNotification kind="error" title="Сохранённые окна не прочитаны" subtitle={String(settingsList.error)} hideCloseButton lowContrast />
        )}
        {settingQuery.isError && (
          <InlineNotification kind="error" title="Сохранённое окно не прочитано" subtitle={String(settingQuery.error)} hideCloseButton lowContrast />
        )}
        {settingsError && (
          <InlineNotification kind="error" title="Окно не записано" subtitle={settingsError} hideCloseButton lowContrast />
        )}
        <h3>Уже посчитанные задачи</h3>
        {history.isLoading && <p role="status">Загрузка истории…</p>}
        {history.isError && <p role="alert">История задач не прочитана: {String(history.error)}</p>}
        {!historyIds.length && <p>У сохранённых окон пока нет расчётов.</p>}
        <ul className="pipeline-history">
          {history.data?.map((item) => {
            const owner = listedSettings.find((setting) => setting.job_ids.includes(item.id));
            const failed = "error" in item ? item.error : "";
            return (
              <li key={item.id}>
                {item.job ? (
                  <>
                    <p>
                      {resultName({
                        recordId: item.job.region.record_id,
                        window: item.job.region,
                        calculatedAt: item.job.created_at,
                      })}
                    </p>
                    <p>
                      Статус сервера: {stageText(item.job.status)}{item.job.stage !== item.job.status ? `. Стадия: ${stageText(item.job.stage)}` : ""}.
                      {item.job.author_name ? ` Запустил ${item.job.author_name}.` : ""}
                    </p>
                    <Button
                      kind="ghost"
                      size="md"
                      onClick={() => {
                        const next = new URLSearchParams(navigation.current.params);
                        next.set("job_id", item.job!.id);
                        next.set("record", item.job!.region.record_id);
                        next.set("source_sha256", item.job!.source_sha256);
                        formatWindow(item.job!.region).forEach((value, index) => next.set(["r0", "r1", "c0", "c1"][index], value));
                        if (owner) next.set("settings_id", owner.id);
                        next.delete("zone_id");
                        navigation.current.setParams(next, { replace: true, preventScrollReset: true });
                      }}
                    >
                      Открыть задачу
                    </Button>
                  </>
                ) : (
                  <p>Связанная задача не прочитана{failed ? `: ${failed}` : "."}</p>
                )}
              </li>
            );
          })}
        </ul>
      </section>
      <section className={compareA && compareB ? "pipeline-compare is-open" : "pipeline-compare"} aria-label="Сравнение двух результатов">
        <h2>Сравнение двух результатов</h2>
        <p>
          Два уже посчитанных результата одной записи и одного SHA. Общая область — их непустое пересечение. Направление: {comparisonCaption().direction}. Победителя нет, точность обнаружения не оценивается.
        </p>
        <div className="analysis-controls">
          <Select id="pipeline-compare-a" labelText="Результат A" value={selectedA ? compareA : ""} onChange={(event) => setCompareA(event.target.value === compareB ? compareA : event.target.value)}>
            <SelectItem value="" text="Не выбран" />
            {runs.data?.map((item) => (
              <SelectItem key={item.id} value={item.id} text={resultName({ recordId: item.region.record_id, window: item.region, calculatedAt: item.calculated_at })} />
            ))}
          </Select>
          <Select id="pipeline-compare-b" labelText="Результат B" value={selectedB ? compareB : ""} onChange={(event) => setCompareB(event.target.value === compareA ? compareB : event.target.value)}>
            <SelectItem value="" text="Не выбран" />
            {runs.data?.map((item) => (
              <SelectItem key={item.id} value={item.id} text={resultName({ recordId: item.region.record_id, window: item.region, calculatedAt: item.calculated_at })} />
            ))}
          </Select>
        </div>
        {runs.isError && <p role="alert">Список результатов не прочитан: {String(runs.error)}</p>}
        {compareA && compareB && compareA === compareB && <p>Нужны два разных результата. Окно не подменено.</p>}
        {comparison && "reason" in comparison && <p role="status">{comparison.reason}</p>}
        {intersection && comparePoint && (
          <p className="mono">
            Пересечение: {windowLabel(selectedA?.region.record_id ?? "", intersection)}. Точка сравнения: строка {comparePoint.row}, столбец {comparePoint.column}.
            {(comparePoint.row !== row || comparePoint.column !== column) && " Точка страницы вне пересечения и не заменяется."}
          </p>
        )}
        {intersection && comparePoint && selectedA && selectedB && (
          <ComparisonView
            a={selectedA}
            b={selectedB}
            sliceA={compareSliceA.data?.run_id === selectedA.id ? compareSliceA.data : undefined}
            sliceB={compareSliceB.data?.run_id === selectedB.id ? compareSliceB.data : undefined}
            loading={compareSliceA.isFetching || compareSliceB.isFetching}
            error={compareSliceA.error ?? compareSliceB.error}
            point={comparePoint}
            columnIndex={comparePoint.column - intersection.column_start}
          />
        )}
      </section>
    </main>
  );
}

function ComparisonView({
  a,
  b,
  sliceA,
  sliceB,
  loading,
  error,
  point,
  columnIndex,
}: {
  a: Run;
  b: Run;
  sliceA: RunSlices | undefined;
  sliceB: RunSlices | undefined;
  loading: boolean;
  error: unknown;
  point: { row: number; column: number } | null;
  columnIndex: number;
}) {
  const processedA = sliceA?.layers.processed_temperature_c?.profile;
  const processedB = sliceB?.layers.processed_temperature_c?.profile;
  const rawA = sliceA?.layers.raw?.profile;
  const rawB = sliceB?.layers.raw?.profile;
  const x = sliceA?.x_m ?? sliceB?.x_m ?? [];
  const aligned = sameAxis(sliceA?.x_m, sliceB?.x_m);
  const delta =
    aligned && processedA && processedB && processedA.length === x.length && processedB.length === x.length
      ? resultDifference(processedA, processedB)
      : null;
  const temperatureScale = finiteExtent([...(processedA ?? []), ...(processedB ?? []), ...(rawA ?? []), ...(rawB ?? [])]);
  const coverageA = sliceA?.layers.processed_temperature_c
    ? breakTimeGaps(
        sliceA.time_s,
        sliceA.layers.processed_temperature_c.coverage.map((value) => (value === null ? null : value * 100)),
      )
    : null;
  const coverageB = sliceB?.layers.processed_temperature_c
    ? breakTimeGaps(
        sliceB.time_s,
        sliceB.layers.processed_temperature_c.coverage.map((value) => (value === null ? null : value * 100)),
      )
    : null;
  const scoresMatch = sameScoreDefinition(a.score, b.score);
  const scoreScale = finiteExtent([...(sliceA?.layers.score?.profile ?? []), ...(sliceB?.layers.score?.profile ?? [])]);
  return (
    <>
      {loading && <p role="status">Загрузка срезов пересечения…</p>}
      {error && <InlineNotification kind="error" title="Срез сравнения не загружен" subtitle={String(error)} hideCloseButton lowContrast />}
      <div className="pipeline-provenance">
        <h3>Что сравнивается</h3>
        <dl className="pipeline-passport">
          <dt>Направление</dt>
          <dd>{comparisonCaption().direction}. {comparisonCaption().winner === null ? "Победителя нет." : ""}</dd>
          <dt>Результат A</dt>
          <dd>{resultName({ recordId: a.region.record_id, window: a.region, calculatedAt: a.calculated_at })}</dd>
          <dt>Результат B</dt>
          <dd>{resultName({ recordId: b.region.record_id, window: b.region, calculatedAt: b.calculated_at })}</dd>
          <dt>Маска источника A</dt>
          <dd>{maskText(sliceA?.source_invalid_reason_profile)}</dd>
          <dt>Маска источника B</dt>
          <dd>{maskText(sliceB?.source_invalid_reason_profile)}</dd>
          <dt>Покрытие A</dt>
          <dd>{coverageText(sliceA?.layers.processed_temperature_c?.coverage)}</dd>
          <dt>Покрытие B</dt>
          <dd>{coverageText(sliceB?.layers.processed_temperature_c?.coverage)}</dd>
          <dt>Сопоставимые пары</dt>
          <dd>{delta ? `${delta.pairs.toLocaleString("ru")} пар, где обе обработки — числа. Пропуски не заменены нулём.` : "Пары обработки не посчитаны: нет двух выровненных слоёв обработки."}</dd>
        </dl>
      </div>
      {sliceA && sliceA.layers.processed_temperature_c && (
        <div className="pipeline-field">
          <Heatmap path={tilePath(a.id, "processed_temperature_c", sliceA)} versionKey={`${a.id}-compare`} expectedSourceSha={a.source_sha256} />
        </div>
      )}
      {sliceB && sliceB.layers.processed_temperature_c && (
        <div className="pipeline-field">
          <Heatmap path={tilePath(b.id, "processed_temperature_c", sliceB)} versionKey={`${b.id}-compare`} expectedSourceSha={b.source_sha256} />
        </div>
      )}
      <p>Профили температуры на одной шкале °C. Цвет полей задаёт сервер для каждого слоя отдельно.</p>
      <div className="line-grid">
        {processedA && (
          <LineChart title="Температура обработки A" x={sliceA!.x_m} y={processedA} xLabel="Расстояние, м" yLabel="°C" yLimits={temperatureScale} syncKey="pipeline-compare" />
        )}
        {processedB && (
          <LineChart title="Температура обработки B" x={sliceB!.x_m} y={processedB} xLabel="Расстояние, м" yLabel="°C" yLimits={temperatureScale} syncKey="pipeline-compare" />
        )}
        {rawA && (
          <LineChart title="Температура исходника A" x={sliceA!.x_m} y={rawA} xLabel="Расстояние, м" yLabel="°C" yLimits={temperatureScale} syncKey="pipeline-compare" />
        )}
        {rawB && (
          <LineChart title="Температура исходника B" x={sliceB!.x_m} y={rawB} xLabel="Расстояние, м" yLabel="°C" yLimits={temperatureScale} syncKey="pipeline-compare" />
        )}
      </div>
      {!processedA && <p>{absentLayer(false).reason} Результат B его не заменяет.</p>}
      {!processedB && <p>{absentLayer(false).reason} Результат A его не заменяет.</p>}
      {processedA && processedB && !delta && <p>Оси или длины профилей пересечения не совпали. Разность не построена и не заменена одним профилем.</p>}
      {delta && (
        <LineChart
          title={`Разность: ${comparisonCaption().direction}`}
          x={x}
          y={delta.delta}
          xLabel="Расстояние, м"
          yLabel="°C"
          yLimits={symmetricScale(delta.delta, undefined)}
          syncKey="pipeline-compare"
        />
      )}
      <p>Направление разности: {comparisonCaption().direction}. Шкала симметричная. Победителя нет. Это разность сигнала, не точность обнаружения.</p>
      <div className="line-grid">
        {sliceA?.layers.processed_temperature_c && (
          <LineChart title="Покрытие A" x={coverageA!.x} y={coverageA!.y} xLabel="Время от начала записи, с" yLabel="%" yLimits={[0, 100]} syncKey="pipeline-compare-coverage" />
        )}
        {sliceB?.layers.processed_temperature_c && (
          <LineChart title="Покрытие B" x={coverageB!.x} y={coverageB!.y} xLabel="Время от начала записи, с" yLabel="%" yLimits={[0, 100]} syncKey="pipeline-compare-coverage" />
        )}
      </div>
      <h3>Score, отдельно</h3>
      {scoresMatch ? (
        <p>Определение score совпадает ({a.score?.name || "без названия"}, {a.score?.unit || "единица не передана"}). Это не температура и не общая метрика с разностью. Победитель не назначается.</p>
      ) : (
        <p>Определения score различаются либо не переданы. Это не одна метрика. Разность score не строится.</p>
      )}
      <dl className="pipeline-passport">
        <dt>Score A</dt>
        <dd>{a.score ? `${a.score.name || "без названия"} · ${a.score.unit || "единица не передана"} · ${a.score.meaning || "смысл не передан"}` : "Score не передан"}{point ? ` · значение ${textNumber(sliceA?.layers.score?.profile[columnIndex])}` : ""}</dd>
        <dt>Score B</dt>
        <dd>{b.score ? `${b.score.name || "без названия"} · ${b.score.unit || "единица не передана"} · ${b.score.meaning || "смысл не передан"}` : "Score не передан"}{point ? ` · значение ${textNumber(sliceB?.layers.score?.profile[columnIndex])}` : ""}</dd>
      </dl>
      {scoresMatch && sliceA?.layers.score && sliceB?.layers.score && (
        <div className="line-grid">
          <LineChart title="Score A" x={sliceA.x_m} y={sliceA.layers.score.profile} xLabel="Расстояние, м" yLabel={a.score?.unit || "score"} yLimits={scoreScale} syncKey="pipeline-compare-score" />
          <LineChart title="Score B" x={sliceB.x_m} y={sliceB.layers.score.profile} xLabel="Расстояние, м" yLabel={b.score?.unit || "score"} yLimits={scoreScale} syncKey="pipeline-compare-score" />
        </div>
      )}
    </>
  );
}

function SourceBlock({
  path,
  sha,
  window,
  axes,
  row,
  column,
  slice,
  retained,
  loading,
  current,
  error,
  onPoint,
}: {
  path: string;
  sha: string;
  window: HalfOpenWindow | null;
  axes: { x: number[]; t: number[]; timeGap: boolean } | null;
  row: number;
  column: number;
  slice: Slices | undefined;
  retained?: { row: number; column: number };
  loading: boolean;
  current: boolean;
  error: unknown;
  onPoint: (row: number, column: number) => void;
}) {
  const localColumn = window ? (slice?.column ?? column) - window.column_start : 0;
  const trendSeries = slice ? breakTimeGaps(slice.time_s, slice.trend) : null;
  const missing =
    slice && window
      ? slice.profile.slice(window.column_start, window.column_end).filter((value) => value === null)
          .length
      : null;
  return (
    <>
      <h2>Исходник</h2>
      <p>Нативное поле, профиль и ряд точки. Заполненный пропуск не показан как измерение.</p>
      <p className="pipeline-selection" role="status">
        {slice && current ? `Срез: строка ${slice.row}, столбец ${slice.column}.` : !slice && retained ? `Последний успешный срез: строка ${retained.row}, столбец ${retained.column}. Новая точка не построена.` : loading ? "Обновление выбранной точки…" : "Срез ещё не получен."}
      </p>
      {error && (
        <InlineNotification
          kind="error"
          title="Исходный срез не загружен"
          subtitle={String(error)}
          hideCloseButton
          lowContrast
        />
      )}
      <div className="pipeline-field">
        {loading && <p role="status">Загрузка исходного поля…</p>}
        {path && sha && (
          <Heatmap
            path={path}
            versionKey={sha}
            expectedSourceSha={sha}
            cursor={
              axes && slice
                ? [axes.x[localColumn], axes.t[(slice?.row ?? row) - (window?.row_start ?? 0)]]
                : undefined
            }
            onSelect={(x, t) => {
              if (!axes || !window) return;
              onPoint(window.row_start + nearest(axes.t, t), window.column_start + nearest(axes.x, x));
            }}
          />
        )}
      </div>
      {slice && current && window && profileReady(slice, window) && (
        <>
          <p>
            Точка {textNumber(slice.distance_m)} м · {slice.timestamp} ·{" "}
            {slice.temperature === null ? "нет измерения" : `${textNumber(slice.temperature)} °C`} ·{" "}
            {qualityLabel(slice.invalid_reason)}
          </p>
          <p>
            В окне профиля пропусков: {missing}. Время с разрывом больше 300 с не соединяется
            выдуманной точкой.
            {axes?.timeGap
              ? " В этом окне такой разрыв есть: соседние строки поля не являются соседними секундами."
              : ""}
          </p>
          <div className="line-grid">
            <LineChart
              title="Точный профиль исходника"
              x={slice.x_m.slice(window.column_start, window.column_end)}
              y={slice.profile.slice(window.column_start, window.column_end)}
              xLabel="Расстояние, м"
              yLabel="°C"
              syncKey={sha}
            />
            <LineChart
              title="Температура выбранной координаты во времени"
              x={trendSeries!.x}
              y={trendSeries!.y}
              xLabel="Время от начала записи, с"
              yLabel="°C"
              syncKey={`${sha}-time`}
            />
          </div>
        </>
      )}
    </>
  );
}

function profileReady(slice: Slices, window: HalfOpenWindow) {
  return (
    slice.x_m.length >= window.column_end &&
    slice.profile.length >= window.column_end &&
    window.column_end - window.column_start >= 2
  );
}

function CleanBlock({
  job,
  run,
  slice,
  retained,
  loading,
  current,
  error,
  view,
  onView,
  changes,
  temperatureScale,
  index,
  zones,
  selectedZone,
  onPoint,
}: {
  job: Job | undefined;
  run: Run | undefined;
  slice: RunSlices | undefined;
  retained?: { row: number; column: number };
  loading: boolean;
  current: boolean;
  error: unknown;
  view: "raw" | "processed_temperature_c" | "difference";
  onView: (view: "raw" | "processed_temperature_c" | "difference") => void;
  changes: ReturnType<typeof profileChanges>;
  temperatureScale: [number, number] | undefined;
  index: { row: number; column: number } | null;
  zones?: ZoneBox[];
  selectedZone?: Run["zones"][number];
  onPoint: (row: number, column: number) => void;
}) {
  const missing = !job?.run_id || job.status !== "complete" || !run;
  const processed = !!run?.layers.includes("processed_temperature_c");
  const coverageSeries = slice?.layers.processed_temperature_c
    ? breakTimeGaps(
        slice.time_s,
        slice.layers.processed_temperature_c.coverage.map((value) => (value === null ? null : value * 100)),
      )
    : null;
  return (
    <>
      <h2>Поле обработки</h2>
      {selectedZone && <p>Контекст {selectedZone.id}: строки [{selectedZone.row_start}, {selectedZone.row_end}), столбцы [{selectedZone.column_start}, {selectedZone.column_end}). Гипотеза алгоритма, не экспертное заключение.</p>}
      <p className="pipeline-selection" role="status">
        {slice && current ? `Срез: строка ${slice.row}, столбец ${slice.column}.` : !slice && retained ? `Последний успешный срез: строка ${retained.row}, столбец ${retained.column}. Новая точка не построена.` : loading ? "Загрузка результата…" : "Срез результата ещё не получен."}
      </p>
      {missing && (
        <p>
          Сохранённой обработки для показа пока нет.
          {job ? ` Статус задачи: ${stageText(job.status)}.` : " Задача ещё не отправлена."}{" "}
          Её даёт кнопка «Проверить пайплайн» по текущему окну и SHA. Исходник вместо результата не
          подставляется.
        </p>
      )}
      {error && (
        <InlineNotification
          kind="error"
          title="Результат обработки не загружен"
          subtitle={String(error)}
          hideCloseButton
          lowContrast
        />
      )}
      {run && slice && current && processed && (
        <>
          <p>
            {resultName({ recordId: run.region.record_id, window: run.region, calculatedAt: run.calculated_at })}.
            Исходник и обработка делят одну температурную шкалу. Разность — обработка − исходник.
            Это изменение сигнала, не качество очистки и не качество обнаружения. Время с разрывом
            больше 300 с не соединяется. Заполненный алгоритмом пропуск не является измерением.
          </p>
          <div className="pipeline-actions">
            {(
              [
                ["raw", "Поле исходника"],
                ["processed_temperature_c", "Поле обработки"],
                ["difference", "Поле разности"],
              ] as const
            ).map(([id, label]) => (
              <Button
                key={id}
                kind={view === id ? "secondary" : "ghost"}
                size="md"
                onClick={() => onView(id)}
              >
                {label}
              </Button>
            ))}
          </div>
          <div className="pipeline-field">
            {loading && <p role="status">Загрузка поля run…</p>}
            <Heatmap
              path={tilePath(run.id, view, slice)}
              versionKey={run.id}
              expectedSourceSha={run.source_sha256}
              zones={zones}
              selectedZoneId={selectedZone?.id}
              cursor={
                index && slice.x_m[index.column] !== undefined && slice.native_time_s[index.row] !== undefined
                  ? [slice.x_m[index.column], slice.native_time_s[index.row]]
                  : undefined
              }
              onSelect={(x, t) => {
                if (!job) return;
                onPoint(
                  job.region.row_start + nearest(slice.native_time_s, t),
                  job.region.column_start + nearest(slice.x_m, x),
                );
              }}
            />
          </div>
          <dl className="pipeline-passport">
            <dt>Сопоставимых пар</dt>
            <dd>{changes ? changes.paired_points : "Нет данных"}</dd>
            <dt>Среднее |обработка − исходник|</dt>
            <dd>{textNumber(changes?.mean_absolute_change_c)} °C</dd>
            <dt>Максимум |обработка − исходник|</dt>
            <dd>{textNumber(changes?.max_absolute_change_c)} °C</dd>
            <dt>Покрытие</dt>
            <dd>
              {changes
                ? `В профиле ${changes.total_points} точек: без исходника ${changes.missing_original}, без обработки ${changes.missing_processed}.`
                : "Слой обработки не передан."}
            </dd>
          </dl>
          <div className="line-grid">
            <LineChart
              title="Исходный профиль"
              x={slice.x_m}
              y={slice.layers.raw.profile}
              xLabel="Расстояние, м"
              yLabel="°C"
              yLimits={temperatureScale}
              syncKey={`${run.id}-temp`}
            />
            <LineChart
              title="Обработанный профиль"
              x={slice.x_m}
              y={slice.layers.processed_temperature_c.profile}
              xLabel="Расстояние, м"
              yLabel="°C"
              yLimits={temperatureScale}
              syncKey={`${run.id}-temp`}
            />
            {slice.layers.difference && (
              <LineChart
                title="Разность: обработка − исходник"
                x={slice.x_m}
                y={slice.layers.difference.profile}
                xLabel="Расстояние, м"
                yLabel="°C"
                yLimits={symmetricScale(slice.layers.difference.profile, run.scales?.difference)}
                syncKey={`${run.id}-difference`}
              />
            )}
          </div>
          <LineChart
            title="Покрытие обработки по времени"
            x={coverageSeries!.x}
            y={coverageSeries!.y}
            xLabel="Время от начала записи, с"
            yLabel="%"
            yLimits={[0, 100]}
            syncKey={`${run.id}-time`}
          />
        </>
      )}
      {run && !processed && <p>{absentLayer(processed).reason}</p>}
    </>
  );
}

function ScoreBlock({
  job,
  run,
  slice,
  retained,
  loading,
  current,
  error,
  view,
  onView,
  index,
  onPoint,
}: {
  job: Job | undefined;
  run: Run | undefined;
  slice: RunSlices | undefined;
  retained?: { row: number; column: number };
  loading: boolean;
  current: boolean;
  error: unknown;
  view: "background_temperature_c" | "score";
  onView: (view: "background_temperature_c" | "score") => void;
  index: { row: number; column: number } | null;
  onPoint: (row: number, column: number) => void;
}) {
  const background = !!run?.layers.includes("background_temperature_c");
  const score = !!run?.layers.includes("score");
  const active = view === "score" && score ? "score" : background ? "background_temperature_c" : null;
  const shown = view === "background_temperature_c" && background ? view : active;
  const trendSeries =
    shown && slice?.layers[shown]
      ? breakTimeGaps(slice.time_s, slice.layers[shown].trend)
      : null;
  return (
    <>
      <h2>Фон и score</h2>
      <p className="pipeline-selection" role="status">
        {slice && current ? `Срез: строка ${slice.row}, столбец ${slice.column}.` : !slice && retained ? `Последний успешный срез: строка ${retained.row}, столбец ${retained.column}. Новая точка не построена.` : loading ? "Загрузка фона и score…" : "Срез результата ещё не получен."}
      </p>
      <p>Качество обнаружения не оценено: эталона аварий нет. Score не является вероятностью.</p>
      {(!job?.run_id || job.status !== "complete") && (
        <p>
          Фона и score ещё нет.
          {job ? ` Статус задачи: ${stageText(job.status)}.` : " Сначала нужен расчёт."} Эти слои
          появляются только если сервер сохранил их в результате.
        </p>
      )}
      {error && (
        <InlineNotification
          kind="error"
          title="Фон и score не загружены"
          subtitle={String(error)}
          hideCloseButton
          lowContrast
        />
      )}
      {run && (
        <>
          {!background && <p>Фон: {absentLayer(background).reason}</p>}
          {!score && <p>Score: {absentLayer(score).reason}</p>}
          {run.score && score && (
            <dl className="pipeline-passport">
              <dt>Название</dt>
              <dd>{run.score.name || "не передано"}</dd>
              <dt>Единица</dt>
              <dd>{run.score.unit || "не передана"}</dd>
              <dt>Смысл</dt>
              <dd>{run.score.meaning || "не передан"}</dd>
              <dt>Порог</dt>
              <dd>{run.score.threshold ?? "не задан"}</dd>
              <dt>Сравнение</dt>
              <dd>
                {run.score.comparison === "ge"
                  ? "значение ≥ порога"
                  : run.score.comparison === "abs_ge"
                    ? "модуль значения ≥ порога"
                    : "правило сравнения не передано"}
              </dd>
            </dl>
          )}
          {shown && current && slice?.layers[shown] && (
            <>
              <div className="pipeline-actions">
                {background && (
                  <Button
                    kind={shown === "background_temperature_c" ? "secondary" : "ghost"}
                    size="md"
                    onClick={() => onView("background_temperature_c")}
                  >
                    Фон
                  </Button>
                )}
                {score && (
                  <Button
                    kind={shown === "score" ? "secondary" : "ghost"}
                    size="md"
                    onClick={() => onView("score")}
                  >
                    Score
                  </Button>
                )}
              </div>
              <div className="pipeline-field">
                {loading && <p role="status">Загрузка слоя…</p>}
                <Heatmap
                  path={tilePath(run.id, shown, slice)}
                  versionKey={run.id}
                  expectedSourceSha={run.source_sha256}
                  cursor={index ? [slice.x_m[index.column], slice.native_time_s[index.row]] : undefined}
                  onSelect={(x, t) => {
                    if (job) onPoint(job.region.row_start + nearest(slice.native_time_s, t), job.region.column_start + nearest(slice.x_m, x));
                  }}
                />
              </div>
              <LineChart
                title={shown === "score" ? "Score выбранного профиля" : "Фон выбранного профиля"}
                x={slice.x_m}
                y={slice.layers[shown].profile}
                xLabel="Расстояние, м"
                yLabel={shown === "score" ? run.score?.unit || "score" : "°C"}
                yLimits={
                  shown === "score"
                    ? run.scales?.score
                    : finiteExtent(slice.layers[shown].profile)
                }
                threshold={shown === "score" ? run.score?.threshold ?? undefined : undefined}
                thresholdMode={run.score?.comparison === "abs_ge" ? "abs_ge" : "ge"}
                syncKey={`${run.id}-${shown}`}
              />
              <p>
                Точное значение среза:{" "}
                {textNumber(
                  index ? slice.layers[shown].profile[index.column] : null,
                )}{" "}
                {shown === "score" ? `${run.score?.unit || "score"}. Score не является вероятностью.` : "°C."}
              </p>
              <LineChart
                title={shown === "score" ? "Score выбранной точки во времени" : "Фон выбранной точки во времени"}
                x={trendSeries!.x}
                y={trendSeries!.y}
                xLabel="Время от начала записи, с"
                yLabel={shown === "score" ? run.score?.unit || "score" : "°C"}
                threshold={shown === "score" ? run.score?.threshold ?? undefined : undefined}
                thresholdMode={run.score?.comparison === "abs_ge" ? "abs_ge" : "ge"}
                syncKey={`${run.id}-time`}
              />
            </>
          )}
        </>
      )}
    </>
  );
}

function ZonesBlock({
  job,
  run,
  zones,
  loading,
  error,
  selected,
  onSelect,
}: {
  job: Job | undefined;
  run: Run | undefined;
  zones: ZoneBox[] | undefined;
  loading: boolean;
  error: unknown;
  selected: string;
  onSelect: (zone: NonNullable<Run["zones"][number]>) => void;
}) {
  return (
    <>
      <h2>Зоны</h2>
      <p>Это предложения алгоритма. Эксперт их не подтверждал. Форма не является доказанной причиной.</p>
      {(!job?.run_id || job.status !== "complete") && (
        <p>
          Зон сохранённого run ещё нет.
          {job ? ` Статус задачи: ${stageText(job.status)}.` : " Сначала нужен расчёт."}
        </p>
      )}
      {loading && <p role="status">Загрузка зон…</p>}
      {error && (
        <InlineNotification
          kind="error"
          title="Зоны не загружены"
          subtitle={String(error)}
          hideCloseButton
          lowContrast
        />
      )}
      {run && run.zones.length === 0 && (
        <p>
          Расчёт завершён; зон по заданному правилу не выделено.{" "}
          {resultName({ recordId: run.region.record_id, window: run.region, calculatedAt: run.calculated_at })}.
        </p>
      )}
      {run?.zones.map((zone) => {
        const boxes = (zones ?? []).filter((item) => item.id === zone.id);
        return (
          <article key={zone.id} className="pipeline-status">
            <h3>{zone.id}</h3>
            <p>Происхождение: предложение закреплённого расчёта, не экспертная разметка.</p>
            <p>Гипотеза формы: {zone.shape_hypothesis || "не указана"}.</p>
            <p className="mono">
              Индексы: строки [{zone.row_start}, {zone.row_end}), столбцы [{zone.column_start},{" "}
              {zone.column_end}).
            </p>
            {boxes.length > 0 ? (
              <ul>
                {boxes.map((box) => (
                  <li key={`${box.id}-${box.segment}`}>
                    Физически: {textNumber(box.x0)}–{textNumber(box.x1)} м, {textNumber(box.t0)}–
                    {textNumber(box.t1)} с от начала записи. {box.label}
                  </li>
                ))}
              </ul>
            ) : (
              <p>Физические границы этого предложения ещё не получены.</p>
            )}
            <Button
              kind={selected === zone.id ? "secondary" : "tertiary"}
              size="md"
              onClick={() => onSelect(zone)}
            >
              Открыть контекст зоны
            </Button>
          </article>
        );
      })}
    </>
  );
}

function tilePath(runId: string, layer: string, slice: RunSlices) {
  return `/api/v2/result-runs/${runId}/tiles?${new URLSearchParams({
    layer,
    width: matchMedia("(max-width: 672px)").matches ? "128" : "256",
    height: "256",
    x0: String(slice.x_m[0]),
    x1: String(slice.x_m.at(-1)),
    t0: String(slice.native_time_s[0]),
    t1: String(slice.native_time_s.at(-1)),
  })}`;
}

