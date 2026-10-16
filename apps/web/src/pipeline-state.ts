/** Draft window, submitted body, saved revision, job link, and idempotency. No network calls. */

export type HalfOpenWindow = {
  row_start: number;
  row_end: number;
  column_start: number;
  column_end: number;
};

export type SubmitBody = {
  region: HalfOpenWindow & { record_id: string };
  source_sha256: string;
  settings_id?: string;
  settings_revision?: number;
};

export type BoundDraft = [string, string, string, string];

export type Attempt = { key: string; body: SubmitBody };

export type PipelineSession = {
  draft: HalfOpenWindow;
  submitted: SubmitBody | null;
  attempt: Attempt | null;
  jobId: string | null;
  shownRunId: string | null;
  /** Saved document revision. Distinct from the submitted snapshot and the job id. */
  savedRevision?: number | null;
};

export type SettingsDraft = {
  title: string;
  record_id: string;
  source_sha256: string;
  window: HalfOpenWindow;
  point: { row: number; column: number };
  method: string;
  method_version: string;
  run_ids: readonly string[];
};

export type CompareRun = {
  id: string;
  record_id: string;
  source_sha256: string;
  window: HalfOpenWindow;
};

export type ScoreDefinition = {
  name?: string;
  unit?: string;
  meaning?: string;
  comparison?: string;
  kind?: string;
};

export const PIPELINE_BLOCKS = [
  { id: "source", label: "Исходник" },
  { id: "clean", label: "Поле обработки" },
  { id: "score", label: "Фон и score" },
  { id: "zones", label: "Зоны" },
] as const;

export type PipelineBlock = (typeof PIPELINE_BLOCKS)[number]["id"];

const knownStages: Record<string, string> = {
  queued: "В очереди",
  running: "Выполняется",
  computing: "Расчёт",
  validating: "Проверка результата",
  complete: "Готово",
  failed: "Не выполнено",
  cancelled: "Отменено",
};

/** Translate a server stage token. Unknown tokens stay verbatim: no invented percent. */
export function stageText(stage: string): string {
  return knownStages[stage] ?? stage;
}

export function parseHalfOpen(
  text: readonly [string, string, string, string],
  shape: readonly [number, number],
): HalfOpenWindow | null {
  const nums = text.map((value) => (/^\d+$/.test(value) ? Number(value) : Number.NaN));
  if (!nums.every((value) => Number.isSafeInteger(value))) return null;
  const [row_start, row_end, column_start, column_end] = nums;
  if (
    !(
      0 <= row_start &&
      row_start < row_end &&
      row_end <= shape[0] &&
      0 <= column_start &&
      column_start + 1 < column_end &&
      column_end <= shape[1]
    )
  )
    return null;
  return { row_start, row_end, column_start, column_end };
}

export function formatWindow(window: HalfOpenWindow): BoundDraft {
  return [
    String(window.row_start),
    String(window.row_end),
    String(window.column_start),
    String(window.column_end),
  ];
}

/** Four bound fields travel as one draft. Editing one index keeps the other three. */
export function editBoundField(draft: BoundDraft, index: 0 | 1 | 2 | 3, value: string): BoundDraft {
  const next: BoundDraft = [draft[0], draft[1], draft[2], draft[3]];
  next[index] = value;
  return next;
}

export function limitedWindow(
  shape: readonly [number, number],
): HalfOpenWindow | null {
  if (shape[0] < 1 || shape[1] < 2) return null;
  const row_end = Math.min(shape[0], 32);
  const column_end = Math.min(shape[1], 64);
  return { row_start: 0, row_end, column_start: 0, column_end };
}

export function submitBody(
  recordId: string,
  sourceSha256: string,
  window: HalfOpenWindow,
  settings?: { id: string; revision: number },
): SubmitBody {
  return {
    region: { record_id: recordId, ...window },
    source_sha256: sourceSha256,
    ...(settings ? { settings_id: settings.id, settings_revision: settings.revision } : {}),
  };
}

export function sameWindow(a: HalfOpenWindow, b: HalfOpenWindow): boolean {
  return (
    a.row_start === b.row_start &&
    a.row_end === b.row_end &&
    a.column_start === b.column_start &&
    a.column_end === b.column_end
  );
}

export function sameBody(a: SubmitBody, b: SubmitBody): boolean {
  return (
    a.source_sha256 === b.source_sha256 &&
    (a.settings_id ?? "") === (b.settings_id ?? "") &&
    (a.settings_revision ?? null) === (b.settings_revision ?? null) &&
    a.region.record_id === b.region.record_id &&
    sameWindow(a.region, b.region)
  );
}

export function sameSettingsDraft(a: SettingsDraft, b: SettingsDraft): boolean {
  return (
    a.title === b.title &&
    a.record_id === b.record_id &&
    a.source_sha256 === b.source_sha256 &&
    sameWindow(a.window, b.window) &&
    a.point.row === b.point.row &&
    a.point.column === b.point.column &&
    a.method === b.method &&
    a.method_version === b.method_version &&
    a.run_ids.length === b.run_ids.length &&
    a.run_ids.every((id, index) => id === b.run_ids[index])
  );
}

/** PATCH body: expected revision plus fields that actually differ. Job ids are server-owned. */
export function changedSettingsPatch(
  saved: SettingsDraft,
  revision: number,
  draft: SettingsDraft,
): Record<string, unknown> | null {
  const patch: Record<string, unknown> = { expected_revision: revision };
  if (draft.title !== saved.title) patch.title = draft.title;
  if (draft.record_id !== saved.record_id) patch.record_id = draft.record_id;
  if (draft.source_sha256 !== saved.source_sha256) patch.source_sha256 = draft.source_sha256;
  if (!sameWindow(draft.window, saved.window)) patch.window = draft.window;
  if (draft.point.row !== saved.point.row || draft.point.column !== saved.point.column) patch.point = draft.point;
  if (draft.method !== saved.method) patch.method = draft.method;
  if (draft.method_version !== saved.method_version) patch.method_version = draft.method_version;
  if (draft.run_ids.length !== saved.run_ids.length || draft.run_ids.some((id, index) => id !== saved.run_ids[index]))
    patch.run_ids = [...draft.run_ids];
  return Object.keys(patch).length === 1 ? null : patch;
}

/** An explicit source is never silently replaced; a job-only link waits for its job. */
export function restoredRecord(
  explicit: string | null,
  jobId: string,
  jobRecord: string | undefined,
  firstRecord: string,
): string | null {
  if (explicit) return explicit;
  return jobId ? jobRecord ?? null : firstRecord;
}

export type LinkedJob = {
  record_id: string;
  source_sha256: string;
  region: HalfOpenWindow;
};

export type LinkedSetting = {
  record_id: string;
  source_sha256: string;
  window: HalfOpenWindow;
  point: { row: number; column: number };
};

export type LinkResolution =
  | { action: "wait" }
  | { action: "keep"; error: "unknown-record" | "invalid-bounds" | null }
  | {
      action: "fill";
      record: string;
      source_sha256: string | null;
      window: HalfOpenWindow | null;
      point: { row: number; column: number } | null;
    };

function copyWindow(window: HalfOpenWindow): HalfOpenWindow {
  return {
    row_start: window.row_start,
    row_end: window.row_end,
    column_start: window.column_start,
    column_end: window.column_end,
  };
}

/**
 * A link that is only a safe job id restores that job's record, SHA, and window
 * once the payload is known. An explicit unknown record or invalid bounds is
 * kept as an error and is not replaced by the job or by another window.
 */
export function resolvePipelineLink(input: {
  record: string | null;
  bounds: readonly [string | null, string | null, string | null, string | null];
  shape: readonly [number, number] | null;
  knownRecords: readonly string[] | null;
  jobId: string;
  job: LinkedJob | null;
  settingPending: boolean;
  setting: LinkedSetting | null;
  firstRecord: string;
}): LinkResolution {
  const boundsPresent = input.bounds.some((value) => value !== null);
  const parsed =
    boundsPresent && input.shape
      ? parseHalfOpen(
          [input.bounds[0] ?? "", input.bounds[1] ?? "", input.bounds[2] ?? "", input.bounds[3] ?? ""],
          input.shape,
        )
      : null;
  const invalidBounds = boundsPresent && input.shape !== null && parsed === null;
  const unknownRecord =
    input.record !== null && input.knownRecords !== null && !input.knownRecords.includes(input.record);
  if (unknownRecord) return { action: "keep", error: "unknown-record" };
  if (invalidBounds) {
    if (input.record === null && input.jobId) {
      if (!input.job) return { action: "wait" };
      return {
        action: "fill",
        record: input.job.record_id,
        source_sha256: input.job.source_sha256,
        window: null,
        point: null,
      };
    }
    return { action: "keep", error: "invalid-bounds" };
  }
  const jobOnly = input.jobId !== "" && input.record === null && !boundsPresent;
  if (jobOnly) {
    if (!input.job) return { action: "wait" };
    return {
      action: "fill",
      record: input.job.record_id,
      source_sha256: input.job.source_sha256,
      window: copyWindow(input.job.region),
      point: null,
    };
  }
  if (input.record === null && !boundsPresent && input.jobId === "") {
    if (input.setting) {
      return {
        action: "fill",
        record: input.setting.record_id,
        source_sha256: input.setting.source_sha256,
        window: copyWindow(input.setting.window),
        point: { row: input.setting.point.row, column: input.setting.point.column },
      };
    }
    if (input.settingPending || input.knownRecords === null) return { action: "wait" };
    return { action: "fill", record: input.firstRecord, source_sha256: null, window: null, point: null };
  }
  return { action: "keep", error: null };
}

/** Non-empty half-open intersection of one record and one SHA. Failures carry a reason and no window. */
export function compareResults(
  a: CompareRun | null,
  b: CompareRun | null,
): { window: HalfOpenWindow } | { reason: string } {
  if (!a || !b) return { reason: "Нет двух сохранённых результатов. Окно не подменено." };
  if (a.record_id !== b.record_id)
    return { reason: "Разные записи. Пересечение не подменяется окном одного результата." };
  if (a.source_sha256 !== b.source_sha256)
    return { reason: "Разный SHA источника. Пересечение не подменяется." };
  const window = {
    row_start: Math.max(a.window.row_start, b.window.row_start),
    row_end: Math.min(a.window.row_end, b.window.row_end),
    column_start: Math.max(a.window.column_start, b.window.column_start),
    column_end: Math.min(a.window.column_end, b.window.column_end),
  };
  if (!(window.row_start < window.row_end && window.column_start < window.column_end))
    return { reason: "Пересечение полуоткрытых окон пустое. Другое окно не подставлено." };
  return { window };
}

/**
 * A saved result is named by the record, the window, and the time the server
 * reported. A method commit is not a name.
 */
export function resultName(input: {
  recordId: string;
  window: HalfOpenWindow;
  calculatedAt?: string | number | null;
}): string {
  return `запись ${input.recordId}, строки [${input.window.row_start}, ${input.window.row_end}), столбцы [${input.window.column_start}, ${input.window.column_end}), ${clockText(input.calculatedAt)}`;
}

function clockText(value: string | number | null | undefined): string {
  if (typeof value === "number" && Number.isFinite(value)) {
    return formatUtc(value > 1e12 ? value : value * 1000);
  }
  if (typeof value === "string" && value.trim()) {
    const text = value.trim();
    if (/^\d{4}-\d{2}-\d{2}T/.test(text)) {
      const parsed = Date.parse(text);
      if (!Number.isNaN(parsed)) return formatUtc(parsed);
    }
    return text;
  }
  return "время не передано";
}

function formatUtc(ms: number): string {
  const date = new Date(ms);
  if (Number.isNaN(date.getTime())) return "время не передано";
  const pad = (part: number) => String(part).padStart(2, "0");
  return `${pad(date.getUTCDate())}.${pad(date.getUTCMonth() + 1)}.${date.getUTCFullYear()} ${pad(date.getUTCHours())}:${pad(date.getUTCMinutes())} UTC`;
}

/** Direction is fixed. Nothing in the comparison is a winner. */
export function comparisonCaption(): { direction: "результат B − результат A"; winner: null } {
  return { direction: "результат B − результат A", winner: null };
}

/** A missing layer stays missing. The source and any other field are not a substitute. */
export function absentLayer(present: boolean): { reason: string | null } {
  if (present) return { reason: null };
  return { reason: "Слой не передан. Исходник и другое поле его не заменяют." };
}

/**
 * The save line is about a reusable name. A calculation already on screen is
 * stored even when no name is selected, and a name that already lists this job
 * is not described as missing.
 */
export function savePrompt(input: {
  savedTitle: string | null;
  draftMatches: boolean;
  authorName?: string | null;
  jobOnScreen: boolean;
  ownerTitles?: readonly string[];
  loadingSelection?: boolean;
}): string {
  if (input.loadingSelection) return "Загрузка сохранённого окна…";
  if (input.savedTitle !== null) {
    const title = input.savedTitle.trim() || "без названия";
    if (input.draftMatches) {
      const author = input.authorName?.trim();
      return `Сохранено «${title}»${author ? ` · ${author}` : ""}.`;
    }
    return `Окно на экране отличается от сохранённого «${title}». Уже отправленный расчёт от этого не меняется.`;
  }
  const owners = (input.ownerTitles ?? []).map((title) => title.trim() || "без названия");
  if (owners.length === 1) {
    return `Эта задача уже входит в сохранённое окно «${owners[0]}». Выберите его в списке, чтобы открыть те же границы. Сам расчёт от этого не меняется.`;
  }
  if (owners.length > 1) {
    return "Эта задача входит в несколько сохранённых окон. Выберите нужное в списке. Сам расчёт от этого не меняется.";
  }
  if (input.jobOnScreen) {
    return "Расчёт уже хранится на сервере. Название, чтобы открыть это окно снова, ещё не записано.";
  }
  return "Это окно ещё не сохранено.";
}

/** Result B minus result A. Missing values stay missing and are not replaced. */
export function resultDifference(
  a: readonly (number | null)[],
  b: readonly (number | null)[],
): { delta: (number | null)[]; pairs: number } {
  const count = Math.min(a.length, b.length);
  const delta: (number | null)[] = [];
  let pairs = 0;
  for (let index = 0; index < count; index += 1) {
    const left = a[index];
    const right = b[index];
    if (typeof left !== "number" || typeof right !== "number") {
      delta.push(null);
      continue;
    }
    pairs += 1;
    delta.push(right - left);
  }
  return { delta, pairs };
}

export function sameScoreDefinition(
  a: ScoreDefinition | null | undefined,
  b: ScoreDefinition | null | undefined,
): boolean {
  if (!a || !b) return false;
  return (
    (a.name ?? "") === (b.name ?? "") &&
    (a.unit ?? "") === (b.unit ?? "") &&
    (a.meaning ?? "") === (b.meaning ?? "") &&
    (a.comparison ?? "") === (b.comparison ?? "") &&
    (a.kind ?? "") === (b.kind ?? "") &&
    (a.name ?? a.unit ?? a.meaning ?? a.kind ?? "") !== ""
  );
}

export function samePoint(
  slice: { row: number; column: number; source_sha256: string } | undefined,
  sha: string,
  row: number,
  column: number,
): boolean {
  return !!slice && slice.source_sha256 === sha && slice.row === row && slice.column === column;
}

type PointSlice = { row: number; column: number; source_sha256: string };

/**
 * Plot a slice only when its point and SHA are the ones just requested.
 * A remembered slice for another point stays as a label and is not plotted.
 * The same point may be plotted from memory when the new read failed.
 */
export function sliceForPoint<T extends PointSlice>(
  fresh: T | undefined,
  remembered: T | undefined,
  sha: string,
  row: number,
  column: number,
): { plot: T | undefined; retained: T | undefined } {
  if (samePoint(fresh, sha, row, column)) return { plot: fresh, retained: undefined };
  if (samePoint(remembered, sha, row, column)) return { plot: remembered, retained: undefined };
  const retained = [remembered, fresh].find(
    (slice) => !!slice && slice.source_sha256 === sha && !samePoint(slice, sha, row, column),
  );
  return { plot: undefined, retained };
}

/** A one-row tile must not include the next observation, including at record end. */
export function sourceTimeBounds(axis: number[], start: number, end: number): [number, number] {
  const first = axis[start], last = axis[end - 1];
  if (start + 1 < end) return [first, last];
  if (start > 0) return [(axis[start - 1] + first) / 2, first];
  return [first, (first + axis[start + 1]) / 2];
}

export function keyForBody(
  previous: Attempt | null,
  body: SubmitBody,
  newKey: () => string,
): string {
  if (previous && sameBody(previous.body, body)) return previous.key;
  return newKey();
}

export function editDraft(
  session: PipelineSession,
  draft: HalfOpenWindow,
): PipelineSession {
  return { ...session, draft };
}

/** Updates the saved revision only. Draft, submitted snapshot, and job id stay as they are. */
export function rememberRevision(session: PipelineSession, revision: number | null): PipelineSession {
  return { ...session, savedRevision: revision };
}

/**
 * A late POST is written only when the URL still shows the submitted record
 * and the job id has not moved since send. The record selected at send time
 * is never written back.
 */
export function applyLateResult(input: {
  urlRecord: string | null;
  submittedRecord: string;
  urlJobId: string | null;
  jobIdAtSend: string | null;
  resultId: string;
}): { record: string | null; jobId: string | null } {
  const record = input.urlRecord;
  if (input.urlRecord !== input.submittedRecord) return { record, jobId: input.urlJobId };
  if (input.urlJobId !== input.jobIdAtSend) return { record, jobId: input.urlJobId };
  return { record, jobId: input.resultId };
}

/**
 * A saved document is written into the URL only while that URL still shows
 * the record that was saved. The record selected at send time is never written back.
 */
export function applySavedSetting(input: {
  urlRecord: string | null;
  submittedRecord: string;
}): { write: boolean } {
  return { write: input.urlRecord !== null && input.urlRecord === input.submittedRecord };
}

/** One explicit send. The helper itself does not call the network. */
export function explicitSubmit(
  session: PipelineSession,
  body: SubmitBody,
  newKey: () => string,
): { session: PipelineSession; post: Attempt } {
  const attempt = { key: keyForBody(session.attempt, body, newKey), body };
  return {
    session: { ...session, submitted: body, attempt },
    post: attempt,
  };
}

/** Status refresh, draft edits, and a lost HTTP response do not start another POST. */
export function followUp(session: PipelineSession): { session: PipelineSession; post: null } {
  return { session, post: null };
}

export function attachJob(
  session: PipelineSession,
  requestBody: SubmitBody,
  job: {
    id: string;
    run_id: string | null;
    source_sha256: string;
    region: HalfOpenWindow & { record_id: string };
  },
): PipelineSession {
  if (!session.submitted || !sameBody(session.submitted, requestBody)) return session;
  const submittedWindow = submitBody(
    session.submitted.region.record_id,
    session.submitted.source_sha256,
    session.submitted.region,
  );
  const jobBody = submitBody(job.region.record_id, job.source_sha256, job.region);
  if (!sameBody(submittedWindow, jobBody)) return session;
  return {
    ...session,
    jobId: job.id,
    shownRunId: job.run_id,
    attempt: null,
  };
}

/** Aborting the browser request leaves the server job untouched. */
export function afterAbort(session: PipelineSession): {
  session: PipelineSession;
  post: null;
  cancelServerJob: false;
} {
  return { session, post: null, cancelServerJob: false };
}

export type PipelineLink = {
  record: string;
  job_id: string;
  block: PipelineBlock;
  window: HalfOpenWindow | null;
  row: number | null;
  column: number | null;
  source_sha256: string;
};

function safeUrl(value: string): URL | null {
  if (
    !value.startsWith("/") ||
    value.startsWith("//") ||
    value.includes("\\") ||
    [...value].some((character) => character.charCodeAt(0) <= 32)
  )
    return null;
  try {
    const url = new URL(value, "https://workbench.invalid");
    if (url.origin !== "https://workbench.invalid" || url.pathname !== "/pipeline")
      return null;
    return url;
  } catch {
    return null;
  }
}

function integer(value: string | null): number | null {
  if (value === null || !/^\d+$/.test(value) || !Number.isSafeInteger(Number(value)))
    return null;
  return Number(value);
}

export function pipelineHref(link: {
  record: string;
  jobId: string;
  block: PipelineBlock;
  window: HalfOpenWindow | null;
  row: number | null;
  column: number | null;
  sourceSha256: string;
}): string {
  const params = new URLSearchParams();
  params.set("record", link.record);
  params.set("job_id", link.jobId);
  params.set("block", link.block);
  if (/^[0-9a-f]{64}$/.test(link.sourceSha256))
    params.set("source_sha256", link.sourceSha256);
  if (link.window) {
    params.set("r0", String(link.window.row_start));
    params.set("r1", String(link.window.row_end));
    params.set("c0", String(link.window.column_start));
    params.set("c1", String(link.window.column_end));
  }
  if (link.row !== null) params.set("row", String(link.row));
  if (link.column !== null) params.set("column", String(link.column));
  return `/pipeline?${params.toString()}`;
}

export function readPipelineLink(value: string): PipelineLink | null {
  const url = safeUrl(value);
  if (!url) return null;
  const job_id = url.searchParams.get("job_id") ?? "";
  if (!/^[A-Za-z0-9_-]{8,128}$/.test(job_id)) return null;
  const blockName = url.searchParams.get("block");
  const block = PIPELINE_BLOCKS.some((item) => item.id === blockName)
    ? (blockName as PipelineBlock)
    : "source";
  const rowStart = integer(url.searchParams.get("r0"));
  const rowEnd = integer(url.searchParams.get("r1"));
  const columnStart = integer(url.searchParams.get("c0"));
  const columnEnd = integer(url.searchParams.get("c1"));
  const window =
    rowStart !== null &&
    rowEnd !== null &&
    columnStart !== null &&
    columnEnd !== null &&
    rowStart < rowEnd &&
    columnStart + 1 < columnEnd
      ? {
          row_start: rowStart,
          row_end: rowEnd,
          column_start: columnStart,
          column_end: columnEnd,
        }
      : null;
  return {
    record: url.searchParams.get("record") ?? "",
    job_id,
    block,
    window,
    row: integer(url.searchParams.get("row")),
    column: integer(url.searchParams.get("column")),
    source_sha256: url.searchParams.get("source_sha256") ?? "",
  };
}
