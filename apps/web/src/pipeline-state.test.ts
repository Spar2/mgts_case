import { describe, expect, it } from "vitest";
import { breakTimeGaps } from "./presentation-model";
import {
  afterAbort,
  applyLateResult,
  applySavedSetting,
  attachJob,
  changedSettingsPatch,
  compareResults,
  editBoundField,
  editDraft,
  explicitSubmit,
  followUp,
  formatWindow,
  limitedWindow,
  parseHalfOpen,
  pipelineHref,
  readPipelineLink,
  rememberRevision,
  resolvePipelineLink,
  restoredRecord,
  absentLayer,
  comparisonCaption,
  savePrompt,
  resultDifference,
  resultName,
  sliceForPoint,
  samePoint,
  sameScoreDefinition,
  PIPELINE_BLOCKS,
  sourceTimeBounds,
  stageText,
  submitBody,
  type PipelineSession,
  type SettingsDraft,
} from "./pipeline-state";
import { STAGES } from "./pipeline-stages";

const shape = [400, 900] as const;
const window = { row_start: 4, row_end: 20, column_start: 6, column_end: 40 };
const sha = "ab".repeat(32);

function session(body = submitBody("5", sha, window)): PipelineSession {
  return {
    draft: window,
    submitted: body,
    attempt: { key: "same-key-1", body },
    jobId: "job-original",
    shownRunId: null,
  };
}

describe("pipeline window, job link, and idempotency", () => {
  it("restores the job source without silently replacing explicit input", () => {
    expect(restoredRecord(null, "job-id", undefined, "6")).toBeNull();
    expect(restoredRecord(null, "job-id", "5", "6")).toBe("5");
    expect(restoredRecord("missing-record", "job-id", "5", "6")).toBe("missing-record");
    expect(restoredRecord(null, "", undefined, "6")).toBe("6");
  });

  it("never treats a previous point or source as the selected slice", () => {
    const value = { row: 7, column: 9, source_sha256: sha };
    expect(samePoint(value, sha, 7, 9)).toBe(true);
    expect(samePoint(value, sha, 8, 9)).toBe(false);
    expect(samePoint(value, sha, 7, 10)).toBe(false);
    expect(samePoint(value, "cd".repeat(32), 7, 9)).toBe(false);
    expect(samePoint(undefined, sha, 7, 9)).toBe(false);
  });

  it("bounds a single native row without borrowing a neighbour or exceeding the record", () => {
    const axis = [0, 0.5, 100, 101];
    for (let row = 0; row < axis.length; row++) {
      const [lo, hi] = sourceTimeBounds(axis, row, row + 1);
      expect(lo).toBeLessThan(hi);
      expect(axis.filter((value) => value >= lo && value <= hi)).toEqual([axis[row]]);
      expect(lo).toBeGreaterThanOrEqual(axis[0]);
      expect(hi).toBeLessThanOrEqual(axis.at(-1)!);
    }
    expect(sourceTimeBounds(axis, 1, 3)).toEqual([0.5, 100]);
  });
  it("round-trips a half-open index window", () => {
    expect(parseHalfOpen(formatWindow(window), shape)).toEqual(window);
    expect(parseHalfOpen(["4", "20", "6", "7"], shape)).toBeNull();
    expect(parseHalfOpen(["4", "4", "6", "40"], shape)).toBeNull();
    const limited = limitedWindow([5000, 8000]);
    expect(limited).toEqual({
      row_start: 0,
      row_end: 32,
      column_start: 0,
      column_end: 64,
    });
    expect(limitedWindow([10, 1])).toBeNull();
  });

  it("restores job_id only from a safe internal link", () => {
    const href = pipelineHref({
      record: "5",
      jobId: "a".repeat(32),
      block: "clean",
      window,
      row: 5,
      column: 8,
      sourceSha256: sha,
    });
    const link = readPipelineLink(href);
    expect(href.startsWith("/pipeline?")).toBe(true);
    expect(link?.job_id).toBe("a".repeat(32));
    expect(link?.window).toEqual(window);
    expect(link?.block).toBe("clean");
    expect(readPipelineLink(`https://evil.example${href}`)).toBeNull();
    expect(readPipelineLink(`//evil.example${href}`)).toBeNull();
    expect(readPipelineLink(`/pipeline?job_id=${"a".repeat(32)}\n`)).toBeNull();
  });

  it("keeps the submitted body when the draft window changes", () => {
    const current = session();
    const edited = editDraft(current, { ...window, row_end: 28 });
    expect(edited.draft.row_end).toBe(28);
    expect(edited.submitted).toEqual(current.submitted);
    expect(edited.attempt).toEqual(current.attempt);
    expect(edited.jobId).toBe(current.jobId);
    expect(followUp(edited).post).toBeNull();
  });

  it("reuses the idempotency key for the same body and replaces it when the body changes", () => {
    const current = session();
    const body = current.submitted!;
    const again = explicitSubmit(current, body, () => "fresh-key");
    expect(again.post.key).toBe("same-key-1");
    expect(again.post.body).toEqual(body);
    const moved = submitBody("5", sha, { ...window, column_end: 48 });
    const changed = explicitSubmit(current, moved, () => "fresh-key");
    expect(changed.post.key).toBe("fresh-key");
    expect(changed.post.body).toEqual(moved);
    expect(followUp(changed.session).post).toBeNull();
    expect(afterAbort(changed.session).post).toBeNull();
    expect(afterAbort(changed.session).cancelServerJob).toBe(false);
  });

  it("does not attach a late response to a different window", () => {
    const current = session();
    const other = submitBody("5", sha, { ...window, row_end: 30 });
    const ignored = attachJob(current, other, {
      id: "late-job",
      run_id: "late-run",
      source_sha256: sha,
      region: other.region,
    });
    expect(ignored).toEqual(current);
    const accepted = attachJob(current, current.submitted!, {
      id: "saved-job",
      run_id: "saved-run",
      source_sha256: sha,
      region: { record_id: "5", ...window },
    });
    expect(accepted.jobId).toBe("saved-job");
    expect(accepted.shownRunId).toBe("saved-run");
    expect(accepted.attempt).toBeNull();
    expect(stageText("computing")).toBe("Расчёт");
    expect(stageText("61%")).toBe("61%");
  });
});

const settingDraft: SettingsDraft = {
  title: "Окно записи 5",
  record_id: "5",
  source_sha256: sha,
  window,
  point: { row: 4, column: 6 },
  method: "median_x3+spatial_z_halo",
  method_version: "237168f4d4a3e11260c9f692880cc477cd7af779",
  run_ids: [],
};

describe("saved settings, late navigation, and comparison", () => {
  it("round-trips four bound fields and keeps the other three when one changes", () => {
    const draft = formatWindow(window);
    expect(parseHalfOpen(draft, shape)).toEqual(window);
    const edited = editBoundField(draft, 2, "9");
    expect(edited).toEqual(["4", "20", "9", "40"]);
    expect(edited[0]).toBe(draft[0]);
    expect(edited[1]).toBe(draft[1]);
    expect(edited[3]).toBe(draft[3]);
    expect(parseHalfOpen(edited, shape)?.column_start).toBe(9);
  });

  it("applies a late result only while the URL record is still the submitted record", () => {
    expect(
      applyLateResult({
        urlRecord: "5",
        submittedRecord: "5",
        urlJobId: "old-job",
        jobIdAtSend: "old-job",
        resultId: "new-job",
      }),
    ).toEqual({ record: "5", jobId: "new-job" });
    const moved = applyLateResult({
      urlRecord: "6",
      submittedRecord: "5",
      urlJobId: null,
      jobIdAtSend: null,
      resultId: "new-job",
    });
    expect(moved).toEqual({ record: "6", jobId: null });
    expect(moved.record).not.toBe("5");
  });

  it("does not write a saved record back after the archive selection moved", () => {
    expect(applySavedSetting({ urlRecord: "6", submittedRecord: "6" })).toEqual({ write: true });
    const moved = applySavedSetting({ urlRecord: "5", submittedRecord: "6" });
    expect(moved).toEqual({ write: false });
    expect(moved).not.toHaveProperty("record");
    expect(applySavedSetting({ urlRecord: null, submittedRecord: "6" }).write).toBe(false);
  });

  it("restores a job-only link and does not replace an explicit error", () => {
    const job = { record_id: "6", source_sha256: sha, region: window };
    const empty = [null, null, null, null] as const;
    expect(
      resolvePipelineLink({
        record: null,
        bounds: empty,
        shape: null,
        knownRecords: ["6"],
        jobId: "a".repeat(32),
        job: null,
        settingPending: false,
        setting: null,
        firstRecord: "5",
      }),
    ).toEqual({ action: "wait" });
    expect(
      resolvePipelineLink({
        record: null,
        bounds: empty,
        shape: null,
        knownRecords: ["6"],
        jobId: "a".repeat(32),
        job,
        settingPending: false,
        setting: {
          record_id: "5",
          source_sha256: sha,
          window: { ...window, row_end: 8 },
          point: { row: 0, column: 0 },
        },
        firstRecord: "5",
      }),
    ).toEqual({ action: "fill", record: "6", source_sha256: sha, window, point: null });
    expect(
      resolvePipelineLink({
        record: "missing-record",
        bounds: empty,
        shape: null,
        knownRecords: ["6"],
        jobId: "a".repeat(32),
        job,
        settingPending: false,
        setting: null,
        firstRecord: "6",
      }),
    ).toEqual({ action: "keep", error: "unknown-record" });
    const invalid = resolvePipelineLink({
      record: "6",
      bounds: ["4", "4", "6", "40"],
      shape,
      knownRecords: ["6"],
      jobId: "a".repeat(32),
      job,
      settingPending: false,
      setting: null,
      firstRecord: "5",
    });
    expect(invalid).toEqual({ action: "keep", error: "invalid-bounds" });
    expect(invalid).not.toHaveProperty("window");
  });

  it("reuses the idempotency key only for an unchanged body and never posts by itself", () => {
    const current = session();
    const again = explicitSubmit(current, current.submitted!, () => "fresh-key");
    expect(again.post.key).toBe("same-key-1");
    const revised = submitBody("5", sha, window, { id: "setting-1", revision: 2 });
    const changed = explicitSubmit(current, revised, () => "fresh-key");
    expect(changed.post.key).toBe("fresh-key");
    expect(changed.post.body.settings_revision).toBe(2);
    expect(followUp(changed.session).post).toBeNull();
    expect(afterAbort(changed.session).post).toBeNull();
    expect(afterAbort(changed.session).cancelServerJob).toBe(false);
  });

  it("names a result by record, window, and time", () => {
    const named = resultName({
      recordId: "6",
      window,
      calculatedAt: "2026-10-15T16:40:00+00:00",
    });
    expect(named).toBe("запись 6, строки [4, 20), столбцы [6, 40), 15.10.2026 16:40 UTC");
    expect(named).not.toMatch(/[0-9a-f]{12}/);
    expect(resultName({ recordId: "6", window, calculatedAt: "не время" })).toContain("не время");
    expect(resultName({ recordId: "6", window, calculatedAt: 0 })).toContain("01.01.1970 00:00 UTC");
    expect(resultName({ recordId: "6", window, calculatedAt: null })).toContain("время не передано");
    expect(comparisonCaption()).toEqual({
      direction: "результат B − результат A",
      winner: null,
    });
    expect(absentLayer(false).reason).toContain("не заменяют");
    expect(absentLayer(true).reason).toBeNull();
  });

  it("breaks a pipeline time series only when the gap is longer than 300 s", () => {
    expect(breakTimeGaps([0, 301], [20, 21])).toEqual({
      x: [0, 150.5, 301],
      y: [20, null, 21],
    });
    expect(breakTimeGaps([0, 300], [20, 21])).toEqual({
      x: [0, 300],
      y: [20, 21],
    });
  });

  it("does not call a stored calculation an unsaved window", () => {
    expect(
      savePrompt({ savedTitle: null, draftMatches: false, jobOnScreen: false }),
    ).toBe("Это окно ещё не сохранено.");
    const stored = savePrompt({ savedTitle: null, draftMatches: false, jobOnScreen: true });
    expect(stored).toContain("уже хранится");
    expect(stored).not.toContain("ещё не сохранено");
    expect(
      savePrompt({
        savedTitle: null,
        draftMatches: false,
        jobOnScreen: true,
        ownerTitles: ["Окно проверки"],
      }),
    ).toContain("Окно проверки");
    expect(
      savePrompt({
        savedTitle: null,
        draftMatches: false,
        jobOnScreen: true,
        ownerTitles: ["Первое", "Второе"],
      }),
    ).toContain("несколько");
    expect(
      savePrompt({
        savedTitle: "Окно проверки",
        draftMatches: true,
        authorName: "Анна",
        jobOnScreen: true,
      }),
    ).toBe("Сохранено «Окно проверки» · Анна.");
    expect(
      savePrompt({ savedTitle: "  ", draftMatches: true, jobOnScreen: false }),
    ).toContain("без названия");
    expect(
      savePrompt({
        savedTitle: "Окно проверки",
        draftMatches: false,
        jobOnScreen: true,
      }),
    ).toContain("не меняется");
    expect(
      savePrompt({
        savedTitle: null,
        draftMatches: false,
        jobOnScreen: true,
        loadingSelection: true,
      }),
    ).toBe("Загрузка сохранённого окна…");
  });

  it("accepts only a non-empty intersection of the same record and SHA", () => {
    const left = { id: "run-a", record_id: "5", source_sha256: sha, window };
    const right = {
      id: "run-b",
      record_id: "5",
      source_sha256: sha,
      window: { row_start: 10, row_end: 24, column_start: 8, column_end: 30 },
    };
    expect(compareResults(left, right)).toEqual({
      window: { row_start: 10, row_end: 20, column_start: 8, column_end: 30 },
    });
    for (const denied of [
      compareResults(left, { ...right, window: { ...window, row_start: 20 } }),
      compareResults(left, { ...right, source_sha256: "cd".repeat(32) }),
      compareResults(left, null),
      compareResults(null, right),
    ]) {
      expect(denied).toHaveProperty("reason");
      expect(typeof (denied as { reason: string }).reason).toBe("string");
      expect((denied as { reason: string }).reason.length).toBeGreaterThan(0);
      expect(denied).not.toHaveProperty("window");
    }
    expect(resultDifference([1, null, 2], [4, 5, null])).toEqual({ delta: [3, null, null], pairs: 1 });
    expect(sameScoreDefinition({ name: "z", unit: "1" }, { name: "z", unit: "1" })).toBe(true);
    expect(sameScoreDefinition({ name: "z", unit: "1" }, { name: "other", unit: "1" })).toBe(false);
  });

  it("keeps saved revision, draft window, submitted snapshot, and job id distinct", () => {
    const current = { ...session(), savedRevision: 4 };
    const edited = editDraft(current, { ...window, row_end: 28 });
    expect(edited.draft.row_end).toBe(28);
    expect(edited.savedRevision).toBe(4);
    expect(edited.submitted).toEqual(current.submitted);
    expect(edited.jobId).toBe("job-original");
    expect(edited.submitted?.region.row_end).toBe(20);
    const noted = rememberRevision(edited, 5);
    expect(noted.savedRevision).toBe(5);
    expect(noted.draft).toEqual(edited.draft);
    expect(noted.submitted).toEqual(edited.submitted);
    expect(noted.jobId).toBe(edited.jobId);
    expect(changedSettingsPatch(settingDraft, 4, settingDraft)).toBeNull();
    expect(changedSettingsPatch(settingDraft, 4, { ...settingDraft, title: "Другое имя" })).toEqual({
      expected_revision: 4,
      title: "Другое имя",
    });
  });

  it("does not plot a failed point as the previous slice", () => {
    const previous = { row: 4, column: 6, source_sha256: sha, profile: [1] };
    const failed = sliceForPoint(undefined, previous, sha, 9, 6);
    expect(failed.plot).toBeUndefined();
    expect(failed.retained).toBe(previous);
    expect(failed.retained?.row).toBe(4);
    const otherSha = sliceForPoint(undefined, previous, "cd".repeat(32), 9, 6);
    expect(otherSha.plot).toBeUndefined();
    expect(otherSha.retained).toBeUndefined();
    const same = sliceForPoint(undefined, previous, sha, 4, 6);
    expect(same.plot).toBe(previous);
    expect(same.retained).toBeUndefined();
    const arrived = { row: 9, column: 6, source_sha256: sha, profile: [2] };
    expect(sliceForPoint(arrived, previous, sha, 9, 6).plot).toBe(arrived);
  });

  it("attaches a settings-linked job and still rejects a different window", () => {
    const body = submitBody("5", sha, window, { id: "a".repeat(32), revision: 2 });
    const current = explicitSubmit(session(), body, () => "settings-key").session;
    const accepted = attachJob(current, body, {
      id: "job-settings",
      run_id: null,
      source_sha256: sha,
      region: { record_id: "5", ...window },
    });
    expect(accepted.jobId).toBe("job-settings");
    expect(accepted.attempt).toBeNull();
    expect(accepted.submitted).toEqual(body);
    const moved = attachJob(current, body, {
      id: "other-job",
      run_id: null,
      source_sha256: sha,
      region: { record_id: "5", ...window, row_end: window.row_end + 1 },
    });
    expect(moved.jobId).toBe(current.jobId);
    expect(moved.attempt).toEqual(current.attempt);
  });
});

describe("pipeline block and stage labels", () => {
  it("does not reuse a stage label or chip on a result block", () => {
    const stageLabels = STAGES.map((stage) => stage.label);
    const stageChips = STAGES.map((stage) => stage.chip);
    for (const block of PIPELINE_BLOCKS) {
      expect(stageLabels).not.toContain(block.label);
      expect(stageChips).not.toContain(block.label);
    }
    expect(PIPELINE_BLOCKS.find((block) => block.id === "clean")?.label).toBe("Поле обработки");
    expect(stageLabels).toContain("Очистка");
  });
});
