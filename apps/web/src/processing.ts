import type { components } from "./api-schema";

export type ProcessingRun = components["schemas"]["RunDetail"];
type Region = components["schemas"]["Region"];

export function processedProfile(
  run: ProcessingRun,
  region: Region,
  row: number,
  sha: string,
  raw: (number | null)[],
) {
  const r = run.region;
  const values =
    run.result && typeof run.result === "object" && "denoised" in run.result
      ? (run.result as { denoised?: unknown }).denoised
      : undefined;
  if (
    !["complete", "imported"].includes(run.status) ||
    run.source_sha256 !== sha ||
    r.record_id !== region.record_id ||
    r.row_start !== row ||
    r.row_end !== row + 1 ||
    r.column_start > region.column_start ||
    r.column_end < region.column_end ||
    r.column_end > raw.length ||
    !Array.isArray(values) ||
    values.length !== raw.length ||
    values.some(
      (v) => v !== null && (typeof v !== "number" || !Number.isFinite(v)),
    )
  ) {
    throw new Error(
      "Результат несовместим с источником, профилем или областью. Другой результат не подставлен.",
    );
  }
  return values.map((v: number | null, i: number) =>
    raw[i] === null ? null : v,
  );
}

export function processingKey(
  region: Region,
  row: number,
  method: string,
  sha?: string,
) {
  return JSON.stringify([region, row, method, sha]);
}
