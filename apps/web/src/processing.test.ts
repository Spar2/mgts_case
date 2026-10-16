import { describe, expect, it } from "vitest";
import {
  processedProfile,
  processingKey,
  type ProcessingRun,
} from "./processing";

const region = {
  record_id: "5",
  row_start: 0,
  row_end: 2,
  column_start: 0,
  column_end: 3,
};
const run: ProcessingRun = {
  id: "selected",
  has_result: true,
  status: "imported",
  method: "fixture",
  method_version: "contract-v1",
  parameters: {},
  limitations: "Test fixture",
  training_origin: "untrained",
  updated_at: 0,
  source_sha256: "sha",
  region: { ...region, row_end: 1 },
  result: { denoised: [10, 11, null] },
};
describe("explicit processing selection", () => {
  it("retains both native and method masks", () => {
    expect(processedProfile(run, region, 0, "sha", [1, null, 3])).toEqual([
      10,
      null,
      null,
    ]);
  });
  it("rejects incompatible results rather than picking another", () => {
    for (const changed of [
      { source_sha256: "other" },
      { status: "running" },
      { region: { ...run.region, row_start: 1, row_end: 2 } },
      { result: { denoised: [1, 2] } },
      { result: { denoised: [1, NaN, 3] } },
      { result: { denoised: [true, 2, 3] } },
    ]) {
      expect(() =>
        processedProfile(
          { ...run, ...changed } as ProcessingRun,
          region,
          0,
          "sha",
          [1, 2, 3],
        ),
      ).toThrow();
    }
  });
  it("invalidates selection for every context change", () => {
    const key = processingKey(region, 0, "median_x3", "sha");
    for (const next of [
      processingKey(region, 1, "median_x3", "sha"),
      processingKey(region, 0, "median_x5", "sha"),
      processingKey({ ...region, column_start: 1 }, 0, "median_x3", "sha"),
      processingKey(region, 0, "median_x3", "new"),
    ])
      expect(next).not.toBe(key);
  });
});
