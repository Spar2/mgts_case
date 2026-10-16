import { describe, expect, it } from "vitest";
import {
  breakTimeGaps,
  compareProfile,
  qualityLabel,
  signalSummary,
} from "./presentation-model";

describe("rehearsal measurements", () => {
  it("finds a native peak without replacing missing observations", () => {
    expect(signalSummary([null, -2, 4, 4, NaN])).toMatchObject({
      total: 5,
      valid: 3,
      missing: 2,
      peak: 2,
      min: -2,
      max: 4,
      range: 6,
    });
    expect(signalSummary([null, NaN])).toMatchObject({
      peak: -1,
      min: null,
      max: null,
      mean: null,
    });
  });
  it("keeps filled gaps masked and uses processed minus raw", () => {
    const comparison = compareProfile(
      [10, 12, 19, 25],
      [2, null, 5, 7],
      [3, 10, null, 5],
    )!;
    expect(comparison.clean).toEqual([3, null, null, 5]);
    expect(comparison.delta).toEqual([1, null, null, -2]);
    expect(comparison.stats).toMatchObject({
      paired_points: 2,
      missing_original: 1,
      missing_processed: 1,
      mean_absolute_change_c: 1.5,
    });
    expect(comparison.stats.top_points[0].x_m).toBe(25);
    expect(comparison.deltaLimits[0]).toBe(-comparison.deltaLimits[1]);
  });
  it("rejects incompatible profiles and retains an all-missing result as unmeasurable", () => {
    expect(compareProfile([1, 2], [1], [1])).toBeNull();
    expect(compareProfile([1], [null], [3])?.stats.paired_points).toBe(0);
    expect(compareProfile([1], [2])).toBeNull();
  });
  it("decodes combined quality reasons", () => {
    expect(qualityLabel(3)).toBe("Sentinel −300; Нечисловое значение");
    expect(qualityLabel(8)).toBe("Код качества 8");
  });
  it("shows time gaps without inventing a temperature or moving original times", () => {
    expect(breakTimeGaps([0, 10, 1000], [2, null, 4])).toEqual({
      x: [0, 10, 505, 1000],
      y: [2, null, null, 4],
    });
    expect(breakTimeGaps([0, 300], [2, 4]).x).toEqual([0, 300]);
  });
});
