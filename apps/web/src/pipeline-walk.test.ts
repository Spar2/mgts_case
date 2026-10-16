import { describe, expect, it } from "vitest";
import {
  PIPELINE_STAGES,
  REDUCED_CAPTION,
  STAGE_COPY,
  STAGE_MS,
  cellCombined,
  pipelineFrame,
  revealCount,
  zoneSpans,
} from "./pipeline-frame";

const BANNED = ["0.8580", "0.8608", "FAR", "Local_Heating", "Extended_Heating", "Contact_Trace", "Local_Cooling", "Extended_Cooling", "HVAC_Diurnal", "Sensor_Artifact", "точность", "жюри"];

describe("pipeline walk timeline", () => {
  it("walks the plate stages in order from elapsed time", () => {
    const order = PIPELINE_STAGES.map((_, index) => pipelineFrame(index * STAGE_MS + STAGE_MS / 2, false).stage);
    expect(order).toEqual([...PIPELINE_STAGES]);
    expect(pipelineFrame(0, false).stage).toBe("trace");
    expect(pipelineFrame(STAGE_MS, false).stage).toBe("channels");
    expect(pipelineFrame(PIPELINE_STAGES.length * STAGE_MS, false).stage).toBe("trace");
  });

  it("reveals the current stage and keeps earlier stages complete", () => {
    const midDetectors = pipelineFrame(2 * STAGE_MS + 1, false);
    expect(midDetectors.stage).toBe("detectors");
    expect(revealCount(midDetectors, 0, 32)).toBe(32);
    expect(revealCount(midDetectors, 1, 3)).toBe(3);
    expect(revealCount(midDetectors, 3, 32)).toBe(0);
    const early = pipelineFrame(STAGE_MS / 2, false);
    expect(revealCount(early, 0, 32)).toBeGreaterThan(0);
    expect(revealCount(early, 0, 32)).toBeLessThan(32);
  });

  it("holds one complete frame when motion is reduced", () => {
    const atStart = pipelineFrame(0, true);
    const later = pipelineFrame(3 * STAGE_MS + 400, true);
    expect(later).toEqual(atStart);
    expect(atStart.stage).toBe("zones");
    expect(atStart.revealed).toBe(PIPELINE_STAGES.length);
    expect(atStart.reducedMotion).toBe(true);
    expect(revealCount(atStart, 0, 32)).toBe(32);
    expect(revealCount(atStart, 4, zoneSpans().length)).toBe(zoneSpans().length);
  });

  it("does not speak plate scores or event names", () => {
    const text = [...STAGE_COPY.map((step) => `${step.name} ${step.text}`), REDUCED_CAPTION].join("\n");
    for (const word of BANNED) expect(text).not.toContain(word);
    expect(zoneSpans().length).toBeGreaterThan(0);
    expect(CELLS_COMBINED()).toBeGreaterThan(0);
  });
});

function CELLS_COMBINED() {
  return Array.from({ length: 32 }, (_, index) => index).filter(cellCombined).length;
}
