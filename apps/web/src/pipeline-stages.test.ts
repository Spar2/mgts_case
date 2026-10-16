import { describe, expect, it } from "vitest";
import {
  STAGES,
  checkStages,
  readStoredSource,
  stageStatus,
  stageStorageKey,
  writeStoredSource,
  type StageId,
} from "./pipeline-stages";

function sources(): Record<StageId, string> {
  return Object.fromEntries(STAGES.map((stage) => [stage.id, stage.source])) as Record<StageId, string>;
}

function memory(): Storage {
  const items = new Map<string, string>();
  return {
    get length() {
      return items.size;
    },
    clear() {
      items.clear();
    },
    getItem(key) {
      return items.get(key) ?? null;
    },
    key(index) {
      return [...items.keys()][index] ?? null;
    },
    removeItem(key) {
      items.delete(key);
    },
    setItem(key, value) {
      items.set(key, value);
    },
  };
}

describe("checkStages", () => {
  it("stops on the default clean line 14", () => {
    const error = checkStages(sources());
    expect(error?.stageId).toBe("clean");
    expect(error?.label).toBe("Очистка");
    expect(error?.line).toBe(14);
    expect(error?.message).toContain("отступ");
    expect(STAGES.find((stage) => stage.id === "clean")?.source.split("\n")[13]).toBe("return None");
  });

  it("accepts the five defaults after the clean return is indented", () => {
    const next = sources();
    next.clean = next.clean.replace("\nreturn None", "\n        return None");
    expect(checkStages(next)).toBeNull();
  });

  it("reports a tab indent and does not reach a later broken stage", () => {
    const next = sources();
    next.preprocess = "def prepare(values):\n\treturn values\n";
    next.anomaly = "def mark(values):\n    return (values\n";
    const error = checkStages(next);
    expect(error?.stageId).toBe("preprocess");
    expect(error?.message).toBe("табуляция вместо отступа");
    expect(error?.line).toBe(2);
  });

  it("reports an unclosed parenthesis on the last line and skips empty stages", () => {
    const next = sources();
    next.preprocess = " \n\n";
    next.clean = "   ";
    next.anomaly = "def mark(values):\n    return (values";
    const error = checkStages(next);
    expect(error?.stageId).toBe("anomaly");
    expect(error?.line).toBe(2);
    expect(error?.message).toBe("скобка не закрыта");
  });
});

describe("stageStatus", () => {
  const error = { stageId: "clean" as const, label: "Очистка", line: 14, message: "после условия ожидался отступ" };

  it("names empty, error, saved and draft", () => {
    expect(stageStatus({ id: "vision", source: " \n", savedSource: null, error })).toBe("empty");
    expect(stageStatus({ id: "clean", source: "return None", savedSource: "return None", error })).toBe("error");
    expect(stageStatus({ id: "preprocess", source: "value = 1", savedSource: "value = 1", error })).toBe("saved");
    expect(stageStatus({ id: "anomaly", source: "value = 1", savedSource: null, error })).toBe("draft");
  });
});

describe("stage storage", () => {
  it("round-trips one tab and leaves the others", () => {
    const storage = memory();
    expect(stageStorageKey("6", "clean")).toBe("dts-pipeline-stage:6:clean");
    expect(readStoredSource("6", "clean", storage)).toBeNull();
    writeStoredSource("6", "clean", "return None\n", storage);
    expect(readStoredSource("6", "clean", storage)).toBe("return None\n");
    expect(readStoredSource("6", "anomaly", storage)).toBeNull();
    expect(readStoredSource("7", "clean", storage)).toBeNull();
  });
});
