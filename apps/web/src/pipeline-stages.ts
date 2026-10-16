export type StageId = "preprocess" | "clean" | "anomaly" | "classify" | "vision";
export type StageStatus = "saved" | "error" | "draft" | "empty";
export type StageSpec = { id: StageId; label: string; chip: string; source: string };
export type StageError = { stageId: StageId; label: string; line: number; message: string };

const CLEAN_SOURCE = [
  "def clean_window(values):",
  '    """Окно температуры волокна.',
  "",
  "    −300 остаётся признаком пропуска.",
  "    Алгоритм не подставляет измерение.",
  '    """',
  "",
  "    ready = []",
  "    for item in values:",
  "        if item is None or item == -300:",
  "            continue",
  "        ready.append(item)",
  "    if not ready:",
  "return None",
].join("\n");

export const STAGES: readonly StageSpec[] = [
  {
    id: "preprocess",
    label: "Предобработка данных",
    chip: "Предобработка",
    source: [
      "def prepare(values):",
      "    return [item for item in values if item is not None]",
    ].join("\n"),
  },
  {
    id: "clean",
    label: "Очистка",
    chip: "Очистка",
    source: CLEAN_SOURCE,
  },
  {
    id: "anomaly",
    label: "Детектор аномалий",
    chip: "Детектор",
    source: [
      "def mark(values):",
      "    if not values:",
      "        return []",
      "    return values",
    ].join("\n"),
  },
  {
    id: "classify",
    label: "Модель классификации",
    chip: "Классификация",
    source: [
      "def label(values):",
      '    return "обычный" if values else "пусто"',
    ].join("\n"),
  },
  {
    id: "vision",
    label: "Компьютерное зрение",
    chip: "Зрение",
    source: "",
  },
];

export function checkStages(sources: Record<StageId, string>): StageError | null {
  for (const stage of STAGES) {
    const error = checkSource(stage, sources[stage.id] ?? "");
    if (error) return error;
  }
  return null;
}

export function stageStatus(input: {
  source: string;
  savedSource: string | null;
  error: StageError | null;
  id: StageId;
}): StageStatus {
  if (!input.source.trim()) return "empty";
  if (input.error?.stageId === input.id) return "error";
  if (input.savedSource === input.source) return "saved";
  return "draft";
}

export function stageStorageKey(recordId: string, id: StageId): string {
  return `dts-pipeline-stage:${recordId}:${id}`;
}

export function readStoredSource(recordId: string, id: StageId, storage: Storage): string | null {
  return storage.getItem(stageStorageKey(recordId, id));
}

export function writeStoredSource(
  recordId: string,
  id: StageId,
  source: string,
  storage: Storage,
): void {
  storage.setItem(stageStorageKey(recordId, id), source);
}

function checkSource(stage: StageSpec, source: string): StageError | null {
  if (!source.trim()) return null;
  const lines = source.split("\n");
  let expectIndent: number | null = null;
  let round = 0;
  let square = 0;
  let curly = 0;
  for (let index = 0; index < lines.length; index += 1) {
    const raw = lines[index] ?? "";
    const line = index + 1;
    const indentText = /^\s*/.exec(raw)?.[0] ?? "";
    if (indentText.includes("\t")) {
      return problem(stage, line, "табуляция вместо отступа");
    }
    const code = codeOf(raw);
    if (!code) continue;
    if (expectIndent !== null && indentText.length <= expectIndent) {
      return problem(stage, line, "после условия ожидался отступ");
    }
    expectIndent = null;
    for (const char of code) {
      if (char === "(") round += 1;
      else if (char === ")") round -= 1;
      else if (char === "[") square += 1;
      else if (char === "]") square -= 1;
      else if (char === "{") curly += 1;
      else if (char === "}") curly -= 1;
    }
    if (code.endsWith(":")) expectIndent = indentText.length;
  }
  if (round !== 0 || square !== 0 || curly !== 0) {
    return problem(stage, lines.length, "скобка не закрыта");
  }
  return null;
}

function codeOf(raw: string): string {
  let quote: "'" | '"' | null = null;
  let out = "";
  for (let index = 0; index < raw.length; index += 1) {
    const char = raw[index];
    if (quote) {
      if (char === "\\") {
        index += 1;
        continue;
      }
      if (char === quote) quote = null;
      continue;
    }
    if (char === "#") break;
    if (char === "'" || char === '"') {
      quote = char;
      continue;
    }
    out += char;
  }
  return out.trim();
}

function problem(stage: StageSpec, line: number, message: string): StageError {
  return { stageId: stage.id, label: stage.label, line, message };
}
