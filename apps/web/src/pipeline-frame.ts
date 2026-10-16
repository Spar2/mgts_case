/** Illustrative reading of the ens_gen2 plate. Not a measurement and not a score. */

export const PIPELINE_STAGES = ["trace", "channels", "detectors", "combine", "zones"] as const;
export type PipelineStage = (typeof PIPELINE_STAGES)[number];

export const STAGE_MS = 2800;
export const LOOP_MS = PIPELINE_STAGES.length * STAGE_MS;
export const TRACE_CELLS = 32;

export const STAGE_COPY: readonly { id: PipelineStage; kicker: string; name: string; text: string }[] = [
  { id: "trace", kicker: "01", name: "След", text: "Температура вдоль волокна остаётся измерением." },
  { id: "channels", kicker: "02", name: "Каналы", text: "Из следа собирают изменение во времени, изменение вдоль волокна и маску годных ячеек." },
  { id: "detectors", kicker: "03", name: "Детекторы", text: "Четыре детектора читают одни и те же каналы, каждый — своё окно." },
  { id: "combine", kicker: "04", name: "Сложение", text: "Там, где окно встречает тёплый след, клетки складываются в одну карту." },
  { id: "zones", kicker: "05", name: "Зоны", text: "Связные участки этой карты остаются зонами для чтения. Тип события не назван." },
];

export const REDUCED_CAPTION = "Схема целиком: след, три канала, четыре детектора, одна карта и зоны для чтения. Тип события не назван.";

const WARM = new Set([7, 8, 9, 18, 19, 20, 21, 22]);
const HOT = new Set([19, 20, 21]);
export const MASK_GAP = new Set([4, 5]);

export const DETECTORS = [
  { id: "A", from: 6, to: 12 },
  { id: "B", from: 16, to: 24 },
  { id: "C", from: 18, to: 22 },
  { id: "D", from: 0, to: 8 },
] as const;

export type PipelineFrame = {
  stage: PipelineStage;
  index: number;
  progress: number;
  revealed: number;
  reducedMotion: boolean;
};

export function pipelineFrame(elapsedMs: number, reducedMotion: boolean): PipelineFrame {
  if (reducedMotion || !Number.isFinite(elapsedMs)) {
    return {
      stage: "zones",
      index: PIPELINE_STAGES.length - 1,
      progress: 1,
      revealed: PIPELINE_STAGES.length,
      reducedMotion: true,
    };
  }
  const t = ((elapsedMs % LOOP_MS) + LOOP_MS) % LOOP_MS;
  const index = Math.min(PIPELINE_STAGES.length - 1, Math.floor(t / STAGE_MS));
  const progress = (t - index * STAGE_MS) / STAGE_MS;
  return {
    stage: PIPELINE_STAGES[index],
    index,
    progress,
    revealed: index + 1,
    reducedMotion: false,
  };
}

/** How many items of a stage are drawn. Earlier stages stay complete; later stages stay empty. */
export function revealCount(frame: PipelineFrame, stageIndex: number, total: number): number {
  if (total <= 0) return 0;
  if (frame.reducedMotion || frame.index > stageIndex) return total;
  if (frame.index < stageIndex || frame.progress <= 0) return 0;
  return Math.max(0, Math.min(total, Math.ceil(frame.progress * total - 1e-9)));
}

export function traceLevel(index: number): 0 | 1 | 2 {
  if (HOT.has(index)) return 2;
  if (WARM.has(index)) return 1;
  return 0;
}

export type ChannelKind = "time" | "space" | "mask";

export const CHANNELS: readonly { kind: ChannelKind; label: string }[] = [
  { kind: "time", label: "Во времени" },
  { kind: "space", label: "Вдоль волокна" },
  { kind: "mask", label: "Годные ячейки" },
];

export function channelLevel(kind: ChannelKind, index: number): 0 | 1 | 2 {
  if (kind === "mask") return MASK_GAP.has(index) ? 0 : 1;
  if (kind === "time") return traceLevel(index);
  const here = traceLevel(index);
  const prev = index === 0 ? 0 : traceLevel(index - 1);
  const next = index === TRACE_CELLS - 1 ? 0 : traceLevel(index + 1);
  return here !== prev || here !== next ? 2 : 0;
}

export function cellCombined(index: number): boolean {
  if (traceLevel(index) === 0) return false;
  return DETECTORS.some((detector) => index >= detector.from && index < detector.to);
}

export function zoneSpans(): { from: number; to: number }[] {
  const spans: { from: number; to: number }[] = [];
  let from = -1;
  for (let index = 0; index <= TRACE_CELLS; index += 1) {
    const on = index < TRACE_CELLS && cellCombined(index);
    if (on && from < 0) from = index;
    if (!on && from >= 0) {
      spans.push({ from, to: index });
      from = -1;
    }
  }
  return spans;
}
