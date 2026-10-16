import { profileChanges } from "./profile-changes";

export function signalSummary(values: (number | null)[]) {
  let min = Infinity,
    max = -Infinity,
    sum = 0,
    valid = 0,
    peak = -1;
  values.forEach((value, index) => {
    if (value === null || !Number.isFinite(value)) return;
    if (value > max) {
      max = value;
      peak = index;
    }
    min = Math.min(min, value);
    sum += value;
    valid++;
  });
  return {
    total: values.length,
    valid,
    missing: values.length - valid,
    peak,
    min: valid ? min : null,
    max: valid ? max : null,
    mean: valid ? sum / valid : null,
    range: valid ? max - min : null,
  };
}

/** A method cannot turn an invalid measurement into an observed temperature. */
export function compareProfile(
  x: number[],
  raw: (number | null)[],
  processed?: (number | null)[],
) {
  if (!processed || x.length !== raw.length || processed.length !== raw.length)
    return null;
  const original = raw.map((v) =>
    v !== null && Number.isFinite(v) ? v : null,
  );
  const clean = processed.map((v, i) =>
    original[i] !== null && v !== null && Number.isFinite(v) ? v : null,
  );
  const delta = original.map((v, i) =>
    v === null || clean[i] === null ? null : clean[i]! - v,
  );
  const stats = profileChanges({
    x_m: x,
    temperature_c: original,
    processed_temperature_c: clean,
  })!;
  const extent = signalSummary([...original, ...clean]);
  const pad = extent.range ? extent.range * 0.06 : 0.1;
  const deltaLimit = Math.max(0.01, stats.max_absolute_change_c ?? 0) * 1.05;
  return {
    original,
    clean,
    delta,
    stats,
    limits: [
      extent.min === null ? 0 : extent.min - pad,
      extent.max === null ? 1 : extent.max + pad,
    ] as [number, number],
    deltaLimits: [-deltaLimit, deltaLimit] as [number, number],
  };
}

export const answerLabels: Record<string, string> = {
  present: "Есть изменение",
  absent: "Изменения не вижу",
  ambiguous: "Неоднозначно",
  unknown: "Недостаточно данных",
};

export function qualityLabel(reason: number) {
  if (reason === 0) return "Валидное измерение";
  const reasons = [
    reason & 1 ? "Sentinel −300" : "",
    reason & 2 ? "Нечисловое значение" : "",
    reason & 4 ? "Отсутствующий хвост" : "",
  ].filter(Boolean);
  return reasons.length ? reasons.join("; ") : `Код качества ${reason}`;
}

/** Null is a break marker, not an interpolated or invented measurement. */
export function breakTimeGaps(
  x: number[],
  y: (number | null)[],
  gapSeconds = 300,
) {
  const time: number[] = [],
    values: (number | null)[] = [];
  x.forEach((value, index) => {
    if (index && value - x[index - 1] > gapSeconds) {
      time.push((x[index - 1] + value) / 2);
      values.push(null);
    }
    time.push(value);
    values.push(y[index]);
  });
  return { x: time, y: values };
}
