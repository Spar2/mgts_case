export type AnalysisRegion = {
  record_id: string;
  row_start: number;
  row_end: number;
  column_start: number;
  column_end: number;
};
export function commonAnalysisRegion(a:{region:AnalysisRegion;source_sha256:string},b:{region:AnalysisRegion;source_sha256:string}):AnalysisRegion|null{
  if(a.source_sha256!==b.source_sha256||a.region.record_id!==b.region.record_id)return null;
  const r={record_id:a.region.record_id,row_start:Math.max(a.region.row_start,b.region.row_start),row_end:Math.min(a.region.row_end,b.region.row_end),column_start:Math.max(a.region.column_start,b.region.column_start),column_end:Math.min(a.region.column_end,b.region.column_end)};
  return r.row_start<r.row_end&&r.column_end-r.column_start>=2?r:null;
}
export type AnalysisContext = {
  region: AnalysisRegion;
  row: number;
  column: number;
  layer: string;
  zone: string;
};
type Run = {
  id: string;
  source_sha256: string;
  region: AnalysisRegion;
  layers: string[];
  zones: (AnalysisRegion & { id: string })[];
};
const bounds = ["r0", "r1", "c0", "c1"] as const;
export function parseAnalysisContext(
  params: URLSearchParams,
  run: Run,
): { ok: true; context: AnalysisContext } | { ok: false; message: string } {
  const fail = (message: string) => ({ ok: false as const, message });
  if (
    params.has("source_sha256") &&
    params.get("source_sha256") !== run.source_sha256
  )
    return fail("SHA источника в ссылке не совпадает с выбранным расчётом.");
  const zone = params.get("zone_id") ?? "";
  const selected = run.zones.find((z) => z.id === zone);
  if (zone && !selected)
    return fail("Зона из ссылки отсутствует в этом расчёте.");
  const number = (key: string) => {
    const text = params.get(key);
    return text !== null &&
      /^\d+$/.test(text) &&
      Number.isSafeInteger(Number(text))
      ? Number(text)
      : NaN;
  };
  const count = bounds.filter((k) => params.has(k)).length;
  if (count && count !== 4)
    return fail("В ссылке нужны все четыре границы окна.");
  const region = count
    ? {
        record_id: run.region.record_id,
        row_start: number("r0"),
        row_end: number("r1"),
        column_start: number("c0"),
        column_end: number("c1"),
      }
    : {
        record_id: run.region.record_id,
        row_start: (selected ?? run.region).row_start,
        row_end: (selected ?? run.region).row_end,
        column_start: (selected ?? run.region).column_start,
        column_end: (selected ?? run.region).column_end,
      };
  const r = run.region;
  if (
    !(
      r.row_start <= region.row_start &&
      region.row_start < region.row_end &&
      region.row_end <= r.row_end &&
      r.column_start <= region.column_start &&
      region.column_start < region.column_end &&
      region.column_end <= r.column_end
    )
  )
    return fail(
      "Границы окна должны находиться внутри исходного покрытия расчёта.",
    );
  if (
    selected &&
    bounds.some(
      (_, i) =>
        region[
          (["row_start", "row_end", "column_start", "column_end"] as const)[i]
        ] !==
        selected[
          (["row_start", "row_end", "column_start", "column_end"] as const)[i]
        ],
    )
  )
    return fail("Границы ссылки не совпадают с выбранной зоной алгоритма.");
  const row = params.has("row") ? number("row") : region.row_start;
  const column = params.has("column") ? number("column") : region.column_start;
  if (
    !(
      region.row_start <= row &&
      row < region.row_end &&
      region.column_start <= column &&
      column < region.column_end
    )
  )
    return fail("Профиль и точка должны находиться внутри выбранного окна.");
  const layer = params.get("layer") ?? "raw";
  if (
    ![
      "raw",
      ...run.layers,
      ...(run.layers.includes("processed_temperature_c") ? ["difference"] : []),
    ].includes(layer)
  )
    return fail("Запрошенный слой не передан в этом расчёте.");
  return { ok: true, context: { region, row, column, layer, zone } };
}
export function analysisParams(
  run: Pick<Run, "id" | "source_sha256">,
  context: AnalysisContext,
  base = new URLSearchParams(),
) {
  const p = new URLSearchParams(base);
  const r = context.region;
  Object.entries({
    run_id: run.id,
    source_sha256: run.source_sha256,
    r0: r.row_start,
    r1: r.row_end,
    c0: r.column_start,
    c1: r.column_end,
    row: context.row,
    column: context.column,
    layer: context.layer,
  }).forEach(([key, value]) => p.set(key, String(value)));
  if (context.zone) p.set("zone_id", context.zone);
  else p.delete("zone_id");
  return p;
}
