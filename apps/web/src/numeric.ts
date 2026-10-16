export type TileMeta = {
  version: number;
  shape: [number, number];
  x_edges?: number[];
  time_edges?: number[];
  x_cells?: [number, number][];
  time_cells?: [number, number][];
  aggregation: string;
  source_sha256: string;
  scale: [number, number];
  palette?: "diverging";
  title?: string;
  unit?: string;
};
export function decode(buffer: ArrayBuffer) {
  if (buffer.byteLength < 4) throw new Error("Неполный заголовок данных");
  const length = new DataView(buffer).getUint32(0, true);
  if (length > buffer.byteLength - 4)
    throw new Error("Неполный заголовок данных");
  const meta = JSON.parse(
    new TextDecoder().decode(new Uint8Array(buffer, 4, length)),
  ) as TileMeta;
  if (
    ![1, 2].includes(meta.version) ||
    meta.shape.length !== 2 ||
    meta.shape.some((v) => !Number.isInteger(v) || v < 1) ||
    meta.shape[0] * meta.shape[1] > 200000
  )
    throw new Error("Неподдерживаемый блок данных");
  const count = meta.shape[0] * meta.shape[1],
    start = Math.ceil((4 + length) / 4) * 4;
  const countStart = start + Math.ceil((count * 5) / 4) * 4;
  if (
    buffer.byteLength !==
    countStart + count * 4 + (meta.version === 2 ? count : 0)
  )
    throw new Error("Неполный блок данных");
  const values = new Float32Array(buffer, start, count),
    mask = new Uint8Array(buffer, start + count * 4, count),
    counts = new Uint32Array(buffer, countStart, count);
  const displayFlags =
    meta.version === 2
      ? new Uint8Array(buffer, countStart + count * 4, count)
      : new Uint8Array(count);
  const nativeMask = mask;
  const effectiveMask = mask.slice();
  for (let i = 0; i < count; i++) {
    // Mixed aggregate bins can contain invalid native cells and valid observations.
    if (counts[i] > 0 && Number.isFinite(values[i])) effectiveMask[i] = 0;
    if (displayFlags[i] !== 0) effectiveMask[i] = 8;
  }
  const xc =
    meta.x_cells ??
    meta
      .x_edges!.slice(0, -1)
      .map((v, i) => [v, meta.x_edges![i + 1]] as [number, number]);
  const tc =
    meta.time_cells ??
    meta
      .time_edges!.slice(0, -1)
      .map((v, i) => [v, meta.time_edges![i + 1]] as [number, number]);
  if (xc.length !== meta.shape[1] || tc.length !== meta.shape[0])
    throw new Error("Неверные координаты блока");
  const extent = [xc[0][0], xc.at(-1)![1], tc[0][0], tc.at(-1)![1]];
  const instances = new Float32Array(count * 6);
  for (let y = 0; y < tc.length; y++)
    for (let x = 0; x < xc.length; x++) {
      const i = y * xc.length + x;
      instances.set(
        [
          (xc[x][0] - extent[0]) / (extent[1] - extent[0]),
          (xc[x][1] - extent[0]) / (extent[1] - extent[0]),
          (tc[y][0] - extent[2]) / (extent[3] - extent[2]),
          (tc[y][1] - extent[2]) / (extent[3] - extent[2]),
          Number.isFinite(values[i]) ? values[i] : 0,
          effectiveMask[i],
        ],
        i * 6,
      );
    }
  return {
    meta,
    instances,
    counts,
    values,
    mask: effectiveMask,
    nativeMask,
    displayFlags,
    extent,
  };
}
/** Keep extrema and break markers; x stays at the original measurement coordinate. */
export function decimate(
  x: number[],
  y: (number | null)[],
  budget = 2000,
): [number[], (number | null)[]] {
  if (x.length <= budget) return [x, y];
  const selected = new Set<number>([0, x.length - 1]),
    step = Math.ceil(x.length / Math.max(1, Math.floor(budget / 4)));
  for (let a = 0; a < x.length; a += step) {
    let lo = a,
      hi = a;
    for (let i = a; i < Math.min(x.length, a + step); i++) {
      if (y[i] === null) {
        selected.add(i);
        if (i) selected.add(i - 1);
        if (i + 1 < x.length) selected.add(i + 1);
      } else {
        if (y[lo] === null || y[i]! < y[lo]!) lo = i;
        if (y[hi] === null || y[i]! > y[hi]!) hi = i;
      }
    }
    selected.add(lo);
    selected.add(hi);
  }
  const indices = [...selected].sort((a, b) => a - b);
  return [indices.map((i) => x[i]), indices.map((i) => y[i])];
}
