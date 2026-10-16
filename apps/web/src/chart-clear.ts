/** Opaque RGB for a WebGL clear. A transparent computed color is not black. */
export function canvasClearRgb(color: string, themeIsDark: boolean): [number, number, number] {
  const match = color.match(
    /rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)(?:\s*,\s*([\d.]+))?\s*\)/i,
  );
  const fallback: [number, number, number] = themeIsDark ? [22, 22, 22] : [255, 255, 255];
  if (!match) return fallback;
  const alpha = match[4] === undefined ? 1 : Number(match[4]);
  if (!(alpha >= 1)) return fallback;
  return [Number(match[1]), Number(match[2]), Number(match[3])];
}
