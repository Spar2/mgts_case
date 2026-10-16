/** Bounded binary LRU; large buffers never enter query or React state. */
export class BinaryCache {
  private entries = new Map<string, ArrayBuffer>();
  bytes = 0;
  generation = 0;
  constructor(readonly budget: number) {}
  get(key: string) {
    const b = this.entries.get(key);
    if (b) {
      this.entries.delete(key);
      this.entries.set(key, b);
    }
    return b;
  }
  put(key: string, b: ArrayBuffer) {
    const old = this.entries.get(key);
    if (old) this.bytes -= old.byteLength;
    this.entries.delete(key);
    if (b.byteLength > this.budget) return;
    this.entries.set(key, b);
    this.bytes += b.byteLength;
    while (this.bytes > this.budget) {
      const oldest = this.entries.keys().next().value!;
      this.bytes -= this.entries.get(oldest)!.byteLength;
      this.entries.delete(oldest);
    }
  }
  clear() {
    this.generation++;
    this.entries.clear();
    this.bytes = 0;
  }
}
export const tiles = new BinaryCache(
  matchMedia("(max-width: 672px)").matches
    ? 32 * 1024 * 1024
    : 64 * 1024 * 1024,
);
// Read-only diagnostics for reproducible browser performance checks; no data or tokens.
Object.defineProperty(globalThis, "__dtsCacheStats", {
  configurable: true,
  value: () => ({ bytes: tiles.bytes, budget: tiles.budget }),
});
export async function binary(
  path: string,
  signal: AbortSignal,
  versionKey = "",
) {
  const generation = tiles.generation;
  let url =
    (path.startsWith("/api/v2/") ? path : "/api/v1" + path) +
    (path.includes("/tiles?") && !path.startsWith("/api/v2/") ? "&format_version=2" : "");
  if (versionKey)
    url +=
      (url.includes("?") ? "&" : "?") +
      "view_version=" +
      encodeURIComponent(versionKey);
  let b = tiles.get(url);
  if (!b) {
    const r = await fetch(url, { signal });
    if (r.status === 401) window.dispatchEvent(new Event("dts-auth-expired"));
    if (!r.ok) throw new Error((await r.json()).detail);
    b = await r.arrayBuffer();
    if (!signal.aborted && generation === tiles.generation) tiles.put(url, b);
  }
  return b.slice(0);
}
