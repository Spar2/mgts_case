import { describe, expect, it, vi } from "vitest";
import { decimate, decode } from "./numeric";
describe("scientific plot reduction", () => {
  it("decodes aligned float32, quality masks and native counts; rejects truncation", () => {
    const header = new TextEncoder().encode(
      JSON.stringify({
        version: 1,
        shape: [1, 2],
        x_edges: [0, 1, 3],
        time_edges: [0, 10],
        scale: [0, 30],
        aggregation: "median",
        source_sha256: "fixture",
      }),
    );
    const start = Math.ceil((4 + header.length) / 4) * 4;
    const countStart = start + 12;
    const buffer = new ArrayBuffer(countStart + 8);
    new DataView(buffer).setUint32(0, header.length, true);
    new Uint8Array(buffer, 4, header.length).set(header);
    new Float32Array(buffer, start, 2).set([21.25, NaN]);
    new Uint8Array(buffer, start + 8, 2).set([0, 8]);
    new Uint32Array(buffer, countStart, 2).set([12, 0]);
    const decoded = decode(buffer);
    expect([...decoded.counts]).toEqual([12, 0]);
    expect([...decoded.mask]).toEqual([0, 8]);
    expect(decoded.values[0]).toBe(21.25);
    expect(decoded.instances[7]).toBe(1);
    expect(() => decode(buffer.slice(0, -1))).toThrow("Неполный блок");
    expect(() => decode(new ArrayBuffer(0))).toThrow("Неполный заголовок");
  });
  it("keeps extremes, original coordinates and gap markers", () => {
    const x = Array.from({ length: 10000 }, (_, i) => i * 0.1),
      y: (number | null)[] = x.map(() => 20);
    y[731] = 200;
    y[1633] = -15;
    y[9000] = null;
    const [xx, yy] = decimate(x, y, 500);
    expect(yy).toContain(200);
    expect(yy).toContain(-15);
    expect(yy).toContain(null);
    expect(xx).toContain(x[731]);
    expect(xx[0]).toBe(0);
    expect(xx.at(-1)).toBe(x.at(-1));
  });
  it("evicts old binary data under a hard byte budget", async () => {
    vi.stubGlobal("matchMedia", () => ({ matches: false }));
    const { BinaryCache } = await import("./cache");
    const cache = new BinaryCache(16);
    cache.put("a", new ArrayBuffer(8));
    cache.put("b", new ArrayBuffer(8));
    cache.get("a");
    cache.put("c", new ArrayBuffer(8));
    expect(cache.get("b")).toBeUndefined();
    expect(cache.bytes).toBe(16);
    cache.put("big", new ArrayBuffer(99));
    expect(cache.bytes).toBe(16);
  });
});
