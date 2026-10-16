import { afterEach, expect, it, vi } from "vitest";

afterEach(() => vi.unstubAllGlobals());

it("isolates the same view by immutable source version and protects cached buffers", async () => {
  vi.resetModules();
  vi.stubGlobal("matchMedia", () => ({ matches: false }));
  let calls = 0;
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => new Response(new Uint8Array([++calls]).buffer)),
  );
  const { binary } = await import("./cache");
  const signal = new AbortController().signal;
  const first = await binary("/records/5/tiles?width=1", signal, "sha-a");
  new Uint8Array(first)[0] = 99;
  expect(
    new Uint8Array(
      await binary("/records/5/tiles?width=1", signal, "sha-a"),
    )[0],
  ).toBe(1);
  expect(
    new Uint8Array(
      await binary("/records/5/tiles?width=1", signal, "sha-b"),
    )[0],
  ).toBe(2);
  expect(calls).toBe(2);
});

it("does not repopulate a cleared account cache with a late response", async () => {
  vi.resetModules();
  vi.stubGlobal("matchMedia", () => ({ matches: false }));
  let release: (response: Response) => void = () => {};
  vi.stubGlobal("fetch", vi.fn(() => new Promise<Response>((resolve) => { release = resolve; })));
  const { binary, tiles } = await import("./cache");
  const pending = binary("/records/5/tiles?width=1", new AbortController().signal, "sha-a");
  tiles.clear();
  release(new Response(new Uint8Array([1]).buffer));
  await pending;
  expect(tiles.bytes).toBe(0);
});
