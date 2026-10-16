import { afterEach, expect, it, vi } from "vitest";
import { control } from "./access";
import { ApiError } from "./api";

vi.mock("./cache", () => ({ tiles: { clear: vi.fn() } }));

afterEach(() => vi.unstubAllGlobals());

it("preserves proxy failure status on a read without inventing data", async () => {
  const fetch = vi.fn().mockResolvedValue(new Response("<html>unavailable</html>", { status: 503 }));
  vi.stubGlobal("fetch", fetch);
  const error = await control("/service-capabilities").catch(e => e);
  expect(error).toBeInstanceOf(ApiError);
  if (!(error instanceof ApiError)) throw error;
  expect(error.status).toBe(503);
  expect(fetch).toHaveBeenCalledTimes(1);
});

it("preserves a failed PATCH attempt and key without automatic replay", async () => {
  const fetch = vi.fn()
    .mockResolvedValueOnce(new Response(JSON.stringify({ mutation_token: "qa-token" })))
    .mockResolvedValueOnce(new Response("<html>bad gateway</html>", { status: 502 }));
  vi.stubGlobal("fetch", fetch);
  const error = await control("/investigations/qa", { expected_revision: 1 }, undefined, "qa-action", "PATCH").catch(e => e);
  expect(error).toBeInstanceOf(ApiError);
  if (!(error instanceof ApiError)) throw error;
  expect(error.status).toBe(502);
  expect(fetch).toHaveBeenCalledTimes(2);
  expect(fetch.mock.calls[1][1]).toMatchObject({ method: "PATCH", headers: { "Idempotency-Key": "qa-action" } });
});
