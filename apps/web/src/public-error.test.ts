import { describe, expect, it } from "vitest";
import {
  PublicApiError,
  clampIndex,
  shouldRetryPublic,
  throwPublicError,
} from "./public-error";

describe("throwPublicError", () => {
  it("uses JSON detail when present", async () => {
    const r = Response.json({ detail: "bad fields" }, { status: 400 });
    await expect(throwPublicError(r)).rejects.toMatchObject({
      status: 400,
      detail: "bad fields",
    });
    await expect(throwPublicError(r)).rejects.toBeInstanceOf(PublicApiError);
  });

  it("falls back on HTML body", async () => {
    const r = new Response("<html>nginx 502</html>", { status: 502 });
    await expect(throwPublicError(r)).rejects.toMatchObject({
      status: 502,
      detail: "Request failed with status 502",
    });
  });

  it("falls back on empty body", async () => {
    const r = new Response(null, { status: 503 });
    await expect(throwPublicError(r)).rejects.toMatchObject({
      status: 503,
      detail: "Request failed with status 503",
    });
  });
});

describe("clampIndex", () => {
  it("clamps boundaries", () => {
    expect(clampIndex(-5, 10)).toBe(0);
    expect(clampIndex(0, 10)).toBe(0);
    expect(clampIndex(9, 10)).toBe(9);
    expect(clampIndex(99, 10)).toBe(9);
    expect(clampIndex(2.9, 10)).toBe(2);
  });

  it("handles NaN and empty length", () => {
    expect(clampIndex(NaN, 10)).toBe(0);
    expect(clampIndex(Infinity, 10)).toBe(0);
    expect(clampIndex(3, 0)).toBe(0);
    expect(clampIndex(3, -1)).toBe(0);
  });
});

describe("shouldRetryPublic", () => {
  it("retries network errors and 503", () => {
    expect(shouldRetryPublic(0, new TypeError("fetch failed"))).toBe(true);
    expect(shouldRetryPublic(0, new PublicApiError(503, "busy"))).toBe(true);
  });

  it("does not retry client errors", () => {
    expect(shouldRetryPublic(0, new PublicApiError(404, "nope"))).toBe(false);
    expect(shouldRetryPublic(0, new PublicApiError(401, "nope"))).toBe(false);
  });

  it("does not retry aborts or exhausted attempts", () => {
    expect(
      shouldRetryPublic(0, new DOMException("aborted", "AbortError")),
    ).toBe(false);
    expect(shouldRetryPublic(3, new TypeError("fetch failed"))).toBe(false);
    expect(shouldRetryPublic(3, new PublicApiError(503, "busy"))).toBe(false);
  });
});
