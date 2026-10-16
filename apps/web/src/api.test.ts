import { afterEach, describe, expect, it, vi } from "vitest";
afterEach(() => vi.unstubAllGlobals());
describe("safe local session recovery", () => {
  it("refreshes only a rejected mutation token and preserves the idempotency key", async () => {
    vi.resetModules();
    const sent: RequestInit[] = [];
    let sessions = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string, options?: RequestInit) => {
        if (url.endsWith("/session"))
          return Response.json({
            mutation_token: ++sessions === 1 ? "old" : "new",
          });
        sent.push(options!);
        return sent.length === 1
          ? Response.json(
              { detail: "Local mutation token required" },
              { status: 403 },
            )
          : Response.json({ revision: 7 });
      }),
    );
    const { mutate } = await import("./api");
    expect(
      await mutate("/reviews", { notes: "evidence" }, "same-action"),
    ).toEqual({ revision: 7 });
    expect(sent).toHaveLength(2);
    expect(sent.map((r) => r.body)).toEqual([
      '{"notes":"evidence"}',
      '{"notes":"evidence"}',
    ]);
    expect(
      sent.map((r) => (r.headers as Record<string, string>)["Idempotency-Key"]),
    ).toEqual(["same-action", "same-action"]);
    expect(
      sent.map(
        (r) => (r.headers as Record<string, string>)["X-Workbench-Token"],
      ),
    ).toEqual(["old", "new"]);
  });
  it("does not retry an untrusted origin or an unconfirmed network write", async () => {
    vi.resetModules();
    let writes = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => {
        if (url.endsWith("/session"))
          return Response.json({ mutation_token: "local" });
        writes++;
        return Response.json({ detail: "Untrusted origin" }, { status: 403 });
      }),
    );
    const { mutate } = await import("./api");
    await expect(mutate("/reviews", {})).rejects.toMatchObject({ status: 403 });
    expect(writes).toBe(1);
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new TypeError("connection lost");
      }),
    );
    await expect(mutate("/reviews", {})).rejects.toThrow("connection lost");
    expect(fetch).toHaveBeenCalledTimes(1);
  });
});
