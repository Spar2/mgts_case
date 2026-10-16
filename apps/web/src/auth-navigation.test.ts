import { describe, expect, it } from "vitest";
import { safeReturnPath } from "./auth-navigation";

describe("safe login return", () => {
  it("preserves a working context without a fragment", () => {
    expect(safeReturnPath("/works/abcd?tab=signal#secret")).toBe("/works/abcd?tab=signal");
    expect(safeReturnPath("/pipeline?job_id=abcdef0123456789&record=5")).toBe("/pipeline?job_id=abcdef0123456789&record=5");
  });
  it("rejects external, login loops and ambiguous targets", () => {
    for (const target of [null, "https://other.invalid", "//other.invalid", "/\\other.invalid", "/login?next=/login", "/try", "/%2f%2fother.invalid", "/works\n"]) expect(safeReturnPath(target)).toBe("/research");
  });
});
