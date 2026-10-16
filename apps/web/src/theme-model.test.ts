import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { runInNewContext } from "node:vm";
import { parseTheme, readTheme, resolveTheme, saveTheme, THEME_STORAGE_KEY } from "./theme-model";
describe("theme preference", () => {
  it("uses the OS only until an explicit choice exists", () => {
    expect(resolveTheme(null, true)).toBe("dark");
    expect(resolveTheme(null, false)).toBe("light");
    expect(resolveTheme("light", true)).toBe("light");
    expect(resolveTheme("dark", false)).toBe("dark");
    expect(parseTheme("g100")).toBeNull();
    expect(parseTheme(null)).toBeNull();
  });
  it("persists only a theme, and tolerates blocked/full storage", () => {
    let pair: string[] = [];
    expect(saveTheme({ setItem: (key, value) => { pair = [key, value]; } }, "dark")).toBe(true);
    expect(pair).toEqual([THEME_STORAGE_KEY, "dark"]);
    expect(readTheme({ getItem: () => "light" })).toBe("light");
    expect(readTheme({ getItem: () => "unknown" })).toBeNull();
    expect(readTheme({ getItem: () => { throw Error("denied"); } })).toBeNull();
    expect(saveTheme({ setItem: () => { throw Error("quota"); } }, "dark")).toBe(false);
  });
});
describe("pre-paint bootstrap agrees with the React preference resolver", () => {
  const html = readFileSync(new URL("../index.html", import.meta.url), "utf8");
  const script = html.match(/<script id="theme-bootstrap">([\s\S]*?)<\/script>/)![1];
  for (const stored of [null, "light", "dark", "invalid", "blocked"]) {
    for (const systemDark of [true, false]) {
      it(`storage=${stored}, system dark=${systemDark}`, () => {
        const dataset: Record<string, string> = {};
        let meta = "";
        runInNewContext(script, {
          localStorage: { getItem: () => { if (stored === "blocked")
              throw Error("denied"); return stored; } },
          window: { matchMedia: () => ({ matches: systemDark }) },
          document: { documentElement: { dataset }, querySelector: () => ({ setAttribute: (_: string, value: string) => { meta = value; } }) },
        });
        const expected = resolveTheme(parseTheme(stored), systemDark);
        expect(dataset.theme).toBe(expected);
        expect(meta).toBe(expected === "dark" ? "#161616" : "#ffffff");
      });
    }
  }
});
