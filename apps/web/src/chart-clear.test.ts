import { describe, expect, it } from "vitest";
import { canvasClearRgb } from "./chart-clear";

describe("heatmap clear color", () => {
  it("does not turn a transparent black page color into opaque black", () => {
    expect(canvasClearRgb("rgba(0, 0, 0, 0)", false)).toEqual([255, 255, 255]);
    expect(canvasClearRgb("rgba(0, 0, 0, 0)", true)).toEqual([22, 22, 22]);
  });

  it("keeps an opaque computed color, including real black", () => {
    expect(canvasClearRgb("rgb(255, 255, 255)", true)).toEqual([255, 255, 255]);
    expect(canvasClearRgb("rgb(22, 22, 22)", false)).toEqual([22, 22, 22]);
    expect(canvasClearRgb("rgb(0, 0, 0)", false)).toEqual([0, 0, 0]);
    expect(canvasClearRgb("rgba(244, 244, 244, 1)", false)).toEqual([244, 244, 244]);
  });

  it("follows the active theme when the color is not rgb()", () => {
    expect(canvasClearRgb("", false)).toEqual([255, 255, 255]);
    expect(canvasClearRgb("transparent", true)).toEqual([22, 22, 22]);
  });
});
