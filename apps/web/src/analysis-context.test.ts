import { describe, expect, it } from "vitest";
import { analysisParams, parseAnalysisContext,commonAnalysisRegion } from "./analysis-context";
const region = {
  record_id: "5",
  row_start: 10,
  row_end: 100,
  column_start: 20,
  column_end: 200,
};
const zone = {
  ...region,
  id: "z",
  row_start: 15,
  row_end: 25,
  column_start: 30,
  column_end: 40,
};
const run = {
  id: "run",
  source_sha256: "a".repeat(64),
  region,
  layers: ["score"],
  zones: [zone],
};
describe("native analysis permalink", () => {
  it('compares native intersection only with identical source identity',()=>{
    const second={...run,region:{...region,row_start:30,row_end:120,column_start:50,column_end:180}};
    expect(commonAnalysisRegion(run,second)).toEqual({...region,row_start:30,row_end:100,column_start:50,column_end:180});
    expect(commonAnalysisRegion(run,{...second,source_sha256:'b'.repeat(64)})).toBeNull();
    expect(commonAnalysisRegion(run,{...second,region:{...second.region,row_start:100}})).toBeNull();
  });
  it("defaults to the selected zone, including its first point", () => {
    const parsed = parseAnalysisContext(new URLSearchParams("zone_id=z"), run);
    expect(parsed.ok && Object.keys(parsed.context.region).sort()).toEqual(
      Object.keys(region).sort(),
    );
    expect(parsed.ok && [parsed.context.row, parsed.context.column]).toEqual([
      15, 30,
    ]);
  });
  it("roundtrips a custom window and keeps comparison", () => {
    const context = {
      region: { ...region, row_end: 80 },
      row: 50,
      column: 100,
      layer: "score",
      zone: "",
    };
    const params = analysisParams(
      run,
      context,
      new URLSearchParams("compare=other&zone_id=z"),
    );
    expect(params.get("compare")).toBe("other");
    expect(params.has("zone_id")).toBe(false);
    expect(parseAnalysisContext(params, run)).toEqual({ ok: true, context });
  });
  for (const query of [
    "r0=10",
    "r0=10&r1=101&c0=20&c1=200",
    "r0=10&r1=10&c0=20&c1=200",
    "row=100",
    "row=-1",
    "column=1e2",
    "zone_id=unknown",
    "zone_id=z&r0=10&r1=100&c0=20&c1=200",
    "layer=difference",
    "source_sha256=wrong",
  ])
    it(`rejects explicit incompatible context: ${query}`, () => {
      expect(parseAnalysisContext(new URLSearchParams(query), run).ok).toBe(
        false,
      );
    });
  it("keeps the last included cell, rejects the excluded endpoint", () => {
    expect(
      parseAnalysisContext(new URLSearchParams("row=99&column=199"), run).ok,
    ).toBe(true);
    expect(
      parseAnalysisContext(new URLSearchParams("column=200"), run).ok,
    ).toBe(false);
  });
});
