import { describe, expect, it } from "vitest";
import fixtures from "./profile-changes.fixtures.json";
import { profileChanges } from "./profile-changes";

describe("window-local descriptive changes shared with the server", () => {
  for (const fixture of fixtures)
    it(fixture.name, () => {
      expect(profileChanges(fixture.profile)).toEqual(fixture.expected);
    });
  it("rejects a truncated processed profile", () => {
    expect(() =>
      profileChanges({
        x_m: [0],
        temperature_c: [1],
        processed_temperature_c: [],
      }),
    ).toThrow("Profile shape differs");
  });
});
