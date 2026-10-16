export type ProfileChanges = {
  schema_version: "dts-profile-changes-v1";
  threshold_c: number;
  total_points: number;
  paired_points: number;
  missing_original: number;
  missing_processed: number;
  changed_points: number;
  max_absolute_change_c: number | null;
  mean_absolute_change_c: number | null;
  top_points: {
    index: number;
    x_m: number;
    original_c: number;
    processed_c: number;
    delta_c: number;
  }[];
};

export function profileChanges(profile: {
  x_m: number[];
  temperature_c: (number | null)[];
  processed_temperature_c?: (number | null)[];
}): ProfileChanges | null {
  const processed = profile.processed_temperature_c;
  if (!processed) return null;
  const raw = profile.temperature_c;
  if (processed.length !== raw.length || profile.x_m.length !== raw.length)
    throw new Error("Profile shape differs");
  const pairs: ProfileChanges["top_points"] = [];
  let missingOriginal = 0,
    missingProcessed = 0;
  raw.forEach((before, index) => {
    const after = processed[index];
    if (before === null) missingOriginal++;
    else if (after === null) missingProcessed++;
    else
      pairs.push({
        index,
        x_m: profile.x_m[index],
        original_c: before,
        processed_c: after,
        delta_c: after - before,
      });
  });
  const absolute = pairs.map((p) => Math.abs(p.delta_c));
  return {
    schema_version: "dts-profile-changes-v1",
    threshold_c: 0.01,
    total_points: raw.length,
    paired_points: pairs.length,
    missing_original: missingOriginal,
    missing_processed: missingProcessed,
    changed_points: absolute.filter((v) => v > 0.01).length,
    max_absolute_change_c: absolute.length ? Math.max(...absolute) : null,
    mean_absolute_change_c: absolute.length
      ? absolute.reduce((sum, v) => sum + v, 0) / absolute.length
      : null,
    top_points: pairs
      .sort(
        (a, b) =>
          Math.abs(b.delta_c) - Math.abs(a.delta_c) || a.index - b.index,
      )
      .slice(0, 5),
  };
}
