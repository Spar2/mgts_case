// Playwright CLI run-code function. Uses the already opened local page.
// This script does not submit expert answers; use an isolated journal for save QA.
async (page) => {
  const origin = new URL(page.url()).origin;
  const errors = [];
  page.on("pageerror", (error) => errors.push(String(error)));
  await page.goto(origin + "/research");
  await page.locator(".heat-figure canvas").waitFor();
  const catalog = await page.evaluate(async () =>
    (await fetch("/api/v1/records")).json(),
  );
  if (catalog.length !== 6) throw new Error("Expected six real records");
  const selected = [];
  for (const record of catalog) {
    await page
      .getByRole("combobox", { name: "Запись", exact: true })
      .selectOption(record.id);
    await page.waitForFunction(
      (sha) => document.querySelector(".provenance code")?.textContent === sha,
      record.source_sha256,
    );
    await page.locator(".heat-figure canvas").waitFor();
    selected.push(record.source);
  }
  await page
    .getByRole("combobox", { name: "Запись", exact: true })
    .selectOption("5");
  await page.waitForFunction(
    (sha) => document.querySelector(".provenance code")?.textContent === sha,
    catalog.find((r) => r.id === "5").source_sha256,
  );
  await page.waitForFunction(text => document.querySelector('.readings')?.textContent?.includes(text),catalog.find(r=>r.id==='5').first_timestamp);
  await page.locator("#row").focus();
  await page.keyboard.press("ControlOrMeta+A");
  await page.locator("#row").pressSequentially("100");
  await page.locator("#row").press("Tab");
  const expected = await page.evaluate(async () =>
    (await fetch("/api/v1/records/5/slices?row=100&column=0")).json(),
  );
  await page.waitForFunction(
    (text) => document.querySelector(".readings")?.textContent?.includes(text),
    expected.timestamp,
  );
  const shown = await page.locator(".readings").innerText();
  if (
    expected.temperature !== null &&
    !shown.includes(expected.temperature.toFixed(4))
  )
    throw new Error("Exact temperature differs");
  await page
    .getByRole("button", { name: "Развернуть поле", exact: true })
    .click();
  if (!(await page.locator(".plots.expanded").count()))
    throw new Error("Field did not expand");
  await page.keyboard.press("Escape");
  if (await page.locator(".plots.expanded").count())
    throw new Error("Escape did not close field");
  const widths = [];
  await page.emulateMedia({ reducedMotion: "reduce" });
  for (const width of [320, 390, 430, 1024, 1440]) {
    await page.setViewportSize({ width, height: 900 });
    await page.waitForFunction(() => document.documentElement.scrollWidth <= innerWidth,undefined,{timeout:3000});
    const dimensions = await page.evaluate(() => ({
      viewport: innerWidth,
      document: document.documentElement.scrollWidth,
    }));
    if (dimensions.document > dimensions.viewport)
      throw new Error("Horizontal overflow: " + JSON.stringify(dimensions));
    widths.push(dimensions);
  }
  await page
    .getByRole("link", { name: "Перейти к экспертной проверке →" })
    .click();
  await page.locator(".heat-figure canvas").waitFor();
  await page.getByRole("heading", { name: "Самостоятельный ответ по сигналу" }).waitFor();
  const queue = await page.evaluate(async () =>
    (await fetch("/api/v1/queue")).json(),
  );
  const card = queue.cards.find((c) => !queue.annotations[c.review_id]);
  await page.goto(
    origin +
      "/review?source=" +
      encodeURIComponent(card.source) +
      "&card=" +
      encodeURIComponent(card.review_id),
  );
  await page.locator(".heat-figure canvas").waitFor();
  if (
    await page
      .getByRole("button", { name: "Раскрыть результаты методов", exact: true })
      .isEnabled()
  )
    throw new Error("Methods visible before own answer");
  const gate = await page.evaluate(
    async (key) => (await fetch("/api/v1/queue/" + key + "/methods")).status,
    card.review_id,
  );
  if (gate !== 403) throw new Error("Server blind gate failed");
  await page.setViewportSize({ width: 390, height: 844 });
  await page
    .getByRole("button", { name: "2 · Ответ", exact: true })
    .click();
  const responseTarget = await page.locator("#review-decision").boundingBox();
  if (responseTarget && (responseTarget.y > 844 || responseTarget.y < -40))
    throw new Error("Mobile answer jump missed form");
  if (errors.length)
    throw new Error("Browser runtime errors: " + errors.join("\n"));
  return {
    records: selected,
    exactSlice: {
      row: expected.row,
      column: expected.column,
      temperature: expected.temperature,
    },
    widths,
    blindGate: gate,
    runtimeErrors: errors,
  };
}
