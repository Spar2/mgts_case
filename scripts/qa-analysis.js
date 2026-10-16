// Read actual saved packages; never insert model results or expert answers.
async (page) => {
  const origin = new URL(page.url()).origin;
  const errors = [];
  const failed = [];
  page.on("requestfailed", (request) =>
    failed.push({ url: request.url(), error: request.failure()?.errorText }),
  );
  page.on("pageerror", (e) => errors.push(String(e)));
  const health = await page.request
    .get(origin + "/api/v1/health")
    .then((r) => r.json());
  if (!health.qa_mode) throw new Error("Isolated QA only");
  const runs = await page.request
    .get(origin + "/api/v2/result-runs")
    .then((r) => r.json());
  const run = runs.find((r) => r.layers.includes("score"));
  if (!run)
    throw new Error("Prepare a real numeric package before this scenario");
  await page.emulateMedia({ reducedMotion: "reduce" });
  const readyField = () =>
    page
      .locator(
        '.heat-figure[data-source-sha256="' + run.source_sha256 + '"] canvas',
      )
      .first()
      .waitFor();
  const layerLabels = {
    raw: "Исходное поле",
    processed_temperature_c: "Обработанное поле",
    difference: "Разность: обработка − исходник",
    score: "Оценка аномальности",
  };
  const metrics = [];
  for (const width of [1440, 430, 390, 320]) {
    await page.setViewportSize({ width, height: 900 });
    const start = Date.now();
    await page.goto(origin + "/analysis?run_id=" + run.id);
    await page
      .getByRole("heading", {
        name: run.method + " / " + run.method_version,
        exact: true,
      })
      .waitFor();
    await page.getByText(/Точка:.*время от начала/).waitFor();
    await readyField();
    await readyField();
    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth > innerWidth + 1,
    );
    if (overflow) throw new Error("Analysis overflow at " + width);
    metrics.push({ width, first_useful_ms: Date.now() - start });
    for (const layer of [
      "raw",
      "processed_temperature_c",
      "difference",
      "score",
    ]) {
      await page.locator("#analysis-layer").selectOption(layer);
      await page.waitForURL((url) => url.searchParams.get("layer") === layer);
      await page
        .getByRole("heading", { name: layerLabels[layer], exact: true })
        .waitFor();
      await readyField();
    }
    if (run.zones.length) {
      const zoneResponse = page.waitForResponse(
        (r) =>
          r.url().includes("/slices?") &&
          r.url().includes("zone_id=" + run.zones[0].id),
      );
      await page.locator("#analysis-zone").selectOption(run.zones[0].id);
      await page
        .getByRole("heading", {
          name: "Выбранная зона: " + run.zones[0].id,
          exact: true,
        })
        .waitFor();
      await readyField();
      const data = await (await zoneResponse).json(),
        zone = run.zones[0];
      if (
        data.x_m.length !== zone.column_end - zone.column_start ||
        data.native_time_s.length !== zone.row_end - zone.row_start
      )
        throw new Error("Zone charts use a different window");
    }
  }
  const r = run.region;
  if (r.row_end - r.row_start < 4 || r.column_end - r.column_start < 4)
    throw new Error("QA requires a window with at least four rows/columns");
  const custom = [
    r.row_start + 1,
    Math.min(r.row_start + 4, r.row_end),
    r.column_start + 1,
    Math.min(r.column_start + 5, r.column_end),
  ];
  for (let i = 0; i < 4; i++)
    await page.locator(`#analysis-bound-${i}`).fill(String(custom[i]));
  const customResponse = page.waitForResponse((response) => {
    const u = new URL(response.url());
    return (
      u.pathname.endsWith("/slices") &&
      u.searchParams.get("row_start") === String(custom[0]) &&
      u.searchParams.get("column_start") === String(custom[2])
    );
  });
  await page
    .getByRole("button", { name: "Применить окно", exact: true })
    .click();
  const customSlices = await (await customResponse).json();
  if (
    customSlices.x_m.length !== custom[3] - custom[2] ||
    customSlices.native_time_s.length !== custom[1] - custom[0]
  )
    throw new Error("Custom slices use a different window");
  await page.locator("#analysis-layer").selectOption("score");
  await page.waitForURL((url) => url.searchParams.get("layer") === "score");
  await readyField();
  const permalink = page.url();
  const savedParams = new URL(permalink).searchParams;
  if (
    savedParams.get("source_sha256") !== run.source_sha256 ||
    savedParams.get("zone_id") !== null ||
    savedParams.get("layer") !== "score"
  )
    throw new Error("Permalink lost source/layer/custom window");
  await page.reload();
  await page.getByText(/Точка:.*время от начала/).waitFor();
  await readyField();
  for (let i = 0; i < 4; i++)
    if (
      (await page.locator(`#analysis-bound-${i}`).inputValue()) !==
      String(custom[i])
    )
      throw new Error("Reload lost custom bounds");
  await page.locator("#analysis-bound-0").fill("-1");
  await page
    .getByRole("button", { name: "Применить окно", exact: true })
    .click();
  await page.getByText("Окно не применено", { exact: true }).waitFor();
  if (page.url() !== permalink)
    throw new Error("Invalid draft changed committed context");
  await page.locator("#analysis-bound-0").fill(String(custom[0]));
  await page
    .getByRole("button", { name: "Всё покрытие расчёта", exact: true })
    .click();
  await readyField();
  await page.goBack();
  await page.getByText(/Точка:.*время от начала/).waitFor();
  await readyField();
  if (
    (await page.locator("#analysis-bound-0").inputValue()) !== String(custom[0])
  )
    throw new Error("Back lost custom context");
  await page.goForward();
  await page.getByText(/Точка:.*время от начала/).waitFor();
  await readyField();
  if (
    (await page.locator("#analysis-bound-0").inputValue()) !==
    String(r.row_start)
  )
    throw new Error("Forward lost full context");
  await page.goto(permalink);
  await page.getByText(/Точка:.*время от начала/).waitFor();
  await readyField();
  const bad = new URL(permalink);
  bad.searchParams.set("source_sha256", "0".repeat(64));
  await page.goto(bad.href);
  await page
    .getByText("Контекст исследования несовместим", { exact: true })
    .waitFor();
  if (await page.locator(".analysis-workspace").count())
    throw new Error("Wrong SHA silently opened another source");
  await page.goto(permalink);
  await page.getByText(/Точка:.*время от начала/).waitFor();
  await readyField();
  let delayed = false;
  await page.route("**/api/v2/result-runs/*/slices?*", async (route) => {
    if (
      new URL(route.request().url()).searchParams.get("row") ===
      String(custom[0] + 1)
    ) {
      delayed = true;
      await new Promise((resolve) => setTimeout(resolve, 700));
    }
    try {
      await route.continue();
    } catch {
      /* Superseded request may have been aborted. */
    }
  });
  await page.locator("#analysis-row").fill(String(custom[0] + 1));
  await page.locator("#analysis-row").press("Enter");
  await page.waitForURL(
    (url) => url.searchParams.get("row") === String(custom[0] + 1),
  );
  const finalResponse = page.waitForResponse(
    (response) =>
      new URL(response.url()).pathname.endsWith("/slices") &&
      new URL(response.url()).searchParams.get("row") === String(custom[0] + 2),
  );
  await page
    .locator(".coordinate-input")
    .first()
    .getByRole("button", { name: "Следующий профиль", exact: true })
    .click();
  await finalResponse;
  await page.getByText(/Точка:.*время от начала/).waitFor();
  await readyField();
  await page.waitForTimeout(850);
  await page.unroute("**/api/v2/result-runs/*/slices?*");
  if (
    !delayed ||
    (await page.locator("#analysis-row").inputValue()) !==
      String(custom[0] + 2) ||
    new URL(page.url()).searchParams.get("row") !== String(custom[0] + 2)
  )
    throw new Error("Late response restored stale point");
  for (const width of [320, 390, 430, 1440]) {
    await page.setViewportSize({ width, height: 900 });
    if (
      await page.evaluate(
        () => document.documentElement.scrollWidth > innerWidth + 1,
      )
    )
      throw new Error("Custom window overflow at " + width);
  }
  await page.setViewportSize({ width: 1440, height: 1000 });
  const preview = page.waitForResponse((r) =>
    r.url().endsWith("/api/v2/cases/preview"),
  );
  await page
    .getByRole("button", { name: "Проверить состав кейса", exact: true })
    .click();
  const checked = await preview;
  if (checked.status() !== 200)
    throw new Error("Case preview status " + checked.status());
  await page
    .getByRole("button", { name: "Зафиксировать кейс", exact: true })
    .click();
  const link = page.getByRole("link", {
    name: "Кейс сохранён — открыть →",
    exact: true,
  });
  await link.waitFor();
  await link.click();
  await page
    .getByRole("heading", { name: "Что изменила обработка", exact: true })
    .waitFor();
  const caseUrl = page.url();
  await page
    .getByRole("link", { name: "Открыть исследование →", exact: true })
    .click();
  await page.getByText(/Точка:.*время от начала/).waitFor();
  await readyField();
  if (
    (await page.locator("#analysis-zone").inputValue()) !== "" ||
    (await page.locator("#analysis-layer").inputValue()) !== "score" ||
    (await page.locator("#analysis-bound-0").inputValue()) !==
      String(custom[0]) ||
    (await page.locator("#analysis-row").inputValue()) !== String(custom[0] + 2)
  )
    throw new Error("Case lost custom window/layer/point");
  await page.goto(caseUrl);
  await page
    .getByRole("link", { name: "Показать на защите →", exact: true })
    .click();
  await page.locator("#case-step-heading").waitFor();
  await page.getByRole("button", { name: "Далее", exact: true }).click();
  await page.getByRole("button", { name: "Далее", exact: true }).click();
  await page
    .getByRole("heading", { name: "Что изменила обработка", exact: true })
    .waitFor();
  if (errors.length) throw new Error(JSON.stringify({ errors, failed }));
  return {
    analysis_layers: true,
    analysis_mobile: true,
    analysis_case: true,
    analysis_custom_window: true,
    analysis_permalink: true,
    analysis_late_response: true,
    metrics,
    errors,
  };
}
