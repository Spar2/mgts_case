// Actual CPU job on native measurements in an isolated schema; no model training.
async (page) => {
  const origin = new URL(page.url()).origin;
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  const health = await (
    await page.request.get(origin + "/api/v1/health")
  ).json();
  if (!health.qa_mode) throw new Error("Isolated QA only");
  await page.emulateMedia({ reducedMotion: "reduce" });
  const records = await (
    await page.request.get(origin + "/api/v1/records")
  ).json();
  const record = records.find((r) => r.id === "5") ?? records[0];
  const detail = await (
    await page.request.get(origin + "/api/v1/records/" + record.id)
  ).json();
  const [rows, columns] = detail.shape;
  if (rows < 2 || columns < 3) throw new Error("QA source too small");
  const base = `${origin}/research?record=${record.id}&r0=0&r1=${Math.min(rows, 16)}&c0=0&c1=${Math.min(columns, 64)}&row=0&column=0&source_sha256=${detail.source_sha256}`;
  await page.setViewportSize({ width: 390, height: 900 });
  await page.goto(base);
  // The calculation lives within the research actions, not a separate hidden UI.
  const button = page.getByRole("button", {
    name: "Запустить расчёт поля",
    exact: true,
  });
  await page
    .locator(".analysis-tools")
    .first()
    .evaluate((node) => {
      node.open = true;
    });
  await button.waitFor();
  await page.waitForFunction(
    () =>
      !document.querySelector(
        'section[aria-label="Расчёт реального поля"] button',
      )?.disabled,
  );
  await page.locator("#compute-bound-0").fill("-1");
  await button.click();
  await page.getByText(/Выберите допустимые исходные границы/).waitFor();
  await page.locator("#compute-bound-0").fill("0");
  // Drop the response after the real commit; manual retry must return the same job.
  let committedId = "",
    calls = 0;
  await page.route("**/api/v2/analysis-jobs", async (route) => {
    calls++;
    const response = await route.fetch();
    if (response.status() !== 202)
      throw new Error("Submit rejected: " + response.status());
    const data = await response.json();
    if (!committedId) {
      committedId = data.id;
      await route.abort("failed");
    } else {
      if (data.id !== committedId)
        throw new Error("Retry created a duplicate job");
      await route.fulfill({ response });
    }
  });
  await button.click();
  await page
    .getByRole("button", { name: "Повторить отправку расчёта", exact: true })
    .waitFor();
  await page
    .getByRole("button", { name: "Повторить отправку расчёта", exact: true })
    .click();
  await page.waitForURL(
    (url) => url.searchParams.get("job_id") === committedId,
  );
  await page.unroute("**/api/v2/analysis-jobs");
  await page.reload();
  await page
    .locator(".analysis-tools")
    .first()
    .evaluate((node) => {
      node.open = true;
    });
  const open = page.getByRole("link", {
    name: "Открыть результат расчёта →",
    exact: true,
  });
  await open.waitFor({ timeout: 30000 });
  const state = await (
    await page.request.get(origin + "/api/v2/analysis-jobs/" + committedId)
  ).json();
  if (state.status !== "complete" || !state.run_id || calls !== 2)
    throw new Error("Actual job/retry failed");
  for (const width of [320, 390, 430, 1440]) {
    await page.setViewportSize({ width, height: 900 });
    if (
      await page.evaluate(
        () => document.documentElement.scrollWidth > innerWidth + 1,
      )
    )
      throw new Error("Compute form overflow at " + width);
  }
  await open.click();
  await page.getByText(/Точка:.*время от начала/).waitFor();
  await page
    .locator(
      `.heat-figure[data-source-sha256="${detail.source_sha256}"] canvas`,
    )
    .first()
    .waitFor();
  if (
    new URL(page.url()).searchParams.get("run_id") !== state.run_id ||
    (await page.locator("#analysis-layer").inputValue()) !==
      "processed_temperature_c"
  )
    throw new Error("Wrong run opened");
  const preview = page.waitForResponse((r) =>
    r.url().endsWith("/api/v2/cases/preview"),
  );
  await page
    .getByRole("button", { name: "Проверить состав кейса", exact: true })
    .click();
  if ((await preview).status() !== 200)
    throw new Error("Calculated case preview failed");
  await page
    .getByRole("button", { name: "Зафиксировать кейс", exact: true })
    .click();
  await page
    .getByRole("link", { name: "Кейс сохранён — открыть →", exact: true })
    .click();
  await page
    .getByRole("heading", { name: "Что изменила обработка", exact: true })
    .waitFor();
  const caseId = page.url().split("/cases/")[1].split("?")[0];
  const saved = await (
    await page.request.get(origin + "/api/v2/cases/" + caseId)
  ).json();
  if (
    saved.run_id !== state.run_id ||
    saved.result.evidence.source_sha256 !== detail.source_sha256
  )
    throw new Error("Case lost job result");
  const bundle = await page.request.get(
    origin + "/api/v2/cases/" + caseId + "/bundle",
  );
  if (
    bundle.status() !== 200 ||
    !bundle.headers()["content-type"].includes("zip")
  )
    throw new Error("Case bundle unavailable");
  if (errors.length) throw new Error(JSON.stringify(errors));
  return {
    native_calculation: true,
    calculation_retry: true,
    calculation_reload: true,
    calculation_mobile: true,
    calculation_case: true,
    errors,
  };
}
