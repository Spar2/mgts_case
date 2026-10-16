// Isolated contract data only; these values are not an ML result or expert label.
async (page) => {
  const origin = new URL(page.url()).origin;
  const tab = await page.context().newPage();
  try {
    const health = await tab.request
      .get(origin + "/api/v1/health")
      .then((r) => r.json());
    if (!health.qa_mode)
      throw new Error("Selected-processing QA requires isolated qa_mode");
    const headers = {
      "X-Workbench-Token": (
        await tab.request.get(origin + "/api/v1/session").then((r) => r.json())
      ).mutation_token,
    };
    const detail = await tab.request
      .get(origin + "/api/v1/records/5")
      .then((r) => r.json());
    const slices = await tab.request
      .get(origin + "/api/v1/records/5/slices?row=5&column=0")
      .then((r) => r.json());
    const region = {
      record_id: "5",
      row_start: 5,
      row_end: 7,
      column_start: 0,
      column_end: slices.profile.length,
    };
    async function create(delta) {
      const response = await tab.request.post(origin + "/api/v1/results", {
        headers,
        data: {
          schema_version: "dts-method-result-v1",
          region: { ...region, row_end: 6 },
          source_sha256: detail.source_sha256,
          method: "qa_selected_contract",
          method_version: "fixture-v1",
          parameters: { delta },
          calculated_at: "2026-10-11",
          training_origin: "untrained",
          limitations: "Isolated contract fixture, not model output",
          result: {
            denoised: slices.profile.map((v) => (v === null ? 999 : v + delta)),
          },
        },
      });
      if (response.status() !== 201)
        throw new Error("QA result import failed: " + response.status());
      return response.json();
    }
    const selected = await create(1);
    const path = `/research?record=5&row=5&column=0&r0=5&r1=7&c0=0&c1=${region.column_end}&run_id=${selected.id}`;
    await tab.goto(origin + path);
    const selectedId = tab.locator('[aria-label="Выбранная обработка"] code');
    await selectedId.waitFor();
    if ((await selectedId.textContent()) !== selected.id)
      throw new Error("URL did not restore selected run");
    await create(2); // A concurrent result must not affect this tab or its report.
    const [response, download] = await Promise.all([
      tab.waitForResponse(
        (r) =>
          r.url().includes("/api/v1/reports?") &&
          r.request().method() === "POST",
      ),
      tab.waitForEvent("download"),
      tab.getByRole("button", { name: "JSON", exact: true }).click(),
    ]);
    if (response.status() !== 200)
      throw new Error("Selected export failed: " + response.status());
    const stream = await download.createReadStream(),
      chunks = [];
    for await (const chunk of stream) chunks.push(chunk);
    const report = JSON.parse(Buffer.concat(chunks).toString("utf8"));
    if (report.profile.processing_run_id !== selected.id)
      throw new Error("Report replaced selected run");
    const expected = slices.profile.map((v) => (v === null ? null : v + 1));
    if (
      JSON.stringify(report.profile.processed_temperature_c) !==
      JSON.stringify(expected)
    )
      throw new Error("Export values or mask changed");
    await tab.locator("#comparison-row").fill("6");
    await selectedId.waitFor({ state: "detached" });
    await tab.goto(origin + path);
    await selectedId.waitFor();
    for (const width of [320, 390, 430, 1024, 1440]) {
      await tab.setViewportSize({ width, height: 900 });
      await tab.waitForFunction(
        () => document.documentElement.scrollWidth <= window.innerWidth + 1,
      );
    }
    await tab.locator("#comparison-method").selectOption("median_x3");
    await selectedId.waitFor({ state: "detached" });
    let release, captured;
    const hold = new Promise((resolve) => (release = resolve)),
      intercepted = new Promise((resolve) => (captured = resolve));
    await tab.route("**/api/v1/denoise", async (route) => {
      captured();
      await hold;
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          run_id: selected.id,
          source_sha256: detail.source_sha256,
          row: 5,
          denoised: expected,
        }),
      });
    });
    const lateResponse = tab.waitForResponse(r => r.url().endsWith('/api/v1/denoise') && r.request().method() === 'POST');
    await tab
      .getByRole("button", { name: "Сравнить обработку", exact: true })
      .click();
    await intercepted;
    await tab.locator("#comparison-method").selectOption("median_x5");
    release();
    await (await lateResponse).finished();
    await tab.waitForFunction(() => Array.from(document.querySelectorAll('button')).some(button => button.textContent?.trim() === 'Сравнить обработку' && !button.disabled));
    await tab.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    await tab.waitForFunction(
      () => !document.querySelector('[aria-label="Выбранная обработка"]'),
    );
    await tab.unroute("**/api/v1/denoise");
    return {
      selected_run_preserved: true,
      native_mask_preserved: true,
      url_restoration: true,
      row_and_method_reset: true,
      mobile_widths: [320, 390, 430, 1024, 1440],
      late_response_rejected: true,
    };
  } catch (error) {
    throw new Error(
      String(error) +
        "; selected UI: " +
        (
          await tab
            .locator(".region-panel")
            .innerText()
            .catch(() => "unavailable")
        ).slice(-1500),
    );
  } finally {
    await tab.close();
  }
}
