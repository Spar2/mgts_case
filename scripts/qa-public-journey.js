async (page) => {
  const allowed = (process.env.QA_BASES || "https://localhost:8443")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
  const base = new URL(page.url()).origin;
  if (!allowed.includes(base))
    throw new Error("Disposable TLS QA stack required");
  const errors = [];
  page.on("pageerror", (error) => errors.push(String(error)));
  await page.goto(base + "/");
  await page
    .getByRole("link", { name: "Открыть реальное окно", exact: true })
    .click();
  await page.locator("#public-row").waitFor();
  await page.locator("#public-row").selectOption("24");
  await page.locator("#public-column").selectOption("40");
  await page
    .getByRole("button", { name: "Медианная обработка", exact: true })
    .click();
  const exact = await page.locator('p[role="status"]').textContent();
  if (!exact.includes("маска:")) throw new Error("Exact inspector missing");
  await page.locator("#demo-answer").selectOption("ambiguous");
  await page.locator("#demo-notes").fill("QA demonstration only");
  const download = page.waitForEvent("download");
  await page
    .getByRole("button", {
      name: "Скачать демонстрационный отчёт",
      exact: true,
    })
    .click();
  await download;
  await page.context().setOffline(true);
  if (!(await page.locator("#public-row").isVisible()))
    throw new Error("Received observations lost offline");
  await page.context().setOffline(false);
  for (const width of [320, 390, 430]) {
    await page.setViewportSize({ width, height: 900 });
    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth > innerWidth + 1,
    );
    if (overflow) throw new Error("Horizontal overflow at " + width);
    await page.locator("#public-column").focus();
    await page.keyboard.press("ArrowDown");
    await page.keyboard.press("Tab");
  }
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.reload();
  await page.locator("#public-row").waitFor();
  if (await page.locator("#demo-notes").inputValue())
    throw new Error("Demo answer unexpectedly persisted");
  const denied = await page.evaluate(async () => {
    const paths = [
      "/api/v1/records",
      "/api/v1/queue",
      "/api/v2/accounts",
      "/api/v2/jobs",
    ];
    return Promise.all(paths.map(async (p) => (await fetch(p)).status));
  });
  if (denied.some((s) => s !== 401))
    throw new Error("Anonymous private API access");
  await page.goto(base + "/request-access");
  await page.locator("#access-name").fill("QA public visitor");
  await page.locator("#access-contact").fill("Disposable QA known channel");
  await page.locator("#access-reason").fill("Browser acceptance only");
  await page
    .getByRole("button", { name: "Отправить заявку", exact: true })
    .click();
  await page
    .getByRole("status")
    .filter({ hasText: "Заявка принята" })
    .waitFor();
  await page.goto(base + "/activate");
  await page
    .getByText("Откройте одноразовую ссылку", { exact: false })
    .waitFor();
  await page.goto(base + "/try");
  await page.locator("#public-row").waitFor();
  let transient503Hits = 0;
  await page.route("**/api/public/v1/package", async (route) => {
    if (transient503Hits++ === 0) {
      await route.fulfill({
        status: 503,
        contentType: "application/json",
        body: JSON.stringify({ detail: "QA transient outage" }),
      });
    } else {
      await route.continue();
    }
  });
  await page.reload();
  await page.getByText("Данные недоступны", { exact: false }).waitFor();
  await page
    .getByRole("button", { name: "Повторить загрузку", exact: true })
    .click();
  await page.locator("#public-row").waitFor();
  await page.unrouteAll({ behavior: "wait" });
  const transient503Retried = true;
  const leakSecrets = [
    "mutation_token",
    "dts_session",
    "linkSecret",
    "password",
    "X-Workbench-Token",
    "Bearer",
  ];
  const pageHtml = await page.content();
  const storageDump = await page.evaluate(() =>
    JSON.stringify({
      local: { ...localStorage },
      session: { ...sessionStorage },
    }),
  );
  await page.locator("#demo-answer").selectOption("ambiguous");
  await page.locator("#demo-notes").fill("QA demonstration only");
  const leakDownloadEvent = page.waitForEvent("download");
  await page
    .getByRole("button", {
      name: "Скачать демонстрационный отчёт",
      exact: true,
    })
    .click();
  const leakDownload = await leakDownloadEvent;
  const leakStream = await leakDownload.createReadStream();
  const leakChunks = [];
  for await (const chunk of leakStream) leakChunks.push(chunk);
  const leakReportHtml = Buffer.concat(leakChunks).toString("utf8");
  const stripAllowedHex = (s) => s.replace(/[0-9a-fA-F]{64}/g, "");
  const stripScripts = (s) =>
    s.replace(/<script[\s\S]*?<\/script>/gi, "");
  const leakHaystacks = [
    stripAllowedHex(stripScripts(pageHtml)),
    stripAllowedHex(storageDump),
    stripAllowedHex(leakReportHtml),
  ];
  for (const secret of leakSecrets) {
    if (leakHaystacks.some((h) => h.includes(secret)))
      throw new Error("Public leak detected: " + secret);
  }
  const leakScan = {
    checked: leakSecrets,
    sources: ["page-content", "web-storage", "demo-report"],
    clean: true,
  };
  if (errors.length) throw new Error(errors.join("\n"));
  return {
    publicJourney: true,
    exactInspector: exact,
    mobileWidths: [320, 390, 430],
    privateDenied: denied,
    offlineRetention: true,
    reloadClearsAnswer: true,
    pageErrors: errors,
    transient503Retried,
    leakScan,
  };
}
