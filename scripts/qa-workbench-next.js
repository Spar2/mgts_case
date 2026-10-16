// Explicit isolated real-data QA; never writes to the working pilot journal.
async (page) => {
  const origin = new URL(page.url()).origin;
  if (origin !== "http://127.0.0.1:8091")
    throw new Error("Isolated QA origin required");
  const health=await page.evaluate(async()=>(await fetch("/api/v1/health")).json());
  if(!health.qa_mode)throw new Error("Explicit DTS_QA_MODE=isolated required; refuses to write to a normal local server");
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  await page.goto(origin + "/demo?package=real-data-v2&candidate=review-00000");
  await page.locator(".region-panel .heat-figure canvas").waitFor();
  const packages = await page.evaluate(async () =>
    (await fetch("/api/v1/packages")).json(),
  );
  const pack = packages.find((p) => p.id === "real-data-v2");
  if (pack.cards !== 340 || pack.shortlist.length !== 24)
    throw new Error("Pack cardinality");
  await page
    .getByRole("button", { name: "Сравнить обработку", exact: true })
    .click();
  await page
    .getByRole("heading", {
      name: "Обработанный профиль · та же область и шкала",
      exact: true,
    })
    .waitFor();
  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name: "Отчёт HTML", exact: true }).click();
  const report = await downloadPromise;
  await report.saveAs(".runtime/next-qa/evidence-report.html");
  await page
    .getByRole("button", { name: "Сохранить участок", exact: true })
    .click();
  await page
    .getByText("Закладка сохранена сервером", { exact: true })
    .waitFor();
  await page
    .getByRole("link", {
      name: "Открыть поле и исторический replay →",
      exact: true,
    })
    .click();
  await page.locator("#row").waitFor();
  const selectedRow = Number(new URL(page.url()).searchParams.get("row"));
  await page.waitForFunction(
    (row) => Number(document.querySelector("#row").value) === row,
    selectedRow,
  );
  await page.reload();
  await page.waitForFunction(
    (row) => Number(document.querySelector("#row")?.value) === row,
    selectedRow,
  );
  const widths = [];
  for (const width of [320, 390, 430, 1024, 1440]) {
    await page.setViewportSize({ width, height: 900 });
    await page.waitForFunction(
      () => document.documentElement.scrollWidth <= innerWidth,
      undefined,
      { timeout: 5000 },
    );
    widths.push(
      await page.evaluate(() => ({
        viewport: innerWidth,
        document: document.documentElement.scrollWidth,
      })),
    );
  }
  await page.goto(origin + "/review?package=real-data-v2&card=review-00000");
  await page
    .getByRole("textbox", { name: "Имя эксперта", exact: true })
    .waitFor();
  const reviewer = "QA ONLY — next " + Date.now();
  await page
    .getByRole("textbox", { name: "Имя эксперта", exact: true })
    .fill(reviewer);
  await page
    .getByRole("textbox", { name: "Имя эксперта", exact: true })
    .press("Tab");
  await page.waitForFunction(
    (name) => sessionStorage.getItem("dts-reviewer") === name,
    reviewer,
  );
  await page.waitForFunction(
    (name) => document.querySelector("#reviewer")?.value === name,
    reviewer,
  );
  await page.locator('label[for="answer-unknown"]').click();
  await page
    .getByLabel(
      "Это доверенное устройство: сохранять черновики после обновления",
      { exact: true },
    )
    .check();
  const notes =
    "QA ONLY: техническая проверка восстановления. Не экспертная разметка.";
  await page
    .getByRole("textbox", { name: "Обоснование по сигналу", exact: true })
    .fill(notes);
  await page.waitForFunction(async () => {
    const db = await new Promise((resolve, reject) => {
      const r = indexedDB.open("dts-workbench-drafts");
      r.onsuccess = () => resolve(r.result);
      r.onerror = () => reject(r.error);
    });
    const rows = await new Promise((resolve) => {
      const r = db.transaction("drafts").objectStore("drafts").getAll();
      r.onsuccess = () => resolve(r.result);
    });
    db.close();
    return rows.some((d) =>
      d.notes.includes("техническая проверка восстановления"),
    );
  });
  await page.reload();
  await page.waitForFunction(
    (text) => document.querySelector("#notes")?.value === text,
    notes,
  );
  await page
    .getByRole("button", { name: "Сохранить ответ", exact: true })
    .click();
  await page.getByText("Сохранено · ревизия 1", { exact: true }).waitFor();
  const independentGate = await page.evaluate(
    async () =>
      (
        await fetch(
          "/api/v1/queue/review-00000/methods?package=real-data-v2&independent=true&reviewer=Another-QA-subject",
        )
      ).status,
  );
  if (independentGate !== 403)
    throw new Error("Other participant sees methods");
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("button", { name: "1 · Сигнал", exact: true }).click();
  if (await page.locator("#review-decision").isVisible())
    throw new Error("Mobile decision visible in signal step");
  await page.getByRole("button", { name: "2 · Ответ", exact: true }).click();
  if (!(await page.locator("#review-decision").isVisible()))
    throw new Error("Mobile decision hidden");
  await page.goto(origin + "/runs");
  await page
    .getByRole("heading", {
      name: "Запуски и происхождение результатов",
      exact: true,
    })
    .waitFor();
  await page.waitForFunction(() =>
    document.querySelector(".run-list")?.textContent?.includes("median_x3"),
  );
  if (errors.length) throw new Error(errors.join("\n"));
  return {
    pack: {
      cards: pack.cards,
      shortlist: pack.shortlist.length,
      verifiedNativeCells: pack.verified_native_cells,
    },
    widths,
    restoredRow: selectedRow,
    independentGate,
    download: report.suggestedFilename(),
    runtimeErrors: errors,
  };
}
