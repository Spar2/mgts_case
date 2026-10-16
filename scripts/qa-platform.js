// Writes only technical records on the explicitly isolated QA backend.
async (page) => {
  const origin = new URL(page.url()).origin;
  const health = await page.evaluate(async () =>
    (await fetch("/api/v1/health")).json(),
  );
  if (origin !== "http://127.0.0.1:8091" || !health.qa_mode)
    throw new Error("Isolated QA required");
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  let stage = "load-candidate";
  const statuses = [];
  page.on("response", (response) => {
    const path = new URL(response.url()).pathname;
    if (path.startsWith("/api/")) {
      statuses.push({ path, status: response.status() });
      if (statuses.length > 30) statuses.shift();
    }
  });
  try {
    const packages = await page.evaluate(async () =>
      (await fetch("/api/v1/packages")).json(),
    );
    const packageId = packages.some((p) => p.id === "real-data-v2")
      ? "real-data-v2"
      : "general";
    await page.goto(origin + "/candidates?package=" + packageId);
    await page.locator(".region-panel .heat-figure canvas").waitFor();
    await page
      .getByRole("textbox", { name: "Название закладки", exact: true })
      .fill("QA: паспорт");
    await page
      .getByRole("button", { name: "Сохранить паспорт и ссылку", exact: true })
      .click();
    const saved = page.getByRole("link", {
      name: "Открыть сохранённый паспорт →",
      exact: true,
    });
    await saved.waitFor();
    const link = await saved.getAttribute("href");
    await page.goto(link);
    stage = "restore-passport-title";
    await page.waitForFunction(
      () => document.querySelector("#bookmark-title")?.value === "QA: паспорт",
    );
    const widths = [];
    for (const width of [320, 390, 430, 1024, 1280, 1281, 1440]) {
      stage = "viewport-" + width;
      await page.setViewportSize({ width, height: 900 });
      await page.waitForFunction(
        () => document.documentElement.scrollWidth <= innerWidth,
      );
      const menu = page.getByRole("button", { name: "Меню", exact: true });
      if (await menu.isVisible()) {
        stage = "navigation-open-" + width;
        await menu.click();
        await page.locator('.shell-tools > summary').click();
        await page
          .getByRole("navigation", { name: "Основная навигация" })
          .getByRole("link", { name: "Аккаунт", exact: true })
          .waitFor();
        await page.waitForFunction(
          () => document.documentElement.scrollWidth <= innerWidth,
        );
        stage = "navigation-account-click-" + width;
        await page
          .getByRole("navigation", { name: "Основная навигация" })
          .getByRole("link", { name: "Аккаунт", exact: true })
          .click();
        await page
          .getByRole("heading", { name: "Аккаунт и безопасность", exact: true })
          .waitFor();
        await page.waitForFunction(
          () => document.documentElement.scrollWidth <= innerWidth,
        );
        await page.goto(link);
        await page.waitForFunction(
          () => document.querySelector("#bookmark-title")?.value === "QA: паспорт",
        );
      }
      widths.push(width);
    }
    stage = "save-review-and-conflict";
    const me = await page.evaluate(async () =>
      (await fetch("/api/v2/auth/me")).json(),
    );
    const suffix =
      "?package=" +
      packageId +
      "&independent=true&reviewer=" +
      encodeURIComponent(me.display_name);
    const q = await page.evaluate(
      async (s) => (await fetch("/api/v1/queue" + s)).json(),
      suffix,
    );
    const card = q.cards[0],
      baseline = q.annotations[card.review_id]?.revision ?? 0;
    const second = await page.context().newPage();
    try {
      for (const tab of [page, second]) {
        await tab.setViewportSize({ width: 1440, height: 900 });
        await tab.goto(
          origin + "/review?package=" + packageId + "&card=" + card.review_id,
        );
        await tab
          .getByRole("heading", { name: "Ваше решение", exact: true })
          .waitFor();
        if (
          !(await tab
            .getByRole("textbox", { name: "Имя эксперта", exact: true })
            .evaluate((e) => e.readOnly))
        )
          throw new Error("Author editable");
        await tab.locator('label[for="answer-unknown"]').click();
        stage = "review-note-entry";
        await tab
          .getByRole("textbox", { name: "Пояснение по сигналу — необязательно", exact: true })
          .fill("QA ONLY: технический ответ, не разметка. " + Date.now());
      }
      stage = "review-network-failure";
      await page.route("**/api/v1/reviews**", (route) => route.abort("failed"));
      await page
        .getByRole("button", { name: "Сохранить ответ", exact: true })
        .click();
      await page.getByText("Ответ не сохранён", { exact: true }).waitFor();
      await page.unroute("**/api/v1/reviews**");
      stage = "review-save-retry";
      await page
        .getByRole("button", { name: "Сохранить ответ", exact: true })
        .click();
      await page
        .getByText("Сохранено · ревизия " + (baseline + 1), { exact: true })
        .waitFor();
      await second
        .getByRole("button", { name: "Сохранить ответ", exact: true })
        .click();
      stage = "review-two-tab-conflict";
      await second
        .getByRole("heading", {
          name: "Ответ изменён в другой вкладке",
          exact: true,
        })
        .waitFor();
    } finally {
      await second.close();
    }
    stage = "assign-expert";
    await page.goto(origin + "/system");
    await page
      .getByRole("heading", {
        name: "Работа с исследовательскими окнами",
        exact: true,
      })
      .waitFor();
    await page
      .getByRole("combobox", { name: "Эксперт для назначения", exact: true })
      .selectOption(me.id);
    await page
      .getByRole("button", { name: "Назначить", exact: true })
      .first()
      .click();
    await page.waitForFunction(() =>
      Array.from(document.querySelectorAll(".job-row")).some((e) =>
        e.textContent.includes("Назначено"),
      ),
    );
    if (errors.length) throw new Error(errors.join("\n"));
    return {
      passport: true,
      authenticated_review: true,
      network_failure: true,
      two_tab_conflict: true,
      assignment: true,
      widths,
      errors,
    };
  } catch (error) {
    const state = await page
      .evaluate(() => ({
        route: location.pathname,
        width: innerWidth,
        scrollWidth: document.documentElement.scrollWidth,
        passportTitle: document.querySelector("#bookmark-title")?.value ?? null,
        assignedRows: Array.from(document.querySelectorAll(".job-row")).filter(
          (e) => e.textContent.includes("Назначено"),
        ).length,
        overflow: Array.from(document.querySelectorAll("body *"))
          .map((e) => {
            const r = e.getBoundingClientRect();
            return {
              tag: e.tagName,
              id: e.id,
              classes: String(e.className).slice(0, 160),
              left: r.left,
              right: r.right,
              width: r.width,
            };
          })
          .filter(
            (e) => e.width > 0 && (e.right > innerWidth + 1 || e.left < -1),
          )
          .slice(0, 20),
      }))
      .catch(() => ({ route: "unavailable" }));
    throw new Error(
      "QA_PLATFORM_DIAGNOSTIC " +
        JSON.stringify({
          stage,
          errorType: error.name,
          state,
          statuses: statuses.slice(-30),
        }),
    );
  }
}
