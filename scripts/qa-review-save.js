// Destructive QA for an explicitly marked, isolated journal only.
// It creates technical revisions, never an expert assessment of the signal.
async (page) => {
  const origin = new URL(page.url()).origin;
  const queue = await page.evaluate(async () =>
    (await fetch("/api/v1/queue")).json(),
  );
  const marker = Object.values(queue.annotations).find(
    (a) => a.reviewer === "QA ONLY — отдельный тестовый журнал",
  );
  if (origin !== "http://127.0.0.1:8091" || !marker)
    throw new Error(
      "Requires the marked isolated QA journal on 8091; refuses to write elsewhere",
    );
  const card = queue.cards.find((c) => c.review_id === marker.review_id);
  const reviewer = "QA ONLY — отдельный тестовый журнал";
  const ownQuery = "?independent=true&reviewer=" + encodeURIComponent(reviewer);
  const ownQueue = await page.evaluate(
    async (suffix) => (await fetch("/api/v1/queue" + suffix)).json(),
    ownQuery,
  );
  const baseline = ownQueue.annotations[card.review_id]?.revision ?? 0;
  const url =
    origin +
    "/review?source=" +
    encodeURIComponent(card.source) +
    "&card=" +
    encodeURIComponent(card.review_id);
  const second = await page.context().newPage();
  try {
    for (const tab of [page, second]) {
      await tab.setViewportSize({ width: 1440, height: 900 });
      await tab.bringToFront();
      await tab.goto(url);
      await tab
        .getByRole("heading", { name: "Ваше решение", exact: true })
        .waitFor();
      await tab
        .getByRole("textbox", { name: "Имя эксперта", exact: true })
        .fill("QA ONLY — отдельный тестовый журнал");
      await tab
        .getByRole("textbox", { name: "Имя эксперта", exact: true })
        .press("Tab");
      await tab.waitForFunction(
        (name) => sessionStorage.getItem("dts-reviewer") === name,
        reviewer,
      );
      await tab.locator('label[for="answer-unknown"]').click();
    }
    const firstNotes =
      "QA: проверка сети, сохранения и конфликта. Это технический тест, а не экспертная оценка; журнал изолирован. Run " +
      Date.now();
    const secondNotes =
      firstNotes + " Второй черновик остаётся в форме при конфликте.";
    await page
      .getByRole("textbox", { name: "Обоснование по сигналу", exact: true })
      .fill(firstNotes);
    await second
      .getByRole("textbox", { name: "Обоснование по сигналу", exact: true })
      .fill(secondNotes);
    await page.route("**/api/v1/reviews**", (route) => route.abort("failed"));
    await page.bringToFront();
    await page
      .getByRole("button", { name: "Сохранить ответ", exact: true })
      .click();
    await page.getByText("Ответ не сохранён", { exact: true }).waitFor();
    if (
      (await page
        .getByRole("textbox", { name: "Обоснование по сигналу", exact: true })
        .inputValue()) !== firstNotes
    )
      throw new Error("Draft lost on network failure");
    await page.unroute("**/api/v1/reviews**");
    await page
      .getByRole("button", { name: "Сохранить ответ", exact: true })
      .click();
    await page
      .getByText("Сохранено · ревизия " + (baseline + 1), { exact: true })
      .waitFor();
    await second.bringToFront();
    await second
      .getByRole("button", { name: "Сохранить ответ", exact: true })
      .click();
    await second
      .getByRole("heading", {
        name: "Ответ изменён в другой вкладке",
        exact: true,
      })
      .waitFor();
    if (
      (await second
        .getByRole("textbox", { name: "Обоснование по сигналу", exact: true })
        .inputValue()) !== secondNotes
    )
      throw new Error("Draft lost on conflict");
    await second
      .getByRole("button", {
        name: "Подготовить мой вариант к повторной отправке",
        exact: true,
      })
      .click();
    await second
      .getByRole("button", { name: "Сохранить ответ", exact: true })
      .click();
    await second
      .getByText("Сохранено · ревизия " + (baseline + 2), { exact: true })
      .waitFor();
    await second.reload();
    await second
      .getByText("Сохранено · ревизия " + (baseline + 2), { exact: true })
      .waitFor();
    await second
      .getByRole("button", { name: "Раскрыть результаты методов", exact: true })
      .click();
    const prepared = await second.evaluate(
      async ({ id, suffix }) =>
        (await fetch("/api/v1/queue/" + id + "/methods" + suffix)).json(),
      { id: card.review_id, suffix: ownQuery },
    );
    if (prepared.methods.length) {
      await second
        .getByRole("combobox", {
          name: "Гипотеза метода на исходной сетке",
          exact: true,
        })
        .waitFor();
      const methods = await second
        .getByRole("combobox", {
          name: "Гипотеза метода на исходной сетке",
          exact: true,
        })
        .locator("option")
        .allTextContents();
      await second
        .getByRole("combobox", {
          name: "Гипотеза метода на исходной сетке",
          exact: true,
        })
        .selectOption({ label: methods[1] });
      await second
        .getByText("Гипотезы на исходной сетке", { exact: false })
        .waitFor();
    } else
      await second
        .getByText("В этой очереди нет подготовленных результатов методов.", {
          exact: false,
        })
        .waitFor();
    const exported = await second.evaluate(
      async (suffix) => (await fetch("/api/v1/export" + suffix)).json(),
      ownQuery,
    );
    return {
      networkDraftPreserved: true,
      conflictDraftPreserved: true,
      revision: baseline + 2,
      reloadPersisted: true,
      methodCount: prepared.methods.length,
      exportVersion: exported.version,
    };
  } finally {
    await second.close();
  }
}
