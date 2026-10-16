// Account writes are limited to the explicitly isolated PostgreSQL QA server.
async (page) => {
  const origin = new URL(page.url()).origin;
  const health = await page.evaluate(async () =>
    (await fetch("/api/v1/health")).json(),
  );
  if (
    !["http://127.0.0.1:8091", "https://localhost:8443"].includes(origin) ||
    (!health.qa_mode && origin !== "https://localhost:8443") ||
    !health.auth_enabled
  )
    throw new Error("Isolated authenticated QA required");
  let forcingOutage = false;
  const errors = [],
    expectedNetworkWarnings = [];
  page.on("pageerror", (error) => {
    // WebKit emits inspector access-control errors for route-aborted fetches,
    // even when the query Promise handles the rejection. Keep these explicit.
    if (forcingOutage && String(error).includes("Fetch API cannot load"))
      expectedNetworkWarnings.push(String(error));
    else errors.push(error.stack || String(error));
  });
  const username = "qa-browser-" + Date.now(),
    display = "QA ONLY — " + username;
  const password = "QA-isolated-" + Date.now() + "-A1b2C3d4";
  await page.goto(origin + "/account");
  await page
    .getByRole("combobox", { name: "Роль участника", exact: true })
    .selectOption("expert");
  await page
    .getByRole("button", { name: "Создать приглашение", exact: true })
    .click();
  await page.locator("#invite-link").waitFor();
  const invitation = await page.locator("#invite-link").inputValue();
  const browser = page.context().browser();
  if (!browser) throw new Error("Browser unavailable");
  const expertContext = await browser.newContext({
    viewport: { width: 390, height: 844 },
    ignoreHTTPSErrors: true,
  });
  const expert = await expertContext.newPage();
  try {
    await expert.goto(invitation);
    await expert.locator("#activate-login").fill(username);
    await expert.locator("#activate-name").fill(display);
    await expert.locator("#activate-password").fill(password);
    await expert.locator("#activate-confirm").fill(password);
    if (new URL(expert.url()).hash)
      throw new Error("Invitation secret remains in URL");
    await expert
      .getByRole("button", { name: "Установить пароль", exact: true })
      .click();
    await expert.getByText("Пароль установлен.", { exact: false }).waitFor();
    await page.goto(origin + "/system");
    const participant = page.locator(".job-row").filter({ hasText: display });
    await participant.waitFor();
    await expert.goto(origin + "/review");
    await expert
      .getByRole("textbox", { name: "Логин", exact: true })
      .fill(username);
    await expert
      .getByRole("textbox", { name: "Пароль", exact: true })
      .fill(password);
    await expert.getByRole("button", { name: "Войти", exact: true }).click();
    await expert.getByRole("button", { name: "Выйти", exact: true }).waitFor();
    await page.setViewportSize({ width: 390, height: 844 });
    await participant
      .getByRole("button", { name: "Отключить доступ", exact: true })
      .click();
    await page
      .getByRole("dialog")
      .getByRole("button", { name: "Отключить доступ", exact: true })
      .click();
    await participant.getByText("Доступ отключён", { exact: false }).waitFor();
    const revoked = await expert.evaluate(
      async () => (await fetch("/api/v2/auth/me")).status,
    );
    if (revoked !== 401) throw new Error("Revoked session survived");
    await expert.goto(origin + "/review");
    await expert
      .getByRole("heading", { name: "Единая аналитическая платформа сенсорных данных · вход", exact: true })
      .waitFor();
    await participant
      .getByRole("button", { name: "Восстановить доступ", exact: true })
      .click();
    await page
      .getByRole("dialog")
      .getByRole("button", { name: "Восстановить доступ", exact: true })
      .click();
    await participant.getByText("Доступ открыт", { exact: false }).waitFor();
    const stillRevoked = await expert.evaluate(
      async () => (await fetch("/api/v2/auth/me")).status,
    );
    if (stillRevoked !== 401) throw new Error("Old cookie was resurrected");
    forcingOutage = true;
    await page.route("**/api/v1/health", (route) => route.abort("failed"));
    await page.goto(origin + "/system");
    await page
      .getByRole("heading", { name: "Сервер временно недоступен", exact: true })
      .waitFor();
    await page.unroute("**/api/v1/health");
    await page
      .getByRole("button", { name: "Повторить подключение", exact: true })
      .click();
    await page
      .getByRole("heading", { name: "Система и путь измерения", exact: true })
      .waitFor();
    forcingOutage = false;
    if (errors.length) throw new Error(errors.join("\n"));
    return {
      invitation: true,
      activation: true,
      fragment_removed: true,
      mobile_login: true,
      disable_revokes: true,
      enable_keeps_old_cookie_revoked: true,
      startup_retry: true,
      expectedNetworkWarnings,
      errors,
    };
  } finally {
    await expertContext.close();
  }
}
