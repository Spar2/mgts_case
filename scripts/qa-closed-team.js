// Runs only against the isolated authenticated QA server; never a team account.
async (page) => {
  const origin = new URL(page.url()).origin;
  const health = await (await page.request.get(origin + "/api/v1/health")).json();
  if (origin !== "http://127.0.0.1:8091" || !health.qa_mode || health.access_mode !== "closed-team") throw new Error("Isolated closed-team QA required");
  const browser = page.context().browser();
  if (!browser) throw new Error("Browser unavailable");
  const anonymous = await browser.newContext({ viewport: { width: 390, height: 844 }, reducedMotion: "reduce" });
  const visitor = await anonymous.newPage();
  const errors = [];
  visitor.on("pageerror", (e) => errors.push(String(e)));
  try {
    const landingApiRequests = [];
    const observeLanding = (request) => {
      if (new URL(request.url()).pathname.startsWith("/api/")) landingApiRequests.push(request.url());
    };
    visitor.on("request", observeLanding);
    await visitor.goto(origin + "/");
    await visitor.getByRole("heading", { name: "Единая аналитическая платформа сенсорных данных", exact: true }).waitFor();
    if (new URL(visitor.url()).pathname !== "/") throw new Error("Landing redirected to login");
    if (await visitor.locator("#main-navigation").count()) throw new Error("Landing exposed work navigation");
    for (const width of [320, 390, 430, 1440]) {
      await visitor.setViewportSize({ width, height: 844 });
      if (await visitor.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1)) throw new Error("Landing overflow: " + width);
      await visitor.getByRole("link", { name: "Войти в платформу", exact: true }).waitFor();
    }
    for (const asset of ["/landing/mgts-original.jpg", "/landing/larp-corp-original.jpg", "/landing/visual-data.json"]) {
      const response = await anonymous.request.get(origin + asset);
      if (response.status() !== 200 || response.headers()["content-type"].includes("text/html")) throw new Error("Published landing asset unavailable: " + asset);
    }
    if (landingApiRequests.length) throw new Error("Landing fetched private APIs");
    visitor.off("request", observeLanding);
    await visitor.getByRole("link", { name: "Войти в платформу", exact: true }).focus();
    await visitor.keyboard.press("Enter");
    await visitor.getByRole("heading", { name: "Единая аналитическая платформа сенсорных данных · вход", exact: true }).waitFor();
    for (const path of ["/try", "/request-access", "/research", "/works", "/monitoring", "/laboratory", "/cases", "/demo", "/account"]) {
      await visitor.goto(origin + path);
      await visitor.getByRole("heading", { name: "Единая аналитическая платформа сенсорных данных · вход", exact: true }).waitFor();
      if (new URL(visitor.url()).pathname !== "/login") throw new Error("Anonymous page not redirected: " + path);
      if (await visitor.locator("#main-navigation").count()) throw new Error("Workspace navigation leaked");
    }
    for (const path of ["/api/v1/records", "/api/v1/queue", "/api/v2/investigations", "/api/v2/result-runs", "/api/v2/cases", "/api/v2/accounts", "/api/v2/jobs", "/api/public/v1/package", "/api/v2/cases/missing/bundle"]) {
      if ((await anonymous.request.get(origin + path)).status() !== 401) throw new Error("Anonymous API opened: " + path);
    }
    for (const width of [320, 390, 430, 1440]) {
      await visitor.setViewportSize({ width, height: 844 });
      const overflow = await visitor.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1);
      if (overflow) throw new Error("Login overflow: " + width);
    }
    const session = await (await page.request.get(origin + "/api/v1/session")).json();
    const csrf = { "X-Workbench-Token": session.mutation_token };
    const invitation = await page.request.post(origin + "/api/v2/invitations", { headers: csrf, data: { role: "viewer" } });
    if (invitation.status() !== 200) throw new Error("Invitation refused");
    const invite = await invitation.json();
    const username = "qa-closed-" + Date.now();
    const password = "QA-closed-isolated-" + Date.now() + "-A1b2C3";
    await visitor.goto(origin + invite.link);
    await visitor.locator("#activate-login").fill(username);
    await visitor.locator("#activate-name").fill("QA closed viewer");
    await visitor.locator("#activate-password").fill(password);
    await visitor.locator("#activate-confirm").fill(password);
    if (new URL(visitor.url()).hash) throw new Error("Invitation secret not removed");
    await visitor.getByRole("button", { name: "Установить пароль", exact: true }).click();
    await visitor.getByRole("status").waitFor();
    await visitor.goto(origin + "/login?next=" + encodeURIComponent("/research?record=5"));
    await visitor.locator("#login-name").fill(username);
    await visitor.locator("#login-password").fill(password);
    await visitor.getByRole("button", { name: "Войти", exact: true }).click();
    await visitor.waitForURL(origin + "/research?record=5");
    await visitor.locator("#main-navigation").waitFor();
    const me = await (await anonymous.request.get(origin + "/api/v2/auth/me")).json();
    if (me.role !== "viewer" || me.username !== username) throw new Error("Account identity not retained");
    if ((await anonymous.request.get(origin + "/api/v1/records")).status() !== 200) throw new Error("Member cannot read archive");
    const viewerCsrf = { "X-Workbench-Token": me.csrf };
    if ((await anonymous.request.post(origin + "/api/v2/jobs", { headers: viewerCsrf, data: { kind: "analysis" } })).status() !== 403) throw new Error("Viewer can submit jobs");
    const secondTab = await anonymous.newPage();
    await secondTab.goto(origin + "/account");
    await secondTab.locator("#main-navigation").waitFor({ state: "attached" });
    await visitor.getByRole("button", { name: "Выйти", exact: true }).click();
    await visitor.getByRole("heading", { name: "Единая аналитическая платформа сенсорных данных · вход", exact: true }).waitFor();
    await secondTab.getByRole("heading", { name: "Единая аналитическая платформа сенсорных данных · вход", exact: true }).waitFor({ timeout: 5000 });
    if (await secondTab.locator("#main-navigation").count()) throw new Error("Other tab retained account UI after logout");
    await secondTab.close();
    if ((await anonymous.request.get(origin + "/api/v1/records")).status() !== 401) throw new Error("Logged-out session still reads archive");
    if (await visitor.evaluate(() => sessionStorage.getItem("dts-reviewer"))) throw new Error("Old reviewer retained");
    if (errors.length) throw new Error("Unhandled browser error: " + errors[0]);
    return { public_landing: true, landing_keyboard: true, landing_mobile: true, closed_pages: true, anonymous_api_denied: true, mobile_login: true, invitation_login_return: true, viewer_readonly: true, logout_denied: true, cross_tab_logout: true };
  } finally {
    await anonymous.close();
  }
}
