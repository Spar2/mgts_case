// Playwright CLI run-code function for /pipeline.
// Isolated QA only. Prints no cookies, passwords, or tokens.
async (page) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(String(error)));
  const fail = (message) => {
    if (errors.length) throw new Error(message + "\n" + JSON.stringify(errors));
    throw new Error(message);
  };
  const origin = new URL(page.url()).origin;
  const healthResponse = await page.request.get(origin + "/api/v1/health");
  if (!healthResponse.ok()) fail("Health check failed");
  const health = await healthResponse.json();
  if (health.qa_mode !== true) fail("Isolated QA only");

  const limited = (region) => {
    if (!region) return false;
    const rows = Number(region.row_end) - Number(region.row_start);
    const cols = Number(region.column_end) - Number(region.column_start);
    return rows >= 1 && rows <= 32 && cols >= 2 && cols <= 64;
  };
  const assertLimited = (region, recordId) => {
    if (!region || String(region.record_id) !== String(recordId))
      fail("Calculation is not for the opened record");
    if (!limited(region)) fail("Calculation exceeded the limited window");
  };
  const chooseBounds = (shape) => {
    const rows = Number(shape?.[0]);
    const cols = Number(shape?.[1]);
    if (!(rows >= 1 && cols >= 2)) return null;
    let rowEnd = Math.min(rows, 32);
    let columnStart = 0;
    let columnEnd = Math.min(cols, 64);
    if (cols >= 3) {
      columnStart = 1;
      columnEnd = Math.min(cols, columnStart + 64);
    } else if (rowEnd > 1) rowEnd = 1;
    const region = {
      row_start: 0,
      row_end: rowEnd,
      column_start: columnStart,
      column_end: columnEnd,
    };
    if (!limited(region)) return null;
    return [
      String(region.row_start),
      String(region.row_end),
      String(region.column_start),
      String(region.column_end),
    ];
  };

  const recordsResponse = await page.request.get(origin + "/api/v1/records");
  if (!recordsResponse.ok()) fail("Records were not readable");
  const records = await recordsResponse.json();
  const ranked = [...records].sort((left, right) => {
    const rank = (id) => (String(id) === "6" ? 0 : String(id) === "5" ? 1 : 2);
    return rank(left?.id) - rank(right?.id);
  });
  let detail = null;
  let bounds = null;
  for (const item of ranked) {
    if (!item || !Array.isArray(item.shape) || item.shape[1] < 2) continue;
    const response = await page.request.get(
      origin + "/api/v1/records/" + encodeURIComponent(item.id),
    );
    if (!response.ok()) continue;
    const full = await response.json();
    const next = chooseBounds(full.shape);
    if (!next || !/^[0-9a-f]{64}$/.test(full.source_sha256 ?? "")) continue;
    detail = full;
    bounds = next;
    break;
  }
  if (!detail || !bounds) fail("No real record with at least two columns");

  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto(
    origin +
      "/pipeline?" +
      new URLSearchParams({
        record: String(detail.id),
        source_sha256: detail.source_sha256,
      }),
  );
  await page.getByRole("heading", { name: "Пайплайн", exact: true }).waitFor();
  await page.waitForFunction(() => {
    const field = document.querySelector("#pipeline-bound-0");
    const button = [...document.querySelectorAll("button")].find(
      (item) => item.textContent.trim() === "Задать отрезок",
    );
    return !!field && field.value !== "" && !!button && !button.disabled;
  });

  for (let index = 0; index < 4; index++) {
    const field = page.locator("#pipeline-bound-" + index);
    await field.fill(bounds[index]);
    if (index < 3) await field.press("Tab");
    await page.evaluate(
      () =>
        new Promise((resolve) =>
          requestAnimationFrame(() => requestAnimationFrame(resolve)),
        ),
    );
    for (let earlier = 0; earlier <= index; earlier++) {
      const value = await page.locator("#pipeline-bound-" + earlier).inputValue();
      if (value !== bounds[earlier])
        fail("Bound " + earlier + " disappeared after editing bound " + index);
    }
  }
  await page.getByRole("button", { name: "Задать отрезок", exact: true }).click();
  await page.waitForFunction(
    (expected) => {
      const url = new URL(location.href);
      return ["r0", "r1", "c0", "c1"].every(
        (key, index) => url.searchParams.get(key) === expected[index],
      );
    },
    bounds,
  );
  for (let index = 0; index < 4; index++) {
    if ((await page.locator("#pipeline-bound-" + index).inputValue()) !== bounds[index])
      fail("Bound " + index + " disappeared after applying the window");
  }

  await page
    .waitForFunction(
      () =>
        [...document.querySelectorAll("button")].some(
          (button) => button.textContent.trim() === "Сохранить настройки",
        ),
      undefined,
      { timeout: 8000 },
    )
    .catch(() => {});
  const nameId = await page.evaluate(() => {
    const fields = [
      ...document.querySelectorAll("input, textarea, [role='textbox']"),
    ];
    for (const node of fields) {
      if (/^pipeline-bound-|^pipeline-row$|^pipeline-column$/.test(node.id))
        continue;
      const label = node.id
        ? (document.querySelector(`label[for="${CSS.escape(node.id)}"]`)
            ?.textContent ?? "")
        : "";
      const labelledBy = (node.getAttribute("aria-labelledby") ?? "")
        .split(/\s+/)
        .map((id) => document.getElementById(id)?.textContent ?? "")
        .join(" ");
      const blob = [
        node.id,
        node.getAttribute("name") ?? "",
        label,
        node.closest("label")?.textContent ?? "",
        node.getAttribute("aria-label") ?? "",
        node.getAttribute("placeholder") ?? "",
        labelledBy,
      ].join(" ");
      if (
        /названи|settings[-_ ]?name|setting[-_ ]?title/i.test(blob) &&
        !/строк|столб|bound/i.test(blob)
      ) {
        if (!node.id) node.id = "qa-pipeline-settings-name";
        return node.id;
      }
    }
    return "";
  });
  const save = page.getByRole("button", {
    name: "Сохранить настройки",
    exact: true,
  });
  const name = nameId ? page.locator("#" + nameId) : page.locator("#missing");
  if (
    (await save.count()) === 0 ||
    !nameId ||
    (await name.count()) === 0 ||
    !(await save.isVisible()) ||
    !(await name.isVisible())
  )
    fail("pipeline settings controls missing");
  const title = "QA pipeline window " + Date.now();
  await name.fill(title);
  if ((await name.inputValue()) !== title) fail("Settings name was not kept");
  if (!(await save.isEnabled())) fail("Settings save stayed disabled");
  const savedPromise = page.waitForResponse(
    (response) => {
      const path = new URL(response.url()).pathname.replace(/\/$/, "");
      const method = response.request().method();
      return (
        (path === "/api/v2/pipeline-settings" ||
          /^\/api\/v2\/pipeline-settings\/[^/]+$/.test(path)) &&
        (method === "POST" || method === "PATCH")
      );
    },
    { timeout: 20000 },
  );
  await save.click();
  const savedResponse = await savedPromise.catch(() => null);
  if (!savedResponse) fail("Settings save did not reach the server");
  if (savedResponse.status() >= 400)
    fail("Settings save failed: " + savedResponse.status());
  let savedPayload = null;
  try {
    savedPayload = savedResponse.request().postDataJSON();
  } catch {
    savedPayload = null;
  }
  if (!savedPayload || savedPayload.title !== title)
    fail("Saved settings name did not match the visible control");
  const savedBody = await savedResponse.json().catch(() => null);
  if (!savedBody || typeof savedBody.id !== "string" || !savedBody.id)
    fail("Settings save returned no id");
  if ((await name.inputValue()) !== title)
    fail("Settings name disappeared after save");

  const checks = page.getByRole("button", { name: "Проверить", exact: true });
  let preflight = "absent";
  if (await checks.count()) {
    const check = checks.first();
    let jobPosts = 0;
    let watchJobs = true;
    const onRequest = (request) => {
      if (!watchJobs) return;
      try {
        const url = new URL(request.url());
        if (
          request.method() === "POST" &&
          url.pathname.replace(/\/$/, "") === "/api/v2/analysis-jobs"
        )
          jobPosts += 1;
      } catch {
        /* Ignore a malformed request URL. */
      }
    };
    page.on("request", onRequest);
    try {
      const before = new URL(page.url()).searchParams.get("job_id");
      const beforeShown = await page.evaluate(
        () =>
          /job_id\s+[A-Za-z0-9_-]{8,128}/.exec(document.body.innerText)?.[0] ??
          "",
      );
      if (!(await check.isEnabled())) {
        await page.waitForFunction(
          () => {
            const button = [...document.querySelectorAll("button")].find(
              (item) => item.textContent.trim() === "Проверить",
            );
            return !!button && !button.disabled;
          },
          undefined,
          { timeout: 15000 },
        );
      }
      await check.click();
      await page
        .waitForResponse(
          (response) => {
            const path = new URL(response.url()).pathname.replace(/\/$/, "");
            return (
              path === "/api/v2/analysis-preflight" ||
              (path === "/api/v2/analysis-jobs" &&
                response.request().method() === "POST")
            );
          },
          { timeout: 8000 },
        )
        .catch(() => null);
      const after = new URL(page.url()).searchParams.get("job_id");
      const afterShown = await page.evaluate(
        () =>
          /job_id\s+[A-Za-z0-9_-]{8,128}/.exec(document.body.innerText)?.[0] ??
          "",
      );
      if (after !== before || jobPosts > 0 || (afterShown && afterShown !== beforeShown))
        fail("Проверить created a job id");
      preflight = "unchanged";
    } finally {
      watchJobs = false;
    }
  }

  const readJobs = async () => {
    const items = [];
    for (let offset = 0, guard = 0; guard < 5; guard++) {
      const response = await page.request.get(
        origin + "/api/v2/jobs?limit=100&offset=" + offset,
      );
      if (!response.ok()) fail("Could not read jobs: " + response.status());
      const body = await response.json();
      items.push(...(body.items ?? []));
      if (body.next_offset == null) break;
      offset = body.next_offset;
    }
    return items.filter((job) => job.kind === "analysis_field");
  };
  const openJob = async (job) => {
    const region = job?.payload?.region;
    assertLimited(region, detail.id);
    if ((job.payload?.source_sha256 ?? "") !== detail.source_sha256)
      fail("Attached calculation is not for the opened record");
    const url = new URL(page.url());
    url.pathname = "/pipeline";
    url.searchParams.set("record", String(detail.id));
    url.searchParams.set("source_sha256", detail.source_sha256);
    url.searchParams.set("job_id", job.id);
    url.searchParams.set("r0", String(region.row_start));
    url.searchParams.set("r1", String(region.row_end));
    url.searchParams.set("c0", String(region.column_start));
    url.searchParams.set("c1", String(region.column_end));
    url.searchParams.set("row", String(region.row_start));
    url.searchParams.set("column", String(region.column_start));
    url.searchParams.set("block", "source");
    await page.goto(url.toString());
    await page.getByRole("heading", { name: "Пайплайн", exact: true }).waitFor();
    if (new URL(page.url()).searchParams.get("job_id") !== job.id)
      fail("Attached job_id was not kept");
    return job.id;
  };
  const exactJob = (jobs) =>
    jobs.find((job) => {
      const region = job.payload?.region;
      return (
        ["queued", "running", "complete"].includes(job.status) &&
        (job.payload?.source_sha256 ?? "") === detail.source_sha256 &&
        String(region?.record_id) === String(detail.id) &&
        String(region?.row_start) === bounds[0] &&
        String(region?.row_end) === bounds[1] &&
        String(region?.column_start) === bounds[2] &&
        String(region?.column_end) === bounds[3] &&
        limited(region)
      );
    });

  const live = [];
  for (let index = 0; index < 4; index++)
    live.push(await page.locator("#pipeline-bound-" + index).inputValue());
  if (live.join(",") !== bounds.join(",")) fail("Bounds changed before calculation");
  assertLimited(
    {
      record_id: detail.id,
      row_start: Number(live[0]),
      row_end: Number(live[1]),
      column_start: Number(live[2]),
      column_end: Number(live[3]),
    },
    detail.id,
  );

  let jobId = "";
  let calculation = "";
  const cleanCode = page.getByRole("textbox", { name: "Код вкладки Очистка" });
  if (await cleanCode.count()) {
    const text = await cleanCode.inputValue();
    if (text.includes("\nreturn None"))
      await cleanCode.fill(text.replace("\nreturn None", "\n        return None"));
  }
  const calculate = page.getByRole("button", { name: "Проверить пайплайн", exact: true });
  const enabled = await page
    .waitForFunction(
      () => {
        const button = [...document.querySelectorAll("button")].find(
          (item) => item.textContent.trim() === "Проверить пайплайн",
        );
        return !!button && !button.disabled;
      },
      undefined,
      { timeout: 20000 },
    )
    .then(() => true)
    .catch(() => false);
  if (enabled && (await calculate.isEnabled())) {
    const accepted = page.waitForResponse(
      (response) =>
        new URL(response.url()).pathname.replace(/\/$/, "") ===
          "/api/v2/analysis-jobs" && response.request().method() === "POST",
      { timeout: 30000 },
    );
    await calculate.click();
    const response = await accepted;
    if (response.status() === 202 || response.status() === 200) {
      let posted = null;
      try {
        posted = response.request().postDataJSON();
      } catch {
        posted = null;
      }
      assertLimited(posted?.region, detail.id);
      if (
        String(posted.region.row_start) !== bounds[0] ||
        String(posted.region.row_end) !== bounds[1] ||
        String(posted.region.column_start) !== bounds[2] ||
        String(posted.region.column_end) !== bounds[3]
      )
        fail("Calculation did not use the applied limited window");
      const body = await response.json();
      assertLimited(body.region, detail.id);
      if (!body.id) fail("Calculation returned no job_id");
      await page.waitForURL(
        (url) => url.searchParams.get("job_id") === body.id,
        { timeout: 30000 },
      );
      jobId = body.id;
      calculation = "started";
    } else if (response.status() === 429 || response.status() === 503) {
      const jobs = await readJobs();
      const active = jobs.find(
        (job) =>
          job.status === "queued" || job.status === "running",
      );
      const candidate =
        exactJob(jobs) ??
        (active &&
        limited(active.payload?.region) &&
        String(active.payload?.region?.record_id) === String(detail.id) &&
        (active.payload?.source_sha256 ?? "") === detail.source_sha256
          ? active
          : null);
      if (!candidate) {
        if (active && !limited(active.payload?.region))
          fail("Refusing to attach a calculation outside the limited window");
        fail("Limited calculation was not started or attached");
      }
      jobId = await openJob(candidate);
      calculation = "attached";
    } else fail("Calculation was not accepted: " + response.status());
  } else {
    const candidate = exactJob(await readJobs());
    if (!candidate) fail("Limited calculation was not started or attached");
    jobId = await openJob(candidate);
    calculation = "attached";
  }
  if (!jobId) fail("job_id missing before reload");
  await page.reload();
  await page.getByRole("heading", { name: "Пайплайн", exact: true }).waitFor();
  if (new URL(page.url()).searchParams.get("job_id") !== jobId)
    fail("job_id changed after reload");
  await page.waitForFunction(
    (id) => (document.querySelector("main.pipeline")?.innerText ?? "").includes(id),
    jobId,
    { timeout: 30000 },
  );

  const tabs = ["Исходник", "Поле обработки", "Фон и score", "Зоны"];
  for (const width of [1440, 430, 390, 320]) {
    await page.setViewportSize({ width, height: 900 });
    let baseline = null;
    for (const name of tabs) {
      await page.locator(".pipeline-chain").getByRole("tab", { name, exact: true }).click();
      const handle = await page.waitForFunction((label) => {
        const selected = [...document.querySelectorAll(".pipeline-chain [role='tab']")].find(
          (node) => node.getAttribute("aria-selected") === "true",
        );
        if (!selected || selected.textContent.trim() !== label) return null;
        return {
          scrollHeight: document.documentElement.scrollHeight,
          scrollWidth: document.documentElement.scrollWidth,
          innerWidth: window.innerWidth,
        };
      }, name);
      const layout = await handle.jsonValue();
      if (layout.scrollWidth > layout.innerWidth + 1)
        fail("Pipeline overflow at " + width + " on " + name);
      if (baseline === null) baseline = layout.scrollHeight;
      else if (layout.scrollHeight !== baseline)
        fail("Pipeline scrollHeight changed at " + width + " on " + name);
    }
  }

  await page.setViewportSize({ width: 1440, height: 900 });
  await page.locator(".pipeline-chain").getByRole("tab", { name: "Исходник", exact: true }).click();
  await page.waitForFunction(() => {
    const host = document.querySelector("#pipeline-panel .heat-canvas");
    const box = host?.getBoundingClientRect();
    const text = document.querySelector("#pipeline-panel")?.innerText ?? "";
    return !!box && box.width > 8 && box.height > 8 && !text.includes("Загрузка исходного поля");
  });
  const previousPoint = await page.evaluate(() => ({
    x: window.scrollX,
    y: window.scrollY,
    top: document.documentElement.scrollTop,
    left: document.documentElement.scrollLeft,
    row: new URL(location.href).searchParams.get("row"),
    column: new URL(location.href).searchParams.get("column"),
  }));
  const clicked = await page.evaluate(() => {
    const host = document.querySelector("#pipeline-panel .heat-canvas");
    const box = host?.getBoundingClientRect();
    if (!host || !box || box.width < 8 || box.height < 8) return false;
    host.dispatchEvent(
      new MouseEvent("click", {
        bubbles: true,
        cancelable: true,
        composed: true,
        clientX: box.left + box.width * 0.8,
        clientY: box.top + box.height * 0.25,
        view: window,
      }),
    );
    return true;
  });
  if (!clicked) fail("Pipeline field is not ready for a point");
  await page.waitForFunction(
    (previous) => {
      const params = new URL(location.href).searchParams;
      return (
        params.get("row") !== previous.row ||
        params.get("column") !== previous.column
      );
    },
    previousPoint,
    { timeout: 10000 },
  );
  await page.evaluate(
    () =>
      new Promise((resolve) =>
        requestAnimationFrame(() => requestAnimationFrame(resolve)),
      ),
  );
  const nextPoint = await page.evaluate(() => ({
    x: window.scrollX,
    y: window.scrollY,
    top: document.documentElement.scrollTop,
    left: document.documentElement.scrollLeft,
  }));
  if (
    Math.abs(nextPoint.x - previousPoint.x) > 1 ||
    Math.abs(nextPoint.y - previousPoint.y) > 1 ||
    Math.abs(nextPoint.top - previousPoint.top) > 1 ||
    Math.abs(nextPoint.left - previousPoint.left) > 1
  )
    fail("Selecting a point scrolled the document");

  let comparison = "not-available";
  const runsResponse = await page.request.get(
    origin +
      "/api/v2/result-runs?record_id=" +
      encodeURIComponent(String(detail.id)),
  );
  const runs = runsResponse.ok() ? await runsResponse.json() : [];
  const compatible = new Set(
    (Array.isArray(runs) ? runs : [])
      .filter(
        (run) =>
          run &&
          run.id &&
          run.source_sha256 === detail.source_sha256 &&
          String(run.region?.record_id) === String(detail.id),
      )
      .map((run) => run.id),
  );
  const choosers = await page.locator("select").evaluateAll((nodes) =>
    nodes.map((node, index) => {
      if (!node.id) node.id = "qa-pipeline-choice-" + index;
      return {
        id: node.id,
        label: [
          node.id,
          document.querySelector(`label[for="${CSS.escape(node.id)}"]`)
            ?.textContent ?? "",
          node.getAttribute("aria-label") ?? "",
        ].join(" "),
        options: [...node.options].map((option) => option.value).filter(Boolean),
      };
    }),
  );
  const resultSelects = choosers.filter((item) =>
    /результат|сравн/i.test(item.label),
  );
  const firstChoice =
    resultSelects.length >= 2
      ? resultSelects[0].options.find((id) => compatible.has(id))
      : "";
  const secondChoice =
    resultSelects.length >= 2
      ? resultSelects[1].options.find((id) => compatible.has(id) && id !== firstChoice)
      : "";
  const boxes = await page.locator("input[type='checkbox']").evaluateAll((nodes) =>
    nodes.map((node, index) => {
      if (!node.id) node.id = "qa-pipeline-result-" + index;
      return {
        id: node.id,
        value: node.value,
      };
    }),
  );
  const boxIds = [];
  const seenRuns = new Set();
  for (const box of boxes) {
    if (!compatible.has(box.value) || seenRuns.has(box.value)) continue;
    seenRuns.add(box.value);
    boxIds.push(box.id);
  }
  const canChoose = (!!firstChoice && !!secondChoice) || boxIds.length >= 2;
  if (canChoose) {
    if (firstChoice && secondChoice) {
      await page.locator("#" + resultSelects[0].id).selectOption(firstChoice);
      await page.locator("#" + resultSelects[1].id).selectOption(secondChoice);
    } else {
      await page.locator("#" + boxIds[0]).setChecked(true);
      await page.locator("#" + boxIds[1]).setChecked(true);
    }
    const compareTab = page.getByRole("tab", { name: "Сравнение", exact: true });
    const compareButton = page.getByRole("button", {
      name: /^(Сравнить|Открыть сравнение|Показать сравнение)$/,
    });
    if (await compareTab.count()) await compareTab.click();
    else if (
      (await compareButton.count()) &&
      (await compareButton.first().isEnabled())
    )
      await compareButton.first().click();
    await page.waitForFunction(
      () =>
        /общая область|сравнение двух|пересечение/i.test(
          document.querySelector("main.pipeline")?.innerText ?? "",
        ),
      undefined,
      { timeout: 10000 },
    );
    const comparisonText = (
      (await page.locator("main.pipeline").innerText()) +
      "\n" +
      ((await page.getByRole("dialog").count())
        ? await page.getByRole("dialog").first().innerText()
        : "")
    ).toLowerCase();
    const denied = comparisonText
      .replace(/не\s+объявля\p{L}*\s+победител\p{L}*/gu, " ")
      .replace(/не\s+выбира\p{L}*\s+победител\p{L}*/gu, " ")
      .replace(/не\s+явля\p{L}*\s+победител\p{L}*/gu, " ")
      .replace(/не\s+победител\p{L}*/gu, " ")
      .replace(/без\s+победител\p{L}*/gu, " ")
      .replace(/победител\p{L}*\s+нет/gu, " ")
      .replace(/нет\s+победител\p{L}*/gu, " ")
      .replace(/no winner|not (?:the |a )?winner|without a winner|does not pick a winner|did not pick a winner|picked no winner/g, " ");
    if (
      /победител|выиграл|winner|picked a winner|лучший результат|лучший метод|выбран лучшим/.test(
        denied,
      )
    )
      fail("Comparison picked a winner");
    comparison = "opened";
  }

  if (errors.length) throw new Error(JSON.stringify(errors));
  return {
    record_id: String(detail.id),
    bounds,
    job_id: jobId,
    calculation,
    settings_saved: true,
    preflight,
    comparison,
    viewports: [1440, 430, 390, 320],
    point_scroll_unchanged: true,
    errors,
  };
}
