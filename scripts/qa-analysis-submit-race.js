// Changing source does not cancel an accepted server task or attach it to another source.
async (page) => {
  const origin = new URL(page.url()).origin;
  if (
    !(await (await page.request.get(origin + "/api/v1/health")).json()).qa_mode
  )
    throw new Error("Isolated QA only");
  const records = await (
    await page.request.get(origin + "/api/v1/records")
  ).json();
  if (records.length < 2) throw new Error("Two native sources required");
  const first = records.find((r) => r.id === "5") ?? records[0],
    second = records.find((r) => r.id !== first.id);
  const detail = await (
    await page.request.get(origin + "/api/v1/records/" + first.id)
  ).json();
  await page.goto(
    origin +
      "/research?" +
      new URLSearchParams({
        record: first.id,
        r0: "0",
        r1: String(Math.min(detail.shape[0], 16)),
        c0: "0",
        c1: String(Math.min(detail.shape[1], 64)),
      }),
  );
  await page.locator(".analysis-tools").evaluate((n) => {
    n.open = true;
  });
  const button = page.getByRole("button", {
    name: "Запустить расчёт поля",
    exact: true,
  });
  await page.waitForFunction(
    () => !document.querySelector(".analysis-job button")?.disabled,
  );
  let release,
    committedId = "";
  const gate = new Promise((resolve) => {
    release = resolve;
  });
  let accepted;
  const committed = new Promise((resolve) => {
    accepted = resolve;
  });
  await page.route("**/api/v2/analysis-jobs", async (route) => {
    const response = await route.fetch();
    if (response.status() !== 202)
      throw new Error("Race submit failed " + response.status());
    committedId = (await response.json()).id;
    accepted();
    await gate;
    try {
      await route.fulfill({ response });
    } catch {
      /* Source navigation already aborted this request. */
    }
  });
  await button.click();
  await committed;
  await page.goto(origin + "/research?record=" + second.id);
  release();
  await page.waitForTimeout(500);
  await page.unroute("**/api/v2/analysis-jobs");
  const url = new URL(page.url());
  if (
    url.searchParams.get("record") !== second.id ||
    url.searchParams.has("job_id")
  )
    throw new Error("Stale job attached to new source");
  const state = await (
    await page.request.get(origin + "/api/v2/analysis-jobs/" + committedId)
  ).json();
  if (state.region.record_id !== first.id)
    throw new Error("Accepted task source changed");
  return { calculation_source_race: true };
}
