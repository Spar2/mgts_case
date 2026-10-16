// Run against the prepared real-data QA runtime; never the working project.
async (page) => {
  const origin = new URL(page.url()).origin;
  await page.context().setStorageState('.runtime/product-qa/browser-auth.json');
  const health = await page.request.get(origin + '/api/v1/health').then(r => r.json());
  if (!health.qa_mode) throw new Error('Isolated QA only');
  const streams = await page.request.get(origin + '/api/v2/streams').then(r => r.json());
  const stream = streams.items.find(s => s.record_id === '4' && s.status === 'complete');
  if (!stream) throw new Error('Completed native replay required');
  await page.goto(origin + '/monitoring?stream_id=' + stream.id);
  const profile = page.getByText(/Точный загруженный профиль/);
  await profile.waitFor();
  const previous = await profile.textContent();
  await page.context().setOffline(true);
  try {
    await page.getByText(/Связь прервана; показан последний подтверждённый снимок/).waitFor();
    if (await profile.textContent() !== previous) throw new Error('Confirmed profile lost offline');
    if (await page.locator('canvas').count() === 0) throw new Error('Charts disappeared');
  } finally {
    await page.context().setOffline(false);
  }
  await page.waitForResponse(r => new URL(r.url()).pathname === '/api/v2/stream/' + stream.id && r.status() === 200);
  await profile.waitFor();
  return {confirmed_profile_retained_during_outage:true,reconnected_without_page_reload:true};
}
