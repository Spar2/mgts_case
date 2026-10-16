// Hold old GET responses while switching records; never writes expert answers.
async (page) => {
  const origin = new URL(page.url()).origin;
  const tab = await page.context().newPage();
  const errors = [], cancelled = [];
  tab.on('pageerror', error => errors.push(String(error)));
  tab.on('requestfailed', request => {
    if (request.url().includes('/records/5/')) cancelled.push(request.failure()?.errorText);
  });
  try {
    const record3 = await tab.request.get(origin + '/api/v1/records/3').then(r => r.json());
    async function ready(id, sha) {
      await tab.waitForFunction(value => document.querySelector('.provenance code')?.textContent === value, sha);
      await tab.locator('.heat-figure canvas').waitFor();
      if (id === '3') await tab.waitForFunction(t => document.querySelector('.readings')?.textContent?.includes(t), record3.timestamps[0]);
    }
    const results = [];
    for (const kind of ['slices', 'tiles']) {
      await tab.goto(origin + '/research?record=5');
      await tab.locator('.heat-figure canvas').waitFor();
      const record5 = await tab.request.get(origin + '/api/v1/records/5').then(r => r.json());
      await tab.waitForFunction(t => document.querySelector('.readings')?.textContent?.includes(t), record5.timestamps[0]);
      let release, captured;
      const held = new Promise(resolve => release = resolve);
      const intercepted = new Promise(resolve => captured = resolve);
      const pattern = kind === 'slices' ? /\/records\/5\/slices\?row=100&/ : /\/records\/5\/tiles\?/;
      await tab.route(pattern, async route => {
        const response = await route.fetch();
        captured();
        await held;
        // A correctly cancelled fetch cannot receive this late response.
        try { await route.fulfill({response}); } catch { /* aborted request */ }
      });
      if (kind === 'slices') {
        await tab.locator('#row').fill('100'); await tab.locator('#row').press('Enter');
      } else await tab.getByRole('button',{name:'Приблизить выбранную точку',exact:true}).click();
      await intercepted;
      await tab.getByRole('combobox',{name:'Запись',exact:true}).selectOption('3');
      await ready('3',record3.source_sha256);
      await tab.waitForFunction(bounds => bounds.every(v => document.querySelector('.heat-figure .chart-heading')?.textContent?.includes(v.toFixed(2))),record3.scale);
      release();
      await tab.unroute(pattern);
      // Flush any stale callback after its held response has been released.
      await tab.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
      const shown = await tab.locator('.provenance code').textContent();
      if (shown !== record3.source_sha256) throw new Error('Old source replaced current provenance');
      const heading = await tab.locator('.heat-figure .chart-heading').innerText();
      for (const bound of record3.scale) if (!heading.includes(bound.toFixed(2))) throw new Error('Old tile replaced current scale');
      const readings = await tab.locator('.readings').innerText();
      if (!readings.includes(record3.timestamps[0])) throw new Error('Old slice replaced current observation');
      results.push({held:kind,currentRecord:'3',provenanceAndScalePreserved:true});
    }
    if (errors.length) throw new Error(errors.join('\n'));
    return {scenarios:results,cancelledRequests:cancelled,runtimeErrors:errors};
  } finally { await tab.close(); }
}
