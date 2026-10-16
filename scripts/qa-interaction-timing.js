// Event Timing probe, not a full-session INP or a phone benchmark.
async (page) => {
  const origin = new URL(page.url()).origin;
  const tab = await page.context().newPage();
  try {
    await tab.goto(origin + '/research?record=5');
    await tab.locator('.readings').waitFor();
    const record = await tab.evaluate(async () => (await fetch('/api/v1/records/5')).json());
    await tab.waitForFunction(timestamp => document.querySelector('.readings')?.textContent?.includes(timestamp), record.timestamps[0]);
    await tab.evaluate(() => {
      window.__dtsInteractions = [];
      window.__dtsEventObserver = new PerformanceObserver(list => {
        for (const entry of list.getEntries()) {
          if (entry.interactionId && entry.name === 'click')
            window.__dtsInteractions.push({durationMs: entry.duration, interactionId: entry.interactionId});
        }
      });
      window.__dtsEventObserver.observe({type:'event', durationThreshold:16});
    });
    for (let row = 1; row <= 30; row++) {
      await tab.getByRole('button', {name:'Следующий профиль', exact:true}).click();
      await tab.waitForFunction(expected => document.querySelector('#row')?.value === String(expected), row);
      await tab.waitForFunction(timestamp => document.querySelector('.readings')?.textContent?.includes(timestamp), record.timestamps[row]);
    }
    // Flush observer delivery via two browser frames, not an arbitrary sleep.
    await tab.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    return await tab.evaluate(() => {
      window.__dtsEventObserver.disconnect();
      return {
        userAgent:navigator.userAgent, origin:location.origin,
        viewport:[innerWidth,innerHeight], devicePixelRatio,
        action:'30 trusted clicks: next native profile on 5.csv',
        thresholdMs:16, entries:window.__dtsInteractions,
        caveat:'Event Timing reports only entries at or above threshold, rounded to 8 ms. These clicks do not constitute full-session INP or hardware phone measurements.'
      };
    });
  } finally { await tab.close(); }
}
