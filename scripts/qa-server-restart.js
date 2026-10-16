// Run twice on the same page, restarting ONLY the isolated QA API between runs.
// Requires the explicit marker; never run against the working journal on 8081.
async (page) => {
  const origin = new URL(page.url()).origin;
  if (origin !== 'http://127.0.0.1:8091') throw new Error('Requires isolated QA on 8091');
  const pending = await page.evaluate(() => window.__dtsRestartQA);
  const reviewer = 'QA ONLY — отдельный тестовый журнал';
  const ownQuery = '?independent=true&reviewer='+encodeURIComponent(reviewer);
  if (!pending) {
    const queue = await page.evaluate(async () => (await fetch('/api/v1/queue')).json());
    const marker = Object.values(queue.annotations).find(a => a.reviewer === 'QA ONLY — отдельный тестовый журнал');
    if (!marker) throw new Error('Explicit isolated journal marker required');
    await page.goto(origin + '/review?card=' + encodeURIComponent(marker.review_id));
    await page.getByRole('heading',{name:'Ваше решение',exact:true}).waitFor();
    await page.getByRole('textbox',{name:'Имя эксперта',exact:true}).fill(reviewer);
    await page.getByRole('textbox',{name:'Имя эксперта',exact:true}).press('Tab');
    const own = await page.evaluate(async suffix => (await fetch('/api/v1/queue'+suffix)).json(), ownQuery);
    const revision = own.annotations[marker.review_id]?.revision ?? 0;
    await page.locator('label[for="answer-unknown"]').click();
    await page.getByRole('textbox',{name:'Обоснование по сигналу',exact:true}).fill('QA ONLY: technical persistence/token restart probe, not an expert assessment. '+Date.now());
    await page.getByRole('button',{name:'Сохранить ответ',exact:true}).click();
    await page.getByText('Сохранено · ревизия '+(revision+1),{exact:true}).waitFor();
    await page.evaluate(state => window.__dtsRestartQA = state, {id:marker.review_id,revision:revision+1});
    return {phase:'saved-before-restart',revision:revision+1};
  }
  const persisted = await page.evaluate(async suffix => (await fetch('/api/v1/queue'+suffix)).json(),ownQuery);
  if (persisted.annotations[pending.id]?.revision !== pending.revision) throw new Error('Journal was not persisted across restart');
  const rejected = [];
  page.on('response', response => { if(response.url().includes('/api/v1/reviews') && response.status()===403) rejected.push(403); });
  await page.getByRole('textbox',{name:'Обоснование по сигналу',exact:true}).fill('QA ONLY: technical post-restart save, not an expert assessment. '+Date.now());
  await page.getByRole('button',{name:'Сохранить ответ',exact:true}).click();
  await page.getByText('Сохранено · ревизия '+(pending.revision+1),{exact:true}).waitFor();
  if (rejected.length !== 1) throw new Error('Expected exactly one rejected old token before refresh');
  const exported = await page.evaluate(async suffix => (await fetch('/api/v1/export'+suffix)).json(),ownQuery);
  if (exported.annotations.find(a=>a.review_id===pending.id)?.revision !== pending.revision+1) throw new Error('Export revision mismatch');
  return {phase:'verified-after-restart',persisted:true,tokenRefreshed:true,oldTokenRejectedOnce:true,revision:pending.revision+1,exportVersion:exported.version};
}
