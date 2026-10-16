// CI contract fixtures only; these assertions do not establish ML quality.
async (page) => {
  const origin = new URL(page.url()).origin;
  const health = await page.request.get(origin + '/api/v1/health').then(r => r.json());
  if (!health.qa_mode) throw new Error('Isolated QA required');
  const runs = await page.request.get(origin + '/api/v2/result-runs').then(r => r.json());
  const run = runs.find(r => r.layers.includes('score'));
  if (!run) throw new Error('Numeric contract fixture required');
  const r = run.region;
  const query = new URLSearchParams({record:r.record_id,source_sha256:run.source_sha256,run_id:run.id,
    r0:String(r.row_start),r1:String(r.row_end),c0:String(r.column_start),c1:String(r.column_end),
    row:String(r.row_start),column:String(r.column_start)});
  await page.goto(origin + '/works?' + query);
  const title = 'QA ONLY CI work ' + Date.now();
  await page.locator('#work-title-new').fill(title);
  await page.getByRole('button', {name:'Создать работу',exact:true}).click();
  await page.waitForURL(/\/works\/[0-9a-f]{32}$/);
  const id = new URL(page.url()).pathname.split('/').at(-1);
  await page.getByRole('heading',{name:title,exact:true}).waitFor();
  await page.locator('.heat-figure canvas').first().waitFor();
  await page.getByText('Контекст и заметки',{exact:true}).click();
  await page.locator('#work-notes').fill('QA explicit saved context');
  await page.getByRole('button',{name:'Сохранить контекст',exact:true}).click();
  await page.getByText(/Контекст сохранён/).waitFor();
  await page.reload();
  await page.getByText('Контекст и заметки',{exact:true}).click();
  if (await page.locator('#work-notes').inputValue() !== 'QA explicit saved context') throw new Error('Notes lost');
  await page.emulateMedia({reducedMotion:'reduce'});
  for (const path of ['/works/'+id,'/laboratory','/monitoring']) {
    await page.goto(origin+path);
    await page.getByRole('heading',{level:1}).waitFor();
    for (const width of [320,390,430,1440]) {
      await page.setViewportSize({width,height:900});
      await page.waitForFunction(()=>document.documentElement.scrollWidth<=innerWidth+1);
      const menu=page.getByRole('button',{name:'Меню',exact:true});
      if(await menu.isVisible())await menu.click();
      await page.getByRole('navigation',{name:'Основная навигация'}).getByRole('link',{name:'Аккаунт',exact:true}).waitFor();
      await page.waitForFunction(()=>document.documentElement.scrollWidth<=innerWidth+1);
      if(await menu.isVisible())await menu.click();
    }
  }
  const saved = await page.request.get(origin+'/api/v2/investigations/'+id).then(r=>r.json());
  if (saved.context.run_ids[0] !== run.id || saved.context.source_sha256 !== run.source_sha256) throw new Error('Context substituted');
  return {workspace_saved_context:true,workspace_direct_links:true,workspace_mobile:true,workspace_no_result_substitution:true};
}
