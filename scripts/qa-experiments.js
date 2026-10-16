// Authenticated, read-only research view; source metrics are never training labels.
async (page) => {
  const origin=new URL(page.url()).origin,errors=[],writes=[];
  const observe=request=>{if(request.method()==='POST'&&/\/api\/(v1\/(denoise|results|jobs|reviews)|v2\/cases)/.test(request.url()))writes.push(request.url());};
  page.on('pageerror',e=>errors.push(String(e)));page.on('request',observe);
  try{
    const response=await page.request.get(origin+'/api/v2/ml-experiments');
    if(response.status()!==200)throw new Error('Research API status '+response.status());
    const pack=await response.json();
    for(const width of [1440,430,390,320]){
      await page.setViewportSize({width,height:900});
      await page.goto(origin+'/experiments?method=n2n_tsr&scale=mad_profile');
      await page.getByRole('heading',{name:'Эксперименты ML: пороги и границы',exact:true}).waitFor();
      await page.locator('.experiment-chart svg').first().waitFor();
      if(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1))throw new Error('Research overflow at '+width);
      if(await page.locator('.experiment-chart:not(.experiment-real-chart) svg').count()!==4)throw new Error('Missing research curves');
      const summary=page.locator('.experiment-summary').first();
      const row=pack.calibration.rows.find(r=>r.method==='n2n_tsr'&&r.scale==='mad_profile');
      if(!(await summary.textContent()).includes(row.threshold.toString())||!(await summary.textContent()).includes(row.hot_iou.toFixed(6)))throw new Error('Summary differs from source evidence');
      const slider=page.getByRole('slider').first();await slider.focus();await slider.press('ArrowRight');
      if(await slider.inputValue()!=='1')throw new Error('Curve keyboard inspection failed');
      await page.locator('#experiment-method').selectOption('zsn2n');
      await page.getByRole('heading',{name:'zsn2n: результат исходного протокола',exact:true}).waitFor();
      await page.locator('#experiment-record').selectOption('5.csv');
      const real=pack.calibration.rows.find(r=>r.method==='zsn2n'&&r.scale==='mad_profile').real_files.find(f=>f.file==='5.csv');
      await page.getByText('Исходник SHA: '+real.source_sha256,{exact:true}).waitFor();
    }
    const download=page.waitForEvent('download');await page.getByRole('button',{name:'Скачать исследовательскую сводку JSON',exact:true}).click();const item=await download;
    if(item.suggestedFilename()!=='dts-ml-experiments-237168f.json')throw new Error('Unexpected research export');
    await page.getByRole('link',{name:'Открыть каталог расчётов этой записи →',exact:true}).click();
    await page.getByRole('heading',{name:'Результаты анализа реальных данных',exact:true}).waitFor();
    if(new URL(page.url()).searchParams.get('record_id')!=='5')throw new Error('Lost record context');
    if(writes.length||errors.length)throw new Error(JSON.stringify({writes,errors}));
    return {research_readonly:true,research_source_values:true,research_keyboard:true,research_mobile:true,research_download:true};
  }finally{page.off('request',observe);}
}
