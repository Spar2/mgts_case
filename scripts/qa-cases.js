// Technical fixtures in an isolated schema; never expert labels for real data.
async (page) => {
  const origin = new URL(page.url()).origin;
  const tab = await page.context().newPage();
  const errors = [];
  tab.on('pageerror', e => errors.push(String(e)));
  try {
    const health = await tab.request.get(origin+'/api/v1/health').then(r=>r.json());
    if (!health.qa_mode || !health.auth_enabled) throw new Error('Isolated authenticated QA required');
    const account = await tab.request.get(origin+'/api/v2/auth/me').then(r=>r.json());
    const session = await tab.request.get(origin+'/api/v1/session').then(r=>r.json());
    const headers = {'X-Workbench-Token':session.mutation_token};
    const packageId='general';
    const suffix='?'+new URLSearchParams({package:packageId,independent:'true',reviewer:account.display_name});
    const queue=await tab.request.get(origin+'/api/v1/queue'+suffix).then(r=>r.json());
    const card=queue.cards[0];
    const record=card.source.replace(/\.csv$/,'');
    const row=card.row_bounds[0];
    const slices=await tab.request.get(origin+`/api/v1/records/${record}/slices?row=${row}&column=0`).then(r=>r.json());
    const region={record_id:record,row_start:card.row_bounds[0],row_end:card.row_bounds[1],column_start:card.column_bounds[0],column_end:card.column_bounds[1]};
    const envelope={schema_version:'dts-method-result-v1',region:{record_id:record,row_start:row,row_end:row+1,column_start:0,column_end:slices.profile.length},source_sha256:card.source_sha256,method:'case-contract-fixture',method_version:'qa-only',parameters:{contract_only:true},calculated_at:'2026-10-11',training_origin:'untrained',limitations:'QA only, no training and no physical claims',result:{denoised:slices.profile.map(v=>v===null?100:v+.25)}};
    await tab.goto(origin+'/runs');
    let writes=0;
    const track=request=>{if(request.url().endsWith('/api/v1/results')&&request.method()==='POST')writes++;};
    tab.on('request',track);
    await tab.locator('input[type="file"]').setInputFiles({name:'contract.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(envelope))});
    await tab.getByRole('heading',{name:'Проверить перед импортом',exact:true}).waitFor();
    if(writes!==0)throw new Error('File selection imported before confirmation');
    const imported=tab.waitForResponse(r=>r.url().endsWith('/api/v1/results')&&r.request().method()==='POST');
    await tab.getByRole('button',{name:'Подтвердить импорт',exact:true}).click();
    const importResponse=await imported;
    if(importResponse.status()!==201)throw new Error('Import failed');
    const run=await importResponse.json();
    const vote={queue_sha256:queue.queue_sha256,review_id:card.review_id,expected_revision:queue.annotations[card.review_id]?.revision??0,reviewer:account.display_name,reviewer_type:'expert',temperature_episode:'ambiguous',phenomenon:'unknown',notes:'QA isolated revision, not a real expert conclusion'};
    const saved=await tab.request.post(origin+'/api/v1/reviews'+suffix,{headers:{...headers,'Idempotency-Key':'case-review-'+Date.now()},data:vote});
    if(saved.status()!==200)throw new Error('Independent fixture review failed');
    const revision=(await saved.json()).revision;
    const path='/research?'+new URLSearchParams({record,row:String(row),column:String(region.column_start),r0:String(region.row_start),r1:String(region.row_end),c0:String(region.column_start),c1:String(region.column_end),run_id:run.id,source_sha256:card.source_sha256});
    await tab.goto(origin+path);
    await tab.getByRole('heading',{name:'Зафиксировать кейс',exact:true}).waitFor();
    await tab.getByText('Выбранный ID графика и отчёта', {exact:true}).waitFor();
    await tab.getByRole('button',{name:'Проверить состав кейса',exact:true}).click();
    await tab.locator('.case-preview').getByText('QA isolated revision, not a real expert conclusion',{exact:false}).waitFor();
    // Lose the response AFTER the server committed, then lose a replay response too.
    let committedCase;
    const attemptKeys=[];
    await tab.route('**/api/v2/cases',async route=>{
      if(route.request().method()!=='POST')return route.continue();
      attemptKeys.push(route.request().headers()['idempotency-key']);
      const persisted=await route.fetch();
      if(persisted.status()!==201)throw new Error('Initial server save failed');
      committedCase=await persisted.json();
      if(attemptKeys.length===1)return route.fulfill({status:503,contentType:'application/json',body:JSON.stringify({detail:'QA: response lost after commit'})});
      return route.abort('failed');
    });
    await tab.getByRole('button',{name:'Зафиксировать кейс',exact:true}).click();
    await tab.getByText(/QA: response lost after commit/).waitFor();
    if(!await tab.getByRole('button',{name:'Проверить состав кейса',exact:true}).isDisabled())throw new Error('Pending action allowed a different preview');
    await tab.getByRole('button',{name:'Повторить сохранение',exact:true}).click();
    await tab.locator('.case-pending').waitFor();
    if(attemptKeys.length!==2)throw new Error('Network replay was not sent');
    await tab.unroute('**/api/v2/cases');
    const creation=tab.waitForResponse(r=>r.url().endsWith('/api/v2/cases')&&r.request().method()==='POST');
    await tab.getByRole('button',{name:'Повторить сохранение',exact:true}).click();
    const created=await creation;
    if(created.status()!==201)throw new Error('Case creation failed: '+created.status());
    const evidenceCase=await created.json();
    attemptKeys.push(created.request().headers()['idempotency-key']);
    if(!attemptKeys[0]||attemptKeys.some(k=>k!==attemptKeys[0])||evidenceCase.id!==committedCase.id)throw new Error('Retry duplicated or changed the pending case');
    if(!await tab.getByRole('button',{name:'Зафиксировать кейс',exact:true}).isDisabled())throw new Error('Confirmed save allowed another creation');
    if(evidenceCase.review.revision!==revision||evidenceCase.run_id!==run.id)throw new Error('Case identity differs');
    await tab.getByRole('link',{name:'Кейс сохранён — открыть →',exact:true}).click();
    await tab.getByText('Неизменяемый снимок',{exact:true}).waitFor();
    await tab.getByRole('heading',{name:'Что изменила обработка',exact:true}).waitFor();
    const changes=evidenceCase.result.evidence.processing_changes;
    const changeRows=await tab.locator('.case-change-table tbody tr').evaluateAll(rows=>rows.map(r=>Array.from(r.querySelectorAll('td')).map(td=>Number(td.textContent))));
    if(JSON.stringify(changeRows)!==JSON.stringify(changes.top_points.map(p=>[p.x_m,p.original_c,p.processed_c,p.delta_c])))throw new Error('Displayed changes differ from saved metrics');
    const widths=[320,390,430,1024,1440];
    for(const width of widths){await tab.setViewportSize({width,height:900});await tab.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));const overflow=await tab.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1);if(overflow)throw new Error('Case overflow at '+width);}
    await tab.emulateMedia({reducedMotion:'reduce'});
    const downloadEvent=tab.waitForEvent('download');
    await tab.getByRole('button',{name:'Скачать пакет',exact:true}).click();
    const download=await downloadEvent;
    const stream=await download.createReadStream();let bytes=0;for await(const chunk of stream)bytes+=chunk.length;
    if(bytes<1000||!download.suggestedFilename().endsWith('.zip'))throw new Error('ZIP download missing');
    const before=await tab.request.get(origin+`/api/v2/cases/${evidenceCase.id}/bundle`).then(r=>r.body());
    const updated=await tab.request.post(origin+'/api/v1/reviews'+suffix,{headers:{...headers,'Idempotency-Key':'case-review-new-'+Date.now()},data:{...vote,expected_revision:revision,notes:'Newer isolated answer'}});
    if(updated.status()!==200)throw new Error('Review update failed');
    const after=await tab.request.get(origin+`/api/v2/cases/${evidenceCase.id}/bundle`).then(r=>r.body());
    if(!before.equals(after))throw new Error('Newer answer changed old ZIP');
    await tab.reload();await tab.getByText('QA isolated revision, not a real expert conclusion',{exact:true}).waitFor();
    // Preserve the snapshot if metadata becomes temporarily unavailable.
    await tab.route('**/api/v2/cases/'+evidenceCase.id,route=>route.fulfill({status:503,contentType:'application/json',body:JSON.stringify({detail:'QA: temporary read outage'})}));
    await tab.getByRole('button',{name:'Повторить загрузку',exact:true}).click();
    await tab.getByText(/QA: temporary read outage/).waitFor();
    if(!await tab.locator('.case-field').count())throw new Error('Read outage removed cached snapshot');
    await tab.unroute('**/api/v2/cases/'+evidenceCase.id);
    await tab.getByRole('button',{name:'Повторить загрузку',exact:true}).click();
    // Frozen presentation must use the saved run/revision, even after newer work.
    const newer=await tab.request.post(origin+'/api/v1/results',{headers,data:{...envelope,method_version:'qa-newer',result:{denoised:envelope.result.denoised.map(v=>v+1)}}});
    if(newer.status()!==201)throw new Error('Newer fixture run failed');
    const frozen=await tab.request.get(origin+`/api/v2/cases/${evidenceCase.id}`).then(r=>r.json());
    const presentationWrites=[];
    // Existing session activity heartbeat is allowed; product writes are forbidden.
    const trackWrites=request=>{const url=new URL(request.url());if(url.origin===origin&&url.pathname!=='/api/v2/auth/activity'&&!['GET','HEAD'].includes(request.method()))presentationWrites.push(request.method()+' '+url.pathname);};
    tab.on('request',trackWrites);
    await tab.getByRole('link',{name:'Показать на защите →',exact:true}).click();
    const stepHeading=tab.locator('#case-step-heading');
    await stepHeading.filter({hasText:'Измерение'}).waitFor();
    await tab.getByRole('button',{name:'Далее',exact:true}).click();
    await stepHeading.filter({hasText:'Наблюдение'}).waitFor();
    await tab.waitForFunction(()=>document.activeElement?.id==='case-step-heading');
    await tab.keyboard.press('ArrowRight');
    await stepHeading.filter({hasText:'Обработка'}).waitFor();
    if(new URL(tab.url()).searchParams.get('step')!=='2')throw new Error('Step missing from URL');
    await tab.reload();await stepHeading.filter({hasText:'Обработка'}).waitFor();
    await tab.getByText(run.id,{exact:true}).waitFor();
    await tab.getByText('Исходный профиль · °C',{exact:true}).waitFor();
    await tab.getByText('Обработка − исходный сигнал · °C',{exact:true}).waitFor();
    await tab.waitForFunction(()=>document.querySelectorAll('.line-figure .uplot').length===3);
    await tab.locator('.line-figure .u-over').first().scrollIntoViewIfNeeded();
    const plot=await tab.locator('.line-figure .u-over').first().boundingBox();
    await tab.mouse.move(plot.x+plot.width*.4,plot.y+plot.height*.5);
    await tab.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));
    const cursors=await tab.locator('.line-figure .u-cursor-x').evaluateAll(nodes=>nodes.map(n=>{const over=n.closest('.uplot').querySelector('.u-over').getBoundingClientRect();return {relative:(n.getBoundingClientRect().left-over.left)/over.width,display:getComputedStyle(n).display};}));
    if(cursors.length!==3||cursors.some(n=>n.display==='none'||Math.abs(n.relative-cursors[0].relative)>.01))throw new Error('Profile cursors are not synchronized: '+JSON.stringify(cursors));
    await tab.getByText('Точные значения профиля',{exact:true}).click();
    const actualRows=await tab.locator('.case-table tbody tr').evaluateAll(rows=>rows.map(row=>Array.from(row.querySelectorAll('td')).map(td=>td.textContent)));
    const profile=frozen.result.evidence.profile;
    const expectedRows=profile.x_m.map((x,i)=>[String(x),String(profile.temperature_c[i]??'Пропуск'),String(profile.processed_temperature_c[i]??'Пропуск')]);
    if(JSON.stringify(actualRows)!==JSON.stringify(expectedRows))throw new Error('Presentation values differ from frozen snapshot');
    for(const width of [320,390,430,1440]){
      await tab.setViewportSize({width,height:900});
      for(let i=0;i<5;i++){
        await tab.getByRole('navigation',{name:'Шаги защиты'}).locator('button').nth(i).click();
        await tab.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));
        if(await tab.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1))throw new Error('Presentation overflow at '+width+' step '+i);
        if(i===2&&width<=430){
          const mobile=tab.locator('.case-change-cards');await mobile.waitFor();
          const mobileValues=await mobile.locator('dd').evaluateAll(nodes=>nodes.map(n=>Number(n.textContent)));
          if(JSON.stringify(mobileValues)!==JSON.stringify(changes.top_points.flatMap(p=>[p.x_m,p.original_c,p.processed_c,p.delta_c])))throw new Error('Mobile change values differ');
        }
      }
    }
    await tab.getByRole('navigation',{name:'Шаги защиты'}).locator('button').nth(3).click();
    await tab.getByText('QA isolated revision, not a real expert conclusion',{exact:true}).waitFor();
    await tab.route('**/api/v2/cases/'+evidenceCase.id,route=>route.fulfill({status:503,contentType:'application/json',body:JSON.stringify({detail:'QA: presentation read outage'})}));
    await tab.getByRole('button',{name:'Повторить загрузку',exact:true}).click();
    await tab.getByText(/QA: presentation read outage/).waitFor();
    await tab.getByText('QA isolated revision, not a real expert conclusion',{exact:true}).waitFor();
    await tab.unroute('**/api/v2/cases/'+evidenceCase.id);
    const recovered=tab.waitForResponse(r=>r.url().endsWith('/api/v2/cases/'+evidenceCase.id)&&r.status()===200);
    await tab.getByRole('button',{name:'Повторить загрузку',exact:true}).click();await recovered;
    await tab.evaluate(()=>{HTMLElement.prototype.requestFullscreen=()=>Promise.reject(new Error('QA fullscreen refusal'));});
    await tab.getByRole('button',{name:'На весь экран',exact:true}).click();
    await tab.getByText(/Полноэкранный режим недоступен/).waitFor();
    await tab.getByRole('navigation',{name:'Шаги защиты'}).locator('button').nth(4).click();
    await tab.route('**/api/v2/cases/'+evidenceCase.id+'/bundle',route=>route.fulfill({status:503,contentType:'application/json',body:JSON.stringify({detail:'QA: presentation download outage'})}));
    await tab.getByRole('button',{name:'Скачать пакет',exact:true}).click();
    await tab.getByText(/QA: presentation download outage/).waitFor();
    await tab.unroute('**/api/v2/cases/'+evidenceCase.id+'/bundle');
    const shownDownload=tab.waitForEvent('download');await tab.getByRole('button',{name:'Скачать пакет',exact:true}).click();await shownDownload;
    if(presentationWrites.length)throw new Error('Presentation issued mutations: '+presentationWrites.join(', '));
    tab.off('request',trackWrites);
    // Raw-only backup remains honest: no fabricated processed profile or delta.
    const rawCase=await tab.request.post(origin+'/api/v2/cases',{headers,data:{title:'QA raw backup',region,profile:row,source_sha256:card.source_sha256}}).then(r=>r.json());
    await tab.goto(origin+`/cases/${rawCase.id}?present=1&step=2`);
    await tab.getByText('Обработка не выбрана. Показан исходный сигнал; разность не вычисляется.',{exact:true}).waitFor();
    if(await tab.getByText('Обработка − исходный сигнал · °C',{exact:true}).count())throw new Error('Raw presentation fabricated delta');
    await tab.keyboard.press('ArrowRight');await tab.getByText('Эксперт ещё не проверил.',{exact:true}).waitFor();
    if(errors.length)throw new Error('Page errors: '+errors.join(';'));
    return {case_id:evidenceCase.id,immutable_zip:true,review_revision:revision,import_preview:true,download_bytes:bytes,widths,reduced_motion:true,save_retry:true,cached_snapshot_survives_outage:true,presentation_snapshot:true,presentation_no_mutations:true,presentation_raw_backup:true,idempotent_commit_retry:true,descriptive_changes:true};
  } catch(error) {
    throw new Error(String(error)+'; route='+new URL(tab.url()).pathname+'; case state='+await tab.locator('.case-page').innerText().then(s=>s.slice(-1800)).catch(()=> 'unavailable'));
  } finally {await tab.close();}
}
