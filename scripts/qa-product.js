// Actual QA runtime only. Private actors are read in memory, never logged.
async (page) => {
  const origin=new URL(page.url()).origin;
  await page.context().setStorageState('.runtime/product-qa/browser-auth.json');
  await page.goto(origin+'/works');
  const health=await page.request.get(origin+'/api/v1/health').then(r=>r.json());
  if(!health.qa_mode)throw new Error('Isolated QA only');
  const errors=[];page.on('pageerror',e=>errors.push(String(e)));
  const experts=await page.request.get(origin+'/api/v2/review-experts').then(r=>r.json());
  const expert=experts.find(e=>e.display_name==='QA ONLY participant 10');if(!expert)throw new Error('Technical expert missing');
  const streamList=await page.request.get(origin+'/api/v2/streams').then(r=>r.json());
  const stream=streamList.items.find(s=>s.record_id==='4'&&s.status==='complete');if(!stream)throw new Error('Completed verified replay missing');
  const runs=await page.request.get(origin+'/api/v2/result-runs').then(r=>r.json());
  const run=runs.find(r=>r.method==='ridge_background'&&r.region.record_id==='5'&&r.region.row_start===1897);if(!run)throw new Error('Verified real result missing');
  const r=run.region;const params=new URLSearchParams({record:r.record_id,source_sha256:run.source_sha256,row:String(r.row_start),column:String(r.column_start),r0:String(r.row_start),r1:String(r.row_end),c0:String(r.column_start),c1:String(r.column_end),run_id:run.id});
  await page.setViewportSize({width:1440,height:1000});await page.emulateMedia({reducedMotion:'reduce'});
  await page.goto(origin+'/works?'+params);await page.getByRole('heading',{name:'Работы команды',exact:true}).waitFor();
  const workTitle='QA ONLY browser coherent work '+Date.now()+' '+Math.random().toString(16).slice(2);
  await page.locator('#work-title-new').fill(workTitle);
  await page.getByRole('button',{name:'Создать работу',exact:true}).click();await page.waitForURL(/\/works\/[0-9a-f]{32}$/);
  const workId=new URL(page.url()).pathname.split('/').at(-1);
  await page.getByRole('heading',{name:workTitle,exact:true}).waitFor();
  await page.locator('.heat-figure canvas').first().waitFor();
  await page.getByText('Контекст и заметки',{exact:true}).click();
  await page.locator('#work-notes').fill('QA technical note; source and run must remain fixed');
  await page.getByRole('button',{name:'Сохранить контекст',exact:true}).click();await page.getByText(/Контекст сохранён/).waitFor();
  await page.reload();await page.locator('.heat-figure canvas').first().waitFor();
  await page.getByText('Контекст и заметки',{exact:true}).click();
  if(await page.locator('#work-notes').inputValue()!=='QA technical note; source and run must remain fixed')throw new Error('Work context lost on reload');
  const sizes=[];
  for(const width of [320,390,430,1440]){
    await page.setViewportSize({width,height:1000});
    const overflowing=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1);
    if(overflowing)throw new Error('Work overflow at '+width);sizes.push(width);
  }
  await page.getByRole('button',{name:'Проверка',exact:true}).click();
  await page.locator('#work-expert').selectOption(expert.id);
  await page.getByRole('button',{name:'Передать сохранённое окно',exact:true}).click();await page.getByText(/отправлен неизменяемый снимок/).waitFor();
  await page.getByRole('link',{name:'Открыть независимую проверку →',exact:true}).click();
  const requestUrl=page.url();
  await page.context().setStorageState('.runtime/product-qa/expert-browser-auth.json');
  await page.goto(requestUrl);await page.getByRole('heading',{name:'Самостоятельный ответ по сигналу',exact:true}).waitFor();
  await page.locator('.review-evidence canvas').first().waitFor();
  if(await page.getByRole('button',{name:'Раскрыть результаты методов',exact:true}).isEnabled())throw new Error('Methods revealed before answer');
  await page.getByText('Недостаточно данных',{exact:true}).click();
  if(!await page.getByLabel('Недостаточно данных',{exact:true}).isChecked())throw new Error('Independent answer not selected');
  await page.getByRole('button',{name:'Сохранить ответ',exact:true}).click();await page.getByText(/Сохранено · ревизия 1/).waitFor();
  await page.getByRole('heading',{name:'Передать проверяемый кейс',exact:true}).waitFor();
  await page.getByRole('button',{name:'Проверить состав кейса',exact:true}).click();await page.getByText(/Автор: QA ONLY participant 10/).waitFor();
  await page.getByRole('button',{name:'Зафиксировать кейс',exact:true}).click();await page.getByRole('link',{name:'Кейс сохранён — открыть →',exact:true}).waitFor();
  await page.getByRole('link',{name:'Кейс сохранён — открыть →',exact:true}).click();await page.getByRole('heading',{name:'Что изменила обработка',exact:true}).waitFor();
  const caseId=new URL(page.url()).pathname.split('/').at(-1);
  const caseData=await page.request.get(origin+'/api/v2/cases/'+caseId).then(r=>r.json());if(caseData.result.evidence.expert_answer.revision!==1)throw new Error('Wrong expert revision in case');
  await page.goto(origin+'/works/'+workId);await page.getByRole('link',{name:'Открыть кейс '+caseId.slice(0,8)+' →',exact:true}).waitFor();
  await page.goto(origin+'/laboratory');await page.getByRole('heading',{name:'Лаборатория',exact:true}).waitFor();
  await page.locator('#lab-run-0').selectOption(run.id);
  const other=runs.find(x=>x.method==='context_local'&&x.source_sha256===run.source_sha256&&JSON.stringify(x.region)===JSON.stringify(run.region));if(!other)throw new Error('Second real result missing');
  await page.locator('#lab-run-1').selectOption(other.id);await page.getByRole('link',{name:'Сравнить в общем окне →',exact:true}).click();await page.locator('.heat-figure canvas').first().waitFor();
  await page.goto(origin+'/laboratory');await page.getByRole('button',{name:'Реальные исследования',exact:true}).click();await page.getByRole('heading',{name:'Real-only исследование',exact:true}).waitFor();
  await page.locator('#lab-track').selectOption('classification');await page.getByText('Обученных классификаторов: 0. Точность: не оценена.',{exact:true}).waitFor();
  await page.goto(origin+'/monitoring?stream_id='+stream.id);await page.getByRole('heading',{name:'Мониторинг измерений',exact:true}).waitFor();await page.getByText(/Точный загруженный профиль/).waitFor();
  for(const width of [320,390,430,1440]){await page.setViewportSize({width,height:1000});if(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1))throw new Error('Monitoring overflow '+width);}
  if(errors.length)throw new Error(JSON.stringify(errors));
  return {work_id:workId,case_id:caseId,real_work_review_case:true,real_lab_comparison:true,real_monitoring_profile:true,mobile_sizes:sizes,page_errors:errors,expert_vote:'technical_QA_only'};
}
