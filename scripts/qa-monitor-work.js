async (page) => {
  const origin=new URL(page.url()).origin;
  await page.context().setStorageState('.runtime/product-qa/browser-auth.json');
  const health=await page.request.get(origin+'/api/v1/health').then(r=>r.json());if(!health.qa_mode)throw new Error('Isolated QA only');
  const streams=await page.request.get(origin+'/api/v2/streams').then(r=>r.json());const stream=streams.items.find(s=>s.record_id==='4'&&s.status==='complete');if(!stream)throw new Error('Verified real replay missing');
  await page.setViewportSize({width:1440,height:1000});
  await page.goto(origin+'/monitoring?stream_id='+stream.id);await page.getByText(/Точный загруженный профиль/).waitFor();
  const accepted=page.waitForResponse(r=>new URL(r.url()).pathname==='/api/v2/analysis-jobs'&&r.request().method()==='POST');
  await page.getByRole('button',{name:'Запустить расчёт поля',exact:true}).click();const response=await accepted;if(response.status()!==202)throw new Error('Analysis was not accepted');const job=await response.json();if(job.region.row_end>stream.verified_persisted+1)throw new Error('Future observations analysed');
  await page.getByRole('link',{name:'Открыть результат расчёта →',exact:true}).waitFor({timeout:180000});
  const final=await page.request.get(origin+'/api/v2/analysis-jobs/'+job.id).then(r=>r.json());if(!final.run_id)throw new Error('No persisted calculation');
  await page.reload();await page.getByText(/Точный загруженный профиль/).waitFor();
  await page.getByRole('link',{name:'Начать разбор результата →',exact:true}).click();
  await page.getByRole('heading',{name:'Работы команды',exact:true}).waitFor();await page.locator('#work-title-new').fill('QA ONLY verified replay analysis work');
  await page.getByRole('button',{name:'Создать работу',exact:true}).click();await page.waitForURL(/\/works\/[0-9a-f]{32}$/);
  const id=new URL(page.url()).pathname.split('/').at(-1);const work=await page.request.get(origin+'/api/v2/investigations/'+id).then(r=>r.json());
  if(work.origin!=='monitoring'||work.stream_id!==stream.id||work.context.run_ids[0]!==final.run_id)throw new Error('Monitoring provenance lost');
  await page.locator('.heat-figure canvas').first().waitFor();await page.getByRole('button',{name:'Проверить состав кейса',exact:true}).click();
  await page.getByRole('button',{name:'Зафиксировать кейс',exact:true}).click();const link=page.getByRole('link',{name:'Кейс сохранён — открыть →',exact:true});await link.waitFor();await link.click();
  const caseId=new URL(page.url()).pathname.split('/').at(-1);const saved=await page.request.get(origin+'/api/v2/cases/'+caseId).then(r=>r.json());if(saved.run_id!==final.run_id)throw new Error('Case chose another result');
  return {stream_id:stream.id,verified_profiles:stream.verified_persisted+1,verified_cells:stream.verified_cells,analysis_job:job.id,run_id:final.run_id,work_id:id,case_id:caseId,replay_analysis_work_case:true,no_equipment_claim:true};
}
