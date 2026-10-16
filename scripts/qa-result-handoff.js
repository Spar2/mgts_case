// Uses a previously calculated real profile; no invented values or training.
async (page) => {
  const origin = new URL(page.url()).origin;
  const health = await page.evaluate(async () => (await fetch('/api/v1/health')).json());
  if (origin !== 'http://127.0.0.1:8091' || !health.qa_mode) throw new Error('Isolated QA required');
  const context = /* QA_CONTEXT */ null;
  const envelope = /* QA_ENVELOPE */ null;
  if (!context || !envelope) throw new Error('Inject actual-context.json and actual-handoff.json into a private copy before running; never invent model output');
  await page.getByRole('button', {name:'Завершить репетицию',exact:true}).click();
  await page.getByRole('button', {name:'Завершить репетицию',exact:true}).waitFor({state:'hidden'});
  await page.goto(origin+'/runs');
  const importedResponse = page.waitForResponse(r=>r.url().endsWith('/results')&&r.request().method()==='POST');
  await page.locator('input[type="file"]').setInputFiles('.runtime/demo-goal-20261010/actual-handoff.json');
  await page.getByRole('button', {name:'Подтвердить импорт',exact:true}).click();
  const response = await importedResponse;
  if(response.status()!==201)throw new Error('Real prepared import failed: '+response.status());
  const run = await response.json();
  await page.goto(`${origin}/demo?session=${context.session}&step=2&row=${context.row}`);
  await page.getByRole('button',{name:'Обновить список результатов',exact:true}).click();
  await page.locator('#demo-result').selectOption(run.id);
  const processedResponse = page.waitForResponse(r=>r.url().endsWith('/process')&&r.request().method()==='POST');
  await page.getByRole('button',{name:'Подключить результат',exact:true}).click();
  const processed = await processedResponse;
  if(processed.status()!==200)throw new Error('Prepared result not connected');
  const actual = await processed.json();
  if(actual.source_sha256!==envelope.source_sha256)throw new Error('Source identity changed');
  const evidence = await page.evaluate(async ({session,row})=>(await (await fetch(`/api/v1/demo/sessions/${session}/evidence?row=${row}`)).json()),context);
  const expected=envelope.result.denoised.map((v,i)=>evidence.selection.profile[i] === null ? null : v);
  if(JSON.stringify(actual.result.denoised)!==JSON.stringify(expected))throw new Error('Prepared values differ');
  await page.getByRole('heading',{name:'Исходный профиль и обработка',exact:true}).waitFor();
  await page.route('**/demo/sessions/*/process',route=>route.fulfill({status:404,contentType:'application/json',body:JSON.stringify({detail:'QA: prepared result unavailable'})}));
  await page.getByRole('button',{name:'Подключить результат',exact:true}).click();
  await page.getByText(/QA: prepared result unavailable/).first().waitFor();
  if(!await page.getByRole('heading',{name:'Исходный профиль и обработка',exact:true}).count())throw new Error('Missing result removed raw data');
  await page.unroute('**/demo/sessions/*/process');
  return {actual_existing_ml_result:true,import_bytes:Buffer.byteLength(JSON.stringify(envelope)),source_sha_verified:true,native_profile_length:expected.length,exact_values:true,native_invalid_mask:true,missing_result_preserves_raw:true};
}
