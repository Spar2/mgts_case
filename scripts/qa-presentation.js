// Technical answers only on an isolated QA database. Never run against team journals.
async (page) => {
  const origin = new URL(page.url()).origin;
  const health = await page.evaluate(async () => (await fetch('/api/v1/health')).json());
  if (origin !== 'http://127.0.0.1:8091' || !health.qa_mode) throw new Error('Isolated QA required');
  const errors = [];
  page.on('pageerror', error => errors.push(String(error)));
  await page.goto(origin + '/demo');
  await page.getByRole('button', {name:'Начать репетицию', exact:true}).click();
  await page.locator('.presentation-signal .heat-figure canvas').waitFor();
  await page.waitForFunction(() => /WebGL2|Canvas2D/.test(document.querySelector('.heat-figure')?.textContent ?? ''));
  const context = await page.evaluate(async () => {
    const id = new URLSearchParams(location.search).get('session');
    const session = await (await fetch('/api/v1/demo/sessions/' + id)).json();
    const me = await (await fetch('/api/v2/auth/me')).json();
    const queueUrl = '/api/v1/queue?package=' + session.package + '&independent=true&reviewer=' + encodeURIComponent(me.display_name);
    const annotations = (await (await fetch(queueUrl)).json()).annotations;
    return {id,session,queueUrl,annotations};
  });
  const beforeRuns = await page.evaluate(async () => (await (await fetch('/api/v1/runs')).json()).map(r=>r.id));
  await page.locator('.presentation-steps button').nth(1).click();
  await page.getByRole('heading', {name:'Основание наблюдения',exact:true}).waitFor();
  const visual = await page.evaluate(() => ({
    outline: getComputedStyle(document.querySelector('.presentation-caption h2')).outlineStyle,
    background: getComputedStyle(document.querySelector('.presentation-workspace')).backgroundColor,
    headingFocused: document.activeElement === document.querySelector('.presentation-caption h2'),
  }));
  if (visual.outline !== 'none' || visual.headingFocused || visual.background !== 'rgba(0, 0, 0, 0)') throw new Error('Heading focus or empty grey grid returned');
  const neighbor = await page.getByRole('link', {name:'Открыть исходные срезы и соседний контекст →',exact:true}).getAttribute('href');
  if (!neighbor.includes('r0=') || !neighbor.includes('c0=')) throw new Error('Native bounds missing from research link');
  await page.locator('.presentation-steps button').nth(2).click();
  await page.waitForFunction(() => {
    const button = [...document.querySelectorAll('button')].find(b => b.textContent.trim() === 'Получить обработку');
    return button && (!button.disabled || document.body.textContent.includes('ML API недоступен'));
  });
  const processButton = page.getByRole('button',{name:'Получить обработку',exact:true});
  const mlAvailable = await processButton.isEnabled();
  let processResponse;
  if (mlAvailable) {
    const response = page.waitForResponse(r => r.url().endsWith('/process') && r.request().method() === 'POST');
    await processButton.click();
    processResponse = await response;
  }
  else await page.getByText(/ML API недоступен/).waitFor();
  const actualMl = processResponse?.status() === 200;
  if (actualMl) {
    const processing = await processResponse.json();
    if (!Array.isArray(processing.result.denoised)) throw new Error('Processed profile missing');
    await page.getByRole('heading',{name:'Исходный профиль и обработка',exact:true}).waitFor();
    await page.getByRole('heading',{name:'Разность: обработка − исходник',exact:true}).waitFor();
  } else {
    if (processResponse && processResponse.status() !== 503) throw new Error('Unexpected processing failure: '+processResponse.status());
    await page.getByRole('heading',{name:'Исходный профиль · точный срез',exact:true}).waitFor();
    if (!await page.getByRole('heading',{name:'Обработка этого профиля не выбрана',exact:true}).count()) throw new Error('Unavailable ML removed raw fallback');
  }
  await page.locator('.presentation-steps button').nth(3).click();
  await page.locator('label[for="demo-answer-unknown"]').click();
  await page.getByRole('textbox',{name:'Обоснование по сигналу',exact:true}).fill('QA ONLY: проверка репетиции; не человеческая разметка.');
  await page.getByRole('button',{name:'Сохранить ответ репетиции',exact:true}).click();
  await page.getByText('Ответ репетиции сохранён отдельно. Рабочая разметка не изменена.',{exact:true}).waitFor();
  const journal = await page.evaluate(async ({queueUrl}) => (await (await fetch(queueUrl)).json()).annotations, context);
  if (JSON.stringify(journal)!==JSON.stringify(context.annotations)) throw new Error('Rehearsal altered the real journal');
  const blocked = await page.evaluate(async () => {
    const csrf = (await (await fetch('/api/v1/session')).json()).mutation_token;
    return (await fetch('/api/v1/reviews',{method:'POST',headers:{'Content-Type':'application/json','X-Workbench-Token':csrf,'Idempotency-Key':'technical-rehearsal-bypass'},body:'{}'})).status;
  });
  if (blocked !== 403) throw new Error('Working mutation was not protected server-side');
  await page.locator('.presentation-steps button').nth(4).click();
  const download = page.waitForEvent('download');
  await page.getByRole('button',{name:'Скачать автономный отчёт',exact:true}).click();
  await (await download).saveAs('.runtime/demo-goal-20261010/rehearsal-report.html');
  const finalUrl = page.url();
  await page.reload();
  await page.getByRole('heading',{name:'Проверяемый итог',exact:true}).waitFor();
  await page.getByText(/QA ONLY: проверка репетиции/).first().waitFor();
  if (page.url()!==finalUrl) throw new Error('Direct route not restored');
  const second = await page.context().newPage();
  try { await second.goto(finalUrl); await second.getByRole('heading',{name:'Проверяемый итог',exact:true}).waitFor(); }
  finally {await second.close();}
  await page.locator('.presentation-caption h2').focus();
  await page.keyboard.press('ArrowLeft');
  await page.getByRole('heading',{name:'Самостоятельное наблюдение',exact:true}).waitFor();
  await page.getByRole('textbox',{name:'Обоснование по сигналу',exact:true}).focus();
  await page.keyboard.press('ArrowLeft');
  if(new URL(page.url()).searchParams.get('step')!=='3') throw new Error('Arrow inside text changed route');
  const widths=[];
  for(const width of [320,390,430,1024,1440]) {
    await page.setViewportSize({width,height:900});
    let geometry;
    for(let i=0;i<5;i++){
      await page.locator('.presentation-steps button').nth(i).click();
      await page.waitForFunction(()=>document.documentElement.scrollWidth<=innerWidth);
      const current = await page.evaluate(() => ({
        pageHeight: document.documentElement.scrollHeight,
        stageHeight: document.querySelector('.presentation-workspace').getBoundingClientRect().height,
        stageTop: document.querySelector('.presentation-workspace').getBoundingClientRect().top + scrollY,
        nextTop: document.querySelector('.presentation-navigation').getBoundingClientRect().top + scrollY,
      }));
      if (!geometry) geometry = current;
      else for (const key of Object.keys(geometry)) if (Math.abs(current[key]-geometry[key])>1) throw new Error(`Step layout moved at ${width}px: ${key}`);
    }
    widths.push(width);
  }
  await page.emulateMedia({reducedMotion:'reduce'});
  if(!await page.evaluate(()=>matchMedia('(prefers-reduced-motion: reduce)').matches))throw new Error('Reduced motion not applied');
  const reduced = await page.locator('.presentation-steps button').first().evaluate(button=>({transition:getComputedStyle(button).transitionDuration,animation:getComputedStyle(button).animationDuration}));
  if ([reduced.transition,reduced.animation].some(value=>value.split(',').some(duration=>parseFloat(duration)>0.001)))throw new Error('Reduced motion leaves a long CSS animation');
  await page.locator('.presentation-steps button').nth(1).click();
  await page.getByRole('heading',{name:'Температура выбранной координаты во времени',exact:true}).waitFor();
  await page.route('**/demo/sessions/*/evidence*', route=>route.abort('failed'));
  const row = Number(await page.getByRole('textbox',{name:'Профиль (исходный индекс)',exact:true}).inputValue());
  const next = row+1<context.session.region.row_end ? row+1 : row-1;
  await page.getByRole('textbox',{name:'Профиль (исходный индекс)',exact:true}).fill(String(next));
  await page.getByRole('textbox',{name:'Профиль (исходный индекс)',exact:true}).press('Enter');
  await page.getByText('Не удалось обновить данные',{exact:true}).waitFor();
  if (!await page.locator('.presentation-signal .heat-figure canvas').count()) throw new Error('Network failure removed the loaded field');
  if (!await page.getByRole('heading',{name:'Исходный профиль · точный срез',exact:true}).count()) throw new Error('Network failure removed the loaded profile');
  await page.unroute('**/demo/sessions/*/evidence*');
  await page.getByRole('button',{name:'Повторить подключение',exact:true}).click();
  await page.waitForFunction(()=>!document.body.textContent.includes('Не удалось обновить данные'));
  const afterRuns = await page.evaluate(async () => (await (await fetch('/api/v1/runs')).json()).map(r=>r.id));
  if(JSON.stringify(beforeRuns)!==JSON.stringify(afterRuns))throw new Error('Comparison created regular history');
  if(errors.length)throw new Error(errors.join('\n'));
  return {record:context.session.region.record_id,candidate:context.session.candidate,source_sha256:context.session.source_sha256,steps:5,actual_ml:actualMl,working_journal_unchanged:true,working_mutation_blocked:true,regular_runs_unchanged:true,direct_link:true,reload:true,keyboard:true,reduced_motion:true,network_recovery:true,widths,errors};
}
