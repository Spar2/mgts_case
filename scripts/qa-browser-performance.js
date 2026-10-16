// Read-only measurements on isolated real records. No FPS claim for a phone.
async (page) => {
  const origin=new URL(page.url()).origin;
  const health=await page.evaluate(async()=>(await fetch('/api/v1/health')).json());
  if(new URL(origin).hostname!=='127.0.0.1'||!health.qa_mode)throw new Error('Isolated loopback QA required');
  const measured=await page.context().newPage();
  const errors=[];measured.on('pageerror',e=>errors.push(String(e)));
  await measured.addInitScript(()=>{
    const counts={active:0,created:0};window.__qaWorkerStats=counts;
    const NativeWorker=window.Worker;
    window.Worker=class extends NativeWorker {
      constructor(...args){super(...args);counts.active++;counts.created++;this.qaActive=true;}
      terminate(){if(this.qaActive){counts.active--;this.qaActive=false;}return super.terminate();}
    };
    window.__qaLongTasks=[];
    if(PerformanceObserver.supportedEntryTypes.includes('longtask'))new PerformanceObserver(list=>{
      for(const entry of list.getEntries())window.__qaLongTasks.push({start:entry.startTime,duration:entry.duration});
    }).observe({type:'longtask',buffered:true});
  });
  const stats=()=>measured.evaluate(()=>({cache:window.__dtsCacheStats?.(),graphics:window.__dtsGraphicsStats?.(),workers:window.__qaWorkerStats,heap:performance.memory?.usedJSHeapSize??null}));
  const ready=(sha)=>measured.waitForFunction(expected=>{
    const figure=document.querySelector('.heat-figure');
    return figure?.dataset.sourceSha256===expected && figure?.querySelector('canvas')?.width>0 && !figure.querySelector('.heat-status') && /WebGL2|Canvas2D/.test(figure.textContent);
  },sha);
  const frames=()=>measured.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
  const samples=[];
  const catalog=await page.evaluate(async()=>(await fetch('/api/v1/records')).json());
  try {
    for(const width of [1440,390]){
      await measured.setViewportSize({width,height:900});
      const begin=Date.now();await measured.goto(origin+'/research?record=5');await ready(catalog.find(r=>r.id==='5').source_sha256);await frames();
      const firstUsefulMs=Date.now()-begin;
      const records=await measured.getByRole('combobox',{name:'Запись',exact:true}).locator('option').evaluateAll(nodes=>nodes.map(n=>n.value));
      const changes=[];
      for(let round=0;round<3;round++)for(const record of records){
        const start=Date.now();
        await measured.getByRole('combobox',{name:'Запись',exact:true}).selectOption(record);
        await measured.waitForFunction(id=>new URLSearchParams(location.search).get('record')===id,record);
        await ready(catalog.find(r=>r.id===record).source_sha256);await frames();
        changes.push({record,round,ms:Date.now()-start,...await stats()});
      }
      const localStart=Date.now();await measured.getByRole('button',{name:'Развернуть поле',exact:true}).click();
      await measured.locator('.plots.expanded').waitFor();await frames();const expandMs=Date.now()-localStart;
      await measured.getByRole('button',{name:'Вернуть панели',exact:true}).click();
      const loaded=await stats();
      const longTasks=await measured.evaluate(()=>window.__qaLongTasks);
      if(width<800)await measured.locator('.mobile-menu').click();
      await measured.getByRole('link',{name:'Система',exact:true}).click();
      await measured.getByRole('heading',{name:'Система и путь измерения',exact:true}).waitFor();await frames();
      const released=await stats();
      if(released.workers.active!==0||released.graphics.webglContexts!==0||released.graphics.bufferBytes!==0||released.graphics.textureBytes!==0)throw new Error('Graphics/worker resources survived route exit');
      if(released.cache.bytes>released.cache.budget)throw new Error('Binary cache exceeded budget');
      samples.push({width,firstUsefulMs,expandMs,changes,loaded,released,longTasks});
    }
    if(errors.length)throw new Error(errors.join('\n'));
    return {userAgent:await measured.evaluate(()=>navigator.userAgent),devicePixelRatio:await measured.evaluate(()=>devicePixelRatio),conditions:'Headless desktop browser; loopback; real six records; mobile viewport is not a physical phone. Useful field waits for decode, canvas and two animation frames. Timings include Playwright action/wait overhead. Heap approximate, no forced GC. Cached blocks warmed by prior QA.',samples,errors};
  } finally {await measured.close();}
}
