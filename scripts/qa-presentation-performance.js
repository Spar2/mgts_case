// Read-only measurement on an already-created isolated rehearsal session.
async (page) => {
  const origin = new URL(page.url()).origin;
  const health = await page.evaluate(async ()=>(await fetch('/api/v1/health')).json());
  if(origin!=='http://127.0.0.1:8091'||!health.qa_mode)throw new Error('Isolated QA required');
  const session = new URL(page.url()).searchParams.get('session');
  const runs=[];
  for(const width of [1440,390]) {
    const measured = await page.context().newPage();
    await measured.setViewportSize({width,height:900});
    const cdp=await measured.context().newCDPSession(measured);
    try {
      await cdp.send('Network.enable');
      await cdp.send('Network.clearBrowserCache');
      await cdp.send('Network.setCacheDisabled',{cacheDisabled:true});
      const start=Date.now();
      await measured.goto(`${origin}/demo?session=${session}&step=2`);
      await measured.locator('.presentation-signal .heat-figure canvas').waitFor();
      await measured.waitForFunction(()=>/WebGL2|Canvas2D/.test(document.querySelector('.heat-figure')?.textContent??''));
      const firstFieldMs=Date.now()-start;
      await cdp.send('Network.setCacheDisabled',{cacheDisabled:false});
      await measured.screenshot({path:`docs/web/screenshots/demo-goal-${width}.png`,fullPage:true});
      const snapshots=[],transitions=[];
      for(let cycle=0;cycle<10;cycle++) {
        for(let step=0;step<5;step++) {
          transitions.push(await measured.evaluate(async step=>{
            const start=performance.now();
            document.querySelectorAll('.presentation-steps button')[step].click();
            await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));
            return performance.now()-start;
          },step));
        }
        await cdp.send('HeapProfiler.collectGarbage');
        const heap=await cdp.send('Runtime.getHeapUsage');
        snapshots.push(await measured.evaluate(heap=>({heap,cache:window.__dtsCacheStats(),graphics:window.__dtsGraphicsStats()}),heap));
      }
      transitions.sort((a,b)=>a-b);
      if(width<672)await measured.locator('.shell button[aria-controls="main-navigation"]').click();
      await measured.getByRole('link',{name:'Исследование',exact:true}).first().click();
      await measured.locator('.heat-figure canvas').waitFor();
      if(width<672)await measured.locator('.shell button[aria-controls="main-navigation"]').click();
      await measured.getByRole('link',{name:'Система',exact:true}).first().click();
      await measured.waitForFunction(()=>{const s=window.__dtsGraphicsStats();return s.webglContexts===0&&s.bufferBytes===0&&s.textureBytes===0&&s.canvasSurfaceBytes===0;});
      if(measured.workers().length!==0)throw new Error('Chart workers remain after leaving graphics');
      const released=await measured.evaluate(()=>window.__dtsGraphicsStats());
      runs.push({width,first_field_ms:firstFieldMs,transition_two_frame_p50_ms:transitions[Math.floor(transitions.length/2)],transition_two_frame_max_ms:transitions.at(-1),snapshots,released});
    } finally {await cdp.detach();await measured.close();}
  }
  return {date:'2026-10-10',browser:page.context().browser().version(),network:'loopback, no throttling, prepared server data; HTTP cache cleared and disabled for first field, enabled for repeated transitions; fresh page application context',device:'Windows desktop; 390px is viewport emulation, not a physical phone',cycles:10,steps_per_cycle:5,heap:'Chromium CDP after explicit GC; excludes GPU, native and process RSS',runs};
}
