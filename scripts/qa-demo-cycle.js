// Start a bounded, read-only collector and return immediately; read results later.
async (page) => {
  const origin=new URL(page.url()).origin;
  await page.setViewportSize({width:1440,height:900});
  await page.emulateMedia({reducedMotion:"reduce"});
  await page.goto(origin+"/demo?package=real-data-v2");
  await page.locator(".region-panel .heat-figure canvas").waitFor();
  await page.evaluate(()=>{
    const stats={startedAt:new Date().toISOString(),userAgent:navigator.userAgent,viewport:[innerWidth,innerHeight],samples:[],errors:[],finished:false};
    window.__dtsDemoCycle=stats;
    const started=performance.now();
    const sample=()=>{stats.samples.push({elapsedMs:performance.now()-started,candidate:document.querySelector("#candidate-select")?.value,cache:window.__dtsCacheStats?.(),graphics:window.__dtsGraphicsStats?.(),heapBytes:performance.memory?.usedJSHeapSize,canvases:document.querySelectorAll("canvas").length});if(performance.now()-started>=310000){clearInterval(timer);stats.finished=true;stats.finishedAt=new Date().toISOString();}};
    addEventListener("error",e=>stats.errors.push(String(e.message)));
    const timer=setInterval(sample,5000);sample();
  });
  await page.getByRole("button",{name:"Автопереход",exact:true}).click();
  return {collectorStarted:true,durationMs:310000};
}
