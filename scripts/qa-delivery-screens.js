// Read-only screenshots of the working local demo; never saves expert answers.
async (page) => {
  const origin = 'http://127.0.0.1:8081';
  async function settle() {
    await page.evaluate(()=>window.scrollTo({top:0,behavior:'instant'}));
    await page.evaluate(async()=>{await document.fonts.ready; await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));});
  }
  await page.setViewportSize({width:1440,height:1000});
  await page.goto(origin+'/demo?package=real-data-v2&candidate=review-00000');
  await page.locator('.region-panel .heat-figure canvas').waitFor();
  await settle();
  await page.screenshot({path:'docs/web/screens/demo-desktop.png',fullPage:true});
  await page.getByRole('button',{name:'Сравнить обработку',exact:true}).click();
  await page.getByRole('heading',{name:'Обработанный профиль · та же область и шкала',exact:true}).waitFor();
  await settle();
  await page.screenshot({path:'docs/web/screens/comparison-desktop.png',fullPage:true});
  await page.setViewportSize({width:390,height:844});
  await page.goto(origin+'/review?package=real-data-v2&card=review-00000');
  await page.getByRole('button',{name:'1 · Сигнал',exact:true}).click();
  await page.locator('.heat-figure canvas').waitFor();
  await settle();
  await page.screenshot({path:'docs/web/screens/review-mobile-signal.png',fullPage:true});
  await page.getByRole('button',{name:'2 · Ответ',exact:true}).click();
  await settle();
  await page.screenshot({path:'docs/web/screens/review-mobile-answer.png',fullPage:true});
  const health=await page.evaluate(async()=>(await fetch('/api/v1/health')).json());
  const queue=await page.evaluate(async()=>(await fetch('/api/v1/queue')).json());
  return {health,legacyAnnotations:Object.keys(queue.annotations).length,screens:4};
}
