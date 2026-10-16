async (page) => {
  const origin=new URL(page.url()).origin;
  const tab=await page.context().newPage();
  try {
    await tab.bringToFront();await tab.goto(origin+'/research');
    await tab.locator('.heat-figure canvas').waitFor();
    await tab.getByRole('combobox',{name:'Скорость replay',exact:true}).selectOption('60');
    await tab.getByRole('button',{name:'Воспроизвести',exact:true}).click();
    await tab.getByRole('button',{name:'Пауза',exact:true}).waitFor();
    const first=await tab.locator('.replay-bar').getAttribute('data-replay-id');
    await tab.getByRole('button',{name:'Пауза',exact:true}).click();
    await tab.getByRole('button',{name:'Воспроизвести',exact:true}).waitFor();
    const paused=await tab.evaluate(async key=>(await fetch('/api/v1/replays/'+key)).json(),first);
    if(paused.playing || paused.speed!==60)throw new Error('Pause/speed command failed');
    await tab.locator('#row').fill('100');await tab.locator('#row').press('Enter');
    await tab.waitForFunction(async key=>(await (await fetch('/api/v1/replays/'+key)).json()).row===100,first);
    await tab.evaluate(async key=>{
      const token=(await(await fetch('/api/v1/session')).json()).mutation_token;
      await fetch('/api/v1/replays/'+key,{method:'DELETE',headers:{'X-Workbench-Token':token}});
    },first);
    await tab.getByRole('button',{name:'Восстановить replay',exact:true}).waitFor();
    await tab.getByRole('button',{name:'Восстановить replay',exact:true}).click();
    await tab.getByRole('button',{name:'Воспроизвести',exact:true}).click();
    await tab.getByRole('button',{name:'Пауза',exact:true}).waitFor();
    const recovered=await tab.locator('.replay-bar').getAttribute('data-replay-id');
    if(first===recovered)throw new Error('Recovery did not create a new session');
    await tab.getByRole('combobox',{name:'Запись',exact:true}).selectOption('3');
    await tab.waitForFunction(async key=>(await fetch('/api/v1/replays/'+key)).status===404,recovered);
    return {pause:true,speed:60,seek:100,lostSessionRecovery:true,recordSwitchReleased:true};
  }finally{await tab.close();}
}
