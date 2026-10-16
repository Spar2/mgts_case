// Exercises real rendering failure paths on fresh pages; does not save answers.
async (page) => {
  const origin=new URL(page.url()).origin;
  const canvasPage=await page.context().newPage();
  const errors=[];
  canvasPage.on('pageerror',e=>errors.push(String(e)));
  try {
    await canvasPage.bringToFront();
    await canvasPage.goto(origin+'/research');
    await canvasPage.locator('.heat-figure canvas').waitFor();
    const lost=await canvasPage.evaluate(()=>{
      const gl=document.querySelector('.heat-figure canvas').getContext('webgl2');
      const extension=gl?.getExtension('WEBGL_lose_context');
      if(!extension)return false;
      extension.loseContext();return true;
    });
    if(lost){
      await canvasPage.getByText('Canvas2D — резервный режим',{exact:false}).waitFor();
      const stats=await canvasPage.evaluate(()=>window.__dtsGraphicsStats());
      if(stats.webglContexts!==0 || stats.bufferBytes!==0)throw new Error('GPU resource accounting not released');
    }
    await canvasPage.addInitScript(()=>{
      const original=HTMLCanvasElement.prototype.getContext;
      HTMLCanvasElement.prototype.getContext=function(type,...args){
        return type==='webgl2'?null:original.call(this,type,...args);
      };
    });
    await canvasPage.reload();
    await canvasPage.getByText('Canvas2D — резервный режим',{exact:false}).waitFor();
    await canvasPage.locator('.heat-figure canvas').waitFor();
    await canvasPage.goto(origin+'/review?source=5.csv');
    await canvasPage.getByRole('heading',{name:'Ваше решение',exact:true}).waitFor();
    await canvasPage.locator('label[for="answer-unknown"]').click();
    await canvasPage.locator('#answer-unknown').focus();
    await canvasPage.keyboard.press('ArrowUp');
    if(!await canvasPage.locator('#answer-ambiguous').isChecked())throw new Error('Radio keyboard navigation failed');
    if(errors.length)throw new Error(errors.join('\n'));
    return {contextLossFallback:lost,unavailableWebglFallback:true,keyboardRadio:true,runtimeErrors:errors};
  }finally {await canvasPage.close();}
}
