async (page) => {
  const origin = new URL(page.url()).origin;
  const session=new URL(page.url()).searchParams.get('session');
  const health=await page.evaluate(async ()=>(await fetch('/api/v1/health')).json());
  if(origin!=='http://127.0.0.1:8091'||!health.qa_mode||!session)throw new Error('Isolated rehearsal required');
  const report=await page.evaluate(async session=>{
    const csrf=(await(await fetch('/api/v1/session')).json()).mutation_token;
    const response=await fetch(`/api/v1/demo/sessions/${session}/report?format=json`,{method:'POST',headers:{'Content-Type':'application/json','X-Workbench-Token':csrf},body:'{}'});
    if(!response.ok)throw new Error('JSON report failed');
    return response.json();
  },session);
  if(!report.rehearsal||!report.source_sha256||!report.profile)throw new Error('Report provenance missing');
  const offline=await page.context().newPage();
  const external=[];
  offline.on('request',request=>{if(/^https?:/.test(request.url()))external.push(request.url());});
  try {
    await offline.goto('file:///C:/dev/mgts/web-design-delivery/.runtime/demo-goal-20261010/rehearsal-report.html');
    await offline.getByText(/РЕПЕТИЦИЯ/).first().waitFor();
    if(await offline.locator('script,link[href],img[src^="http"]').count())throw new Error('Report requires external assets or scripts');
    if(!await offline.locator('svg').count())throw new Error('Native figures missing');
    if(external.length)throw new Error('Report requested network');
    return {json_provenance:true,html_offline_rendered:true,inline_figures:true,no_external_requests:true,rehearsal_explicit:true};
  } finally {await offline.close();}
}
