for(const root of ['m3p3A','qVvEA','T48mcW','ElvPP','hN1eX','gDbeh','C7Gl9']) {
 Update(root,{placeholder:true});
 Get(root,n=>{if(n.type==='text' && n.textGrowth==='auto')Update(n.id,{width:undefined,height:undefined});});
 Update(root,{placeholder:false});
}
Print(Get('vn1BE',{depth:1})); Print(Get('i4mFN',{depth:1}));
Get('m3p3A',(n,c)=>c.depth<2 && Print(n.id,n.name,c.bounds));
for(const root of ['m3p3A','qVvEA','T48mcW','ElvPP','hN1eX','gDbeh','C7Gl9'])Get(root,(n,c)=>c.problems && Print('BOUNDARY',n.id,n.name,c.parentCtx?.node.name,c.problems));
TakeScreenshot(['HQDiT','Ih4TR']);
Export(['m3p3A','qVvEA','HQDiT','Ih4TR'],'png','C:/dev/mgts/hackathon-delivery/design/landing-r4/previews',{scale:1});
