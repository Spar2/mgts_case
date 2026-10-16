const roots=['m3p3A','qVvEA','T48mcW','ElvPP','hN1eX','gDbeh','C7Gl9','pp4J8','D8oTd','T0xeyJ'];
for(const root of roots) {
 Update(root,{placeholder:true});
 Get(root,(n,c)=>{if(n.type==='path' && n.width==='fill_container')Update(n.id,{width:c.parentCtx.bounds.width});});
 const plots=Get(root,(n,c)=>n.metadata?.shape && n.metadata?.type==='scientific-plot'?{id:n.id,w:c.bounds.width}:undefined);
 for(const p of plots)Get(p.id,(n,c)=>{
  if(n.name==='Выбранный момент · 17:35:15')Update(n.id,{width:p.w});
  if(n.name==='Выбранное расстояние · 840 м')Update(n.id,{x:p.w*42.5/86});
 });
 const axes=Get(root,(n,c)=>n.name==='Температурная ось · °C'?{id:n.id,h:c.bounds.height}:undefined);
 for(const a of axes) {
  Update(a.id,{padding:[a.h/4-6.6,0]});
  Get(a.id,n=>{if(n.type==='text'){if(n.content==='32' || n.content==='16')Update(n.id,{enabled:false});else Update(n.id,{lineHeight:1.2});}});
 }
 Update(root,{placeholder:false});
 Get(root,(n,c)=>{if(n.enabled===false){c.skipChildren();return;}if(c.problems && !(n.type==='rectangle' && n.fill?.type==='image'))Print('BOUNDARY',n.id,n.name,c.parentCtx?.node.name,c.problems);});
}
Get((n,c)=>{if(c.depth===0){Print('ROOT',n.id,n.name,c.bounds);c.skipChildren();}});
Export([...roots,'QZS9a','HQDiT','JJxfR','A3ENu2','p8ZGU'],'png','C:/dev/mgts/hackathon-delivery/design/landing-r4/previews',{scale:1});
