const roots=['m3p3A','qVvEA','T48mcW','ElvPP','hN1eX','gDbeh','C7Gl9','pp4J8','D8oTd','T0xeyJ'];
for(const root of roots) {
 Update(root,{placeholder:true});
 Get(root,(n,c)=>{
  if(n.type==='text' && n.content?.includes('↗'))Update(n.id,{content:n.content.replace('↗','→')});
  if(n.type==='frame' && n.name?.startsWith('МГТС ·'))Update(n.id,{width:144,height:36});
  if(n.type==='frame' && n.name?.startsWith('LARP CORP ·'))Update(n.id,{width:116,height:54});
  if(n.type==='rectangle' && n.fill?.url?.includes('mgts-original'))Update(n.id,{x:0,y:-103});
  if(n.type==='rectangle' && n.fill?.url?.includes('larp-corp-original'))Update(n.id,{x:-14,y:-48});
 });
 Update(root,{placeholder:false});
}
const pos=FindEmptySpace({width:1440,height:1000,nodeId:'m3p3A',direction:'top',padding:100});
const first=Insert(document,{type:'frame',name:'Landing R4 · Первый экран 1440',x:pos.x,y:pos.y,width:1440,height:'fit_content',layout:'vertical',fill:'#ffffff',clip:true,placeholder:true});
for(const id of ['JvaNg','QZS9a','A3ENu2'])Copy(id,first);
Update(first,{placeholder:false});
const all=[...roots,first];
for(const root of all)Get(root,(n,c)=>{if(n.enabled===false){c.skipChildren();return;}if(c.problems && !(n.type==='rectangle' && n.fill?.type==='image'))Print('BOUNDARY',n.id,n.name,c.parentCtx?.node.name,c.problems);});
Print({first});
Get('qVvEA',(n,c)=>c.depth===1 && Print('MOBILE_SECTION',n.id,n.name,c.bounds));
Export(all,'png','C:/dev/mgts/hackathon-delivery/design/landing-r4/png',{scale:2});
Export(['HQDiT','JJxfR','p8ZGU',first],'png','C:/dev/mgts/hackathon-delivery/design/landing-r4/previews',{scale:1});
Get('qVvEA',(n,c)=>c.depth===1 && Export([n.id],'png','C:/dev/mgts/hackathon-delivery/design/landing-r4/mobile-review',{scale:1}));
