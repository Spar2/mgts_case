const roots=['m3p3A','qVvEA','T48mcW','ElvPP','hN1eX','gDbeh','C7Gl9','pp4J8','D8oTd','T0xeyJ','hWzL5'];
for(const root of roots) {
 Update(root,{placeholder:true});
 const top=Get(root,{depth:1});
 for(const n of top.children ?? [])if(n.type==='frame' && !n.fill)Update(n.id,{fill:top.fill ?? '#ffffff'});
 if(root==='D8oTd')Get(root,n=>n.name==='МГТС и LARP CORP' && Update(n.id,{gap:12}));
 Get(root,n=>n.name==='Ограничение публичного примера' && Update(n.id,{content:'Иллюстрация реальных данных. Часовой пояс не указан. Рабочий архив доступен участникам после входа.'}));
 Update(root,{placeholder:false});
 Get(root,(n,c)=>{if(n.enabled===false){c.skipChildren();return;}if(c.problems && !(n.type==='rectangle' && n.fill?.type==='image'))Print('BOUNDARY',n.id,n.name,c.parentCtx?.node.name,c.problems);});
}
Export(roots,'png','C:/dev/mgts/hackathon-delivery/design/landing-r4/png',{scale:2});
Export(['hWzL5','HQDiT','JJxfR'],'png','C:/dev/mgts/hackathon-delivery/design/landing-r4/previews',{scale:1});
Get('qVvEA',(n,c)=>c.depth===1 && Export([n.id],'png','C:/dev/mgts/hackathon-delivery/design/landing-r4/mobile-review',{scale:1}));
Get('D8oTd',(n,c)=>c.depth===1 && n.name==='Первый экран · мобильный' && Export([n.id],'png','C:/dev/mgts/hackathon-delivery/design/landing-r4/mobile-review',{scale:1}));
