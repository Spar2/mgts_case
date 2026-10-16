const roots=['m3p3A','qVvEA','T48mcW','ElvPP','hN1eX','gDbeh','C7Gl9'];
for(const id of roots)Update(id,{placeholder:true});
for(const root of ['m3p3A','qVvEA']) {
 const plots=Get(root,(n,c)=>n.name==='Пространственный профиль · точные значения'?{id:n.id,parent:c.parentCtx.node.id,index:c.index,w:c.bounds.width,h:c.bounds.height}:undefined);
 for(const item of plots) {
  const w=Insert(item.parent,{type:'frame',name:'Профиль с температурной осью',layout:'horizontal',width:'fill_container',height:item.h,gap:8});Move(w,item.parent,item.index);
  const a=Insert(w,{type:'frame',name:'Температурная ось · °C',layout:'vertical',width:28,height:item.h,justifyContent:'space_between'});
  for(const t of ['32','28','24','20','16'])Insert(a,{type:'text',name:t+' °C',content:t,fontFamily:'IBM Plex Sans',fontSize:11,fill:'#525252'});
  Move(item.id,w);Update(item.id,{width:item.w-36});
  Get(item.id,(n,c)=>c.depth===1 && n.type==='path' && Update(n.id,{width:item.w-36}));
 }
}
Get('qVvEA',(n,c)=>n.name==='Первый экран · мобильный' && Update(n.id,{fill:'#ffffff'}));
const mobileTabs=Get('qVvEA',n=>n.name==='Выбор возможностей'?n.id:undefined)[0];
const selected=Get(mobileTabs,{depth:1}).children.find(n=>n.name==='Очистка от шумов').id;
for(const n of Get(mobileTabs,{depth:1}).children)if(n.id!==selected)Update(n.id,{enabled:false});
Update(selected,{fill:'#ffffff',stroke:'#8d8d8d',strokeWidth:1,padding:[14,16]});
Get(selected,n=>n.type==='text' && Update(n.id,{content:'Очистка от шумов     ⌄',fontSize:16}));
Update(mobileTabs,{strokeWidth:0});
const mobileHeat=Get('qVvEA',n=>n.metadata?.type==='scientific-plot' && n.metadata.shape?n.id:undefined)[0];
Update(mobileHeat,{width:'fill_container'});
Get(mobileHeat,(n,c)=>c.depth===1 && n.type==='path' && Update(n.id,{width:'fill_container'}));
for(const root of roots)Update(root,{placeholder:false});
for(const root of roots)Get(root,(n,c)=>c.problems && !n.name.includes('оригинал') && !n.name.includes('Оригинал') && Print('BOUNDARY',n.id,n.name,c.parentCtx?.node.name,c.problems));
const at=FindEmptySpace({width:390,height:700,nodeId:'C7Gl9',direction:'right',padding:100});
const open=Insert(document,{type:'frame',name:'Landing R4 · Mobile · выбор возможностей',x:at.x,y:at.y,width:390,height:'fit_content',layout:'vertical',clip:true,fill:'#f4f4f4',padding:24,gap:24,placeholder:true});
Insert(open,{type:'text',name:'Выбор возможностей',content:'Возможности платформы',fontFamily:'IBM Plex Sans',fontSize:28,fontWeight:'500',textGrowth:'fixed-width',width:'fill_container',fill:'#161616'});
const select=Insert(open,{type:'frame',name:'Выбор · раскрыто',layout:'vertical',width:'fill_container',stroke:'#8d8d8d',strokeWidth:1});
for(const label of ['Предобработка','Очистка от шумов','Аномалии','Компьютерное зрение','Трансформеры']) {
 const r=Insert(select,{type:'frame',name:label,layout:'horizontal',width:'fill_container',padding:[16,16],fill:label==='Очистка от шумов'?'#edf5ff':'#ffffff',stroke:'#e0e0e0',strokeWidth:{bottom:1}});
 Insert(r,{type:'text',name:label+' · вариант',content:label+(label==='Очистка от шумов'?'  ✓':''),fontFamily:'IBM Plex Sans',fontSize:16,fill:label==='Очистка от шумов'?'#0043ce':'#161616',textGrowth:'fixed-width',width:'fill_container'});
}
Insert(open,{type:'text',name:'Поведение выбора',content:'После выбора список закрывается, ниже появляется описание и пример. Фокус остаётся на переключателе.',fontFamily:'IBM Plex Sans',fontSize:16,lineHeight:1.5,textGrowth:'fixed-width',width:'fill_container',fill:'#525252'});
Update(open,{placeholder:false});
Print({open});
Export(['m3p3A','qVvEA','HQDiT','JJxfR',open],'png','C:/dev/mgts/hackathon-delivery/design/landing-r4/previews',{scale:1});
