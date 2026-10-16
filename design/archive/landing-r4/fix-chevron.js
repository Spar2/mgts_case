for(const root of ['qVvEA','D8oTd','T0xeyJ']) {
 Update(root,{placeholder:true});
 const tabs=Get(root,n=>n.name==='Выбор возможностей'?n.id:undefined)[0];
 const active=Get(tabs,{depth:1}).children.find(n=>n.enabled!==false).id;
 Update(active,{alignItems:'center',gap:12});
 Get(active,n=>n.type==='text' && Update(n.id,{content:'Очистка от шумов'}));
 Insert(active,{type:'icon',name:'Раскрыть список возможностей',library:'lucide',icon:'chevron-down',width:20,height:20,fill:'#0f62fe'});
 Update(root,{placeholder:false});
 Get(root,(n,c)=>{if(n.enabled===false){c.skipChildren();return;}if(c.problems && !(n.type==='rectangle' && n.fill?.type==='image'))Print('BOUNDARY',n.id,n.name,c.problems);});
}
Export(['qVvEA','D8oTd','T0xeyJ'],'png','C:/dev/mgts/hackathon-delivery/design/landing-r4/png',{scale:2});
Export(['B0IuXV'],'png','C:/dev/mgts/hackathon-delivery/design/landing-r4/mobile-review',{scale:1});
