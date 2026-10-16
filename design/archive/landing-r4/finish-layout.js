function fitGraphics(root) {
 const maps=Get(root,(n,c)=>n.metadata?.type==='scientific-plot' && n.metadata.shape?{id:n.id,width:c.bounds.width}:undefined);
 for(const m of maps)Get(m.id,(n,c)=>c.depth===1 && n.type==='path' && Update(n.id,{width:m.width}));
 const lines=Get(root,(n,c)=>n.name==='Пространственный профиль · точные значения'?{id:n.id,parent:c.parentCtx.node.id}:undefined);
 for(const line of lines){Update(line.id,{width:'fill_container'});const w=Get(line.id,(n,c)=>c.depth===0?c.bounds.width:undefined)[0];Get(line.id,(n,c)=>c.depth===1 && n.type==='path' && Update(n.id,{width:w}));}
}
for(const root of ['m3p3A','qVvEA']){Update(root,{placeholder:true});fitGraphics(root);Update(root,{placeholder:false});}
const pos=FindEmptySpace({width:320,height:5700,nodeId:'qVvEA',direction:'bottom',padding:100});
const small=Copy('qVvEA',document,{name:'Landing R4 · Mobile 320',width:320,x:pos.x,y:pos.y,placeholder:true});
Get(small,n=>{if(n.name==='Главный заголовок')Update(n.id,{fontSize:32});if(n.name==='Пространственный профиль · точные значения')Update(n.id,{width:'fill_container'});});
fitGraphics(small);Update(small,{placeholder:false});
const pos430=FindEmptySpace({width:430,height:5500,nodeId:small,direction:'right',padding:100});
const wide=Copy('qVvEA',document,{name:'Landing R4 · Mobile 430',width:430,x:pos430.x,y:pos430.y,placeholder:true});fitGraphics(wide);Update(wide,{placeholder:false});
const all=['m3p3A','qVvEA','T48mcW','ElvPP','hN1eX','gDbeh','C7Gl9','pp4J8',small,wide];
for(const root of all)Get(root,(n,c)=>{if(n.enabled===false){c.skipChildren();return;}if(c.problems && !(n.type==='rectangle' && n.fill?.type==='image'))Print('BOUNDARY',n.id,n.name,c.parentCtx?.node.name,c.problems);});
Print({small,wide});
Get('m3p3A',(n,c)=>c.depth===1 && Print('SECTION',n.id,n.name,c.bounds));
TakeScreenshot(['HQDiT']);
Export([...all,'QZS9a','HQDiT'],'png','C:/dev/mgts/hackathon-delivery/design/landing-r4/previews',{scale:1});
