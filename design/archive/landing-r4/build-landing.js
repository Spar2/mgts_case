const C={ink:'#161616',muted:'#525252',line:'#e0e0e0',blue:'#0f62fe',wash:'#edf5ff',layer:'#f4f4f4',white:'#ffffff'};
function frame(p,name,props={}) {return Insert(p,{type:'frame',name,layout:'vertical',width:'fill_container',height:'fit_content',...props});}
function txt(p,name,content,size=16,props={}) {return Insert(p,{type:'text',name,content,fontFamily:'IBM Plex Sans',fontSize:size,lineHeight:1.4,fill:C.ink,textGrowth:'fixed-width',width:'fill_container',...props});}
function button(p,label,primary=true,props={}) {const b=frame(p,label,{layout:'horizontal',width:'fit_content',padding:[16,22],fill:primary?C.blue:C.white,stroke:primary?C.blue:C.ink,strokeWidth:1,...props});txt(b,label+' · подпись',label,16,{textGrowth:'auto',fill:primary?C.white:C.ink,href:label.indexOf('Войти')>=0?'https://xeon-staging.tail5a7687.ts.net/login':undefined});return b;}
function tag(p,label,research=false) {const t=frame(p,label,{width:'fit_content',padding:[4,10],fill:research?'#fff1e6':C.wash});txt(t,label+' · статус',label,13,{textGrowth:'auto',fill:research?'#8a3800':'#0043ce'});return t;}
function pair(p,name,value,label) {const f=frame(p,name,{gap:4});txt(f,name+' · значение',value,28,{fontWeight:'500',lineHeight:1.15});txt(f,name+' · пояснение',label,14,{fill:C.muted});return f;}
function brands(p,mobile=false) {
 const b=frame(p,'МГТС и LARP CORP',{layout:'horizontal',gap:24,alignItems:'center'});
 const mg=frame(b,'МГТС · исходный знак',{width:124,height:36,layout:'none',clip:true,fill:'#008bd2'});
 Insert(mg,{type:'rectangle',name:'МГТС · оригинальное изображение',x:-7,y:-104,width:144,height:144,fill:{type:'image',url:'assets/mgts-original.jpg',mode:'stretch'}});
 const la=frame(b,'LARP CORP · исходный знак',{width:104,height:48,layout:'none',clip:true});
 Insert(la,{type:'rectangle',name:'LARP CORP · оригинальное изображение',x:-20,y:-51,width:144,height:144,fill:{type:'image',url:'assets/larp-corp-original.jpg',mode:'stretch'}});
 if(!mobile)txt(b,'Подпись знаков','Задача МГТС. Решение команды LARP CORP.',14,{fill:C.muted});
 return b;
}
function heatmap(p,w,h,select=true) {
 const f=frame(p,'Температурное поле · 51 × 86 исходных ячеек',{layout:'none',width:w,height:h,clip:true,fill:'#e0e0e0',metadata:{type:'scientific-plot',source:D.source,source_sha256:D.source_sha256,package_sha256:D.package_sha256,shape:D.shape,aggregation:'none',palette:'cividis-32',color_domain:[D.lo,D.hi],time_axis:'physical time; cell boundaries are midpoints'}});
 for(const [i,part] of D.paths.entries()) Insert(f,{type:'path',name:'Температура · диапазон '+i,width:w,height:h,x:0,y:0,viewBox:[0,0,800,200],geometry:part.geometry,fill:part.color});
 if(select) {
  Insert(f,{type:'rectangle',name:'Выбранный момент · 17:35:15',x:0,y:h/2,width:w,height:1,fill:'#ffffff'});
  Insert(f,{type:'rectangle',name:'Выбранное расстояние · 840 м',x:w*42.5/86,y:0,width:1,height:h,fill:'#ffffff'});
 }
 return f;
}
function field(p,mobile=false) {
 const w=mobile?302:780;
 const row=frame(p,'Поле с временной осью',{layout:'horizontal',gap:10});
 const axis=frame(row,'Подписи времени',{width:48,height:mobile?170:212,justifyContent:'space_between'});
 for(const t of ['17:31','17:35','17:39'])txt(axis,t,t,12,{fill:C.muted});
 const area=frame(row,'Числовое поле',{width:'fill_container',gap:10});
 heatmap(area,w,mobile?170:212);
 const ticks=frame(area,'Пространственная ось',{layout:'horizontal',justifyContent:'space_between'});
 for(const t of ['798','840','883 м'])txt(ticks,t,t,12,{textGrowth:'auto',fill:C.muted});
 const legend=frame(p,'Температурная шкала',{layout:'horizontal',gap:12,alignItems:'center'});
 txt(legend,'Минимум','16,4 °C',12,{textGrowth:'auto',fill:C.muted});
 const bar=frame(legend,'Cividis · общая шкала',{layout:'horizontal',width:mobile?160:220,height:8});
 for(const part of D.paths)Insert(bar,{type:'rectangle',name:part.color+' · шкала',width:'fill_container',height:8,fill:part.color});
 txt(legend,'Максимум','31,6 °C',12,{textGrowth:'auto',fill:C.muted});
}
function profile(p,width=600,height=160,processed=true) {
 const f=frame(p,'Пространственный профиль · точные значения',{layout:'none',width,height,fill:C.white,metadata:{type:'scientific-plot',source_sha256:D.source_sha256,row:1444,x_m:D.x_m,raw:D.raw_profile,processed:processed?D.processed_profile:undefined,method:processed?'median_x3':null,provenance:D.provenance}});
 const line=v=>v.map((y,i)=>(i?'L':'M')+(i/85*width).toFixed(3)+','+((32-y)/16*height).toFixed(3)).join(' ');
 for(const y of [20,24,28])Insert(f,{type:'path',name:y+' °C · опорная линия',width,height,viewBox:[0,0,width,height],geometry:'M0,'+((32-y)/16*height)+'H'+width,stroke:C.line,strokeWidth:1});
 Insert(f,{type:'path',name:'Исходный профиль · 17:35:15',width,height,viewBox:[0,0,width,height],geometry:line(D.raw_profile),stroke:processed?'#8d8d8d':C.blue,strokeWidth:1.8});
 if(processed)Insert(f,{type:'path',name:'median_x3 · опубликованный референс',width,height,viewBox:[0,0,width,height],geometry:line(D.processed_profile),stroke:C.blue,strokeWidth:2.4});
 return f;
}
function graphLegend(p,processed=true) {
 const f=frame(p,'Легенда профиля',{layout:'horizontal',gap:20});
 txt(f,'Исходный сигнал','━  Исходный сигнал',13,{textGrowth:'auto',fill:processed?'#6f6f6f':C.blue});
 if(processed)txt(f,'Референсная обработка','━  median_x3',13,{textGrowth:'auto',fill:C.blue});
}
function proof(p,mobile=false) {
 const panel=frame(p,'Реальный сигнал · демонстрационный фрагмент',{padding:mobile?20:28,gap:20,stroke:C.line,strokeWidth:1,fill:C.white});
 const top=frame(panel,'Паспорт демонстрации',{layout:mobile?'vertical':'horizontal',gap:8,justifyContent:'space_between'});
 txt(top,'Источник демонстрации','5.csv / опубликованное окно',16,{fontWeight:'500',width:mobile?'fill_container':'fit_content',textGrowth:mobile?'fixed-width':'auto'});
 txt(top,'Дата','16.10.2025 · 798–883 м',14,{fill:C.muted,width:mobile?'fill_container':'fit_content',textGrowth:mobile?'fixed-width':'auto'});
 const content=frame(panel,'Поле и точка',{layout:mobile?'vertical':'horizontal',gap:32});
 const plot=frame(content,'Температурное поле',{width:mobile?'fill_container':838,gap:12});
 field(plot,mobile);
 const inspector=frame(content,'Точная точка',{width:mobile?'fill_container':300,gap:18});
 tag(inspector,'Исходные измерения');
 txt(inspector,'Температура','29,0009 °C',36,{fontWeight:'500',lineHeight:1.1});
 txt(inspector,'Координаты точки','840 м · 17:35:15',17);
 txt(inspector,'Что можно исследовать','Один участок в трёх представлениях: поле, профиль и изменение температуры во времени.',16,{fill:C.muted,lineHeight:1.5});
 txt(panel,'Ограничение публичного примера','Иллюстрация по реальным данным. Рабочий архив доступен участникам после входа.',13,{fill:C.muted});
 return panel;
}
const featureInfo={
 preprocess:{tab:'Предобработка',title:'Сначала понять,\nчто измерено.',description:'Координаты, время и маски качества сохраняют связь графика с исходной записью.',detail:'Пропуск остаётся пропуском. Невалидные значения не превращаются в температуру.',label:'Исходные данные',mode:'raw'},
 clean:{tab:'Очистка от шумов',title:'Увидеть, что\nизменила обработка.',description:'Сравните исходный профиль с конкретным результатом. Общая шкала и точные значения помогают проверить каждое изменение.',detail:'В примере показан референсный median_x3. Изменение сигнала само по себе не доказывает качество очистки.',label:'Доступно в рабочем контуре',mode:'compare'},
 anomaly:{tab:'Аномалии',title:'От отклонения\nк проверке экспертом.',description:'Исследуйте score и кандидатов на исходных координатах, когда эти слои переданы вместе с расчётом.',detail:'Кандидат алгоритма остаётся гипотезой. Независимый ответ эксперта сохраняется отдельно.',label:'Результат зависит от переданных слоёв',mode:'workflow'},
 vision:{tab:'Компьютерное зрение',title:'Исследовать поле\nкак изображение.',description:'В лаборатории сравниваются подходы к пространственным и временным структурам температурного поля.',detail:'Исследовательское направление. Подтверждённая точность обнаружения аварий не заявлена.',label:'Исследование',mode:'research'},
 transformer:{tab:'Трансформеры',title:'Сравнивать идеи\nпо единому протоколу.',description:'Версии, параметры, источники обучения и ограничения помогают сопоставлять фактические эксперименты.',detail:'Исследовательское направление. Новый подход не выбирается победителем автоматически.',label:'Исследование',mode:'research'}
};
function feature(p,key,mobile=false) {
 const info=featureInfo[key];
 const section=frame(p,'Возможности · '+info.tab,{padding:mobile?[48,24]:[72,72],gap:32,fill:C.layer});
 txt(section,'Заголовок возможностей','От измерения к объяснению.',mobile?32:44,{fontWeight:'500',lineHeight:1.15});
 txt(section,'Вводный текст','Выберите этап и посмотрите, что с ним можно сделать.',17,{fill:C.muted});
 const tabs=frame(section,'Выбор возможностей',{layout:mobile?'vertical':'horizontal',gap:mobile?0:8,stroke:C.line,strokeWidth:{bottom:1}});
 for(const [id,v] of Object.entries(featureInfo)) {
  const tab=frame(tabs,v.tab,{layout:'horizontal',width:mobile?'fill_container':'fill_container',padding:mobile?[13,12]:[16,12],fill:id===key?C.white:C.layer,stroke:id===key?C.blue:C.line,strokeWidth:id===key?{bottom:3}:{bottom:1}});
  txt(tab,v.tab+' · вкладка',v.tab,14,{fontWeight:id===key?'600':'400',fill:id===key?C.blue:C.muted});
 }
 const body=frame(section,'Описание и подтверждение',{layout:mobile?'vertical':'horizontal',gap:mobile?28:72});
 const text=frame(body,'Смысл возможности',{width:mobile?'fill_container':390,gap:20});
 tag(text,info.label,info.mode==='research');
 txt(text,'Название возможности',info.title,mobile?30:36,{fontWeight:'500',lineHeight:1.15});
 txt(text,'Описание возможности',info.description,17,{fill:C.muted,lineHeight:1.5});
 txt(text,'Граница утверждения',info.detail,14,{fill:C.muted,lineHeight:1.5});
 const v=frame(body,'Пример · '+info.tab,{width:'fill_container',padding:mobile?16:24,gap:18,fill:C.white,stroke:C.line,strokeWidth:1});
 if(info.mode==='compare') {
  txt(v,'Название графика','Один профиль. Один выбранный результат.',18,{fontWeight:'500'});
  txt(v,'Паспорт профиля','5.csv · 17:35:15 · температура, °C',13,{fill:C.muted});
  graphLegend(v);
  profile(v,mobile?278:716,mobile?150:200);
  const ticks=frame(v,'Ось расстояния',{layout:'horizontal',justifyContent:'space_between'});
  for(const l of ['798','840','883 м'])txt(ticks,l,l,12,{textGrowth:'auto',fill:C.muted});
  const readings=frame(v,'Сопоставление в точке',{layout:mobile?'vertical':'horizontal',gap:20,padding:[16,0,0,0],stroke:C.line,strokeWidth:{top:1}});
  pair(readings,'Исходник','29,0009 °C','840 м / исходное значение');
  pair(readings,'Обработка','27,5301 °C','Та же точка / median_x3');
  txt(v,'Примечание обработки','Референсная обработка опубликованного окна; не финальная модель команды.',12,{fill:C.muted});
 } else if(info.mode==='raw') {
  txt(v,'Паспорт исходника','Координаты и качество',18,{fontWeight:'500'});
  const facts=[['Пространственная ось','798–883 м · 86 точек'],['Временная ось','51 измерение · 17:31–17:39'],['Маска качества','4 386 валидных значений в окне'],['Часовой пояс','Не указан в источнике']];
  for(const [label,value] of facts){const r=frame(v,label,{gap:4,padding:[8,0],stroke:C.line,strokeWidth:{bottom:1}});txt(r,label,label,13,{fill:C.muted});txt(r,value,value,17);}
  txt(v,'Примечание SHA','SHA исходника хранится вместе с результатом и кейсом.',14,{fill:C.muted});
 } else if(info.mode==='workflow') {
  txt(v,'Путь кандидата','Алгоритм предлагает. Эксперт решает.',18,{fontWeight:'500'});
  for(const [n,h,s] of [['01','Кандидат','Окно и результат расчёта'],['02','Независимая проверка','Метод скрыт до самостоятельного ответа'],['03','Зафиксированный кейс','Ответ, автор, ревизия и происхождение']]) {
   const r=frame(v,h,{layout:'horizontal',gap:16,padding:[16,0],stroke:C.line,strokeWidth:{bottom:1}});
   txt(r,n,n,24,{textGrowth:'auto',fill:C.blue,fontFamily:'IBM Plex Mono'});
   const t=frame(r,h+' · текст',{gap:4});txt(t,h,h,18,{fontWeight:'500'});txt(t,s,s,14,{fill:C.muted});
  }
  txt(v,'Отсутствующий score','Score появится только при наличии совместимого числового слоя.',14,{fill:C.muted});
 } else {
  txt(v,'Научный статус','Лаборатория подходов',18,{fontWeight:'500'});
  tag(v,'Исследовательский статус',true);
  txt(v,'Предмет исследования',key==='vision'?'Пространственные структуры':'Временные зависимости',mobile?25:30,{fontWeight:'500'});
  txt(v,'Метаданные эксперимента','Данные → версия → параметры → результат → независимая проверка',17,{fill:C.muted});
  const f=frame(v,'Реальное поле для исследования',{gap:10});heatmap(f,mobile?278:716,160,false);
  txt(v,'Подпись реального поля','Реальное поле 5.csv. Изображение не является результатом нейросети.',12,{fill:C.muted});
  txt(v,'Качество не оценено','Качество обнаружения без экспертного эталона: не оценено.',15,{fontWeight:'500'});
 }
 return section;
}
function caseSection(p,mobile=false) {
 const s=frame(p,'Проверяемый кейс',{padding:mobile?[48,24]:[72,72],gap:40});
 const head=frame(s,'Объяснение кейса',{layout:mobile?'vertical':'horizontal',gap:32,alignItems:'start'});
 txt(head,'Заголовок кейса','Один кейс.\nВсё можно проверить.',mobile?32:48,{fontWeight:'500',lineHeight:1.12,width:mobile?'fill_container':610});
 txt(head,'Польза кейса','Передайте специалисту весь разбор: точные значения, выбранный расчёт, зафиксированный ответ и происхождение. Кейс сохраняет снимок и не меняется после новых вычислений.',18,{fill:C.muted,lineHeight:1.5});
 const body=frame(s,'Разбор и передача',{layout:mobile?'vertical':'horizontal',gap:48});
 const paper=frame(body,'Автономный отчёт · пример состава',{width:mobile?'fill_container':740,padding:mobile?20:32,gap:20,fill:C.layer,stroke:C.line,strokeWidth:1});
 txt(paper,'Отчёт','Разбор температурного участка',24,{fontWeight:'500'});
 txt(paper,'Паспорт','5.csv / 798–883 м / 16.10.2025',14,{fill:C.muted});
 graphLegend(paper);
 profile(paper,mobile?302:676,160);
 const meta=frame(paper,'Проверяемые связи',{gap:12,padding:[18,0,0,0],stroke:C.line,strokeWidth:{top:1}});
 for(const t of ['Исходник · SHA-256','Расчёт · выбранный run_id','Ответ · автор и ревизия','Пакет · HTML + JSON + manifest']) txt(meta,t,t,14,{fill:C.muted});
 txt(paper,'Статус иллюстрации','Иллюстрация состава пакета. Экспертный ответ не подставлен.',12,{fill:C.muted});
 const details=frame(body,'От окна к пакету',{width:'fill_container',gap:24});
 for(const [num,title,desc] of [['01','Выберите окно','Сохраните источник, профиль и область исследования.'],['02','Подключите результат','Сравните именно выбранный расчёт, с его версией и ограничениями.'],['03','Передайте на проверку','Получите независимый ответ специалиста.'],['04','Сохраните и передайте','Скачайте автономный пакет для числовой сверки.']]) {
  const r=frame(details,title,{layout:'horizontal',gap:20});txt(r,num,num,20,{textGrowth:'auto',fontFamily:'IBM Plex Mono',fill:C.blue});
  const t=frame(r,title+' · описание',{gap:6});txt(t,title,title,20,{fontWeight:'500'});txt(t,desc,desc,15,{fill:C.muted,lineHeight:1.5});
 }
 return s;
}
function team(p,mobile=false) {
 const s=frame(p,'Команда LARP CORP',{padding:mobile?[48,24]:[64,72],gap:32,fill:C.layer});
 txt(s,'Команда','За платформой стоит команда.',mobile?32:44,{fontWeight:'500',lineHeight:1.15});
 txt(s,'Описание команды','LARP CORP объединяет работу с данными, продуктом и инфраструктурой.',17,{fill:C.muted});
 const row=frame(s,'Участники',{layout:mobile?'vertical':'horizontal',gap:mobile?24:40});
 for(const [name,role,desc] of [['Тихон','Фронтенд и продукт','Исследование сигнала, интерфейс и передача кейса.'],['Захар','Машинное обучение','Методы обработки, эксперименты и анализ результатов.'],['Григорий','Инфраструктура','Сервисы, база данных и командный стенд.']]) {
  const f=frame(row,name,{gap:10,padding:[20,0,0,0],stroke:C.line,strokeWidth:{top:1}});txt(f,name,name,28,{fontWeight:'500'});txt(f,role,role,16,{fill:C.blue});txt(f,desc,desc,16,{fill:C.muted,lineHeight:1.5});
 }
 return s;
}
function closing(p,mobile=false) {
 const s=frame(p,'Вход участников',{padding:mobile?[48,24]:[56,72],gap:24,stroke:C.line,strokeWidth:{bottom:1}});
 const row=frame(s,'Призыв к действию',{layout:mobile?'vertical':'horizontal',gap:24,alignItems:mobile?'start':'center',justifyContent:'space_between'});
 const t=frame(row,'Приглашение к работе',{width:mobile?'fill_container':850,gap:12});txt(t,'Призыв','Перейдём к вашим данным.',mobile?32:44,{fontWeight:'500',lineHeight:1.15});txt(t,'Условие доступа','Участники входят под своим аккаунтом. Для подключения обратитесь к администратору команды за приглашением.',17,{fill:C.muted});
 button(row,'Войти в платформу  ↗',true,{width:mobile?'fill_container':'fit_content'});
 const foot=frame(p,'Подвал',{layout:mobile?'vertical':'horizontal',gap:16,padding:mobile?[24,24]:[24,72],justifyContent:'space_between'});
 txt(foot,'Название и команда','DTS Workbench / LARP CORP',14,{textGrowth:'auto',fill:C.muted});txt(foot,'Дисклеймер','Хакатон · распределённые волоконные сенсоры',14,{fill:C.muted,textGrowth:mobile?'fixed-width':'auto'});
}
const desktop='m3p3A';
Update(desktop,{placeholder:true,height:'fit_content'});
Get(desktop,(n,c)=>{if(c.depth>0 && n.name.indexOf('Оригинал')<0)Update(n.id,{x:undefined,y:undefined});});
Update('QZS9a',{height:'fit_content',fill:C.white,padding:[56,72,32,72]});
Update('JvaNg',{layout:'horizontal'});Update('T2Q2g',{layout:'horizontal'});Update('tY8fi',{layout:'horizontal'});Update('qCAio',{layout:'horizontal'});Update('P3jf5Q',{layout:'horizontal'});Update('l4sAye',{layout:'horizontal'});
Update('Bv7BM',{fill:{type:'image',url:'assets/mgts-original.jpg',mode:'stretch'}});Update('j5Smx',{fill:{type:'image',url:'assets/larp-corp-original.jpg',mode:'stretch'}});
Update('JPY7m',{href:'https://xeon-staging.tail5a7687.ts.net/login'});
const proofWrap=frame(desktop,'Сигнал на первом экране',{padding:[0,72,40,72]});proof(proofWrap);
const facts=frame(desktop,'Данные и проверяемость',{layout:'horizontal',gap:48,padding:[28,72,40,72],stroke:C.line,strokeWidth:{top:1}});
pair(facts,'Реальные записи','6 записей','Архив измерений для исследования');
pair(facts,'Точное окно','51 × 86','Ячейки в показанном фрагменте');
pair(facts,'Происхождение','SHA-256','Связь источника, расчёта и кейса');
pair(facts,'Автономная передача','HTML + JSON','Пакет открывается без сервера');
const cleanSection=feature(desktop,'clean');caseSection(desktop);team(desktop);closing(desktop);
Update(desktop,{placeholder:false});
const mobilePos=FindEmptySpace({width:390,height:5500,nodeId:desktop,direction:'right',padding:100});
const mobile=Insert(document,{type:'frame',name:'Landing R4 · Mobile 390',x:mobilePos.x,y:mobilePos.y,width:390,height:'fit_content',layout:'vertical',clip:true,fill:C.white,placeholder:true});
const nav=frame(mobile,'Мобильная навигация',{layout:'horizontal',height:64,padding:[0,24],alignItems:'center',justifyContent:'space_between',stroke:C.line,strokeWidth:{bottom:1}});txt(nav,'Продукт','DTS Workbench',18,{textGrowth:'auto',fontWeight:'600'});button(nav,'Войти',true,{padding:[10,14]});
const hero=frame(mobile,'Первый экран · мобильный',{padding:[32,24,28,24],gap:22});
txt(hero,'Главный заголовок','Единая аналитическая платформа сенсорных данных',36,{fontWeight:'500',lineHeight:1.12});
txt(hero,'Описание','Исследуйте измерения, сравнивайте обработку и передавайте эксперту воспроизводимый разбор.',17,{fill:C.muted,lineHeight:1.45});
txt(hero,'Кейс хакатона','Кейс МГТС: «Анализ данных распределённых волоконных сенсоров»',13,{fill:C.muted});brands(hero,true);button(hero,'Посмотреть возможности  ↓',true,{width:'fill_container'});
const preview=frame(mobile,'Реальный сигнал · мобильный',{padding:[0,24,32,24]});
const mobilePanel=frame(preview,'Опубликованное окно',{gap:16,padding:16,stroke:C.line,strokeWidth:1});
txt(mobilePanel,'Источник','5.csv / 798–883 м',16,{fontWeight:'500'});txt(mobilePanel,'Дата записи','16.10.2025 · 17:31–17:39',13,{fill:C.muted});
const fieldArea=frame(mobilePanel,'Поле · мобильное',{gap:10});heatmap(fieldArea,308,190);const ticks=frame(fieldArea,'Расстояние',{layout:'horizontal',justifyContent:'space_between'});for(const t of ['798','840','883 м'])txt(ticks,t,t,12,{textGrowth:'auto',fill:C.muted});
txt(mobilePanel,'Диапазон температуры','16,4–31,6 °C · исходное поле',13,{fill:C.muted});
const readout=frame(mobilePanel,'Точная точка',{padding:[14,0,0,0],gap:6,stroke:C.line,strokeWidth:{top:1}});txt(readout,'Температура','29,0009 °C',28,{fontWeight:'500'});txt(readout,'Точка','840 м · 17:35:15',14,{fill:C.muted});
txt(mobilePanel,'Публичное ограничение','Иллюстрация реального окна. Рабочий архив открывается после входа.',13,{fill:C.muted});
const mf=frame(mobile,'Мобильные факты',{padding:[24,24],gap:20,stroke:C.line,strokeWidth:{top:1}});pair(mf,'Реальные записи','6 реальных записей','Точные значения и исходные координаты');pair(mf,'Передача кейса','Автономный пакет','HTML, JSON и проверяемое происхождение');
feature(mobile,'clean',true);caseSection(mobile,true);team(mobile,true);closing(mobile,true);Update(mobile,{placeholder:false});
const variants=[];
for(const key of ['preprocess','anomaly','vision','transformer']) {
 const pos=FindEmptySpace({width:1440,height:840,nodeId:variants.length?variants[variants.length-1]:mobile,direction:'right',padding:100});
 const root=Insert(document,{type:'frame',name:'Landing R4 · Вкладка '+featureInfo[key].tab,x:pos.x,y:pos.y,width:1440,height:'fit_content',layout:'vertical',clip:true,fill:C.layer,placeholder:true});
 feature(root,key);Update(root,{placeholder:false});variants.push(root);
}
const boardPos=FindEmptySpace({width:1440,height:1000,nodeId:variants[variants.length-1],direction:'right',padding:100});
const board=Insert(document,{type:'frame',name:'Landing R4 · Спецификация взаимодействий',x:boardPos.x,y:boardPos.y,width:1440,height:'fit_content',layout:'vertical',clip:true,fill:C.white,placeholder:true,padding:56,gap:32});
txt(board,'Название спецификации','Как лендинг работает в браузере',40,{fontWeight:'500'});
for(const [title,desc] of [['Навигация','Якоря ведут к возможностям, сценарию и команде. Вход открывает /login. Лендинг не открывает рабочие данные.'],['Вкладки','Выбор мышью или клавиатурой. Tab фокусирует группу, стрелки меняют вкладку; aria-selected и связь с панелью обязательны.'],['Точный сигнал','Публичный фрагмент не запрашивает рабочие API. Наведение и касание показывают одну реальную точку, стрелки перемещают выбор.'],['Отклик','Только opacity и transform, 160–200 мс. Reduced motion отключает переходы. Без автопрокрутки, бесконечных частиц и счётчиков точности.'],['Телефон','Вертикальная компоновка, поле с точным инспектором. Вкладки в мобильной реализации выбираются раскрываемым списком, раскрытое состояние показано в макете.'],['Загрузка и отказ','Сначала статичный график. Интерактивный код грузится по необходимости. Ошибка не скрывает уже показанный сигнал и не создаёт вымышленный результат.'],['Научные границы','median_x3 — референс. CV и трансформеры — исследовательские направления. Метрики точности аварий не заявлены.'],['Команда','Имена и роли взяты из диалога; состав, фамилии и контакты нужно подтвердить до публикации.']]) {
 const r=frame(board,title,{layout:'horizontal',gap:40,padding:[16,0],stroke:C.line,strokeWidth:{top:1}});txt(r,title,title,21,{fontWeight:'500',width:300});txt(r,title+' · правило',desc,17,{fill:C.muted});
}
Update(board,{placeholder:false});
Print({desktop,mobile,cleanSection,variants,board});
for(const root of [desktop,mobile,...variants,board])Get(root,(n,c)=>c.problems && Print('BOUNDARY',n.id,n.name,c.parentCtx?.node.name,c.problems));
TakeScreenshot([cleanSection,hero]);
Export([desktop,mobile,...variants,board],'png','C:/dev/mgts/hackathon-delivery/design/landing-r4/previews',{scale:1});
