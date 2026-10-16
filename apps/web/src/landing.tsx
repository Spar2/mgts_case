import { ThemeSwitch } from "./theme";
import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import Autoencoder from "./autoencoder";
import PipelineWalk from "./pipeline-walk";
import { usePreviewIntro } from "./preview-intro";
import "./landing.scss";

type Preview = {
  source_sha256: string;
  lo: number; hi: number; row_timestamp: string;
  x_m: number[]; raw_profile: number[]; processed_profile: number[];
  time_s: number[]; timestamps: string[];
  raw_c: (number | null)[][]; processed_c: (number | null)[][];
  invalid_reason: number[][];
  paths: { color: string; geometry: string }[];
};
const features = [
  { name: "Предобработка", title: "Подготовка измерений", text: "Проверка координат, времени и невалидных значений перед анализом.", status: "Данные", detail: "Исходные значения и маски сохраняются." },
  { name: "Очистка от шумов", title: "Исходник и обработка", text: "Сравнение температурных профилей на общей шкале.", status: "Обработка", detail: "В примере — референсный median_x3." },
  { name: "Аномалии", title: "Поиск участков для проверки", text: "Поле, точный профиль и время помогают исследовать температурные пики.", status: "Анализ", detail: "Выделенный участок требует оценки эксперта." },
  { name: "Компьютерное зрение", title: "Структуры на температурном поле", text: "Исследование пространственных и временных форм сигнала.", status: "Исследование", detail: "Показана схема входа модели, не результат обнаружения." },
  { name: "Трансформеры", title: "Зависимости во времени", text: "Исследование последовательностей измерений в выбранной точке.", status: "Исследование", detail: "Показан входной ряд. Прогноз модели не представлен." },
];

function Profile({ data, processed = true, row }: { data: Preview; processed?: boolean; row?: number }) {
  const path = (values: (number | null)[]) => {
    let connected = false;
    return values.map((v, i) => {
      if (v === null || !Number.isFinite(v)) { connected = false; return ""; }
      const command = connected ? "L" : "M";
      connected = true;
      return `${command}${48 + i / (values.length - 1) * 700},${190 - (v - 16) / 16 * 158}`;
    }).join(" ");
  };
  return <svg className="lp-profile" viewBox="0 0 780 230" role="img" aria-label="Пространственный профиль: исходная температура и референсная медианная обработка">
    {[20,24,28,32].map(v => <g key={v}><line x1="48" x2="748" y1={190-(v-16)/16*158} y2={190-(v-16)/16*158} stroke="var(--dts-chart-grid)"/><text x="8" y={194-(v-16)/16*158}>{v}°</text></g>)}
    <path d={path(row === undefined ? data.raw_profile : data.raw_c[row].map((v,i) => data.invalid_reason[row][i] ? null : v))} fill="none" stroke="var(--dts-chart-muted)" strokeWidth="1.7" />
    {processed && <path d={path(row === undefined ? data.processed_profile : data.processed_c[row])} fill="none" stroke="var(--dts-chart-series)" strokeWidth="2.3" />}
    {[798,840,883].map((v,i) => <text key={v} x={48+i*350} y="218" textAnchor={i===2?"end":"start"}>{v} м</text>)}
  </svg>;
}

function FeatureVisual({ data, tab, row, point }: { data: Preview; tab: number; row: number; point: number }) {
  if (tab === 0) return <div className="lp-data-preview">
    <div className="lp-chart-title"><span>Измерения · {data.timestamps[row].slice(11)}</span><span>5.csv</span></div>
    <table><caption>Фрагмент исходной строки</caption><thead><tr><th scope="col">Расстояние</th><th scope="col">Температура</th><th scope="col">Качество</th></tr></thead><tbody>{[Math.max(0,Math.min(81,point-2)),Math.max(0,Math.min(81,point-2))+1,Math.max(0,Math.min(81,point-2))+2,Math.max(0,Math.min(81,point-2))+3,Math.max(0,Math.min(81,point-2))+4].map(i=><tr key={i}><td>{data.x_m[i]} м</td><td>{data.invalid_reason[row][i] ? "—" : data.raw_c[row][i]?.toFixed(4)} °C</td><td>{data.invalid_reason[row][i] ? "Пропуск" : "Валидно"}</td></tr>)}</tbody></table>
    <p className="lp-chart-note">Координаты и маска качества из исходной записи.</p>
  </div>;
  if (tab === 1) return <><div className="lp-chart-title"><span>Профиль температуры</span><span>{data.timestamps[row].slice(11)} · °C</span></div><Profile data={data} row={row}/><div className="lp-legend"><span><i/>Исходный сигнал</span><span><i className="lp-blue"/>median_x3</span></div><p className="lp-chart-note">Изменение сигнала не является оценкой качества очистки.</p></>;
  if (tab === 4) {
    let connected = false;
    const path = data.raw_c.map((values,i)=>{
      const v = values[point];
      if(v===null || data.invalid_reason[i][point]) { connected=false; return ""; }
      const cmd = connected?"L":"M"; connected=true;
      return `${cmd}${48+(data.time_s[i]-data.time_s[0])/(data.time_s.at(-1)!-data.time_s[0])*700},${190-(v-16)/16*158}`;
    }).join(" ");
    return <><div className="lp-chart-title"><span>Временной ряд · {data.x_m[point]} м</span><span>Вход модели</span></div><svg className="lp-profile" viewBox="0 0 780 230" role="img" aria-label="Реальные температуры выбранной точки во времени, без прогноза модели">{[20,24,28,32].map(v=><g key={v}><line x1="48" x2="748" y1={190-(v-16)/16*158} y2={190-(v-16)/16*158} stroke="var(--dts-chart-grid)"/><text x="8" y={194-(v-16)/16*158}>{v}°</text></g>)}<path d={path} fill="none" stroke="var(--dts-chart-series)" strokeWidth="2"/><text x="48" y="224">17:31</text><text x="748" y="224" textAnchor="end">17:39</text></svg><div className="lp-sequence" aria-label="Схема обработки последовательности"><span>Измерения</span><span aria-hidden="true">→</span><span>Временные окна</span><span aria-hidden="true">→</span><span>Модель</span></div><p className="lp-chart-note">Схема исследовательского подхода. Результат модели здесь не показан.</p></>;
  }
  return <><div className="lp-chart-title"><span>{tab===2?"Температурные пики":"Разбиение поля на фрагменты"}</span><span>{tab===2?"Исходные данные":"Схема входа модели"}</span></div><svg className="lp-feature-field" viewBox="0 0 800 200" preserveAspectRatio="none" shapeRendering="crispEdges" role="img" aria-label={tab===2?"Исходное температурное поле без алгоритмической разметки":"Реальное поле с иллюстративной сеткой фрагментов для компьютерного зрения"}>{data.paths.map(p=><path key={p.color} d={p.geometry} fill={p.color}/>)}{tab===3&&<g stroke="white" strokeWidth="2" fill="none">{[200,400,600].map(x=><line key={x} x1={x} x2={x} y1="0" y2="200"/>)}{[67,133].map(y=><line key={y} x1="0" x2="800" y1={y} y2={y}/>)}</g>}</svg><div className="lp-axis"><span>798 м</span><span>883 м</span></div><p className="lp-chart-note">{tab===2?"16.10.2025 · 17:31–17:39. Яркость показывает температуру, а не вероятность аварии.":"Сетка иллюстрирует подготовку входа. Это не маски событий и не вывод модели."}</p></>;
}

export default function Landing() {
  const [data, setData] = useState<Preview | null>(null);
  const [error, setError] = useState(false);
  const [attempt, setAttempt] = useState(0);
  const [tab, setTab] = useState(0);
  const [point, setPoint] = useState(42);
  const [row, setRow] = useState(25);
  const inspectorRef = useRef<HTMLElement>(null);
  const distanceRef = useRef<HTMLInputElement>(null);
  const timeRef = useRef<HTMLInputElement>(null);
  usePreviewIntro({
    ready: !!data, inspector: inspectorRef, distance: distanceRef, time: timeRef,
    point, row, maxPoint: data ? data.x_m.length-1 : 85, maxRow: data ? data.time_s.length-1 : 50,
    select: (nextPoint, nextRow) => { setPoint(nextPoint); setRow(nextRow); },
  });
  useEffect(() => {
    const controller = new AbortController();
    fetch("/landing/visual-data.json", { signal: controller.signal }).then(r => {
      if (!r.ok) throw new Error("preview");
      return r.json();
    }).then(d => { setData(d); setError(false); }).catch(e => { if (e.name !== "AbortError") setError(true); });
    return () => controller.abort();
  }, [attempt]);
  const feature = features[tab];
  const timestamp = data?.timestamps[row];
  const value = data && !data.invalid_reason[row][point] ? data.raw_c[row][point] : null;
  const timeStart = data ? data.time_s[0] - (data.time_s[1]-data.time_s[0])/2 : 0;
  const timeEnd = data ? data.time_s.at(-1)! + (data.time_s.at(-1)!-data.time_s.at(-2)!)/2 : 1;
  const partners = <div className="lp-partners"><div className="lp-logos"><div className="lp-mgts"><img src="/landing/mgts-original.jpg" alt="МГТС" width="144" height="144"/></div><span className="lp-collab" aria-label="совместно с">×</span><div className="lp-larp"><img src="/landing/larp-corp-original.jpg" alt="LARP CORP" width="144" height="144"/></div></div><p>Создано для кейса МГТС<br/><strong>«Анализ данных распределённых волоконных сенсоров»</strong></p></div>;
  return <div className="lp">
    <a className="lp-skip" href="#landing-main">Перейти к содержимому</a>
    <header className="lp-header">
      <Link className="lp-wordmark" to="/" aria-label="Единая аналитическая платформа сенсорных данных, главная"><span className="lp-mark" aria-hidden="true">▥</span><span>Единая аналитическая платформа сенсорных данных</span></Link>
      <nav aria-label="Навигация лендинга"><a href="#capabilities">Возможности</a><a href="#workflow">Как это работает</a></nav>
      <Link className="lp-login" to="/login">Войти <span aria-hidden="true">↗</span></Link>
      <ThemeSwitch />
    </header>
    <main id="landing-main">
      <section className="lp-hero lp-wrap">
        <h1>Единая аналитическая платформа сенсорных данных</h1>
        <div className="lp-intro"><p>Температурные поля, обработка сигнала и экспертная проверка в одном приложении.</p></div>
        <div className="lp-signal">
          <div className="lp-signal-heading"><span><i aria-hidden="true"/> РЕАЛЬНЫЙ ФРАГМЕНТ ИЗМЕРЕНИЯ</span><span>5.csv / 798–883 м</span></div>
          <div className="lp-field-layout"><div className="lp-field">
            <div className="lp-field-time"><span>17:31:04</span><span>17:35:15</span><span>17:39:26</span></div>
            <div className="lp-field-plot">{data ? <svg viewBox="0 0 800 200" preserveAspectRatio="none" shapeRendering="crispEdges" role="img" aria-label="Температурное поле: выберите точку по расстоянию и времени. С клавиатуры используйте ползунки справа." onClick={e => {
              const box = e.currentTarget.getBoundingClientRect();
              const x = Math.max(0,Math.min(1,(e.clientX-box.left)/box.width));
              const y = Math.max(0,Math.min(1,(e.clientY-box.top)/box.height));
              setPoint(Math.min(85,Math.floor(x*86)));
              const time = timeStart + y*(timeEnd-timeStart);
              setRow(data.time_s.reduce((best,t,i) => Math.abs(t-time)<Math.abs(data.time_s[best]-time)?i:best,0));
            }}>
              {data.paths.map(p => <path key={p.color} d={p.geometry} fill={p.color}/>)}
              <line x1="0" x2="800" y1={(data.time_s[row]-timeStart)/(timeEnd-timeStart)*200} y2={(data.time_s[row]-timeStart)/(timeEnd-timeStart)*200} stroke="white" strokeDasharray="4 4" opacity=".8"/>
              <line x1={(point+.5)/86*800} x2={(point+.5)/86*800} y1="0" y2="200" stroke="white" strokeDasharray="4 4"/>
            </svg> : <div className="lp-placeholder" role="status">{error ? <>Фрагмент не загрузился. <button onClick={() => { setError(false); setAttempt(a=>a+1); }}>Повторить</button></> : "Загружаем реальный фрагмент…"}</div>}
            <div className="lp-axis"><span>798 м</span><span>840 м</span><span>883 м</span></div></div>
          </div><aside className="lp-inspector" ref={inspectorRef}><p className="lp-small">ТОЧНОЕ НАБЛЮДЕНИЕ</p><strong>{value === null ? "—" : value.toFixed(4)}<small> °C</small></strong><label htmlFor="preview-point">Расстояние · {798+point} м</label><input id="preview-point" ref={distanceRef} type="range" min="0" max="85" defaultValue="42" disabled={!data} onChange={e => setPoint(Math.round(Number(e.target.value)))}/><label htmlFor="preview-time">Время · {timestamp?.slice(11) ?? "—"}</label><input id="preview-time" ref={timeRef} type="range" min="0" max="50" defaultValue="25" disabled={!data} onChange={e => setRow(Math.round(Number(e.target.value)))}/><p>16.10.2025 · {timestamp?.slice(11) ?? "—"}</p><p className="lp-muted">{data && value === null ? "Нет валидного измерения." : "Исходное измерение."}<br/>Часовой пояс не указан.</p><div className="lp-scale"/><div className="lp-axis"><span>16,42 °C</span><span>31,64 °C</span></div></aside></div>
          <div className="lp-signal-footer"><span>Нажмите на поле: время ↓ · расстояние →</span><span>Реальное окно · без интерполяции</span></div>
        </div>
      </section>
      <div className="lp-facts lp-wrap">{[["6","исходных записей"],["51 × 86","ячеек в этом фрагменте"],["SHA-256","связь источника и результата"],["HTML + JSON","автономная передача кейса"]].map(([n,t])=><div key={n}><strong>{n}</strong><span>{t}</span></div>)}</div>
      <Autoencoder />
      <PipelineWalk />
      <section id="capabilities" className="lp-capabilities lp-wrap">
        <p className="lp-kicker">01 / ВОЗМОЖНОСТИ</p><h2>Инструменты анализа</h2>
        <div className="lp-tabs" role="tablist" aria-label="Возможности платформы">{features.map((f,i)=><button key={f.name} id={`feature-${i}`} role="tab" aria-selected={tab===i} aria-controls="feature-panel" tabIndex={tab===i?0:-1} onClick={()=>setTab(i)} onKeyDown={e=>{if(["ArrowRight","ArrowLeft","Home","End"].includes(e.key)){e.preventDefault();const next=e.key==="Home"?0:e.key==="End"?4:(i+(e.key==="ArrowRight"?1:4))%5;setTab(next);document.getElementById(`feature-${next}`)?.focus();}}}>{f.name}</button>)}</div>
        <label className="lp-mobile-tabs">Выберите возможность<select value={tab} onChange={e=>setTab(Number(e.target.value))}>{features.map((f,i)=><option key={f.name} value={i}>{f.name}</option>)}</select></label>
        <div id="feature-panel" role="tabpanel" aria-labelledby={`feature-${tab}`} className="lp-feature-panel">
          <div className="lp-feature-copy"><p className="lp-tag">{feature.status}</p><h3>{feature.title}</h3><p>{feature.text}</p><p className="lp-feature-detail">{feature.detail}</p></div>
          <div className="lp-chart-panel">{data ? <FeatureVisual data={data} tab={tab} row={row} point={point}/> : <p className="lp-placeholder">{error?"Данные временно недоступны":"Загружаем данные…"}</p>}</div>
        </div>
      </section>
      <section id="workflow" className="lp-workflow"><div className="lp-wrap"><p className="lp-kicker">02 / ПРОВЕРЯЕМЫЙ КЕЙС</p><h2>Отчёт с исходными данными</h2><p className="lp-section-intro">Сохраните окно, выбранный расчёт и ответ эксперта в автономный пакет.</p><div className="lp-workflow-grid"><div className="lp-report"><div className="lp-report-top">DTS / ПАСПОРТ ИССЛЕДОВАНИЯ<span>ПРИМЕР</span></div><h3>Температурное поле<br/>5.csv / 798–883 м</h3><p>16.10.2025 · 17:31:04–17:39:26</p>{data&&<Profile data={data}/>}<dl><div><dt>Обработка</dt><dd>median_x3 · референс</dd></div><div><dt>Экспертная проверка</dt><dd>Не включена в пример</dd></div><div><dt>SHA источника</dt><dd className="lp-hash">{data?.source_sha256??"Загружается…"}</dd></div></dl><p className="lp-chart-note">Иллюстрация состава кейса. Рабочие кейсы доступны после входа.</p></div><ol className="lp-steps">{[["Исходные измерения","Окно, координаты, время и маски качества сохраняются вместе с SHA источника."],["Явно выбранная обработка","Метод, версия и конкретный запуск. Новый расчёт не подменяет сохранённый результат."],["Независимый ответ эксперта","Автор и ревизия фиксируются в кейсе. Отсутствие заключения обозначается явно."],["Автономный пакет","HTML, JSON и manifest с SHA. Разбор можно открыть и численно проверить без сайта."]].map(([title,text],i)=><li key={title}><span>0{i+1}</span><div><h3>{title}</h3><p>{text}</p></div></li>)}</ol></div></div></section>
      <section className="lp-access lp-wrap"><p className="lp-kicker">КОМАНДНОЕ РАБОЧЕЕ МЕСТО</p><h2>Продолжите исследование<br/>в платформе.</h2><div className="lp-access-action"><p>Вход для участников команды. Приглашение можно получить у администратора.</p><Link className="lp-button" to="/login">Войти в платформу <span aria-hidden="true">↗</span></Link></div>{partners}</section>
    </main><footer className="lp-footer"><span>Единая аналитическая платформа сенсорных данных</span><span>LARP CORP</span><a href="#landing-main">Наверх ↑</a></footer>
  </div>;
}
