import { useEffect, useState } from "react";
import {
  CHANNELS,
  channelLevel,
  cellCombined,
  DETECTORS,
  pipelineFrame,
  REDUCED_CAPTION,
  revealCount,
  STAGE_COPY,
  TRACE_CELLS,
  traceLevel,
  zoneSpans,
} from "./pipeline-frame";

const CELLS = Array.from({ length: TRACE_CELLS }, (_, index) => index);
const FIELD = 768;
const STEP = FIELD / TRACE_CELLS;
const TRACE_Y = 8;
const TRACE_H = 40;
const CH_TOP = 60;
const CH_H = 16;
const CH_STEP = 24;
const DET_TOP = 144;
const DET_H = 10;
const DET_STEP = 16;
const MAP_Y = 216;
const MAP_H = 24;
const FIELD_H = 252;

function cellX(index: number) {
  return index * STEP;
}

function fillFor(level: 0 | 1 | 2) {
  if (level === 2) return "var(--walk-hot)";
  if (level === 1) return "var(--walk-mid)";
  return "var(--walk-low)";
}

function prefersReducedMotion() {
  return typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

export default function PipelineWalk() {
  const [reduced, setReduced] = useState(prefersReducedMotion);
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const media = window.matchMedia("(prefers-reduced-motion: reduce)");
    const onChange = () => setReduced(media.matches);
    media.addEventListener("change", onChange);
    return () => media.removeEventListener("change", onChange);
  }, []);

  useEffect(() => {
    if (reduced) return undefined;
    let frame = 0;
    let origin = 0;
    const tick = (now: number) => {
      if (origin === 0) origin = now;
      setElapsed(now - origin);
      frame = window.requestAnimationFrame(tick);
    };
    frame = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(frame);
  }, [reduced]);

  const scene = pipelineFrame(elapsed, reduced);
  const shownTrace = revealCount(scene, 0, TRACE_CELLS);
  const shownChannels = revealCount(scene, 1, CHANNELS.length);
  const shownDetectors = revealCount(scene, 2, DETECTORS.length);
  const shownMap = revealCount(scene, 3, TRACE_CELLS);
  const shownZones = revealCount(scene, 4, zoneSpans().length);
  const caption = scene.reducedMotion ? REDUCED_CAPTION : STAGE_COPY[scene.index].text;

  return (
    <section className="lp-walk lp-wrap" data-pipeline-walk data-stage={scene.stage} aria-labelledby="pipeline-walk-title">
      <p className="lp-kicker">КАК ЧИТАЕТСЯ СЛЕД</p>
      <h2 id="pipeline-walk-title">От сырого следа к зонам</h2>
      <p className="lp-section-intro">Один и тот же след смотрят несколько детекторов. Карты складываются, и связные участки остаются зонами для чтения.</p>
      <div className="lp-walk-board">
        <ol className="lp-walk-steps">
          {STAGE_COPY.map((step, index) => {
            const on = scene.reducedMotion || index <= scene.index;
            return (
              <li key={step.id} aria-current={!scene.reducedMotion && index === scene.index ? "step" : undefined} data-on={on ? "yes" : "no"}>
                <span>{step.kicker}</span>
                {step.name}
              </li>
            );
          })}
        </ol>
        <div className="lp-walk-picture">
          <div className="lp-walk-chart">
            <div className="lp-walk-labels">
              <span style={{ top: `${((TRACE_Y + TRACE_H / 2) / FIELD_H) * 100}%` }}>След</span>
              {CHANNELS.map((channel, row) => (
                <span key={channel.kind} style={{ top: `${((CH_TOP + row * CH_STEP + CH_H / 2) / FIELD_H) * 100}%` }}>{channel.label}</span>
              ))}
              {DETECTORS.map((detector, index) => (
                <span key={detector.id} data-lit={index < shownDetectors ? "yes" : "no"} style={{ top: `${((DET_TOP + index * DET_STEP + DET_H / 2) / FIELD_H) * 100}%` }}>{detector.id}</span>
              ))}
              <span style={{ top: `${((MAP_Y + MAP_H / 2) / FIELD_H) * 100}%` }}>Одна карта</span>
            </div>
            <svg className="lp-walk-svg" viewBox={`0 0 ${FIELD} ${FIELD_H}`} preserveAspectRatio="none" role="img" aria-label={caption}>
              {CELLS.map((index) => (
                <rect key={`trace-${index}`} x={cellX(index) + 0.6} y={TRACE_Y} width={STEP - 1.2} height={TRACE_H} fill={index < shownTrace ? fillFor(traceLevel(index)) : "var(--walk-low)"} stroke="var(--dts-subtle-border)" strokeWidth="1" />
              ))}
              {!scene.reducedMotion && scene.stage === "trace" && shownTrace < TRACE_CELLS && (
                <rect x={cellX(shownTrace)} y={TRACE_Y} width="3" height={TRACE_H} fill="var(--walk-hot)" />
              )}
              {CHANNELS.map((channel, row) => {
                const y = CH_TOP + row * CH_STEP;
                const lit = row < shownChannels;
                return (
                  <g key={channel.kind}>
                    {CELLS.map((index) => (
                      <rect key={`${channel.kind}-${index}`} x={cellX(index) + 0.6} y={y} width={STEP - 1.2} height={CH_H} fill={lit ? fillFor(channelLevel(channel.kind, index)) : "var(--walk-low)"} stroke="var(--dts-subtle-border)" strokeWidth="1" />
                    ))}
                  </g>
                );
              })}
              {DETECTORS.map((detector, index) => {
                const y = DET_TOP + index * DET_STEP;
                const lit = index < shownDetectors;
                return (
                  <g key={detector.id}>
                    <line x1="0" x2={FIELD} y1={y + DET_H} y2={y + DET_H} stroke="var(--dts-subtle-border)" strokeWidth="1" />
                    {lit && <rect x={cellX(detector.from)} y={y} width={(detector.to - detector.from) * STEP} height={DET_H} fill="var(--walk-hot)" />}
                  </g>
                );
              })}
              {CELLS.map((index) => (
                <rect key={`map-${index}`} x={cellX(index) + 0.6} y={MAP_Y} width={STEP - 1.2} height={MAP_H} fill={index < shownMap && cellCombined(index) ? "var(--walk-hot)" : "var(--walk-low)"} stroke="var(--dts-subtle-border)" strokeWidth="1" />
              ))}
              {zoneSpans().map((span, index) => index < shownZones && (
                <rect key={`zone-${span.from}`} x={cellX(span.from)} y={MAP_Y - 3} width={(span.to - span.from) * STEP} height={MAP_H + 6} fill="none" stroke="var(--cds-text-primary)" strokeWidth="2" />
              ))}
            </svg>
          </div>
          <p className="lp-walk-caption">{caption}</p>
        </div>
      </div>
      <p className="lp-chart-note">Схема показа по пластине команды. Чисел измерения и заключения по событию здесь нет.</p>
    </section>
  );
}
