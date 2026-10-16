import { chartColors, useAppTheme } from "./theme";
import { useEffect, useRef } from "react";
import uPlot from "uplot";

/** Only the exact selected profile, never a field matrix. No interpolation across nulls. */
export function ProfileOverlay({
  x,
  raw,
  clean,
  limits,
  syncKey,
}: {
  x: number[];
  raw: (number | null)[];
  clean: (number | null)[];
  limits: [number, number];
  syncKey: string;
}) {
  const { theme } = useAppTheme();
  const host = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const container = host.current!;
    const paint = chartColors();
    const plot = new uPlot(
      {
        width: Math.max(200, container.clientWidth),
        height: 240,
        legend: { show: false },
        cursor: {
          drag: { x: false, y: false },
          sync: { key: syncKey, scales: ["x", null] },
        },
        scales: { x: { time: false }, y: { range: () => limits } },
        series: [
          {},
          { label: "Исходник", stroke: paint.raw, width: 1, spanGaps: false },
          { label: "Обработка", stroke: paint.series, width: 2, spanGaps: false },
        ],
        axes: [
          { label: "Расстояние, м", size: 42, stroke: paint.axis, ticks: { stroke: paint.grid }, grid: { show: false } },
          { label: "°C", size: 52, stroke: paint.axis, ticks: { stroke: paint.grid }, grid: { stroke: paint.grid } },
        ],
      },
      [x, raw, clean],
      container,
    );
    const observer = new ResizeObserver(() =>
      plot.setSize({
        width: Math.max(200, container.clientWidth),
        height: 240,
      }),
    );
    observer.observe(container);
    return () => {
      observer.disconnect();
      plot.destroy();
    };
  }, [x, raw, clean, limits, syncKey, theme]);
  return (
    <figure className="demo-overlay">
      <h3>Исходный профиль и обработка</h3>
      <div className="demo-legend">
        <span>Исходник</span>
        <span>Обработка</span>
      </div>
      <div
        ref={host}
        role="img"
        aria-label={`Исходный профиль и обработка: ${x.length} координат на общих осях. Точные значения доступны в таблице.`}
      />
      <figcaption>
        Общие оси, исходные координаты. Пропуски не соединяются линией.
      </figcaption>
    </figure>
  );
}
