import { canvasClearRgb } from "./chart-clear";
import { chartColors, useAppTheme } from "./theme";
import { useEffect, useRef, useState } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import { binary } from "./cache";
import type { decode, TileMeta } from "./numeric";
import { colors } from "./palette";

type Tile = ReturnType<typeof decode>;
const gpuAllocations = new Map<
  HTMLCanvasElement,
  { buffers: number; textures: number }
>();
Object.defineProperty(globalThis, "__dtsGraphicsStats", {
  configurable: true,
  value: () => ({
    webglContexts: gpuAllocations.size,
    bufferBytes: [...gpuAllocations.values()].reduce(
      (sum, v) => sum + v.buffers,
      0,
    ),
    textureBytes: [...gpuAllocations.values()].reduce(
      (sum, v) => sum + v.textures,
      0,
    ),
    canvasSurfaceBytes: [...document.querySelectorAll("canvas")].reduce(
      (sum, c) => sum + c.width * c.height * 4,
      0,
    ),
    caveat:
      "Requested buffers/textures and one RGBA surface per canvas; excludes driver, compositor and double buffering.",
  }),
});
const vs = `#version 300 es
in vec2 quad; in vec4 cell; in vec2 measurement; flat out float value; flat out float invalid;
void main(){vec2 p=vec2(mix(cell.x,cell.y,quad.x),mix(cell.z,cell.w,quad.y));gl_Position=vec4(p.x*2.-1.,1.-p.y*2.,0.,1.);value=measurement.x;invalid=measurement.y;}`;
const fs = `#version 300 es
precision highp float; flat in float value; flat in float invalid; uniform vec2 scale; uniform sampler2D colors; out vec4 color;
void main(){if(invalid>.5){color=vec4(.82,.82,.82,1.);}else{float t=clamp((value-scale.x)/(scale.y-scale.x),0.,1.);color=texture(colors,vec2(t,.5));}}`;
function gpu(canvas: HTMLCanvasElement, data: Tile) {
  const palette = colors(data.meta.palette);
  const gl = canvas.getContext("webgl2", {
    antialias: false,
    alpha: false,
    preserveDrawingBuffer: false,
  });
  if (!gl) throw new Error("WebGL2 недоступен");
  function shader(type: number, source: string) {
    const s = gl!.createShader(type)!;
    gl!.shaderSource(s, source);
    gl!.compileShader(s);
    if (!gl!.getShaderParameter(s, gl!.COMPILE_STATUS))
      throw new Error(gl!.getShaderInfoLog(s)!);
    return s;
  }
  const vertex = shader(gl.VERTEX_SHADER, vs),
    fragment = shader(gl.FRAGMENT_SHADER, fs),
    program = gl.createProgram()!;
  gl.attachShader(program, vertex);
  gl.attachShader(program, fragment);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS))
    throw new Error(gl.getProgramInfoLog(program)!);
  gl.useProgram(program);
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  const corners = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, corners);
  gl.bufferData(
    gl.ARRAY_BUFFER,
    new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]),
    gl.STATIC_DRAW,
  );
  const q = gl.getAttribLocation(program, "quad");
  gl.enableVertexAttribArray(q);
  gl.vertexAttribPointer(q, 2, gl.FLOAT, false, 0, 0);
  const cells = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, cells);
  gl.bufferData(gl.ARRAY_BUFFER, data.instances, gl.STATIC_DRAW);
  const c = gl.getAttribLocation(program, "cell"),
    m = gl.getAttribLocation(program, "measurement");
  gl.enableVertexAttribArray(c);
  gl.vertexAttribPointer(c, 4, gl.FLOAT, false, 24, 0);
  gl.vertexAttribDivisor(c, 1);
  gl.enableVertexAttribArray(m);
  gl.vertexAttribPointer(m, 2, gl.FLOAT, false, 24, 16);
  gl.vertexAttribDivisor(m, 1);
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA,
    256,
    1,
    0,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    new Uint8Array(palette),
  );
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  const scaleLocation = gl.getUniformLocation(program, "scale");
  const colorsLocation = gl.getUniformLocation(program, "colors");
  gl.uniform1i(colorsLocation, 0);
  gl.uniform2f(scaleLocation, data.meta.scale[0], data.meta.scale[1]);
  gpuAllocations.set(canvas, {
    buffers: data.instances.byteLength + 32,
    textures: palette.length,
  });
  function draw() {
    // A canvas size change drops the bound program, array, and texture.
    gl!.useProgram(program);
    gl!.bindVertexArray(vao);
    gl!.activeTexture(gl!.TEXTURE0);
    gl!.bindTexture(gl!.TEXTURE_2D, texture);
    gl!.uniform1i(colorsLocation, 0);
    gl!.uniform2f(scaleLocation, data.meta.scale[0], data.meta.scale[1]);
    gl!.viewport(0, 0, canvas.width, canvas.height);
    const rgb = canvasClearRgb(
      getComputedStyle(document.body).backgroundColor,
      document.documentElement.dataset.theme === "dark",
    );
    gl!.clearColor(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, 1);
    gl!.clear(gl!.COLOR_BUFFER_BIT);
    gl!.drawArraysInstanced(
      gl!.TRIANGLE_STRIP,
      0,
      4,
      data.instances.length / 6,
    );
  }
  return {
    draw,
    dispose() {
      gpuAllocations.delete(canvas);
      gl.deleteTexture(texture);
      gl.deleteBuffer(corners);
      gl.deleteBuffer(cells);
      gl.deleteVertexArray(vao);
      gl.deleteProgram(program);
      gl.deleteShader(vertex);
      gl.deleteShader(fragment);
    },
  };
}
function fallback(canvas: HTMLCanvasElement, data: Tile) {
  const palette = colors(data.meta.palette);
  const ctx = canvas.getContext("2d")!;
  return {
    draw() {
      ctx.fillStyle = chartColors().background;
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      for (let i = 0; i < data.instances.length; i += 6) {
        const [x0, x1, y0, y1, value, invalid] = data.instances.subarray(
          i,
          i + 6,
        );
        const p =
          Math.round(
            Math.min(
              1,
              Math.max(
                0,
                (value - data.meta.scale[0]) /
                  (data.meta.scale[1] - data.meta.scale[0]),
              ),
            ) * 255,
          ) * 4;
        ctx.fillStyle = invalid
          ? "#d0d0d0"
          : `rgb(${palette[p]},${palette[p + 1]},${palette[p + 2]})`;
        ctx.fillRect(
          x0 * canvas.width,
          y0 * canvas.height,
          Math.max(1, (x1 - x0) * canvas.width),
          Math.max(1, (y1 - y0) * canvas.height),
        );
      }
    },
    dispose() {},
  };
}
export function Heatmap({
  path,
  onSelect,
  cursor,
  versionKey,
  expectedSourceSha,
  zones = [],
  selectedZoneId,
}: {
  path: string;
  onSelect?: (x: number, t: number) => void;
  cursor?: [number, number];
  versionKey?: string;
  expectedSourceSha?: string;
  zones?: {
    id: string;
    segment: number;
    x0: number;
    x1: number;
    t0: number;
    t1: number;
    label: string;
  }[];
  selectedZoneId?: string;
}) {
  const { theme } = useAppTheme();
  const redraw = useRef<(() => void) | undefined>(undefined);
  useEffect(() => { redraw.current?.(); }, [theme]);
  const host = useRef<HTMLDivElement>(null),
    tile = useRef<Tile | null>(null);
  const [meta, setMeta] = useState<TileMeta>(),
    [extent, setExtent] = useState<number[]>(),
    [validCount, setValidCount] = useState(0),
    [status, setStatus] = useState("Загрузка поля…");
  useEffect(() => {
    const controller = new AbortController(),
      worker = new Worker(new URL("./data.worker.ts", import.meta.url), {
        type: "module",
      });
    let renderer: ReturnType<typeof gpu> | null = null,
      canvas: HTMLCanvasElement | null = null,
      observer: ResizeObserver | null = null,
      frame = 0,
      disposed = false;
    setStatus("Загрузка поля…");
    setMeta(undefined);
    tile.current = null;
    const render = (data: Tile, forceFallback = false) => {
      const container = host.current;
      if (disposed || !container) return;
      renderer?.dispose();
      observer?.disconnect();
      canvas?.remove();
      canvas = document.createElement("canvas");
      canvas.setAttribute("aria-hidden", "true");
      container.prepend(canvas);
      try {
        renderer = forceFallback ? fallback(canvas, data) : gpu(canvas, data);
        setStatus(forceFallback ? "Canvas2D — резервный режим" : "WebGL2");
      } catch {
        canvas.remove();
        canvas = document.createElement("canvas");
        canvas.setAttribute("aria-hidden", "true");
        container.prepend(canvas);
        renderer = fallback(canvas, data);
        setStatus("Canvas2D — резервный режим");
      }
      redraw.current = () => renderer?.draw();
      canvas.addEventListener(
        "webglcontextlost",
        (event) => {
          event.preventDefault();
          if (!disposed) render(data, true);
        },
        { once: true },
      );
      const paint = () => {
        if (disposed || !canvas || !host.current) return;
        const box = host.current.getBoundingClientRect();
        if (box.width < 2 || box.height < 2) return;
        const ratio = Math.min(devicePixelRatio, box.width < 672 ? 1.5 : 2);
        canvas.width = Math.max(1, Math.round(box.width * ratio));
        canvas.height = Math.max(1, Math.round(box.height * ratio));
        renderer?.draw();
      };
      paint();
      observer = new ResizeObserver(() => {
        if (disposed) return;
        cancelAnimationFrame(frame);
        frame = requestAnimationFrame(paint);
      });
      observer.observe(container);
    };
    worker.onmessage = (event: MessageEvent<Tile & { error?: string }>) => {
      if (disposed) return;
      if (event.data.error) {
        setStatus(event.data.error);
        return;
      }
      if (
        expectedSourceSha &&
        event.data.meta.source_sha256 !== expectedSourceSha
      ) {
        setStatus(
          "Версия источника изменилась. Обновите запись перед исследованием.",
        );
        return;
      }
      tile.current = event.data;
      setMeta(event.data.meta);
      setExtent(event.data.extent);
      setValidCount(event.data.counts.reduce((sum, n) => sum + n, 0));
      render(event.data);
    };
    worker.onerror = () => {
      if (!disposed)
        setStatus(
          "Не удалось обработать блок данных. Обновите страницу или выберите другую область.",
        );
    };
    binary(path, controller.signal, versionKey)
      .then((buffer) => {
        if (!disposed) worker.postMessage({ buffer }, [buffer]);
      })
      .catch((e) => {
        if (!controller.signal.aborted) setStatus(String(e));
      });
    return () => {
      disposed = true;
      redraw.current = undefined;
      controller.abort();
      worker.terminate();
      observer?.disconnect();
      cancelAnimationFrame(frame);
      renderer?.dispose();
      canvas?.remove();
      tile.current = null;
    };
  }, [path, versionKey, expectedSourceSha]);
  return (
    <figure className="heat-figure" data-source-sha256={meta?.source_sha256}>
      <div className="chart-heading">
        <span>{meta?.title ?? "Температурное поле"}</span>
        <span className="mono">
          {meta
            ? `${meta.scale[0].toFixed(2)}–${meta.scale[1].toFixed(2)} ${meta.unit ?? "°C"}`
            : status}
        </span>
      </div>
      <div className="heat-axis">
        <div className="time-axis">
          <span>Время от начала, с</span>
          {extent &&
            [0, 0.5, 1].map((v) => (
              <small key={v} style={{ top: `${v * 100}%` }}>
                {(extent[2] + v * (extent[3] - extent[2])).toFixed(0)}
              </small>
            ))}
        </div>
        <div
          ref={host}
          className="heat-canvas"
          onClick={(e) => {
            const d = tile.current;
            if (!d || !onSelect) return;
            const b = e.currentTarget.getBoundingClientRect();
            onSelect(
              d.extent[0] +
                ((e.clientX - b.left) / b.width) * (d.extent[1] - d.extent[0]),
              d.extent[2] +
                ((e.clientY - b.top) / b.height) * (d.extent[3] - d.extent[2]),
            );
          }}
        >
          {!meta && (
            <div className="heat-status" role="status">
              {status}
            </div>
          )}
          {meta && extent && cursor && (
            <>
              <div
                className="cursor-v"
                style={{
                  left: `${Math.max(0, Math.min(100, ((cursor[0] - extent[0]) / (extent[1] - extent[0])) * 100))}%`,
                }}
              />
              <div
                className="cursor-h"
                style={{
                  top: `${Math.max(0, Math.min(100, ((cursor[1] - extent[2]) / (extent[3] - extent[2])) * 100))}%`,
                }}
              />
            </>
          )}
          {meta && extent && zones.length > 0 && (
            <svg
              className="heat-zones"
              viewBox="0 0 100 100"
              preserveAspectRatio="none"
              aria-hidden="true"
            >
              {zones.slice(0, 200).map((zone) => {
                const x0 = Math.max(zone.x0, extent[0]),
                  x1 = Math.min(zone.x1, extent[1]),
                  t0 = Math.max(zone.t0, extent[2]),
                  t1 = Math.min(zone.t1, extent[3]);
                if (x1 <= x0 || t1 <= t0) return null;
                return (
                  <rect
                    key={zone.id + ":" + zone.segment}
                    x={((x0 - extent[0]) / (extent[1] - extent[0])) * 100}
                    y={((t0 - extent[2]) / (extent[3] - extent[2])) * 100}
                    width={((x1 - x0) / (extent[1] - extent[0])) * 100}
                    height={((t1 - t0) / (extent[3] - extent[2])) * 100}
                    fill="none"
                    stroke={zone.id === selectedZoneId ? "var(--dts-chart-series)" : "var(--cds-text-primary)"}
                    strokeWidth={zone.id === selectedZoneId ? 2 : 1}
                    vectorEffect="non-scaling-stroke"
                    strokeDasharray="4 3"
                  >
                    <title>
                      {zone.id}: {zone.label} — прямоугольник гипотезы
                    </title>
                  </rect>
                );
              })}
            </svg>
          )}
        </div>
        <div
          className="colorbar"
          style={{
            background: `linear-gradient(to top,${[0, 64, 128, 192, 255].map((i) => `rgb(${colors(meta?.palette)[i * 4]},${colors(meta?.palette)[i * 4 + 1]},${colors(meta?.palette)[i * 4 + 2]})`).join(",")})`,
          }}
        />
      </div>
      <div className="axis-labels">
        <span>{extent?.[0].toFixed(1)} м</span>
        <span>Расстояние по волокну, м</span>
        <span>{extent?.[1].toFixed(1)} м</span>
      </div>
      <figcaption>
        {meta?.aggregation === "native_method_hypothesis"
          ? "Гипотезы на исходной сетке · синий: охлаждение, красный: нагрев, нейтральный: нет кандидата"
          : meta?.aggregation === "native_observations"
            ? "Исходные ячейки · без интерполяции"
            : meta?.aggregation === "signed_max_absolute_valid"
              ? "Знаковый экстремум по модулю в физическом бине; это не медиана и не вероятность"
              : "Медиана валидных наблюдений по физическим бинам"}{" "}
        · серый: нет валидных данных · {meta ? status : ""}
        {meta &&
          ` · валидных наблюдений в блоке: ${validCount.toLocaleString("ru")}`}
      </figcaption>
    </figure>
  );
}
export function LineChart({
  title,
  x,
  y,
  xLabel,
  yLimits,
  syncKey,
  yLabel = "°C",
  threshold,
  thresholdMode = "ge",
}: {
  title: string;
  x: number[];
  y: (number | null)[];
  xLabel: string;
  yLimits?: [number, number];
  syncKey?: string;
  yLabel?: string;
  threshold?: number;
  thresholdMode?: "ge" | "abs_ge";
}) {
  const { theme } = useAppTheme();
  const ref = useRef<HTMLDivElement>(null);
  const plotRef = useRef<uPlot | undefined>(undefined);
  useEffect(() => {
    const host = ref.current!;
    const observer = new ResizeObserver(() =>
      plotRef.current?.setSize({ width: host.clientWidth, height: 180 }),
    );
    observer.observe(host);
    return () => {
      observer.disconnect();
      plotRef.current?.destroy();
      plotRef.current = undefined;
    };
  }, []);
  useEffect(() => {
    plotRef.current?.destroy();
    plotRef.current = undefined;
  }, [syncKey, xLabel, yLabel, threshold, thresholdMode, theme]);
  useEffect(() => {
    const host = ref.current!,
      worker = new Worker(new URL("./data.worker.ts", import.meta.url), {
        type: "module",
      });
    worker.onmessage = (
      e: MessageEvent<{ line: [number[], (number | null)[]] }>,
    ) => {
      const data: uPlot.AlignedData =
        threshold === undefined
          ? e.data.line
          : [
              ...e.data.line,
              e.data.line[0].map(() => threshold),
              ...(thresholdMode === "abs_ge"
                ? [e.data.line[0].map(() => -threshold)]
                : []),
            ];
      if (plotRef.current) {
        plotRef.current.setData(data);
        if (yLimits)
          plotRef.current.setScale("y", { min: yLimits[0], max: yLimits[1] });
        return;
      }
      const paint = chartColors();
      plotRef.current = new uPlot(
        {
          width: host.clientWidth,
          height: 180,
          legend: { show: false },
          cursor: {
            drag: { x: false, y: false },
            ...(syncKey ? { sync: { key: syncKey, scales: ["x", null] } } : {}),
          },
          scales: { x: { time: false } },
          series: [
            {},
            { label: yLabel, stroke: paint.series, width: 1, spanGaps: false },
            ...(threshold === undefined
              ? []
              : [
                  { label: "Порог", stroke: paint.threshold, width: 1, dash: [5, 5] },
                ]),
            ...(threshold !== undefined && thresholdMode === "abs_ge"
              ? [
                  {
                    label: "Отрицательный порог",
                    stroke: paint.threshold,
                    width: 1,
                    dash: [5, 5],
                  },
                ]
              : []),
          ],
          axes: [
            { label: xLabel, size: 40, stroke: paint.axis, ticks: { stroke: paint.grid }, grid: { show: false } },
            { label: yLabel, size: 55, stroke: paint.axis, ticks: { stroke: paint.grid }, grid: { stroke: paint.grid } },
          ],
        },
        data,
        host,
      );
      if (yLimits)
        plotRef.current.setScale("y", { min: yLimits[0], max: yLimits[1] });
    };
    worker.postMessage({ x, y, budget: Math.max(200, host.clientWidth * 2) });
    return () => {
      worker.terminate();
    };
  }, [
    x,
    y,
    xLabel,
    yLabel,
    threshold,
    thresholdMode,
    yLimits?.[0],
    yLimits?.[1],
    syncKey,
    theme,
  ]);
  return (
    <figure className="line-figure">
      <h3>{title}</h3>
      <div
        ref={ref}
        role="img"
        aria-label={`${title}. ${x.length} исходных наблюдений; точные значения в инспекторе.`}
      />
    </figure>
  );
}
