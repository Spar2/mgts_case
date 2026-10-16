import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import {
  Button,
  InlineNotification,
  Select,
  SelectItem,
  Tag,
} from "@carbon/react";
import { control, useAccount } from "./access";
import { type AnalysisRun } from "./analysis";
import { analysisParams, commonAnalysisRegion } from "./analysis-context";
import Experiments from "./experiments";

type Metric = {
  coverage: number;
  mean_abs_change_c: number | null;
  bias_vs_raw_c: number | null;
  roughness_proxy_c: number | null;
};
type Window = {
  window: {
    id: string;
    record_id: string;
    source_sha256: string;
    row_start: number;
    row_end: number;
    column_start: number;
    column_end: number;
    split: number;
  };
  denoising: Record<string, Metric>;
  detection: Record<
    string,
    {
      threshold: number;
      score_coverage: number;
      marked_observation_fraction: number | null;
    }
  >;
};
type Study = {
  source_commit: string;
  manifest_sha256: string;
  data: {
    report: {
      windows: Window[];
      limits: string[];
      calibration: {
        target_macro_marked_observation_fraction: number;
        thresholds: Record<string, number>;
      };
    };
    "temporal-report": {
      device: string;
      selected_weights_sha256: string;
      internal_holdout: Record<
        string,
        {
          mse_noisy_target_c2: number;
          excluded_target_neighbour_mean_mse_c2: number;
        }
      >;
      limits: string[];
    };
    "classification-report": {
      status: string;
      models_trained: number;
      requirements: string;
    };
  };
};
const fmt = (value: number | null | undefined) =>
  value == null ? "Не оценено" : value.toFixed(5);
function RealStudy() {
  const account = useAccount();
  const query = useQuery({
    queryKey: ["real-study", account?.id],
    queryFn: ({ signal }) =>
      control<Study>("/research-studies/real-next", undefined, signal),
  });
  const [track, setTrack] = useState("denoising"),
    [windowId, setWindowId] = useState("5-split-2");
  const pack = query.data,
    w =
      pack?.data.report.windows.find((w) => w.window.id === windowId) ??
      pack?.data.report.windows[0];
  return (
    <section className="laboratory-study">
      <h2>Real-only исследование</h2>
      <Tag type="gray">Ранее известные записи · исследование</Tag>
      <p>
        Чистого эталона нет. Внутренние проверочные окна уже наблюдались; это не
        независимая оценка аварийной точности.
      </p>
      {query.error && (
        <InlineNotification
          kind="error"
          title="Исследование недоступно"
          subtitle={String(query.error)}
          hideCloseButton
        />
      )}
      {pack && w && (
        <>
          <div className="lab-controls">
            <Select
              id="lab-track"
              labelText="Направление"
              value={track}
              onChange={(e) => setTrack(e.target.value)}
            >
              <SelectItem value="denoising" text="Очистка" />
              <SelectItem value="detection" text="Обнаружение" />
              <SelectItem value="classification" text="Классификация" />
            </Select>
            <Select
              id="lab-window"
              labelText="Фактическое окно"
              value={windowId}
              onChange={(e) => setWindowId(e.target.value)}
            >
              {pack.data.report.windows.map((row) => (
                <SelectItem
                  key={row.window.id}
                  value={row.window.id}
                  text={`${row.window.record_id}.csv / ${["train", "development", "внутренняя проверка"][row.window.split]}`}
                />
              ))}
            </Select>
          </div>
          <p>
            SHA исходника: <code>{w.window.source_sha256}</code> · исходные
            строки [{w.window.row_start}, {w.window.row_end}), столбцы [
            {w.window.column_start}, {w.window.column_end})
          </p>
          {track === "denoising" && (
            <>
              <h3>Что изменили фильтры</h3>
              <p>
                Изменение и шероховатость — не качество очистки. Доступность CNN
                может отличаться; сравнение не объявляет победителя.
              </p>
              <div className="table-scroll">
                <table>
                  <thead>
                    <tr>
                      <th>Метод</th>
                      <th>Доступность</th>
                      <th>Среднее |изменение|, °C</th>
                      <th>Смещение, °C</th>
                      <th>Шероховатость, °C</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(w.denoising).map(([method, m]) => (
                      <tr key={method}>
                        <th>{method}</th>
                        <td>{(100 * m.coverage).toFixed(1)}%</td>
                        <td>{fmt(m.mean_abs_change_c)}</td>
                        <td>{fmt(m.bias_vs_raw_c)}</td>
                        <td>{fmt(m.roughness_proxy_c)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <h3>Временная реконструкция на GPU</h3>
              <p>
                {pack.data["temporal-report"].device} · двусторонний контекст,
                офлайн. Ошибка к шумному измерению не подтверждает очистку
                чистого сигнала.
              </p>
              <div className="table-scroll">
                <table>
                  <thead>
                    <tr>
                      <th>Запись</th>
                      <th>CNN MSE, °C²</th>
                      <th>Среднее соседей MSE, °C²</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(
                      pack.data["temporal-report"].internal_holdout,
                    )
                      .sort(([a], [b]) => a.localeCompare(b))
                      .map(([id, m]) => (
                        <tr key={id}>
                          <th>{id}.csv</th>
                          <td>{fmt(m.mse_noisy_target_c2)}</td>
                          <td>
                            {fmt(m.excluded_target_neighbour_mean_mse_c2)}
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
              <p>
                Вывод текущего цикла: общего преимущества CNN не установлено;
                рабочая пара не переключена.
              </p>
            </>
          )}
          {track === "detection" && (
            <>
              <h3>Доля отметок и доступность score</h3>
              <p>
                Нет экспертного эталона — нет FAR, precision или recall аварий.
                Одинаковая доля отметок — способ сравнения чувствительности, не
                доказательство качества.
              </p>
              <div className="table-scroll">
                <table>
                  <thead>
                    <tr>
                      <th>Метод</th>
                      <th>Порог</th>
                      <th>Доступность</th>
                      <th>Отмеченная доля</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(w.detection).map(([method, m]) => (
                      <tr key={method}>
                        <th>{method}</th>
                        <td>{fmt(m.threshold)}</td>
                        <td>{(m.score_coverage * 100).toFixed(1)}%</td>
                        <td>
                          {m.marked_observation_fraction == null
                            ? "Нет оценки"
                            : (m.marked_observation_fraction * 100).toFixed(3) +
                              "%"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p>
                Отдельная калибровка по development:{" "}
                {Object.entries(pack.data.report.calibration.thresholds)
                  .map(([m, v]) => m + " = " + fmt(v))
                  .join("; ")}
                .
              </p>
            </>
          )}
          {track === "classification" && (
            <>
              <h3>Ожидает подтверждённых ответов</h3>
              <p>
                Обученных классификаторов:{" "}
                {pack.data["classification-report"].models_trained}. Точность:
                не оценена.
              </p>
              <p>{pack.data["classification-report"].requirements}</p>
              <Link to="/review?quick=1">Открыть независимую проверку →</Link>
              <p>
                Автоматические гипотезы, пропуски карточек и технические
                QA-ответы не заменяют экспертные метки.
              </p>
            </>
          )}
          <details>
            <summary>Происхождение и ограничения</summary>
            <p>Коммит исследования: {pack.source_commit}</p>
            <p>
              Manifest SHA: <code>{pack.manifest_sha256}</code>
            </p>
            <p>
              Weights SHA:{" "}
              <code>
                {pack.data["temporal-report"].selected_weights_sha256}
              </code>
            </p>
            <ul>
              {pack.data.report.limits.map((limit) => (
                <li key={limit}>{limit}</li>
              ))}
            </ul>
          </details>
          <Button
            kind="tertiary"
            onClick={() => {
              const url = URL.createObjectURL(
                new Blob([JSON.stringify(pack, null, 2)], {
                  type: "application/json",
                }),
              );
              const link = document.createElement("a");
              link.href = url;
              link.download = "dts-real-study.json";
              link.click();
              URL.revokeObjectURL(url);
            }}
          >
            Скачать исследовательскую сводку
          </Button>
        </>
      )}
    </section>
  );
}

export default function Laboratory() {
  // Historical view owns its existing main landmark; keep one main per page.
  const account = useAccount(),
    [tab, setTab] = useState("results"),
    [first, setFirst] = useState(""),
    [second, setSecond] = useState("");
  const catalog = useQuery({
    queryKey: ["lab-results", account?.id],
    queryFn: ({ signal }) =>
      control<AnalysisRun[]>("/result-runs", undefined, signal),
  });
  const a = catalog.data?.find((r) => r.id === first),
    b = catalog.data?.find((r) => r.id === second),
    common = a && b ? commonAnalysisRegion(a, b) : null;
  const comparison =
    a && b && common
      ? "/analysis?" +
        analysisParams(
          a,
          {
            region: common,
            row: common.row_start,
            column: common.column_start,
            layer: "raw",
            zone: "",
          },
          new URLSearchParams({ compare: b.id }),
        )
      : "";
  const Container=tab==='historical'?'div':'main';
  return (
    <Container className="page laboratory-page">
      <h1>Лаборатория</h1>
      <p>
        Сохранённые числовые результаты и воспроизводимые исследования. У
        каждого запуска свой источник, версия и ограничения.
      </p>
      <nav className="work-tabs" aria-label="Разделы лаборатории">
        {[
          ["results", "Расчёты"],
          ["real", "Реальные исследования"],
          ["historical", "Исторический benchmark"],
        ].map(([key, label]) => (
          <button
            key={key}
            onClick={() => setTab(key)}
            aria-current={tab === key ? "page" : undefined}
          >
            {label}
          </button>
        ))}
      </nav>
      {tab === "real" && <RealStudy />}
      {tab === "historical" && (
        <>
          <Tag type="warm-gray">
            Синтетическая калибровка и реальные диагностические окна
          </Tag>
          <Experiments />
        </>
      )}
      {tab === "results" && (
        <>
          <div className="lab-controls">
            {[
              [first, setFirst, "Первый результат"],
              [second, setSecond, "Второй результат"],
            ].map(([value, setter, label], i) => (
              <Select
                key={i}
                id={"lab-run-" + i}
                labelText={label as string}
                value={value as string}
                onChange={(e) =>
                  (setter as (v: string) => void)(e.target.value)
                }
              >
                <SelectItem value="" text="Выберите запуск" />
                {catalog.data?.map((r) => (
                  <SelectItem
                    key={r.id}
                    value={r.id}
                    text={`${r.region.record_id}.csv / ${r.method} / ${r.id.slice(0, 8)}`}
                  />
                ))}
              </Select>
            ))}
          </div>
          {first && second && !common && (
            <InlineNotification
              kind="warning"
              title="Нет совместимого общего окна"
              subtitle="Нужны одинаковые источник/SHA и пересечение исходных границ. Метрики не объединяются."
              hideCloseButton
            />
          )}
          {comparison && (
            <p>
              <Link className="text-link" to={comparison}>
                Сравнить в общем окне →
              </Link>
            </p>
          )}
          {catalog.error && (
            <InlineNotification
              kind="error"
              title="Результаты не загружены"
              subtitle={String(catalog.error)}
              hideCloseButton
            />
          )}
          {catalog.data?.length === 0 && (
            <p>
              Пока нет импортированных расчётов. Архив доступен; отсутствующий
              результат не заменяется демонстрационным.
            </p>
          )}
          <div className="lab-run-list">
            {catalog.data?.map((r) => (
              <article key={r.id}>
                <h2>{r.method}</h2>
                <p>
                  {r.region.record_id}.csv · версия {r.method_version} ·{" "}
                  {r.research_status} · обучение: {r.training_origin}
                </p>
                <p>Слои: {r.layers.join(", ") || "Нет числовых слоёв"}</p>
                <p>{r.limitations}</p>
                <p>
                  <Link to={"/analysis?run_id=" + r.id}>
                    Открыть точные данные →
                  </Link>
                </p>
                <p>
                  <Link
                    to={
                      "/works?" +
                      new URLSearchParams({
                        record: r.region.record_id,
                        source_sha256: r.source_sha256,
                        row: String(r.region.row_start),
                        column: String(r.region.column_start),
                        r0: String(r.region.row_start),
                        r1: String(r.region.row_end),
                        c0: String(r.region.column_start),
                        c1: String(r.region.column_end),
                        run_id: r.id,
                      })
                    }
                  >
                    Начать работу с результатом →
                  </Link>
                </p>
              </article>
            ))}
          </div>
        </>
      )}
    </Container>
  );
}
