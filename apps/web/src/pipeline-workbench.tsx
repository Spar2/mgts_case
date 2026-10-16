import { useRef, type JSX, type ReactNode, type UIEvent } from "react";
import { Button } from "@carbon/react";
import type { StageError, StageId, StageStatus } from "./pipeline-stages";
import "./pipeline.scss";

export type WorkbenchStage = {
  id: StageId;
  label: string;
  chip: string;
  status: StageStatus;
  source: string;
};

const STATUS_MARK: Record<StageStatus, { word: string; tone: string }> = {
  saved: { word: "сохранено", tone: "is-saved" },
  error: { word: "ошибка", tone: "is-error" },
  draft: { word: "черновик", tone: "is-draft" },
  empty: { word: "пусто", tone: "is-empty" },
};

export function PipelineWorkbench({
  stages,
  active,
  error,
  recordLabel,
  canSave,
  canCheck,
  canSegment,
  checking,
  onActive,
  onSource,
  onSave,
  onCheck,
  onSegment,
  segment,
  original,
  checked,
}: {
  stages: WorkbenchStage[];
  active: StageId;
  error: StageError | null;
  recordLabel: string;
  canSave: boolean;
  canCheck: boolean;
  canSegment: boolean;
  checking: boolean;
  onActive: (id: StageId) => void;
  onSource: (source: string) => void;
  onSave: () => void;
  onCheck: () => void;
  onSegment: () => void;
  segment: ReactNode;
  original: ReactNode;
  checked: ReactNode;
}): JSX.Element {
  const linesRef = useRef<HTMLDivElement>(null);
  const selected = stages.find((stage) => stage.id === active);
  const label = selected?.label ?? "";
  const source = selected?.source ?? "";
  const errorLine = error !== null && error.stageId === active ? error.line : undefined;
  const lineCount = Math.max(source.split("\n").length, 1);

  function onEditorScroll(event: UIEvent<HTMLTextAreaElement>) {
    const gutter = linesRef.current;
    if (gutter) gutter.scrollTop = event.currentTarget.scrollTop;
  }

  return (
    <section className="pipeline-workbench" aria-label="Рабочее место пайплайна">
      <div role="tablist" aria-label="Вкладки пайплайна" className="pipeline-workbench-tabs">
        {stages.map((stage) => (
          <button
            key={stage.id}
            type="button"
            role="tab"
            aria-selected={stage.id === active}
            className={stage.id === active ? "pipeline-workbench-tab is-active" : "pipeline-workbench-tab"}
            onClick={() => onActive(stage.id)}
          >
            {stage.label}
          </button>
        ))}
      </div>
      <div className="pipeline-workbench-strip">
        <div className="pipeline-workbench-chips">
          {stages.map((stage) => {
            const mark = STATUS_MARK[stage.status];
            return (
              <span key={stage.id} className={`pipeline-workbench-chip ${mark.tone}`}>
                {stage.chip} · {mark.word}
              </span>
            );
          })}
        </div>
        <p className="pipeline-workbench-hint">сохраняется только эта вкладка</p>
        <Button kind="primary" type="button" disabled={!canSave} onClick={onSave}>
          {`Сохранить вкладку «${label}»`}
        </Button>
      </div>
      <div className="pipeline-workbench-body">
        <div className="pipeline-workbench-code">
          <div className="pipeline-workbench-editor">
            <div className="pipeline-workbench-lines" aria-hidden="true" ref={linesRef}>
              {Array.from({ length: lineCount }, (_, index) => {
                const number = index + 1;
                return (
                  <div
                    key={number}
                    className={number === errorLine ? "pipeline-workbench-line is-error" : "pipeline-workbench-line"}
                  >
                    {number}
                  </div>
                );
              })}
            </div>
            <textarea
              className="pipeline-workbench-input"
              spellCheck={false}
              wrap="off"
              aria-label={`Код вкладки ${label}`}
              value={source}
              onChange={(event) => onSource(event.target.value)}
              onScroll={onEditorScroll}
            />
          </div>
          {error !== null && (
            <div role="alert" className="pipeline-workbench-alert">
              {`Вкладка «${error.label}», строка ${error.line}. ${/[.!?]$/.test(error.message) ? error.message : `${error.message}.`} Следующие вкладки интерпретатор не запускал.`}
            </div>
          )}
          {error !== null && (
            <p className="pipeline-workbench-note">Исправьте указанную строку. «Проверить пайплайн» отправит закреплённый расчёт только после этого.</p>
          )}
        </div>
        <div className="pipeline-workbench-preview">
          <p className="pipeline-workbench-caption">Оригинальный отрезок</p>
          <div className="pipeline-workbench-segment">{segment}</div>
          <div className="pipeline-workbench-actions">
            <Button kind="tertiary" type="button" disabled={!canSegment} onClick={onSegment}>
              Задать отрезок
            </Button>
            <Button kind="primary" type="button" disabled={!canCheck || checking} onClick={onCheck}>
              Проверить пайплайн
            </Button>
          </div>
          <div className="pipeline-workbench-map">{original}</div>
          <p className="pipeline-workbench-caption">После проверки</p>
          <div className="pipeline-workbench-map">{checked}</div>
          <p className="pipeline-workbench-note">изменение сигнала, не доказанная очистка</p>
          <p className="pipeline-workbench-note">{recordLabel}</p>
        </div>
      </div>
    </section>
  );
}
