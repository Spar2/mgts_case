/** Documentation recipe only. Not imported by the application; no API calls. */
import { useId, type FormEvent } from "react";
import { Button, InlineNotification, TextInput } from "@carbon/react";
import { Link } from "react-router-dom";

export type SavePanelStatus = "idle" | "saving" | "saved" | "error" | "conflict";

export type SavePanelProps = {
  title: string;
  status: SavePanelStatus;
  error?: string;
  validationMessage?: string;
  savedHref?: string;
  onTitleChange: (value: string) => void;
  onSubmit: () => void;
  onReload?: () => void;
};

export function SavePanel({
  title, status, error, validationMessage, savedHref,
  onTitleChange, onSubmit, onReload,
}: SavePanelProps) {
  const id = useId();
  const locked = status === "saving" || status === "saved";
  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (locked || status === "conflict" || validationMessage || !title.trim()) return;
    // The parent owns request identity, API, role checks and commit confirmation.
    onSubmit();
  }

  return (
    <section className="guide-save-panel" aria-labelledby={`${id}-heading`}>
      <h2 id={`${id}-heading`}>Сохранить работу</h2>
      <p>Название поможет найти работу в каталоге.</p>
      <form onSubmit={submit} aria-busy={status === "saving"}>
        <TextInput
          id={`${id}-title`}
          labelText="Название работы"
          value={title}
          required
          readOnly={locked}
          invalid={Boolean(validationMessage)}
          invalidText={validationMessage}
          helperText="Название обязательно. Заполните его перед сохранением."
          onChange={(event) => onTitleChange(event.target.value)}
        />
        {(status === "error" || status === "conflict") && (
          <InlineNotification
            kind="error"
            title={status === "conflict" ? "Работа изменилась" : "Не удалось сохранить"}
            subtitle={error ?? (status === "conflict"
              ? "Ознакомьтесь с актуальной версией перед новым сохранением."
              : "Проверьте связь. Введённое название осталось в форме.")}
            hideCloseButton
          />
        )}
        <div className="guide-save-panel__actions">
          {status !== "saved" && (
            <Button
              type="submit"
              kind="primary"
              size="lg"
              disabled={locked || status === "conflict" || Boolean(validationMessage) || !title.trim()}
            >
              {status === "saving" ? "Сохраняем…" : status === "error" ? "Повторить сохранение" : "Сохранить работу"}
            </Button>
          )}
          {status === "conflict" && onReload && (
            <Button type="button" kind="tertiary" size="lg" onClick={onReload}>
              Открыть актуальную версию
            </Button>
          )}
          {status === "saved" && savedHref && <Link to={savedHref}>Открыть сохранённую работу →</Link>}
        </div>
        <p role="status" aria-atomic="true">
          {status === "saving" ? "Сохранение выполняется." : status === "saved" ? "Работа сохранена сервером." : ""}
        </p>
      </form>
    </section>
  );
}
