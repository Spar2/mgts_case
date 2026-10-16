import { useEffect, useState } from "react";
import { Button, TextInput } from "@carbon/react";

/** Keep editing separate from the selected observation; commit an entire number. */
export function CoordinateInput({
  id,
  label,
  value,
  min,
  max,
  integer = false,
  disabled = false,
  previousLabel,
  nextLabel,
  onCommit,
  onPrevious,
  onNext,
}: {
  id: string;
  label: string;
  value: number;
  min: number;
  max: number;
  integer?: boolean;
  disabled?: boolean;
  previousLabel?: string;
  nextLabel?: string;
  onCommit: (value: number) => void;
  onPrevious: () => void;
  onNext: () => void;
}) {
  const [draft, setDraft] = useState(String(value));
  const [focused, setFocused] = useState(false);
  const [invalid, setInvalid] = useState(false);
  useEffect(() => {
    if (!focused && !invalid) setDraft(String(value));
  }, [value, focused, invalid]);
  function commit() {
    const number = Number(draft.trim().replace(",", "."));
    const ok =
      draft.trim() !== "" &&
      Number.isFinite(number) &&
      number >= min &&
      number <= max &&
      (!integer || Number.isInteger(number));
    setInvalid(!ok);
    if (ok) onCommit(number);
  }
  return (
    <div className="coordinate-input">
      <TextInput
        id={id}
        labelText={label}
        inputMode={integer ? "numeric" : "decimal"}
        value={draft}
        disabled={disabled}
        onFocus={() => setFocused(true)}
        onChange={(e) => {
          setDraft(e.target.value);
          setInvalid(false);
        }}
        onBlur={() => {
          commit();
          setFocused(false);
        }}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            e.preventDefault();
            e.currentTarget.blur();
          }
        }}
        invalid={invalid}
        invalidText={`Введите ${integer ? "целое число" : "координату"} от ${min} до ${max}`}
        helperText="Enter или завершите ввод для выбора наблюдения."
      />
      <div className="coordinate-steps">
        <Button
          kind="tertiary"
          size="sm"
          disabled={disabled || value <= min}
          aria-label={
            previousLabel ??
            (integer ? "Предыдущий профиль" : "Предыдущая координата")
          }
          onClick={onPrevious}
        >
          ← Назад
        </Button>
        <Button
          kind="tertiary"
          size="sm"
          disabled={disabled || value >= max}
          aria-label={
            nextLabel ??
            (integer ? "Следующий профиль" : "Следующая координата")
          }
          onClick={onNext}
        >
          Вперёд →
        </Button>
      </div>
    </div>
  );
}
