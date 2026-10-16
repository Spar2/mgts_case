import { ThemeSwitch } from "./theme";
import { Component, type ReactNode } from "react";
import { Button, InlineNotification } from "@carbon/react";
export class ErrorBoundary extends Component<
  { children: ReactNode },
  { failed: boolean }
> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    return this.state.failed ? (
      <main className="page">
        <div className="auth-theme-tools"><ThemeSwitch /></div>
        <h1>Рабочая область не открылась</h1>
        <InlineNotification
          kind="error"
          title="Не удалось отобразить интерфейс"
          subtitle="Обновите приложение. Исходные записи и подтверждённые сервером ответы сохраняются; несохранённый текст этой вкладки будет потерян."
          hideCloseButton
        />
        <Button onClick={() => location.reload()}>Обновить приложение</Button>
      </main>
    ) : (
      this.props.children
    );
  }
}
