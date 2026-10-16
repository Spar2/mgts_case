import { ThemeSwitch, chartColors, useAppTheme } from "./theme";
import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import { Link, useLocation } from "react-router-dom";
import {
  Button,
  InlineNotification,
  TextInput,
  TextArea,
  Select,
  SelectItem,
} from "@carbon/react";
import { useQuery } from "@tanstack/react-query";
import { LineChart } from "./charts";
import { palette } from "./palette";
import {
  PublicApiError,
  clampIndex,
  shouldRetryPublic,
  throwPublicError,
} from "./public-error";

import type { components } from "./api-schema";
type Package = components["schemas"]["PublicPackage"];
async function post(path: string, body: unknown) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    let detail = "Проверьте поля и повторите действие.";
    try {
      const data = (await response.json()) as { detail?: unknown };
      if (typeof data.detail === "string" && data.detail.length > 0)
        detail = data.detail;
    } catch {
      // Не-JSON тело: общий текст, без SyntaxError наружу.
    }
    throw new PublicApiError(response.status, detail);
  }
  return response;
}
function ErrorMessage({ message }: { message: string }) {
  return message ? (
    <InlineNotification
      kind="error"
      title="Действие не выполнено"
      subtitle={message}
      hideCloseButton
    />
  ) : null;
}
function publicMessage(error: unknown): string {
  if (error instanceof DOMException && error.name === "AbortError") return "";
  if (error instanceof PublicApiError) {
    if (error.status === 503)
      return `Сервис временно недоступен. ${error.detail} Повторите загрузку.`;
    if (error.status === 401) return "Войдите, чтобы продолжить.";
    return "Данные недоступны. Повторите загрузку.";
  }
  return "Данные недоступны. Повторите загрузку.";
}
export function PublicLayout({ children }: { children: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <header className={`shell ${open ? "navigation-open" : ""}`}>
        <Link className="brand" to="/">
          Единая аналитическая платформа сенсорных данных
        </Link>
        <button
          className="mobile-menu"
          aria-expanded={open}
          aria-controls="public-navigation"
          onClick={() => setOpen((v) => !v)}
        >
          Меню
        </button>
        <nav
          id="public-navigation"
          aria-label="Публичная навигация"
          onClick={() => setOpen(false)}
        >
          <Link to="/login">Войти</Link>
        </nav>
        <ThemeSwitch />
      </header>
      <main className="page public-page">{children}</main>
    </>
  );
}
export { default as Landing } from "./landing";
function edges(axis: number[], index: number) {
  const step = index ? axis[index] - axis[index - 1] : axis[1] - axis[0];
  return [
    index ? (axis[index - 1] + axis[index]) / 2 : axis[0] - step / 2,
    index + 1 < axis.length
      ? (axis[index] + axis[index + 1]) / 2
      : axis[index] + step / 2,
  ];
}
function PublicField({
  data,
  processed,
  row,
  column,
  select,
}: {
  data: Package;
  processed: boolean;
  row: number;
  column: number;
  select: (r: number, c: number) => void;
}) {
  const { theme } = useAppTheme();
  const ref = useRef<HTMLCanvasElement>(null);
  const values = processed ? data.processed_c : data.raw_c;
  const scale = useMemo(() => {
    const v = data.raw_c.flat().filter((x): x is number => x !== null);
    return [Math.min(...v), Math.max(...v)];
  }, [data]);
  useEffect(() => {
    const canvas = ref.current!,
      ctx = canvas.getContext("2d");
    if (!ctx) return;
    const xs = edges(data.x_m, 0)[0],
      xe = edges(data.x_m, data.x_m.length - 1)[1];
    const ts = edges(data.time_s, 0)[0],
      te = edges(data.time_s, data.time_s.length - 1)[1];
    ctx.fillStyle = chartColors().background;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    values.forEach((line, r) =>
      line.forEach((value, c) => {
        const x = edges(data.x_m, c),
          t = edges(data.time_s, r);
        const i =
          Math.max(
            0,
            Math.min(
              255,
              Math.round(
                (255 * ((value ?? scale[0]) - scale[0])) /
                  (scale[1] - scale[0] || 1),
              ),
            ),
          ) * 4;
        ctx.fillStyle =
          value === null
            ? "#d0d0d0"
            : `rgb(${palette[i]},${palette[i + 1]},${palette[i + 2]})`;
        ctx.fillRect(
          ((x[0] - xs) / (xe - xs)) * canvas.width,
          ((t[0] - ts) / (te - ts)) * canvas.height,
          ((x[1] - x[0]) / (xe - xs)) * canvas.width + 0.2,
          ((t[1] - t[0]) / (te - ts)) * canvas.height + 0.2,
        );
      }),
    );
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 2;
    ctx.strokeRect(
      ((data.x_m[column] - xs) / (xe - xs)) * canvas.width - 4,
      ((data.time_s[row] - ts) / (te - ts)) * canvas.height - 4,
      8,
      8,
    );
  }, [data, values, row, column, scale, theme]);
  return (
    <figure>
      <canvas
        ref={ref}
        width={1024}
        height={400}
        className="public-field"
        role="img"
        aria-label="Температурное поле; точные значения доступны в выборе профиля и координаты"
        onClick={(e) => {
          const bounds = e.currentTarget.getBoundingClientRect();
          const x =
            data.x_m[0] +
            ((e.clientX - bounds.left) / bounds.width) *
              (data.x_m.at(-1)! - data.x_m[0]);
          const t =
            data.time_s[0] +
            ((e.clientY - bounds.top) / bounds.height) *
              (data.time_s.at(-1)! - data.time_s[0]);
          const nearest = (a: number[], v: number) =>
            a.reduce(
              (best, val, i) =>
                Math.abs(val - v) < Math.abs(a[best] - v) ? i : best,
              0,
            );
          select(nearest(data.time_s, t), nearest(data.x_m, x));
        }}
      />
      <figcaption>
        {data.x_m[0]}–{data.x_m.at(-1)} м · {data.timestamps[0]}–
        {data.timestamps.at(-1)} · шкала {scale[0].toFixed(2)}–
        {scale[1].toFixed(2)} °C · серый: пропуск
      </figcaption>
    </figure>
  );
}
export function TryDemo() {
  const query = useQuery({
    queryKey: ["public-package"],
    queryFn: async ({ signal }) => {
      const r = await fetch("/api/public/v1/package", { signal });
      if (!r.ok) await throwPublicError(r);
      try {
        return (await r.json()) as Package;
      } catch {
        throw new PublicApiError(
          r.status,
          "Данные недоступны. Повторите загрузку.",
        );
      }
    },
    retry: false,
    staleTime: Infinity,
  });
  const [row, setRow] = useState(25),
    [column, setColumn] = useState(42),
    [processed, setProcessed] = useState(false);
  const [answer, setAnswer] = useState("unknown"),
    [notes, setNotes] = useState(""),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false);
  const attemptsRef = useRef(0);
  const data = query.data,
    values = data && (processed ? data.processed_c : data.raw_c);
  const safeRow = data ? clampIndex(row, data.time_s.length) : 0;
  const safeColumn = data ? clampIndex(column, data.x_m.length) : 0;
  const shortPacket =
    !!data &&
    !!values &&
    (data.x_m.length === 0 ||
      data.time_s.length === 0 ||
      values.length === 0 ||
      values[safeRow] === undefined);
  const message = query.isError ? publicMessage(query.error) : "";
  return (
    <PublicLayout>
      <h1>Исследуйте реальное температурное окно</h1>
      <p>
        5.csv / review-00018 · доступен только опубликованный участок · ответ
        хранится в памяти этой страницы.
      </p>
      {query.isError && message !== "" && (
        <>
          <ErrorMessage message={message} />
          <Button
            disabled={
              !shouldRetryPublic(attemptsRef.current, query.error) ||
              query.isFetching
            }
            onClick={() => {
              attemptsRef.current += 1;
              void query.refetch();
            }}
          >
            Повторить загрузку ({attemptsRef.current}/3)
          </Button>
        </>
      )}
      {!data && query.isPending && <p role="status">Загружаем наблюдения…</p>}
      {data && values && shortPacket && !query.isPending && (
        <p role="status">Данные недоступны. Повторите загрузку.</p>
      )}
      {data && values && !shortPacket && (
        <>
          <div className="public-actions">
            <Button
              kind={processed ? "tertiary" : "primary"}
              onClick={() => setProcessed(false)}
            >
              Исходные данные
            </Button>
            <Button
              kind={processed ? "primary" : "tertiary"}
              onClick={() => setProcessed(true)}
            >
              Медианная обработка
            </Button>
          </div>
          <PublicField
            data={data}
            processed={processed}
            row={safeRow}
            column={safeColumn}
            select={(r, c) => {
              setRow(clampIndex(r, data.time_s.length));
              setColumn(clampIndex(c, data.x_m.length));
            }}
          />
          <div className="public-controls">
            <Select
              id="public-row"
              labelText="Профиль / время"
              value={safeRow}
              onChange={(e) =>
                setRow(
                  clampIndex(Number(e.target.value), data.timestamps.length),
                )
              }
            >
              {data.timestamps.map((t, i) => (
                <SelectItem key={i} value={i} text={t} />
              ))}
            </Select>
            <Select
              id="public-column"
              labelText="Координата, м"
              value={safeColumn}
              onChange={(e) =>
                setColumn(clampIndex(Number(e.target.value), data.x_m.length))
              }
            >
              {data.x_m.map((x, i) => (
                <SelectItem key={i} value={i} text={String(x)} />
              ))}
            </Select>
          </div>
          <p role="status">
            Точное значение: {values[safeRow][safeColumn] ?? "Пропуск"} °C ·
            исходное: {data.raw_c[safeRow][safeColumn] ?? "Пропуск"} · маска:{" "}
            {data.invalid_reason[safeRow][safeColumn]} · часовой пояс неизвестен
          </p>
          <div className="public-facts">
            <LineChart
              title="Профиль температуры"
              x={data.x_m}
              y={values[safeRow]}
              xLabel="Расстояние, м"
            />
            <LineChart
              title="Температура во времени"
              x={data.time_s}
              y={values.map((line) => line[safeColumn])}
              xLabel="Время от начала записи, с"
            />
          </div>
          <details>
            <summary>Происхождение и точные данные профиля</summary>
            <p className="public-sha">
              SHA источника: {data.source_sha256}
              <br />
              SHA пакета: {data.package_sha256}
            </p>
            <p>
              {data.provenance.method} · {data.provenance.method_version} ·{" "}
              {data.provenance.calculated_at}. Метод не обучался; обработка не
              подтверждает событие.
            </p>
            <div className="public-table">
              <table>
                <thead>
                  <tr>
                    <th>м</th>
                    <th>Исходное, °C</th>
                    <th>Обработка, °C</th>
                    <th>Маска</th>
                  </tr>
                </thead>
                <tbody>
                  {data.x_m.map((x, c) => (
                    <tr key={x}>
                      <td>{x}</td>
                      <td>{data.raw_c[safeRow][c] ?? "Пропуск"}</td>
                      <td>{data.processed_c[safeRow][c] ?? "Пропуск"}</td>
                      <td>{data.invalid_reason[safeRow][c]}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
          <section>
            <h2>Ваш демонстрационный ответ</h2>
            <Select
              id="demo-answer"
              labelText="Есть температурный эпизод?"
              value={answer}
              onChange={(e) => setAnswer(e.target.value)}
            >
              <SelectItem value="unknown" text="Недостаточно данных" />
              <SelectItem value="present" text="Есть" />
              <SelectItem value="absent" text="Нет" />
              <SelectItem value="ambiguous" text="Неоднозначно" />
            </Select>
            <TextArea
              id="demo-notes"
              labelText="Обоснование"
              maxLength={4000}
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
            />
            <Button
              disabled={busy}
              onClick={() => {
                setBusy(true);
                setError("");
                void post("/api/public/v1/report", { answer, notes })
                  .then((r) => r.blob())
                  .then((blob) => {
                    const url = URL.createObjectURL(blob),
                      a = document.createElement("a");
                    a.href = url;
                    a.download = "dts-demo-report.html";
                    a.click();
                    setTimeout(() => URL.revokeObjectURL(url), 1000);
                  })
                  .catch((e: unknown) =>
                    setError(
                      e instanceof PublicApiError
                        ? e.detail
                        : "Проверьте поля и повторите действие.",
                    ),
                  )
                  .finally(() => setBusy(false));
              }}
            >
              Скачать демонстрационный отчёт
            </Button>
            <ErrorMessage message={error} />
          </section>
        </>
      )}
    </PublicLayout>
  );
}
export function RequestAccess() {
  const [name, setName] = useState(""),
    [contact, setContact] = useState(""),
    [reason, setReason] = useState("");
  const [busy, setBusy] = useState(false),
    [done, setDone] = useState(false),
    [error, setError] = useState("");
  return (
    <PublicLayout>
      <h1>Запросить доступ к проекту</h1>
      <p>
        Администратор рассмотрит заявку и свяжется с вами лично. Права
        автоматически не выдаются. Закрытые заявки удаляются через 30 дней.
      </p>
      {done ? (
        <p role="status">
          Заявка принята. Ожидайте ответа по указанному контакту.
        </p>
      ) : (
        <form
          className="public-form"
          onSubmit={(e) => {
            e.preventDefault();
            setBusy(true);
            setError("");
            void post("/api/public/v1/access-requests", {
              name,
              contact,
              reason,
            })
              .then(() => setDone(true))
              .catch((e: unknown) =>
                    setError(
                      e instanceof PublicApiError
                        ? e.detail
                        : "Проверьте поля и повторите действие.",
                    ),
                  )
              .finally(() => setBusy(false));
          }}
        >
          <TextInput
            id="access-name"
            labelText="Имя"
            autoComplete="name"
            required
            maxLength={100}
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
          <TextInput
            id="access-contact"
            labelText="Контакт для обратной связи"
            helperText="Например, Telegram или электронная почта"
            required
            maxLength={200}
            value={contact}
            onChange={(e) => setContact(e.target.value)}
          />
          <TextArea
            id="access-reason"
            labelText="Для чего нужен доступ"
            required
            maxLength={2000}
            value={reason}
            onChange={(e) => setReason(e.target.value)}
          />
          <Button type="submit" disabled={busy}>
            {busy ? "Отправляем…" : "Отправить заявку"}
          </Button>
          <ErrorMessage message={error} />
        </form>
      )}
    </PublicLayout>
  );
}
// Capture once, before React mounts: StrictMode cannot discard the secret on remount.
const secretPath = location.pathname;
let linkSecret = "";
if (["/activate", "/recover"].includes(secretPath)) {
  linkSecret = location.hash.slice(1);
  history.replaceState(null, "", location.pathname);
}
export function Activate() {
  const recovery = useLocation().pathname === "/recover";
  const [username, setUsername] = useState(""),
    [name, setName] = useState(""),
    [password, setPassword] = useState(""),
    [confirm, setConfirm] = useState("");
  const [busy, setBusy] = useState(false),
    [done, setDone] = useState(false),
    [error, setError] = useState("");
  function submit(e: FormEvent) {
    e.preventDefault();
    if (password !== confirm) {
      setError("Пароли не совпадают.");
      return;
    }
    setBusy(true);
    setError("");
    void post("/api/v2/auth/" + (recovery ? "recover" : "activate"), {
      token: linkSecret,
      password,
      ...(!recovery ? { username, display_name: name } : {}),
    })
      .then(() => {
        linkSecret = "";
        setPassword("");
        setConfirm("");
        setDone(true);
      })
      .catch((e: unknown) =>
                    setError(
                      e instanceof PublicApiError
                        ? e.detail
                        : "Проверьте поля и повторите действие.",
                    ),
                  )
      .finally(() => setBusy(false));
  }
  return (
    <PublicLayout>
      <h1>{recovery ? "Восстановить доступ" : "Принять приглашение"}</h1>
      {done ? (
        <>
          <p role="status">Пароль установлен. Войдите обычным способом.</p>
          <Link to="/login">Перейти ко входу</Link>
        </>
      ) : !linkSecret ? (
        <p>
          Откройте одноразовую ссылку от администратора. После обновления этой
          страницы потребуется открыть её снова.
        </p>
      ) : (
        <form className="public-form" onSubmit={submit}>
          {!recovery && (
            <>
              <TextInput
                id="activate-login"
                autoComplete="username"
                required
                labelText="Ваш логин"
                maxLength={100}
                value={username}
                onChange={(e) => setUsername(e.target.value)}
              />
              <TextInput
                id="activate-name"
                autoComplete="name"
                required
                labelText="Имя"
                maxLength={100}
                value={name}
                onChange={(e) => setName(e.target.value)}
              />
            </>
          )}
          <TextInput
            id="activate-password"
            type="password"
            autoComplete="new-password"
            required
            minLength={12}
            maxLength={512}
            labelText="Новый пароль"
            helperText="Минимум 12 символов. Вставка и менеджеры паролей разрешены."
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
          <TextInput
            id="activate-confirm"
            type="password"
            autoComplete="new-password"
            required
            labelText="Повторите пароль"
            value={confirm}
            onChange={(e) => setConfirm(e.target.value)}
          />
          <Button type="submit" disabled={busy}>
            {busy ? "Сохраняем…" : "Установить пароль"}
          </Button>
          <ErrorMessage message={error} />
        </form>
      )}
    </PublicLayout>
  );
}
