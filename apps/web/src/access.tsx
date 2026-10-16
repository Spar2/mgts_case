import { ThemeSwitch } from "./theme";
import {
  createContext,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { useQueryClient } from "@tanstack/react-query";
import { Button, TextInput, InlineNotification, Loading } from "@carbon/react";
import { tiles } from "./cache";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { safeReturnPath } from "./auth-navigation";
import { ApiError, readFailure } from "./api";

type Account = {
  id: string;
  display_name: string;
  username: string;
  role: string;
  csrf: string;
  project_id: string;
  last_seen: number;
  expires: number;
};
const AccountContext = createContext<Account | null>(null);
export const useAccount = () => useContext(AccountContext);
function notifyAccountChange() {
  if (typeof BroadcastChannel === "undefined") return;
  const channel = new BroadcastChannel("dts-account-change");
  channel.postMessage("changed");
  channel.close();
}
export async function control<T>(
  path: string,
  body?: unknown,
  signal?: AbortSignal,
  idempotencyKey?: string,
  method: "POST" | "PATCH" = "POST",
): Promise<T> {
  let session: { mutation_token: string } | null = null;
  if (body !== undefined) {
    const response = await fetch("/api/v1/session", { signal });
    if (response.status === 401)
      window.dispatchEvent(new Event("dts-auth-expired"));
    if (!response.ok)
      throw new ApiError(response.status, {
        detail:
          "Не удалось подтвердить сессию. Проверьте связь и повторите действие.",
      });
    session = (await response.json()) as { mutation_token: string };
  }
  const response = await fetch("/api/v2" + path, {
    method: body === undefined ? "GET" : method,
    headers:
      body === undefined
        ? {}
        : {
            "Content-Type": "application/json",
            "X-Workbench-Token": session!.mutation_token,
            ...(idempotencyKey ? { "Idempotency-Key": idempotencyKey } : {}),
          },
    body: body === undefined ? undefined : JSON.stringify(body),
    signal,
  });
  const value = response.ok ? await response.json() : await readFailure(response);
  if (response.status === 401)
    window.dispatchEvent(new Event("dts-auth-expired"));
  if (!response.ok)
    throw new ApiError(response.status, {
      detail:
        typeof value.detail === "string"
          ? value.detail
          : "Не удалось выполнить действие. Проверьте данные и повторите.",
    });
  return value;
}
export function AccessGate({ children }: { children: ReactNode }) {
  const route = useLocation();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const securityEpoch = useRef(0);
  const [account, setAccount] = useState<Account | null>(null),
    [ready, setReady] = useState(false),
    [enabled, setEnabled] = useState(false);
  const [username, setUsername] = useState(""),
    [password, setPassword] = useState(""),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false);
  const [connectionError, setConnectionError] = useState(""),
    [attempt, setAttempt] = useState(0);
  useEffect(() => {
    const expire = () => {
      securityEpoch.current += 1;
      setAccount(null);
      setEnabled(true);
      setReady(true);
      setPassword("");
      void queryClient.cancelQueries();
      queryClient.clear();
      tiles.clear();
      sessionStorage.removeItem("dts-reviewer");
      setError("Сессия истекла. Войдите снова; серверные ответы сохранены.");
    };
    window.addEventListener("dts-auth-expired", expire);
    return () => window.removeEventListener("dts-auth-expired", expire);
  }, [queryClient]);
  useEffect(() => {
    const controller = new AbortController();
    const epoch = securityEpoch.current;
    setReady(false);
    setConnectionError("");
    void fetch("/api/v1/health", { signal: controller.signal })
      .then((r) => {
        if (!r.ok) throw new Error("Сервер не отвечает");
        return r.json();
      })
      .then(async (health) => {
        if (controller.signal.aborted || epoch !== securityEpoch.current) return;
        setEnabled(!!health.auth_enabled);
        if (!health.auth_enabled && health.access_mode !== "isolated-test")
          throw new Error("Сервер не настроен для командного входа");
        if (health.auth_enabled) {
          const r = await fetch("/api/v2/auth/me", {
            signal: controller.signal,
          });
          if (r.ok) {
            const a = (await r.json()) as Account;
            if (controller.signal.aborted || epoch !== securityEpoch.current) return;
            sessionStorage.setItem("dts-reviewer", a.display_name);
            setAccount(a);
          } else if (r.status === 401) {
            setAccount(null);
            queryClient.clear();
            tiles.clear();
            sessionStorage.removeItem("dts-reviewer");
          } else throw new Error("Не удалось проверить сессию");
        }
        setReady(true);
      })
      .catch(() => {
        if (controller.signal.aborted) return;
        setConnectionError(
          "Не удалось связаться с сервером. Сохранённые на сервере ответы остаются в журнале.",
        );
        setReady(true);
      });
    return () => controller.abort();
  }, [attempt, queryClient]);
  useEffect(() => {
    if (typeof BroadcastChannel === "undefined") return;
    const channel = new BroadcastChannel("dts-account-change");
    channel.onmessage = (event) => {
      if (event.data !== "changed") return;
      void queryClient.cancelQueries();
      queryClient.clear();
      tiles.clear();
      sessionStorage.removeItem("dts-reviewer");
      // A full reload also disposes graph buffers and per-account UI state.
      location.reload();
    };
    return () => channel.close();
  }, [queryClient]);
  useEffect(() => {
    if (!ready || connectionError || !enabled) return;
    if (!account && route.pathname !== "/login") {
      navigate("/login?next=" + encodeURIComponent(route.pathname + route.search), { replace: true });
    } else if (account && route.pathname === "/login") {
      navigate(safeReturnPath(new URLSearchParams(route.search).get("next")), { replace: true });
    }
  }, [ready, connectionError, enabled, account, route.pathname, route.search, navigate]);
  const [warning, setWarning] = useState(false);
  useEffect(() => {
    if (!account) return;
    let lastSent = 0;
    const activity = (event: Event) => {
      if (!event.isTrusted || Date.now() - lastSent < 60000) return;
      lastSent = Date.now();
      void control("/auth/activity", {}).catch(() => {});
    };
    const timer = window.setInterval(() => {
      void control<Account>("/auth/me")
        .then((a) => {
          if (a.id !== account.id || a.project_id !== account.project_id) {
            queryClient.clear();
            tiles.clear();
            location.replace("/research");
            return;
          }
          setWarning(
            Math.min(a.expires, a.last_seen + 1800) - Date.now() / 1000 < 120,
          );
        })
        .catch(() => {});
    }, 30000);
    window.addEventListener("pointerdown", activity);
    window.addEventListener("keydown", activity);
    return () => {
      clearInterval(timer);
      window.removeEventListener("pointerdown", activity);
      window.removeEventListener("keydown", activity);
    };
  }, [account, queryClient]);
  if (!ready)
    return (
      <main className="page">
        <div className="auth-theme-tools"><ThemeSwitch /></div>
        <Loading description="Проверяем доступ" withOverlay={false} />
      </main>
    );
  if (connectionError)
    return (
      <main className="page">
        <div className="auth-theme-tools"><ThemeSwitch /></div>
        <h1>Сервер временно недоступен</h1>
        <p role="alert">{connectionError}</p>
        <Button onClick={() => setAttempt((v) => v + 1)}>
          Повторить подключение
        </Button>
      </main>
    );
  if (enabled && !account)
    return (
      <main className="page">
        <div className="auth-theme-tools"><Link to="/">← На главную</Link><ThemeSwitch /></div>
        <h1>Единая аналитическая платформа сенсорных данных · вход</h1>
        <p>Для доступа получите одноразовое приглашение у администратора команды.</p>
        <p>
          Учётная запись определяет проект, права и независимый журнал эксперта.
        </p>
        <form
          className="review-form"
          onSubmit={(e) => {
            e.preventDefault();
            if (busy || !username.trim() || password.length < 12) return;
            setBusy(true);
            setError("");
            void fetch("/api/v2/auth/login", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ username, password }),
            })
              .then(async (r) => {
                const data = r.ok ? await r.json() : await readFailure(r);
                if (!r.ok)
                  throw new Error(
                    r.status === 503
                      ? "Сервер временно недоступен. Повторите вход позже."
                      : r.status === 429
                        ? "Слишком много попыток входа. Подождите минуту и повторите."
                      : r.status === 401
                        ? "Не удалось войти. Проверьте логин и пароль; если доступ отключён, обратитесь к администратору."
                      : typeof data.detail === "string"
                        ? data.detail
                        : "Проверьте логин и пароль.",
                  );
                const a = await control<Account>("/auth/me");
                sessionStorage.setItem("dts-reviewer", a.display_name);
                setPassword("");
                queryClient.clear();
                tiles.clear();
                setAccount(a);
                notifyAccountChange();
              })
              .catch((e) => setError(String(e)))
              .finally(() => setBusy(false));
          }}
        >
          <TextInput
            id="login-name"
            autoComplete="username"
            maxLength={100}
            labelText="Логин"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
          />
          <TextInput
            id="login-password"
            type="password"
            autoComplete="current-password"
            maxLength={512}
            labelText="Пароль"
            helperText={password.length > 0 && password.length < 12 ? "Минимум 12 символов. Пока введено меньше — кнопка входа недоступна." : "Пароль аккаунта — минимум 12 символов."}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
          <Button
            type="submit"
            disabled={busy || !username.trim() || password.length < 12}
          >
            Войти
          </Button>
          {import.meta.env.DEV && (
            <Button
              kind="tertiary"
              type="button"
              disabled={busy}
              onClick={() => {
                setBusy(true);
                setError("");
                const next = safeReturnPath(
                  new URLSearchParams(route.search).get("next"),
                );
                void fetch("/__local/enter", { method: "POST" })
                  .then(async (response) => {
                    if (!response.ok) throw new Error("Локальный вход не выполнен");
                    location.assign(next);
                  })
                  .catch((reason) => {
                    setError(String(reason));
                    setBusy(false);
                  });
              }}
            >
              Локальный вход
            </Button>
          )}
          {error && (
            <InlineNotification
              kind="error"
              title="Вход не выполнен"
              subtitle={error}
              hideCloseButton
            />
          )}
          <p>
            <Link to="/activate">Создать аккаунт по приглашению</Link>
          </p>
          <p>
            <Link to="/recover">Восстановить доступ</Link>
          </p>
          <p>
            Нет аккаунта? Попросите администратора прислать одноразовую ссылку.
            По ней вы сможете выбрать логин и установить собственный пароль.
          </p>
        </form>
      </main>
    );
  return (
    <AccountContext.Provider value={account}>
      {warning && (
        <InlineNotification
          kind="warning"
          title="Сессия скоро завершится"
          subtitle="До завершения менее двух минут. Продолжите работу или войдите снова после истечения 12 часов."
          hideCloseButton
        />
      )}
      {children}
    </AccountContext.Provider>
  );
}
export function AccountActions() {
  const account = useAccount();
  const [busy, setBusy] = useState(false),
    [error, setError] = useState("");
  return account ? (
    <span className="account-actions">
      {account.display_name} · {account.role}{" "}
      <button
        disabled={busy}
        onClick={() => {
          setBusy(true);
          setError("");
          void control("/auth/logout", {})
            .then(() => {
              notifyAccountChange();
              tiles.clear();
              sessionStorage.removeItem("dts-reviewer");
              location.assign("/");
            })
            .catch(() => setError("Выход не выполнен. Повторите."))
            .finally(() => setBusy(false));
        }}
      >
        Выйти
      </button>
      {error && <span role="alert">{error}</span>}
    </span>
  ) : null;
}
