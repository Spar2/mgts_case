import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Button,
  TextInput,
  Select,
  SelectItem,
  InlineNotification,
} from "@carbon/react";
import { control, useAccount } from "./access";

type Invitation = {
  id: string;
  role: string;
  expires: number;
  consumed: number | null;
  revoked: boolean;
};
type Session = {
  id: string;
  label: string;
  current: boolean;
  last_seen: number;
  expires: number;
};
type Request = {
  id: string;
  name: string;
  contact: string;
  reason: string;
  status: string;
};
export function InvitationsPanel() {
  const account = useAccount();
  const [role, setRole] = useState("viewer"),
    [password, setPassword] = useState(""),
    [link, setLink] = useState(""),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false);
  const query = useQuery({
    queryKey: ["invitations"],
    queryFn: () => control<Invitation[]>("/invitations"),
    enabled: account?.role === "admin",
  });
  if (account?.role !== "admin") return null;
  return (
    <section>
      <h2>Пригласить участника</h2>
      <p>
        Ссылка действует 24 часа и используется один раз. Передайте её лично;
        участник сам выберет пароль.
      </p>
      <form
        className="public-form"
        onSubmit={(e) => {
          e.preventDefault();
          setBusy(true);
          setError("");
          setLink("");
          void control<{ link: string }>("/invitations", {
            role,
            project_id: account.project_id,
            ...(role === "admin" ? { current_password: password } : {}),
          })
            .then((result) => {
              setLink(location.origin + result.link);
              setPassword("");
              void query.refetch();
            })
            .catch((e) => setError(String(e)))
            .finally(() => setBusy(false));
        }}
      >
        <Select
          id="invite-role"
          labelText="Роль участника"
          value={role}
          onChange={(e) => setRole(e.target.value)}
        >
          {["viewer", "expert", "analyst", "admin"].map((v) => (
            <SelectItem key={v} value={v} text={v} />
          ))}
        </Select>
        {role === "admin" && (
          <TextInput
            id="invite-confirm"
            type="password"
            autoComplete="current-password"
            labelText="Ваш пароль администратора"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
        )}
        <Button disabled={busy} type="submit">
          Создать приглашение
        </Button>
      </form>
      {link && (
        <div role="status">
          <p>Скопируйте ссылку сейчас — повторно секрет не показывается.</p>
          <TextInput
            id="invite-link"
            labelText="Одноразовая ссылка"
            readOnly
            value={link}
          />
          <Button
            kind="ghost"
            onClick={() =>
              void navigator.clipboard
                .writeText(link)
                .catch(() => setError("Выделите ссылку и скопируйте вручную."))
            }
          >
            Скопировать
          </Button>
          <Button kind="ghost" onClick={() => setLink("")}>
            Скрыть ссылку
          </Button>
        </div>
      )}
      {error && (
        <InlineNotification
          kind="error"
          title="Ошибка"
          subtitle={error}
          hideCloseButton
        />
      )}
      {query.data?.map((i) => (
        <article className="job-row" key={i.id}>
          <p>
            {i.role} · до {new Date(i.expires * 1000).toLocaleString()} ·{" "}
            {i.revoked ? "Отозвано" : i.consumed ? "Принято" : "Ожидает"}
          </p>
          {!i.revoked && !i.consumed && (
            <Button
              kind="ghost"
              onClick={() =>
                void control(`/invitations/${i.id}/revoke`, {})
                  .then(() => query.refetch())
                  .catch((e) => setError(String(e)))
              }
            >
              Отозвать
            </Button>
          )}
        </article>
      ))}
    </section>
  );
}
export default function AccountSecurity() {
  const account = useAccount();
  const [current, setCurrent] = useState(""),
    [password, setPassword] = useState(""),
    [confirm, setConfirm] = useState("");
  const [error, setError] = useState(""),
    [busy, setBusy] = useState(false),
    [recovery, setRecovery] = useState("");
  const sessions = useQuery({
    queryKey: ["account-sessions"],
    queryFn: () => control<Session[]>("/auth/sessions"),
  });
  const requests = useQuery({
    queryKey: ["access-requests"],
    queryFn: () => control<Request[]>("/access-requests"),
    enabled: account?.role === "admin" && account.project_id === "local",
  });
  const users = useQuery({
    queryKey: ["recovery-users"],
    queryFn: () => control<{ id: string; display_name: string }[]>("/accounts"),
    enabled: account?.role === "admin",
  });
  return (
    <main className="page public-page">
      <h1>Аккаунт и безопасность</h1>
      <p>
        {account?.display_name} · {account?.role} · проект {account?.project_id}
      </p>
      {error && (
        <InlineNotification
          kind="error"
          title="Действие не выполнено"
          subtitle={error}
          hideCloseButton
        />
      )}
      <section>
        <h2>Сменить пароль</h2>
        <p>
          После смены все сессии будут завершены. Серверные ответы сохраняются.
        </p>
        <form
          className="public-form"
          onSubmit={(e) => {
            e.preventDefault();
            if (password !== confirm) {
              setError("Пароли не совпадают");
              return;
            }
            setBusy(true);
            setError("");
            void control("/auth/password", {
              current_password: current,
              password,
            })
              .then(() => location.assign("/login"))
              .catch((e) => setError(String(e)))
              .finally(() => setBusy(false));
          }}
        >
          <TextInput
            id="current-password"
            type="password"
            labelText="Текущий пароль"
            autoComplete="current-password"
            required
            value={current}
            onChange={(e) => setCurrent(e.target.value)}
          />
          <TextInput
            id="changed-password"
            type="password"
            labelText="Новый пароль: минимум 12 символов"
            autoComplete="new-password"
            minLength={12}
            maxLength={512}
            required
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
          <TextInput
            id="changed-confirm"
            type="password"
            labelText="Повторите новый пароль"
            autoComplete="new-password"
            required
            value={confirm}
            onChange={(e) => setConfirm(e.target.value)}
          />
          <Button type="submit" disabled={busy}>
            Сменить пароль и завершить сессии
          </Button>
        </form>
      </section>
      <section>
        <h2>Активные сессии</h2>
        <Button
          kind="tertiary"
          onClick={() =>
            void control("/auth/sessions/others/revoke", {})
              .then(() => sessions.refetch())
              .catch((e) => setError(String(e)))
          }
        >
          Завершить остальные сессии
        </Button>
        {sessions.data?.map((s) => (
          <article className="job-row" key={s.id}>
            <p>
              {s.label} · {s.current ? "Эта сессия" : "Другая сессия"} · до{" "}
              {new Date(
                Math.min(s.expires, s.last_seen + 1800) * 1000,
              ).toLocaleString()}
            </p>
            <Button
              kind="ghost"
              onClick={() =>
                void control(`/auth/sessions/${s.id}/revoke`, {})
                  .then(() => {
                    if (s.current) location.assign("/login");
                    else void sessions.refetch();
                  })
                  .catch((e) => setError(String(e)))
              }
            >
              Завершить
            </Button>
          </article>
        ))}
      </section>
      <InvitationsPanel />
      {account?.role === "admin" && (
        <section>
          <h2>Восстановление участника</h2>
          <p>
            Сначала подтвердите личность через ранее известный канал связи.
            Ссылка действует 30 минут; пароль участник задаёт самостоятельно.
          </p>
          {users.data?.map((u) => (
            <Button
              kind="ghost"
              key={u.id}
              onClick={() =>
                void control<{ link: string }>(`/accounts/${u.id}/recovery`, {})
                  .then((r) => setRecovery(location.origin + r.link))
                  .catch((e) => setError(String(e)))
              }
            >
              Выдать ссылку: {u.display_name}
            </Button>
          ))}
          {recovery && (
            <>
              <TextInput
                id="recovery-link"
                readOnly
                labelText="Одноразовая ссылка восстановления"
                value={recovery}
              />
              <Button kind="ghost" onClick={() => setRecovery("")}>
                Скрыть ссылку
              </Button>
            </>
          )}
        </section>
      )}
      {requests.data && (
        <section>
          <h2>Заявки доступа</h2>
          {requests.data.map((r) => (
            <article className="job-row" key={r.id}>
              <h3>{r.name}</h3>
              <p>
                {r.contact} · {r.reason} · {r.status}
              </p>
              {r.status === "pending" && (
                <>
                  {["approved", "declined"].map((status) => (
                    <Button
                      kind="ghost"
                      key={status}
                      onClick={() =>
                        void control(`/access-requests/${r.id}/close`, {
                          status,
                        })
                          .then(() => requests.refetch())
                          .catch((e) => setError(String(e)))
                      }
                    >
                      {status === "approved"
                        ? "Рассмотрено: одобрить"
                        : "Отклонить"}
                    </Button>
                  ))}
                </>
              )}
            </article>
          ))}
        </section>
      )}
    </main>
  );
}
