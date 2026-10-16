import { useEffect, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api, mutate } from "./api";

export function rehearsalChanged() {
  window.dispatchEvent(new Event("dts-rehearsal-changed"));
}

export function DemoGuardBanner() {
  const client = useQueryClient();
  const [leaveError, setLeaveError] = useState("");
  const [leaving, setLeaving] = useState(false);
  const state = useQuery({
    queryKey: ["rehearsal-state"],
    queryFn: ({ signal }) =>
      api<{ active: boolean; session_id: string | null }>(
        "/demo/state",
        signal,
      ),
    refetchInterval: 5000,
  });
  useEffect(() => {
    const refresh = () => {
      void client.invalidateQueries({ queryKey: ["rehearsal-state"] });
    };
    window.addEventListener("dts-rehearsal-changed", refresh);
    return () => window.removeEventListener("dts-rehearsal-changed", refresh);
  }, [client]);
  if (!state.data?.active) return null;
  return (
    <aside className="rehearsal-banner" aria-label="Защита рабочего журнала">
      <span>
        Репетиция: рабочие ответы и изменения заблокированы сервером во всех
        вкладках.
      </span>
      <Link to={`/demo?session=${state.data.session_id}`}>
        Вернуться к показу
      </Link>
      <button
        disabled={leaving}
        onClick={() => {
          setLeaving(true);
          setLeaveError("");
          void mutate("/demo/leave", {})
            .then(rehearsalChanged)
            .catch(() =>
              setLeaveError(
                "Не удалось завершить репетицию. Проверьте связь и повторите.",
              ),
            )
            .finally(() => setLeaving(false));
        }}
      >
        Завершить репетицию
      </button>
      {leaveError && <span role="alert">{leaveError}</span>}
    </aside>
  );
}
