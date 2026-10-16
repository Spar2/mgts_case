import type { components } from "./api-schema";
export type ReviewBody = components["schemas"]["Review"];
export type RecordMeta = components["schemas"]["RecordMeta"];
export type RecordDetail = components["schemas"]["RecordDetail"];
export type Slices = components["schemas"]["Slices"];
export type Card = components["schemas"]["Card"];
export type Annotation = components["schemas"]["Annotation"];
export type Queue = components["schemas"]["ReviewQueue"];
export type Replay = components["schemas"]["ReplayState"];
export class ApiError extends Error {
  constructor(
    public status: number,
    public body: { detail: string; current?: Annotation },
  ) {
    super(body.detail);
  }
}
let tokenPromise: Promise<string> | undefined;
// nginx 413/502 отдаёт HTML вместо JSON — не даём SyntaxError затмить статус.
export async function readFailure(r: Response): Promise<{ detail: string; current?: Annotation }> {
  try {
    return await r.json();
  } catch {
    return { detail: r.statusText || `Request failed with status ${r.status}` };
  }
}
// Успешный статус не гарантирует JSON-тело (усечённый прокси-ответ,
// captive portal): не даём SyntaxError вырваться наружу без статуса.
async function readJson<T>(r: Response): Promise<T> {
  try {
    return (await r.json()) as T;
  } catch {
    throw new ApiError(
      r.status,
      { detail: r.statusText || `Request failed with status ${r.status}` },
    );
  }
}
export async function download(path: string, body: unknown, filename: string) {
  for (let attempt = 0; attempt < 2; attempt++) {
    const response = await fetch("/api/v1" + path, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Workbench-Token": await token(),
      },
      body: JSON.stringify(body),
    });
    if (!response.ok) {
      const failure = await readFailure(response);
      if (
        response.status === 403 &&
        failure.detail === "Local mutation token required" &&
        attempt === 0
      ) {
        tokenPromise = undefined;
        continue;
      }
      throw new ApiError(response.status, failure);
    }
    const url = URL.createObjectURL(await response.blob()),
      link = document.createElement("a");
    link.href = url;
    link.download = filename;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    return;
  }
}
async function token() {
  return (tokenPromise ??= fetch("/api/v1/session")
    .then(async (r) => {
      if (!r.ok) throw new Error("Не удалось открыть локальную сессию");
      try {
        return (await r.json()).mutation_token as string;
      } catch {
        throw new Error("Не удалось открыть локальную сессию");
      }
    })
    .catch((e) => {
      tokenPromise = undefined;
      throw e;
    }));
}
export async function api<T>(path: string, signal?: AbortSignal): Promise<T> {
  const r = await fetch("/api/v1" + path, { signal });
  if (r.status === 401) window.dispatchEvent(new Event("dts-auth-expired"));
  if (!r.ok) throw new ApiError(r.status, await readFailure(r));
  return readJson<T>(r);
}
export async function mutate<T>(
  path: string,
  body: unknown,
  key?: string,
  method = "POST",
  signal?: AbortSignal,
): Promise<T> {
  for (let attempt = 0; attempt < 2; attempt++) {
    const r = await fetch("/api/v1" + path, {
      method,
      signal,
      keepalive: method === "DELETE",
      headers: {
        "Content-Type": "application/json",
        "X-Workbench-Token": await token(),
        ...(key ? { "Idempotency-Key": key } : {}),
      },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    if (!r.ok) {
      if (r.status === 401) window.dispatchEvent(new Event("dts-auth-expired"));
      const failure = await readFailure(r);
      // A rejected token has made no mutation. Refresh once after a server restart.
      if (
        r.status === 403 &&
        failure.detail === "Local mutation token required" &&
        attempt === 0
      ) {
        tokenPromise = undefined;
        continue;
      }
      throw new ApiError(r.status, failure);
    }
    return readJson<T>(r);
  }
  throw new Error("Не удалось восстановить локальную сессию");
}
