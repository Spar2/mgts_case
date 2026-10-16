export class PublicApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly detail: string,
  ) {
    super(detail);
    this.name = "PublicApiError";
  }
}

export async function throwPublicError(r: Response): Promise<never> {
  let detail: string;
  try {
    const b: unknown = await r.json();
    const d = (b as { detail?: unknown } | null | undefined)?.detail;
    detail =
      typeof d === "string" && d.length > 0
        ? d
        : `Request failed with status ${r.status}`;
  } catch {
    detail = `Request failed with status ${r.status}`;
  }
  throw new PublicApiError(r.status, detail);
}

export function clampIndex(index: number, length: number): number {
  if (!Number.isFinite(index) || length <= 0) return 0;
  const i = Math.trunc(index);
  return Math.min(Math.max(0, i), length - 1);
}

export function shouldRetryPublic(
  attempt: number,
  error: unknown,
): boolean {
  if (error instanceof DOMException && error.name === "AbortError")
    return false;
  if (attempt >= 3) return false;
  if (error instanceof TypeError) return true;
  return error instanceof PublicApiError && error.status === 503;
}
