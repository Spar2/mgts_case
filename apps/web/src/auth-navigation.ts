const routes = new Set(["works", "research", "review", "candidates", "runs", "analysis", "pipeline", "experiments", "cases", "demo", "system", "account", "laboratory", "monitoring"]);

/** Keep return links inside the application; never accept an external URL. */
export function safeReturnPath(value: string | null): string {
  if (!value || !value.startsWith("/") || value.startsWith("//") || value.includes("\\") || [...value].some((character) => character.charCodeAt(0) <= 32)) return "/research";
  try {
    const url = new URL(value, "https://workbench.invalid");
    if (url.origin !== "https://workbench.invalid" || !routes.has(url.pathname.split("/")[1])) return "/research";
    return url.pathname + url.search;
  } catch {
    return "/research";
  }
}
