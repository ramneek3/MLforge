const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";
const ADMIN_KEY_STORAGE = "mlforge-admin-key";

// The admin key is never bundled into production builds: anything prefixed VITE_ is public.
// Local dev can still set VITE_API_KEY; otherwise the user is asked once per tab.
function adminKey(): string | null {
  if (import.meta.env.DEV && import.meta.env.VITE_API_KEY) return import.meta.env.VITE_API_KEY;
  const stored = sessionStorage.getItem(ADMIN_KEY_STORAGE);
  if (stored) return stored;
  const entered = window.prompt("Admin API key (required to train or promote models)")?.trim();
  if (entered) sessionStorage.setItem(ADMIN_KEY_STORAGE, entered);
  return entered || null;
}

function errorMessage(detail: unknown, fallback: string): string {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return detail.map((item) => item.msg || JSON.stringify(item)).join("; ");
  return fallback;
}

async function request<T>(path: string, options: RequestInit = {}, auth = false): Promise<T> {
  const headers: Record<string, string> = { ...(options.headers as Record<string, string> | undefined) };
  if (options.body) {
    headers["Content-Type"] = "application/json";
  }
  if (auth) {
    const key = adminKey();
    if (!key) throw new Error("Admin API key required.");
    headers["X-API-Key"] = key;
  }
  const response = await fetch(`${API_BASE}${path}`, { ...options, headers });
  if (!response.ok) {
    if (auth && response.status === 401) sessionStorage.removeItem(ADMIN_KEY_STORAGE);
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(errorMessage(body.detail, "Request failed"));
  }
  return response.json() as Promise<T>;
}

export const api = {
  health: () => request("/health"),
  summary: () => request("/monitoring/summary"),
  drift: () => request("/monitoring/drift"),
  models: (name: string) => request(`/models/${name}/versions`),
  experiments: () => request("/experiments"),
  train: (payload: Record<string, unknown>) =>
    request("/train", { method: "POST", body: JSON.stringify(payload) }, true),
  promote: (name: string, version: string, action: string) =>
    request(
      `/models/${name}/promote`,
      {
        method: "POST",
        body: JSON.stringify({ version, action }),
      },
      true,
    ),
};
