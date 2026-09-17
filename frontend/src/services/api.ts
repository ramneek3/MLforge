const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";
const API_KEY = import.meta.env.VITE_API_KEY || "changeme-admin-api-key";

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
    headers["X-API-Key"] = API_KEY;
  }
  const response = await fetch(`${API_BASE}${path}`, { ...options, headers });
  if (!response.ok) {
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
