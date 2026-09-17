import { useState } from "react";
import { useAsync } from "../hooks/useAsync";
import { api } from "../services/api";
import type { ModelVersion } from "../types";

export function ModelsPage() {
  const { data, error, loading, reload } = useAsync<ModelVersion[]>(() => api.models("churn-model") as Promise<ModelVersion[]>);
  const [message, setMessage] = useState("");

  async function act(version: string, action: string) {
    try {
      const result = await api.promote("churn-model", version, action);
      setMessage(`${action} → ${JSON.stringify(result)}`);
      reload();
    } catch (err) {
      setMessage((err as Error).message);
    }
  }

  if (loading) return <p>Loading models…</p>;
  if (error) return <p className="text-rose-400">{error}</p>;
  return (
    <div>
      <h2 className="text-xl font-semibold mb-4">Model registry</h2>
      {message && <p className="mb-3 text-sm text-slate-300">{message}</p>}
      <div className="overflow-x-auto rounded-lg border border-line">
        <table className="w-full text-sm">
          <thead className="bg-panel text-slate-400">
            <tr>
              {["Name", "Version", "Type", "Status", "F1", "ROC-AUC", "Accuracy", "Created", "Actions"].map((h) => (
                <th key={h} className="text-left px-3 py-2 font-medium">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {(data || []).map((row) => (
              <tr key={row.version} className="border-t border-line">
                <td className="px-3 py-2">{row.model_name}</td>
                <td className="px-3 py-2">{row.version}</td>
                <td className="px-3 py-2">{row.model_type || "—"}</td>
                <td className="px-3 py-2">{row.status}</td>
                <td className="px-3 py-2">{row.f1?.toFixed(3) ?? "—"}</td>
                <td className="px-3 py-2">{row.roc_auc?.toFixed(3) ?? "—"}</td>
                <td className="px-3 py-2">{row.accuracy?.toFixed(3) ?? "—"}</td>
                <td className="px-3 py-2">{row.created_at || "—"}</td>
                <td className="px-3 py-2 space-x-2">
                  <button className="text-accent" onClick={() => act(row.version, "stage")}>
                    Stage
                  </button>
                  <button className="text-emerald-400" onClick={() => act(row.version, "promote")}>
                    Promote
                  </button>
                  <button className="text-rose-400" onClick={() => act(row.version, "reject")}>
                    Reject
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
