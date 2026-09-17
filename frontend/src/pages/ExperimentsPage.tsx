import { useAsync } from "../hooks/useAsync";
import { api } from "../services/api";
import type { ExperimentRun } from "../types";

export function ExperimentsPage() {
  const { data, error, loading } = useAsync<{ experiments: ExperimentRun[] }>(
    () => api.experiments() as Promise<{ experiments: ExperimentRun[] }>,
  );
  if (loading) return <p>Loading experiments…</p>;
  if (error) return <p className="text-rose-400">{error}</p>;
  return (
    <div>
      <h2 className="text-xl font-semibold mb-4">Experiment runs</h2>
      <div className="space-y-3">
        {(data?.experiments || []).map((run) => (
          <div key={run.run_id} className="rounded-lg border border-line bg-panel p-4">
            <p className="text-sm">
              <span className="text-slate-400">Run</span> {run.run_id.slice(0, 8)} · {run.model_type} · {run.status} ·{" "}
              {run.duration_seconds.toFixed(1)}s
            </p>
            <p className="text-xs text-slate-400 mt-2">
              F1 {run.metrics.f1?.toFixed(3)} · ROC-AUC {run.metrics.roc_auc?.toFixed(3)} · Recall{" "}
              {run.metrics.recall?.toFixed(3)}
            </p>
            <p className="text-xs text-slate-500 mt-1">
              {Object.entries(run.params)
                .filter(([key]) => key.startsWith("hp_"))
                .map(([key, value]) => `${key.replace("hp_", "")}=${value}`)
                .join(" · ")}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
