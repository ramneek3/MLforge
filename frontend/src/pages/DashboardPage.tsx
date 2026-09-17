import { MetricCard } from "../components/MetricCard";
import { useAsync } from "../hooks/useAsync";
import { api } from "../services/api";
import type { MonitoringSummary } from "../types";

function fmt(value?: number | null, digits = 3) {
  return value == null ? "—" : value.toFixed(digits);
}

export function DashboardPage() {
  const { data, error, loading } = useAsync<MonitoringSummary>(() => api.summary() as Promise<MonitoringSummary>);
  if (loading) return <p>Loading dashboard…</p>;
  if (error) return <p className="text-rose-400">{error}</p>;
  if (!data) return null;
  return (
    <div>
      <h2 className="text-xl font-semibold mb-4">Production overview</h2>
      <div className="grid grid-cols-1 md:grid-cols-3 xl:grid-cols-4 gap-4">
        <MetricCard label="Production model" value={data.production_model_version || "none"} />
        <MetricCard label="Production F1" value={fmt(data.production_f1)} />
        <MetricCard label="ROC-AUC" value={fmt(data.production_roc_auc)} />
        <MetricCard label="Total predictions" value={String(data.total_predictions)} />
        <MetricCard label="API latency (ms)" value={fmt(data.avg_latency_ms, 1)} />
        <MetricCard label="Error rate" value={fmt(data.error_rate * 100, 2) + "%"} />
        <MetricCard label="Drift status" value={data.drift_detected ? "detected" : "stable"} />
        <MetricCard label="Retrain needed" value={data.retraining_required ? "yes" : "no"} />
      </div>
    </div>
  );
}
