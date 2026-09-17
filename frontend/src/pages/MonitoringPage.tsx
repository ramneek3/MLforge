import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useAsync } from "../hooks/useAsync";
import { api } from "../services/api";
import type { DriftReport, MonitoringSummary } from "../types";
import { MetricCard } from "../components/MetricCard";

export function MonitoringPage() {
  const summary = useAsync<MonitoringSummary>(() => api.summary() as Promise<MonitoringSummary>);
  const drift = useAsync<DriftReport>(() => api.drift() as Promise<DriftReport>);

  if (summary.loading || drift.loading) return <p>Loading monitoring…</p>;
  if (summary.error) return <p className="text-rose-400">{summary.error}</p>;
  const s = summary.data;
  const d = drift.data;
  const volume = [
    { name: "predictions", value: s?.total_predictions || 0 },
    { name: "positive", value: Math.round((s?.positive_prediction_rate || 0) * (s?.total_predictions || 0)) },
  ];
  return (
    <div>
      <h2 className="text-xl font-semibold mb-4">Monitoring</h2>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <MetricCard label="Request volume" value={String(s?.total_predictions ?? 0)} />
        <MetricCard label="Latency ms" value={(s?.avg_latency_ms ?? 0).toFixed(1)} />
        <MetricCard label="Error rate" value={`${((s?.error_rate ?? 0) * 100).toFixed(2)}%`} />
        <MetricCard label="Drift" value={d?.drift_detected ? "yes" : "no"} />
      </div>
      <div className="h-64 rounded-lg border border-line bg-panel p-4 mb-6">
        <p className="text-sm mb-2">Prediction distribution</p>
        <ResponsiveContainer width="100%" height="90%">
          <BarChart data={volume}>
            <CartesianGrid strokeDasharray="3 3" stroke="#223049" />
            <XAxis dataKey="name" stroke="#94a3b8" />
            <YAxis stroke="#94a3b8" />
            <Tooltip />
            <Bar dataKey="value" fill="#5b8def" />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="rounded-lg border border-line">
        <table className="w-full text-sm">
          <thead className="bg-panel text-slate-400">
            <tr>
              <th className="text-left px-3 py-2">Feature</th>
              <th className="text-left px-3 py-2">Type</th>
              <th className="text-left px-3 py-2">Score</th>
              <th className="text-left px-3 py-2">Drift</th>
            </tr>
          </thead>
          <tbody>
            {(d?.features || []).map((row) => (
              <tr key={row.feature} className="border-t border-line">
                <td className="px-3 py-2">{row.feature}</td>
                <td className="px-3 py-2">{row.type}</td>
                <td className="px-3 py-2">{row.score}</td>
                <td className="px-3 py-2">{row.drift}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {d?.reason && <p className="mt-3 text-sm text-slate-400">{d.reason}</p>}
    </div>
  );
}
