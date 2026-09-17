import { FormEvent, useState } from "react";
import { api } from "../services/api";

export function TrainingPage() {
  const [model, setModel] = useState("xgboost");
  const [nEstimators, setNEstimators] = useState(150);
  const [maxDepth, setMaxDepth] = useState(4);
  const [status, setStatus] = useState("Idle");
  const [result, setResult] = useState("");

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    setStatus("Training… this can take a minute.");
    try {
      const payload = {
        trigger: "manual",
        model_type: model,
        hyperparameters: {
          [model]: {
            n_estimators: nEstimators,
            max_depth: maxDepth,
          },
        },
      };
      const body = await api.train(payload);
      setStatus("Complete");
      setResult(JSON.stringify(body, null, 2));
    } catch (err) {
      setStatus("Failed");
      setResult((err as Error).message);
    }
  }

  return (
    <div className="max-w-2xl">
      <h2 className="text-xl font-semibold mb-4">Start training</h2>
      <form onSubmit={onSubmit} className="space-y-4 rounded-lg border border-line bg-panel p-5">
        <label className="block text-sm">
          Candidate emphasis
          <select className="mt-1 w-full bg-ink border border-line rounded px-3 py-2" value={model} onChange={(e) => setModel(e.target.value)}>
            <option value="logistic_regression">Logistic Regression</option>
            <option value="random_forest">Random Forest</option>
            <option value="xgboost">XGBoost</option>
          </select>
        </label>
        <label className="block text-sm">
          n_estimators
          <input
            type="number"
            className="mt-1 w-full bg-ink border border-line rounded px-3 py-2"
            value={nEstimators}
            onChange={(e) => setNEstimators(Number(e.target.value))}
          />
        </label>
        <label className="block text-sm">
          max_depth
          <input
            type="number"
            className="mt-1 w-full bg-ink border border-line rounded px-3 py-2"
            value={maxDepth}
            onChange={(e) => setMaxDepth(Number(e.target.value))}
          />
        </label>
        <button className="rounded bg-accent px-4 py-2 text-sm font-medium text-white" type="submit">
          Train candidates
        </button>
      </form>
      <p className="mt-4 text-sm text-slate-300">{status}</p>
      {result && <pre className="mt-3 overflow-auto rounded bg-black/40 p-3 text-xs">{result}</pre>}
    </div>
  );
}
