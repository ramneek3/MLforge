export interface ModelVersion {
  model_name: string;
  version: string;
  status: string;
  model_type?: string | null;
  accuracy?: number | null;
  precision?: number | null;
  recall?: number | null;
  f1?: number | null;
  roc_auc?: number | null;
  created_at?: string | null;
  run_id?: string | null;
}

export interface MonitoringSummary {
  total_predictions: number;
  error_rate: number;
  avg_latency_ms: number;
  production_model_version?: string | null;
  production_f1?: number | null;
  production_roc_auc?: number | null;
  positive_prediction_rate: number;
  drift_detected: boolean;
  retraining_required: boolean;
}

export interface ExperimentRun {
  run_id: string;
  status: string;
  duration_seconds: number;
  model_type?: string;
  params: Record<string, string>;
  metrics: Record<string, number>;
}

export interface DriftReport {
  drift_detected: boolean;
  retraining_required: boolean;
  reason: string;
  engine?: string;
  features: Array<{ feature: string; type: string; score: number; drift: string }>;
}
