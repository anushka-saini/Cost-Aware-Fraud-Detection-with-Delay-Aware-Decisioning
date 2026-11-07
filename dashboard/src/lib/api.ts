// Locally, requests go through the Vite proxy: /api -> the FastAPI service.
// A hosted dashboard (static files, e.g. on Vercel) calls the API directly:
// build it with VITE_API_URL=https://your-api.example.com
const BASE = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, '') || '/api'
const TOKEN_KEY = 'fraud.token'

export type Tier = 'ALLOW' | 'REVIEW' | 'BLOCK'
export type Role = 'analyst' | 'admin'
export type Level = 'ok' | 'warning' | 'alert' | 'unknown'

export interface Session { access_token: string; username: string; role: Role }

export interface DemoAccount { username: string; password: string; role: Role }
export interface ServiceStatus {
  status: 'ok' | 'degraded'
  checked_at: string
  api_version: string
  environment: string
  demo_mode: boolean
  uptime_seconds: number
  worker_pid: number
  model: { name: string; version: string; thresholds: { t_review: number; t_block: number }; feature_count: number; train_days: number[] | null; created_at: string | null }
  database: { ok: boolean; backend: string; latency_ms: number }
  monitoring: { period: string; status: string; checked_at: string } | null
  activity: { transactions_scored_last_24h: number; open_cases: number } | null
  demo_accounts: DemoAccount[] | null
  demo_reset: { every_hours: number; last_reset_at: string | null; next_reset_at: string | null } | null
}

export interface CaseRow {
  id: number; status: 'open' | 'approved' | 'declined' | 'escalated'; amount: number; fraud_probability: number
  anomaly_score: number | null; tier: Tier; fallback_applied: boolean; opened_at: string; closed_at: string | null
  decided_by: string | null; assessment_id: string; reasons: string[]; type: string | null; step: number | null
  name_orig: string | null; name_dest: string | null; age_seconds: number
}
export interface CaseList { total: number; limit: number; offset: number; counts_by_status: Record<string, number>; cases: CaseRow[] }

export interface Contribution { feature: string; label: string; feature_value: number | boolean; shap_value: number }
export interface Explanation { base_value: number; units: 'probability' | 'log-odds'; contributions: Contribution[] }
export interface AuditRow {
  id: number; at: string; actor: string; role: string; action: string; entity_type: string; entity_id: string
  detail: Record<string, unknown>; request_id: string | null
}
export interface CaseDetail extends CaseRow {
  transaction: null | {
    id: number; step: number; type: string; amount: number; name_orig: string | null; name_dest: string
    old_balance_orig: number | null; new_balance_orig: number | null; old_balance_dest: number | null; new_balance_dest: number | null
  }
  scores: {
    fraud_probability: number; raw_score: number; anomaly_score: number | null; t_review: number; t_block: number
    supervised_tier: Tier | null; fallback_applied: boolean; model_version: string; anomaly_review_threshold: number | null
  }
  features: Record<string, number | boolean>
  explanation: Explanation | null
  recipient_history: { transaction_id: number; step: number; type: string | null; amount: number; name_orig: string | null }[]
  similar_cases: { id: number; status: string; amount: number; fraud_probability: number; decided_by: string | null }[]
  notes: { author: string; note: string; created_at: string }[]
  decisions: { actor: string; role: string; action: string; note: string; created_at: string }[]
  audit_trail: AuditRow[]
}

export interface StreamEvent {
  event: 'transaction'; position: number; assessment_id: string; transaction_id: number; fraud_probability: number
  raw_score: number; anomaly_score: number | null; tier: Tier; reasons: string[]; amount: number; type: string; step: number
  name_orig: string; name_dest: string; actual_fraud: number; created_at: string
}

export interface PsiRow { name: string; kind: 'feature' | 'score'; psi: number; level: Level }
export interface DriftReport {
  period: string; days: number[]; status: Level; n_rows: number; created_at: string
  psi: PsiRow[]
  rates: { name: string; observed: number; expected: number | null; band_low?: number; band_high?: number; level: Level }[]
  alerts: { type: string; name: string; level: Level; value: number }[]
  thresholds: { psi_warning: number; psi_alert: number }
}
export interface Drift {
  status: Level | 'no_reports'; reports: DriftReport[]
  live: { assessments: number; tier_counts: Record<Tier, number>; review_rate: number | null; block_rate: number | null
    customer_decisions: number; override_rate: number | null }
}

export interface Overview {
  kpis: {
    scored_last_24h: number; scored_total: number; block_rate: number | null; review_rate: number | null
    review_queue: number; escalated: number; oldest_open_case_seconds: number | null; override_rate: number | null
    active_model: { name: string; version: string }; drift_status: Level | 'no_reports'
  }
  scored_by_hour: number[]
  tier_mix: ({ from: number; to: number } & Record<Tier, number>)[]
  drift_alerts: { source: string; period: string; type: string; name: string; level: Level; value: number }[]
  recent_events: AuditRow[]
}

export interface SimulatorData {
  split: 'eval_2' | 'held_out'; days: number[]; model: { name: string; version: string }
  n_rows: number; n_fraud: number; n_legit: number
  deployed: { t_review: number; t_block: number; cost_false_block: number; cost_review: number; total_cost: number
    fraud_allowed: number; legit_reviewed: number; legit_reviewed_pct: number; legit_blocked: number
    block_precision: number; block_recall: number }
  fraud: [number, number][]
  legit_histogram: [number, number][]
  note: string
}

export interface ModelVersion {
  name: string; version: string; is_active: boolean; activated_at: string | null; activated_by: string | null
  created_at?: string; calibration?: string; thresholds?: { t_review: number; t_block: number }; train_days?: number[]
  metrics?: { eval_1_pr_auc?: number; eval_2_pr_auc?: number; held_out?: Record<string, number> }
  protocol?: Record<string, unknown>; git_hash?: string
}

export interface OperatingPoint {
  t_review: number; t_block: number; total_cost: number; fraud_allowed: number; legit_reviewed_pct: number
  legit_blocked: number; block_precision: number; block_recall: number; block_f1: number
}
export interface ModelSummary extends OperatingPoint {
  key: string; label: string; deployed: boolean; weight_name: string; weight: number | string; calibration: string
  pr_auc: number; ci_low: number; ci_high: number; seed_mean: number; seed_std: number; seed_values: number[]
  seed_costs: number[]; seed_fraud_allowed: number[]
  recall_by_amount_quartile: { quartile: string; n_fraud: number; recall_at_review: number; recall_at_block: number }[]
  old: null | (OperatingPoint & { pr_auc: number; ci_low: number; ci_high: number; weight: number; flaw: string; deployed: boolean })
}
export interface Interval { difference: number; ci_low: number; ci_high: number; significant: boolean }
export interface ResultsSummary {
  available: boolean; missing_sections?: string[]
  split: string; n_rows: number; n_fraud: number; protocol: string; features: string
  models: ModelSummary[]
  paired_tests: ({ a: string; b: string } & Interval)[]
  balance_ablation?: {
    removed_features: string[]
    with_balance_features: OperatingPoint & { pr_auc: number; ci_low: number; ci_high: number }
    without_balance_features: OperatingPoint & { pr_auc: number; ci_low: number; ci_high: number }
    pr_auc_difference_without_minus_with: Interval
  }
  deep_learning?: {
    autoencoder: { seed_42: { pr_auc: number; ci_low: number; ci_high: number; share_of_fraud_in_top_1pct: number
      share_of_fraud_in_top_5pct: number }; seeds: { values: number[]; mean: number; std: number }; base_rate: number }
    hybrid: OperatingPoint & { label: string; pr_auc: number; ci_low: number; ci_high: number; seed_mean: number; seed_std: number }
    hybrid_minus_supervised: Interval
    supervised_blind_spot: { n: number; in_top_1pct_most_anomalous: number; in_top_5pct_most_anomalous: number
      in_top_10pct_most_anomalous: number; amount: number }
  }
  rolling?: {
    by_delay: { delay_days: number
      supervised: { mean_daily_pr_auc: number; ci_low: number; ci_high: number }
      hybrid: { mean_daily_pr_auc: number; ci_low: number; ci_high: number }
      hybrid_minus_supervised: Interval }[]
    champion_challenger: Record<string, { n_promotions: number; mean_daily_pr_auc: number; always_retrain_mean_daily_pr_auc: number
      decisions: { scoring_day: number; validation_day: number; challenger_trained_through_day: number; decision: 'promote' | 'keep'
        reason: string; pr_auc_difference?: number; ci_low?: number; ci_high?: number; serving_model_trained_through_day: number
        serving_pr_auc_on_scoring_day: number }[] }>
  }
  stress_test?: {
    baseline: { pr_auc: number; total_cost: number; fraud_allowed: number; legit_reviewed_pct: number }
    scenarios: { kind: 'multiply' | 'add'; value: number; pr_auc: number; total_cost: number; fraud_allowed: number
      legit_reviewed_pct: number; would_alert: boolean }[]
  }
  latency?: { sequential: Record<string, { n: number; p50_ms: number; p95_ms: number } | null>
    concurrent: { clients: number; p50_ms: number; p95_ms: number; requests_per_second: number } }
}

export class ApiError extends Error {
  constructor(public status: number, message: string, public requestId?: string) { super(message) }
}

export const getToken = () => { try { return localStorage.getItem(TOKEN_KEY) } catch { return null } }
export const setToken = (token: string | null) => {
  try { token ? localStorage.setItem(TOKEN_KEY, token) : localStorage.removeItem(TOKEN_KEY) } catch { /* private mode */ }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers)
  const token = getToken()
  if (token) headers.set('Authorization', `Bearer ${token}`)
  if (init.body) headers.set('Content-Type', 'application/json')
  let response: Response
  try {
    response = await fetch(BASE + path, { ...init, headers })
  } catch {
    throw new ApiError(0, 'Could not reach the API. Is the service running on port 8000?')
  }
  if (!response.ok) {
    let detail = `The API returned HTTP ${response.status}.`
    let requestId: string | undefined
    try {
      const body = await response.json()
      requestId = body.request_id
      if (typeof body.detail === 'string') detail = body.detail
      if (Array.isArray(body.errors) && body.errors.length) {
        detail = body.errors.map((e: { field: string; message: string }) => `${e.field}: ${e.message}`).join('; ')
      }
    } catch { /* the Vite proxy answers 502/504 with no JSON when nothing is listening */
      if (response.status >= 500) detail = 'Could not reach the API. Is the service running on port 8000?'
    }
    if (response.status === 401 && token) {
      setToken(null)
      window.dispatchEvent(new Event('fraud:signed-out'))
    }
    throw new ApiError(response.status, detail, requestId)
  }
  return response.json() as Promise<T>
}

const query = (params: Record<string, string | number | undefined | null>) => {
  const search = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) if (value !== undefined && value !== null && value !== '') search.set(key, String(value))
  const text = search.toString()
  return text ? `?${text}` : ''
}
const post = <T>(path: string, body: unknown) => request<T>(path, { method: 'POST', body: JSON.stringify(body) })

// Version 3 results for the Model lab page (GET /results/v3).
export interface Ci { ci_low: number; ci_high: number; n_bootstrap: number }
export interface Test { difference: number; ci_low: number; ci_high: number; significant: boolean; p_holm?: number; significant_after_holm?: boolean }
export type Verdict = 'reference' | 'significantly better' | 'significantly worse' | 'comparable'
export interface LeaderboardRow {
  key: string; label: string; kind: string; tests_on: string
  pr_auc: ({ pr_auc: number } & Ci) | null; roc_auc: ({ roc_auc: number } & Ci) | null
  seed_mean: number | null; seed_std: number | null
  day_13_pr_auc: ({ pr_auc: number } & Ci) | null; day_13_cost: number | null; day_13_fraud_allowed: number | null
  day_13_cost_vs_v2: Test | null; vs_v3: Test | null; vs_v2: Test | null
  verdict_vs_v3: Verdict; verdict_vs_v2: Verdict
}
export interface HeldOutSide {
  pr_auc: { pr_auc: number } & Ci; roc_auc: { roc_auc: number } & Ci
  t_review: number; t_block: number; total_cost: number; fraud_allowed: number; legit_reviewed_pct: number
  legit_blocked: number; block_precision: number; block_recall: number
}
export interface HistoryRow {
  version: string; model: string; pr_auc: number; ci_low: number; ci_high: number; total_cost: number
  fraud_allowed: number; legit_blocked: number; honest: boolean; note: string; source: string
}
// The latest run of the retraining job (scripts/retrain.py).
export interface RetrainingGate {
  gate_rule: string; runs_in_log: number
  latest: {
    run_at: string; as_of_day: number; label_maturity_days: number; last_fully_labelled_day: number
    train_days: number[]; validation_days: number[]; n_train_rows: number; n_train_fraud: number
    feedback: { feedback_labels_found?: number; matured_feedback_labels: number; labels_changed: number }
    champion: { name: string; version: string; trained_through_day: number; validation_pr_auc: number }
    challenger: { validation_pr_auc: number; calibration: string; t_review: number; t_block: number }
    pr_auc_difference: number; ci_low: number; ci_high: number
    decision: 'promote' | 'keep'; reason: string; registered_version: string | null
  }
}
export interface CalibrationBin { from: number; to: number; n: number; n_fraud: number; mean_predicted: number; observed_rate: number }
// Rule baselines (scripts/rule_baselines.py): one block of measures per set of days.
export interface RuleResult {
  n_rows: number; n_fraud: number; recall: number | null; precision: number | null; total_cost: number
  fraud_allowed: number; fraud_allowed_amount: number; review_rate: number; block_rate: number
  legit_blocked: number; legit_reviewed_pct: number
}
export interface RuleRow {
  key: string; label: string; description: string; tier: string
  design_days: RuleResult; days_11_13: RuleResult; days_14_16: RuleResult
}
export interface RuleBaselines {
  design_days: number[]; how_rules_were_made: string; tier_rule: string
  cut_off_search: { cut_off: number; candidate_rule: string }
  rules: RuleRow[]; best_rule: string; best_rule_chosen_by: string; caveat: string
  v3: { days_11_13: RuleResult & { note: string }; days_14_16: RuleResult & { note: string } }
}
export interface V3Summary {
  available: boolean; missing_sections?: string[]
  v3_key: string
  validation: { evaluated_on: string; significance_rule: string; stacking_note: string; leaderboard: LeaderboardRow[] }
  phase2: {
    calibration: string; isotonic_rejected_because: string | null; threshold_rule: string
    point: Omit<HeldOutSide, 'pr_auc' | 'roc_auc'>; odds_factor_to_cliff: number; baselines: Record<string, number>
  }
  held_out: {
    days: number[]; n_rows: number; n_fraud: number; decision_rule: string; verdict: string; deploy_v3: boolean
    v3: HeldOutSide; v2: HeldOutSide; difference: { pr_auc: Test; roc_auc: Test; cost: Test }; baselines: Record<string, number>
  }
  history: HistoryRow[]
  curves: null | {
    curves_on: string; calibration_on: string; pr_note: string; roc_note: string
    curves: Record<string, { label: string; pr: { recall: number; precision: number }[]; roc: { fpr: number; tpr: number }[] }>
    calibration: Record<string, { label: string; bins: CalibrationBin[]; brier: number; ece: number; mean_predicted: number; base_rate: number }>
  }
  rules: RuleBaselines | null
  retraining: RetrainingGate | null
  sources: Record<string, string>
}

// Explainability page (GET /explainability) and POST /explain.
export interface FeatureImportance { feature: string; fraud_mean_abs: number; fraud_mean: number; legit_mean_abs: number; legit_mean: number }
export interface Scenario {
  id: string; title: string; rule: string; group_size: number; actual_fraud: number; expected_tier: Tier
  transaction: { step: number; type: string; nameOrig: string; nameDest: string; amount: number
    oldbalanceOrg: number; newbalanceOrig: number; oldbalanceDest: number; newbalanceDest: number }
  features: Record<string, number | boolean>
}
export interface BreakdownRow {
  group: string; n_rows: number; n_fraud: number; recall: number | null; block_recall: number | null
  fraud_allowed: number; review_rate: number; block_rate: number; legit_blocked: number
  amount_from?: number; amount_to?: number
}
export interface ExplainabilityData {
  global: null | { _source: string; days: number[]; n_fraud: number; n_legit_sample: number; units: string; model: { name: string; version: string }; features: FeatureImportance[] }
  breakdown: null | { _source: string; days: number[]; n_rows: number; n_fraud: number; what_this_is: string; check: string
    quartiles_of: string; by_type: BreakdownRow[]; by_amount_quartile: BreakdownRow[] }
  scenarios: Scenario[]; scenarios_source: string | null
}
export type ExplainResult = Explanation & { fraud_probability: number; raw_score: number }

export const api = {
  login: (username: string, password: string) => post<Session>('/auth/login', { username, password }),
  me: () => request<{ username: string; role: Role }>('/auth/me'),
  status: () => request<ServiceStatus>('/status'),
  overview: () => request<Overview>('/overview'),
  cases: (params: { status?: string; sort?: string; q?: string; type?: string; min_amount?: number; limit?: number; offset?: number }) =>
    request<CaseList>(`/cases${query(params)}`),
  caseDetail: (id: number) => request<CaseDetail>(`/cases/${id}`),
  decide: (id: number, action: string, note: string) => post<{ status: string }>(`/cases/${id}/decision`, { action, note }),
  decideMany: (case_ids: number[], action: string, note: string) =>
    post<{ decided: number[]; failed: { case_id: number; error: string }[] }>('/cases/bulk-decision', { case_ids, action, note }),
  addNote: (id: number, note: string) => post<{ status: string }>(`/cases/${id}/notes`, { note }),
  drift: () => request<Drift>('/monitoring/drift'),
  simulator: (split: string) => request<SimulatorData>(`/simulator/scores${query({ split })}`),
  propose: (body: Record<string, unknown>) => post<{ status: string; applied: boolean }>('/thresholds/proposals', body),
  models: () => request<{ active: string; versions: ModelVersion[] }>('/models'),
  rollback: (version: string | null, reason: string) => post<{ active: string; from: string }>('/models/rollback', { version, reason }),
  summary: () => request<ResultsSummary>('/results/summary'),
  v3: () => request<V3Summary>('/results/v3'),
  explainability: () => request<ExplainabilityData>('/explainability'),
  explain: (body: Record<string, unknown>) => post<ExplainResult>('/explain', body),
  audit: (params: { actor?: string; action?: string; entity_type?: string; q?: string; limit?: number; offset?: number }) =>
    request<{ total: number; limit: number; offset: number; actions: string[]; rows: AuditRow[] }>(`/audit${query(params)}`),
  auditCsv: async (params: { actor?: string; action?: string; q?: string }) => {
    const response = await fetch(`${BASE}/audit/export.csv${query(params)}`, { headers: { Authorization: `Bearer ${getToken()}` } })
    if (!response.ok) throw new ApiError(response.status, 'Could not export the audit log.')
    return response.blob()
  },
}

export function streamUrl(speed: number, restart: boolean) {
  const params = query({ token: getToken() ?? '', speed, restart: restart ? 'true' : undefined })
  if (BASE.startsWith('http')) {
    // The API is on another host: same address, WebSocket scheme.
    return `${BASE.replace(/^http/, 'ws')}/stream${params}`
  }
  const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
  return `${protocol}://${window.location.host}${BASE}/stream${params}`
}
