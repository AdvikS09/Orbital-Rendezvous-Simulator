/**
 * TypeScript mirror of the Pydantic models in
 * `web-platform/backend/app/schemas/`.
 *
 * Kept by hand rather than generated, because the set is small and the
 * hand-written version carries the same explanatory comments as the Python
 * side. `tests/contract.test.ts` checks these against the live OpenAPI schema,
 * so drift is caught rather than discovered.
 */

// --------------------------------------------------------------- formatting
export type NumberFormatName =
  | 'auto'
  | 'fixed'
  | 'significant'
  | 'scientific'
  | 'percent'
  | 'duration'
  | 'distance'
  | 'velocity'
  | 'angle'
  | 'integer'

export type Severity = 'info' | 'success' | 'warning' | 'error'

export interface Metric {
  key: string
  label: string
  value: number | string | null
  unit: string
  fmt: NumberFormatName
  precision: number | null
  help: string | null
  highlight: boolean
}

export interface Interpretation {
  severity: Severity
  text: string
  title: string | null
}

// -------------------------------------------------------------------- plots
export type PlotKind =
  | 'line'
  | 'scatter3d'
  | 'orbit3d'
  | 'heatmap'
  | 'dual_axis'
  | 'log_log'

export type DashStyle = 'solid' | 'dash' | 'dot' | 'dashdot'

export interface SeriesSpec {
  key: string
  label: string
  x_key: string | null
  y_key: string
  z_key: string | null
  unit: string
  color_index: number
  dash: DashStyle
  secondary_y: boolean
  width: number
  mode: 'lines' | 'markers' | 'lines+markers'
}

export interface AxisSpec {
  title: string
  unit: string
  log: boolean
}

export interface HeatmapMatrix {
  z: (number | null)[][]
  x: string[]
  y: string[]
  colorbar_title?: string
}

export interface PlotSpec {
  id: string
  title: string
  kind: PlotKind
  x: AxisSpec
  y: AxisSpec
  y2: AxisSpec | null
  z: AxisSpec | null
  x_key: string | null
  series: SeriesSpec[]
  caption: string | null
  show_earth: boolean
  show_target_marker: boolean
  keep_out_radius_m: number | null
  reference_line: Record<string, unknown> | null
  matrix: HeatmapMatrix | null
  height: number
}

// ------------------------------------------------------------------- tables
export interface ColumnSpec {
  key: string
  label: string
  unit: string
  fmt: NumberFormatName
  precision: number | null
  numeric: boolean
}

export interface TableSpec {
  id: string
  label: string
  columns: ColumnSpec[]
  rows: Record<string, unknown>[]
  caption: string | null
  downloadable: boolean
}

// --------------------------------------------------------------- provenance
export interface SolverInfo {
  integrator: string
  deterministic: boolean
  random_seed: number | null
  force_model: string | null
  model: string | null
  n_steps: number | null
  dt_s: number | null
  duration_s: number | null
  engine_version: string
  units: string
  frames: string
}

export interface ProvenanceInfo {
  software: string
  external_validation: string
  licence: string
}

// ---------------------------------------------------------------- scenarios
export type ParamKind = 'number' | 'slider' | 'checkbox' | 'select'
export type Track = 'guided' | 'professional' | 'free'
export type UserType = 'student' | 'teacher' | 'engineer_researcher' | 'other'

export interface ParamSpec {
  name: string
  label: string
  kind: ParamKind
  default: number | string | boolean
  minimum: number | null
  maximum: number | null
  step: number | null
  unit: string
  help: string
  options: (string | number)[]
  advanced: boolean
  fmt: NumberFormatName
  precision: number | null
  /** Bound enforced by the engine's validator; tighter than the widget range. */
  hard_minimum: number | null
  hard_maximum: number | null
}

export interface ScenarioSummary {
  id: string
  title: string
  category: string
  category_label: string
  feature_category: string
  summary: string
  levels: string[]
  result_kind: string
  model: string
  param_count: number
}

export interface ScenarioDetail extends Omit<ScenarioSummary, 'param_count'> {
  explanation: string
  reading_guide: string
  references: string[]
  assumptions: string[]
  limitations: string[]
  params: ParamSpec[]
  defaults: Record<string, number | string | boolean>
  estimated_steps: number | null
  estimated_seconds: number | null
}

// -------------------------------------------------------------- simulations
export type ParamValues = Record<string, number | string | boolean>

export interface SimulationRequest {
  params: ParamValues
  session_token?: string | null
  activity_id?: string | null
  step_id?: string | null
}

export interface InputEcho {
  name: string
  label: string
  value: number | string | boolean | null
  unit: string
}

export interface SimulationResult {
  status: 'ok'
  scenario_id: string
  scenario_title: string
  result_kind: string
  inputs: InputEcho[]
  derived: Metric[]
  metrics: Metric[]
  /** Raw numeric arrays. The frontend plots these; it never recomputes them. */
  series: Record<string, (number | null)[]>
  plots: PlotSpec[]
  tables: TableSpec[]
  interpretations: Interpretation[]
  warnings: string[]
  assumptions: string[]
  limitations: string[]
  reading_guide: string
  solver: SolverInfo
  provenance: ProvenanceInfo
  runtime_ms: number
  n_steps: number
  completed_at: string
  extras: Record<string, unknown>
}

export interface FieldError {
  field: string | null
  message: string
}

export interface ApiErrorBody {
  status: 'error'
  error_class: string
  message: string
  detail: FieldError[]
  warnings: string[]
  rejected: boolean
  retry_after_s: number | null
}

export interface PreflightResult {
  ok: boolean
  errors: FieldError[]
  warnings: string[]
  n_steps: number
  estimated_seconds: number
  params: ParamValues
}

// ------------------------------------------------------------------- health
export interface HealthResponse {
  status: 'ok' | 'degraded'
  engine_version: string
  api_version: string
  app_version: string
  environment: string
  database_ok: boolean
  database: string
  schema_version: number
  scenario_count: number
  warnings: string[]
}

// ------------------------------------------------------------------ session
export interface SessionRequest {
  pilot_code: string
  participant_code: string
  user_type: UserType
  consent_given: boolean
  consent_version?: string
}

export interface SessionResponse {
  session_token: string
  participant_ref: string
  user_type: UserType
  organisation_name: string
  pilot_code: string | null
  pilot_code_registered: boolean
  is_return_visit: boolean
  consent_version: string
  runs_remaining: number | null
}

export interface Acknowledgement {
  status: 'ok'
  recorded: boolean
  message: string
}

// --------------------------------------------------------------- assessment
export interface AssessmentItem {
  id: string
  question: string
  options: string[]
  topic: string
  correct_index: number | null
  explanation: string | null
}

export interface AssessmentInstrument {
  id: string
  version: string
  title: string
  description: string
  items: AssessmentItem[]
  confidence_question: string
  confidence_labels: string[]
  max_score: number
  review_status: string
}

export interface AssessmentSubmission {
  instrument_id?: string
  phase: 'pre' | 'post'
  responses: Record<string, number>
  confidence_rating?: number | null
}

export interface AssessmentResult {
  status: 'ok'
  recorded: boolean
  instrument_id: string
  phase: 'pre' | 'post'
  score: number
  max_score: number
  previous_score: number | null
  change: number | null
  items: AssessmentItem[]
  caveat: string
}

// --------------------------------------------------------------- activities
export interface ActivityStep {
  id: string
  title: string
  scenario_id: string
  brief: string
  predict_prompt: string
  predict_options: string[]
  params: ParamValues
  look_for: string
  takeaway: string
}

export interface Activity {
  id: string
  title: string
  level: string
  duration_minutes: number
  summary: string
  learning_points: string[]
  prerequisites: string
  review_status: string
  steps: ActivityStep[]
}

// ----------------------------------------------------------------- feedback
export interface FeedbackRequest {
  usefulness_rating?: number | null
  usability_rating?: number | null
  clarity_rating?: number | null
  would_recommend?: number | null
  most_useful?: string | null
  improvement?: string | null
  bug_report?: string | null
  feature_request?: string | null
}

// -------------------------------------------------------------------- admin
export interface SummaryMetrics {
  organisations: number
  unique_users: number
  active_users: number
  returning_users: number
  return_rate: number
  completed_runs: number
  failed_runs: number
  failure_rate: number
  runs_per_user: number
  measured_impact_users: number
  feedback_submissions: number
}

export interface AdminReport {
  summary: SummaryMetrics
  usage_by_date: Record<string, unknown>[]
  usage_by_organisation: Record<string, unknown>[]
  usage_by_pilot_code: Record<string, unknown>[]
  usage_by_scenario: Record<string, unknown>[]
  usage_by_user_type: Record<string, unknown>[]
  usage_by_feature: Record<string, unknown>[]
  simulations_per_user: Record<string, unknown>[]
  assessment_changes: Record<string, unknown>[]
  feedback: Record<string, unknown>[]
  feedback_ratings: Record<string, unknown>
  bugs_and_requests: Record<string, unknown>[]
  environment: string
  is_demo_environment: boolean
  include_demo: boolean
  metric_definitions: Record<string, string>
  warnings: string[]
}
