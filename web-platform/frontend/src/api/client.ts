import type {
  Acknowledgement,
  Activity,
  AdminReport,
  ApiErrorBody,
  AssessmentInstrument,
  AssessmentResult,
  AssessmentSubmission,
  FeedbackRequest,
  HealthResponse,
  ParamValues,
  PreflightResult,
  ScenarioDetail,
  ScenarioSummary,
  SessionRequest,
  SessionResponse,
  SimulationResult,
  Track,
} from './types'

export const API_BASE = '/api/v1'

const SESSION_KEY = 'orbital.session_token'

export class ApiError extends Error {
  readonly status: number
  readonly body: ApiErrorBody | null

  constructor(status: number, body: ApiErrorBody | null, fallback: string) {
    super(body?.message ?? fallback)
    this.name = 'ApiError'
    this.status = status
    this.body = body
  }

  get fieldErrors(): Record<string, string> {
    const out: Record<string, string> = {}
    for (const detail of this.body?.detail ?? []) {
      if (detail.field) out[detail.field] = detail.message
    }
    return out
  }

  get generalErrors(): string[] {
    const details = this.body?.detail ?? []
    const unattributed = details.filter((d) => !d.field).map((d) => d.message)
    return unattributed.length > 0 ? unattributed : [this.message]
  }

  get rejected(): boolean {
    return this.body?.rejected ?? false
  }

  get isRateLimited(): boolean {
    return this.status === 429 || this.body?.error_class === 'RateLimited'
  }
}

export function getSessionToken(): string | null {
  try {
    return window.sessionStorage.getItem(SESSION_KEY)
  } catch {
    return null
  }
}

export function setSessionToken(token: string | null): void {
  try {
    if (token === null) window.sessionStorage.removeItem(SESSION_KEY)
    else window.sessionStorage.setItem(SESSION_KEY, token)
  } catch {}
}

interface RequestOptions {
  method?: 'GET' | 'POST'
  body?: unknown
  signal?: AbortSignal
  auth?: { username: string; password: string }
  raw?: boolean
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const headers: Record<string, string> = { Accept: 'application/json' }

  const token = getSessionToken()
  if (token) headers['X-Session-Token'] = token

  if (options.body !== undefined) headers['Content-Type'] = 'application/json'

  if (options.auth) {
    headers['Authorization'] = `Basic ${btoa(
      `${options.auth.username}:${options.auth.password}`,
    )}`
  }

  let response: Response
  try {
    response = await fetch(`${API_BASE}${path}`, {
      method: options.method ?? 'GET',
      headers,
      body: options.body === undefined ? undefined : JSON.stringify(options.body),
      signal: options.signal,
    })
  } catch (cause) {
    if ((cause as Error)?.name === 'AbortError') throw cause
    throw new ApiError(
      0,
      null,
      'Could not reach the simulation server. Check that the backend is running.',
    )
  }

  if (options.raw) {
    if (!response.ok) {
      throw new ApiError(response.status, null, `Request failed (${response.status}).`)
    }
    return response as unknown as T
  }

  const text = await response.text()
  let parsed: unknown = null
  if (text) {
    try {
      parsed = JSON.parse(text)
    } catch {
      parsed = null
    }
  }

  if (!response.ok) {
    const body =
      parsed && typeof parsed === 'object' && 'error_class' in (parsed as object)
        ? (parsed as ApiErrorBody)
        : null
    throw new ApiError(
      response.status,
      body,
      `The request failed (HTTP ${response.status}).`,
    )
  }

  if (
    parsed &&
    typeof parsed === 'object' &&
    (parsed as { status?: string }).status === 'error'
  ) {
    throw new ApiError(response.status, parsed as ApiErrorBody, 'The request failed.')
  }

  return parsed as T
}

export const api = {
  health: (signal?: AbortSignal) => request<HealthResponse>('/health', { signal }),

  listScenarios: (track?: Track, signal?: AbortSignal) =>
    request<ScenarioSummary[]>(
      track ? `/scenarios?track=${encodeURIComponent(track)}` : '/scenarios',
      { signal },
    ),

  categories: (signal?: AbortSignal) =>
    request<Record<string, string>>('/scenarios/categories', { signal }),

  getScenario: (id: string, signal?: AbortSignal) =>
    request<ScenarioDetail>(`/scenarios/${encodeURIComponent(id)}`, { signal }),

  preflight: (id: string, params: ParamValues, signal?: AbortSignal) =>
    request<PreflightResult>(`/scenarios/${encodeURIComponent(id)}/preflight`, {
      method: 'POST',
      body: { params },
      signal,
    }),

  runSimulation: (
    id: string,
    params: ParamValues,
    extra: { activity_id?: string; step_id?: string } = {},
    signal?: AbortSignal,
  ) =>
    request<SimulationResult>(`/simulations/${encodeURIComponent(id)}`, {
      method: 'POST',
      body: { params, ...extra },
      signal,
    }),

  startSession: (payload: SessionRequest) =>
    request<SessionResponse>('/sessions', { method: 'POST', body: payload }),

  selectTrack: (track: Track) =>
    request<Acknowledgement>('/sessions/track', {
      method: 'POST',
      body: { track },
    }),

  submitFeedback: (payload: FeedbackRequest) =>
    request<Acknowledgement>('/feedback', { method: 'POST', body: payload }),

  getAssessment: (instrumentId = 'orbital_basics', signal?: AbortSignal) =>
    request<AssessmentInstrument>(
      `/assessments/${encodeURIComponent(instrumentId)}`,
      { signal },
    ),

  submitAssessment: (instrumentId: string, payload: AssessmentSubmission) =>
    request<AssessmentResult>(
      `/assessments/${encodeURIComponent(instrumentId)}/submit`,
      { method: 'POST', body: payload },
    ),

  listActivities: (signal?: AbortSignal) =>
    request<Activity[]>('/activities', { signal }),

  getActivity: (id: string, signal?: AbortSignal) =>
    request<Activity>(`/activities/${encodeURIComponent(id)}`, { signal }),

  completeActivity: (id: string) =>
    request<Acknowledgement>(`/activities/${encodeURIComponent(id)}/complete`, {
      method: 'POST',
    }),

  exportConfig: (id: string, params: ParamValues) =>
    request<Response>(`/exports/${encodeURIComponent(id)}/config`, {
      method: 'POST',
      body: { params },
      raw: true,
    }),

  exportNotes: (id: string, params: ParamValues, warnings: string[]) =>
    request<Response>(`/exports/${encodeURIComponent(id)}/notes`, {
      method: 'POST',
      body: { params, warnings },
      raw: true,
    }),

  adminReport: (
    auth: { username: string; password: string },
    query: Record<string, string | number | boolean | undefined> = {},
  ) => {
    const search = new URLSearchParams()
    for (const [key, value] of Object.entries(query)) {
      if (value !== undefined && value !== null && value !== '') {
        search.set(key, String(value))
      }
    }
    const suffix = search.toString() ? `?${search}` : ''
    return request<AdminReport>(`/admin/report${suffix}`, { auth })
  },

  adminOrganisations: (
    auth: { username: string; password: string },
    includeDemo = false,
  ) =>
    request<Record<string, unknown>[]>(
      `/admin/organisations?include_demo=${includeDemo ? 'true' : 'false'}`,
      { auth },
    ),

  adminExport: (
    auth: { username: string; password: string },
    query: Record<string, string | number | boolean | undefined> = {},
  ) => {
    const search = new URLSearchParams()
    for (const [key, value] of Object.entries(query)) {
      if (value !== undefined && value !== null && value !== '') {
        search.set(key, String(value))
      }
    }
    const suffix = search.toString() ? `?${search}` : ''
    return request<Response>(`/admin/export${suffix}`, { auth, raw: true })
  },
}

export async function downloadResponse(
  response: Response,
  fallbackName: string,
): Promise<void> {
  const disposition = response.headers.get('Content-Disposition') ?? ''
  const match = /filename="?([^"]+)"?/.exec(disposition)
  const filename = match?.[1] ?? fallbackName

  const blob = await response.blob()
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
  URL.revokeObjectURL(url)
}
