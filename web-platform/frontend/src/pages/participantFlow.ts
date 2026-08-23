/**
 * Participant-flow state and routing constants.
 *
 * The Streamlit application kept the journey in `st.session_state`. The React
 * application has no server-rendered session, so the same small set of facts
 * lives in `sessionStorage`: consent, user type, the summary of the anonymous
 * session, the chosen track and the run count.
 *
 * Nothing here identifies anyone. The participant code itself is never stored
 * -- only the opaque session token (held separately by `@/api/client`) and the
 * server's short participant reference.
 */

import type { Track, UserType } from '@/api/types'

/** Mirrors `pilot.ui.common.CONSENT_VERSION`. */
export const CONSENT_VERSION = '1.0'

/** Every route the participant flow navigates to, in one place. */
export const ROUTES = {
  landing: '/',
  consent: '/consent',
  signIn: '/sign-in',
  track: '/track',
  assessmentPre: '/assessment/pre',
  workspace: '/workspace',
  feedback: '/feedback',
  confirmation: '/confirmation',
} as const

/** Mirrors `pilot.analytics.USER_TYPES`. */
export const USER_TYPES: readonly UserType[] = [
  'student',
  'teacher',
  'engineer_researcher',
  'other',
]

/** Mirrors `pilot.analytics.USER_TYPE_LABELS`. */
export const USER_TYPE_LABELS: Record<UserType, string> = {
  student: 'Student',
  teacher: 'Teacher',
  engineer_researcher: 'Engineer / researcher',
  other: 'Other',
}

/** Mirrors `pilot.ui.common.TRACK_LABELS`. */
export const TRACK_LABELS: Record<Track, string> = {
  guided: 'Guided educational activities',
  professional: 'Professional / research exploration',
  free: 'Free simulation mode',
}

/**
 * The track suggested for a user type. A nudge, never a constraint: every
 * track stays selectable whatever the answer.
 */
export function suggestedTrack(userType: UserType | null): Track {
  if (userType === 'student' || userType === 'teacher') return 'guided'
  if (userType === 'engineer_researcher') return 'professional'
  return 'free'
}

export interface ParticipantFlowState {
  consentGiven: boolean
  consentVersion: string | null
  userType: UserType | null
  /** Short non-reversible reference the server returns for a participant. */
  participantRef: string | null
  organisationName: string | null
  pilotCode: string | null
  pilotCodeRegistered: boolean
  isReturnVisit: boolean
  /** False when the participant chose to carry on without a usage session. */
  tracked: boolean
  track: Track | null
  assessmentPreDone: boolean
  completedRuns: number
}

export const EMPTY_FLOW_STATE: ParticipantFlowState = {
  consentGiven: false,
  consentVersion: null,
  userType: null,
  participantRef: null,
  organisationName: null,
  pilotCode: null,
  pilotCodeRegistered: false,
  isReturnVisit: false,
  tracked: false,
  track: null,
  assessmentPreDone: false,
  completedRuns: 0,
}

const STORAGE_KEY = 'orbital.participant_flow'

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

/** Read the flow state, falling back to the empty state on any problem. */
export function readFlowState(): ParticipantFlowState {
  let raw: string | null = null
  try {
    raw = window.sessionStorage.getItem(STORAGE_KEY)
  } catch {
    // Private browsing can deny storage. The flow still works within one page
    // load; only cross-page memory is lost.
    return { ...EMPTY_FLOW_STATE }
  }
  if (!raw) return { ...EMPTY_FLOW_STATE }

  let parsed: unknown = null
  try {
    parsed = JSON.parse(raw)
  } catch {
    return { ...EMPTY_FLOW_STATE }
  }
  if (!isRecord(parsed)) return { ...EMPTY_FLOW_STATE }

  return { ...EMPTY_FLOW_STATE, ...(parsed as Partial<ParticipantFlowState>) }
}

/** Merge a patch into the stored flow state and return the result. */
export function writeFlowState(
  patch: Partial<ParticipantFlowState>,
): ParticipantFlowState {
  const next = { ...readFlowState(), ...patch }
  try {
    window.sessionStorage.setItem(STORAGE_KEY, JSON.stringify(next))
  } catch {
    /* ignore: the flow degrades to a single page load */
  }
  return next
}

/** Forget everything about this participant's journey. */
export function clearFlowState(): void {
  try {
    window.sessionStorage.removeItem(STORAGE_KEY)
  } catch {
    /* ignore */
  }
}
