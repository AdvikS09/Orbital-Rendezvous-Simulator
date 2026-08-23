/**
 * Constants shared by the participant flow pages.
 *
 * State is NOT held here — the anonymous session, the consent record and the
 * run count belong to `@/state/session`. What lives here is the small set of
 * labels and destinations the flow pages have in common, mirrored from the
 * Streamlit application so the two cannot drift.
 */

import type { Track, UserType } from '@/api/types'

/** Mirrors `pilot.ui.common.CONSENT_VERSION`. */
export const CONSENT_VERSION = '1.0'

/** The routes the participant flow navigates between, matching `@/App`. */
export const ROUTES = {
  landing: '/',
  consent: '/consent',
  signIn: '/sign-in',
  track: '/track',
  assessmentPre: '/assessment/pre',
  activities: '/activities',
  scenarios: '/scenarios',
  feedback: '/feedback',
  confirmation: '/done',
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
 * The track suggested for a user type, mirroring `flow.track_selection`.
 * A nudge, not a constraint: every track stays selectable whatever the answer.
 */
export function suggestedTrack(userType: UserType | null): Track {
  if (userType === 'student' || userType === 'teacher') return 'guided'
  if (userType === 'engineer_researcher') return 'professional'
  return 'free'
}

/**
 * Where a chosen track leads.
 *
 * The guided track is offered the anonymous pre-assessment first, exactly as in
 * `flow.track_selection`; the other two go straight to the scenario catalogue.
 */
export function trackDestination(track: Track, assessmentDone: boolean): string {
  if (track === 'guided') {
    return assessmentDone ? ROUTES.activities : ROUTES.assessmentPre
  }
  return ROUTES.scenarios
}
