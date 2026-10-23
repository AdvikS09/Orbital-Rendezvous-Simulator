import type { Track, UserType } from '@/api/types'

export const CONSENT_VERSION = '1.0'

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

export const USER_TYPES: readonly UserType[] = [
  'student',
  'teacher',
  'engineer_researcher',
  'other',
]

export const USER_TYPE_LABELS: Record<UserType, string> = {
  student: 'Student',
  teacher: 'Teacher',
  engineer_researcher: 'Engineer / researcher',
  other: 'Other',
}

export const TRACK_LABELS: Record<Track, string> = {
  guided: 'Guided educational activities',
  professional: 'Professional / research exploration',
  free: 'Free simulation mode',
}

export function suggestedTrack(userType: UserType | null): Track {
  if (userType === 'student' || userType === 'teacher') return 'guided'
  if (userType === 'engineer_researcher') return 'professional'
  return 'free'
}

export function trackDestination(track: Track, assessmentDone: boolean): string {
  if (track === 'guided') {
    return assessmentDone ? ROUTES.activities : ROUTES.assessmentPre
  }
  return ROUTES.scenarios
}
