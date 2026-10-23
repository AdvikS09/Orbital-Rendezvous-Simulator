import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react'

import { getSessionToken, setSessionToken } from '@/api/client'
import type { SessionResponse, Track, UserType } from '@/api/types'

const CONSENT_KEY = 'orbital.consent_version'
const SESSION_META_KEY = 'orbital.session_meta'

export interface SessionState {
  token: string | null
  participantRef: string | null
  userType: UserType | null
  organisationName: string | null
  pilotCode: string | null
  pilotCodeRegistered: boolean
  isReturnVisit: boolean
  track: Track | null
  consentVersion: string | null
  completedRuns: number
}

interface SessionContextValue extends SessionState {
  hasConsented: boolean
  isSignedIn: boolean
  giveConsent: (version: string) => void
  beginSession: (response: SessionResponse) => void
  setTrack: (track: Track) => void
  recordCompletedRun: () => void
  endSession: () => void
}

const EMPTY: SessionState = {
  token: null,
  participantRef: null,
  userType: null,
  organisationName: null,
  pilotCode: null,
  pilotCodeRegistered: false,
  isReturnVisit: false,
  track: null,
  consentVersion: null,
  completedRuns: 0,
}

const SessionContext = createContext<SessionContextValue | null>(null)

function readStored<T>(key: string): T | null {
  try {
    const raw = window.sessionStorage.getItem(key)
    return raw ? (JSON.parse(raw) as T) : null
  } catch {
    return null
  }
}

function writeStored(key: string, value: unknown): void {
  try {
    window.sessionStorage.setItem(key, JSON.stringify(value))
  } catch {}
}

export function SessionProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<SessionState>(() => {
    const token = getSessionToken()
    const meta = readStored<Partial<SessionState>>(SESSION_META_KEY)
    const consent = readStored<string>(CONSENT_KEY)
    return {
      ...EMPTY,
      ...(meta ?? {}),
      token,
      consentVersion: consent,
      completedRuns: 0,
    }
  })

  useEffect(() => {
    if (state.token) {
      const { completedRuns: _ignored, ...meta } = state
      writeStored(SESSION_META_KEY, meta)
    }
  }, [state])

  const giveConsent = useCallback((version: string) => {
    writeStored(CONSENT_KEY, version)
    setState((previous) => ({ ...previous, consentVersion: version }))
  }, [])

  const beginSession = useCallback((response: SessionResponse) => {
    setSessionToken(response.session_token)
    setState((previous) => ({
      ...previous,
      token: response.session_token,
      participantRef: response.participant_ref,
      userType: response.user_type,
      organisationName: response.organisation_name,
      pilotCode: response.pilot_code,
      pilotCodeRegistered: response.pilot_code_registered,
      isReturnVisit: response.is_return_visit,
      consentVersion: response.consent_version,
    }))
  }, [])

  const setTrack = useCallback((track: Track) => {
    setState((previous) => ({ ...previous, track }))
  }, [])

  const recordCompletedRun = useCallback(() => {
    setState((previous) => ({ ...previous, completedRuns: previous.completedRuns + 1 }))
  }, [])

  const endSession = useCallback(() => {
    setSessionToken(null)
    try {
      window.sessionStorage.removeItem(SESSION_META_KEY)
      window.sessionStorage.removeItem(CONSENT_KEY)
    } catch {}
    setState({ ...EMPTY })
  }, [])

  const value = useMemo<SessionContextValue>(
    () => ({
      ...state,
      hasConsented: state.consentVersion !== null,
      isSignedIn: state.token !== null,
      giveConsent,
      beginSession,
      setTrack,
      recordCompletedRun,
      endSession,
    }),
    [state, giveConsent, beginSession, setTrack, recordCompletedRun, endSession],
  )

  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>
}

export function useSession(): SessionContextValue {
  const context = useContext(SessionContext)
  if (context === null) {
    throw new Error('useSession must be used inside a SessionProvider')
  }
  return context
}
