import { useState } from 'react'
import type { FormEvent } from 'react'
import { useNavigate } from 'react-router-dom'

import { ApiError, api } from '@/api/client'
import type { SessionResponse, UserType } from '@/api/types'
import { Button, Callout, Panel } from '@/components/ui'
import { formatNumber } from '@/formatting/number'
import { useSession } from '@/state/session'
import {
  CONSENT_VERSION,
  ROUTES,
  USER_TYPES,
  USER_TYPE_LABELS,
} from './participantFlow'
import './SignInPage.css'

export const MIN_CODE_LENGTH = 3
export const MAX_CODE_LENGTH = 40

const EMAIL_MARKER = '@'

const CODE_PATTERN = /^[\p{L}\p{N}_-]+$/u

const COUNT = { notation: 'fixed', decimals: 0 } as const

export interface CodeCheck {
  code: string
  error: string | null
}

export function validateParticipantCode(input: string): CodeCheck {
  const normalised = input.trim()
  if (!normalised) {
    return { code: '', error: 'Enter an anonymous participant code.' }
  }
  if (normalised.length < MIN_CODE_LENGTH) {
    return {
      code: '',
      error:
        'The participant code must be at least ' +
        `${formatNumber(MIN_CODE_LENGTH, COUNT)} characters.`,
    }
  }
  if (normalised.length > MAX_CODE_LENGTH) {
    return {
      code: '',
      error:
        'The participant code must be at most ' +
        `${formatNumber(MAX_CODE_LENGTH, COUNT)} characters.`,
    }
  }
  if (normalised.includes(EMAIL_MARKER)) {
    return {
      code: '',
      error:
        'That looks like an email address. Please use an anonymous code that ' +
        'is not personal information.',
    }
  }
  if (!CODE_PATTERN.test(normalised)) {
    return {
      code: '',
      error:
        'Use letters, digits, hyphens and underscores only, so the code cannot ' +
        'contain personal information.',
    }
  }
  return { code: normalised.toUpperCase(), error: null }
}

export function validatePilotCode(input: string): CodeCheck {
  const normalised = input.trim()
  if (!normalised) {
    return { code: '', error: 'Enter the organisation or pilot code you were given.' }
  }
  if (normalised.length > MAX_CODE_LENGTH) {
    return {
      code: '',
      error:
        'The pilot code must be at most ' +
        `${formatNumber(MAX_CODE_LENGTH, COUNT)} characters.`,
    }
  }
  if (!CODE_PATTERN.test(normalised)) {
    return { code: '', error: 'Use letters, digits, hyphens and underscores only.' }
  }
  return { code: normalised.toUpperCase(), error: null }
}

const PILOT_HELP =
  'The code your teacher or pilot coordinator gave you. If you are here on ' +
  'your own, enter SELF.'

const PARTICIPANT_HELP =
  'Make one up and write it down — you will need the same code to be ' +
  'recognised if you come back. Do NOT use your name or email.'

const CODE_NOTE =
  'Pick something memorable but not personal. Your code is never stored — only ' +
  'a one-way hash of it.'

const DATABASE_UNAVAILABLE =
  'Sign-in could not be completed because the usage database is not reachable. ' +
  'You can still use the simulator — press Continue without tracking below — ' +
  'but your session will not be counted in the pilot.'

const UNREGISTERED_TAIL =
  ' is not registered as a pilot on this deployment, so your session is ' +
  'grouped under “Unregistered pilot code”. That is fine — everything works ' +
  'normally.'

const RETURN_VISIT = 'Welcome back. Your earlier work is recognised.'

const UNATTRIBUTED_REJECTION =
  'The server did not accept those codes. Check the pilot code with your ' +
  'coordinator, and make sure your participant code contains only letters, ' +
  'digits, hyphens and underscores.'

interface FieldErrors {
  pilot_code: string | null
  participant_code: string | null
}

const NO_FIELD_ERRORS: FieldErrors = { pilot_code: null, participant_code: null }

export function SignInPage() {
  const navigate = useNavigate()
  const { hasConsented, beginSession } = useSession()

  const [pilotCode, setPilotCode] = useState('')
  const [participantCode, setParticipantCode] = useState('')
  const [userType, setUserType] = useState<UserType>('student')

  const [fieldErrors, setFieldErrors] = useState<FieldErrors>(NO_FIELD_ERRORS)
  const [generalErrors, setGeneralErrors] = useState<string[]>([])
  const [databaseUnavailable, setDatabaseUnavailable] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [startedSession, setStartedSession] = useState<SessionResponse | null>(null)

  function continueWithoutTracking() {
    navigate(ROUTES.track)
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setGeneralErrors([])
    setDatabaseUnavailable(false)

    const pilot = validatePilotCode(pilotCode)
    const participant = validateParticipantCode(participantCode)
    setFieldErrors({
      pilot_code: pilot.error,
      participant_code: participant.error,
    })
    if (pilot.error || participant.error) return

    setSubmitting(true)
    try {
      const response = await api.startSession({
        pilot_code: pilot.code,
        participant_code: participant.code,
        user_type: userType,
        consent_given: true,
        consent_version: CONSENT_VERSION,
      })

      beginSession(response)

      if (response.is_return_visit || !response.pilot_code_registered) {
        setStartedSession(response)
      } else {
        navigate(ROUTES.track)
      }
    } catch (error) {
      if (error instanceof ApiError) {
        if (error.status === 503 || error.status === 0) {
          setDatabaseUnavailable(true)
        } else {
          const mapped = error.fieldErrors
          setFieldErrors({
            pilot_code: mapped['pilot_code'] ?? null,
            participant_code: mapped['participant_code'] ?? null,
          })
          const unattributed = error.generalErrors.filter(
            (message) =>
              message !== mapped['pilot_code'] &&
              message !== mapped['participant_code'],
          )
          setGeneralErrors(
            error.body === null && error.status === 422
              ? [UNATTRIBUTED_REJECTION]
              : unattributed,
          )
        }
      } else {
        setGeneralErrors([
          'Sign-in could not be completed. Please try again.',
        ])
      }
    } finally {
      setSubmitting(false)
    }
  }

  if (!hasConsented) {
    return (
      <div className="sign-in-page">
        <header className="sign-in-page__header">
          <h1>Sign in anonymously</h1>
        </header>
        <Callout severity="warning">
          <p>Please read the privacy notice first.</p>
        </Callout>
        <div className="sign-in-page__actions">
          <Button variant="primary" onClick={() => navigate(ROUTES.consent)}>
            Go to the privacy notice
          </Button>
        </div>
      </div>
    )
  }

  if (startedSession !== null) {
    return (
      <div className="sign-in-page">
        <header className="sign-in-page__header">
          <h1>Sign in anonymously</h1>
          <p className="sign-in-page__subtitle">
            Your anonymous session has started.
          </p>
        </header>

        {startedSession.is_return_visit ? (
          <Callout severity="success">
            <p>{RETURN_VISIT}</p>
          </Callout>
        ) : null}

        {startedSession.pilot_code_registered ? null : (
          <Callout severity="info">
            <p>
              The code <strong>{startedSession.pilot_code}</strong>
              {UNREGISTERED_TAIL}
            </p>
          </Callout>
        )}

        <div className="sign-in-page__actions">
          <Button variant="primary" onClick={() => navigate(ROUTES.track)}>
            Continue
          </Button>
        </div>
      </div>
    )
  }

  return (
    <div className="sign-in-page">
      <header className="sign-in-page__header">
        <h1>Sign in anonymously</h1>
        <p className="sign-in-page__subtitle">
          Three short answers. None of them identify you.
        </p>
      </header>

      {generalErrors.length > 0 ? (
        <Callout severity="error">
          <ul className="sign-in-page__error-list">
            {generalErrors.map((message) => (
              <li key={message}>{message}</li>
            ))}
          </ul>
        </Callout>
      ) : null}

      {databaseUnavailable ? (
        <Callout severity="error">
          <p>{DATABASE_UNAVAILABLE}</p>
          <div className="sign-in-page__actions">
            <Button
              type="button"
              variant="secondary"
              onClick={continueWithoutTracking}
            >
              Continue without tracking
            </Button>
          </div>
        </Callout>
      ) : null}

      <Panel title="Your details">
        <form className="sign-in-page__form" onSubmit={handleSubmit} noValidate>
          <div className="sign-in-page__field">
            <label htmlFor="pilot-code">Organisation or pilot code</label>
            <input
              className="sign-in-page__input"
              id="pilot-code"
              name="pilot_code"
              type="text"
              value={pilotCode}
              onChange={(event) => setPilotCode(event.target.value)}
              placeholder="e.g. SCHOOL-A or SELF"
              autoComplete="off"
              autoCapitalize="characters"
              spellCheck={false}
              aria-invalid={fieldErrors.pilot_code !== null}
              aria-describedby={
                fieldErrors.pilot_code === null
                  ? 'pilot-code-help'
                  : 'pilot-code-help pilot-code-error'
              }
            />
            <p className="sign-in-page__help" id="pilot-code-help">
              {PILOT_HELP}
            </p>
            {fieldErrors.pilot_code === null ? null : (
              <p
                className="sign-in-page__error"
                id="pilot-code-error"
                role="alert"
              >
                {fieldErrors.pilot_code}
              </p>
            )}
          </div>

          <div className="sign-in-page__field">
            <label htmlFor="participant-code">Anonymous participant code</label>
            <input
              className="sign-in-page__input"
              id="participant-code"
              name="participant_code"
              type="text"
              value={participantCode}
              onChange={(event) => setParticipantCode(event.target.value)}
              placeholder="e.g. BLUE-FALCON-42"
              autoComplete="off"
              autoCapitalize="characters"
              spellCheck={false}
              aria-invalid={fieldErrors.participant_code !== null}
              aria-describedby={
                fieldErrors.participant_code === null
                  ? 'participant-code-help participant-code-note'
                  : 'participant-code-help participant-code-note participant-code-error'
              }
            />
            <p className="sign-in-page__help" id="participant-code-help">
              {PARTICIPANT_HELP}
            </p>
            <p className="sign-in-page__help" id="participant-code-note">
              {CODE_NOTE}
            </p>
            {fieldErrors.participant_code === null ? null : (
              <p
                className="sign-in-page__error"
                id="participant-code-error"
                role="alert"
              >
                {fieldErrors.participant_code}
              </p>
            )}
          </div>

          <fieldset className="sign-in-page__fieldset">
            <legend>Which best describes you?</legend>
            <div className="sign-in-page__radios">
              {USER_TYPES.map((value) => (
                <div className="sign-in-page__radio" key={value}>
                  <input
                    type="radio"
                    id={`user-type-${value}`}
                    name="user_type"
                    value={value}
                    checked={userType === value}
                    onChange={() => setUserType(value)}
                  />
                  <label htmlFor={`user-type-${value}`}>
                    {USER_TYPE_LABELS[value]}
                  </label>
                </div>
              ))}
            </div>
          </fieldset>

          <div className="sign-in-page__actions">
            <Button
              type="submit"
              variant="primary"
              loading={submitting}
              loadingLabel="Signing in"
              aria-describedby="sign-in-note"
            >
              Continue
            </Button>
            <Button
              type="button"
              variant="secondary"
              onClick={() => navigate(ROUTES.consent)}
            >
              Back
            </Button>
          </div>

          <p className="sign-in-page__note" id="sign-in-note">
            Nothing is recorded until you press Continue.
          </p>
        </form>
      </Panel>
    </div>
  )
}
