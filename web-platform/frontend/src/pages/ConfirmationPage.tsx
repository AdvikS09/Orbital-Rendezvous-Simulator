import { useNavigate } from 'react-router-dom'

import { Button, Callout } from '@/components/ui'
import { formatNumber } from '@/formatting/number'
import { useSession } from '@/state/session'
import { ROUTES, trackDestination } from './participantFlow'
import './ConfirmationPage.css'

const FEEDBACK_FIRST =
  'This is a pilot, and the point of a pilot is to change what gets built. ' +
  'Your responses are pooled with everyone else’s and reviewed by the project ' +
  'maintainer. Feedback '
const FEEDBACK_FIRST_TAIL =
  ' influence later releases — which scenarios get added, which explanations ' +
  'get rewritten, what gets fixed first.'

const FEEDBACK_SECOND =
  'To be straight with you about it: this is a small open-source project, so ' +
  'no promise is made that any particular suggestion will be implemented, or ' +
  'implemented soon. Bug reports are the most likely to lead to a change, and ' +
  'quickly.'

const FEEDBACK_THIRD =
  'Nothing you submitted identifies you, so we cannot contact you about your ' +
  'feedback or tell you what became of it.'

const COMING_BACK_LEAD = 'Use the '
const COMING_BACK_EMPHASIS = 'same participant code'
const COMING_BACK_TAIL =
  ' next time and your return visit will be recognised. That matters for the ' +
  'pilot: repeat use across separate days is one of the things being measured.'

const USING_IT_YOURSELF =
  'The whole thing is open source under the MIT licence. You can run it ' +
  'locally with no analytics at all, read the physics, and reuse it in your ' +
  'own teaching or research.'

export function ConfirmationPage() {
  const navigate = useNavigate()
  const { completedRuns, track, endSession } = useSession()

  function finish() {
    endSession()
    navigate(ROUTES.landing)
  }

  return (
    <div className="confirmation-page">
      <header className="confirmation-page__header">
        <h1>Thank you</h1>
        <p className="confirmation-page__subtitle">Your session is complete.</p>
      </header>

      <Callout severity="success">
        <p>
          You completed{' '}
          <span className="num">
            {formatNumber(completedRuns, { notation: 'fixed', decimals: 0 })}
          </span>{' '}
          {completedRuns === 1 ? 'simulation' : 'simulations'} in this session.
        </p>
      </Callout>

      <section
        className="confirmation-page__section"
        aria-labelledby="confirmation-feedback"
      >
        <h2 id="confirmation-feedback">What happens to your feedback</h2>
        <p>
          {FEEDBACK_FIRST}
          <strong>may</strong>
          {FEEDBACK_FIRST_TAIL}
        </p>
        <p>{FEEDBACK_SECOND}</p>
        <p>{FEEDBACK_THIRD}</p>
      </section>

      <section
        className="confirmation-page__section"
        aria-labelledby="confirmation-coming-back"
      >
        <h2 id="confirmation-coming-back">Coming back</h2>
        <p>
          {COMING_BACK_LEAD}
          <strong>{COMING_BACK_EMPHASIS}</strong>
          {COMING_BACK_TAIL}
        </p>
      </section>

      <section
        className="confirmation-page__section"
        aria-labelledby="confirmation-using-it"
      >
        <h2 id="confirmation-using-it">Using this yourself</h2>
        <p>{USING_IT_YOURSELF}</p>
      </section>

      <hr />

      <div className="confirmation-page__actions">
        <Button
          type="button"
          variant="primary"
          onClick={() => navigate(trackDestination(track ?? 'free', true))}
        >
          Run more simulations
        </Button>
        <Button type="button" variant="secondary" onClick={finish}>
          Finish and start a new session
        </Button>
      </div>

      <p className="confirmation-page__note">
        Finishing clears this browser’s copy of your session. Your anonymous
        records stay in the pilot, because they cannot be traced back to you.
      </p>
    </div>
  )
}
