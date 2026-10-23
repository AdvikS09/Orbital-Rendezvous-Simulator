import { useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { Button, Callout, Disclosure, Panel } from '@/components/ui'
import { formatNumber } from '@/formatting/number'
import { useSession } from '@/state/session'
import { CONSENT_VERSION, ROUTES } from './participantFlow'
import './ConsentPage.css'

const RETENTION_DAYS = 730

const LOG_RAW_PARAMETERS: boolean = false

const SUBTITLE =
  'A short privacy notice, and your consent to anonymous usage tracking.'

const INTRO_LEAD =
  'This platform is being piloted, which means we want to know whether it is ' +
  'actually useful. To find that out we record '
const INTRO_EMPHASIS = 'anonymous, aggregate'
const INTRO_TAIL = ' usage information. Here is exactly what that means.'

const RECORDED =
  'an irreversible code derived from your pilot and participant codes, your ' +
  'user type, which scenarios you ran, whether they succeeded, how long they ' +
  'took, your assessment scores, and your feedback.'

const NEVER_RECORDED =
  'your name, email address, exact age, IP address, browser details, or the ' +
  'numerical parameters you type into simulations. Your participant code ' +
  'itself is not stored — only a one-way cryptographic hash of it, which ' +
  'cannot be turned back into the code you chose.'

const WHY_COLLECT =
  'To report how many people used the platform, which parts they used, whether ' +
  'simulations succeeded, whether learners’ understanding changed, and what ' +
  'people asked for next. Nothing is used for advertising, sold, or shared ' +
  'with third parties.'

const HOW_ANONYMITY =
  'Your participant code is combined with your pilot code and a secret key ' +
  'held by the server, and passed through a one-way cryptographic function. ' +
  'Only the result is stored. Typing the same code again on another day ' +
  'produces the same result, which is how return visits are counted — but the ' +
  'stored value cannot be reversed to recover your code.'

const PARAMETERS_STAY_YOURS =
  'The numbers you type into simulations are not recorded. The analytics store ' +
  'only broad bands — for example “separation between 100 m and 1 km” — so ' +
  'nobody can reconstruct what you were working on.'

const RAW_PARAMETER_NOTE =
  'raw parameter logging has been switched on by the operator of this ' +
  'instance, so the values you enter '
const RAW_PARAMETER_NOTE_TAIL = ' being recorded.'

const FREE_TEXT =
  'Anything you type into the feedback box is stored as you wrote it, so ' +
  'please do not put personal information there.'

const WITHDRAWING =
  'Because the data is anonymous, we cannot identify your records to remove ' +
  'them individually. If you want nothing recorded at all, please close this ' +
  'page instead of continuing — the simulator can also be run entirely offline ' +
  'from the source repository, with no analytics.'

const CONSENT_LABEL =
  'I have read the above and I consent to anonymous usage information being ' +
  'recorded.'

export function ConsentPage() {
  const navigate = useNavigate()
  const { giveConsent } = useSession()
  const [agreed, setAgreed] = useState(false)

  function accept() {
    giveConsent(CONSENT_VERSION)
    navigate(ROUTES.signIn)
  }

  return (
    <div className="consent-page">
      <header className="consent-page__header">
        <h1>Before you begin</h1>
        <p className="consent-page__subtitle">{SUBTITLE}</p>
      </header>

      <p className="consent-page__intro">
        {INTRO_LEAD}
        <strong>{INTRO_EMPHASIS}</strong>
        {INTRO_TAIL}
      </p>

      <Callout severity="info" title="What is and is not recorded">
        <p>
          <strong>What is recorded:</strong> {RECORDED}
        </p>
        <p>
          <strong>What is never recorded:</strong> {NEVER_RECORDED}
        </p>
      </Callout>

      <Disclosure title="The full detail">
        <div className="consent-page__detail">
          <p>
            <strong>Why we collect anything at all.</strong> {WHY_COLLECT}
          </p>
          <p>
            <strong>How anonymity works.</strong> {HOW_ANONYMITY}
          </p>
          <p>
            <strong>Your orbital parameters stay yours.</strong>{' '}
            {PARAMETERS_STAY_YOURS}
          </p>
          {LOG_RAW_PARAMETERS ? (
            <p>
              <strong>Note for this deployment:</strong> {RAW_PARAMETER_NOTE}
              <em>are</em>
              {RAW_PARAMETER_NOTE_TAIL}
            </p>
          ) : null}
          <p>
            <strong>Free text.</strong> {FREE_TEXT}
          </p>
          <p>
            <strong>Retention.</strong> Pilot data is kept for at most{' '}
            <span className="num">
              {formatNumber(RETENTION_DAYS, { notation: 'fixed', decimals: 0 })}
            </span>{' '}
            days and then deleted.
          </p>
          <p>
            <strong>Withdrawing.</strong> {WITHDRAWING}
          </p>
        </div>
      </Disclosure>

      <hr />

      <Panel title="Your consent">
        <div className="consent-page__agree">
          <input
            className="consent-page__checkbox"
            type="checkbox"
            id="consent-checkbox"
            checked={agreed}
            onChange={(event) => setAgreed(event.target.checked)}
            aria-describedby="consent-requirement"
          />
          <label htmlFor="consent-checkbox">{CONSENT_LABEL}</label>
        </div>

        <div className="consent-page__actions">
          <Button
            variant="primary"
            onClick={accept}
            disabled={!agreed}
            aria-describedby="consent-requirement"
          >
            Continue
          </Button>
          <Button variant="secondary" onClick={() => navigate(ROUTES.landing)}>
            Back
          </Button>
        </div>

        <p className="consent-page__requirement" id="consent-requirement">
          {agreed
            ? 'Consent recorded for this session. You can stop at any time by closing the page.'
            : 'You need to tick the box to continue. Until then the Continue button stays disabled.'}
        </p>
      </Panel>
    </div>
  )
}
