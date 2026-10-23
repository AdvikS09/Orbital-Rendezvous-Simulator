import type { FormEvent } from 'react'
import { useCallback, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { ApiError, api } from '@/api/client'
import type { UserType } from '@/api/types'
import { NumberValue } from '@/components/NumberValue'
import { Button, Callout, Panel } from '@/components/ui'
import { useSession } from '@/state/session'
import { ROUTES, USER_TYPE_LABELS } from './participantFlow'
import './FeedbackPage.css'

interface ScaleFieldProps {
  id: string
  label: string
  help: string
  min: number
  max: number
  value: number
  onChange: (value: number) => void
  disabled?: boolean
}

function ScaleField({
  id,
  label,
  help,
  min,
  max,
  value,
  onChange,
  disabled = false,
}: ScaleFieldProps) {
  const helpId = `${id}-help`
  const valueId = `${id}-value`
  return (
    <div className="feedback-page__scale">
      <label htmlFor={id}>{label}</label>
      <div className="feedback-page__scale-row">
        <input
          type="range"
          id={id}
          min={min}
          max={max}
          step={1}
          value={value}
          disabled={disabled}
          aria-describedby={`${helpId} ${valueId}`}
          onChange={(event) => onChange(Number(event.target.value))}
        />
        <output id={valueId} htmlFor={id} className="feedback-page__scale-value">
          <NumberValue value={value} fmt="integer" />
        </output>
      </div>
      <p className="feedback-page__help" id={helpId}>
        {help}
      </p>
    </div>
  )
}

interface TextFieldProps {
  id: string
  label: string
  placeholder: string
  value: string
  onChange: (value: string) => void
  disabled?: boolean
}

function TextField({
  id,
  label,
  placeholder,
  value,
  onChange,
  disabled = false,
}: TextFieldProps) {
  return (
    <div className="feedback-page__text">
      <label htmlFor={id}>{label}</label>
      <textarea
        id={id}
        rows={3}
        value={value}
        placeholder={placeholder}
        disabled={disabled}
        onChange={(event) => onChange(event.target.value)}
      />
    </div>
  )
}

export function FeedbackPage() {
  const navigate = useNavigate()
  const { userType } = useSession()
  const resolvedUserType: UserType = userType ?? 'other'
  const isProfessional = resolvedUserType === 'engineer_researcher'

  const [usefulness, setUsefulness] = useState(3)
  const [usability, setUsability] = useState(3)
  const [clarity, setClarity] = useState(3)
  const [recommend, setRecommend] = useState(7)
  const [mostUseful, setMostUseful] = useState('')
  const [improvement, setImprovement] = useState('')
  const [bugReport, setBugReport] = useState('')
  const [featureRequest, setFeatureRequest] = useState('')

  const [submitting, setSubmitting] = useState(false)
  const [submitError, setSubmitError] = useState<string | null>(null)
  const [notRecorded, setNotRecorded] = useState(false)

  const handleSubmit = useCallback(
    async (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault()
      setSubmitting(true)
      setSubmitError(null)
      try {
        const acknowledgement = await api.submitFeedback({
          usefulness_rating: usefulness,
          usability_rating: usability,
          clarity_rating: clarity,
          would_recommend: recommend,
          most_useful: mostUseful,
          improvement,
          bug_report: bugReport,
          feature_request: featureRequest,
        })
        if (acknowledgement.recorded) {
          navigate(ROUTES.confirmation)
        } else {
          setNotRecorded(true)
          window.scrollTo({ top: 0 })
        }
      } catch (cause) {
        setSubmitError(
          cause instanceof ApiError
            ? cause.message
            : 'Your feedback could not be submitted.',
        )
      } finally {
        setSubmitting(false)
      }
    },
    [
      bugReport,
      clarity,
      featureRequest,
      improvement,
      mostUseful,
      navigate,
      recommend,
      usability,
      usefulness,
    ],
  )

  const usefulnessLabel = isProfessional
    ? 'How useful was this for real engineering or research work?'
    : 'How useful was this for understanding orbital mechanics?'

  const clarityLabel = isProfessional
    ? 'How clear were the assumptions and limitations?'
    : 'How clear were the explanations?'

  const improvementPlaceholder = isProfessional
    ? 'Missing physics, missing diagnostics, workflow friction.'
    : 'Anything confusing, missing, or in the way.'

  const featurePlaceholder = isProfessional
    ? 'Impulsive or optimal guidance, differential drag, navigation errors, eccentric reference orbits…'
    : 'Another activity, a different topic…'

  return (
    <div className="feedback-page">
      <header className="feedback-page__header">
        <h1>How was it?</h1>
        <p className="feedback-page__subtitle">
          Six questions, about two minutes. All of it is anonymous.
        </p>
        <p className="feedback-page__help">
          Answering as: {USER_TYPE_LABELS[resolvedUserType]}
        </p>
      </header>

      {notRecorded ? (
        <Callout
          severity="warning"
          title="Your feedback could not be saved"
          actions={
            <Button variant="primary" onClick={() => navigate(ROUTES.confirmation)}>
              Continue
            </Button>
          }
        >
          <p>
            The database is not reachable. Sorry — nothing you can do about that
            from here.
          </p>
        </Callout>
      ) : null}

      {submitError !== null ? (
        <Callout severity="error" title="Not submitted">
          <p>{submitError}</p>
        </Callout>
      ) : null}

      <form className="feedback-page__form" onSubmit={handleSubmit}>
        <Panel title="Ratings" headingLevel={2}>
          <ScaleField
            id="feedback-usefulness"
            label={usefulnessLabel}
            help="1 = not useful at all, 5 = very useful"
            min={1}
            max={5}
            value={usefulness}
            onChange={setUsefulness}
            disabled={submitting}
          />
          <ScaleField
            id="feedback-usability"
            label="How easy was the platform to use?"
            help="1 = very difficult, 5 = very easy"
            min={1}
            max={5}
            value={usability}
            onChange={setUsability}
            disabled={submitting}
          />
          <ScaleField
            id="feedback-clarity"
            label={clarityLabel}
            help="1 = not clear at all, 5 = very clear"
            min={1}
            max={5}
            value={clarity}
            onChange={setClarity}
            disabled={submitting}
          />
          <ScaleField
            id="feedback-recommend"
            label="How likely are you to recommend this to a colleague or classmate?"
            help="0 = not at all likely, 10 = extremely likely"
            min={0}
            max={10}
            value={recommend}
            onChange={setRecommend}
            disabled={submitting}
          />
        </Panel>

        <Panel title="In your own words" headingLevel={2}>
          <TextField
            id="feedback-most-useful"
            label="What was most useful?"
            placeholder="A scenario, an explanation, a plot — anything."
            value={mostUseful}
            onChange={setMostUseful}
            disabled={submitting}
          />
          <TextField
            id="feedback-improvement"
            label="What would you change?"
            placeholder={improvementPlaceholder}
            value={improvement}
            onChange={setImprovement}
            disabled={submitting}
          />
          <TextField
            id="feedback-bug"
            label="Did anything go wrong?"
            placeholder="Errors, wrong-looking results, things that would not load. Say which scenario if you can."
            value={bugReport}
            onChange={setBugReport}
            disabled={submitting}
          />
          <TextField
            id="feedback-feature"
            label="What should be added next?"
            placeholder={featurePlaceholder}
            value={featureRequest}
            onChange={setFeatureRequest}
            disabled={submitting}
          />
          <p className="feedback-page__help">
            Please do not put your name, email address or anything else personal
            in these boxes — they are stored exactly as written.
          </p>
        </Panel>

        <div className="feedback-page__actions">
          <Button
            type="submit"
            variant="primary"
            loading={submitting}
            loadingLabel="Submitting your feedback"
            disabled={submitting}
          >
            Submit feedback
          </Button>
          <Button
            onClick={() => navigate(ROUTES.scenarios)}
            disabled={submitting}
          >
            Back to the simulator
          </Button>
          <Button
            variant="ghost"
            onClick={() => navigate(ROUTES.confirmation)}
            disabled={submitting}
          >
            Skip feedback
          </Button>
        </div>
      </form>
    </div>
  )
}
