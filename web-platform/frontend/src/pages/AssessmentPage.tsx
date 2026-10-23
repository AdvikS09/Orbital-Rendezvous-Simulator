import type { FormEvent } from 'react'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { ApiError, api } from '@/api/client'
import type { AssessmentInstrument, AssessmentResult } from '@/api/types'
import { formatNumber, rawValue } from '@/formatting/number'
import { NumberValue } from '@/components/NumberValue'
import { Button, Callout, Disclosure, Panel, Spinner } from '@/components/ui'
import { ROUTES } from './participantFlow'
import './AssessmentPage.css'

const PRE_SCORE_KEY = 'orbital.assessment_pre_score'

function readPreScore(): number | null {
  try {
    const raw = window.sessionStorage.getItem(PRE_SCORE_KEY)
    if (raw === null) return null
    const parsed = Number(raw)
    return Number.isFinite(parsed) ? parsed : null
  } catch {
    return null
  }
}

function writePreScore(score: number): void {
  try {
    window.sessionStorage.setItem(PRE_SCORE_KEY, rawValue(score))
  } catch {}
}

const DEFAULT_CONFIDENCE = 3

export function AssessmentPage() {
  const { phase } = useParams<{ phase: string }>()
  const navigate = useNavigate()
  const resolvedPhase: 'pre' | 'post' = phase === 'post' ? 'post' : 'pre'
  const isPre = resolvedPhase === 'pre'

  const [instrument, setInstrument] = useState<AssessmentInstrument | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [responses, setResponses] = useState<Record<string, number>>({})
  const [confidence, setConfidence] = useState(DEFAULT_CONFIDENCE)
  const [submitting, setSubmitting] = useState(false)
  const [submitError, setSubmitError] = useState<string | null>(null)
  const [result, setResult] = useState<AssessmentResult | null>(null)

  const previousScore = useMemo(() => (isPre ? null : readPreScore()), [isPre])

  useEffect(() => {
    const controller = new AbortController()
    setInstrument(null)
    setLoadError(null)
    setResponses({})
    setResult(null)
    setConfidence(DEFAULT_CONFIDENCE)

    api
      .getAssessment('orbital_basics', controller.signal)
      .then(setInstrument)
      .catch((cause: unknown) => {
        if (controller.signal.aborted) return
        setLoadError(
          cause instanceof ApiError
            ? cause.message
            : 'The questions could not be loaded.',
        )
      })

    return () => controller.abort()
  }, [resolvedPhase])

  const answered = Object.keys(responses).length
  const unanswered = (instrument?.items.length ?? 0) - answered

  const handleSubmit = useCallback(
    async (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault()
      if (instrument === null) return
      setSubmitting(true)
      setSubmitError(null)
      try {
        const outcome = await api.submitAssessment(instrument.id, {
          instrument_id: instrument.id,
          phase: resolvedPhase,
          responses,
          confidence_rating: confidence,
        })
        if (isPre) writePreScore(outcome.score)
        setResult(outcome)
        window.scrollTo({ top: 0 })
      } catch (cause) {
        setSubmitError(
          cause instanceof ApiError
            ? cause.message
            : 'Your answers could not be submitted.',
        )
      } finally {
        setSubmitting(false)
      }
    },
    [confidence, instrument, isPre, resolvedPhase, responses],
  )

  const handleSkip = useCallback(() => {
    navigate(isPre ? ROUTES.scenarios : ROUTES.feedback)
  }, [isPre, navigate])

  if (loadError !== null) {
    return (
      <div className="assessment-page assessment-page--message">
        <Callout severity="error" title="Questions unavailable">
          <p>{loadError}</p>
        </Callout>
        <Button onClick={handleSkip}>Continue without them</Button>
      </div>
    )
  }

  if (instrument === null) {
    return (
      <div className="assessment-page assessment-page--message">
        <p className="assessment-page__loading">
          <Spinner label="Loading the questions" />
          <span aria-hidden="true">Loading the questions…</span>
        </p>
      </div>
    )
  }

  if (result !== null) {
    if (isPre) {
      return (
        <div className="assessment-page assessment-page--message">
          <h1>Recorded</h1>
          <Callout severity="success" title="Your answers have been recorded">
            <p>
              They are stored against an anonymous reference, not against you.
              You will answer the same questions at the end.
            </p>
          </Callout>
          <Button variant="primary" onClick={() => navigate(ROUTES.scenarios)}>
            Continue to the simulator
          </Button>
        </div>
      )
    }

    const before = result.previous_score ?? previousScore
    const after = result.score
    const change = result.change ?? (before === null ? null : after - before)

    return (
      <div className="assessment-page">
        <header className="assessment-page__header">
          <h1>How you did</h1>
          <p className="assessment-page__subtitle">
            Your answers have been recorded anonymously.
          </p>
        </header>

        <Panel title="Score" headingLevel={2}>
          <dl className="assessment-page__scores">
            <div className="assessment-page__score">
              <dt>Before</dt>
              <dd>
                {before === null ? (
                  <span className="assessment-page__missing">not recorded</span>
                ) : (
                  <>
                    <NumberValue value={before} fmt="integer" /> of{' '}
                    <NumberValue value={result.max_score} fmt="integer" />
                  </>
                )}
              </dd>
            </div>
            <div className="assessment-page__score">
              <dt>After</dt>
              <dd>
                <NumberValue value={after} fmt="integer" /> of{' '}
                <NumberValue value={result.max_score} fmt="integer" />
              </dd>
            </div>
            <div className="assessment-page__score">
              <dt>Change</dt>
              <dd>
                {change === null ? (
                  <span className="assessment-page__missing">not available</span>
                ) : (
                  <span className="num">
                    {formatNumber(change, {
                      notation: 'fixed',
                      decimals: 0,
                      signed: true,
                    })}
                  </span>
                )}
              </dd>
            </div>
          </dl>
          {result.caveat ? (
            <p className="assessment-page__caption">{result.caveat}</p>
          ) : null}
        </Panel>

        <Disclosure
          title="The answers, with explanations"
          defaultOpen
          icon="book"
          count={result.items.length}
        >
          <ol className="assessment-page__answers">
            {result.items.map((item) => {
              const correct =
                item.correct_index === null
                  ? null
                  : (item.options[item.correct_index] ?? null)
              return (
                <li key={item.id}>
                  <p className="assessment-page__question">{item.question}</p>
                  {correct !== null ? (
                    <p className="assessment-page__correct">
                      Correct answer: <em>{correct}</em>
                    </p>
                  ) : null}
                  {item.explanation ? (
                    <p className="assessment-page__caption">{item.explanation}</p>
                  ) : null}
                </li>
              )
            })}
          </ol>
        </Disclosure>

        <div className="assessment-page__actions">
          <Button variant="primary" onClick={() => navigate(ROUTES.feedback)}>
            Continue to feedback
          </Button>
        </div>
      </div>
    )
  }

  const confidenceLabel =
    instrument.confidence_labels[confidence - 1] ??
    formatNumber(confidence, { notation: 'fixed', decimals: 0 })

  return (
    <div className="assessment-page">
      <header className="assessment-page__header">
        <h1>
          {isPre ? 'A few questions before you start' : 'The same questions again'}
        </h1>
        <p className="assessment-page__subtitle">
          <NumberValue value={instrument.items.length} fmt="integer" />{' '}
          multiple-choice questions · anonymous · no time limit
        </p>
        {isPre ? (
          <p className="assessment-page__intro">
            These measure what the activities teach, not you. Nobody sees your
            individual answers — a teacher only ever sees pooled results for a
            whole group. Answer what you think now; you will see the same
            questions at the end so the change can be measured.
          </p>
        ) : (
          <>
            <p className="assessment-page__intro">
              You saw these at the start. Answering them again is what makes the
              change measurable. You will get the explanations afterwards.
            </p>
            {previousScore !== null ? (
              <p className="assessment-page__caption">
                For reference, you scored{' '}
                <NumberValue value={previousScore} fmt="integer" /> out of{' '}
                <NumberValue value={instrument.max_score} fmt="integer" /> at the
                start.
              </p>
            ) : null}
          </>
        )}
      </header>

      <form className="assessment-page__form" onSubmit={handleSubmit}>
        <ol className="assessment-page__items">
          {instrument.items.map((item) => (
            <li key={item.id}>
              <fieldset className="assessment-page__item">
                <legend className="assessment-page__question">
                  {item.question}
                </legend>
                {item.options.map((option, optionIndex) => {
                  const optionId = `${item.id}-option-${optionIndex}`
                  return (
                    <div className="assessment-page__option" key={optionId}>
                      <input
                        type="radio"
                        id={optionId}
                        name={item.id}
                        checked={responses[item.id] === optionIndex}
                        onChange={() =>
                          setResponses((current) => ({
                            ...current,
                            [item.id]: optionIndex,
                          }))
                        }
                      />
                      <label htmlFor={optionId}>{option}</label>
                    </div>
                  )
                })}
              </fieldset>
            </li>
          ))}
        </ol>

        <div className="assessment-page__confidence">
          <label htmlFor="assessment-confidence">
            {instrument.confidence_question}
          </label>
          <input
            type="range"
            id="assessment-confidence"
            min={1}
            max={instrument.confidence_labels.length}
            step={1}
            value={confidence}
            aria-valuetext={confidenceLabel}
            aria-describedby="assessment-confidence-value"
            onChange={(event) => setConfidence(Number(event.target.value))}
          />
          <output
            id="assessment-confidence-value"
            htmlFor="assessment-confidence"
            className="assessment-page__confidence-value"
          >
            {confidenceLabel}
          </output>
        </div>

        {answered > 0 && unanswered > 0 ? (
          <Callout severity="warning">
            <NumberValue value={unanswered} fmt="integer" /> of the questions are
            unanswered. You can submit anyway — they will be marked as incorrect —
            or go back and answer them.
          </Callout>
        ) : null}

        {submitError !== null ? (
          <Callout severity="error" title="Not submitted">
            <p>{submitError}</p>
          </Callout>
        ) : null}

        <div className="assessment-page__actions">
          <Button
            type="submit"
            variant="primary"
            loading={submitting}
            loadingLabel="Submitting your answers"
            disabled={submitting}
          >
            Submit answers
          </Button>
          <Button variant="ghost" onClick={handleSkip} disabled={submitting}>
            Skip this — I would rather just use the simulator
          </Button>
        </div>
      </form>
    </div>
  )
}
