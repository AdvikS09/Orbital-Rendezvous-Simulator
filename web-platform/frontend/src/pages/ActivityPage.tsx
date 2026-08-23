/**
 * A guided activity, one step at a time.
 *
 * Route: `/activities/:activityId`.
 *
 * The structure is predict, run, observe. The prediction is required: the Run
 * button stays disabled until one is chosen, and the caption says so. That is
 * not friction for its own sake — committing to an answer before seeing the
 * result is what separates an experiment from an animation, and a learner who
 * has not committed will read whatever came back as the thing they expected.
 *
 * Predictions are held in this component and never sent anywhere. Nothing is
 * recorded about which one was chosen, which is what makes it safe to be wrong.
 */

import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { ApiError, api } from '@/api/client'
import type {
  Activity,
  ParamValues,
  ScenarioDetail,
  SimulationResult,
} from '@/api/types'
import { NumberValue } from '@/components/NumberValue'
import { Button, Callout, Disclosure, Panel, Spinner } from '@/components/ui'
import { ParamPanel } from '@/components/controls/ParamPanel'
import { ResultView } from '@/components/results'
import { useSession } from '@/state/session'
import { ROUTES } from './participantFlow'
import { Prose } from './Prose'
import './ActivityPage.css'

function messageFor(cause: unknown, fallback: string): string {
  return cause instanceof ApiError ? cause.message : fallback
}

export function ActivityPage() {
  const { activityId = '' } = useParams<{ activityId: string }>()
  const navigate = useNavigate()
  const { recordCompletedRun } = useSession()

  const [activity, setActivity] = useState<Activity | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [index, setIndex] = useState(0)

  const [scenario, setScenario] = useState<ScenarioDetail | null>(null)
  const [scenarioError, setScenarioError] = useState<string | null>(null)
  const [params, setParams] = useState<ParamValues>({})

  /** Keyed by step id. Held here only; never sent to the server. */
  const [predictions, setPredictions] = useState<Record<string, string>>({})

  const [result, setResult] = useState<SimulationResult | null>(null)
  const [running, setRunning] = useState(false)
  const [runError, setRunError] = useState<ApiError | null>(null)

  const [finished, setFinished] = useState(false)
  const [finishWarning, setFinishWarning] = useState<string | null>(null)

  // ------------------------------------------------------------- load
  useEffect(() => {
    const controller = new AbortController()
    setActivity(null)
    setLoadError(null)
    setIndex(0)
    setFinished(false)

    api
      .getActivity(activityId, controller.signal)
      .then(setActivity)
      .catch((cause: unknown) => {
        if (controller.signal.aborted) return
        setLoadError(messageFor(cause, 'This activity could not be loaded.'))
      })

    return () => controller.abort()
  }, [activityId])

  const step = activity?.steps[index]

  // The parameters for a step are the scenario defaults with the activity's
  // starting values laid over them, so a step only has to state what it changes.
  useEffect(() => {
    if (!step) return
    const controller = new AbortController()
    setScenario(null)
    setScenarioError(null)
    setResult(null)
    setRunError(null)

    api
      .getScenario(step.scenario_id, controller.signal)
      .then((loaded) => {
        setScenario(loaded)
        setParams({ ...loaded.defaults, ...step.params })
      })
      .catch((cause: unknown) => {
        if (controller.signal.aborted) return
        setScenarioError(
          messageFor(cause, 'The scenario for this step could not be loaded.'),
        )
      })

    return () => controller.abort()
  }, [step])

  const prediction = step ? (predictions[step.id] ?? null) : null
  const stepCount = activity?.steps.length ?? 0
  const isLastStep = stepCount > 0 && index === stepCount - 1

  const fieldErrors = useMemo(() => runError?.fieldErrors ?? {}, [runError])

  // ParamPanel reports one changed parameter by name; it holds no state itself.
  const handleParamChange = useCallback(
    (name: string, value: number | string | boolean) => {
      setParams((current) => ({ ...current, [name]: value }))
    },
    [],
  )

  const handlePredict = useCallback(
    (choice: string) => {
      if (!step) return
      setPredictions((current) => ({ ...current, [step.id]: choice }))
    },
    [step],
  )

  const handleRun = useCallback(async () => {
    if (!step || !activity) return
    setRunning(true)
    setRunError(null)
    try {
      const outcome = await api.runSimulation(step.scenario_id, params, {
        activity_id: activity.id,
        step_id: step.id,
      })
      setResult(outcome)
      recordCompletedRun()
    } catch (cause) {
      setResult(null)
      setRunError(
        cause instanceof ApiError
          ? cause
          : new ApiError(0, null, 'The simulation could not be completed.'),
      )
    } finally {
      setRunning(false)
    }
  }, [activity, params, recordCompletedRun, step])

  const goToStep = useCallback((next: number) => {
    setIndex(next)
    setResult(null)
    setRunError(null)
    window.scrollTo({ top: 0 })
  }, [])

  const handleFinish = useCallback(async () => {
    if (!activity) return
    try {
      await api.completeActivity(activity.id)
      setFinishWarning(null)
    } catch (cause) {
      // Completion is an analytics record, not the point of the activity. If it
      // cannot be stored the learner has still finished, so say so and move on.
      setFinishWarning(
        messageFor(
          cause,
          'Your progress could not be recorded. The activity is still complete.',
        ),
      )
    }
    setFinished(true)
    window.scrollTo({ top: 0 })
  }, [activity])

  // ----------------------------------------------------------- render
  if (loadError !== null) {
    return (
      <div className="activity-page activity-page--message">
        <Callout severity="error" title="Activity unavailable">
          <p>{loadError}</p>
        </Callout>
        <p>
          <Link to={ROUTES.activities}>Back to the activities</Link>
        </p>
      </div>
    )
  }

  if (activity === null || !step) {
    return (
      <div className="activity-page activity-page--message">
        <p className="activity-page__loading">
          <Spinner label="Loading the activity" />
          <span aria-hidden="true">Loading the activity…</span>
        </p>
      </div>
    )
  }

  if (finished) {
    return (
      <div className="activity-page activity-page--message">
        <Callout severity="success" title="Activity complete">
          <p>{activity.title}</p>
        </Callout>
        {finishWarning !== null ? (
          <Callout severity="warning">{finishWarning}</Callout>
        ) : null}
        <div className="activity-page__actions">
          <Button variant="primary" onClick={() => navigate('/assessment/post')}>
            Answer the closing questions
          </Button>
          <Button onClick={() => navigate(ROUTES.activities)}>
            Back to the activities
          </Button>
        </div>
      </div>
    )
  }

  return (
    <div className="activity-page">
      <header className="activity-page__header">
        <div className="activity-page__title-row">
          <div>
            <h1>{activity.title}</h1>
            <p className="activity-page__step-label">
              Step <NumberValue value={index + 1} fmt="integer" /> of{' '}
              <NumberValue value={stepCount} fmt="integer" /> — {step.title}
            </p>
          </div>
          <Button icon="close" onClick={() => navigate(ROUTES.activities)}>
            Leave activity
          </Button>
        </div>
        <progress
          className="activity-page__progress"
          value={index + 1}
          max={stepCount}
          aria-label="Activity progress"
        />
      </header>

      <Prose text={step.brief} className="activity-page__brief" />

      {/* --------------------------------------------------- prediction */}
      <Panel
        title="Predict first"
        subtitle={step.predict_prompt}
        headingLevel={2}
        icon="target"
      >
        <fieldset className="activity-page__predict">
          <legend className="visually-hidden">{step.predict_prompt}</legend>
          {step.predict_options.map((option) => {
            const optionId = `predict-${step.id}-${option.replace(/\W+/g, '-')}`
            return (
              <div className="activity-page__option" key={option}>
                <input
                  type="radio"
                  id={optionId}
                  name={`predict-${step.id}`}
                  value={option}
                  checked={prediction === option}
                  onChange={() => handlePredict(option)}
                />
                <label htmlFor={optionId}>{option}</label>
              </div>
            )
          })}
        </fieldset>
        <p className="activity-page__caption">
          There is no penalty for being wrong, and no record is kept of what you
          choose. Committing to a prediction is what makes the result
          informative.
        </p>
      </Panel>

      {/* --------------------------------------------------- parameters */}
      <Disclosure title="Parameters for this step" icon="settings">
        <p className="activity-page__caption">
          Pre-set for the activity. Change them if you want to experiment.
        </p>
        {scenarioError !== null ? (
          <Callout severity="error">{scenarioError}</Callout>
        ) : scenario === null ? (
          <p className="activity-page__loading">
            <Spinner label="Loading the parameters" />
            <span aria-hidden="true">Loading the parameters…</span>
          </p>
        ) : (
          <ParamPanel
            params={scenario.params}
            values={params}
            onChange={handleParamChange}
            errors={fieldErrors}
            disabled={running}
            legend={`Parameters for ${step.title}`}
          />
        )}
      </Disclosure>

      {/* ---------------------------------------------------------- run */}
      <div className="activity-page__run">
        <Button
          variant="primary"
          icon="play"
          onClick={handleRun}
          loading={running}
          loadingLabel="Running this step"
          disabled={prediction === null || running || scenario === null}
        >
          Run this step
        </Button>
        {prediction === null ? (
          <p className="activity-page__caption">
            Choose a prediction above to unlock the Run button.
          </p>
        ) : null}
      </div>

      {/* ------------------------------------------------------ results */}
      <section className="activity-page__results" aria-label="Results">
        {running ? (
          <p className="activity-page__loading">
            <Spinner size="md" label="Running the simulation" />
            <span aria-hidden="true">Running the simulation…</span>
          </p>
        ) : null}

        {runError !== null && !running ? (
          <Callout
            severity={runError.isRateLimited ? 'warning' : 'error'}
            title={
              runError.isRateLimited
                ? 'Run limit reached'
                : 'The simulation did not run'
            }
          >
            {runError.generalErrors.map((message, position) => (
              <p key={position}>{message}</p>
            ))}
            {runError.rejected ? (
              <p>
                Nothing was run. Adjust the values above and try again — the
                engine itself has no such limits when you run it locally.
              </p>
            ) : null}
          </Callout>
        ) : null}

        {result !== null && !running ? (
          <>
            <Callout severity="info" title="What to look for">
              <Prose text={step.look_for} />
            </Callout>
            <p className="activity-page__caption">
              Completed in{' '}
              <NumberValue value={result.runtime_ms} fmt="integer" unit="ms" />
              {' · '}
              <NumberValue value={result.n_steps} fmt="integer" /> integration
              steps.
            </p>
            <ResultView
              result={result}
              params={params}
              scenarioId={step.scenario_id}
            />
            <Callout severity="success" title="Takeaway">
              <Prose text={step.takeaway} />
            </Callout>
          </>
        ) : null}
      </section>

      {/* --------------------------------------------------- navigation */}
      <nav className="activity-page__nav" aria-label="Activity steps">
        <Button
          icon="chevronLeft"
          onClick={() => goToStep(index - 1)}
          disabled={index === 0}
        >
          Previous step
        </Button>
        {isLastStep ? (
          <Button variant="primary" icon="check" onClick={handleFinish}>
            Finish activity
          </Button>
        ) : (
          <Button
            variant="primary"
            icon="chevronRight"
            iconPosition="end"
            onClick={() => goToStep(index + 1)}
          >
            Next step
          </Button>
        )}
      </nav>
    </div>
  )
}
