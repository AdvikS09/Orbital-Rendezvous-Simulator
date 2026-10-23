import { useCallback, useEffect, useMemo, useState } from 'react'
import { useParams } from 'react-router-dom'
import { ApiError, api } from '@/api/client'
import type {
  ParamValues,
  PreflightResult,
  ScenarioDetail,
  SimulationResult,
} from '@/api/types'
import { NumberValue } from '@/components/NumberValue'
import {
  Badge,
  Button,
  Callout,
  Disclosure,
  Panel,
  Spinner,
  TrackChips,
} from '@/components/ui'
import { ParamPanel } from '@/components/controls/ParamPanel'
import { ResultView } from '@/components/results'
import { useSession } from '@/state/session'
import { Prose } from './Prose'
import './ScenarioPage.css'

const PREFLIGHT_DELAY_MS = 350

function messageFor(cause: unknown, fallback: string): string {
  return cause instanceof ApiError ? cause.message : fallback
}

function RunErrorCallout({ error }: { error: ApiError }) {
  if (error.isRateLimited) {
    const retryAfter = error.body?.retry_after_s ?? null
    return (
      <Callout severity="warning" title="Run limit reached">
        <p>{error.message}</p>
        {retryAfter !== null ? (
          <p>
            You can run again in <NumberValue value={retryAfter} fmt="duration" />.
          </p>
        ) : null}
        <p>
          The limit belongs to this hosted pilot. The engine itself has no such
          limits when you run it locally.
        </p>
      </Callout>
    )
  }

  const messages = error.generalErrors
  return (
    <Callout severity="error" title="The simulation did not run">
      {messages.length === 1 ? (
        <p>{messages[0]}</p>
      ) : (
        <ul className="scenario-page__error-list">
          {messages.map((message, index) => (
            <li key={index}>{message}</li>
          ))}
        </ul>
      )}
      {error.rejected ? (
        <p>
          Nothing was run. Adjust the values above and try again — the engine
          itself has no such limits when you run it locally.
        </p>
      ) : null}
    </Callout>
  )
}

export function ScenarioPage() {
  const { scenarioId = '' } = useParams<{ scenarioId: string }>()
  const { recordCompletedRun } = useSession()

  const [detail, setDetail] = useState<ScenarioDetail | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [params, setParams] = useState<ParamValues>({})
  const [preflight, setPreflight] = useState<PreflightResult | null>(null)
  const [checking, setChecking] = useState(false)
  const [result, setResult] = useState<SimulationResult | null>(null)
  const [running, setRunning] = useState(false)
  const [runError, setRunError] = useState<ApiError | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    setDetail(null)
    setLoadError(null)
    setParams({})
    setPreflight(null)
    setResult(null)
    setRunError(null)

    api
      .getScenario(scenarioId, controller.signal)
      .then((loaded) => {
        setDetail(loaded)
        setParams({ ...loaded.defaults })
      })
      .catch((cause: unknown) => {
        if (controller.signal.aborted) return
        setLoadError(messageFor(cause, 'This scenario could not be loaded.'))
      })

    return () => controller.abort()
  }, [scenarioId])

  useEffect(() => {
    if (detail === null) return
    if (Object.keys(params).length === 0) return

    const controller = new AbortController()
    setChecking(true)

    const timer = window.setTimeout(() => {
      api
        .preflight(detail.id, params, controller.signal)
        .then((check) => {
          if (controller.signal.aborted) return
          setPreflight(check)
          setChecking(false)
        })
        .catch((cause: unknown) => {
          if (controller.signal.aborted) return
          setChecking(false)
          setPreflight({
            ok: false,
            errors:
              cause instanceof ApiError && (cause.body?.detail.length ?? 0) > 0
                ? (cause.body?.detail ?? [])
                : [
                    {
                      field: null,
                      message: messageFor(
                        cause,
                        'These parameters could not be checked.',
                      ),
                    },
                  ],
            warnings: [],
            n_steps: 0,
            estimated_seconds: 0,
            params,
          })
        })
    }, PREFLIGHT_DELAY_MS)

    return () => {
      window.clearTimeout(timer)
      controller.abort()
    }
  }, [detail, params])

  const fieldErrors = useMemo(() => {
    const collected: Record<string, string> = {}
    for (const item of preflight?.errors ?? []) {
      if (item.field) collected[item.field] = item.message
    }
    return { ...collected, ...(runError?.fieldErrors ?? {}) }
  }, [preflight, runError])

  const generalPreflightErrors = useMemo(
    () => (preflight?.errors ?? []).filter((item) => !item.field),
    [preflight],
  )

  const blocked = preflight !== null && !preflight.ok

  const handleParamChange = useCallback(
    (name: string, value: number | string | boolean) => {
      setParams((current) => ({ ...current, [name]: value }))
    },
    [],
  )

  const handleRun = useCallback(async () => {
    if (detail === null) return
    setRunning(true)
    setRunError(null)
    try {
      const outcome = await api.runSimulation(detail.id, params)
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
  }, [detail, params, recordCompletedRun])

  const handleReset = useCallback(() => {
    if (detail === null) return
    setParams({ ...detail.defaults })
    setPreflight(null)
    setResult(null)
    setRunError(null)
  }, [detail])

  if (loadError !== null) {
    return (
      <div className="scenario-page scenario-page--message">
        <Callout severity="error" title="Scenario unavailable">
          <p>{loadError}</p>
        </Callout>
      </div>
    )
  }

  if (detail === null) {
    return (
      <div className="scenario-page scenario-page--message">
        <p className="scenario-page__loading">
          <Spinner label="Loading the scenario" />
          <span aria-hidden="true">Loading the scenario…</span>
        </p>
      </div>
    )
  }

  return (
    <div className="scenario-page">
      <header className="scenario-page__header">
        <div className="scenario-page__tags">
          <Badge tone="neutral">{detail.category_label}</Badge>
          <TrackChips levels={detail.levels} withTitle />
        </div>
        <h1>{detail.title}</h1>
        <p className="scenario-page__summary">{detail.summary}</p>

        <Disclosure title="Background" icon="book">
          <Prose text={detail.explanation} />
          {detail.references.length > 0 ? (
            <p className="scenario-page__references">
              References: {detail.references.join('; ')}
            </p>
          ) : null}
        </Disclosure>
      </header>

      <div className="scenario-page__layout">
        <aside className="scenario-page__controls" aria-label="Parameters">
          <Panel title="Parameters" headingLevel={2} icon="settings">
            <ParamPanel
              params={detail.params}
              values={params}
              onChange={handleParamChange}
              errors={fieldErrors}
              disabled={running}
              legend={`Parameters for ${detail.title}`}
            />

            <div className="scenario-page__estimate" aria-live="polite">
              {checking ? (
                <span className="scenario-page__estimate-text">
                  <Spinner label="Checking these parameters" inherit />
                  <span aria-hidden="true">Checking…</span>
                </span>
              ) : preflight !== null && preflight.ok ? (
                <span className="scenario-page__estimate-text">
                  Estimated{' '}
                  <NumberValue value={preflight.n_steps} fmt="integer" />{' '}
                  integration steps (~
                  <NumberValue
                    value={preflight.estimated_seconds}
                    fmt="fixed"
                    precision={1}
                    unit="s"
                  />
                  ).
                </span>
              ) : null}
            </div>

            {generalPreflightErrors.map((item, index) => (
              <Callout severity="error" key={index}>
                {item.message}
              </Callout>
            ))}

            {(preflight?.warnings ?? []).map((warning, index) => (
              <Callout severity="warning" key={index}>
                {warning}
              </Callout>
            ))}

            <div className="scenario-page__actions">
              <Button
                variant="primary"
                icon="play"
                onClick={handleRun}
                loading={running}
                loadingLabel="Running the simulation"
                disabled={blocked || running}
              >
                Run simulation
              </Button>
              <Button icon="reset" onClick={handleReset} disabled={running}>
                Reset
              </Button>
            </div>

            {blocked ? (
              <p className="scenario-page__blocked">
                Running is disabled until the marked parameters are within range.
              </p>
            ) : null}
          </Panel>
        </aside>

        <section className="scenario-page__results" aria-label="Results">
          {running ? (
            <p className="scenario-page__loading">
              <Spinner size="md" label="Running the simulation" />
              <span aria-hidden="true">Running the simulation…</span>
            </p>
          ) : null}

          {runError !== null && !running ? (
            <RunErrorCallout error={runError} />
          ) : null}

          {result !== null && !running ? (
            <>
              <p className="scenario-page__completed">
                Completed in{' '}
                <NumberValue value={result.runtime_ms} fmt="integer" unit="ms" />
                {' · '}
                <NumberValue value={result.n_steps} fmt="integer" /> integration
                steps.
              </p>
              <ResultView
                result={result}
                params={params}
                scenarioId={detail.id}
              />
            </>
          ) : null}

          {result === null && runError === null && !running ? (
            <Callout severity="info" title="Nothing has been run yet">
              <p>
                Set the parameters, then run. Results appear here and stay until
                the next run.
              </p>
            </Callout>
          ) : null}
        </section>
      </div>
    </div>
  )
}
