import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { api } from '@/api/client'
import { STATUS_STATEMENT, VALIDATION_STATEMENT } from '@/components/Layout'
import { Button, Callout, Disclosure, Panel } from '@/components/ui'
import { ROUTES } from './participantFlow'
import './LandingPage.css'

const SUBTITLE =
  'An open-source tool for exploring how spacecraft move near one another in ' +
  'orbit — and how far a simple model can be trusted.'

const INTRO_FIRST =
  'This platform runs a real orbital-mechanics simulation engine in your ' +
  'browser. Nothing to install, no Python required.'

const INTRO_QUESTION =
  'when two spacecraft are close together, how well does the simple, cheap ' +
  'model engineers actually fly with agree with a more faithful one?'

const CAPABILITIES: readonly string[] = [
  'Propagate orbits and check that energy and angular momentum are conserved',
  'Switch on Earth’s equatorial bulge (J2) and atmospheric drag and see what each does',
  'Explore relative motion in a target-centred frame, where nothing moves the way intuition expects',
  'Fly a PD-controlled rendezvous and measure what it costs in propellant',
  'Sweep controller gains to map the speed-versus-fuel trade-off',
  'Compare the linear model against a non-linear reference and measure the gap',
  'Check keep-out-zone and debris clearance for a trajectory',
  'Download your results and a configuration file that reproduces the run exactly',
]

const AUDIENCES: readonly { lead: string; rest: string }[] = [
  {
    lead: 'Students',
    rest:
      ' meeting orbital mechanics for the first time — guided activities that ' +
      'ask you to predict before you run',
  },
  {
    lead: 'Teachers',
    rest:
      ' wanting a simulator that runs in a browser, with optional anonymous ' +
      'before-and-after assessments',
  },
  {
    lead: 'Engineers and researchers',
    rest:
      ' who want direct access to the model, the numerical diagnostics, and ' +
      'reproducible exports',
  },
]

const LIMITATIONS: readonly { lead: string; rest: string }[] = [
  {
    lead: 'The relative-motion model is linear.',
    rest:
      ' The Clohessy-Wiltshire equations assume a circular reference orbit and ' +
      'a small separation. Accuracy degrades as the spacecraft get further ' +
      'apart, and degrades fastest for an offset directly above or below the ' +
      'target.',
  },
  {
    lead: 'The reference model is better, not perfect.',
    rest:
      ' It adds non-linear gravity, J2 and drag, but has no higher-order ' +
      'gravity terms, no third-body effects and no solar radiation pressure.',
  },
  {
    lead: 'The atmosphere is crude.',
    rest:
      ' A single-scale-height exponential model calibrated at 400 km. Real ' +
      'density varies by an order of magnitude with solar activity, which is ' +
      'not modelled at all.',
  },
  {
    lead: 'Navigation is perfect.',
    rest:
      ' The controller is given exact, noise-free knowledge of the state. There ' +
      'is no sensor model and no estimator, so nothing here says anything about ' +
      'robustness to measurement error.',
  },
  {
    lead: 'Guidance is one PD controller.',
    rest: ' No impulsive targeting, no LQR, no model-predictive control.',
  },
  {
    lead: 'Nothing avoids anything.',
    rest: ' Keep-out zones and debris are detected and reported, never avoided.',
  },
  {
    lead: 'Identical spacecraft.',
    rest:
      ' Target and chaser share a ballistic coefficient, so differential drag ' +
      'is not exercised.',
  },
]

export function LandingPage() {
  const navigate = useNavigate()
  const [engineVersion, setEngineVersion] = useState<string | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    api
      .health(controller.signal)
      .then((health) => setEngineVersion(health.engine_version))
      .catch(() => {
      })
    return () => controller.abort()
  }, [])

  return (
    <div className="landing-page">
      <header className="landing-page__header">
        <h1>Orbital Rendezvous Simulator</h1>
        <p className="landing-page__subtitle">{SUBTITLE}</p>
      </header>

      <section className="landing-page__intro" aria-labelledby="landing-intro">
        <h2 className="visually-hidden" id="landing-intro">
          About this platform
        </h2>
        <p>{INTRO_FIRST}</p>
        <p>
          It is built around one question that matters in spaceflight:{' '}
          <strong>{INTRO_QUESTION}</strong> You can explore that question
          directly, or work through guided activities that build up to it.
        </p>
      </section>

      <div className="landing-page__columns">
        <Panel title="What you can do here">
          <ul className="landing-page__list">
            {CAPABILITIES.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </Panel>

        <Panel title="Who it is for">
          <ul className="landing-page__list">
            {AUDIENCES.map((item) => (
              <li key={item.lead}>
                <strong>{item.lead}</strong>
                {item.rest}
              </li>
            ))}
          </ul>
        </Panel>
      </div>

      <section
        className="landing-page__not"
        aria-labelledby="landing-what-this-is-not"
      >
        <h2 id="landing-what-this-is-not">What this is not</h2>
        <Callout severity="warning">
          <p>{STATUS_STATEMENT}</p>
          <p>{VALIDATION_STATEMENT}</p>
        </Callout>

        <Disclosure
          title="Known limitations of the physics — worth reading before you start"
          count={LIMITATIONS.length}
        >
          <ul className="landing-page__list landing-page__list--limitations">
            {LIMITATIONS.map((item) => (
              <li key={item.lead}>
                <strong>{item.lead}</strong>
                {item.rest}
              </li>
            ))}
          </ul>
        </Disclosure>
      </section>

      <hr />

      <div className="landing-page__start">
        <Button
          variant="primary"
          onClick={() => navigate(ROUTES.consent)}
          aria-describedby="landing-start-note"
        >
          Start
        </Button>
        <p className="landing-page__note" id="landing-start-note">
          {engineVersion === null ? null : <>Engine v{engineVersion}. </>}
          Open source under the MIT licence. The next page explains what is
          recorded and asks your consent.
        </p>
      </div>
    </div>
  )
}
