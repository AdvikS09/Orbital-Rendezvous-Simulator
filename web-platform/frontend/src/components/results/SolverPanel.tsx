import type { ReactNode } from 'react'
import { useId } from 'react'
import type { ProvenanceInfo, SolverInfo } from '@/api/types'
import { Icon } from '@/components/ui/Icon'
import { NumberValue } from '@/components/NumberValue'
import './SolverPanel.css'

export interface SolverPanelProps {
  solver: SolverInfo
  provenance: ProvenanceInfo
  runtimeMs?: number | null
  nSteps?: number | null
  completedAt?: string | null
  headingLevel?: 2 | 3 | 4
}

interface Row {
  label: string
  value: ReactNode
}

function textOrDash(value: string | null | undefined): ReactNode {
  const text = (value ?? '').trim()
  return text.length > 0 ? text : <span className="solver-panel__absent">Not reported</span>
}

export function SolverPanel({
  solver,
  provenance,
  runtimeMs,
  nSteps,
  completedAt,
  headingLevel = 2,
}: SolverPanelProps) {
  const baseId = useId()
  const Heading = `h${headingLevel}` as const

  const steps = nSteps ?? solver.n_steps

  const rows: Row[] = [
    { label: 'Integrator', value: textOrDash(solver.integrator) },
    {
      label: 'Deterministic',
      value: (
        <span className="solver-panel__flag">
          <Icon name={solver.deterministic ? 'check' : 'warning'} size={14} />
          <span>{solver.deterministic ? 'Yes' : 'No'}</span>
          {solver.random_seed !== null ? (
            <span className="solver-panel__seed">
              seed{' '}
              <NumberValue value={solver.random_seed} fmt="integer" precision={0} unit="" />
            </span>
          ) : null}
        </span>
      ),
    },
    { label: 'Force model', value: textOrDash(solver.force_model) },
    { label: 'Dynamics model', value: textOrDash(solver.model) },
    {
      label: 'Steps',
      value:
        steps === null || steps === undefined ? (
          textOrDash(null)
        ) : (
          <NumberValue value={steps} fmt="integer" precision={0} unit="" />
        ),
    },
    {
      label: 'Step size',
      value:
        solver.dt_s === null ? (
          textOrDash(null)
        ) : (
          <NumberValue value={solver.dt_s} fmt="auto" precision={4} unit="s" />
        ),
    },
    {
      label: 'Propagation span',
      value:
        solver.duration_s === null ? (
          textOrDash(null)
        ) : (
          <NumberValue value={solver.duration_s} fmt="duration" precision={2} unit="" />
        ),
    },
    { label: 'Engine version', value: textOrDash(solver.engine_version) },
    { label: 'Units', value: textOrDash(solver.units) },
    { label: 'Frames', value: textOrDash(solver.frames) },
    {
      label: 'Runtime',
      value:
        runtimeMs === null || runtimeMs === undefined ? (
          textOrDash(null)
        ) : (
          <NumberValue value={runtimeMs} fmt="auto" precision={4} unit="ms" />
        ),
    },
  ]

  if (completedAt) {
    rows.push({
      label: 'Completed',
      value: (
        <time className="solver-panel__time" dateTime={completedAt}>
          {completedAt}
        </time>
      ),
    })
  }

  return (
    <section className="solver-panel" aria-labelledby={baseId}>
      <Heading className="solver-panel__heading" id={baseId}>
        Solver and provenance
      </Heading>

      <dl className="solver-panel__grid">
        {rows.map((row) => (
          <div className="solver-panel__row" key={row.label}>
            <dt className="solver-panel__term">{row.label}</dt>
            <dd className="solver-panel__definition">{row.value}</dd>
          </div>
        ))}
      </dl>

      <div className="solver-panel__provenance">
        <h3 className="solver-panel__provenance-heading">
          <Icon name="shield" size={15} />
          <span>Validation status</span>
        </h3>
        <p className="solver-panel__provenance-text">{provenance.external_validation}</p>
        <dl className="solver-panel__provenance-meta">
          <div className="solver-panel__row">
            <dt className="solver-panel__term">Software</dt>
            <dd className="solver-panel__definition">{textOrDash(provenance.software)}</dd>
          </div>
          <div className="solver-panel__row">
            <dt className="solver-panel__term">Licence</dt>
            <dd className="solver-panel__definition">{textOrDash(provenance.licence)}</dd>
          </div>
        </dl>
      </div>
    </section>
  )
}
