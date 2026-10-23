import { useId } from 'react'
import type { InputEcho, ParamValues, SimulationResult } from '@/api/types'
import { PlotlyChart } from '@/charts/PlotlyChart'
import { Callout } from '@/components/ui/Callout'
import { Disclosure } from '@/components/ui/Disclosure'
import { AssumptionsPanel } from './AssumptionsPanel'
import { ExportBar } from './ExportBar'
import { InterpretationList, renderMarkdown } from './InterpretationList'
import { MetricGrid } from './MetricGrid'
import { ResultTables } from './ResultTables'
import { SolverPanel } from './SolverPanel'
import './ResultView.css'

export interface ResultViewProps {
  result: SimulationResult
  params?: ParamValues
  scenarioId?: string
}

function paramsFromInputs(inputs: InputEcho[]): ParamValues {
  const values: ParamValues = {}
  for (const input of inputs) {
    if (input.value !== null) values[input.name] = input.value
  }
  return values
}

export function ResultView({ result, params, scenarioId }: ResultViewProps) {
  const warningsId = useId()
  const tablesId = useId()
  const id = scenarioId ?? result.scenario_id
  const exportParams = params ?? paramsFromInputs(result.inputs)
  const hasMetrics = result.metrics.length > 0 || result.derived.length > 0
  const hasConditions = result.assumptions.length > 0 || result.limitations.length > 0

  return (
    <article className="result-view" aria-label={`Result: ${result.scenario_title}`}>
      {result.warnings.length > 0 ? (
        <section className="result-view__section" aria-labelledby={warningsId}>
          <h2 className="visually-hidden" id={warningsId}>
            Warnings from this run
          </h2>
          <Callout severity="warning" title="Warnings from this run">
            <ul className="result-view__warnings">
              {result.warnings.map((warning, index) => (
                <li key={index}>{warning}</li>
              ))}
            </ul>
          </Callout>
        </section>
      ) : null}

      {hasMetrics ? (
        <section className="result-view__section">
          <MetricGrid metrics={result.metrics} title="Results" headingLevel={2} />
          {result.derived.length > 0 ? (
            <MetricGrid
              metrics={result.derived}
              title="Derived quantities"
              headingLevel={3}
            />
          ) : null}
        </section>
      ) : null}

      {result.plots.length > 0 ? (
        <section className="result-view__section" aria-label="Plots">
          <div className="result-view__plots">
            {result.plots.map((plot) => (
              <PlotlyChart key={plot.id} plot={plot} series={result.series} />
            ))}
          </div>
        </section>
      ) : null}

      {result.interpretations.length > 0 ? (
        <section className="result-view__section">
          <InterpretationList interpretations={result.interpretations} headingLevel={2} />
        </section>
      ) : null}

      {result.reading_guide.trim().length > 0 ? (
        <section className="result-view__section">
          <Disclosure title="How to read this result" icon="book">
            <div className="result-view__reading-guide">
              {renderMarkdown(result.reading_guide, {
                className: 'result-view__reading-guide-paragraph',
              })}
            </div>
          </Disclosure>
        </section>
      ) : null}

      {hasConditions ? (
        <section className="result-view__section">
          <AssumptionsPanel
            assumptions={result.assumptions}
            limitations={result.limitations}
            headingLevel={2}
          />
        </section>
      ) : null}

      <section className="result-view__section">
        <SolverPanel
          solver={result.solver}
          provenance={result.provenance}
          runtimeMs={result.runtime_ms}
          nSteps={result.n_steps}
          completedAt={result.completed_at}
          headingLevel={2}
        />
      </section>

      {result.tables.length > 0 ? (
        <section className="result-view__section" aria-labelledby={tablesId}>
          <h2 className="result-view__section-heading" id={tablesId}>
            Tables
          </h2>
          <ResultTables tables={result.tables} headingLevel={3} />
        </section>
      ) : null}

      <section className="result-view__section">
        <ExportBar
          result={result}
          scenarioId={id}
          params={exportParams}
          headingLevel={2}
        />
      </section>
    </article>
  )
}
