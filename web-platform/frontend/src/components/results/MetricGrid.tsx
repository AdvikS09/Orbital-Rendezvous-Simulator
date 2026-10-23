import { useId } from 'react'
import type { Metric } from '@/api/types'
import { Icon } from '@/components/ui/Icon'
import { NumberValue } from '@/components/NumberValue'
import './MetricGrid.css'

export interface MetricGridProps {
  metrics: Metric[]
  title?: string
  headingLevel?: 2 | 3 | 4
  ariaLabel?: string
}

interface HelpTipProps {
  label: string
  text: string
}

function HelpTip({ label, text }: HelpTipProps) {
  const tooltipId = useId()
  return (
    <span className="metric-grid__help">
      <button
        type="button"
        className="metric-grid__help-button"
        aria-label={`Explain ${label}`}
        aria-describedby={tooltipId}
      >
        <Icon name="info" size={13} />
      </button>
      <span role="tooltip" id={tooltipId} className="metric-grid__tooltip">
        {text}
      </span>
    </span>
  )
}

function MetricValue({ metric }: { metric: Metric }) {
  return (
    <span className="metric-grid__value num num--right">
      <NumberValue
        value={metric.value}
        fmt={metric.fmt}
        precision={metric.precision}
        unit={metric.unit}
      />
    </span>
  )
}

function HighlightCard({ metric }: { metric: Metric }) {
  return (
    <li className="metric-grid__card">
      <span className="metric-grid__label">
        <span className="metric-grid__label-text">{metric.label}</span>
        {metric.help ? <HelpTip label={metric.label} text={metric.help} /> : null}
      </span>
      <MetricValue metric={metric} />
    </li>
  )
}

export function MetricGrid({
  metrics,
  title,
  headingLevel = 2,
  ariaLabel,
}: MetricGridProps) {
  const generatedId = useId()

  if (metrics.length === 0) return null

  const Heading = `h${headingLevel}` as const

  const highlighted = metrics.filter((metric) => metric.highlight)
  const supporting = metrics.filter((metric) => !metric.highlight)

  const cards = highlighted
  const rest = highlighted.length === 0 ? metrics : supporting

  return (
    <section
      className="metric-grid"
      aria-labelledby={title ? generatedId : undefined}
      aria-label={title ? undefined : (ariaLabel ?? 'Results')}
    >
      {title ? (
        <Heading className="metric-grid__heading" id={generatedId}>
          {title}
        </Heading>
      ) : null}

      {cards.length > 0 ? (
        <ul className="metric-grid__cards" role="list">
          {cards.map((metric) => (
            <HighlightCard key={metric.key} metric={metric} />
          ))}
        </ul>
      ) : null}

      {rest.length > 0 ? (
        <dl className="metric-grid__list">
          {rest.map((metric) => (
            <div className="metric-grid__row" key={metric.key}>
              <dt className="metric-grid__term">
                <span className="metric-grid__label-text">{metric.label}</span>
                {metric.help ? <HelpTip label={metric.label} text={metric.help} /> : null}
              </dt>
              <dd className="metric-grid__definition">
                <MetricValue metric={metric} />
              </dd>
            </div>
          ))}
        </dl>
      ) : null}
    </section>
  )
}
