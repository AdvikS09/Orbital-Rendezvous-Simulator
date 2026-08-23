/**
 * The headline numbers of a result.
 *
 * Metrics arrive already classified by the engine: `highlight` marks the two or
 * three figures that answer the question the run was asking, and the rest are
 * supporting detail. Those two groups are presented differently, because a grid
 * of twenty equally weighted tiles has no hierarchy and therefore says nothing
 * about which number matters.
 *
 * The `help` text of every metric is reachable: an info button beside the
 * label, wired with `aria-describedby` to a tooltip that is announced whether
 * or not it is visible. A `title` attribute would be invisible to keyboard
 * users and unreliable on touch, so it is not used.
 */

import { useId } from 'react'
import type { Metric } from '@/api/types'
import { Icon } from '@/components/ui/Icon'
import { NumberValue } from '@/components/NumberValue'
import './MetricGrid.css'

export interface MetricGridProps {
  metrics: Metric[]
  /** Optional heading above the group. */
  title?: string
  /** Heading level, so the group sits correctly in the document outline. */
  headingLevel?: 2 | 3 | 4
  /** Accessible label used when no visible title is given. */
  ariaLabel?: string
}

interface HelpTipProps {
  label: string
  text: string
}

/**
 * An info affordance whose text is always in the accessibility tree.
 *
 * The tooltip is hidden with opacity rather than `display: none` or
 * `visibility: hidden`, either of which would remove it from the tree and
 * break the `aria-describedby` reference it exists to serve.
 */
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

  // With nothing marked as headline, promoting everything to a card would be a
  // claim about importance the engine never made; the compact list is honest.
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
