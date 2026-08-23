/**
 * Headline figures for the administrator report.
 *
 * A metric here is worthless without its definition. "Unique users" and "active
 * users" differ by whether the reporting period applies, and a reader who
 * assumes the wrong one draws the wrong conclusion, so every tile that has a
 * definition carries it on an info affordance beside the label.
 *
 * That affordance is a real button with `aria-describedby`, not a `title`
 * attribute: a `title` is invisible to keyboard users, unreliable on touch, and
 * announced inconsistently. The tooltip is hidden with opacity rather than
 * `display: none`, because either of the latter would drop it out of the
 * accessibility tree and break the reference it exists to serve.
 */

import { useId } from 'react'

import type { NumberFormatName } from '@/api/types'
import { NumberValue } from '@/components/NumberValue'
import { Icon } from '@/components/ui'
import './MetricTiles.css'

export interface MetricTile {
  /** Stable key, used for React reconciliation only. */
  key: string
  label: string
  /** A string passes through verbatim; anything numeric is formatted. */
  value: number | string | null
  /** Presentation. Counts should pass 'integer', fractions 'percent'. */
  fmt?: NumberFormatName
  precision?: number | null
  unit?: string
  /** The metric's definition. Omit only where there is genuinely nothing to say. */
  help?: string
}

export interface MetricTilesProps {
  tiles: readonly MetricTile[]
  /** Visible heading above the group. */
  title?: string
  headingLevel?: 2 | 3 | 4
  /** Accessible name used when there is no visible title. */
  ariaLabel?: string
  /** Preferred column count at full width. The grid still reflows below it. */
  columns?: 2 | 3 | 4
  className?: string
}

function HelpTip({ label, text }: { label: string; text: string }) {
  const tooltipId = useId()
  return (
    <span className="metric-tiles__help">
      <button
        type="button"
        className="metric-tiles__help-button"
        aria-label={`Explain ${label}`}
        aria-describedby={tooltipId}
      >
        <Icon name="info" size={13} />
      </button>
      <span role="tooltip" id={tooltipId} className="metric-tiles__tooltip">
        {text}
      </span>
    </span>
  )
}

export function MetricTiles({
  tiles,
  title,
  headingLevel = 2,
  ariaLabel,
  columns = 4,
  className,
}: MetricTilesProps) {
  const headingId = useId()

  if (tiles.length === 0) return null

  const Heading = `h${headingLevel}` as const

  const classes = ['metric-tiles', className ?? ''].filter(Boolean).join(' ')

  return (
    <section
      className={classes}
      aria-labelledby={title ? headingId : undefined}
      aria-label={title ? undefined : (ariaLabel ?? 'Summary metrics')}
    >
      {title ? (
        <Heading className="metric-tiles__heading" id={headingId}>
          {title}
        </Heading>
      ) : null}

      <dl className={`metric-tiles__grid metric-tiles__grid--${columns}`}>
        {tiles.map((tile) => (
          <div className="metric-tiles__tile" key={tile.key}>
            <dt className="metric-tiles__label">
              <span className="metric-tiles__label-text">{tile.label}</span>
              {tile.help ? <HelpTip label={tile.label} text={tile.help} /> : null}
            </dt>
            <dd className="metric-tiles__value">
              <NumberValue
                value={tile.value}
                fmt={tile.fmt}
                precision={tile.precision ?? null}
                unit={tile.unit ?? ''}
              />
            </dd>
          </div>
        ))}
      </dl>
    </section>
  )
}
