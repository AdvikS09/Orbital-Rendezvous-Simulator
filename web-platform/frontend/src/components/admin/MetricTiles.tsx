import { useId } from 'react'

import type { NumberFormatName } from '@/api/types'
import { NumberValue } from '@/components/NumberValue'
import { Icon } from '@/components/ui'
import './MetricTiles.css'

export interface MetricTile {
  key: string
  label: string
  value: number | string | null
  fmt?: NumberFormatName
  precision?: number | null
  unit?: string
  help?: string
}

export interface MetricTilesProps {
  tiles: readonly MetricTile[]
  title?: string
  headingLevel?: 2 | 3 | 4
  ariaLabel?: string
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
