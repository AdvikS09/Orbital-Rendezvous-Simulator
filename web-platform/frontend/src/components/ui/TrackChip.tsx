import { Icon } from './Icon'
import { trackMeta } from '@/content/tracks'
import './TrackChip.css'

export interface TrackChipProps {
  level: string
  withTitle?: boolean
}

export function TrackChip({ level, withTitle = false }: TrackChipProps) {
  const meta = trackMeta(level)
  return (
    <span
      className={`track-chip track-chip--${meta.id}`}
      title={withTitle ? meta.description : undefined}
    >
      <Icon name={meta.icon} size={12} className="track-chip__icon" />
      <span className="track-chip__label">{meta.label}</span>
    </span>
  )
}

export interface TrackChipsProps {
  levels: string[]
  withTitle?: boolean
}

export function TrackChips({ levels, withTitle = false }: TrackChipsProps) {
  if (levels.length === 0) return null
  return (
    <span className="track-chips">
      {levels.map((level) => (
        <TrackChip key={level} level={level} withTitle={withTitle} />
      ))}
    </span>
  )
}
