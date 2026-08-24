/**
 * A small static chip naming a track a scenario belongs to.
 *
 * Not interactive: it is a label, not a filter. The three tracks are told apart
 * by their word and their glyph; the faint colour on the icon only assists and
 * is never the sole difference, so the chips stay legible without colour.
 */

import { Icon } from './Icon'
import { trackMeta } from '@/content/tracks'
import './TrackChip.css'

export interface TrackChipProps {
  /** The API level string: 'guided' | 'professional' | 'free'. */
  level: string
  /** Attach the track's one-line meaning as a native tooltip. */
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

/** The tracks a scenario belongs to, in the order the API returns them. */
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
