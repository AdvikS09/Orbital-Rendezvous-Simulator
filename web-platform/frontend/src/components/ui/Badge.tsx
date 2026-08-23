import type { ReactNode } from 'react'
import { Icon } from './Icon'
import type { IconName } from './Icon'
import './Badge.css'

export type BadgeTone = 'neutral' | 'accent' | 'success' | 'warning' | 'danger'

export interface BadgeProps {
  tone?: BadgeTone
  /** Optional supporting glyph. The text remains the carrier of meaning. */
  icon?: IconName
  /** Use tabular figures. For counts and short numeric labels. */
  mono?: boolean
  /**
   * Prefix read only by assistive technology, for badges whose meaning would
   * otherwise depend on their colour alone, e.g. 'Status: '.
   */
  srPrefix?: string
  className?: string
  children: ReactNode
}

/** Small static label. Not interactive: never put a click handler on it. */
export function Badge({
  tone = 'neutral',
  icon,
  mono = false,
  srPrefix,
  className,
  children,
}: BadgeProps) {
  const classes = [
    'badge',
    `badge--${tone}`,
    mono ? 'badge--mono' : '',
    className ?? '',
  ]
    .filter(Boolean)
    .join(' ')

  return (
    <span className={classes}>
      {srPrefix ? <span className="visually-hidden">{srPrefix}</span> : null}
      {icon ? <Icon name={icon} size={12} className="badge__icon" /> : null}
      {children}
    </span>
  )
}
