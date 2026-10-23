import type { ReactNode } from 'react'
import { Icon } from './Icon'
import type { IconName } from './Icon'
import './Badge.css'

export type BadgeTone = 'neutral' | 'accent' | 'success' | 'warning' | 'danger'

export interface BadgeProps {
  tone?: BadgeTone
  icon?: IconName
  mono?: boolean
  srPrefix?: string
  className?: string
  children: ReactNode
}

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
