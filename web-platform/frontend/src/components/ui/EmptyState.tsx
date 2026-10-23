import { useId } from 'react'
import type { ReactNode } from 'react'
import { Icon } from './Icon'
import type { IconName } from './Icon'
import './EmptyState.css'

export interface EmptyStateProps {
  icon?: IconName
  heading: string
  description?: ReactNode
  action?: ReactNode
  compact?: boolean
  className?: string
}

export function EmptyState({
  icon = 'orbit',
  heading,
  description,
  action,
  compact = false,
  className,
}: EmptyStateProps) {
  const headingId = `${useId()}-heading`
  const classes = [
    'empty-state',
    compact ? 'empty-state--compact' : '',
    className ?? '',
  ]
    .filter(Boolean)
    .join(' ')

  return (
    <div className={classes} role="group" aria-labelledby={headingId}>
      <Icon name={icon} size={28} className="empty-state__icon" />
      <p className="empty-state__heading" id={headingId}>
        {heading}
      </p>
      {description !== undefined ? (
        <p className="empty-state__description">{description}</p>
      ) : null}
      {action !== undefined ? <div className="empty-state__action">{action}</div> : null}
    </div>
  )
}
