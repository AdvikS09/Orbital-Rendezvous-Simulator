import { useId } from 'react'
import type { ReactNode } from 'react'
import { Icon } from './Icon'
import type { IconName } from './Icon'
import './EmptyState.css'

export interface EmptyStateProps {
  icon?: IconName
  heading: string
  /** What is missing and what the reader can do next. One or two sentences. */
  description?: ReactNode
  /** Usually a single Button. */
  action?: ReactNode
  /** Reduce the vertical padding, for an empty state inside a small panel. */
  compact?: boolean
  className?: string
}

/**
 * Placeholder for a region with nothing in it yet.
 *
 * Rendered as a labelled group so a screen reader announces the heading and
 * description together rather than reading a stray fragment of text.
 */
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
