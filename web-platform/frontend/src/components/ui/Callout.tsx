import type { ReactNode } from 'react'

// Severity is defined once, in the API contract, so a severity added
// server-side cannot be silently unrenderable here. Imported for local use and
// re-exported so existing callers keep working.
import type { Severity } from '@/api/types'

import { Icon, SEVERITY_ICON } from './Icon'
import './Callout.css'

export type { Severity }

/**
 * Read out before the message body. Without it the severity of a callout would
 * be carried by its tint alone, which is invisible to a screen reader and to
 * anyone printing in greyscale.
 */
const SEVERITY_LABEL: Record<Severity, string> = {
  info: 'Note:',
  success: 'Success:',
  warning: 'Warning:',
  error: 'Error:',
}

export interface CalloutProps {
  severity: Severity
  title?: ReactNode
  /** Rendered under the title, or as the whole message when there is no title. */
  children: ReactNode
  /** Buttons or links placed after the message. */
  actions?: ReactNode
  id?: string
  className?: string
}

/**
 * Inline status message.
 *
 * Errors and warnings use `role="alert"`, which interrupts and is announced
 * immediately, because they report a failed run or a parameter the solver
 * could not honour. Information and success use `role="status"` so they are
 * announced politely once the user is idle.
 */
export function Callout({
  severity,
  title,
  children,
  actions,
  id,
  className,
}: CalloutProps) {
  const isUrgent = severity === 'error' || severity === 'warning'
  const classes = ['callout', `callout--${severity}`, className ?? '']
    .filter(Boolean)
    .join(' ')

  return (
    <div className={classes} id={id} role={isUrgent ? 'alert' : 'status'}>
      <Icon name={SEVERITY_ICON[severity]} size={16} className="callout__icon" />
      <div className="callout__content">
        <span className="visually-hidden">{SEVERITY_LABEL[severity]} </span>
        {title !== undefined ? <p className="callout__title">{title}</p> : null}
        <div className="callout__body">{children}</div>
        {actions !== undefined ? <div className="callout__actions">{actions}</div> : null}
      </div>
    </div>
  )
}
