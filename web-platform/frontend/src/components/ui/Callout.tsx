import type { ReactNode } from 'react'

import type { Severity } from '@/api/types'

import { Icon, SEVERITY_ICON } from './Icon'
import './Callout.css'

export type { Severity }

const SEVERITY_LABEL: Record<Severity, string> = {
  info: 'Note:',
  success: 'Success:',
  warning: 'Warning:',
  error: 'Error:',
}

export interface CalloutProps {
  severity: Severity
  title?: ReactNode
  children: ReactNode
  actions?: ReactNode
  id?: string
  className?: string
}

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
