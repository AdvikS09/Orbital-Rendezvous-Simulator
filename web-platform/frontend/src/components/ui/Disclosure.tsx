import type { ReactNode, SyntheticEvent } from 'react'
import { formatFixed } from '@/formatting/number'
import { Badge } from './Badge'
import { Icon } from './Icon'
import type { IconName } from './Icon'
import './Disclosure.css'

export interface DisclosureProps {
  title: ReactNode
  defaultOpen?: boolean
  count?: number
  icon?: IconName
  flush?: boolean
  onToggle?: (open: boolean) => void
  id?: string
  className?: string
  children: ReactNode
}

export function Disclosure({
  title,
  defaultOpen = false,
  count,
  icon,
  flush = false,
  onToggle,
  id,
  className,
  children,
}: DisclosureProps) {
  const classes = ['disclosure', className ?? ''].filter(Boolean).join(' ')
  const panelClasses = ['disclosure__panel', flush ? 'disclosure__panel--flush' : '']
    .filter(Boolean)
    .join(' ')

  function handleToggle(event: SyntheticEvent<HTMLDetailsElement>) {
    onToggle?.(event.currentTarget.open)
  }

  return (
    <details className={classes} id={id} open={defaultOpen} onToggle={handleToggle}>
      <summary className="disclosure__summary">
        <Icon name="chevronRight" size={14} className="disclosure__chevron" />
        {icon ? <Icon name={icon} size={14} className="disclosure__icon" /> : null}
        <span className="disclosure__title">{title}</span>
        {count !== undefined ? (
          <Badge tone="neutral" mono srPrefix="Items: ">
            {formatFixed(count, 0)}
          </Badge>
        ) : null}
      </summary>
      <div className={panelClasses}>{children}</div>
    </details>
  )
}
