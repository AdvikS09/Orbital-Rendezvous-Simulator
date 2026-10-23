import { useId } from 'react'
import type { ReactNode } from 'react'
import { Icon } from './Icon'
import type { IconName } from './Icon'
import './Panel.css'

export interface PanelProps {
  title?: ReactNode
  subtitle?: ReactNode
  icon?: IconName
  actions?: ReactNode
  footer?: ReactNode
  padded?: boolean
  headingLevel?: 2 | 3 | 4
  flush?: boolean
  id?: string
  className?: string
  children: ReactNode
}

export function Panel({
  title,
  subtitle,
  icon,
  actions,
  footer,
  padded = true,
  headingLevel = 2,
  flush = false,
  id,
  className,
  children,
}: PanelProps) {
  const generatedId = useId()
  const titleId = `${generatedId}-title`
  const Heading = `h${headingLevel}` as 'h2' | 'h3' | 'h4'

  const classes = ['panel', flush ? 'panel--flush' : '', className ?? '']
    .filter(Boolean)
    .join(' ')

  const bodyClasses = ['panel__body', padded ? 'panel__body--padded' : '']
    .filter(Boolean)
    .join(' ')

  const hasHeader = title !== undefined || actions !== undefined

  const inner = (
    <>
      {hasHeader ? (
        <div className="panel__header">
          <div className="panel__heading-group">
            {title !== undefined ? (
              <Heading className="panel__title" id={titleId}>
                {icon ? <Icon name={icon} size={15} className="panel__title-icon" /> : null}
                {title}
              </Heading>
            ) : null}
            {subtitle !== undefined ? <p className="panel__subtitle">{subtitle}</p> : null}
          </div>
          {actions !== undefined ? <div className="panel__actions">{actions}</div> : null}
        </div>
      ) : null}
      <div className={bodyClasses}>{children}</div>
      {footer !== undefined ? <div className="panel__footer">{footer}</div> : null}
    </>
  )

  if (title === undefined) {
    return (
      <div className={classes} id={id}>
        {inner}
      </div>
    )
  }

  return (
    <section className={classes} id={id} aria-labelledby={titleId}>
      {inner}
    </section>
  )
}
