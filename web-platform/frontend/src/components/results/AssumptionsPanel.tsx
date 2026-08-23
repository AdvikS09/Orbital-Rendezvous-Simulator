/**
 * The conditions under which the numbers above are true.
 *
 * This panel is never collapsed by default and has no toggle that hides it. A
 * result separated from its assumptions is how a wrong belief forms: the figure
 * is remembered, the caveat is not, and six weeks later it is quoted in a
 * design review as though it were unconditional. Keeping both on screen at the
 * same time costs a little vertical space and prevents that.
 */

import { useId } from 'react'
import { Icon } from '@/components/ui/Icon'
import './AssumptionsPanel.css'

export interface AssumptionsPanelProps {
  assumptions: string[]
  limitations: string[]
  headingLevel?: 2 | 3 | 4
}

interface ColumnProps {
  id: string
  heading: string
  icon: 'shield' | 'warning'
  items: string[]
  emptyText: string
  variant: 'assumptions' | 'limitations'
}

function Column({ id, heading, icon, items, emptyText, variant }: ColumnProps) {
  return (
    <div className={`assumptions-panel__column assumptions-panel__column--${variant}`}>
      <h3 className="assumptions-panel__column-heading" id={id}>
        <Icon name={icon} size={15} />
        <span>{heading}</span>
      </h3>
      {items.length > 0 ? (
        <ul className="assumptions-panel__items" aria-labelledby={id}>
          {items.map((item, index) => (
            <li className="assumptions-panel__item" key={index}>
              {item}
            </li>
          ))}
        </ul>
      ) : (
        <p className="assumptions-panel__empty">{emptyText}</p>
      )}
    </div>
  )
}

export function AssumptionsPanel({
  assumptions,
  limitations,
  headingLevel = 2,
}: AssumptionsPanelProps) {
  const baseId = useId()

  if (assumptions.length === 0 && limitations.length === 0) return null

  const Heading = `h${headingLevel}` as const
  const panelId = `${baseId}-panel`
  const assumptionsId = `${baseId}-assumptions`
  const limitationsId = `${baseId}-limitations`

  return (
    <section className="assumptions-panel" aria-labelledby={panelId}>
      <Heading className="assumptions-panel__heading" id={panelId}>
        Conditions on this result
      </Heading>
      <div className="assumptions-panel__columns">
        <Column
          id={assumptionsId}
          heading="Assumptions behind this result"
          icon="shield"
          items={assumptions}
          emptyText="The engine reported no assumptions for this scenario."
          variant="assumptions"
        />
        <Column
          id={limitationsId}
          heading="Limitations you should know about"
          icon="warning"
          items={limitations}
          emptyText="The engine reported no limitations for this scenario."
          variant="limitations"
        />
      </div>
    </section>
  )
}
