import { useCallback, useId, useRef } from 'react'
import type { KeyboardEvent, ReactNode } from 'react'
import { formatFixed } from '@/formatting/number'
import { Badge } from './Badge'
import { Icon } from './Icon'
import type { IconName } from './Icon'
import './Tabs.css'

export interface TabItem {
  value: string
  label: string
  icon?: IconName
  count?: number
  disabled?: boolean
}

export interface TabsProps {
  items: readonly TabItem[]
  value: string
  onChange: (value: string) => void
  /** Accessible name for the tab list, e.g. 'Result views'. */
  label: string
  /**
   * Prefix for the generated tab and panel ids. Pass the same value to every
   * TabPanel in the group so aria-controls and aria-labelledby line up.
   */
  idPrefix?: string
  className?: string
}

export function tabId(prefix: string, value: string): string {
  return `${prefix}-tab-${value}`
}

export function tabPanelId(prefix: string, value: string): string {
  return `${prefix}-panel-${value}`
}

/**
 * ARIA tab list with automatic activation: arrow keys move focus and select in
 * one step. That is the right pattern here because switching a tab only swaps
 * already-fetched results into view, so there is no cost to selecting as the
 * user arrows past.
 */
export function Tabs({ items, value, onChange, label, idPrefix, className }: TabsProps) {
  const generatedPrefix = useId()
  const prefix = idPrefix ?? generatedPrefix
  const refs = useRef(new Map<string, HTMLButtonElement>())

  const registerRef = useCallback((tabValue: string, node: HTMLButtonElement | null) => {
    if (node) refs.current.set(tabValue, node)
    else refs.current.delete(tabValue)
  }, [])

  const selectAt = useCallback(
    (index: number) => {
      const target = items[index]
      if (!target) return
      refs.current.get(target.value)?.focus()
      if (target.value !== value) onChange(target.value)
    },
    [items, onChange, value],
  )

  /** Move to the next enabled tab in `direction`, wrapping at the ends. */
  const move = useCallback(
    (start: number, direction: 1 | -1) => {
      const count = items.length
      if (count === 0) return
      for (let offset = 1; offset <= count; offset += 1) {
        const index = (((start + direction * offset) % count) + count) % count
        const candidate = items[index]
        if (candidate && candidate.disabled !== true) {
          selectAt(index)
          return
        }
      }
    },
    [items, selectAt],
  )

  /** Move to the first or last enabled tab. */
  const moveToEdge = useCallback(
    (edge: 'first' | 'last') => {
      const count = items.length
      for (let position = 0; position < count; position += 1) {
        const index = edge === 'first' ? position : count - 1 - position
        const candidate = items[index]
        if (candidate && candidate.disabled !== true) {
          selectAt(index)
          return
        }
      }
    },
    [items, selectAt],
  )

  function handleKeyDown(event: KeyboardEvent<HTMLButtonElement>, index: number) {
    switch (event.key) {
      case 'ArrowRight':
        event.preventDefault()
        move(index, 1)
        break
      case 'ArrowLeft':
        event.preventDefault()
        move(index, -1)
        break
      case 'Home':
        event.preventDefault()
        moveToEdge('first')
        break
      case 'End':
        event.preventDefault()
        moveToEdge('last')
        break
      default:
        break
    }
  }

  const classes = ['tabs', className ?? ''].filter(Boolean).join(' ')

  return (
    <div className={classes} role="tablist" aria-label={label}>
      {items.map((item, index) => {
        const selected = item.value === value
        return (
          <button
            key={item.value}
            type="button"
            role="tab"
            id={tabId(prefix, item.value)}
            aria-selected={selected}
            aria-controls={tabPanelId(prefix, item.value)}
            /* Roving tabindex: the group is one tab stop, arrows move within it. */
            tabIndex={selected ? 0 : -1}
            disabled={item.disabled === true}
            className="tabs__tab"
            ref={(node) => registerRef(item.value, node)}
            onClick={() => onChange(item.value)}
            onKeyDown={(event) => handleKeyDown(event, index)}
          >
            {item.icon ? <Icon name={item.icon} size={14} className="tabs__icon" /> : null}
            {item.label}
            {item.count !== undefined ? (
              <Badge tone={selected ? 'accent' : 'neutral'} mono srPrefix="Items: ">
                {formatFixed(item.count, 0)}
              </Badge>
            ) : null}
          </button>
        )
      })}
    </div>
  )
}

export interface TabPanelProps {
  /** Must match the `value` of the tab that controls this panel. */
  value: string
  /** The currently selected tab value. */
  selected: string
  /** The same prefix passed to the Tabs this panel belongs to. */
  idPrefix: string
  /**
   * Keep the panel mounted while hidden. Useful for a plot that is expensive
   * to rebuild; the default unmounts, so hidden panels cost nothing.
   */
  keepMounted?: boolean
  className?: string
  children: ReactNode
}

/** The region a tab controls. Hidden panels are removed from the tab order. */
export function TabPanel({
  value,
  selected,
  idPrefix,
  keepMounted = false,
  className,
  children,
}: TabPanelProps) {
  const isSelected = value === selected
  if (!isSelected && !keepMounted) return null

  const classes = ['tabs__panel', className ?? ''].filter(Boolean).join(' ')

  return (
    <div
      role="tabpanel"
      id={tabPanelId(idPrefix, value)}
      aria-labelledby={tabId(idPrefix, value)}
      hidden={!isSelected}
      /* Focusable so a keyboard user can reach panel content that has no
         interactive elements of its own, such as a table of results. */
      tabIndex={isSelected ? 0 : -1}
      className={classes}
    >
      {children}
    </div>
  )
}
