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
  label: string
  idPrefix?: string
  className?: string
}

export function tabId(prefix: string, value: string): string {
  return `${prefix}-tab-${value}`
}

export function tabPanelId(prefix: string, value: string): string {
  return `${prefix}-panel-${value}`
}

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
  value: string
  selected: string
  idPrefix: string
  keepMounted?: boolean
  className?: string
  children: ReactNode
}

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
      tabIndex={isSelected ? 0 : -1}
      className={classes}
    >
      {children}
    </div>
  )
}
