import { useEffect, useId, useState } from 'react'
import type { ParamSpec } from '@/api/types'
import { Icon } from '@/components/ui'
import { NumberValue } from '@/components/NumberValue'
import { MINUS, rawValue } from '@/formatting/number'
import './NumberField.css'

export function parseNumericInput(text: string): number | null {
  const trimmed = text.trim().replace(new RegExp(MINUS, 'g'), '-')
  if (trimmed === '') return null
  const parsed = Number(trimmed)
  return Number.isFinite(parsed) ? parsed : null
}

export function toInputText(value: number): string {
  return rawValue(value)
}

export interface NumberFieldProps {
  param: ParamSpec
  value: number
  onChange: (value: number) => void
  error?: string
  disabled?: boolean
}

export function NumberField({
  param,
  value,
  onChange,
  error,
  disabled = false,
}: NumberFieldProps) {
  const reactId = useId()
  const inputId = `${reactId}-${param.name}`
  const helpId = `${inputId}-help`
  const boundsId = `${inputId}-bounds`
  const errorId = `${inputId}-error`

  const [draft, setDraft] = useState<string | null>(null)

  useEffect(() => {
    setDraft((current) =>
      current !== null && parseNumericInput(current) === value ? current : null,
    )
  }, [value])

  const text = draft ?? toInputText(value)
  const parsed = draft === null ? value : parseNumericInput(draft)

  const parseMessage =
    parsed === null
      ? draft !== null && draft.trim() === ''
        ? 'Enter a value.'
        : 'Not a number. Use digits, a decimal point, or scientific form such as 4.6e-5.'
      : null

  const message = error ?? parseMessage
  const invalid = message !== null && message !== undefined

  const hasLowerBound = param.hard_minimum !== null
  const hasUpperBound = param.hard_maximum !== null
  const showBounds = hasLowerBound || hasUpperBound

  const describedBy = [
    param.help ? helpId : '',
    showBounds ? boundsId : '',
    invalid ? errorId : '',
  ]
    .filter(Boolean)
    .join(' ')

  function handleChange(next: string) {
    setDraft(next)
    const candidate = parseNumericInput(next)
    if (candidate !== null) onChange(candidate)
  }

  function handleBlur() {
    if (draft !== null && parseNumericInput(draft) !== null) setDraft(null)
  }

  return (
    <div className={`number-field${invalid ? ' number-field--invalid' : ''}`}>
      <label className="number-field__label" htmlFor={inputId}>
        {param.label}
        {param.unit ? <span className="visually-hidden"> in {param.unit}</span> : null}
      </label>

      <div className="number-field__shell">
        <input
          id={inputId}
          name={param.name}
          className="number-field__input num"
          type="text"
          inputMode="decimal"
          autoComplete="off"
          spellCheck={false}
          value={text}
          disabled={disabled}
          aria-invalid={invalid || undefined}
          aria-describedby={describedBy || undefined}
          onChange={(event) => handleChange(event.target.value)}
          onBlur={handleBlur}
        />
        {param.unit ? (
          <span className="number-field__unit" aria-hidden="true">
            {param.unit}
          </span>
        ) : null}
      </div>

      {param.help ? (
        <p className="number-field__help" id={helpId}>
          {param.help}
        </p>
      ) : null}

      {showBounds ? (
        <p className="number-field__bounds" id={boundsId}>
          {hasLowerBound && hasUpperBound ? (
            <>
              <span className="number-field__bounds-label">Accepted range</span>
              <NumberValue
                value={param.hard_minimum}
                unit={param.unit}
                fmt={param.fmt}
                precision={param.precision}
                muted
              />
              <span className="number-field__bounds-to">to</span>
              <NumberValue
                value={param.hard_maximum}
                unit={param.unit}
                fmt={param.fmt}
                precision={param.precision}
                muted
              />
            </>
          ) : hasLowerBound ? (
            <>
              <span className="number-field__bounds-label">Minimum</span>
              <NumberValue
                value={param.hard_minimum}
                unit={param.unit}
                fmt={param.fmt}
                precision={param.precision}
                muted
              />
            </>
          ) : (
            <>
              <span className="number-field__bounds-label">Maximum</span>
              <NumberValue
                value={param.hard_maximum}
                unit={param.unit}
                fmt={param.fmt}
                precision={param.precision}
                muted
              />
            </>
          )}
        </p>
      ) : null}

      {invalid ? (
        <p className="number-field__error" id={errorId} role="alert">
          <Icon name="warning" size={13} className="number-field__error-icon" />
          <span>{message}</span>
        </p>
      ) : null}
    </div>
  )
}
