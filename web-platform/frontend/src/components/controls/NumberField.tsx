/**
 * A labelled numeric input.
 *
 * Three decisions worth stating, because they are easy to get wrong:
 *
 * 1. The control is `type="text"` with `inputMode="decimal"`, not
 *    `type="number"`. A number input silently discards `4.6e-5` in several
 *    browsers, hides the value behind spinner buttons, and reports an empty
 *    string for anything it cannot parse. Orbital parameters are routinely
 *    entered in scientific form, so the text is parsed here instead.
 * 2. Nothing is clamped. The engine owns its bounds and returns a message that
 *    explains which bound was hit and why; quietly snapping a typed value to a
 *    limit would hide the fact that the user asked for something else.
 * 3. Unparseable text is an error, never a zero. Typing `4.6e` mid-entry leaves
 *    the last valid value in place and shows what is wrong.
 */

import { useEffect, useId, useState } from 'react'
import type { ParamSpec } from '@/api/types'
import { Icon } from '@/components/ui/Icon'
import { NumberValue } from '@/components/NumberValue'
import { MINUS, rawValue } from '@/formatting/number'
import './NumberField.css'

/**
 * Parse text typed by a user into a number, or null if it is not one.
 *
 * Accepts plain decimals and scientific notation, and tolerates the Unicode
 * minus sign, because a value copied out of the results table carries one.
 */
export function parseNumericInput(text: string): number | null {
  const trimmed = text.trim().replace(new RegExp(MINUS, 'g'), '-')
  if (trimmed === '') return null
  const parsed = Number(trimmed)
  return Number.isFinite(parsed) ? parsed : null
}

/** The editable representation: full precision, no presentation formatting. */
export function toInputText(value: number): string {
  return rawValue(value)
}

export interface NumberFieldProps {
  param: ParamSpec
  value: number
  onChange: (value: number) => void
  /** Validation message from the server. Takes precedence over parse errors. */
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

  /*
   * While the user is typing, the text they have entered is the truth: a draft
   * of `1.2e` is not yet a number and must not be rewritten. Once the draft
   * parses, the parsed number is pushed upward and the draft is kept so the
   * caret does not jump. The draft is dropped whenever the value changes from
   * outside, which is what makes 'reset to defaults' visible in the field.
   */
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
    // A draft that parses is equivalent to the value it produced, so drop it
    // and let the canonical representation take over.
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
