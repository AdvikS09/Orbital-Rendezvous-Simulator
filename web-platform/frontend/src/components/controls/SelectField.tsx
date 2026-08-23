/**
 * A native `<select>` with label and help text.
 *
 * Native rather than a custom listbox: it is keyboard and screen-reader correct
 * on every platform for free, and it behaves properly on a phone. The option
 * sets here are short and closed, so there is nothing a custom widget would add.
 *
 * Numeric options are formatted through the shared formatter, because an option
 * list is display like any other. `<option>` cannot hold markup, so the text
 * form is used rather than the component.
 */

import { useId } from 'react'
import type { ParamSpec } from '@/api/types'
import { Icon } from '@/components/ui'
import { formatValueText } from '@/components/NumberValue'
import { rawValue } from '@/formatting/number'
import './SelectField.css'

/** A stable string key for an option, used as the DOM value. */
function optionKey(option: string | number): string {
  return typeof option === 'number' ? rawValue(option) : option
}

export interface SelectFieldProps {
  param: ParamSpec
  value: string | number
  onChange: (value: string | number) => void
  error?: string
  disabled?: boolean
}

export function SelectField({
  param,
  value,
  onChange,
  error,
  disabled = false,
}: SelectFieldProps) {
  const reactId = useId()
  const selectId = `${reactId}-${param.name}`
  const helpId = `${selectId}-help`
  const errorId = `${selectId}-error`

  const invalid = error !== null && error !== undefined && error !== ''

  const describedBy = [param.help ? helpId : '', invalid ? errorId : '']
    .filter(Boolean)
    .join(' ')

  const currentKey = optionKey(value)
  const known = param.options.some((option) => optionKey(option) === currentKey)

  function handleChange(key: string) {
    const chosen = param.options.find((option) => optionKey(option) === key)
    if (chosen !== undefined) onChange(chosen)
  }

  return (
    <div className={`select-field${invalid ? ' select-field--invalid' : ''}`}>
      <label className="select-field__label" htmlFor={selectId}>
        {param.label}
        {param.unit ? <span className="visually-hidden"> in {param.unit}</span> : null}
      </label>

      <div className="select-field__shell">
        <select
          id={selectId}
          name={param.name}
          className="select-field__select"
          value={known ? currentKey : ''}
          disabled={disabled}
          aria-invalid={invalid || undefined}
          aria-describedby={describedBy || undefined}
          onChange={(event) => handleChange(event.target.value)}
        >
          {/*
            Only present when the current value is not one of the options, which
            happens when a saved run is loaded against a changed scenario. It is
            never selectable, so the mismatch is visible rather than silently
            rewritten.
          */}
          {known ? null : (
            <option value="" disabled>
              Select an option
            </option>
          )}
          {param.options.map((option) => (
            <option key={optionKey(option)} value={optionKey(option)}>
              {typeof option === 'number'
                ? formatValueText(option, {
                    unit: param.unit,
                    fmt: param.fmt,
                    precision: param.precision,
                  })
                : option}
            </option>
          ))}
        </select>
        <Icon name="chevronDown" size={14} className="select-field__chevron" />
      </div>

      {param.help ? (
        <p className="select-field__help" id={helpId}>
          {param.help}
        </p>
      ) : null}

      {invalid ? (
        <p className="select-field__error" id={errorId} role="alert">
          <Icon name="warning" size={13} className="select-field__error-icon" />
          <span>{error}</span>
        </p>
      ) : null}
    </div>
  )
}
