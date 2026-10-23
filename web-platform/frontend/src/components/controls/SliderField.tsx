import { useEffect, useId, useState } from 'react'
import type { ParamSpec } from '@/api/types'
import { Icon } from '@/components/ui'
import { NumberValue, formatValueText } from '@/components/NumberValue'
import { NumberField, parseNumericInput, toInputText } from './NumberField'
import './SliderField.css'

export interface SliderFieldProps {
  param: ParamSpec
  value: number
  onChange: (value: number) => void
  error?: string
  disabled?: boolean
}

export function SliderField({
  param,
  value,
  onChange,
  error,
  disabled = false,
}: SliderFieldProps) {
  const reactId = useId()
  const sliderId = `${reactId}-${param.name}`
  const entryId = `${sliderId}-entry`
  const helpId = `${sliderId}-help`
  const noteId = `${sliderId}-note`
  const errorId = `${sliderId}-error`

  const [draft, setDraft] = useState<string | null>(null)

  useEffect(() => {
    setDraft((current) =>
      current !== null && parseNumericInput(current) === value ? current : null,
    )
  }, [value])

  const minimum = param.minimum ?? param.hard_minimum
  const maximum = param.maximum ?? param.hard_maximum

  if (minimum === null || maximum === null || !(maximum > minimum)) {
    return (
      <NumberField
        param={param}
        value={value}
        onChange={onChange}
        error={error}
        disabled={disabled}
      />
    )
  }

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

  const outsideRange = value < minimum || value > maximum
  const trackValue = Math.min(maximum, Math.max(minimum, value))

  const describedBy = [
    param.help ? helpId : '',
    outsideRange ? noteId : '',
    invalid ? errorId : '',
  ]
    .filter(Boolean)
    .join(' ')

  const valueText = formatValueText(value, {
    unit: param.unit,
    fmt: param.fmt,
    precision: param.precision,
  })

  function handleEntry(next: string) {
    setDraft(next)
    const candidate = parseNumericInput(next)
    if (candidate !== null) onChange(candidate)
  }

  function handleBlur() {
    if (draft !== null && parseNumericInput(draft) !== null) setDraft(null)
  }

  return (
    <div className={`slider-field${invalid ? ' slider-field--invalid' : ''}`}>
      <div className="slider-field__head">
        <label className="slider-field__label" htmlFor={sliderId}>
          {param.label}
          {param.unit ? <span className="visually-hidden"> in {param.unit}</span> : null}
        </label>

        <div className="slider-field__value">
          <label className="visually-hidden" htmlFor={entryId}>
            {param.label} exact value
            {param.unit ? ` in ${param.unit}` : ''}
          </label>
          <input
            id={entryId}
            name={param.name}
            className="slider-field__entry num num--right"
            type="text"
            inputMode="decimal"
            autoComplete="off"
            spellCheck={false}
            size={10}
            value={text}
            disabled={disabled}
            aria-invalid={invalid || undefined}
            aria-describedby={describedBy || undefined}
            onChange={(event) => handleEntry(event.target.value)}
            onBlur={handleBlur}
          />
          {param.unit ? (
            <span className="slider-field__unit" aria-hidden="true">
              {param.unit}
            </span>
          ) : null}
        </div>
      </div>

      <input
        id={sliderId}
        className="slider-field__track"
        type="range"
        min={minimum}
        max={maximum}
        step={param.step ?? 'any'}
        value={toInputText(trackValue)}
        disabled={disabled}
        aria-valuetext={valueText}
        aria-describedby={describedBy || undefined}
        onChange={(event) => {
          setDraft(null)
          const candidate = parseNumericInput(event.target.value)
          if (candidate !== null) onChange(candidate)
        }}
      />

      <div className="slider-field__ends">
        <NumberValue
          value={minimum}
          unit={param.unit}
          fmt={param.fmt}
          precision={param.precision}
          muted
        />
        <NumberValue
          value={maximum}
          unit={param.unit}
          fmt={param.fmt}
          precision={param.precision}
          align="right"
          muted
        />
      </div>

      {param.help ? (
        <p className="slider-field__help" id={helpId}>
          {param.help}
        </p>
      ) : null}

      {outsideRange ? (
        <p className="slider-field__note" id={noteId}>
          Outside the slider range. The value is kept as typed and checked by the
          engine.
        </p>
      ) : null}

      {invalid ? (
        <p className="slider-field__error" id={errorId} role="alert">
          <Icon name="warning" size={13} className="slider-field__error-icon" />
          <span>{message}</span>
        </p>
      ) : null}
    </div>
  )
}
