import { useId } from 'react'
import type { ParamSpec } from '@/api/types'
import { Icon } from '@/components/ui'
import './CheckboxField.css'

export interface CheckboxFieldProps {
  param: ParamSpec
  value: boolean
  onChange: (value: boolean) => void
  error?: string
  disabled?: boolean
}

export function CheckboxField({
  param,
  value,
  onChange,
  error,
  disabled = false,
}: CheckboxFieldProps) {
  const reactId = useId()
  const inputId = `${reactId}-${param.name}`
  const helpId = `${inputId}-help`
  const errorId = `${inputId}-error`

  const invalid = error !== null && error !== undefined && error !== ''

  const describedBy = [param.help ? helpId : '', invalid ? errorId : '']
    .filter(Boolean)
    .join(' ')

  return (
    <div className={`checkbox-field${invalid ? ' checkbox-field--invalid' : ''}`}>
      <div className="checkbox-field__row">
        <input
          id={inputId}
          name={param.name}
          className="checkbox-field__input"
          type="checkbox"
          checked={value}
          disabled={disabled}
          aria-invalid={invalid || undefined}
          aria-describedby={describedBy || undefined}
          onChange={(event) => onChange(event.target.checked)}
        />
        <label className="checkbox-field__label" htmlFor={inputId}>
          {param.label}
        </label>
      </div>

      {param.help ? (
        <p className="checkbox-field__help" id={helpId}>
          {param.help}
        </p>
      ) : null}

      {invalid ? (
        <p className="checkbox-field__error" id={errorId} role="alert">
          <Icon name="warning" size={13} className="checkbox-field__error-icon" />
          <span>{error}</span>
        </p>
      ) : null}
    </div>
  )
}
