/**
 * Dispatch a single `ParamSpec` to the control that suits its kind.
 *
 * The parameter set is defined by the backend, so a value arriving here is
 * typed only as `number | string | boolean` and may be missing entirely when a
 * scenario gains a parameter between one visit and the next. Each branch
 * coerces defensively and falls back to the spec's own default, so a control
 * always renders something the user can act on rather than an empty box.
 */

import type { ParamSpec } from '@/api/types'
import { CheckboxField } from './CheckboxField'
import { NumberField } from './NumberField'
import { SelectField } from './SelectField'
import { SliderField } from './SliderField'

export type ParamValue = number | string | boolean

/** Coerce to a finite number, falling back when the value is not usable. */
function toNumber(value: ParamValue | undefined, fallback: number): number {
  if (typeof value === 'number') return Number.isFinite(value) ? value : fallback
  if (typeof value === 'string') {
    const parsed = Number(value.trim())
    if (value.trim() !== '' && Number.isFinite(parsed)) return parsed
  }
  if (typeof value === 'boolean') return value ? 1 : 0
  return fallback
}

function toBoolean(value: ParamValue | undefined, fallback: boolean): boolean {
  if (typeof value === 'boolean') return value
  if (typeof value === 'number') return value !== 0
  if (typeof value === 'string') {
    const lowered = value.trim().toLowerCase()
    if (lowered === 'true' || lowered === '1' || lowered === 'yes') return true
    if (lowered === 'false' || lowered === '0' || lowered === 'no') return false
  }
  return fallback
}

function toChoice(
  value: ParamValue | undefined,
  fallback: string | number,
): string | number {
  if (typeof value === 'string' || typeof value === 'number') return value
  return fallback
}

export interface ParamControlProps {
  param: ParamSpec
  value: ParamValue | undefined
  onChange: (value: ParamValue) => void
  /** Message for this field, usually from the server's validation response. */
  error?: string
  disabled?: boolean
}

export function ParamControl({
  param,
  value,
  onChange,
  error,
  disabled = false,
}: ParamControlProps) {
  switch (param.kind) {
    case 'checkbox':
      return (
        <CheckboxField
          param={param}
          value={toBoolean(value, toBoolean(param.default, false))}
          onChange={onChange}
          error={error}
          disabled={disabled}
        />
      )

    case 'select': {
      const firstOption = param.options[0]
      const fallback = toChoice(param.default, firstOption ?? '')
      return (
        <SelectField
          param={param}
          value={toChoice(value, fallback)}
          onChange={onChange}
          error={error}
          disabled={disabled}
        />
      )
    }

    case 'slider':
      return (
        <SliderField
          param={param}
          value={toNumber(value, toNumber(param.default, 0))}
          onChange={onChange}
          error={error}
          disabled={disabled}
        />
      )

    case 'number':
    default:
      return (
        <NumberField
          param={param}
          value={toNumber(value, toNumber(param.default, 0))}
          onChange={onChange}
          error={error}
          disabled={disabled}
        />
      )
  }
}
