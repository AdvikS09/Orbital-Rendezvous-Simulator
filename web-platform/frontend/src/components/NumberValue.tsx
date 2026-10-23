import {
  NBSP,
  formatAngle,
  formatByUnit,
  formatDuration,
  formatDistance,
  formatFixed,
  formatNumber,
  formatPercent,
  formatScientific,
  formatSignificant,
  formatVelocity,
  rawValue,
} from '@/formatting/number'
import type { NumberFormatName } from '@/api/types'
import './NumberValue.css'

export interface NumberValueParts {
  body: string
  unit: string | null
}

export interface FormatValueOptions {
  unit?: string | null
  fmt?: NumberFormatName | null
  precision?: number | null
}

const DURATION_UNITS = ['min', 's', 'h', 'd'] as const
const DISTANCE_UNITS = ['km', 'm'] as const

function splitTrailingUnit(
  text: string,
  candidates: readonly string[],
): NumberValueParts {
  const ordered = candidates
    .filter((candidate) => candidate.length > 0)
    .sort((a, b) => b.length - a.length)

  for (const candidate of ordered) {
    const suffix = `${NBSP}${candidate}`
    if (text.endsWith(suffix)) {
      return { body: text.slice(0, text.length - suffix.length), unit: candidate }
    }
  }
  return { body: text, unit: null }
}

function isMissing(value: number | null | undefined): value is null | undefined {
  return value === null || value === undefined || !Number.isFinite(value)
}

export function formatValueParts(
  value: number | null | undefined,
  options: FormatValueOptions = {},
): NumberValueParts {
  const unit = options.unit ?? ''
  const fmt = options.fmt ?? null
  const precision = options.precision ?? null

  if (isMissing(value)) {
    return { body: formatNumber(null), unit: null }
  }

  switch (fmt) {
    case 'fixed':
      return { body: formatFixed(value, precision ?? 3), unit: unit || null }
    case 'integer':
      return { body: formatFixed(value, 0), unit: unit || null }
    case 'significant':
      return { body: formatSignificant(value, precision ?? 4), unit: unit || null }
    case 'scientific':
      return { body: formatScientific(value, precision ?? 4), unit: unit || null }
    case 'auto':
      return {
        body: formatNumber(value, { notation: 'auto', significant: precision ?? 4 }),
        unit: unit || null,
      }
    case 'percent':
      return { body: formatPercent(value, precision ?? 1), unit: null }
    case 'angle':
      return { body: formatAngle(value, precision ?? 2), unit: null }
    case 'duration':
      return splitTrailingUnit(
        formatDuration(value, precision === null ? {} : { decimals: precision }),
        DURATION_UNITS,
      )
    case 'distance':
      return splitTrailingUnit(
        formatDistance(value, precision === null ? {} : { significant: precision }),
        DISTANCE_UNITS,
      )
    case 'velocity':
      return splitTrailingUnit(formatVelocity(value, precision ?? 4), ['m/s'])
    default:
      break
  }

  return splitTrailingUnit(formatByUnit(value, unit, precision ?? 4), [
    ...DURATION_UNITS,
    ...DISTANCE_UNITS,
    'm/s',
    'm/s\u00B2',
    unit,
  ])
}

export function formatValueText(
  value: number | null | undefined,
  options: FormatValueOptions = {},
): string {
  const { body, unit } = formatValueParts(value, options)
  return unit ? `${body}${NBSP}${unit}` : body
}

export interface NumberValueProps {
  value: number | string | null
  unit?: string
  fmt?: NumberFormatName
  precision?: number | null
  align?: 'left' | 'right'
  muted?: boolean
  className?: string
}

export function NumberValue({
  value,
  unit,
  fmt,
  precision,
  align = 'left',
  muted = false,
  className,
}: NumberValueProps) {
  const numeric = typeof value === 'string' ? null : value
  const { body, unit: displayUnit } =
    typeof value === 'string'
      ? { body: value, unit: unit || null }
      : formatValueParts(value, { unit, fmt, precision })
  const missing = typeof value === 'string' ? false : isMissing(value)

  const classes = [
    'num',
    'number-value',
    align === 'right' ? 'num--right' : '',
    muted ? 'number-value--muted' : '',
    className ?? '',
  ]
    .filter(Boolean)
    .join(' ')

  const exact =
    missing || numeric === null
      ? undefined
      : unit
        ? `${rawValue(numeric)} ${unit}`
        : rawValue(numeric)

  return (
    <span className={classes} title={exact}>
      {body}
      {displayUnit ? (
        <span className="num--unit number-value__unit">
          {NBSP}
          {displayUnit}
        </span>
      ) : null}
    </span>
  )
}
