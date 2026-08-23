/**
 * The one component that puts a number on the screen.
 *
 * Nothing else in the application may render a numeric value directly. Every
 * metric, bound, axis end label and input echo goes through here, which is what
 * keeps `2e-12` from ever reaching a user: it arrives as a proper scientific
 * rendering with Unicode superscripts, and its unit sits in its own span so it
 * can be styled down without shrinking the digits.
 *
 * The element carries a `title` holding the full-precision machine value, so a
 * displayed figure rounded to four significant figures is always one hover away
 * from the number the solver actually produced.
 */

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

/** A formatted number split from the unit it is measured in. */
export interface NumberValueParts {
  /** Digits, sign, and any scientific mantissa and exponent. */
  body: string
  /** The unit as it should be displayed, or null when there is none. */
  unit: string | null
}

export interface FormatValueOptions {
  unit?: string | null
  fmt?: NumberFormatName | null
  precision?: number | null
}

/**
 * Units a formatter may choose for itself. `formatDuration` picks between
 * seconds and days depending on magnitude, and `formatDistance` between metres
 * and kilometres, so the unit shown is not necessarily the unit supplied.
 */
const DURATION_UNITS = ['min', 's', 'h', 'd'] as const
const DISTANCE_UNITS = ['km', 'm'] as const

/**
 * Peel a trailing unit off a formatted string.
 *
 * Only the listed candidates are considered, and only when preceded by the
 * non-breaking space the formatter uses to attach a unit. Scientific notation
 * contains non-breaking spaces of its own, so splitting on the last one blindly
 * would mistake an exponent for a unit.
 */
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

/**
 * Format a value into its display body and unit.
 *
 * Exported because a handful of places need the formatted text rather than an
 * element: `aria-valuetext` on a slider, and the label of a `<select>` option,
 * neither of which can hold markup.
 */
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
    // The next three carry their own symbol, glued to the digits without a
    // space. Repeating the supplied unit beside them would double it up.
    case 'percent':
      return { body: formatPercent(value, precision ?? 1), unit: null }
    case 'angle':
      return { body: formatAngle(value, precision ?? 2), unit: null }
    // These choose their own unit by magnitude, so it is read back off the
    // formatted string rather than assumed.
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

  // No explicit format: the unit decides the presentation.
  return splitTrailingUnit(formatByUnit(value, unit, precision ?? 4), [
    ...DURATION_UNITS,
    ...DISTANCE_UNITS,
    'm/s',
    'm/s\u00B2',
    unit,
  ])
}

/** The same formatting as a single string, for attributes and option labels. */
export function formatValueText(
  value: number | null | undefined,
  options: FormatValueOptions = {},
): string {
  const { body, unit } = formatValueParts(value, options)
  return unit ? `${body}${NBSP}${unit}` : body
}

export interface NumberValueProps {
  value: number | null
  /** The unit the value is supplied in, e.g. 'm', 'm/s', 's'. */
  unit?: string
  /** Presentation chosen by the API. Omit to let the unit decide. */
  fmt?: NumberFormatName
  /** Decimal places or significant figures, depending on the format. */
  precision?: number | null
  /** Right alignment for numeric table columns. */
  align?: 'left' | 'right'
  /** Secondary emphasis, for bounds and supporting figures. */
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
  const { body, unit: displayUnit } = formatValueParts(value, { unit, fmt, precision })
  const missing = isMissing(value)

  const classes = [
    'num',
    'number-value',
    align === 'right' ? 'num--right' : '',
    muted ? 'number-value--muted' : '',
    className ?? '',
  ]
    .filter(Boolean)
    .join(' ')

  // The exact value, unrounded, for anyone who needs to check a displayed
  // figure against the solver output.
  const exact = missing ? undefined : unit ? `${rawValue(value)} ${unit}` : rawValue(value)

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
