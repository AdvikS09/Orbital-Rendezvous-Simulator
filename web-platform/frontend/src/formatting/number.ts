/**
 * The single shared number-formatting utility.
 *
 * Every user-facing number in this application goes through this module. The
 * Streamlit version rendered values with Python's `%g`, which produced strings
 * like `2e-12` and `1.1060555202205516e-13` in the interface. That is machine
 * notation, not scientific notation, and it is hard to read at a glance.
 *
 * Here, a value like 2e-12 is rendered as `2.0 x 10^-12` using real Unicode
 * superscripts, and the caller can choose fixed, significant-figure, scientific
 * or automatic presentation.
 *
 * Two rules that matter:
 *
 * 1. Presentation formatting NEVER touches exported data. `rawValue()` returns
 *    the full-precision machine representation for CSV and JSON, so a rounded
 *    display value can never leak into someone's analysis.
 * 2. Formatting never invents precision. Asking for 6 significant figures from
 *    a value only known to 3 does not make it more accurate, so callers pass
 *    the precision that is actually meaningful for the quantity.
 */

/** Unicode superscript digits, minus sign and decimal point. */
const SUPERSCRIPTS: Record<string, string> = {
  '0': '⁰',
  '1': '¹',
  '2': '²',
  '3': '³',
  '4': '⁴',
  '5': '⁵',
  '6': '⁶',
  '7': '⁷',
  '8': '⁸',
  '9': '⁹',
  '-': '⁻',
  '+': '⁺',
}

/** The multiplication sign used in scientific notation (never the letter x). */
export const TIMES = '×'

/** Non-breaking space, so a value never wraps away from its unit. */
export const NBSP = ' '

/** The minus sign U+2212, which aligns with digits, not the hyphen. */
export const MINUS = '−'

export type Notation = 'auto' | 'fixed' | 'significant' | 'scientific'

export interface FormatOptions {
  /** How to present the number. Default 'auto'. */
  notation?: Notation
  /** Decimal places for 'fixed'. Default 3. */
  decimals?: number
  /** Significant figures for 'significant' and 'scientific'. Default 4. */
  significant?: number
  /** Unit appended after a non-breaking space. */
  unit?: string
  /** Render the sign for positive values too. */
  signed?: boolean
  /** What to show for null, undefined or NaN. Default an em dash. */
  placeholder?: string
  /**
   * In 'auto', values whose magnitude falls outside
   * [autoLowerBound, autoUpperBound) switch to scientific notation.
   */
  autoLowerBound?: number
  autoUpperBound?: number
  /** Use the proper Unicode minus sign rather than a hyphen. Default true. */
  unicodeMinus?: boolean
}

const DEFAULTS: Required<Omit<FormatOptions, 'unit'>> = {
  notation: 'auto',
  decimals: 3,
  significant: 4,
  signed: false,
  placeholder: '—',
  autoLowerBound: 1e-3,
  autoUpperBound: 1e5,
  unicodeMinus: true,
}

/** Convert an integer exponent to Unicode superscript, e.g. -12 -> superscript. */
export function toSuperscript(exponent: number | string): string {
  return String(exponent)
    .split('')
    .map((character) => SUPERSCRIPTS[character] ?? character)
    .join('')
}

function isMissing(value: unknown): boolean {
  return (
    value === null ||
    value === undefined ||
    (typeof value === 'number' && !Number.isFinite(value))
  )
}

function applyMinus(text: string, unicodeMinus: boolean): string {
  return unicodeMinus ? text.replace(/-/g, MINUS) : text
}

function withSign(text: string, value: number, signed: boolean): string {
  if (!signed || value < 0) return text
  return `+${text}`
}

function withUnit(text: string, unit?: string): string {
  return unit ? `${text}${NBSP}${unit}` : text
}

/**
 * Split a number into mantissa and exponent at the requested significant
 * figures. Returns strings so trailing zeros are preserved.
 */
function splitScientific(
  value: number,
  significant: number,
): { mantissa: string; exponent: number } {
  if (value === 0) return { mantissa: (0).toFixed(Math.max(0, significant - 1)), exponent: 0 }
  const exponential = value.toExponential(Math.max(0, significant - 1))
  const [mantissa, exponent] = exponential.split('e')
  return { mantissa: mantissa ?? '0', exponent: Number(exponent ?? 0) }
}

/**
 * Format a number for display.
 *
 * Examples:
 *   formatNumber(2e-12)                     -> '2.000 x 10^-12' (with real superscripts)
 *   formatNumber(5553.62, { decimals: 2 })  -> '5553.62'
 *   formatNumber(2.3257, { unit: 'm/s' })   -> '2.326 m/s'
 */
export function formatNumber(value: number | null | undefined, options: FormatOptions = {}): string {
  const config = { ...DEFAULTS, ...options }
  if (isMissing(value)) return config.placeholder

  const numeric = Number(value)
  const magnitude = Math.abs(numeric)

  let notation = config.notation
  if (notation === 'auto') {
    if (numeric === 0) {
      notation = 'fixed'
    } else if (magnitude < config.autoLowerBound || magnitude >= config.autoUpperBound) {
      notation = 'scientific'
    } else {
      notation = 'significant'
    }
  }

  let body: string
  switch (notation) {
    case 'scientific': {
      const { mantissa, exponent } = splitScientific(numeric, config.significant)
      const mantissaText = applyMinus(mantissa, config.unicodeMinus)
      // An exponent of zero adds noise; show the mantissa alone.
      body =
        exponent === 0
          ? mantissaText
          : `${mantissaText}${NBSP}${TIMES}${NBSP}10${toSuperscript(exponent)}`
      break
    }
    case 'fixed':
      body = applyMinus(numeric.toFixed(config.decimals), config.unicodeMinus)
      break
    case 'significant': {
      // toPrecision can itself return exponential form; fall back cleanly.
      const precise = numeric.toPrecision(config.significant)
      if (precise.includes('e')) {
        const { mantissa, exponent } = splitScientific(numeric, config.significant)
        body = `${applyMinus(mantissa, config.unicodeMinus)}${NBSP}${TIMES}${NBSP}10${toSuperscript(exponent)}`
      } else {
        // Strip trailing zeros introduced by toPrecision, but keep integers whole.
        const trimmed = precise.includes('.') ? precise.replace(/\.?0+$/, '') : precise
        body = applyMinus(trimmed === '' || trimmed === '-' ? '0' : trimmed, config.unicodeMinus)
      }
      break
    }
    default:
      body = applyMinus(String(numeric), config.unicodeMinus)
  }

  return withUnit(withSign(body, numeric, config.signed), options.unit)
}

/** Always scientific, regardless of magnitude. */
export function formatScientific(
  value: number | null | undefined,
  significant = 4,
  unit?: string,
): string {
  return formatNumber(value, { notation: 'scientific', significant, unit })
}

/** Fixed decimal places. */
export function formatFixed(
  value: number | null | undefined,
  decimals = 2,
  unit?: string,
): string {
  return formatNumber(value, { notation: 'fixed', decimals, unit })
}

/** Significant figures without switching to exponential form unless forced. */
export function formatSignificant(
  value: number | null | undefined,
  significant = 4,
  unit?: string,
): string {
  return formatNumber(value, { notation: 'significant', significant, unit })
}

/** A fraction in [0, 1] rendered as a percentage. */
export function formatPercent(
  value: number | null | undefined,
  decimals = 1,
): string {
  if (isMissing(value)) return DEFAULTS.placeholder
  return `${applyMinus((Number(value) * 100).toFixed(decimals), true)}%`
}

/** An angle in degrees, with the degree sign. */
export function formatAngle(
  value: number | null | undefined,
  decimals = 2,
): string {
  if (isMissing(value)) return DEFAULTS.placeholder
  return `${applyMinus(Number(value).toFixed(decimals), true)}°`
}

/**
 * A duration in seconds, rendered in whichever unit reads most naturally.
 * Orbital periods are minutes; long propagations are hours.
 */
export function formatDuration(
  value: number | null | undefined,
  options: { forceUnit?: 's' | 'min' | 'h' | 'd'; decimals?: number } = {},
): string {
  if (isMissing(value)) return DEFAULTS.placeholder
  const seconds = Number(value)
  const magnitude = Math.abs(seconds)
  const decimals = options.decimals ?? 1

  const unit =
    options.forceUnit ??
    (magnitude < 120 ? 's' : magnitude < 7200 ? 'min' : magnitude < 172800 ? 'h' : 'd')

  const divisor = unit === 's' ? 1 : unit === 'min' ? 60 : unit === 'h' ? 3600 : 86400
  const scaled = seconds / divisor
  const body =
    unit === 's' && Math.abs(scaled) < 1e-3
      ? formatScientific(scaled, 3)
      : applyMinus(scaled.toFixed(decimals), true)
  return `${body}${NBSP}${unit}`
}

/**
 * A distance in metres, rendered in m or km as appropriate.
 * Sub-millimetre values fall back to scientific notation in metres, because
 * rendering 1.5e-7 m as "0.000 km" would hide the result.
 */
export function formatDistance(
  value: number | null | undefined,
  options: { forceUnit?: 'm' | 'km'; decimals?: number; significant?: number } = {},
): string {
  if (isMissing(value)) return DEFAULTS.placeholder
  const metres = Number(value)
  const magnitude = Math.abs(metres)

  if (options.forceUnit === 'km' || (!options.forceUnit && magnitude >= 10_000)) {
    return formatNumber(metres / 1000, {
      notation: 'significant',
      significant: options.significant ?? 4,
      unit: 'km',
    })
  }
  if (magnitude !== 0 && magnitude < 1e-3) {
    return formatScientific(metres, options.significant ?? 3, 'm')
  }
  return formatNumber(metres, {
    notation: options.decimals === undefined ? 'significant' : 'fixed',
    significant: options.significant ?? 4,
    decimals: options.decimals ?? 3,
    unit: 'm',
  })
}

/** A velocity in metres per second. */
export function formatVelocity(
  value: number | null | undefined,
  significant = 4,
): string {
  if (isMissing(value)) return DEFAULTS.placeholder
  const speed = Number(value)
  if (speed !== 0 && Math.abs(speed) < 1e-3) {
    return formatScientific(speed, 3, 'm/s')
  }
  return formatSignificant(speed, significant, 'm/s')
}

/** An acceleration in metres per second squared. */
export function formatAcceleration(
  value: number | null | undefined,
  significant = 4,
): string {
  return formatNumber(value, { notation: 'auto', significant, unit: 'm/s²' })
}

/**
 * The machine representation, for CSV and JSON export.
 *
 * Deliberately plain: no superscripts, no unicode minus, no thousands
 * separators, full precision. Presentation formatting must never reach an
 * export, because someone will load it into a spreadsheet and compute with it.
 */
export function rawValue(value: number | null | undefined): string {
  if (value === null || value === undefined) return ''
  if (typeof value === 'number' && Number.isNaN(value)) return 'NaN'
  if (typeof value === 'number' && !Number.isFinite(value)) {
    return value > 0 ? 'Infinity' : '-Infinity'
  }
  return String(value)
}

/**
 * Format a value the way its metric label implies, given a unit string coming
 * from the API. Keeps result tables consistent without every call site
 * repeating the same decision.
 */
export function formatByUnit(
  value: number | null | undefined,
  unit: string | null | undefined,
  significant = 4,
): string {
  if (isMissing(value)) return DEFAULTS.placeholder
  switch ((unit ?? '').trim()) {
    case 'm':
      return formatDistance(value, { significant })
    case 'km':
      return formatNumber(value, { notation: 'significant', significant, unit: 'km' })
    case 'm/s':
      return formatVelocity(value, significant)
    case 'm/s^2':
    case 'm/s²':
      return formatAcceleration(value, significant)
    case 's':
      return formatDuration(value)
    case 'deg':
      return formatAngle(value)
    case '%':
      return formatPercent(value)
    case '':
    case '-':
      return formatNumber(value, { notation: 'auto', significant })
    default:
      return formatNumber(value, { notation: 'auto', significant, unit: unit ?? undefined })
  }
}
