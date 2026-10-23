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

export const TIMES = '×'

export const NBSP = ' '

export const MINUS = '−'

export type Notation = 'auto' | 'fixed' | 'significant' | 'scientific'

export interface FormatOptions {
  notation?: Notation
  decimals?: number
  significant?: number
  unit?: string
  signed?: boolean
  placeholder?: string
  autoLowerBound?: number
  autoUpperBound?: number
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

function splitScientific(
  value: number,
  significant: number,
): { mantissa: string; exponent: number } {
  if (value === 0) return { mantissa: (0).toFixed(Math.max(0, significant - 1)), exponent: 0 }
  const exponential = value.toExponential(Math.max(0, significant - 1))
  const [mantissa, exponent] = exponential.split('e')
  return { mantissa: mantissa ?? '0', exponent: Number(exponent ?? 0) }
}

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
      const precise = numeric.toPrecision(config.significant)
      if (precise.includes('e')) {
        const { mantissa, exponent } = splitScientific(numeric, config.significant)
        body = `${applyMinus(mantissa, config.unicodeMinus)}${NBSP}${TIMES}${NBSP}10${toSuperscript(exponent)}`
      } else {
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

export function formatScientific(
  value: number | null | undefined,
  significant = 4,
  unit?: string,
): string {
  return formatNumber(value, { notation: 'scientific', significant, unit })
}

export function formatFixed(
  value: number | null | undefined,
  decimals = 2,
  unit?: string,
): string {
  return formatNumber(value, { notation: 'fixed', decimals, unit })
}

export function formatSignificant(
  value: number | null | undefined,
  significant = 4,
  unit?: string,
): string {
  return formatNumber(value, { notation: 'significant', significant, unit })
}

export function formatPercent(
  value: number | null | undefined,
  decimals = 1,
): string {
  if (isMissing(value)) return DEFAULTS.placeholder
  return `${applyMinus((Number(value) * 100).toFixed(decimals), true)}%`
}

export function formatAngle(
  value: number | null | undefined,
  decimals = 2,
): string {
  if (isMissing(value)) return DEFAULTS.placeholder
  return `${applyMinus(Number(value).toFixed(decimals), true)}°`
}

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

export function formatAcceleration(
  value: number | null | undefined,
  significant = 4,
): string {
  return formatNumber(value, { notation: 'auto', significant, unit: 'm/s²' })
}

export function rawValue(value: number | null | undefined): string {
  if (value === null || value === undefined) return ''
  if (typeof value === 'number' && Number.isNaN(value)) return 'NaN'
  if (typeof value === 'number' && !Number.isFinite(value)) {
    return value > 0 ? 'Infinity' : '-Infinity'
  }
  return String(value)
}

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
