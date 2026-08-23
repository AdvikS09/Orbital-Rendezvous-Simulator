import { describe, expect, it } from 'vitest'

import {
  MINUS,
  NBSP,
  TIMES,
  formatAcceleration,
  formatAngle,
  formatByUnit,
  formatDistance,
  formatDuration,
  formatFixed,
  formatNumber,
  formatPercent,
  formatScientific,
  formatSignificant,
  formatVelocity,
  rawValue,
  toSuperscript,
} from '../src/formatting/number'

/**
 * The behaviour these tests pin down was a specific defect in the Streamlit
 * version: energy drift rendered as `2e-12` rather than as scientific
 * notation. Anything that reintroduces machine notation into the interface
 * should fail here.
 */

describe('superscripts', () => {
  it('converts digits and a negative sign', () => {
    expect(toSuperscript(-12)).toBe('⁻¹²')
    expect(toSuperscript(3)).toBe('³')
    expect(toSuperscript(0)).toBe('⁰')
    expect(toSuperscript(105)).toBe('¹⁰⁵')
  })
})

describe('scientific notation', () => {
  it('renders 2e-12 as proper scientific notation, never as machine notation', () => {
    const output = formatScientific(2e-12, 2)
    expect(output).toBe(`2.0${NBSP}${TIMES}${NBSP}10⁻¹²`)
    expect(output).not.toContain('e-')
    expect(output).not.toContain('E-')
  })

  it('renders the real energy-drift value from the engine readably', () => {
    // tests/baseline_engine_outputs.json: max_energy_drift_rel
    const output = formatScientific(1.1060555202205516e-13, 3)
    expect(output).toBe(`1.11${NBSP}${TIMES}${NBSP}10⁻¹³`)
  })

  it('uses a unicode minus in the mantissa, not a hyphen', () => {
    const output = formatScientific(-4.2e-5, 2)
    expect(output.startsWith(MINUS)).toBe(true)
    expect(output.startsWith('-')).toBe(false)
  })

  it('omits a zero exponent rather than printing x 10^0', () => {
    expect(formatScientific(3.14159, 3)).toBe('3.14')
  })

  it('handles zero', () => {
    expect(formatScientific(0, 3)).toBe('0.00')
  })

  it('handles large positive exponents', () => {
    expect(formatScientific(3.986004418e14, 4)).toBe(`3.986${NBSP}${TIMES}${NBSP}10¹⁴`)
  })
})

describe('auto notation', () => {
  it('uses scientific below 1e-3', () => {
    expect(formatNumber(1.5e-7)).toContain(TIMES)
  })

  it('uses plain digits in the readable middle range', () => {
    expect(formatNumber(2.3257791301763304)).toBe('2.326')
    expect(formatNumber(369.32370625238775)).toBe('369.3')
  })

  it('uses scientific at or above 1e5', () => {
    expect(formatNumber(6378137)).toContain(TIMES)
  })

  it('never emits machine notation for any magnitude', () => {
    const values = [1e-30, 2e-12, 1.5e-7, 0.001, 1, 5553.62, 1e5, 6.6743e-11, 3.986e14]
    for (const value of values) {
      const output = formatNumber(value)
      expect(output, `value ${value} produced ${output}`).not.toMatch(/\de[+-]?\d/i)
    }
  })
})

describe('missing values', () => {
  it.each([null, undefined, NaN, Infinity, -Infinity])('renders %s as a dash', (value) => {
    expect(formatNumber(value as number)).toBe('—')
  })

  it('accepts a custom placeholder', () => {
    expect(formatNumber(null, { placeholder: 'not measured' })).toBe('not measured')
  })
})

describe('units', () => {
  it('appends the unit after a non-breaking space so it cannot wrap away', () => {
    const output = formatNumber(2.326, { unit: 'm/s' })
    expect(output).toBe(`2.326${NBSP}m/s`)
    expect(output).not.toContain(' m/s')
  })
})

describe('fixed and significant', () => {
  it('formats fixed decimals', () => {
    expect(formatFixed(5553.624271252228, 2)).toBe('5553.62')
    expect(formatFixed(0.5, 3)).toBe('0.500')
  })

  it('formats significant figures and trims trailing zeros', () => {
    expect(formatSignificant(2.3257791301763304, 4)).toBe('2.326')
    expect(formatSignificant(1000, 4)).toBe('1000')
  })

  it('shows a signed value when asked', () => {
    expect(formatNumber(4, { notation: 'fixed', decimals: 0, signed: true })).toBe('+4')
    expect(formatNumber(-4, { notation: 'fixed', decimals: 0, signed: true })).toBe(`${MINUS}4`)
  })
})

describe('percentages and angles', () => {
  it('formats a fraction as a percentage', () => {
    expect(formatPercent(0.5)).toBe('50.0%')
    expect(formatPercent(0)).toBe('0.0%')
  })

  it('formats an angle with the degree sign', () => {
    expect(formatAngle(51.6)).toBe('51.60°')
  })
})

describe('durations', () => {
  it('uses seconds for short spans', () => {
    expect(formatDuration(45)).toBe(`45.0${NBSP}s`)
  })

  it('uses minutes for an orbital period', () => {
    expect(formatDuration(5553.62)).toBe(`92.6${NBSP}min`)
  })

  it('uses hours for long propagations', () => {
    expect(formatDuration(36000)).toBe(`10.0${NBSP}h`)
  })

  it('honours a forced unit', () => {
    expect(formatDuration(5553.62, { forceUnit: 's', decimals: 2 })).toBe(`5553.62${NBSP}s`)
  })
})

describe('distances', () => {
  it('uses metres for close separations', () => {
    expect(formatDistance(369.32)).toBe(`369.3${NBSP}m`)
  })

  it('switches to kilometres for large distances', () => {
    expect(formatDistance(50_000)).toBe(`50${NBSP}km`)
  })

  it('keeps sub-millimetre results visible in scientific notation', () => {
    // A converged rendezvous ends ~1.5e-7 m from the target. Rendering that as
    // "0.000 km" would hide the entire result.
    const output = formatDistance(1.495996307381497e-7)
    expect(output).toBe(`1.50${NBSP}${TIMES}${NBSP}10⁻⁷${NBSP}m`)
  })
})

describe('velocity and acceleration', () => {
  it('formats a delta-v', () => {
    expect(formatVelocity(2.3257791301763304)).toBe(`2.326${NBSP}m/s`)
  })

  it('keeps a tiny final velocity error readable', () => {
    expect(formatVelocity(6.581593002429279e-10)).toBe(`6.58${NBSP}${TIMES}${NBSP}10⁻¹⁰${NBSP}m/s`)
  })

  it('formats a control acceleration with the squared unit', () => {
    expect(formatAcceleration(0.01701831014059112)).toBe(`0.01702${NBSP}m/s²`)
  })
})

describe('formatByUnit', () => {
  it.each([
    [369.32, 'm', `369.3${NBSP}m`],
    [2.3257, 'm/s', `2.326${NBSP}m/s`],
    [51.6, 'deg', '51.60°'],
    [0.25, '%', '25.0%'],
  ])('formats %s as %s', (value, unit, expected) => {
    expect(formatByUnit(value as number, unit as string)).toBe(expected)
  })

  it('falls back to auto notation for an unknown unit', () => {
    expect(formatByUnit(1.23456, 'widgets')).toBe(`1.235${NBSP}widgets`)
  })
})

describe('rawValue for exports', () => {
  it('returns full machine precision with no presentation formatting', () => {
    expect(rawValue(1.1060555202205516e-13)).toBe('1.1060555202205516e-13')
    expect(rawValue(2.3257791301763304)).toBe('2.3257791301763304')
  })

  it('uses a plain hyphen, never a unicode minus, so parsers accept it', () => {
    const output = rawValue(-42.5)
    expect(output).toBe('-42.5')
    expect(output).not.toContain(MINUS)
  })

  it('never contains a superscript or a multiplication sign', () => {
    for (const value of [1e-30, 2e-12, 6378137, -1.5]) {
      const output = rawValue(value)
      expect(output).not.toContain(TIMES)
      expect(output).not.toMatch(/[⁰¹²³⁴⁵⁶⁷⁸⁹⁻]/)
    }
  })

  it('renders missing values as an empty cell', () => {
    expect(rawValue(null)).toBe('')
    expect(rawValue(undefined)).toBe('')
  })

  it('round-trips through Number() without loss', () => {
    const values = [1.1060555202205516e-13, 2.3257791301763304, 5553.624271252228]
    for (const value of values) {
      expect(Number(rawValue(value))).toBe(value)
    }
  })
})

describe('no emoji anywhere in formatted output', () => {
  it('never introduces pictographic characters', () => {
    const outputs = [
      formatNumber(2e-12),
      formatDistance(369.32),
      formatDuration(5553.62),
      formatVelocity(2.3257),
      formatAcceleration(0.017),
      formatPercent(0.5),
      formatAngle(51.6),
    ]
    for (const output of outputs) {
      expect(output).not.toMatch(/\p{Extended_Pictographic}/u)
    }
  })
})
