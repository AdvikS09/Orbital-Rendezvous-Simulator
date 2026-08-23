/**
 * Component tests.
 *
 * Concentrated on the two things this migration was asked to get right and the
 * ones a screenshot cannot verify: that numbers are presented as scientific
 * notation with units rather than machine notation, and that every control is
 * reachable and announced.
 */

import { describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import type { Metric, ParamSpec, TableSpec } from '@/api/types'
import { NumberValue } from '@/components/NumberValue'
import { ParamControl } from '@/components/controls/ParamControl'
import { ParamPanel } from '@/components/controls/ParamPanel'
import { MetricGrid } from '@/components/results/MetricGrid'
import { ResultTables } from '@/components/results/ResultTables'
import { toCsv } from '@/components/results/csv'
import { Callout } from '@/components/ui/Callout'
import { TIMES } from '@/formatting/number'

// --------------------------------------------------------------- fixtures
function numberParam(overrides: Partial<ParamSpec> = {}): ParamSpec {
  return {
    name: 'altitude_km',
    label: 'Target altitude',
    kind: 'number',
    default: 400,
    minimum: 200,
    maximum: 1200,
    step: 25,
    unit: 'km',
    help: "Height above Earth's surface of the target's circular orbit.",
    options: [],
    advanced: false,
    fmt: 'auto',
    precision: null,
    hard_minimum: 150,
    hard_maximum: 40000,
    ...overrides,
  }
}

function metric(overrides: Partial<Metric> = {}): Metric {
  return {
    key: 'max_energy_drift_rel',
    label: 'Energy drift (relative)',
    value: 2e-12,
    unit: '',
    fmt: 'scientific',
    precision: 3,
    help: 'How much the total orbital energy wandered.',
    highlight: true,
    ...overrides,
  }
}

// ------------------------------------------------------------- NumberValue
describe('NumberValue', () => {
  it('renders a tiny value as scientific notation, never machine notation', () => {
    render(<NumberValue value={2e-12} fmt="scientific" precision={2} />)
    const text = document.body.textContent ?? ''
    expect(text).toContain(TIMES)
    expect(text).toContain('10⁻¹²')
    expect(text).not.toMatch(/\de-\d/)
  })

  it('shows the unit alongside the value', () => {
    render(<NumberValue value={2.3257} unit="m/s" />)
    expect(document.body.textContent).toContain('m/s')
  })

  it('exposes full precision for inspection without displaying it', () => {
    const { container } = render(<NumberValue value={1.1060555202205516e-13} />)
    const element = container.querySelector('[title]')
    expect(element?.getAttribute('title')).toContain('1.1060555202205516e-13')
    // ...but the visible text is the readable form.
    expect(container.textContent).not.toContain('1.1060555202205516e-13')
  })

  it('renders a missing value as a dash rather than zero', () => {
    render(<NumberValue value={null} />)
    expect(document.body.textContent).toContain('—')
    expect(document.body.textContent).not.toContain('0')
  })
})

// ------------------------------------------------------------------ inputs
describe('NumberField', () => {
  it('associates its label with the input', () => {
    render(
      <ParamControl param={numberParam()} value={400} onChange={vi.fn()} />,
    )
    const input = screen.getByLabelText(/Target altitude/i)
    expect(input).toBeInTheDocument()
  })

  it('shows the unit next to the field, not inside the value', () => {
    render(<ParamControl param={numberParam()} value={400} onChange={vi.fn()} />)
    const input = screen.getByLabelText(/Target altitude/i) as HTMLInputElement
    // The typed value must be the bare number; the unit is separate markup.
    expect(input.value).toBe('400')
    expect(document.body.textContent).toContain('km')
  })

  it('accepts scientific input such as 4.6e-5', async () => {
    const user = userEvent.setup()
    const onChange = vi.fn()
    render(
      <ParamControl
        param={numberParam({ name: 'kp', label: 'Kp', unit: '1/s^2', default: 4.6e-5 })}
        value={4.6e-5}
        onChange={onChange}
      />,
    )
    const input = screen.getByLabelText(/Kp/i)
    await user.clear(input)
    await user.type(input, '1.5e-4')
    expect(onChange).toHaveBeenCalled()
    const last = onChange.mock.calls.at(-1)
    expect(Number(last?.[last.length - 1])).toBeCloseTo(1.5e-4, 12)
  })

  it('reports a non-numeric entry rather than silently zeroing it', async () => {
    const user = userEvent.setup()
    render(<ParamControl param={numberParam()} value={400} onChange={vi.fn()} />)
    const input = screen.getByLabelText(/Target altitude/i)
    await user.clear(input)
    await user.type(input, 'abc')
    expect(input).toHaveAttribute('aria-invalid', 'true')
    expect(screen.getByRole('alert')).toBeInTheDocument()
  })

  it('marks a server-reported error as invalid and announces it', () => {
    render(
      <ParamControl
        param={numberParam()}
        value={-7000}
        onChange={vi.fn()}
        error="Altitude must be at least 150 km (got -7000 km)."
      />,
    )
    const input = screen.getByLabelText(/Target altitude/i)
    expect(input).toHaveAttribute('aria-invalid', 'true')
    expect(screen.getByRole('alert')).toHaveTextContent(/at least 150/)
  })
})

describe('SliderField', () => {
  it('gives the range and the numeric entry separate accessible names', () => {
    render(
      <ParamControl
        param={numberParam({ kind: 'slider' })}
        value={400}
        onChange={vi.fn()}
      />,
    )
    const slider = screen.getByRole('slider')
    expect(slider).toHaveAccessibleName()
    // The value is announced as formatted text, not a bare number.
    expect(slider).toHaveAttribute('aria-valuetext')
  })

  it('renders the value in its own element so it cannot overlap the handle', () => {
    const { container } = render(
      <ParamControl
        param={numberParam({ kind: 'slider' })}
        value={400}
        onChange={vi.fn()}
      />,
    )
    const slider = container.querySelector('input[type="range"]')
    const valueCell = container.querySelector('.num, [class*="value"]')
    expect(slider).not.toBeNull()
    expect(valueCell).not.toBeNull()
    // They must not be the same node, and the value must not be a child of the
    // track -- that is how the Streamlit version overlapped them.
    expect(slider?.contains(valueCell ?? null)).toBe(false)
  })
})

describe('ParamPanel', () => {
  const params = [
    numberParam(),
    numberParam({ name: 'dt_s', label: 'Timestep', unit: 's', advanced: true }),
    numberParam({ name: 'kd', label: 'Kd', unit: '1/s', advanced: true }),
  ]

  it('hides advanced parameters behind a disclosure', () => {
    render(
      <ParamPanel
        params={params}
        values={{ altitude_km: 400, dt_s: 5, kd: 0.0136 }}
        onChange={vi.fn()}
        errors={{}}
      />,
    )
    expect(screen.getByLabelText(/Target altitude/i)).toBeVisible()
    expect(screen.getByText(/Advanced controls/i)).toBeInTheDocument()
  })

  it('reports the parameter name and the new value on change', async () => {
    const user = userEvent.setup()
    const onChange = vi.fn()
    render(
      <ParamPanel
        params={[numberParam()]}
        values={{ altitude_km: 400 }}
        onChange={onChange}
        errors={{}}
      />,
    )
    const input = screen.getByLabelText(/Target altitude/i)
    await user.clear(input)
    await user.type(input, '500')
    expect(onChange).toHaveBeenCalled()
    expect(onChange.mock.calls.at(-1)?.[0]).toBe('altitude_km')
  })
})

// ----------------------------------------------------------------- results
describe('MetricGrid', () => {
  it('formats a tiny metric as scientific notation', () => {
    render(<MetricGrid metrics={[metric()]} />)
    const text = document.body.textContent ?? ''
    expect(text).toContain('10⁻¹²')
    expect(text).not.toMatch(/\de-\d/)
  })

  it('renders a non-converging metric as a dash, not as zero', () => {
    render(
      <MetricGrid
        metrics={[
          metric({ key: 'convergence_time_s', label: 'Time to converge', value: null, unit: 's', fmt: 'duration' }),
        ]}
      />,
    )
    expect(document.body.textContent).toContain('—')
  })

  it('exposes the help text to assistive technology', () => {
    render(<MetricGrid metrics={[metric()]} />)
    expect(
      screen.getByText(/total orbital energy wandered/i),
    ).toBeInTheDocument()
  })
})

describe('ResultTables', () => {
  const table: TableSpec = {
    id: 'metrics',
    label: 'Rendezvous metrics',
    columns: [
      { key: 't_s', label: 't_s', unit: 's', fmt: 'auto', precision: null, numeric: true },
      { key: 'pos_err_m', label: 'pos_err_m', unit: 'm', fmt: 'auto', precision: null, numeric: true },
    ],
    rows: [
      { t_s: 0, pos_err_m: 369.32370625238775 },
      { t_s: 5, pos_err_m: 1.495996307381497e-7 },
    ],
    caption: null,
    downloadable: true,
  }

  it('renders a scrollable region so the page never scrolls sideways', () => {
    const { container } = render(<ResultTables tables={[table]} />)
    const region = container.querySelector('[role="region"]')
    expect(region).not.toBeNull()
    expect(region).toHaveAttribute('tabindex', '0')
  })

  it('formats cell values rather than printing raw floats', () => {
    render(<ResultTables tables={[table]} />)
    const text = document.body.textContent ?? ''
    expect(text).not.toContain('369.32370625238775')
    expect(text).not.toMatch(/1\.495996307381497e-7/)
  })
})

describe('CSV export', () => {
  const table: TableSpec = {
    id: 'metrics',
    label: 'Metrics',
    columns: [
      { key: 'a', label: 'a', unit: 'm', fmt: 'auto', precision: null, numeric: true },
      { key: 'note', label: 'note', unit: '', fmt: 'auto', precision: null, numeric: false },
    ],
    rows: [
      { a: 1.1060555202205516e-13, note: 'plain' },
      { a: -42.5, note: 'has, comma' },
    ],
    caption: null,
    downloadable: true,
  }

  it('writes machine values, never the display string', () => {
    const csv = toCsv(table)
    expect(csv).toContain('1.1060555202205516e-13')
    expect(csv).not.toContain(TIMES)
    expect(csv).not.toMatch(/[⁰¹²³⁴⁵⁶⁷⁸⁹⁻]/)
    // A plain hyphen, so a spreadsheet parses it as a number.
    expect(csv).toContain('-42.5')
    expect(csv).not.toContain('−42.5')
  })

  it('quotes a field containing a comma', () => {
    expect(toCsv(table)).toContain('"has, comma"')
  })
})

// -------------------------------------------------------------- accessibility
describe('Callout', () => {
  it('announces severity in text, not by colour alone', () => {
    render(<Callout severity="warning">Separation exceeds the characterised range.</Callout>)
    expect(document.body.textContent).toMatch(/Warning/i)
  })

  it('uses role=alert for problems so they are announced immediately', () => {
    render(<Callout severity="error">Altitude must be at least 150 km.</Callout>)
    expect(screen.getByRole('alert')).toBeInTheDocument()
  })

  it('uses role=status for non-urgent notes', () => {
    render(<Callout severity="info">The integrator conserved energy.</Callout>)
    expect(screen.getByRole('status')).toBeInTheDocument()
  })
})

describe('no emoji in rendered output', () => {
  it('renders none anywhere in these components', () => {
    render(
      <>
        <MetricGrid metrics={[metric()]} />
        <Callout severity="warning">Check the separation.</Callout>
        <ParamControl param={numberParam()} value={400} onChange={vi.fn()} />
      </>,
    )
    const text = document.body.textContent ?? ''
    expect(text).not.toMatch(/\p{Extended_Pictographic}/u)
  })
})
