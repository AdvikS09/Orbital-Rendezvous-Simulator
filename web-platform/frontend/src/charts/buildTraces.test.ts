import { describe, expect, it } from 'vitest'
import type { PlotSpec, SeriesSpec } from '@/api/types'
import { axisLabel, buildLayout, buildTraces } from './buildTraces'

function series(overrides: Partial<SeriesSpec>): SeriesSpec {
  return {
    key: 's',
    label: 'series',
    x_key: null,
    y_key: 'y',
    z_key: null,
    unit: 'm',
    color_index: 0,
    dash: 'solid',
    secondary_y: false,
    width: 2.5,
    mode: 'lines',
    ...overrides,
  }
}

function plot(overrides: Partial<PlotSpec>): PlotSpec {
  return {
    id: 'p',
    title: 'Plot',
    kind: 'line',
    x: { title: 'time', unit: 's', log: false },
    y: { title: 'altitude', unit: 'km', log: false },
    y2: null,
    z: null,
    x_key: 't_s',
    series: [series({})],
    caption: null,
    show_earth: false,
    show_target_marker: false,
    keep_out_radius_m: null,
    reference_line: null,
    matrix: null,
    height: 380,
    ...overrides,
  }
}

const data = { t_s: [0, 1, 2], y: [1, null, 3], x_m: [1000, 2000, 3000], z_m: [0, 0, 0] }

describe('buildTraces', () => {
  it('labels axes with units', () => {
    expect(axisLabel({ title: 'altitude', unit: 'km', log: false })).toBe('altitude [km]')
    expect(axisLabel({ title: 'ratio', unit: '', log: false })).toBe('ratio')
  })

  it('builds a line trace from the plot x_key', () => {
    const [trace] = buildTraces(plot({}), data)
    expect(trace?.['type']).toBe('scatter')
    expect(trace?.['x']).toEqual([0, 1, 2])
    expect(trace?.['y']).toEqual([1, null, 3])
    expect(trace?.['yaxis']).toBe('y')
  })

  it('puts secondary traces on y2', () => {
    const spec = plot({
      kind: 'dual_axis',
      y2: { title: 'velocity', unit: 'm/s', log: false },
      series: [series({}), series({ secondary_y: true })],
    })
    const traces = buildTraces(spec, data)
    expect(traces[1]?.['yaxis']).toBe('y2')
    const layout = buildLayout(spec, true)
    expect((layout['yaxis2'] as Record<string, unknown>)['side']).toBe('right')
  })

  it('sets both axes to log for log_log', () => {
    const layout = buildLayout(plot({ kind: 'log_log' }), true)
    expect((layout['xaxis'] as Record<string, unknown>)['type']).toBe('log')
    expect((layout['yaxis'] as Record<string, unknown>)['type']).toBe('log')
  })

  it('adds a target marker and keep-out sphere', () => {
    const traces = buildTraces(
      plot({
        kind: 'scatter3d',
        show_target_marker: true,
        keep_out_radius_m: 300,
        series: [series({ x_key: 'x_m', y_key: 'y', z_key: 'z_m' })],
      }),
      data,
    )
    expect(traces).toHaveLength(3)
    expect(traces[1]?.['name']).toBe('target')
    expect(traces[2]?.['type']).toBe('surface')
  })

  it('converts orbit positions to kilometres for display', () => {
    const traces = buildTraces(
      plot({
        kind: 'orbit3d',
        show_earth: true,
        series: [series({ x_key: 'x_m', y_key: 'y', z_key: 'z_m' })],
      }),
      data,
    )
    expect(traces[0]?.['x']).toEqual([1, 2, 3])
    expect(traces[1]?.['name']).toBe('Earth')
  })

  it('builds a heatmap from the matrix', () => {
    const traces = buildTraces(
      plot({
        kind: 'heatmap',
        matrix: { z: [[1, 2]], x: ['a', 'b'], y: ['c'], colorbar_title: 'seconds' },
      }),
      data,
    )
    expect(traces[0]?.['type']).toBe('heatmap')
    expect(traces[0]?.['colorscale']).toBe('Viridis')
  })

  it('spans a constant reference line across the x extent', () => {
    const traces = buildTraces(plot({ reference_line: { y: 2, label: 'target' } }), data)
    expect(traces).toHaveLength(2)
    expect(traces[1]?.['x']).toEqual([0, 2])
    expect(traces[1]?.['y']).toEqual([2, 2])
  })

  it('survives a missing series key', () => {
    const traces = buildTraces(plot({ series: [series({ y_key: 'absent' })] }), data)
    expect(traces[0]?.['y']).toEqual([])
  })
})
