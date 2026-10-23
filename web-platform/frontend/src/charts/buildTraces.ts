import type { AxisSpec, PlotSpec, SeriesSpec } from '@/api/types'
import type { PlotData, PlotLayout } from 'plotly.js-dist-min'
import { DASH_MAP, SERIES_COLORS, chartPalette, plotlyLayout, seriesColor } from './theme'

export type SeriesData = Record<string, (number | null)[]>

export interface BuildOptions {
  colors?: readonly string[]
}

const EARTH_RADIUS_KM = 6378.137

const M_TO_KM = 1 / 1000

const EMPTY: readonly (number | null)[] = []

function column(series: SeriesData, key: string | null | undefined): readonly (number | null)[] {
  if (key === null || key === undefined || key === '') {
    return EMPTY
  }
  return series[key] ?? EMPTY
}

function xColumn(
  plot: PlotSpec,
  spec: SeriesSpec,
  series: SeriesData,
  fallbackLength: number,
): readonly (number | null)[] {
  const own = column(series, spec.x_key)
  if (own.length > 0) {
    return own
  }
  const shared = column(series, plot.x_key)
  if (shared.length > 0) {
    return shared
  }
  return Array.from({ length: fallbackLength }, (_unused, index) => index)
}

function scaled(values: readonly (number | null)[], factor: number): (number | null)[] {
  if (factor === 1) {
    return values.slice()
  }
  return values.map((value) => (value === null ? null : value * factor))
}

function finiteExtent(values: readonly (number | null)[]): [number, number] | null {
  let low = Number.POSITIVE_INFINITY
  let high = Number.NEGATIVE_INFINITY
  for (const value of values) {
    if (value === null || !Number.isFinite(value)) {
      continue
    }
    if (value < low) {
      low = value
    }
    if (value > high) {
      high = value
    }
  }
  return low <= high ? [low, high] : null
}

export function axisLabel(axis: AxisSpec | null | undefined): string {
  if (!axis) {
    return ''
  }
  return axis.unit ? axis.title + ' [' + axis.unit + ']' : axis.title
}

function valueTemplate(unit: string): string {
  const suffix = unit ? ' ' + unit : ''
  return '%{y:.6~g}' + suffix + '<extra></extra>'
}

function spatialTemplate(label: string, unit: string): string {
  const suffix = unit ? ' ' + unit : ''
  return (
    '%{x:.4~g}, %{y:.4~g}, %{z:.4~g}' + suffix + '<extra>' + label + '</extra>'
  )
}

function cartesianTrace(
  plot: PlotSpec,
  spec: SeriesSpec,
  series: SeriesData,
  palette: readonly string[],
): PlotData {
  const y = column(series, spec.y_key)
  const x = xColumn(plot, spec, series, y.length)
  const colour = seriesColor(spec.color_index, palette)
  const onSecondary = spec.secondary_y && plot.y2 !== null

  return {
    type: 'scatter',
    mode: spec.mode,
    name: spec.label,
    x: x.slice(),
    y: y.slice(),
    line: { color: colour, width: spec.width, dash: DASH_MAP[spec.dash], shape: 'linear' },
    marker: { color: colour, size: Math.max(4, spec.width + 1) },
    yaxis: onSecondary ? 'y2' : 'y',
    hovertemplate: valueTemplate(spec.unit),
    connectgaps: false,
  }
}

function spatialTrace(
  spec: SeriesSpec,
  series: SeriesData,
  palette: readonly string[],
  factor: number,
  unit: string,
): PlotData {
  const colour = seriesColor(spec.color_index, palette)
  return {
    type: 'scatter3d',
    mode: spec.mode === 'markers' ? 'markers' : 'lines',
    name: spec.label,
    x: scaled(column(series, spec.x_key), factor),
    y: scaled(column(series, spec.y_key), factor),
    z: scaled(column(series, spec.z_key), factor),
    line: { color: colour, width: spec.width, dash: DASH_MAP[spec.dash] },
    marker: { color: colour, size: 2 },
    hovertemplate: spatialTemplate(spec.label, unit),
    connectgaps: false,
  }
}

function sphereSurface(options: {
  radius: number
  colour: string
  opacity: number
  name: string
  lonSteps?: number
  latSteps?: number
}): PlotData {
  const lonSteps = options.lonSteps ?? 36
  const latSteps = options.latSteps ?? 18
  const xs: number[][] = []
  const ys: number[][] = []
  const zs: number[][] = []

  for (let i = 0; i <= latSteps; i += 1) {
    const polar = (Math.PI * i) / latSteps
    const rowX: number[] = []
    const rowY: number[] = []
    const rowZ: number[] = []
    for (let j = 0; j <= lonSteps; j += 1) {
      const azimuth = (2 * Math.PI * j) / lonSteps
      rowX.push(options.radius * Math.sin(polar) * Math.cos(azimuth))
      rowY.push(options.radius * Math.sin(polar) * Math.sin(azimuth))
      rowZ.push(options.radius * Math.cos(polar))
    }
    xs.push(rowX)
    ys.push(rowY)
    zs.push(rowZ)
  }

  return {
    type: 'surface',
    name: options.name,
    x: xs,
    y: ys,
    z: zs,
    opacity: options.opacity,
    colorscale: [
      [0, options.colour],
      [1, options.colour],
    ],
    showscale: false,
    showlegend: false,
    hoverinfo: 'skip',
    lighting: { ambient: 0.85, diffuse: 0.4, specular: 0.04, roughness: 0.9 },
    contours: {
      x: { highlight: false },
      y: { highlight: false },
      z: { highlight: false },
    },
  }
}

function targetMarker(palette: readonly string[]): PlotData {
  return {
    type: 'scatter3d',
    mode: 'markers',
    name: 'target',
    x: [0],
    y: [0],
    z: [0],
    marker: { symbol: 'diamond', size: 6, color: seriesColor(2, palette) },
    hovertemplate: 'target<extra></extra>',
    showlegend: true,
  }
}

function numberField(record: Record<string, unknown>, key: string): number | null {
  const value = record[key]
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function arrayField(record: Record<string, unknown>, key: string): number[] | null {
  const value = record[key]
  if (!Array.isArray(value)) {
    return null
  }
  const numbers = value.filter((item): item is number => typeof item === 'number')
  return numbers.length === value.length ? numbers : null
}

function referenceLineTrace(
  plot: PlotSpec,
  series: SeriesData,
  palette: readonly string[],
): PlotData | null {
  const spec = plot.reference_line
  if (!spec) {
    return null
  }

  const rawLabel = spec['label']
  const base: PlotData = {
    type: 'scatter',
    mode: 'lines',
    name: typeof rawLabel === 'string' ? rawLabel : 'reference',
    line: { color: seriesColor(6, palette), width: 1.5, dash: 'dash' },
    hoverinfo: 'skip',
    showlegend: true,
  }

  const explicitX = arrayField(spec, 'x')
  const explicitY = arrayField(spec, 'y')
  if (explicitX && explicitY) {
    return { ...base, x: explicitX, y: explicitY }
  }

  const xKey = typeof spec['x_key'] === 'string' ? spec['x_key'] : null
  const yKey = typeof spec['y_key'] === 'string' ? spec['y_key'] : null
  const first = plot.series[0]

  if (yKey) {
    const y = column(series, yKey)
    let x: readonly (number | null)[]
    if (xKey) {
      x = column(series, xKey)
    } else if (first) {
      x = xColumn(plot, first, series, y.length)
    } else {
      x = Array.from({ length: y.length }, (_unused, index) => index)
    }
    return { ...base, x: x.slice(), y: y.slice() }
  }

  if (!first) {
    return null
  }
  const y = column(series, first.y_key)
  const x = xColumn(plot, first, series, y.length)

  const constantY = numberField(spec, 'y')
  if (constantY !== null) {
    const extent = finiteExtent(x)
    return extent ? { ...base, x: extent, y: [constantY, constantY] } : null
  }

  const constantX = numberField(spec, 'x')
  if (constantX !== null) {
    const extent = finiteExtent(y)
    return extent ? { ...base, x: [constantX, constantX], y: extent } : null
  }

  return null
}

export function buildTraces(
  plot: PlotSpec,
  series: SeriesData,
  options: BuildOptions = {},
): PlotData[] {
  const palette = options.colors ?? SERIES_COLORS

  if (plot.kind === 'heatmap') {
    const matrix = plot.matrix
    if (!matrix) {
      return []
    }
    return [
      {
        type: 'heatmap',
        z: matrix.z,
        x: matrix.x,
        y: matrix.y,
        colorscale: 'Viridis',
        colorbar: {
          title: { text: matrix.colorbar_title ?? '', side: 'right' },
          thickness: 12,
          outlinewidth: 0,
          ticks: 'outside',
          ticklen: 3,
        },
        xgap: 1,
        ygap: 1,
        hoverongaps: false,
        hovertemplate:
          plot.x.title +
          ': %{x}<br>' +
          plot.y.title +
          ': %{y}<br>' +
          plot.title +
          ': %{z:.4~g}<extra></extra>',
      },
    ]
  }

  if (plot.kind === 'scatter3d') {
    const traces: PlotData[] = plot.series.map((spec) =>
      spatialTrace(spec, series, palette, 1, spec.unit),
    )
    if (plot.show_target_marker) {
      traces.push(targetMarker(palette))
    }
    if (plot.keep_out_radius_m !== null && plot.keep_out_radius_m > 0) {
      traces.push(
        sphereSurface({
          radius: plot.keep_out_radius_m,
          colour: seriesColor(1, palette),
          opacity: 0.16,
          name: 'keep-out sphere',
        }),
      )
    }
    return traces
  }

  if (plot.kind === 'orbit3d') {
    const traces: PlotData[] = plot.series.map((spec) =>
      spatialTrace(spec, series, palette, M_TO_KM, 'km'),
    )
    if (plot.show_earth) {
      traces.push(
        sphereSurface({
          radius: EARTH_RADIUS_KM,
          colour: seriesColor(0, palette),
          opacity: 0.22,
          name: 'Earth',
          lonSteps: 48,
          latSteps: 24,
        }),
      )
    }
    return traces
  }

  const traces: PlotData[] = plot.series.map((spec) => cartesianTrace(plot, spec, series, palette))
  const reference = referenceLineTrace(plot, series, palette)
  if (reference) {
    traces.push(reference)
  }
  return traces
}

function is3d(kind: PlotSpec['kind']): boolean {
  return kind === 'scatter3d' || kind === 'orbit3d'
}

export function buildLayout(plot: PlotSpec, dark: boolean, reducedMotion = false): PlotLayout {
  const base = plotlyLayout(dark)
  const palette = chartPalette(dark)
  const logX = plot.x.log || plot.kind === 'log_log'
  const logY = plot.y.log || plot.kind === 'log_log'

  const layout: PlotLayout = {
    ...base,
    height: plot.height,
    showlegend: plot.series.length > 1 || plot.show_target_marker,
    transition: { duration: reducedMotion ? 0 : 220, easing: 'cubic-in-out' },
  }

  if (is3d(plot.kind)) {
    const axis3d = {
      gridcolor: palette.grid,
      zerolinecolor: palette.zeroline,
      backgroundcolor: 'rgba(0,0,0,0)',
      showbackground: false,
      tickfont: { family: palette.font, size: 10, color: palette.muted },
      titlefont: { family: palette.font, size: 11, color: palette.text },
    }
    delete layout['xaxis']
    delete layout['yaxis']
    layout['hovermode'] = 'closest'
    layout['margin'] = { l: 4, r: 4, t: 28, b: 4 }
    layout['scene'] = {
      aspectmode: 'data',
      bgcolor: 'rgba(0,0,0,0)',
      xaxis: { ...axis3d, title: { text: axisLabel(plot.x) } },
      yaxis: { ...axis3d, title: { text: axisLabel(plot.y) } },
      zaxis: { ...axis3d, title: { text: axisLabel(plot.z) } },
      camera: { eye: { x: 1.55, y: 1.55, z: 0.85 } },
    }
    return layout
  }

  if (plot.kind === 'heatmap') {
    layout['hovermode'] = 'closest'
    layout['showlegend'] = false
  }

  const xaxis = { ...(base['xaxis'] as Record<string, unknown>) }
  xaxis['title'] = { text: axisLabel(plot.x), standoff: 10 }
  if (logX) {
    xaxis['type'] = 'log'
    xaxis['exponentformat'] = 'power'
  }
  layout['xaxis'] = xaxis

  const yaxis = { ...(base['yaxis'] as Record<string, unknown>) }
  yaxis['title'] = { text: axisLabel(plot.y), standoff: 12 }
  if (logY) {
    yaxis['type'] = 'log'
    yaxis['exponentformat'] = 'power'
  }
  layout['yaxis'] = yaxis

  if (plot.y2) {
    layout['yaxis2'] = {
      ...(base['yaxis'] as Record<string, unknown>),
      title: { text: axisLabel(plot.y2), standoff: 12 },
      overlaying: 'y',
      side: 'right',
      showgrid: false,
      zeroline: false,
      type: plot.y2.log ? 'log' : 'linear',
    }
    layout['margin'] = { l: 68, r: 68, t: 34, b: 52 }
  }

  return layout
}
