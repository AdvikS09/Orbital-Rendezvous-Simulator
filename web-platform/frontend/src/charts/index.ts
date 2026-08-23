/**
 * Charting layer.
 *
 * Renders the `PlotSpec` and numeric arrays returned by the API. It selects and
 * styles; the Python engine computes. Import from here rather than reaching
 * into the individual modules.
 */

export { PlotlyChart } from './PlotlyChart'
export type { PlotlyChartProps } from './PlotlyChart'

export { buildTraces, buildLayout, axisLabel } from './buildTraces'
export type { SeriesData, BuildOptions } from './buildTraces'

export {
  SERIES_COLORS,
  DASH_MAP,
  plotlyLayout,
  chartPalette,
  seriesPalette,
  seriesColor,
  readCssVar,
  isDarkTheme,
  prefersReducedMotion,
} from './theme'
export type { ChartPalette } from './theme'

export { loadPlotly, isPlotlyRequested, resetPlotlyLoader } from './plotly'
export type { PlotlyModule } from './plotly'
