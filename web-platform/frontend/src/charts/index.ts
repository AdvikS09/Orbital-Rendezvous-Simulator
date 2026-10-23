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
