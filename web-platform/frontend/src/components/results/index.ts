/**
 * Result rendering.
 *
 * `ResultView` composes the whole thing; the individual panels are exported so
 * a page can assemble a different arrangement (an activity step, for instance,
 * that shows only the metrics it asked the reader to predict).
 */

export { MetricGrid } from './MetricGrid'
export type { MetricGridProps } from './MetricGrid'

export {
  InterpretationList,
  renderInlineMarkdown,
  renderMarkdown,
} from './InterpretationList'
export type { InterpretationListProps, MarkdownOptions } from './InterpretationList'

export { AssumptionsPanel } from './AssumptionsPanel'
export type { AssumptionsPanelProps } from './AssumptionsPanel'

export { SolverPanel } from './SolverPanel'
export type { SolverPanelProps } from './SolverPanel'

export { DEFAULT_ROW_CAP, ResultTables } from './ResultTables'
export type { ResultTablesProps } from './ResultTables'

export { ExportBar } from './ExportBar'
export type { ExportBarProps } from './ExportBar'

export { ResultView } from './ResultView'
export type { ResultViewProps } from './ResultView'

export { csvFilename, quoteField, toCsv } from './csv'
export type { CsvContext } from './csv'
