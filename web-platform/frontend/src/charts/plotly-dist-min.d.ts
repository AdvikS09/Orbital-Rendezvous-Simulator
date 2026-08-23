/**
 * Ambient types for `plotly.js-dist-min`.
 *
 * The dist-min package ships a pre-built UMD bundle with no declaration file,
 * and the community `@types/plotly.js` package describes the full source build
 * rather than this one. Rather than pull in a large third-party type surface
 * that would drift from the bundle we actually load, this declares the narrow
 * slice the charting layer uses.
 *
 * Trace and layout objects are open records: Plotly's own option set is far
 * wider than anything we would usefully enumerate here, and every object we
 * hand it is constructed in `buildTraces.ts` and `theme.ts` where the shape is
 * visible at the point of construction.
 */

declare module 'plotly.js-dist-min' {
  /** One trace. Open by design; see the note above. */
  export interface PlotData {
    [option: string]: unknown
  }

  /** A layout object. Open by design; see the note above. */
  export interface PlotLayout {
    [option: string]: unknown
  }

  /** A config object. Open by design; see the note above. */
  export interface PlotConfig {
    [option: string]: unknown
  }

  /** A div that Plotly has drawn into. */
  export interface PlotlyHTMLElement extends HTMLElement {
    data?: PlotData[]
    layout?: PlotLayout
  }

  export function newPlot(
    root: HTMLElement,
    data: PlotData[],
    layout?: PlotLayout,
    config?: PlotConfig,
  ): Promise<PlotlyHTMLElement>

  export function react(
    root: HTMLElement,
    data: PlotData[],
    layout?: PlotLayout,
    config?: PlotConfig,
  ): Promise<PlotlyHTMLElement>

  export function relayout(root: HTMLElement, layout: PlotLayout): Promise<PlotlyHTMLElement>

  export function purge(root: HTMLElement): void

  export function downloadImage(
    root: HTMLElement,
    options: { format: string; filename: string; width?: number; height?: number; scale?: number },
  ): Promise<string>

  export const Plots: {
    resize(root: HTMLElement): void
  }

  export const version: string

  const Plotly: {
    newPlot: typeof newPlot
    react: typeof react
    relayout: typeof relayout
    purge: typeof purge
    downloadImage: typeof downloadImage
    Plots: typeof Plots
    version: string
  }

  export default Plotly
}
