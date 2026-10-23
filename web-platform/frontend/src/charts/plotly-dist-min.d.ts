declare module 'plotly.js-dist-min' {
  export interface PlotData {
    [option: string]: unknown
  }

  export interface PlotLayout {
    [option: string]: unknown
  }

  export interface PlotConfig {
    [option: string]: unknown
  }

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
