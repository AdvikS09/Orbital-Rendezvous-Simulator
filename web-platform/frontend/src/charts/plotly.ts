export type PlotlyModule = typeof import('plotly.js-dist-min')

let pending: Promise<PlotlyModule> | null = null

function normalise(mod: PlotlyModule): PlotlyModule {
  const candidate = (mod as { default?: unknown }).default
  if (candidate && typeof (candidate as PlotlyModule).react === 'function') {
    return candidate as PlotlyModule
  }
  return mod
}

export function loadPlotly(): Promise<PlotlyModule> {
  if (pending === null) {
    pending = import('plotly.js-dist-min')
      .then(normalise)
      .catch((error: unknown) => {
        pending = null
        throw error
      })
  }
  return pending
}

export function isPlotlyRequested(): boolean {
  return pending !== null
}

export function resetPlotlyLoader(): void {
  pending = null
}
