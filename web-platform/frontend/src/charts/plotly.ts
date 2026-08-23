/**
 * Lazy loader for Plotly.
 *
 * `plotly.js-dist-min` is roughly 3 MB. Loading it eagerly would delay first
 * paint for every visitor, including those who only read the scenario
 * catalogue and never run a simulation. It is therefore pulled in with a
 * dynamic `import()` the first time a chart is actually drawn; Vite splits it
 * into its own chunk (see `manualChunks` in vite.config.ts).
 *
 * The promise is memoised, so a results page with eight charts triggers one
 * network request and one parse, not eight. A failed load clears the memo so a
 * later attempt (a retry, or simply navigating to another result) can succeed
 * after a transient network fault.
 */

export type PlotlyModule = typeof import('plotly.js-dist-min')

let pending: Promise<PlotlyModule> | null = null

/**
 * Resolve the module namespace regardless of interop shape.
 *
 * The dist bundle is UMD. Depending on whether it is served by Vite's dev
 * server or through the Rollup CommonJS interop in a production build, the
 * namespace object may expose `newPlot` directly or only under `default`.
 * Normalising here means callers never have to care.
 */
function normalise(mod: PlotlyModule): PlotlyModule {
  const candidate = (mod as { default?: unknown }).default
  if (candidate && typeof (candidate as PlotlyModule).react === 'function') {
    return candidate as PlotlyModule
  }
  return mod
}

/** Load Plotly, reusing the in-flight or already-resolved module. */
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

/** True once the bundle has been requested. Used to skip the spinner on repeat charts. */
export function isPlotlyRequested(): boolean {
  return pending !== null
}

/** Drop the memo. Exists for tests; nothing in the application calls it. */
export function resetPlotlyLoader(): void {
  pending = null
}
