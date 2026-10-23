import { useEffect, useMemo, useRef, useState } from 'react'
import type { PlotSpec } from '@/api/types'
import { Callout } from '@/components/ui/Callout'
import { Spinner } from '@/components/ui/Spinner'
import { axisLabel, buildLayout, buildTraces } from './buildTraces'
import type { SeriesData } from './buildTraces'
import { loadPlotly } from './plotly'
import type { PlotlyModule } from './plotly'
import { isDarkTheme, prefersReducedMotion, seriesPalette } from './theme'
import './PlotlyChart.css'

type ChartStatus = 'loading' | 'ready' | 'failed'

export interface PlotlyChartProps {
  plot: PlotSpec
  series: SeriesData
  className?: string
}

function useDarkTheme(): boolean {
  const [dark, setDark] = useState<boolean>(isDarkTheme)

  useEffect(() => {
    if (typeof MutationObserver === 'undefined' || typeof document === 'undefined') {
      return
    }
    const observer = new MutationObserver(() => {
      setDark(isDarkTheme())
    })
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['data-theme'],
    })
    return () => {
      observer.disconnect()
    }
  }, [])

  return dark
}

function chartDescription(plot: PlotSpec): string {
  const x = axisLabel(plot.x)
  const y = axisLabel(plot.y)
  if (plot.kind === 'scatter3d' || plot.kind === 'orbit3d') {
    const z = axisLabel(plot.z)
    return `${plot.title}. Three-dimensional plot with axes ${x}, ${y} and ${z}. The plotted values are listed in the result tables below.`
  }
  if (plot.kind === 'heatmap') {
    return `${plot.title}. Grid of ${y} against ${x}, shaded by value. The plotted values are listed in the result tables below.`
  }
  return `${plot.title}. Plot of ${y} against ${x}. The plotted values are listed in the result tables below.`
}

export function PlotlyChart({ plot, series, className }: PlotlyChartProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const plotlyRef = useRef<PlotlyModule | null>(null)
  const [status, setStatus] = useState<ChartStatus>('loading')
  const dark = useDarkTheme()

  const config = useMemo(
    () => ({
      displaylogo: false,
      responsive: true,
      displayModeBar: true,
      scrollZoom: false,
      doubleClick: 'reset' as const,
      modeBarButtonsToRemove: [
        'sendDataToCloud',
        'lasso2d',
        'select2d',
        'toggleSpikelines',
        'hoverClosestCartesian',
        'hoverCompareCartesian',
      ],
      toImageButtonOptions: {
        format: 'png',
        filename: plot.id,
        scale: 2,
      },
    }),
    [plot.id],
  )

  useEffect(() => {
    let active = true
    loadPlotly()
      .then((module) => {
        if (!active) {
          return
        }
        plotlyRef.current = module
        setStatus('ready')
      })
      .catch(() => {
        if (active) {
          setStatus('failed')
        }
      })
    return () => {
      active = false
    }
  }, [])

  useEffect(() => {
    const node = containerRef.current
    const plotly = plotlyRef.current
    if (status !== 'ready' || node === null || plotly === null) {
      return
    }

    const reduced = prefersReducedMotion()
    const data = buildTraces(plot, series, { colors: seriesPalette(dark) })
    const layout = buildLayout(plot, dark, reduced)

    plotly.react(node, data, layout, config).catch(() => {
      setStatus('failed')
    })
  }, [status, plot, series, dark, config])

  useEffect(() => {
    const node = containerRef.current
    return () => {
      const plotly = plotlyRef.current
      if (node !== null && plotly !== null) {
        plotly.purge(node)
      }
    }
  }, [])

  useEffect(() => {
    const node = containerRef.current
    if (status !== 'ready' || node === null || typeof ResizeObserver === 'undefined') {
      return
    }

    let frame = 0
    const observer = new ResizeObserver(() => {
      window.cancelAnimationFrame(frame)
      frame = window.requestAnimationFrame(() => {
        const plotly = plotlyRef.current
        const current = containerRef.current
        if (plotly !== null && current !== null) {
          plotly.Plots.resize(current)
        }
      })
    })
    observer.observe(node)

    return () => {
      window.cancelAnimationFrame(frame)
      observer.disconnect()
    }
  }, [status])

  const rootClass = className ? `plotly-chart ${className}` : 'plotly-chart'

  return (
    <figure className={rootClass}>
      {status === 'failed' ? (
        <Callout severity="warning" title="Chart could not be drawn" className="plotly-chart__fallback">
          The plotting library did not load. Nothing is wrong with the result itself: the
          values behind this chart are unaffected and remain available in the result tables
          and the data download.
        </Callout>
      ) : (
        <div className="plotly-chart__frame" style={{ minHeight: plot.height }}>
          <div
            ref={containerRef}
            className="plotly-chart__canvas"
            role="img"
            aria-label={chartDescription(plot)}
            aria-busy={status === 'loading'}
          />
          {status === 'loading' ? (
            <div className="plotly-chart__loading">
              <Spinner size="md" label="Loading chart" />
              <span className="plotly-chart__loading-text" aria-hidden="true">
                Loading chart
              </span>
            </div>
          ) : null}
        </div>
      )}
      {plot.caption ? (
        <figcaption className="plotly-chart__caption">{plot.caption}</figcaption>
      ) : null}
    </figure>
  )
}
