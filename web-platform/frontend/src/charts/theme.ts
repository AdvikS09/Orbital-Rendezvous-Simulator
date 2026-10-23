import type { DashStyle } from '@/api/types'
import type { PlotLayout } from 'plotly.js-dist-min'

export const SERIES_COLORS: readonly string[] = [
  '#0072B2',
  '#D55E00',
  '#009E73',
  '#CC79A7',
  '#E69F00',
  '#56B4E9',
  '#000000',
]

export const DASH_MAP: Record<DashStyle, string> = {
  solid: 'solid',
  dash: 'dash',
  dot: 'dot',
  dashdot: 'dashdot',
}

export interface ChartPalette {
  font: string
  text: string
  muted: string
  grid: string
  zeroline: string
  axisLine: string
  surface: string
  border: string
}

const DARK_FALLBACK: ChartPalette = {
  font: '"Inter", "Segoe UI", system-ui, sans-serif',
  text: '#a7b6cc',
  muted: '#7286a3',
  grid: '#1e2b42',
  zeroline: '#3a4d6e',
  axisLine: '#2a3a56',
  surface: '#16223a',
  border: '#2a3a56',
}

const LIGHT_FALLBACK: ChartPalette = {
  font: '"Inter", "Segoe UI", system-ui, sans-serif',
  text: '#40506a',
  muted: '#64748b',
  grid: '#e3e9f2',
  zeroline: '#b3c0d3',
  axisLine: '#d0d9e6',
  surface: '#ffffff',
  border: '#d0d9e6',
}

export function readCssVar(name: string, fallback: string): string {
  if (typeof document === 'undefined' || typeof getComputedStyle !== 'function') {
    return fallback
  }
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim()
  return value.length > 0 ? value : fallback
}

export function chartPalette(dark: boolean): ChartPalette {
  const fallback = dark ? DARK_FALLBACK : LIGHT_FALLBACK
  return {
    font: readCssVar('--font-sans', fallback.font),
    text: readCssVar('--text-secondary', fallback.text),
    muted: readCssVar('--text-muted', fallback.muted),
    grid: readCssVar('--border-subtle', fallback.grid),
    zeroline: readCssVar('--border-strong', fallback.zeroline),
    axisLine: readCssVar('--border-default', fallback.axisLine),
    surface: readCssVar('--surface-overlay', fallback.surface),
    border: readCssVar('--border-default', fallback.border),
  }
}

export function seriesPalette(dark: boolean): string[] {
  const colors = SERIES_COLORS.slice()
  colors[6] = readCssVar('--series-7', dark ? '#f0f0f0' : '#1a1a1a')
  return colors
}

export function seriesColor(index: number, palette: readonly string[] = SERIES_COLORS): string {
  const size = palette.length
  if (size === 0) {
    return '#0072B2'
  }
  const wrapped = ((index % size) + size) % size
  return palette[wrapped] ?? '#0072B2'
}

export function prefersReducedMotion(): boolean {
  if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') {
    return false
  }
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches
}

export function isDarkTheme(): boolean {
  if (typeof document === 'undefined') {
    return true
  }
  return document.documentElement.getAttribute('data-theme') !== 'light'
}

export function plotlyLayout(dark: boolean): PlotLayout {
  const palette = chartPalette(dark)
  const font = { family: palette.font, size: 12, color: palette.text }

  const axis = {
    gridcolor: palette.grid,
    zerolinecolor: palette.zeroline,
    linecolor: palette.axisLine,
    tickcolor: palette.axisLine,
    tickfont: { family: palette.font, size: 11, color: palette.muted },
    titlefont: { family: palette.font, size: 12, color: palette.text },
    ticks: 'outside',
    ticklen: 4,
    showline: true,
    zeroline: true,
    automargin: true,
  }

  return {
    autosize: true,
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font,
    colorway: seriesPalette(dark),
    hovermode: 'x unified',
    hoverlabel: {
      bgcolor: palette.surface,
      bordercolor: palette.border,
      font: { family: palette.font, size: 12, color: readCssVar('--text-primary', dark ? '#e8eef7' : '#0f1a2b') },
      align: 'left',
    },
    margin: { l: 68, r: 24, t: 34, b: 52 },
    legend: {
      orientation: 'h',
      x: 0,
      y: 1.06,
      xanchor: 'left',
      yanchor: 'bottom',
      bgcolor: 'rgba(0,0,0,0)',
      font: { family: palette.font, size: 11, color: palette.text },
      itemsizing: 'constant',
    },
    xaxis: { ...axis },
    yaxis: { ...axis },
    transition: { duration: 0 },
  }
}
