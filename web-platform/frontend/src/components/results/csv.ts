import type { ColumnSpec, TableSpec } from '@/api/types'
import { rawValue } from '@/formatting/number'

export interface CsvContext {
  scenarioId?: string | null
  scenarioTitle?: string | null
  units?: string | null
  solver?: string | null
  timestamp?: string | null
}

const CRLF = '\r\n'
const NEEDS_QUOTING = /[",\r\n]/

export function quoteField(field: string): string {
  if (NEEDS_QUOTING.test(field) || field !== field.trim()) {
    return `"${field.replace(/"/g, '""')}"`
  }
  return field
}

function cellText(value: unknown, column: ColumnSpec): string {
  if (value === null || value === undefined) return ''

  if (column.numeric) {
    if (typeof value === 'number') return rawValue(value)
    const parsed = Number(value)
    return Number.isFinite(parsed) ? rawValue(parsed) : String(value)
  }

  if (typeof value === 'number') return rawValue(value)
  if (typeof value === 'boolean') return value ? 'true' : 'false'
  if (typeof value === 'string') return value
  return JSON.stringify(value) ?? ''
}

function commentLine(label: string, value: string | null | undefined): string | null {
  const text = (value ?? '').trim()
  if (!text) return null
  return `# ${label}: ${text.replace(/[\r\n]+/g, ' ')}`
}

export function toCsv(table: TableSpec, context: CsvContext = {}): string {
  const header = [
    '# Orbital Rendezvous Simulator - exported table',
    commentLine('table', `${table.label} (${table.id})`),
    commentLine(
      'scenario',
      context.scenarioTitle && context.scenarioId
        ? `${context.scenarioTitle} (${context.scenarioId})`
        : (context.scenarioTitle ?? context.scenarioId ?? null),
    ),
    commentLine('generated', context.timestamp ?? new Date().toISOString()),
    commentLine('units', context.units),
    commentLine('solver', context.solver),
    commentLine('caption', table.caption),
    '# Values are full machine precision, not the rounded figures shown on screen.',
  ].filter((line): line is string => line !== null)

  const columnHeader = table.columns
    .map((column) => quoteField(column.unit ? `${column.label} (${column.unit})` : column.label))
    .join(',')

  const body = table.rows.map((row) =>
    table.columns.map((column) => quoteField(cellText(row[column.key], column))).join(','),
  )

  return [...header, columnHeader, ...body].join(CRLF) + CRLF
}

export function csvFilename(table: TableSpec, scenarioId?: string | null): string {
  const stem = [scenarioId, table.id].filter(Boolean).join('_')
  return `${stem.replace(/[^A-Za-z0-9._-]+/g, '-')}.csv`
}
